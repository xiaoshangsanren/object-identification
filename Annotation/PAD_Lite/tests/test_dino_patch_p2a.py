from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from PAD_Lite.dino_patch_models import masked_average_patch_tokens
from PAD_Lite.dino_patch_weighted_models import masked_softmax_patch_pool


class DinoPatchP2aTests(unittest.TestCase):
    def test_zero_initialized_scorer_matches_masked_average(self) -> None:
        patches = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]]
        )
        mask = torch.tensor([[True, True, False]])
        scorer = nn.Linear(2, 1)
        nn.init.zeros_(scorer.weight)
        nn.init.zeros_(scorer.bias)

        pooled, weights = masked_softmax_patch_pool(patches, mask, scorer)
        expected = masked_average_patch_tokens(patches, mask)
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 0.0]]))

    def test_padding_attention_is_exactly_zero(self) -> None:
        patches = torch.randn(2, 5, 3)
        mask = torch.tensor(
            [[True, False, True, False, True], [False, True, True, True, False]]
        )
        scorer = nn.Linear(3, 1)
        _, weights = masked_softmax_patch_pool(patches, mask, scorer)
        self.assertTrue(bool((weights[~mask] == 0).all()))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_content_scorer_changes_weights_per_image(self) -> None:
        patches = torch.tensor(
            [
                [[5.0, 0.0], [1.0, 0.0]],
                [[-2.0, 0.0], [4.0, 0.0]],
            ]
        )
        mask = torch.ones(2, 2, dtype=torch.bool)
        scorer = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            scorer.weight.copy_(torch.tensor([[1.0, 0.0]]))
        _, weights = masked_softmax_patch_pool(patches, mask, scorer)
        self.assertGreater(float(weights[0, 0]), float(weights[0, 1]))
        self.assertLess(float(weights[1, 0]), float(weights[1, 1]))

    def test_scorer_receives_gradient_from_pooled_feature(self) -> None:
        patches = torch.tensor(
            [[[1.0, 0.0], [3.0, 1.0], [-2.0, 4.0]]], requires_grad=False
        )
        mask = torch.ones(1, 3, dtype=torch.bool)
        scorer = nn.Linear(2, 1)
        nn.init.zeros_(scorer.weight)
        nn.init.zeros_(scorer.bias)
        pooled, _ = masked_softmax_patch_pool(patches, mask, scorer)
        pooled[0, 0].backward()
        self.assertIsNotNone(scorer.weight.grad)
        self.assertGreater(float(scorer.weight.grad.abs().sum()), 0.0)

    def test_empty_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one valid patch"):
            masked_softmax_patch_pool(
                torch.ones(1, 2, 3),
                torch.zeros(1, 2, dtype=torch.bool),
                nn.Linear(3, 1),
            )


if __name__ == "__main__":
    unittest.main()
