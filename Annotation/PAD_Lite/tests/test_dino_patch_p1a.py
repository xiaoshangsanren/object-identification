from __future__ import annotations

import unittest

import torch
from PIL import Image

from PAD_Lite.dino_patch_engine import PatchLetterboxTransform
from PAD_Lite.dino_patch_models import masked_average_patch_tokens


class DinoPatchP1aTests(unittest.TestCase):
    def test_square_image_uses_all_patches(self) -> None:
        result = PatchLetterboxTransform(336, 14, train=False)(
            Image.new("RGB", (200, 200), "white")
        )
        self.assertEqual(tuple(result["image"].shape), (3, 336, 336))
        self.assertEqual(tuple(result["valid_patch_mask"].shape), (576,))
        self.assertEqual(int(result["valid_patch_mask"].sum()), 576)

    def test_landscape_padding_is_excluded(self) -> None:
        result = PatchLetterboxTransform(336, 14, train=False)(
            Image.new("RGB", (400, 100), "white")
        )
        mask = result["valid_patch_mask"].reshape(24, 24)
        self.assertEqual(int(mask.sum()), 144)
        self.assertFalse(bool(mask[:9].any()))
        self.assertTrue(bool(mask[9:15].all()))
        self.assertFalse(bool(mask[15:].any()))

    def test_portrait_padding_is_excluded(self) -> None:
        result = PatchLetterboxTransform(336, 14, train=False)(
            Image.new("RGB", (100, 400), "white")
        )
        mask = result["valid_patch_mask"].reshape(24, 24)
        self.assertEqual(int(mask.sum()), 144)
        self.assertFalse(bool(mask[:, :9].any()))
        self.assertTrue(bool(mask[:, 9:15].all()))
        self.assertFalse(bool(mask[:, 15:].any()))

    def test_masked_average_ignores_padding_tokens(self) -> None:
        patches = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
        mask = torch.tensor([[True, True, False]])
        actual = masked_average_patch_tokens(patches, mask)
        expected = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(actual, expected)

    def test_empty_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one valid patch"):
            masked_average_patch_tokens(
                torch.ones(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool)
            )


if __name__ == "__main__":
    unittest.main()
