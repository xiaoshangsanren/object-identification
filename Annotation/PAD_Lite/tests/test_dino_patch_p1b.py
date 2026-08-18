from __future__ import annotations

import unittest

import numpy as np

from PAD_Lite.dino_patch_fusion import fuse_normalized_features


class DinoPatchP1bTests(unittest.TestCase):
    def test_equal_weight_joint_cosine_matches_branch_score_average(self) -> None:
        cls_query = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        cls_support = np.asarray([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        patch_query = np.asarray([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
        patch_support = np.asarray([[1.0, 0.0], [0.0, -2.0]], dtype=np.float32)

        fused_query = fuse_normalized_features(cls_query, patch_query, 0.5)
        fused_support = fuse_normalized_features(cls_support, patch_support, 0.5)
        cls_query = cls_query / np.linalg.norm(cls_query, axis=1, keepdims=True)
        cls_support = cls_support / np.linalg.norm(cls_support, axis=1, keepdims=True)
        patch_query = patch_query / np.linalg.norm(patch_query, axis=1, keepdims=True)
        patch_support = patch_support / np.linalg.norm(
            patch_support, axis=1, keepdims=True
        )
        expected = 0.5 * (cls_query @ cls_support.T) + 0.5 * (
            patch_query @ patch_support.T
        )
        np.testing.assert_allclose(fused_query @ fused_support.T, expected, atol=1e-6)

    def test_fused_features_are_unit_normalized(self) -> None:
        cls = np.asarray([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        patch = np.asarray([[0.0, 4.0], [-1.0, 1.0]], dtype=np.float32)
        fused = fuse_normalized_features(cls, patch, 0.5)
        np.testing.assert_allclose(
            np.linalg.norm(fused, axis=1), np.ones(2), atol=1e-6
        )
        self.assertEqual(fused.shape, (2, 4))

    def test_invalid_weight_is_rejected(self) -> None:
        features = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "cls_weight"):
            fuse_normalized_features(features, features, 1.1)

    def test_mismatched_sample_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample counts differ"):
            fuse_normalized_features(
                np.ones((2, 3), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32),
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
