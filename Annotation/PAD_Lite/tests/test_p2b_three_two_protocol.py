from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from PAD_Lite.data import load_fold
from PAD_Lite.metrics import compute_retrieval_metrics
from PAD_Lite.p2b_three_two_protocol import build_three_two_payloads


def _source_fold(fold_number: int) -> dict:
    pairs = {
        fold: [f"c{fold}a", f"c{fold}b"] for fold in range(1, 6)
    }
    novel = pairs[fold_number]
    base = [name for fold in range(1, 6) if fold != fold_number for name in pairs[fold]]
    return {
        "version": "pad_lite_class_holdout_fold_v1",
        "fold": fold_number,
        "seed": 2026,
        "image_root": "train",
        "base_classes": base,
        "novel_classes": novel,
        "partitions": {
            "base_train": {name: [f"{name}_train.jpg"] for name in base},
            "base_val": {name: [f"{name}_val.jpg"] for name in base},
            "novel_support": {name: [f"{name}_support.jpg"] for name in novel},
            "novel_query": {name: [f"{name}_query.jpg"] for name in novel},
        },
    }


class P2bThreeTwoProtocolTests(unittest.TestCase):
    def test_builds_all_ten_unique_four_way_episodes(self) -> None:
        episodes = build_three_two_payloads([_source_fold(i) for i in range(1, 6)])
        self.assertEqual(len(episodes), 10)
        self.assertEqual(
            {tuple(item["source_test_folds"]) for item in episodes},
            {(a, b) for a in range(1, 6) for b in range(a + 1, 6)},
        )
        for episode in episodes:
            self.assertEqual(len(episode["base_classes"]), 6)
            self.assertEqual(len(episode["novel_classes"]), 4)
            self.assertFalse(
                set(episode["base_classes"]) & set(episode["novel_classes"])
            )

    def test_load_fold_accepts_generated_episode_ten(self) -> None:
        episode = build_three_two_payloads(
            [_source_fold(i) for i in range(1, 6)]
        )[-1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fold_10.json").write_text(json.dumps(episode), encoding="utf-8")
            loaded = load_fold(root, 10)
        self.assertEqual(loaded["fold"], 10)
        self.assertEqual(len(loaded["novel_classes"]), 4)

    def test_fixed_gallery_ids_are_propagated_to_four_way_episodes(self) -> None:
        folds = [_source_fold(i) for i in range(1, 6)]
        for fold in folds:
            fold["fixed_gallery"] = {
                "version": "fixed_semantic_balanced_gallery_v1",
                "manifest": "fixed.json",
                "manifest_sha256": "abc",
                "sample_ids_by_class": {
                    class_name: [f"{class_name}_crop"]
                    for class_name in fold["novel_classes"]
                },
            }
        episode = build_three_two_payloads(folds)[0]
        self.assertEqual(
            set(episode["fixed_gallery"]["sample_ids_by_class"]),
            set(episode["novel_classes"]),
        )
        self.assertEqual(
            episode["fixed_gallery"]["version"],
            "fixed_semantic_balanced_gallery_v1",
        )

    def test_four_way_metrics_report_candidate_recall(self) -> None:
        support = np.eye(4, dtype=np.float32)
        query = np.asarray(
            [[0.1, 0.2, 0.9, 0.8], [0.7, 0.6, 0.5, 0.4]], dtype=np.float32
        )
        metrics, _ = compute_retrieval_metrics(
            support,
            np.arange(4),
            query,
            np.asarray([3, 1]),
            np.zeros(2, dtype=bool),
            ["a", "b", "c", "d"],
            1,
        )
        self.assertEqual(metrics["rank1_all"], 0.0)
        self.assertEqual(metrics["recall_at_2"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
