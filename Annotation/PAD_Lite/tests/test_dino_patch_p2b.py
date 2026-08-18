from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PAD_Lite.dino_patch_fusion import _validate_sources
from PAD_Lite.dino_patch_weighted_fusion import (
    P2A_SOURCE_VARIANT,
    validate_p2a_source,
    validate_p2b_settings,
)


def _bundle() -> dict:
    return {
        "class_names": ["a", "b"],
        "support_labels": [0, 1],
        "query_labels": [0],
    }


def _prediction(sample_id: str = "sample-1") -> list[dict]:
    return [
        {
            "sample_id": sample_id,
            "source_image": "source.jpg",
            "class_name": "a",
            "tiny": False,
            "crowded": False,
            "clipped": False,
        }
    ]


class DinoPatchP2bTests(unittest.TestCase):
    def test_fixed_equal_weight_configuration_is_accepted(self) -> None:
        weight = validate_p2b_settings(
            {
                "cls_weight": 0.5,
                "patch_weight": 0.5,
                "candidate_scope": "all_classes",
                "calibration": False,
                "prototype_gate": False,
                "query_margin_gate": False,
                "reuse_cached_features": True,
                "novel_query_tuning": False,
            }
        )
        self.assertEqual(weight, 0.5)

    def test_weight_scan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cls_weight=0.5"):
            validate_p2b_settings({"cls_weight": 0.6, "patch_weight": 0.4})

    def test_calibration_gate_and_query_tuning_are_rejected(self) -> None:
        for field in ("calibration", "prototype_gate", "query_margin_gate"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_p2b_settings({field: True})
        with self.assertRaisesRegex(ValueError, "Novel Query"):
            validate_p2b_settings({"novel_query_tuning": True})

    def test_query_metadata_misalignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query order differs"):
            _validate_sources(
                _bundle(),
                _bundle(),
                _prediction("p0-sample"),
                _prediction("p2a-sample"),
            )

    def test_p2a_source_provenance_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "variant": P2A_SOURCE_VARIANT,
                        "text_anchor_used": False,
                        "cls_feature_used": False,
                    }
                ),
                encoding="utf-8",
            )
            payload = validate_p2a_source(root)
            self.assertEqual(payload["variant"], P2A_SOURCE_VARIANT)
            (root / "run_config.json").write_text(
                json.dumps({"variant": "p1a_patch_average_only"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected P2a source"):
                validate_p2a_source(root)


if __name__ == "__main__":
    unittest.main()
