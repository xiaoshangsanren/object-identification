from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from PAD_Lite.experiment_cli import _initialise_run, main
from PAD_Lite.experiment_registry import (
    get_experiment,
    load_registry,
    resolve_experiment_config,
)


class ExperimentRegistryTests(unittest.TestCase):
    def test_registry_contains_historical_switchable_modes(self) -> None:
        registry = load_registry()
        self.assertGreaterEqual(len(registry), 58)
        required = {
            "clip_b0_frozen_224",
            "clip_b1_no_text_224_u00",
            "clip_b1_no_text_224_u12",
            "clip_b2_text_anchor_224_u02",
            "dino_s_b0_center_crop_224",
            "dino_s_b1_no_text_center_crop_224_u00",
            "dino_s_b1_no_text_center_crop_224_u12",
            "dino_s_b2_text_anchor_center_crop_224_u00",
            "dino_s_b2_text_anchor_center_crop_224_u12",
            "dino_s_b1_no_text_letterbox_336_u00",
            "dino_s_b2_text_anchor_letterbox_336_u00",
            "dino_b_b2_text_anchor_letterbox_336_u00",
            "dino_s_p1a_patch_average_letterbox_336",
            "dino_s_p1b_cls_patch_average_equal_fusion_336",
            "dino_s_p2a_weighted_patch_letterbox_336",
            "dino_s_p2b_cls_weighted_patch_equal_fusion_336",
        }
        self.assertTrue(required <= set(registry))

    def test_text_and_no_text_presets_resolve_to_different_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            b1 = get_experiment("dino_s_b1_no_text_letterbox_336_u00")
            b2 = get_experiment("dino_s_b2_text_anchor_letterbox_336_u00")
            b1_config = resolve_experiment_config(b1, root / "b1")
            b2_config = resolve_experiment_config(b2, root / "b2")

        self.assertEqual(b1.variant, "b1")
        self.assertFalse(b1.text_anchor)
        self.assertEqual(b1_config["b1"]["anchor_weight"], 0)
        self.assertEqual(b2.variant, "b2")
        self.assertTrue(b2.text_anchor)
        self.assertGreater(b2_config["b2"]["anchor_weight"], 0)
        self.assertIn("prompt", b2_config["b2"])
        self.assertEqual(b1_config["data"]["image_size"], 336)
        self.assertEqual(b2_config["data"]["image_size"], 336)

    def test_matrix_presets_keep_each_unfreeze_depth_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero = get_experiment("dino_s_b2_text_anchor_center_crop_224_u00")
            twelve = get_experiment("dino_s_b2_text_anchor_center_crop_224_u12")
            zero_config = resolve_experiment_config(zero, root / "zero")
            twelve_config = resolve_experiment_config(twelve, root / "twelve")
        self.assertEqual(zero_config["model"]["unfreeze_last_blocks"], 0)
        self.assertEqual(twelve_config["model"]["unfreeze_last_blocks"], 12)

    def test_dry_run_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "must_not_exist"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "run",
                        "dino_s_b1_no_text_letterbox_336_u00",
                        "--fold",
                        "1",
                        "--output-root",
                        str(output_root),
                        "--dry-run",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertFalse(payload["text_anchor"])
            self.assertEqual(payload["folds"], [1])
            self.assertFalse(output_root.exists())

    def test_new_run_saves_snapshot_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = get_experiment("dino_s_b2_text_anchor_letterbox_336_u00")
            run_root = Path(temporary) / "managed_run"
            config = resolve_experiment_config(spec, run_root)
            _initialise_run(run_root, "managed_run", spec, config, {})

            manifest = json.loads(
                (run_root / "experiment_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["experiment_name"], spec.name)
            self.assertTrue(manifest["text_anchor"])
            self.assertTrue((run_root / "resolved_config.json").is_file())
            self.assertTrue(
                (
                    run_root
                    / "source_snapshot"
                    / "PAD_Lite"
                    / "src"
                    / "experiment_cli.py"
                ).is_file()
            )
            self.assertTrue(
                (
                    run_root
                    / "source_snapshot"
                    / "PAD_Lite"
                    / "experiments"
                    / spec.preset_path.name
                ).is_file()
            )

            with self.assertRaises(FileExistsError):
                _initialise_run(run_root, "managed_run", spec, config, {})


if __name__ == "__main__":
    unittest.main()
