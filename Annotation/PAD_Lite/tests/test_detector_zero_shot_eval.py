from __future__ import annotations

from pathlib import Path

import numpy as np

from PAD_Lite.detector_zero_shot_eval import (
    ImageRecord,
    _average_precision,
    box_iou,
    evaluate_predictions,
)


def _record(name: str, boxes: list[list[float]], classes: tuple[str, ...]) -> ImageRecord:
    return ImageRecord(
        image_path=Path(name),
        width=100,
        height=100,
        boxes=np.asarray(boxes, dtype=np.float32),
        class_names=classes,
    )


def test_box_iou_exact_and_disjoint() -> None:
    overlaps = box_iou(
        [0, 0, 10, 10],
        np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
    )
    np.testing.assert_allclose(overlaps, [1.0, 0.0])


def test_perfect_predictions_have_unit_ap_and_recall() -> None:
    records = [_record("one.jpg", [[10, 10, 50, 50]], ("t-72",))]
    predictions = {
        "one.jpg": [
            {
                "box": [10, 10, 50, 50],
                "score": 0.9,
                "source_class_id": 7,
                "source_class_name": "truck",
            }
        ]
    }
    assert _average_precision(records, predictions, 0.5) == 1.0
    metrics = evaluate_predictions(records, predictions, [0.5, 0.75], 0.3, 1024)
    assert metrics["map_50_95"] == 1.0
    assert metrics["operating_point"]["recall"] == 1.0
    assert metrics["recall_by_original_vehicle_class"]["t-72"]["recall"] == 1.0
    assert metrics["primary_crop_metrics"]["all_targets_detected_image_rate_iou50"] == 1.0
    assert metrics["primary_crop_metrics"]["complete_crop_recall"] == 1.0
    assert metrics["matched_box_quality_iou50"]["iou"]["mean"] == 1.0
    assert metrics["matched_box_quality_iou50"]["target_coverage"]["mean"] == 1.0
    assert metrics["matched_box_quality_iou50"]["crop_purity"]["mean"] == 1.0


def test_unmatched_prediction_is_false_positive() -> None:
    records = [_record("one.jpg", [[10, 10, 50, 50]], ("t-72",))]
    predictions = {
        "one.jpg": [
            {
                "box": [60, 60, 90, 90],
                "score": 0.9,
                "source_class_id": 2,
                "source_class_name": "car",
            }
        ]
    }
    metrics = evaluate_predictions(records, predictions, [0.5], 0.3, 1024)
    assert metrics["ap_50"] == 0.0
    assert metrics["operating_point"]["tp"] == 0
    assert metrics["operating_point"]["fp"] == 1
    assert metrics["operating_point"]["fn"] == 1
    assert metrics["primary_crop_metrics"]["zero_detected_image_rate"] == 1.0
    assert metrics["matched_box_quality_iou50"]["iou"]["mean"] is None


def test_image_level_success_requires_every_target() -> None:
    records = [
        _record(
            "two.jpg",
            [[0, 0, 20, 20], [60, 60, 90, 90]],
            ("t-72", "t-80"),
        )
    ]
    predictions = {
        "two.jpg": [
            {
                "box": [0, 0, 20, 20],
                "score": 0.9,
                "source_class_id": 7,
                "source_class_name": "truck",
            }
        ]
    }
    metrics = evaluate_predictions(records, predictions, [0.5, 0.75], 0.3, 1024)
    assert metrics["primary_crop_metrics"]["object_recall_iou50"] == 0.5
    assert metrics["primary_crop_metrics"]["all_targets_detected_image_rate_iou50"] == 0.0
    assert metrics["primary_crop_metrics"]["all_complete_crop_image_rate"] == 0.0


def test_target_coverage_detects_vehicle_cut_by_crop() -> None:
    records = [_record("cut.jpg", [[0, 0, 100, 100]], ("t-72",))]
    predictions = {
        "cut.jpg": [
            {
                "box": [0, 0, 80, 100],
                "score": 0.9,
                "source_class_id": 7,
                "source_class_name": "truck",
            }
        ]
    }
    metrics = evaluate_predictions(records, predictions, [0.5, 0.75], 0.3, 1024)
    assert metrics["primary_crop_metrics"]["object_recall_iou50"] == 1.0
    assert metrics["primary_crop_metrics"]["complete_crop_recall"] == 0.0
    assert metrics["matched_box_quality_iou50"]["iou"]["mean"] == 0.8
    assert metrics["matched_box_quality_iou50"]["target_coverage"]["mean"] == 0.8
    assert metrics["matched_box_quality_iou50"]["crop_purity"]["mean"] == 1.0
    assert metrics["matched_box_quality_iou50"]["vehicle_cut_rate_among_matches"] == 1.0
