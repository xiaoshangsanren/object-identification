from __future__ import annotations

from pathlib import Path

import numpy as np

from PAD_Lite.detector_finetune_eval import build_split_payloads
from PAD_Lite.detector_zero_shot_eval import ImageRecord


def _records() -> list[ImageRecord]:
    rows = []
    for class_name in ("t-72", "t-80"):
        for index in range(20):
            rows.append(
                ImageRecord(
                    image_path=Path(f"{class_name}_{index:03d}.jpg"),
                    width=100,
                    height=100,
                    boxes=np.asarray([[10, 10, 80, 80]], dtype=np.float32),
                    class_names=(class_name,),
                )
            )
    return rows


def test_fivefold_partitions_are_disjoint_and_complete() -> None:
    records = _records()
    payloads = build_split_payloads(records, 5, 0.1, 2026, "test_v1")
    all_names = {record.image_path.name for record in records}
    assert len(payloads) == 5
    test_occurrences = {name: 0 for name in all_names}
    for payload in payloads:
        train = set(payload["partitions"]["train"])
        validation = set(payload["partitions"]["validation"])
        test = set(payload["partitions"]["test"])
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == all_names
        for name in test:
            test_occurrences[name] += 1
    assert set(test_occurrences.values()) == {1}


def test_fivefold_split_is_reproducible() -> None:
    records = _records()
    first = build_split_payloads(records, 5, 0.1, 2026, "test_v1")
    second = build_split_payloads(records, 5, 0.1, 2026, "test_v1")
    assert first == second
