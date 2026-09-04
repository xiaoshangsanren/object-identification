"""Unit tests for the leave-one-out gallery protocol."""

from __future__ import annotations

import torch
import pytest

from fg_retrain.retrieval import evaluate_leave_one_out


def test_query_is_excluded_and_nearest_other_image_is_used() -> None:
    """验证留一法会排除Query自身。

    作用:
        使用特征``[4,2]``检查每个Query的Top-K中不存在自身且Rank-1全部正确。
    参数:
        无。
    返回值:
        无；断言失败时由pytest报告。
    """

    features = torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    result = evaluate_leave_one_out(
        features,
        labels,
        class_names=["class zero", "class one"],
        device=torch.device("cpu"),
        query_chunk_size=2,
        recall_ks=(1, 2),
        compute_map=True,
        save_top_k=3,
    )

    assert result.metrics["gallery_size_per_query"] == 3
    assert result.metrics["micro_rank1"] == 1.0
    assert result.metrics["macro_rank1"] == 1.0
    for query_index, record in enumerate(result.per_query):
        assert record["top_neighbors"][0]["test_row_index"] != query_index
        assert all(n["test_row_index"] != query_index for n in record["top_neighbors"])


def test_macro_and_micro_rank1_are_computed_separately() -> None:
    """验证Macro与Micro Rank-1采用不同加权方式。

    作用:
        使用不均衡标签``[5]``构造已知结果，检查两种聚合指标。
    参数:
        无。
    返回值:
        无；断言失败时由pytest报告。
    """

    # Class 0 has three correct queries; class 1 has two incorrect queries.
    features = torch.tensor(
        [
            [1.00, 0.00],
            [1.00, 0.00],
            [1.00, 0.00],
            [0.00, 1.00],
            [0.00, -1.00],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1])
    result = evaluate_leave_one_out(
        features,
        labels,
        class_names=["majority", "minority"],
        device=torch.device("cpu"),
        query_chunk_size=5,
        recall_ks=(1,),
        compute_map=False,
        save_top_k=1,
    )

    assert result.metrics["micro_rank1"] == pytest.approx(3 / 5)
    assert result.metrics["macro_rank1"] == pytest.approx(0.5)
