"""Tests for all-class-pair Recall@1..5 difficulty analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fg_retrain.pairwise_difficulty import (
    evaluate_pairwise_difficulty,
    save_pairwise_difficulty,
)


def _synthetic_pairwise_result():
    """构建具有一个明确困难类别对的合成结果。

    作用:
        创建特征``[6,3]``；A和B类互相混淆，C类与它们明显分离。
    参数:
        无。
    返回值:
        ``PairwiseDifficultyEvaluation``；包含3个无向类别对。
    """

    features = torch.tensor(
        [
            [1.00, 0.00, 0.00],
            [0.80, 0.60, 0.00],
            [0.995, 0.10, 0.00],
            [0.75, 0.66, 0.00],
            [0.00, 0.00, 1.00],
            [0.00, 0.10, 0.995],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    return evaluate_pairwise_difficulty(
        features=features,
        labels=labels,
        class_names=["A", "B", "C"],
        device=torch.device("cpu"),
        query_chunk_size=2,
        recall_ks=(1, 2, 3, 4, 5),
        top_n_per_class=2,
    )


def test_pairwise_recall_reports_rank1_through_rank5() -> None:
    """验证每个类别方向都保存Recall@1至Recall@5。

    作用:
        检查困难A/B对的方向性和双向平衡累计召回率。
    参数:
        无。
    返回值:
        无；断言失败时由pytest报告。
    """

    result = _synthetic_pairwise_result()
    assert result.summary["num_unordered_pairs"] == 3
    hardest = result.pairs[0]
    assert (hardest["class_a_name"], hardest["class_b_name"]) == ("A", "B")
    assert hardest["a_to_b"]["recall_at_k"] == pytest.approx(
        {"1": 0.0, "2": 0.5, "3": 1.0, "4": 1.0, "5": 1.0}
    )
    assert hardest["balanced_recall_at_k"] == pytest.approx(
        {"1": 0.0, "2": 0.5, "3": 1.0, "4": 1.0, "5": 1.0}
    )
    assert hardest["difficulty_score_r1_to_r5"] == pytest.approx(0.3)
    assert result.per_class[0]["top_hardest"][0]["other_class_name"] == "B"


def test_pairwise_outputs_include_json_and_csv(tmp_path: Path) -> None:
    """验证成对难度的四种文件产物均可读取。

    作用:
        将3个合成类别对写入临时目录并检查JSON条数和CSV表头。
    参数:
        tmp_path: pytest提供的临时目录，不包含主实验数据。
    返回值:
        无；断言失败时由pytest报告。
    """

    result = _synthetic_pairwise_result()
    artifacts = save_pairwise_difficulty(tmp_path, result)
    assert set(artifacts) == {
        "summary",
        "all_pairs_json",
        "per_class_json",
        "all_pairs_csv",
    }
    pairs = json.loads((tmp_path / artifacts["all_pairs_json"]).read_text(encoding="utf-8"))
    assert len(pairs) == 3
    csv_header = (tmp_path / artifacts["all_pairs_csv"]).read_text(
        encoding="utf-8-sig"
    ).splitlines()[0]
    assert "balanced_recall_at_1" in csv_header
    assert "balanced_recall_at_5" in csv_header
