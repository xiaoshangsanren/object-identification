"""Correctness tests for LaFG L_aux and its PK sampler."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from fg_retrain.laux_data import PKBatchSampler
from fg_retrain.laux_loss import lafg_auxiliary_contrastive_loss


def test_pk_sampler_yields_exactly_two_instances_per_selected_class() -> None:
    """验证PK采样器严格执行每类2张。

    作用:
        从标签``[20]``生成多个批次并检查每批大小``[6]``及每类计数。
    参数:
        无。
    返回值:
        无；断言失败时由pytest报告。
    """

    labels = [label for label in range(5) for _ in range(4)]
    sampler = PKBatchSampler(labels, classes_per_batch=3, batches_per_epoch=4, seed=7)
    for batch in sampler:
        counts = Counter(labels[index] for index in batch)
        assert len(batch) == 6
        assert len(counts) == 3
        assert set(counts.values()) == {2}


def test_laux_prefers_the_unique_same_class_image() -> None:
    """验证Laux选中唯一同类positive且可以反向传播。

    作用:
        用视觉特征``[4,2]``和标签``[4]``检查损失、pair accuracy及梯度有限性。
    参数:
        无。
    返回值:
        无；断言失败时由pytest报告。
    """

    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    output = lafg_auxiliary_contrastive_loss(embeddings, labels, temperature=0.1)
    assert output.loss.item() < 0.01
    assert output.positive_selection_accuracy.item() == 1.0
    output.loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_laux_rejects_more_than_one_positive_per_anchor() -> None:
    """验证Laux拒绝一个anchor存在多个positive的非法批次。

    作用:
        向损失输入特征``[4,8]``和不满足P×2的标签``[4]``，检查其抛出异常。
    参数:
        无。
    返回值:
        无；预期捕获``ValueError``。
    """

    with pytest.raises(ValueError, match="exactly one"):
        lafg_auxiliary_contrastive_loss(
            torch.randn(4, 8),
            torch.tensor([0, 0, 0, 1]),
            temperature=0.1,
        )
