"""LaFG's visual-only auxiliary contrastive objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LauxOutput:
    """Laux损失及视觉特征几何诊断量。

    作用:
        将参与反向传播的损失与不参与梯度的训练监控指标一起返回。
    参数:
        loss: 标量Laux张量，用于反向传播。
        positive_selection_accuracy: 批内将唯一positive排到第一位的比例，标量。
        mean_positive_similarity: ``B``个anchor与各自positive的平均余弦相似度，标量。
        mean_hardest_negative_similarity: ``B``个anchor最难negative的平均相似度，标量。
    返回值:
        诊断结果容器本身。
    """

    loss: torch.Tensor
    positive_selection_accuracy: torch.Tensor
    mean_positive_similarity: torch.Tensor
    mean_hardest_negative_similarity: torch.Tensor


def lafg_auxiliary_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> LauxOutput:
    """计算唯一正样本、全部异类负样本的LaFG Laux。

    作用:
        归一化视觉特征，构造余弦相似度矩阵``[B,B]``和平方距离矩阵``[B,B]``；
        屏蔽对角线后，让每个anchor在其余``B-1``张图中选中唯一同类positive。
    参数:
        embeddings: ViT视觉特征，形状``[B,D]``，当前``D=768``。
        labels: 类别ID，形状``[B]``；每个出现的类别必须恰好重复2次。
        temperature: 距离softmax温度标量``tau``。
    返回值:
        ``LauxOutput``；``loss``及三个诊断量均为标量张量。
    """

    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Expected embeddings (B,D) and matching labels (B,).")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    normalized = F.normalize(embeddings.float(), p=2, dim=1)
    similarity = normalized @ normalized.T
    batch_size = len(normalized)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=normalized.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_counts = positive_mask.sum(dim=1)
    if not torch.all(positive_counts == 1):
        raise ValueError(
            "LaFG L_aux requires exactly one same-class positive for every anchor; "
            f"observed counts={positive_counts.detach().cpu().tolist()}"
        )

    positive_indices = positive_mask.to(torch.int64).argmax(dim=1)
    squared_distance = 2.0 - 2.0 * similarity
    logits = -squared_distance / temperature
    logits = logits.masked_fill(self_mask, -torch.inf)
    row_indices = torch.arange(batch_size, device=normalized.device)
    positive_logits = logits[row_indices, positive_indices]
    log_denominator = torch.logsumexp(logits, dim=1)
    loss = -(positive_logits - log_denominator).mean()

    with torch.no_grad():
        predictions = logits.argmax(dim=1)
        negative_mask = ~(self_mask | positive_mask)
        hardest_negative = similarity.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
        diagnostics = LauxOutput(
            loss=loss,
            positive_selection_accuracy=predictions.eq(positive_indices).float().mean(),
            mean_positive_similarity=similarity[row_indices, positive_indices].mean(),
            mean_hardest_negative_similarity=hardest_negative.mean(),
        )
    return diagnostics
