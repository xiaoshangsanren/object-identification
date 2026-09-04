"""Exact leave-one-out image-to-image retrieval metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RetrievalEvaluation:
    """留一法检索的汇总指标与逐Query结果容器。

    作用:
        同时返回机器可读的整体指标和可写入JSONL的逐图邻居排序。
    参数:
        metrics: 汇总指标字典，包含标量指标和长度为``[C]``的逐类别结果。
        per_query: 长度为``[N]``的Query记录列表；每项含长度为``[K]``的Top-K邻居。
    返回值:
        评测结果容器本身。
    """

    metrics: dict[str, Any]
    per_query: list[dict[str, Any]]


@torch.inference_mode()
def evaluate_leave_one_out(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    query_chunk_size: int = 512,
    recall_ks: tuple[int, ...] = (1, 5, 10),
    compute_map: bool = True,
    save_top_k: int = 10,
) -> RetrievalEvaluation:
    """对每张测试图执行“其余测试图作Gallery”的精确留一法检索。

    作用:
        分块计算Query与全部Gallery的余弦相似度``[Q,N]``，将对角线自相似度置为
        负无穷，再计算Micro/Macro Rank-1、Recall@K、mAP和逐Query Top-K。
    参数:
        features: 全部测试图的视觉特征，形状``[N,D]``。
        labels: 与特征逐行对应的类别标签，形状``[N]``。
        class_names: 类别名称列表，形状``[C]``。
        device: 放置Gallery特征和相似度分块的CPU或CUDA设备。
        query_chunk_size: 单次参与矩阵乘法的Query数``Q``。
        recall_ks: 需要计算的K值列表，形状``[R]``。
        compute_map: 是否对完整``N-1``邻居排序并计算mAP。
        save_top_k: 每个Query写入结果的邻居数``K``。
    返回值:
        ``RetrievalEvaluation``：``metrics``为汇总字典，``per_query``长度为``[N]``；
        每个Query的有效Gallery固定为``N-1``张，不包含Query自身。
    """

    if features.ndim != 2:
        raise ValueError("features must have shape (N, D).")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must have shape (N,) matching features.")
    if len(features) < 2:
        raise ValueError("Leave-one-out retrieval needs at least two images.")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= len(class_names)):
        raise ValueError("A label index is outside class_names.")

    num_images = len(features)
    gallery_size = num_images - 1
    recall_ks = tuple(sorted(set(int(k) for k in recall_ks)))
    effective_recall_ks = {k: min(k, gallery_size) for k in recall_ks}
    stored_k = min(save_top_k, gallery_size)
    required_top_k = max([stored_k, *effective_recall_ks.values()])

    gallery_features = F.normalize(features.float(), dim=1).to(device)
    labels_cpu = labels.long().cpu()
    gallery_labels = labels_cpu.to(device)
    class_counts = Counter(labels_cpu.tolist())

    correct_rank1 = torch.zeros(num_images, dtype=torch.bool)
    has_positive = torch.tensor(
        [class_counts[int(label)] > 1 for label in labels_cpu.tolist()],
        dtype=torch.bool,
    )
    recall_hits = {k: torch.zeros(num_images, dtype=torch.bool) for k in recall_ks}
    average_precision = torch.full((num_images,), float("nan"), dtype=torch.float32)
    per_query: list[dict[str, Any]] = []

    for start in range(0, num_images, query_chunk_size):
        end = min(start + query_chunk_size, num_images)
        query_features = gallery_features[start:end]
        query_labels = gallery_labels[start:end]
        similarities = query_features @ gallery_features.T

        local_rows = torch.arange(end - start, device=device)
        global_rows = torch.arange(start, end, device=device)
        similarities[local_rows, global_rows] = -torch.inf

        top_scores, top_indices = similarities.topk(required_top_k, dim=1, largest=True)
        top_labels = gallery_labels[top_indices]
        relevance_top = top_labels.eq(query_labels[:, None])
        chunk_rank1 = relevance_top[:, 0].cpu()
        correct_rank1[start:end] = chunk_rank1

        for requested_k, effective_k in effective_recall_ks.items():
            recall_hits[requested_k][start:end] = relevance_top[:, :effective_k].any(dim=1).cpu()

        first_positive_ranks: list[int | None]
        if compute_map:
            ranked_indices = similarities.argsort(dim=1, descending=True)[:, :gallery_size]
            ranked_labels = gallery_labels[ranked_indices]
            relevance = ranked_labels.eq(query_labels[:, None])
            ranks = torch.arange(1, gallery_size + 1, device=device, dtype=torch.float32)
            precisions = relevance.cumsum(dim=1).float() / ranks[None, :]
            positive_counts = relevance.sum(dim=1)
            ap = (precisions * relevance).sum(dim=1) / positive_counts.clamp_min(1)
            ap = torch.where(positive_counts > 0, ap, torch.full_like(ap, float("nan")))
            average_precision[start:end] = ap.cpu()
            first_positive_ranks = []
            for row in relevance:
                positions = torch.nonzero(row, as_tuple=False)
                first_positive_ranks.append(int(positions[0, 0].item()) + 1 if len(positions) else None)
        else:
            first_positive_ranks = []
            for row in relevance_top:
                positions = torch.nonzero(row, as_tuple=False)
                first_positive_ranks.append(int(positions[0, 0].item()) + 1 if len(positions) else None)

        top_indices_cpu = top_indices[:, :stored_k].cpu()
        top_scores_cpu = top_scores[:, :stored_k].float().cpu()
        for local_index, query_index in enumerate(range(start, end)):
            label = int(labels_cpu[query_index].item())
            neighbors = []
            for rank, (neighbor_index_tensor, score_tensor) in enumerate(
                zip(top_indices_cpu[local_index], top_scores_cpu[local_index]),
                start=1,
            ):
                neighbor_index = int(neighbor_index_tensor.item())
                neighbor_label = int(labels_cpu[neighbor_index].item())
                neighbors.append(
                    {
                        "rank": rank,
                        "test_row_index": neighbor_index,
                        "label": neighbor_label,
                        "class_name": class_names[neighbor_label],
                        "cosine_similarity": float(score_tensor.item()),
                        "is_relevant": neighbor_label == label,
                    }
                )
            per_query.append(
                {
                    "query_test_row_index": query_index,
                    "query_label": label,
                    "query_class_name": class_names[label],
                    "gallery_size": gallery_size,
                    "has_positive_in_gallery": bool(has_positive[query_index].item()),
                    "rank1_correct": bool(chunk_rank1[local_index].item()),
                    "first_positive_rank": first_positive_ranks[local_index],
                    "average_precision": (
                        float(average_precision[query_index].item())
                        if compute_map and has_positive[query_index]
                        else None
                    ),
                    "top_neighbors": neighbors,
                }
            )

    valid_indices = torch.nonzero(has_positive, as_tuple=False).flatten()
    if len(valid_indices) == 0:
        raise ValueError("No query has another same-class image in its gallery.")

    per_class: list[dict[str, Any]] = []
    class_rank1_values: list[float] = []
    for label, class_name in enumerate(class_names):
        mask = labels_cpu.eq(label) & has_positive
        query_count = int(mask.sum().item())
        if query_count == 0:
            continue
        class_rank1 = float(correct_rank1[mask].float().mean().item())
        class_rank1_values.append(class_rank1)
        class_metrics: dict[str, Any] = {
            "label": label,
            "class_name": class_name,
            "num_queries": query_count,
            "rank1": class_rank1,
        }
        for k in recall_ks:
            class_metrics[f"recall_at_{k}"] = float(recall_hits[k][mask].float().mean().item())
        if compute_map:
            class_metrics["mAP"] = float(torch.nanmean(average_precision[mask]).item())
        per_class.append(class_metrics)

    valid_mask = has_positive
    metrics: dict[str, Any] = {
        "protocol": "test_leave_one_out_all_other_test_images_gallery",
        "num_queries": num_images,
        "valid_query_count": int(valid_mask.sum().item()),
        "gallery_size_per_query": gallery_size,
        "num_classes": len(set(labels_cpu.tolist())),
        "micro_rank1": float(correct_rank1[valid_mask].float().mean().item()),
        "macro_rank1": float(sum(class_rank1_values) / len(class_rank1_values)),
        "recall_at_k": {
            str(k): float(recall_hits[k][valid_mask].float().mean().item()) for k in recall_ks
        },
        "per_class": per_class,
    }
    if compute_map:
        metrics["mAP"] = float(torch.nanmean(average_precision[valid_mask]).item())
    return RetrievalEvaluation(metrics=metrics, per_query=per_query)
