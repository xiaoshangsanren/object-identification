"""Class-pair retrieval difficulty analysis for fine-grained embeddings."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseDifficultyEvaluation:
    """完整类别对指标和逐类别困难邻居的分析结果。

    作用:
        保存所有无向类别对、每个类别到其他类别的方向性指标，以及汇总统计。
    参数:
        summary: 全局汇总字典，包含类别对数量、指标定义和最困难类别对。
        pairs: 长度``[C*(C-1)/2]``的无向类别对指标列表。
        per_class: 长度``[C]``的逐类别困难邻居列表；每类保留最困难的``H``个方向性结果。
        confusions_per_class: 长度``[C]``的逐类别Top-1错分统计；每类包含
            对其余``C-1``类的错分次数/比例和按次数排序的Top-5。
    返回值:
        成对难度评测容器本身。
    """

    summary: dict[str, Any]
    pairs: list[dict[str, Any]]
    per_class: list[dict[str, Any]]
    confusions_per_class: list[dict[str, Any]]


def _direction_metrics(
    query_indices: torch.Tensor,
    other_label: int,
    first_positive_ranks: torch.Tensor,
    margins: torch.Tensor,
    global_top1_labels: torch.Tensor,
    recall_ks: tuple[int, ...],
) -> dict[str, Any]:
    """计算一个有向类别关系A到B的检索难度。

    作用:
        汇总A类Query在仅含“其他A类+B类”的Gallery中的首个A类命中名次，
        并统计全196类Gallery中第一名被预测为B的次数。
    参数:
        query_indices: A类Query在测试集中的行号，形状``[N_A]``。
        other_label: 被比较类别B的标量标签。
        first_positive_ranks: 所有Query对所有目标类别的首个同类名次矩阵，形状``[N,C]``。
        margins: 最佳同类相似度减最佳目标类别相似度，形状``[N,C]``。
        global_top1_labels: 全类别Gallery的Top-1预测标签，形状``[N]``。
        recall_ks: 需要统计的检索截断位置，当前为``[1,2,3,4,5]``。
    返回值:
        方向性指标字典，包含``recall_at_k``、首个正例名次、相似度边界和全局混淆率。
    """

    ranks = first_positive_ranks[query_indices, other_label]
    direction_margins = margins[query_indices, other_label]
    query_count = len(query_indices)
    recall = {
        str(k): float(ranks.le(k).float().mean().item()) for k in recall_ks
    }
    global_confusion_count = int(
        global_top1_labels[query_indices].eq(other_label).sum().item()
    )
    return {
        "num_queries": query_count,
        "recall_at_k": recall,
        "mean_first_positive_rank": float(ranks.float().mean().item()),
        "median_first_positive_rank": float(ranks.float().median().item()),
        "mean_similarity_margin": float(direction_margins.mean().item()),
        "median_similarity_margin": float(direction_margins.median().item()),
        "global_top1_confusion_count": global_confusion_count,
        "global_top1_confusion_rate": global_confusion_count / query_count,
    }


@torch.inference_mode()
def evaluate_pairwise_difficulty(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    query_chunk_size: int = 256,
    recall_ks: tuple[int, ...] = (1, 2, 3, 4, 5),
    top_n_per_class: int = 20,
) -> PairwiseDifficultyEvaluation:
    """计算所有类别对的双类别受限检索Recall@1至Recall@5。

    作用:
        对A类Query和候选B类，构造``Gallery=(A中除Query自身的图片)+B类全部图片``；
        统计第一个A类positive在该Gallery中的名次。遍历所有``C*(C-1)/2``对，得到
        A到B、B到A、双向平衡Recall@K以及前1到前5平均失败率。
    参数:
        features: 全部测试图的视觉特征，形状``[N,D]``。
        labels: 与特征对应的类别标签，形状``[N]``。
        class_names: 类别名称列表，形状``[C]``。
        device: 分块相似度矩阵``[Q,N]``的计算设备。
        query_chunk_size: 每个相似度分块的Query数量``Q``。
        recall_ks: 截断位置列表，形状``[R]``，默认``[1,2,3,4,5]``。
        top_n_per_class: 汇总文件为每个类别直接展示的最困难邻居数。
    返回值:
        ``PairwiseDifficultyEvaluation``；pairs长度``[C*(C-1)/2]``，per_class长度``[C]``，
        每个类别保存Top-H困难邻居。
    """

    if features.ndim != 2:
        raise ValueError("features must have shape [N,D].")
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must have shape [N] and match features.")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    recall_ks = tuple(sorted(set(int(k) for k in recall_ks)))
    if not recall_ks or any(k <= 0 for k in recall_ks):
        raise ValueError("recall_ks must contain positive integers.")

    labels_cpu = labels.long().cpu()
    num_images, _ = features.shape
    num_classes = len(class_names)
    if labels_cpu.numel() and (
        labels_cpu.min().item() < 0 or labels_cpu.max().item() >= num_classes
    ):
        raise ValueError("A label falls outside class_names.")

    class_indices = [
        torch.nonzero(labels_cpu.eq(label), as_tuple=False).flatten()
        for label in range(num_classes)
    ]
    invalid_classes = [label for label, indices in enumerate(class_indices) if len(indices) < 2]
    if invalid_classes:
        raise ValueError(
            "Pairwise retrieval requires at least two test images in every class; "
            f"invalid labels={invalid_classes}"
        )

    gallery_features = F.normalize(features.float(), dim=1).to(device)
    best_similarity_to_class = torch.empty((num_images, num_classes), dtype=torch.float32)
    outranking_negative_counts = torch.empty((num_images, num_classes), dtype=torch.int32)
    class_indices_device = [indices.to(device) for indices in class_indices]

    for start in range(0, num_images, query_chunk_size):
        end = min(start + query_chunk_size, num_images)
        similarities = gallery_features[start:end] @ gallery_features.T
        local_rows = torch.arange(end - start, device=device)
        global_rows = torch.arange(start, end, device=device)
        similarities[local_rows, global_rows] = -torch.inf

        chunk_best = torch.empty((end - start, num_classes), device=device)
        for label, indices in enumerate(class_indices_device):
            chunk_best[:, label] = similarities[:, indices].max(dim=1).values
        query_labels = labels_cpu[start:end].to(device)
        best_positive = chunk_best.gather(1, query_labels[:, None]).squeeze(1)

        chunk_counts = torch.empty(
            (end - start, num_classes), dtype=torch.int32, device=device
        )
        for label, indices in enumerate(class_indices_device):
            chunk_counts[:, label] = similarities[:, indices].gt(
                best_positive[:, None]
            ).sum(dim=1)
        best_similarity_to_class[start:end] = chunk_best.cpu()
        outranking_negative_counts[start:end] = chunk_counts.cpu()

    own_best_similarity = best_similarity_to_class.gather(1, labels_cpu[:, None]).squeeze(1)
    margins = own_best_similarity[:, None] - best_similarity_to_class
    first_positive_ranks = outranking_negative_counts + 1
    global_top1_labels = best_similarity_to_class.argmax(dim=1)

    class_prototypes = []
    for indices in class_indices:
        prototype = F.normalize(features[indices].float(), dim=1).mean(dim=0)
        class_prototypes.append(F.normalize(prototype, dim=0))
    prototype_similarity = torch.stack(class_prototypes) @ torch.stack(class_prototypes).T

    pairs: list[dict[str, Any]] = []
    directed_by_class: list[list[dict[str, Any]]] = [[] for _ in range(num_classes)]
    for class_a in range(num_classes):
        for class_b in range(class_a + 1, num_classes):
            a_to_b = _direction_metrics(
                class_indices[class_a],
                class_b,
                first_positive_ranks,
                margins,
                global_top1_labels,
                recall_ks,
            )
            b_to_a = _direction_metrics(
                class_indices[class_b],
                class_a,
                first_positive_ranks,
                margins,
                global_top1_labels,
                recall_ks,
            )
            balanced_recall = {
                str(k): (a_to_b["recall_at_k"][str(k)] + b_to_a["recall_at_k"][str(k)])
                / 2.0
                for k in recall_ks
            }
            total_queries = a_to_b["num_queries"] + b_to_a["num_queries"]
            micro_recall = {
                str(k): (
                    a_to_b["recall_at_k"][str(k)] * a_to_b["num_queries"]
                    + b_to_a["recall_at_k"][str(k)] * b_to_a["num_queries"]
                )
                / total_queries
                for k in recall_ks
            }
            difficulty_score = sum(1.0 - balanced_recall[str(k)] for k in recall_ks) / len(
                recall_ks
            )
            pair = {
                "class_a_label": class_a,
                "class_a_name": class_names[class_a],
                "class_b_label": class_b,
                "class_b_name": class_names[class_b],
                "num_queries": total_queries,
                "a_to_b": a_to_b,
                "b_to_a": b_to_a,
                "balanced_recall_at_k": balanced_recall,
                "micro_recall_at_k": micro_recall,
                "difficulty_score_r1_to_r5": difficulty_score,
                "prototype_cosine_similarity": float(
                    prototype_similarity[class_a, class_b].item()
                ),
                "global_top1_mutual_confusion_count": (
                    a_to_b["global_top1_confusion_count"]
                    + b_to_a["global_top1_confusion_count"]
                ),
            }
            pairs.append(pair)
            directed_by_class[class_a].append(
                {
                    "other_label": class_b,
                    "other_class_name": class_names[class_b],
                    **a_to_b,
                    "balanced_recall_at_k": balanced_recall,
                    "pair_difficulty_score_r1_to_r5": difficulty_score,
                    "prototype_cosine_similarity": pair["prototype_cosine_similarity"],
                }
            )
            directed_by_class[class_b].append(
                {
                    "other_label": class_a,
                    "other_class_name": class_names[class_a],
                    **b_to_a,
                    "balanced_recall_at_k": balanced_recall,
                    "pair_difficulty_score_r1_to_r5": difficulty_score,
                    "prototype_cosine_similarity": pair["prototype_cosine_similarity"],
                }
            )

    pairs.sort(
        key=lambda item: (
            -item["difficulty_score_r1_to_r5"],
            item["balanced_recall_at_k"][str(max(recall_ks))],
            -item["prototype_cosine_similarity"],
        )
    )
    for rank, pair in enumerate(pairs, start=1):
        pair["difficulty_rank"] = rank

    per_class: list[dict[str, Any]] = []
    confusions_per_class: list[dict[str, Any]] = []
    for label, neighbors in enumerate(directed_by_class):
        neighbors.sort(
            key=lambda item: (
                -item["pair_difficulty_score_r1_to_r5"],
                item["recall_at_k"][str(max(recall_ks))],
                -item["prototype_cosine_similarity"],
            )
        )
        per_class.append(
            {
                "label": label,
                "class_name": class_names[label],
                "num_queries": len(class_indices[label]),
                "top_hardest": neighbors[: min(top_n_per_class, len(neighbors))],
            }
        )

        confusion_rows = [
            {
                "predicted_label": neighbor["other_label"],
                "predicted_class_name": neighbor["other_class_name"],
                "confusion_count": neighbor["global_top1_confusion_count"],
                "confusion_rate": neighbor["global_top1_confusion_rate"],
            }
            for neighbor in neighbors
        ]
        confusion_rows.sort(
            key=lambda item: (
                -item["confusion_count"],
                -item["confusion_rate"],
                item["predicted_label"],
            )
        )
        for rank, row in enumerate(confusion_rows, start=1):
            row["error_rank"] = rank
        nonzero_confusions = [
            row for row in confusion_rows if row["confusion_count"] > 0
        ]
        error_count = sum(row["confusion_count"] for row in confusion_rows)
        query_count = len(class_indices[label])
        confusions_per_class.append(
            {
                "true_label": label,
                "true_class_name": class_names[label],
                "num_queries": query_count,
                "correct_top1_count": query_count - error_count,
                "correct_top1_rate": (query_count - error_count) / query_count,
                "error_top1_count": error_count,
                "error_top1_rate": error_count / query_count,
                "top5_error_classifications": nonzero_confusions[:5],
                "all_other_class_confusions": confusion_rows,
            }
        )

    scores = torch.tensor(
        [pair["difficulty_score_r1_to_r5"] for pair in pairs], dtype=torch.float32
    )
    summary = {
        "protocol": "two_class_restricted_leave_one_out_retrieval",
        "definition": (
            "For A->B, each A query retrieves from all other A images plus all B images. "
            "Recall@K is one when an A positive appears in the first K positions."
        ),
        "num_images": num_images,
        "num_classes": num_classes,
        "num_unordered_pairs": len(pairs),
        "recall_ks": list(recall_ks),
        "primary_difficulty_metric": (
            "mean of (1 - balanced bidirectional Recall@K) over K=1..5; higher is harder"
        ),
        "difficulty_score_mean": float(scores.mean().item()),
        "difficulty_score_median": float(scores.median().item()),
        "top_hardest_pairs": pairs[: min(100, len(pairs))],
    }
    return PairwiseDifficultyEvaluation(
        summary=summary,
        pairs=pairs,
        per_class=per_class,
        confusions_per_class=confusions_per_class,
    )


def save_pairwise_difficulty(
    output_dir: Path,
    evaluation: PairwiseDifficultyEvaluation,
) -> dict[str, str]:
    """把成对难度结果保存为JSON和便于排序查看的CSV。

    作用:
        写入全类别对、逐类别困难邻居、摘要以及扁平CSV四个文件。
    参数:
        output_dir: 当前E0或E1实验输出目录。
        evaluation: 成对难度评测对象；pairs为``[C*(C-1)/2]``，per_class为``[C]``。
    返回值:
        四个产物名称组成的字典，不包含主数据张量。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "pairwise_difficulty_summary.json"
    pairs_path = output_dir / "pairwise_difficulty_all_pairs.json"
    per_class_path = output_dir / "pairwise_difficulty_per_class.json"
    csv_path = output_dir / "pairwise_difficulty_all_pairs.csv"
    confusion_path = output_dir / "top1_confusion_per_class.json"
    confusion_csv_path = output_dir / "top1_confusion_all_directions.csv"
    for path, value in (
        (summary_path, evaluation.summary),
        (pairs_path, evaluation.pairs),
        (per_class_path, evaluation.per_class),
        (confusion_path, evaluation.confusions_per_class),
    ):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    recall_ks = tuple(int(k) for k in evaluation.summary["recall_ks"])
    fieldnames = [
        "difficulty_rank",
        "class_a_label",
        "class_a_name",
        "class_b_label",
        "class_b_name",
        "num_queries",
        "difficulty_score_r1_to_r5",
        "prototype_cosine_similarity",
        "global_top1_mutual_confusion_count",
    ]
    fieldnames.extend(f"balanced_recall_at_{k}" for k in recall_ks)
    fieldnames.extend(f"micro_recall_at_{k}" for k in recall_ks)
    fieldnames.extend(f"a_to_b_recall_at_{k}" for k in recall_ks)
    fieldnames.extend(f"b_to_a_recall_at_{k}" for k in recall_ks)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in evaluation.pairs:
            row = {key: pair[key] for key in fieldnames if key in pair}
            for k in recall_ks:
                row[f"balanced_recall_at_{k}"] = pair["balanced_recall_at_k"][str(k)]
                row[f"micro_recall_at_{k}"] = pair["micro_recall_at_k"][str(k)]
                row[f"a_to_b_recall_at_{k}"] = pair["a_to_b"]["recall_at_k"][str(k)]
                row[f"b_to_a_recall_at_{k}"] = pair["b_to_a"]["recall_at_k"][str(k)]
            writer.writerow(row)

    confusion_fieldnames = [
        "true_label",
        "true_class_name",
        "predicted_label",
        "predicted_class_name",
        "num_queries",
        "confusion_count",
        "confusion_rate",
        "error_rank",
    ]
    with confusion_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=confusion_fieldnames)
        writer.writeheader()
        for class_result in evaluation.confusions_per_class:
            for confusion in class_result["all_other_class_confusions"]:
                writer.writerow(
                    {
                        "true_label": class_result["true_label"],
                        "true_class_name": class_result["true_class_name"],
                        "predicted_label": confusion["predicted_label"],
                        "predicted_class_name": confusion["predicted_class_name"],
                        "num_queries": class_result["num_queries"],
                        "confusion_count": confusion["confusion_count"],
                        "confusion_rate": confusion["confusion_rate"],
                        "error_rank": confusion["error_rank"],
                    }
                )

    return {
        "summary": summary_path.name,
        "all_pairs_json": pairs_path.name,
        "per_class_json": per_class_path.name,
        "all_pairs_csv": csv_path.name,
        "top1_confusion_per_class_json": confusion_path.name,
        "top1_confusion_all_directions_csv": confusion_csv_path.name,
    }


def build_pairwise_metrics_section(
    evaluation: PairwiseDifficultyEvaluation,
    artifacts: dict[str, str],
    top_pair_count: int = 20,
) -> dict[str, Any]:
    """构建写入主metrics.json的紧凑成对难度摘要。

    作用:
        避免将全部``C*(C-1)/2``对重复塞入主指标文件，仅保留定义、统计、文件索引和
        最困难的若干类别对。
    参数:
        evaluation: 完整成对难度结果；pairs长度``[C*(C-1)/2]``。
        artifacts: ``save_pairwise_difficulty``返回的产物文件名字典。
        top_pair_count: 主指标中保留的困难类别对数量``H``。
    返回值:
        可直接赋给``metrics['pairwise_difficulty']``的字典，Top列表长度``[H]``。
    """

    return {
        **{key: value for key, value in evaluation.summary.items() if key != "top_hardest_pairs"},
        "artifacts": artifacts,
        "top_hardest_pairs": evaluation.summary["top_hardest_pairs"][:top_pair_count],
    }
