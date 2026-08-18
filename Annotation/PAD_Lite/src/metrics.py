from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _average_precision(relevant: np.ndarray, scores: np.ndarray) -> float:
    """
    方法作用：
        执行 _average_precision 对应的处理流程。
    
    输入参数：
        relevant (np.ndarray)：相关性布尔标记，形状为 [N]。
        scores (np.ndarray)：与标记逐一对应的检索得分，形状为 [N]。
    
    返回值：
        float：单个查询或类别的 AP 标量；没有正样本时返回 NaN。
    """
    relevant = relevant.astype(bool, copy=False)
    positives = int(relevant.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = relevant[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / (np.arange(len(ranked)) + 1)
    return float(precision[ranked].sum() / positives)


def top_k_prototype_scores(
    query_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    num_classes: int,
    top_k: int,
) -> np.ndarray:
    """
    方法作用：
        执行 top_k_prototype_scores 对应的处理流程。
    
    输入参数：
        query_features (np.ndarray)：归一化查询特征，形状为 [Nq, D]。
        support_features (np.ndarray)：归一化支持集特征，形状为 [Ns, D]。
        support_labels (np.ndarray)：支持集类别标签，形状为 [Ns]。
        num_classes (int)：新类别数量 C。
        top_k (int)：计算每类得分时选取的最高相似度数量 K。
    
    返回值：
        np.ndarray：查询样本对各类别的得分矩阵，形状为 [Nq, C]。
    """
    columns = []
    for class_id in range(num_classes):
        prototypes = support_features[support_labels == class_id]
        if len(prototypes) == 0:
            raise ValueError(f"No support prototype for local class {class_id}")
        similarities = query_features @ prototypes.T
        k = min(max(1, top_k), similarities.shape[1])
        selected = np.partition(
            similarities, similarities.shape[1] - k, axis=1
        )[:, -k:]
        columns.append(selected.mean(axis=1))
    return np.stack(columns, axis=1)


def _masked_accuracy(
    predictions: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> float | None:
    """
    方法作用：
        执行 _masked_accuracy 对应的处理流程。
    
    输入参数：
        predictions (np.ndarray)：预测类别编号，形状为 [N]。
        labels (np.ndarray)：真实类别编号，形状为 [N]。
        mask (np.ndarray)：指定参与统计样本的布尔掩码，形状为 [N]。
    
    返回值：
        float | None：方法执行得到的结果。
    """
    if not mask.any():
        return None
    return float((predictions[mask] == labels[mask]).mean())


def compute_retrieval_metrics(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    query_tiny: np.ndarray,
    class_names: Sequence[str],
    prototype_top_k: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    方法作用：
        执行 compute_retrieval_metrics 对应的处理流程。
    
    输入参数：
        support_features (np.ndarray)：归一化支持集特征，形状为 [Ns, D]。
        support_labels (np.ndarray)：支持集标签，形状为 [Ns]。
        query_features (np.ndarray)：归一化查询集特征，形状为 [Nq, D]。
        query_labels (np.ndarray)：查询集真实标签，形状为 [Nq]。
        query_tiny (np.ndarray)：查询样本是否为微小目标的掩码，形状为 [Nq]。
        class_names (Sequence[str])：按本地类别编号排列的 C 个类别名称。
        prototype_top_k (int)：方法所需的 prototype_top_k 参数。
    
    返回值：
        tuple[dict[str, Any], np.ndarray]：指标字典和类别得分矩阵；得分矩阵
        形状为 [Nq, C]。其中 Ns、Nq、C、D 分别表示支持样本数、查询
        样本数、类别数和特征维度。
    """
    class_scores = top_k_prototype_scores(
        query_features,
        support_features,
        support_labels,
        num_classes=len(class_names),
        top_k=prototype_top_k,
    )
    predictions = class_scores.argmax(axis=1)
    class_order = np.argsort(-class_scores, axis=1, kind="stable")
    recall_at = {}
    for k in (2, 3, 5):
        effective_k = min(k, len(class_names))
        hits = (class_order[:, :effective_k] == query_labels[:, None]).any(axis=1)
        recall_at[f"recall_at_{k}"] = float(hits.mean())
    core_mask = ~query_tiny
    tiny_mask = query_tiny

    per_class_rank1 = {}
    class_aps = []
    for class_id, class_name in enumerate(class_names):
        mask = query_labels == class_id
        per_class_rank1[class_name] = _masked_accuracy(predictions, query_labels, mask)
        class_ap = _average_precision(mask, class_scores[:, class_id])
        if not math.isnan(class_ap):
            class_aps.append(class_ap)

    prototype_similarities = query_features @ support_features.T
    retrieval_aps = []
    for index, label in enumerate(query_labels):
        relevant = support_labels == label
        retrieval_aps.append(
            _average_precision(relevant, prototype_similarities[index])
        )

    metrics = {
        "query_count": int(len(query_labels)),
        "support_count": int(len(support_labels)),
        "rank1_all": _masked_accuracy(
            predictions, query_labels, np.ones(len(query_labels), dtype=bool)
        ),
        "rank1_core": _masked_accuracy(predictions, query_labels, core_mask),
        "rank1_tiny": _masked_accuracy(predictions, query_labels, tiny_mask),
        "class_map": float(np.mean(class_aps)),
        "retrieval_map": float(np.nanmean(retrieval_aps)),
        "per_class_rank1": per_class_rank1,
        "tiny_query_count": int(tiny_mask.sum()),
        "prototype_top_k": int(prototype_top_k),
        **recall_at,
    }
    return metrics, class_scores


@torch.inference_mode()
def encode_loader(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    方法作用：
        执行 encode_loader 对应的处理流程。
    
    输入参数：
        encoder (nn.Module)：方法所需的 encoder 参数。
        loader (DataLoader)：逐批提供 image [B, 3, H, W]、label [B]
        及元数据的数据加载器。
        device (torch.device)：执行计算的 PyTorch 设备。
        amp (bool)：方法所需的 amp 参数。
    
    返回值：
        tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]：依次返回全部特征
        [N, D]、标签 [N] 和长度为 N 的样本元数据列表。
    """
    encoder.eval()
    feature_batches = []
    label_batches = []
    metadata: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            features = encoder.encode(images) if hasattr(encoder, "encode") else encoder(images)
        feature_batches.append(features.float().cpu())
        label_batches.append(batch["label"].long().cpu())
        batch_size = len(batch["sample_id"])
        for index in range(batch_size):
            metadata.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "source_image": batch["source_image"][index],
                    "class_name": batch["class_name"][index],
                    "tiny": bool(batch["tiny"][index]),
                    "crowded": bool(batch["crowded"][index]),
                    "clipped": bool(batch["clipped"][index]),
                }
            )
    if not feature_batches:
        raise ValueError("Cannot encode an empty dataset")
    features = torch.cat(feature_batches).numpy().astype(np.float32, copy=False)
    labels = torch.cat(label_batches).numpy().astype(np.int64, copy=False)
    return features, labels, metadata


def save_retrieval_outputs(
    output_dir: Path,
    metrics: dict[str, Any],
    class_names: Sequence[str],
    class_scores: np.ndarray,
    query_labels: np.ndarray,
    query_metadata: list[dict[str, Any]],
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
) -> None:
    """
    方法作用：
        执行 save_retrieval_outputs 对应的处理流程。
    
    输入参数：
        output_dir (Path)：当前实验的输出目录。
        metrics (dict[str, Any])：方法所需的 metrics 参数。
        class_names (Sequence[str])：方法所需的 class_names 参数。
        class_scores (np.ndarray)：查询样本的类别得分，形状为 [Nq, C]。
        query_labels (np.ndarray)：查询样本真实标签，形状为 [Nq]。
        query_metadata (list[dict[str, Any]])：方法所需的 query_metadata 参数。
        support_features (np.ndarray)：支持集特征，形状为 [Ns, D]。
        support_labels (np.ndarray)：支持集标签，形状为 [Ns]。
        query_features (np.ndarray)：查询集特征，形状为 [Nq, D]。
    
    返回值：
        None：方法直接完成相应操作。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    predicted_ids = class_scores.argmax(axis=1)
    for index, item in enumerate(query_metadata):
        scores = {
            class_names[class_id]: round(float(class_scores[index, class_id]), 6)
            for class_id in range(len(class_names))
        }
        predictions.append(
            {
                **item,
                "true_class": class_names[int(query_labels[index])],
                "predicted_class": class_names[int(predicted_ids[index])],
                "correct": bool(predicted_ids[index] == query_labels[index]),
                "class_scores": scores,
            }
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "class_names": list(class_names),
            "support_features": torch.from_numpy(support_features),
            "support_labels": torch.from_numpy(support_labels),
            "query_features": torch.from_numpy(query_features),
            "query_labels": torch.from_numpy(query_labels),
        },
        output_dir / "features.pt",
    )


def summarize_folds(variant_root: Path) -> dict[str, Any]:
    """
    方法作用：
        执行 summarize_folds 对应的处理流程。
    
    输入参数：
        variant_root (Path)：方法所需的 variant_root 参数。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    fold_metrics = []
    for fold_number in range(1, 6):
        path = variant_root / f"fold_{fold_number:02d}" / "metrics.json"
        if path.is_file():
            fold_metrics.append(json.loads(path.read_text(encoding="utf-8")))
    metric_names = (
        "rank1_all",
        "rank1_core",
        "rank1_tiny",
        "class_map",
        "retrieval_map",
    )
    aggregate = {}
    for name in metric_names:
        values = [float(item[name]) for item in fold_metrics if item.get(name) is not None]
        aggregate[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "fold_values": values,
        }
    summary = {
        "completed_folds": len(fold_metrics),
        "aggregate": aggregate,
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    (variant_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
