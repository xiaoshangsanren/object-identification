from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ANNOTATION_ROOT, serializable_config
from .data import (
    CropIndex,
    build_eval_loader,
    build_train_loader,
    load_fold,
    records_for_partition,
)
from .dino_engine import _dino_backbone_name
from .dino_models import load_local_dino
from .dino_patch_engine import (
    PatchCropDataset,
    PatchLetterboxTransform,
    _batch_to_device,
)
from .dino_patch_weighted_models import DinoMaskedWeightedPatchModel
from .engine import _amp_enabled, choose_device, seed_everything
from .losses import batch_hard_triplet_loss
from .metrics import compute_retrieval_metrics, save_retrieval_outputs, summarize_folds


P2A_VARIANT = "p2a_masked_weighted_patch_only"
P2A_REPRESENTATION = "dynamic_masked_weighted_patch_only"
CHECKPOINT_FORMAT = "annotation_pad_lite_dino_patch_p2a_v1"


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        创建父目录并以 UTF-8 缩进格式写入 JSON。
    输入参数：
        path (Path)：目标文件；payload (Any)：可 JSON 序列化对象。
    返回值：
        None：写入完成后无返回数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_annotation_path(value: str | Path) -> Path:
    """
    方法作用：
        将相对路径按 Annotation 根目录解析为绝对路径。
    输入参数：
        value (str|Path)：绝对路径或相对 Annotation 的路径。
    返回值：
        Path：展开用户目录并解析后的绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _output_dir(config: dict[str, Any], fold_number: int) -> Path:
    """
    方法作用：
        计算 P2a 当前折的输出目录。
    输入参数：
        config：包含 output_root 的配置；fold_number：折号。
    返回值：
        Path：output_root/fold_XX。
    """
    return config["paths"]["output_root"] / f"fold_{fold_number:02d}"


def _make_loaders(
    config: dict[str, Any],
    fold: dict[str, Any],
    crop_index: CropIndex,
    seed: int,
):
    """
    方法作用：
        构造 P2a 基础类训练/验证 DataLoader 与 PK 采样器。
    输入参数：
        config：数据/模型/P2a 配置；fold：当前折定义；crop_index：裁剪索引；
        seed：采样随机种子。
    返回值：
        tuple：train_loader、sampler、val_loader、Mtr、Mval；批次主数据为
        image [B,3,S,S]、mask [B,N]、label [B]。
    """
    settings = config["p2a"]
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"].get("patch_size", 14))
    base_classes = list(fold["base_classes"])
    class_to_local = {name: index for index, name in enumerate(base_classes)}
    train_records = records_for_partition(
        crop_index,
        fold,
        "base_train",
        exclude_tiny=bool(config["data"].get("exclude_tiny_train", True)),
    )
    val_records = records_for_partition(
        crop_index,
        fold,
        "base_val",
        exclude_tiny=bool(config["data"].get("exclude_tiny_val", False)),
    )
    train_loader, sampler = build_train_loader(
        PatchCropDataset(
            train_records,
            PatchLetterboxTransform(image_size, patch_size, train=True),
            class_to_local,
        ),
        classes_per_batch=int(settings["classes_per_batch"]),
        instances_per_class=int(settings["instances_per_class"]),
        workers=int(config["data"]["workers"]),
        seed=seed,
    )
    val_loader = build_eval_loader(
        PatchCropDataset(
            val_records,
            PatchLetterboxTransform(image_size, patch_size, train=False),
            class_to_local,
        ),
        batch_size=int(config["data"]["eval_batch_size"]),
        workers=int(config["data"]["workers"]),
    )
    return train_loader, sampler, val_loader, len(train_records), len(val_records)


def _weight_batch_statistics(
    weights: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, float]:
    """
    方法作用：
        汇总一个批次中 Patch 权重的峰值、归一化熵、有效 Patch 数及填充质量。
    输入参数：
        weights (torch.Tensor)：Patch 权重 [B,N]；masks (torch.Tensor)：有效掩码 [B,N]。
    返回值：
        dict[str,float]：批次平均注意力统计标量。
    """
    weights = weights.float()
    masks = masks.bool()
    valid_counts = masks.sum(dim=1).float()
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
    normalized_entropy = entropy / valid_counts.log().clamp_min(1e-12)
    return {
        "attention_max": float(weights.max(dim=1).values.mean()),
        "attention_normalized_entropy": float(normalized_entropy.mean()),
        "attention_effective_patches": float(entropy.exp().mean()),
        "attention_invalid_mass": float((weights * (~masks)).sum(dim=1).mean()),
    }


def train_p2a_one_epoch(
    model: DinoMaskedWeightedPatchModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
    amp: bool,
) -> dict[str, float]:
    """
    方法作用：
        运行一个 P2a Epoch，以 ID+Triplet 训练 Patch Scorer 与检索头并记录权重统计。
    输入参数：
        model：P2a 模型；loader：image [B,3,S,S]、mask [B,N]、label [B]；
        optimizer/scaler：训练状态；device：设备；settings：损失配置；amp：混合精度开关。
    返回值：
        dict[str,float]：样本平均损失、准确率和 Patch 注意力统计。
    """
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            embeddings, weights = model.encode_with_weights(images, masks)
            logits = model.classifier(embeddings)
            id_loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=float(settings.get("label_smoothing", 0.0)),
            )
            triplet_loss = batch_hard_triplet_loss(
                embeddings,
                labels,
                margin=float(settings["triplet_margin"]),
            )
            loss = (
                float(settings["id_weight"]) * id_loss
                + float(settings["triplet_weight"]) * triplet_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.head_parameters(),
            max_norm=float(settings.get("gradient_clip", 5.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        sample_count += batch_size
        totals["loss"] += float(loss.detach()) * batch_size
        totals["id_loss"] += float(id_loss.detach()) * batch_size
        totals["triplet_loss"] += float(triplet_loss.detach()) * batch_size
        totals["accuracy"] += float((logits.argmax(dim=1) == labels).sum())
        weight_stats = _weight_batch_statistics(weights.detach(), masks)
        for key, value in weight_stats.items():
            totals[key] += value * batch_size
    return {
        key: value / max(1, sample_count)
        for key, value in totals.items()
    }


@torch.inference_mode()
def evaluate_p2a_classifier(
    model: DinoMaskedWeightedPatchModel,
    loader,
    device: torch.device,
    amp: bool,
) -> dict[str, float | None]:
    """
    方法作用：
        在基础类验证集上评估 P2a 分类性能和 Patch 权重分布。
    输入参数：
        model：P2a 模型；loader：验证批次；device：设备；amp：混合精度开关。
        批次张量形状为 image [B,3,S,S]、mask [B,N]、label [B]。
    返回值：
        dict[str,float|None]：loss、all/core/tiny accuracy 及注意力统计。
    """
    model.eval()
    total_loss = 0.0
    total = correct = core_total = core_correct = tiny_total = tiny_correct = 0
    attention_totals: defaultdict[str, float] = defaultdict(float)
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            embeddings, weights = model.encode_with_weights(images, masks)
            logits = model.classifier(embeddings)
            loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        tiny = batch["tiny"].bool().to(device)
        core = ~tiny
        batch_size = labels.numel()
        total_loss += float(loss) * batch_size
        total += batch_size
        correct += int((predictions == labels).sum())
        core_total += int(core.sum())
        core_correct += int(((predictions == labels) & core).sum())
        tiny_total += int(tiny.sum())
        tiny_correct += int(((predictions == labels) & tiny).sum())
        for key, value in _weight_batch_statistics(weights, masks).items():
            attention_totals[key] += value * batch_size
    result: dict[str, float | None] = {
        "loss": total_loss / max(1, total),
        "accuracy_all": correct / max(1, total),
        "accuracy_core": core_correct / core_total if core_total else None,
        "accuracy_tiny": tiny_correct / tiny_total if tiny_total else None,
    }
    result.update(
        {key: value / max(1, total) for key, value in attention_totals.items()}
    )
    return result


@torch.inference_mode()
def encode_weighted_loader(
    model: DinoMaskedWeightedPatchModel,
    loader,
    device: torch.device,
    amp: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    """
    方法作用：
        编码完整数据集，并同时保留每张图的 Patch 权重和有效掩码。
    输入参数：
        model：P2a 模型；loader：M 个样本的加载器；device：设备；amp：混合精度开关。
    返回值：
        tuple：特征 [M,D]、标签 [M]、长度 M 的元数据、权重 [M,N]、掩码 [M,N]。
    """
    model.eval()
    features = []
    labels = []
    attention_weights = []
    valid_masks = []
    metadata: list[dict[str, Any]] = []
    for batch in loader:
        images, masks, batch_labels = _batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            encoded, weights = model.encode_with_weights(images, masks)
        features.append(encoded.float().cpu())
        labels.append(batch_labels.cpu())
        attention_weights.append(weights.float().cpu())
        valid_masks.append(masks.cpu())
        for index in range(len(batch["sample_id"])):
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
    if not features:
        raise ValueError("Cannot encode an empty dataset")
    return (
        torch.cat(features).numpy().astype(np.float32, copy=False),
        torch.cat(labels).numpy().astype(np.int64, copy=False),
        metadata,
        torch.cat(attention_weights).numpy().astype(np.float32, copy=False),
        torch.cat(valid_masks).numpy().astype(bool, copy=False),
    )


def _attention_summary(weights: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    """
    方法作用：
        对整个数据集的 Patch 权重计算熵、峰值、有效 Patch 数和无效区域质量。
    输入参数：
        weights (np.ndarray)：权重矩阵 [M,N]；masks (np.ndarray)：布尔掩码 [M,N]。
    返回值：
        dict[str,Any]：数据集级注意力描述统计。
    """
    invalid_mass = (weights * (~masks)).sum(axis=1)
    valid_counts = masks.sum(axis=1).astype(np.float32)
    entropy = -(np.maximum(weights, 1e-12) * np.log(np.maximum(weights, 1e-12))).sum(
        axis=1
    )
    normalized_entropy = entropy / np.maximum(np.log(valid_counts), 1e-12)
    effective_patches = np.exp(entropy)
    max_weights = weights.max(axis=1)
    return {
        "sample_count": int(len(weights)),
        "invalid_attention_mass_max": float(invalid_mass.max(initial=0.0)),
        "invalid_attention_mass_mean": float(invalid_mass.mean()),
        "max_weight_mean": float(max_weights.mean()),
        "max_weight_min": float(max_weights.min()),
        "max_weight_max": float(max_weights.max()),
        "normalized_entropy_mean": float(normalized_entropy.mean()),
        "normalized_entropy_min": float(normalized_entropy.min()),
        "effective_patch_count_mean": float(effective_patches.mean()),
        "effective_patch_count_min": float(effective_patches.min()),
        "valid_patch_count_mean": float(valid_counts.mean()),
    }


def _true_class_margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    方法作用：
        计算每个 Query 的真实类得分减去最强错误类得分。
    输入参数：
        scores (np.ndarray)：类别得分 [Nq,C]；labels (np.ndarray)：真实标签 [Nq]。
    返回值：
        np.ndarray：真实类分类间隔，形状为 [Nq]。
    """
    rows = np.arange(len(labels))
    true_scores = scores[rows, labels]
    competitors = scores.copy()
    competitors[rows, labels] = -np.inf
    return true_scores - competitors.max(axis=1)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """
    方法作用：
        在样本数和方差足够时计算两组一维统计量的 Pearson 相关系数。
    输入参数：
        left、right (np.ndarray)：对齐的一维数组，形状均为 [N]。
    返回值：
        float|None：相关系数；数据退化时返回 None。
    """
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _compare_with_p0(
    config: dict[str, Any],
    fold_number: int,
    class_names: list[str],
    query_labels: np.ndarray,
    query_metadata: list[dict[str, Any]],
    patch_scores: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    方法作用：
        对齐 P0 CLS 与 P2a Patch 预测，统计两分支互补纠错和分类间隔相关性。
    输入参数：
        config：含 P0 缓存根目录的配置；fold_number：折号；class_names：C 个类名；
        query_labels [Nq]；query_metadata：长度 Nq；patch_scores [Nq,C]。
    返回值：
        tuple：诊断字典、P0 预测标签 [Nq]、P2a 预测标签 [Nq]。
    """
    raw_root = config["paths"].get("p0_features_root")
    if raw_root is None:
        raise KeyError("P2a config requires paths.p0_features_root")
    p0_root = _resolve_annotation_path(raw_root) / f"fold_{fold_number:02d}"
    predictions_path = p0_root / "predictions.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing P0 predictions: {predictions_path}")
    p0_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if len(p0_predictions) != len(query_metadata):
        raise ValueError("P0 and P2a query counts differ")

    p0_scores = np.empty_like(patch_scores)
    p0_predicted_ids = np.empty(len(query_labels), dtype=np.int64)
    for index, (p0_item, patch_item) in enumerate(
        zip(p0_predictions, query_metadata, strict=True)
    ):
        if p0_item["sample_id"] != patch_item["sample_id"]:
            raise ValueError(f"P0/P2a query order differs at index {index}")
        if p0_item["true_class"] != class_names[int(query_labels[index])]:
            raise ValueError(f"P0/P2a label differs at index {index}")
        for class_id, class_name in enumerate(class_names):
            p0_scores[index, class_id] = float(p0_item["class_scores"][class_name])
        p0_predicted_ids[index] = class_names.index(p0_item["predicted_class"])

    patch_predicted_ids = patch_scores.argmax(axis=1)
    cls_correct = p0_predicted_ids == query_labels
    patch_correct = patch_predicted_ids == query_labels
    per_class = {}
    for class_id, class_name in enumerate(class_names):
        class_mask = query_labels == class_id
        cls_wrong_patch_correct = int((~cls_correct & patch_correct & class_mask).sum())
        cls_correct_patch_wrong = int((cls_correct & ~patch_correct & class_mask).sum())
        per_class[class_name] = {
            "query_count": int(class_mask.sum()),
            "cls_wrong_patch_correct": cls_wrong_patch_correct,
            "cls_correct_patch_wrong": cls_correct_patch_wrong,
            "patch_net_rescue_potential": (
                cls_wrong_patch_correct - cls_correct_patch_wrong
            ),
        }
    diagnostics = {
        "cls_correct_patch_correct": int((cls_correct & patch_correct).sum()),
        "cls_correct_patch_wrong": int((cls_correct & ~patch_correct).sum()),
        "cls_wrong_patch_correct": int((~cls_correct & patch_correct).sum()),
        "cls_wrong_patch_wrong": int((~cls_correct & ~patch_correct).sum()),
        "patch_net_rescue_potential": int(
            (~cls_correct & patch_correct).sum() - (cls_correct & ~patch_correct).sum()
        ),
        "true_class_margin_correlation": _safe_correlation(
            _true_class_margins(p0_scores, query_labels),
            _true_class_margins(patch_scores, query_labels),
        ),
        "per_class": per_class,
        "p0_predictions": str(predictions_path),
    }
    return diagnostics, p0_predicted_ids, patch_predicted_ids


def _save_attention_visualizations(
    output_dir: Path,
    query_records,
    metadata: list[dict[str, Any]],
    weights: np.ndarray,
    masks: np.ndarray,
    query_labels: np.ndarray,
    class_names: list[str],
    p0_predictions: np.ndarray,
    patch_predictions: np.ndarray,
    image_size: int,
    patch_size: int,
    per_group: int,
) -> list[dict[str, Any]]:
    """
    方法作用：
        按 P0/P2a 正误组合导出代表性 Patch 权重热力图。
    输入参数：
        output_dir：输出目录；query_records/metadata：Nq 个 Query；weights、masks [Nq,N]；
        query_labels、p0_predictions、patch_predictions [Nq]；class_names：C 类；
        image_size=S；patch_size=P；per_group：每组上限。
    返回值：
        list[dict[str,Any]]：每张导出图的分组、样本、类别和最大权重索引记录。
    """
    if per_group <= 0:
        return []
    from matplotlib import colormaps

    records_by_id = {record.sample_id: record for record in query_records}
    cls_correct = p0_predictions == query_labels
    patch_correct = patch_predictions == query_labels
    groups = {
        "cls_wrong_patch_correct": ~cls_correct & patch_correct,
        "cls_correct_patch_wrong": cls_correct & ~patch_correct,
        "both_correct": cls_correct & patch_correct,
        "both_wrong": ~cls_correct & ~patch_correct,
    }
    transform = PatchLetterboxTransform(image_size, patch_size, train=False)
    target_root = output_dir / "attention_samples"
    target_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    grid = image_size // patch_size
    for group_name, group_mask in groups.items():
        candidates = np.flatnonzero(group_mask)
        if len(candidates) == 0:
            continue
        candidates = sorted(
            candidates,
            key=lambda item: float(weights[item].max()),
            reverse=True,
        )[:per_group]
        for query_index in candidates:
            item = metadata[query_index]
            record = records_by_id[item["sample_id"]]
            with Image.open(record.path) as source:
                canvas, geometry_mask = transform._letterbox(source.convert("RGB"))
            if not np.array_equal(geometry_mask.numpy(), masks[query_index]):
                raise ValueError("Visualization mask differs from encoded query mask")
            heat = weights[query_index].reshape(grid, grid)
            maximum = float(heat.max())
            normalized = heat / maximum if maximum > 0 else heat
            heat_image = Image.fromarray(normalized.astype(np.float32), mode="F").resize(
                (image_size, image_size), resample=Image.Resampling.BILINEAR
            )
            heat_full = np.asarray(heat_image, dtype=np.float32)
            color = colormaps["turbo"](heat_full)[..., :3] * 255.0
            base = np.asarray(canvas, dtype=np.float32)
            alpha = (0.70 * heat_full)[..., None]
            overlay = np.clip(base * (1.0 - alpha) + color * alpha, 0, 255).astype(
                np.uint8
            )
            filename = f"{group_name}_{item['sample_id']}.png"
            Image.fromarray(overlay, mode="RGB").save(target_root / filename)
            index_rows.append(
                {
                    "group": group_name,
                    "file": filename,
                    "sample_id": item["sample_id"],
                    "true_class": class_names[int(query_labels[query_index])],
                    "p0_predicted_class": class_names[int(p0_predictions[query_index])],
                    "p2a_predicted_class": class_names[
                        int(patch_predictions[query_index])
                    ],
                    "max_patch_weight": maximum,
                }
            )
    _write_json(target_root / "index.json", index_rows)
    return index_rows


def evaluate_p2a_novel_retrieval(
    model: DinoMaskedWeightedPatchModel,
    crop_index: CropIndex,
    fold: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """
    方法作用：
        在未见类 Support/Query 上评测 P2a，保存检索结果、Patch 权重、热力图及 P0 对照。
    输入参数：
        model：P2a 模型；crop_index：裁剪索引；fold：当前折；config：评测配置；
        device：设备；output_dir：结果目录。主特征为 Support [Ns,D]、Query [Nq,D]。
    返回值：
        dict[str,Any]：Rank-1/mAP/Recall@K、注意力统计和分支互补性指标。
    """
    novel_classes = list(fold["novel_classes"])
    class_to_local = {name: index for index, name in enumerate(novel_classes)}
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"].get("patch_size", 14))
    transform = PatchLetterboxTransform(image_size, patch_size, train=False)
    support_records = records_for_partition(
        crop_index,
        fold,
        "novel_support",
        exclude_tiny=bool(config["data"].get("exclude_tiny_support", True)),
    )
    query_records = records_for_partition(crop_index, fold, "novel_query")
    if {record.class_name for record in support_records} != set(novel_classes):
        raise ValueError("Support loses a novel class after tiny filtering")
    support_loader = build_eval_loader(
        PatchCropDataset(support_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    query_loader = build_eval_loader(
        PatchCropDataset(query_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    amp = _amp_enabled(config, device)
    (
        support_features,
        support_labels,
        _,
        support_weights,
        support_masks,
    ) = encode_weighted_loader(model, support_loader, device, amp)
    (
        query_features,
        query_labels,
        query_metadata,
        query_weights,
        query_masks,
    ) = encode_weighted_loader(model, query_loader, device, amp)
    query_tiny = np.asarray([item["tiny"] for item in query_metadata], dtype=bool)
    metrics, class_scores = compute_retrieval_metrics(
        support_features=support_features,
        support_labels=support_labels,
        query_features=query_features,
        query_labels=query_labels,
        query_tiny=query_tiny,
        class_names=novel_classes,
        prototype_top_k=int(config["retrieval"]["prototype_top_k"]),
    )
    diagnostics_cfg = config.get("diagnostics", {})
    compare_with_p0 = bool(diagnostics_cfg.get("compare_with_p0", True))
    patch_predictions = class_scores.argmax(axis=1)
    complementarity = None
    p0_predictions = None
    if compare_with_p0:
        complementarity, p0_predictions, patch_predictions = _compare_with_p0(
            config,
            int(fold["fold"]),
            novel_classes,
            query_labels,
            query_metadata,
            class_scores,
        )
    attention = {
        "support": _attention_summary(support_weights, support_masks),
        "query": _attention_summary(query_weights, query_masks),
    }
    visualizations = []
    if compare_with_p0:
        visualizations = _save_attention_visualizations(
            output_dir,
            query_records,
            query_metadata,
            query_weights,
            query_masks,
            query_labels,
            novel_classes,
            p0_predictions,
            patch_predictions,
            image_size,
            patch_size,
            int(diagnostics_cfg.get("heatmaps_per_group", 4)),
        )
    metrics.update(
        {
            "variant": P2A_VARIANT,
            "representation": P2A_REPRESENTATION,
            "visual_backbone": _dino_backbone_name(config),
            "fold": int(fold["fold"]),
            "novel_classes": novel_classes,
            "text_anchor_used": False,
            "cls_feature_used": False,
            "attention": attention,
            "p0_comparison_enabled": compare_with_p0,
            "attention_visualization_count": len(visualizations),
        }
    )
    if complementarity is not None:
        metrics["p0_complementarity"] = complementarity
    save_retrieval_outputs(
        output_dir,
        metrics,
        novel_classes,
        class_scores,
        query_labels,
        query_metadata,
        support_features,
        support_labels,
        query_features,
    )
    torch.save(
        {
            "sample_ids": [item["sample_id"] for item in query_metadata],
            "weights": torch.from_numpy(query_weights),
            "valid_patch_masks": torch.from_numpy(query_masks),
            "query_labels": torch.from_numpy(query_labels),
            "class_names": novel_classes,
        },
        output_dir / "query_attention.pt",
    )
    _write_json(output_dir / "attention_summary.json", attention)
    if complementarity is not None:
        _write_json(output_dir / "p0_complementarity.json", complementarity)
    return metrics


def _checkpoint_payload(
    epoch: int,
    model: DinoMaskedWeightedPatchModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    base_classes: list[str],
    best_accuracy: float,
) -> dict[str, Any]:
    """
    方法作用：
        组装包含 Patch Scorer 的可恢复 P2a 检查点。
    输入参数：
        epoch：轮次；model：P2a 模型；optimizer/scheduler/scaler：训练状态；
        base_classes：C 个基础类；best_accuracy：最佳验证准确率。
    返回值：
        dict[str,Any]：适配器与完整优化状态。
    """
    return {
        "format": CHECKPOINT_FORMAT,
        "variant": P2A_VARIANT,
        "epoch": epoch,
        "base_classes": base_classes,
        "best_accuracy": best_accuracy,
        "model_state": model.adapter_state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }


def _load_checkpoint(
    path: Path,
    model: DinoMaskedWeightedPatchModel,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict[str, Any]:
    """
    方法作用：
        加载并校验 P2a 检查点，确保其中存在 Patch Scorer，可选恢复训练状态。
    输入参数：
        path：检查点；model：待恢复模型；optimizer/scheduler/scaler：可选训练组件。
    返回值：
        dict[str,Any]：检查点原始内容。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported P2a checkpoint: {path}")
    state = checkpoint["model_state"]
    if not any(name.startswith("patch_scorer.") for name in state):
        raise ValueError(f"P2a checkpoint has no Patch scorer: {path}")
    model.load_state_dict(state, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def run_p2a_fold(
    config: dict[str, Any],
    fold_number: int,
    device_name: str,
    resume: bool,
    eval_only: bool,
) -> dict[str, Any]:
    """
    方法作用：
        执行单折 P2a 训练、基础类选模和未见类 Patch 检索评测。
    输入参数：
        config：完整配置；fold_number：折号；device_name：设备；resume：恢复开关；
        eval_only：只评测开关。
    返回值：
        dict[str,Any]：单折检索、注意力与最佳验证指标。
    """
    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + fold_number
    seed_everything(seed)
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    crop_index = CropIndex(config["paths"]["crop_root"])
    settings = config["p2a"]
    train_loader, sampler, val_loader, train_count, val_count = _make_loaders(
        config, fold, crop_index, seed
    )
    dino, _ = load_local_dino(config["paths"]["dino_model"])
    expected_patch_size = int(config["model"].get("patch_size", 14))
    actual_patch_size = int(dino.config.patch_size)
    if actual_patch_size != expected_patch_size:
        raise ValueError(
            f"Configured patch_size={expected_patch_size}, model uses {actual_patch_size}"
        )
    model = DinoMaskedWeightedPatchModel(
        dino,
        num_classes=len(fold["base_classes"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.head_parameters(),
                "lr": float(settings["head_lr"]),
                "name": "weighted_patch_head",
            }
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(settings.get("min_lr", 1e-7)),
    )
    amp = _amp_enabled(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir = _output_dir(config, fold_number)
    output_dir.mkdir(parents=True, exist_ok=True)
    head_count = sum(parameter.numel() for parameter in model.head_parameters())
    scorer_count = sum(parameter.numel() for parameter in model.scorer_parameters())
    run_config = {
        "format": "annotation_pad_lite_dino_patch_weighted_fold_v1",
        "variant": P2A_VARIANT,
        "representation": P2A_REPRESENTATION,
        "fold": fold_number,
        "base_classes": list(fold["base_classes"]),
        "novel_classes": list(fold["novel_classes"]),
        "device": str(device),
        "config": serializable_config(config),
        "train_samples": train_count,
        "val_samples": val_count,
        "parameter_groups": {
            "weighted_patch_head": head_count,
            "patch_scorer": scorer_count,
            "dino_backbone": 0,
        },
        "trainable_parameters": head_count,
        "scorer_zero_initialized": True,
        "text_anchor_used": False,
        "cls_feature_used": False,
    }
    _write_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_accuracy = float("-inf")
    best_checkpoint = output_dir / "best.pt"
    last_checkpoint = output_dir / "last.pt"
    if eval_only:
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Missing P2a checkpoint: {best_checkpoint}")
        best_accuracy = float(_load_checkpoint(best_checkpoint, model)["best_accuracy"])
    else:
        if resume and last_checkpoint.is_file():
            checkpoint = _load_checkpoint(
                last_checkpoint, model, optimizer, scheduler, scaler
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_accuracy = float(checkpoint["best_accuracy"])
            history_path = output_dir / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
        for epoch in range(start_epoch, epochs + 1):
            sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = train_p2a_one_epoch(
                model, train_loader, optimizer, scaler, device, settings, amp
            )
            val_metrics = evaluate_p2a_classifier(model, val_loader, device, amp)
            scheduler.step()
            selection = (
                val_metrics["accuracy_core"]
                if val_metrics["accuracy_core"] is not None
                else val_metrics["accuracy_all"]
            )
            record = {
                "epoch": epoch,
                "backbone_enabled": False,
                "seconds": round(time.time() - started, 3),
                "learning_rates": {
                    group.get("name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)
            payload = _checkpoint_payload(
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                list(fold["base_classes"]),
                max(best_accuracy, float(selection)),
            )
            torch.save(payload, last_checkpoint)
            if float(selection) > best_accuracy:
                best_accuracy = float(selection)
                payload["best_accuracy"] = best_accuracy
                torch.save(payload, best_checkpoint)
            _write_json(output_dir / "history.json", history)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        _load_checkpoint(best_checkpoint, model)

    metrics = evaluate_p2a_novel_retrieval(
        model, crop_index, fold, config, device, output_dir
    )
    metrics["best_base_val_accuracy"] = best_accuracy
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_p2a(
    config: dict[str, Any],
    folds: list[int],
    device_name: str,
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        运行指定折的 P2a 实验并生成跨折汇总。
    输入参数：
        config：P2a 配置；folds：折号列表；device_name：设备；resume/eval_only：运行模式。
    返回值：
        dict[str,Any]：逐折 results 和聚合 summary。
    """
    if config["data"].get("input_mode") != "letterbox":
        raise ValueError("P2a requires data.input_mode=letterbox")
    if int(config["model"].get("unfreeze_last_blocks", 0)) != 0:
        raise ValueError("P2a requires a frozen DINO backbone")
    results = []
    for fold in folds:
        metrics = run_p2a_fold(config, fold, device_name, resume, eval_only)
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    root = config["paths"]["output_root"]
    summary = summarize_folds(root)
    summary.update(
        {
            "variant": P2A_VARIANT,
            "representation": P2A_REPRESENTATION,
            "visual_backbone": _dino_backbone_name(config),
        }
    )
    _write_json(root / "summary.json", summary)
    return {"results": results, "summary": summary}
