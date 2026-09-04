from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANNOTATION_ROOT, serializable_config
from .data import CropIndex, build_eval_loader, build_train_loader, load_fold, records_for_partition
from .dino_engine import run_dino_variant
from .dino_patch_engine import PatchCropDataset, PatchLetterboxTransform, _batch_to_device
from .dino_patch_weighted_engine import (
    _attention_summary,
    _weight_batch_statistics,
    run_p2a,
)
from .dino_patch_weighted_fusion import run_p2b
from .dino_patch_weighted_models import masked_softmax_patch_pool
from .engine import choose_device, seed_everything
from .frozen_backbone_comparison import (
    load_comparison_config,
    run_frozen_backbone,
)
from .losses import batch_hard_triplet_loss
from .metrics import compute_retrieval_metrics, save_retrieval_outputs, summarize_folds


FORMAT_VERSION = "annotation_pad_lite_fixed_gallery_fourway_v1.00"
CHECKPOINT_FORMAT = "annotation_pad_lite_convnext_p2b_v1.00"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/fixed_gallery_fourway_v1_00.json"
)


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：按Annotation根目录解析配置路径。
    输入参数：value，相对或绝对路径。
    返回值：Path，绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    """
    方法作用：流式计算文件SHA-256。
    输入参数：path，普通文件。
    返回值：str，十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：创建父目录并写入UTF-8格式化JSON。
    输入参数：path；payload，可JSON序列化对象。
    返回值：无。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_fourway_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：读取四路固定Gallery实验配置并解析所有路径。
    输入参数：path，配置JSON。
    返回值：dict，路径字段均为绝对Path。
    """
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Four-way config schema_version must be 1")
    required = {"paths", "protocol", "raw", "dino_p2b", "convnext_p2b"}
    if missing := required - set(config):
        raise ValueError(f"Four-way config misses sections: {sorted(missing)}")
    for key, value in config["paths"].items():
        config["paths"][key] = _resolve_path(value)
    config["config_path"] = config_path
    return config


def validate_fixed_gallery(config: dict[str, Any]) -> dict[str, Any]:
    """
    方法作用：验证五折固定Gallery哈希、每类数量及Support/Query源图隔离。
    输入参数：config，四路实验配置。
    返回值：dict，逐折类别、样本ID、数量及隔离诊断。
    """
    manifest = config["paths"]["fixed_gallery_manifest"]
    expected_hash = str(config["protocol"]["fixed_gallery_manifest_sha256"])
    actual_hash = _sha256(manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"Fixed Gallery hash mismatch: {actual_hash} != {expected_hash}")
    index = CropIndex(config["paths"]["crop_root"])
    expected_per_class = int(config["protocol"]["gallery_per_class"])
    folds: list[dict[str, Any]] = []
    for fold_number in range(1, int(config["protocol"]["fold_count"]) + 1):
        fold = load_fold(config["paths"]["splits_root"], fold_number)
        fixed = fold.get("fixed_gallery")
        if not isinstance(fixed, dict) or fixed.get("manifest_sha256") != expected_hash:
            raise ValueError(f"Fold {fold_number} does not reference the expected fixed Gallery")
        support = records_for_partition(index, fold, "novel_support", exclude_tiny=True)
        query = records_for_partition(index, fold, "novel_query")
        counts = {
            name: sum(record.class_name == name for record in support)
            for name in fold["novel_classes"]
        }
        if any(value != expected_per_class for value in counts.values()):
            raise ValueError(f"Fold {fold_number} fixed Gallery counts differ: {counts}")
        support_ids = {record.sample_id for record in support}
        query_ids = {record.sample_id for record in query}
        support_sources = {record.source_image for record in support}
        query_sources = {record.source_image for record in query}
        if support_ids & query_ids or support_sources & query_sources:
            raise ValueError(f"Fold {fold_number} has fixed Gallery/Query leakage")
        folds.append(
            {
                "fold": fold_number,
                "novel_classes": list(fold["novel_classes"]),
                "gallery_count": len(support),
                "gallery_counts_by_class": counts,
                "gallery_sample_ids": [record.sample_id for record in support],
                "query_count": len(query),
                "sample_id_overlap": 0,
                "source_image_overlap": 0,
            }
        )
    return {
        "format": FORMAT_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": actual_hash,
        "gallery_per_class": expected_per_class,
        "folds": folds,
    }


class RetrievalHead(nn.Module):
    """
    方法作用：以Projection、BNNeck和ID分类器把冻结主干特征转为检索嵌入。
    输入参数：hidden_dim；embedding_dim；num_classes；dropout。
    返回值：RetrievalHead模块；主输入[B,H]，嵌入[B,D]，logits[B,C]。
    """

    def __init__(self, hidden_dim: int, embedding_dim: int, num_classes: int, dropout: float) -> None:
        """
        方法作用：初始化轻量检索头。
        输入参数：特征维H、嵌入维D、基础类数C、Dropout概率。
        返回值：无。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        方法作用：生成L2归一化检索嵌入。
        输入参数：features，Tensor[B,H]。
        返回值：Tensor[B,D]。
        """
        return F.normalize(self.bnneck(self.projection(self.dropout(features))), dim=-1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：同时生成检索嵌入和基础类ID logits。
        输入参数：features，Tensor[B,H]。
        返回值：tuple，embedding[B,D]、logits[B,C]。
        """
        embedding = self.encode(features)
        return embedding, self.classifier(embedding)


class FrozenConvNextP2B(nn.Module):
    """
    方法作用：冻结原始ConvNeXt-Tiny，训练全局头和动态加权Patch头并执行50/50 P2B融合。
    输入参数：convnext；基础类数；嵌入维；dropout。
    返回值：模型；图像[B,3,S,S]、mask[B,N]映射为两个[B,D]分支。
    """

    def __init__(self, convnext: nn.Module, num_classes: int, embedding_dim: int, dropout: float) -> None:
        """
        方法作用：初始化冻结主干、两个检索头和零初始化Patch Scorer。
        输入参数：ConvNeXt模型、类别数C、嵌入维D、Dropout概率。
        返回值：无。
        """
        super().__init__()
        self.convnext = convnext
        hidden_dim = int(convnext.config.hidden_sizes[-1])
        self.global_head = RetrievalHead(hidden_dim, embedding_dim, num_classes, dropout)
        self.patch_head = RetrievalHead(hidden_dim, embedding_dim, num_classes, dropout)
        self.patch_scorer = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.patch_scorer.weight)
        nn.init.zeros_(self.patch_scorer.bias)
        for parameter in self.convnext.parameters():
            parameter.requires_grad_(False)
        self.convnext.eval()

    def train(self, mode: bool = True):
        """
        方法作用：切换轻量头训练状态并强制冻结主干保持eval。
        输入参数：mode，是否训练。
        返回值：FrozenConvNextP2B自身。
        """
        super().train(mode)
        self.convnext.eval()
        return self

    def _frozen_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：无梯度提取ConvNeXt全局池化特征和最终空间token。
        输入参数：pixel_values，Tensor[B,3,S,S]。
        返回值：tuple，全局[B,768]、局部[B,N,768]。
        """
        with torch.no_grad():
            output = self.convnext(pixel_values=pixel_values)
        tokens = output.last_hidden_state.float().flatten(2).transpose(1, 2)
        return output.pooler_output.float(), tokens

    def encode_branches(
        self, pixel_values: torch.Tensor, valid_patch_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        方法作用：编码全局与动态Patch分支并返回Patch权重。
        输入参数：图像[B,3,S,S]；有效掩码[B,N]。
        返回值：global[B,D]、patch[B,D]、weights[B,N]。
        """
        global_features, tokens = self._frozen_features(pixel_values)
        if tokens.shape[:2] != valid_patch_mask.shape:
            raise ValueError(
                f"ConvNeXt tokens/mask differ: {tuple(tokens.shape)} vs {tuple(valid_patch_mask.shape)}"
            )
        pooled, weights = masked_softmax_patch_pool(tokens, valid_patch_mask, self.patch_scorer)
        return self.global_head.encode(global_features), self.patch_head.encode(pooled), weights

    def forward(
        self, pixel_values: torch.Tensor, valid_patch_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        方法作用：生成两个分支的嵌入、分类logits及Patch权重。
        输入参数：图像[B,3,S,S]；有效掩码[B,N]。
        返回值：global/patch嵌入[B,D]、两个logits[B,C]、weights[B,N]。
        """
        global_embedding, patch_embedding, weights = self.encode_branches(
            pixel_values, valid_patch_mask
        )
        return (
            global_embedding,
            patch_embedding,
            self.global_head.classifier(global_embedding),
            self.patch_head.classifier(patch_embedding),
            weights,
        )

    def head_parameters(self) -> list[nn.Parameter]:
        """
        方法作用：返回两个检索头和Patch Scorer的全部可训练参数。
        输入参数：无。
        返回值：list[nn.Parameter]。
        """
        modules = (self.global_head, self.patch_head, self.patch_scorer)
        return [parameter for module in modules for parameter in module.parameters()]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """
        方法作用：导出不包含冻结ConvNeXt主干的轻量权重。
        输入参数：无。
        返回值：dict，CPU权重张量。
        """
        prefixes = ("global_head.", "patch_head.", "patch_scorer.")
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
        }


def _make_convnext_loaders(
    config: dict[str, Any], fold: dict[str, Any], index: CropIndex, seed: int
) -> tuple[Any, Any, Any, int, int]:
    """
    方法作用：构造ConvNeXt+P2B基础类训练/验证加载器。
    输入参数：config；fold；CropIndex；随机种子。
    返回值：train_loader、PK sampler、val_loader、训练数、验证数；图像[B,3,S,S]、mask[B,N]。
    """
    settings = config["convnext_p2b"]
    classes = list(fold["base_classes"])
    class_map = {name: index for index, name in enumerate(classes)}
    train_records = records_for_partition(
        index, fold, "base_train", exclude_tiny=bool(settings["exclude_tiny_train"])
    )
    val_records = records_for_partition(
        index, fold, "base_val", exclude_tiny=bool(settings["exclude_tiny_val"])
    )
    train_loader, sampler = build_train_loader(
        PatchCropDataset(
            train_records,
            PatchLetterboxTransform(
                int(settings["image_size"]), int(settings["patch_stride"]), train=True
            ),
            class_map,
        ),
        classes_per_batch=int(settings["classes_per_batch"]),
        instances_per_class=int(settings["instances_per_class"]),
        workers=int(settings["workers"]),
        seed=seed,
    )
    val_loader = build_eval_loader(
        PatchCropDataset(
            val_records,
            PatchLetterboxTransform(
                int(settings["image_size"]), int(settings["patch_stride"]), train=False
            ),
            class_map,
        ),
        int(settings["eval_batch_size"]),
        int(settings["workers"]),
    )
    return train_loader, sampler, val_loader, len(train_records), len(val_records)


def _train_convnext_epoch(
    model: FrozenConvNextP2B,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
) -> dict[str, float]:
    """
    方法作用：以两个分支各自的ID+Triplet损失训练一轮ConvNeXt P2B轻量头。
    输入参数：model；批次图像[B,3,S,S]/mask[B,N]/label[B]；优化状态；device；配置。
    返回值：dict，平均损失、两个分类准确率及Patch注意力统计。
    """
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    amp = bool(settings["amp"]) and device.type == "cuda"
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            global_embedding, patch_embedding, global_logits, patch_logits, weights = model(
                images, masks
            )
            global_id = F.cross_entropy(
                global_logits, labels, label_smoothing=float(settings["label_smoothing"])
            )
            patch_id = F.cross_entropy(
                patch_logits, labels, label_smoothing=float(settings["label_smoothing"])
            )
            global_triplet = batch_hard_triplet_loss(
                global_embedding, labels, margin=float(settings["triplet_margin"])
            )
            patch_triplet = batch_hard_triplet_loss(
                patch_embedding, labels, margin=float(settings["triplet_margin"])
            )
            loss = 0.5 * (
                float(settings["id_weight"]) * global_id
                + float(settings["triplet_weight"]) * global_triplet
                + float(settings["id_weight"]) * patch_id
                + float(settings["triplet_weight"]) * patch_triplet
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.head_parameters(), float(settings["gradient_clip"]))
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(labels.numel())
        count += batch_size
        for key, value in (
            ("loss", loss),
            ("global_id_loss", global_id),
            ("patch_id_loss", patch_id),
            ("global_triplet_loss", global_triplet),
            ("patch_triplet_loss", patch_triplet),
        ):
            totals[key] += float(value.detach()) * batch_size
        totals["global_accuracy"] += float((global_logits.argmax(1) == labels).sum())
        totals["patch_accuracy"] += float((patch_logits.argmax(1) == labels).sum())
        for key, value in _weight_batch_statistics(weights.detach(), masks).items():
            totals[key] += value * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.inference_mode()
def _evaluate_convnext_classifier(
    model: FrozenConvNextP2B, loader: Any, device: torch.device, amp: bool
) -> dict[str, float]:
    """
    方法作用：在基础类验证集评估两个轻量分类头。
    输入参数：model；批次图像[B,3,S,S]/mask[B,N]/label[B]；device；amp。
    返回值：dict，全局、Patch及平均准确率。
    """
    model.eval()
    total = global_correct = patch_correct = 0
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, _, global_logits, patch_logits, _ = model(images, masks)
        total += int(labels.numel())
        global_correct += int((global_logits.argmax(1) == labels).sum())
        patch_correct += int((patch_logits.argmax(1) == labels).sum())
    global_accuracy = global_correct / max(total, 1)
    patch_accuracy = patch_correct / max(total, 1)
    return {
        "global_accuracy": global_accuracy,
        "patch_accuracy": patch_accuracy,
        "mean_branch_accuracy": 0.5 * (global_accuracy + patch_accuracy),
    }


@torch.inference_mode()
def _encode_convnext_loader(
    model: FrozenConvNextP2B, loader: Any, device: torch.device, amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    方法作用：编码全局、Patch与50/50融合特征，并保留注意力和元数据。
    输入参数：model；M样本loader；device；amp。
    返回值：global[M,D]、patch[M,D]、fused[M,2D]、labels[M]、metadata、weights[M,N]、masks[M,N]。
    """
    model.eval()
    global_parts: list[torch.Tensor] = []
    patch_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            global_embedding, patch_embedding, weights = model.encode_branches(images, masks)
        global_parts.append(global_embedding.float().cpu())
        patch_parts.append(patch_embedding.float().cpu())
        label_parts.append(labels.cpu())
        weight_parts.append(weights.float().cpu())
        mask_parts.append(masks.cpu())
        for item_index in range(len(batch["sample_id"])):
            metadata.append(
                {
                    "sample_id": batch["sample_id"][item_index],
                    "source_image": batch["source_image"][item_index],
                    "class_name": batch["class_name"][item_index],
                    "tiny": bool(batch["tiny"][item_index]),
                    "crowded": bool(batch["crowded"][item_index]),
                    "clipped": bool(batch["clipped"][item_index]),
                }
            )
    global_features = torch.cat(global_parts).numpy().astype(np.float32, copy=False)
    patch_features = torch.cat(patch_parts).numpy().astype(np.float32, copy=False)
    fused = np.concatenate(
        (math.sqrt(0.5) * global_features, math.sqrt(0.5) * patch_features), axis=1
    )
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)
    return (
        global_features,
        patch_features,
        fused.astype(np.float32, copy=False),
        torch.cat(label_parts).numpy().astype(np.int64, copy=False),
        metadata,
        torch.cat(weight_parts).numpy().astype(np.float32, copy=False),
        torch.cat(mask_parts).numpy().astype(bool, copy=False),
    )


def _evaluate_convnext_p2b_fold(
    model: FrozenConvNextP2B,
    config: dict[str, Any],
    fold: dict[str, Any],
    index: CropIndex,
    device: torch.device,
    output_dir: Path,
    best_val: float,
) -> dict[str, Any]:
    """
    方法作用：在固定Gallery/Query上评测ConvNeXt P2B并保存逐图结果。
    输入参数：model；config；fold；CropIndex；device；输出目录；最佳Base Val准确率。
    返回值：dict，融合Rank-1/mAP、两个来源分支及注意力统计。
    """
    settings = config["convnext_p2b"]
    classes = list(fold["novel_classes"])
    class_map = {name: class_index for class_index, name in enumerate(classes)}
    transform = PatchLetterboxTransform(
        int(settings["image_size"]), int(settings["patch_stride"]), train=False
    )
    support_records = records_for_partition(
        index, fold, "novel_support", exclude_tiny=bool(settings["exclude_tiny_support"])
    )
    query_records = records_for_partition(index, fold, "novel_query")
    support_loader = build_eval_loader(
        PatchCropDataset(support_records, transform, class_map),
        int(settings["eval_batch_size"]),
        int(settings["workers"]),
    )
    query_loader = build_eval_loader(
        PatchCropDataset(query_records, transform, class_map),
        int(settings["eval_batch_size"]),
        int(settings["workers"]),
    )
    amp = bool(settings["amp"]) and device.type == "cuda"
    sg, sp, sf, support_labels, support_meta, support_weights, support_masks = (
        _encode_convnext_loader(model, support_loader, device, amp)
    )
    qg, qp, qf, query_labels, query_meta, query_weights, query_masks = (
        _encode_convnext_loader(model, query_loader, device, amp)
    )
    query_tiny = np.asarray([item["tiny"] for item in query_meta], dtype=bool)
    top_k = int(config["protocol"]["prototype_top_k"])
    global_metrics, _ = compute_retrieval_metrics(
        sg, support_labels, qg, query_labels, query_tiny, classes, top_k
    )
    patch_metrics, _ = compute_retrieval_metrics(
        sp, support_labels, qp, query_labels, query_tiny, classes, top_k
    )
    metrics, scores = compute_retrieval_metrics(
        sf, support_labels, qf, query_labels, query_tiny, classes, top_k
    )
    metrics.update(
        {
            "format": FORMAT_VERSION,
            "variant": "convnext_p2b",
            "representation": "trained_global_head_plus_dynamic_weighted_patch_equal_fusion",
            "visual_backbone": "convnext-tiny-frozen",
            "fold": int(fold["fold"]),
            "novel_classes": classes,
            "support_count": len(support_records),
            "query_count": len(query_records),
            "training_performed": True,
            "backbone_trainable_parameters": 0,
            "best_base_val_mean_branch_accuracy": best_val,
            "source_rank1": {
                "trained_global_head": global_metrics["rank1_all"],
                "trained_weighted_patch": patch_metrics["rank1_all"],
                "p2b_equal_fusion": metrics["rank1_all"],
            },
            "attention": {
                "support": _attention_summary(support_weights, support_masks),
                "query": _attention_summary(query_weights, query_masks),
            },
            "fixed_gallery_manifest_sha256": config["protocol"][
                "fixed_gallery_manifest_sha256"
            ],
            "input_mode": "letterbox",
            "image_size": int(settings["image_size"]),
        }
    )
    save_retrieval_outputs(
        output_dir,
        metrics,
        classes,
        scores,
        query_labels,
        query_meta,
        sf,
        support_labels,
        qf,
    )
    _write_json(
        output_dir / "gallery_samples.json",
        {
            "manifest_sha256": config["protocol"]["fixed_gallery_manifest_sha256"],
            "records": support_meta,
        },
    )
    return metrics


def run_convnext_p2b(
    config: dict[str, Any],
    run_root: Path,
    device_name: str,
    folds: Sequence[int] | None = None,
    output_name: str = "convnext_p2b",
) -> dict[str, Any]:
    """
    方法作用：五折训练冻结ConvNeXt的P2B轻量头并使用固定Gallery评测。
    输入参数：config；run_root；device_name；folds，默认按protocol.fold_count；output_name，输出子目录名。
    返回值：dict，五折汇总。
    """
    from transformers import ConvNextModel

    output_root = run_root / output_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite ConvNeXt P2B run: {output_root}")
    device = choose_device(device_name)
    settings = config["convnext_p2b"]
    index = CropIndex(config["paths"]["crop_root"])
    results: list[dict[str, Any]] = []
    selected_folds = list(folds) if folds is not None else list(
        range(1, int(config["protocol"]["fold_count"]) + 1)
    )
    for fold_number in selected_folds:
        seed = int(config.get("seed", 2026)) + fold_number
        seed_everything(seed)
        fold = load_fold(config["paths"]["splits_root"], fold_number)
        train_loader, sampler, val_loader, train_count, val_count = _make_convnext_loaders(
            config, fold, index, seed
        )
        backbone = ConvNextModel.from_pretrained(
            str(config["paths"]["convnext_model"]), local_files_only=True
        )
        model = FrozenConvNextP2B(
            backbone,
            len(fold["base_classes"]),
            int(settings["embedding_dim"]),
            float(settings["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=float(settings["head_lr"]),
            weight_decay=float(settings["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(settings["epochs"]),
            eta_min=float(settings["min_lr"]),
        )
        amp = bool(settings["amp"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        fold_root = output_root / f"fold_{fold_number:02d}"
        fold_root.mkdir(parents=True)
        history: list[dict[str, Any]] = []
        best_val = float("-inf")
        best_path = fold_root / "best.pt"
        for epoch in range(1, int(settings["epochs"]) + 1):
            sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = _train_convnext_epoch(
                model, train_loader, optimizer, scaler, device, settings
            )
            val_metrics = _evaluate_convnext_classifier(model, val_loader, device, amp)
            scheduler.step()
            row = {
                "epoch": epoch,
                "seconds": round(time.time() - started, 3),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "epoch": epoch,
                "base_classes": list(fold["base_classes"]),
                "best_accuracy": max(best_val, val_metrics["mean_branch_accuracy"]),
                "model_state": model.adapter_state_dict(),
            }
            torch.save(checkpoint, fold_root / "last.pt")
            if val_metrics["mean_branch_accuracy"] > best_val:
                best_val = float(val_metrics["mean_branch_accuracy"])
                checkpoint["best_accuracy"] = best_val
                torch.save(checkpoint, best_path)
            _write_json(fold_root / "history.json", history)
            print(
                f"convnext_p2b fold={fold_number} epoch={epoch:02d} "
                f"val={val_metrics['mean_branch_accuracy']:.4f}",
                flush=True,
            )
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Invalid ConvNeXt P2B checkpoint: {best_path}")
        model.load_state_dict(checkpoint["model_state"], strict=False)
        metrics = _evaluate_convnext_p2b_fold(
            model, config, fold, index, device, fold_root, best_val
        )
        _write_json(
            fold_root / "run_config.json",
            {
                "format": FORMAT_VERSION,
                "variant": "convnext_p2b",
                "fold": fold_number,
                "training_performed": True,
                "backbone_frozen": True,
                "train_samples": train_count,
                "val_samples": val_count,
                "trainable_head_parameters": sum(
                    parameter.numel() for parameter in model.head_parameters()
                ),
                "config": serializable_config(config),
                "checkpoint_sha256": _sha256(best_path),
            },
        )
        results.append(metrics)
        del model, backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = summarize_folds(output_root)
    summary.update(
        {
            "format": FORMAT_VERSION,
            "variant": "convnext_p2b",
            "training_performed": True,
            "backbone_frozen": True,
        }
    )
    _write_json(output_root / "summary.json", summary)
    return summary


def _copy_checkpoint(source: Path, destination: Path) -> dict[str, str]:
    """
    方法作用：复制旧训练权重到新固定Gallery评测目录并记录双端摘要。
    输入参数：source；destination。
    返回值：dict，路径及SHA-256。
    """
    if not source.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = _sha256(source)
    if _sha256(destination) != source_hash:
        raise RuntimeError(f"Checkpoint copy hash differs: {destination}")
    return {"source": str(source), "destination": str(destination), "sha256": source_hash}


def run_dino_p2b_fixed(config: dict[str, Any], run_root: Path, device_name: str) -> dict[str, Any]:
    """
    方法作用：复用既有DINO B2/P2A权重，用当前固定Gallery重编码并执行P2B融合。
    输入参数：config；run_root；device_name。
    返回值：dict，DINO P2B五折汇总。
    """
    component_root = run_root / "dino_p2b_components"
    output_root = run_root / "dino_p2b"
    if component_root.exists() or output_root.exists():
        raise FileExistsError(f"Refusing to overwrite DINO P2B fixed-Gallery run: {run_root}")
    from .config import load_config

    b2_config = load_config(
        Path(__file__).resolve().parents[1] / "configs/dino_letterbox_text_anchor.json"
    )
    b2_config["data"]["image_size"] = int(config["dino_p2b"]["image_size"])
    b2_config["data"]["resize_short_edge"] = int(config["dino_p2b"]["resize_short_edge"])
    b2_config["data"]["eval_batch_size"] = int(config["dino_p2b"]["eval_batch_size"])
    b2_config["data"]["workers"] = int(config["dino_p2b"]["workers"])
    b2_config["paths"]["output_root"] = component_root / "b2"
    copied: list[dict[str, str]] = []
    for fold in range(1, 6):
        copied.append(
            _copy_checkpoint(
                config["paths"]["dino_b2_checkpoint_root"] / f"fold_{fold:02d}/best.pt",
                b2_config["paths"]["output_root"] / f"dino/b2/fold_{fold:02d}/best.pt",
            )
        )
    run_dino_variant(b2_config, "b2", [1, 2, 3, 4, 5], device_name, eval_only=True)

    p2a_config = load_config(
        Path(__file__).resolve().parents[1] / "configs/dino_patch_p2a_weighted_letterbox_336.json"
    )
    p2a_config["paths"]["output_root"] = component_root / "p2a"
    p2a_config["paths"]["p0_features_root"] = b2_config["paths"]["output_root"] / "dino/b2"
    p2a_config["diagnostics"] = {
        "compare_with_p0": True,
        "save_query_attention": True,
        "heatmaps_per_group": 0,
    }
    for fold in range(1, 6):
        copied.append(
            _copy_checkpoint(
                config["paths"]["dino_p2a_checkpoint_root"] / f"fold_{fold:02d}/best.pt",
                p2a_config["paths"]["output_root"] / f"fold_{fold:02d}/best.pt",
            )
        )
    run_p2a(p2a_config, [1, 2, 3, 4, 5], device_name, eval_only=True)

    p2b_config = load_config(
        Path(__file__).resolve().parents[1] / "configs/dino_patch_p2b_equal_fusion_336.json"
    )
    p2b_config["paths"]["p0_features_root"] = b2_config["paths"]["output_root"] / "dino/b2"
    p2b_config["paths"]["p2a_features_root"] = p2a_config["paths"]["output_root"]
    p2b_config["paths"]["output_root"] = output_root
    result = run_p2b(p2b_config, [1, 2, 3, 4, 5], device_name)
    _write_json(
        component_root / "checkpoint_provenance.json",
        {
            "format": FORMAT_VERSION,
            "training_performed_in_this_run": False,
            "fixed_gallery_manifest_sha256": config["protocol"][
                "fixed_gallery_manifest_sha256"
            ],
            "copied_checkpoints": copied,
        },
    )
    return result["summary"]


def run_raw_backbone(
    config: dict[str, Any], run_root: Path, backbone: str, device_name: str
) -> dict[str, Any]:
    """
    方法作用：使用现有冻结评测器运行原始DINO或原始ConvNeXt。
    输入参数：四路config；run_root；backbone；device_name。
    返回值：dict，五折汇总。
    """
    base_path = Path(__file__).resolve().parents[1] / "configs/frozen_dino_convnext_p2b_comparison.json"
    frozen = load_comparison_config(base_path)
    frozen["paths"]["crop_root"] = config["paths"]["crop_root"]
    frozen["paths"]["splits_root"] = config["paths"]["splits_root"]
    frozen["paths"]["dino_model"] = config["paths"]["dino_model"]
    frozen["paths"]["convnext_model"] = config["paths"]["convnext_model"]
    frozen["data"].update(deepcopy(config["raw"]))
    frozen["retrieval"]["prototype_top_k"] = int(config["protocol"]["prototype_top_k"])
    return run_frozen_backbone(backbone, frozen, [1, 2, 3, 4, 5], device_name, run_root)


def _method_summary(root: Path) -> dict[str, Any]:
    """
    方法作用：从五折metrics和predictions计算Macro/Micro Rank-1。
    输入参数：root，含fold_01至fold_05。
    返回值：dict，逐折、宏平均、微平均和逐类正确数。
    """
    fold_rank1: list[float] = []
    correct = total = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    protocol: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_root = root / f"fold_{fold:02d}"
        metrics = json.loads((fold_root / "metrics.json").read_text(encoding="utf-8"))
        predictions = json.loads((fold_root / "predictions.json").read_text(encoding="utf-8"))
        fold_rank1.append(float(metrics["rank1_all"]))
        correct += sum(bool(item["correct"]) for item in predictions)
        total += len(predictions)
        for item in predictions:
            target = per_class[str(item["true_class"])]
            target[1] += 1
            target[0] += int(bool(item["correct"]))
        protocol.append(
            {
                "fold": fold,
                "novel_classes": metrics["novel_classes"],
                "support_count": int(metrics.get("support_count", 20)),
                "query_count": len(predictions),
            }
        )
    return {
        "fold_rank1": fold_rank1,
        "macro_rank1": float(np.mean(fold_rank1)),
        "std_rank1": float(np.std(fold_rank1)),
        "micro_rank1": correct / max(total, 1),
        "correct_query_count": correct,
        "query_count": total,
        "per_class_rank1": {
            name: values[0] / values[1] for name, values in sorted(per_class.items())
        },
        "protocol": protocol,
    }


def summarize_fourway(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """
    方法作用：验证四方法共享协议并生成统一JSON和Markdown结果。
    输入参数：config；run_root。
    返回值：dict，四方法完整结果。
    """
    roots = {
        "dino_raw": run_root / "dino_raw",
        "dino_p2b": run_root / "dino_p2b",
        "convnext_raw": run_root / "convnext_raw",
        "convnext_p2b": run_root / "convnext_p2b",
    }
    methods = {name: _method_summary(root) for name, root in roots.items()}
    reference = methods["dino_raw"]["protocol"]
    for name, result in methods.items():
        if result["protocol"] != reference:
            raise ValueError(f"Four-way protocol differs for {name}")
    gallery = validate_fixed_gallery(config)
    payload = {
        "format": FORMAT_VERSION,
        "experiment": config["name"],
        "fixed_gallery": gallery,
        "methods": methods,
        "comparisons": {
            "dino_p2b_minus_dino_raw_macro": methods["dino_p2b"]["macro_rank1"]
            - methods["dino_raw"]["macro_rank1"],
            "convnext_p2b_minus_convnext_raw_macro": methods["convnext_p2b"]["macro_rank1"]
            - methods["convnext_raw"]["macro_rank1"],
        },
    }
    _write_json(run_root / "summary.json", payload)
    labels = {
        "dino_raw": "原始 DINOv2-Small",
        "dino_p2b": "DINOv2-Small + P2B",
        "convnext_raw": "原始 ConvNeXt-Tiny",
        "convnext_p2b": "冻结 ConvNeXt-Tiny + P2B轻量头",
    }
    lines = [
        "# 固定Gallery四路细粒度检索结果 V1.00",
        "",
        f"固定Gallery清单SHA-256：`{gallery['manifest_sha256']}`。每折两类、每类10张Gallery；全部方法共享相同Query。",
        "",
        "| 方法 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Macro Rank-1 | Micro Rank-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("dino_raw", "dino_p2b", "convnext_raw", "convnext_p2b"):
        row = methods[name]
        folds = " | ".join(f"{value:.2%}" for value in row["fold_rank1"])
        lines.append(
            f"| {labels[name]} | {folds} | {row['macro_rank1']:.2%} | {row['micro_rank1']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## P2B增量",
            "",
            f"- DINO + P2B 相对原始DINO：{payload['comparisons']['dino_p2b_minus_dino_raw_macro']:+.2%} Macro Rank-1。",
            f"- ConvNeXt + P2B 相对原始ConvNeXt：{payload['comparisons']['convnext_p2b_minus_convnext_raw_macro']:+.2%} Macro Rank-1。",
            "",
            "说明：原始模型完全零训练；DINO+P2B复用既有B2/P2A权重；ConvNeXt+P2B冻结原始主干，仅在每折Base Train上训练Projection、BNNeck、ID头和Patch Scorer。",
            "",
        ]
    )
    (run_root / "FOUR_WAY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：构建固定Gallery四路实验分阶段命令行。
    输入参数：无。
    返回值：ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="Fixed-Gallery DINO/ConvNeXt four-way retrieval")
    parser.add_argument(
        "stage",
        choices=("validate", "raw", "dino-p2b", "convnext-p2b", "summarize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("dino_raw", "convnext_raw"), default="dino_raw")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    方法作用：验证协议、运行四个分支或汇总最终结果。
    输入参数：argv；None时读取系统命令行。
    返回值：int，成功为0。
    """
    args = build_parser().parse_args(argv)
    config = load_fourway_config(args.config)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        result = validate_fixed_gallery(config)
        _write_json(run_root / "fixed_gallery_protocol.json", result)
    elif args.stage == "raw":
        result = run_raw_backbone(config, run_root, args.backbone, args.device)
    elif args.stage == "dino-p2b":
        result = run_dino_p2b_fixed(config, run_root, args.device)
    elif args.stage == "convnext-p2b":
        result = run_convnext_p2b(config, run_root, args.device)
    else:
        result = summarize_fourway(config, run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
