from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .config import serializable_config
from .data import (
    CropIndex,
    CropRecord,
    build_eval_loader,
    build_train_loader,
    load_fold,
    records_for_partition,
)
from .dino_engine import DINO_MEAN, DINO_MEAN_RGB, DINO_STD, _dino_backbone_name
from .dino_models import load_local_dino
from .dino_patch_models import DinoMaskedPatchAverageModel
from .engine import _amp_enabled, choose_device, seed_everything
from .losses import batch_hard_triplet_loss
from .metrics import compute_retrieval_metrics, save_retrieval_outputs, summarize_folds


P1A_VARIANT = "p1a_patch_average_only"


class PatchLetterboxTransform:
    """
    方法作用：
        保持宽高比缩放并填充图像，同时生成与 DINO Patch 网格对齐的
        有效内容掩码。
    输入参数：
        构造参数见 __init__；调用时输入单张 PIL RGB 图像 [H, W, 3]。
    返回值：
        PatchLetterboxTransform：可返回图像张量 [3, S, S] 与掩码 [N] 的变换器。
    """

    def __init__(self, image_size: int, patch_size: int, train: bool) -> None:
        """
        方法作用：
            初始化输出尺寸、Patch 尺寸和训练期图像增强。
        输入参数：
            image_size (int)：正方形输出边长 S，且 S 必须整除 patch_size。
            patch_size (int)：DINO 单个 Patch 的边长 P。
            train (bool)：是否启用翻转、颜色扰动和模糊增强。
        返回值：
            None：完成变换器初始化。
        """
        if image_size < 1 or patch_size < 1:
            raise ValueError("image_size and patch_size must be positive")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.train_mode = bool(train)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.04,
        )
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        """
        方法作用：
            把原图等比缩放到 S×S 画布，并按 Patch 中心是否落在原图内容区生成掩码。
        输入参数：
            image (PIL.Image.Image)：单张 RGB 图像，空间尺寸为 [H, W]。
        返回值：
            tuple[PIL.Image.Image, torch.Tensor]：Letterbox 图像 [S, S, 3]
            和布尔掩码 [N]，其中 N=(S/P)^2。
        """
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Cannot letterbox an empty image: {image.size}")
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = min(self.image_size, max(1, round(width * scale)))
        resized_height = min(self.image_size, max(1, round(height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        right = left + resized_width
        bottom = top + resized_height
        canvas = Image.new("RGB", (self.image_size, self.image_size), DINO_MEAN_RGB)
        canvas.paste(resized, (left, top))

        grid = self.image_size // self.patch_size
        # A boundary token is valid when its patch center belongs to the
        # resized content rectangle. Fully padded tokens are always excluded.
        centers = (
            torch.arange(grid, dtype=torch.float32) * self.patch_size
            + self.patch_size / 2
        )
        valid_x = (centers >= left) & (centers < right)
        valid_y = (centers >= top) & (centers < bottom)
        mask = (valid_y[:, None] & valid_x[None, :]).reshape(-1)
        if not bool(mask.any()):
            raise ValueError("Letterbox geometry produced an empty patch mask")
        return canvas, mask

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        """
        方法作用：
            执行 Letterbox、可选增强、张量化与 DINO 标准化。
        输入参数：
            image (PIL.Image.Image)：单张 RGB 图像 [H, W, 3]。
        返回值：
            dict[str, torch.Tensor]：image 为 [3, S, S]，valid_patch_mask 为 [N]。
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"PatchLetterboxTransform expects PIL.Image, got {type(image)}")
        image, mask = self._letterbox(image)
        if self.train_mode:
            if random.random() < 0.5:
                image = TF.hflip(image)
            image = self.color_jitter(image)
            if random.random() < 0.15:
                image = self.gaussian_blur(image)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, DINO_MEAN, DINO_STD)
        return {"image": tensor, "valid_patch_mask": mask}


class PatchCropDataset(Dataset):
    """
    方法作用：
        读取目标裁剪图，并同时返回训练标签、样本元数据和有效 Patch 掩码。
    输入参数：
        构造参数见 __init__；索引读取输入为整数样本位置。
    返回值：
        PatchCropDataset：单样本 image [3,S,S]、mask [N] 的数据集实例。
    """

    def __init__(
        self,
        records: Sequence[CropRecord],
        transform: PatchLetterboxTransform,
        class_to_local_id: dict[str, int],
    ) -> None:
        """
        方法作用：
            保存裁剪记录、图像变换和全局类名到当前折局部标签的映射。
        输入参数：
            records (Sequence[CropRecord])：长度为 M 的样本记录序列。
            transform (PatchLetterboxTransform)：图像与 Patch 掩码变换器。
            class_to_local_id (dict[str,int])：类名到 [0,C-1] 标签的映射。
        返回值：
            None：完成数据集初始化。
        """
        self.records = list(records)
        self.transform = transform
        self.class_to_local_id = dict(class_to_local_id)
        missing = {record.class_name for record in self.records} - set(
            self.class_to_local_id
        )
        if missing:
            raise ValueError(f"Dataset class map is missing: {sorted(missing)}")

    def __len__(self) -> int:
        """
        方法作用：
            返回当前数据集的样本数量。
        输入参数：
            无。
        返回值：
            int：裁剪记录数量 M。
        """
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        方法作用：
            读取并变换指定裁剪图，组合标签和可追踪元数据。
        输入参数：
            index (int)：样本下标，范围 [0,M-1]。
        返回值：
            dict[str,Any]：image [3,S,S]、valid_patch_mask [N]、标量 label
            及 sample_id/tiny/crowded/clipped 等元数据。
        """
        record = self.records[index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            transformed = self.transform(image)
        return {
            **transformed,
            "label": self.class_to_local_id[record.class_name],
            "global_class_id": record.class_id,
            "class_name": record.class_name,
            "sample_id": record.sample_id,
            "source_image": record.source_image,
            "tiny": record.tiny,
            "crowded": record.crowded,
            "clipped": record.clipped,
        }


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        以 UTF-8、缩进格式写入 JSON 结果，并创建父目录。
    输入参数：
        path (Path)：目标 JSON 路径。
        payload (Any)：可 JSON 序列化的数据对象。
    返回值：
        None：文件写入完成后无返回数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _output_dir(config: dict[str, Any], fold_number: int) -> Path:
    """
    方法作用：
        计算 P1a 指定折的输出目录。
    输入参数：
        config (dict[str,Any])：包含 paths.output_root 的实验配置。
        fold_number (int)：折号。
    返回值：
        Path：形如 output_root/fold_XX 的路径。
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
        根据当前折构造 P1a 基础类训练/验证 DataLoader 和 PK 采样器。
    输入参数：
        config (dict[str,Any])：数据、模型和 P1a 超参数配置。
        fold (dict[str,Any])：当前折的基础类与分区定义。
        crop_index (CropIndex)：裁剪样本索引。
        seed (int)：PK 采样随机种子。
    返回值：
        tuple：train_loader、sampler、val_loader、训练样本数 Mtr、验证样本数 Mval；
        每个批次 image [B,3,S,S]、mask [B,N]、label [B]。
    """
    settings = config["p1a"]
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


def _batch_to_device(batch: dict[str, Any], device: torch.device):
    """
    方法作用：
        从 DataLoader 批次中取出主张量并搬到计算设备。
    输入参数：
        batch (dict[str,Any])：包含 image [B,3,S,S]、mask [B,N]、label [B]。
        device (torch.device)：目标 CPU/CUDA 设备。
    返回值：
        tuple[torch.Tensor,...]：图像 [B,3,S,S]、布尔掩码 [B,N]、长整型标签 [B]。
    """
    return (
        batch["image"].to(device, non_blocking=True),
        batch["valid_patch_mask"].bool().to(device, non_blocking=True),
        batch["label"].long().to(device, non_blocking=True),
    )


def train_p1a_one_epoch(
    model: DinoMaskedPatchAverageModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
    amp: bool,
) -> dict[str, float]:
    """
    方法作用：
        运行一个 P1a 训练 Epoch，联合优化 ID Loss 与 Batch-Hard Triplet Loss。
    输入参数：
        model：P1a 模型；loader：批次 image [B,3,S,S]、mask [B,N]、label [B]；
        optimizer：优化器；scaler：AMP 梯度缩放器；device：计算设备；
        settings：损失权重等 P1a 配置；amp：是否使用混合精度。
    返回值：
        dict[str,float]：按样本平均的 loss、id_loss、triplet_loss 和 accuracy。
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
            embeddings, logits = model(images, masks)
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
    return {
        "loss": totals["loss"] / max(1, sample_count),
        "id_loss": totals["id_loss"] / max(1, sample_count),
        "triplet_loss": totals["triplet_loss"] / max(1, sample_count),
        "accuracy": totals["accuracy"] / max(1, sample_count),
    }


@torch.inference_mode()
def evaluate_p1a_classifier(
    model: DinoMaskedPatchAverageModel,
    loader,
    device: torch.device,
    amp: bool,
) -> dict[str, float | None]:
    """
    方法作用：
        在基础类验证集上计算 P1a ID 分类损失以及全部/core/tiny 准确率。
    输入参数：
        model：P1a 模型；loader：验证批次 image [B,3,S,S]、mask [B,N]、label [B]；
        device：计算设备；amp：是否启用混合精度。
    返回值：
        dict[str,float|None]：验证损失与分组准确率；无对应分组时为 None。
    """
    model.eval()
    total_loss = 0.0
    total = correct = core_total = core_correct = tiny_total = tiny_correct = 0
    for batch in loader:
        images, masks, labels = _batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            _, logits = model(images, masks)
            loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        tiny = batch["tiny"].bool().to(device)
        core = ~tiny
        total_loss += float(loss) * labels.numel()
        total += labels.numel()
        correct += int((predictions == labels).sum())
        core_total += int(core.sum())
        core_correct += int(((predictions == labels) & core).sum())
        tiny_total += int(tiny.sum())
        tiny_correct += int(((predictions == labels) & tiny).sum())
    return {
        "loss": total_loss / max(1, total),
        "accuracy_all": correct / max(1, total),
        "accuracy_core": core_correct / core_total if core_total else None,
        "accuracy_tiny": tiny_correct / tiny_total if tiny_total else None,
    }


@torch.inference_mode()
def encode_patch_loader(
    model: DinoMaskedPatchAverageModel,
    loader,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    方法作用：
        批量编码整个 Patch 数据集并汇总标签与样本元数据。
    输入参数：
        model：P1a 模型；loader：N 个样本的批量加载器；device：计算设备；
        amp：是否启用混合精度。批次主数据为 image [B,3,S,S]、mask [B,N]。
    返回值：
        tuple[np.ndarray,...]：归一化特征 [M,D]、标签 [M]、长度 M 的元数据列表。
    """
    model.eval()
    features = []
    labels = []
    metadata: list[dict[str, Any]] = []
    for batch in loader:
        images, masks, batch_labels = _batch_to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            encoded = model.encode(images, masks)
        features.append(encoded.float().cpu())
        labels.append(batch_labels.cpu())
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
    )


def evaluate_p1a_novel_retrieval(
    model: DinoMaskedPatchAverageModel,
    crop_index: CropIndex,
    fold: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """
    方法作用：
        使用当前折未见类 Support/Query 评测 P1a Patch 检索并保存预测与特征。
    输入参数：
        model：已训练 P1a 模型；crop_index：裁剪索引；fold：当前折定义；
        config：评测配置；device：计算设备；output_dir：结果目录。
        主数据形成 Support [Ns,D]、Query [Nq,D] 和类别得分 [Nq,C]。
    返回值：
        dict[str,Any]：Rank-1、mAP、Recall@K 及实验元信息。
    """
    novel_classes = list(fold["novel_classes"])
    class_to_local = {name: index for index, name in enumerate(novel_classes)}
    transform = PatchLetterboxTransform(
        int(config["data"]["image_size"]),
        int(config["model"].get("patch_size", 14)),
        train=False,
    )
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
    support_features, support_labels, _ = encode_patch_loader(
        model, support_loader, device, amp
    )
    query_features, query_labels, query_metadata = encode_patch_loader(
        model, query_loader, device, amp
    )
    query_tiny = np.asarray(
        [item["tiny"] for item in query_metadata], dtype=bool
    )
    metrics, class_scores = compute_retrieval_metrics(
        support_features=support_features,
        support_labels=support_labels,
        query_features=query_features,
        query_labels=query_labels,
        query_tiny=query_tiny,
        class_names=novel_classes,
        prototype_top_k=int(config["retrieval"]["prototype_top_k"]),
    )
    metrics.update(
        {
            "variant": P1A_VARIANT,
            "representation": "masked_patch_average_only",
            "visual_backbone": _dino_backbone_name(config),
            "fold": int(fold["fold"]),
            "novel_classes": novel_classes,
        }
    )
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
    return metrics


def _checkpoint_payload(
    epoch: int,
    model: DinoMaskedPatchAverageModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    base_classes: list[str],
    best_accuracy: float,
) -> dict[str, Any]:
    """
    方法作用：
        组装可恢复的 P1a 训练检查点。
    输入参数：
        epoch：当前轮次；model：P1a 模型；optimizer/scheduler/scaler：训练状态；
        base_classes：长度 C 的基础类名；best_accuracy：最佳验证准确率。
    返回值：
        dict[str,Any]：模型适配器、优化器、调度器和 AMP 状态字典。
    """
    return {
        "format": "annotation_pad_lite_dino_patch_p1a_v1",
        "variant": P1A_VARIANT,
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
    model: DinoMaskedPatchAverageModel,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict[str, Any]:
    """
    方法作用：
        从磁盘加载 P1a 检查点，并可选恢复完整训练状态。
    输入参数：
        path (Path)：检查点路径；model：待恢复模型；optimizer、scheduler、scaler：
        可选训练组件。
    返回值：
        dict[str,Any]：反序列化后的检查点内容。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "annotation_pad_lite_dino_patch_p1a_v1":
        raise ValueError(f"Unsupported P1a checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def run_p1a_fold(
    config: dict[str, Any],
    fold_number: int,
    device_name: str,
    resume: bool,
    eval_only: bool,
) -> dict[str, Any]:
    """
    方法作用：
        从头或断点运行单折 P1a 训练、基础类选模和未见类检索评测。
    输入参数：
        config：完整实验配置；fold_number：折号；device_name：设备名；
        resume：是否恢复 last.pt；eval_only：是否只加载 best.pt 评测。
    返回值：
        dict[str,Any]：该折未见类检索指标和最佳基础类验证准确率。
    """
    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + fold_number
    seed_everything(seed)
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    crop_index = CropIndex(config["paths"]["crop_root"])
    settings = config["p1a"]
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
    model = DinoMaskedPatchAverageModel(
        dino,
        num_classes=len(fold["base_classes"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.head_parameters(), "lr": float(settings["head_lr"]), "name": "patch_head"}],
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
    run_config = {
        "format": "annotation_pad_lite_dino_patch_fold_v1",
        "variant": P1A_VARIANT,
        "representation": "masked_patch_average_only",
        "fold": fold_number,
        "base_classes": list(fold["base_classes"]),
        "novel_classes": list(fold["novel_classes"]),
        "device": str(device),
        "config": serializable_config(config),
        "train_samples": train_count,
        "val_samples": val_count,
        "parameter_groups": {"patch_head": head_count, "dino_backbone": 0},
        "trainable_parameters": head_count,
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
            raise FileNotFoundError(f"Missing P1a checkpoint: {best_checkpoint}")
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
            train_metrics = train_p1a_one_epoch(
                model, train_loader, optimizer, scaler, device, settings, amp
            )
            val_metrics = evaluate_p1a_classifier(model, val_loader, device, amp)
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

    metrics = evaluate_p1a_novel_retrieval(
        model, crop_index, fold, config, device, output_dir
    )
    metrics["best_base_val_accuracy"] = best_accuracy
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_p1a(
    config: dict[str, Any],
    folds: list[int],
    device_name: str,
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        按顺序运行指定折的 P1a，并生成跨折汇总。
    输入参数：
        config：P1a 配置；folds：折号列表；device_name：设备；
        resume：是否恢复；eval_only：是否仅评测。
    返回值：
        dict[str,Any]：逐折 results 和均值/标准差 summary。
    """
    if config["data"].get("input_mode") != "letterbox":
        raise ValueError("P1a currently requires data.input_mode=letterbox")
    if int(config["model"].get("unfreeze_last_blocks", 0)) != 0:
        raise ValueError("P1a requires a frozen DINO backbone")
    results = []
    for fold in folds:
        metrics = run_p1a_fold(config, fold, device_name, resume, eval_only)
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    root = config["paths"]["output_root"]
    summary = summarize_folds(root)
    summary.update(
        {
            "variant": P1A_VARIANT,
            "representation": "masked_patch_average_only",
            "visual_backbone": _dino_backbone_name(config),
        }
    )
    _write_json(root / "summary.json", summary)
    return {"results": results, "summary": summary}
