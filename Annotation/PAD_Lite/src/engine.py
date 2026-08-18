from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import serializable_config
from .data import (
    CropDataset,
    CropIndex,
    build_eval_loader,
    build_eval_transform,
    build_train_loader,
    build_train_transform,
    load_fold,
    records_for_partition,
)
from .losses import batch_hard_triplet_loss, text_anchor_contrastive_loss
from .metrics import (
    compute_retrieval_metrics,
    encode_loader,
    save_retrieval_outputs,
    summarize_folds,
)
from .models import (
    FrozenClipEncoder,
    RetrievalModel,
    TextAnchorPrompt,
    load_local_clip,
    trainable_parameter_count,
)


def choose_device(requested: str) -> torch.device:
    """
    方法作用：
        根据传入的设备名称选择模型运行设备，支持 auto、cpu 和 cuda。

    输入参数：
        requested (str):
            设备名称字符串，例如 "auto"、"cpu" 或 "cuda:0"。

    返回值：
        torch.device:
            解析后的 PyTorch 设备对象。
    """
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def seed_everything(seed: int) -> None:
    """
    方法作用：
        执行 seed_everything 对应的处理流程。
    
    输入参数：
        seed (int)：方法所需的 seed 参数。
    
    返回值：
        None：方法直接完成相应操作。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _amp_enabled(config: dict[str, Any], device: torch.device) -> bool:
    """
    方法作用：
        判断当前是否启用自动混合精度训练或推理，只有在 CUDA 设备上且配置允许时才开启。

    输入参数：
        config (dict[str, Any]):
            实验配置字典。
        device (torch.device):
            当前运行设备。

    返回值：
        bool:
            如果满足启用 AMP 的条件则返回 True，否则返回 False。
    """
    return bool(config["data"].get("amp", True)) and device.type == "cuda"


def _fold_output_root(config: dict[str, Any], variant: str, fold_number: int) -> Path:
    """
    方法作用：
        根据实验版本和折编号生成对应的输出目录路径。

    输入参数：
        config (dict[str, Any]):
            实验配置字典。
        variant (str):
            实验版本，如 b0/b1/b2。
        fold_number (int):
            当前折编号。

    返回值：
        Path:
            对应的输出目录路径对象。
    """
    return config["paths"]["output_root"] / variant / f"fold_{fold_number:02d}"


def _write_run_config(
    output_dir: Path,
    config: dict[str, Any],
    variant: str,
    fold: dict,
    device: torch.device,
) -> None:
    """
    方法作用：
        将当前实验的配置、折信息和设备信息写入输出目录，便于后续复现和排查问题。

    输入参数：
        output_dir (Path):
            要写入配置文件的输出目录。
        config (dict[str, Any]):
            实验配置字典。
        variant (str):
            当前实验版本。
        fold (dict):
            当前折的划分信息。
        device (torch.device):
            当前运行设备。

    返回值：
        None:
            直接将 run_config.json 写入磁盘。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "fold": fold["fold"],
        "base_classes": fold["base_classes"],
        "novel_classes": fold["novel_classes"],
        "device": str(device),
        "config": serializable_config(config),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_novel_retrieval(
    encoder,
    crop_index: CropIndex,
    fold: dict,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    variant: str,
) -> dict[str, Any]:
    """
    方法作用：
        使用训练好的编码器对新类别样本进行检索评估，
        生成支持集特征、查询集特征、检索指标和逐条预测结果。

    输入参数：
        encoder:
            将图像批次 [B, 3, H, W] 转换为归一化特征 [B, D] 的编码器。
        crop_index (CropIndex):
            裁剪样本索引对象。
        fold (dict):
            当前折的划分信息。
        config (dict[str, Any]):
            实验配置字典。
        device (torch.device):
            运行设备。
        output_dir (Path):
            结果写出目录。
        variant (str):
            当前实验版本，如 b0/b1/b2。

    返回值：
        dict[str, Any]:
            由支持特征 [Ns, D] 和查询特征 [Nq, D] 计算出的 Rank-1、mAP、
            样本数量等检索指标。
    """
    novel_classes = list(fold["novel_classes"])
    class_to_local = {name: index for index, name in enumerate(novel_classes)}
    eval_transform = build_eval_transform(int(config["data"]["image_size"]))
    support_records = records_for_partition(
        crop_index,
        fold,
        "novel_support",
        exclude_tiny=bool(config["data"].get("exclude_tiny_support", True)),
    )
    query_records = records_for_partition(
        crop_index,
        fold,
        "novel_query",
        exclude_tiny=False,
    )
    support_classes = {record.class_name for record in support_records}
    if support_classes != set(novel_classes):
        raise ValueError(
            f"Support loses a novel class after filtering: {support_classes}"
        )
    support_dataset = CropDataset(support_records, eval_transform, class_to_local)
    query_dataset = CropDataset(query_records, eval_transform, class_to_local)
    workers = int(config["data"]["workers"])
    batch_size = int(config["data"]["eval_batch_size"])
    support_loader = build_eval_loader(support_dataset, batch_size, workers)
    query_loader = build_eval_loader(query_dataset, batch_size, workers)

    support_features, support_labels, _ = encode_loader(
        encoder, support_loader, device, _amp_enabled(config, device)
    )
    query_features, query_labels, query_metadata = encode_loader(
        encoder, query_loader, device, _amp_enabled(config, device)
    )
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
    metrics.update(
        {
            "variant": variant,
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


def run_b0_fold(
    config: dict[str, Any], fold_number: int, device_name: str
) -> dict[str, Any]:
    """
    方法作用：
        运行不需要训练的 B0 基线实验，直接用冻结 CLIP 提取特征并评估新类别检索效果。

    输入参数：
        config (dict[str, Any]):
            实验配置。
        fold_number (int):
            当前折编号。
        device_name (str):
            设备名称。

    返回值：
        dict[str, Any]:
            当前折的检索评估结果。
    """
    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + fold_number
    seed_everything(seed)
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    crop_index = CropIndex(config["paths"]["crop_root"])
    output_dir = _fold_output_root(config, "b0", fold_number)
    _write_run_config(output_dir, config, "b0", fold, device)
    clip_model, _ = load_local_clip(config["paths"]["clip_model"])
    encoder = FrozenClipEncoder(clip_model).to(device).eval()
    metrics = evaluate_novel_retrieval(
        encoder, crop_index, fold, config, device, output_dir, "b0"
    )
    return metrics


@torch.inference_mode()
def evaluate_base_classifier(
    model: RetrievalModel,
    loader,
    device: torch.device,
    amp: bool,
) -> dict[str, float | None]:
    """
    方法作用：
        在基础类别验证集上评估当前模型的分类性能，
        包括整体准确率、核心样本准确率和微小目标准确率。

    输入参数：
        model (RetrievalModel):
            当前待评估的检索模型。
        loader:
            验证集 DataLoader，每批提供 image [B, 3, H, W]、label [B]
            和 tiny [B]。
        device (torch.device):
            运行设备。
        amp (bool):
            是否启用自动混合精度。

    返回值：
        dict[str, float | None]:
            包含整套验证数据的标量平均损失和不同分组准确率；模型前向产生
            embedding [B, D] 和 logits [B, C]。
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    core_total = 0
    core_correct = 0
    tiny_total = 0
    tiny_correct = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            _, logits = model(images)
            loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        tiny = batch["tiny"].bool().to(device)
        core = ~tiny
        total_loss += float(loss.item()) * labels.numel()
        total += labels.numel()
        correct += int((predictions == labels).sum().item())
        core_total += int(core.sum().item())
        core_correct += int(((predictions == labels) & core).sum().item())
        tiny_total += int(tiny.sum().item())
        tiny_correct += int(((predictions == labels) & tiny).sum().item())
    return {
        "loss": total_loss / max(1, total),
        "accuracy_all": correct / max(1, total),
        "accuracy_core": core_correct / core_total if core_total else None,
        "accuracy_tiny": tiny_correct / tiny_total if tiny_total else None,
    }


def train_one_epoch(
    model: RetrievalModel,
    prompt: TextAnchorPrompt | None,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
    amp: bool,
) -> dict[str, float]:
    """
    方法作用：
        在单轮训练中对模型进行一次前向传播、损失计算、反向传播和参数更新，
        同时统计训练损失和分类准确率。

    输入参数：
        model (RetrievalModel):
            当前训练的检索模型。
        prompt (TextAnchorPrompt | None):
            可选的文本锚点提示模块，B2 会用到。
        loader:
            训练集 DataLoader，每批提供 image [B, 3, H, W] 和 label [B]。
        optimizer (torch.optim.Optimizer):
            优化器对象。
        scaler (torch.amp.GradScaler):
            混合精度缩放器。
        device (torch.device):
            运行设备。
        settings (dict[str, Any]):
            当前实验版本的训练超参数字典。
        amp (bool):
            是否启用 AMP。

    返回值：
        dict[str, float]:
            包含整轮标量平均损失和准确率的统计字典；每批前向产生
            embedding [B, D]、logits [B, C]，B、D、C 分别为批量大小、
            嵌入维度和基础类别数。
    """
    model.train()
    if prompt is not None:
        prompt.train()
    totals: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            embeddings, logits = model(images)
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
            anchor_loss = embeddings.new_zeros(())
            if prompt is not None:
                anchors = prompt(model.clip)
                anchor_loss = text_anchor_contrastive_loss(
                    embeddings,
                    anchors,
                    labels,
                    temperature=float(settings["anchor_temperature"]),
                )
            total_loss = (
                float(settings["id_weight"]) * id_loss
                + float(settings["triplet_weight"]) * triplet_loss
                + float(settings.get("anchor_weight", 0.0)) * anchor_loss
            )
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ],
            max_norm=float(settings.get("gradient_clip", 5.0)),
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        sample_count += batch_size
        totals["loss"] += float(total_loss.detach().item()) * batch_size
        totals["id_loss"] += float(id_loss.detach().item()) * batch_size
        totals["triplet_loss"] += float(triplet_loss.detach().item()) * batch_size
        totals["anchor_loss"] += float(anchor_loss.detach().item()) * batch_size
        totals["accuracy"] += float((logits.argmax(dim=1) == labels).sum().item())
    return {
        "loss": totals["loss"] / max(1, sample_count),
        "id_loss": totals["id_loss"] / max(1, sample_count),
        "triplet_loss": totals["triplet_loss"] / max(1, sample_count),
        "anchor_loss": totals["anchor_loss"] / max(1, sample_count),
        "accuracy": totals["accuracy"] / max(1, sample_count),
    }


def _optimizer(
    model: RetrievalModel,
    prompt: TextAnchorPrompt | None,
    settings: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    方法作用：
        根据模型头部、主干网络和可学习文本提示分别设置不同学习率的优化器参数组。

    输入参数：
        model (RetrievalModel):
            当前模型实例。
        prompt (TextAnchorPrompt | None):
            可选的文本提示模块。
        settings (dict[str, Any]):
            包含学习率和权重衰减配置的字典。

    返回值：
        torch.optim.Optimizer:
            配置好参数组的 AdamW 优化器。
    """
    groups = [
        {
            "params": model.head_parameters(),
            "lr": float(settings["head_lr"]),
            "name": "head",
        }
    ]
    if model.backbone_parameters():
        groups.append(
            {
                "params": model.backbone_parameters(),
                "lr": float(settings["backbone_lr"]),
                "name": "backbone",
            }
        )
    if prompt is not None:
        groups.append(
            {
                "params": list(prompt.parameters()),
                "lr": float(settings["prompt_lr"]),
                "name": "prompt",
            }
        )
    return torch.optim.AdamW(
        groups,
        weight_decay=float(settings["weight_decay"]),
    )


def _checkpoint_payload(
    variant: str,
    epoch: int,
    model: RetrievalModel,
    prompt: TextAnchorPrompt | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    base_classes: list[str],
    best_accuracy: float,
) -> dict[str, Any]:
    """
    方法作用：
        将当前训练轮次的模型参数、优化器状态和训练元信息打包成检查点内容，
        便于恢复训练和保存最佳模型。

    输入参数：
        variant (str):
            当前实验版本。
        epoch (int):
            当前训练轮次。
        model (RetrievalModel):
            当前模型实例。
        prompt (TextAnchorPrompt | None):
            文本提示模块。
        optimizer (torch.optim.Optimizer):
            优化器。
        scheduler:
            学习率调度器。
        scaler:
            混合精度缩放器。
        base_classes (list[str]):
            基础类别名称列表。
        best_accuracy (float):
            当前已记录的最佳验证准确率。

    返回值：
        dict[str, Any]:
            可保存为检查点的状态字典。
    """
    return {
        "format": "annotation_pad_lite_adapter_v1",
        "variant": variant,
        "epoch": epoch,
        "base_classes": base_classes,
        "best_accuracy": best_accuracy,
        "model_state": model.adapter_state_dict(),
        "prompt_state": prompt.state_dict() if prompt is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }


def _load_checkpoint(
    path: Path,
    model: RetrievalModel,
    prompt: TextAnchorPrompt | None,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict[str, Any]:
    """
    方法作用：
        从检查点文件中恢复模型、优化器、调度器和混合精度缩放器状态。

    输入参数：
        path (Path):
            检查点文件路径。
        model (RetrievalModel):
            当前模型实例。
        prompt (TextAnchorPrompt | None):
            文本提示模块。
        optimizer:
            可选的优化器对象。
        scheduler:
            可选的学习率调度器对象。
        scaler:
            可选的混合精度缩放器对象。

    返回值：
        dict[str, Any]:
            加载后的检查点内容字典。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "annotation_pad_lite_adapter_v1":
        raise ValueError(f"Unsupported checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if prompt is not None:
        if checkpoint.get("prompt_state") is None:
            raise ValueError(f"Checkpoint has no text prompt state: {path}")
        prompt.load_state_dict(checkpoint["prompt_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def run_training_fold(
    config: dict[str, Any],
    variant: str,
    fold_number: int,
    device_name: str,
    resume: bool,
    eval_only: bool,
) -> dict[str, Any]:
    """
    方法作用：
        运行指定交叉验证折上的 B1 或 B2 模型训练与评估流程。

        函数会读取当前折的数据划分，构建训练集和验证集，
        加载本地 CLIP 模型，并创建图像检索模型。

        对于 B2 实验，还会额外创建用于文本锚点对比学习的
        TextAnchorPrompt 模块。

        在正常训练模式下，函数会逐轮训练模型，在基础类别验证集上
        评估分类准确率，并保存最后一次训练检查点 last.pt 和验证效果
        最佳的检查点 best.pt。

        函数也支持从 last.pt 恢复训练，或者跳过训练、直接加载
        best.pt 进行评估。

        训练结束后，函数会加载最佳模型，在新类别数据上执行检索评估，
        保存评估指标并返回结果。

    输入参数：
        config (dict[str, Any]):
            实验配置字典，包含数据集路径、模型路径、输出路径、
            图像尺寸、批次设置、训练轮数、学习率和损失函数参数等。

        variant (str):
            要运行的训练版本，只能是 "b1" 或 "b2"。

            "b1" 使用图像分类与度量学习方式训练检索模型；
            "b2" 在 B1 的基础上增加可学习的文本提示和
            文本锚点对比学习。

        fold_number (int):
            当前运行的交叉验证折编号，用于加载对应的数据划分，
            一般取值为 1～5。

        device_name (str):
            模型运行设备的名称，例如 "auto"、"cpu"、"cuda"
            或 "cuda:0"。"auto" 表示自动选择 CUDA 或 CPU。

        resume (bool):
            是否恢复之前未完成的训练。

            当值为 True 且当前输出目录中存在 last.pt 时，
            函数会恢复模型、优化器、学习率调度器和梯度缩放器的状态，
            并从下一轮继续训练。

        eval_only (bool):
            是否只执行评估。

            当值为 True 时，函数不会进行训练，而是直接加载当前折的
            best.pt 检查点并执行新类别检索评估。如果 best.pt 不存在，
            则抛出 FileNotFoundError。

    返回值：
        dict[str, Any]:
            返回当前交叉验证折的评估指标字典。

            返回内容主要包括：
            - variant：当前实验版本；
            - fold：当前交叉验证折编号；
            - novel_classes：当前折的新类别列表；
            - rank1_all：全部查询样本的 Rank-1 准确率；
            - rank1_core：普通尺寸样本的 Rank-1 准确率；
            - rank1_tiny：微小目标样本的 Rank-1 准确率；
            - class_map：按类别计算的平均精度均值；
            - retrieval_map：检索任务的平均精度均值；
            - best_base_val_accuracy：训练期间基础类别验证集上的
              最佳分类准确率。

            具体字段可能根据 evaluate_novel_retrieval 的实现有所扩展。
            评估结果同时会被保存到当前实验目录的 metrics.json 文件中。
    """
    if variant not in {"b1", "b2"}:
        raise ValueError(f"Training variant must be b1 or b2, got {variant}")
    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + fold_number
    seed_everything(seed)
    # 加载当前折的数据划分，构建训练集和验证集，并创建输出目录。
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    # 根据当前折编号和实验版本，生成输出目录路径，并创建目录。
    output_dir = _fold_output_root(config, variant, fold_number)
    # 将当前实验的配置、折编号、基础类别和设备信息写入 run_config.json 文件，便于后续复现。
    _write_run_config(output_dir, config, variant, fold, device)
    # 创建 CropIndex 实例，用于管理裁剪图像的索引和路径。
    crop_index = CropIndex(config["paths"]["crop_root"])
    # 加载本地 CLIP 模型和处理器，用于图像特征提取和文本处理。
    settings = config[variant]
    # 取出基础类别
    base_classes = list(fold["base_classes"])
    # 创建类别到本地索引的映射
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
    train_dataset = CropDataset(
        train_records,
        build_train_transform(int(config["data"]["image_size"])),
        class_to_local,
    )
    val_dataset = CropDataset(
        val_records,
        build_eval_transform(int(config["data"]["image_size"])),
        class_to_local,
    )
    train_loader, batch_sampler = build_train_loader(
        train_dataset,
        classes_per_batch=int(settings["classes_per_batch"]),
        instances_per_class=int(settings["instances_per_class"]),
        workers=int(config["data"]["workers"]),
        seed=seed,
    )
    val_loader = build_eval_loader(
        val_dataset,
        batch_size=int(config["data"]["eval_batch_size"]),
        workers=int(config["data"]["workers"]),
    )

    clip_model, processor = load_local_clip(config["paths"]["clip_model"])
    model = RetrievalModel(
        clip_model,
        num_classes=len(base_classes),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    )
    last_n_blocks = int(config["model"]["unfreeze_last_blocks"])
    model.configure_backbone(last_n_blocks, enabled=True)
    prompt = None
    if variant == "b2":
        if int(config["model"]["embedding_dim"]) != int(clip_model.config.projection_dim):
            raise ValueError(
                "B2 embedding_dim must equal CLIP projection_dim for text anchoring"
            )
        prompt_cfg = settings["prompt"]
        prompt = TextAnchorPrompt(
            clip_model,
            processor.tokenizer,
            num_classes=len(base_classes),
            context_tokens=int(prompt_cfg["context_tokens"]),
            prefix=str(prompt_cfg["prefix"]),
            suffix=str(prompt_cfg["suffix"]),
            init_std=float(prompt_cfg["init_std"]),
        )
        if any(name.lower() in prompt.template.lower() for name in base_classes):
            raise ValueError("A class name entered the B2 prompt template")

    model.to(device)
    if prompt is not None:
        prompt.to(device)
    warmup_epochs = int(settings["warmup_epochs"])
    if warmup_epochs > 0:
        model.set_backbone_enabled(False)
    optimizer = _optimizer(model, prompt, settings)
    epochs = int(settings["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(settings.get("min_lr", 1e-7)),
    )
    amp = _amp_enabled(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_accuracy = float("-inf")
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"

    if eval_only:
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Missing best checkpoint: {best_checkpoint}")
        checkpoint = _load_checkpoint(best_checkpoint, model, prompt)
        best_accuracy = float(checkpoint["best_accuracy"])
    else:
        if resume and last_checkpoint.is_file():
            checkpoint = _load_checkpoint(
                last_checkpoint, model, prompt, optimizer, scheduler, scaler
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_accuracy = float(checkpoint["best_accuracy"])
            history_path = output_dir / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))

        print(
            json.dumps(
                {
                    "variant": variant,
                    "fold": fold_number,
                    "device": str(device),
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "trainable_parameters": trainable_parameter_count(model)
                    + (trainable_parameter_count(prompt) if prompt is not None else 0),
                    "prompt_template": prompt.template if prompt is not None else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        for epoch in range(start_epoch, epochs + 1):
            backbone_enabled = epoch > warmup_epochs
            model.set_backbone_enabled(backbone_enabled)
            batch_sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = train_one_epoch(
                model,
                prompt,
                train_loader,
                optimizer,
                scaler,
                device,
                settings,
                amp,
            )
            val_metrics = evaluate_base_classifier(model, val_loader, device, amp)
            scheduler.step()
            selection_accuracy = (
                val_metrics["accuracy_core"]
                if val_metrics["accuracy_core"] is not None
                else val_metrics["accuracy_all"]
            )
            epoch_record = {
                "epoch": epoch,
                "backbone_enabled": backbone_enabled,
                "seconds": round(time.time() - started, 3),
                "learning_rates": {
                    group.get("name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_record)
            payload = _checkpoint_payload(
                variant,
                epoch,
                model,
                prompt,
                optimizer,
                scheduler,
                scaler,
                base_classes,
                max(best_accuracy, float(selection_accuracy)),
            )
            torch.save(payload, last_checkpoint)
            if float(selection_accuracy) > best_accuracy:
                best_accuracy = float(selection_accuracy)
                payload["best_accuracy"] = best_accuracy
                torch.save(payload, best_checkpoint)
            (output_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        _load_checkpoint(best_checkpoint, model, prompt)

    model.eval()
    metrics = evaluate_novel_retrieval(
        model, crop_index, fold, config, device, output_dir, variant
    )
    metrics["best_base_val_accuracy"] = best_accuracy
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_variant(
    config: dict[str, Any],
    variant: str,
    folds: list[int],
    device_name: str,
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        按照指定的实验版本和交叉验证折依次运行实验。

        当 variant 为 "b0" 时，直接运行不需要训练的 B0 基线实验；
        当 variant 为 "b1" 或 "b2" 时，运行对应版本的模型训练和评估。

        每一折运行完成后，会保存并打印该折的评估指标；
        所有指定折运行结束后，会汇总输出目录中已有的各折结果，
        计算平均值和标准差，并生成整体实验摘要。

    输入参数：
        config (dict[str, Any]):
            实验配置字典，包含数据集路径、模型路径、输出目录、
            训练参数和检索评估参数等配置。

        variant (str):
            要运行的实验版本，可取 "b0"、"b1" 或 "b2"。
            "b0" 直接使用冻结的模型进行特征提取和检索评估；
            "b1"、"b2" 会先进行模型训练或加载检查点，再进行评估。

        folds (list[int]):
            需要运行的交叉验证折编号列表，例如 [1, 2, 3, 4, 5]。
            函数会按照列表顺序逐折运行实验。

        device_name (str):
            模型运行设备的名称，例如 "auto"、"cpu"、"cuda"
            或 "cuda:0"。

        resume (bool):
            是否从已有的 last.pt 检查点恢复训练。
            默认为 False，主要适用于 B1 和 B2。

        eval_only (bool):
            是否跳过训练，只加载已有的 best.pt 检查点进行评估。
            默认为 False，主要适用于 B1 和 B2。

    返回值：
        dict[str, Any]:
            返回包含各折实验结果和整体汇总结果的字典，结构如下：

            {
                "folds": [
                    每一折的评估指标字典
                ],
                "summary": {
                    "completed_folds": 已完成并生成指标文件的折数,
                    "aggregate": {
                        指标名称: {
                            "mean": 各折指标的平均值,
                            "std": 各折指标的标准差,
                            "fold_values": 各折的原始指标值
                        }
                    }
                }
            }

            汇总的主要指标包括 rank1_all、rank1_core、
            rank1_tiny、class_map 和 retrieval_map。
    
    """
    results = []
    for fold_number in folds:
        if variant == "b0":
            metrics = run_b0_fold(config, fold_number, device_name)
        else:
            metrics = run_training_fold(
                config,
                variant,
                fold_number,
                device_name,
                resume,
                eval_only,
            )
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize_folds(config["paths"]["output_root"] / variant)
    return {"folds": results, "summary": summary}
