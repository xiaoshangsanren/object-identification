from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import serializable_config
from .data import (
    CropDataset,
    CropIndex,
    build_eval_loader,
    build_train_loader,
    load_fold,
    records_for_partition,
)
from .dino_models import DinoRetrievalModel, FrozenDinoEncoder, load_local_dino
from .engine import (
    _amp_enabled,
    _checkpoint_payload,
    _load_checkpoint,
    _optimizer,
    choose_device,
    evaluate_base_classifier,
    seed_everything,
)
from .losses import batch_hard_triplet_loss, text_anchor_contrastive_loss
from .metrics import summarize_folds
from .models import TextAnchorPrompt, load_local_clip


DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
DINO_MEAN_RGB = tuple(round(channel * 255) for channel in DINO_MEAN)
DINO_INPUT_MODES = {"center_crop", "letterbox"}


def _dino_backbone_name(config: dict[str, Any]) -> str:
    """
    方法作用：
        从配置中读取并校验 DINOv2 主干名称。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
    
    返回值：
        str：方法执行得到的结果。
    """
    configured = str(config.get("model", {}).get("backbone_name", "")).strip()
    if configured:
        return configured
    return Path(config["paths"]["dino_model"]).name


class DinoLetterbox:
    """
    方法作用：
        按原始宽高比缩放图片并填充为正方形，避免裁掉目标。
    
    输入参数：
        image_size (int)：初始化实例所需的 image_size 参数。
        fill (tuple[int, int, int])：初始化实例所需的 fill 参数。
    
    返回值：
        DinoLetterbox：初始化后的类实例。
    """

    def __init__(
        self,
        image_size: int,
        fill: tuple[int, int, int] = DINO_MEAN_RGB,
    ) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            image_size (int)：方法所需的 image_size 参数。
            fill (tuple[int, int, int])：方法所需的 fill 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.image_size = int(image_size)
        self.fill = tuple(int(channel) for channel in fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        方法作用：
            对输入图像执行保持宽高比的缩放与填充。
        
        输入参数：
            self (Any)：当前实例。
        image (Image.Image)：尺寸为 [W, H] 的单张 PIL 图像。
        
        返回值：
        Image.Image：填充后的正方形 PIL 图像，尺寸为 [image_size, image_size]。
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"DinoLetterbox expects PIL.Image, got {type(image).__name__}")
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
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.fill)
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


def _validate_dino_input_mode(input_mode: str) -> str:
    """
    方法作用：
        校验并规范化 DINOv2 的图像输入模式。
    
    输入参数：
        input_mode (str)：方法所需的 input_mode 参数。
    
    返回值：
        str：方法执行得到的结果。
    """
    resolved = str(input_mode).strip().lower()
    if resolved not in DINO_INPUT_MODES:
        raise ValueError(
            f"Unsupported DINO input_mode={input_mode!r}; "
            f"expected one of {sorted(DINO_INPUT_MODES)}"
        )
    return resolved


def build_dino_eval_transform(
    image_size: int,
    resize_short_edge: int,
    input_mode: str = "center_crop",
):
    """
    方法作用：
        创建 DINOv2 评估阶段的图像预处理流程。
    
    输入参数：
        image_size (int)：输出张量的高和宽 H=W=image_size。
        resize_short_edge (int)：方法所需的 resize_short_edge 参数。
        input_mode (str)：方法所需的 input_mode 参数。
    
    返回值：
        transforms.Compose：将单张 PIL 图片转换为 [3, H, W] 张量的评估变换。
    """
    input_mode = _validate_dino_input_mode(input_mode)
    spatial_transform = (
        DinoLetterbox(image_size)
        if input_mode == "letterbox"
        else transforms.Compose(
            [
                transforms.Resize(
                    resize_short_edge,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ]
        )
    )
    return transforms.Compose(
        [
            spatial_transform,
            transforms.ToTensor(),
            transforms.Normalize(DINO_MEAN, DINO_STD),
        ]
    )


def build_dino_train_transform(
    image_size: int,
    input_mode: str = "center_crop",
):
    """
    方法作用：
        创建 DINOv2 训练阶段的数据增强流程。
    
    输入参数：
        image_size (int)：增强后张量的高和宽 H=W=image_size。
        input_mode (str)：方法所需的 input_mode 参数。
    
    返回值：
        transforms.Compose：将单张 PIL 图片增强并转换为 [3, H, W] 张量的训练变换。
    """
    input_mode = _validate_dino_input_mode(input_mode)
    if input_mode == "letterbox":
        # The experiment is intended to preserve the complete vehicle. Avoid
        # RandomResizedCrop, perspective warping, and erasing here because each
        # can remove the same fine-grained parts that letterboxing retains.
        return transforms.Compose(
            [
                DinoLetterbox(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.20,
                    hue=0.04,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.15,
                ),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.75, 1.333333),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.RandomPerspective(
                        distortion_scale=0.15,
                        p=1.0,
                        interpolation=InterpolationMode.BICUBIC,
                    )
                ],
                p=0.25,
            ),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.04,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(DINO_MEAN, DINO_STD),
            transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def _fold_output_root(config: dict[str, Any], variant: str, fold: int) -> Path:
    """
    方法作用：
        生成当前实验版本和折编号的输出目录。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        variant (str)：实验版本名称。
        fold (int)：交叉验证折信息或折编号。
    
    返回值：
        Path：方法执行得到的结果。
    """
    return config["paths"]["output_root"] / "dino" / variant / f"fold_{fold:02d}"


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        将数据以格式化 JSON 写入指定文件。
    
    输入参数：
        path (Path)：目标文件或目录路径。
        payload (Any)：待写入或处理的数据。
    
    返回值：
        None：方法直接完成相应操作。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_config(
    config: dict[str, Any],
    variant: str,
    fold: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    方法作用：
        整理 DINOv2 当前实验的可序列化运行配置。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        variant (str)：实验版本名称。
        fold (dict[str, Any])：交叉验证折信息或折编号。
        device (torch.device)：执行计算的 PyTorch 设备。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    return {
        "format": "annotation_pad_lite_dino_fold_v1",
        "visual_backbone": _dino_backbone_name(config),
        "variant": variant,
        "fold": int(fold["fold"]),
        "base_classes": list(fold["base_classes"]),
        "novel_classes": list(fold["novel_classes"]),
        "device": str(device),
        "config": serializable_config(config),
    }


def run_dino_b0_fold(
    config: dict[str, Any], fold_number: int, device_name: str
) -> dict[str, Any]:
    """
    方法作用：
        运行单折冻结 DINOv2 基线检索评估。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        fold_number (int)：交叉验证折编号。
        device_name (str)：运行设备名称。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    device = choose_device(device_name)
    seed_everything(int(config.get("seed", 2026)) + fold_number)
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    crop_index = CropIndex(config["paths"]["crop_root"])
    output_dir = _fold_output_root(config, "b0", fold_number)
    _write_json(output_dir / "run_config.json", _run_config(config, "b0", fold, device))
    dino_model, _ = load_local_dino(config["paths"]["dino_model"])
    encoder = FrozenDinoEncoder(dino_model).to(device).eval()
    metrics = evaluate_dino_novel_retrieval(
        encoder,
        crop_index,
        fold,
        config,
        device,
        output_dir,
        "b0",
    )
    return metrics


def evaluate_dino_novel_retrieval(
    encoder,
    crop_index: CropIndex,
    fold: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    variant: str,
) -> dict[str, Any]:
    """
    方法作用：
        使用 DINOv2 编码器评估新类别少样本检索效果。
    
    输入参数：
        encoder (Any)：将图像批次 [B, 3, H, W] 编码为特征 [B, D] 的编码器。
        crop_index (CropIndex)：方法所需的 crop_index 参数。
        fold (dict[str, Any])：交叉验证折信息或折编号。
        config (dict[str, Any])：实验配置字典。
        device (torch.device)：执行计算的 PyTorch 设备。
        output_dir (Path)：当前实验的输出目录。
        variant (str)：实验版本名称。
    
    返回值：
        dict[str, Any]：由支持特征 [Ns, D] 与查询特征 [Nq, D] 计算出的
        Rank-1、mAP、样本数量等检索指标。
    """
    # The shared evaluator constructs CLIP transforms, so reproduce its small
    # dataset assembly here and temporarily provide DINO-specific loaders.
    from .metrics import compute_retrieval_metrics, encode_loader, save_retrieval_outputs

    novel_classes = list(fold["novel_classes"])
    class_to_local = {name: index for index, name in enumerate(novel_classes)}
    transform = build_dino_eval_transform(
        int(config["data"]["image_size"]),
        int(config["data"]["resize_short_edge"]),
        str(config["data"].get("input_mode", "center_crop")),
    )
    support_records = records_for_partition(
        crop_index,
        fold,
        "novel_support",
        exclude_tiny=bool(config["data"].get("exclude_tiny_support", True)),
    )
    query_records = records_for_partition(crop_index, fold, "novel_query")
    support_classes = {record.class_name for record in support_records}
    if support_classes != set(novel_classes):
        raise ValueError(f"Support loses a novel class after filtering: {support_classes}")
    support_loader = build_eval_loader(
        CropDataset(support_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    query_loader = build_eval_loader(
        CropDataset(query_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    support_features, support_labels, _ = encode_loader(
        encoder, support_loader, device, _amp_enabled(config, device)
    )
    query_features, query_labels, query_metadata = encode_loader(
        encoder, query_loader, device, _amp_enabled(config, device)
    )
    query_tiny = torch.tensor(
        [item["tiny"] for item in query_metadata], dtype=torch.bool
    ).numpy()
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


def train_dino_one_epoch(
    model: DinoRetrievalModel,
    prompt: TextAnchorPrompt | None,
    clip_text_model,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
    amp: bool,
) -> dict[str, float]:
    """
    方法作用：
        完成 DINOv2 检索模型的一轮训练并统计指标。
    
    输入参数：
        model (DinoRetrievalModel)：待训练、评估或读取参数的模型。
        prompt (TextAnchorPrompt | None)：方法所需的 prompt 参数。
        clip_text_model (Any)：方法所需的 clip_text_model 参数。
        loader (Any)：逐批提供 image [B, 3, H, W] 和 label [B] 的训练加载器。
        optimizer (torch.optim.Optimizer)：方法所需的 optimizer 参数。
        scaler (torch.amp.GradScaler)：方法所需的 scaler 参数。
        device (torch.device)：执行计算的 PyTorch 设备。
        settings (dict[str, Any])：当前实验版本的训练参数。
        amp (bool)：方法所需的 amp 参数。
    
    返回值：
        dict[str, float]：对整轮样本聚合后的标量损失和准确率；前向过程中
        模型产生 embedding [B, D]、logits [B, C]，B、D、C 分别为批量大小、
        嵌入维度和基础类别数。
    """
    model.train()
    if prompt is not None:
        prompt.train()
    if clip_text_model is not None:
        clip_text_model.eval()
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
                if clip_text_model is None:
                    raise RuntimeError("DINO-B2 requires the frozen CLIP text teacher")
                anchors = prompt(clip_text_model)
                anchor_loss = text_anchor_contrastive_loss(
                    embeddings,
                    anchors,
                    labels,
                    temperature=float(settings["anchor_temperature"]),
                )
            loss = (
                float(settings["id_weight"]) * id_loss
                + float(settings["triplet_weight"]) * triplet_loss
                + float(settings.get("anchor_weight", 0.0)) * anchor_loss
            )
        scaler.scale(loss).backward()
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
        totals["loss"] += float(loss.detach().item()) * batch_size
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


def run_dino_training_fold(
    config: dict[str, Any],
    variant: str,
    fold_number: int,
    device_name: str,
    resume: bool,
    eval_only: bool,
) -> dict[str, Any]:
    """
    方法作用：
        完成单折 DINOv2 模型的训练、选模和新类评估。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        variant (str)：实验版本名称。
        fold_number (int)：交叉验证折编号。
        device_name (str)：运行设备名称。
        resume (bool)：是否从最近检查点恢复训练。
        eval_only (bool)：是否跳过训练并仅执行评估。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    if variant not in {"b1", "b2"}:
        raise ValueError(f"DINO training only supports b1/b2, got {variant}")
    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + fold_number
    seed_everything(seed)
    fold = load_fold(config["paths"]["splits_root"], fold_number)
    crop_index = CropIndex(config["paths"]["crop_root"])
    settings = config[variant]
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
    train_loader, batch_sampler = build_train_loader(
        CropDataset(
            train_records,
            build_dino_train_transform(
                int(config["data"]["image_size"]),
                str(config["data"].get("input_mode", "center_crop")),
            ),
            class_to_local,
        ),
        classes_per_batch=int(settings["classes_per_batch"]),
        instances_per_class=int(settings["instances_per_class"]),
        workers=int(config["data"]["workers"]),
        seed=seed,
    )
    val_loader = build_eval_loader(
        CropDataset(
            val_records,
            build_dino_eval_transform(
                int(config["data"]["image_size"]),
                int(config["data"]["resize_short_edge"]),
                str(config["data"].get("input_mode", "center_crop")),
            ),
            class_to_local,
        ),
        batch_size=int(config["data"]["eval_batch_size"]),
        workers=int(config["data"]["workers"]),
    )

    dino_model, _ = load_local_dino(config["paths"]["dino_model"])
    model = DinoRetrievalModel(
        dino_model,
        num_classes=len(base_classes),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    )
    model.configure_backbone(
        int(config["model"]["unfreeze_last_blocks"]), enabled=True
    )

    prompt = None
    clip_text_model = None
    if variant == "b2":
        if int(config["model"]["embedding_dim"]) != 512:
            raise ValueError("DINO-B2 embedding_dim must match CLIP projection_dim=512")
        clip_text_model, processor = load_local_clip(config["paths"]["clip_model"])
        for parameter in clip_text_model.parameters():
            parameter.requires_grad_(False)
        clip_text_model.eval()
        prompt_cfg = settings["prompt"]
        prompt = TextAnchorPrompt(
            clip_text_model,
            processor.tokenizer,
            num_classes=len(base_classes),
            context_tokens=int(prompt_cfg["context_tokens"]),
            prefix=str(prompt_cfg["prefix"]),
            suffix=str(prompt_cfg["suffix"]),
            init_std=float(prompt_cfg["init_std"]),
        )
        if any(name.lower() in prompt.template.lower() for name in base_classes):
            raise ValueError("A class name entered the DINO-B2 prompt template")

    model.to(device)
    if clip_text_model is not None:
        clip_text_model.to(device)
    if prompt is not None:
        prompt.to(device)
    head_parameter_count = sum(
        parameter.numel() for parameter in model.head_parameters()
    )
    backbone_parameter_count = sum(
        parameter.numel() for parameter in model.backbone_parameters()
    )
    prompt_parameter_count = (
        sum(parameter.numel() for parameter in prompt.parameters())
        if prompt is not None
        else 0
    )
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

    output_dir = _fold_output_root(config, variant, fold_number)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = _run_config(config, variant, fold, device)
    run_config.update(
        {
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "parameter_groups": {
                "head": head_parameter_count,
                "selected_backbone": backbone_parameter_count,
                "prompt": prompt_parameter_count,
            },
            "trainable_parameters_during_warmup": (
                head_parameter_count + prompt_parameter_count
            ),
            "trainable_parameters_after_warmup": (
                head_parameter_count
                + backbone_parameter_count
                + prompt_parameter_count
            ),
            "prompt_template": prompt.template if prompt is not None else None,
        }
    )
    _write_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

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
                last_checkpoint,
                model,
                prompt,
                optimizer,
                scheduler,
                scaler,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_accuracy = float(checkpoint["best_accuracy"])
            history_path = output_dir / "history.json"
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))

        for epoch in range(start_epoch, epochs + 1):
            model.set_backbone_enabled(epoch > warmup_epochs)
            batch_sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = train_dino_one_epoch(
                model,
                prompt,
                clip_text_model,
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
            record = {
                "epoch": epoch,
                "backbone_enabled": epoch > warmup_epochs,
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
            payload["visual_backbone"] = _dino_backbone_name(config)
            torch.save(payload, last_checkpoint)
            if float(selection_accuracy) > best_accuracy:
                best_accuracy = float(selection_accuracy)
                payload["best_accuracy"] = best_accuracy
                torch.save(payload, best_checkpoint)
            _write_json(output_dir / "history.json", history)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        _load_checkpoint(best_checkpoint, model, prompt)

    model.eval()
    metrics = evaluate_dino_novel_retrieval(
        model,
        crop_index,
        fold,
        config,
        device,
        output_dir,
        variant,
    )
    metrics["best_base_val_accuracy"] = best_accuracy
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_dino_variant(
    config: dict[str, Any],
    variant: str,
    folds: list[int],
    device_name: str,
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        依次运行指定 DINOv2 实验版本的多个交叉验证折。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        variant (str)：实验版本名称。
        folds (list[int])：需要运行的交叉验证折编号列表。
        device_name (str)：运行设备名称。
        resume (bool)：是否从最近检查点恢复训练。
        eval_only (bool)：是否跳过训练并仅执行评估。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    if "dino_model" not in config["paths"]:
        raise ValueError("DINO config requires paths.dino_model")
    results = []
    for fold in folds:
        if variant == "b0":
            metrics = run_dino_b0_fold(config, fold, device_name)
        else:
            metrics = run_dino_training_fold(
                config,
                variant,
                fold,
                device_name,
                resume,
                eval_only,
            )
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    variant_root = config["paths"]["output_root"] / "dino" / variant
    summary = summarize_folds(variant_root)
    summary.update({"visual_backbone": _dino_backbone_name(config), "variant": variant})
    _write_json(variant_root / "summary.json", summary)
    return {"results": results, "summary": summary}
