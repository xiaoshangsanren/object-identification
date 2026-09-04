"""Train and evaluate the E1 ImageNet ViT + visual-only LaFG L_aux baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from torch import nn

from fg_retrain.data import build_dataloader, load_stanford_cars_parquet
from fg_retrain.laux_config import LauxConfig, ResolvedLauxConfig
from fg_retrain.laux_data import build_train_dataloader
from fg_retrain.laux_loss import lafg_auxiliary_contrastive_loss
from fg_retrain.modeling import PureVisualViT, extract_embeddings, load_image_processor
from fg_retrain.pairwise_difficulty import (
    build_pairwise_metrics_section,
    evaluate_pairwise_difficulty,
    save_pairwise_difficulty,
)
from fg_retrain.retrieval import evaluate_leave_one_out


def build_parser() -> argparse.ArgumentParser:
    """构建E1训练命令行参数解析器。

    作用:
        接收配置文件、设备覆盖、断点策略和仅用于验证的轮数覆盖。
    参数:
        无。
    返回值:
        配置完成的``argparse.ArgumentParser``。
    """

    parser = argparse.ArgumentParser(
        description="Train ImageNet ViT with visual-only LaFG L_aux and evaluate retrieval."
    )
    parser.add_argument("--config", type=Path, required=True, help="E1 JSON configuration.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Optional runtime device override.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reject an existing checkpoint instead of resuming it.",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
        help="Test-only epoch override; outputs are marked as a smoke run.",
    )
    return parser


def repository_root() -> Path:
    """定位项目仓库根目录。

    作用:
        根据E1入口源文件位置解析根目录，不依赖启动时的工作目录。
    参数:
        无。
    返回值:
        仓库根目录绝对``Path``。
    """

    return Path(__file__).resolve().parents[4]


def select_device(requested: str) -> torch.device:
    """校验并返回E1计算设备。

    作用:
        在请求CUDA但CUDA不可用时提前终止，避免训练静默回退到CPU。
    参数:
        requested: ``cuda``或``cpu``字符串。
    返回值:
        ``torch.device``实例。
    """

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """初始化E1随机状态。

    作用:
        设置Python、PyTorch及所有可见CUDA设备的随机种子。
    参数:
        seed: 整数随机种子。
    返回值:
        无。
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    """保存带缩进的UTF-8 JSON。

    作用:
        自动创建父目录并写入配置、清单、指标或摘要。
    参数:
        path: 目标JSON路径。
        value: 可JSON序列化的对象。
    返回值:
        无。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """保存完整JSONL记录集合。

    作用:
        为每条记录写一行JSON，当前主要用于逐Query检索结果。
    参数:
        path: 目标JSONL路径。
        records: 长度``[N]``的记录列表；检索场景中每项含Top-K邻居``[K]``。
    返回值:
        无。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加并立即刷新一条训练历史。

    作用:
        每完成一个epoch就持久化标量指标，降低异常退出造成的日志丢失。
    参数:
        path: ``train_history.jsonl``路径。
        record: 单轮训练指标字典，所有值均为标量或字符串。
    返回值:
        无。
    """

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def reconcile_history(path: Path, completed_epochs: int) -> None:
    """使训练历史与最近可恢复检查点保持一致。

    作用:
        断点续训前删除epoch编号大于持久化检查点的日志，避免重复或虚假记录。
    参数:
        path: 训练历史JSONL路径。
        completed_epochs: 检查点已完成的epoch数量标量。
    返回值:
        无；历史文件不存在时直接返回。
    """

    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["epoch"]) <= completed_epochs:
            retained.append(json.dumps(record, ensure_ascii=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    """计算方法规范文件的SHA-256。

    作用:
        将本次实验绑定到确定版本的``LaFG_Laux_execution_method.md``。
    参数:
        path: 待计算哈希的文件路径。
    返回值:
        64字符十六进制SHA-256字符串。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """采集E1软件环境和全部可见GPU。

    作用:
        记录Python、平台、PyTorch、Transformers、Datasets、CUDA及GPU名称。
    参数:
        device: 本次训练使用的主设备。
    返回值:
        可JSON序列化的环境信息字典；不包含图像主数据。
    """

    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        result["visible_cuda_devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
    return result


def unwrap_model(model: nn.Module) -> PureVisualViT:
    """从可选DataParallel包装中取出原始ViT。

    作用:
        统一单GPU和多GPU情况下的权重保存、加载接口。
    参数:
        model: ``PureVisualViT``或``nn.DataParallel``包装模型；前向输出``[B,768]``。
    返回值:
        原始``PureVisualViT``实例。
    """

    if isinstance(model, nn.DataParallel):
        return model.module  # type: ignore[return-value]
    return model  # type: ignore[return-value]


def atomic_torch_save(value: Any, path: Path) -> None:
    """原子保存PyTorch对象。

    作用:
        先写临时文件，完整写入后再替换目标，避免中断留下损坏checkpoint。
    参数:
        value: 可由``torch.save``序列化的对象，可能包含模型参数张量。
        path: 最终输出路径。
    返回值:
        无。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    completed_epochs: int,
    config: ResolvedLauxConfig,
) -> None:
    """保存从下一轮继续训练所需的完整状态。

    作用:
        持久化ViT权重、SGD、学习率调度器、AMP scaler、轮次和配置。
    参数:
        path: 检查点路径。
        model: 当前ViT或DataParallel模型；视觉前向为``[B,3,224,224] -> [B,768]``。
        optimizer: 当前SGD优化器。
        scheduler: 当前StepLR调度器。
        scaler: 当前自动混合精度GradScaler。
        completed_epochs: 已完整训练的epoch数量。
        config: 已解析E1配置。
    返回值:
        无；检查点原子写入磁盘。
    """

    atomic_torch_save(
        {
            "completed_epochs": completed_epochs,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config.as_serializable_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> int:
    """从E1检查点恢复全部训练状态。

    作用:
        恢复ViT、优化器、调度器和AMP scaler，使下一轮训练与中断前连续。
    参数:
        path: 检查点路径。
        model: 待恢复的ViT或DataParallel模型；参数形状必须与checkpoint一致。
        optimizer: 待恢复的SGD优化器。
        scheduler: 待恢复的StepLR调度器。
        scaler: 待恢复的AMP GradScaler。
    返回值:
        已完成的epoch数，也是训练循环下一轮的零基索引。
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["completed_epochs"])


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    temperature: float,
    amp: bool,
) -> dict[str, float]:
    """使用纯视觉Laux完成一个PK采样训练轮次。

    作用:
        遍历训练批次，执行ViT前向、Laux、AMP反向传播和SGD更新，并累计几何诊断量。
    参数:
        model: ViT或DataParallel模型；输入图像``[B,3,224,224]``，输出特征``[B,768]``。
        loader: PK DataLoader；每批图像``[B=2P,3,224,224]``、标签``[B]``。
        optimizer: 更新全部可训练ViT参数的SGD。
        scaler: AMP梯度缩放器。
        device: 批次张量放置设备。
        temperature: Laux距离softmax温度标量。
        amp: 是否在CUDA上启用float16自动混合精度。
    返回值:
        单轮标量指标字典：平均loss、pair accuracy、正负相似度、图像数、批次数、耗时和吞吐率。
    """

    model.train()
    totals = Counter()
    started = time.perf_counter()

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            embeddings = model(images)
            output = lafg_auxiliary_contrastive_loss(embeddings, labels, temperature)
        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = len(labels)
        totals["images"] += batch_size
        totals["batches"] += 1
        totals["loss_sum"] += float(output.loss.detach().item()) * batch_size
        totals["selection_sum"] += (
            float(output.positive_selection_accuracy.detach().item()) * batch_size
        )
        totals["positive_similarity_sum"] += (
            float(output.mean_positive_similarity.detach().item()) * batch_size
        )
        totals["hardest_negative_similarity_sum"] += (
            float(output.mean_hardest_negative_similarity.detach().item()) * batch_size
        )

    elapsed = time.perf_counter() - started
    image_count = float(totals["images"])
    return {
        "loss": totals["loss_sum"] / image_count,
        "positive_selection_accuracy": totals["selection_sum"] / image_count,
        "mean_positive_similarity": totals["positive_similarity_sum"] / image_count,
        "mean_hardest_negative_similarity": (
            totals["hardest_negative_similarity_sum"] / image_count
        ),
        "images_seen": int(totals["images"]),
        "num_batches": int(totals["batches"]),
        "seconds": elapsed,
        "images_per_second": image_count / elapsed,
    }


def main() -> None:
    """执行E1的训练、保存和留一法检索评测。

    作用:
        加载官方train/test，构建P×2批次，完整微调ViT，保存可恢复状态；随后提取测试特征
        ``[8041,768]``，以每张Query对其余8040张Gallery进行检索并保存结果。
    参数:
        无；从命令行读取``--config``、设备和断点选项。
    返回值:
        无；训练权重、历史、特征``[N,768]``、逐Query记录``[N]``及指标写入输出目录。
    """

    args = build_parser().parse_args()
    config = LauxConfig.from_json(args.config.resolve()).resolved(repository_root())
    device = select_device(args.device or config.device)
    seed_everything(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_last.pt"
    history_path = output_dir / "train_history.jsonl"
    method_path = repository_root() / "others" / "LaFG_Laux_execution_method.md"

    target_epochs = args.smoke_epochs if args.smoke_epochs is not None else config.epochs
    if target_epochs <= 0 or target_epochs > config.epochs:
        raise ValueError("--smoke-epochs must be in [1, configured epochs].")
    is_smoke_run = target_epochs != config.epochs
    if is_smoke_run:
        output_dir = output_dir / f"smoke_{target_epochs}_epochs"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "checkpoint_last.pt"
        history_path = output_dir / "train_history.jsonl"

    save_json(output_dir / "config_resolved.json", config.as_serializable_dict())
    save_json(output_dir / "environment.json", runtime_environment(device))
    save_json(
        output_dir / "method_provenance.json",
        {
            "method_document": str(method_path),
            "method_document_sha256": file_sha256(method_path),
            "loss": "LaFG visual-only L_aux",
            "dataset_protocol": (
                "Stanford Cars-196 official train (known classes) for training; "
                "official test leave-one-out for retrieval"
            ),
            "paper_unspecified_engineering_choices": {
                "temperature": config.temperature,
                "classes_per_batch": config.classes_per_batch,
                "color_jitter_strengths": {
                    "brightness": config.color_jitter_brightness,
                    "contrast": config.color_jitter_contrast,
                    "saturation": config.color_jitter_saturation,
                    "hue": config.color_jitter_hue,
                },
            },
            "is_smoke_run": is_smoke_run,
        },
    )

    print("[1/5] Loading Stanford Cars-196 train and test parquet splits")
    train_data = load_stanford_cars_parquet(config.dataset_root, config.train_split)
    test_data = load_stanford_cars_parquet(config.dataset_root, config.test_split)
    if train_data.class_names != test_data.class_names:
        raise RuntimeError("Train and test ClassLabel tables differ.")
    train_counts = Counter(int(label) for label in train_data.dataset["label"])
    test_counts = Counter(int(label) for label in test_data.dataset["label"])
    save_json(
        output_dir / "data_manifest.json",
        {
            "train_images": len(train_data.dataset),
            "test_images": len(test_data.dataset),
            "num_classes": len(train_data.class_names),
            "train_class_counts": {str(k): v for k, v in sorted(train_counts.items())},
            "test_class_counts": {str(k): v for k, v in sorted(test_counts.items())},
            "train_parquet_files": [str(path) for path in train_data.parquet_files],
            "test_parquet_files": [str(path) for path in test_data.parquet_files],
        },
    )
    print(
        f"      train={len(train_data.dataset)}, test={len(test_data.dataset)}, "
        f"classes={len(train_data.class_names)}"
    )

    print("[2/5] Building ImageNet ViT and strict P×2 batches")
    base_model = PureVisualViT(config.model_path).to(device)
    if config.gradient_checkpointing:
        base_model.backbone.gradient_checkpointing_enable()
    train_loader, train_sampler = build_train_dataloader(
        train_data.dataset,
        config,
        use_cuda=device.type == "cuda",
    )
    expected_batches = config.batches_per_epoch or math.ceil(
        len(train_data.dataset) / config.train_batch_size
    )
    if len(train_loader) != expected_batches:
        raise RuntimeError("Unexpected PK batch count.")

    model: nn.Module = base_model
    if config.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)
        print(f"      DataParallel devices={list(range(torch.cuda.device_count()))}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.amp and device.type == "cuda",
    )

    start_epoch = 0
    if checkpoint_path.exists():
        if args.no_resume:
            raise FileExistsError(f"Checkpoint exists and --no-resume was set: {checkpoint_path}")
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)
        print(f"      resumed after epoch {start_epoch}")
    reconcile_history(history_path, start_epoch)

    print(
        f"[3/5] Training L_aux: epochs={start_epoch + 1}..{target_epochs}, "
        f"P={config.classes_per_batch}, K=2, batch={config.train_batch_size}, "
        f"batches/epoch={len(train_loader)}"
    )
    training_started = time.perf_counter()
    for epoch in range(start_epoch, target_epochs):
        train_sampler.set_epoch(epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            temperature=config.temperature,
            amp=config.amp,
        )
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            **epoch_metrics,
        }
        append_jsonl(history_path, record)
        print(
            f"      epoch={epoch + 1:03d}/{target_epochs} "
            f"loss={epoch_metrics['loss']:.5f} "
            f"pair_acc={epoch_metrics['positive_selection_accuracy']:.4f} "
            f"pos_sim={epoch_metrics['mean_positive_similarity']:.4f} "
            f"hardneg_sim={epoch_metrics['mean_hardest_negative_similarity']:.4f} "
            f"sec={epoch_metrics['seconds']:.1f}"
        )
        if (epoch + 1) % config.checkpoint_every_epochs == 0 or epoch + 1 == target_epochs:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                completed_epochs=epoch + 1,
                config=config,
            )
    training_seconds = time.perf_counter() - training_started

    atomic_torch_save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "model": "google/vit-base-patch16-224 without ImageNet classifier",
            "embedding": "L2-normalized final-layer CLS token",
            "completed_epochs": target_epochs,
            "is_smoke_run": is_smoke_run,
            "config": config.as_serializable_dict(),
        },
        output_dir / "model_final.pt",
    )
    save_json(
        output_dir / "training_summary.json",
        {
            "completed_epochs": target_epochs,
            "training_seconds_this_process": training_seconds,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "classes_per_batch": config.classes_per_batch,
            "instances_per_class": 2,
            "train_batch_size": config.train_batch_size,
            "batches_per_epoch": len(train_loader),
            "is_smoke_run": is_smoke_run,
        },
    )

    print("[4/5] Extracting final pure-visual test embeddings")
    processor = load_image_processor(config.model_path)
    test_loader = build_dataloader(
        test_data.dataset,
        processor,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        use_cuda=device.type == "cuda",
    )
    features, labels = extract_embeddings(model, test_loader, device, amp=config.amp)
    if config.save_embeddings:
        atomic_torch_save(
            {
                "features": features,
                "labels": labels,
                "class_names": test_data.class_names,
                "split": config.test_split,
            },
            output_dir / "test_embeddings.pt",
        )

    print("[5/5] Evaluating each test query against all other test images")
    evaluation = evaluate_leave_one_out(
        features=features,
        labels=labels,
        class_names=test_data.class_names,
        device=device,
        query_chunk_size=config.retrieval_query_chunk_size,
        recall_ks=config.recall_ks,
        compute_map=config.compute_map,
        save_top_k=config.save_top_k,
    )
    evaluation.metrics.update(
        {
            "experiment_name": config.experiment_name,
            "training_objective": "LaFG visual-only L_aux",
            "train_split": config.train_split,
            "test_split": config.test_split,
            "train_images": len(train_data.dataset),
            "completed_epochs": target_epochs,
            "is_smoke_run": is_smoke_run,
            "model": "google/vit-base-patch16-224, fully fine-tuned",
            "embedding": "L2-normalized final-layer CLS token",
        }
    )
    print("      computing all class-pair Recall@1..5 difficulty metrics")
    pairwise_evaluation = evaluate_pairwise_difficulty(
        features=features,
        labels=labels,
        class_names=test_data.class_names,
        device=device,
        query_chunk_size=config.pairwise_query_chunk_size,
        recall_ks=config.pairwise_recall_ks,
        top_n_per_class=config.pairwise_top_n_per_class,
    )
    pairwise_artifacts = save_pairwise_difficulty(output_dir, pairwise_evaluation)
    evaluation.metrics["pairwise_difficulty"] = build_pairwise_metrics_section(
        pairwise_evaluation,
        pairwise_artifacts,
        top_pair_count=20,
    )
    save_json(output_dir / "metrics.json", evaluation.metrics)
    save_jsonl(output_dir / "per_query.jsonl", evaluation.per_query)
    print(json.dumps({k: v for k, v in evaluation.metrics.items() if k != "per_class"}, indent=2))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
