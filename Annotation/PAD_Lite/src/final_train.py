from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import PAD_LITE_ROOT, load_config, serializable_config
from .data import CropDataset, CropIndex, build_train_loader, build_train_transform
from .engine import (
    _checkpoint_payload,
    _load_checkpoint,
    _optimizer,
    _amp_enabled,
    choose_device,
    seed_everything,
    train_one_epoch,
)
from .models import RetrievalModel, TextAnchorPrompt, load_local_clip


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


def train_final_model(
    config: dict[str, Any],
    variant: str,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    """
    方法作用：
        执行 train_final_model 对应的处理流程。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        variant (str)：实验版本名称。
        device_name (str)：运行设备名称。
        resume (bool)：是否从最近检查点恢复训练。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    if variant not in {"b1", "b2"}:
        raise ValueError(f"Final training only supports b1/b2, got {variant}")

    device = choose_device(device_name)
    seed = int(config.get("seed", 2026)) + (101 if variant == "b1" else 102)
    seed_everything(seed)
    settings = config[variant]
    crop_index = CropIndex(config["paths"]["crop_root"])
    class_names = list(crop_index.classes)
    class_to_local = {name: index for index, name in enumerate(class_names)}

    # Final deployment training intentionally consumes every validated crop,
    # including tiny samples. Cross-validation has already selected the epoch
    # budget, so Gallery images are never used for model selection.
    records = list(crop_index.records)
    dataset = CropDataset(
        records,
        build_train_transform(int(config["data"]["image_size"])),
        class_to_local,
    )
    loader, batch_sampler = build_train_loader(
        dataset,
        classes_per_batch=int(settings["classes_per_batch"]),
        instances_per_class=int(settings["instances_per_class"]),
        workers=int(config["data"]["workers"]),
        seed=seed,
    )

    clip_model, processor = load_local_clip(config["paths"]["clip_model"])
    model = RetrievalModel(
        clip_model,
        num_classes=len(class_names),
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    )
    last_n_blocks = int(config["model"]["unfreeze_last_blocks"])
    model.configure_backbone(last_n_blocks, enabled=True)

    prompt = None
    if variant == "b2":
        prompt_cfg = settings["prompt"]
        prompt = TextAnchorPrompt(
            clip_model,
            processor.tokenizer,
            num_classes=len(class_names),
            context_tokens=int(prompt_cfg["context_tokens"]),
            prefix=str(prompt_cfg["prefix"]),
            suffix=str(prompt_cfg["suffix"]),
            init_std=float(prompt_cfg["init_std"]),
        )
        if any(name.lower() in prompt.template.lower() for name in class_names):
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

    output_dir = config["paths"]["output_root"] / "final" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pt"
    final_checkpoint = output_dir / "final.pt"
    history_path = output_dir / "history.json"
    history: list[dict[str, Any]] = []
    start_epoch = 1
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
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    run_config = {
        "format": "annotation_pad_lite_final_run_v1",
        "variant": variant,
        "training_scope": "all_russian_classes",
        "classes": class_names,
        "class_count": len(class_names),
        "sample_count": len(records),
        "tiny_sample_count": sum(record.tiny for record in records),
        "device": str(device),
        "seed": seed,
        "fixed_epoch_budget": epochs,
        "gallery_used_for_training_or_selection": False,
        "config": serializable_config(config),
    }
    _write_json(output_dir / "run_config.json", run_config)
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        backbone_enabled = epoch > warmup_epochs
        model.set_backbone_enabled(backbone_enabled)
        batch_sampler.set_epoch(epoch)
        started = time.time()
        train_metrics = train_one_epoch(
            model,
            prompt,
            loader,
            optimizer,
            scaler,
            device,
            settings,
            amp,
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "backbone_enabled": backbone_enabled,
            "seconds": round(time.time() - started, 3),
            "learning_rates": {
                group.get("name", str(index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "train": train_metrics,
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
            class_names,
            float(train_metrics["accuracy"]),
        )
        payload.update(
            {
                "training_scope": "all_russian_classes",
                "sample_count": len(records),
                "tiny_sample_count": sum(record.tiny for record in records),
                "seed": seed,
            }
        )
        torch.save(payload, last_checkpoint)
        _write_json(history_path, history)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    if not last_checkpoint.is_file():
        raise FileNotFoundError(f"Final checkpoint was not produced: {last_checkpoint}")
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    torch.save(checkpoint, final_checkpoint)
    summary = {
        **run_config,
        "completed_epochs": int(checkpoint["epoch"]),
        "final_train_metrics": history[-1]["train"] if history else None,
        "final_checkpoint": str(final_checkpoint),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建并配置当前命令行程序的参数解析器。
    
    输入参数：
        无。
    
    返回值：
        argparse.ArgumentParser：方法执行得到的结果。
    """
    parser = argparse.ArgumentParser(
        description="Train one PAD-Lite deployment model on all Russian classes."
    )
    parser.add_argument("variant", choices=("b1", "b2"))
    parser.add_argument(
        "--config",
        default=str(PAD_LITE_ROOT / "configs" / "default.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    """
    方法作用：
        解析命令行参数并执行当前脚本的主流程。
    
    输入参数：
        无。
    
    返回值：
        None：方法直接完成相应操作。
    """
    args = build_parser().parse_args()
    config = load_config(args.config)
    result = train_final_model(config, args.variant, args.device, args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
