from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT
from .detector_zero_shot_eval import (
    ImageRecord,
    _normalize_label,
    _sha256,
    evaluate_predictions,
    load_dataset,
    run_detr_inference,
    run_yolo_inference,
)


DEFAULT_CONFIG = PAD_LITE_ROOT / "configs" / "detector_finetune_yolo_detr.json"


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：
        将相对路径按Annotation根目录解析为绝对路径。

    输入参数：
        value (str|Path)：配置中的原始路径。

    返回值：
        Path：解析后的绝对路径。
    """

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def load_finetune_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：
        读取并校验检测器五折微调配置，解析全部文件系统路径。

    输入参数：
        path (str|Path)：阶段2配置文件路径。

    返回值：
        dict[str,Any]：包含paths、split、yolo、detr和evaluation的配置。
    """

    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Detector fine-tune schema_version must be 1")
    for section in ("paths", "split", "yolo", "detr", "evaluation"):
        if section not in payload:
            raise ValueError(f"Detector fine-tune config is missing {section}")
    for key in ("image_root", "prepared_root", "yolo_model", "detr_model", "output_root"):
        if key not in payload["paths"]:
            raise ValueError(f"Detector fine-tune config is missing paths.{key}")
        payload["paths"][key] = _resolve_path(payload["paths"][key])
    if int(payload["split"]["fold_count"]) != 5:
        raise ValueError("The current detector protocol requires exactly five folds")
    payload["config_path"] = config_path
    return payload


def _stable_key(seed: int, *parts: str) -> str:
    """
    方法作用：
        为图片划分生成与Python哈希随机化无关的稳定排序键。

    输入参数：
        seed (int)：实验种子；parts (str...)：车型、文件名或折号等组成部分。

    返回值：
        str：SHA-256十六进制稳定键。
    """

    text = ":".join([str(seed), *[str(part) for part in parts]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_stratum(record: ImageRecord) -> str:
    """
    方法作用：
        从一张原图的全部目标提取分层划分标签，并拒绝无类别记录。

    输入参数：
        record (ImageRecord)：含N个目标类别的整图记录。

    返回值：
        str：排序去重后的车型组合标签；当前数据通常为单一车型。
    """

    names = sorted(set(record.class_names))
    if not names:
        raise ValueError(f"Image has no class names: {record.image_path}")
    return "+".join(names)


def build_split_payloads(
    records: list[ImageRecord],
    fold_count: int,
    validation_fraction: float,
    seed: int,
    version: str,
) -> list[dict[str, Any]]:
    """
    方法作用：
        按原始车型分层生成外层五折Test，并从其余图片确定约10%的Validation。

    输入参数：
        records：I张原始整图；fold_count：外层折数；validation_fraction：全数据验证比例；
        seed：稳定划分种子；version：协议版本名。

    返回值：
        list[dict[str,Any]]：每折互斥Train/Validation/Test图片名列表。
    """

    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not 0.0 < validation_fraction < 1.0 / fold_count:
        raise ValueError("validation_fraction must be positive and smaller than one test fold")
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(_record_stratum(record), []).append(record)
    outer_assignment: dict[str, int] = {}
    for class_name, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: _stable_key(seed, "outer", class_name, item.image_path.name),
        )
        for index, record in enumerate(ordered):
            outer_assignment[record.image_path.name] = index % fold_count + 1

    payloads: list[dict[str, Any]] = []
    all_names = {record.image_path.name for record in records}
    for fold in range(1, fold_count + 1):
        train_names: set[str] = set()
        validation_names: set[str] = set()
        test_names: set[str] = set()
        for class_name, group in sorted(groups.items()):
            test_group = [
                record
                for record in group
                if outer_assignment[record.image_path.name] == fold
            ]
            remaining = [
                record
                for record in group
                if outer_assignment[record.image_path.name] != fold
            ]
            validation_count = max(1, int(round(len(group) * validation_fraction)))
            if validation_count >= len(remaining):
                raise ValueError(f"Not enough {class_name} images for train/validation")
            remaining.sort(
                key=lambda item: _stable_key(
                    seed, "validation", str(fold), class_name, item.image_path.name
                )
            )
            validation_group = remaining[:validation_count]
            train_group = remaining[validation_count:]
            test_names.update(record.image_path.name for record in test_group)
            validation_names.update(record.image_path.name for record in validation_group)
            train_names.update(record.image_path.name for record in train_group)
        if train_names & validation_names or train_names & test_names or validation_names & test_names:
            raise ValueError(f"Fold {fold} contains overlapping partitions")
        if train_names | validation_names | test_names != all_names:
            raise ValueError(f"Fold {fold} does not cover the full dataset")
        partitions = {
            "train": sorted(train_names),
            "validation": sorted(validation_names),
            "test": sorted(test_names),
        }
        by_name = {record.image_path.name: record for record in records}
        payloads.append(
            {
                "version": version,
                "fold": fold,
                "seed": seed,
                "stratification": "original_vehicle_class",
                "detection_class": "military_vehicle",
                "partitions": partitions,
                "counts": {
                    partition: {
                        "images": len(names),
                        "objects": sum(len(by_name[name].boxes) for name in names),
                    }
                    for partition, names in partitions.items()
                },
            }
        )
    return payloads


def _write_text_checked(path: Path, text: str) -> None:
    """
    方法作用：
        首次写入固定数据文件；若已存在则要求内容完全一致以防静默改折。

    输入参数：
        path (Path)：固定文件路径；text (str)：期望内容。

    返回值：
        None：文件已创建或已验证一致。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"Prepared file differs; refusing overwrite: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _ensure_image_link(source: Path, destination: Path) -> None:
    """
    方法作用：
        为YOLO统一数据目录创建原图软连接，并验证已有链接目标。

    输入参数：
        source (Path)：原始图片；destination (Path)：准备目录中的软链接。

    返回值：
        None：链接已创建或已确认指向同一图片。
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Image link points elsewhere: {destination}")
        return
    destination.symlink_to(source.resolve())


def _yolo_label_text(record: ImageRecord) -> str:
    """
    方法作用：
        将一张图的N个VOC xyxy真实框转换为单类别YOLO归一化标签。

    输入参数：
        record (ImageRecord)：宽W、高H、boxes形状(N,4)的原图记录。

    返回值：
        str：每行`0 cx cy w h`的标签文本。
    """

    rows = []
    for x1, y1, x2, y2 in record.boxes.tolist():
        center_x = ((x1 + x2) / 2.0) / record.width
        center_y = ((y1 + y2) / 2.0) / record.height
        width = (x2 - x1) / record.width
        height = (y2 - y1) / record.height
        rows.append(f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
    return "\n".join(rows) + "\n"


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """
    方法作用：
        生成共享五折manifest、YOLO单类标签、图片软链接和每折data.yaml。

    输入参数：
        config (dict)：阶段2配置，读取I张整图和N个VOC目标框。

    返回值：
        dict[str,Any]：准备目录、总样本数及五折计数摘要。
    """

    records = load_dataset(config["paths"]["image_root"])
    split = config["split"]
    payloads = build_split_payloads(
        records,
        fold_count=int(split["fold_count"]),
        validation_fraction=float(split["validation_fraction_of_all_images"]),
        seed=int(split["seed"]),
        version=str(split["version"]),
    )
    root = config["paths"]["prepared_root"]
    image_dir = root / "images"
    label_dir = root / "labels"
    for record in records:
        _ensure_image_link(record.image_path, image_dir / record.image_path.name)
        _write_text_checked(label_dir / f"{record.image_path.stem}.txt", _yolo_label_text(record))

    fold_rows = []
    for payload in payloads:
        fold = int(payload["fold"])
        fold_dir = root / "folds" / f"fold_{fold:02d}"
        _write_text_checked(
            fold_dir / "manifest.json",
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        lists: dict[str, Path] = {}
        for partition, names in payload["partitions"].items():
            list_path = fold_dir / f"{partition}.txt"
            # 保留准备目录中的images路径，Ultralytics才能按images→labels规则找到标签；
            # 不能对图片软链接调用resolve()，否则会回到只有VOC XML的原始train目录。
            text = "\n".join(str(image_dir / name) for name in names) + "\n"
            _write_text_checked(list_path, text)
            lists[partition] = list_path.resolve()
        data_yaml = {
            "path": str(root.resolve()),
            "train": str(lists["train"]),
            "val": str(lists["validation"]),
            "test": str(lists["test"]),
            "names": {0: str(config["evaluation"]["class_name"])},
        }
        _write_text_checked(
            fold_dir / "data.yaml",
            json.dumps(data_yaml, ensure_ascii=False, indent=2) + "\n",
        )
        fold_rows.append({"fold": fold, "counts": payload["counts"]})
    summary = {
        "version": split["version"],
        "seed": int(split["seed"]),
        "image_root": str(config["paths"]["image_root"]),
        "prepared_root": str(root),
        "image_count": len(records),
        "object_count": sum(len(record.boxes) for record in records),
        "folds": fold_rows,
    }
    _write_text_checked(
        root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


def _load_fold(
    config: dict[str, Any], fold: int, records: list[ImageRecord]
) -> dict[str, list[ImageRecord]]:
    """
    方法作用：
        按固定manifest将原图记录映射为某折Train/Validation/Test对象列表。

    输入参数：
        config：阶段2配置；fold：1到5；records：I张整图记录。

    返回值：
        dict[str,list[ImageRecord]]：三个互斥数据分区。
    """

    path = config["paths"]["prepared_root"] / "folds" / f"fold_{fold:02d}" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_name = {record.image_path.name: record for record in records}
    return {
        partition: [by_name[name] for name in names]
        for partition, names in payload["partitions"].items()
    }


class DetrVocDataset:
    """
    方法作用：
        将ImageRecord适配为Hugging Face DETR所需的COCO风格单类检测样本。

    输入参数：
        records (list[ImageRecord])：I张图，每张含boxes (N_i,4)。

    返回值：
        DetrVocDataset：按下标返回PIL整图与COCO annotations。
    """

    def __init__(self, records: list[ImageRecord]) -> None:
        """
        方法作用：
            保存本数据分区的原图记录。

        输入参数：
            records (list[ImageRecord])：I张训练或验证整图。

        返回值：
            None：初始化对象状态。
        """

        self.records = records

    def __len__(self) -> int:
        """
        方法作用：
            返回当前分区的原图数量。

        输入参数：
            无。

        返回值：
            int：图片数量I。
        """

        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        """
        方法作用：
            读取一张RGB原图并将N个xyxy框转换为COCO xywh标注。

        输入参数：
            index (int)：范围0到I-1的样本下标。

        返回值：
            tuple：RGB PIL图像(H,W,3)与含N个annotation的目标字典。
        """

        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        annotations = []
        for x1, y1, x2, y2 in record.boxes.tolist():
            width = x2 - x1
            height = y2 - y1
            annotations.append(
                {
                    "image_id": index,
                    "category_id": 0,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
        return image, {"image_id": index, "annotations": annotations}


class DetrCollator:
    """
    方法作用：
        使用DETR图像处理器批量缩放、补边并编码检测标签。

    输入参数：
        processor：本地DETR AutoImageProcessor。

    返回值：
        DetrCollator：可作为DataLoader collate_fn的对象。
    """

    def __init__(self, processor: Any) -> None:
        """
        方法作用：
            保存冻结的图像预处理器。

        输入参数：
            processor (Any)：DETR图像和检测标注处理器。

        返回值：
            None：初始化对象状态。
        """

        self.processor = processor

    def __call__(self, batch: list[tuple[Image.Image, dict[str, Any]]]) -> dict[str, Any]:
        """
        方法作用：
            将B张可变尺寸图像编码为pixel_values (B,3,H_pad,W_pad)和B个标签字典。

        输入参数：
            batch：B个(PIL图像, COCO目标)样本。

        返回值：
            dict[str,Any]：含pixel_values、pixel_mask和labels的训练批次。
        """

        images, annotations = zip(*batch)
        try:
            return self.processor(
                images=list(images), annotations=list(annotations), return_tensors="pt"
            )
        finally:
            for image in images:
                image.close()


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """
    方法作用：
        将DETR输入张量和每图目标字典递归移动到训练设备。

    输入参数：
        batch：pixel_values形状(B,3,H,W)、pixel_mask(B,H,W)及B个labels；device：设备。

    返回值：
        dict[str,Any]：位于目标设备、保持相同形状的批次。
    """

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [
                {name: tensor.to(device) for name, tensor in item.items()}
                for item in value
            ]
        else:
            moved[key] = value.to(device)
    return moved


def _build_detr_loader(
    records: list[ImageRecord],
    processor: Any,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> Any:
    """
    方法作用：
        创建可复现的DETR训练或验证DataLoader。

    输入参数：
        records：I张整图；processor：DETR处理器；batch_size：批大小B；workers：进程数；
        shuffle：是否打乱；seed：生成器种子。

    返回值：
        DataLoader：每批pixel_values形状(B,3,H_pad,W_pad)。
    """

    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        DetrVocDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=DetrCollator(processor),
        generator=generator,
        # GPU已被其他任务占满时，即使本次指定CPU，pin_memory也可能触发CUDA初始化。
        # 该数据集规模较小，关闭锁页内存换取CPU/GPU两种启动方式的一致稳定性。
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def _load_single_class_detr(model_path: Path, class_name: str, device: str) -> Any:
    """
    方法作用：
        从本地COCO预训练权重构造单一military_vehicle类别的DETR检测器。

    输入参数：
        model_path (Path)：DETR本地目录；class_name (str)：唯一前景类别；device (str)：设备。

    返回值：
        Any：分类头为1个前景类加背景类的DetrForObjectDetection。
    """

    from transformers import AutoConfig, AutoModelForObjectDetection

    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    model_config.id2label = {0: class_name}
    model_config.label2id = {class_name: 0}
    return AutoModelForObjectDetection.from_pretrained(
        str(model_path),
        config=model_config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    ).to(device)


def _validation_loss(model: Any, loader: Any, device: str) -> float:
    """
    方法作用：
        在Validation整图上计算平均DETR总损失，用于早停和最佳权重选择。

    输入参数：
        model：DETR模型；loader：V个验证批次；device：推理设备。

    返回值：
        float：按批样本数加权的平均验证损失。
    """

    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            output = model(**moved)
            batch_count = len(moved["labels"])
            total += float(output.loss.detach().cpu()) * batch_count
            count += batch_count
    return total / max(count, 1)


def train_detr_fold(
    config: dict[str, Any],
    fold: int,
    partitions: dict[str, list[ImageRecord]],
    device: str,
    fold_root: Path,
) -> tuple[Path, float, list[dict[str, float]]]:
    """
    方法作用：
        从COCO预训练DETR-R50开始，在单折Train上微调并按Validation损失保存最佳权重。

    输入参数：
        config：阶段2配置；fold：折号；partitions：Train/Validation/Test整图；
        device：训练设备；fold_root：本折独立输出目录。

    返回值：
        tuple：最佳本地模型目录、训练秒数、逐Epoch损失历史。
    """

    import torch
    from transformers import AutoImageProcessor

    settings = config["detr"]
    fold_root.mkdir(parents=True, exist_ok=True)
    model_path = config["paths"]["detr_model"]
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    class_name = str(config["evaluation"]["class_name"])
    model = _load_single_class_detr(model_path, class_name, device)
    train_loader = _build_detr_loader(
        partitions["train"],
        processor,
        batch_size=int(settings["batch_size"]),
        workers=int(settings["workers"]),
        shuffle=True,
        seed=int(config["split"]["seed"]) + fold,
    )
    validation_loader = _build_detr_loader(
        partitions["validation"],
        processor,
        batch_size=int(settings["eval_batch_size"]),
        workers=int(settings["workers"]),
        shuffle=False,
        seed=int(config["split"]["seed"]) + fold,
    )
    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_parameters if "backbone" in name else other_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": float(settings["learning_rate"])},
            {"params": backbone_parameters, "lr": float(settings["backbone_learning_rate"])},
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["epochs"])
    )
    best_dir = fold_root / "best_model"
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    began = time.perf_counter()
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            moved = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**moved)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite DETR loss at fold {fold}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["max_grad_norm"]))
            optimizer.step()
            batch_count = len(moved["labels"])
            running += float(loss.detach().cpu()) * batch_count
            seen += batch_count
        scheduler.step()
        train_loss = running / max(seen, 1)
        validation_loss = _validation_loss(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"DETR fold {fold:02d} epoch {epoch:03d}: "
            f"train_loss={train_loss:.6f} val_loss={validation_loss:.6f}",
            flush=True,
        )
        (fold_root / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            stale_epochs = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["patience"]):
            print(f"DETR fold {fold:02d}: early stop at epoch {epoch}", flush=True)
            break
    elapsed = time.perf_counter() - began
    if not (best_dir / "config.json").is_file():
        raise RuntimeError(f"DETR fold {fold} did not save a best model")
    return best_dir, elapsed, history


def train_yolo_fold(
    config: dict[str, Any],
    fold: int,
    device: str,
    fold_root: Path,
) -> tuple[Path, float, int]:
    """
    方法作用：
        从当前通用YOLO26n权重开始，在单折单类别数据上微调并返回best.pt。

    输入参数：
        config：阶段2配置；fold：折号；device：训练设备；fold_root：独立输出目录。

    返回值：
        tuple：最佳YOLO权重路径、训练秒数和实际完成Epoch数。
    """

    from ultralytics import YOLO

    settings = config["yolo"]
    data_yaml = config["paths"]["prepared_root"] / "folds" / f"fold_{fold:02d}" / "data.yaml"
    model = YOLO(str(config["paths"]["yolo_model"]))
    began = time.perf_counter()
    result = model.train(
        data=str(data_yaml),
        epochs=int(settings["epochs"]),
        patience=int(settings["patience"]),
        imgsz=int(settings["image_size"]),
        batch=int(settings["batch_size"]),
        workers=int(settings["workers"]),
        device=device,
        seed=int(config["split"]["seed"]) + fold,
        deterministic=True,
        project=str(fold_root),
        name="train",
        exist_ok=True,
        pretrained=True,
        plots=False,
        verbose=True,
    )
    elapsed = time.perf_counter() - began
    save_dir = Path(getattr(result, "save_dir", fold_root / "train"))
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        best_path = fold_root / "train" / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"YOLO best.pt not found below {fold_root}")
    results_csv = best_path.parent.parent / "results.csv"
    trained_epochs = 0
    if results_csv.is_file():
        trained_epochs = max(
            sum(1 for line in results_csv.read_text(encoding="utf-8").splitlines() if line.strip()) - 1,
            0,
        )
    return best_path.resolve(), elapsed, trained_epochs


def _parse_folds(value: str) -> list[int]:
    """
    方法作用：
        将all或逗号分隔文本转换为1到5的折号列表。

    输入参数：
        value (str)：例如all、1或1,3,5。

    返回值：
        list[int]：去重后的折号。
    """

    if value.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds = []
    for part in value.split(","):
        fold = int(part.strip())
        if fold not in range(1, 6):
            raise ValueError("fold must be all or comma-separated 1..5")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError("At least one fold is required")
    return folds


def _write_predictions(
    path: Path,
    records: list[ImageRecord],
    predictions: dict[str, list[dict[str, Any]]],
) -> None:
    """
    方法作用：
        保存单折Test逐图真实框和模型预测，便于复核检测结果。

    输入参数：
        path：JSONL路径；records：I张Test图及boxes (N_i,4)；predictions：每图预测。

    返回值：
        None：逐图结果写入磁盘。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = {
                "image": str(record.image_path),
                "ground_truth": [
                    {"box": box.tolist(), "class_name": name}
                    for box, name in zip(record.boxes, record.class_names)
                ],
                "predictions": predictions.get(record.image_path.name, []),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _metrics_from_predictions(
    config: dict[str, Any],
    records: list[ImageRecord],
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    方法作用：
        使用与阶段1完全相同的定义计算单折Test检测指标。

    输入参数：
        config：含IoU和分数阈值；records：I张Test整图；predictions：逐图预测。

    返回值：
        dict[str,Any]：AP、Precision、Recall、F1和逐车型Recall。
    """

    settings = config["evaluation"]
    return evaluate_predictions(
        records,
        predictions,
        iou_thresholds=[float(value) for value in settings["iou_thresholds"]],
        operating_threshold=float(settings["operating_score_threshold"]),
        small_object_area=float(settings["small_object_area"]),
    )


def _seed_everything(seed: int) -> None:
    """
    方法作用：
        固定Python、NumPy和PyTorch随机状态以提高每折复现性。

    输入参数：
        seed (int)：本折随机种子。

    返回值：
        None：随机状态已更新。
    """

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_results(detector_root: Path, detector: str) -> dict[str, Any]:
    """
    方法作用：
        读取已完成折结果并计算检测器的折间均值和标准差。

    输入参数：
        detector_root (Path)：yolo或detr输出根目录；detector (str)：模型名。

    返回值：
        dict[str,Any]：逐折结果及主要指标mean/std。
    """

    rows = []
    for path in sorted(detector_root.glob("fold_*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise FileNotFoundError(f"No completed folds in {detector_root}")
    getters = {
        "ap50": lambda row: row["metrics"]["ap_50"],
        "map50_95": lambda row: row["metrics"]["map_50_95"],
        "precision_at_030": lambda row: row["metrics"]["operating_point"]["precision"],
        "recall_at_030": lambda row: row["metrics"]["operating_point"]["recall"],
        "f1_at_030": lambda row: row["metrics"]["operating_point"]["f1"],
        "fp_per_image_at_030": lambda row: row["metrics"]["operating_point"]["false_positives_per_image"],
        "milliseconds_per_image": lambda row: row["runtime"]["milliseconds_per_image"],
    }
    aggregate = {}
    for name, getter in getters.items():
        values = np.asarray([float(getter(row)) for row in rows], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    summary = {
        "detector": detector,
        "training_performed": True,
        "completed_fold_count": len(rows),
        "folds": rows,
        "aggregate": aggregate,
    }
    (detector_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_detector(args: argparse.Namespace) -> Path:
    """
    方法作用：
        在指定设备上顺序完成一个检测器的所选五折训练、Test推理和汇总。

    输入参数：
        args：detector、fold、device、run_root及resume等CLI参数。

    返回值：
        Path：本检测器输出根目录。
    """

    import torch

    config = load_finetune_config(args.config)
    prepare_dataset(config)
    records = load_dataset(config["paths"]["image_root"])
    run_root = _resolve_path(args.run_root)
    detector_root = run_root / args.detector
    detector_root.mkdir(parents=True, exist_ok=True)
    snapshot = {
        **config,
        "paths": {key: str(value) for key, value in config["paths"].items()},
        "config_path": str(config["config_path"]),
        "runtime": {"detector": args.detector, "device": args.device, "fold": args.fold},
    }
    (detector_root / "resolved_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    class_name = _normalize_label(str(config["evaluation"]["class_name"]))
    for fold in _parse_folds(args.fold):
        fold_root = detector_root / f"fold_{fold:02d}"
        result_path = fold_root / "result.json"
        if result_path.is_file() and args.resume:
            print(f"{args.detector.upper()} fold {fold:02d}: completed, skip", flush=True)
            continue
        fold_root.mkdir(parents=True, exist_ok=True)
        partitions = _load_fold(config, fold, records)
        _seed_everything(int(config["split"]["seed"]) + fold)
        if args.detector == "yolo":
            checkpoint, training_seconds, trained_epochs = train_yolo_fold(
                config, fold, args.device, fold_root
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predictions, runtime = run_yolo_inference(
                partitions["test"],
                checkpoint,
                {class_name},
                args.device,
                batch_size=int(config["yolo"]["batch_size"]),
                image_size=int(config["yolo"]["image_size"]),
                score_floor=float(config["evaluation"]["ap_score_floor"]),
                max_detections=int(config["evaluation"]["max_detections_per_image"]),
            )
            history = None
        else:
            checkpoint, training_seconds, history = train_detr_fold(
                config, fold, partitions, args.device, fold_root
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            predictions, runtime = run_detr_inference(
                partitions["test"],
                checkpoint,
                {class_name},
                args.device,
                batch_size=int(config["detr"]["eval_batch_size"]),
                score_floor=float(config["evaluation"]["ap_score_floor"]),
                max_detections=int(config["evaluation"]["max_detections_per_image"]),
            )
        metrics = _metrics_from_predictions(config, partitions["test"], predictions)
        _write_predictions(fold_root / "test_predictions.jsonl", partitions["test"], predictions)
        weight_path = checkpoint if checkpoint.is_file() else checkpoint / "model.safetensors"
        result = {
            "detector": args.detector,
            "fold": fold,
            "training_performed": True,
            "partition_counts": {key: len(value) for key, value in partitions.items()},
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(weight_path),
            "training_seconds": training_seconds,
            "trained_epochs": len(history) if history is not None else trained_epochs,
            "runtime": runtime,
            "metrics": metrics,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{args.detector.upper()} fold {fold:02d}: "
            f"AP50={metrics['ap_50']:.4f} mAP50:95={metrics['map_50_95']:.4f} "
            f"Recall@0.30={metrics['operating_point']['recall']:.4f}",
            flush=True,
        )
    _aggregate_results(detector_root, args.detector)
    return detector_root


def validate_preparation(config: dict[str, Any], device: str) -> dict[str, Any]:
    """
    方法作用：
        在不保存训练权重的情况下验证YOLO数据、DETR单类标签编码和一次前后向传播。

    输入参数：
        config：阶段2配置；device：通常为cpu的验证设备。

    返回值：
        dict[str,Any]：数据计数、YOLO类别和有限DETR损失检查结果。
    """

    from ultralytics import YOLO
    summary = prepare_dataset(config)
    records = load_dataset(config["paths"]["image_root"])
    partitions = _load_fold(config, 1, records)
    yolo = YOLO(str(config["paths"]["yolo_model"]))
    yolo_names = {str(key): str(value) for key, value in yolo.model.names.items()}
    smoke_root = config["paths"]["prepared_root"] / "smoke_validation"
    smoke_train = partitions["train"][:8]
    smoke_validation = partitions["validation"][:4]
    smoke_train_list = smoke_root / "train.txt"
    smoke_validation_list = smoke_root / "validation.txt"
    _write_text_checked(
        smoke_train_list,
        "\n".join(
            str(config["paths"]["prepared_root"] / "images" / record.image_path.name)
            for record in smoke_train
        )
        + "\n",
    )
    _write_text_checked(
        smoke_validation_list,
        "\n".join(
            str(config["paths"]["prepared_root"] / "images" / record.image_path.name)
            for record in smoke_validation
        )
        + "\n",
    )
    smoke_yaml = smoke_root / "data.yaml"
    _write_text_checked(
        smoke_yaml,
        json.dumps(
            {
                "path": str(config["paths"]["prepared_root"]),
                "train": str(smoke_train_list.resolve()),
                "val": str(smoke_validation_list.resolve()),
                "names": {0: str(config["evaluation"]["class_name"])},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    yolo_result = yolo.train(
        data=str(smoke_yaml),
        epochs=1,
        patience=1,
        imgsz=160,
        batch=2,
        workers=0,
        device=device,
        seed=int(config["split"]["seed"]),
        deterministic=True,
        project=str(smoke_root),
        name="train_output",
        exist_ok=True,
        pretrained=True,
        plots=False,
        verbose=False,
    )
    yolo_smoke_save_dir = Path(getattr(yolo_result, "save_dir", smoke_root / "train_output"))
    yolo_smoke_best = yolo_smoke_save_dir / "weights" / "best.pt"
    if not yolo_smoke_best.is_file():
        yolo_smoke_best = smoke_root / "train_output" / "weights" / "best.pt"
    if not yolo_smoke_best.is_file():
        raise FileNotFoundError("YOLO smoke training did not produce best.pt")

    class_name = str(config["evaluation"]["class_name"])
    smoke_config = deepcopy(config)
    smoke_config["detr"].update(
        {
            "epochs": 1,
            "patience": 1,
            "batch_size": 1,
            "eval_batch_size": 1,
            "workers": 0,
        }
    )
    smoke_partitions = {
        "train": partitions["train"][:2],
        "validation": partitions["validation"][:1],
        "test": partitions["test"][:1],
    }
    detr_smoke_root = smoke_root / "detr_train_output"
    detr_checkpoint, _, detr_history = train_detr_fold(
        smoke_config, 1, smoke_partitions, device, detr_smoke_root
    )
    detr_predictions, _ = run_detr_inference(
        smoke_partitions["test"],
        detr_checkpoint,
        {_normalize_label(class_name)},
        device,
        batch_size=1,
        score_floor=float(config["evaluation"]["ap_score_floor"]),
        max_detections=int(config["evaluation"]["max_detections_per_image"]),
    )
    loss = float(detr_history[-1]["train_loss"])
    if not math.isfinite(loss):
        raise RuntimeError("DETR smoke training produced non-finite loss")
    result = {
        "status": "ok",
        "source_weights_updated": False,
        "formal_training_performed": False,
        "prepared_summary": summary,
        "fold_01_counts": {key: len(value) for key, value in partitions.items()},
        "yolo_source_classes": yolo_names,
        "yolo_smoke_training": {
            "images": len(smoke_train),
            "validation_images": len(smoke_validation),
            "epochs": 1,
            "image_size": 160,
            "best_checkpoint": str(yolo_smoke_best.resolve()),
        },
        "detr_smoke_training": {
            "images": len(smoke_partitions["train"]),
            "validation_images": len(smoke_partitions["validation"]),
            "test_images": len(smoke_partitions["test"]),
            "epochs": len(detr_history),
            "best_checkpoint": str(detr_checkpoint.resolve()),
            "train_loss": loss,
            "test_prediction_count": sum(len(rows) for rows in detr_predictions.values()),
        },
    }
    path = config["paths"]["prepared_root"] / "validation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def summarize_run(config: dict[str, Any], run_root: Path) -> Path:
    """
    方法作用：
        合并YOLO和DETR五折汇总并生成可直接阅读的Markdown表格。

    输入参数：
        config：阶段2配置；run_root：同时包含yolo和detr子目录的Run。

    返回值：
        Path：生成的Markdown报告路径。
    """

    summaries = {}
    for detector in ("yolo", "detr"):
        path = run_root / detector / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing detector summary: {path}")
        summaries[detector] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "experiment": config["name"],
        "training_performed": True,
        "run_root": str(run_root),
        "detectors": summaries,
    }
    (run_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 实验1阶段2：YOLO与DETR五折微调结果",
        "",
        "两模型从各自通用预训练权重开始，在相同图像级划分上微调；10种车型统一为 `military_vehicle`。",
        "",
        "| 模型 | 完成折数 | AP50 | mAP50:95 | Precision@0.30 | Recall@0.30 | F1@0.30 | ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for detector in ("yolo", "detr"):
        summary = summaries[detector]
        aggregate = summary["aggregate"]
        lines.append(
            f"| {detector.upper()} | {summary['completed_fold_count']} | "
            f"{aggregate['ap50']['mean']:.4f} | {aggregate['map50_95']['mean']:.4f} | "
            f"{aggregate['precision_at_030']['mean']:.4f} | "
            f"{aggregate['recall_at_030']['mean']:.4f} | "
            f"{aggregate['f1_at_030']['mean']:.4f} | "
            f"{aggregate['milliseconds_per_image']['mean']:.2f} |"
        )
    report = run_root / "RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建阶段2数据准备、链路验证、单模型训练和汇总子命令。

    输入参数：
        无。

    返回值：
        argparse.ArgumentParser：包含prepare、validate、run、summarize的解析器。
    """

    parser = argparse.ArgumentParser(description="Five-fold YOLO/DETR domain fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "run", "summarize"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name == "validate":
            child.add_argument("--device", default="cpu")
        elif name == "run":
            child.add_argument("--detector", choices=("yolo", "detr"), required=True)
            child.add_argument("--fold", default="all")
            child.add_argument("--device", default="cuda:0")
            child.add_argument("--run-root", type=Path, required=True)
            child.add_argument("--resume", action="store_true")
        elif name == "summarize":
            child.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        按CLI子命令执行阶段2准备、验证、训练或结果汇总。

    输入参数：
        argv (list[str]|None)：可选命令行参数。

    返回值：
        int：成功时为0。
    """

    args = build_parser().parse_args(argv)
    config = load_finetune_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(config), ensure_ascii=False, indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_preparation(config, args.device), ensure_ascii=False, indent=2))
    elif args.command == "run":
        print(f"Detector results: {run_detector(args)}")
    else:
        report = summarize_run(config, _resolve_path(args.run_root))
        print(f"Combined report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
