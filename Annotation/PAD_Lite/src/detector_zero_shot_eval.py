from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT


DEFAULT_CONFIG = PAD_LITE_ROOT / "configs" / "detector_zero_shot_yolo_detr.json"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """
    方法作用：
        保存一张原始整图及其 Pascal VOC 标注，作为检测评测的最小数据单元。

    输入参数：
        image_path (Path)：原始 RGB 图片路径。
        width、height (int)：原图宽高。
        boxes (np.ndarray)：全部真实框，形状为 (N, 4)，坐标顺序为 xyxy。
        class_names (tuple[str,...])：N 个真实框对应的原始车型名。

    返回值：
        ImageRecord：不可变的图像评测记录。
    """

    image_path: Path
    width: int
    height: int
    boxes: np.ndarray
    class_names: tuple[str, ...]


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：
        将配置中的相对路径按 Annotation 根目录解析为绝对路径。

    输入参数：
        value (str|Path)：配置文件中的路径值。

    返回值：
        Path：解析后的绝对路径。
    """

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：
        读取零样本检测配置，校验字段并解析数据、模型和输出路径。

    输入参数：
        path (str|Path)：实验 JSON 配置路径。

    返回值：
        dict[str,Any]：可直接用于实验的配置字典。
    """

    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Detector evaluation schema_version must be 1")
    for section in ("paths", "evaluation"):
        if section not in payload:
            raise ValueError(f"Detector evaluation config is missing {section}")
    for key in ("image_root", "yolo_model", "detr_model", "output_root"):
        if key not in payload["paths"]:
            raise ValueError(f"Detector evaluation config is missing paths.{key}")
        payload["paths"][key] = _resolve_path(payload["paths"][key])
    payload["config_path"] = config_path
    return payload


def _parse_voc_record(xml_path: Path) -> ImageRecord:
    """
    方法作用：
        解析一个 Pascal VOC XML，并读取对应原始图片的全部有效目标框。

    输入参数：
        xml_path (Path)：单张图片对应的 XML 路径。

    返回值：
        ImageRecord：boxes 形状为 (N,4) 的整图记录。
    """

    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or f"{xml_path.stem}.jpg").strip()
    image_path = xml_path.parent / filename
    if not image_path.is_file():
        candidates = list(xml_path.parent.glob(f"{xml_path.stem}.*"))
        candidates = [item for item in candidates if item.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot resolve image for {xml_path}")
        image_path = candidates[0]

    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0
    if width <= 0 or height <= 0:
        with Image.open(image_path) as image:
            width, height = image.size

    boxes: list[list[float]] = []
    names: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        coords = [
            float(box.findtext(key, "nan"))
            for key in ("xmin", "ymin", "xmax", "ymax")
        ]
        x1, y1, x2, y2 = coords
        x1 = min(max(x1, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        x2 = min(max(x2, 0.0), float(width))
        y2 = min(max(y2, 0.0), float(height))
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        names.append(name)
    if not boxes:
        raise ValueError(f"No valid object boxes in {xml_path}")
    return ImageRecord(
        image_path=image_path.resolve(),
        width=width,
        height=height,
        boxes=np.asarray(boxes, dtype=np.float32),
        class_names=tuple(names),
    )


def load_dataset(image_root: Path) -> list[ImageRecord]:
    """
    方法作用：
        从原始 train 目录加载所有一一对应的 JPG/XML 整图检测记录。

    输入参数：
        image_root (Path)：Russian-Military-Vehicles/train 目录。

    返回值：
        list[ImageRecord]：按文件名稳定排序的整图记录列表。
    """

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    xml_paths = sorted(image_root.glob("*.xml"), key=lambda item: item.name.casefold())
    if not xml_paths:
        raise FileNotFoundError(f"No XML annotations found in {image_root}")
    records = [_parse_voc_record(path) for path in xml_paths]
    image_names = [record.image_path.name.casefold() for record in records]
    if len(set(image_names)) != len(image_names):
        raise ValueError("Duplicate source image names found in XML annotations")
    return records


def select_smoke_records(records: list[ImageRecord], limit: int) -> list[ImageRecord]:
    """
    方法作用：
        从全量数据的排序范围中等距抽取少量图片，使冒烟测试覆盖多个车型前缀。

    输入参数：
        records (list[ImageRecord])：完整整图记录列表。
        limit (int)：冒烟测试最大图片数。

    返回值：
        list[ImageRecord]：不重复、保持原排序的抽样记录。
    """

    if limit <= 0 or limit >= len(records):
        return records
    indices = np.linspace(0, len(records) - 1, num=limit, dtype=np.int64)
    return [records[int(index)] for index in sorted(set(indices.tolist()))]


def _normalize_label(value: str) -> str:
    """
    方法作用：
        统一不同模型类别文本的大小写、下划线和空白表示。

    输入参数：
        value (str)：模型类别名称。

    返回值：
        str：小写且使用单空格的类别名。
    """

    return " ".join(str(value).replace("_", " ").strip().lower().split())


def _device_text(device: str) -> str:
    """
    方法作用：
        将 auto 设备请求解析为明确的 cpu 或 cuda:0 字符串。

    输入参数：
        device (str)：用户请求的设备。

    返回值：
        str：PyTorch与Ultralytics可使用的设备字符串。
    """

    import torch

    if device.strip().lower() != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _synchronize(device: str) -> None:
    """
    方法作用：
        在CUDA推理计时边界同步设备；CPU模式不执行操作。

    输入参数：
        device (str)：当前推理设备。

    返回值：
        None：仅同步执行状态。
    """

    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def run_yolo_inference(
    records: list[ImageRecord],
    model_path: Path,
    candidate_names: set[str],
    device: str,
    batch_size: int,
    image_size: int,
    score_floor: float,
    max_detections: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """
    方法作用：
        使用当前Annotation的通用YOLO权重进行零样本候选框推理，不更新参数。

    输入参数：
        records：B张整图记录；model_path：YOLO权重；candidate_names：车辆类别白名单；
        device：推理设备；batch_size：每批B；image_size：YOLO输入边长；
        score_floor：AP预测最低分；max_detections：每图最大框数。

    返回值：
        tuple：图片名到预测框列表的映射，其中每个box为长度4的xyxy；以及速度和模型信息。
    """

    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    model = YOLO(str(model_path))
    names = getattr(model.model, "names", {}) or {}
    allowed_ids = [
        int(class_id)
        for class_id, label in names.items()
        if _normalize_label(str(label)) in candidate_names
    ]
    if not allowed_ids:
        raise ValueError(f"YOLO contains none of candidate classes: {sorted(candidate_names)}")

    predictions: dict[str, list[dict[str, Any]]] = {}
    elapsed = 0.0
    for start in tqdm(range(0, len(records), batch_size), desc="YOLO zero-shot"):
        batch = records[start : start + batch_size]
        _synchronize(device)
        began = time.perf_counter()
        results = model.predict(
            source=[str(record.image_path) for record in batch],
            imgsz=image_size,
            conf=score_floor,
            classes=allowed_ids,
            max_det=max_detections,
            device=device,
            batch=len(batch),
            verbose=False,
        )
        _synchronize(device)
        elapsed += time.perf_counter() - began
        for record, result in zip(batch, results):
            rows: list[dict[str, Any]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                labels = boxes.cls.detach().cpu().numpy().astype(int)
                result_names = getattr(result, "names", names) or names
                for box, score, label_id in zip(xyxy, scores, labels):
                    rows.append(
                        {
                            "box": [float(value) for value in box.tolist()],
                            "score": float(score),
                            "source_class_id": int(label_id),
                            "source_class_name": str(result_names.get(int(label_id), label_id)),
                        }
                    )
            predictions[record.image_path.name] = rows
    return predictions, {
        "model_path": str(model_path),
        "source_classes": {str(key): str(value) for key, value in names.items()},
        "allowed_class_ids": allowed_ids,
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / max(len(records), 1),
        "preprocessing": f"Ultralytics letterbox imgsz={image_size}",
    }


def run_detr_inference(
    records: list[ImageRecord],
    model_path: Path,
    candidate_names: set[str],
    device: str,
    batch_size: int,
    score_floor: float,
    max_detections: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """
    方法作用：
        使用本地COCO预训练DETR进行零样本整图推理，并筛选与Annotation相同的车辆类别。

    输入参数：
        records：B张整图记录；model_path：DETR本地目录；candidate_names：车辆白名单；
        device：推理设备；batch_size：每批B；score_floor：AP最低分；
        max_detections：每图保留的最高分框数。

    返回值：
        tuple：图片名到预测列表的映射，box形状为(4,)；以及速度和模型信息。
    """

    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if not model_path.is_dir():
        raise FileNotFoundError(f"DETR model directory not found: {model_path}")
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device)
    model.eval()
    id2label = {
        int(key): str(value) for key, value in dict(model.config.id2label).items()
    }
    allowed_ids = {
        class_id
        for class_id, label in id2label.items()
        if _normalize_label(label) in candidate_names
    }
    if not allowed_ids:
        raise ValueError(f"DETR contains none of candidate classes: {sorted(candidate_names)}")

    predictions: dict[str, list[dict[str, Any]]] = {}
    elapsed = 0.0
    for start in tqdm(range(0, len(records), batch_size), desc="DETR zero-shot"):
        batch = records[start : start + batch_size]
        images: list[Image.Image] = []
        try:
            for record in batch:
                with Image.open(record.image_path) as image:
                    images.append(image.convert("RGB"))
            _synchronize(device)
            began = time.perf_counter()
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor(
                [[record.height, record.width] for record in batch],
                dtype=torch.int64,
                device=device,
            )
            processed = processor.post_process_object_detection(
                outputs,
                threshold=score_floor,
                target_sizes=target_sizes,
            )
            _synchronize(device)
            elapsed += time.perf_counter() - began
        finally:
            for image in images:
                image.close()

        for record, result in zip(batch, processed):
            rows = []
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            labels = result["labels"].detach().cpu().numpy().astype(int)
            order = np.argsort(-scores)
            for index in order:
                label_id = int(labels[index])
                if label_id not in allowed_ids:
                    continue
                rows.append(
                    {
                        "box": [float(value) for value in boxes[index].tolist()],
                        "score": float(scores[index]),
                        "source_class_id": label_id,
                        "source_class_name": id2label.get(label_id, str(label_id)),
                    }
                )
                if len(rows) >= max_detections:
                    break
            predictions[record.image_path.name] = rows
    return predictions, {
        "model_path": str(model_path),
        "source_classes": {str(key): value for key, value in id2label.items()},
        "allowed_class_ids": sorted(allowed_ids),
        "elapsed_seconds": elapsed,
        "milliseconds_per_image": 1000.0 * elapsed / max(len(records), 1),
        "preprocessing": processor.to_json_string(),
    }


def box_iou(box: Iterable[float], boxes: np.ndarray) -> np.ndarray:
    """
    方法作用：
        计算单个预测框与N个真实框之间的交并比。

    输入参数：
        box (Iterable[float])：单个xyxy框，形状(4,)；boxes (np.ndarray)：形状(N,4)。

    返回值：
        np.ndarray：N个IoU值，形状(N,)。
    """

    candidate = np.asarray(list(box), dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    left_top = np.maximum(candidate[:2], boxes[:, :2])
    right_bottom = np.minimum(candidate[2:], boxes[:, 2:])
    intersection_size = np.maximum(right_bottom - left_top, 0.0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    candidate_area = max(float(candidate[2] - candidate[0]), 0.0) * max(
        float(candidate[3] - candidate[1]), 0.0
    )
    target_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    union = candidate_area + target_area - intersection
    return intersection / np.maximum(union, 1e-12)


def _pair_overlap_quality(
    prediction_box: Iterable[float], ground_truth_box: Iterable[float]
) -> dict[str, float]:
    """
    方法作用：
        计算单个预测框相对单个GT框的IoU、目标覆盖率和裁剪纯净度。

    输入参数：
        prediction_box (Iterable[float])：预测xyxy框，形状(4,)；
        ground_truth_box (Iterable[float])：XML真实xyxy框，形状(4,)。

    返回值：
        dict[str,float]：intersection、iou、target_coverage和crop_purity。
    """

    prediction = np.asarray(list(prediction_box), dtype=np.float64)
    target = np.asarray(list(ground_truth_box), dtype=np.float64)
    left_top = np.maximum(prediction[:2], target[:2])
    right_bottom = np.minimum(prediction[2:], target[2:])
    intersection_size = np.maximum(right_bottom - left_top, 0.0)
    intersection = float(intersection_size[0] * intersection_size[1])
    prediction_area = max(float(prediction[2] - prediction[0]), 0.0) * max(
        float(prediction[3] - prediction[1]), 0.0
    )
    target_area = max(float(target[2] - target[0]), 0.0) * max(
        float(target[3] - target[1]), 0.0
    )
    union = prediction_area + target_area - intersection
    return {
        "intersection": intersection,
        "iou": intersection / max(union, 1e-12),
        "target_coverage": intersection / max(target_area, 1e-12),
        "crop_purity": intersection / max(prediction_area, 1e-12),
    }


def _match_prediction_details(
    record: ImageRecord,
    rows: list[dict[str, Any]],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, Any]:
    """
    方法作用：
        按置信度执行预测框与GT的一对一贪心匹配，并保留每对框的重合质量。

    输入参数：
        record：含GT boxes (N,4)的整图；rows：M个预测；
        iou_threshold：成功匹配IoU；score_threshold：最低预测置信度。

    返回值：
        dict[str,Any]：TP/FP/FN、候选框数、匹配GT集合和逐匹配框质量。
    """

    matched: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positive = 0
    candidate_count = 0
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        if float(row["score"]) < score_threshold:
            continue
        candidate_count += 1
        overlaps = box_iou(row["box"], record.boxes)
        available = [index for index in range(len(overlaps)) if index not in matched]
        if not available:
            false_positive += 1
            continue
        best = max(available, key=lambda index: float(overlaps[index]))
        if float(overlaps[best]) < iou_threshold:
            false_positive += 1
            continue
        matched.add(best)
        quality = _pair_overlap_quality(row["box"], record.boxes[best])
        matches.append(
            {
                "ground_truth_index": best,
                "class_name": record.class_names[best],
                "score": float(row["score"]),
                **quality,
            }
        )
    return {
        "tp": len(matches),
        "fp": false_positive,
        "fn": len(record.boxes) - len(matched),
        "candidate_count": candidate_count,
        "matched": matched,
        "matches": matches,
    }


def _match_predictions(
    record: ImageRecord,
    rows: list[dict[str, Any]],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[int, int, int, set[int]]:
    """
    方法作用：
        按置信度从高到低将预测框与单张图的GT框一对一贪心匹配。

    输入参数：
        record：含GT boxes (N,4)的整图记录；rows：M个预测；
        iou_threshold：成功匹配IoU；score_threshold：运行点置信度阈值。

    返回值：
        tuple：TP、FP、FN，以及被匹配GT下标集合。
    """

    details = _match_prediction_details(
        record,
        rows,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    return details["tp"], details["fp"], details["fn"], details["matched"]


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    """
    方法作用：
        汇总匹配框质量分布，直接报告均值、中位数和低质量10%分位数。

    输入参数：
        values (list[float])：IoU、覆盖率或纯净度标量列表。

    返回值：
        dict：count、mean、median、p10、min和max；空列表对应值为None。
    """

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _average_precision(
    records: list[ImageRecord],
    predictions: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> float:
    """
    方法作用：
        按COCO式101点插值计算单类别检测AP。

    输入参数：
        records：I张图及总计N个GT框；predictions：每图M_i个预测；
        iou_threshold：本次AP匹配所用IoU阈值。

    返回值：
        float：范围0到1的101点插值AP。
    """

    by_name = {record.image_path.name: record for record in records}
    ranked = [
        (float(row["score"]), image_name, row)
        for image_name, rows in predictions.items()
        for row in rows
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    total_gt = sum(len(record.boxes) for record in records)
    if total_gt == 0:
        return 0.0
    matched = {name: set() for name in by_name}
    true_flags: list[float] = []
    false_flags: list[float] = []
    for _, image_name, row in ranked:
        record = by_name[image_name]
        overlaps = box_iou(row["box"], record.boxes)
        available = [index for index in range(len(overlaps)) if index not in matched[image_name]]
        if available:
            best = max(available, key=lambda index: float(overlaps[index]))
        else:
            best = -1
        is_true = best >= 0 and float(overlaps[best]) >= iou_threshold
        if is_true:
            matched[image_name].add(best)
        true_flags.append(float(is_true))
        false_flags.append(float(not is_true))
    if not true_flags:
        return 0.0
    cumulative_true = np.cumsum(np.asarray(true_flags, dtype=np.float64))
    cumulative_false = np.cumsum(np.asarray(false_flags, dtype=np.float64))
    recall = cumulative_true / float(total_gt)
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, 1e-12)
    samples = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in samples
    ]
    return float(np.mean(interpolated))


def evaluate_predictions(
    records: list[ImageRecord],
    predictions: dict[str, list[dict[str, Any]]],
    iou_thresholds: list[float],
    operating_threshold: float,
    small_object_area: float,
    target_coverage_threshold: float = 0.90,
    crop_purity_threshold: float = 0.50,
) -> dict[str, Any]:
    """
    方法作用：
        汇总零样本检测AP、默认阈值Precision/Recall及按原车型拆分的Recall。

    输入参数：
        records：I张图、N个GT框；predictions：每图预测框；iou_thresholds：AP阈值列表；
        operating_threshold：默认置信度运行点；small_object_area：小目标面积上界；
        target_coverage_threshold：车辆被完整保留的最低GT覆盖率；
        crop_purity_threshold：可用裁剪中车辆占预测框面积的最低比例。

    返回值：
        dict[str,Any]：目标找全率、整图成功率、框质量、误检、AP和逐车型召回率。
    """

    if not 0.0 < target_coverage_threshold <= 1.0:
        raise ValueError("target_coverage_threshold must be in (0, 1]")
    if not 0.0 < crop_purity_threshold <= 1.0:
        raise ValueError("crop_purity_threshold must be in (0, 1]")
    ap_by_iou = {
        f"{threshold:.2f}": _average_precision(records, predictions, threshold)
        for threshold in iou_thresholds
    }
    totals = {"tp": 0, "fp": 0, "fn": 0}
    totals_iou75 = {"tp": 0, "fp": 0, "fn": 0}
    per_class: dict[str, dict[str, int]] = {}
    small_total = 0
    small_matched = 0
    candidate_count = 0
    images_with_gt = 0
    all_detected_iou50 = 0
    all_detected_iou75 = 0
    all_complete_crop = 0
    all_usable_crop = 0
    zero_detected_images = 0
    complete_crop_count = 0
    usable_crop_count = 0
    matched_ious: list[float] = []
    matched_coverages: list[float] = []
    matched_purities: list[float] = []
    for record in records:
        details = _match_prediction_details(
            record,
            predictions.get(record.image_path.name, []),
            iou_threshold=0.5,
            score_threshold=operating_threshold,
        )
        details_iou75 = _match_prediction_details(
            record,
            predictions.get(record.image_path.name, []),
            iou_threshold=0.75,
            score_threshold=operating_threshold,
        )
        for key in totals:
            totals[key] += int(details[key])
            totals_iou75[key] += int(details_iou75[key])
        candidate_count += int(details["candidate_count"])
        matched = details["matched"]
        matches = details["matches"]
        matched_ious.extend(float(item["iou"]) for item in matches)
        matched_coverages.extend(float(item["target_coverage"]) for item in matches)
        matched_purities.extend(float(item["crop_purity"]) for item in matches)
        complete_indices = {
            int(item["ground_truth_index"])
            for item in matches
            if float(item["target_coverage"]) >= target_coverage_threshold
        }
        usable_indices = {
            int(item["ground_truth_index"])
            for item in matches
            if float(item["target_coverage"]) >= target_coverage_threshold
            and float(item["crop_purity"]) >= crop_purity_threshold
        }
        complete_crop_count += len(complete_indices)
        usable_crop_count += len(usable_indices)
        if len(record.boxes) > 0:
            images_with_gt += 1
            all_detected_iou50 += int(details["fn"] == 0)
            all_detected_iou75 += int(details_iou75["fn"] == 0)
            all_complete_crop += int(len(complete_indices) == len(record.boxes))
            all_usable_crop += int(len(usable_indices) == len(record.boxes))
            zero_detected_images += int(details["tp"] == 0)
        for index, class_name in enumerate(record.class_names):
            row = per_class.setdefault(class_name, {"gt": 0, "matched": 0})
            row["gt"] += 1
            if index in matched:
                row["matched"] += 1
            box = record.boxes[index]
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if area < small_object_area:
                small_total += 1
                if index in matched:
                    small_matched += 1
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    recall_iou75 = totals_iou75["tp"] / max(
        totals_iou75["tp"] + totals_iou75["fn"], 1
    )
    ground_truth_count = sum(len(record.boxes) for record in records)
    matched_count = len(matched_ious)
    return {
        "image_count": len(records),
        "ground_truth_count": ground_truth_count,
        "prediction_count_at_ap_floor": sum(len(rows) for rows in predictions.values()),
        "primary_crop_metrics": {
            "score_threshold": operating_threshold,
            "object_recall_iou50": recall,
            "object_recall_iou75": recall_iou75,
            "all_targets_detected_image_count_iou50": all_detected_iou50,
            "all_targets_detected_image_rate_iou50": all_detected_iou50
            / max(images_with_gt, 1),
            "all_targets_detected_image_count_iou75": all_detected_iou75,
            "all_targets_detected_image_rate_iou75": all_detected_iou75
            / max(images_with_gt, 1),
            "complete_crop_count": complete_crop_count,
            "complete_crop_recall": complete_crop_count / max(ground_truth_count, 1),
            "all_complete_crop_image_count": all_complete_crop,
            "all_complete_crop_image_rate": all_complete_crop / max(images_with_gt, 1),
            "usable_crop_count": usable_crop_count,
            "usable_crop_recall": usable_crop_count / max(ground_truth_count, 1),
            "all_usable_crop_image_count": all_usable_crop,
            "all_usable_crop_image_rate": all_usable_crop / max(images_with_gt, 1),
            "zero_detected_image_count": zero_detected_images,
            "zero_detected_image_rate": zero_detected_images / max(images_with_gt, 1),
        },
        "matched_box_quality_iou50": {
            "matched_box_count": matched_count,
            "iou": _distribution_summary(matched_ious),
            "target_coverage": _distribution_summary(matched_coverages),
            "crop_purity": _distribution_summary(matched_purities),
            "target_coverage_threshold": target_coverage_threshold,
            "crop_purity_threshold": crop_purity_threshold,
            "vehicle_cut_count": matched_count - complete_crop_count,
            "vehicle_cut_rate_among_matches": (matched_count - complete_crop_count)
            / max(matched_count, 1),
        },
        "ap_by_iou": ap_by_iou,
        "map_50_95": float(np.mean(list(ap_by_iou.values()))),
        "ap_50": ap_by_iou.get("0.50", 0.0),
        "ap_75": ap_by_iou.get("0.75", 0.0),
        "operating_point": {
            "score_threshold": operating_threshold,
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives_per_image": totals["fp"] / max(len(records), 1),
            "candidate_crop_count": candidate_count,
            "candidate_crops_per_image": candidate_count / max(len(records), 1),
            "valid_crops_per_image": totals["tp"] / max(len(records), 1),
            "small_object_gt": small_total,
            "small_object_recall": small_matched / max(small_total, 1),
        },
        "recall_by_original_vehicle_class": {
            name: {
                **counts,
                "recall": counts["matched"] / max(counts["gt"], 1),
            }
            for name, counts in sorted(per_class.items())
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        创建父目录并写入UTF-8格式化JSON结果。

    输入参数：
        path (Path)：输出路径；payload (Any)：可JSON序列化的数据。

    返回值：
        None：结果写入磁盘。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_predictions(
    path: Path,
    records: list[ImageRecord],
    predictions: dict[str, list[dict[str, Any]]],
) -> None:
    """
    方法作用：
        将逐图GT与预测写为JSON Lines，便于后续复核和可视化。

    输入参数：
        path (Path)：JSONL路径；records：每图GT boxes (N,4)；predictions：每图预测。

    返回值：
        None：逐图结果写入磁盘。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "image": str(record.image_path),
                "width": record.width,
                "height": record.height,
                "ground_truth": [
                    {"box": box.tolist(), "class_name": class_name}
                    for box, class_name in zip(record.boxes, record.class_names)
                ],
                "predictions": predictions.get(record.image_path.name, []),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    """
    方法作用：
        流式计算模型或配置文件的SHA-256，用于实验可复现记录。

    输入参数：
        path (Path)：普通文件路径。

    返回值：
        str：十六进制SHA-256摘要。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_weight_path(model_name: str, model_path: Path) -> Path:
    """
    方法作用：
        确定YOLO单文件权重或DETR目录中的实际权重文件。

    输入参数：
        model_name (str)：yolo或detr；model_path (Path)：模型路径。

    返回值：
        Path：用于哈希记录的实际权重文件。
    """

    if model_name == "yolo":
        return model_path
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_path / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No DETR weight file found in {model_path}")


def _detector_display_label(model_name: str, result: dict[str, Any]) -> str:
    """
    方法作用：
        根据本次实际加载的权重路径生成报告中的模型名，避免切换YOLO版本后标题仍被写死。

    输入参数：
        model_name (str)：检测器键，yolo或detr；
        result (dict[str,Any])：单个检测器的运行时与指标结果。

    返回值：
        str：供Markdown报告展示的模型名。
    """

    if model_name == "yolo":
        stem = Path(result["runtime"]["model_path"]).stem
        return f"YOLO{stem[4:]}" if stem.lower().startswith("yolo") else stem
    if model_name == "detr":
        return "DETR-R50"
    return model_name


def _write_crop_evaluation_report(run_root: Path, summary: dict[str, Any]) -> None:
    """
    方法作用：
        生成以“找全车辆和裁剪质量”为主、标准AP为辅的阶段1Markdown报告。

    输入参数：
        run_root (Path)：本次实验输出目录；summary：YOLO/DETR完整指标字典。

    返回值：
        None：报告写入run_root/RESULTS.md。
    """

    labels = {
        name: _detector_display_label(name, result)
        for name, result in summary["results"].items()
    }
    lines = [
        "# 实验1阶段1：未微调车辆候选框评测（截取指标版）",
        "",
        f"数据：{summary['selected_image_count']}张整图、"
        f"{summary['selected_ground_truth_count']}个XML车辆框；两个检测器均未训练。",
        "",
        "## 主指标：是否找全并正确截取车辆",
        "",
        "| 模型 | 对象Recall@IoU50 | 整图全部找全@IoU50 | 对象Recall@IoU75 | "
        "完整裁剪Recall | 整图全部完整裁剪 | 可用裁剪Recall | 零检出图片率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["results"].items():
        primary = result["metrics"]["primary_crop_metrics"]
        lines.append(
            f"| {labels.get(name, name)} | {primary['object_recall_iou50'] * 100:.2f}% | "
            f"{primary['all_targets_detected_image_rate_iou50'] * 100:.2f}% | "
            f"{primary['object_recall_iou75'] * 100:.2f}% | "
            f"{primary['complete_crop_recall'] * 100:.2f}% | "
            f"{primary['all_complete_crop_image_rate'] * 100:.2f}% | "
            f"{primary['usable_crop_recall'] * 100:.2f}% | "
            f"{primary['zero_detected_image_rate'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "完整裁剪要求IoU不低于0.50且车辆覆盖率不低于90%；可用裁剪还要求车辆占预测框面积不低于50%。",
            "",
            "## 匹配框重合质量（IoU不低于0.50的正确框）",
            "",
            "| 模型 | 平均IoU | IoU中位数 | IoU P10 | 平均车辆覆盖率 | "
            "平均裁剪纯净度 | 车辆被切比例 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in summary["results"].items():
        quality = result["metrics"]["matched_box_quality_iou50"]
        lines.append(
            f"| {labels.get(name, name)} | {quality['iou']['mean']:.4f} | "
            f"{quality['iou']['median']:.4f} | {quality['iou']['p10']:.4f} | "
            f"{quality['target_coverage']['mean'] * 100:.2f}% | "
            f"{quality['crop_purity']['mean'] * 100:.2f}% | "
            f"{quality['vehicle_cut_rate_among_matches'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 辅助指标：误检、标准检测指标和效率",
            "",
            "| 模型 | Precision@0.30 | FP/图 | 候选裁剪/图 | AP50 | AP75 | "
            "mAP50:95 | 小目标Recall | ms/图 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in summary["results"].items():
        metrics = result["metrics"]
        operating = metrics["operating_point"]
        lines.append(
            f"| {labels.get(name, name)} | {operating['precision'] * 100:.2f}% | "
            f"{operating['false_positives_per_image']:.3f} | "
            f"{operating['candidate_crops_per_image']:.3f} | {metrics['ap_50']:.4f} | "
            f"{metrics['ap_75']:.4f} | {metrics['map_50_95']:.4f} | "
            f"{operating['small_object_recall'] * 100:.2f}% | "
            f"{result['runtime']['milliseconds_per_image']:.2f} |"
        )
    lines.extend(
        [
            "",
            "说明：整图全部找全率要求一张图内每个XML目标都成功匹配，任何一个目标漏检都将该图片记为失败。",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> Path:
    """
    方法作用：
        执行smoke或full零样本实验，保存配置、逐图预测和指标汇总。

    输入参数：
        args (argparse.Namespace)：CLI配置，主数据流为I张整图及每图N_i个GT/预测框。

    返回值：
        Path：本次唯一输出目录。
    """

    config = load_evaluation_config(args.config)
    evaluation = config["evaluation"]
    all_records = load_dataset(config["paths"]["image_root"])
    limit = args.limit
    if args.mode == "smoke" and limit is None:
        limit = int(evaluation["smoke_limit"])
    records = select_smoke_records(all_records, int(limit)) if limit else all_records
    if not records:
        raise ValueError("No evaluation images selected")

    device = _device_text(args.device)
    batch_size = int(args.batch_size or evaluation["batch_size"])
    if args.mode == "smoke" and args.batch_size is None:
        batch_size = min(batch_size, 2)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = config["paths"]["output_root"] / f"{args.mode}_{run_id}"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_root}")
    run_root.mkdir(parents=True)

    candidate_names = {_normalize_label(item) for item in evaluation["candidate_class_names"]}
    common = {
        "records": records,
        "candidate_names": candidate_names,
        "device": device,
        "batch_size": batch_size,
        "score_floor": float(evaluation["ap_score_floor"]),
        "max_detections": int(evaluation["max_detections_per_image"]),
    }
    detector_names = [item.strip().lower() for item in args.detectors.split(",") if item.strip()]
    invalid = set(detector_names) - {"yolo", "detr"}
    if invalid:
        raise ValueError(f"Unknown detectors: {sorted(invalid)}")

    summary: dict[str, Any] = {
        "experiment": config["name"],
        "mode": args.mode,
        "training_performed": False,
        "device": device,
        "selected_image_count": len(records),
        "full_dataset_image_count": len(all_records),
        "selected_ground_truth_count": sum(len(record.boxes) for record in records),
        "candidate_class_names": sorted(candidate_names),
        "config_path": str(config["config_path"]),
        "results": {},
    }
    for detector_name in detector_names:
        model_path = config["paths"][f"{detector_name}_model"]
        weight_path = _model_weight_path(detector_name, model_path)
        if detector_name == "yolo":
            predictions, runtime = run_yolo_inference(
                model_path=model_path,
                image_size=int(evaluation["yolo_image_size"]),
                **common,
            )
        else:
            predictions, runtime = run_detr_inference(model_path=model_path, **common)
        metrics = evaluate_predictions(
            records,
            predictions,
            iou_thresholds=[float(item) for item in evaluation["iou_thresholds"]],
            operating_threshold=float(evaluation["operating_score_threshold"]),
            small_object_area=float(evaluation["small_object_area"]),
            target_coverage_threshold=float(
                evaluation.get("target_coverage_threshold", 0.90)
            ),
            crop_purity_threshold=float(evaluation.get("crop_purity_threshold", 0.50)),
        )
        _write_predictions(run_root / f"{detector_name}_predictions.jsonl", records, predictions)
        summary["results"][detector_name] = {
            "weight_sha256": _sha256(weight_path),
            "runtime": runtime,
            "metrics": metrics,
        }
        print(
            f"{detector_name.upper()}: "
            f"ObjectRecall@IoU50={metrics['primary_crop_metrics']['object_recall_iou50']:.4f}, "
            f"AllTargetsImageRate={metrics['primary_crop_metrics']['all_targets_detected_image_rate_iou50']:.4f}, "
            f"MeanMatchedIoU={metrics['matched_box_quality_iou50']['iou']['mean']:.4f}"
        )
    _write_json(run_root / "summary.json", summary)
    _write_crop_evaluation_report(run_root, summary)
    _write_json(
        run_root / "resolved_config.json",
        {
            **config,
            "paths": {key: str(value) for key, value in config["paths"].items()},
            "config_path": str(config["config_path"]),
            "runtime": {
                "mode": args.mode,
                "device": device,
                "batch_size": batch_size,
                "limit": limit,
                "detectors": detector_names,
            },
        },
    )
    print(f"Results: {run_root}")
    return run_root


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建零样本检测评测命令行参数解析器。

    输入参数：
        无。

    返回值：
        argparse.ArgumentParser：支持smoke/full、设备和样本数覆盖的解析器。
    """

    parser = argparse.ArgumentParser(
        description="Zero-shot YOLO/DETR evaluation on raw Russian vehicle images"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detectors", default="yolo,detr")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析命令行并启动零样本检测实验。

    输入参数：
        argv (list[str]|None)：可选命令行参数列表。

    返回值：
        int：成功时返回0。
    """

    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
