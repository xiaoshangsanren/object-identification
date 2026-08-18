from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image

from .config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_MODEL_ROOT,
    DEFAULT_VEHICLE_CLASS_IDS,
    DEFAULT_YOLO_MODEL_DIR,
)

ProgressCallback = Callable[[float, str], None]

YOLO_MODEL_DIR = Path(os.getenv("TVS_YOLO_MODEL_DIR", DEFAULT_YOLO_MODEL_DIR))
YOLO_DATASET_DIR = Path("datasets/yolo_uploads")
YOLO_RUNS_DIR = Path("runs/yolo_finetune")
YOLO_TRAINING_LOG_DIR = Path(
    os.getenv(
        "YOLO_TRAINING_LOG_DIR",
        str(Path(os.getenv("LOG_DIR", "logs")) / "yolo_training"),
    )
)
ACTIVE_TRAINING_STATES = {"queued", "running"}
DEFAULT_MODEL_CHOICES = (
    DEFAULT_DETECTOR_MODEL,
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
)
TANK_ONLY_TRAIN_STEM = "tank_only_train"
TANK_ONLY_CLASS_NAME = "tank"
TANK_ONLY_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TANK_ONLY_FULL_IMAGE_LABEL = "0 0.500000 0.500000 1.000000 1.000000"


def _safe_name(value: str, default: str = "yolo_custom") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip())
    value = value.strip("._-")
    return value or default


def list_yolo_models() -> list[str]:
    choices: list[str] = []
    for item in DEFAULT_MODEL_CHOICES:
        if item and item not in choices:
            choices.append(item)

    model_root = Path(DEFAULT_MODEL_ROOT)
    search_roots = [
        YOLO_MODEL_DIR,
        model_root,
        YOLO_RUNS_DIR,
        Path("runs"),
    ]
    seen_roots = set()
    unique_roots = []
    for root in search_roots:
        resolved = root.resolve()
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        unique_roots.append(root)

    for root in unique_roots:
        if not root.exists():
            continue
        for pattern in ("*.pt", "*.engine", "**/weights/best.pt", "**/weights/last.pt"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    value = str(path)
                    if value not in choices:
                        choices.append(value)
    return choices


def _assert_safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes target directory: {member_name}") from exc
    return target


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = _assert_safe_target(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def _extract_rar(archive: Path, destination: Path) -> None:
    try:
        import rarfile
    except ImportError as exc:
        raise RuntimeError(
            "RAR extraction requires the Python package `rarfile` and a usable "
            "unrar/7z backend. Verify the offline package or install the configured backend."
        ) from exc

    with rarfile.RarFile(archive) as rf:
        for info in rf.infolist():
            target = _assert_safe_target(destination, info.filename)
            if info.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with rf.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)


def extract_dataset_archive(archive_path: str | Path, destination: Path) -> Path:
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        _extract_zip(archive, destination)
    elif suffix == ".rar":
        _extract_rar(archive, destination)
    else:
        raise ValueError("Only .zip and .rar YOLO datasets are supported.")
    return destination


def is_tank_only_train_archive(archive_path: str | Path) -> bool:
    return Path(archive_path).stem.casefold() == TANK_ONLY_TRAIN_STEM


def _looks_like_yolo_label_text(text: str) -> bool:
    found = False
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            return False
        try:
            values = [float(item) for item in parts[:5]]
        except ValueError:
            return False
        if values[0] < 0 or not all(0.0 <= value <= 1.0 for value in values[1:5]):
            return False
        found = True
    return found


def _json_annotation_format(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    annotations = payload.get("annotations")
    if isinstance(payload.get("images"), list) and isinstance(annotations, list):
        if any(isinstance(item, dict) and item.get("bbox") for item in annotations):
            return "COCO JSON"
    shapes = payload.get("shapes")
    if isinstance(shapes, list):
        if any(isinstance(item, dict) and item.get("points") for item in shapes):
            return "LabelMe/X-AnyLabeling JSON"
    containers = [payload.get("objects"), payload.get("annotations"), payload.get("bboxes")]
    if any(
        isinstance(value, list)
        and any(
            isinstance(item, dict)
            and any(key in item for key in ("bbox", "box", "xyxy", "x1", "xmin"))
            for item in value
        )
        for value in containers
    ):
        return "通用 JSON bbox"
    if any(key in payload for key in ("bbox", "box", "xyxy", "x1")):
        return "通用 JSON bbox"
    return None


def _inspect_annotation_bytes(name: str, raw: bytes) -> str | None:
    suffix = Path(name).suffix.lower()
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return None
    if suffix == ".json":
        try:
            return _json_annotation_format(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            return None
    if suffix == ".xml":
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        return "Pascal VOC XML" if root.findall(".//object/bndbox") else None
    if suffix == ".txt" and Path(name).name.casefold() != "classes.txt":
        return "YOLO TXT" if _looks_like_yolo_label_text(text) else None
    return None


def inspect_tank_only_train_archive(archive_path: str | Path | None) -> dict[str, Any]:
    """Inspect a training upload without materializing a second dataset copy."""

    if not archive_path:
        return {
            "special_case": False,
            "class_labels": None,
            "image_count": 0,
            "annotation_files": 0,
            "annotation_formats": [],
            "message": "等待上传训练数据集。",
        }
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if not is_tank_only_train_archive(archive):
        return {
            "special_case": False,
            "class_labels": None,
            "image_count": 0,
            "annotation_files": 0,
            "annotation_formats": [],
            "message": "普通 YOLO 数据集：按现有 data.yaml / YOLO 目录规则处理。",
        }

    suffix = archive.suffix.lower()
    members: list[tuple[str, bytes | None]] = []
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                member_suffix = Path(info.filename).suffix.lower()
                raw = handle.read(info) if member_suffix in {".json", ".xml", ".txt"} else None
                members.append((info.filename, raw))
    elif suffix == ".rar":
        try:
            import rarfile
        except ImportError as exc:
            raise RuntimeError("RAR inspection requires the Python package `rarfile`.") from exc
        with rarfile.RarFile(archive) as handle:
            for info in handle.infolist():
                if info.isdir():
                    continue
                member_suffix = Path(info.filename).suffix.lower()
                raw = handle.read(info) if member_suffix in {".json", ".xml", ".txt"} else None
                members.append((info.filename, raw))
    else:
        raise ValueError("Only .zip and .rar YOLO datasets are supported.")

    image_count = sum(Path(name).suffix.lower() in TANK_ONLY_IMAGE_SUFFIXES for name, _ in members)
    if image_count == 0:
        raise RuntimeError("tank_only_train 压缩包中没有可用图片。")
    formats: list[str] = []
    annotation_files = 0
    for name, raw in members:
        if raw is None:
            continue
        detected = _inspect_annotation_bytes(name, raw)
        if detected is None:
            continue
        annotation_files += 1
        if detected not in formats:
            formats.append(detected)

    if annotation_files:
        format_text = "、".join(formats)
        message = (
            f"tank_only_train：检测到 {image_count} 张图片、{annotation_files} 个有效标注文件"
            f"（{format_text}），训练时将自动转换为单类别 tank 的 YOLO 标注。"
        )
    else:
        message = (
            f"tank_only_train：检测到 {image_count} 张图片；不存在标注，视为已裁剪。"
            "训练时将为每张图片生成覆盖整图的 tank 目标框。"
        )
    return {
        "special_case": True,
        "class_labels": TANK_ONLY_CLASS_NAME,
        "image_count": image_count,
        "annotation_files": annotation_files,
        "annotation_formats": formats,
        "message": message,
    }


def _read_training_archive_members(archive: Path) -> list[tuple[str, bytes | None]]:
    metadata_suffixes = {".json", ".xml", ".txt", ".yaml", ".yml"}
    members: list[tuple[str, bytes | None]] = []
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                members.append((info.filename, handle.read(info) if suffix in metadata_suffixes else None))
        return members
    if archive.suffix.lower() == ".rar":
        try:
            import rarfile
        except ImportError as exc:
            raise RuntimeError("RAR inspection requires the Python package `rarfile`.") from exc
        with rarfile.RarFile(archive) as handle:
            for info in handle.infolist():
                if info.isdir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                members.append((info.filename, handle.read(info) if suffix in metadata_suffixes else None))
        return members
    raise ValueError("Only .zip and .rar YOLO datasets are supported.")


def _append_unique(values: list[str], candidates: Any) -> None:
    for candidate in candidates:
        text = str(candidate).strip()
        if text and text not in values:
            values.append(text)


def _discover_json_class_names(payload: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(payload, dict):
        return names
    categories = payload.get("categories")
    if isinstance(categories, list):
        ordered = sorted(
            (item for item in categories if isinstance(item, dict)),
            key=lambda item: _class_id_sort_key(str(item.get("id", ""))),
        )
        _append_unique(names, (item.get("name", "") for item in ordered))
    shapes = payload.get("shapes")
    if isinstance(shapes, list):
        _append_unique(
            names,
            (item.get("label", "") for item in shapes if isinstance(item, dict)),
        )
    for key in ("objects", "annotations", "bboxes"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        _append_unique(
            names,
            (
                item.get("label", item.get("name", item.get("class_name", "")))
                for item in items
                if isinstance(item, dict)
            ),
        )
    return names


def inspect_yolo_training_archive(archive_path: str | Path | None) -> dict[str, Any]:
    """Inspect any supported training archive and discover its class metadata."""

    special = inspect_tank_only_train_archive(archive_path)
    if special.get("special_case") or not archive_path:
        return special
    archive = Path(archive_path)
    members = _read_training_archive_members(archive)
    image_count = sum(Path(name).suffix.lower() in TANK_ONLY_IMAGE_SUFFIXES for name, _ in members)
    formats: list[str] = []
    class_names: list[str] = []
    max_yolo_class_id = -1
    annotation_files = 0
    native_yolo = False

    normalized_names = [name.replace("\\", "/").casefold() for name, _ in members]
    has_images_directory = any("/images/" in f"/{name}/" for name in normalized_names)
    has_labels_directory = any("/labels/" in f"/{name}/" for name in normalized_names)
    native_yolo = has_images_directory and has_labels_directory

    for name, raw in members:
        if raw is None:
            continue
        suffix = Path(name).suffix.lower()
        filename = Path(name).name.casefold()
        try:
            text = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            continue
        if suffix in {".yaml", ".yml"}:
            try:
                payload = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if isinstance(payload, dict) and "train" in payload and "names" in payload:
                native_yolo = True
                _append_unique(class_names, _normalize_class_names(payload.get("names")))
            continue
        if filename == "classes.txt":
            _append_unique(class_names, (line.strip() for line in text.splitlines()))
            continue

        detected = _inspect_annotation_bytes(name, raw)
        if detected is None:
            continue
        annotation_files += 1
        if detected not in formats:
            formats.append(detected)
        if suffix == ".json":
            try:
                _append_unique(class_names, _discover_json_class_names(json.loads(text)))
            except json.JSONDecodeError:
                pass
        elif suffix == ".xml":
            try:
                root = ET.fromstring(text)
            except ET.ParseError:
                continue
            _append_unique(class_names, (node.text or "" for node in root.findall(".//object/name")))
        elif suffix == ".txt":
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    max_yolo_class_id = max(max_yolo_class_id, int(float(parts[0])))
                except ValueError:
                    continue

    if max_yolo_class_id >= 0 and len(class_names) <= max_yolo_class_id:
        for class_id in range(len(class_names), max_yolo_class_id + 1):
            class_names.append(f"class_{class_id}")
    if image_count == 0:
        message = "压缩包中没有检测到受支持的图片。"
    elif annotation_files == 0 and not native_yolo:
        message = (
            f"普通数据集：检测到 {image_count} 张图片，但不存在受支持的目标框标注。"
            "普通数据集不会把无标注图片视为整图目标。"
        )
    else:
        format_text = "、".join(formats) if formats else "标准 YOLO 数据集"
        class_text = "、".join(class_names) if class_names else "未提供类别名"
        message = (
            f"普通数据集：检测到 {image_count} 张图片；格式：{format_text}；"
            f"类别：{class_text}。训练时将自动适配为 YOLO 格式。"
        )
    return {
        "special_case": False,
        "class_labels": "\n".join(class_names) if class_names else None,
        "class_names": class_names,
        "image_count": image_count,
        "annotation_files": annotation_files,
        "annotation_formats": formats,
        "native_yolo": native_yolo,
        "message": message,
    }


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _looks_like_yolo_yaml(path: Path) -> bool:
    data = _load_yaml(path)
    if not data:
        return False
    return "train" in data and "names" in data


def find_dataset_yaml(root: Path) -> Path | None:
    candidates = []
    for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        candidates.extend(root.rglob(name))
    candidates.extend(root.rglob("*.yaml"))
    candidates.extend(root.rglob("*.yml"))

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if _looks_like_yolo_yaml(path):
            return path
    return None


def _read_class_names(root: Path) -> list[str]:
    for path in root.rglob("classes.txt"):
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        names = [name for name in names if name]
        if names:
            return names

    max_class_id = _scan_max_class_id(root)
    if max_class_id >= 0:
        return [f"class_{idx}" for idx in range(max_class_id + 1)]
    return ["class_0"]


def _scan_max_class_id(root: Path) -> int:
    max_class_id = -1
    label_root = root / "labels"
    label_files = label_root.rglob("*.txt") if label_root.exists() else root.rglob("*.txt")
    for label_file in label_files:
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                max_class_id = max(max_class_id, int(float(parts[0])))
            except ValueError:
                continue
    return max_class_id


def _write_classes_txt(root: Path, names: list[str]) -> None:
    if not names:
        return
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _find_split_dir(base: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _resolve_images_first_layout(root: Path) -> tuple[Path, str, str | None] | None:
    train_images = root / "images" / "train"
    train_labels = root / "labels" / "train"
    if train_images.is_dir() and train_labels.is_dir():
        val_dir = _find_split_dir(root / "images", ("val", "valid", "validation"))
        val_rel = None if val_dir is None else f"images/{val_dir.name}"
        return root, "images/train", val_rel

    for candidate in root.rglob("images/train"):
        if not candidate.is_dir():
            continue
        dataset_root = candidate.parent.parent
        label_dir = dataset_root / "labels" / "train"
        if not label_dir.is_dir():
            continue
        val_dir = _find_split_dir(dataset_root / "images", ("val", "valid", "validation"))
        val_rel = None if val_dir is None else f"images/{val_dir.name}"
        return dataset_root, "images/train", val_rel
    return None


def _resolve_split_first_layout(root: Path) -> tuple[Path, str, str | None] | None:
    for train_dir in root.rglob("train"):
        if not train_dir.is_dir():
            continue
        images_dir = train_dir / "images"
        labels_dir = train_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        dataset_root = train_dir.parent
        val_rel = None
        for split_name in ("val", "valid", "validation"):
            split_dir = dataset_root / split_name
            if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
                val_rel = f"{split_name}/images"
                break
        return dataset_root, "train/images", val_rel
    return None


def _summarize_dataset_layout(root: Path) -> str:
    interesting_dirs: list[str] = []
    interesting_files: list[str] = []
    patterns = re.compile(r"images|labels|train|val|valid|test|data\.ya?ml", re.IGNORECASE)

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not relative:
            continue
        if path.is_dir() and patterns.search(relative):
            interesting_dirs.append(relative + "/")
        elif path.is_file() and patterns.search(relative):
            interesting_files.append(relative)

    lines: list[str] = []
    if interesting_dirs:
        lines.append("found dirs:")
        lines.extend(f"  - {item}" for item in interesting_dirs[:20])
    if interesting_files:
        lines.append("found files:")
        lines.extend(f"  - {item}" for item in interesting_files[:20])
    return "\n".join(lines) if lines else "no YOLO-like train/val/images/labels paths were found."


def _valid_xyxy(
    values: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = values
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _scale_if_normalized(values: list[float], width: int, height: int) -> list[float]:
    if values and all(0.0 <= value <= 1.0 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return values


def _bbox_value_to_xyxy(
    value: Any,
    width: int,
    height: int,
    *,
    xywh_default: bool,
) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            raw = [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
            raw = _scale_if_normalized(raw, width, height)
            return _valid_xyxy(tuple(raw), width, height)
        if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
            raw = [float(value[key]) for key in ("xmin", "ymin", "xmax", "ymax")]
            raw = _scale_if_normalized(raw, width, height)
            return _valid_xyxy(tuple(raw), width, height)
        if all(key in value for key in ("x", "y", "width", "height")):
            raw = [float(value[key]) for key in ("x", "y", "width", "height")]
            raw = _scale_if_normalized(raw, width, height)
            x, y, box_width, box_height = raw
            return _valid_xyxy((x, y, x + box_width, y + box_height), width, height)
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        raw = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    raw = _scale_if_normalized(raw, width, height)
    if xywh_default:
        x, y, box_width, box_height = raw
        raw = [x, y, x + box_width, y + box_height]
    return _valid_xyxy(tuple(raw), width, height)


def _json_boxes(
    payload: Any,
    width: int,
    height: int,
) -> tuple[list[tuple[float, float, float, float]], str | None]:
    if not isinstance(payload, dict):
        return [], None
    boxes: list[tuple[float, float, float, float]] = []
    shapes = payload.get("shapes")
    if isinstance(shapes, list):
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            points = shape.get("points")
            if not isinstance(points, list) or len(points) < 2:
                continue
            try:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
            except (IndexError, TypeError, ValueError):
                continue
            box = _valid_xyxy((min(xs), min(ys), max(xs), max(ys)), width, height)
            if box is not None:
                boxes.append(box)
        return boxes, "LabelMe/X-AnyLabeling JSON"

    items: list[Any] = []
    for key in ("objects", "annotations", "bboxes"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(value)
    if not items:
        items = [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        box = None
        if "xyxy" in item:
            box = _bbox_value_to_xyxy(item.get("xyxy"), width, height, xywh_default=False)
        elif "bbox" in item:
            box = _bbox_value_to_xyxy(item.get("bbox"), width, height, xywh_default=True)
        elif "box" in item:
            box = _bbox_value_to_xyxy(item.get("box"), width, height, xywh_default=False)
        else:
            box = _bbox_value_to_xyxy(item, width, height, xywh_default=False)
        if box is not None:
            boxes.append(box)
    return boxes, "通用 JSON bbox" if boxes else None


def _xml_boxes(
    path: Path,
    width: int,
    height: int,
) -> list[tuple[float, float, float, float]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for node in root.findall(".//object/bndbox"):
        try:
            values = tuple(float(node.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        box = _valid_xyxy(values, width, height)
        if box is not None:
            boxes.append(box)
    return boxes


def _yolo_lines(path: Path) -> list[str]:
    lines: list[str] = []
    try:
        source_lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in source_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            values = [float(item) for item in parts[1:5]]
        except ValueError:
            continue
        if not all(0.0 <= value <= 1.0 for value in values):
            continue
        x_center, y_center, box_width, box_height = values
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return lines


def _xyxy_to_yolo_line(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = box
    return (
        f"0 {((x1 + x2) / 2.0) / width:.6f} {((y1 + y2) / 2.0) / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def _paired_annotation(image_path: Path, candidates: dict[str, list[Path]]) -> Path | None:
    matches = candidates.get(image_path.stem.casefold(), [])
    for candidate in matches:
        if candidate.parent == image_path.parent:
            return candidate
    return matches[0] if len(matches) == 1 else None


def _build_coco_box_map(root: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            continue
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            continue
        images_by_id = {
            item.get("id"): item
            for item in payload["images"]
            if isinstance(item, dict) and item.get("id") is not None
        }
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            image = images_by_id.get(annotation.get("image_id"))
            if image is None:
                continue
            try:
                width = int(image.get("width"))
                height = int(image.get("height"))
            except (TypeError, ValueError):
                continue
            box = _bbox_value_to_xyxy(
                annotation.get("bbox"), width, height, xywh_default=True
            )
            if box is None:
                continue
            file_name = str(image.get("file_name") or "").replace("\\", "/").casefold()
            if not file_name:
                continue
            for key in {file_name, Path(file_name).name.casefold()}:
                result.setdefault(key, []).append(box)
    return result


def prepare_tank_only_train_dataset(extracted_root: Path) -> tuple[Path, dict[str, Any]]:
    """Convert a tank-only archive into a deterministic one-class YOLO dataset."""

    image_paths = sorted(
        (
            path
            for path in extracted_root.rglob("*")
            if path.is_file() and path.suffix.lower() in TANK_ONLY_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(extracted_root).as_posix().casefold(),
    )
    if not image_paths:
        raise RuntimeError("tank_only_train 压缩包中没有可用图片。")

    annotation_paths: dict[str, dict[str, list[Path]]] = {}
    for suffix in (".json", ".xml", ".txt"):
        by_stem: dict[str, list[Path]] = {}
        for path in sorted(extracted_root.rglob(f"*{suffix}")):
            if path.name.casefold() == "classes.txt":
                continue
            by_stem.setdefault(path.stem.casefold(), []).append(path)
        annotation_paths[suffix] = by_stem
    coco_box_map = _build_coco_box_map(extracted_root)

    prepared_samples: list[tuple[Path, list[str], str]] = []
    format_counts: dict[str, int] = {}
    annotated_images = 0
    full_image_boxes = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"无效图片尺寸：{image_path}")

        relative_key = image_path.relative_to(extracted_root).as_posix().casefold()
        boxes = coco_box_map.get(relative_key) or coco_box_map.get(image_path.name.casefold()) or []
        lines: list[str] = []
        source_format: str | None = None
        if boxes:
            lines = [_xyxy_to_yolo_line(box, width, height) for box in boxes]
            source_format = "COCO JSON"

        if not lines:
            json_path = _paired_annotation(image_path, annotation_paths[".json"])
            if json_path is not None:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                json_boxes, json_format = _json_boxes(payload, width, height)
                if json_boxes:
                    lines = [_xyxy_to_yolo_line(box, width, height) for box in json_boxes]
                    source_format = json_format

        if not lines:
            xml_path = _paired_annotation(image_path, annotation_paths[".xml"])
            if xml_path is not None:
                xml_boxes = _xml_boxes(xml_path, width, height)
                if xml_boxes:
                    lines = [_xyxy_to_yolo_line(box, width, height) for box in xml_boxes]
                    source_format = "Pascal VOC XML"

        if not lines:
            txt_path = _paired_annotation(image_path, annotation_paths[".txt"])
            if txt_path is not None:
                lines = _yolo_lines(txt_path)
                if lines:
                    source_format = "YOLO TXT"

        if not lines:
            lines = [TANK_ONLY_FULL_IMAGE_LABEL]
            source_format = "已裁剪整图"
            full_image_boxes += 1
        else:
            annotated_images += 1
        format_counts[source_format] = format_counts.get(source_format, 0) + 1
        prepared_samples.append((image_path, lines, source_format))

    shuffled = list(prepared_samples)
    random.Random(42).shuffle(shuffled)
    if len(shuffled) >= 2:
        val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.2)))
        validation = shuffled[:val_count]
        training = shuffled[val_count:]
    else:
        training = shuffled
        validation = []

    dataset_root = extracted_root / "tank_only_yolo"
    if dataset_root.exists():
        raise FileExistsError(f"Prepared dataset already exists: {dataset_root}")
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    sample_index = 0
    for split, samples in (("train", training), ("val", validation)):
        for image_path, lines, _ in sorted(samples, key=lambda item: item[0].as_posix().casefold()):
            sample_index += 1
            name = f"tank_{sample_index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, dataset_root / "images" / split / name)
            (dataset_root / "labels" / split / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val" if validation else "images/train",
        "names": [TANK_ONLY_CLASS_NAME],
        "nc": 1,
    }
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_classes_txt(dataset_root, [TANK_ONLY_CLASS_NAME])

    if annotated_images == 0:
        message = "不存在标注，视为已裁剪；已为每张图片生成覆盖整图的 tank 目标框。"
    elif full_image_boxes:
        message = (
            f"已自动适配 {annotated_images} 张图片的目标框；其余 {full_image_boxes} 张不存在标注，"
            "视为已裁剪并生成覆盖整图的 tank 目标框。"
        )
    else:
        message = f"已自动适配全部 {annotated_images} 张图片的目标框。"
    summary = {
        "special_case": TANK_ONLY_TRAIN_STEM,
        "class_names": [TANK_ONLY_CLASS_NAME],
        "images": len(prepared_samples),
        "annotated_images": annotated_images,
        "full_image_box_images": full_image_boxes,
        "train_images": len(training),
        "val_images": len(validation),
        "annotation_formats": format_counts,
        "message": message,
    }
    (dataset_root / "preparation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data_yaml, summary


def _has_native_yolo_dataset(root: Path) -> bool:
    return (
        find_dataset_yaml(root) is not None
        or _resolve_images_first_layout(root) is not None
        or _resolve_split_first_layout(root) is not None
    )


def _read_explicit_classes(root: Path) -> list[str]:
    for path in sorted(root.rglob("classes.txt")):
        names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
        names = [name for name in names if name]
        if names:
            return names
    return []


def _xyxy_to_normalized_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def _json_labeled_boxes(
    payload: Any, width: int, height: int
) -> tuple[list[tuple[str, tuple[float, float, float, float]]], str | None]:
    if not isinstance(payload, dict):
        return [], None
    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    shapes = payload.get("shapes")
    if isinstance(shapes, list):
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            points = shape.get("points")
            if not isinstance(points, list) or len(points) < 2:
                continue
            try:
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
            except (IndexError, TypeError, ValueError):
                continue
            box = _valid_xyxy((min(xs), min(ys), max(xs), max(ys)), width, height)
            if box is not None:
                label = str(shape.get("label") or "class_0").strip() or "class_0"
                objects.append((label, _xyxy_to_normalized_box(box, width, height)))
        return objects, "LabelMe/X-AnyLabeling JSON"

    items: list[Any] = []
    for key in ("objects", "annotations", "bboxes"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(value)
    if not items:
        items = [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        if "xyxy" in item:
            box = _bbox_value_to_xyxy(item.get("xyxy"), width, height, xywh_default=False)
        elif "bbox" in item:
            box = _bbox_value_to_xyxy(item.get("bbox"), width, height, xywh_default=True)
        elif "box" in item:
            box = _bbox_value_to_xyxy(item.get("box"), width, height, xywh_default=False)
        else:
            box = _bbox_value_to_xyxy(item, width, height, xywh_default=False)
        if box is None:
            continue
        label = str(
            item.get(
                "label",
                item.get("name", item.get("class_name", f"class_{item.get('category_id', 0)}")),
            )
        ).strip()
        objects.append((label or "class_0", _xyxy_to_normalized_box(box, width, height)))
    return objects, "通用 JSON bbox" if objects else None


def _xml_labeled_boxes(
    path: Path, width: int, height: int
) -> list[tuple[str, tuple[float, float, float, float]]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    for object_node in root.findall(".//object"):
        box_node = object_node.find("bndbox")
        if box_node is None:
            continue
        try:
            values = tuple(
                float(box_node.findtext(key, ""))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError:
            continue
        box = _valid_xyxy(values, width, height)
        if box is None:
            continue
        label = str(object_node.findtext("name", "class_0")).strip() or "class_0"
        objects.append((label, _xyxy_to_normalized_box(box, width, height)))
    return objects


def _yolo_labeled_boxes(
    path: Path, class_hints: list[str]
) -> list[tuple[str, tuple[float, float, float, float]]]:
    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return objects
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = tuple(float(item) for item in parts[1:5])
        except ValueError:
            continue
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in values):
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            continue
        label = class_hints[class_id] if class_id < len(class_hints) else f"class_{class_id}"
        objects.append((label, values))
    return objects


def _build_coco_labeled_box_map(
    root: Path,
) -> dict[str, list[tuple[str, tuple[float, float, float, float]]]]:
    result: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            continue
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            continue
        category_names = {
            item.get("id"): str(item.get("name") or f"class_{item.get('id')}").strip()
            for item in (payload.get("categories") or [])
            if isinstance(item, dict) and item.get("id") is not None
        }
        images_by_id = {
            item.get("id"): item
            for item in payload["images"]
            if isinstance(item, dict) and item.get("id") is not None
        }
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            image = images_by_id.get(annotation.get("image_id"))
            if image is None:
                continue
            try:
                width, height = int(image.get("width")), int(image.get("height"))
            except (TypeError, ValueError):
                continue
            box = _bbox_value_to_xyxy(annotation.get("bbox"), width, height, xywh_default=True)
            if box is None:
                continue
            category_id = annotation.get("category_id")
            label = category_names.get(category_id, f"class_{category_id if category_id is not None else 0}")
            file_name = str(image.get("file_name") or "").replace("\\", "/").casefold()
            if not file_name:
                continue
            labeled_box = (label, _xyxy_to_normalized_box(box, width, height))
            for key in {file_name, Path(file_name).name.casefold()}:
                result.setdefault(key, []).append(labeled_box)
    return result


def prepare_common_annotated_dataset(
    extracted_root: Path,
    preferred_class_names: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Adapt common flat detection annotations to a standard YOLO dataset."""

    image_paths = sorted(
        (
            path
            for path in extracted_root.rglob("*")
            if path.is_file() and path.suffix.lower() in TANK_ONLY_IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(extracted_root).as_posix().casefold(),
    )
    if not image_paths:
        raise RuntimeError("压缩包中没有可用图片。")
    class_hints = _normalize_class_names(preferred_class_names or []) or _read_explicit_classes(extracted_root)
    if not class_hints:
        max_flat_yolo_id = _scan_max_class_id(extracted_root)
        if max_flat_yolo_id >= 0:
            class_hints = [f"class_{class_id}" for class_id in range(max_flat_yolo_id + 1)]
    annotations_by_suffix: dict[str, dict[str, list[Path]]] = {}
    for suffix in (".json", ".xml", ".txt"):
        by_stem: dict[str, list[Path]] = {}
        for path in sorted(extracted_root.rglob(f"*{suffix}")):
            if path.name.casefold() == "classes.txt":
                continue
            by_stem.setdefault(path.stem.casefold(), []).append(path)
        annotations_by_suffix[suffix] = by_stem
    coco_map = _build_coco_labeled_box_map(extracted_root)

    samples: list[
        tuple[Path, list[tuple[str, tuple[float, float, float, float]]], str]
    ] = []
    detected_names: list[str] = []
    format_counts: dict[str, int] = {}
    background_images = 0
    total_boxes = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        relative_key = image_path.relative_to(extracted_root).as_posix().casefold()
        objects = list(coco_map.get(relative_key) or coco_map.get(image_path.name.casefold()) or [])
        source_format: str | None = "COCO JSON" if objects else None
        if not objects:
            json_path = _paired_annotation(image_path, annotations_by_suffix[".json"])
            if json_path is not None:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = None
                objects, source_format = _json_labeled_boxes(payload, width, height)
        if not objects:
            xml_path = _paired_annotation(image_path, annotations_by_suffix[".xml"])
            if xml_path is not None:
                objects = _xml_labeled_boxes(xml_path, width, height)
                source_format = "Pascal VOC XML" if objects else None
        if not objects:
            txt_path = _paired_annotation(image_path, annotations_by_suffix[".txt"])
            if txt_path is not None:
                objects = _yolo_labeled_boxes(txt_path, class_hints)
                source_format = "YOLO TXT" if objects else None
        if objects:
            total_boxes += len(objects)
            _append_unique(detected_names, (label for label, _ in objects))
            format_counts[source_format or "未知格式"] = format_counts.get(source_format or "未知格式", 0) + 1
        else:
            background_images += 1
            source_format = "无目标背景图"
        samples.append((image_path, objects, source_format))

    if total_boxes == 0:
        raise RuntimeError(
            "普通数据集没有检测到任何有效目标框。只有文件名为 tank_only_train 的特例"
            "才会把无标注图片解释为已裁剪整图目标。"
        )

    rename_map: dict[str, str] = {}
    preferred = class_hints
    if preferred:
        if all(name in preferred for name in detected_names):
            class_names = preferred
        elif len(preferred) == len(detected_names):
            rename_map = dict(zip(detected_names, preferred))
            class_names = preferred
        else:
            raise RuntimeError(
                f"自动识别到 {len(detected_names)} 个类别 {detected_names}，"
                f"但输入了 {len(preferred)} 个类别标签 {preferred}。"
            )
    else:
        class_names = detected_names
    class_ids = {name: index for index, name in enumerate(class_names)}

    shuffled = list(samples)
    random.Random(42).shuffle(shuffled)
    if len(shuffled) >= 2:
        val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.2)))
        validation, training = shuffled[:val_count], shuffled[val_count:]
    else:
        training, validation = shuffled, []

    dataset_root = extracted_root / "adapted_yolo"
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    sample_index = 0
    for split, split_samples in (("train", training), ("val", validation)):
        for image_path, objects, _ in sorted(split_samples, key=lambda item: item[0].as_posix().casefold()):
            sample_index += 1
            name = f"sample_{sample_index:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, dataset_root / "images" / split / name)
            lines = []
            for original_label, values in objects:
                label = rename_map.get(original_label, original_label)
                if label not in class_ids:
                    raise RuntimeError(f"类别 {label!r} 不在最终类别列表中。")
                x_center, y_center, box_width, box_height = values
                lines.append(
                    f"{class_ids[label]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )
            label_text = "\n".join(lines)
            if label_text:
                label_text += "\n"
            (dataset_root / "labels" / split / f"{Path(name).stem}.txt").write_text(
                label_text, encoding="utf-8"
            )

    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val" if validation else "images/train",
        "names": class_names,
        "nc": len(class_names),
    }
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_classes_txt(dataset_root, class_names)
    summary = {
        "special_case": None,
        "adapted_common_format": True,
        "class_names": class_names,
        "images": len(samples),
        "boxes": total_boxes,
        "background_images": background_images,
        "train_images": len(training),
        "val_images": len(validation),
        "annotation_formats": format_counts,
        "message": (
            f"已自动识别并转换 {len(samples)} 张图片、{total_boxes} 个目标框、"
            f"{len(class_names)} 个类别：{', '.join(class_names)}。"
        ),
    }
    (dataset_root / "preparation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data_yaml, summary


def ensure_dataset_yaml(root: Path, preferred_class_names: list[str] | None = None) -> Path:
    preferred_class_names = _normalize_class_names(preferred_class_names or [])
    data_yaml = find_dataset_yaml(root)
    if data_yaml is not None:
        data = _load_yaml(data_yaml) or {}
        dataset_root = data_yaml.parent
        resolved_root = data.get("path")
        if resolved_root:
            dataset_root = (data_yaml.parent / str(resolved_root)).resolve()
        names = preferred_class_names or _normalize_class_names(data.get("names")) or _read_class_names(dataset_root)
        max_class_id = _scan_max_class_id(dataset_root)
        if max_class_id >= len(names):
            raise RuntimeError(
                f"Provided class labels count is {len(names)}, but dataset labels reference class id {max_class_id}. "
                "Please provide enough labels, for example one name per class id."
            )
        data["path"] = str(dataset_root.resolve())
        data["names"] = names
        data["nc"] = len(names)
        data_yaml.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        if preferred_class_names:
            _write_classes_txt(dataset_root, names)
        return data_yaml

    layout = _resolve_images_first_layout(root) or _resolve_split_first_layout(root)
    if layout is None:
        layout_summary = _summarize_dataset_layout(root)
        raise RuntimeError(
            "Dataset archive must contain data.yaml or YOLO folders "
            "images/train + labels/train, or train/images + train/labels.\n"
            f"{layout_summary}"
        )

    root, train_rel, val_rel = layout
    names = preferred_class_names or _read_class_names(root)
    max_class_id = _scan_max_class_id(root)
    if max_class_id >= len(names):
        raise RuntimeError(
            f"Provided class labels count is {len(names)}, but dataset labels reference class id {max_class_id}. "
            "Please provide enough labels, for example one name per class id."
        )
    data = {
        "path": str(root.resolve()),
        "train": train_rel,
        "val": val_rel or train_rel,
        "names": names,
        "nc": len(names),
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), "utf-8")
    if preferred_class_names:
        _write_classes_txt(root, names)
    return data_yaml


def _class_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return (1, str(value).strip())


def _normalize_class_id_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _normalize_class_id_map(value: Any) -> dict[str, str]:
    items: list[tuple[str, str]] = []
    if isinstance(value, dict):
        raw_items = value.items()
    elif isinstance(value, (list, tuple)):
        raw_items = enumerate(value)
    else:
        return {}

    for raw_key, raw_name in raw_items:
        key = _normalize_class_id_text(raw_key)
        name = str(raw_name).strip()
        if not key or not name:
            continue
        items.append((key, name))

    items.sort(key=lambda item: _class_id_sort_key(item[0]))
    return {key: name for key, name in items}


def _normalize_class_names(value: Any) -> list[str]:
    if isinstance(value, str):
        names: list[str] = []
        seen: set[str] = set()
        for item in re.split(r"[\r\n,;，；]+", value):
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names
    if isinstance(value, dict):
        return list(_normalize_class_id_map(value).values())
    if isinstance(value, (list, tuple)):
        return [str(name).strip() for name in value if str(name).strip()]
    return []


def parse_class_names_input(value: str | None) -> list[str]:
    return _normalize_class_names(value)


def _class_names_to_map(names: list[str]) -> dict[str, str]:
    return {str(index): name for index, name in enumerate(names)}


def _normalize_recommended_class_ids(
    value: Any,
    class_id_map: dict[str, str],
) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"all", "*", "none"}:
            return "all"
        parts = re.split(r"[\s,;，；]+", cleaned)
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    elif value is None:
        parts = []
    else:
        parts = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = _normalize_class_id_text(part)
        if not key or key in seen:
            continue
        if class_id_map and key not in class_id_map:
            continue
        seen.add(key)
        normalized.append(key)
    return ",".join(normalized)


def _build_hint_text(class_id_map: dict[str, str], recommended_ids: str) -> str:
    if not class_id_map:
        return "当前模型没有可用的类别元数据。"
    mapping_lines = [f"{key}: {value}" for key, value in class_id_map.items()]
    if recommended_ids:
        prefix = f"当前模型类别 ID 建议填写: {recommended_ids}"
    else:
        prefix = "当前模型已提供类别映射。"
    return prefix + "\n类别映射:\n" + "\n".join(mapping_lines)


def build_class_info(
    names: list[str] | None,
    *,
    class_id_map: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    recommended_class_ids: Any = None,
    hint_text: str | None = None,
) -> dict[str, Any]:
    normalized = [str(name).strip() for name in (names or []) if str(name).strip()]
    normalized_class_id_map = _normalize_class_id_map(class_id_map)
    if not normalized_class_id_map:
        normalized_class_id_map = _class_names_to_map(normalized)
    normalized_names = list(normalized_class_id_map.values())
    recommended_ids = _normalize_recommended_class_ids(
        recommended_class_ids,
        normalized_class_id_map,
    )
    if not recommended_ids and normalized_class_id_map:
        recommended_ids = ",".join(normalized_class_id_map.keys())
    resolved_hint_text = str(hint_text or "").strip() or _build_hint_text(
        normalized_class_id_map,
        recommended_ids,
    )
    return {
        "class_names": normalized_names,
        "class_id_map": normalized_class_id_map,
        "recommended_class_ids": recommended_ids,
        "hint_text": resolved_hint_text,
    }


def load_dataset_class_info(data_yaml: str | Path) -> dict[str, Any]:
    data = _load_yaml(Path(data_yaml)) or {}
    names = _normalize_class_names(data.get("names"))
    if not names:
        names = _read_class_names(Path(data_yaml).parent)
    return build_class_info(names)


def get_yolo_model_info(model_path: str | Path | None) -> dict[str, Any]:
    raw_value = str(model_path or "").strip()
    if not raw_value:
        return {
            **build_class_info([]),
            "model_path": "",
            "metadata_path": "",
            "metadata_source": "none",
        }

    candidate = Path(raw_value).expanduser()
    metadata_candidates = []
    if candidate.suffix:
        metadata_candidates.append(candidate.with_suffix(".json"))
    if not candidate.is_absolute():
        basename = candidate.name
        metadata_candidates.append(YOLO_MODEL_DIR / Path(basename).with_suffix(".json"))
        metadata_candidates.append(Path(DEFAULT_MODEL_ROOT) / Path(basename).with_suffix(".json"))

    seen_paths: set[Path] = set()
    for metadata_path in metadata_candidates:
        resolved = metadata_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        info = build_class_info(
            _normalize_class_names(payload.get("class_names")),
            class_id_map=payload.get("class_id_map"),
            recommended_class_ids=payload.get("recommended_class_ids"),
            hint_text=payload.get("hint_text"),
        )
        info.update(
            {
                "model_path": raw_value,
                "metadata_path": str(metadata_path),
                "metadata_source": "sidecar_json",
            }
        )
        return info

    built_in_names = {
        "2": "car",
        "3": "motorcycle",
        "5": "bus",
        "7": "truck",
    }
    basename = candidate.name or raw_value
    if basename in {"yolo26n.pt", "yolo11n.pt", "yolov8n.pt"}:
        recommended_ids = ",".join(str(item) for item in DEFAULT_VEHICLE_CLASS_IDS)
        info = build_class_info(
            [],
            class_id_map=built_in_names,
            recommended_class_ids=DEFAULT_VEHICLE_CLASS_IDS,
            hint_text=(
                f"当前是通用 YOLO 模型。做车辆检索时，候选类别 ID 建议填写: {recommended_ids}\n"
                "类别映射:\n2: car\n3: motorcycle\n5: bus\n7: truck"
            ),
        )
        info.update(
            {
                "model_path": raw_value,
                "metadata_path": "",
                "metadata_source": "builtin_vehicle_default",
            }
        )
        return info

    info = build_class_info(
        [],
        hint_text=(
            "当前模型没有同名 .json 类别元数据，无法自动提醒新增类别 ID。"
            "如果这是本系统训练出的模型，请保留模型旁边的同名 .json 文件。"
        ),
    )
    info.update(
        {
            "model_path": raw_value,
            "metadata_path": "",
            "metadata_source": "unknown",
        }
    )
    return info


def _copy_best_model(save_dir: Path, run_name: str) -> Path:
    weights_dir = save_dir / "weights"
    best = weights_dir / "best.pt"
    if not best.exists():
        best_files = sorted(save_dir.rglob("best.pt"), key=lambda item: item.stat().st_mtime)
        if not best_files:
            raise FileNotFoundError(f"Cannot find best.pt under {save_dir}")
        best = best_files[-1]

    YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output = YOLO_MODEL_DIR / f"{_safe_name(run_name)}_best.pt"
    if output.exists():
        output = YOLO_MODEL_DIR / f"{_safe_name(run_name)}_{time.strftime('%Y%m%d_%H%M%S')}_best.pt"
    shutil.copy2(best, output)
    return output


@dataclass
class TrainingJob:
    job_id: str
    status: str
    progress: float
    message: str
    created_at: float
    updated_at: float
    params: dict[str, Any]
    log_path: str
    summary: dict[str, Any] | None = None
    error: str | None = None
    traceback_text: str | None = None
    saved_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["progress_percent"] = int(round(float(self.progress) * 100))
        return data


class TrainingJobManager:
    """Small in-process backend for long-running YOLO fine-tune jobs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, TrainingJob] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(
        self,
        archive_path: str | Path,
        base_model: str,
        run_name: str,
        class_names: list[str] | None,
        epochs: int,
        imgsz: int,
        batch: int,
        patience: int,
        workers: int,
        device: str,
    ) -> dict[str, Any]:
        with self._lock:
            running = [
                job
                for job in self._jobs.values()
                if job.status in ACTIVE_TRAINING_STATES
            ]
            if running:
                active = running[-1]
                raise RuntimeError(
                    "已有 YOLO 训练任务正在运行，请等待完成后再提交。"
                    f"当前任务: {active.job_id}"
                )

            stamp = time.strftime("%Y%m%d_%H%M%S")
            job_id = f"train_{stamp}_{uuid.uuid4().hex[:8]}"
            log_path = YOLO_TRAINING_LOG_DIR / f"{job_id}.log"
            params = {
                "archive_path": str(archive_path),
                "base_model": base_model,
                "run_name": run_name,
                "class_names": list(class_names or []),
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "patience": int(patience),
                "workers": int(workers),
                "device": device,
            }
            now = time.time()
            job = TrainingJob(
                job_id=job_id,
                status="queued",
                progress=0.0,
                message="训练任务已提交，等待后台线程启动。",
                created_at=now,
                updated_at=now,
                params=params,
                log_path=str(log_path),
            )
            self._jobs[job_id] = job
            self._append_log_locked(job, job.message)
            self._write_state_locked(job)

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id,),
                name=f"yolo-training-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            return job.to_dict()

    def get(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            job = self._select_job_locked(job_id)
            if job is None:
                return {
                    "job_id": "",
                    "status": "idle",
                    "progress": 0.0,
                    "progress_percent": 0,
                    "message": "暂无训练任务。",
                    "log_path": "",
                    "summary": None,
                    "error": None,
                    "saved_model": None,
                }
            return job.to_dict()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                job.to_dict()
                for job in sorted(
                    self._jobs.values(),
                    key=lambda item: item.created_at,
                    reverse=True,
                )
            ]

    def _select_job_locked(self, job_id: str | None) -> TrainingJob | None:
        if job_id:
            return self._jobs.get(str(job_id).strip())
        if not self._jobs:
            return None
        return max(self._jobs.values(), key=lambda item: item.created_at)

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.progress = 0.01
            job.message = "后台训练已启动。"
            job.updated_at = time.time()
            self._append_log_locked(job, job.message)
            self._write_state_locked(job)

        try:
            params = self.get(job_id)["params"]
            summary = train_yolo_from_archive(
                archive_path=params["archive_path"],
                base_model=params["base_model"],
                run_name=params["run_name"],
                class_names=params.get("class_names"),
                epochs=params["epochs"],
                imgsz=params["imgsz"],
                batch=params["batch"],
                patience=params["patience"],
                workers=params["workers"],
                device=params["device"],
                progress_callback=lambda value, message: self.update_progress(
                    job_id, value, message
                ),
            )
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "error"
                job.progress = max(float(job.progress), 0.0)
                job.message = f"训练失败：{exc}"
                job.error = str(exc)
                job.traceback_text = traceback.format_exc(limit=20)
                job.updated_at = time.time()
                self._append_log_locked(job, job.message)
                self._append_log_locked(job, job.traceback_text)
                self._write_state_locked(job)
            return

        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.progress = 1.0
            job.summary = summary
            job.saved_model = str(summary.get("saved_model") or "")
            job.message = str(
                summary.get("message") or f"训练完成：{job.saved_model}"
            )
            job.updated_at = time.time()
            self._append_log_locked(job, job.message)
            self._write_state_locked(job)

    def update_progress(self, job_id: str, value: float, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ACTIVE_TRAINING_STATES:
                return
            progress = max(0.0, min(float(value), 1.0))
            job.progress = max(float(job.progress), progress)
            if message:
                job.message = message
            job.updated_at = time.time()
            self._append_log_locked(
                job, f"{int(round(job.progress * 100)):3d}% {job.message}"
            )
            self._write_state_locked(job)

    def _append_log_locked(self, job: TrainingJob, message: str) -> None:
        path = Path(job.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = str(message).rstrip().splitlines() or [""]
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"[{stamp}] {line}\n")

    def _write_state_locked(self, job: TrainingJob) -> None:
        path = Path(job.log_path).with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            "utf-8",
        )


_TRAINING_MANAGER = TrainingJobManager()


def start_yolo_training_job(
    archive_path: str | Path,
    base_model: str,
    run_name: str,
    class_names: list[str] | None,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    return _TRAINING_MANAGER.start(
        archive_path=archive_path,
        base_model=base_model,
        run_name=run_name,
        class_names=class_names,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
    )


def get_yolo_training_job(job_id: str | None = None) -> dict[str, Any]:
    return _TRAINING_MANAGER.get(job_id)


def list_yolo_training_jobs() -> list[dict[str, Any]]:
    return _TRAINING_MANAGER.list()


def latest_yolo_training_log() -> Path | None:
    if not YOLO_TRAINING_LOG_DIR.exists():
        return None
    logs = sorted(
        YOLO_TRAINING_LOG_DIR.glob("*.log"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    return logs[0] if logs else None


def train_yolo_from_archive(
    archive_path: str | Path,
    base_model: str,
    run_name: str,
    class_names: list[str] | None,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    workers: int,
    device: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    def emit(value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = _safe_name(run_name, f"yolo_custom_{stamp}")
    archive = Path(archive_path)
    dataset_root = YOLO_DATASET_DIR / f"{_safe_name(archive.stem)}_{stamp}"
    extract_root = dataset_root / "extracted"

    emit(0.05, "extracting dataset")
    extract_dataset_archive(archive, extract_root)

    emit(0.15, "checking YOLO dataset")
    dataset_preparation: dict[str, Any] | None = None
    if is_tank_only_train_archive(archive):
        data_yaml, dataset_preparation = prepare_tank_only_train_dataset(extract_root)
        emit(0.20, dataset_preparation["message"])
    elif _has_native_yolo_dataset(extract_root):
        data_yaml = ensure_dataset_yaml(extract_root, preferred_class_names=class_names)
    else:
        data_yaml, dataset_preparation = prepare_common_annotated_dataset(
            extract_root,
            preferred_class_names=class_names,
        )
        emit(0.20, dataset_preparation["message"])
    class_info = load_dataset_class_info(data_yaml)

    emit(0.25, "training YOLO")
    model = YOLO(base_model)

    def emit_epoch_progress(trainer: Any) -> None:
        epoch = getattr(trainer, "epoch", None)
        try:
            current_epoch = int(epoch) + 1
        except (TypeError, ValueError):
            current_epoch = 0
        total_epochs = max(1, int(epochs))
        if current_epoch <= 0:
            return
        epoch_ratio = min(current_epoch / total_epochs, 1.0)
        emit(
            0.25 + 0.65 * epoch_ratio,
            f"training YOLO epoch {current_epoch}/{total_epochs}",
        )

    if hasattr(model, "add_callback"):
        try:
            model.add_callback(
                "on_train_epoch_end",
                lambda trainer: emit_epoch_progress(trainer),
            )
        except Exception as exc:
            emit(0.25, f"training YOLO; epoch callback unavailable: {exc}")

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "patience": int(patience),
        "workers": int(workers),
        "project": str(YOLO_RUNS_DIR),
        "name": run_name,
        "exist_ok": False,
        "verbose": True,
    }
    if device.strip():
        train_kwargs["device"] = device.strip()
    model.train(**train_kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", YOLO_RUNS_DIR / run_name))
    emit(0.95, "saving best model")
    saved_model = _copy_best_model(save_dir, run_name)

    summary = {
        "status": "ok",
        "saved_model": str(saved_model),
        "base_model": base_model,
        "data_yaml": str(data_yaml),
        "save_dir": str(save_dir),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "class_names": class_info["class_names"],
        "class_id_map": class_info["class_id_map"],
        "recommended_class_ids": class_info["recommended_class_ids"],
        "hint_text": class_info["hint_text"],
        "dataset_preparation": dataset_preparation,
        "message": "训练完成，已保存 best.pt，可在 YOLO 模型下拉框选择。",
    }
    (saved_model.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        "utf-8",
    )
    emit(1.0, "done")
    return summary
