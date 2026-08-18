from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_NAMES = (
    "bm-21",
    "bmd-2",
    "bmp-1",
    "bmp-2",
    "btr-70",
    "btr-80",
    "mt-lb",
    "t-64",
    "t-72",
    "t-80",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
RENAMED_IMAGE_PATTERN = re.compile(r"^(?P<class_name>.+)_(?P<number>[0-9]{3})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic PAD-Lite image/object metadata from the cleaned "
            "Russian-Military-Vehicles Pascal VOC annotations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Russian-Military-Vehicles dataset root containing train/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <dataset-root>/pad_lite_training_v1.json.",
    )
    parser.add_argument(
        "--context-padding",
        type=float,
        default=0.08,
        help="Context padding ratio recorded in the suggested crop box.",
    )
    parser.add_argument(
        "--tiny-area-threshold",
        type=float,
        default=0.02,
        help="Objects below this image-area ratio are flagged as tiny.",
    )
    return parser.parse_args()


def require_text(parent: ET.Element, path: str, source: Path) -> str:
    value = parent.findtext(path)
    if value is None or not value.strip():
        raise ValueError(f"Missing required XML field {path!r}: {source}")
    return value.strip()


def parse_int(parent: ET.Element, path: str, source: Path) -> int:
    value = require_text(parent, path, source)
    try:
        return int(round(float(value)))
    except ValueError as exc:
        raise ValueError(f"Invalid integer XML field {path!r}={value!r}: {source}") from exc


def convert_bbox(
    raw_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    context_padding: float,
) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = raw_bbox

    # Pascal VOC coordinates are treated as one-based at the left/top edge.
    # right/bottom are stored as exclusive zero-based coordinates for PIL/torch.
    left = max(0, min(width - 1, xmin - 1))
    top = max(0, min(height - 1, ymin - 1))
    right = max(left + 1, min(width, xmax))
    bottom = max(top + 1, min(height, ymax))

    bbox_width = right - left
    bbox_height = bottom - top
    area_ratio = (bbox_width * bbox_height) / float(width * height)

    pad_x = bbox_width * context_padding
    pad_y = bbox_height * context_padding
    crop_left = max(0, int(math.floor(left - pad_x)))
    crop_top = max(0, int(math.floor(top - pad_y)))
    crop_right = min(width, int(math.ceil(right + pad_x)))
    crop_bottom = min(height, int(math.ceil(bottom + pad_y)))

    was_clipped = xmin < 1 or ymin < 1 or xmax > width or ymax > height

    return {
        "bbox": [left, top, right, bottom],
        "crop_bbox": [crop_left, crop_top, crop_right, crop_bottom],
        "area_ratio": round(area_ratio, 6),
        "clipped": was_clipped,
    }


def build_metadata(
    dataset_root: Path,
    context_padding: float,
    tiny_area_threshold: float,
) -> dict[str, Any]:
    train_root = dataset_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_root}")

    xml_paths = sorted(train_root.glob("*.xml"), key=lambda path: path.name.lower())
    if not xml_paths:
        raise ValueError(f"No XML annotations found under {train_root}")

    sample_records: list[dict[str, Any]] = []
    quality_counts: Counter[str] = Counter()

    for image_index, xml_path in enumerate(xml_paths, start=1):
        root = ET.parse(xml_path).getroot()
        objects = root.findall("./object")
        if not objects:
            raise ValueError(
                "Cleaned dataset unexpectedly contains an XML without <object>: "
                f"{xml_path}"
            )

        xml_filename = require_text(root, "./filename", xml_path)
        image_path = train_root / xml_filename
        if not image_path.is_file():
            fallback = xml_path.with_suffix(".jpg")
            if not fallback.is_file():
                raise FileNotFoundError(
                    f"Missing image referenced by {xml_path}: {image_path}"
                )
            image_path = fallback

        width = parse_int(root, "./size/width", xml_path)
        height = parse_int(root, "./size/height", xml_path)
        depth = parse_int(root, "./size/depth", xml_path)
        if width <= 0 or height <= 0 or depth <= 0:
            raise ValueError(f"Invalid image dimensions in {xml_path}")

        renamed_match = RENAMED_IMAGE_PATTERN.fullmatch(image_path.stem)
        if renamed_match is None:
            raise ValueError(
                "Image does not follow <class>_<NNN>.<extension>: "
                f"{image_path}"
            )

        relative_image_path = image_path.relative_to(dataset_root).as_posix()
        relative_xml_path = xml_path.relative_to(dataset_root).as_posix()
        image_id = f"img_{image_index:04d}"
        image_class_names: set[str] = set()

        for object_index, object_node in enumerate(objects):
            class_name = require_text(object_node, "./name", xml_path)
            if class_name not in CLASS_TO_ID:
                raise ValueError(f"Unknown class {class_name!r}: {xml_path}")

            bbox_node = object_node.find("./bndbox")
            if bbox_node is None:
                raise ValueError(f"Missing bndbox in object {object_index}: {xml_path}")
            raw_bbox = (
                parse_int(bbox_node, "./xmin", xml_path),
                parse_int(bbox_node, "./ymin", xml_path),
                parse_int(bbox_node, "./xmax", xml_path),
                parse_int(bbox_node, "./ymax", xml_path),
            )
            bbox = convert_bbox(raw_bbox, width, height, context_padding)
            left, top, right, bottom = bbox["bbox"]
            if right <= left or bottom <= top:
                raise ValueError(f"Invalid converted bbox in {xml_path}: {raw_bbox}")

            sample_id = f"sample_{len(sample_records) + 1:05d}"

            class_id = CLASS_TO_ID[class_name]
            is_tiny = bbox["area_ratio"] < tiny_area_threshold
            is_crowded = len(objects) > 1
            if is_tiny:
                quality_counts["tiny_objects"] += 1
            else:
                quality_counts["core_objects"] += 1
            if is_crowded:
                quality_counts["crowded_objects"] += 1
            if bbox["clipped"]:
                quality_counts["clipped_boxes"] += 1

            image_class_names.add(class_name)
            sample_records.append(
                {
                    "id": sample_id,
                    "image_id": image_id,
                    "object_index": object_index,
                    "image": relative_image_path,
                    "xml": relative_xml_path,
                    "image_size": [width, height],
                    "class_id": class_id,
                    "class": class_name,
                    "anchor_id": f"ta_{class_id:02d}",
                    "bbox": bbox["bbox"],
                    "crop_bbox": bbox["crop_bbox"],
                    "area_ratio": bbox["area_ratio"],
                    "tiny": is_tiny,
                    "crowded": is_crowded,
                    "clipped": bbox["clipped"],
                }
            )

        if len(image_class_names) != 1:
            raise ValueError(
                f"Expected one class per source image, found {image_class_names}: "
                f"{xml_path}"
            )
        xml_class_name = next(iter(image_class_names))
        if renamed_match.group("class_name") != xml_class_name:
            raise ValueError(
                f"Image class prefix does not match XML class {xml_class_name!r}: "
                f"{image_path}"
            )

    classes = [
        {"id": class_id, "name": class_name, "anchor_id": f"ta_{class_id:02d}"}
        for class_name, class_id in CLASS_TO_ID.items()
    ]

    return {
        "meta": {
            "version": "pad_lite_simple_v1",
            "dataset": "Russian-Military-Vehicles",
            "source_format": "Pascal VOC",
            "task": "object_crop_classification_with_text_anchor",
            "bbox_format": "xyxy_0based_right_bottom_exclusive",
            "crop_context_padding": context_padding,
            "tiny_area_threshold": tiny_area_threshold,
            "text_anchor": {
                "template": "A photo of a {learnable_tokens} military vehicle.",
                "learnable_tokens": 4,
                "class_name_in_template": False,
                "training_only": True,
            },
            "counts": {
                "images": len(xml_paths),
                "samples": len(sample_records),
                "classes": len(CLASS_NAMES),
                "tiny": quality_counts["tiny_objects"],
                "crowded": quality_counts["crowded_objects"],
                "clipped": quality_counts["clipped_boxes"],
            },
        },
        "classes": classes,
        "samples": sample_records,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else dataset_root / "pad_lite_training_v1.json"
    )
    if not 0.0 <= args.context_padding <= 0.5:
        raise ValueError("--context-padding must be between 0.0 and 0.5")
    if not 0.0 < args.tiny_area_threshold < 1.0:
        raise ValueError("--tiny-area-threshold must be between 0.0 and 1.0")

    metadata = build_metadata(
        dataset_root=dataset_root,
        context_padding=args.context_padding,
        tiny_area_threshold=args.tiny_area_threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = metadata["meta"]["counts"]
    print(
        json.dumps(
            {
                "output": str(output),
                "source_image_count": stats["images"],
                "object_sample_count": stats["samples"],
                "class_count": stats["classes"],
                "tiny_object_count": stats["tiny"],
                "crowded_object_count": stats["crowded"],
                "clipped_bbox_count": stats["clipped"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
