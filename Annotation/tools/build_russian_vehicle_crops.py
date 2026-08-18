from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop one military-vehicle image per annotated object using the "
            "validated crop_bbox fields in pad_lite_training_v1.json."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/pad_lite_training_v1.json.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def validate_box(box: list[int], width: int, height: int, sample_id: str) -> None:
    if len(box) != 4:
        raise ValueError(f"Crop box for {sample_id} must contain four integers")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"Crop box outside image for {sample_id}: {box}, image={width}x{height}"
        )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else dataset_root / "pad_lite_training_v1.json"
    )
    if not 80 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 80 and 100")
    if output_root.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_root}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("version") != "pad_lite_simple_v1":
        raise ValueError(f"Unsupported source manifest: {manifest_path}")

    staging_root = output_root.parent / f".{output_root.name}.building"
    if staging_root.exists():
        raise FileExistsError(
            f"Incomplete staging directory already exists: {staging_root}"
        )
    (staging_root / "images").mkdir(parents=True)

    crop_records = []
    class_counts: Counter[str] = Counter()
    source_images: set[str] = set()
    seen_crop_names: set[str] = set()
    try:
        for sample in data["samples"]:
            source_path = dataset_root / sample["image"]
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source image: {source_path}")

            class_name = sample["class"]
            object_number = int(sample["object_index"]) + 1
            crop_name = f"{source_path.stem}_obj{object_number:02d}.jpg"
            if crop_name in seen_crop_names:
                raise ValueError(f"Duplicate crop filename: {crop_name}")
            seen_crop_names.add(crop_name)
            relative_crop = Path("images") / class_name / crop_name
            crop_path = staging_root / relative_crop
            crop_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source_path) as image:
                image.load()
                expected_width, expected_height = sample["image_size"]
                if image.size != (expected_width, expected_height):
                    raise ValueError(
                        f"Image size mismatch for {source_path}: "
                        f"file={image.size}, manifest={(expected_width, expected_height)}"
                    )
                crop_box = [int(value) for value in sample["crop_bbox"]]
                validate_box(crop_box, image.width, image.height, sample["id"])
                crop = image.crop(tuple(crop_box)).convert("RGB")
                crop.save(
                    crop_path,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                    optimize=True,
                )

            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            crop_id = crop_path.stem
            crop_records.append(
                {
                    "id": crop_id,
                    "file": relative_crop.as_posix(),
                    "source_image": source_path.name,
                    "class_id": sample["class_id"],
                    "class": class_name,
                    "anchor_id": sample["anchor_id"],
                    "object_index": sample["object_index"],
                    "crop_bbox": crop_box,
                    "size": [crop_width, crop_height],
                    "tiny": sample["tiny"],
                    "crowded": sample["crowded"],
                    "clipped": sample["clipped"],
                }
            )
            class_counts[class_name] += 1
            source_images.add(source_path.name)

        crop_manifest = {
            "meta": {
                "version": "russian_pad_crops_v1",
                "source_manifest": manifest_path.name,
                "crop_method": "pascal_voc_bbox_with_context",
                "context_padding": data["meta"]["crop_context_padding"],
                "image_format": "JPEG",
                "jpeg_quality": args.jpeg_quality,
                "resize_applied": False,
                "fold_mapping_key": "source_image",
                "counts": {
                    "source_images": len(source_images),
                    "crops": len(crop_records),
                    "classes": len(class_counts),
                },
            },
            "classes": data["classes"],
            "samples": crop_records,
        }
        (staging_root / "crop_manifest.json").write_text(
            json.dumps(crop_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_root.rename(output_root)
    except Exception:
        print(
            f"error: crop generation stopped; partial output kept at {staging_root}",
            file=sys.stderr,
        )
        raise

    print(
        json.dumps(
            {
                "output": str(output_root),
                "source_images": len(source_images),
                "crops": len(crop_records),
                "classes": len(class_counts),
                "class_crop_counts": dict(sorted(class_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        ValueError,
        json.JSONDecodeError,
        UnidentifiedImageError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
