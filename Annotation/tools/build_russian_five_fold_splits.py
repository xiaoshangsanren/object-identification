from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic five-fold class-held-out splits for the cleaned "
            "Russian-Military-Vehicles PAD-Lite manifest."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/pad_lite_training_v1.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/pad_lite_splits_v1/.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-val-ratio", type=float, default=0.10)
    parser.add_argument("--novel-support-images", type=int, default=10)
    return parser.parse_args()


def tagged_rng(seed: int, tag: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_source_images(
    manifest_path: Path,
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("meta", {}).get("version") != "pad_lite_simple_v1":
        raise ValueError(f"Unsupported PAD-Lite manifest: {manifest_path}")

    classes = [item["name"] for item in data["classes"]]
    images_by_class: dict[str, set[str]] = {name: set() for name in classes}
    image_to_class: dict[str, str] = {}
    objects_per_image: Counter[str] = Counter()
    for sample in data["samples"]:
        class_name = sample["class"]
        image_name = Path(sample["image"]).name
        if class_name not in images_by_class:
            raise ValueError(f"Unknown class in sample {sample['id']}: {class_name}")
        previous_class = image_to_class.setdefault(image_name, class_name)
        if previous_class != class_name:
            raise ValueError(
                f"Source image belongs to multiple classes: {image_name}"
            )
        images_by_class[class_name].add(image_name)
        objects_per_image[image_name] += 1

    sorted_images = {
        class_name: sorted(image_names)
        for class_name, image_names in images_by_class.items()
    }
    return classes, sorted_images, dict(objects_per_image)


def split_class_images(
    class_name: str,
    image_names: list[str],
    seed: int,
    base_val_ratio: float,
    novel_support_images: int,
) -> dict[str, list[str]]:
    base_order = list(image_names)
    tagged_rng(seed, f"{class_name}:base").shuffle(base_order)
    val_count = max(1, int(math.ceil(len(base_order) * base_val_ratio)))

    novel_order = list(image_names)
    tagged_rng(seed, f"{class_name}:novel").shuffle(novel_order)
    if len(novel_order) <= novel_support_images:
        raise ValueError(
            f"Class {class_name} has {len(novel_order)} images, not enough for "
            f"{novel_support_images} support images plus query images"
        )

    return {
        "base_train": sorted(base_order[val_count:]),
        "base_val": sorted(base_order[:val_count]),
        "novel_support": sorted(novel_order[:novel_support_images]),
        "novel_query": sorted(novel_order[novel_support_images:]),
    }


def flatten(partition: dict[str, list[str]]) -> list[str]:
    return [name for class_names in partition.values() for name in class_names]


def partition_counts(
    partition: dict[str, list[str]], objects_per_image: dict[str, int]
) -> dict[str, int]:
    images = flatten(partition)
    return {
        "source_images": len(images),
        "object_samples": sum(objects_per_image[name] for name in images),
    }


def validate_fold(
    fold: dict[str, Any],
    all_classes: set[str],
    all_images: set[str],
    support_images_per_class: int,
) -> None:
    base_classes = set(fold["base_classes"])
    novel_classes = set(fold["novel_classes"])
    if base_classes & novel_classes or base_classes | novel_classes != all_classes:
        raise ValueError(f"Invalid class partition in fold {fold['fold']}")

    partitions = fold["partitions"]
    expected_keys = {
        "base_train": base_classes,
        "base_val": base_classes,
        "novel_support": novel_classes,
        "novel_query": novel_classes,
    }
    image_sets: dict[str, set[str]] = {}
    for partition_name, expected_classes in expected_keys.items():
        partition = partitions[partition_name]
        if set(partition) != expected_classes:
            raise ValueError(
                f"Wrong classes in {partition_name}, fold {fold['fold']}"
            )
        names = flatten(partition)
        if len(names) != len(set(names)):
            raise ValueError(
                f"Duplicate image inside {partition_name}, fold {fold['fold']}"
            )
        image_sets[partition_name] = set(names)

    partition_names = list(image_sets)
    for index, left_name in enumerate(partition_names):
        for right_name in partition_names[index + 1 :]:
            overlap = image_sets[left_name] & image_sets[right_name]
            if overlap:
                raise ValueError(
                    f"Image leakage between {left_name} and {right_name}, "
                    f"fold {fold['fold']}: {sorted(overlap)[:5]}"
                )
    covered = set().union(*image_sets.values())
    if covered != all_images:
        raise ValueError(
            f"Fold {fold['fold']} does not cover every source image exactly once"
        )
    for class_name, images in partitions["novel_support"].items():
        if len(images) != support_images_per_class:
            raise ValueError(
                f"Fold {fold['fold']} class {class_name} has {len(images)} "
                "support images"
            )


def build_splits(
    classes: list[str],
    images_by_class: dict[str, list[str]],
    objects_per_image: dict[str, int],
    seed: int,
    base_val_ratio: float,
    novel_support_images: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(classes) != 10:
        raise ValueError(f"Expected 10 classes for five 2-class folds, got {len(classes)}")

    per_class_splits = {
        class_name: split_class_images(
            class_name,
            images_by_class[class_name],
            seed,
            base_val_ratio,
            novel_support_images,
        )
        for class_name in classes
    }

    shuffled_classes = list(classes)
    tagged_rng(seed, "class_folds").shuffle(shuffled_classes)
    novel_pairs = [
        sorted(shuffled_classes[index : index + 2])
        for index in range(0, len(shuffled_classes), 2)
    ]

    all_classes = set(classes)
    all_images = {
        image_name
        for class_images in images_by_class.values()
        for image_name in class_images
    }
    folds: list[dict[str, Any]] = []
    for fold_number, novel_classes in enumerate(novel_pairs, start=1):
        base_classes = [name for name in classes if name not in novel_classes]
        partitions = {
            "base_train": {
                name: per_class_splits[name]["base_train"] for name in base_classes
            },
            "base_val": {
                name: per_class_splits[name]["base_val"] for name in base_classes
            },
            "novel_support": {
                name: per_class_splits[name]["novel_support"]
                for name in novel_classes
            },
            "novel_query": {
                name: per_class_splits[name]["novel_query"] for name in novel_classes
            },
        }
        counts = {
            name: partition_counts(partition, objects_per_image)
            for name, partition in partitions.items()
        }
        fold = {
            "version": "pad_lite_class_holdout_fold_v1",
            "fold": fold_number,
            "seed": seed,
            "image_root": "train",
            "base_classes": base_classes,
            "novel_classes": novel_classes,
            "partitions": partitions,
            "counts": counts,
        }
        validate_fold(
            fold,
            all_classes,
            all_images,
            novel_support_images,
        )
        folds.append(fold)

    novel_appearances = Counter(
        class_name for fold in folds for class_name in fold["novel_classes"]
    )
    if novel_appearances != Counter({name: 1 for name in classes}):
        raise ValueError(f"Classes are not held out exactly once: {novel_appearances}")

    summary = {
        "version": "pad_lite_class_holdout_5fold_v1",
        "source_manifest": "pad_lite_training_v1.json",
        "seed": seed,
        "rules": {
            "folds": 5,
            "base_classes_per_fold": 8,
            "novel_classes_per_fold": 2,
            "base_val_ratio": base_val_ratio,
            "novel_support_source_images_per_class": novel_support_images,
            "split_unit": "source_image",
            "gallery_used": False,
        },
        "class_image_counts": {
            name: len(images_by_class[name]) for name in classes
        },
        "folds": [
            {
                "fold": fold["fold"],
                "file": f"fold_{fold['fold']:02d}.json",
                "novel_classes": fold["novel_classes"],
                "base_classes": fold["base_classes"],
                "counts": fold["counts"],
            }
            for fold in folds
        ],
    }
    return summary, folds


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else dataset_root / "pad_lite_training_v1.json"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "pad_lite_splits_v1"
    )
    if not 0.0 < args.base_val_ratio < 0.5:
        raise ValueError("--base-val-ratio must be between 0 and 0.5")
    if args.novel_support_images < 1:
        raise ValueError("--novel-support-images must be positive")

    classes, images_by_class, objects_per_image = load_source_images(manifest_path)
    summary, folds = build_splits(
        classes,
        images_by_class,
        objects_per_image,
        args.seed,
        args.base_val_ratio,
        args.novel_support_images,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_path = output_dir / f"fold_{fold['fold']:02d}.json"
        fold_path.write_text(
            json.dumps(fold, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
