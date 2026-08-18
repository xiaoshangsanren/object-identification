from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(directory: Path) -> list[Path]:
    """
    方法作用：
        遍历指定目录，筛选出所有支持的图像文件并按路径排序返回。

    输入参数：
        directory (Path):
            需要扫描的图片目录。

    返回值：
        list[Path]:
            按文件名排序后的图片路径列表。
    """
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _sha256(path: Path) -> str:
    """
    方法作用：
        计算指定文件的 SHA-256 哈希值，用于判断图片是否重复。

    输入参数：
        path (Path):
            需要计算哈希的文件路径。

    返回值：
        str:
            文件对应的哈希字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_files(paths: list[Path], destination: Path) -> None:
    """
    方法作用：
        执行 _copy_files 对应的处理流程。
    
    输入参数：
        paths (list[Path])：方法所需的 paths 参数。
        destination (Path)：方法所需的 destination 参数。
    
    返回值：
        None：方法直接完成相应操作。
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def build_split(
    source_gallery: Path,
    evaluation_gallery: Path,
    split_root: Path,
    seed: int,
) -> dict[str, Any]:
    """
    方法作用：
        根据原始 Gallery 数据构建固定的 10 类 holdout 划分，
        将每个类别的大约 1/3 数据划为测试集，其余作为支持集，
        并生成可复现的 split_manifest.json 结果文件。

    输入参数：
        source_gallery (Path):
            原始 Gallery 数据目录。
        evaluation_gallery (Path):
            评估用 Gallery 输出目录。
        split_root (Path):
            测试集输出目录。
        seed (int):
            随机种子，用于控制划分和打乱顺序。

    返回值：
        dict[str, Any]:
            包含划分元信息、类别统计和测试顺序的结果字典。
    """
    if evaluation_gallery.exists() or split_root.exists():
        raise FileExistsError(
            "Gallery split output already exists; remove the versioned output explicitly "
            "before creating a different split"
        )
    classes_payload = json.loads(
        (source_gallery / "classes.json").read_text(encoding="utf-8")
    )
    classes = classes_payload.get("classes", [])
    if len(classes) != 10:
        raise ValueError(f"Expected exactly 10 Gallery classes, found {len(classes)}")

    rng = random.Random(seed)
    # 按类别依次构建支持集和测试集，并记录每个类别的划分统计信息。
    entries: list[dict[str, Any]] = []
    test_order: list[dict[str, str]] = []
    seen_hash_classes: dict[str, str] = {}
    for class_item in classes:
        class_id = str(class_item["id"])
        folder = str(class_item["folder"])
        source_paths = _images(source_gallery / "prototypes" / folder)
        if len(source_paths) < 6:
            raise ValueError(f"Class {class_id} has too few images: {len(source_paths)}")

        groups_by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in source_paths:
            content_hash = _sha256(path)
            previous_class = seen_hash_classes.setdefault(content_hash, class_id)
            if previous_class != class_id:
                raise ValueError(
                    f"Exact duplicate image is labelled as both {previous_class} and {class_id}"
                )
            groups_by_hash[content_hash].append(path)
        groups = list(groups_by_hash.values())
        rng.shuffle(groups)
        target_test_count = max(1, int(math.floor(len(source_paths) / 3.0 + 0.5)))
        test_paths: list[Path] = []
        support_paths: list[Path] = []
        for group in groups:
            if len(test_paths) < target_test_count:
                test_paths.extend(group)
            else:
                support_paths.extend(group)
        if not support_paths:
            raise ValueError(f"Split removed all support images for {class_id}")

        _copy_files(
            support_paths,
            evaluation_gallery / "prototypes" / folder,
        )
        _copy_files(test_paths, split_root / "test" / folder)
        entry = {
            "id": class_id,
            "folder": folder,
            "source_count": len(source_paths),
            "support_count": len(support_paths),
            "test_count": len(test_paths),
            "support": [path.name for path in support_paths],
            "test": [path.name for path in test_paths],
        }
        entries.append(entry)
        test_order.extend(
            {"class_id": class_id, "folder": folder, "file": path.name}
            for path in test_paths
        )

    rng.shuffle(test_order)
    # 将划分后的结果目录和元信息写入输出路径，便于后续评估脚本读取。
    evaluation_gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_gallery / "classes.json",
        evaluation_gallery / "classes.json",
    )
    shutil.copytree(
        source_gallery / "negatives",
        evaluation_gallery / "negatives",
    )
    shutil.copytree(
        source_gallery / "unknown_tanks",
        evaluation_gallery / "unknown_tanks",
    )

    payload: dict[str, Any] = {
        "format": "annotation_gallery_10class_holdout_v1",
        "seed": seed,
        "source_gallery": str(source_gallery.resolve()),
        "evaluation_gallery": str(evaluation_gallery.resolve()),
        "split_root": str(split_root.resolve()),
        "policy": {
            "test_fraction": "approximately 1/3 per class",
            "support_fraction": "approximately 2/3 per class",
            "exact_duplicate_hash_groups_kept_together": True,
            "test_order_shuffled": True,
        },
        "classes": entries,
        "test_order": test_order,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["split_sha256"] = hashlib.sha256(canonical).hexdigest()
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    """
    方法作用：
        解析命令行参数并执行当前脚本的主流程。
    
    输入参数：
        无。
    
    返回值：
        None：方法直接完成相应操作。
    """
    parser = argparse.ArgumentParser(description="Build the fixed 10-class Gallery holdout.")
    parser.add_argument(
        "--source-gallery",
        type=Path,
        default=ANNOTATION_ROOT / "Resource" / "clip_gallery",
    )
    parser.add_argument(
        "--evaluation-gallery",
        type=Path,
        default=ANNOTATION_ROOT / "Resource" / "clip_gallery_eval_10class_v1",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PAD_LITE_ROOT / "evaluation_data" / "gallery_10class_eval_v1",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    payload = build_split(
        args.source_gallery.resolve(),
        args.evaluation_gallery.resolve(),
        args.split_root.resolve(),
        args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
