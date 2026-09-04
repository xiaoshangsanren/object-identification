from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "datasets/processed/russian_pad"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / "datasets/processed/russian_pad_min64_v1"
DEFAULT_SOURCE_SPLITS = (
    PROJECT_ROOT.parent
    / "datasets/Russian-Military-Vehicles/pad_lite_splits_v1"
)
DEFAULT_OUTPUT_SPLITS = (
    PROJECT_ROOT.parent
    / "datasets/Russian-Military-Vehicles/pad_lite_splits_min64_v1"
)


def _read_json(path: Path) -> Any:
    """
    方法作用：
        读取 UTF-8 JSON 文件。

    输入参数：
        path (Path)：JSON 文件路径。

    返回值：
        Any：反序列化后的 JSON 数据。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        以稳定、可读的格式写入 UTF-8 JSON 文件。

    输入参数：
        path (Path)：输出 JSON 文件路径。
        payload (Any)：待序列化的数据。

    返回值：
        None：仅完成文件写入。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """
    方法作用：
        计算文件 SHA-256，用于记录数据来源和验证可追溯性。

    输入参数：
        path (Path)：待计算摘要的文件。

    返回值：
        str：小写十六进制 SHA-256 摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_file(source: Path, destination: Path) -> None:
    """
    方法作用：
        在新数据视图中创建指向原文件的符号链接，不复制和修改原图片。

    输入参数：
        source (Path)：原始图片路径。
        destination (Path)：新视图中的链接路径。

    返回值：
        None：仅创建符号链接。
    """
    if not source.is_file():
        raise FileNotFoundError(f"Missing source image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _copy_and_filter_splits(
    source_root: Path,
    output_root: Path,
    kept_source_images: set[str],
) -> None:
    """
    方法作用：
        复制原五折划分，并移除已经没有任何合格裁剪图的源图片。

    输入参数：
        source_root (Path)：原五折划分目录。
        output_root (Path)：新五折划分目录。
        kept_source_images (set[str])：至少含一张合格裁剪图的源图片名集合。

    返回值：
        None：写出独立的 min64 五折划分。
    """
    if output_root.exists():
        raise FileExistsError(f"Output splits already exist: {output_root}")
    output_root.mkdir(parents=True)
    for source_path in sorted(source_root.glob("*.json")):
        payload = _read_json(source_path)
        if source_path.name.startswith("fold_"):
            for partition in payload.get("partitions", {}).values():
                for class_name, names in partition.items():
                    partition[class_name] = [
                        name for name in names if name in kept_source_images
                    ]
            payload.pop("fixed_gallery", None)
            payload["resolution_filter"] = {
                "rule": "retain source images with at least one crop whose "
                "min(width, height) >= 64",
                "source_split": str(source_path.resolve()),
                "source_split_sha256": _sha256(source_path),
            }
        _write_json(output_root / source_path.name, payload)


def build_filtered_view(
    source_root: Path,
    output_root: Path,
    source_splits: Path,
    output_splits: Path,
    min_side: int,
) -> dict[str, Any]:
    """
    方法作用：
        构建不改动原数据的分辨率过滤视图，并把低分辨率样本放入隔离区。

    输入参数：
        source_root (Path)：原 russian_pad 数据根目录。
        output_root (Path)：新过滤数据视图根目录。
        source_splits (Path)：原五折划分目录。
        output_splits (Path)：新五折划分目录。
        min_side (int)：合格图片最短边阈值，单位为像素。

    返回值：
        dict[str, Any]：包含保留数、隔离数和分类统计的执行摘要。
    """
    if min_side <= 0:
        raise ValueError("min_side must be positive")
    if output_root.exists():
        raise FileExistsError(f"Output view already exists: {output_root}")

    source_manifest_path = source_root / "crop_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    samples = source_manifest["samples"]
    kept = [item for item in samples if min(map(int, item["size"])) >= min_side]
    isolated = [item for item in samples if min(map(int, item["size"])) < min_side]
    if len(kept) + len(isolated) != len(samples):
        raise AssertionError("Resolution partition is incomplete")

    output_parent = output_root.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_parent)
    )
    try:
        for item in kept:
            relative_path = Path(item["file"])
            _link_file(source_root / relative_path, temporary_root / relative_path)
        for item in isolated:
            relative_path = Path(item["file"])
            _link_file(
                source_root / relative_path,
                temporary_root / "isolated_lt64" / relative_path,
            )

        manifest = json.loads(json.dumps(source_manifest))
        manifest["samples"] = kept
        manifest["meta"]["counts"]["crops"] = len(kept)
        manifest["meta"]["counts"]["source_images"] = len(
            {item["source_image"] for item in kept}
        )
        manifest["meta"]["resolution_filter"] = {
            "minimum_short_side_px": min_side,
            "rule": f"min(width, height) >= {min_side}",
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "original_crop_count": len(samples),
            "kept_crop_count": len(kept),
            "isolated_crop_count": len(isolated),
        }
        filtered_manifest_path = temporary_root / "crop_manifest.json"
        _write_json(filtered_manifest_path, manifest)

        isolated_rows = [
            {
                "id": item["id"],
                "file": item["file"],
                "isolated_file": str(Path("isolated_lt64") / item["file"]),
                "class": item["class"],
                "source_image": item["source_image"],
                "width": int(item["size"][0]),
                "height": int(item["size"][1]),
                "short_side": min(map(int, item["size"])),
                "reason": f"min(width, height) < {min_side}",
            }
            for item in sorted(
                isolated,
                key=lambda row: (min(map(int, row["size"])), row["id"]),
            )
        ]
        isolation_manifest = {
            "format": "russian_pad_resolution_isolation_v1",
            "policy": {
                "threshold_px": min_side,
                "comparison": "strictly_less_than",
                "measurement": "min(width, height)",
                "files_are_symlinks": True,
                "source_data_modified": False,
            },
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "count": len(isolated_rows),
            "class_counts": dict(
                sorted(Counter(row["class"] for row in isolated_rows).items())
            ),
            "images": isolated_rows,
        }
        _write_json(
            temporary_root / "isolated_lt64_manifest.json",
            isolation_manifest,
        )

        readme = f"""# Russian PAD min{min_side} 数据视图

- 保留条件：`min(width, height) >= {min_side}`。
- 合格样本：{len(kept)} 张，位于 `images/`；均为指向原图的符号链接。
- 隔离样本：{len(isolated)} 张，位于 `isolated_lt64/images/`。
- 原始 `russian_pad` 未被移动、覆盖或删除。
- 完整隔离列表见 `isolated_lt64_manifest.json`。
"""
        (temporary_root / "README.md").write_text(readme, encoding="utf-8")

        os.replace(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    kept_sources = {item["source_image"] for item in kept}
    _copy_and_filter_splits(
        source_splits,
        output_splits,
        kept_sources,
    )
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_splits": str(source_splits),
        "output_splits": str(output_splits),
        "minimum_short_side_px": min_side,
        "original_count": len(samples),
        "kept_count": len(kept),
        "isolated_count": len(isolated),
        "isolated_class_counts": dict(
            sorted(Counter(item["class"] for item in isolated).items())
        ),
        "source_images_removed_completely": sorted(
            {item["source_image"] for item in samples} - kept_sources
        ),
    }
    _write_json(output_root / "build_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        构建分辨率过滤数据视图命令行参数解析器。

    输入参数：
        无。

    返回值：
        argparse.ArgumentParser：命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="Build a non-destructive min-resolution russian_pad view."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-splits", type=Path, default=DEFAULT_SOURCE_SPLITS)
    parser.add_argument("--output-splits", type=Path, default=DEFAULT_OUTPUT_SPLITS)
    parser.add_argument("--min-side", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    方法作用：
        执行低分辨率图片隔离并打印构建摘要。

    输入参数：
        argv (Sequence[str] | None)：命令行参数；None 时读取 sys.argv。

    返回值：
        int：成功时返回 0。
    """
    args = build_parser().parse_args(argv)
    summary = build_filtered_view(
        args.source_root.resolve(),
        args.output_root.resolve(),
        args.source_splits.resolve(),
        args.output_splits.resolve(),
        args.min_side,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
