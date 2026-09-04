from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT_VERSION = "russian_pad_monocular_pointcloud_v1.00"
DEFAULT_SOURCE = PROJECT_ROOT / "datasets/processed/russian_pad"
DEFAULT_MANIFEST = DEFAULT_SOURCE / "crop_manifest.json"
DEFAULT_SPLITS = (
    PROJECT_ROOT / "datasets/Russian-Military-Vehicles/pad_lite_splits_v1"
)
DEFAULT_MODEL = (
    PROJECT_ROOT / "Annotation/Resource/models/depth-anything-v2-small-hf"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/processed/russian_pad_pointcloud_v1"


def _read_json(path: Path) -> dict[str, Any]:
    """
    方法作用：读取顶层为对象的UTF-8 JSON。
    输入参数：path。
    返回值：dict。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：写入稳定、可读的UTF-8 JSON。
    输入参数：path；payload。
    返回值：None。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    """
    方法作用：流式计算文件SHA-256。
    输入参数：path。
    返回值：十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_pixel_indices(
    height: int, width: int, point_count: int
) -> np.ndarray:
    """
    方法作用：以近似等间距二维网格确定性抽取像素，不依赖随机种子。
    输入参数：图像height/width；目标point_count=P。
    返回值：扁平像素下标``[min(HW,P)]``，无重复且升序不作保证。
    """
    if height <= 0 or width <= 0 or point_count <= 0:
        raise ValueError("height, width and point_count must be positive")
    available = int(height * width)
    target = min(available, int(point_count))
    if target == available:
        return np.arange(available, dtype=np.int64)
    rows = min(height, max(1, int(round(math.sqrt(target * height / width)))))
    columns = min(width, max(1, int(math.ceil(target / rows))))
    y = np.unique(np.rint(np.linspace(0, height - 1, rows)).astype(np.int64))
    x = np.unique(np.rint(np.linspace(0, width - 1, columns)).astype(np.int64))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    selected = np.unique((yy * width + xx).reshape(-1))
    if len(selected) > target:
        keep = np.rint(np.linspace(0, len(selected) - 1, target)).astype(np.int64)
        selected = selected[keep]
    if len(selected) < target:
        remaining = np.setdiff1d(
            np.arange(available, dtype=np.int64), selected, assume_unique=True
        )
        take = np.rint(np.linspace(0, len(remaining) - 1, target - len(selected))).astype(
            np.int64
        )
        selected = np.concatenate((selected, remaining[take]))
    if len(np.unique(selected)) != target:
        raise RuntimeError("Deterministic pixel sampler produced duplicate indices")
    return selected.astype(np.int64, copy=False)


def depth_to_fixed_pointcloud(
    rgb: np.ndarray,
    relative_depth: np.ndarray,
    point_count: int,
) -> dict[str, np.ndarray | float | int]:
    """
    方法作用：把RGB和模型原生相对深度转换为对象中心化的定长2.5D点云。
    输入参数：rgb``[H,W,3] uint8``；relative_depth``[H,W]``；点数P。
    返回值：xyz/rgb/pixel_xy/depth/mask均以P为首维，并含稳健归一化统计。
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {rgb.shape}")
    if relative_depth.shape != rgb.shape[:2]:
        raise ValueError("Depth and RGB spatial shapes differ")
    if not np.isfinite(relative_depth).all():
        raise ValueError("Depth map contains NaN or Inf")
    height, width = relative_depth.shape
    low, high = np.percentile(relative_depth.astype(np.float64), (2.0, 98.0))
    spread = max(float(high - low), 1e-6)
    depth_normalized = np.clip((relative_depth - low) / spread, 0.0, 1.0).astype(
        np.float32
    )
    indices = deterministic_pixel_indices(height, width, point_count)
    y = indices // width
    x = indices % width
    denominator = float(max(height, width, 1))
    xyz_valid = np.stack(
        (
            (x.astype(np.float32) - (width - 1) / 2.0) / denominator,
            -((y.astype(np.float32) - (height - 1) / 2.0) / denominator),
            depth_normalized[y, x] - 0.5,
        ),
        axis=1,
    )
    # 每张图的相对深度只在仿射意义上有效，因此中心化并归一到单位球。
    center = np.median(xyz_valid, axis=0)
    xyz_valid = xyz_valid - center
    radii = np.linalg.norm(xyz_valid, axis=1)
    unit_scale = max(float(np.percentile(radii, 95.0)), 1e-6)
    xyz_valid = (xyz_valid / unit_scale).astype(np.float32)
    valid_count = len(indices)
    xyz = np.zeros((point_count, 3), dtype=np.float32)
    colors = np.zeros((point_count, 3), dtype=np.uint8)
    pixels = np.zeros((point_count, 2), dtype=np.uint16)
    depths = np.zeros((point_count,), dtype=np.float32)
    mask = np.zeros((point_count,), dtype=np.bool_)
    xyz[:valid_count] = xyz_valid
    colors[:valid_count] = rgb[y, x]
    pixels[:valid_count] = np.stack((x, y), axis=1).astype(np.uint16)
    depths[:valid_count] = depth_normalized[y, x]
    mask[:valid_count] = True
    return {
        "xyz": xyz,
        "rgb": colors,
        "pixel_xy": pixels,
        "relative_depth": depths,
        "valid_mask": mask,
        "dense_relative_depth": depth_normalized.astype(np.float16),
        "valid_count": valid_count,
        "raw_depth_p02": float(low),
        "raw_depth_p98": float(high),
        "xyz_center": center.astype(np.float32),
        "xyz_unit_scale": unit_scale,
    }


def write_binary_ply(
    path: Path, xyz: np.ndarray, rgb: np.ndarray, valid_mask: np.ndarray
) -> None:
    """
    方法作用：输出带RGB的二进制Little-Endian PLY，便于CloudCompare/Open3D查看。
    输入参数：xyz/rgb``[P,3]``；valid_mask``[P]``。
    返回值：None。
    """
    points = np.asarray(xyz[valid_mask], dtype="<f4")
    colors = np.asarray(rgb[valid_mask], dtype=np.uint8)
    if len(points) != len(colors):
        raise ValueError("PLY point/color count mismatch")
    payload = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    payload["x"], payload["y"], payload["z"] = points.T
    payload["red"], payload["green"], payload["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(header)
        payload.tofile(stream)


class RelativeDepthEncoder:
    """
    方法作用：离线加载Depth Anything V2并批量输出原图大小的相对深度。
    输入参数：本地模型目录、设备和是否使用FP16 autocast。
    返回值：可调用encoder，输入B张PIL图，返回B个``[Hi,Wi]``数组。
    """

    def __init__(self, model_root: Path, device: torch.device, use_fp16: bool) -> None:
        """
        方法作用：只从本地文件载入处理器与深度模型，禁止隐式联网。
        输入参数：model_root；device；use_fp16。
        返回值：None。
        """
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(
            model_root, local_files_only=True
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_root, local_files_only=True
        ).to(device).eval()
        self.device = device
        self.use_fp16 = bool(use_fp16 and device.type == "cuda")

    @torch.inference_mode()
    def __call__(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        """
        方法作用：保持长宽比地边缘填充成正方形，批量估深后裁回原尺寸。
        输入参数：B张RGB PIL图。
        返回值：B个float32数组，各为``[Hi,Wi]``。
        """
        square_images: list[Image.Image] = []
        crop_windows: list[tuple[int, int, int, int]] = []
        square_sizes: list[tuple[int, int]] = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            height, width = rgb.shape[:2]
            side = max(height, width)
            left = (side - width) // 2
            right = side - width - left
            top = (side - height) // 2
            bottom = side - height - top
            padded = np.pad(
                rgb,
                ((top, bottom), (left, right), (0, 0)),
                mode="edge",
            )
            square_images.append(Image.fromarray(padded, mode="RGB"))
            crop_windows.append((left, top, left + width, top + height))
            square_sizes.append((side, side))
        inputs = self.processor(images=square_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            outputs = self.model(**inputs)
        rows = self.processor.post_process_depth_estimation(
            outputs, target_sizes=square_sizes
        )
        result: list[np.ndarray] = []
        for row, (left, top, right, bottom) in zip(
            rows, crop_windows, strict=True
        ):
            depth = row["predicted_depth"].float().cpu().numpy()
            result.append(depth[top:bottom, left:right].astype(np.float32))
        for image in square_images:
            image.close()
        return result


def build_fivefold_protocols(
    split_root: Path,
    samples: Sequence[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """
    方法作用：把原五折source-image划分无泄漏映射到crop级点云记录。
    输入参数：原split根、点云sample清单、输出根。
    返回值：五折计数与完整性摘要。
    """
    by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_sample = {str(item["sample_id"]): item for item in samples}
    for item in samples:
        by_source.setdefault((str(item["class"]), str(item["source_image"])), []).append(
            item
        )
    summary: list[dict[str, Any]] = []
    for fold in range(1, 6):
        source = _read_json(split_root / f"fold_{fold:02d}.json")
        partitions: dict[str, list[dict[str, Any]]] = {}
        for partition in ("base_train", "base_val", "novel_query"):
            records: list[dict[str, Any]] = []
            for class_name, source_names in source["partitions"][partition].items():
                for source_name in source_names:
                    records.extend(by_source.get((class_name, source_name), []))
            partitions[partition] = sorted(records, key=lambda item: item["sample_id"])
        support_ids = {
            sample_id
            for values in source["fixed_gallery"]["sample_ids_by_class"].values()
            for sample_id in values
        }
        fixed_gallery = [by_sample[item] for item in sorted(support_ids)]
        partitions["novel_support"] = fixed_gallery
        support_source_names = {
            str(item["source_image"]) for item in fixed_gallery
        }
        query_source_names = {
            str(item["source_image"]) for item in partitions["novel_query"]
        }
        if support_source_names & query_source_names:
            raise ValueError(f"Fold {fold} has source-image leakage between gallery/query")
        partition_support_ids = {
            str(item["sample_id"]) for item in partitions["novel_support"]
        }
        if not support_ids <= partition_support_ids:
            raise ValueError(f"Fold {fold} fixed gallery is outside novel_support")
        compact = {
            name: [
                {
                    key: item[key]
                    for key in (
                        "sample_id", "class", "class_id", "source_image",
                        "image", "pointcloud_npz", "pointcloud_ply",
                    )
                }
                for item in records
            ]
            for name, records in partitions.items()
        }
        counts = {name: len(records) for name, records in compact.items()}
        expected = source["counts"]
        # base计数仍与原折一致；固定语义Gallery后来替换过support源图，旧JSON中的
        # novel_support/query object_samples未同步更新，故novel计数以当前精确ID为准。
        for name in ("base_train", "base_val"):
            count = counts[name]
            expected_object_samples = int(expected[name]["object_samples"])
            if expected_object_samples != count:
                raise ValueError(
                    f"Fold {fold} {name} object-sample count mismatch: "
                    f"{count} != {expected_object_samples}"
                )
        payload = {
            "format": FORMAT_VERSION,
            "fold": fold,
            "base_classes": source["base_classes"],
            "novel_classes": source["novel_classes"],
            "counts": counts,
            "partitions": compact,
            "fixed_gallery": [
                {
                    key: item[key]
                    for key in (
                        "sample_id", "class", "class_id", "source_image",
                        "image", "pointcloud_npz", "pointcloud_ply",
                    )
                }
                for item in fixed_gallery
            ],
            "source_split": str((split_root / f"fold_{fold:02d}.json").resolve()),
            "source_split_sha256": _sha256(split_root / f"fold_{fold:02d}.json"),
        }
        _write_json(output_root / "protocols/fivefold" / f"fold_{fold:02d}.json", payload)
        summary.append(
            {
                "fold": fold,
                "base_classes": source["base_classes"],
                "novel_classes": source["novel_classes"],
                "counts": counts,
            }
        )
    result = {"format": FORMAT_VERSION, "fold_count": 5, "folds": summary}
    _write_json(output_root / "protocols/fivefold/summary.json", result)
    return result


def render_contact_sheet(
    preview_items: Sequence[tuple[dict[str, Any], np.ndarray]],
    output_path: Path,
) -> None:
    """
    方法作用：每类选一张图并绘制RGB、相对深度和RGB点云三联图。
    输入参数：样本与dense depth序列；输出PNG。
    返回值：None。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(preview_items)
    figure = plt.figure(figsize=(12, max(3, rows * 3)))
    for index, (item, depth) in enumerate(preview_items):
        rgb = np.asarray(Image.open(item["source_path"]).convert("RGB"))
        cloud = np.load(item["npz_path"], allow_pickle=False)
        mask = cloud["valid_mask"]
        xyz = cloud["xyz"][mask]
        colors = cloud["rgb"][mask].astype(np.float32) / 255.0
        stride = max(1, len(xyz) // 1800)
        axis = figure.add_subplot(rows, 3, index * 3 + 1)
        axis.imshow(rgb)
        axis.set_title(f"{item['class']} / {item['sample_id']}")
        axis.axis("off")
        axis = figure.add_subplot(rows, 3, index * 3 + 2)
        axis.imshow(depth, cmap="turbo")
        axis.set_title("per-image relative depth")
        axis.axis("off")
        axis = figure.add_subplot(rows, 3, index * 3 + 3, projection="3d")
        axis.scatter(
            xyz[::stride, 0], xyz[::stride, 2], xyz[::stride, 1],
            c=colors[::stride], s=0.8, depthshade=False,
        )
        axis.view_init(elev=18, azim=-72)
        axis.set_title("normalized 2.5D cloud")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_dataset(
    source_root: Path,
    crop_manifest_path: Path,
    split_root: Path,
    model_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    point_count: int,
    use_fp16: bool,
) -> dict[str, Any]:
    """
    方法作用：全量生成NPZ/PLY、五折映射、预览和可追溯数据清单。
    输入参数：数据/模型/输出路径、设备、批量、定长点数及FP16开关。
    返回值：数据集汇总字典。
    """
    source_root = source_root.resolve()
    model_root = model_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite point-cloud dataset: {output_root}")
    manifest = _read_json(crop_manifest_path.resolve())
    samples = list(manifest["samples"])
    staging = output_root.parent / f".{output_root.name}.staging_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    encoder = RelativeDepthEncoder(model_root, device, use_fp16)
    output_samples: list[dict[str, Any]] = []
    preview_by_class: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    try:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            pil_images: list[Image.Image] = []
            rgb_arrays: list[np.ndarray] = []
            source_paths: list[Path] = []
            for item in batch:
                source_path = source_root / item["file"]
                with Image.open(source_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                pil_images.append(image)
                rgb_arrays.append(np.asarray(image, dtype=np.uint8))
                source_paths.append(source_path)
            depths = encoder(pil_images)
            for item, source_path, rgb, depth in zip(
                batch, source_paths, rgb_arrays, depths, strict=True
            ):
                cloud = depth_to_fixed_pointcloud(rgb, depth, point_count)
                class_name = str(item["class"])
                stem = Path(item["file"]).stem
                npz_relative = Path("points_npz") / class_name / f"{stem}.npz"
                ply_relative = Path("points_ply") / class_name / f"{stem}.ply"
                npz_path = staging / npz_relative
                ply_path = staging / ply_relative
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz_path,
                    xyz=cloud["xyz"],
                    rgb=cloud["rgb"],
                    pixel_xy=cloud["pixel_xy"],
                    relative_depth=cloud["relative_depth"],
                    valid_mask=cloud["valid_mask"],
                    dense_relative_depth=cloud["dense_relative_depth"],
                    image_size_wh=np.asarray([rgb.shape[1], rgb.shape[0]], np.int32),
                )
                write_binary_ply(
                    ply_path,
                    cloud["xyz"],  # type: ignore[arg-type]
                    cloud["rgb"],  # type: ignore[arg-type]
                    cloud["valid_mask"],  # type: ignore[arg-type]
                )
                result_item = {
                    "sample_id": str(item["id"]),
                    "class": class_name,
                    "class_id": int(item["class_id"]),
                    "source_image": str(item["source_image"]),
                    "image": str(item["file"]),
                    "image_width": int(rgb.shape[1]),
                    "image_height": int(rgb.shape[0]),
                    "short_side_lt64": min(rgb.shape[:2]) < 64,
                    "valid_point_count": int(cloud["valid_count"]),
                    "point_count": point_count,
                    "pointcloud_npz": npz_relative.as_posix(),
                    "pointcloud_ply": ply_relative.as_posix(),
                    "raw_depth_p02": float(cloud["raw_depth_p02"]),
                    "raw_depth_p98": float(cloud["raw_depth_p98"]),
                    "xyz_unit_scale": float(cloud["xyz_unit_scale"]),
                    "source_sha256": _sha256(source_path),
                    # 以下仅用于本次构建预览，不写入最终manifest。
                    "source_path": str(source_path),
                    "npz_path": str(npz_path),
                }
                output_samples.append(result_item)
                preview_by_class.setdefault(class_name, (result_item, depth))
            for image in pil_images:
                image.close()
            completed = min(start + len(batch), len(samples))
            print(f"point-cloud conversion: {completed}/{len(samples)}", flush=True)

        # 协议路径相对于最终数据根，与staging目录名无关。
        serializable_samples = [
            {key: value for key, value in item.items() if key not in {"source_path", "npz_path"}}
            for item in output_samples
        ]
        class_counts = dict(sorted(Counter(item["class"] for item in serializable_samples).items()))
        dataset_manifest = {
            "format": FORMAT_VERSION,
            "description": "Depth Anything V2 relative-depth 2.5D pseudo point clouds",
            "limitations": [
                "single-view visible surface only; no back-side geometry",
                "relative depth has no metric scale and is normalized per image",
                "camera intrinsics are unavailable; xyz uses normalized image coordinates",
                "vehicle crops retain some background because no expert foreground mask is used",
            ],
            "source_root": str(source_root),
            "source_manifest": str(crop_manifest_path.resolve()),
            "source_manifest_sha256": _sha256(crop_manifest_path.resolve()),
            "model_root": str(model_root),
            "model_sha256": _sha256(model_root / "model.safetensors"),
            "model_repo": "depth-anything/Depth-Anything-V2-Small-hf",
            "depth_type": "relative_affine_ambiguous",
            "depth_preprocess": "edge_letterbox_to_square_then_crop_back",
            "xyz_definition": "normalized_image_xy_plus_robust_normalized_model_depth_then_unit_sphere",
            "point_count": point_count,
            "sample_count": len(serializable_samples),
            "class_counts": class_counts,
            "short_side_lt64_count": sum(
                bool(item["short_side_lt64"]) for item in serializable_samples
            ),
            "samples": serializable_samples,
        }
        _write_json(staging / "manifest.json", dataset_manifest)
        protocol_summary = build_fivefold_protocols(
            split_root.resolve(), serializable_samples, staging
        )
        render_contact_sheet(
            [preview_by_class[key] for key in sorted(preview_by_class)],
            staging / "pointcloud_quality_preview_10class.png",
        )
        readme = f"""# Russian PAD 单目相对深度伪点云 v1.00

- 原图：`{source_root}`，未修改。
- 样本：{len(serializable_samples)}张、{len(class_counts)}类。
- 深度模型：Depth Anything V2 Small，约24.8M参数，本地离线加载；边缘填充为正方形后估深，再裁回原始长宽比，不拉伸车辆。
- 每图：固定{point_count}点；不足时以`valid_mask=false`补零。
- `points_npz/`：训练输入，包含`xyz [P,3]`、`rgb [P,3]`、`pixel_xy [P,2]`、`relative_depth [P]`、`valid_mask [P]`和原尺寸稠密相对深度。
- `points_ply/`：同一点集的二进制RGB PLY，可用CloudCompare/Open3D查看。
- `protocols/fivefold/`：与原PAD-Lite五折和固定Gallery逐样本对齐。

## 重要限制

这不是LiDAR或多视角重建真值，而是单张RGB经相对深度模型得到的2.5D可见表面：没有真实尺度、背面结构、相机内参和跨视角配准。它适合检验“额外深度形状先验是否帮助检索”，不能证明模型获得了真实三维车辆结构。

建议后续至少比较：`XYZ only`、`XYZ+RGB`、`P2B+XYZ融合`、`flat-Z负对照`和`shuffle-depth负对照`。只有真实depth优于两个负对照，才能认为收益来自深度几何，而不是新增参数或RGB捷径。
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        result = {
            "format": FORMAT_VERSION,
            "sample_count": len(serializable_samples),
            "point_count": point_count,
            "class_counts": class_counts,
            "short_side_lt64_count": dataset_manifest["short_side_lt64_count"],
            "model_sha256": dataset_manifest["model_sha256"],
            "fivefold_count": protocol_summary["fold_count"],
            "output_root": str(output_root),
        }
        _write_json(staging / "build_summary.json", result)
        staging.replace(output_root)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：构建RGB到单目伪点云数据集的命令行。
    输入参数：无。
    返回值：ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="Convert Russian PAD crops to relative-depth 2.5D point clouds"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    方法作用：解析参数并执行全量单目伪点云转换。
    输入参数：argv；None时读取系统命令行。
    返回值：int，成功为0。
    """
    args = build_parser().parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    result = build_dataset(
        args.source,
        args.manifest,
        args.splits,
        args.model,
        args.output,
        device,
        int(args.batch_size),
        int(args.point_count),
        not args.no_fp16,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
