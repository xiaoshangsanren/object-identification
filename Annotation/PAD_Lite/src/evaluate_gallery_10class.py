from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT
from .metrics import compute_retrieval_metrics


APP_ROOT = ANNOTATION_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from target_video_search.algorithms.gallery_temporal import GalleryTemporalMatcher
from target_video_search.config import (
    PAD_LITE_B0_ALGORITHM,
    PAD_LITE_B1_ALGORITHM,
    PAD_LITE_B2_ALGORITHM,
    DetectionConfig,
)
from target_video_search.pad_lite_embedder import PADLiteEmbedder


VARIANT_TO_ALGORITHM = {
    "b0": PAD_LITE_B0_ALGORITHM,
    "b1": PAD_LITE_B1_ALGORITHM,
    "b2": PAD_LITE_B2_ALGORITHM,
}


def _open_images(paths: list[Path]) -> list[Image.Image]:
    """
    方法作用：
        执行 _open_images 对应的处理流程。
    
    输入参数：
        paths (list[Path])：长度为 N 的图片路径列表。
    
    返回值：
        list[Image.Image]：长度为 N 的 RGB PIL 图片列表，各图片保留自身 [W, H] 尺寸。
    """
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    return images


def _classification_summary(
    confusion: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    方法作用：
        执行 _classification_summary 对应的处理流程。
    
    输入参数：
        confusion (np.ndarray)：真实类别×预测类别的混淆矩阵，形状为 [C, C]。
        class_names (list[str])：长度为 C、与矩阵行列编号对应的类别名称。
    
    返回值：
        dict[str, Any]：由 [C, C] 混淆矩阵计算出的宏平均指标和 C 个类别的逐类指标。
    """
    per_class: dict[str, Any] = {}
    f1_values = []
    recall_values = []
    precision_values = []
    for index, name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        actual = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        recall = true_positive / actual if actual else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recall_values.append(recall)
        precision_values.append(precision)
        f1_values.append(f1)
        per_class[name] = {
            "test_count": actual,
            "correct": true_positive,
            "accuracy_recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return {
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def evaluate_variant(
    variant: str,
    manifest_path: Path,
    checkpoint_root: Path,
    clip_model: Path,
    output_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """
    方法作用：
        执行 evaluate_variant 对应的处理流程。
    
    输入参数：
        variant (str)：实验版本名称。
        manifest_path (Path)：方法所需的 manifest_path 参数。
        checkpoint_root (Path)：方法所需的 checkpoint_root 参数。
        clip_model (Path)：CLIP 模型实例。
        output_root (Path)：实验输出根目录。
        device (str)：执行计算的 PyTorch 设备。
        batch_size (int)：方法所需的 batch_size 参数。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "annotation_gallery_10class_holdout_v1":
        raise ValueError(f"Unsupported split manifest: {manifest_path}")
    algorithm = VARIANT_TO_ALGORITHM[variant]
    gallery_root = Path(manifest["evaluation_gallery"])
    split_root = Path(manifest["split_root"])
    config = DetectionConfig(
        algorithm=algorithm,
        gallery_root=str(gallery_root),
        clip_local_dir=str(clip_model),
        clip_pretrained=str(clip_model),
        clip_backend="transformers",
        pad_lite_checkpoint_root=str(checkpoint_root),
        clip_batch_size=batch_size,
        device=device,
        half=False,
    )
    embedder = PADLiteEmbedder(config)
    matcher = GalleryTemporalMatcher(config, embedder)
    matcher.prepare()

    class_names = [item.class_id for item in matcher.classes]
    class_to_id = {name: index for index, name in enumerate(class_names)}
    query_paths: list[Path] = []
    query_labels: list[int] = []
    query_items: list[dict[str, str]] = []
    for item in manifest["test_order"]:
        path = split_root / "test" / item["folder"] / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing Gallery test image: {path}")
        query_paths.append(path)
        query_labels.append(class_to_id[item["class_id"]])
        query_items.append(item)

    query_tensor = embedder.encode_images(_open_images(query_paths), batch_size=batch_size)
    query_features = query_tensor.numpy().astype(np.float32, copy=False)
    support_features = np.concatenate(matcher.class_features, axis=0)
    support_labels = np.concatenate(
        [
            np.full(len(features), index, dtype=np.int64)
            for index, features in enumerate(matcher.class_features)
        ]
    )
    labels = np.asarray(query_labels, dtype=np.int64)
    metrics, class_scores = compute_retrieval_metrics(
        support_features=support_features,
        support_labels=support_labels,
        query_features=query_features,
        query_labels=labels,
        query_tiny=np.zeros(len(labels), dtype=bool),
        class_names=class_names,
        prototype_top_k=3,
    )
    predictions = class_scores.argmax(axis=1)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    prediction_rows: list[dict[str, Any]] = []
    for index, (truth, prediction) in enumerate(zip(labels, predictions, strict=True)):
        confusion[int(truth), int(prediction)] += 1
        prediction_rows.append(
            {
                **query_items[index],
                "true_class": class_names[int(truth)],
                "predicted_class": class_names[int(prediction)],
                "correct": bool(truth == prediction),
                "class_scores": {
                    name: round(float(class_scores[index, class_id]), 6)
                    for class_id, name in enumerate(class_names)
                },
            }
        )

    classification = _classification_summary(confusion, class_names)
    result = {
        "format": "annotation_gallery_10class_result_v1",
        "variant": variant,
        "algorithm": algorithm,
        "closed_set_class_count": len(class_names),
        "split_sha256": manifest["split_sha256"],
        "gallery_used_for_training_or_model_selection": False,
        "support_count": int(len(support_labels)),
        "test_count": int(len(labels)),
        "metrics": metrics,
        **classification,
        "class_names": class_names,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "model": embedder.summary_info(),
        "gallery": matcher.summary_info(),
    }
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = []
    for name in ("b0", "b1", "b2"):
        path = output_root / name / "metrics.json"
        if path.is_file():
            item = json.loads(path.read_text(encoding="utf-8"))
            completed.append(
                {
                    "variant": name,
                    "rank1": item["metrics"]["rank1_all"],
                    "class_map": item["metrics"]["class_map"],
                    "retrieval_map": item["metrics"]["retrieval_map"],
                    "macro_precision": item["macro_precision"],
                    "macro_recall": item["macro_recall"],
                    "macro_f1": item["macro_f1"],
                    "test_count": item["test_count"],
                    "split_sha256": item["split_sha256"],
                }
            )
    summary = {
        "format": "annotation_gallery_10class_summary_v1",
        "completed_variants": len(completed),
        "results": completed,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """
    方法作用：
        解析命令行参数并执行当前脚本的主流程。
    
    输入参数：
        无。
    
    返回值：
        None：方法直接完成相应操作。
    """
    parser = argparse.ArgumentParser(description="Evaluate PAD-Lite on the 10-class Gallery split.")
    parser.add_argument("variant", choices=("b0", "b1", "b2"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PAD_LITE_ROOT
        / "evaluation_data"
        / "gallery_10class_eval_v1"
        / "split_manifest.json",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=(
            PAD_LITE_ROOT
            / "outputs"
            / "01_baseline_and_finetuning"
            / "final"
        ),
    )
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=ANNOTATION_ROOT / "Resource" / "models" / "clip-vit-base-patch32",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PAD_LITE_ROOT
            / "outputs"
            / "04_gallery_and_recognizer_evaluation"
            / "gallery_10class"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = evaluate_variant(
        args.variant,
        args.manifest.resolve(),
        args.checkpoint_root.resolve(),
        args.clip_model.resolve(),
        args.output_root.resolve(),
        args.device,
        args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
