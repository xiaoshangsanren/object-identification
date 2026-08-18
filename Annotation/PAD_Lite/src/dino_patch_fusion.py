from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import ANNOTATION_ROOT, serializable_config
from .metrics import (
    compute_retrieval_metrics,
    save_retrieval_outputs,
    summarize_folds,
)


P1B_VARIANT = "p1b_cls_patch_average_equal_fusion"
P1B_REPRESENTATION = "equal_weight_cls_plus_masked_patch_average"
METADATA_KEYS = (
    "sample_id",
    "source_image",
    "class_name",
    "tiny",
    "crowded",
    "clipped",
)


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        创建父目录并写入 UTF-8 JSON。
    输入参数：
        path：目标路径；payload：可序列化对象。
    返回值：
        None：无返回数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_annotation_path(value: str | Path) -> Path:
    """
    方法作用：
        将路径解析为相对 Annotation 根目录的绝对路径。
    输入参数：
        value (str|Path)：原始路径。
    返回值：
        Path：解析后的绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    """
    方法作用：
        流式计算源特征或配置文件的 SHA-256，用于实验溯源。
    输入参数：
        path (Path)：待校验文件。
    返回值：
        str：64 位十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_numpy(value: Any, dtype) -> np.ndarray:
    """
    方法作用：
        将 Torch 张量或类数组统一转成指定 dtype 的 NumPy 数组。
    输入参数：
        value (Any)：张量/数组，主特征通常为 [N,D]；dtype：目标 NumPy 类型。
    返回值：
        np.ndarray：保持原形状的数组。
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    """
    方法作用：
        对二维特征矩阵逐样本做 L2 归一化。
    输入参数：
        features (np.ndarray)：特征 [N,D]。
    返回值：
        np.ndarray：单位长度特征 [N,D]。
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {features.shape}")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot normalize a zero feature vector")
    return features / norms


def fuse_normalized_features(
    cls_features: np.ndarray,
    patch_features: np.ndarray,
    cls_weight: float,
) -> np.ndarray:
    """
    方法作用：
        按固定权重缩放并拼接 CLS 全局特征与 Patch 局部特征，构造联合余弦空间。
    输入参数：
        cls_features (np.ndarray)：全局特征 [N,Dc]；patch_features：局部特征 [N,Dp]；
        cls_weight (float)：CLS 权重 λ，Patch 权重为 1-λ。
    返回值：
        np.ndarray：L2 归一化融合特征 [N,Dc+Dp]。
    """

    cls_features = _l2_normalize(cls_features)
    patch_features = _l2_normalize(patch_features)
    if cls_features.shape[0] != patch_features.shape[0]:
        raise ValueError(
            "CLS and Patch sample counts differ: "
            f"{cls_features.shape[0]} vs {patch_features.shape[0]}"
        )
    if not 0.0 <= cls_weight <= 1.0:
        raise ValueError("cls_weight must be in [0, 1]")
    patch_weight = 1.0 - cls_weight
    fused = np.concatenate(
        (
            math.sqrt(cls_weight) * cls_features,
            math.sqrt(patch_weight) * patch_features,
        ),
        axis=1,
    )
    return _l2_normalize(fused).astype(np.float32, copy=False)


def _load_features(path: Path) -> dict[str, Any]:
    """
    方法作用：
        加载并校验某一分支保存的 Support/Query 特征包。
    输入参数：
        path (Path)：features.pt 路径。
    返回值：
        dict[str,Any]：class_names、Support [Ns,D]、Query [Nq,D] 及标签 [Ns]/[Nq]。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached feature file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "class_names",
        "support_features",
        "support_labels",
        "query_features",
        "query_labels",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return {
        "class_names": list(payload["class_names"]),
        "support_features": _as_numpy(payload["support_features"], np.float32),
        "support_labels": _as_numpy(payload["support_labels"], np.int64),
        "query_features": _as_numpy(payload["query_features"], np.float32),
        "query_labels": _as_numpy(payload["query_labels"], np.int64),
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    """
    方法作用：
        读取一个分支的逐 Query 预测记录。
    输入参数：
        path (Path)：predictions.json 路径。
    返回值：
        list[dict[str,Any]]：长度 Nq 的预测与元数据列表。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached prediction file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction file must contain a list: {path}")
    return payload


def _validate_sources(
    cls_bundle: dict[str, Any],
    patch_bundle: dict[str, Any],
    cls_predictions: list[dict[str, Any]],
    patch_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    方法作用：
        校验 CLS/Patch 类别顺序、标签和 Query 元数据完全对齐，防止错误融合。
    输入参数：
        cls_bundle、patch_bundle：两分支特征包；cls_predictions、patch_predictions：
        两个长度 Nq 的预测列表。
    返回值：
        list[dict[str,Any]]：对齐后的长度 Nq 元数据列表。
    """
    if cls_bundle["class_names"] != patch_bundle["class_names"]:
        raise ValueError("CLS and Patch class order differs")
    for key in ("support_labels", "query_labels"):
        if not np.array_equal(cls_bundle[key], patch_bundle[key]):
            raise ValueError(f"CLS and Patch {key} differ")
    if len(cls_predictions) != len(patch_predictions):
        raise ValueError("CLS and Patch prediction counts differ")
    if len(cls_predictions) != len(cls_bundle["query_labels"]):
        raise ValueError("Prediction and query-feature counts differ")

    metadata: list[dict[str, Any]] = []
    for index, (cls_item, patch_item) in enumerate(
        zip(cls_predictions, patch_predictions, strict=True)
    ):
        for key in METADATA_KEYS:
            if cls_item.get(key) != patch_item.get(key):
                raise ValueError(
                    f"CLS/Patch query order differs at index {index}, field {key}"
                )
        metadata.append({key: cls_item[key] for key in METADATA_KEYS})
    return metadata


def _true_class_margins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    方法作用：
        计算真实类别得分相对最强竞争类别的间隔。
    输入参数：
        scores (np.ndarray)：类别得分 [Nq,C]；labels (np.ndarray)：标签 [Nq]。
    返回值：
        np.ndarray：分类间隔 [Nq]。
    """
    rows = np.arange(len(labels))
    true_scores = scores[rows, labels]
    competitors = scores.copy()
    competitors[rows, labels] = -np.inf
    return true_scores - competitors.max(axis=1)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """
    方法作用：
        在非退化条件下计算两个对齐向量的 Pearson 相关性。
    输入参数：
        left、right (np.ndarray)：一维数组 [N]。
    返回值：
        float|None：相关系数，样本不足或零方差时为 None。
    """
    if len(left) < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correctness_counts(
    cls_predictions: np.ndarray,
    patch_predictions: np.ndarray,
    fusion_predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """
    方法作用：
        统计 CLS、Patch、融合分支的正误组合，以及融合相对 CLS 的纠错和伤害。
    输入参数：
        cls_predictions、patch_predictions、fusion_predictions、labels：形状均为 [Nq]；
        class_names：长度 C 的类别名称。
    返回值：
        dict[str,Any]：总体及逐类别互补性计数。
    """
    cls_correct = cls_predictions == labels
    patch_correct = patch_predictions == labels
    fusion_correct = fusion_predictions == labels
    rescued_mask = ~cls_correct & fusion_correct
    harmed_mask = cls_correct & ~fusion_correct

    per_class = {}
    for class_id, class_name in enumerate(class_names):
        class_mask = labels == class_id
        rescued = int((rescued_mask & class_mask).sum())
        harmed = int((harmed_mask & class_mask).sum())
        per_class[class_name] = {
            "query_count": int(class_mask.sum()),
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
        }
    return {
        "cls_patch_complementarity": {
            "cls_correct_patch_correct": int((cls_correct & patch_correct).sum()),
            "cls_correct_patch_wrong": int((cls_correct & ~patch_correct).sum()),
            "cls_wrong_patch_correct": int((~cls_correct & patch_correct).sum()),
            "cls_wrong_patch_wrong": int((~cls_correct & ~patch_correct).sum()),
        },
        "fusion_vs_cls": {
            "rescued": int(rescued_mask.sum()),
            "harmed": int(harmed_mask.sum()),
            "net_rescue": int(rescued_mask.sum() - harmed_mask.sum()),
            "unchanged_correct": int((cls_correct & fusion_correct).sum()),
            "unchanged_wrong": int((~cls_correct & ~fusion_correct).sum()),
            "per_class": per_class,
        },
    }


def _source_fold_root(config: dict[str, Any], key: str, fold_number: int) -> Path:
    """
    方法作用：
        从配置中解析某个缓存来源的当前折目录。
    输入参数：
        config：路径配置；key：缓存根目录字段；fold_number：折号。
    返回值：
        Path：缓存根目录/fold_XX。
    """
    raw = config["paths"].get(key)
    if raw is None:
        raise KeyError(f"P1b config requires paths.{key}")
    return _resolve_annotation_path(raw) / f"fold_{fold_number:02d}"


def evaluate_p1b_fold(
    config: dict[str, Any],
    fold_number: int,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """
    方法作用：
        从 P0 CLS 与 P1a 平均 Patch 缓存执行固定 50/50 P1b 融合评测。
    输入参数：
        config：P1b 和缓存路径配置；fold_number：折号；device_name：记录的请求设备。
        主数据为 Support [Ns,Dc/Dp]、Query [Nq,Dc/Dp]、融合得分 [Nq,C]。
    返回值：
        dict[str,Any]：融合检索指标、权重与互补诊断。
    """

    settings = config.get("p1b")
    if not isinstance(settings, dict):
        raise ValueError("P1b config requires a p1b section")
    cls_weight = float(settings.get("cls_weight", 0.5))
    if not math.isclose(cls_weight, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "P1b is the pre-registered equal-weight ablation; cls_weight must be 0.5"
        )
    if bool(settings.get("calibration", False)):
        raise ValueError("P1b must not use calibration; calibration belongs to P4")

    cls_root = _source_fold_root(config, "p0_features_root", fold_number)
    patch_root = _source_fold_root(config, "p1a_features_root", fold_number)
    cls_feature_path = cls_root / "features.pt"
    patch_feature_path = patch_root / "features.pt"
    cls_bundle = _load_features(cls_feature_path)
    patch_bundle = _load_features(patch_feature_path)
    cls_prediction_path = cls_root / "predictions.json"
    patch_prediction_path = patch_root / "predictions.json"
    metadata = _validate_sources(
        cls_bundle,
        patch_bundle,
        _load_predictions(cls_prediction_path),
        _load_predictions(patch_prediction_path),
    )

    class_names = cls_bundle["class_names"]
    support_labels = cls_bundle["support_labels"]
    query_labels = cls_bundle["query_labels"]
    query_tiny = np.asarray([item["tiny"] for item in metadata], dtype=bool)
    top_k = int(config["retrieval"]["prototype_top_k"])

    cls_metrics, cls_scores = compute_retrieval_metrics(
        cls_bundle["support_features"],
        support_labels,
        cls_bundle["query_features"],
        query_labels,
        query_tiny,
        class_names,
        top_k,
    )
    patch_metrics, patch_scores = compute_retrieval_metrics(
        patch_bundle["support_features"],
        support_labels,
        patch_bundle["query_features"],
        query_labels,
        query_tiny,
        class_names,
        top_k,
    )
    fused_support = fuse_normalized_features(
        cls_bundle["support_features"],
        patch_bundle["support_features"],
        cls_weight,
    )
    fused_query = fuse_normalized_features(
        cls_bundle["query_features"],
        patch_bundle["query_features"],
        cls_weight,
    )
    metrics, fusion_scores = compute_retrieval_metrics(
        fused_support,
        support_labels,
        fused_query,
        query_labels,
        query_tiny,
        class_names,
        top_k,
    )

    cls_predictions = cls_scores.argmax(axis=1)
    patch_predictions = patch_scores.argmax(axis=1)
    fusion_predictions = fusion_scores.argmax(axis=1)
    diagnostics = _correctness_counts(
        cls_predictions,
        patch_predictions,
        fusion_predictions,
        query_labels,
        class_names,
    )
    diagnostics["true_class_margin_correlation"] = {
        "cls_vs_patch": _safe_correlation(
            _true_class_margins(cls_scores, query_labels),
            _true_class_margins(patch_scores, query_labels),
        ),
        "cls_vs_fusion": _safe_correlation(
            _true_class_margins(cls_scores, query_labels),
            _true_class_margins(fusion_scores, query_labels),
        ),
    }
    diagnostics["source_rank1"] = {
        "p0_cls": cls_metrics["rank1_all"],
        "p1a_patch_average": patch_metrics["rank1_all"],
        "p1b_equal_fusion": metrics["rank1_all"],
    }

    metrics.update(
        {
            "variant": P1B_VARIANT,
            "representation": P1B_REPRESENTATION,
            "fold": fold_number,
            "novel_classes": class_names,
            "cls_weight": cls_weight,
            "patch_weight": 1.0 - cls_weight,
            "calibration_used": False,
            "text_anchor_inherited_from_p0": True,
            "text_encoder_used_during_p1b": False,
            "cached_feature_reuse": True,
            "diagnostics": diagnostics,
        }
    )
    output_dir = config["paths"]["output_root"] / f"fold_{fold_number:02d}"
    run_config = {
        "format": "annotation_pad_lite_dino_patch_fusion_fold_v1",
        "variant": P1B_VARIANT,
        "representation": P1B_REPRESENTATION,
        "fold": fold_number,
        "device_requested": device_name,
        "execution_device": "cpu_cached_features",
        "config": serializable_config(config),
        "source_artifacts": {
            "p0_cls_features": str(cls_feature_path),
            "p0_cls_features_sha256": _sha256(cls_feature_path),
            "p1a_patch_features": str(patch_feature_path),
            "p1a_patch_features_sha256": _sha256(patch_feature_path),
            "p0_predictions": str(cls_prediction_path),
            "p1a_predictions": str(patch_prediction_path),
        },
        "trainable_parameters": 0,
        "training_performed": False,
        "novel_query_tuning": False,
    }
    _write_json(output_dir / "run_config.json", run_config)
    save_retrieval_outputs(
        output_dir,
        metrics,
        class_names,
        fusion_scores,
        query_labels,
        metadata,
        fused_support,
        support_labels,
        fused_query,
    )
    _write_json(output_dir / "fusion_diagnostics.json", diagnostics)
    return metrics


def run_p1b(
    config: dict[str, Any],
    folds: list[int],
    device_name: str = "cpu",
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        对指定折运行无训练的 P1b 缓存特征融合并汇总结果。
    输入参数：
        config：P1b 配置；folds：折号；device_name：请求设备；resume/eval_only：
        P1b 不支持的训练状态开关。
    返回值：
        dict[str,Any]：逐折 results 与跨折 summary。
    """
    if resume or eval_only:
        raise ValueError("P1b has no training checkpoint; run it as a new evaluation")
    results = []
    for fold in folds:
        metrics = evaluate_p1b_fold(config, fold, device_name)
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    root = config["paths"]["output_root"]
    summary = summarize_folds(root)
    summary.update(
        {
            "variant": P1B_VARIANT,
            "representation": P1B_REPRESENTATION,
            "cls_weight": 0.5,
            "patch_weight": 0.5,
            "calibration_used": False,
            "training_performed": False,
        }
    )
    _write_json(root / "summary.json", summary)
    return {"results": results, "summary": summary}
