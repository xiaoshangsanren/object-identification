from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ANNOTATION_ROOT, serializable_config
from .dino_patch_fusion import (
    _correctness_counts,
    _load_features,
    _load_predictions,
    _safe_correlation,
    _sha256,
    _true_class_margins,
    _validate_sources,
    _write_json,
    fuse_normalized_features,
)
from .metrics import compute_retrieval_metrics, save_retrieval_outputs, summarize_folds


P2B_VARIANT = "p2b_cls_weighted_patch_equal_fusion"
P2B_REPRESENTATION = "equal_weight_cls_plus_dynamic_masked_weighted_patch"
P2A_SOURCE_VARIANT = "p2a_masked_weighted_patch_only"


def _resolve_annotation_path(value: str | Path) -> Path:
    """
    方法作用：
        将相对路径按 Annotation 根目录解析为绝对路径。
    输入参数：
        value (str|Path)：原始路径。
    返回值：
        Path：解析后的绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _source_fold_root(config: dict[str, Any], key: str, fold_number: int) -> Path:
    """
    方法作用：
        取得 P2b 某缓存分支的指定折目录。
    输入参数：
        config：路径配置；key：缓存路径字段；fold_number：折号。
    返回值：
        Path：缓存根目录/fold_XX。
    """
    raw = config["paths"].get(key)
    if raw is None:
        raise KeyError(f"P2b config requires paths.{key}")
    return _resolve_annotation_path(raw) / f"fold_{fold_number:02d}"


def validate_p2b_settings(settings: Any) -> float:
    """
    方法作用：
        校验 P2b 严格采用 50/50、全类别、无门控、无 Query 调参的注册协议。
    输入参数：
        settings (Any)：p2b 配置字典。
    返回值：
        float：校验通过后的 CLS 权重 0.5。
    """

    if not isinstance(settings, dict):
        raise ValueError("P2b config requires a p2b section")
    cls_weight = float(settings.get("cls_weight", 0.5))
    patch_weight = float(settings.get("patch_weight", 0.5))
    if not math.isclose(cls_weight, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("P2b fixed ablation requires cls_weight=0.5")
    if not math.isclose(patch_weight, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("P2b fixed ablation requires patch_weight=0.5")
    if not math.isclose(cls_weight + patch_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("P2b fusion weights must sum to 1")
    if bool(settings.get("calibration", False)):
        raise ValueError("P2b must not use calibration; calibration belongs to P4")
    if bool(settings.get("prototype_gate", False)) or bool(
        settings.get("query_margin_gate", False)
    ):
        raise ValueError("P2b is ungated; candidate gates belong to P3/P4")
    if str(settings.get("candidate_scope", "all_classes")) != "all_classes":
        raise ValueError("P2b must fuse all candidate classes")
    if bool(settings.get("novel_query_tuning", False)):
        raise ValueError("P2b forbids Novel Query tuning")
    if not bool(settings.get("reuse_cached_features", True)):
        raise ValueError("P2b must reuse the registered P0 and P2a feature caches")
    return cls_weight


def validate_p2a_source(fold_root: Path) -> dict[str, Any]:
    """
    方法作用：
        校验 Patch 缓存确实来自不使用文本/CLS 的 P2a，避免错误消融来源。
    输入参数：
        fold_root (Path)：P2a 当前折目录。
    返回值：
        dict[str,Any]：通过校验的 P2a run_config。
    """

    path = fold_root / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing P2a provenance file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("variant") != P2A_SOURCE_VARIANT:
        raise ValueError(
            f"Expected P2a source variant {P2A_SOURCE_VARIANT!r}, "
            f"got {payload.get('variant')!r}"
        )
    if bool(payload.get("text_anchor_used", False)):
        raise ValueError("P2a Patch source must not use a text anchor")
    if bool(payload.get("cls_feature_used", False)):
        raise ValueError("P2a Patch source must be Patch-only")
    return payload


def evaluate_p2b_fold(
    config: dict[str, Any],
    fold_number: int,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """
    方法作用：
        从 P0 CLS 与 P2a 加权 Patch 缓存执行固定 50/50 P2b 全类别融合评测。
    输入参数：
        config：P2b 与缓存配置；fold_number：折号；device_name：记录用请求设备。
        主特征为 Support [Ns,Dc/Dp]、Query [Nq,Dc/Dp]，类别得分 [Nq,C]。
    返回值：
        dict[str,Any]：融合检索指标、来源信息和互补诊断。
    """

    cls_weight = validate_p2b_settings(config.get("p2b"))
    cls_root = _source_fold_root(config, "p0_features_root", fold_number)
    patch_root = _source_fold_root(config, "p2a_features_root", fold_number)
    p2a_provenance = validate_p2a_source(patch_root)

    cls_feature_path = cls_root / "features.pt"
    patch_feature_path = patch_root / "features.pt"
    cls_prediction_path = cls_root / "predictions.json"
    patch_prediction_path = patch_root / "predictions.json"
    cls_bundle = _load_features(cls_feature_path)
    patch_bundle = _load_features(patch_feature_path)
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
        "p2a_weighted_patch": patch_metrics["rank1_all"],
        "p2b_equal_fusion": metrics["rank1_all"],
    }

    metrics.update(
        {
            "variant": P2B_VARIANT,
            "representation": P2B_REPRESENTATION,
            "fold": fold_number,
            "novel_classes": class_names,
            "cls_weight": cls_weight,
            "patch_weight": 1.0 - cls_weight,
            "candidate_scope": "all_classes",
            "calibration_used": False,
            "prototype_gate_used": False,
            "query_margin_gate_used": False,
            "novel_query_tuning": False,
            "text_anchor_inherited_from_p0": True,
            "text_encoder_used_during_p2b": False,
            "cached_feature_reuse": True,
            "diagnostics": diagnostics,
        }
    )

    output_dir = config["paths"]["output_root"] / f"fold_{fold_number:02d}"
    p2a_run_config_path = patch_root / "run_config.json"
    run_config = {
        "format": "annotation_pad_lite_dino_patch_weighted_fusion_fold_v1",
        "variant": P2B_VARIANT,
        "representation": P2B_REPRESENTATION,
        "fold": fold_number,
        "device_requested": device_name,
        "execution_device": "cpu_cached_features",
        "config": serializable_config(config),
        "source_artifacts": {
            "p0_cls_features": str(cls_feature_path),
            "p0_cls_features_sha256": _sha256(cls_feature_path),
            "p2a_patch_features": str(patch_feature_path),
            "p2a_patch_features_sha256": _sha256(patch_feature_path),
            "p2a_run_config": str(p2a_run_config_path),
            "p2a_run_config_sha256": _sha256(p2a_run_config_path),
            "p2a_source_variant": p2a_provenance["variant"],
            "p0_predictions": str(cls_prediction_path),
            "p2a_predictions": str(patch_prediction_path),
        },
        "trainable_parameters": 0,
        "training_performed": False,
        "novel_query_tuning": False,
        "calibration_used": False,
        "gating_used": False,
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


def run_p2b(
    config: dict[str, Any],
    folds: list[int],
    device_name: str = "cpu",
    resume: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """
    方法作用：
        在指定折运行无训练的 P2b 缓存融合并生成跨折汇总。
    输入参数：
        config：P2b 配置；folds：折号列表；device_name：请求设备；resume/eval_only：
        P2b 不支持的检查点开关。
    返回值：
        dict[str,Any]：逐折 results 和聚合 summary。
    """
    if resume or eval_only:
        raise ValueError("P2b has no training checkpoint; run it as a new evaluation")
    results = []
    for fold in folds:
        metrics = evaluate_p2b_fold(config, fold, device_name)
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    root = config["paths"]["output_root"]
    summary = summarize_folds(root)
    summary.update(
        {
            "variant": P2B_VARIANT,
            "representation": P2B_REPRESENTATION,
            "cls_weight": 0.5,
            "patch_weight": 0.5,
            "candidate_scope": "all_classes",
            "calibration_used": False,
            "gating_used": False,
            "novel_query_tuning": False,
            "training_performed": False,
        }
    )
    _write_json(root / "summary.json", summary)
    return {"results": results, "summary": summary}
