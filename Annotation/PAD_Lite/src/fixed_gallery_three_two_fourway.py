from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ANNOTATION_ROOT
from .data import CropIndex, load_fold, records_for_partition
from .fixed_gallery_fourway import run_convnext_p2b
from .frozen_backbone_comparison import (
    load_comparison_config,
    run_frozen_backbone,
)
from .p2b_three_two_protocol import (
    _stage_config,
    build_episode_splits,
    evaluate_p2b_fold,
    load_protocol_config,
    run_stage,
    summarize_episodes,
)


FORMAT_VERSION = "annotation_pad_lite_fixed_gallery_3train_2test_fourway_v1.00"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/fixed_gallery_three_train_two_test_fourway_v1_00.json"
)
METHOD_ORDER = ("dino_raw", "dino_p2b", "convnext_raw", "convnext_p2b")
METHOD_LABELS = {
    "dino_raw": "原始 DINOv2-Small",
    "dino_p2b": "DINOv2-Small + P2B",
    "convnext_raw": "原始 ConvNeXt-Tiny",
    "convnext_p2b": "冻结 ConvNeXt-Tiny + P2B轻量头",
}


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：按 Annotation 根目录解析配置中的相对或绝对路径。
    输入参数：value，路径文本或 Path。
    返回值：Path，解析后的绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：创建父目录并写入 UTF-8 格式化 JSON。
    输入参数：path；payload，可 JSON 序列化对象。
    返回值：None。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """
    方法作用：流式计算普通文件的 SHA-256。
    输入参数：path，文件路径。
    返回值：str，十六进制摘要。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：加载 3折训练/2折测试四路实验配置并解析路径。
    输入参数：path，配置 JSON。
    返回值：dict，路径字段已转成绝对 Path。
    """
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Three-two four-way config schema_version must be 1")
    required = {"paths", "protocol", "raw", "convnext_p2b"}
    if missing := required - set(config):
        raise ValueError(f"Config misses sections: {sorted(missing)}")
    for key, value in config["paths"].items():
        config["paths"][key] = _resolve_path(value)
    config["config_path"] = config_path
    return config


def _dino_protocol(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """
    方法作用：构造使用新输出目录和固定 Episode 划分的 DINO P2B 协议。
    输入参数：config；run_root，本次结果根目录。
    返回值：dict，可交给原 3折/2折训练引擎的协议。
    """
    protocol = load_protocol_config(config["paths"]["dino_protocol_config"])
    protocol["paths"]["generated_splits_root"] = config["paths"]["episode_splits_root"]
    protocol["paths"]["output_root"] = run_root / "dino_p2b_components"
    return protocol


def validate_protocol(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """
    方法作用：验证十个 Episode 的3/2折类别隔离、固定 Gallery 数量、哈希和样本泄漏。
    输入参数：config；run_root。
    返回值：dict，十个 Episode 的类别与样本计数诊断。
    """
    expected_hash = str(config["protocol"]["fixed_gallery_manifest_sha256"])
    actual_hash = _sha256(config["paths"]["fixed_gallery_manifest"])
    if actual_hash != expected_hash:
        raise ValueError(f"Fixed Gallery hash mismatch: {actual_hash} != {expected_hash}")
    build_episode_splits(_dino_protocol(config, run_root))
    index = CropIndex(config["paths"]["crop_root"])
    rows: list[dict[str, Any]] = []
    for episode in range(1, int(config["protocol"]["episode_count"]) + 1):
        fold = load_fold(config["paths"]["episode_splits_root"], episode)
        if len(fold["base_classes"]) != int(config["protocol"]["base_class_count"]):
            raise ValueError(f"Episode {episode} base class count differs")
        if len(fold["novel_classes"]) != int(config["protocol"]["novel_class_count"]):
            raise ValueError(f"Episode {episode} novel class count differs")
        fixed = fold.get("fixed_gallery", {})
        if fixed.get("manifest_sha256") != expected_hash:
            raise ValueError(f"Episode {episode} does not reference fixed Gallery")
        support = records_for_partition(index, fold, "novel_support", exclude_tiny=True)
        query = records_for_partition(index, fold, "novel_query")
        base_train = records_for_partition(index, fold, "base_train", exclude_tiny=True)
        base_val = records_for_partition(index, fold, "base_val")
        counts = {
            name: sum(record.class_name == name for record in support)
            for name in fold["novel_classes"]
        }
        if any(
            value != int(config["protocol"]["gallery_per_class"])
            for value in counts.values()
        ):
            raise ValueError(f"Episode {episode} fixed Gallery counts differ: {counts}")
        train_sources = {record.source_image for record in base_train + base_val}
        test_sources = {record.source_image for record in support + query}
        if train_sources & test_sources:
            raise ValueError(f"Episode {episode} train/test source image leakage")
        if {record.sample_id for record in support} & {record.sample_id for record in query}:
            raise ValueError(f"Episode {episode} Gallery/Query sample leakage")
        rows.append(
            {
                "episode": episode,
                "source_train_folds": fold["source_train_folds"],
                "source_test_folds": fold["source_test_folds"],
                "base_classes": fold["base_classes"],
                "novel_classes": fold["novel_classes"],
                "base_train_count": len(base_train),
                "base_val_count": len(base_val),
                "gallery_count": len(support),
                "gallery_counts_by_class": counts,
                "query_count": len(query),
                "train_test_source_overlap": 0,
                "gallery_query_sample_overlap": 0,
            }
        )
    result = {
        "format": FORMAT_VERSION,
        "fixed_gallery_manifest": str(config["paths"]["fixed_gallery_manifest"]),
        "fixed_gallery_manifest_sha256": actual_hash,
        "episode_count": len(rows),
        "episodes": rows,
    }
    _write_json(run_root / "protocol_validation.json", result)
    return result


def run_raw(
    config: dict[str, Any], run_root: Path, backbone: str, device_name: str
) -> dict[str, Any]:
    """
    方法作用：在十个四分类 Episode 上运行零训练原始 DINO 或 ConvNeXt。
    输入参数：config；run_root；backbone；device_name。
    返回值：dict，十个 Episode 汇总指标。
    """
    base_path = Path(__file__).resolve().parents[1] / "configs/frozen_dino_convnext_p2b_comparison.json"
    frozen = load_comparison_config(base_path)
    frozen["paths"]["crop_root"] = config["paths"]["crop_root"]
    frozen["paths"]["splits_root"] = config["paths"]["episode_splits_root"]
    frozen["paths"]["dino_model"] = config["paths"]["dino_model"]
    frozen["paths"]["convnext_model"] = config["paths"]["convnext_model"]
    frozen["data"].update(deepcopy(config["raw"]))
    frozen["data"]["fold_count"] = int(config["protocol"]["episode_count"])
    frozen["retrieval"]["prototype_top_k"] = int(config["protocol"]["prototype_top_k"])
    episodes = list(range(1, int(config["protocol"]["episode_count"]) + 1))
    return run_frozen_backbone(backbone, frozen, episodes, device_name, run_root)


def run_dino_stage(
    config: dict[str, Any],
    run_root: Path,
    stage: str,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    """
    方法作用：运行 DINO P2B 的 P0(B2)、P2A 或最终 P2B 阶段。
    输入参数：config；run_root；stage；device_name；resume。
    返回值：dict，当前阶段十 Episode 汇总。
    """
    protocol = _dino_protocol(config, run_root)
    episodes = list(range(1, int(config["protocol"]["episode_count"]) + 1))
    if stage in {"p0", "p2a"}:
        return run_stage(protocol, stage, episodes, device_name, resume, False, None)
    if stage != "p2b":
        raise ValueError(f"Unsupported DINO stage: {stage}")
    stage_config = _stage_config(protocol, "p2b")
    stage_config["paths"]["output_root"] = run_root / "dino_p2b"
    root = stage_config["paths"]["output_root"]
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite DINO P2B evaluation: {root}")
    for episode in episodes:
        metrics = evaluate_p2b_fold(stage_config, episode, device_name)
        print(
            f"dino_p2b episode={episode:02d} Rank-1={metrics['rank1_all']:.4f}",
            flush=True,
        )
    return summarize_episodes(root, len(episodes))


def run_convnext(
    config: dict[str, Any], run_root: Path, device_name: str
) -> dict[str, Any]:
    """
    方法作用：在每个 Episode 的6个基础类上训练 ConvNeXt P2B 轻量头并在4个未见类测试。
    输入参数：config；run_root；device_name。
    返回值：dict，十 Episode 汇总指标。
    """
    conv_config = deepcopy(config)
    conv_config["paths"]["splits_root"] = config["paths"]["episode_splits_root"]
    conv_config["protocol"]["fold_count"] = int(config["protocol"]["episode_count"])
    episodes = list(range(1, int(config["protocol"]["episode_count"]) + 1))
    return run_convnext_p2b(conv_config, run_root, device_name, episodes)


def _method_summary(root: Path, episode_count: int) -> dict[str, Any]:
    """
    方法作用：从十个 Episode 的逐图预测计算 Macro/Micro Rank-1及逐类结果。
    输入参数：root；episode_count。
    返回值：dict，逐Episode、宏/微平均、正确数和协议字段。
    """
    episode_rank1: list[float] = []
    correct = total = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    protocol: list[dict[str, Any]] = []
    for episode in range(1, episode_count + 1):
        episode_root = root / f"fold_{episode:02d}"
        metrics = json.loads((episode_root / "metrics.json").read_text(encoding="utf-8"))
        predictions = json.loads((episode_root / "predictions.json").read_text(encoding="utf-8"))
        episode_rank1.append(float(metrics["rank1_all"]))
        correct += sum(bool(item["correct"]) for item in predictions)
        total += len(predictions)
        for item in predictions:
            values = per_class[str(item["true_class"])]
            values[0] += int(bool(item["correct"]))
            values[1] += 1
        protocol.append(
            {
                "episode": episode,
                "novel_classes": metrics["novel_classes"],
                "support_count": int(metrics["support_count"]),
                "query_count": len(predictions),
            }
        )
    return {
        "episode_rank1": episode_rank1,
        "macro_rank1": float(np.mean(episode_rank1)),
        "std_rank1": float(np.std(episode_rank1)),
        "micro_rank1": correct / max(total, 1),
        "correct_query_count": correct,
        "pooled_query_count": total,
        "per_class_rank1": {
            name: values[0] / values[1] for name, values in sorted(per_class.items())
        },
        "protocol": protocol,
    }


def summarize(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """
    方法作用：汇总四策略十 Episode 的固定 Gallery 四分类结果并生成报告。
    输入参数：config；run_root。
    返回值：dict，四方法结果、差值和协议验证。
    """
    episode_count = int(config["protocol"]["episode_count"])
    roots = {
        "dino_raw": run_root / "dino_raw",
        "dino_p2b": run_root / "dino_p2b",
        "convnext_raw": run_root / "convnext_raw",
        "convnext_p2b": run_root / "convnext_p2b",
    }
    methods = {
        name: _method_summary(root, episode_count) for name, root in roots.items()
    }
    reference = methods["dino_raw"]["protocol"]
    for name, result in methods.items():
        if result["protocol"] != reference:
            raise ValueError(f"Shared Episode protocol differs for {name}")
    validation = validate_protocol(config, run_root)
    payload = {
        "format": FORMAT_VERSION,
        "experiment": config["name"],
        "protocol_validation": validation,
        "methods": methods,
        "comparisons": {
            "dino_p2b_minus_dino_raw_macro": methods["dino_p2b"]["macro_rank1"]
            - methods["dino_raw"]["macro_rank1"],
            "convnext_p2b_minus_convnext_raw_macro": methods["convnext_p2b"]["macro_rank1"]
            - methods["convnext_raw"]["macro_rank1"],
        },
    }
    _write_json(run_root / "summary.json", payload)
    lines = [
        "# 固定 Gallery：3折微调、2折测试四路结果 V1.00",
        "",
        "协议：枚举全部10种3折训练/2折测试组合；每个 Episode 用6个类别训练轻量头，在4个未见类别上检索分类。每个测试类固定10张 Gallery，Gallery 与 Query 无样本/源图重叠。",
        "",
        f"固定 Gallery SHA-256：`{validation['fixed_gallery_manifest_sha256']}`。",
        "",
        "## Episode 对应关系",
        "",
        "| Episode | 训练折 | 测试折 | 四个测试类别 | Query数 |",
        "|---|---|---|---|---:|",
    ]
    for episode in validation["episodes"]:
        train_folds = ",".join(str(value) for value in episode["source_train_folds"])
        test_folds = ",".join(str(value) for value in episode["source_test_folds"])
        classes = ", ".join(episode["novel_classes"])
        lines.append(
            f"| E{episode['episode']} | {train_folds} | {test_folds} | {classes} | {episode['query_count']} |"
        )
    lines.extend(
        [
        "",
        "## Rank-1 结果",
        "",
        "| 方法 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | Macro Rank-1 | Micro Rank-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in METHOD_ORDER:
        row = methods[name]
        episodes = " | ".join(f"{value:.2%}" for value in row["episode_rank1"])
        lines.append(
            f"| {METHOD_LABELS[name]} | {episodes} | {row['macro_rank1']:.2%} | {row['micro_rank1']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## P2B 增量",
            "",
            f"- DINO P2B - 原始 DINO：{payload['comparisons']['dino_p2b_minus_dino_raw_macro']:+.2%} Macro Rank-1。",
            f"- ConvNeXt P2B - 原始 ConvNeXt：{payload['comparisons']['convnext_p2b_minus_convnext_raw_macro']:+.2%} Macro Rank-1。",
            "",
            "说明：原始 DINO/ConvNeXt 是零训练基线；P2B 在每个 Episode 都从预训练主干重新初始化轻量头，使用当前 Episode 的6个基础类训练，测试4类从未进入该 Episode 的训练。不同 Episode 会重复测试同一个类别，因此 Micro 是按全部 Episode-Query 预测池化，并非1081张唯一图片的一次性统计。",
            "",
        ]
    )
    (run_root / "THREE_TWO_FOURWAY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：构建3折训练/2折测试四路固定 Gallery 实验命令行。
    输入参数：None。
    返回值：ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="Fixed-Gallery 3-train/2-test four-way experiment")
    parser.add_argument(
        "stage",
        choices=(
            "validate",
            "raw",
            "dino-p0",
            "dino-p2a",
            "dino-p2b",
            "convnext-p2b",
            "summarize",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("dino_raw", "convnext_raw"), default="dino_raw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    方法作用：验证协议、运行四路分阶段实验或生成汇总报告。
    输入参数：argv；None 时读取系统命令行。
    返回值：int，成功为0。
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        result = validate_protocol(config, run_root)
    elif args.stage == "raw":
        result = run_raw(config, run_root, args.backbone, args.device)
    elif args.stage == "dino-p0":
        result = run_dino_stage(config, run_root, "p0", args.device, args.resume)
    elif args.stage == "dino-p2a":
        result = run_dino_stage(config, run_root, "p2a", args.device, args.resume)
    elif args.stage == "dino-p2b":
        result = run_dino_stage(config, run_root, "p2b", args.device, args.resume)
    elif args.stage == "convnext-p2b":
        result = run_convnext(config, run_root, args.device)
    else:
        result = summarize(config, run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
