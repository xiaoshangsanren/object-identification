from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from .cli import DEFAULT_CONFIG, parse_folds
from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config
from .engine import run_training_fold


DEFAULT_OUTPUT_ROOT = PAD_LITE_ROOT / "outputs" / "clip_backbone_sweep"


def parse_blocks(value: str) -> list[int]:
    """
    方法作用：
        解析需要扫描的主干网络解冻层数。
    
    输入参数：
        value (str)：待解析或转换的输入值。
    
    返回值：
        list[int]：方法执行得到的结果。
    """
    normalized = value.strip().lower()
    if normalized == "all":
        return list(range(13))
    blocks: list[int] = []
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError("block range start must not exceed end")
            candidates = range(start, end + 1)
        else:
            candidates = (int(item),)
        for block in candidates:
            if block not in range(13):
                raise argparse.ArgumentTypeError("CLIP block count must be in 0..12")
            if block not in blocks:
                blocks.append(block)
    if not blocks:
        raise argparse.ArgumentTypeError("at least one block count is required")
    return blocks


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        将数据以格式化 JSON 写入指定文件。
    
    输入参数：
        path (Path)：目标文件或目录路径。
        payload (Any)：待写入或处理的数据。
    
    返回值：
        None：方法直接完成相应操作。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _block_root(output_root: Path, blocks: int) -> Path:
    """
    方法作用：
        生成指定主干解冻层数对应的实验输出目录。
    
    输入参数：
        output_root (Path)：实验输出根目录。
        blocks (int)：方法所需的 blocks 参数。
    
    返回值：
        Path：方法执行得到的结果。
    """
    return output_root / f"unfreeze_{blocks:02d}"


def _fold_metrics_path(block_root: Path, fold: int) -> Path:
    """
    方法作用：
        生成指定实验折的指标文件路径。
    
    输入参数：
        block_root (Path)：方法所需的 block_root 参数。
        fold (int)：交叉验证折信息或折编号。
    
    返回值：
        Path：方法执行得到的结果。
    """
    return block_root / "b1" / f"fold_{fold:02d}" / "metrics.json"


def _rank1_summary(output_root: Path, blocks: int) -> dict[str, Any]:
    """
    方法作用：
        读取各折指标并汇总 Rank-1 结果。
    
    输入参数：
        output_root (Path)：实验输出根目录。
        blocks (int)：方法所需的 blocks 参数。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    fold_rows = []
    for fold in range(1, 6):
        metrics_path = _fold_metrics_path(_block_root(output_root, blocks), fold)
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold": fold,
                "novel_classes": list(metrics["novel_classes"]),
                "rank1": float(metrics["rank1_all"]),
            }
        )
    values = [row["rank1"] for row in fold_rows]
    payload = {
        "format": "annotation_clip_backbone_rank1_v1",
        "visual_backbone": "clip-vit-base-patch32",
        "training": "retrieval_head + ID loss + Batch-Hard Triplet loss",
        "unfreeze_last_blocks": blocks,
        "completed_folds": len(fold_rows),
        "folds": fold_rows,
        "rank1_mean": statistics.fmean(values) if values else None,
        "rank1_std": statistics.pstdev(values) if values else None,
    }
    _write_json(_block_root(output_root, blocks) / "rank1_summary.json", payload)
    return payload


def _baseline_rank1(config: dict[str, Any]) -> dict[str, Any]:
    """
    方法作用：
        读取基线实验的 Rank-1 汇总结果。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    baseline_root = config["paths"]["output_root"] / "b0"
    fold_rows = []
    for fold in range(1, 6):
        path = baseline_root / f"fold_{fold:02d}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen CLIP B0 result: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fold_rows.append(
            {
                "fold": fold,
                "novel_classes": list(metrics["novel_classes"]),
                "rank1": float(metrics["rank1_all"]),
            }
        )
    values = [row["rank1"] for row in fold_rows]
    return {
        "label": "B0: frozen CLIP, no retrieval-head training",
        "folds": fold_rows,
        "rank1_mean": statistics.fmean(values),
        "rank1_std": statistics.pstdev(values),
    }


def aggregate_rank1(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """
    方法作用：
        汇总不同主干解冻设置下的 Rank-1 指标。
    
    输入参数：
        config (dict[str, Any])：实验配置字典。
        output_root (Path)：实验输出根目录。
    
    返回值：
        dict[str, Any]：方法执行得到的结果。
    """
    baseline = _baseline_rank1(config)
    experiments = []
    for block in range(13):
        path = _block_root(output_root, block) / "rank1_summary.json"
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["completed_folds"] != 5:
            continue
        row["delta_vs_b0"] = row["rank1_mean"] - baseline["rank1_mean"]
        experiments.append(row)
    best = max(experiments, key=lambda item: item["rank1_mean"]) if experiments else None
    payload = {
        "format": "annotation_clip_backbone_rank1_sweep_v1",
        "metric": "rank1_all",
        "baseline": baseline,
        "experiments": experiments,
        "best_unfreeze_last_blocks": (
            best["unfreeze_last_blocks"] if best is not None else None
        ),
        "best_rank1_mean": best["rank1_mean"] if best is not None else None,
    }
    _write_json(output_root / "rank1_results.json", payload)
    _write_markdown(output_root / "RANK1_RESULTS.md", payload)
    return payload


def _percent(value: float | None) -> str:
    """
    方法作用：
        将可选浮点指标格式化为百分数字符串。
    
    输入参数：
        value (float | None)：待解析或转换的输入值。
    
    返回值：
        str：方法执行得到的结果。
    """
    return "—" if value is None or math.isnan(value) else f"{100.0 * value:.2f}%"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """
    方法作用：
        将主干扫描结果写成 Markdown 报告。
    
    输入参数：
        path (Path)：目标文件或目录路径。
        payload (dict[str, Any])：待写入或处理的数据。
    
    返回值：
        None：方法直接完成相应操作。
    """
    baseline = payload["baseline"]
    baseline_by_fold = {row["fold"]: row for row in baseline["folds"]}
    lines = [
        "# CLIP Backbone 解冻深度消融：Rank-1",
        "",
        "本报告只采用全部 Query 的 `rank1_all`，不使用其他评测指标。",
        "",
        "| 解冻最后 n 个 Block | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 相对 B0 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| B0（冻结、无训练） | "
        + " | ".join(_percent(baseline_by_fold[i]["rank1"]) for i in range(1, 6))
        + f" | **{_percent(baseline['rank1_mean'])}** | — |",
    ]
    for experiment in payload["experiments"]:
        by_fold = {row["fold"]: row for row in experiment["folds"]}
        delta = experiment["delta_vs_b0"] * 100.0
        lines.append(
            f"| {experiment['unfreeze_last_blocks']} | "
            + " | ".join(_percent(by_fold[i]["rank1"]) for i in range(1, 6))
            + f" | **{_percent(experiment['rank1_mean'])}** | {delta:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"当前最高结果：解冻最后 `{payload['best_unfreeze_last_blocks']}` 个 Block，"
            f"五折平均 Rank-1 为 `{_percent(payload['best_rank1_mean'])}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _cleanup_checkpoints(block_root: Path, fold: int) -> None:
    """
    方法作用：
        清理扫描实验产生的中间检查点文件。
    
    输入参数：
        block_root (Path)：方法所需的 block_root 参数。
        fold (int)：交叉验证折信息或折编号。
    
    返回值：
        None：方法直接完成相应操作。
    """
    fold_root = block_root / "b1" / f"fold_{fold:02d}"
    for name in ("best.pt", "last.pt"):
        path = fold_root / name
        if path.is_file():
            path.unlink()


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建并配置当前命令行程序的参数解析器。
    
    输入参数：
        无。
    
    返回值：
        argparse.ArgumentParser：方法执行得到的结果。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sweep CLIP visual-backbone unfreezing depth with fixed B1 training "
            "and report only five-fold rank1_all."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blocks", type=parse_blocks, default=list(range(13)))
    parser.add_argument("--fold", type=parse_folds, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析命令行参数并执行当前脚本的主流程。
    
    输入参数：
        argv (list[str] | None)：待解析的命令行参数；为 None 时读取系统命令行。
    
    返回值：
        int：方法执行得到的结果。
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_root = args.output_root.expanduser()
    output_root = (
        output_root.resolve()
        if output_root.is_absolute()
        else (ANNOTATION_ROOT / output_root).resolve()
    )
    if args.aggregate_only:
        print(json.dumps(aggregate_rank1(config, output_root), ensure_ascii=False, indent=2))
        return 0
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.workers is not None and args.workers < 0:
        raise ValueError("--workers cannot be negative")

    for blocks in args.blocks:
        block_config = deepcopy(config)
        block_root = _block_root(output_root, blocks)
        block_config["paths"]["output_root"] = block_root
        block_config["model"]["unfreeze_last_blocks"] = blocks
        if args.epochs is not None:
            block_config["b1"]["epochs"] = args.epochs
            block_config["b1"]["warmup_epochs"] = min(
                int(block_config["b1"]["warmup_epochs"]), args.epochs - 1
            )
        if args.workers is not None:
            block_config["data"]["workers"] = args.workers

        for fold in args.fold:
            metrics_path = _fold_metrics_path(block_root, fold)
            if args.resume and metrics_path.is_file():
                print(
                    json.dumps(
                        {"unfreeze_last_blocks": blocks, "fold": fold, "status": "reused"}
                    ),
                    flush=True,
                )
            else:
                run_training_fold(
                    block_config,
                    variant="b1",
                    fold_number=fold,
                    device_name=args.device,
                    resume=args.resume,
                    eval_only=False,
                )
            _rank1_summary(output_root, blocks)
            if not args.keep_checkpoints:
                _cleanup_checkpoints(block_root, fold)

    result = aggregate_rank1(config, output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
