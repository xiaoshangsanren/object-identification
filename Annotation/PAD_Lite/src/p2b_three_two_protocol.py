from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config, serializable_config
from .dino_engine import run_dino_training_fold
from .dino_patch_weighted_engine import run_p2a_fold
from .dino_patch_weighted_fusion import evaluate_p2b_fold


PROTOCOL_VERSION = "pad_lite_3train_2test_4way_v1"
DEFAULT_CONFIG = (
    PAD_LITE_ROOT
    / "configs"
    / "p2b_three_train_two_test_4way.json"
)
METRIC_NAMES = (
    "rank1_all",
    "rank1_core",
    "rank1_tiny",
    "recall_at_2",
    "recall_at_3",
    "recall_at_5",
    "class_map",
    "retrieval_map",
)


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        创建父目录并写入格式化 JSON。
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


def _annotation_path(value: str | Path) -> Path:
    """
    方法作用：
        将协议配置路径按 Annotation 根目录解析。
    输入参数：
        value (str|Path)：原始路径。
    返回值：
        Path：绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def load_protocol_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """
    方法作用：
        读取并验证 3-train/2-test 协议配置，解析全部路径。
    输入参数：
        path (str|Path)：协议 JSON 路径。
    返回值：
        dict[str,Any]：已解析、要求 10 Episode 的协议配置。
    """
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Protocol schema_version must be 1")
    required_paths = {
        "source_splits_root",
        "generated_splits_root",
        "output_root",
        "p0_base_config",
        "p2a_base_config",
        "p2b_base_config",
    }
    missing = required_paths - set(payload.get("paths", {}))
    if missing:
        raise ValueError(f"Protocol config is missing paths: {sorted(missing)}")
    resolved = deepcopy(payload)
    for key in required_paths:
        resolved["paths"][key] = _annotation_path(resolved["paths"][key])
    if int(resolved["protocol"].get("episode_count", 0)) != 10:
        raise ValueError("The all-combination 3-train/2-test protocol requires 10 episodes")
    resolved["config_path"] = config_path
    return resolved


def _source_folds(source_root: Path) -> list[dict[str, Any]]:
    """
    方法作用：
        加载原始五折划分文件。
    输入参数：
        source_root (Path)：包含 fold_01～fold_05.json 的目录。
    返回值：
        list[dict[str,Any]]：按折号排列的 5 个划分载荷。
    """
    folds = []
    for fold_number in range(1, 6):
        path = source_root / f"fold_{fold_number:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source fold: {path}")
        folds.append(json.loads(path.read_text(encoding="utf-8")))
    return folds


def _class_partitions(
    source_folds: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, int]]:
    """
    方法作用：
        汇总每类稳定的基础训练/验证分区及它被留出的原始折号。
    输入参数：
        source_folds：5 个原始折载荷。
    返回值：
        tuple：类名到 base_train/base_val 样本列表的映射，以及类名到 held-out 折号的映射。
    """
    base_by_class: dict[str, dict[str, list[str]]] = {}
    held_out_by_class: dict[str, int] = {}
    for fold in source_folds:
        fold_number = int(fold["fold"])
        for class_name in fold["novel_classes"]:
            if class_name in held_out_by_class:
                raise ValueError(f"Class {class_name} is held out by multiple source folds")
            held_out_by_class[class_name] = fold_number
        for class_name in fold["base_classes"]:
            candidate = {
                "base_train": list(fold["partitions"]["base_train"][class_name]),
                "base_val": list(fold["partitions"]["base_val"][class_name]),
            }
            previous = base_by_class.setdefault(class_name, candidate)
            if previous != candidate:
                raise ValueError(
                    f"Base train/val partition for {class_name} differs by source fold"
                )
    if set(base_by_class) != set(held_out_by_class):
        raise ValueError("Base and held-out class sets are inconsistent")
    return base_by_class, held_out_by_class


def build_three_two_payloads(
    source_folds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    方法作用：
        枚举 C(5,2)=10 个 Episode；每个 Episode 用 3 折对应 6 类训练、2 折对应 4 类测试。
    输入参数：
        source_folds：5 个原始二类 held-out 划分。
    返回值：
        list[dict[str,Any]]：10 个无类交叠、无源图泄漏的四分类 Episode 载荷。
    """

    if len(source_folds) != 5:
        raise ValueError("The 3-train/2-test protocol requires exactly five source folds")
    by_number = {int(fold["fold"]): fold for fold in source_folds}
    if set(by_number) != set(range(1, 6)):
        raise ValueError("Source fold numbers must be exactly 1..5")
    if any(len(fold["novel_classes"]) != 2 for fold in source_folds):
        raise ValueError("Every source fold must contain exactly two held-out classes")

    base_by_class, held_out_by_class = _class_partitions(source_folds)
    all_classes = set(held_out_by_class)
    episodes = []
    for episode_number, test_folds in enumerate(
        itertools.combinations(range(1, 6), 2), start=1
    ):
        train_folds = tuple(fold for fold in range(1, 6) if fold not in test_folds)
        base_classes = [
            class_name
            for fold_number in train_folds
            for class_name in by_number[fold_number]["novel_classes"]
        ]
        novel_classes = [
            class_name
            for fold_number in test_folds
            for class_name in by_number[fold_number]["novel_classes"]
        ]
        if len(base_classes) != 6 or len(novel_classes) != 4:
            raise ValueError("Every episode must contain six base and four novel classes")
        if set(base_classes) & set(novel_classes):
            raise ValueError("Base and novel classes overlap")
        if set(base_classes) | set(novel_classes) != all_classes:
            raise ValueError("Episode classes do not cover the original ten classes")

        partitions = {
            "base_train": {
                name: list(base_by_class[name]["base_train"])
                for name in base_classes
            },
            "base_val": {
                name: list(base_by_class[name]["base_val"])
                for name in base_classes
            },
            "novel_support": {},
            "novel_query": {},
        }
        for class_name in novel_classes:
            holder = by_number[held_out_by_class[class_name]]
            partitions["novel_support"][class_name] = list(
                holder["partitions"]["novel_support"][class_name]
            )
            partitions["novel_query"][class_name] = list(
                holder["partitions"]["novel_query"][class_name]
            )

        train_sources = {
            item
            for partition in ("base_train", "base_val")
            for names in partitions[partition].values()
            for item in names
        }
        test_sources = {
            item
            for partition in ("novel_support", "novel_query")
            for names in partitions[partition].values()
            for item in names
        }
        if train_sources & test_sources:
            raise ValueError("Train and test source images overlap")
        episodes.append(
            {
                "version": "pad_lite_class_holdout_fold_v1",
                "protocol_version": PROTOCOL_VERSION,
                "fold": episode_number,
                "seed": int(source_folds[0].get("seed", 2026)),
                "image_root": source_folds[0].get("image_root", "train"),
                "source_train_folds": list(train_folds),
                "source_test_folds": list(test_folds),
                "base_classes": base_classes,
                "novel_classes": novel_classes,
                "partitions": partitions,
            }
        )
    return episodes


def build_episode_splits(config: dict[str, Any]) -> dict[str, Any]:
    """
    方法作用：
        生成并落盘 10 个 Episode 划分；已有文件必须与重建结果完全一致。
    输入参数：
        config：含原始和生成划分目录的协议配置。
    返回值：
        dict[str,Any]：Episode 类别、来源折及样本数汇总。
    """
    source_root = config["paths"]["source_splits_root"]
    output_root = config["paths"]["generated_splits_root"]
    episodes = build_three_two_payloads(_source_folds(source_root))
    rows = []
    for payload in episodes:
        path = output_root / f"fold_{int(payload['fold']):02d}.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(
                    f"Generated split differs from existing file; refusing overwrite: {path}"
                )
        else:
            _write_json(path, payload)
        rows.append(
            {
                "episode": payload["fold"],
                "train_folds": payload["source_train_folds"],
                "test_folds": payload["source_test_folds"],
                "base_classes": payload["base_classes"],
                "novel_classes": payload["novel_classes"],
                "counts": {
                    partition: sum(len(items) for items in by_class.values())
                    for partition, by_class in payload["partitions"].items()
                },
            }
        )
    summary = {
        "format": PROTOCOL_VERSION,
        "source_splits_root": str(source_root),
        "episode_count": len(rows),
        "episodes": rows,
    }
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise FileExistsError(
                f"Generated summary differs from existing file: {summary_path}"
            )
    else:
        _write_json(summary_path, summary)
    return summary


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    方法作用：
        按点路径对阶段配置做深拷贝覆盖。
    输入参数：
        config：目标配置；dotted_key：嵌套字段；value：新值。
    返回值：
        None：原位更新配置。
    """
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise KeyError(f"Override parent does not exist: {dotted_key}")
        target = child
    target[keys[-1]] = deepcopy(value)


def _stage_config(protocol: dict[str, Any], stage: str) -> dict[str, Any]:
    """
    方法作用：
        为 p0、p2a 或 p2b 生成独立输入/输出/缓存路径配置。
    输入参数：
        protocol：协议配置；stage：p0、p2a 或 p2b。
    返回值：
        dict[str,Any]：可直接传给相应底层引擎的阶段配置。
    """
    base_path = protocol["paths"][f"{stage}_base_config"]
    config = load_config(base_path)
    for dotted_key, value in protocol.get(f"{stage}_overrides", {}).items():
        _set_dotted(config, dotted_key, value)
    config["paths"]["splits_root"] = protocol["paths"]["generated_splits_root"]
    output_root = protocol["paths"]["output_root"]
    if stage == "p0":
        config["paths"]["output_root"] = output_root / "p0"
    elif stage == "p2a":
        config["paths"]["output_root"] = output_root / "p2a"
    elif stage == "p2b":
        config["paths"]["output_root"] = output_root / "p2b"
        config["paths"]["p0_features_root"] = output_root / "p0" / "dino" / "b2"
        config["paths"]["p2a_features_root"] = output_root / "p2a"
    else:
        raise ValueError(f"Unknown protocol stage: {stage}")
    config["protocol"] = {
        **deepcopy(protocol["protocol"]),
        "name": protocol["name"],
        "config_path": str(protocol["config_path"]),
    }
    return config


def parse_episodes(value: str, episode_count: int = 10) -> list[int]:
    """
    方法作用：
        将 all 或逗号分隔文本解析为 Episode 编号列表。
    输入参数：
        value (str)：选择文本；episode_count (int)：合法 Episode 总数。
    返回值：
        list[int]：去重后的合法编号。
    """
    if value.strip().lower() == "all":
        return list(range(1, episode_count + 1))
    episodes = []
    for item in value.split(","):
        try:
            episode = int(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError("episodes must be all or comma-separated integers") from exc
        if episode not in range(1, episode_count + 1):
            raise argparse.ArgumentTypeError(f"episode must be in 1..{episode_count}")
        if episode not in episodes:
            episodes.append(episode)
    if not episodes:
        raise argparse.ArgumentTypeError("at least one episode is required")
    return episodes


def summarize_episodes(root: Path, episode_count: int = 10) -> dict[str, Any]:
    """
    方法作用：
        汇总各 Episode 指标，计算 Macro 均值/标准差和按 Query 池化的 Micro Rank-1。
    输入参数：
        root (Path)：某阶段结果目录；episode_count (int)：Episode 数量。
    返回值：
        dict[str,Any]：聚合指标、完成数、总 Query 数及可选融合纠错统计。
    """
    rows = []
    for episode in range(1, episode_count + 1):
        path = root / f"fold_{episode:02d}" / "metrics.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        aggregate[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "episode_values": values,
        }
    query_count = sum(int(row["query_count"]) for row in rows)
    correct_count = sum(
        int(round(float(row["rank1_all"]) * int(row["query_count"]))) for row in rows
    )
    summary: dict[str, Any] = {
        "format": PROTOCOL_VERSION,
        "completed_episodes": len(rows),
        "episode_count": episode_count,
        "aggregate": aggregate,
        "micro_rank1_all": correct_count / query_count if query_count else None,
        "pooled_query_count": query_count,
        "pooled_correct_count": correct_count,
    }
    if rows and all("diagnostics" in row for row in rows):
        rescued = sum(int(row["diagnostics"]["fusion_vs_cls"]["rescued"]) for row in rows)
        harmed = sum(int(row["diagnostics"]["fusion_vs_cls"]["harmed"]) for row in rows)
        summary["fusion_vs_cls"] = {
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
        }
    _write_json(root / "summary.json", summary)
    return summary


def _ensure_new_or_resume(path: Path, resume: bool, eval_only: bool) -> None:
    """
    方法作用：
        防止普通新训练覆盖已有检查点。
    输入参数：
        path：检查点路径；resume：恢复开关；eval_only：只评测开关。
    返回值：
        None：允许继续时无返回，否则抛出 FileExistsError。
    """
    if path.is_file() and not (resume or eval_only):
        raise FileExistsError(
            f"Existing checkpoint found: {path}. Use --resume or --eval-only."
        )


def run_stage(
    protocol: dict[str, Any],
    stage: str,
    episodes: Iterable[int],
    device: str,
    resume: bool,
    eval_only: bool,
    epochs: int | None,
) -> dict[str, Any]:
    """
    方法作用：
        对选定 Episode 执行 p0 训练、p2a 训练或 p2b 缓存融合，并汇总阶段结果。
    输入参数：
        protocol：协议；stage：阶段；episodes：编号迭代器；device：设备；
        resume/eval_only：运行模式；epochs：可选训练轮数覆盖。
    返回值：
        dict[str,Any]：该阶段的 Episode 汇总。
    """
    build_episode_splits(protocol)
    config = _stage_config(protocol, stage)
    if epochs is not None:
        if epochs < 1:
            raise ValueError("--epochs must be positive")
        if stage == "p0":
            config["b2"]["epochs"] = epochs
            config["b2"]["warmup_epochs"] = min(
                int(config["b2"]["warmup_epochs"]), epochs - 1
            )
        elif stage == "p2a":
            config["p2a"]["epochs"] = epochs
        else:
            raise ValueError("--epochs only applies to p0 or p2a")
    selected = list(episodes)
    if stage == "p0":
        root = config["paths"]["output_root"] / "dino" / "b2"
        for episode in selected:
            _ensure_new_or_resume(
                root / f"fold_{episode:02d}" / "best.pt", resume, eval_only
            )
            metrics = run_dino_training_fold(
                config, "b2", episode, device, resume, eval_only
            )
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    elif stage == "p2a":
        root = config["paths"]["output_root"]
        for episode in selected:
            _ensure_new_or_resume(
                root / f"fold_{episode:02d}" / "best.pt", resume, eval_only
            )
            metrics = run_p2a_fold(config, episode, device, resume, eval_only)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    elif stage == "p2b":
        if resume or eval_only:
            raise ValueError("P2b is cached-feature evaluation and has no checkpoint")
        if epochs is not None:
            raise ValueError("--epochs does not apply to P2b")
        root = config["paths"]["output_root"]
        for episode in selected:
            metrics = evaluate_p2b_fold(config, episode, device)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return summarize_episodes(root, int(protocol["protocol"]["episode_count"]))


def _print_plan(
    protocol: dict[str, Any], stage: str, episodes: list[int], device: str
) -> None:
    """
    方法作用：
        在 dry-run 模式打印解析后的协议、Episode 和阶段配置，不执行训练。
    输入参数：
        protocol：协议配置；stage：阶段；episodes：编号列表；device：设备。
    返回值：
        None：仅向标准输出打印 JSON。
    """
    payload = {
        "protocol": protocol["name"],
        "stage": stage,
        "episodes": episodes,
        "device": device,
        "generated_splits_root": str(protocol["paths"]["generated_splits_root"]),
        "output_root": str(protocol["paths"]["output_root"]),
        "resolved_stage_config": (
            serializable_config(_stage_config(protocol, stage))
            if stage in {"p0", "p2a", "p2b"}
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        构建 build-splits/p0/p2a/p2b/summary 协议 CLI。
    输入参数：
        无。
    返回值：
        argparse.ArgumentParser：协议参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="python -m PAD_Lite.p2b_three_two_protocol",
        description="Run the 10-combination 3-train/2-test four-way P2b protocol.",
    )
    parser.add_argument("stage", choices=("build-splits", "p0", "p2a", "p2b", "summary"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析协议命令、选择 Episode，并执行划分、阶段运行或汇总。
    输入参数：
        argv (list[str]|None)：显式命令行；None 时读取系统命令行。
    返回值：
        int：成功状态码 0。
    """
    args = build_parser().parse_args(argv)
    protocol = load_protocol_config(args.config)
    if args.output_root is not None:
        protocol["paths"]["output_root"] = _annotation_path(args.output_root)
    episode_count = int(protocol["protocol"]["episode_count"])
    episodes = parse_episodes(args.episodes, episode_count)
    if args.dry_run:
        _print_plan(protocol, args.stage, episodes, args.device)
        return 0
    if args.stage == "build-splits":
        result = build_episode_splits(protocol)
    elif args.stage == "summary":
        output_root = protocol["paths"]["output_root"]
        result = {
            stage: summarize_episodes(
                output_root / ("p0/dino/b2" if stage == "p0" else stage),
                episode_count,
            )
            for stage in ("p0", "p2a", "p2b")
        }
        _write_json(output_root / "protocol_summary.json", result)
    else:
        result = run_stage(
            protocol,
            args.stage,
            episodes,
            args.device,
            args.resume,
            args.eval_only,
            args.epochs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
