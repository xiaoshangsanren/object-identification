"""Backfill pairwise difficulty artifacts from an existing embedding file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fg_retrain.pairwise_difficulty import (
    build_pairwise_metrics_section,
    evaluate_pairwise_difficulty,
    save_pairwise_difficulty,
)


def build_parser() -> argparse.ArgumentParser:
    """构建历史embedding成对难度分析命令行。

    作用:
        接收特征文件、输出目录、设备、分块大小和逐类别展示数量。
    参数:
        无。
    返回值:
        配置完成的``argparse.ArgumentParser``。
    """

    parser = argparse.ArgumentParser(
        description="Compute all class-pair Recall@1..5 from saved test embeddings."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--top-n-per-class", type=int, default=20)
    parser.add_argument(
        "--update-metrics",
        action="store_true",
        help="Add a compact pairwise_difficulty section to metrics.json if it exists.",
    )
    return parser


def _save_json(path: Path, value: Any) -> None:
    """保存UTF-8 JSON文件。

    作用:
        使用缩进格式更新历史实验的``metrics.json``。
    参数:
        path: 目标JSON路径。
        value: 可JSON序列化对象。
    返回值:
        无。
    """

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """从已保存测试特征生成所有类别对Recall@1至Recall@5。

    作用:
        加载特征``[N,D]``和标签``[N]``，计算全类别对指标并写入指定实验目录；
        可选地为原``metrics.json``添加产物索引和最困难类别对摘要。
    参数:
        无；从命令行读取embedding、输出目录和计算设置。
    返回值:
        无；结果写入四个pairwise difficulty文件。
    """

    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    payload = torch.load(args.embeddings.resolve(), map_location="cpu", weights_only=False)
    features = payload["features"]
    labels = payload["labels"]
    class_names = list(payload["class_names"])
    evaluation = evaluate_pairwise_difficulty(
        features=features,
        labels=labels,
        class_names=class_names,
        device=torch.device(args.device),
        query_chunk_size=args.query_chunk_size,
        recall_ks=(1, 2, 3, 4, 5),
        top_n_per_class=args.top_n_per_class,
    )
    output_dir = args.output_dir.resolve()
    artifacts = save_pairwise_difficulty(output_dir, evaluation)
    if args.update_metrics:
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Cannot update missing metrics file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["pairwise_difficulty"] = build_pairwise_metrics_section(
            evaluation, artifacts, top_pair_count=20
        )
        _save_json(metrics_path, metrics)
    print(json.dumps({**evaluation.summary, "top_hardest_pairs": evaluation.summary["top_hardest_pairs"][:5]}, ensure_ascii=False, indent=2))
    print(f"Pairwise outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
