"""Command-line entry point for the E0 pure-vision ViT baseline."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers

from fg_retrain.config import ExperimentConfig
from fg_retrain.data import build_dataloader, load_stanford_cars_parquet
from fg_retrain.modeling import PureVisualViT, extract_embeddings, load_image_processor
from fg_retrain.pairwise_difficulty import (
    build_pairwise_metrics_section,
    evaluate_pairwise_difficulty,
    save_pairwise_difficulty,
)
from fg_retrain.retrieval import evaluate_leave_one_out


def build_parser() -> argparse.ArgumentParser:
    """构建E0命令行参数解析器。

    作用:
        接收配置路径、可选设备覆盖和运行目录名称，核心实验参数仍保存在JSON中。
    参数:
        无。
    返回值:
        配置完成的``argparse.ArgumentParser``。
    """

    parser = argparse.ArgumentParser(
        description="ImageNet ViT leave-one-out retrieval on Stanford Cars-196 test split."
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON path.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Optional runtime override for the configured device.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name; default is a timestamp.",
    )
    return parser


def repository_root() -> Path:
    """定位项目仓库根目录。

    作用:
        根据当前源文件位置解析根目录，使入口不依赖调用者工作目录。
    参数:
        无。
    返回值:
        仓库根目录绝对``Path``。
    """

    return Path(__file__).resolve().parents[4]


def select_device(requested: str) -> torch.device:
    """校验并创建PyTorch计算设备。

    作用:
        当请求CUDA但CUDA不可用时提前报错。
    参数:
        requested: ``cuda``或``cpu``字符串。
    返回值:
        ``torch.device``实例。
    """

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """设置E0使用的随机种子。

    作用:
        初始化Python、PyTorch及所有可见CUDA设备的随机状态。
    参数:
        seed: 整数随机种子。
    返回值:
        无。
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_directory(output_root: Path, run_name: str | None) -> Path:
    """创建不会覆盖历史结果的运行目录。

    作用:
        使用指定名称或时间戳创建子目录；名称冲突时自动添加数字后缀。
    参数:
        output_root: E0输出根目录。
        run_name: 可选运行名称；为null时使用当前时间。
    返回值:
        本次新建的运行目录绝对``Path``。
    """

    stem = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def save_json(path: Path, value: Any) -> None:
    """保存易读的UTF-8 JSON文件。

    作用:
        使用缩进和非ASCII字符原样输出配置或指标。
    参数:
        path: 目标JSON路径。
        value: 可JSON序列化的对象。
    返回值:
        无。
    """

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """将逐Query记录保存为JSONL。

    作用:
        每行写入一个紧凑JSON对象，便于流式读取大量检索结果。
    参数:
        path: 目标JSONL路径。
        records: 长度``[N]``的逐Query记录列表，每项含Top-K邻居``[K]``。
    返回值:
        无。
    """

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """采集E0复现实验所需的运行环境。

    作用:
        记录Python、平台、PyTorch、Transformers、Datasets、CUDA和GPU信息。
    参数:
        device: 本次特征提取和检索使用的设备。
    返回值:
        可序列化的环境信息字典；不包含图像主数据。
    """

    result: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_device_index": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                "torch_cuda_version": torch.version.cuda,
            }
        )
    return result


def main() -> None:
    """执行完整E0纯视觉留一法检索实验。

    作用:
        解析配置，加载测试集和ViT，提取特征``[N,768]``，执行Gallery大小``N-1``的
        检索评测，并保存指标、逐Query结果和特征。
    参数:
        无；从命令行读取``--config``等参数。
    返回值:
        无；实验产物写入新建的输出目录。
    """

    args = build_parser().parse_args()
    raw_config = ExperimentConfig.from_json(args.config.resolve())
    config = raw_config.resolved(repository_root())
    device = select_device(args.device or config.device)
    seed_everything(config.seed)

    if not config.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.dataset_root}")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {config.model_path}")

    run_dir = make_run_directory(config.output_dir, args.run_name)
    save_json(run_dir / "config_resolved.json", config.as_serializable_dict())
    save_json(run_dir / "environment.json", runtime_environment(device))

    print(f"[1/4] Loading Stanford Cars-196 {config.split} split")
    data = load_stanford_cars_parquet(config.dataset_root, config.split)
    print(f"      images={len(data.dataset)}, classes={len(data.class_names)}")

    print(f"[2/4] Loading pure-vision ViT from {config.model_path}")
    processor = load_image_processor(config.model_path)
    model = PureVisualViT(config.model_path).to(device)
    dataloader = build_dataloader(
        data.dataset,
        processor,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_cuda=device.type == "cuda",
    )

    print("[3/4] Extracting normalized CLS embeddings")
    features, labels = extract_embeddings(model, dataloader, device, amp=config.amp)
    if config.save_embeddings:
        torch.save(
            {
                "features": features,
                "labels": labels,
                "class_names": data.class_names,
                "split": config.split,
                "model_path": str(config.model_path),
            },
            run_dir / "test_embeddings.pt",
        )

    print("[4/4] Evaluating: each query uses every other test image as gallery")
    evaluation = evaluate_leave_one_out(
        features=features,
        labels=labels,
        class_names=data.class_names,
        device=device,
        query_chunk_size=config.retrieval_query_chunk_size,
        recall_ks=config.recall_ks,
        compute_map=config.compute_map,
        save_top_k=config.save_top_k,
    )
    evaluation.metrics["experiment_name"] = config.experiment_name
    evaluation.metrics["model"] = "google/vit-base-patch16-224 (ImageNet-21k pretraining)"
    evaluation.metrics["embedding"] = "L2-normalized final-layer CLS token"
    evaluation.metrics["test_parquet_files"] = [str(path) for path in data.parquet_files]
    print("      computing all class-pair Recall@1..5 difficulty metrics")
    pairwise_evaluation = evaluate_pairwise_difficulty(
        features=features,
        labels=labels,
        class_names=data.class_names,
        device=device,
        query_chunk_size=config.pairwise_query_chunk_size,
        recall_ks=config.pairwise_recall_ks,
        top_n_per_class=config.pairwise_top_n_per_class,
    )
    pairwise_artifacts = save_pairwise_difficulty(run_dir, pairwise_evaluation)
    evaluation.metrics["pairwise_difficulty"] = build_pairwise_metrics_section(
        pairwise_evaluation,
        pairwise_artifacts,
        top_pair_count=20,
    )
    save_json(run_dir / "metrics.json", evaluation.metrics)
    save_jsonl(run_dir / "per_query.jsonl", evaluation.per_query)

    print(json.dumps({k: v for k, v in evaluation.metrics.items() if k != "per_class"}, indent=2))
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
