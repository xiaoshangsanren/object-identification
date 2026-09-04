from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANNOTATION_ROOT, serializable_config
from .data import (
    CropDataset,
    CropIndex,
    build_eval_loader,
    load_fold,
    records_for_partition,
)
from .dino_engine import build_dino_eval_transform
from .dino_models import FrozenDinoEncoder, load_local_dino
from .engine import choose_device, seed_everything
from .metrics import (
    compute_retrieval_metrics,
    encode_loader,
    save_retrieval_outputs,
    summarize_folds,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "frozen_dino_convnext_p2b_comparison.json"
)
RAW_BACKBONES = ("dino_raw", "convnext_raw")


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：
        将配置中的相对路径按 Annotation 根目录解析为绝对路径。

    输入参数：
        value (str|Path)：配置文件中的路径。

    返回值：
        Path：解析后的绝对路径。
    """

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def load_comparison_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：
        读取三模型冻结检索对比配置，并校验实验不会包含训练参数。

    输入参数：
        path (str|Path)：JSON 配置文件路径。

    返回值：
        dict[str,Any]：路径已解析的实验配置。
    """

    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Frozen comparison schema_version must be 1")
    for section in ("paths", "data", "retrieval"):
        if section not in config:
            raise ValueError(f"Frozen comparison config is missing {section}")
    for key in (
        "crop_root",
        "splits_root",
        "dino_model",
        "convnext_model",
        "p2b_results_root",
        "output_root",
    ):
        if key not in config["paths"]:
            raise ValueError(f"Frozen comparison config is missing paths.{key}")
        config["paths"][key] = _resolve_path(config["paths"][key])
    if int(config["data"].get("fold_count", 0)) != 5:
        raise ValueError("Frozen comparison requires exactly five folds")
    if int(config["retrieval"].get("prototype_top_k", 0)) < 1:
        raise ValueError("retrieval.prototype_top_k must be positive")
    forbidden = {"training", "optimizer", "loss"}.intersection(config)
    if forbidden:
        raise ValueError(f"Frozen comparison forbids training sections: {sorted(forbidden)}")
    config["config_path"] = config_path
    return config


def _sha256(path: Path) -> str:
    """
    方法作用：
        流式计算单个模型权重文件的 SHA-256 摘要。

    输入参数：
        path (Path)：普通模型权重文件路径。

    返回值：
        str：十六进制 SHA-256 摘要。
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_file(model_root: Path) -> Path:
    """
    方法作用：
        从 Hugging Face 本地模型目录中定位实际权重文件。

    输入参数：
        model_root (Path)：本地模型目录。

    返回值：
        Path：model.safetensors 或 pytorch_model.bin 权重文件。
    """

    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No local model weight file in {model_root}")


class FrozenConvNextEncoder(nn.Module):
    """
    方法作用：
        冻结原始 ConvNeXt 主干，并输出 L2 归一化的全局池化特征。

    输入参数：
        convnext (nn.Module)：本地预训练 ConvNeXtModel。

    返回值：
        FrozenConvNextEncoder：不可训练的 ConvNeXt 检索编码器。
    """

    def __init__(self, convnext: nn.Module) -> None:
        """
        方法作用：
            冻结 ConvNeXt 全部参数并切换至评估模式。

        输入参数：
            convnext (nn.Module)：预训练 ConvNeXtModel。

        返回值：
            None：仅初始化模块状态。
        """

        super().__init__()
        self.convnext = convnext
        for parameter in self.convnext.parameters():
            parameter.requires_grad_(False)
        self.convnext.eval()

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            提取 ConvNeXt 最终全局池化特征并执行 L2 归一化。

        输入参数：
            pixel_values (torch.Tensor)：图像批次，形状 [B,3,H,W]。

        返回值：
            torch.Tensor：归一化特征，形状 [B,768]（ConvNeXt-Tiny）。
        """

        output = self.convnext(pixel_values=pixel_values)
        features = output.pooler_output.float()
        return F.normalize(features, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            调用冻结编码器前向传播。

        输入参数：
            pixel_values (torch.Tensor)：图像批次，形状 [B,3,H,W]。

        返回值：
            torch.Tensor：归一化全局特征，形状 [B,768]。
        """

        return self.encode(pixel_values)


def load_frozen_encoder(
    backbone: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, Path, int]:
    """
    方法作用：
        从本地权重创建原始 DINOv2 或原始 ConvNeXt 冻结编码器。

    输入参数：
        backbone (str)：dino_raw 或 convnext_raw；config：实验配置；
        device (torch.device)：模型运行设备。

    返回值：
        tuple[nn.Module,Path,int]：冻结编码器、实际权重文件和总参数量。
    """

    if backbone == "dino_raw":
        model_root = config["paths"]["dino_model"]
        model, _ = load_local_dino(model_root)
        encoder: nn.Module = FrozenDinoEncoder(model)
    elif backbone == "convnext_raw":
        from transformers import ConvNextModel

        model_root = config["paths"]["convnext_model"]
        model = ConvNextModel.from_pretrained(str(model_root), local_files_only=True)
        encoder = FrozenConvNextEncoder(model)
    else:
        raise ValueError(f"Unsupported frozen backbone: {backbone}")
    encoder = encoder.to(device).eval()
    trainable = sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"{backbone} unexpectedly has {trainable} trainable parameters")
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    return encoder, _weight_file(model_root), parameter_count


def build_frozen_transform(config: dict[str, Any]):
    """
    方法作用：
        构建 DINOv2 与 ConvNeXt 共用的原生 224 CenterCrop 评估预处理。

    输入参数：
        config (dict[str,Any])：包含 image_size、resize_short_edge 的配置。

    返回值：
        torchvision.transforms.Compose：PIL 图像到 [3,H,W] 张量的变换。
    """

    return build_dino_eval_transform(
        int(config["data"]["image_size"]),
        int(config["data"]["resize_short_edge"]),
        "center_crop",
    )


def _sync(device: torch.device) -> None:
    """
    方法作用：
        在 CUDA 设备上同步队列，使推理计时包含真实执行时间。

    输入参数：
        device (torch.device)：当前执行设备。

    返回值：
        None：仅执行同步操作。
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_frozen_fold(
    backbone: str,
    encoder: nn.Module,
    config: dict[str, Any],
    crop_index: CropIndex,
    fold_number: int,
    device: torch.device,
    output_root: Path,
    parameter_count: int,
    weight_path: Path,
) -> dict[str, Any]:
    """
    方法作用：
        在单折 Novel Support/Query 上执行冻结特征 Gallery 检索。

    输入参数：
        backbone (str)：冻结模型名称；encoder：输入 [B,3,H,W]、输出 [B,D]；
        config：实验配置；crop_index：裁剪图索引；fold_number：折号；
        device：执行设备；output_root：本次运行目录；parameter_count：参数量；
        weight_path：原始权重文件。

    返回值：
        dict[str,Any]：Rank-1、mAP、逐类别结果及推理开销。
    """

    fold = load_fold(config["paths"]["splits_root"], fold_number)
    class_names = list(fold["novel_classes"])
    class_to_local = {name: index for index, name in enumerate(class_names)}
    transform = build_frozen_transform(config)
    support_records = records_for_partition(
        crop_index,
        fold,
        "novel_support",
        exclude_tiny=bool(config["data"].get("exclude_tiny_support", True)),
    )
    query_records = records_for_partition(crop_index, fold, "novel_query")
    support_loader = build_eval_loader(
        CropDataset(support_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    query_loader = build_eval_loader(
        CropDataset(query_records, transform, class_to_local),
        int(config["data"]["eval_batch_size"]),
        int(config["data"]["workers"]),
    )
    amp = bool(config["data"].get("amp", True)) and device.type == "cuda"
    _sync(device)
    began = time.perf_counter()
    support_features, support_labels, _ = encode_loader(encoder, support_loader, device, amp)
    query_features, query_labels, query_metadata = encode_loader(
        encoder, query_loader, device, amp
    )
    _sync(device)
    elapsed = time.perf_counter() - began
    query_tiny = np.asarray([item["tiny"] for item in query_metadata], dtype=bool)
    metrics, class_scores = compute_retrieval_metrics(
        support_features,
        support_labels,
        query_features,
        query_labels,
        query_tiny,
        class_names,
        int(config["retrieval"]["prototype_top_k"]),
    )
    total_images = len(support_records) + len(query_records)
    metrics.update(
        {
            "variant": backbone,
            "visual_backbone": backbone,
            "fold": fold_number,
            "novel_classes": class_names,
            "training_performed": False,
            "trainable_parameter_count": 0,
            "parameter_count": parameter_count,
            "weight_path": str(weight_path),
            "weight_sha256": _sha256(weight_path),
            "input_mode": "center_crop",
            "image_size": int(config["data"]["image_size"]),
            "feature_extraction_seconds": elapsed,
            "milliseconds_per_encoded_image": 1000.0 * elapsed / max(total_images, 1),
        }
    )
    fold_root = output_root / backbone / f"fold_{fold_number:02d}"
    save_retrieval_outputs(
        fold_root,
        metrics,
        class_names,
        class_scores,
        query_labels,
        query_metadata,
        support_features,
        support_labels,
        query_features,
    )
    run_config = {
        "format": "annotation_pad_lite_frozen_backbone_comparison_v1",
        "backbone": backbone,
        "fold": fold_number,
        "training_performed": False,
        "device": str(device),
        "config": serializable_config(config),
    }
    (fold_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_frozen_backbone(
    backbone: str,
    config: dict[str, Any],
    folds: Sequence[int],
    device_name: str,
    output_root: Path,
) -> dict[str, Any]:
    """
    方法作用：
        加载一次冻结模型并依次执行指定折次，过程中不创建优化器或反向传播。

    输入参数：
        backbone (str)：dino_raw 或 convnext_raw；config：实验配置；
        folds (Sequence[int])：待评测折号；device_name：运行设备；
        output_root (Path)：本次对比实验根目录。

    返回值：
        dict[str,Any]：当前冻结模型的跨折汇总。
    """

    if backbone not in RAW_BACKBONES:
        raise ValueError(f"backbone must be one of {RAW_BACKBONES}")
    device = choose_device(device_name)
    seed_everything(int(config.get("seed", 2026)))
    crop_index = CropIndex(config["paths"]["crop_root"])
    encoder, weight_path, parameter_count = load_frozen_encoder(backbone, config, device)
    for fold_number in folds:
        metrics = _evaluate_frozen_fold(
            backbone,
            encoder,
            config,
            crop_index,
            fold_number,
            device,
            output_root,
            parameter_count,
            weight_path,
        )
        print(
            f"{backbone} fold {fold_number:02d}: "
            f"Rank-1={metrics['rank1_all']:.4f} "
            f"mAP={metrics['retrieval_map']:.4f} "
            f"ms/image={metrics['milliseconds_per_encoded_image']:.2f}"
        )
    summary = summarize_folds(output_root / backbone)
    summary.update(
        {
            "variant": backbone,
            "training_performed": False,
            "trainable_parameter_count": 0,
            "parameter_count": parameter_count,
            "weight_path": str(weight_path),
            "weight_sha256": _sha256(weight_path),
        }
    )
    (output_root / backbone / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_method_folds(root: Path) -> list[dict[str, Any]]:
    """
    方法作用：
        读取一个方法的五折 metrics.json，并保证折次完整。

    输入参数：
        root (Path)：包含 fold_01 至 fold_05 的方法结果目录。

    返回值：
        list[dict[str,Any]]：按折号排序的五折指标。
    """

    rows = []
    for fold in range(1, 6):
        path = root / f"fold_{fold:02d}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing comparison result: {path}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _micro_rank1(root: Path, rows: Sequence[dict[str, Any]]) -> tuple[float, int, int]:
    """
    方法作用：
        依据逐样本预测统计跨折 Micro Rank-1，避免折大小不同造成偏差。

    输入参数：
        root (Path)：方法结果目录；rows：五折指标。

    返回值：
        tuple[float,int,int]：Micro Rank-1、正确Query数、总Query数。
    """

    correct = 0
    total = 0
    for row in rows:
        fold = int(row["fold"])
        path = root / f"fold_{fold:02d}" / "predictions.json"
        predictions = json.loads(path.read_text(encoding="utf-8"))
        if len(predictions) != int(row["query_count"]):
            raise ValueError(f"Prediction count differs from query_count: {path}")
        correct += sum(bool(item["correct"]) for item in predictions)
        total += len(predictions)
    return correct / max(total, 1), correct, total


def _validate_shared_protocol(method_folds: dict[str, list[dict[str, Any]]]) -> None:
    """
    方法作用：
        验证三种方法逐折使用相同类别、Support数量和Query数量。

    输入参数：
        method_folds (dict[str,list[dict]])：三种方法的五折指标。

    返回值：
        None：协议一致时正常结束，不一致时抛出异常。
    """

    names = list(method_folds)
    reference = method_folds[names[0]]
    for name in names[1:]:
        for expected, actual in zip(reference, method_folds[name]):
            keys = ("fold", "novel_classes", "support_count", "query_count")
            for key in keys:
                if expected.get(key) != actual.get(key):
                    raise ValueError(
                        f"Protocol mismatch for {name} fold {expected.get('fold')}: {key}"
                    )


def summarize_three_way(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """
    方法作用：
        汇总原始DINO、原始ConvNeXt与已有P2B的同协议五折性能。

    输入参数：
        config (dict[str,Any])：对比配置；output_root (Path)：本次输出目录。

    返回值：
        dict[str,Any]：三方法Macro/Micro Rank-1、逐折结果和两组差值。
    """

    roots = {
        "dino_raw": output_root / "dino_raw",
        "convnext_raw": output_root / "convnext_raw",
        "p2b": config["paths"]["p2b_results_root"],
    }
    method_folds = {name: _load_method_folds(root) for name, root in roots.items()}
    _validate_shared_protocol(method_folds)
    methods: dict[str, Any] = {}
    for name, rows in method_folds.items():
        fold_values = [float(row["rank1_all"]) for row in rows]
        micro, correct, query_count = _micro_rank1(roots[name], rows)
        methods[name] = {
            "training_performed_in_this_run": False,
            "representation": (
                "frozen_pretrained_cls"
                if name == "dino_raw"
                else "frozen_pretrained_global_pool"
                if name == "convnext_raw"
                else "existing_trained_cls_plus_weighted_patch"
            ),
            "fold_rank1": fold_values,
            "macro_rank1": float(np.mean(fold_values)),
            "std_rank1": float(np.std(fold_values)),
            "micro_rank1": micro,
            "correct_query_count": correct,
            "query_count": query_count,
            "mean_retrieval_map": float(np.mean([float(row["retrieval_map"]) for row in rows])),
        }
        if name != "p2b":
            summary = json.loads((roots[name] / "summary.json").read_text(encoding="utf-8"))
            methods[name]["parameter_count"] = int(summary["parameter_count"])
            methods[name]["weight_sha256"] = summary["weight_sha256"]
            methods[name]["mean_milliseconds_per_encoded_image"] = float(
                np.mean([float(row["milliseconds_per_encoded_image"]) for row in rows])
            )
    comparisons = {
        "experiment_2a_dino_raw_minus_convnext_raw": (
            methods["dino_raw"]["macro_rank1"] - methods["convnext_raw"]["macro_rank1"]
        ),
        "experiment_2b_p2b_minus_convnext_raw": (
            methods["p2b"]["macro_rank1"] - methods["convnext_raw"]["macro_rank1"]
        ),
        "p2b_minus_dino_raw": (
            methods["p2b"]["macro_rank1"] - methods["dino_raw"]["macro_rank1"]
        ),
    }
    payload = {
        "experiment": config.get("name", "frozen_dino_convnext_p2b_comparison"),
        "protocol": {
            "fold_count": 5,
            "classes_per_fold": 2,
            "prototype_top_k": int(config["retrieval"]["prototype_top_k"]),
            "same_support_query_for_all_methods": True,
            "convnext_training_performed": False,
            "raw_dino_training_performed": False,
            "p2b_reused_existing_results": True,
        },
        "methods": methods,
        "comparisons": comparisons,
        "output_root": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    labels = {"dino_raw": "原始DINOv2-Small", "convnext_raw": "原始ConvNeXt-Tiny", "p2b": "P2B"}
    lines = [
        "# 实验2：原始DINO、原始ConvNeXt与P2B性能对比",
        "",
        "ConvNeXt-Tiny与原始DINO在本次运行中均全冻结、零训练；P2B复用已有五折结果。",
        "",
        "| 方法 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Macro Rank-1 | Micro Rank-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("dino_raw", "convnext_raw", "p2b"):
        row = methods[name]
        values = " | ".join(f"{value * 100:.2f}%" for value in row["fold_rank1"])
        lines.append(
            f"| {labels[name]} | {values} | {row['macro_rank1'] * 100:.2f}% | "
            f"{row['micro_rank1'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 两组预定比较",
            "",
            f"- 实验2A：原始DINO - 原始ConvNeXt = {comparisons['experiment_2a_dino_raw_minus_convnext_raw'] * 100:+.2f}个百分点。",
            f"- 实验2B：P2B - 原始ConvNeXt = {comparisons['experiment_2b_p2b_minus_convnext_raw'] * 100:+.2f}个百分点。",
            f"- P2B - 原始DINO = {comparisons['p2b_minus_dino_raw'] * 100:+.2f}个百分点。",
            "",
            "注意：P2B包含领域训练和Patch融合，而两个原始backbone没有训练，因此结果是现成方案性能比较，不是纯backbone公平性结论。",
        ]
    )
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _parse_folds(value: str) -> list[int]:
    """
    方法作用：
        将 all 或逗号分隔折号解析为去重有序整数列表。

    输入参数：
        value (str)：all、1 或 1,2,3 等文本。

    返回值：
        list[int]：范围1至5的折号列表。
    """

    if value.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > 5 for fold in folds):
        raise ValueError("folds must be all or comma-separated values in 1..5")
    return folds


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建冻结三模型对比的 run/summarize 命令行解析器。

    输入参数：
        无。

    返回值：
        argparse.ArgumentParser：配置完整的命令行解析器。
    """

    parser = argparse.ArgumentParser(
        description="Compare frozen raw DINO, frozen raw ConvNeXt, and existing P2B."
    )
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone", choices=RAW_BACKBONES, default="dino_raw")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", required=True)
    return parser


def main() -> None:
    """
    方法作用：
        解析命令并执行冻结模型评测或三方法最终汇总。

    输入参数：
        无；参数来自命令行。

    返回值：
        None：结果写入 --run-root 指定目录。
    """

    args = build_parser().parse_args()
    config = load_comparison_config(args.config)
    output_root = Path(args.run_root).expanduser().resolve()
    if args.command == "run":
        run_frozen_backbone(
            args.backbone,
            config,
            _parse_folds(args.folds),
            args.device,
            output_root,
        )
    else:
        summary = summarize_three_way(config, output_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
