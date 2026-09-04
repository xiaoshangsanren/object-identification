from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CropDataset, CropIndex, build_eval_loader, load_fold, records_for_partition
from .dino_engine import build_dino_eval_transform
from .dino_models import FrozenDinoEncoder, load_local_dino
from .dino_patch_engine import PatchCropDataset, PatchLetterboxTransform
from .dino_patch_models import FrozenVisualHeads
from .engine import choose_device, seed_everything
from .fixed_gallery_fourway import FrozenConvNextP2B, load_fourway_config
from .frozen_backbone_comparison import FrozenConvNextEncoder


FORMAT_VERSION = "annotation_pad_lite_fixed_gallery_latency_v1.00"
METHOD_ORDER = ("dino_raw", "dino_p2b", "convnext_raw", "convnext_p2b")
METHOD_LABELS = {
    "dino_raw": "原始 DINOv2-Small",
    "dino_p2b": "DINOv2-Small + P2B",
    "convnext_raw": "原始 ConvNeXt-Tiny",
    "convnext_p2b": "冻结 ConvNeXt-Tiny + P2B轻量头",
}


class SharedDinoP2BEncoder(nn.Module):
    """
    方法作用：共享一次 DINOv2 前向，同时重放 B2 CLS 头和 P2A Patch 头并进行 P2B 融合。
    输入参数：dino，冻结的 DINOv2-Small；heads，当前折冻结的 B2/P2A 轻量头。
    返回值：编码器；输入图像 [B,3,S,S]、有效 Patch 掩码 [B,N]，输出融合特征 [B,1024]。
    """

    def __init__(self, dino: nn.Module, heads: FrozenVisualHeads) -> None:
        """
        方法作用：保存冻结主干和当前折轻量头，强制所有参数处于推理状态。
        输入参数：dino；heads。
        返回值：None。
        """
        super().__init__()
        self.dino = dino
        self.heads = heads
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def encode(
        self, pixel_values: torch.Tensor, valid_patch_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        方法作用：提取 CLS/Patch Token 并生成等权融合的 P2B 检索特征。
        输入参数：pixel_values [B,3,S,S]；valid_patch_masks [B,N]。
        返回值：L2 归一化 P2B 特征 [B,1024]。
        """
        hidden = self.dino(pixel_values=pixel_values).last_hidden_state
        _, _, fused, _ = self.heads.encode(
            hidden[:, 0], hidden[:, 1:], valid_patch_masks
        )
        return fused

    def forward(
        self, pixel_values: torch.Tensor, valid_patch_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        方法作用：转发到 encode，供统一推理接口调用。
        输入参数：pixel_values [B,3,S,S]；valid_patch_masks [B,N]。
        返回值：P2B 特征 [B,1024]。
        """
        return self.encode(pixel_values, valid_patch_masks)


class ConvNextP2BInferenceEncoder(nn.Module):
    """
    方法作用：把已训练 FrozenConvNextP2B 包装为只输出 50/50 融合检索特征的推理模型。
    输入参数：model，当前折 ConvNeXt P2B 模型。
    返回值：编码器；输入图像 [B,3,S,S]、有效 Patch 掩码 [B,N]，输出 [B,1024]。
    """

    def __init__(self, model: FrozenConvNextP2B) -> None:
        """
        方法作用：保存并冻结当前折 ConvNeXt P2B 模型。
        输入参数：model。
        返回值：None。
        """
        super().__init__()
        self.model = model
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def encode(
        self, pixel_values: torch.Tensor, valid_patch_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        方法作用：计算全局/Patch特征并进行 P2B 等权拼接与归一化。
        输入参数：pixel_values [B,3,S,S]；valid_patch_masks [B,N]。
        返回值：L2 归一化融合特征 [B,1024]。
        """
        global_features, patch_features, _ = self.model.encode_branches(
            pixel_values, valid_patch_masks
        )
        fused = torch.cat(
            (math.sqrt(0.5) * global_features, math.sqrt(0.5) * patch_features),
            dim=-1,
        )
        return F.normalize(fused, dim=-1)

    def forward(
        self, pixel_values: torch.Tensor, valid_patch_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        方法作用：转发到 encode，供统一推理接口调用。
        输入参数：pixel_values [B,3,S,S]；valid_patch_masks [B,N]。
        返回值：融合特征 [B,1024]。
        """
        return self.encode(pixel_values, valid_patch_masks)


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：将计时结果写成 UTF-8 格式化 JSON。
    输入参数：path，输出路径；payload，可序列化对象。
    返回值：None。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sync(device: torch.device) -> None:
    """
    方法作用：同步 CUDA 队列，确保墙钟计时覆盖实际 GPU 执行。
    输入参数：device，当前计算设备。
    返回值：None。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prototype_scores(
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    top_k: int,
) -> torch.Tensor:
    """
    方法作用：在 GPU 上以每类 Top-K Gallery 相似度均值计算类别得分。
    输入参数：query_features [B,D]；support_features [Ns,D]；support_labels [Ns]；类别数 C；K。
    返回值：类别得分 [B,C]。
    """
    columns: list[torch.Tensor] = []
    for class_id in range(num_classes):
        prototypes = support_features[support_labels == class_id]
        if not len(prototypes):
            raise ValueError(f"No Gallery prototype for local class {class_id}")
        similarities = query_features @ prototypes.T
        effective_k = min(max(1, top_k), similarities.shape[1])
        columns.append(similarities.topk(effective_k, dim=1).values.mean(dim=1))
    return torch.stack(columns, dim=1)


def _model_input(
    encoder: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    masked: bool,
) -> torch.Tensor:
    """
    方法作用：把一个批次移入 GPU 并执行当前方法的视觉编码。
    输入参数：encoder；batch，image [B,3,S,S]及可选mask [B,N]；device；masked。
    返回值：归一化检索特征 [B,D]。
    """
    images = batch["image"].to(device, non_blocking=True)
    if masked:
        masks = batch["valid_patch_mask"].to(device, non_blocking=True).bool()
        return encoder.encode(images, masks)
    return encoder.encode(images)


def _load_feature_bank(
    run_root: Path, method: str, fold_number: int, device: torch.device
) -> dict[str, Any]:
    """
    方法作用：读取当前方法/折已经离线缓存的 Gallery 和 Query 特征。
    输入参数：run_root；method；fold_number；device。
    返回值：dict；Gallery 特征 [Ns,D]/标签 [Ns] 已置于 device，Query 特征保留在 CPU。
    """
    path = run_root / method / f"fold_{fold_number:02d}" / "features.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached feature bank: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    bank["support_features"] = bank["support_features"].float().to(device)
    bank["support_labels"] = bank["support_labels"].long().to(device)
    return bank


def _load_encoder(
    config: dict[str, Any],
    run_root: Path,
    method: str,
    fold: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, Any, bool, dict[str, Any]]:
    """
    方法作用：按方法和折加载真实推理模型、评估变换及输入掩码模式。
    输入参数：config；run_root；method；fold；device。
    返回值：encoder；图像变换；是否需要mask；模型/输入元数据。
    """
    if method == "dino_raw":
        backbone, _ = load_local_dino(config["paths"]["dino_model"])
        encoder: nn.Module = FrozenDinoEncoder(backbone).to(device).eval()
        settings = config["raw"]
        transform = build_dino_eval_transform(
            int(settings["image_size"]), int(settings["resize_short_edge"]), "center_crop"
        )
        return encoder, transform, False, {"input_size": 224, "input_mode": "center_crop"}
    if method == "convnext_raw":
        from transformers import ConvNextModel

        backbone = ConvNextModel.from_pretrained(
            str(config["paths"]["convnext_model"]), local_files_only=True
        )
        encoder = FrozenConvNextEncoder(backbone).to(device).eval()
        settings = config["raw"]
        transform = build_dino_eval_transform(
            int(settings["image_size"]), int(settings["resize_short_edge"]), "center_crop"
        )
        return encoder, transform, False, {"input_size": 224, "input_mode": "center_crop"}
    if method == "dino_p2b":
        backbone, _ = load_local_dino(config["paths"]["dino_model"])
        fold_number = int(fold["fold"])
        b2_path = (
            run_root
            / "dino_p2b_components/b2/dino/b2"
            / f"fold_{fold_number:02d}"
            / "best.pt"
        )
        p2a_path = (
            run_root
            / "dino_p2b_components/p2a"
            / f"fold_{fold_number:02d}"
            / "best.pt"
        )
        b2_state = torch.load(b2_path, map_location="cpu", weights_only=False)["model_state"]
        p2a_state = torch.load(p2a_path, map_location="cpu", weights_only=False)["model_state"]
        heads = FrozenVisualHeads(b2_state, p2a_state).to(device)
        encoder = SharedDinoP2BEncoder(backbone.to(device), heads).eval()
        settings = config["dino_p2b"]
        transform = PatchLetterboxTransform(
            int(settings["image_size"]), 14, train=False
        )
        return encoder, transform, True, {"input_size": 336, "input_mode": "letterbox"}
    if method == "convnext_p2b":
        from transformers import ConvNextModel

        settings = config["convnext_p2b"]
        backbone = ConvNextModel.from_pretrained(
            str(config["paths"]["convnext_model"]), local_files_only=True
        )
        model = FrozenConvNextP2B(
            backbone,
            len(fold["base_classes"]),
            int(settings["embedding_dim"]),
            float(settings["dropout"]),
        )
        checkpoint_path = run_root / method / f"fold_{int(fold['fold']):02d}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        encoder = ConvNextP2BInferenceEncoder(model.to(device)).eval()
        transform = PatchLetterboxTransform(
            int(settings["image_size"]), int(settings["patch_stride"]), train=False
        )
        return encoder, transform, True, {"input_size": 320, "input_mode": "letterbox"}
    raise ValueError(f"Unsupported latency method: {method}")


def _validate_first_batch(
    encoder: nn.Module,
    loader: Any,
    bank: dict[str, Any],
    device: torch.device,
    masked: bool,
    amp: bool,
) -> dict[str, float | int]:
    """
    方法作用：用首批 Query 验证新计时入口与原实验缓存特征保持一致。
    输入参数：encoder；loader；bank；device；masked；amp。
    返回值：dict，比较样本数、最大绝对误差和平均余弦相似度。
    """
    batch = next(iter(loader))
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp and device.type == "cuda",
    ):
        actual = _model_input(encoder, batch, device, masked).float()
    _sync(device)
    expected = bank["query_features"][: len(actual)].float().to(device)
    if actual.shape != expected.shape:
        raise ValueError(f"Feature shape differs: {tuple(actual.shape)} != {tuple(expected.shape)}")
    cosine = F.cosine_similarity(actual, expected, dim=1)
    result = {
        "sample_count": int(len(actual)),
        "max_abs_error": float((actual - expected).abs().max().item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
    }
    if result["mean_cosine_similarity"] < 0.999:
        raise RuntimeError(f"Benchmark encoder does not reproduce cached features: {result}")
    return result


def _benchmark_loader(
    encoder: nn.Module,
    loader: Any,
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    num_classes: int,
    top_k: int,
    device: torch.device,
    masked: bool,
    amp: bool,
    warmup_batches: int,
    repeats: int,
) -> dict[str, Any]:
    """
    方法作用：测量读图、预处理、H2D、视觉前向和 Gallery 类别打分的端到端墙钟耗时。
    输入参数：encoder；loader，Query批次；Gallery特征 [Ns,D]/标签 [Ns]；C；K；device；模式与计时次数。
    返回值：dict，重复耗时、平均/标准差 ms/image、吞吐和最后一次预测 [N]。
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    encoder.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                features = _model_input(encoder, batch, device, masked)
                _prototype_scores(features, support_features, support_labels, num_classes, top_k)
            if batch_index + 1 >= warmup_batches:
                break
    _sync(device)

    elapsed_values: list[float] = []
    final_predictions: list[torch.Tensor] = []
    final_labels: list[torch.Tensor] = []
    sample_count = 0
    for repeat_index in range(repeats):
        predictions: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        count = 0
        _sync(device)
        began = time.perf_counter()
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp and device.type == "cuda",
                ):
                    features = _model_input(encoder, batch, device, masked)
                    scores = _prototype_scores(
                        features, support_features, support_labels, num_classes, top_k
                    )
                predictions.append(scores.argmax(dim=1))
                labels.append(batch["label"].long())
                count += int(len(batch["label"]))
        _sync(device)
        elapsed_values.append(time.perf_counter() - began)
        if repeat_index == repeats - 1:
            final_predictions = [value.cpu() for value in predictions]
            final_labels = labels
            sample_count = count
    milliseconds = [1000.0 * value / max(sample_count, 1) for value in elapsed_values]
    mean_ms = statistics.fmean(milliseconds)
    predictions_tensor = torch.cat(final_predictions)
    labels_tensor = torch.cat(final_labels)
    return {
        "query_count": sample_count,
        "repeats": repeats,
        "elapsed_seconds": elapsed_values,
        "milliseconds_per_image_each_repeat": milliseconds,
        "mean_milliseconds_per_image": mean_ms,
        "std_milliseconds_per_image": statistics.pstdev(milliseconds),
        "images_per_second": 1000.0 / mean_ms,
        "rank1_check": float((predictions_tensor == labels_tensor).float().mean().item()),
    }


def _weighted_method_summary(folds: Sequence[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    """
    方法作用：按 Query 数加权汇总五折指定批大小的平均时延。
    输入参数：folds，五折结果；batch_size，待汇总批大小。
    返回值：dict，加权 ms/image、images/s及逐折数值。
    """
    key = f"batch_{batch_size}"
    total_queries = sum(int(row[key]["query_count"]) for row in folds)
    weighted_ms = sum(
        float(row[key]["mean_milliseconds_per_image"]) * int(row[key]["query_count"])
        for row in folds
    ) / max(total_queries, 1)
    return {
        "batch_size": batch_size,
        "query_count": total_queries,
        "mean_milliseconds_per_image": weighted_ms,
        "images_per_second": 1000.0 / weighted_ms,
        "fold_milliseconds_per_image": [
            float(row[key]["mean_milliseconds_per_image"]) for row in folds
        ],
    }


def _write_markdown(run_root: Path, payload: dict[str, Any]) -> Path:
    """
    方法作用：生成独立时延表，并把相同内容追加到四路准确率报告。
    输入参数：run_root；payload，完整计时结果。
    返回值：Path，独立 Markdown 路径。
    """
    batch_sizes = [int(value) for value in payload["protocol"]["batch_sizes"]]
    lines = [
        "# 固定 Gallery 四路细粒度识别时延",
        "",
        "口径：Gallery 特征已离线缓存；模型加载与 Gallery 建库不计时。端到端时延包含 Query 读图、预处理、CPU→GPU、视觉模型前向，以及与固定 Gallery 做 Top-3 原型类别打分。每折模型加载后预热，随后完整 Query 重复计时。",
        "",
        "| 方法 | 输入 | Batch=1 平均耗时 | Batch=1 吞吐 | Batch=32 摊销耗时 | Batch=32 吞吐 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        result = payload["methods"][method]
        batch_one = result["aggregate"]["batch_1"]
        batch_many = result["aggregate"][f"batch_{batch_sizes[-1]}"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {result['model']['input_size']}×{result['model']['input_size']} "
            f"{result['model']['input_mode']} | {batch_one['mean_milliseconds_per_image']:.2f} ms/图 | "
            f"{batch_one['images_per_second']:.2f} 图/s | "
            f"{batch_many['mean_milliseconds_per_image']:.2f} ms/图 | "
            f"{batch_many['images_per_second']:.2f} 图/s |"
        )
    lines.extend(
        [
            "",
            f"环境：{payload['environment']['gpu_name']}；PyTorch {payload['environment']['torch_version']}；"
            f"CUDA {payload['environment']['cuda_version']}；数据加载 workers=0；"
            f"每种配置重复 {payload['protocol']['repeats']} 次。Batch=1 是当前最接近逐帧在线调用的主指标；"
            f"Batch={batch_sizes[-1]} 只表示离线批处理摊销速度。该时延不包含前级 YOLO 检测与裁剪。",
            "",
            "## Batch=1 逐折平均耗时（ms/图）",
            "",
            "| 方法 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | 加权平均 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        row = payload["methods"][method]["aggregate"]["batch_1"]
        folds = " | ".join(f"{value:.2f}" for value in row["fold_milliseconds_per_image"])
        lines.append(
            f"| {METHOD_LABELS[method]} | {folds} | {row['mean_milliseconds_per_image']:.2f} |"
        )
    lines.append("")
    content = "\n".join(lines)
    output_path = run_root / "LATENCY_RESULTS.md"
    output_path.write_text(content, encoding="utf-8")

    report_path = run_root / "FOUR_WAY_RESULTS.md"
    if report_path.is_file():
        marker = "\n# 固定 Gallery 四路细粒度识别时延\n"
        report = report_path.read_text(encoding="utf-8")
        if marker in report:
            report = report.split(marker, 1)[0].rstrip() + "\n"
        report_path.write_text(report.rstrip() + "\n\n" + content + "\n", encoding="utf-8")
    return output_path


def run_latency_benchmark(
    config: dict[str, Any],
    run_root: Path,
    device_name: str,
    batch_sizes: Sequence[int],
    repeats: int,
    warmup_batches: int,
) -> dict[str, Any]:
    """
    方法作用：在统一固定 Gallery/Query 协议下运行四种细粒度模型的公平时延测试。
    输入参数：config；run_root；device_name；batch_sizes；repeats；warmup_batches。
    返回值：dict，环境、协议、四方法逐折和聚合时延。
    """
    if 1 not in batch_sizes:
        raise ValueError("batch_sizes must contain 1 for online single-image latency")
    device = choose_device(device_name)
    seed_everything(int(config.get("seed", 2026)))
    index = CropIndex(config["paths"]["crop_root"])
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
    else:
        gpu_name = "CPU"
    payload: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "environment": {
            "platform": platform.platform(),
            "device": str(device),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "protocol": {
            "scope": "fine_grained_query_only_after_detection_and_crop",
            "gallery_features_cached": True,
            "model_loading_timed": False,
            "included_stages": [
                "image_read",
                "preprocess",
                "cpu_to_gpu",
                "visual_forward",
                "gallery_top3_prototype_scoring",
            ],
            "excluded_stages": ["yolo_detection", "crop", "model_load", "gallery_encoding"],
            "batch_sizes": list(batch_sizes),
            "workers": 0,
            "warmup_batches": warmup_batches,
            "repeats": repeats,
            "query_count_across_folds": 1081,
        },
        "methods": {},
    }
    for method in METHOD_ORDER:
        fold_rows: list[dict[str, Any]] = []
        print(f"[latency] method={method}", flush=True)
        for fold_number in range(1, int(config["protocol"]["fold_count"]) + 1):
            fold = load_fold(config["paths"]["splits_root"], fold_number)
            class_names = list(fold["novel_classes"])
            class_map = {name: class_id for class_id, name in enumerate(class_names)}
            query_records = records_for_partition(index, fold, "novel_query")
            encoder, transform, masked, model_meta = _load_encoder(
                config, run_root, method, fold, device
            )
            dataset_cls = PatchCropDataset if masked else CropDataset
            dataset = dataset_cls(query_records, transform, class_map)
            bank = _load_feature_bank(run_root, method, fold_number, device)
            amp = True and device.type == "cuda"
            validation_loader = build_eval_loader(dataset, min(32, len(dataset)), workers=0)
            reproduction = _validate_first_batch(
                encoder, validation_loader, bank, device, masked, amp
            )
            row: dict[str, Any] = {
                "fold": fold_number,
                "novel_classes": class_names,
                "feature_reproduction": reproduction,
            }
            expected_rank1 = json.loads(
                (
                    run_root / method / f"fold_{fold_number:02d}" / "metrics.json"
                ).read_text(encoding="utf-8")
            )["rank1_all"]
            for batch_size in batch_sizes:
                loader = build_eval_loader(dataset, int(batch_size), workers=0)
                timing = _benchmark_loader(
                    encoder,
                    loader,
                    bank["support_features"],
                    bank["support_labels"],
                    len(class_names),
                    int(config["protocol"]["prototype_top_k"]),
                    device,
                    masked,
                    amp,
                    warmup_batches,
                    repeats,
                )
                timing["reference_rank1"] = float(expected_rank1)
                timing["rank1_delta"] = timing["rank1_check"] - float(expected_rank1)
                # AMP 下极小的特征舍入差异可能改变低 margin 样本的并列顺序；
                # 首批特征余弦一致性才是推理入口是否正确的主校验。
                if abs(timing["rank1_delta"]) > 0.025:
                    raise RuntimeError(
                        f"Rank-1 differs for {method} fold {fold_number}: "
                        f"{timing['rank1_check']} != {expected_rank1}"
                    )
                row[f"batch_{batch_size}"] = timing
                print(
                    f"[latency] {method} fold={fold_number} batch={batch_size} "
                    f"ms/image={timing['mean_milliseconds_per_image']:.3f}",
                    flush=True,
                )
            fold_rows.append(row)
            del encoder, bank
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload["methods"][method] = {
            "label": METHOD_LABELS[method],
            "model": model_meta,
            "folds": fold_rows,
            "aggregate": {
                f"batch_{batch_size}": _weighted_method_summary(fold_rows, int(batch_size))
                for batch_size in batch_sizes
            },
        }
    actual_queries = payload["methods"]["dino_raw"]["aggregate"]["batch_1"]["query_count"]
    payload["protocol"]["query_count_across_folds"] = actual_queries
    _write_json(run_root / "latency_benchmark.json", payload)
    _write_markdown(run_root, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：构建固定 Gallery 四路时延测试命令行。
    输入参数：None。
    返回值：ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="Benchmark four-way fixed-Gallery latency")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 32))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    方法作用：解析参数、执行四方法时延测试并输出聚合结果。
    输入参数：argv，None 时读取系统命令行。
    返回值：int，成功为 0。
    """
    args = build_parser().parse_args(argv)
    config = load_fourway_config(args.config)
    payload = run_latency_benchmark(
        config,
        args.run_root.expanduser().resolve(),
        args.device,
        args.batch_sizes,
        args.repeats,
        args.warmup_batches,
    )
    compact = {
        method: payload["methods"][method]["aggregate"] for method in METHOD_ORDER
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
