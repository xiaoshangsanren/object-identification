from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import PAD_LITE_ROOT
from .data import CropIndex, load_fold, records_for_partition
from .dino_patch_engine import PatchLetterboxTransform


DEFAULT_EXPERIMENT_ROOT = (
    PAD_LITE_ROOT
    / "outputs"
    / "03_p2b_patch_reranking"
    / "p2b_three_train_two_test_4way_v1"
)
INDEX_FIELDS = (
    "episode",
    "sample_id",
    "source_image",
    "file",
    "tiny",
    "crowded",
    "clipped",
    "p0_predicted_class",
    "p0_correct",
    "p2a_predicted_class",
    "p2a_correct",
    "p2b_predicted_class",
    "p2b_correct",
    "gallery_reference_sample_id",
    "gallery_reference_source_image",
    "gallery_reference_class",
    "gallery_reference_same_class",
    "gallery_reference_cosine_similarity",
    "max_patch_weight",
)


def _read_json(path: Path) -> Any:
    """
    方法作用：
        读取 UTF-8 JSON 文件并检查存在性。
    输入参数：
        path (Path)：JSON 路径。
    返回值：
        Any：反序列化内容。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """
    方法作用：
        写入缩进 UTF-8 JSON。
    输入参数：
        path：目标路径；payload：可序列化对象。
    返回值：
        None：无返回数据。
    """
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    """
    方法作用：
        加载指定字号的 DejaVu 字体，不存在时回退 PIL 默认字体。
    输入参数：
        size (int)：字号。
    返回值：
        ImageFont.ImageFont：可用于面板绘制的字体。
    """
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """
    方法作用：
        将预测列表转换为 sample_id 唯一索引。
    输入参数：
        path (Path)：predictions.json。
    返回值：
        dict[str,dict]：Nq 个 sample_id 到预测记录的映射。
    """
    rows = _read_json(path)
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate prediction sample_id in {path}: {sample_id}")
        result[sample_id] = row
    return result


def _status_label(method: str, prediction: dict[str, Any]) -> tuple[str, bool]:
    """
    方法作用：
        生成某分支预测类别与正误状态的面板文本。
    输入参数：
        method (str)：分支名；prediction：单个 Query 预测记录。
    返回值：
        tuple[str,bool]：显示文本和是否正确。
    """
    correct = bool(prediction["correct"])
    state = "CORRECT" if correct else "WRONG"
    return f"{method}: {prediction['predicted_class']}  {state}", correct


def _top1_gallery_indices(
    support_features: np.ndarray,
    query_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    方法作用：
        在全局 Support 中为每个 Query 找到余弦相似度最高的参考图。
    输入参数：
        support_features (np.ndarray)：支持特征 [Ns,D]；query_features：查询特征 [Nq,D]。
    返回值：
        tuple[np.ndarray,np.ndarray]：Top-1 Support 下标 [Nq] 与余弦相似度 [Nq]。
    """

    support = np.asarray(support_features, dtype=np.float32)
    query = np.asarray(query_features, dtype=np.float32)
    if support.ndim != 2 or query.ndim != 2:
        raise ValueError("Support and query features must both be rank-2 arrays")
    if support.shape[0] == 0 or query.shape[0] == 0:
        raise ValueError("Support and query feature arrays must not be empty")
    if support.shape[1] != query.shape[1]:
        raise ValueError("Support and query feature dimensions differ")
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    if np.any(support_norm <= 0.0) or np.any(query_norm <= 0.0):
        raise ValueError("Cannot compute cosine similarity for zero-norm features")
    similarities = (query / query_norm) @ (support / support_norm).T
    indices = similarities.argmax(axis=1)
    scores = similarities[np.arange(len(query)), indices]
    return indices.astype(np.int64, copy=False), scores.astype(np.float32, copy=False)


def _attention_overlay(
    base: Image.Image,
    weights: np.ndarray,
    image_size: int,
    patch_size: int,
) -> tuple[Image.Image, float]:
    """
    方法作用：
        把一维 Patch 权重还原为网格、插值并以 turbo 颜色叠加到 Query 图像。
    输入参数：
        base：S×S RGB 图；weights (np.ndarray)：Patch 权重 [N]；
        image_size：S；patch_size：P，其中 N=(S/P)^2。
    返回值：
        tuple[Image.Image,float]：S×S 热力叠加图和该图最大 Patch 权重。
    """
    from matplotlib import colormaps

    grid = image_size // patch_size
    if weights.shape != (grid * grid,):
        raise ValueError(
            f"Expected {grid * grid} patch weights, got shape {weights.shape}"
        )
    heat = weights.reshape(grid, grid)
    maximum = float(heat.max(initial=0.0))
    normalized = heat / maximum if maximum > 0.0 else heat
    heat_image = Image.fromarray(normalized.astype(np.float32), mode="F").resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    heat_full = np.asarray(heat_image, dtype=np.float32)
    color = colormaps["turbo"](heat_full)[..., :3] * 255.0
    base_array = np.asarray(base, dtype=np.float32)
    alpha = (0.72 * heat_full)[..., None]
    overlay = np.clip(
        base_array * (1.0 - alpha) + color * alpha,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB"), maximum


def _annotated_panel(
    gallery_reference: Image.Image,
    overlay: Image.Image,
    episode: int,
    sample_id: str,
    true_class: str,
    metadata: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    gallery_metadata: dict[str, Any],
) -> Image.Image:
    """
    方法作用：
        组合左侧 Gallery Top-1、右侧 Query 热力图及 P0/P2a/P2b 预测标注。
    输入参数：
        gallery_reference、overlay：两个 S×S RGB 图；episode/sample_id/true_class：身份；
        metadata：质量标记；predictions：三个分支预测；gallery_metadata：参考图信息。
    返回值：
        Image.Image：尺寸 [2S, S+132, 3] 的可视化面板。
    """
    image_size = gallery_reference.width
    if (
        gallery_reference.size != (image_size, image_size)
        or overlay.size != gallery_reference.size
    ):
        raise ValueError("Gallery reference and overlay panels must be equal square images")
    width = image_size * 2
    header_height = 132
    panel = Image.new("RGB", (width, header_height + image_size), (20, 24, 29))
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(17)
    body_font = _load_font(14)
    small_font = _load_font(12)

    quality = " ".join(
        (
            f"tiny={str(bool(metadata['tiny'])).lower()}",
            f"crowded={str(bool(metadata['crowded'])).lower()}",
            f"clipped={str(bool(metadata['clipped'])).lower()}",
        )
    )
    draw.text(
        (10, 7),
        f"Episode {episode:02d} | {sample_id} | TRUE: {true_class}",
        font=title_font,
        fill=(245, 245, 245),
    )
    draw.text((10, 31), quality, font=small_font, fill=(190, 198, 208))

    badge_width = width // 3
    for index, method in enumerate(("P0", "P2a", "P2b")):
        label, correct = _status_label(method, predictions[method])
        x0 = index * badge_width + 4
        x1 = (index + 1) * badge_width - 4
        color = (30, 112, 65) if correct else (158, 42, 43)
        draw.rounded_rectangle((x0, 50, x1, 78), radius=5, fill=color)
        draw.text((x0 + 7, 56), label, font=body_font, fill=(255, 255, 255))

    gallery_state = "SAME CLASS" if gallery_metadata["same_class"] else "DIFFERENT CLASS"
    gallery_color = (98, 220, 145) if gallery_metadata["same_class"] else (255, 132, 132)
    draw.text(
        (10, 89),
        f"P2b TOP-1 GALLERY: {gallery_metadata['sample_id']}",
        font=small_font,
        fill=(220, 225, 230),
    )
    draw.text(
        (10, 106),
        f"CLASS: {gallery_metadata['class_name']}  |  COS: {gallery_metadata['similarity']:.4f}  |  {gallery_state}",
        font=small_font,
        fill=gallery_color,
    )
    draw.text(
        (image_size + 10, 89),
        "QUERY P2a PATCH ATTENTION",
        font=small_font,
        fill=(220, 225, 230),
    )
    panel.paste(gallery_reference, (0, header_height))
    panel.paste(overlay, (image_size, header_height))
    return panel


def _episode_numbers(experiment_root: Path, target_class: str) -> list[int]:
    """
    方法作用：
        查找包含目标类别且已有 P2a 注意力缓存的 Episode。
    输入参数：
        experiment_root：协议结果根目录；target_class：目标类名。
    返回值：
        list[int]：匹配的 Episode 编号。
    """
    episodes = []
    for path in sorted((experiment_root / "p2a").glob("fold_*/query_attention.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if target_class in payload["class_names"]:
            episodes.append(int(path.parent.name.split("_")[-1]))
    if not episodes:
        raise ValueError(f"No episodes contain target class {target_class!r}")
    return episodes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    方法作用：
        以固定字段和 UTF-8 BOM 写入热力图索引 CSV。
    输入参数：
        path：CSV 路径；rows：长度 Nq 的索引记录。
    返回值：
        None：无返回数据。
    """
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(
    path: Path,
    target_class: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """
    方法作用：
        生成按 Episode 分组、可浏览全部面板的 HTML 索引页。
    输入参数：
        path：HTML 路径；target_class：类名；rows：面板索引；summary：统计摘要。
    返回值：
        None：写入 HTML 后无返回数据。
    """
    cards = []
    current_episode = None
    for row in rows:
        episode = int(row["episode"])
        if episode != current_episode:
            if current_episode is not None:
                cards.append("</div>")
            episode_summary = summary["episodes"][f"{episode:02d}"]
            cards.append(
                f"<h2>Episode {episode:02d} — "
                f"P0 {episode_summary['p0_correct']}/{episode_summary['count']}, "
                f"P2a {episode_summary['p2a_correct']}/{episode_summary['count']}, "
                f"P2b {episode_summary['p2b_correct']}/{episode_summary['count']}, "
                f"Gallery Top-1 same class "
                f"{episode_summary['gallery_top1_same_class']}/{episode_summary['count']}</h2>"
                "<div class='grid'>"
            )
            current_episode = episode
        p2b_class = "correct" if row["p2b_correct"] else "wrong"
        cards.append(
            f"<figure class='{p2b_class}'>"
            f"<a href='{html.escape(str(row['file']))}'>"
            f"<img loading='lazy' src='{html.escape(str(row['file']))}'></a>"
            f"<figcaption>{html.escape(str(row['sample_id']))}</figcaption>"
            "</figure>"
        )
    if current_episode is not None:
        cards.append("</div>")
    total = summary["total"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(target_class)} attention heatmaps</title>
<style>
body {{ background:#11151a; color:#eef2f6; font:15px sans-serif; margin:24px; }}
h1,h2 {{ margin:20px 0 10px; }}
.note {{ color:#b8c1cc; max-width:1100px; line-height:1.5; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:12px; }}
figure {{ margin:0; padding:5px; background:#20262d; border:3px solid #8d2f32; }}
figure.correct {{ border-color:#247a4b; }}
img {{ display:block; width:100%; height:auto; }}
figcaption {{ padding:5px 2px 1px; color:#d9e0e8; }}
</style>
</head>
<body>
<h1>{html.escape(target_class)} — Gallery Top-1 and P2a attention heatmaps</h1>
<p class="note">The left side is the globally nearest Gallery support image under P2b fused-feature cosine similarity; the right side is the current Query with its P2a Patch-Scorer heatmap. Green/red badges indicate whether P0, P2a, and P2b predicted the true class. P2b classification uses Top-3 class-prototype aggregation, so its predicted class can differ from the single nearest Gallery image class. Card border follows P2b correctness. Heat intensity is normalized independently per image and must not be compared as an absolute score across images.</p>
<p>Total {total['count']} heatmaps; P0 {total['p0_correct']}/{total['count']}, P2a {total['p2a_correct']}/{total['count']}, P2b {total['p2b_correct']}/{total['count']}; Gallery Top-1 same class {total['gallery_top1_same_class']}/{total['count']}.</p>
{''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def export_heatmaps(
    experiment_root: Path,
    output_root: Path,
    target_class: str,
) -> dict[str, Any]:
    """
    方法作用：
        为目标类别导出所有相关 Episode 的 Gallery 参考图+P2a 热力图面板及索引。
    输入参数：
        experiment_root：P0/P2a/P2b 结果根目录；output_root：新输出目录；
        target_class：要导出的类别。
        主缓存含 Query 权重 [Nq,N]、P2b Support [Ns,D]、Query [Nq,D]。
    返回值：
        dict[str,Any]：逐 Episode 与总计准确数、可视化参数和输出路径。
    """
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing heatmap directory: {output_root}"
        )
    output_root.mkdir(parents=True)
    episodes = _episode_numbers(experiment_root, target_class)
    first_config = _read_json(
        experiment_root / "p2a" / f"fold_{episodes[0]:02d}" / "run_config.json"
    )
    config = first_config["config"]
    crop_index = CropIndex(Path(config["paths"]["crop_root"]))
    records_by_id = {record.sample_id: record for record in crop_index.records}
    image_size = int(config["data"]["image_size"])
    patch_size = int(config["model"]["patch_size"])
    transform = PatchLetterboxTransform(image_size, patch_size, train=False)
    splits_root = Path(config["paths"]["splits_root"])
    exclude_tiny_support = bool(config["data"].get("exclude_tiny_support", True))

    rows: list[dict[str, Any]] = []
    episode_stats: dict[str, dict[str, int]] = {}
    total_stats: defaultdict[str, int] = defaultdict(int)
    for episode in episodes:
        p2a_dir = experiment_root / "p2a" / f"fold_{episode:02d}"
        attention = torch.load(
            p2a_dir / "query_attention.pt",
            map_location="cpu",
            weights_only=False,
        )
        class_names = list(attention["class_names"])
        class_id = class_names.index(target_class)
        labels = attention["query_labels"].numpy()
        selected = np.flatnonzero(labels == class_id)
        sample_ids = list(attention["sample_ids"])
        weights = attention["weights"].numpy().astype(np.float32, copy=False)
        masks = attention["valid_patch_masks"].numpy().astype(bool, copy=False)
        fold = load_fold(splits_root, episode)
        support_records = records_for_partition(
            crop_index,
            fold,
            "novel_support",
            exclude_tiny=exclude_tiny_support,
        )
        p2b_features = torch.load(
            experiment_root / "p2b" / f"fold_{episode:02d}" / "features.pt",
            map_location="cpu",
            weights_only=False,
        )
        support_features = p2b_features["support_features"].numpy()
        query_features = p2b_features["query_features"].numpy()
        support_labels = p2b_features["support_labels"].numpy()
        query_feature_labels = p2b_features["query_labels"].numpy()
        if list(p2b_features["class_names"]) != class_names:
            raise ValueError(f"P2b class order differs from P2a attention in episode {episode}")
        if len(support_records) != len(support_features):
            raise ValueError(f"Support records and P2b features differ in episode {episode}")
        if len(sample_ids) != len(query_features) or not np.array_equal(
            labels, query_feature_labels
        ):
            raise ValueError(f"Query attention and P2b features differ in episode {episode}")
        expected_support_labels = np.asarray(
            [class_names.index(record.class_name) for record in support_records],
            dtype=np.int64,
        )
        if not np.array_equal(expected_support_labels, support_labels):
            raise ValueError(f"Support record order differs from P2b features in episode {episode}")
        gallery_indices, gallery_similarities = _top1_gallery_indices(
            support_features,
            query_features,
        )
        prediction_maps = {
            "P0": _predictions_by_id(
                experiment_root
                / "p0"
                / "dino"
                / "b2"
                / f"fold_{episode:02d}"
                / "predictions.json"
            ),
            "P2a": _predictions_by_id(p2a_dir / "predictions.json"),
            "P2b": _predictions_by_id(
                experiment_root
                / "p2b"
                / f"fold_{episode:02d}"
                / "predictions.json"
            ),
        }
        episode_dir = output_root / f"episode_{episode:02d}"
        episode_dir.mkdir()
        stats: defaultdict[str, int] = defaultdict(int)
        for query_index in selected:
            sample_id = sample_ids[int(query_index)]
            record = records_by_id.get(sample_id)
            if record is None:
                raise KeyError(f"Missing crop record for {sample_id}")
            if record.class_name != target_class:
                raise ValueError(f"Attention label and crop class disagree for {sample_id}")
            predictions = {
                method: mapping[sample_id]
                for method, mapping in prediction_maps.items()
            }
            if any(item["true_class"] != target_class for item in predictions.values()):
                raise ValueError(f"Prediction true class mismatch for {sample_id}")
            with Image.open(record.path) as source:
                query_crop, geometry_mask = transform._letterbox(source.convert("RGB"))
            if not np.array_equal(
                geometry_mask.numpy(), masks[int(query_index)]
            ):
                raise ValueError(f"Stored and reconstructed masks differ for {sample_id}")
            overlay, max_patch_weight = _attention_overlay(
                query_crop,
                weights[int(query_index)],
                image_size,
                patch_size,
            )
            gallery_index = int(gallery_indices[int(query_index)])
            gallery_record = support_records[gallery_index]
            gallery_similarity = float(gallery_similarities[int(query_index)])
            gallery_same_class = gallery_record.class_name == target_class
            with Image.open(gallery_record.path) as source:
                gallery_reference, _ = transform._letterbox(source.convert("RGB"))
            panel = _annotated_panel(
                gallery_reference,
                overlay,
                episode,
                sample_id,
                target_class,
                {
                    "tiny": record.tiny,
                    "crowded": record.crowded,
                    "clipped": record.clipped,
                },
                predictions,
                {
                    "sample_id": gallery_record.sample_id,
                    "class_name": gallery_record.class_name,
                    "similarity": gallery_similarity,
                    "same_class": gallery_same_class,
                },
            )
            p2b_correct = bool(predictions["P2b"]["correct"])
            state = "correct" if p2b_correct else "wrong"
            filename = f"{sample_id}__p2b_{state}.png"
            relative_file = Path(f"episode_{episode:02d}") / filename
            panel.save(output_root / relative_file, format="PNG", optimize=True)
            row = {
                "episode": episode,
                "sample_id": sample_id,
                "source_image": record.source_image,
                "file": relative_file.as_posix(),
                "tiny": record.tiny,
                "crowded": record.crowded,
                "clipped": record.clipped,
                "p0_predicted_class": predictions["P0"]["predicted_class"],
                "p0_correct": bool(predictions["P0"]["correct"]),
                "p2a_predicted_class": predictions["P2a"]["predicted_class"],
                "p2a_correct": bool(predictions["P2a"]["correct"]),
                "p2b_predicted_class": predictions["P2b"]["predicted_class"],
                "p2b_correct": p2b_correct,
                "gallery_reference_sample_id": gallery_record.sample_id,
                "gallery_reference_source_image": gallery_record.source_image,
                "gallery_reference_class": gallery_record.class_name,
                "gallery_reference_same_class": gallery_same_class,
                "gallery_reference_cosine_similarity": gallery_similarity,
                "max_patch_weight": max_patch_weight,
            }
            rows.append(row)
            stats["count"] += 1
            total_stats["count"] += 1
            if gallery_same_class:
                stats["gallery_top1_same_class"] += 1
                total_stats["gallery_top1_same_class"] += 1
            for method, key in (("P0", "p0"), ("P2a", "p2a"), ("P2b", "p2b")):
                if predictions[method]["correct"]:
                    stats[f"{key}_correct"] += 1
                    total_stats[f"{key}_correct"] += 1
        episode_stats[f"{episode:02d}"] = dict(stats)

    rows.sort(key=lambda row: (int(row["episode"]), str(row["sample_id"])))
    summary = {
        "format": "pad_lite_class_attention_heatmaps_gallery_reference_v2",
        "target_class": target_class,
        "experiment_root": str(experiment_root),
        "output_root": str(output_root),
        "episodes": episode_stats,
        "total": dict(total_stats),
        "visualization": {
            "source": "P2a query_attention.pt",
            "input": f"{image_size}x{image_size} letterbox crop",
            "patch_grid": f"{image_size // patch_size}x{image_size // patch_size}",
            "colormap": "turbo",
            "per_image_max_normalization": True,
            "overlay_max_alpha": 0.72,
            "left_panel": "global P2b fused-feature cosine Top-1 Gallery support image",
            "right_panel": "P2a Patch attention overlay on the current Query image",
            "classification_note": "P2b class prediction uses Top-3 prototype aggregation and may differ from the single Gallery Top-1 class",
        },
    }
    _write_json(output_root / "index.json", rows)
    _write_csv(output_root / "index.csv", rows)
    _write_json(output_root / "summary.json", summary)
    _write_html(output_root / "index.html", target_class, rows, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        构建类别热力图导出 CLI。
    输入参数：
        无。
    返回值：
        argparse.ArgumentParser：类名、实验根目录和输出目录解析器。
    """
    parser = argparse.ArgumentParser(
        description="Export all cached P2a attention heatmaps for one query class."
    )
    parser.add_argument("--class-name", default="t-72")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析导出参数、确定默认输出目录并执行热力图导出。
    输入参数：
        argv (list[str]|None)：显式命令行；None 时读取系统命令行。
    返回值：
        int：成功状态码 0。
    """
    args = build_parser().parse_args(argv)
    experiment_root = args.experiment_root.expanduser().resolve()
    target_class = str(args.class_name)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else experiment_root / f"{target_class}_attention_heatmaps_gallery_reference"
    )
    summary = export_heatmaps(experiment_root, output_root, target_class)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
