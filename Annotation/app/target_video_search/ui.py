from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import (
    DEFAULT_CLIP_BACKEND,
    DEFAULT_CLIP_GALLERY_ROOT,
    DEFAULT_HIGHLIGHT_SIMILARITY_THRESHOLD,
    DEFAULT_CLIP_LOCAL_DIR,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_DINO_LOCAL_DIR,
    DEFAULT_GROUNDING_MODEL_DIR,
    DEFAULT_GROUNDING_PROMPT,
    DINO_GALLERY_TEMPORAL_ALGORITHM,
    DetectionConfig,
    GALLERY_ALGORITHMS,
    GALLERY_CLIP_ALGORITHM,
    LEGACY_CLIP_ALGORITHM,
    PAD_LITE_ALGORITHMS,
    PAD_LITE_B0_ALGORITHM,
    PAD_LITE_B1_ALGORITHM,
    PAD_LITE_B2_ALGORITHM,
    SUPPORTED_ALGORITHMS,
    TEXT_GROUNDING_TEMPORAL_ALGORITHM,
    parse_class_ids,
    parse_target_box,
)
from .algorithms import GalleryResourceError, inspect_gallery_resources
from .dino_embedder import inspect_dino_model
from .grounding_dino import inspect_grounding_model
from .pad_lite_embedder import inspect_pad_lite_model
from .diagnostics import collect_runtime_diagnostics
from .training import (
    YOLO_TRAINING_LOG_DIR,
    get_yolo_training_job,
    get_yolo_model_info,
    inspect_yolo_training_archive,
    latest_yolo_training_log,
    list_yolo_models,
    parse_class_names_input,
    start_yolo_training_job,
)

ERROR_PATTERN = re.compile(
    r"error|exception|traceback|failed|fatal|cannot|not found|no module|"
    r"cuda|nvidia|conda|git-lfs|permission denied|connection|timeout|"
    r"out of memory|oom",
    re.IGNORECASE,
)


def analyze_yolo_training_upload(
    archive_path: str | None,
) -> tuple[Any, str]:
    try:
        info = inspect_yolo_training_archive(archive_path)
    except Exception as exc:
        return gr.update(), f"数据集检查失败：{exc}"
    if info.get("class_labels"):
        return gr.update(value=str(info["class_labels"])), str(info["message"])
    return gr.update(), str(info["message"])


def algorithm_section_open_states(
    selected_algorithm: str,
) -> tuple[bool, bool, bool, bool, bool]:
    return (
        selected_algorithm != TEXT_GROUNDING_TEMPORAL_ALGORITHM,
        selected_algorithm == LEGACY_CLIP_ALGORITHM,
        selected_algorithm == GALLERY_CLIP_ALGORITHM
        or selected_algorithm in PAD_LITE_ALGORITHMS,
        selected_algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM,
        selected_algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM,
    )


def inspect_dino_gallery_resources(
    dino_model_dir: str,
    gallery_root: str,
) -> dict[str, Any]:
    return {
        "model": inspect_dino_model(dino_model_dir),
        "gallery": inspect_gallery_resources(gallery_root),
    }


def format_grounding_model_status(model_dir: str) -> str:
    status = inspect_grounding_model(model_dir)
    if status["ready"]:
        return f"模型可用：{status['root']}"
    missing = ", ".join(status["missing"]) or "模型目录不存在"
    return f"模型不可用：缺少 {missing}"
BENIGN_WARNING_PATTERN = re.compile(
    r"StarletteDeprecationWarning|HTTP_422_UNPROCESSABLE_ENTITY|"
    r"HTTP_422_UNPROCESSABLE_CONTENT|HF_HUB_ENABLE_HF_TRANSFER",
    re.IGNORECASE,
)

def _empty_target_box_state(image_path: str | None = None) -> dict[str, Any]:
    return {
        "image_path": str(image_path or ""),
        "start": None,
        "box": None,
    }


def _coerce_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(round(float(value)))


def _load_target_box_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    path = Path(str(image_path))
    if not path.is_file():
        return None
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _normalize_target_box_for_image(
    image: Image.Image,
    raw_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if raw_box is None:
        return None
    width, height = image.size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [int(v) for v in raw_box]
    left_raw = min(x1, x2)
    top_raw = min(y1, y2)
    right_raw = max(x1, x2)
    bottom_raw = max(y1, y2)
    if right_raw <= 0 or bottom_raw <= 0 or left_raw >= width or top_raw >= height:
        return None
    left = max(0, min(width - 1, left_raw))
    top = max(0, min(height - 1, top_raw))
    right = max(1, min(width, right_raw))
    bottom = max(1, min(height, bottom_raw))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _box_from_clicks(
    image: Image.Image,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    raw_box = (
        min(int(start[0]), int(end[0])),
        min(int(start[1]), int(end[1])),
        max(int(start[0]), int(end[0])) + 1,
        max(int(start[1]), int(end[1])) + 1,
    )
    return _normalize_target_box_for_image(image, raw_box)


def _render_target_box_preview(
    image_path: str | None,
    box: tuple[int, int, int, int] | None = None,
    start: tuple[int, int] | None = None,
) -> np.ndarray | None:
    image = _load_target_box_image(image_path)
    if image is None:
        return None
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    if box is not None:
        left, top, right, bottom = box
        draw.rectangle(
            [(left, top), (max(left, right - 1), max(top, bottom - 1))],
            outline=(255, 64, 64),
            width=4,
        )
    elif start is not None:
        x, y = [int(v) for v in start]
        radius = max(4, round(min(preview.size) * 0.01))
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=(255, 64, 64),
            width=3,
        )
    return np.asarray(preview)


def _target_box_outputs(
    state: dict[str, Any],
    image_path: str | None,
    status: str,
) -> tuple[dict[str, Any], np.ndarray | None, str, int | None, int | None, int | None, int | None]:
    box = state.get("box")
    x1 = y1 = x2 = y2 = None
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
    return (
        state,
        _render_target_box_preview(
            image_path,
            box=box,
            start=state.get("start"),
        ),
        status,
        x1,
        y1,
        x2,
        y2,
    )


def init_target_box_ui(image_path: str | None):
    if not image_path:
        return _target_box_outputs(
            _empty_target_box_state(),
            None,
            "请先上传目标图片，再在图上点击两次设置矩形框。",
        )
    state = _empty_target_box_state(image_path)
    return _target_box_outputs(
        state,
        image_path,
        "可在目标图上点击两次设置矩形框，也可手动填写坐标后点击应用。",
    )


def handle_target_image_select(
    image_path: str | None,
    state: dict[str, Any] | None,
    evt: gr.SelectData,
):
    image = _load_target_box_image(image_path)
    if image is None:
        return _target_box_outputs(
            _empty_target_box_state(),
            None,
            "目标图片不可用，请重新上传。",
        )

    next_state = dict(state or _empty_target_box_state(image_path))
    if next_state.get("image_path") != str(image_path):
        next_state = _empty_target_box_state(image_path)

    index = getattr(evt, "index", None)
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return _target_box_outputs(
            next_state,
            image_path,
            "未读取到点击坐标，请重试。",
        )
    point = (int(index[0]), int(index[1]))

    if next_state.get("start") is None or next_state.get("box") is not None:
        next_state["image_path"] = str(image_path or "")
        next_state["start"] = point
        next_state["box"] = None
        return _target_box_outputs(
            next_state,
            image_path,
            f"已记录第 1 个角点: ({point[0]}, {point[1]})。请点击对角点完成矩形框。",
        )

    start = tuple(next_state["start"])
    box = _box_from_clicks(image, start, point)
    next_state["image_path"] = str(image_path or "")
    next_state["start"] = None
    next_state["box"] = box
    if box is None:
        return _target_box_outputs(
            next_state,
            image_path,
            "目标框无效，请重新点击两个角点。",
        )
    return _target_box_outputs(
        next_state,
        image_path,
        f"目标框已设置: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}。",
    )


def apply_target_box_inputs(
    image_path: str | None,
    x1: Any,
    y1: Any,
    x2: Any,
    y2: Any,
):
    image = _load_target_box_image(image_path)
    state = _empty_target_box_state(image_path)
    if image is None:
        return _target_box_outputs(state, None, "目标图片不可用，请重新上传。")
    try:
        box = parse_target_box([x1, y1, x2, y2])
    except ValueError as exc:
        return _target_box_outputs(
            state,
            image_path,
            f"目标框坐标无效: {exc}",
        )
    if box is None:
        return _target_box_outputs(
            state,
            image_path,
            "未填写目标框坐标，已清除目标框。",
        )
    normalized = _normalize_target_box_for_image(image, box)
    state["box"] = normalized
    if normalized is None:
        return _target_box_outputs(
            state,
            image_path,
            "目标框坐标越界或宽高为 0，请重新调整。",
        )
    return _target_box_outputs(
        state,
        image_path,
        f"已应用目标框: x1={normalized[0]}, y1={normalized[1]}, x2={normalized[2]}, y2={normalized[3]}。",
    )


def clear_target_box(image_path: str | None):
    state = _empty_target_box_state(image_path)
    if not image_path:
        return _target_box_outputs(state, None, "请先上传目标图片。")
    return _target_box_outputs(
        state,
        image_path,
        "已清除目标框。可重新点击两次设置矩形框。",
    )


def _log_dir() -> Path:
    return Path(os.getenv("LOG_DIR", "logs"))


def _read_tail(path: Path, lines: int) -> list[str]:
    if not path.exists() or not path.is_file():
        return [f"日志文件不存在: {path}"]
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return [f"读取日志失败: {path}: {exc}"]
    return content.splitlines()[-max(1, int(lines)) :]


def _filter_benign_warnings(lines: list[str]) -> tuple[list[str], int]:
    visible: list[str] = []
    filtered = 0
    for line in lines:
        if BENIGN_WARNING_PATTERN.search(line):
            filtered += 1
            continue
        visible.append(line)
    return visible, filtered


def _latest_install_log(log_dir: Path) -> Path | None:
    logs = sorted(
        log_dir.glob("install_*.log"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    return logs[0] if logs else None


def _latest_train_log(log_dir: Path) -> Path | None:
    latest = latest_yolo_training_log()
    if latest is not None:
        return latest
    train_log_dir = log_dir / "yolo_training"
    if not train_log_dir.exists():
        return None
    logs = sorted(
        train_log_dir.glob("*.log"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    return logs[0] if logs else None


def _normalize_log_kind(kind: str) -> str:
    value = str(kind or "app").strip().lower()
    if value in {"app", "application"} or "应用" in value:
        return "app"
    if value in {"install", "installer"} or "安装" in value:
        return "install"
    if value in {"train", "training"} or "训练" in value:
        return "train"
    if value == "all" or "全部" in value:
        return "all"
    if value in {"list", "files"} or "列表" in value:
        return "list"
    return value


def _format_log(path: Path, lines: int) -> tuple[str, str, str]:
    raw_tail = _read_tail(path, lines)
    tail, filtered_warning_count = _filter_benign_warnings(raw_tail)
    traceback_start = None
    for index, line in enumerate(tail):
        if "Traceback (most recent call last):" in line:
            traceback_start = index

    if traceback_start is not None:
        traceback_block = tail[traceback_start:]
        summary = ["最近一次 Traceback:"]
        summary.extend(traceback_block[-40:])
        if any(
            "BaseModelOutputWithPooling" in line and "float" in line
            for line in traceback_block
        ):
            summary.append("")
            summary.append(
                "判断: 这是旧版 CLIP 特征提取错误。当前版本已经包含对应修复；"
                "若仍出现该错误，请执行交付包完整性校验并确认没有混用旧文件。"
            )
        if any(
            "Dataset archive must contain data.yaml" in line
            for line in traceback_block
        ):
            summary.append("")
            summary.append(
                "判断: 这是 YOLO 训练数据集目录结构不符合要求。"
            )
            summary.append(
                "支持的自动识别布局包括: images/train + labels/train，"
                "或 train/images + train/labels。"
            )
        return str(path), "\n".join(summary), "\n".join(tail)

    error_lines = [
        f"{index}: {line}"
        for index, line in enumerate(tail, start=1)
        if ERROR_PATTERN.search(line)
    ]
    if not error_lines:
        error_lines = ["未在当前显示范围内匹配到明显错误关键词。"]
    if filtered_warning_count:
        error_lines.insert(
            0,
            f"已过滤 {filtered_warning_count} 条已知无害依赖警告"
            "（Gradio/Starlette 或 HuggingFace 弃用提示）。",
        )
    return str(path), "\n".join(error_lines[-80:]), "\n".join(tail)


def read_frontend_logs(kind: str, lines: int) -> tuple[str, str, str]:
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = max(20, min(int(lines or 160), 2000))
    kind = _normalize_log_kind(kind)

    if kind == "list":
        files = sorted(
            [path for path in log_dir.glob("**/*") if path.is_file()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        listing = "\n".join(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))}"
            f"  {path.stat().st_size} bytes  {path}"
            for path in files
        )
        return str(log_dir), "请选择应用日志或安装日志查看错误原因。", listing or "暂无日志文件。"

    if kind == "install":
        latest = _latest_install_log(log_dir)
        if latest is None:
            return str(log_dir), "未找到安装日志。", "暂无 logs/install_*.log。"
        return _format_log(latest, lines)

    if kind == "train":
        latest = _latest_train_log(log_dir)
        if latest is None:
            return str(log_dir / "yolo_training"), "未找到训练日志。", "暂无 YOLO 训练任务日志。"
        return _format_log(latest, lines)

    if kind == "all":
        sections = []
        error_sections = []
        for label, path in [
            ("应用日志", log_dir / "app.log"),
            ("安装日志", _latest_install_log(log_dir)),
            ("训练日志", _latest_train_log(log_dir)),
        ]:
            if path is None:
                continue
            file_path, errors, tail = _format_log(path, lines)
            error_sections.append(f"## {label}: {file_path}\n{errors}")
            sections.append(f"## {label}: {file_path}\n{tail}")
        return str(log_dir), "\n\n".join(error_sections), "\n\n".join(sections)

    return _format_log(log_dir / "app.log", lines)


def clear_frontend_logs(kind: str) -> tuple[str, str, str]:
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    kind = _normalize_log_kind(kind)
    targets: list[Path] = []
    if kind in {"app", "all"}:
        targets.append(log_dir / "app.log")
    if kind in {"install", "all"}:
        targets.extend(log_dir.glob("install_*.log"))
    if kind in {"train", "all"}:
        train_log_dirs = [YOLO_TRAINING_LOG_DIR, log_dir / "yolo_training"]
        seen_dirs: set[Path] = set()
        for train_log_dir in train_log_dirs:
            resolved = train_log_dir.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            targets.extend(train_log_dir.glob("*.log"))
            targets.extend(train_log_dir.glob("*.json"))

    if not targets:
        return str(log_dir), "当前日志类型不支持清空。", ""

    cleared = []
    for path in targets:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            cleared.append(str(path))
        except Exception as exc:
            cleared.append(f"{path}: 清空失败: {exc}")

    return str(log_dir), "已清空:\n" + "\n".join(cleared), ""


def refresh_yolo_models(current_model: str | None = None):
    import gradio as gr

    choices = list_yolo_models()
    value = current_model if current_model in choices else choices[0]
    return gr.update(choices=choices, value=value)


def refresh_two_yolo_model_dropdowns(
    detector_current: str | None = None,
    train_current: str | None = None,
):
    return refresh_yolo_models(detector_current), refresh_yolo_models(train_current)


def _model_class_map_json(model_info: dict[str, Any]) -> str:
    return json.dumps(model_info.get("class_id_map") or {}, ensure_ascii=False, indent=2)


def get_detector_model_metadata_outputs(model_path: str | None):
    model_info = get_yolo_model_info(model_path)
    return (
        str(model_info.get("recommended_class_ids") or ""),
        str(model_info.get("hint_text") or ""),
        _model_class_map_json(model_info),
    )


def refresh_detector_model_bundle(current_model: str | None = None):
    import gradio as gr

    choices = list_yolo_models()
    value = current_model if current_model in choices else choices[0]
    recommended_ids, hint_text, class_map_json = get_detector_model_metadata_outputs(value)
    return (
        gr.update(choices=choices, value=value),
        recommended_ids,
        hint_text,
        class_map_json,
    )


def refresh_detector_train_and_metadata(
    detector_current: str | None = None,
    train_current: str | None = None,
):
    detector_update, recommended_ids, hint_text, class_map_json = (
        refresh_detector_model_bundle(detector_current)
    )
    train_update = refresh_yolo_models(train_current)
    return detector_update, recommended_ids, hint_text, class_map_json, train_update


def apply_recommended_class_ids(recommended_ids: str | None) -> str:
    return str(recommended_ids or "").strip()


def _read_training_log_tail(log_path: str | None, lines: int = 80) -> str:
    if not log_path:
        return "暂无训练日志。"
    return "\n".join(_read_tail(Path(log_path), lines))


def _should_apply_saved_training_model(
    job: dict[str, Any],
    applied_job_id: str | None,
) -> bool:
    return (
        str(job.get("status") or "") == "done"
        and bool(str(job.get("saved_model") or "").strip())
        and bool(str(job.get("job_id") or "").strip())
        and str(job.get("job_id") or "") != str(applied_job_id or "")
    )


def _training_job_outputs(
    job: dict[str, Any],
    applied_job_id: str | None = None,
):
    import gradio as gr

    status = str(job.get("status") or "idle")
    job_id = str(job.get("job_id") or "")
    progress_percent = int(job.get("progress_percent") or 0)
    message = str(job.get("message") or "")
    log_path = str(job.get("log_path") or "")
    saved_model = str(job.get("saved_model") or "")
    summary = json.dumps(job, ensure_ascii=False, indent=2)
    status_text = f"{status}: {message}" if message else status
    log_tail = _read_training_log_tail(log_path)

    should_apply_model = _should_apply_saved_training_model(job, applied_job_id)
    if status == "done" and saved_model:
        choices = list_yolo_models()
        if should_apply_model:
            model_info = get_yolo_model_info(saved_model)
            detector_update = gr.update(choices=choices, value=saved_model)
            base_update = gr.update(choices=choices, value=saved_model)
            class_ids_update = gr.update(
                value=str(model_info.get("recommended_class_ids") or "all")
            )
            recommended_ids_update = gr.update(
                value=str(model_info.get("recommended_class_ids") or "")
            )
            hint_update = gr.update(value=str(model_info.get("hint_text") or ""))
            class_map_update = gr.update(value=_model_class_map_json(model_info))
        else:
            detector_update = gr.update(choices=choices)
            base_update = gr.update(choices=choices)
            class_ids_update = gr.update()
            recommended_ids_update = gr.update()
            hint_update = gr.update()
            class_map_update = gr.update()
    else:
        detector_update = gr.update()
        base_update = gr.update()
        class_ids_update = gr.update()
        recommended_ids_update = gr.update()
        hint_update = gr.update()
        class_map_update = gr.update()

    next_applied_job_id = job_id if should_apply_model else str(applied_job_id or "")

    return (
        job_id,
        summary,
        status_text,
        progress_percent,
        log_path,
        log_tail,
        detector_update,
        class_ids_update,
        recommended_ids_update,
        hint_update,
        class_map_update,
        base_update,
        next_applied_job_id,
    )


def _build_config(
    algorithm: str,
    detector_model: str,
    detector_imgsz: int,
    proposal_conf: float,
    class_ids: str,
    target_box_x1: Any,
    target_box_y1: Any,
    target_box_x2: Any,
    target_box_y2: Any,
    clip_model: str,
    clip_pretrained: str,
    clip_backend: str,
    clip_local_dir: str,
    similarity_threshold: float,
    highlight_similarity_threshold: float,
    text_weight: float,
    gallery_root: str,
    gallery_min_similarity: float,
    gallery_class_margin: float,
    gallery_negative_margin: float,
    gallery_known_bias: float,
    gallery_temporal_window: int,
    gallery_track_iou: float,
    dino_local_dir: str,
    dino_batch_size: int,
    dino_gallery_root: str,
    dino_gallery_min_similarity: float,
    dino_gallery_class_margin: float,
    dino_gallery_background_margin: float,
    dino_gallery_unknown_margin: float,
    dino_gallery_known_bias: float,
    dino_gallery_temporal_window: int,
    dino_gallery_track_iou: float,
    dino_background_filter_enabled: bool,
    dino_hysteresis_enabled: bool,
    dino_hysteresis_max_span_sec: float,
    dino_hysteresis_anchor_min_frames: int,
    dino_hysteresis_anchor_similarity: float,
    grounding_model_dir: str,
    grounding_prompt: str,
    grounding_box_threshold: float,
    grounding_text_threshold: float,
    grounding_sample_fps: float,
    grounding_track_iou: float,
    grounding_max_gap_samples: int,
    target_sample_fps: float,
    frame_stride: int,
    yolo_batch_size: int,
    clip_batch_size: int,
    hold_boxes: bool,
    use_half: bool,
    use_nvenc: bool,
    output_dir: str,
) -> DetectionConfig:
    resolved_clip_local_dir = clip_local_dir.strip() or DEFAULT_CLIP_LOCAL_DIR
    resolved_clip_pretrained = clip_pretrained.strip() or DEFAULT_CLIP_PRETRAINED
    resolved_clip_backend = clip_backend.strip() or DEFAULT_CLIP_BACKEND
    if (
        resolved_clip_backend == "open_clip"
        and os.getenv("ALLOW_OPEN_CLIP", "0") != "1"
    ):
        resolved_clip_backend = "transformers"
        resolved_clip_pretrained = resolved_clip_local_dir
    target_box = (
        parse_target_box([target_box_x1, target_box_y1, target_box_x2, target_box_y2])
        if algorithm == LEGACY_CLIP_ALGORITHM
        else None
    )
    dino_selected = algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
    selected_gallery_root = dino_gallery_root if dino_selected else gallery_root
    selected_min_similarity = (
        dino_gallery_min_similarity if dino_selected else gallery_min_similarity
    )
    selected_class_margin = (
        dino_gallery_class_margin if dino_selected else gallery_class_margin
    )
    selected_background_margin = (
        dino_gallery_background_margin if dino_selected else gallery_negative_margin
    )
    selected_unknown_margin = (
        dino_gallery_unknown_margin if dino_selected else gallery_negative_margin
    )
    selected_known_bias = dino_gallery_known_bias if dino_selected else gallery_known_bias
    selected_temporal_window = (
        dino_gallery_temporal_window if dino_selected else gallery_temporal_window
    )
    selected_track_iou = dino_gallery_track_iou if dino_selected else gallery_track_iou

    return DetectionConfig(
        algorithm=algorithm,
        detector_model=detector_model.strip() or DEFAULT_DETECTOR_MODEL,
        detector_imgsz=int(detector_imgsz),
        proposal_conf=float(proposal_conf),
        class_ids=parse_class_ids(class_ids),
        target_box=target_box,
        clip_model=clip_model.strip() or DEFAULT_CLIP_MODEL,
        clip_pretrained=resolved_clip_pretrained,
        clip_backend=resolved_clip_backend,
        clip_local_dir=resolved_clip_local_dir,
        similarity_threshold=float(similarity_threshold),
        highlight_similarity_threshold=float(highlight_similarity_threshold),
        text_weight=float(text_weight),
        gallery_root=selected_gallery_root.strip() or DEFAULT_CLIP_GALLERY_ROOT,
        gallery_min_similarity=float(selected_min_similarity),
        gallery_class_margin=float(selected_class_margin),
        gallery_background_margin=float(selected_background_margin),
        gallery_unknown_margin=float(selected_unknown_margin),
        gallery_known_bias=float(selected_known_bias),
        gallery_temporal_window=max(1, int(selected_temporal_window)),
        gallery_track_iou=float(selected_track_iou),
        target_sample_fps=float(target_sample_fps),
        frame_stride=int(frame_stride),
        yolo_batch_size=int(yolo_batch_size),
        clip_batch_size=int(clip_batch_size),
        dino_local_dir=dino_local_dir.strip() or DEFAULT_DINO_LOCAL_DIR,
        dino_batch_size=max(1, int(dino_batch_size)),
        dino_background_filter_enabled=bool(dino_background_filter_enabled),
        dino_hysteresis_enabled=bool(dino_hysteresis_enabled),
        dino_hysteresis_max_span_sec=max(0.0, float(dino_hysteresis_max_span_sec)),
        dino_hysteresis_anchor_min_frames=max(
            1, int(dino_hysteresis_anchor_min_frames)
        ),
        dino_hysteresis_anchor_similarity=float(
            dino_hysteresis_anchor_similarity
        ),
        grounding_model_dir=grounding_model_dir.strip() or DEFAULT_GROUNDING_MODEL_DIR,
        grounding_prompt=grounding_prompt.strip(),
        grounding_box_threshold=float(grounding_box_threshold),
        grounding_text_threshold=float(grounding_text_threshold),
        grounding_sample_fps=max(0.1, float(grounding_sample_fps)),
        grounding_track_iou=float(grounding_track_iou),
        grounding_max_gap_samples=max(0, int(grounding_max_gap_samples)),
        hold_boxes=bool(hold_boxes),
        half=bool(use_half),
        use_nvenc=bool(use_nvenc),
        output_dir=Path(output_dir or "outputs"),
    )


def build_demo():
    import gradio as gr

    def update_algorithm_sections(selected_algorithm: str):
        yolo_visible, legacy_open, clip_gallery_open, dino_gallery_open, grounding_visible = (
            algorithm_section_open_states(selected_algorithm)
        )
        return (
            gr.update(visible=yolo_visible, open=True),
            gr.update(open=legacy_open),
            gr.update(open=clip_gallery_open),
            gr.update(open=dino_gallery_open),
            gr.update(visible=grounding_visible, open=True),
        )

    yolo_model_choices = list_yolo_models()
    initial_detector_model = (
        DEFAULT_DETECTOR_MODEL
        if DEFAULT_DETECTOR_MODEL in yolo_model_choices
        else yolo_model_choices[0]
    )
    initial_model_recommended_ids, initial_model_hint, initial_model_class_map = (
        get_detector_model_metadata_outputs(initial_detector_model)
    )

    def run_detection(
        video_path: str | None,
        detection_algorithm: str,
        target_image_path: str | None,
        target_box_x1: Any,
        target_box_y1: Any,
        target_box_x2: Any,
        target_box_y2: Any,
        description: str,
        detector_model: str,
        detector_imgsz: int,
        proposal_conf: float,
        class_ids: str,
        clip_model: str,
        clip_pretrained: str,
        clip_backend: str,
        clip_local_dir: str,
        similarity_threshold: float,
        highlight_similarity_threshold: float,
        text_weight: float,
        gallery_root: str,
        gallery_min_similarity: float,
        gallery_class_margin: float,
        gallery_negative_margin: float,
        gallery_known_bias: float,
        gallery_temporal_window: int,
        gallery_track_iou: float,
        dino_local_dir: str,
        dino_batch_size: int,
        dino_gallery_root: str,
        dino_gallery_min_similarity: float,
        dino_gallery_class_margin: float,
        dino_gallery_background_margin: float,
        dino_gallery_unknown_margin: float,
        dino_gallery_known_bias: float,
        dino_gallery_temporal_window: int,
        dino_gallery_track_iou: float,
        dino_background_filter_enabled: bool,
        dino_hysteresis_enabled: bool,
        dino_hysteresis_max_span_sec: float,
        dino_hysteresis_anchor_min_frames: int,
        dino_hysteresis_anchor_similarity: float,
        grounding_model_dir: str,
        grounding_prompt: str,
        grounding_box_threshold: float,
        grounding_text_threshold: float,
        grounding_sample_fps: float,
        grounding_track_iou: float,
        grounding_max_gap_samples: int,
        target_sample_fps: float,
        frame_stride: int,
        yolo_batch_size: int,
        clip_batch_size: int,
        hold_boxes: bool,
        use_half: bool,
        use_nvenc: bool,
        output_dir: str,
        progress=gr.Progress(track_tqdm=False),
    ) -> tuple[str, str, str]:
        if not video_path:
            raise gr.Error("请先上传输入视频。")
        if detection_algorithm not in SUPPORTED_ALGORITHMS:
            raise gr.Error(f"不支持的识别算法: {detection_algorithm}")
        if detection_algorithm == LEGACY_CLIP_ALGORITHM and not target_image_path:
            raise gr.Error("请先上传目标图片。")
        if detection_algorithm in GALLERY_ALGORITHMS:
            selected_gallery_root = (
                dino_gallery_root
                if detection_algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
                else gallery_root
            )
            gallery_info = inspect_gallery_resources(selected_gallery_root)
            if not gallery_info["ready"]:
                raise gr.Error(
                    f"{detection_algorithm} 资源不完整: "
                    + ", ".join(gallery_info["missing"])
                )
        if detection_algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM:
            dino_info = inspect_dino_model(dino_local_dir)
            if not dino_info["ready"]:
                raise gr.Error(
                    "dino_gallery_temporal 模型资源不完整: "
                    + ", ".join(dino_info["missing"])
                )
        if detection_algorithm in PAD_LITE_ALGORITHMS:
            pad_info = inspect_pad_lite_model(
                DetectionConfig().pad_lite_checkpoint_root,
                detection_algorithm,
            )
            if not pad_info["ready"]:
                raise gr.Error(
                    f"{detection_algorithm} Final模型资源不完整: "
                    + ", ".join(pad_info["missing"])
                )
        if detection_algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            if not str(grounding_prompt or "").strip():
                raise gr.Error("text_grounding_temporal 必须提供非空文本描述。")
            grounding_info = inspect_grounding_model(grounding_model_dir)
            if not grounding_info["ready"]:
                raise gr.Error(
                    "Grounding DINO 模型资源不完整: "
                    + ", ".join(grounding_info["missing"])
                )

        try:
            config = _build_config(
                algorithm=detection_algorithm,
                detector_model=detector_model,
                detector_imgsz=detector_imgsz,
                proposal_conf=proposal_conf,
                class_ids=class_ids,
                target_box_x1=target_box_x1,
                target_box_y1=target_box_y1,
                target_box_x2=target_box_x2,
                target_box_y2=target_box_y2,
                clip_model=clip_model,
                clip_pretrained=clip_pretrained,
                clip_backend=clip_backend,
                clip_local_dir=clip_local_dir,
                similarity_threshold=similarity_threshold,
                highlight_similarity_threshold=highlight_similarity_threshold,
                text_weight=text_weight,
                gallery_root=gallery_root,
                gallery_min_similarity=gallery_min_similarity,
                gallery_class_margin=gallery_class_margin,
                gallery_negative_margin=gallery_negative_margin,
                gallery_known_bias=gallery_known_bias,
                gallery_temporal_window=gallery_temporal_window,
                gallery_track_iou=gallery_track_iou,
                dino_local_dir=dino_local_dir,
                dino_batch_size=dino_batch_size,
                dino_gallery_root=dino_gallery_root,
                dino_gallery_min_similarity=dino_gallery_min_similarity,
                dino_gallery_class_margin=dino_gallery_class_margin,
                dino_gallery_background_margin=dino_gallery_background_margin,
                dino_gallery_unknown_margin=dino_gallery_unknown_margin,
                dino_gallery_known_bias=dino_gallery_known_bias,
                dino_gallery_temporal_window=dino_gallery_temporal_window,
                dino_gallery_track_iou=dino_gallery_track_iou,
                dino_background_filter_enabled=dino_background_filter_enabled,
                dino_hysteresis_enabled=dino_hysteresis_enabled,
                dino_hysteresis_max_span_sec=dino_hysteresis_max_span_sec,
                dino_hysteresis_anchor_min_frames=dino_hysteresis_anchor_min_frames,
                dino_hysteresis_anchor_similarity=dino_hysteresis_anchor_similarity,
                grounding_model_dir=grounding_model_dir,
                grounding_prompt=grounding_prompt,
                grounding_box_threshold=grounding_box_threshold,
                grounding_text_threshold=grounding_text_threshold,
                grounding_sample_fps=grounding_sample_fps,
                grounding_track_iou=grounding_track_iou,
                grounding_max_gap_samples=grounding_max_gap_samples,
                target_sample_fps=target_sample_fps,
                frame_stride=frame_stride,
                yolo_batch_size=yolo_batch_size,
                clip_batch_size=clip_batch_size,
                hold_boxes=hold_boxes,
                use_half=use_half,
                use_nvenc=use_nvenc,
                output_dir=output_dir,
            )
        except ValueError as exc:
            raise gr.Error(f"目标框配置无效: {exc}") from exc
        out_dir = Path(output_dir or "outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        from .pipeline import TargetVideoAnnotator, build_output_path

        output_path = build_output_path(out_dir, detection_algorithm)

        def on_progress(frame_count: int, total_frames: int, message: str) -> None:
            if total_frames:
                progress(min(frame_count / total_frames, 1.0), desc=message)
            else:
                progress(0.0, desc=message)

        try:
            annotator = TargetVideoAnnotator(config)
            summary = annotator.process_video(
                video_path=video_path,
                target_image_path=target_image_path,
                output_path=output_path,
                description=description,
                progress_callback=on_progress,
            )
        except (GalleryResourceError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        compact: dict[str, Any] = {
            key: summary[key]
            for key in [
                "output_video",
                "output_json",
                "target_box",
                "total_frames",
                "fps",
                "duration_sec",
                "sample_stride",
                "sampled_frames",
                "matched_sample_frames",
                "matched_boxes",
                "known_boxes",
                "unknown_boxes",
                "rejected_boxes",
                "low_confidence_boxes",
                "temporal_propagated_boxes",
                "processing_sec",
                "processing_fps",
                "video_seconds_per_processing_second",
            ]
            if key in summary
        }
        compact["config"] = summary.get("config", {})
        compact["gallery_info"] = summary.get("gallery_info")
        compact["algorithm"] = detection_algorithm
        return (
            summary["output_video"],
            json.dumps(compact, ensure_ascii=False, indent=2),
            summary["output_json"],
        )

    def start_yolo_training(
        dataset_archive: str | None,
        base_model: str,
        run_name: str,
        class_labels: str,
        epochs: int,
        train_imgsz: int,
        train_batch: int,
        patience: int,
        workers: int,
        train_device: str,
        applied_job_id: str | None,
    ):
        if not dataset_archive:
            raise gr.Error("请上传 YOLO 数据集 zip 或 rar。")

        try:
            job = start_yolo_training_job(
                archive_path=dataset_archive,
                base_model=base_model,
                run_name=run_name,
                class_names=parse_class_names_input(class_labels),
                epochs=int(epochs),
                imgsz=int(train_imgsz),
                batch=int(train_batch),
                patience=int(patience),
                workers=int(workers),
                device=train_device,
            )
        except Exception as exc:
            error = {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            return (
                "",
                json.dumps(error, ensure_ascii=False, indent=2),
                f"训练失败：{exc}",
                0,
                "",
                "",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                str(applied_job_id or ""),
            )
        return _training_job_outputs(job, applied_job_id)

    def refresh_yolo_training(
        job_id: str | None,
        applied_job_id: str | None,
    ):
        job = get_yolo_training_job(job_id)
        return _training_job_outputs(job, applied_job_id)

    with gr.Blocks(title="目标视频检测") as demo:
        gr.Markdown("## 目标视频检测")
        target_box_state = gr.State(_empty_target_box_state())

        gr.Markdown("### 1. 视频与候选检测")
        with gr.Row():
            with gr.Column(scale=7):
                video = gr.Video(
                    label="输入视频",
                    sources=["upload"],
                    height=420,
                )
                output_video = gr.Video(label="标注后视频", height=420)
            with gr.Column(scale=5):
                detection_algorithm = gr.Radio(
                    label="识别算法",
                    choices=[
                        LEGACY_CLIP_ALGORITHM,
                        GALLERY_CLIP_ALGORITHM,
                        DINO_GALLERY_TEMPORAL_ALGORITHM,
                        PAD_LITE_B0_ALGORITHM,
                        PAD_LITE_B1_ALGORITHM,
                        PAD_LITE_B2_ALGORITHM,
                        TEXT_GROUNDING_TEMPORAL_ALGORITHM,
                    ],
                    value=LEGACY_CLIP_ALGORITHM,
                )
                with gr.Accordion(
                    "YOLO 候选设置", open=True
                ) as yolo_candidate_settings:
                    with gr.Row():
                        detector_model = gr.Dropdown(
                            label="YOLO 模型/引擎",
                            choices=yolo_model_choices,
                            value=initial_detector_model,
                            allow_custom_value=True,
                        )
                        refresh_yolo_models_button = gr.Button("刷新模型")
                    with gr.Row():
                        detector_model_recommended_ids = gr.Textbox(
                            label="当前模型推荐类别 ID",
                            value=initial_model_recommended_ids,
                            interactive=False,
                        )
                        apply_recommended_ids_button = gr.Button("填入推荐类别 ID")
                    detector_model_class_hint = gr.Textbox(
                        label="当前模型类别提示",
                        value=initial_model_hint,
                        lines=3,
                        interactive=False,
                    )
                    with gr.Accordion("当前模型类别映射", open=False):
                        detector_model_class_map = gr.Code(
                            label="类别 ID 与名称",
                            value=initial_model_class_map,
                            language="json",
                        )
                    with gr.Row():
                        detector_imgsz = gr.Slider(
                            label="检测输入尺寸",
                            minimum=320,
                            maximum=1280,
                            step=32,
                            value=640,
                        )
                        proposal_conf = gr.Slider(
                            label="候选框置信度",
                            minimum=0.05,
                            maximum=0.80,
                            step=0.01,
                            value=0.30,
                        )
                    with gr.Row():
                        class_ids = gr.Textbox(label="候选类别 ID", value="2,3,5,7")
                        yolo_batch_size = gr.Slider(
                            label="YOLO 批大小",
                            minimum=1,
                            maximum=64,
                            step=1,
                            value=16,
                        )
                    use_half = gr.Checkbox(
                        label="使用 FP16 半精度推理（仅在已验证兼容的 GPU 上启用）",
                        value=False,
                    )
                with gr.Accordion(
                    "Text Grounding 设置", open=True, visible=False
                ) as text_grounding_temporal_settings:
                    grounding_prompt = gr.Textbox(
                        label="文本描述",
                        value=DEFAULT_GROUNDING_PROMPT,
                        lines=2,
                        info="使用英文名词短语，并用句点分隔不同目标。",
                    )
                    with gr.Row():
                        grounding_model_dir = gr.Textbox(
                            label="Grounding DINO 本地目录",
                            value=DEFAULT_GROUNDING_MODEL_DIR,
                            scale=4,
                        )
                        refresh_grounding_model_button = gr.Button(
                            "刷新模型状态", scale=1
                        )
                    grounding_model_status = gr.Textbox(
                        label="模型状态",
                        value=format_grounding_model_status(
                            DEFAULT_GROUNDING_MODEL_DIR
                        ),
                        interactive=False,
                        lines=1,
                    )
                    with gr.Row():
                        grounding_box_threshold = gr.Slider(
                            label="Box 阈值",
                            minimum=0.05,
                            maximum=0.90,
                            step=0.01,
                            value=0.30,
                        )
                        grounding_text_threshold = gr.Slider(
                            label="Text 阈值",
                            minimum=0.05,
                            maximum=0.90,
                            step=0.01,
                            value=0.25,
                        )
                        grounding_sample_fps = gr.Slider(
                            label="采样 FPS",
                            minimum=0.5,
                            maximum=10.0,
                            step=0.5,
                            value=2.0,
                        )
                    with gr.Row():
                        grounding_track_iou = gr.Slider(
                            label="跟踪 IoU",
                            minimum=0.05,
                            maximum=0.90,
                            step=0.05,
                            value=0.30,
                        )
                        grounding_max_gap_samples = gr.Slider(
                            label="最大漏检采样数",
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=2,
                        )

        with gr.Row():
            with gr.Column(scale=7):
                summary = gr.Code(label="检测统计", language="json")
            with gr.Column(scale=5):
                json_file = gr.File(label="JSON 结果")

        gr.HTML(
            '<div style="height:1px;margin:1.25rem 0;'
            'background:linear-gradient(90deg,transparent,#707070,transparent)"></div>'
        )
        gr.Markdown("### 2. 目标匹配设置")
        with gr.Accordion("CLIP 目标图片与参数", open=True) as legacy_clip_settings:
            gr.Markdown("原 CLIP 使用单张目标图片、目标框和可选描述进行匹配。")
            with gr.Row():
                with gr.Column(scale=6):
                    target_image = gr.Image(label="目标图片", type="filepath")
                    target_box_preview = gr.Image(
                        label="目标框预览",
                        interactive=False,
                    )
                with gr.Column(scale=5):
                    target_box_status = gr.Textbox(
                        label="目标框状态",
                        value="请先上传目标图片，再在图上点击两次设置矩形框。",
                        interactive=False,
                    )
                    with gr.Row():
                        target_box_x1 = gr.Number(label="x1", value=None, precision=0)
                        target_box_y1 = gr.Number(label="y1", value=None, precision=0)
                        target_box_x2 = gr.Number(label="x2", value=None, precision=0)
                        target_box_y2 = gr.Number(label="y2", value=None, precision=0)
                    with gr.Row():
                        apply_target_box_button = gr.Button("应用目标框坐标")
                        clear_target_box_button = gr.Button("清除目标框")
                    description = gr.Textbox(
                        label="可选描述",
                        placeholder="例如：IS-2 heavy tank",
                        lines=2,
                    )
                    highlight_similarity_threshold = gr.Slider(
                        label="红框高亮阈值",
                        minimum=0.05,
                        maximum=0.95,
                        step=0.01,
                        value=DEFAULT_HIGHLIGHT_SIMILARITY_THRESHOLD,
                    )

            with gr.Row():
                clip_model = gr.Textbox(label="CLIP 模型", value=DEFAULT_CLIP_MODEL)
                clip_pretrained = gr.Textbox(
                    label="CLIP 权重或本地目录", value=DEFAULT_CLIP_PRETRAINED
                )
            with gr.Row():
                clip_backend = gr.Dropdown(
                    label="CLIP 后端",
                    choices=["open_clip", "transformers"],
                    value=DEFAULT_CLIP_BACKEND,
                )
                clip_local_dir = gr.Textbox(
                    label="CLIP 本地目录",
                    value=DEFAULT_CLIP_LOCAL_DIR,
                )
            with gr.Row():
                similarity_threshold = gr.Slider(
                    label="相似度阈值",
                    minimum=0.05,
                    maximum=0.60,
                    step=0.01,
                    value=0.24,
                )
                text_weight = gr.Slider(
                    label="描述权重",
                    minimum=0.0,
                    maximum=0.8,
                    step=0.05,
                    value=0.25,
                )
                clip_batch_size = gr.Slider(
                    label="CLIP 批大小",
                    minimum=8,
                    maximum=512,
                    step=8,
                    value=128,
                )
                hold_boxes = gr.Checkbox(label="采样帧之间保持框", value=True)

        with gr.Accordion(
            "clip_gallery_temporal 参数", open=False
        ) as gallery_temporal_settings:
            with gr.Row():
                gallery_root = gr.Textbox(
                    label="Gallery 目录",
                    value=DEFAULT_CLIP_GALLERY_ROOT,
                )
                refresh_gallery_button = gr.Button("刷新原型库")
            with gr.Row():
                gallery_min_similarity = gr.Slider(
                    label="最低型号相似度",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.50,
                )
                gallery_class_margin = gr.Slider(
                    label="第一二名间隔",
                    minimum=0.0,
                    maximum=0.20,
                    step=0.005,
                    value=0.01,
                )
                gallery_negative_margin = gr.Slider(
                    label="负样本间隔",
                    minimum=0.0,
                    maximum=0.20,
                    step=0.005,
                    value=0.01,
                )
                gallery_known_bias = gr.Slider(
                    label="已知型号确认偏置",
                    minimum=0.0,
                    maximum=0.10,
                    step=0.005,
                    value=0.02,
                )
            with gr.Row():
                gallery_temporal_window = gr.Slider(
                    label="时序窗口",
                    minimum=1,
                    maximum=15,
                    step=1,
                    value=5,
                )
                gallery_track_iou = gr.Slider(
                    label="跟踪 IoU",
                    minimum=0.05,
                    maximum=0.90,
                    step=0.05,
                    value=0.30,
                )
            gallery_status = gr.JSON(
                label="原型库状态",
                value=inspect_gallery_resources(DEFAULT_CLIP_GALLERY_ROOT),
            )

        with gr.Accordion(
            "dino_gallery_temporal 参数", open=False
        ) as dino_gallery_temporal_settings:
            with gr.Row():
                dino_local_dir = gr.Textbox(
                    label="DINO 本地模型目录",
                    value=DEFAULT_DINO_LOCAL_DIR,
                )
                dino_batch_size = gr.Slider(
                    label="DINO 批大小",
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=64,
                )
            with gr.Row():
                dino_gallery_root = gr.Textbox(
                    label="Gallery 目录",
                    value=DEFAULT_CLIP_GALLERY_ROOT,
                )
                refresh_dino_gallery_button = gr.Button("刷新 DINO 与原型库")
            with gr.Row():
                dino_gallery_min_similarity = gr.Slider(
                    label="最低型号相似度",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.00,
                )
                dino_gallery_class_margin = gr.Slider(
                    label="低置信界线（不触发 Unknown）",
                    minimum=0.0,
                    maximum=0.20,
                    step=0.005,
                    value=0.01,
                )
                dino_gallery_known_bias = gr.Slider(
                    label="已知型号确认偏置",
                    minimum=0.0,
                    maximum=0.10,
                    step=0.005,
                    value=0.00,
                )
            with gr.Row():
                dino_gallery_background_margin = gr.Slider(
                    label="背景拒绝间隔",
                    minimum=0.0,
                    maximum=0.20,
                    step=0.005,
                    value=0.01,
                )
                dino_gallery_unknown_margin = gr.Slider(
                    label="Unknown 判定间隔",
                    minimum=-0.20,
                    maximum=0.20,
                    step=0.005,
                    value=-0.02,
                )
            with gr.Row():
                dino_gallery_temporal_window = gr.Slider(
                    label="局部分数平均窗口",
                    minimum=1,
                    maximum=15,
                    step=1,
                    value=5,
                )
                dino_gallery_track_iou = gr.Slider(
                    label="跟踪 IoU",
                    minimum=0.05,
                    maximum=0.90,
                    step=0.05,
                    value=0.30,
                )
            with gr.Row():
                dino_background_filter_enabled = gr.Checkbox(
                    label="启用 DINO 背景过滤",
                    value=True,
                    info="关闭后不再用 negatives 拒绝 YOLO 候选；适合模糊视频，但可能增加误检。",
                )
                dino_hysteresis_enabled = gr.Checkbox(
                    label="启用 Temporal Hysteresis（双向时序传播）",
                    value=True,
                    info="关闭后使用单遍 DINO 分类；类别间隔不足只标记 low，不判为 Unknown。",
                )
            with gr.Row():
                dino_hysteresis_max_span_sec = gr.Slider(
                    label="时序传播最大跨度（秒）",
                    minimum=1,
                    maximum=30,
                    step=1,
                    value=10,
                )
                dino_hysteresis_anchor_min_frames = gr.Slider(
                    label="锚点连续帧数",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=3,
                )
                dino_hysteresis_anchor_similarity = gr.Slider(
                    label="锚点图像相似度",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.80,
                )
            dino_gallery_status = gr.JSON(
                label="DINO 模型与原型库状态",
                value=inspect_dino_gallery_resources(
                    DEFAULT_DINO_LOCAL_DIR,
                    DEFAULT_CLIP_GALLERY_ROOT,
                ),
            )

        with gr.Accordion("视频处理与输出", open=False):
            with gr.Row():
                target_sample_fps = gr.Slider(
                    label="采样 FPS", minimum=1, maximum=30, step=1, value=15
                )
                frame_stride = gr.Number(
                    label="固定跳帧间隔", value=0, precision=0
                )
                use_nvenc = gr.Checkbox(label="使用 FFmpeg NVENC 写视频", value=True)
            output_dir = gr.Textbox(label="输出目录", value="outputs")

        run_button = gr.Button("开始检测", variant="primary")

        with gr.Accordion("YOLO 新类别训练", open=False):
            dataset_archive = gr.File(
                label="YOLO 数据集压缩包",
                file_types=[".zip", ".rar"],
                type="filepath",
            )
            train_dataset_status = gr.Textbox(
                label="数据集识别状态",
                value="等待上传训练数据集。",
                interactive=False,
                lines=2,
            )
            with gr.Row():
                train_base_model = gr.Dropdown(
                    label="基础 YOLO 模型",
                    choices=yolo_model_choices,
                    value=DEFAULT_DETECTOR_MODEL if DEFAULT_DETECTOR_MODEL in yolo_model_choices else yolo_model_choices[0],
                    allow_custom_value=True,
                )
                train_run_name = gr.Textbox(label="训练名称", value="custom_yolo")
                refresh_train_models_button = gr.Button("刷新基础模型")
            train_class_labels = gr.Textbox(
                label="新类别标签",
                placeholder="例如: tank, armored_vehicle 或按行输入；tank_only_train 将自动填入 tank",
                lines=3,
            )
            with gr.Row():
                train_epochs = gr.Slider(label="Epochs", minimum=1, maximum=300, step=1, value=50)
                train_imgsz = gr.Slider(label="训练尺寸", minimum=320, maximum=1280, step=32, value=640)
                train_batch = gr.Number(label="Batch", value=8, precision=0)
            with gr.Row():
                train_patience = gr.Number(label="Patience", value=20, precision=0)
                train_workers = gr.Number(label="Workers", value=8, precision=0)
                train_device = gr.Textbox(label="训练设备", value="0")
            with gr.Row():
                train_button = gr.Button("开始训练并保存模型", variant="primary")
                refresh_train_status_button = gr.Button("刷新训练状态")
            train_job_id = gr.Textbox(
                label="训练任务 ID",
                value="",
                interactive=False,
            )
            train_model_applied_job_id = gr.State("")
            with gr.Row():
                train_status = gr.Textbox(
                    label="训练进程状态",
                    value="等待上传 YOLO 数据集",
                    interactive=False,
                )
                train_progress = gr.Slider(
                    label="训练进度",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0,
                    interactive=False,
                )
            train_log_path = gr.Textbox(label="训练日志文件", interactive=False)
            train_log_tail = gr.Textbox(label="训练日志", lines=10, interactive=False)
            train_summary = gr.Code(label="训练结果", language="json")

        with gr.Accordion("日志与错误原因", open=False):
            with gr.Row():
                log_kind = gr.Dropdown(
                    label="日志类型",
                    choices=[
                        ("应用日志", "app"),
                        ("安装日志", "install"),
                        ("训练日志", "train"),
                        ("全部日志", "all"),
                        ("日志列表", "list"),
                    ],
                    value="app",
                )
                log_lines = gr.Slider(
                    label="显示行数", minimum=20, maximum=1000, step=20, value=160
                )
                refresh_logs = gr.Button("刷新日志")
                clear_logs = gr.Button("清空日志")
            log_file = gr.Textbox(label="日志文件", interactive=False)
            log_errors = gr.Textbox(label="可能的错误原因", lines=8, interactive=False)
            log_tail = gr.Textbox(label="尾部日志", lines=18, interactive=False)

        with gr.Accordion("系统与设备测试", open=False):
            gr.Markdown(
                "读取容器、GPU、运行库和外部资源状态；不会加载模型或执行视频识别。"
            )
            refresh_runtime_info = gr.Button("刷新环境信息")
            runtime_info = gr.JSON(label="当前运行环境")

        inputs = [
            video,
            detection_algorithm,
            target_image,
            target_box_x1,
            target_box_y1,
            target_box_x2,
            target_box_y2,
            description,
            detector_model,
            detector_imgsz,
            proposal_conf,
            class_ids,
            clip_model,
            clip_pretrained,
            clip_backend,
            clip_local_dir,
            similarity_threshold,
            highlight_similarity_threshold,
            text_weight,
            gallery_root,
            gallery_min_similarity,
            gallery_class_margin,
            gallery_negative_margin,
            gallery_known_bias,
            gallery_temporal_window,
            gallery_track_iou,
            dino_local_dir,
            dino_batch_size,
            dino_gallery_root,
            dino_gallery_min_similarity,
            dino_gallery_class_margin,
            dino_gallery_background_margin,
            dino_gallery_unknown_margin,
            dino_gallery_known_bias,
            dino_gallery_temporal_window,
            dino_gallery_track_iou,
            dino_background_filter_enabled,
            dino_hysteresis_enabled,
            dino_hysteresis_max_span_sec,
            dino_hysteresis_anchor_min_frames,
            dino_hysteresis_anchor_similarity,
            grounding_model_dir,
            grounding_prompt,
            grounding_box_threshold,
            grounding_text_threshold,
            grounding_sample_fps,
            grounding_track_iou,
            grounding_max_gap_samples,
            target_sample_fps,
            frame_stride,
            yolo_batch_size,
            clip_batch_size,
            hold_boxes,
            use_half,
            use_nvenc,
            output_dir,
        ]
        run_button.click(run_detection, inputs=inputs, outputs=[output_video, summary, json_file])
        refresh_gallery_button.click(
            inspect_gallery_resources,
            inputs=[gallery_root],
            outputs=[gallery_status],
        )
        refresh_dino_gallery_button.click(
            inspect_dino_gallery_resources,
            inputs=[dino_local_dir, dino_gallery_root],
            outputs=[dino_gallery_status],
        )
        refresh_grounding_model_button.click(
            format_grounding_model_status,
            inputs=[grounding_model_dir],
            outputs=[grounding_model_status],
        )
        detection_algorithm.change(
            update_algorithm_sections,
            inputs=[detection_algorithm],
            outputs=[
                yolo_candidate_settings,
                legacy_clip_settings,
                gallery_temporal_settings,
                dino_gallery_temporal_settings,
                text_grounding_temporal_settings,
            ],
        )
        target_box_outputs = [
            target_box_state,
            target_box_preview,
            target_box_status,
            target_box_x1,
            target_box_y1,
            target_box_x2,
            target_box_y2,
        ]
        target_image.change(
            init_target_box_ui,
            inputs=[target_image],
            outputs=target_box_outputs,
        )
        target_image.select(
            handle_target_image_select,
            inputs=[target_image, target_box_state],
            outputs=target_box_outputs,
        )
        apply_target_box_button.click(
            apply_target_box_inputs,
            inputs=[
                target_image,
                target_box_x1,
                target_box_y1,
                target_box_x2,
                target_box_y2,
            ],
            outputs=target_box_outputs,
        )
        clear_target_box_button.click(
            clear_target_box,
            inputs=[target_image],
            outputs=target_box_outputs,
        )
        refresh_yolo_models_button.click(
            refresh_detector_model_bundle,
            inputs=[detector_model],
            outputs=[
                detector_model,
                detector_model_recommended_ids,
                detector_model_class_hint,
                detector_model_class_map,
            ],
        )
        detector_model.change(
            get_detector_model_metadata_outputs,
            inputs=[detector_model],
            outputs=[
                detector_model_recommended_ids,
                detector_model_class_hint,
                detector_model_class_map,
            ],
        )
        apply_recommended_ids_button.click(
            apply_recommended_class_ids,
            inputs=[detector_model_recommended_ids],
            outputs=[class_ids],
        )
        refresh_train_models_button.click(
            refresh_detector_train_and_metadata,
            inputs=[detector_model, train_base_model],
            outputs=[
                detector_model,
                detector_model_recommended_ids,
                detector_model_class_hint,
                detector_model_class_map,
                train_base_model,
            ],
        )
        training_outputs = [
            train_job_id,
            train_summary,
            train_status,
            train_progress,
            train_log_path,
            train_log_tail,
            detector_model,
            class_ids,
            detector_model_recommended_ids,
            detector_model_class_hint,
            detector_model_class_map,
            train_base_model,
            train_model_applied_job_id,
        ]
        dataset_archive.change(
            analyze_yolo_training_upload,
            inputs=[dataset_archive],
            outputs=[train_class_labels, train_dataset_status],
        )
        train_button.click(
            start_yolo_training,
            inputs=[
                dataset_archive,
                train_base_model,
                train_run_name,
                train_class_labels,
                train_epochs,
                train_imgsz,
                train_batch,
                train_patience,
                train_workers,
                train_device,
                train_model_applied_job_id,
            ],
            outputs=training_outputs,
        )
        refresh_train_status_button.click(
            refresh_yolo_training,
            inputs=[train_job_id, train_model_applied_job_id],
            outputs=training_outputs,
        )
        if hasattr(gr, "Timer"):
            try:
                training_timer = gr.Timer(value=3.0, active=True)
            except TypeError:
                training_timer = gr.Timer(3.0)
            training_timer.tick(
                refresh_yolo_training,
                inputs=[train_job_id, train_model_applied_job_id],
                outputs=training_outputs,
            )
        refresh_logs.click(
            read_frontend_logs,
            inputs=[log_kind, log_lines],
            outputs=[log_file, log_errors, log_tail],
        )
        clear_logs.click(
            clear_frontend_logs,
            inputs=[log_kind],
            outputs=[log_file, log_errors, log_tail],
        )
        refresh_runtime_info.click(
            collect_runtime_diagnostics,
            inputs=[],
            outputs=[runtime_info],
        )

    return demo
