from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_SHARED_MODEL_ROOT = "/home/zq/Downloads/upload/project/object_recognition/models"
DEFAULT_VEHICLE_CLASS_IDS = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck.
LEGACY_CLIP_ALGORITHM = "CLIP"
GALLERY_CLIP_ALGORITHM = "clip_gallery_temporal"
DINO_GALLERY_TEMPORAL_ALGORITHM = "dino_gallery_temporal"
PAD_LITE_B0_ALGORITHM = "pad_lite_b0"
PAD_LITE_B1_ALGORITHM = "pad_lite_b1"
PAD_LITE_B2_ALGORITHM = "pad_lite_b2"
PAD_LITE_ALGORITHMS = frozenset(
    {PAD_LITE_B0_ALGORITHM, PAD_LITE_B1_ALGORITHM, PAD_LITE_B2_ALGORITHM}
)
TEXT_GROUNDING_TEMPORAL_ALGORITHM = "text_grounding_temporal"
GALLERY_ALGORITHMS = frozenset(
    {GALLERY_CLIP_ALGORITHM, DINO_GALLERY_TEMPORAL_ALGORITHM, *PAD_LITE_ALGORITHMS}
)
SUPPORTED_ALGORITHMS = frozenset(
    {
        LEGACY_CLIP_ALGORITHM,
        GALLERY_CLIP_ALGORITHM,
        DINO_GALLERY_TEMPORAL_ALGORITHM,
        *PAD_LITE_ALGORITHMS,
        TEXT_GROUNDING_TEMPORAL_ALGORITHM,
    }
)


def _default_model_root() -> str:
    explicit = os.getenv("TVS_MODEL_ROOT", "").strip()
    if explicit:
        return explicit
    shared_root = Path(DEFAULT_SHARED_MODEL_ROOT)
    if shared_root.exists():
        return str(shared_root)
    return "models"


def _default_yolo_model_dir(model_root: str) -> str:
    explicit = os.getenv("TVS_YOLO_MODEL_DIR", "").strip()
    if explicit:
        return explicit
    return str(Path(model_root) / "yolo")


def _default_detector_model(yolo_model_dir: str) -> str:
    explicit = os.getenv("TVS_DETECTOR_MODEL", "").strip()
    if explicit:
        return explicit

    candidates = [
        Path(yolo_model_dir) / "yolo26n.pt",
        Path(yolo_model_dir) / "yolo11n.pt",
        Path(yolo_model_dir) / "yolov8n.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "yolo26n.pt"


DEFAULT_MODEL_ROOT = _default_model_root()
DEFAULT_YOLO_MODEL_DIR = _default_yolo_model_dir(DEFAULT_MODEL_ROOT)
DEFAULT_DETECTOR_MODEL = _default_detector_model(DEFAULT_YOLO_MODEL_DIR)
DEFAULT_CLIP_MODEL = os.getenv("TVS_CLIP_MODEL", "ViT-B-32")
DEFAULT_CLIP_LOCAL_DIR = os.getenv(
    "TVS_CLIP_LOCAL_DIR",
    str(Path(DEFAULT_MODEL_ROOT) / "clip-vit-base-patch32"),
)
DEFAULT_CLIP_PRETRAINED = os.getenv("TVS_CLIP_PRETRAINED", DEFAULT_CLIP_LOCAL_DIR)
DEFAULT_CLIP_BACKEND = os.getenv("TVS_CLIP_BACKEND", "transformers")
DEFAULT_HIGHLIGHT_SIMILARITY_THRESHOLD = float(
    os.getenv("TVS_HIGHLIGHT_SIMILARITY_THRESHOLD", "0.24")
)
DEFAULT_CLIP_GALLERY_ROOT = os.getenv(
    "TVS_CLIP_GALLERY_ROOT",
    str(Path("Resource") / "clip_gallery"),
)
DEFAULT_DINO_LOCAL_DIR = os.getenv(
    "TVS_DINO_LOCAL_DIR",
    str(Path(DEFAULT_MODEL_ROOT) / "dinov2-small"),
)
DEFAULT_PAD_LITE_CHECKPOINT_ROOT = os.getenv(
    "TVS_PAD_LITE_CHECKPOINT_ROOT",
    str(Path("PAD_Lite") / "outputs" / "final"),
)
DEFAULT_GROUNDING_MODEL_DIR = os.getenv(
    "TVS_GROUNDING_MODEL_DIR",
    str(Path(DEFAULT_MODEL_ROOT) / "grounding-dino-tiny"),
)
DEFAULT_GROUNDING_PROMPT = "tank . Tiger 2 tank . armored vehicle ."


@dataclass(slots=True)
class DetectionConfig:
    """Runtime knobs for the video target search pipeline."""

    detector_model: str = DEFAULT_DETECTOR_MODEL
    detector_imgsz: int = 640
    proposal_conf: float = 0.30
    class_ids: tuple[int, ...] | None = field(
        default_factory=lambda: tuple(DEFAULT_VEHICLE_CLASS_IDS)
    )

    clip_model: str = DEFAULT_CLIP_MODEL
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED
    clip_backend: str = DEFAULT_CLIP_BACKEND
    clip_local_dir: str = DEFAULT_CLIP_LOCAL_DIR
    similarity_threshold: float = 0.24
    highlight_similarity_threshold: float = DEFAULT_HIGHLIGHT_SIMILARITY_THRESHOLD
    text_weight: float = 0.25

    algorithm: str = LEGACY_CLIP_ALGORITHM
    gallery_root: str = DEFAULT_CLIP_GALLERY_ROOT
    gallery_min_similarity: float = 0.50
    gallery_class_margin: float = 0.01
    gallery_background_margin: float = 0.01
    gallery_unknown_margin: float | None = None
    gallery_known_bias: float = 0.02
    gallery_temporal_window: int = 5
    gallery_track_iou: float = 0.30

    target_sample_fps: float = 15.0
    frame_stride: int = 0
    yolo_batch_size: int = 16
    clip_batch_size: int = 128
    dino_local_dir: str = DEFAULT_DINO_LOCAL_DIR
    dino_batch_size: int = 64
    pad_lite_checkpoint_root: str = DEFAULT_PAD_LITE_CHECKPOINT_ROOT
    dino_background_filter_enabled: bool = True
    dino_hysteresis_enabled: bool = True
    dino_hysteresis_max_span_sec: float = 10.0
    dino_hysteresis_anchor_min_frames: int = 3
    dino_hysteresis_anchor_similarity: float = 0.80
    grounding_model_dir: str = DEFAULT_GROUNDING_MODEL_DIR
    grounding_prompt: str = DEFAULT_GROUNDING_PROMPT
    grounding_box_threshold: float = 0.30
    grounding_text_threshold: float = 0.25
    grounding_sample_fps: float = 2.0
    grounding_track_iou: float = 0.30
    grounding_max_gap_samples: int = 2
    box_padding: float = 0.08
    hold_boxes: bool = True
    target_box: tuple[int, int, int, int] | None = None

    device: str = "auto"
    # FP16 is opt-in. Some CUDA devices (confirmed on GTX 1650 with YOLO26)
    # can silently return zero detections in half precision even though FP32
    # inference is correct.
    half: bool = False
    use_nvenc: bool = True
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.gallery_unknown_margin is None:
            self.gallery_unknown_margin = (
                -0.02
                if self.algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
                else 0.01
            )


def parse_class_ids(value: str | Iterable[int] | None) -> tuple[int, ...] | None:
    """Parse UI/CLI class filters."""

    if value is None:
        return tuple(DEFAULT_VEHICLE_CLASS_IDS)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "vehicle", "vehicles", "car", "cars"}:
            return tuple(DEFAULT_VEHICLE_CLASS_IDS)
        if cleaned in {"all", "*", "none"}:
            return None
        parts = [part.strip() for part in cleaned.replace(";", ",").split(",")]
        ids = [int(part) for part in parts if part]
        return tuple(ids) if ids else tuple(DEFAULT_VEHICLE_CLASS_IDS)
    ids = tuple(int(item) for item in value)
    return ids or tuple(DEFAULT_VEHICLE_CLASS_IDS)


def parse_target_box(
    value: str | Iterable[int | float | None] | None,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "null", "auto"}:
            return None
        parts = [part.strip() for part in re.split(r"[\s,;，；]+", cleaned) if part.strip()]
    else:
        parts = list(value)
        if not any(part not in {None, ""} for part in parts):
            return None
    if len(parts) != 4:
        raise ValueError("Target box must contain exactly 4 values: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError("Target box values must be numeric: x1,y1,x2,y2") from exc
    if x1 == x2 or y1 == y2:
        raise ValueError("Target box width and height must be greater than 0")
    return (x1, y1, x2, y2)


def resolve_frame_stride(
    fps: float, target_sample_fps: float, explicit_stride: int = 0
) -> int:
    if explicit_stride and explicit_stride > 0:
        return max(1, int(explicit_stride))
    if not fps or fps <= 0 or target_sample_fps <= 0:
        return 1
    return max(1, round(float(fps) / float(target_sample_fps)))
