from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..config import (
    DINO_GALLERY_TEMPORAL_ALGORITHM,
    DetectionConfig,
    GALLERY_CLIP_ALGORITHM,
)
from .base import MatchBatchResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class GalleryResourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GalleryClass:
    class_id: str
    folder: str
    label: str
    aliases: tuple[str, ...]


@dataclass(slots=True)
class _Track:
    track_id: int
    last_box: np.ndarray
    last_frame_index: int
    class_history: deque[np.ndarray]
    background_history: deque[float]
    unknown_history: deque[float]
    observations: int = 0


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_gallery_classes(root: str | Path) -> list[GalleryClass]:
    root = Path(root)
    manifest_path = root / "classes.json"
    if not manifest_path.is_file():
        raise GalleryResourceError(f"缺少类别文件: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GalleryResourceError(f"无法读取类别文件 {manifest_path}: {exc}") from exc
    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise GalleryResourceError("classes.json 必须包含非空 classes 数组。")

    classes: list[GalleryClass] = []
    seen_ids: set[str] = set()
    seen_folders: set[str] = set()
    for index, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise GalleryResourceError(f"classes[{index}] 必须是对象。")
        class_id = str(item.get("id", "")).strip()
        folder = str(item.get("folder", "")).strip()
        label = str(item.get("label", "")).strip()
        aliases = item.get("aliases", [])
        if not class_id or not folder or not label:
            raise GalleryResourceError(
                f"classes[{index}] 必须包含非空 id、folder 和 label。"
            )
        if class_id in seen_ids or folder in seen_folders:
            raise GalleryResourceError(f"类别 id 或目录重复: {class_id}/{folder}")
        if not isinstance(aliases, list):
            raise GalleryResourceError(f"类别 {class_id} 的 aliases 必须是数组。")
        seen_ids.add(class_id)
        seen_folders.add(folder)
        classes.append(
            GalleryClass(
                class_id=class_id,
                folder=folder,
                label=label,
                aliases=tuple(str(value) for value in aliases),
            )
        )
    return classes


def inspect_gallery_resources(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    info: dict[str, Any] = {
        "root": str(root_path),
        "ready": False,
        "class_count": 0,
        "prototype_counts": {},
        "negative_count": 0,
        "unknown_count": 0,
        "missing": [],
    }
    try:
        classes = load_gallery_classes(root_path)
    except GalleryResourceError as exc:
        info["missing"] = [str(exc)]
        return info

    info["class_count"] = len(classes)
    missing: list[str] = []
    prototype_counts: dict[str, int] = {}
    for definition in classes:
        count = len(_image_paths(root_path / "prototypes" / definition.folder))
        prototype_counts[definition.class_id] = count
        if count == 0:
            missing.append(f"prototypes/{definition.folder}")
    info["prototype_counts"] = prototype_counts

    negative_count = len(_image_paths(root_path / "negatives"))
    unknown_count = len(_image_paths(root_path / "unknown_tanks"))
    info["negative_count"] = negative_count
    info["unknown_count"] = unknown_count
    if negative_count == 0:
        missing.append("negatives")
    if unknown_count == 0:
        missing.append("unknown_tanks")
    info["missing"] = missing
    info["ready"] = not missing
    return info


def top_k_class_scores(
    candidate_features: np.ndarray,
    class_features: list[np.ndarray],
    top_k: int = 3,
) -> np.ndarray:
    if candidate_features.ndim != 2:
        raise ValueError("candidate_features must have shape [N,D]")
    columns: list[np.ndarray] = []
    for prototypes in class_features:
        if prototypes.ndim != 2 or prototypes.shape[0] == 0:
            raise ValueError("each class must contain at least one prototype")
        similarities = candidate_features @ prototypes.T
        k = min(max(1, int(top_k)), similarities.shape[1])
        selected = np.partition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
        columns.append(selected.mean(axis=1))
    return np.stack(columns, axis=1)


def top_k_bank_scores(
    candidate_features: np.ndarray,
    bank_features: np.ndarray,
    top_k: int = 3,
) -> np.ndarray:
    """Score a reference bank with the same Top-K mean used by known classes."""
    if bank_features.ndim != 2 or bank_features.shape[0] == 0:
        return np.full(candidate_features.shape[0], -1.0, dtype=np.float32)
    similarities = candidate_features @ bank_features.T
    k = min(max(1, int(top_k)), similarities.shape[1])
    selected = np.partition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
    return selected.mean(axis=1)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def decide_gallery_candidate(
    class_ids: list[str],
    class_labels: list[str],
    class_scores: np.ndarray,
    background_score: float,
    unknown_score: float,
    min_similarity: float,
    class_margin_threshold: float,
    background_margin_threshold: float,
    unknown_margin_threshold: float,
    known_bias: float = 0.0,
    class_margin_as_unknown: bool = True,
    background_filter_enabled: bool = True,
) -> dict[str, Any]:
    order = np.argsort(-class_scores)
    best_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else best_index
    best_score = float(class_scores[best_index])
    second_score = float(class_scores[second_index]) if len(order) > 1 else -1.0
    class_margin = best_score - second_score
    effective_known_score = best_score + float(known_bias)
    background_margin = effective_known_score - float(background_score)
    unknown_margin = effective_known_score - float(unknown_score)

    status = "known"
    subtype_id = class_ids[best_index]
    subtype_label = class_labels[best_index]
    reason = "known"
    if (
        background_filter_enabled
        and background_margin < float(background_margin_threshold)
    ):
        status = "rejected_background"
        subtype_id = None
        subtype_label = "Background"
        reason = "background_margin"
    elif best_score < float(min_similarity):
        status = "unknown"
        subtype_id = "Unknown"
        subtype_label = "Unknown"
        reason = "min_similarity"
    elif class_margin_as_unknown and class_margin < float(class_margin_threshold):
        status = "unknown"
        subtype_id = "Unknown"
        subtype_label = "Unknown"
        reason = "class_margin"
    elif unknown_margin < float(unknown_margin_threshold):
        status = "unknown"
        subtype_id = "Unknown"
        subtype_label = "Unknown"
        reason = "unknown_margin"

    return {
        "gallery_status": status,
        "decision_reason": reason,
        "subtype_id": subtype_id,
        "subtype_label": subtype_label,
        "best_candidate_id": class_ids[best_index],
        "best_candidate_label": class_labels[best_index],
        "score": best_score,
        "known_bias": float(known_bias),
        "effective_known_score": effective_known_score,
        "second_score": second_score,
        "class_margin": class_margin,
        "background_score": float(background_score),
        "unknown_score": float(unknown_score),
        "negative_margin": min(background_margin, unknown_margin),
        "background_margin": background_margin,
        "unknown_margin": unknown_margin,
        "low_confidence": bool(
            not class_margin_as_unknown
            and class_margin < float(class_margin_threshold)
        ),
        "temporal_propagated": False,
    }


class SimpleIoUTracker:
    def __init__(
        self,
        class_ids: list[str],
        class_labels: list[str],
        temporal_window: int,
        iou_threshold: float,
        min_similarity: float,
        class_margin: float,
        background_margin: float,
        unknown_margin: float,
        known_bias: float = 0.0,
        class_margin_as_unknown: bool = True,
        retain_class_scores: bool = False,
        background_filter_enabled: bool = True,
        max_gap: int = 2,
    ) -> None:
        self.class_ids = class_ids
        self.class_labels = class_labels
        self.temporal_window = max(1, int(temporal_window))
        self.iou_threshold = float(iou_threshold)
        self.min_similarity = float(min_similarity)
        self.class_margin = float(class_margin)
        self.background_margin = float(background_margin)
        self.unknown_margin = float(unknown_margin)
        self.known_bias = float(known_bias)
        self.class_margin_as_unknown = bool(class_margin_as_unknown)
        self.retain_class_scores = bool(retain_class_scores)
        self.background_filter_enabled = bool(background_filter_enabled)
        self.max_gap = max(0, int(max_gap))
        self.tracks: dict[int, _Track] = {}
        self.next_track_id = 1

    def _new_track(self, candidate: dict[str, Any], frame_index: int) -> _Track:
        track = _Track(
            track_id=self.next_track_id,
            last_box=np.asarray(candidate["xyxy"], dtype=np.float32),
            last_frame_index=int(frame_index),
            class_history=deque(maxlen=self.temporal_window),
            background_history=deque(maxlen=self.temporal_window),
            unknown_history=deque(maxlen=self.temporal_window),
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def update_frame(
        self,
        frame_index: int,
        sample_stride: int,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        allowed_delta = max(1, int(sample_stride)) * (self.max_gap + 1)
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if int(frame_index) - track.last_frame_index > allowed_delta
        ]
        for track_id in expired:
            del self.tracks[track_id]

        pair_scores: list[tuple[float, int, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_box = np.asarray(candidate["xyxy"], dtype=np.float32)
            for track_id, track in self.tracks.items():
                score = box_iou(candidate_box, track.last_box)
                if score >= self.iou_threshold:
                    pair_scores.append((score, candidate_index, track_id))
        pair_scores.sort(reverse=True)

        assigned_candidates: set[int] = set()
        assigned_tracks: set[int] = set()
        candidate_tracks: dict[int, _Track] = {}
        for _, candidate_index, track_id in pair_scores:
            if candidate_index in assigned_candidates or track_id in assigned_tracks:
                continue
            assigned_candidates.add(candidate_index)
            assigned_tracks.add(track_id)
            candidate_tracks[candidate_index] = self.tracks[track_id]

        decisions: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            track = candidate_tracks.get(candidate_index)
            if track is None:
                track = self._new_track(candidate, frame_index)
            track.last_box = np.asarray(candidate["xyxy"], dtype=np.float32)
            track.last_frame_index = int(frame_index)
            track.class_history.append(np.asarray(candidate["class_scores"], dtype=np.float32))
            track.background_history.append(float(candidate["background_score"]))
            track.unknown_history.append(float(candidate["unknown_score"]))
            track.observations += 1

            averaged_classes = np.stack(list(track.class_history), axis=0).mean(axis=0)
            averaged_background = float(np.mean(track.background_history))
            averaged_unknown = float(np.mean(track.unknown_history))
            decision = decide_gallery_candidate(
                self.class_ids,
                self.class_labels,
                averaged_classes,
                averaged_background,
                averaged_unknown,
                self.min_similarity,
                self.class_margin,
                self.background_margin,
                self.unknown_margin,
                self.known_bias,
                self.class_margin_as_unknown,
                self.background_filter_enabled,
            )
            item = {
                    **candidate["proposal"],
                    **decision,
                    "track_id": track.track_id,
                    "track_observations": track.observations,
                }
            if isinstance(candidate.get("embedding"), np.ndarray):
                item["_embedding"] = np.asarray(
                    candidate["embedding"], dtype=np.float32
                )
            if self.retain_class_scores:
                item["_class_scores"] = np.asarray(
                    averaged_classes, dtype=np.float32
                )
            decisions.append(item)
        return decisions


class GalleryTemporalMatcher:
    algorithm = GALLERY_CLIP_ALGORITHM
    class_margin_as_unknown = True
    retain_candidate_embeddings = False
    retain_candidate_class_scores = False

    def __init__(self, config: DetectionConfig, embedder: Any) -> None:
        self.config = config
        self.algorithm = str(config.algorithm)
        self.embedder = embedder
        self.root = Path(config.gallery_root)
        self.classes: list[GalleryClass] = []
        self.class_features: list[np.ndarray] = []
        self.negative_features = np.empty((0, 0), dtype=np.float32)
        self.unknown_features = np.empty((0, 0), dtype=np.float32)
        self.tracker: SimpleIoUTracker | None = None
        self.info: dict[str, Any] | None = None
        self.embedding_batch_size = int(config.clip_batch_size)

    @staticmethod
    def _open_images(paths: list[Path]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        return images

    def _encode_paths(self, paths: list[Path]) -> np.ndarray:
        images = self._open_images(paths)
        features = self.embedder.encode_images(
            images,
            batch_size=self.embedding_batch_size,
        )
        if features is None:
            return np.empty((0, 0), dtype=np.float32)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def prepare(
        self,
        target_image: Image.Image | None = None,
        description: str | None = None,
    ) -> None:
        del target_image, description
        info = inspect_gallery_resources(self.root)
        if not info["ready"]:
            missing = ", ".join(info["missing"])
            raise GalleryResourceError(f"{self.algorithm} 资源不完整: {missing}")
        self.classes = load_gallery_classes(self.root)
        self.class_features = [
            self._encode_paths(_image_paths(self.root / "prototypes" / item.folder))
            for item in self.classes
        ]
        self.negative_features = self._encode_paths(_image_paths(self.root / "negatives"))
        self.unknown_features = self._encode_paths(_image_paths(self.root / "unknown_tanks"))
        self.tracker = SimpleIoUTracker(
            class_ids=[item.class_id for item in self.classes],
            class_labels=[item.label for item in self.classes],
            temporal_window=self.config.gallery_temporal_window,
            iou_threshold=self.config.gallery_track_iou,
            min_similarity=self.config.gallery_min_similarity,
            class_margin=self.config.gallery_class_margin,
            background_margin=self.config.gallery_background_margin,
            unknown_margin=self.config.gallery_unknown_margin,
            known_bias=self.config.gallery_known_bias,
            class_margin_as_unknown=self.class_margin_as_unknown,
            retain_class_scores=self.retain_candidate_class_scores,
            background_filter_enabled=(
                self.config.dino_background_filter_enabled
                if self.algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
                else True
            ),
        )
        self.info = {
            **info,
            "algorithm": self.algorithm,
            "embedding_backend": str(getattr(self.embedder, "backend", "clip")),
        }
        embedder_info = getattr(self.embedder, "summary_info", None)
        if callable(embedder_info):
            self.info["embedding_model"] = embedder_info()

    def match_batch(
        self,
        crop_features: Any,
        crop_color_signatures: list[np.ndarray],
        proposals: list[dict[str, Any]],
        frame_indices: list[int],
        sample_stride: int,
    ) -> MatchBatchResult:
        del crop_color_signatures
        visible = {int(index): [] for index in frame_indices}
        decisions = {int(index): [] for index in frame_indices}
        if crop_features is None:
            return MatchBatchResult(visible, decisions)
        if self.tracker is None:
            raise RuntimeError("GalleryTemporalMatcher.prepare() must be called first")

        candidate_features = crop_features.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        class_scores = top_k_class_scores(candidate_features, self.class_features)
        background_scores = top_k_bank_scores(candidate_features, self.negative_features)
        unknown_scores = top_k_bank_scores(candidate_features, self.unknown_features)

        grouped: dict[int, list[dict[str, Any]]] = {int(index): [] for index in frame_indices}
        for index, proposal in enumerate(proposals):
            candidate = {
                    "proposal": proposal,
                    "xyxy": proposal["xyxy"],
                    "class_scores": class_scores[index],
                    "background_score": float(background_scores[index]),
                    "unknown_score": float(unknown_scores[index]),
                }
            if self.retain_candidate_embeddings:
                candidate["embedding"] = candidate_features[index]
            grouped[int(proposal["frame_index"])].append(candidate)

        for frame_index in sorted(grouped):
            frame_decisions = self.tracker.update_frame(
                frame_index,
                sample_stride,
                grouped[frame_index],
            )
            decisions[frame_index] = frame_decisions
            visible[frame_index] = [
                item
                for item in frame_decisions
                if item["gallery_status"] != "rejected_background"
            ]
        return MatchBatchResult(visible, decisions)

    def summary_info(self) -> dict[str, Any] | None:
        return dict(self.info) if self.info is not None else None


class DinoGalleryTemporalMatcher(GalleryTemporalMatcher):
    algorithm = DINO_GALLERY_TEMPORAL_ALGORITHM
    class_margin_as_unknown = False
    retain_candidate_embeddings = False

    def __init__(self, config: DetectionConfig, embedder: Any) -> None:
        super().__init__(config, embedder)
        self.embedding_batch_size = int(config.dino_batch_size)
        self.retain_candidate_embeddings = bool(config.dino_hysteresis_enabled)
        self.retain_candidate_class_scores = bool(config.dino_hysteresis_enabled)

    def prepare(
        self,
        target_image: Image.Image | None = None,
        description: str | None = None,
    ) -> None:
        super().prepare(target_image=target_image, description=description)
        if self.info is not None:
            self.info["embedding_backend"] = "dinov2"
