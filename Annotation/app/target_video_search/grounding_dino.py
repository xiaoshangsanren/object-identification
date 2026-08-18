from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import DetectionConfig

REQUIRED_GROUNDING_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def inspect_grounding_model(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir)
    missing = [
        name for name in REQUIRED_GROUNDING_FILES if not (root / name).is_file()
    ]
    return {
        "root": str(root),
        "ready": root.is_dir() and not missing,
        "required_files": list(REQUIRED_GROUNDING_FILES),
        "missing": missing,
    }


def split_grounding_phrases(prompt: str) -> list[str]:
    return [part.strip() for part in str(prompt or "").split(".") if part.strip()]


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0, float(box_a[3] - box_a[1])
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0, float(box_b[3] - box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def nms_grounding_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    ordered = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        candidate_box = np.asarray(candidate["xyxy"], dtype=np.float32)
        if any(
            box_iou(candidate_box, np.asarray(item["xyxy"], dtype=np.float32))
            > float(iou_threshold)
            for item in kept
        ):
            continue
        kept.append(candidate)
    return kept


class GroundingDinoProposalProvider:
    backend = "grounding_dino"

    def __init__(self, config: DetectionConfig) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = self._choose_device(config.device)
        self.model_dir = Path(config.grounding_model_dir).expanduser()
        self.prompt = str(config.grounding_prompt or "").strip()
        self.phrases = split_grounding_phrases(self.prompt)
        if not self.phrases:
            raise ValueError("text_grounding_temporal 必须提供非空文本描述。")
        status = inspect_grounding_model(self.model_dir)
        if not status["ready"]:
            missing = ", ".join(status["missing"]) or "模型目录不存在"
            raise RuntimeError(
                f"Grounding DINO 本地模型资源不完整: {self.model_dir} ({missing})"
            )
        self.box_threshold = float(config.grounding_box_threshold)
        self.text_threshold = float(config.grounding_text_threshold)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def _choose_device(self, configured: str) -> str:
        if configured and configured.lower() != "auto":
            return configured
        return "cuda:0" if self.torch.cuda.is_available() else "cpu"

    @staticmethod
    def _label_text(value: Any, phrases: list[str]) -> str:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, str):
            return value.strip() or "target"
        try:
            index = int(value)
        except (TypeError, ValueError):
            return "target"
        return phrases[index] if 0 <= index < len(phrases) else str(index)

    def predict(self, frame: np.ndarray, frame_index: int) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels = processed.get("text_labels")
        if labels is None:
            labels = processed.get("labels", [])

        detections: list[dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            box_values = box.detach().float().cpu().tolist() if hasattr(box, "detach") else list(box)
            score_value = float(score.detach().float().cpu().item()) if hasattr(score, "detach") else float(score)
            phrase = self._label_text(label, self.phrases)
            detections.append(
                {
                    "frame_index": int(frame_index),
                    "xyxy": [float(value) for value in box_values],
                    "score": score_value,
                    "grounding_score": score_value,
                    "phrase": phrase,
                    "class_id": -1,
                    "class_name": phrase,
                    "detector_conf": score_value,
                    "temporal_propagated": False,
                }
            )
        return nms_grounding_detections(detections, iou_threshold=0.50)


@dataclass
class _GroundingTrack:
    track_id: int
    last_box: np.ndarray
    boxes: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=3))
    observations: deque[tuple[str, float]] = field(
        default_factory=lambda: deque(maxlen=3)
    )
    total_observations: int = 0
    missing_samples: int = 0


class GroundingTemporalTracker:
    def __init__(self, iou_threshold: float = 0.30, max_gap_samples: int = 2) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_gap_samples = max(0, int(max_gap_samples))
        self.tracks: dict[int, _GroundingTrack] = {}
        self.next_track_id = 1

    def _new_track(self, detection: dict[str, Any]) -> _GroundingTrack:
        box = np.asarray(detection["xyxy"], dtype=np.float32)
        track = _GroundingTrack(track_id=self.next_track_id, last_box=box)
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track

    @staticmethod
    def _append(track: _GroundingTrack, detection: dict[str, Any]) -> None:
        box = np.asarray(detection["xyxy"], dtype=np.float32)
        track.last_box = box
        track.boxes.append(box)
        track.observations.append(
            (str(detection.get("phrase") or "target"), float(detection["score"]))
        )
        track.total_observations += 1
        track.missing_samples = 0

    @staticmethod
    def _render(track: _GroundingTrack, frame_index: int, propagated: bool) -> dict[str, Any]:
        averaged_box = np.stack(list(track.boxes), axis=0).mean(axis=0)
        phrase_scores: dict[str, list[float]] = defaultdict(list)
        for phrase, score in track.observations:
            phrase_scores[phrase].append(score)
        phrase, values = max(
            phrase_scores.items(),
            key=lambda item: (float(np.mean(item[1])), item[0]),
        )
        score = float(np.mean(values))
        return {
            "frame_index": int(frame_index),
            "xyxy": [float(value) for value in averaged_box.tolist()],
            "score": score,
            "grounding_score": score,
            "phrase": phrase,
            "class_id": -1,
            "class_name": phrase,
            "detector_conf": score,
            "track_id": track.track_id,
            "track_observations": track.total_observations,
            "temporal_propagated": bool(propagated),
        }

    def update_frame(
        self,
        frame_index: int,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pair_scores: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            box = np.asarray(detection["xyxy"], dtype=np.float32)
            for track_id, track in self.tracks.items():
                score = box_iou(box, track.last_box)
                if score >= self.iou_threshold:
                    pair_scores.append((score, detection_index, track_id))
        pair_scores.sort(reverse=True)

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, _GroundingTrack] = {}
        for _, detection_index, track_id in pair_scores:
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assignments[detection_index] = self.tracks[track_id]

        visible: list[dict[str, Any]] = []
        updated_track_ids: set[int] = set()
        for detection_index, detection in enumerate(detections):
            track = assignments.get(detection_index) or self._new_track(detection)
            self._append(track, detection)
            updated_track_ids.add(track.track_id)
            visible.append(self._render(track, frame_index, propagated=False))

        for track_id, track in list(self.tracks.items()):
            if track_id in updated_track_ids:
                continue
            track.missing_samples += 1
            if track.missing_samples <= self.max_gap_samples:
                visible.append(self._render(track, frame_index, propagated=True))
            else:
                del self.tracks[track_id]
        visible.sort(key=lambda item: int(item["track_id"]))
        return visible
