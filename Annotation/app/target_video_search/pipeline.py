from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .algorithms import create_match_strategy
from .config import (
    DINO_GALLERY_TEMPORAL_ALGORITHM,
    DetectionConfig,
    GALLERY_ALGORITHMS,
    LEGACY_CLIP_ALGORITHM,
    PAD_LITE_ALGORITHMS,
    TEXT_GROUNDING_TEMPORAL_ALGORITHM,
    resolve_frame_stride,
)
from .dino_embedder import DinoEmbedder
from .pad_lite_embedder import PADLiteEmbedder
from .drawing import draw_matches
from .grounding_dino import GroundingDinoProposalProvider, GroundingTemporalTracker
from .temporal_hysteresis import (
    apply_dino_temporal_hysteresis,
    strip_internal_decision_fields,
)
from .video_io import make_video_writer

ProgressCallback = Callable[[int, int, str], None]


def build_output_path(
    output_dir: str | Path,
    algorithm: str,
    stamp: str | None = None,
) -> Path:
    method = re.sub(r"[^A-Za-z0-9_-]+", "_", str(algorithm)).strip("_")
    if not method:
        raise ValueError("识别算法名称不能为空。")
    resolved_stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{method}_{resolved_stamp}.mp4"


def _choose_torch_device(device: str) -> str:
    import torch

    if device and device.lower() != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _is_default_open_clip_weight(value: str) -> bool:
    return value.strip().lower() in {
        "laion2b_s34b_b79k",
        "laion/clip-vit-b-32-laion2b-s34b-b79k",
    }


def _normalize_target_box_to_image(
    box: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    width, height = [int(v) for v in image_size]
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
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


def _crop_target_image_to_box(
    image: Image.Image,
    box: tuple[int, int, int, int] | None,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    normalized_box = _normalize_target_box_to_image(box, image.size)
    if normalized_box is None:
        return image, None
    return image.crop(normalized_box), normalized_box


def _mean_rgb_signature(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if array.size == 0:
        return np.zeros(3, dtype=np.float32)
    return array.reshape(-1, 3).mean(axis=0)


def _color_similarity(
    target_signature: np.ndarray | None,
    candidate_signature: np.ndarray | None,
) -> float:
    if target_signature is None or candidate_signature is None:
        return 0.0
    distance = float(np.linalg.norm(target_signature - candidate_signature))
    max_distance = float(np.sqrt(3.0 * (255.0**2)))
    normalized = 1.0 - min(max(distance / max_distance, 0.0), 1.0)
    return normalized


def _match_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item.get("score", 0.0)),
        float(item.get("color_similarity", 0.0)),
        float(item.get("image_score", 0.0)),
        float(item.get("detector_conf", 0.0)),
    )


def _mark_primary_match(
    matches: list[dict[str, Any]],
    threshold: float,
    score_tolerance: float = 1e-6,
) -> None:
    for item in matches:
        item["is_primary_highlight"] = False
    if not matches:
        return
    top_score = float(matches[0].get("score", 0.0))
    tied_top_matches = [
        item
        for item in matches
        if abs(float(item.get("score", 0.0)) - top_score) <= float(score_tolerance)
    ]
    top_match = max(tied_top_matches, key=_match_sort_key)
    if float(top_match.get("score", 0.0)) >= float(threshold):
        top_match["is_primary_highlight"] = True
        chosen_index = matches.index(top_match)
        if chosen_index != 0:
            matches[0], matches[chosen_index] = matches[chosen_index], matches[0]


class ClipEmbedder:
    def __init__(self, config: DetectionConfig) -> None:
        import torch

        self.torch = torch
        self.device = _choose_torch_device(config.device)
        self.half_cuda = config.half and self.device.startswith("cuda")
        backend = (config.clip_backend or "transformers").strip().lower()
        local_dir = Path(config.clip_local_dir).expanduser() if config.clip_local_dir else None
        pretrained_path = Path(config.clip_pretrained).expanduser()
        use_transformers = (
            backend == "transformers"
            or (local_dir is not None and local_dir.exists())
            or pretrained_path.is_dir()
        )
        self.backend = "transformers" if use_transformers else "open_clip"

        if self.backend == "transformers":
            from transformers import CLIPModel, CLIPProcessor

            if local_dir is not None:
                model_dir = local_dir
            else:
                model_dir = pretrained_path
            if not model_dir.is_dir():
                raise RuntimeError(
                    "CLIP 本地模型目录不存在，当前系统不能访问 Hugging Face 时必须使用 "
                    f"ModelScope 本地模型目录: {model_dir}\n"
                    "离线交付包中的CLIP模型目录不完整。请重新复制完整的 "
                    "Resource/models/clip-vit-base-patch32 目录，并执行交付包的完整性校验。"
                )
            self.processor = CLIPProcessor.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
            self.model = CLIPModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
            ).to(self.device)
            self.preprocess = None
            self.tokenizer = None
        else:
            import open_clip

            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    config.clip_model,
                    pretrained=config.clip_pretrained,
                    device=self.device,
                )
            except RuntimeError as exc:
                if _is_default_open_clip_weight(str(config.clip_pretrained)):
                    raise RuntimeError(
                        "当前配置仍在使用 open_clip 在线权重 "
                        f"{config.clip_pretrained!r}，会访问 Hugging Face 并失败。\n"
                        "请把 CLIP 后端改为 transformers，CLIP 本地目录改为 "
                        "models/clip-vit-base-patch32，然后重启应用。"
                    ) from exc
                raise
            self.tokenizer = open_clip.get_tokenizer(config.clip_model)
        self.model.eval()
        if self.half_cuda:
            self.model.half()

    def _autocast(self):
        if not self.half_cuda:
            return nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=self.torch.float16)

    def _feature_tensor(self, output: Any, feature_name: str):
        if self.torch.is_tensor(output):
            return output

        preferred_attrs = (
            ("image_embeds", "pooler_output", "last_hidden_state")
            if feature_name == "image"
            else ("text_embeds", "pooler_output", "last_hidden_state")
        )
        for attr in preferred_attrs:
            value = getattr(output, attr, None)
            if value is None:
                continue
            if attr == "last_hidden_state" and value.ndim == 3:
                return value[:, 0]
            return value

        if isinstance(output, (tuple, list)) and output:
            value = output[0]
            if self.torch.is_tensor(value):
                return value[:, 0] if value.ndim == 3 else value

        raise TypeError(
            f"Cannot extract {feature_name} feature tensor from "
            f"{type(output).__name__}. Expected Tensor or model output with "
            "image_embeds/text_embeds/pooler_output."
        )

    def encode_images(self, images: list[Image.Image], batch_size: int) -> Any:
        if not images:
            return None
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                if self.backend == "transformers":
                    inputs = self.processor(
                        images=[image.convert("RGB") for image in batch],
                        return_tensors="pt",
                    ).to(self.device)
                    if self.half_cuda:
                        inputs["pixel_values"] = inputs["pixel_values"].half()
                    with self._autocast():
                        batch_features = self.model.get_image_features(
                            pixel_values=inputs["pixel_values"]
                        )
                else:
                    tensor = self.torch.stack(
                        [self.preprocess(image.convert("RGB")) for image in batch]
                    ).to(self.device)
                    if self.half_cuda:
                        tensor = tensor.half()
                    with self._autocast():
                        batch_features = self.model.encode_image(tensor)
                batch_features = self._feature_tensor(batch_features, "image")
                batch_features = batch_features.float()
                batch_features = batch_features / batch_features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                features.append(batch_features.cpu())
        return self.torch.cat(features, dim=0)

    def encode_text(self, text: str | None) -> Any:
        if not text or not text.strip():
            return None
        with self.torch.inference_mode():
            if self.backend == "transformers":
                inputs = self.processor(
                    text=[text.strip()],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                with self._autocast():
                    text_features = self.model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
            else:
                tokens = self.tokenizer([text.strip()]).to(self.device)
                with self._autocast():
                    text_features = self.model.encode_text(tokens)
            text_features = self._feature_tensor(text_features, "text")
            text_features = text_features.float()
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
        return text_features.cpu()


def create_image_embedder(config: DetectionConfig) -> Any:
    if config.algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM:
        return DinoEmbedder(config)
    if config.algorithm in PAD_LITE_ALGORITHMS:
        return PADLiteEmbedder(config)
    return ClipEmbedder(config)


class YoloProposalDetector:
    def __init__(self, config: DetectionConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.device = _choose_torch_device(config.device)
        self.model = YOLO(config.detector_model)

    def predict(self, frames: list[np.ndarray]):
        kwargs: dict[str, Any] = {
            "source": frames,
            "imgsz": self.config.detector_imgsz,
            "conf": self.config.proposal_conf,
            "device": self.device,
            "batch": max(1, len(frames)),
            "verbose": False,
        }
        if self.config.class_ids is not None:
            kwargs["classes"] = list(self.config.class_ids)
        if self.config.half and self.device.startswith("cuda"):
            kwargs["half"] = True
        return self.model.predict(**kwargs)


def create_proposal_provider(config: DetectionConfig) -> Any:
    if config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
        return GroundingDinoProposalProvider(config)
    return YoloProposalDetector(config)


class TargetVideoAnnotator:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self.detector = create_proposal_provider(self.config)
        if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            self.embedder = None
            self.matcher = None
            self.grounding_tracker = GroundingTemporalTracker(
                iou_threshold=self.config.grounding_track_iou,
                max_gap_samples=self.config.grounding_max_gap_samples,
            )
            self.embedding_backend = "grounding_dino"
            self.embedding_model = self.config.grounding_model_dir
            self.embedding_batch_size = 1
            return

        self.embedder = create_image_embedder(self.config)
        if self.config.algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM:
            self.embedding_backend = "dinov2"
            self.embedding_model = self.config.dino_local_dir
            self.embedding_batch_size = int(self.config.dino_batch_size)
        elif self.config.algorithm in PAD_LITE_ALGORITHMS:
            self.embedding_backend = str(self.embedder.backend)
            self.embedding_model = (
                str(self.embedder.checkpoint_path)
                if self.embedder.checkpoint_path is not None
                else self.config.clip_local_dir
            )
            self.embedding_batch_size = int(self.config.clip_batch_size)
        else:
            self.embedding_backend = "clip"
            self.embedding_model = self.config.clip_local_dir
            self.embedding_batch_size = int(self.config.clip_batch_size)
        self.matcher = create_match_strategy(self.config, self.embedder)

    def process_video(
        self,
        video_path: str | Path,
        target_image_path: str | Path | None,
        output_path: str | Path | None = None,
        description: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = build_output_path(output_dir, self.config.algorithm)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        applied_target_box = None
        resolved_target_image_path: Path | None = None
        if self.config.algorithm == LEGACY_CLIP_ALGORITHM:
            if not target_image_path:
                raise ValueError("CLIP 模式必须提供目标图片。")
            resolved_target_image_path = Path(target_image_path)
            target_image = ImageOps.exif_transpose(
                Image.open(resolved_target_image_path)
            ).convert("RGB")
            target_image, applied_target_box = _crop_target_image_to_box(
                target_image,
                self.config.target_box,
            )
            self.matcher.prepare(target_image=target_image, description=description)
        elif self.config.algorithm in GALLERY_ALGORITHMS:
            self.matcher.prepare()
        elif self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            if not str(self.config.grounding_prompt or "").strip():
                raise ValueError("text_grounding_temporal 必须提供非空文本描述。")
        else:
            raise ValueError(f"不支持的识别算法: {self.config.algorithm}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Cannot read video dimensions")

        selected_sample_fps = (
            self.config.grounding_sample_fps
            if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM
            else self.config.target_sample_fps
        )
        sample_stride = resolve_frame_stride(
            fps, selected_sample_fps, self.config.frame_stride
        )
        processing_batch_size = (
            1
            if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM
            else self.config.yolo_batch_size
        )
        start_time = time.perf_counter()
        sample_frames: list[np.ndarray] = []
        sample_indices: list[int] = []
        sample_results: dict[int, list[dict[str, Any]]] = {}
        sample_decisions: dict[int, list[dict[str, Any]]] = {}
        frame_results: list[dict[str, Any]] = []
        decision_results: list[dict[str, Any]] = []
        current_matches: list[dict[str, Any]] = []
        frames_written = 0
        sampled_frames = 0

        def emit(frame_count: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(frame_count, total_frames, message)

        dino_hysteresis_active = (
            self.config.algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
            and self.config.dino_hysteresis_enabled
        )
        if dino_hysteresis_active:
            def flush_dino_samples() -> None:
                nonlocal sampled_frames
                if not sample_frames:
                    return
                batch_result = self._process_sample_batch(
                    frames=sample_frames,
                    frame_indices=sample_indices,
                    sample_stride=sample_stride,
                )
                sample_decisions.update(batch_result.decisions_by_frame)
                sampled_frames += len(sample_frames)
                emit(
                    min(total_frames, (sample_indices[-1] + 1) // 2),
                    "analyzing video",
                )
                sample_frames.clear()
                sample_indices.clear()

            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % sample_stride == 0:
                        sample_frames.append(frame)
                        sample_indices.append(frame_index)
                    if len(sample_frames) >= processing_batch_size:
                        flush_dino_samples()
                    frame_index += 1
                flush_dino_samples()
            finally:
                cap.release()

            raw_decision_results = [
                {
                    "frame_index": index,
                    "time_sec": index / fps,
                    "decisions": sample_decisions.get(index, []),
                }
                for index in sorted(sample_decisions)
            ]
            emit(total_frames // 2, "applying temporal hysteresis")
            apply_dino_temporal_hysteresis(
                raw_decision_results,
                max_span_sec=self.config.dino_hysteresis_max_span_sec,
                anchor_min_frames=self.config.dino_hysteresis_anchor_min_frames,
                anchor_similarity=self.config.dino_hysteresis_anchor_similarity,
            )

            for frame_item in raw_decision_results:
                clean_decisions = [
                    strip_internal_decision_fields(item)
                    for item in frame_item["decisions"]
                ]
                index = int(frame_item["frame_index"])
                visible = [
                    item
                    for item in clean_decisions
                    if item.get("gallery_status") != "rejected_background"
                ]
                sample_results[index] = visible
                frame_results.append(
                    {
                        "frame_index": index,
                        "time_sec": float(frame_item["time_sec"]),
                        "matches": visible,
                    }
                )
                decision_results.append(
                    {
                        "frame_index": index,
                        "time_sec": float(frame_item["time_sec"]),
                        "decisions": clean_decisions,
                    }
                )

            render_cap = cv2.VideoCapture(str(video_path))
            if not render_cap.isOpened():
                raise RuntimeError(f"Cannot reopen video for rendering: {video_path}")
            writer = make_video_writer(
                output_path,
                fps,
                width,
                height,
                use_nvenc=self.config.use_nvenc,
            )
            frame_index = 0
            try:
                while True:
                    ok, frame = render_cap.read()
                    if not ok:
                        break
                    if frame_index in sample_results:
                        current_matches = sample_results[frame_index]
                    matches = (
                        current_matches
                        if self.config.hold_boxes
                        else sample_results.get(frame_index, [])
                    )
                    writer.write(draw_matches(frame, matches, frame_index))
                    frames_written += 1
                    if frames_written % 128 == 0:
                        emit(
                            min(total_frames, total_frames // 2 + frames_written // 2),
                            "annotating video",
                        )
                    frame_index += 1
                emit(total_frames, "annotating video")
            finally:
                render_cap.release()
                writer.close()
        else:
            writer = make_video_writer(
                output_path,
                fps,
                width,
                height,
                use_nvenc=self.config.use_nvenc,
            )
            segment: list[tuple[int, np.ndarray]] = []

            def flush_segment() -> None:
                nonlocal current_matches, frames_written, sampled_frames
                if sample_frames:
                    batch_result = self._process_sample_batch(
                        frames=sample_frames,
                        frame_indices=sample_indices,
                        sample_stride=sample_stride,
                    )
                    sample_results.update(batch_result.visible_by_frame)
                    sample_decisions.update(batch_result.decisions_by_frame)
                    sampled_frames += len(sample_frames)
                    sample_frames.clear()
                    sample_indices.clear()

                for item_index, item_frame in segment:
                    if item_index in sample_results:
                        current_matches = sample_results[item_index]
                        frame_results.append(
                            {
                                "frame_index": item_index,
                                "time_sec": item_index / fps,
                                "matches": current_matches,
                            }
                        )
                        if self.config.algorithm in GALLERY_ALGORITHMS:
                            decision_results.append(
                                {
                                    "frame_index": item_index,
                                    "time_sec": item_index / fps,
                                    "decisions": sample_decisions.get(item_index, []),
                                }
                            )
                    matches = (
                        current_matches
                        if self.config.hold_boxes
                        else sample_results.get(item_index, [])
                    )
                    writer.write(draw_matches(item_frame, matches, item_index))
                    frames_written += 1
                segment.clear()
                emit(frames_written, "annotating video")

            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    segment.append((frame_index, frame))
                    if frame_index % sample_stride == 0:
                        sample_frames.append(frame)
                        sample_indices.append(frame_index)
                    if len(sample_frames) >= processing_batch_size:
                        flush_segment()
                    frame_index += 1
                flush_segment()
            finally:
                cap.release()
                writer.close()

        elapsed = time.perf_counter() - start_time
        matched_frames = sum(1 for item in frame_results if item["matches"])
        matched_boxes = sum(len(item["matches"]) for item in frame_results)
        flat_decisions = [
            decision
            for frame_item in decision_results
            for decision in frame_item["decisions"]
        ]
        known_boxes = sum(
            1 for item in flat_decisions if item.get("gallery_status") == "known"
        )
        unknown_boxes = sum(
            1 for item in flat_decisions if item.get("gallery_status") == "unknown"
        )
        rejected_boxes = sum(
            1
            for item in flat_decisions
            if item.get("gallery_status") == "rejected_background"
        )
        low_confidence_boxes = sum(
            1
            for item in flat_decisions
            if item.get("gallery_status") == "known" and item.get("low_confidence")
        )
        temporal_propagated_boxes = sum(
            1 for item in flat_decisions if item.get("temporal_propagated")
        )
        if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            temporal_propagated_boxes = sum(
                1
                for frame_item in frame_results
                for item in frame_item["matches"]
                if item.get("temporal_propagated")
            )
        summary = {
            "algorithm": self.config.algorithm,
            "proposal_backend": (
                "grounding_dino"
                if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM
                else "yolo"
            ),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "video_path": str(video_path),
            "target_image_path": (
                str(resolved_target_image_path)
                if resolved_target_image_path is not None
                else None
            ),
            "description": (
                self.config.grounding_prompt
                if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM
                else (description or "")
            ),
            "output_video": str(output_path),
            "target_box": list(applied_target_box) if applied_target_box is not None else None,
            "total_frames": frames_written,
            "source_total_frames": total_frames,
            "fps": fps,
            "duration_sec": frames_written / fps if fps else 0.0,
            "sample_stride": sample_stride,
            "sampled_frames": sampled_frames,
            "sample_fps": fps / sample_stride if sample_stride else fps,
            "matched_sample_frames": matched_frames,
            "matched_boxes": matched_boxes,
            "known_boxes": known_boxes,
            "unknown_boxes": unknown_boxes,
            "rejected_boxes": rejected_boxes,
            "low_confidence_boxes": low_confidence_boxes,
            "temporal_propagated_boxes": temporal_propagated_boxes,
            "processing_sec": elapsed,
            "processing_fps": frames_written / elapsed if elapsed > 0 else None,
            "video_seconds_per_processing_second": (
                (frames_written / fps) / elapsed if fps and elapsed > 0 else None
            ),
            "config": {
                "algorithm": self.config.algorithm,
                "detector_model": self.config.detector_model,
                "detector_imgsz": self.config.detector_imgsz,
                "proposal_conf": self.config.proposal_conf,
                "class_ids": self.config.class_ids,
                "clip_model": self.config.clip_model,
                "clip_pretrained": self.config.clip_pretrained,
                "clip_backend": self.config.clip_backend,
                "clip_local_dir": self.config.clip_local_dir,
                "similarity_threshold": self.config.similarity_threshold,
                "highlight_similarity_threshold": self.config.highlight_similarity_threshold,
                "text_weight": self.config.text_weight,
                "gallery_root": self.config.gallery_root,
                "gallery_min_similarity": self.config.gallery_min_similarity,
                "gallery_class_margin": self.config.gallery_class_margin,
                "gallery_background_margin": self.config.gallery_background_margin,
                "gallery_unknown_margin": self.config.gallery_unknown_margin,
                "gallery_known_bias": self.config.gallery_known_bias,
                "gallery_temporal_window": self.config.gallery_temporal_window,
                "gallery_track_iou": self.config.gallery_track_iou,
                "target_sample_fps": self.config.target_sample_fps,
                "frame_stride": self.config.frame_stride,
                "yolo_batch_size": self.config.yolo_batch_size,
                "clip_batch_size": self.config.clip_batch_size,
                "dino_local_dir": self.config.dino_local_dir,
                "dino_batch_size": self.config.dino_batch_size,
                "pad_lite_checkpoint_root": self.config.pad_lite_checkpoint_root,
                "dino_background_filter_enabled": self.config.dino_background_filter_enabled,
                "dino_hysteresis_enabled": self.config.dino_hysteresis_enabled,
                "dino_hysteresis_max_span_sec": self.config.dino_hysteresis_max_span_sec,
                "dino_hysteresis_anchor_min_frames": self.config.dino_hysteresis_anchor_min_frames,
                "dino_hysteresis_anchor_similarity": self.config.dino_hysteresis_anchor_similarity,
                "grounding_model_dir": self.config.grounding_model_dir,
                "grounding_prompt": self.config.grounding_prompt,
                "grounding_box_threshold": self.config.grounding_box_threshold,
                "grounding_text_threshold": self.config.grounding_text_threshold,
                "grounding_sample_fps": self.config.grounding_sample_fps,
                "grounding_track_iou": self.config.grounding_track_iou,
                "grounding_max_gap_samples": self.config.grounding_max_gap_samples,
                "hold_boxes": self.config.hold_boxes,
                "half": self.config.half,
                "target_box": list(applied_target_box) if applied_target_box is not None else None,
                "use_nvenc": self.config.use_nvenc,
            },
            "gallery_info": (
                self.matcher.summary_info() if self.matcher is not None else None
            ),
            "detections": frame_results,
        }
        if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            summary.update(
                {
                    "grounding_model": self.config.grounding_model_dir,
                    "grounding_prompt": self.config.grounding_prompt,
                    "grounding_box_threshold": self.config.grounding_box_threshold,
                    "grounding_text_threshold": self.config.grounding_text_threshold,
                }
            )
        if self.config.algorithm in GALLERY_ALGORITHMS:
            summary["proposal_decisions"] = decision_results
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        summary["output_json"] = str(json_path)
        return summary

    def _process_sample_batch(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        sample_stride: int,
    ):
        from .algorithms.base import MatchBatchResult

        empty = {int(index): [] for index in frame_indices}
        if not frames:
            return MatchBatchResult(empty, {key: [] for key in empty})

        if self.config.algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM:
            visible: dict[int, list[dict[str, Any]]] = {}
            for frame, frame_index in zip(frames, frame_indices):
                detections = self.detector.predict(frame, int(frame_index))
                visible[int(frame_index)] = self.grounding_tracker.update_frame(
                    int(frame_index), detections
                )
            return MatchBatchResult(
                visible_by_frame=visible,
                decisions_by_frame={key: list(value) for key, value in visible.items()},
            )

        yolo_results = self.detector.predict(frames)
        crops: list[Image.Image] = []
        crop_color_signatures: list[np.ndarray] = []
        proposals: list[dict[str, Any]] = []

        for frame, frame_index, yolo_result in zip(frames, frame_indices, yolo_results):
            boxes = getattr(yolo_result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            names = getattr(yolo_result, "names", {}) or {}

            for box, conf, cls_id in zip(xyxy, confs, classes):
                crop = self._crop(frame, box)
                if crop is None:
                    continue
                crops.append(crop)
                crop_color_signatures.append(_mean_rgb_signature(crop))
                proposals.append(
                    {
                        "frame_index": int(frame_index),
                        "xyxy": [float(v) for v in box.tolist()],
                        "detector_conf": float(conf),
                        "class_id": int(cls_id),
                        "class_name": str(names.get(int(cls_id), cls_id)),
                    }
                )

        crop_features = self.embedder.encode_images(
            crops, batch_size=self.embedding_batch_size
        )
        if crop_features is None:
            return MatchBatchResult(empty, {key: [] for key in empty})

        return self.matcher.match_batch(
            crop_features=crop_features,
            crop_color_signatures=crop_color_signatures,
            proposals=proposals,
            frame_indices=frame_indices,
            sample_stride=sample_stride,
        )

    def _crop(self, frame: np.ndarray, box: np.ndarray) -> Image.Image | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        pad_x = box_width * self.config.box_padding
        pad_y = box_height * self.config.box_padding
        ix1 = max(0, int(round(x1 - pad_x)))
        iy1 = max(0, int(round(y1 - pad_y)))
        ix2 = min(width, int(round(x2 + pad_x)))
        iy2 = min(height, int(round(y2 + pad_y)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return None
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
