from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from ..config import DetectionConfig, LEGACY_CLIP_ALGORITHM
from .base import MatchBatchResult


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def mean_rgb_signature(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    if array.size == 0:
        return np.zeros(3, dtype=np.float32)
    return array.reshape(-1, 3).mean(axis=0)


def color_similarity(
    target_signature: np.ndarray | None,
    candidate_signature: np.ndarray | None,
) -> float:
    if target_signature is None or candidate_signature is None:
        return 0.0
    distance = float(np.linalg.norm(target_signature - candidate_signature))
    max_distance = float(np.sqrt(3.0 * (255.0**2)))
    return 1.0 - min(max(distance / max_distance, 0.0), 1.0)


def match_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item.get("score", 0.0)),
        float(item.get("color_similarity", 0.0)),
        float(item.get("image_score", 0.0)),
        float(item.get("detector_conf", 0.0)),
    )


def mark_primary_match(
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
    top_match = max(tied_top_matches, key=match_sort_key)
    if float(top_match.get("score", 0.0)) >= float(threshold):
        top_match["is_primary_highlight"] = True
        chosen_index = matches.index(top_match)
        if chosen_index != 0:
            matches[0], matches[chosen_index] = matches[chosen_index], matches[0]


class LegacyClipMatcher:
    algorithm = LEGACY_CLIP_ALGORITHM

    def __init__(self, config: DetectionConfig, embedder: Any) -> None:
        self.config = config
        self.embedder = embedder
        self.target_image_features = None
        self.target_text_features = None
        self.target_color_signature: np.ndarray | None = None

    def prepare(
        self,
        target_image: Image.Image | None = None,
        description: str | None = None,
    ) -> None:
        if target_image is None:
            raise ValueError("CLIP 模式必须提供目标图片。")
        self.target_image_features = self.embedder.encode_images(
            [target_image], batch_size=1
        )
        self.target_text_features = self.embedder.encode_text(description)
        self.target_color_signature = mean_rgb_signature(target_image)

    def match_batch(
        self,
        crop_features: Any,
        crop_color_signatures: list[np.ndarray],
        proposals: list[dict[str, Any]],
        frame_indices: list[int],
        sample_stride: int,
    ) -> MatchBatchResult:
        del sample_stride
        results = {int(index): [] for index in frame_indices}
        if crop_features is None or self.target_image_features is None:
            return MatchBatchResult(results, {key: [] for key in results})

        image_scores = (crop_features @ self.target_image_features.T).squeeze(1)
        text_scores = None
        combined_scores = image_scores
        if (
            self.target_text_features is not None
            and self.config.text_weight > 0
            and crop_features.shape[-1] == self.target_text_features.shape[-1]
        ):
            text_scores = (crop_features @ self.target_text_features.T).squeeze(1)
            weight = max(0.0, min(1.0, self.config.text_weight))
            combined_scores = (1.0 - weight) * image_scores + weight * text_scores

        combined_np = combined_scores.detach().cpu().numpy()
        image_np = image_scores.detach().cpu().numpy()
        text_np = (
            np.full_like(image_np, np.nan, dtype=np.float32)
            if text_scores is None
            else text_scores.detach().cpu().numpy()
        )
        color_np = np.asarray(
            [
                color_similarity(self.target_color_signature, signature)
                for signature in crop_color_signatures
            ],
            dtype=np.float32,
        )

        for proposal, score, image_score, text_score, candidate_color in zip(
            proposals, combined_np, image_np, text_np, color_np
        ):
            if float(score) < self.config.similarity_threshold:
                continue
            item = {
                **proposal,
                "score": float(score),
                "image_score": float(image_score),
                "text_score": safe_float(text_score),
                "color_similarity": float(candidate_color),
            }
            results[item["frame_index"]].append(item)

        for frame_index in results:
            results[frame_index].sort(key=match_sort_key, reverse=True)
            mark_primary_match(
                results[frame_index],
                self.config.highlight_similarity_threshold,
            )
        return MatchBatchResult(results, {key: list(value) for key, value in results.items()})

    def summary_info(self) -> dict[str, Any] | None:
        return None
