from __future__ import annotations

from typing import Any

from ..config import (
    DINO_GALLERY_TEMPORAL_ALGORITHM,
    DetectionConfig,
    GALLERY_CLIP_ALGORITHM,
    LEGACY_CLIP_ALGORITHM,
    PAD_LITE_ALGORITHMS,
)
from .base import MatchStrategy
from .gallery_temporal import DinoGalleryTemporalMatcher, GalleryTemporalMatcher
from .legacy import LegacyClipMatcher


def create_match_strategy(config: DetectionConfig, embedder: Any) -> MatchStrategy:
    algorithm = str(config.algorithm or LEGACY_CLIP_ALGORITHM).strip()
    if algorithm == LEGACY_CLIP_ALGORITHM:
        return LegacyClipMatcher(config, embedder)
    if algorithm == GALLERY_CLIP_ALGORITHM or algorithm in PAD_LITE_ALGORITHMS:
        return GalleryTemporalMatcher(config, embedder)
    if algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM:
        return DinoGalleryTemporalMatcher(config, embedder)
    raise ValueError(f"不支持的识别算法: {algorithm}")
