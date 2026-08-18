from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(slots=True)
class MatchBatchResult:
    visible_by_frame: dict[int, list[dict[str, Any]]]
    decisions_by_frame: dict[int, list[dict[str, Any]]]


class MatchStrategy(Protocol):
    algorithm: str

    def prepare(
        self,
        target_image: Image.Image | None = None,
        description: str | None = None,
    ) -> None: ...

    def match_batch(
        self,
        crop_features: Any,
        crop_color_signatures: list[np.ndarray],
        proposals: list[dict[str, Any]],
        frame_indices: list[int],
        sample_stride: int,
    ) -> MatchBatchResult: ...

    def summary_info(self) -> dict[str, Any] | None: ...
