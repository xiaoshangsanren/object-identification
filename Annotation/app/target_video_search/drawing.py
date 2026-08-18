from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def draw_matches(frame: np.ndarray, matches: Iterable[dict], frame_index: int) -> np.ndarray:
    annotated = frame.copy()
    cv2.putText(
        annotated,
        f"frame {frame_index}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    for item in matches:
        x1, y1, x2, y2 = [int(v) for v in item["xyxy"]]
        score = float(item["score"])
        gallery_status = item.get("gallery_status")
        if gallery_status in {"known", "unknown"}:
            suffix = ""
            if gallery_status == "known" and item.get("temporal_propagated"):
                suffix = " (temporal)"
            elif gallery_status == "known" and item.get("low_confidence"):
                suffix = " (low)"
            label = f"{item.get('subtype_label', 'Unknown')}{suffix} {score:.2f}"
            color = (50, 205, 80) if gallery_status == "known" else (0, 165, 255)
            text_color = (10, 24, 36)
        else:
            cls_name = item.get("class_name", "target")
            label = f"{cls_name} sim {score:.2f}"
            is_primary_highlight = bool(item.get("is_primary_highlight"))
            color = (0, 0, 255) if is_primary_highlight else (0, 210, 255)
            text_color = (245, 245, 245) if is_primary_highlight else (10, 24, 36)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, y_text),
            (min(x1 + tw + 10, annotated.shape[1] - 1), y_text + th + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y_text + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA,
        )
    return annotated
