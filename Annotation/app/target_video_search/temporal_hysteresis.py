from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalAnchor:
    track_id: int
    subtype_id: str
    subtype_label: str
    start_sec: float
    end_sec: float
    feature: np.ndarray


@dataclass(slots=True)
class _DecisionRef:
    time_sec: float
    decision: dict[str, Any]


def _normalized_mean(features: list[np.ndarray]) -> np.ndarray:
    centroid = np.stack(features, axis=0).mean(axis=0).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-8:
        return centroid
    return centroid / norm


def _cosine(feature: np.ndarray, anchor: np.ndarray) -> float:
    left = np.asarray(feature, dtype=np.float32)
    right = np.asarray(anchor, dtype=np.float32)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _anchor_distance(anchor: TemporalAnchor, time_sec: float) -> float:
    if time_sec < anchor.start_sec:
        return anchor.start_sec - time_sec
    if time_sec > anchor.end_sec:
        return time_sec - anchor.end_sec
    return 0.0


def _is_direct_anchor_candidate(decision: dict[str, Any]) -> bool:
    return (
        decision.get("gallery_status") == "known"
        and not bool(decision.get("temporal_propagated"))
        and decision.get("subtype_id") not in {None, "Unknown"}
        and isinstance(decision.get("_embedding"), np.ndarray)
    )


def build_temporal_anchors(
    decision_results: list[dict[str, Any]],
    min_frames: int,
) -> list[TemporalAnchor]:
    """Build anchors from uninterrupted direct-known runs within each track."""
    by_track: dict[int, list[_DecisionRef]] = {}
    for frame_item in decision_results:
        time_sec = float(frame_item["time_sec"])
        for decision in frame_item.get("decisions", []):
            track_id = int(decision.get("track_id", -1))
            by_track.setdefault(track_id, []).append(_DecisionRef(time_sec, decision))

    required = max(1, int(min_frames))
    anchors: list[TemporalAnchor] = []

    def finish_run(track_id: int, run: list[_DecisionRef]) -> None:
        if len(run) < required:
            return
        embeddings = [
            np.asarray(item.decision["_embedding"], dtype=np.float32) for item in run
        ]
        anchors.append(
            TemporalAnchor(
                track_id=track_id,
                subtype_id=str(run[0].decision["subtype_id"]),
                subtype_label=str(run[0].decision["subtype_label"]),
                start_sec=run[0].time_sec,
                end_sec=run[-1].time_sec,
                feature=_normalized_mean(embeddings),
            )
        )

    for track_id, refs in by_track.items():
        refs.sort(key=lambda item: item.time_sec)
        run: list[_DecisionRef] = []
        run_label: str | None = None
        for ref in refs:
            decision = ref.decision
            label = str(decision.get("subtype_id"))
            if _is_direct_anchor_candidate(decision) and (
                run_label is None or label == run_label
            ):
                run.append(ref)
                run_label = label
                continue
            finish_run(track_id, run)
            run = [ref] if _is_direct_anchor_candidate(decision) else []
            run_label = label if run else None
        finish_run(track_id, run)

    return sorted(anchors, key=lambda item: (item.start_sec, item.track_id))


def _eligible_for_propagation(decision: dict[str, Any]) -> bool:
    if decision.get("gallery_status") == "rejected_background":
        return False
    if (
        decision.get("gallery_status") == "unknown"
        and decision.get("decision_reason") == "min_similarity"
    ):
        return False
    return decision.get("gallery_status") in {"known", "unknown"}


def _apply_anchor_result(
    decision: dict[str, Any],
    anchors: list[TemporalAnchor],
    similarities: list[float],
    distance_sec: float,
) -> None:
    anchor = anchors[0]
    decision["original_gallery_status"] = decision.get("gallery_status")
    decision["original_decision_reason"] = decision.get("decision_reason")
    decision["gallery_status"] = "known"
    decision["decision_reason"] = "temporal_hysteresis"
    decision["subtype_id"] = anchor.subtype_id
    decision["subtype_label"] = anchor.subtype_label
    decision["temporal_propagated"] = True
    decision["temporal_anchor_track_ids"] = sorted(
        {int(anchor.track_id) for anchor in anchors}
    )
    decision["temporal_anchor_distance_sec"] = float(distance_sec)
    decision["temporal_anchor_similarity"] = float(min(similarities))


def apply_dino_temporal_hysteresis(
    decision_results: list[dict[str, Any]],
    *,
    max_span_sec: float,
    anchor_min_frames: int,
    anchor_similarity: float,
) -> list[TemporalAnchor]:
    """Apply bidirectional anchor propagation in-place and return the anchors used."""
    span = max(0.0, float(max_span_sec))
    similarity_threshold = float(anchor_similarity)
    for frame_item in decision_results:
        for decision in frame_item.get("decisions", []):
            decision.setdefault("low_confidence", False)
            decision.setdefault("temporal_propagated", False)

    anchors = build_temporal_anchors(decision_results, anchor_min_frames)
    if not anchors or span <= 0:
        return anchors

    for frame_item in decision_results:
        time_sec = float(frame_item["time_sec"])
        for decision in frame_item.get("decisions", []):
            if not _eligible_for_propagation(decision):
                continue
            feature = decision.get("_embedding")
            if not isinstance(feature, np.ndarray):
                continue

            nearby = [
                anchor
                for anchor in anchors
                if _anchor_distance(anchor, time_sec) <= span
            ]
            same_track = [
                anchor
                for anchor in nearby
                if anchor.track_id == int(decision.get("track_id", -1))
            ]
            selected: list[TemporalAnchor] = []

            if same_track:
                if any(
                    anchor.start_sec <= time_sec <= anchor.end_sec
                    for anchor in same_track
                ):
                    continue
                previous = [anchor for anchor in same_track if anchor.end_sec < time_sec]
                following = [anchor for anchor in same_track if anchor.start_sec > time_sec]
                if previous:
                    selected = [max(previous, key=lambda item: item.end_sec)]
                elif following:
                    selected = [min(following, key=lambda item: item.start_sec)]
                else:
                    continue
            else:
                previous = [anchor for anchor in nearby if anchor.end_sec <= time_sec]
                following = [anchor for anchor in nearby if anchor.start_sec >= time_sec]
                if not previous or not following:
                    continue
                before = max(previous, key=lambda item: item.end_sec)
                after = min(following, key=lambda item: item.start_sec)
                if before.subtype_id != after.subtype_id:
                    continue
                conflicts = [
                    anchor
                    for anchor in anchors
                    if anchor.subtype_id != before.subtype_id
                    and anchor.end_sec >= before.end_sec
                    and anchor.start_sec <= after.start_sec
                ]
                if conflicts:
                    continue
                selected = [before, after]

            similarities = [_cosine(feature, anchor.feature) for anchor in selected]
            if not similarities or min(similarities) < similarity_threshold:
                continue
            distance_sec = max(_anchor_distance(anchor, time_sec) for anchor in selected)
            _apply_anchor_result(decision, selected, similarities, distance_sec)

    return anchors


def strip_internal_decision_fields(decision: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in decision.items() if not key.startswith("_")}
