from __future__ import annotations

import argparse
import json
from pathlib import Path

from target_video_search.config import (
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
    DEFAULT_PAD_LITE_CHECKPOINT_ROOT,
    DINO_GALLERY_TEMPORAL_ALGORITHM,
    DetectionConfig,
    GALLERY_CLIP_ALGORITHM,
    LEGACY_CLIP_ALGORITHM,
    PAD_LITE_B0_ALGORITHM,
    PAD_LITE_B1_ALGORITHM,
    PAD_LITE_B2_ALGORITHM,
    TEXT_GROUNDING_TEMPORAL_ALGORITHM,
    parse_class_ids,
    parse_target_box,
)
from target_video_search.pipeline import TargetVideoAnnotator


def validate_target_image_argument(algorithm: str, target_image: str | None) -> None:
    if algorithm == LEGACY_CLIP_ALGORITHM and not str(target_image or "").strip():
        raise ValueError("--target-image is required when --algorithm CLIP is selected")


def validate_grounding_prompt_argument(algorithm: str, prompt: str | None) -> None:
    if (
        algorithm == TEXT_GROUNDING_TEMPORAL_ALGORITHM
        and not str(prompt or "").strip()
    ):
        raise ValueError(
            "--grounding-prompt is required when --algorithm text_grounding_temporal is selected"
        )


def resolve_gallery_parameters(
    algorithm: str,
    min_similarity: float | None,
    class_margin: float | None,
    negative_margin: float | None,
    background_margin: float | None,
    unknown_margin: float | None,
    known_bias: float | None,
) -> tuple[float, float, float, float, float]:
    dino_selected = algorithm == DINO_GALLERY_TEMPORAL_ALGORITHM
    resolved_background_margin = (
        background_margin
        if background_margin is not None
        else (negative_margin if negative_margin is not None else 0.01)
    )
    resolved_unknown_margin = (
        unknown_margin
        if unknown_margin is not None
        else (
            negative_margin
            if negative_margin is not None
            else (-0.02 if dino_selected else 0.01)
        )
    )
    return (
        float(min_similarity if min_similarity is not None else (0.0 if dino_selected else 0.50)),
        float(class_margin if class_margin is not None else 0.01),
        float(resolved_background_margin),
        float(resolved_unknown_margin),
        float(known_bias if known_bias is not None else (0.0 if dino_selected else 0.02)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a target vehicle in a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--algorithm",
        choices=[
            LEGACY_CLIP_ALGORITHM,
            GALLERY_CLIP_ALGORITHM,
            DINO_GALLERY_TEMPORAL_ALGORITHM,
            PAD_LITE_B0_ALGORITHM,
            PAD_LITE_B1_ALGORITHM,
            PAD_LITE_B2_ALGORITHM,
            TEXT_GROUNDING_TEMPORAL_ALGORITHM,
        ],
        default=LEGACY_CLIP_ALGORITHM,
    )
    parser.add_argument("--target-image", default="", help="Reference target image path.")
    parser.add_argument("--description", default="", help="Optional text description.")
    parser.add_argument("--output", default="", help="Output annotated MP4 path.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--proposal-conf", type=float, default=0.30)
    parser.add_argument("--classes", default="2,3,5,7")
    parser.add_argument("--target-box", default="", help="Optional target box x1,y1,x2,y2")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    parser.add_argument("--clip-backend", default=DEFAULT_CLIP_BACKEND)
    parser.add_argument("--clip-local-dir", default=DEFAULT_CLIP_LOCAL_DIR)
    parser.add_argument("--similarity-threshold", type=float, default=0.24)
    parser.add_argument(
        "--highlight-similarity-threshold",
        type=float,
        default=DEFAULT_HIGHLIGHT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument("--text-weight", type=float, default=0.25)
    parser.add_argument("--gallery-root", default=DEFAULT_CLIP_GALLERY_ROOT)
    parser.add_argument("--gallery-min-similarity", type=float, default=None)
    parser.add_argument("--gallery-class-margin", type=float, default=None)
    parser.add_argument("--gallery-negative-margin", type=float, default=None)
    parser.add_argument("--gallery-background-margin", type=float, default=None)
    parser.add_argument("--gallery-unknown-margin", type=float, default=None)
    parser.add_argument("--gallery-known-bias", type=float, default=None)
    parser.add_argument("--gallery-temporal-window", type=int, default=5)
    parser.add_argument("--gallery-track-iou", type=float, default=0.30)
    parser.add_argument("--target-sample-fps", type=float, default=15.0)
    parser.add_argument("--frame-stride", type=int, default=0)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    parser.add_argument("--clip-batch-size", type=int, default=128)
    parser.add_argument("--dino-local-dir", default=DEFAULT_DINO_LOCAL_DIR)
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument(
        "--pad-lite-checkpoint-root",
        default=DEFAULT_PAD_LITE_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--no-dino-background-filter",
        action="store_true",
        help="Keep DINO YOLO proposals instead of rejecting them with the background bank.",
    )
    parser.add_argument(
        "--no-dino-hysteresis",
        action="store_true",
        help="Disable DINO temporal hysteresis and use single-pass low-confidence classification.",
    )
    parser.add_argument("--dino-hysteresis-max-span-sec", type=float, default=10.0)
    parser.add_argument("--dino-hysteresis-anchor-min-frames", type=int, default=3)
    parser.add_argument("--dino-hysteresis-anchor-similarity", type=float, default=0.80)
    parser.add_argument("--grounding-model-dir", default=DEFAULT_GROUNDING_MODEL_DIR)
    parser.add_argument("--grounding-prompt", default=DEFAULT_GROUNDING_PROMPT)
    parser.add_argument("--grounding-box-threshold", type=float, default=0.30)
    parser.add_argument("--grounding-text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-sample-fps", type=float, default=2.0)
    parser.add_argument("--grounding-track-iou", type=float, default=0.30)
    parser.add_argument("--grounding-max-gap-samples", type=int, default=2)
    parser.add_argument("--no-hold-boxes", action="store_true")
    parser.add_argument("--no-nvenc", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()
    try:
        validate_target_image_argument(args.algorithm, args.target_image)
        validate_grounding_prompt_argument(args.algorithm, args.grounding_prompt)
    except ValueError as exc:
        parser.error(str(exc))

    (
        gallery_min_similarity,
        gallery_class_margin,
        gallery_background_margin,
        gallery_unknown_margin,
        gallery_known_bias,
    ) = resolve_gallery_parameters(
        args.algorithm,
        args.gallery_min_similarity,
        args.gallery_class_margin,
        args.gallery_negative_margin,
        args.gallery_background_margin,
        args.gallery_unknown_margin,
        args.gallery_known_bias,
    )

    config = DetectionConfig(
        algorithm=args.algorithm,
        detector_model=args.detector_model,
        detector_imgsz=args.detector_imgsz,
        proposal_conf=args.proposal_conf,
        class_ids=parse_class_ids(args.classes),
        target_box=(
            parse_target_box(args.target_box)
            if args.algorithm == LEGACY_CLIP_ALGORITHM
            else None
        ),
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_backend=args.clip_backend,
        clip_local_dir=args.clip_local_dir,
        similarity_threshold=args.similarity_threshold,
        highlight_similarity_threshold=args.highlight_similarity_threshold,
        text_weight=args.text_weight,
        gallery_root=args.gallery_root,
        gallery_min_similarity=gallery_min_similarity,
        gallery_class_margin=gallery_class_margin,
        gallery_background_margin=gallery_background_margin,
        gallery_unknown_margin=gallery_unknown_margin,
        gallery_known_bias=gallery_known_bias,
        gallery_temporal_window=max(1, args.gallery_temporal_window),
        gallery_track_iou=args.gallery_track_iou,
        target_sample_fps=args.target_sample_fps,
        frame_stride=args.frame_stride,
        yolo_batch_size=args.yolo_batch_size,
        clip_batch_size=args.clip_batch_size,
        dino_local_dir=args.dino_local_dir,
        dino_batch_size=max(1, args.dino_batch_size),
        pad_lite_checkpoint_root=args.pad_lite_checkpoint_root,
        dino_background_filter_enabled=not args.no_dino_background_filter,
        dino_hysteresis_enabled=not args.no_dino_hysteresis,
        dino_hysteresis_max_span_sec=max(0.0, args.dino_hysteresis_max_span_sec),
        dino_hysteresis_anchor_min_frames=max(
            1, args.dino_hysteresis_anchor_min_frames
        ),
        dino_hysteresis_anchor_similarity=float(
            args.dino_hysteresis_anchor_similarity
        ),
        grounding_model_dir=args.grounding_model_dir,
        grounding_prompt=args.grounding_prompt,
        grounding_box_threshold=float(args.grounding_box_threshold),
        grounding_text_threshold=float(args.grounding_text_threshold),
        grounding_sample_fps=max(0.1, float(args.grounding_sample_fps)),
        grounding_track_iou=float(args.grounding_track_iou),
        grounding_max_gap_samples=max(0, int(args.grounding_max_gap_samples)),
        hold_boxes=not args.no_hold_boxes,
        use_nvenc=not args.no_nvenc,
        device=args.device,
        half=not args.no_half,
        output_dir=Path(args.output_dir),
    )
    annotator = TargetVideoAnnotator(config)

    def progress(frame_count: int, total_frames: int, message: str) -> None:
        if total_frames:
            pct = frame_count / total_frames * 100
            print(f"\r{message}: {frame_count}/{total_frames} ({pct:.1f}%)", end="")
        else:
            print(f"\r{message}: {frame_count}", end="")

    summary = annotator.process_video(
        video_path=args.video,
        target_image_path=args.target_image or None,
        output_path=args.output or None,
        description=args.description,
        progress_callback=progress,
    )
    print()
    compact = {
        key: summary[key]
        for key in [
            "output_video",
            "output_json",
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
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
