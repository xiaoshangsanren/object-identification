from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cli import parse_folds
from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config
from .dino_engine import run_dino_variant


DEFAULT_CONFIG = PAD_LITE_ROOT / "configs" / "dino_default.json"


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        创建并配置当前命令行程序的参数解析器。
    
    输入参数：
        无。
    
    返回值：
        argparse.ArgumentParser：方法执行得到的结果。
    """
    parser = argparse.ArgumentParser(
        prog="python -m PAD_Lite.dino_cli",
        description="Run DINOv2-based PAD-Lite B0/B1/B2 five-fold experiments.",
    )
    parser.add_argument("variant", choices=("b0", "b1", "b2"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=parse_folds, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Override data.image_size. DINOv2-small uses 14x14 patches, so "
            "resolution experiments should use a positive multiple of 14."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override data.eval_batch_size without changing training batches.",
    )
    parser.add_argument(
        "--resize-short-edge",
        type=int,
        default=None,
        help=(
            "Override data.resize_short_edge for center-crop evaluation. "
            "It is ignored by the letterbox input mode."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=None,
        help=(
            "Override model.unfreeze_last_blocks. Use 0 to train only the "
            "retrieval head; DINOv2-small accepts 0..12."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析命令行参数并执行当前脚本的主流程。
    
    输入参数：
        argv (list[str] | None)：待解析的命令行参数；为 None 时读取系统命令行。
    
    返回值：
        int：方法执行得到的结果。
    """
    args = build_parser().parse_args(argv)
    if args.variant == "b0" and (args.resume or args.eval_only or args.epochs):
        raise ValueError("DINO-B0 is frozen and has no training options")
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config[args.variant]["epochs"] = args.epochs
        config[args.variant]["warmup_epochs"] = min(
            int(config[args.variant]["warmup_epochs"]), args.epochs - 1
        )
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("--workers cannot be negative")
        config["data"]["workers"] = args.workers
    if args.image_size is not None:
        if args.image_size < 14 or args.image_size % 14:
            raise ValueError("DINO --image-size must be a positive multiple of 14")
        config["data"]["image_size"] = args.image_size
    if args.eval_batch_size is not None:
        if args.eval_batch_size < 1:
            raise ValueError("--eval-batch-size must be positive")
        config["data"]["eval_batch_size"] = args.eval_batch_size
    if args.resize_short_edge is not None:
        if args.resize_short_edge < 1:
            raise ValueError("--resize-short-edge must be positive")
        if args.resize_short_edge < int(config["data"]["image_size"]):
            raise ValueError("--resize-short-edge cannot be smaller than --image-size")
        config["data"]["resize_short_edge"] = args.resize_short_edge
    if args.unfreeze_last_blocks is not None:
        if args.variant == "b0":
            raise ValueError("--unfreeze-last-blocks does not apply to DINO-B0")
        if args.unfreeze_last_blocks not in range(13):
            raise ValueError("DINO --unfreeze-last-blocks must be in 0..12")
        config["model"]["unfreeze_last_blocks"] = args.unfreeze_last_blocks
    if args.output_root is not None:
        output = args.output_root.expanduser()
        config["paths"]["output_root"] = (
            output.resolve()
            if output.is_absolute()
            else (ANNOTATION_ROOT / output).resolve()
        )
    result = run_dino_variant(
        config,
        variant=args.variant,
        folds=args.fold,
        device_name=args.device,
        resume=args.resume,
        eval_only=args.eval_only,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
