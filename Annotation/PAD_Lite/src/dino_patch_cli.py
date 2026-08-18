from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cli import parse_folds
from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config
from .dino_patch_engine import run_p1a


DEFAULT_CONFIG = (
    PAD_LITE_ROOT
    / "configs"
    / "dino_patch_safe_letterbox_336.json"
)
P1B_DEFAULT_CONFIG = (
    PAD_LITE_ROOT
    / "configs"
    / "dino_patch_p1b_equal_fusion_336.json"
)
P2A_DEFAULT_CONFIG = (
    PAD_LITE_ROOT
    / "configs"
    / "dino_patch_p2a_weighted_letterbox_336.json"
)
P2B_DEFAULT_CONFIG = (
    PAD_LITE_ROOT
    / "configs"
    / "dino_patch_p2b_equal_fusion_336.json"
)


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        构建 P1a/P1b/P2a/P2b 兼容命令行参数解析器。
    输入参数：
        无。
    返回值：
        argparse.ArgumentParser：包含实验版本、折、设备、恢复及输出参数的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="python -m PAD_Lite.dino_patch_cli",
        description="Run PAD-Lite DINO Patch-Safe ablations.",
    )
    parser.add_argument("variant", choices=("p1a", "p1b", "p2a", "p2b"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--fold", type=parse_folds, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析 Patch 实验参数、加载对应配置并分派到 P1a/P1b/P2a/P2b 引擎。
    输入参数：
        argv (list[str]|None)：命令行参数列表；None 时读取系统命令行。
    返回值：
        int：执行成功返回 0；配置或运行异常由上层入口处理。
    """
    args = build_parser().parse_args(argv)
    default_configs = {
        "p1a": DEFAULT_CONFIG,
        "p1b": P1B_DEFAULT_CONFIG,
        "p2a": P2A_DEFAULT_CONFIG,
        "p2b": P2B_DEFAULT_CONFIG,
    }
    config_path = args.config or default_configs[args.variant]
    config = load_config(config_path)
    if args.variant not in config:
        raise ValueError(f"Patch config requires a {args.variant} section")
    if args.epochs is not None:
        if args.variant in {"p1b", "p2b"}:
            raise ValueError(
                f"--epochs does not apply to {args.variant.upper()} cached-feature fusion"
            )
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config[args.variant]["epochs"] = args.epochs
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("--workers cannot be negative")
        config["data"]["workers"] = args.workers
    if args.output_root is not None:
        output = args.output_root.expanduser()
        config["paths"]["output_root"] = (
            output.resolve()
            if output.is_absolute()
            else (ANNOTATION_ROOT / output).resolve()
        )
    if args.variant == "p1a":
        result = run_p1a(
            config,
            folds=args.fold,
            device_name=args.device,
            resume=args.resume,
            eval_only=args.eval_only,
        )
    elif args.variant == "p1b":
        if args.resume or args.eval_only:
            raise ValueError("P1b has no checkpoint; run it as a new evaluation")
        from .dino_patch_fusion import run_p1b

        result = run_p1b(config, folds=args.fold, device_name=args.device)
    elif args.variant == "p2a":
        from .dino_patch_weighted_engine import run_p2a

        result = run_p2a(
            config,
            folds=args.fold,
            device_name=args.device,
            resume=args.resume,
            eval_only=args.eval_only,
        )
    else:
        if args.resume or args.eval_only:
            raise ValueError("P2b has no checkpoint; run it as a new evaluation")
        from .dino_patch_weighted_fusion import run_p2b

        result = run_p2b(config, folds=args.fold, device_name=args.device)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
