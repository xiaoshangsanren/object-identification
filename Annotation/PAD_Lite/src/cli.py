from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config
from .engine import run_variant


DEFAULT_CONFIG = PAD_LITE_ROOT / "configs" / "default.json"


def parse_folds(value: str) -> list[int]:
    """
    方法作用：
        将命令行中的折编号参数解析为 1~5 的整数列表。

    输入参数：
        value (str):
            例如 "all"、"1" 或 "1,3,5" 的折参数字符串。

    返回值：
        list[int]:
            解析后的折编号列表。
    """
    if value.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds = []
    for item in value.split(","):
        fold = int(item.strip())
        if fold not in range(1, 6):
            raise argparse.ArgumentTypeError("folds must be all or comma-separated 1..5")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise argparse.ArgumentTypeError("at least one fold is required")
    return folds


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
        prog="python -m PAD_Lite",
        description="Run Annotation PAD-Lite B0/B1/B2 five-fold experiments.",
    )
    parser.add_argument("variant", choices=("b0", "b1", "b2"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fold",
        type=parse_folds,
        default=[1, 2, 3, 4, 5],
        help="all, a single fold such as 1, or a list such as 1,3,5",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume B1/B2 from fold last.pt when it exists.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate B1/B2 best.pt without further training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the selected B1/B2 epoch count for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override DataLoader worker count.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the experiment output directory.",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=None,
        help=(
            "Override model.unfreeze_last_blocks. Use 0 to train only the "
            "retrieval head; CLIP ViT-B/32 accepts 0..12."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
        方法作用：
            解析命令行参数，加载实验配置，并根据参数对训练轮数、
            DataLoader 工作进程数和输出目录等配置进行覆盖。
            随后运行指定的 PAD-Lite B0、B1 或 B2 五折实验，
            并以 JSON 格式输出实验结果摘要。
        输入参数：
            argv (list[str] | None):
                要解析的命令行参数列表。
                当值为 None 时,argparse 会自动读取 sys.argv 中的参数；
                传入字符串列表时，则解析该列表，通常用于测试。
        返回值：
            int:
                执行成功时返回状态码 0。
                参数或实验执行出现异常时，由程序入口捕获异常并以状态码 1 退出。
    """
    # 创建命令行解释器
    args = build_parser().parse_args(argv)
    # 检查参数组合是否合法。
    if args.variant == "b0" and (args.resume or args.eval_only):
        raise ValueError("B0 has no training checkpoint; --resume/--eval-only do not apply")
    # 读取 --config 指定的 JSON 配置文件，并将配置内容保存到 config。
    config = load_config(args.config)
    # 是否设置epochs参数，如果设置了则覆盖配置文件中的训练轮数和预热轮数。
    if args.epochs is not None:
        # b0 模型不支持设置训练轮数，因此如果指定了 --epochs 参数则抛出异常。
        if args.variant == "b0":
            raise ValueError("--epochs does not apply to B0")
        # 检查指定的训练轮数是否为正数，如果不是则抛出异常。
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        # 将指定的训练轮数覆盖配置文件中的训练轮数。
        config[args.variant]["epochs"] = args.epochs
        # 将预热轮数设置为指定训练轮数减 1，确保预热轮数不超过训练轮数。
        config[args.variant]["warmup_epochs"] = min(
            int(config[args.variant]["warmup_epochs"]), args.epochs - 1
        )
    # 检查是否设置了 --workers 参数，如果设置了则覆盖配置文件中的 DataLoader 工作进程数。
    if args.workers is not None:
        # 检查指定的工作进程数是否为负数，如果是则抛出异常。
        if args.workers < 0:
            raise ValueError("--workers cannot be negative")
        # 将指定的工作进程数覆盖配置文件中的 DataLoader 工作进程数。
        config["data"]["workers"] = args.workers
    if args.unfreeze_last_blocks is not None:
        if args.variant == "b0":
            raise ValueError("--unfreeze-last-blocks does not apply to B0")
        if args.unfreeze_last_blocks not in range(13):
            raise ValueError("CLIP --unfreeze-last-blocks must be in 0..12")
        config["model"]["unfreeze_last_blocks"] = args.unfreeze_last_blocks
    # 检查是否设置了 --output-root 参数，如果设置了则覆盖配置文件中的输出目录。
    if args.output_root is not None:
        # 将指定的输出目录路径进行展开，处理用户主目录符号（~）。
        output = args.output_root.expanduser()
        # 检查指定的输出目录是否为文件，如果是则抛出异常。
        config["paths"]["output_root"] = (
            output.resolve()
            if output.is_absolute()
            else (ANNOTATION_ROOT / output).resolve()
        )

    # 调用 run_variant 函数运行指定的 PAD-Lite B0/B1/B2 五折实验，并将结果保存到 result。
    result = run_variant(
        config,
        variant=args.variant,
        folds=args.fold,
        device_name=args.device,
        resume=args.resume,
        eval_only=args.eval_only,
    )

    # 将实验结果摘要以 JSON 格式输出到标准输出，确保非 ASCII 字符正确显示，并使用缩进格式化。
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
