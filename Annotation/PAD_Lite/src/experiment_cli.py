from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, serializable_config
from .experiment_registry import (
    ExperimentSpec,
    get_experiment,
    load_registry,
    resolve_experiment_config,
)


DEFAULT_RUNS_ROOT = (
    PAD_LITE_ROOT
    / "outputs"
    / "03_p2b_patch_reranking"
    / "experiment_runs"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_folds(value: str) -> list[int]:
    """
    方法作用：
        将 all 或逗号分隔文本解析为去重后的 1～5 折列表。
    输入参数：
        value (str)：例如 all、1 或 1,3,5。
    返回值：
        list[int]：按输入顺序排列的折号。
    """

    if value.strip().lower() == "all":
        return [1, 2, 3, 4, 5]
    folds: list[int] = []
    for item in value.split(","):
        try:
            fold = int(item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "folds must be all or comma-separated 1..5"
            ) from exc
        if fold not in range(1, 6):
            raise argparse.ArgumentTypeError(
                "folds must be all or comma-separated 1..5"
            )
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise argparse.ArgumentTypeError("at least one fold is required")
    return folds


def _write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    """
    方法作用：
        写入格式化 JSON，并按 overwrite 策略防止覆盖实验元数据。
    输入参数：
        path：目标路径；payload：可序列化对象；overwrite：是否允许覆盖。
    返回值：
        None：写入完成后无返回数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing metadata: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_hash(payload: Any) -> str:
    """
    方法作用：
        对规范化 JSON 内容计算可复现 SHA-256。
    输入参数：
        payload (Any)：可 JSON 序列化对象。
    返回值：
        str：十六进制 SHA-256。
    """
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    """
    方法作用：
        流式计算文件 SHA-256。
    输入参数：
        path (Path)：文件路径。
    返回值：
        str：十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp() -> str:
    """
    方法作用：
        生成带微秒的 UTC 时间戳字符串。
    输入参数：
        无。
    返回值：
        str：YYYYMMDDTHHMMSS_microsecondsZ 格式时间戳。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _new_run_id() -> str:
    """
    方法作用：
        为未显式命名的新实验生成唯一 Run ID。
    输入参数：
        无。
    返回值：
        str：基于 UTC 时间戳的 Run ID。
    """
    return _utc_timestamp()


def _resolve_user_path(path: Path) -> Path:
    """
    方法作用：
        将用户路径展开并按 Annotation 根目录解析。
    输入参数：
        path (Path)：绝对或相对路径。
    返回值：
        Path：绝对路径。
    """
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (ANNOTATION_ROOT / expanded).resolve()
    )


def _resolve_run_root(
    spec: ExperimentSpec,
    run_id: str | None,
    output_root: Path | None,
) -> tuple[Path, str]:
    """
    方法作用：
        根据实验、可选 Run ID 和输出覆盖路径确定本次 Run 根目录。
    输入参数：
        spec：实验预设；run_id：可选标识；output_root：可选显式输出目录。
    返回值：
        tuple[Path,str]：Run 目录和最终 Run ID。
    """
    if output_root is not None:
        resolved_id = run_id or output_root.name
        return _resolve_user_path(output_root), resolved_id
    resolved_id = run_id or _new_run_id()
    return (DEFAULT_RUNS_ROOT / spec.name / resolved_id).resolve(), resolved_id


def _validate_run_id(run_id: str) -> None:
    """
    方法作用：
        校验 Run ID 只包含允许的安全字符。
    输入参数：
        run_id (str)：待校验标识。
    返回值：
        None：合法时无返回，非法时抛出 ValueError。
    """
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dot, underscore, and hyphen"
        )


def _runtime_overrides(
    config: dict[str, Any],
    spec: ExperimentSpec,
    epochs: int | None,
    workers: int | None,
) -> dict[str, Any]:
    """
    方法作用：
        把 CLI 的 epochs/workers 覆盖应用到新 Run 配置，并记录实际覆盖项。
    输入参数：
        config：可变配置；spec：实验定义；epochs：可选轮数；workers：可选进程数。
    返回值：
        dict[str,Any]：实际应用的覆盖项。
    """
    applied: dict[str, Any] = {}
    if epochs is not None:
        if spec.variant in {"b0", "p1b", "p2b"}:
            raise ValueError(f"--epochs does not apply to {spec.variant.upper()}")
        if epochs < 1:
            raise ValueError("--epochs must be positive")
        section = spec.variant
        config[section]["epochs"] = epochs
        if "warmup_epochs" in config[section]:
            config[section]["warmup_epochs"] = min(
                int(config[section]["warmup_epochs"]), epochs - 1
            )
        applied["epochs"] = epochs
    if workers is not None:
        if workers < 0:
            raise ValueError("--workers cannot be negative")
        config["data"]["workers"] = workers
        applied["workers"] = workers
    return applied


def _restore_saved_config(run_root: Path) -> dict[str, Any]:
    """
    方法作用：
        从受管 Run 恢复不可变配置并重新构造 Path 对象。
    输入参数：
        run_root (Path)：已有 Run 目录。
    返回值：
        dict[str,Any]：恢复后的运行配置。
    """
    config_path = run_root / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume/evaluate an unmanaged run without {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("crop_root", "splits_root", "clip_model", "output_root"):
        config["paths"][key] = Path(config["paths"][key]).resolve()
    if "dino_model" in config["paths"]:
        config["paths"]["dino_model"] = Path(config["paths"]["dino_model"]).resolve()
    if "config_path" in config:
        config["config_path"] = Path(config["config_path"]).resolve()
    config["paths"]["output_root"] = run_root.resolve()
    return config


def _snapshot_sources(run_root: Path) -> dict[str, str]:
    """
    方法作用：
        复制当前兼容入口、src、配置和预设到 Run，并记录逐文件哈希。
    输入参数：
        run_root (Path)：新 Run 根目录。
    返回值：
        dict[str,str]：快照相对路径到 SHA-256 的映射。
    """
    package_root = PAD_LITE_ROOT
    source_root = package_root / "src"
    snapshot_package = run_root / "source_snapshot" / "PAD_Lite"
    hashes: dict[str, str] = {}
    source_files = sorted(source_root.glob("*.py"))
    compatibility_files = sorted(package_root.glob("*.py"))
    config_files = sorted((package_root / "configs").glob("*.json"))
    preset_files = sorted((package_root / "experiments").glob("*.json"))
    snapshot_files = (
        compatibility_files + source_files + config_files + preset_files
    )
    for source in snapshot_files:
        relative = source.relative_to(package_root)
        destination = snapshot_package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(Path("PAD_Lite") / relative)] = _file_hash(destination)
    _write_json(
        run_root / "source_snapshot" / "snapshot_manifest.json",
        {"format": "pad_lite_source_snapshot_v1", "sha256": hashes},
        overwrite=False,
    )
    return hashes


def _initialise_run(
    run_root: Path,
    run_id: str,
    spec: ExperimentSpec,
    config: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    """
    方法作用：
        初始化全新受管 Run，写入解析配置、源码快照和实验清单。
    输入参数：
        run_root：Run 目录；run_id：标识；spec：实验定义；config：解析配置；
        runtime_overrides：CLI 覆盖记录。
    返回值：
        None：初始化完成后无返回数据。
    """
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"Run directory already contains data: {run_root}. "
            "Choose another --run-id; use --resume or --eval-only only for this same run."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    resolved = serializable_config(config)
    _write_json(run_root / "resolved_config.json", resolved, overwrite=False)
    source_hashes = _snapshot_sources(run_root)
    manifest = {
        "format": "pad_lite_experiment_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": spec.name,
        "run_id": run_id,
        "family": spec.family,
        "variant": spec.variant,
        "text_anchor": spec.text_anchor,
        "description": spec.description,
        "tags": list(spec.tags),
        "preset_path": str(spec.preset_path),
        "base_config": str(spec.base_config),
        "preset_overrides": spec.overrides,
        "runtime_overrides": runtime_overrides,
        "output_root": str(run_root),
        "resolved_config_sha256": _json_hash(resolved),
        "source_file_count": len(source_hashes),
    }
    _write_json(run_root / "experiment_manifest.json", manifest, overwrite=False)


def _verify_managed_run(run_root: Path, spec: ExperimentSpec) -> dict[str, Any]:
    """
    方法作用：
        校验已有目录是当前实验的受管 Run。
    输入参数：
        run_root：Run 目录；spec：预期实验定义。
    返回值：
        dict[str,Any]：实验清单内容。
    """
    manifest_path = run_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume/evaluate an unmanaged directory: {manifest_path} is missing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_name") != spec.name:
        raise ValueError(
            f"Run belongs to {manifest.get('experiment_name')!r}, not {spec.name!r}"
        )
    return manifest


def _start_event(
    run_root: Path,
    spec: ExperimentSpec,
    mode: str,
    folds: list[int],
    device: str,
) -> tuple[Path, dict[str, Any]]:
    """
    方法作用：
        创建一次 train/resume/eval 调用的 running 事件记录。
    输入参数：
        run_root：Run 目录；spec：实验；mode：模式；folds：折号；device：设备字符串。
    返回值：
        tuple[Path,dict]：事件文件路径和可继续更新的事件内容。
    """
    payload = {
        "format": "pad_lite_run_event_v1",
        "invocation_id": _utc_timestamp(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "experiment_name": spec.name,
        "mode": mode,
        "folds": folds,
        "device": device,
        "pid": os.getpid(),
    }
    path = run_root / "events" / f"{payload['invocation_id']}_{mode}.json"
    _write_json(path, payload, overwrite=False)
    return path, payload


def _finish_event(
    path: Path,
    payload: dict[str, Any],
    status: str,
    *,
    summary: Any = None,
    error: str | None = None,
) -> None:
    """
    方法作用：
        将事件标记为 completed/failed 并追加结果摘要或错误。
    输入参数：
        path：事件文件；payload：事件内容；status：终态；summary：可选摘要；error：可选错误。
    返回值：
        None：更新文件后无返回数据。
    """
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = status
    if summary is not None:
        payload["summary"] = summary
    if error is not None:
        payload["error"] = error
    _write_json(path, payload)


def _dispatch(
    spec: ExperimentSpec,
    config: dict[str, Any],
    folds: list[int],
    device: str,
    resume: bool,
    eval_only: bool,
) -> dict[str, Any]:
    """
    方法作用：
        根据 family/variant 把统一实验请求分派给 CLIP、DINO 或 Patch 引擎。
    输入参数：
        spec：实验定义；config：运行配置；folds：折号；device：设备；
        resume/eval_only：运行模式。
    返回值：
        dict[str,Any]：底层引擎返回的逐折结果和汇总。
    """
    if spec.family == "clip":
        from .engine import run_variant

        return run_variant(config, spec.variant, folds, device, resume, eval_only)
    if spec.family == "dino":
        from .dino_engine import run_dino_variant

        return run_dino_variant(config, spec.variant, folds, device, resume, eval_only)
    if spec.family == "dino_patch":
        if spec.variant == "p1a":
            from .dino_patch_engine import run_p1a

            return run_p1a(config, folds, device, resume, eval_only)
        if spec.variant == "p1b":
            from .dino_patch_fusion import run_p1b

            return run_p1b(config, folds, device, resume, eval_only)
        if spec.variant == "p2a":
            from .dino_patch_weighted_engine import run_p2a

            return run_p2a(config, folds, device, resume, eval_only)
        if spec.variant == "p2b":
            from .dino_patch_weighted_fusion import run_p2b

            return run_p2b(config, folds, device, resume, eval_only)
    raise ValueError(f"Unsupported experiment family: {spec.family}")


def _print_experiment_list(args: argparse.Namespace) -> int:
    """
    方法作用：
        按 family/关键词筛选并打印已注册实验。
    输入参数：
        args (argparse.Namespace)：list 子命令参数。
    返回值：
        int：成功状态码 0。
    """
    specs = list(load_registry().values())
    if args.family:
        specs = [spec for spec in specs if spec.family == args.family]
    if args.contains:
        needle = args.contains.lower()
        specs = [
            spec
            for spec in specs
            if needle in spec.name.lower()
            or needle in spec.description.lower()
            or any(needle in tag.lower() for tag in spec.tags)
        ]
    specs.sort(key=lambda item: item.name)
    if args.json:
        print(json.dumps([spec.as_dict() for spec in specs], ensure_ascii=False, indent=2))
        return 0
    print(f"{'NAME':58} {'FAMILY':11} {'VARIANT':7} {'TEXT':5} DESCRIPTION")
    for spec in specs:
        description = spec.description.replace("\n", " ")
        print(
            f"{spec.name:58} {spec.family:11} {spec.variant:7} "
            f"{('yes' if spec.text_anchor else 'no'):5} {description}"
        )
    print(f"\n{len(specs)} experiment preset(s)")
    return 0


def _show_experiment(args: argparse.Namespace) -> int:
    """
    方法作用：
        打印单个预设及其完全解析后的配置预览。
    输入参数：
        args：包含实验 name 的命令行命名空间。
    返回值：
        int：成功状态码 0。
    """
    spec = get_experiment(args.name)
    preview_root = DEFAULT_RUNS_ROOT / spec.name / "<run-id>"
    config = resolve_experiment_config(spec, preview_root)
    payload = spec.as_dict()
    payload["new_run_root_template"] = str(preview_root)
    payload["resolved_config"] = serializable_config(config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _list_runs(args: argparse.Namespace) -> int:
    """
    方法作用：
        列出某个实验已有受管 Run 及其最后事件状态。
    输入参数：
        args：包含实验 name 的命令行命名空间。
    返回值：
        int：成功状态码 0。
    """
    spec = get_experiment(args.name)
    root = DEFAULT_RUNS_ROOT / spec.name
    rows = []
    if root.is_dir():
        for run_root in sorted(root.iterdir()):
            manifest_path = run_root / "experiment_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            events = sorted((run_root / "events").glob("*.json"))
            last_status = None
            if events:
                last_status = json.loads(events[-1].read_text(encoding="utf-8")).get(
                    "status"
                )
            rows.append(
                {
                    "run_id": manifest.get("run_id", run_root.name),
                    "created_at_utc": manifest.get("created_at_utc"),
                    "last_status": last_status,
                    "output_root": str(run_root),
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    """
    方法作用：
        校验运行模式，创建/恢复受管 Run，记录事件并执行实验。
    输入参数：
        args：run 子命令的实验名、折、设备、覆盖项和恢复参数。
    返回值：
        int：成功状态码 0。
    """
    spec = get_experiment(args.name)
    if args.resume and args.eval_only:
        raise ValueError("--resume and --eval-only are mutually exclusive")
    if (args.resume or args.eval_only) and args.run_id is None and args.output_root is None:
        raise ValueError("--resume/--eval-only requires --run-id or --output-root")
    if spec.variant in {"b0", "p1b", "p2b"} and (args.resume or args.eval_only):
        raise ValueError(
            f"{spec.variant.upper()} has no training checkpoint; start a new run instead"
        )
    if (args.resume or args.eval_only) and args.epochs is not None:
        raise ValueError(
            "--epochs cannot change an existing managed run; start a new run instead"
        )
    if args.run_id is not None:
        _validate_run_id(args.run_id)
    run_root, run_id = _resolve_run_root(spec, args.run_id, args.output_root)
    _validate_run_id(run_id)

    existing = args.resume or args.eval_only
    if existing:
        _verify_managed_run(run_root, spec)
        config = _restore_saved_config(run_root)
    else:
        config = resolve_experiment_config(spec, run_root)
    runtime_overrides = _runtime_overrides(
        config, spec, epochs=args.epochs, workers=args.workers
    )

    mode = (
        "eval"
        if args.eval_only or spec.variant in {"p1b", "p2b"}
        else ("resume" if args.resume else "train")
    )
    plan = {
        "experiment_name": spec.name,
        "family": spec.family,
        "variant": spec.variant,
        "text_anchor": spec.text_anchor,
        "mode": mode,
        "folds": args.fold,
        "device": args.device,
        "run_id": run_id,
        "output_root": str(run_root),
        "runtime_overrides": runtime_overrides,
    }
    if args.dry_run:
        plan["resolved_config"] = serializable_config(config)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if not existing:
        _initialise_run(run_root, run_id, spec, config, runtime_overrides)
    event_path, event = _start_event(
        run_root, spec, mode, args.fold, args.device
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    try:
        result = _dispatch(
            spec,
            config,
            args.fold,
            args.device,
            resume=args.resume,
            eval_only=args.eval_only,
        )
    except Exception as exc:
        _finish_event(event_path, event, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    summary = result.get("summary", result)
    _finish_event(event_path, event, "completed", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    方法作用：
        构建 list/show/runs/run 四个子命令的统一实验 CLI。
    输入参数：
        无。
    返回值：
        argparse.ArgumentParser：完整参数解析器。
    """
    parser = argparse.ArgumentParser(
        prog="python -m PAD_Lite.experiment_cli",
        description=(
            "List and run immutable PAD-Lite experiment presets. Every new run "
            "uses an independent output directory and saves a source snapshot."
        ),
    )

    # 准备创建子命令工具
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 创建 list 子命令解析器
    list_parser = subparsers.add_parser("list", help="List registered experiments")
    list_parser.add_argument("--family", choices=("clip", "dino", "dino_patch"))
    list_parser.add_argument("--contains", default=None)
    list_parser.add_argument("--json", action="store_true")
    # 将 list 子命令的处理函数设置为 _print_experiment_list
    list_parser.set_defaults(handler=_print_experiment_list)

    # 创建 show 子命令解析器
    show_parser = subparsers.add_parser("show", help="Show one resolved preset")
    show_parser.add_argument("name")
    # 将 show 子命令的处理函数设置为 _show_experiment
    show_parser.set_defaults(handler=_show_experiment)

    # 创建 runs 子命令解析器
    runs_parser = subparsers.add_parser("runs", help="List managed runs for a preset")
    runs_parser.add_argument("name")
    # 将 runs 子命令的处理函数设置为 _list_runs
    runs_parser.set_defaults(handler=_list_runs)

    # 创建 run 子命令解析器
    run_parser = subparsers.add_parser("run", help="Run one named experiment")
    run_parser.add_argument("name")
    run_parser.add_argument("--fold", type=parse_folds, default=[1, 2, 3, 4, 5])
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--output-root", type=Path, default=None)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--eval-only", action="store_true")
    run_parser.add_argument("--epochs", type=int, default=None)
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--dry-run", action="store_true")
    # 将 run 子命令的处理函数设置为 _run_experiment
    run_parser.set_defaults(handler=_run_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    方法作用：
        解析统一实验 CLI，并调用所选子命令处理器。
    输入参数：
        argv (list[str]|None)：显式参数列表；None 时读取系统命令行。
    返回值：
        int：子命令退出状态码。
    """
    args = build_parser().parse_args(argv)
    # 根据arg.command调用对应的处理函数，并返回其退出状态码
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
