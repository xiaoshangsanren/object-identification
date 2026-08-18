from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ANNOTATION_ROOT, PAD_LITE_ROOT, load_config


PRESET_ROOT = PAD_LITE_ROOT / "experiments"
EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SUPPORTED_FAMILIES = {"clip", "dino", "dino_patch"}
FAMILY_VARIANTS = {
    "clip": {"b0", "b1", "b2"},
    "dino": {"b0", "b1", "b2"},
    "dino_patch": {"p1a", "p1b", "p2a", "p2b"},
}


@dataclass(frozen=True)
class ExperimentSpec:
    """
    方法作用：
        表示一个不可变、可复现的 PAD-Lite 命名实验定义。
    输入参数：
        name/family/variant：实验身份；base_config：基础配置；description/tags：说明；
        text_anchor：语义锚点开关；overrides：点路径覆盖；preset_path：来源文件。
    返回值：
        ExperimentSpec：只读实验规格实例。
    """

    name: str
    family: str
    variant: str
    base_config: Path
    description: str
    text_anchor: bool
    overrides: dict[str, Any]
    tags: tuple[str, ...]
    preset_path: Path
    legacy_output_root: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """
        方法作用：
            把实验规格转换为可 JSON 序列化的独立字典。
        输入参数：
            无。
        返回值：
            dict[str,Any]：包含全部实验字段的深拷贝字典。
        """
        return {
            "name": self.name,
            "family": self.family,
            "variant": self.variant,
            "base_config": str(self.base_config),
            "description": self.description,
            "text_anchor": self.text_anchor,
            "overrides": deepcopy(self.overrides),
            "tags": list(self.tags),
            "preset_path": str(self.preset_path),
            "legacy_output_root": (
                str(self.legacy_output_root)
                if self.legacy_output_root is not None
                else None
            ),
        }


def _annotation_path(value: str | Path) -> Path:
    """
    方法作用：
        将预设中的相对路径按 Annotation 根目录解析。
    输入参数：
        value (str|Path)：原始路径。
    返回值：
        Path：绝对路径。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def _expand_payload(payload: dict[str, Any], preset_path: Path) -> Iterable[dict[str, Any]]:
    """
    方法作用：
        将单实验预设原样产生，或把 matrix.values 展开为多个独立实验载荷。
    输入参数：
        payload：预设 JSON；preset_path：用于错误定位的来源路径。
    返回值：
        Iterable[dict[str,Any]]：逐个产生展开后的实验载荷。
    """
    matrix = payload.get("matrix")
    if matrix is None:
        yield payload
        return

    if not isinstance(matrix, dict):
        raise ValueError(f"{preset_path}: matrix must be an object")
    field = str(matrix.get("field", ""))
    values = matrix.get("values")
    name_template = matrix.get("name_template")
    if not field or not isinstance(values, list) or not values or not name_template:
        raise ValueError(
            f"{preset_path}: matrix requires field, non-empty values, and name_template"
        )

    for matrix_value in values:
        expanded = deepcopy(payload)
        expanded.pop("matrix", None)
        expanded["name"] = str(name_template).format(value=matrix_value)
        expanded["description"] = str(expanded.get("description", "")).format(
            value=matrix_value
        )
        overrides = dict(expanded.get("overrides", {}))
        overrides[field] = matrix_value
        expanded["overrides"] = overrides
        if expanded.get("legacy_output_root") is not None:
            expanded["legacy_output_root"] = str(
                expanded["legacy_output_root"]
            ).format(value=matrix_value)
        yield expanded


def _parse_spec(payload: dict[str, Any], preset_path: Path) -> ExperimentSpec:
    """
    方法作用：
        校验 schema、命名、family/variant、文本锚点和路径后构造实验规格。
    输入参数：
        payload：展开后的预设载荷；preset_path：来源 JSON 路径。
    返回值：
        ExperimentSpec：验证通过的不可变实验规格。
    """
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"{preset_path}: schema_version must be 1")

    name = str(payload.get("name", ""))
    family = str(payload.get("family", ""))
    variant = str(payload.get("variant", ""))
    if not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{preset_path}: invalid experiment name {name!r}")
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"{preset_path}: unsupported family {family!r}")
    if variant not in FAMILY_VARIANTS[family]:
        raise ValueError(
            f"{preset_path}: variant {variant!r} is invalid for family {family!r}"
        )

    text_anchor = bool(payload.get("text_anchor", False))
    if family in {"clip", "dino"} and text_anchor != (variant == "b2"):
        raise ValueError(
            f"{preset_path}: B2 must declare text_anchor=true and B0/B1 false"
        )
    if family == "dino_patch":
        expected_text_anchor = variant in {"p1b", "p2b"}
        if text_anchor != expected_text_anchor:
            raise ValueError(
                f"{preset_path}: Patch-only variants have no text anchor; "
                "P1b/P2b inherit the training-only text-enhanced P0 CLS branch"
            )

    base_config_raw = payload.get("base_config")
    if not base_config_raw:
        raise ValueError(f"{preset_path}: base_config is required")
    base_config = _annotation_path(str(base_config_raw))
    if not base_config.is_file():
        raise FileNotFoundError(f"{preset_path}: missing base config {base_config}")

    overrides = payload.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"{preset_path}: overrides must be an object")
    if "paths.output_root" in overrides:
        raise ValueError(
            f"{preset_path}: paths.output_root is managed by the experiment runner"
        )

    legacy_raw = payload.get("legacy_output_root")
    return ExperimentSpec(
        name=name,
        family=family,
        variant=variant,
        base_config=base_config,
        description=str(payload.get("description", "")).strip(),
        text_anchor=text_anchor,
        overrides=deepcopy(overrides),
        tags=tuple(str(tag) for tag in payload.get("tags", [])),
        preset_path=preset_path.resolve(),
        legacy_output_root=(
            _annotation_path(str(legacy_raw)) if legacy_raw is not None else None
        ),
    )


def load_registry(preset_root: Path = PRESET_ROOT) -> dict[str, ExperimentSpec]:
    """
    方法作用：
        加载并验证目录中所有单实验或矩阵实验预设，检查名称冲突。
    输入参数：
        preset_root (Path)：实验 JSON 目录。
    返回值：
        dict[str,ExperimentSpec]：实验名到规格的注册表。
    """

    registry: dict[str, ExperimentSpec] = {}
    if not preset_root.is_dir():
        raise FileNotFoundError(f"Experiment preset directory is missing: {preset_root}")
    for preset_path in sorted(preset_root.glob("*.json")):
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{preset_path}: preset root must be an object")
        for expanded in _expand_payload(payload, preset_path):
            spec = _parse_spec(expanded, preset_path)
            if spec.name in registry:
                raise ValueError(
                    f"Duplicate experiment name {spec.name!r}: "
                    f"{registry[spec.name].preset_path} and {preset_path}"
                )
            registry[spec.name] = spec
    if not registry:
        raise ValueError(f"No experiment presets were found under {preset_root}")
    return registry


def get_experiment(name: str) -> ExperimentSpec:
    """
    方法作用：
        按精确名称取得实验，并在找不到时给出模糊候选。
    输入参数：
        name (str)：实验名。
    返回值：
        ExperimentSpec：对应实验规格。
    """
    registry = load_registry()
    try:
        return registry[name]
    except KeyError as exc:
        matches = [item for item in registry if name.lower() in item.lower()][:8]
        suffix = f"; possible matches: {', '.join(matches)}" if matches else ""
        raise KeyError(f"Unknown experiment {name!r}{suffix}") from exc


def _set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    方法作用：
        按 data.image_size 形式的点路径修改已有嵌套配置字段。
    输入参数：
        config：目标配置；dotted_key：已存在字段的点路径；value：新值。
    返回值：
        None：原位修改配置。
    """
    keys = dotted_key.split(".")
    if not keys or any(not key for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            raise KeyError(f"Override path does not exist: {dotted_key}")
        target = child
    leaf = keys[-1]
    if leaf not in target:
        raise KeyError(f"Override path does not exist: {dotted_key}")
    target[leaf] = deepcopy(value)


def resolve_experiment_config(
    spec: ExperimentSpec,
    output_root: Path,
) -> dict[str, Any]:
    """
    方法作用：
        加载基础配置、应用预设覆盖并设置独立输出目录，得到引擎最终配置。
    输入参数：
        spec (ExperimentSpec)：实验规格；output_root (Path)：本次 Run 输出目录。
    返回值：
        dict[str,Any]：可直接传给对应训练/评测引擎的配置。
    """

    config = load_config(spec.base_config)
    for dotted_key, value in spec.overrides.items():
        _set_dotted_value(config, dotted_key, value)
    config["paths"]["output_root"] = output_root.resolve()

    if spec.variant in {"b1", "b2"}:
        anchor_weight = float(config[spec.variant].get("anchor_weight", 0.0))
        if spec.text_anchor and anchor_weight <= 0:
            raise ValueError(f"{spec.name}: text-anchor preset requires anchor_weight > 0")
        if not spec.text_anchor and anchor_weight != 0:
            raise ValueError(f"{spec.name}: no-text preset requires anchor_weight = 0")
    return config
