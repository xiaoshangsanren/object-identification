from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PAD_LITE_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = PAD_LITE_ROOT.parent


def _resolve_path(value: str | Path) -> Path:
    """
    方法作用：
        将相对路径转换为基于 Annotation 根目录的绝对路径。

    输入参数：
        value (str | Path):
            需要解析的配置路径。

    返回值：
        Path:
            解析后的绝对路径对象。
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ANNOTATION_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """
    方法作用：
        读取并检查实验配置文件，完成路径解析和必需字段校验。

    输入参数：
        path (str | Path):
            配置文件的路径。

    返回值：
        dict[str, Any]:
            处理后的实验配置字典。
    """
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_sections = {"paths", "data", "model", "retrieval", "b1", "b2"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")

    resolved = deepcopy(config)
    for key in ("crop_root", "splits_root", "clip_model", "output_root"):
        if key not in resolved["paths"]:
            raise ValueError(f"Config paths.{key} is required")
        resolved["paths"][key] = _resolve_path(resolved["paths"][key])
    if "dino_model" in resolved["paths"]:
        resolved["paths"]["dino_model"] = _resolve_path(
            resolved["paths"]["dino_model"]
        )
    resolved["config_path"] = config_path
    return resolved


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    方法作用：
        将配置字典中的 Path 对象递归转换为字符串，便于序列化保存到 JSON。

    输入参数：
        config (dict[str, Any]):
            原始配置字典。

    返回值：
        dict[str, Any]:
            可 JSON 序列化的配置字典。
    """
    def convert(value: Any) -> Any:
        """
        方法作用：
            将配置值递归转换为可进行 JSON 序列化的数据。
        
        输入参数：
            value (Any)：待解析或转换的输入值。
        
        返回值：
            Any：方法执行得到的结果。
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(config)
