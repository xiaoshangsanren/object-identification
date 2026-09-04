"""Configuration utilities for the standalone retrieval baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """E0纯视觉ViT留一法检索实验的原始配置。

    作用:
        保存JSON中读取的路径、设备、批大小和检索评测参数，并在运行前完成合法性校验。
    参数:
        experiment_name: 实验名称。
        dataset_root: Stanford Cars-196数据集路径。
        split: 用于评测的数据划分名称，E0固定为``test``。
        model_path: 本地ImageNet预训练ViT路径。
        output_dir: 实验输出根目录。
        batch_size: 特征提取批大小``B``。
        num_workers: DataLoader工作进程数。
        device: ``cuda``或``cpu``。
        amp: 是否在CUDA上启用半精度自动混合精度。
        retrieval_query_chunk_size: 每次构造相似度矩阵的Query数量``Q``。
        recall_ks: 需要计算的Recall@K列表。
        compute_map: 是否计算mAP。
        save_top_k: 每个Query保存的邻居数量。
        save_embeddings: 是否保存测试特征``[N,D]``。
        seed: 随机种子。
    返回值:
        配置对象本身不直接返回主数据；各字段供E0入口读取。
    """

    experiment_name: str
    dataset_root: str
    split: str
    model_path: str
    output_dir: str
    batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True
    retrieval_query_chunk_size: int = 512
    recall_ks: tuple[int, ...] = (1, 5, 10)
    compute_map: bool = True
    save_top_k: int = 10
    save_embeddings: bool = True
    seed: int = 42

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """从JSON文件创建并校验E0配置。

        作用:
            读取JSON，将``recall_ks``转换为整数元组，然后调用``validate``。
        参数:
            cls: 当前配置类。
            path: 配置JSON文件路径。
        返回值:
            ``ExperimentConfig``配置实例。
        """

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        if "recall_ks" in raw:
            raw["recall_ks"] = tuple(int(k) for k in raw["recall_ks"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """校验E0实验参数是否满足协议。

        作用:
            阻止非test划分、非法设备、非正批大小或非法Recall@K进入实验。
        参数:
            self: 当前E0配置实例。
        返回值:
            无；参数非法时抛出``ValueError``。
        """

        if self.split != "test":
            raise ValueError("Stage E0 is defined on the official test split only.")
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative.")
        if self.retrieval_query_chunk_size <= 0:
            raise ValueError("retrieval_query_chunk_size must be positive.")
        if not self.recall_ks or any(k <= 0 for k in self.recall_ks):
            raise ValueError("recall_ks must contain positive integers.")
        if self.save_top_k <= 0:
            raise ValueError("save_top_k must be positive.")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be either 'cuda' or 'cpu'.")

    def resolved(self, repo_root: Path) -> "ResolvedExperimentConfig":
        """把配置中的相对路径解析为绝对路径。

        作用:
            以仓库根目录为基准解析数据、模型和输出路径。
        参数:
            self: 当前E0配置实例。
            repo_root: 项目仓库根目录。
        返回值:
            ``ResolvedExperimentConfig``绝对路径配置实例。
        """

        values = asdict(self)
        values.update(
            dataset_root=_resolve_path(repo_root, self.dataset_root),
            model_path=_resolve_path(repo_root, self.model_path),
            output_dir=_resolve_path(repo_root, self.output_dir),
        )
        return ResolvedExperimentConfig(**values)


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """所有资源路径均已解析的E0运行配置。

    作用:
        向数据加载、模型加载和检索评测代码提供无需依赖当前工作目录的配置。
    参数:
        字段与``ExperimentConfig``一致；``dataset_root``、``model_path``和
        ``output_dir``已经是绝对``Path``，``recall_ks``为整数元组。
    返回值:
        配置对象本身不返回主数据。
    """

    experiment_name: str
    dataset_root: Path
    split: str
    model_path: Path
    output_dir: Path
    batch_size: int
    num_workers: int
    device: str
    amp: bool
    retrieval_query_chunk_size: int
    recall_ks: tuple[int, ...]
    compute_map: bool
    save_top_k: int
    save_embeddings: bool
    seed: int

    def as_serializable_dict(self) -> dict[str, Any]:
        """把已解析配置转换为可写入JSON的字典。

        作用:
            将``Path``转为字符串、将元组转为列表，用于保存实验快照。
        参数:
            self: 已解析的E0配置实例。
        返回值:
            可被``json.dump``序列化的配置字典。
        """

        result = asdict(self)
        for key in ("dataset_root", "model_path", "output_dir"):
            result[key] = str(result[key])
        result["recall_ks"] = list(self.recall_ks)
        return result


def _resolve_path(repo_root: Path, value: str) -> Path:
    """解析单个配置路径。

    作用:
        展开用户目录，并把相对路径拼接到仓库根目录后规范化。
    参数:
        repo_root: 项目仓库根目录。
        value: JSON中读取的绝对或相对路径字符串。
    返回值:
        解析后的绝对``Path``。
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()
