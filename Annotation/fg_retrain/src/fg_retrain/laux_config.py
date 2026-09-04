"""Configuration for the standalone ViT + LaFG L_aux experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LauxConfig:
    """E1 ViT加纯视觉LaFG Laux实验的原始配置。

    作用:
        保存数据、模型、P×2采样、优化器、增强、评测和检查点的完整参数。
    参数:
        experiment_name: 实验名称。
        dataset_root/model_path/output_dir: 数据、预训练模型和结果目录。
        train_split/test_split: 训练与测试划分名称。
        classes_per_batch: 每批类别数``P``。
        instances_per_class: 每类图像数``K``，LaFG固定为2。
        batches_per_epoch: 每轮批次数；null时按训练集大小自动计算。
        epochs/temperature: 总训练轮数和Laux温度系数。
        learning_rate/momentum/weight_decay: SGD优化器参数。
        lr_step_size/lr_gamma: StepLR衰减周期和比例。
        train_resize/train_crop: 训练缩放和随机裁剪尺寸。
        color_jitter_brightness/color_jitter_contrast/color_jitter_saturation/color_jitter_hue:
            ColorJitter四个强度参数。
        eval_batch_size/num_workers: 测试批大小``B_eval``和加载进程数。
        device/amp/data_parallel/gradient_checkpointing: 计算设备与显存相关设置。
        retrieval_query_chunk_size: 检索相似度矩阵的Query分块大小``Q``。
        recall_ks/compute_map/save_top_k/save_embeddings: 检索指标及结果保存设置。
        checkpoint_every_epochs: 检查点保存周期。
        seed: 随机种子。
    返回值:
        配置对象本身不返回图像主数据。
    """

    experiment_name: str
    dataset_root: str
    model_path: str
    output_dir: str
    train_split: str = "train"
    test_split: str = "test"
    classes_per_batch: int = 64
    instances_per_class: int = 2
    batches_per_epoch: int | None = None
    epochs: int = 200
    temperature: float = 0.1
    learning_rate: float = 1.0e-5
    momentum: float = 0.9
    weight_decay: float = 1.0e-4
    lr_step_size: int = 5
    lr_gamma: float = 0.9
    train_resize: int = 256
    train_crop: int = 224
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1
    eval_batch_size: int = 128
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True
    data_parallel: bool = True
    gradient_checkpointing: bool = False
    retrieval_query_chunk_size: int = 512
    recall_ks: tuple[int, ...] = (1, 2, 4, 5, 8, 10)
    compute_map: bool = True
    save_top_k: int = 10
    save_embeddings: bool = True
    checkpoint_every_epochs: int = 5
    seed: int = 42

    @classmethod
    def from_json(cls, path: Path) -> "LauxConfig":
        """从JSON创建并校验E1配置。

        作用:
            读取配置文件，将Recall@K列表转成元组并执行合法性校验。
        参数:
            cls: 当前配置类。
            path: E1配置JSON路径。
        返回值:
            ``LauxConfig``实例。
        """

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        if "recall_ks" in raw:
            raw["recall_ks"] = tuple(int(k) for k in raw["recall_ks"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """校验配置是否满足LaFG唯一正样本规则。

        作用:
            强制每类2张并检查划分、批次、温度、增强、设备和评测参数。
        参数:
            self: 当前E1配置实例。
        返回值:
            无；配置非法时抛出``ValueError``。
        """

        if self.train_split == self.test_split:
            raise ValueError("Training and test splits must be different.")
        if self.instances_per_class != 2:
            raise ValueError("LaFG L_aux requires exactly two instances per selected class.")
        if self.classes_per_batch < 2:
            raise ValueError("classes_per_batch must be at least two.")
        if self.batches_per_epoch is not None and self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive or null.")
        if self.epochs <= 0 or self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, temperature, and learning_rate must be positive.")
        if self.lr_step_size <= 0 or not 0 < self.lr_gamma <= 1:
            raise ValueError("Invalid learning-rate schedule.")
        if self.train_crop > self.train_resize:
            raise ValueError("train_crop cannot exceed train_resize.")
        if self.eval_batch_size <= 0 or self.num_workers < 0:
            raise ValueError("Invalid evaluation loader settings.")
        if not self.recall_ks or any(k <= 0 for k in self.recall_ks):
            raise ValueError("recall_ks must contain positive integers.")
        if self.checkpoint_every_epochs <= 0:
            raise ValueError("checkpoint_every_epochs must be positive.")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'.")

    def resolved(self, repo_root: Path) -> "ResolvedLauxConfig":
        """将E1配置中的相对路径解析为绝对路径。

        作用:
            以仓库根目录解析数据集、模型和输出目录。
        参数:
            self: 当前E1原始配置。
            repo_root: 项目仓库根目录。
        返回值:
            ``ResolvedLauxConfig``绝对路径配置。
        """

        values = asdict(self)
        for key in ("dataset_root", "model_path", "output_dir"):
            path = Path(values[key]).expanduser()
            if not path.is_absolute():
                path = repo_root / path
            values[key] = path.resolve()
        return ResolvedLauxConfig(**values)


@dataclass(frozen=True)
class ResolvedLauxConfig:
    """路径已解析、可直接执行的E1配置。

    作用:
        向训练数据流、模型、Laux损失、优化器和评测阶段提供统一只读参数。
    参数:
        字段与``LauxConfig``一致；三个路径字段为绝对``Path``，``recall_ks``为元组。
    返回值:
        配置对象本身不返回主数据。
    """

    experiment_name: str
    dataset_root: Path
    model_path: Path
    output_dir: Path
    train_split: str
    test_split: str
    classes_per_batch: int
    instances_per_class: int
    batches_per_epoch: int | None
    epochs: int
    temperature: float
    learning_rate: float
    momentum: float
    weight_decay: float
    lr_step_size: int
    lr_gamma: float
    train_resize: int
    train_crop: int
    color_jitter_brightness: float
    color_jitter_contrast: float
    color_jitter_saturation: float
    color_jitter_hue: float
    eval_batch_size: int
    num_workers: int
    device: str
    amp: bool
    data_parallel: bool
    gradient_checkpointing: bool
    retrieval_query_chunk_size: int
    recall_ks: tuple[int, ...]
    compute_map: bool
    save_top_k: int
    save_embeddings: bool
    checkpoint_every_epochs: int
    seed: int

    @property
    def train_batch_size(self) -> int:
        """计算P×2采样得到的训练批大小。

        作用:
            将每批类别数与每类样本数相乘。
        参数:
            self: 已解析的E1配置。
        返回值:
            整数批大小``B_train=P×2``。
        """

        return self.classes_per_batch * self.instances_per_class

    def as_serializable_dict(self) -> dict[str, Any]:
        """把E1配置转换为可序列化实验快照。

        作用:
            将Path转为字符串、元组转为列表，并补充实际训练批大小。
        参数:
            self: 已解析的E1配置。
        返回值:
            可被``json.dump``写入的配置字典。
        """

        result = asdict(self)
        for key in ("dataset_root", "model_path", "output_dir"):
            result[key] = str(result[key])
        result["recall_ks"] = list(self.recall_ks)
        result["train_batch_size"] = self.train_batch_size
        return result
