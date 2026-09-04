"""Stanford Cars parquet loading and ViT batch preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import BaseImageProcessor


@dataclass(frozen=True)
class StanfordCarsData:
    """已加载的Stanford Cars数据划分及其类别元数据。

    作用:
        把Hugging Face数据表、196个类别名称和实际Parquet分片路径组合为一个返回对象。
    参数:
        dataset: 一维样本表，长度为``[N]``；每行至少包含PIL图像和标量类别标签。
        class_names: 类别名称列表，形状为``[C]``。
        parquet_files: 构成当前划分的Parquet文件列表，形状为``[S]``。
    返回值:
        数据容器本身；不改变图像或标签内容。
    """

    dataset: Dataset
    class_names: list[str]
    parquet_files: list[Path]


def load_stanford_cars_parquet(dataset_root: Path, split: str) -> StanfordCarsData:
    """加载Stanford Cars某个官方划分的全部Parquet分片。

    作用:
        查找``parquet/<split>/*.parquet``，合并所有分片并读取ClassLabel名称。
    参数:
        dataset_root: Stanford Cars-196根目录。
        split: 划分名称，通常为``train``或``test``。
    返回值:
        ``StanfordCarsData``；其中``dataset``长度为``[N]``，标签列形状为``[N]``，
        ``class_names``形状为``[C]``。
    """

    parquet_dir = dataset_root / "parquet" / split
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found under {parquet_dir}")

    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in parquet_files]},
        split=split,
    )
    label_feature = dataset.features.get("label")
    class_names = list(getattr(label_feature, "names", []) or [])
    if not class_names:
        raise ValueError("The parquet label column has no ClassLabel names.")

    labels = dataset["label"]
    if labels and (min(labels) < 0 or max(labels) >= len(class_names)):
        raise ValueError("A label index falls outside the ClassLabel name table.")
    return StanfordCarsData(dataset, class_names, parquet_files)


class IndexedImageDataset(TorchDataset[dict[str, Any]]):
    """为特征提取暴露稳定行号、RGB图像和标签的数据集包装器。

    作用:
        保持官方测试集顺序，并把每一行解码为RGB PIL图像。
    参数:
        dataset: Hugging Face数据表，长度为``[N]``。
    返回值:
        可按索引读取的PyTorch Dataset；单样本包含图像``[H,W,3]``、标量标签和标量行号。
    """

    def __init__(self, dataset: Dataset) -> None:
        """初始化测试图像包装器。

        作用:
            保存底层Hugging Face Dataset引用。
        参数:
            self: 当前包装器实例。
            dataset: 长度为``[N]``的数据表。
        返回值:
            无。
        """

        self.dataset = dataset

    def __len__(self) -> int:
        """返回样本总数。

        作用:
            让DataLoader获知当前划分的长度。
        参数:
            self: 当前包装器实例。
        返回值:
            整数``N``。
        """

        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取并解码一条测试样本。

        作用:
            按稳定行号读取图片，将图片转换为RGB并保留类别标签。
        参数:
            self: 当前包装器实例。
            index: 测试集行号，范围``[0,N)``。
        返回值:
            字典：``image``为RGB图像``[H,W,3]``，``label``和``index``均为标量。
        """

        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return {"index": index, "image": image, "label": int(row["label"])}


class ViTBatchCollator:
    """把测试样本列表整理为ViT批输入。

    作用:
        调用预训练权重配套的图像处理器，对PIL图像执行缩放、张量化和归一化。
    参数:
        image_processor: 与本地ViT权重匹配的Hugging Face图像处理器。
    返回值:
        可调用的collator；调用后产生图像``[B,3,224,224]``以及标签、行号``[B]``。
    """

    def __init__(self, image_processor: BaseImageProcessor) -> None:
        """保存ViT图像处理器。

        作用:
            为后续批量预处理保存无状态处理器。
        参数:
            self: 当前collator实例。
            image_processor: ViT图像处理器。
        返回值:
            无。
        """

        self.image_processor = image_processor

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """整理一个测试批次。

        作用:
            批量处理PIL图像，并把行号和标签转换为LongTensor。
        参数:
            self: 当前collator实例。
            samples: 长度为``[B]``的样本列表，每项包含图像``[H_i,W_i,3]``。
        返回值:
            ``pixel_values``形状``[B,3,224,224]``；``indices``和``labels``形状``[B]``。
        """

        processed = self.image_processor(
            images=[sample["image"] for sample in samples],
            return_tensors="pt",
        )
        return {
            "indices": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
            "labels": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "pixel_values": processed["pixel_values"],
        }


def build_dataloader(
    dataset: Dataset,
    image_processor: BaseImageProcessor,
    batch_size: int,
    num_workers: int,
    use_cuda: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """构建顺序固定且不打乱的特征提取DataLoader。

    作用:
        将数据集、ViT预处理器和批参数组合，确保输出顺序与原始测试行号一致。
    参数:
        dataset: 长度为``[N]``的Hugging Face数据表。
        image_processor: ViT图像处理器。
        batch_size: 每批样本数``B``。
        num_workers: 数据加载工作进程数。
        use_cuda: 是否启用pin memory以加速CPU到GPU拷贝。
    返回值:
        DataLoader；每批``pixel_values``为``[B,3,224,224]``，标签和行号为``[B]``。
    """

    return DataLoader(
        IndexedImageDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
        collate_fn=ViTBatchCollator(image_processor),
    )
