"""PK sampling and image transforms for the pure-vision LaFG baseline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fg_retrain.laux_config import ResolvedLauxConfig


class TransformedCarsDataset(TorchDataset[dict[str, Any]]):
    """按需解码Parquet图像并执行E1训练增强的数据集包装器。

    作用:
        避免预先展开全部图像，在被采样时才将图片转换为ViT训练张量。
    参数:
        dataset: Hugging Face训练数据表，长度为``[N_train]``。
        transform: 将PIL图像``[H,W,3]``转换为张量``[3,224,224]``的增强流水线。
    返回值:
        PyTorch Dataset；单样本包含图像``[3,224,224]``、标量标签和标量行号。
    """

    def __init__(self, dataset: Dataset, transform: transforms.Compose) -> None:
        """保存训练数据表及增强流水线。

        作用:
            初始化延迟解码的数据集包装器。
        参数:
            self: 当前数据集实例。
            dataset: 长度``[N_train]``的训练数据表。
            transform: 单图输入``[H,W,3]``、输出``[3,224,224]``的变换。
        返回值:
            无。
        """

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """返回训练样本总数。

        作用:
            让DataLoader和采样器获得数据集大小。
        参数:
            self: 当前数据集实例。
        返回值:
            整数``N_train``。
        """

        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取并增强一张训练图像。

        作用:
            按行号解码RGB图像并执行随机增强。
        参数:
            self: 当前数据集实例。
            index: 训练集行号，范围``[0,N_train)``。
        返回值:
            字典：``pixel_values``为``[3,224,224]``；``label``和``index``为标量。
        """

        row = self.dataset[index]
        return {
            "index": index,
            "pixel_values": self.transform(row["image"].convert("RGB")),
            "label": int(row["label"]),
        }


class PKBatchSampler(Sampler[list[int]]):
    """生成“P个类别×每类严格2张”的LaFG批次采样器。

    作用:
        每个批次随机选取互不重复的P类，再为每类无放回选择2个实例，保证每个anchor
        在批内恰好存在一个同类positive。
    参数:
        labels: 训练集逐样本类别标签，形状``[N_train]``。
        classes_per_batch: 每批类别数量``P``。
        instances_per_class: 每类实例数，必须等于2。
        batches_per_epoch: 每轮生成的批次数``S``；null时取``ceil(N_train/(2P))``。
        seed: 控制类别及实例选择的随机种子。
    返回值:
        BatchSampler；每次迭代返回长度``[B=2P]``的训练集行号列表。
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        instances_per_class: int = 2,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        """建立类别到样本行号的索引。

        作用:
            验证每类至少2张、P不超过类别总数，并确定每轮批次数。
        参数:
            self: 当前采样器实例。
            labels: 标签序列``[N_train]``。
            classes_per_batch: 每批类别数``P``。
            instances_per_class: 每类实例数，固定为2。
            batches_per_epoch: 可选的每轮批次数``S``。
            seed: 基础随机种子。
        返回值:
            无。
        """

        if instances_per_class != 2:
            raise ValueError("This sampler implements LaFG's fixed K=2 protocol.")
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            class_to_indices[int(label)].append(index)
        too_small = [label for label, indices in class_to_indices.items() if len(indices) < 2]
        if too_small:
            raise ValueError(f"Classes with fewer than two images: {too_small}")
        if classes_per_batch > len(class_to_indices):
            raise ValueError("classes_per_batch exceeds the number of training classes.")

        self.class_to_indices = dict(class_to_indices)
        self.classes = sorted(self.class_to_indices)
        self.classes_per_batch = classes_per_batch
        self.instances_per_class = instances_per_class
        self.batch_size = classes_per_batch * instances_per_class
        self.batches_per_epoch = batches_per_epoch or math.ceil(len(labels) / self.batch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置采样器当前轮次。

        作用:
            使用``seed+epoch``产生可复现但逐轮变化的采样序列。
        参数:
            self: 当前采样器实例。
            epoch: 从0开始的训练轮次标量。
        返回值:
            无。
        """

        self.epoch = epoch

    def __len__(self) -> int:
        """返回每轮批次数。

        作用:
            让训练循环获知一个epoch包含多少个step。
        参数:
            self: 当前采样器实例。
        返回值:
            整数``S=batches_per_epoch``。
        """

        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        """生成当前epoch的全部P×2批次。

        作用:
            随机选类、每类选2张，并打乱批内图像顺序。
        参数:
            self: 当前采样器实例，使用其中的``seed``和``epoch``。
        返回值:
            迭代器；共``S``项，每项是形状``[B=2P]``的样本行号列表。
        """

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        class_tensor = torch.tensor(self.classes, dtype=torch.long)
        for _ in range(self.batches_per_epoch):
            permutation = torch.randperm(len(class_tensor), generator=generator)
            selected_classes = class_tensor[permutation[: self.classes_per_batch]].tolist()
            batch: list[int] = []
            for label in selected_classes:
                indices = self.class_to_indices[int(label)]
                selected = torch.randperm(len(indices), generator=generator)[:2].tolist()
                batch.extend(indices[position] for position in selected)
            shuffle_order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in shuffle_order]


def build_train_transform(config: ResolvedLauxConfig) -> transforms.Compose:
    """构建LaFG视觉训练增强流水线。

    作用:
        依次执行256方形缩放、224随机裁剪、随机水平翻转、颜色扰动、张量化和ViT归一化。
    参数:
        config: 已解析E1配置，提供尺寸与ColorJitter强度。
    返回值:
        ``transforms.Compose``；单图输入``[H,W,3]``，输出float张量``[3,224,224]``。
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (config.train_resize, config.train_resize),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop((config.train_crop, config.train_crop)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def _collate_train(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """整理训练批次并再次校验每类恰好2张。

    作用:
        堆叠增强图像，构造标签和行号张量；采样协议被破坏时立即终止。
    参数:
        samples: 长度``[B=2P]``的样本列表；每项图像为``[3,224,224]``。
    返回值:
        字典：图像``pixel_values``为``[B,3,224,224]``，标签和行号均为``[B]``。
    """

    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    counts = Counter(labels.tolist())
    if not counts or any(count != 2 for count in counts.values()):
        raise RuntimeError("A training batch violated the one-positive-per-anchor rule.")
    return {
        "indices": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
        "labels": labels,
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
    }


def seed_worker(worker_id: int) -> None:
    """初始化DataLoader工作进程的随机种子。

    作用:
        从PyTorch分配的worker种子派生增强所用的torch随机状态。
    参数:
        worker_id: DataLoader工作进程编号标量；仅用于符合回调签名。
    返回值:
        无。
    """

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed)


def build_train_dataloader(
    dataset: Dataset,
    config: ResolvedLauxConfig,
    use_cuda: bool,
) -> tuple[DataLoader[dict[str, torch.Tensor]], PKBatchSampler]:
    """构建E1严格P×2训练DataLoader。

    作用:
        组合训练数据集、增强、PK采样器、collator和多进程加载设置。
    参数:
        dataset: 长度``[N_train]``的Hugging Face训练表。
        config: 已解析E1配置。
        use_cuda: 是否启用pin memory。
    返回值:
        二元组：DataLoader每批图像为``[B=2P,3,224,224]``、标签为``[B]``；
        ``PKBatchSampler``用于训练循环逐轮调用``set_epoch``。
    """

    labels = [int(label) for label in dataset["label"]]
    sampler = PKBatchSampler(
        labels=labels,
        classes_per_batch=config.classes_per_batch,
        instances_per_class=config.instances_per_class,
        batches_per_epoch=config.batches_per_epoch,
        seed=config.seed,
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TransformedCarsDataset(dataset, build_train_transform(config)),
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=use_cuda,
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate_train,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    return loader, sampler
