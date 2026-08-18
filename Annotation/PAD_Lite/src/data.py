from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class CropRecord:
    """
    方法作用：
        保存单个目标裁剪样本的路径、类别及质量标记等元数据。
    
    输入参数：
        sample_id (str)：初始化实例所需的 sample_id 参数。
        path (Path)：目标文件或目录路径。
        source_image (str)：初始化实例所需的 source_image 参数。
        class_id (int)：初始化实例所需的 class_id 参数。
        class_name (str)：初始化实例所需的 class_name 参数。
        anchor_id (str)：初始化实例所需的 anchor_id 参数。
        tiny (bool)：初始化实例所需的 tiny 参数。
        crowded (bool)：初始化实例所需的 crowded 参数。
        clipped (bool)：初始化实例所需的 clipped 参数。
    
    返回值：
        CropRecord：初始化后的类实例。
    """
    sample_id: str
    path: Path
    source_image: str
    class_id: int
    class_name: str
    anchor_id: str
    tiny: bool
    crowded: bool
    clipped: bool


class CropIndex:
    """
    方法作用：
        加载裁剪样本清单，并按来源图片建立可查询的样本索引。
    
    输入参数：
        crop_root (Path)：初始化实例所需的 crop_root 参数。
    
    返回值：
        CropIndex：初始化后的类实例。
    """

    def __init__(self, crop_root: Path) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            crop_root (Path)：方法所需的 crop_root 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        self.root = crop_root
        manifest_path = crop_root / "crop_manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if raw.get("meta", {}).get("version") != "russian_pad_crops_v1":
            raise ValueError(f"Unsupported crop manifest: {manifest_path}")
        self.classes = [item["name"] for item in raw["classes"]]
        self.records: list[CropRecord] = []
        self.by_source: dict[str, list[CropRecord]] = defaultdict(list)
        for item in raw["samples"]:
            record = CropRecord(
                sample_id=item["id"],
                path=crop_root / item["file"],
                source_image=item["source_image"],
                class_id=int(item["class_id"]),
                class_name=item["class"],
                anchor_id=item["anchor_id"],
                tiny=bool(item["tiny"]),
                crowded=bool(item["crowded"]),
                clipped=bool(item["clipped"]),
            )
            if not record.path.is_file():
                raise FileNotFoundError(f"Missing crop: {record.path}")
            self.records.append(record)
            self.by_source[record.source_image].append(record)

    def records_for_sources(
        self, source_names: Sequence[str], exclude_tiny: bool = False
    ) -> list[CropRecord]:
        """
        方法作用：
            根据来源图片名称获取对应的目标裁剪记录。
        
        输入参数：
            self (Any)：当前实例。
            source_names (Sequence[str])：方法所需的 source_names 参数。
            exclude_tiny (bool)：方法所需的 exclude_tiny 参数。
        
        返回值：
            list[CropRecord]：方法执行得到的结果。
        """
        records: list[CropRecord] = []
        for source_name in source_names:
            if source_name not in self.by_source:
                raise KeyError(f"Split references unknown source image: {source_name}")
            for record in self.by_source[source_name]:
                if exclude_tiny and record.tiny:
                    continue
                records.append(record)
        return records


def load_fold(splits_root: Path, fold_number: int) -> dict:
    """
    方法作用：
        读取指定折编号的划分配置文件，确认当前折的基础类别、新类别和样本分区信息。

    输入参数：
        splits_root (Path):
            存放 fold_*.json 文件的目录。
        fold_number (int):
            需要加载的折编号。

    返回值：
        dict:
            当前折的划分字典。
    """
    if not isinstance(fold_number, int) or fold_number < 1:
        raise ValueError(f"fold must be a positive integer, got {fold_number}")
    path = splits_root / f"fold_{fold_number:02d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold file: {path}")
    fold = json.loads(path.read_text(encoding="utf-8"))
    if fold.get("version") != "pad_lite_class_holdout_fold_v1":
        raise ValueError(f"Unsupported fold file: {path}")
    return fold


def partition_source_names(fold: dict, partition: str) -> list[str]:
    """
    方法作用：
        执行 partition_source_names 对应的处理流程。
    
    输入参数：
        fold (dict)：交叉验证折信息或折编号。
        partition (str)：方法所需的 partition 参数。
    
    返回值：
        list[str]：方法执行得到的结果。
    """
    by_class = fold["partitions"][partition]
    return [name for class_names in by_class.values() for name in class_names]


def records_for_partition(
    index: CropIndex,
    fold: dict,
    partition: str,
    exclude_tiny: bool = False,
) -> list[CropRecord]:
    """
    方法作用：
        执行 records_for_partition 对应的处理流程。
    
    输入参数：
        index (CropIndex)：待访问的样本索引。
        fold (dict)：交叉验证折信息或折编号。
        partition (str)：方法所需的 partition 参数。
        exclude_tiny (bool)：方法所需的 exclude_tiny 参数。
    
    返回值：
        list[CropRecord]：方法执行得到的结果。
    """
    return index.records_for_sources(
        partition_source_names(fold, partition),
        exclude_tiny=exclude_tiny,
    )


def build_eval_transform(image_size: int):
    """
    方法作用：
        执行 build_eval_transform 对应的处理流程。
    
    输入参数：
        image_size (int)：输出图像张量的高和宽 H=W=image_size。
    
    返回值：
        transforms.Compose：将单张 PIL 图片转换为 [3, H, W] 归一化张量的评估变换。
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_train_transform(image_size: int):
    """
    方法作用：
        执行 build_train_transform 对应的处理流程。
    
    输入参数：
        image_size (int)：增强后图像张量的高和宽 H=W=image_size。
    
    返回值：
        transforms.Compose：将单张 PIL 图片增强并转换为 [3, H, W] 张量的训练变换。
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
                ratio=(0.75, 1.333333),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.RandomPerspective(
                        distortion_scale=0.15,
                        p=1.0,
                        interpolation=InterpolationMode.BICUBIC,
                    )
                ],
                p=0.25,
            ),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.04,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
            transforms.RandomErasing(
                p=0.20,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


class CropDataset(Dataset):
    """
    方法作用：
        将裁剪记录封装为可供 PyTorch 加载和预处理的数据集。
    
    输入参数：
        records (Sequence[CropRecord])：初始化实例所需的 records 参数。
        transform (Any)：初始化实例所需的 transform 参数。
        class_to_local_id (dict[str, int])：初始化实例所需的 class_to_local_id 参数。
    
    返回值：
        CropDataset：初始化后的类实例。
    """

    def __init__(
        self,
        records: Sequence[CropRecord],
        transform,
        class_to_local_id: dict[str, int],
    ) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            records (Sequence[CropRecord])：方法所需的 records 参数。
            transform (Any)：方法所需的 transform 参数。
            class_to_local_id (dict[str, int])：方法所需的 class_to_local_id 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        self.records = list(records)
        self.transform = transform
        self.class_to_local_id = dict(class_to_local_id)
        missing_classes = {
            record.class_name for record in self.records
        } - set(self.class_to_local_id)
        if missing_classes:
            raise ValueError(f"Dataset class map is missing: {sorted(missing_classes)}")

    def __len__(self) -> int:
        """
        方法作用：
            返回当前数据集或采样器包含的元素数量。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            int：方法执行得到的结果。
        """
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        """
        方法作用：
            读取指定索引的样本并完成图像预处理。
        
        输入参数：
            self (Any)：当前实例。
            index (int)：待访问的样本索引。
        
        返回值：
            dict：单样本字典；其中 image 为 [3, H, W]，label 为标量类别编号，
            其余字段为样本标识和质量元数据。
        """
        record = self.records[index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": self.class_to_local_id[record.class_name],
            "global_class_id": record.class_id,
            "class_name": record.class_name,
            "sample_id": record.sample_id,
            "source_image": record.source_image,
            "tiny": record.tiny,
            "crowded": record.crowded,
            "clipped": record.clipped,
        }


class PKBatchSampler(BatchSampler):
    """
    方法作用：
        按照每批 P 个类别、每类 K 个实例的规则生成均衡批次。
    
    输入参数：
        labels (Sequence[int])：初始化实例所需的 labels 参数。
        classes_per_batch (int)：初始化实例所需的 classes_per_batch 参数。
        instances_per_class (int)：初始化实例所需的 instances_per_class 参数。
        seed (int)：初始化实例所需的 seed 参数。
    
    返回值：
        PKBatchSampler：初始化后的类实例。
    """

    def __init__(
        self,
        labels: Sequence[int],
        classes_per_batch: int,
        instances_per_class: int,
        seed: int,
    ) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            labels (Sequence[int])：方法所需的 labels 参数。
            classes_per_batch (int)：方法所需的 classes_per_batch 参数。
            instances_per_class (int)：方法所需的 instances_per_class 参数。
            seed (int)：方法所需的 seed 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        self.classes_per_batch = int(classes_per_batch)
        self.instances_per_class = int(instances_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_class[int(label)].append(index)
        if len(self.by_class) < self.classes_per_batch:
            raise ValueError(
                f"P={self.classes_per_batch} exceeds available classes={len(self.by_class)}"
            )
        if self.instances_per_class < 2:
            raise ValueError("K must be at least 2 for triplet loss")
        self.batch_size = self.classes_per_batch * self.instances_per_class
        self.num_batches = max(1, math.ceil(len(labels) / self.batch_size))

    def __len__(self) -> int:
        """
        方法作用：
            返回当前数据集或采样器包含的元素数量。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            int：方法执行得到的结果。
        """
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """
        方法作用：
            设置采样器当前轮次，以改变确定性随机采样序列。
        
        输入参数：
            self (Any)：当前实例。
            epoch (int)：当前训练轮次。
        
        返回值：
            None：方法直接完成相应操作。
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """
        方法作用：
            逐批生成符合 P-K 规则的训练样本索引。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            Iterator[list[int]]：每次返回长度为 B=P×K 的样本索引列表。
        """
        rng = random.Random(self.seed + self.epoch * 1009)
        classes = sorted(self.by_class)
        for _ in range(self.num_batches):
            selected_classes = rng.sample(classes, self.classes_per_batch)
            batch: list[int] = []
            for label in selected_classes:
                candidates = self.by_class[label]
                if len(candidates) >= self.instances_per_class:
                    batch.extend(rng.sample(candidates, self.instances_per_class))
                else:
                    batch.extend(
                        rng.choice(candidates) for _ in range(self.instances_per_class)
                    )
            rng.shuffle(batch)
            yield batch


def build_train_loader(
    dataset: CropDataset,
    classes_per_batch: int,
    instances_per_class: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, PKBatchSampler]:
    """
    方法作用：
        执行 build_train_loader 对应的处理流程。
    
    输入参数：
        dataset (CropDataset)：方法所需的 dataset 参数。
        classes_per_batch (int)：方法所需的 classes_per_batch 参数。
        instances_per_class (int)：方法所需的 instances_per_class 参数。
        workers (int)：数据加载工作进程数量。
        seed (int)：方法所需的 seed 参数。
    
    返回值：
        tuple[DataLoader, PKBatchSampler]：训练加载器和采样器；加载器每批输出
        image [B, 3, H, W] 与 label [B]，其中 B=classes_per_batch×instances_per_class。
    """
    labels = [dataset.class_to_local_id[item.class_name] for item in dataset.records]
    sampler = PKBatchSampler(
        labels,
        classes_per_batch=classes_per_batch,
        instances_per_class=instances_per_class,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return loader, sampler


def build_eval_loader(dataset: CropDataset, batch_size: int, workers: int) -> DataLoader:
    """
    方法作用：
        执行 build_eval_loader 对应的处理流程。
    
    输入参数：
        dataset (CropDataset)：方法所需的 dataset 参数。
        batch_size (int)：方法所需的 batch_size 参数。
        workers (int)：数据加载工作进程数量。
    
    返回值：
        DataLoader：按顺序输出批数据的加载器；每批包含 image [B, 3, H, W]
        和 label [B]，最后一批的 B 可以小于 batch_size。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
