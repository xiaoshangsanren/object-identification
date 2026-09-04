"""Pure-vision ImageNet ViT feature extractor."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification


class PureVisualViT(nn.Module):
    """不含分类头和文本分支的纯视觉ViT检索模型。

    作用:
        从ImageNet分类权重中只保留ViT视觉主干，输出最终层CLS检索特征。
    参数:
        model_path: 本地``google/vit-base-patch16-224``权重目录。
    返回值:
        PyTorch模型；前向输入图像``[B,3,224,224]``，输出归一化特征``[B,768]``。
    """

    def __init__(self, model_path: Path) -> None:
        """加载预训练权重并移除ImageNet分类头。

        作用:
            先按原始分类模型读取完整checkpoint，再仅注册其中的``vit``视觉主干。
        参数:
            self: 当前模型实例。
            model_path: 本地预训练模型目录。
        返回值:
            无。
        """

        super().__init__()
        pretrained_model = ViTForImageClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        # Keep only the pretrained visual transformer. The ImageNet classifier
        # is deliberately not registered in this retrieval model.
        self.backbone = pretrained_model.vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """将一个图像批次编码为归一化CLS特征。

        作用:
            执行ViT前向，取得``last_hidden_state``的第0个CLS token并进行L2归一化。
        参数:
            self: 当前模型实例。
            pixel_values: 归一化图像张量，形状``[B,3,224,224]``。
        返回值:
            float32单位向量，形状``[B,768]``。
        """

        output = self.backbone(pixel_values=pixel_values)
        cls_embedding = output.last_hidden_state[:, 0, :]
        return F.normalize(cls_embedding.float(), dim=-1)


def load_image_processor(model_path: Path) -> AutoImageProcessor:
    """加载与ViT权重匹配的测试图像处理器。

    作用:
        从本地``preprocessor_config.json``读取缩放、均值和标准差，不访问网络。
    参数:
        model_path: 本地ViT模型目录。
    返回值:
        Hugging Face ``AutoImageProcessor``；可把``B``张PIL图像转换为``[B,3,224,224]``。
    """

    return AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )


@torch.inference_mode()
def extract_embeddings(
    model: PureVisualViT,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按数据集原始顺序提取全部纯视觉特征。

    作用:
        在inference mode和可选AMP下遍历DataLoader，并检查批次行号连续且未被打乱。
    参数:
        model: 纯视觉ViT或其DataParallel包装；输入``[B,3,224,224]``、输出``[B,768]``。
        dataloader: 特征提取DataLoader；每批包含图像``[B,3,224,224]``和标签、行号``[B]``。
        device: 执行前向的CPU或CUDA设备。
        amp: 是否在CUDA前向阶段启用float16自动混合精度。
    返回值:
        二元组：CPU视觉特征``[N,768]``和对应类别标签``[N]``。
    """

    model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    expected_index = 0

    for batch in dataloader:
        indices = batch["indices"]
        expected = torch.arange(expected_index, expected_index + len(indices))
        if not torch.equal(indices, expected):
            raise RuntimeError("Feature loader changed test-set row order.")
        expected_index += len(indices)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            features = model(pixel_values)
        feature_batches.append(features.cpu())
        label_batches.append(batch["labels"].clone())

    if not feature_batches:
        raise ValueError("The test split is empty.")
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)
