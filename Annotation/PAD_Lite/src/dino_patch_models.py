from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_average_patch_tokens(
    patch_tokens: torch.Tensor,
    valid_patch_mask: torch.Tensor,
) -> torch.Tensor:
    """
    方法作用：
        仅对落在有效图像内容区域内的 DINOv2 Patch Token 做平均池化，
        排除 Letterbox 填充区域。
    输入参数：
        patch_tokens (torch.Tensor)：Patch 特征，形状为 [B, N, Dv]。
        valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
    返回值：
        torch.Tensor：每张图的平均 Patch 特征，形状为 [B, Dv]。
        B、N、Dv 分别表示批量大小、Patch 数和 DINO 隐藏维度。
    """
    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must have shape [batch, patches, channels]")
    if valid_patch_mask.ndim != 2:
        raise ValueError("valid_patch_mask must have shape [batch, patches]")
    if patch_tokens.shape[:2] != valid_patch_mask.shape:
        raise ValueError(
            "patch token and mask shapes disagree: "
            f"{tuple(patch_tokens.shape)} vs {tuple(valid_patch_mask.shape)}"
        )
    mask = valid_patch_mask.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
    counts = mask.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every image must contain at least one valid patch")
    return (patch_tokens * mask.unsqueeze(-1)).sum(dim=1) / counts


class DinoMaskedPatchAverageModel(nn.Module):
    """
    方法作用：
        实现 P1a：冻结 DINOv2 主干，使用有效 Patch 平均特征，并训练
        Projection、BNNeck 与 ID 分类头。
    输入参数：
        构造参数见 __init__；主数据输入为图像 [B, 3, H, W] 和掩码 [B, N]。
    返回值：
        DinoMaskedPatchAverageModel：P1a Patch 检索模型实例。
    """

    def __init__(
        self,
        dino_model: nn.Module,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        """
        方法作用：
            初始化冻结主干、Patch 投影层、BNNeck 和基础类分类头。
        输入参数：
            dino_model (nn.Module)：已加载的 DINOv2 主干。
            num_classes (int)：当前训练折的基础类别数 C。
            embedding_dim (int)：输出检索嵌入维度 D。
            dropout (float)：投影前的 Dropout 概率。
        返回值：
            None：完成模型状态初始化。
        """
        super().__init__()
        self.dino = dino_model
        hidden_dim = int(dino_model.config.hidden_size)
        self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self.dino.eval()

    def train(self, mode: bool = True):
        """
        方法作用：
            切换头部训练模式，同时强制冻结的 DINOv2 主干保持 eval 模式。
        输入参数：
            mode (bool)：True 表示训练头部，False 表示整体评估。
        返回值：
            DinoMaskedPatchAverageModel：当前模型实例，便于链式调用。
        """
        super().train(mode)
        # The backbone is a fixed feature extractor in P1a. Keeping it in eval
        # mode also makes the experimental condition explicit and reproducible.
        self.dino.eval()
        return self

    def _patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            无梯度运行 DINOv2，并去掉序列首部的 CLS Token。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
        返回值：
            torch.Tensor：Patch Token，形状为 [B, N, Dv]。
        """
        with torch.no_grad():
            hidden = self.dino(pixel_values=pixel_values).last_hidden_state
        return hidden[:, 1:].float()

    def encode(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        方法作用：
            对有效 Patch 做平均池化，再经过 Projection 和 BNNeck 得到
            L2 归一化检索嵌入。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
            valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
        返回值：
            torch.Tensor：归一化 Patch 检索特征，形状为 [B, D]。
        """
        patches = self._patch_tokens(pixel_values)
        if patches.shape[1] != valid_patch_mask.shape[1]:
            raise ValueError(
                f"DINO returned {patches.shape[1]} patches, but the mask has "
                f"{valid_patch_mask.shape[1]} entries"
            )
        pooled = masked_average_patch_tokens(patches, valid_patch_mask)
        projected = self.projection(self.dropout(pooled))
        return F.normalize(self.bnneck(projected), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：
            执行 P1a 前向传播，并用检索嵌入计算基础类 ID logits。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
            valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
        返回值：
            tuple[torch.Tensor, torch.Tensor]：检索嵌入 [B, D] 和分类
            logits [B, C]。
        """
        embedding = self.encode(pixel_values, valid_patch_mask)
        return embedding, self.classifier(embedding)

    def head_parameters(self) -> list[nn.Parameter]:
        """
        方法作用：
            收集 P1a 可训练的 Projection、BNNeck 和分类头参数。
        输入参数：
            无。
        返回值：
            list[nn.Parameter]：供优化器使用的可训练参数列表。
        """
        modules = (self.projection, self.bnneck, self.classifier)
        return [parameter for module in modules for parameter in module.parameters()]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """
        方法作用：
            导出部署/恢复所需的轻量头部权重，不包含冻结 DINOv2 主干。
        输入参数：
            无。
        返回值：
            dict[str, torch.Tensor]：名称到 CPU 权重张量的映射。
        """
        prefixes = ("projection.", "bnneck.", "classifier.")
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name.startswith(prefixes)
        }
