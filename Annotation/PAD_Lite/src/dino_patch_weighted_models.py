from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax_patch_pool(
    patch_tokens: torch.Tensor,
    valid_patch_mask: torch.Tensor,
    patch_scorer: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    方法作用：
        用 Patch Scorer 为每张图的有效 Patch 计算 Softmax 权重并加权池化，
        保证填充 Patch 的权重严格为零。
    输入参数：
        patch_tokens (torch.Tensor)：Patch 特征，形状为 [B, N, Dv]。
        valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
        patch_scorer (nn.Module)：把每个 [Dv] Token 映射为标量的打分器。
    返回值：
        tuple[torch.Tensor, torch.Tensor]：池化特征 [B, Dv] 和 Patch
        权重 [B, N]。
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
    mask = valid_patch_mask.to(device=patch_tokens.device, dtype=torch.bool)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("Every image must contain at least one valid patch")

    scores = patch_scorer(patch_tokens).squeeze(-1)
    scores = scores.masked_fill(~mask, float("-inf"))
    # Compute the normalization in FP32 for stable AMP training. Casting back
    # preserves the feature dtype while invalid positions remain exactly zero.
    weights = torch.softmax(scores.float(), dim=-1).to(dtype=patch_tokens.dtype)
    weights = weights.masked_fill(~mask, 0.0)
    pooled = (weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
    return pooled, weights


class DinoMaskedWeightedPatchModel(nn.Module):
    """
    方法作用：
        实现 P2a：冻结 DINOv2，通过可学习 Patch Scorer 动态聚合局部特征，
        再训练 Projection、BNNeck 与 ID 分类头。
    输入参数：
        构造参数见 __init__；主数据输入为图像 [B, 3, H, W] 和掩码 [B, N]。
    返回值：
        DinoMaskedWeightedPatchModel：P2a 加权 Patch 检索模型实例。
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
            初始化冻结 DINOv2、零初始化 Patch Scorer 和检索/分类头。
        输入参数：
            dino_model (nn.Module)：已加载的 DINOv2 主干。
            num_classes (int)：基础类别数 C。
            embedding_dim (int)：检索嵌入维度 D。
            dropout (float)：投影前的 Dropout 概率。
        返回值：
            None：完成模型初始化。
        """
        super().__init__()
        self.dino = dino_model
        hidden_dim = int(dino_model.config.hidden_size)

        # Keep the P1a head construction order so a fixed random seed gives the
        # same projection/classifier initialization. The new scorer is appended
        # and then reset to zero, making epoch zero exactly uniform pooling.
        self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.patch_scorer = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.patch_scorer.weight)
        nn.init.zeros_(self.patch_scorer.bias)

        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self.dino.eval()

    def train(self, mode: bool = True):
        """
        方法作用：
            切换可训练头部模式，同时保持冻结 DINOv2 处于 eval 模式。
        输入参数：
            mode (bool)：头部是否进入训练模式。
        返回值：
            DinoMaskedWeightedPatchModel：当前模型实例。
        """
        super().train(mode)
        self.dino.eval()
        return self

    def _patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            无梯度提取 DINOv2 Patch Token，并移除 CLS Token。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
        返回值：
            torch.Tensor：Patch Token，形状为 [B, N, Dv]。
        """
        with torch.no_grad():
            hidden = self.dino(pixel_values=pixel_values).last_hidden_state
        return hidden[:, 1:].float()

    def encode_with_weights(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：
            计算动态 Patch 权重、加权池化并生成归一化检索嵌入。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
            valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
        返回值：
            tuple[torch.Tensor, torch.Tensor]：检索嵌入 [B, D] 和 Patch
            权重 [B, N]。
        """
        patches = self._patch_tokens(pixel_values)
        if patches.shape[1] != valid_patch_mask.shape[1]:
            raise ValueError(
                f"DINO returned {patches.shape[1]} patches, but the mask has "
                f"{valid_patch_mask.shape[1]} entries"
            )
        pooled, weights = masked_softmax_patch_pool(
            patches, valid_patch_mask, self.patch_scorer
        )
        projected = self.projection(self.dropout(pooled))
        embedding = F.normalize(self.bnneck(projected), dim=-1)
        return embedding, weights

    def encode(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        方法作用：
            生成归一化检索嵌入，忽略仅用于诊断的 Patch 权重返回值。
        输入参数：
            pixel_values (torch.Tensor)：归一化图像，形状为 [B, 3, H, W]。
            valid_patch_mask (torch.Tensor)：有效 Patch 掩码，形状为 [B, N]。
        返回值：
            torch.Tensor：检索嵌入，形状为 [B, D]。
        """
        embedding, _ = self.encode_with_weights(pixel_values, valid_patch_mask)
        return embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：
            执行 P2a 前向传播并计算基础类 ID 分类结果。
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
            收集 Patch Scorer、Projection、BNNeck 和分类头的全部参数。
        输入参数：
            无。
        返回值：
            list[nn.Parameter]：完整可训练头部参数列表。
        """
        modules = (
            self.patch_scorer,
            self.projection,
            self.bnneck,
            self.classifier,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def scorer_parameters(self) -> list[nn.Parameter]:
        """
        方法作用：
            单独取得 Patch Scorer 参数，供梯度或优化器诊断使用。
        输入参数：
            无。
        返回值：
            list[nn.Parameter]：Patch Scorer 参数列表。
        """
        return list(self.patch_scorer.parameters())

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """
        方法作用：
            导出 P2a 轻量适配器权重，不保存冻结 DINOv2 主干。
        输入参数：
            无。
        返回值：
            dict[str, torch.Tensor]：名称到 CPU 权重张量的映射。
        """
        prefixes = (
            "patch_scorer.",
            "projection.",
            "bnneck.",
            "classifier.",
        )
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name.startswith(prefixes)
        }
