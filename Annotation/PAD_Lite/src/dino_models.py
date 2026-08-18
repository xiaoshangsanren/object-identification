from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_local_dino(model_dir: Path):
    """
    方法作用：
        从本地目录加载 DINOv2 模型与图像处理器。
    
    输入参数：
        model_dir (Path)：方法所需的 model_dir 参数。
    
    返回值：
        Any：方法执行得到的结果。
    """
    from transformers import AutoImageProcessor, Dinov2Model

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local DINOv2 model directory does not exist: {model_dir}")
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model = Dinov2Model.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    return model, processor


def _dino_encoder_layers(dino_model: nn.Module) -> nn.ModuleList:
    """
    方法作用：
        获取 DINOv2 主干网络中的 Transformer 编码层。
    
    输入参数：
        dino_model (nn.Module)：DINOv2 模型实例。
    
    返回值：
        nn.ModuleList：方法执行得到的结果。
    """
    layers = dino_model.encoder.layer
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Unsupported DINOv2 encoder layer layout")
    return layers


class FrozenDinoEncoder(nn.Module):
    """
    方法作用：
        冻结 DINOv2 参数并将其作为归一化图像特征提取器。
    
    输入参数：
        dino_model (nn.Module)：DINOv2 模型实例。
    
    返回值：
        FrozenDinoEncoder：初始化后的类实例。
    """

    def __init__(self, dino_model: nn.Module) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            dino_model (nn.Module)：DINOv2 模型实例。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        super().__init__()
        self.dino = dino_model
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self.dino.eval()

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            将输入图像编码为归一化的检索特征向量。
        
        输入参数：
            self (Any)：当前实例。
            pixel_values (torch.Tensor)：经过预处理的批量图像，形状为 [B, 3, H, W]。
        
        返回值：
            torch.Tensor：L2 归一化的 DINOv2 CLS 特征，形状为 [B, D_dino]。
        """
        output = self.dino(pixel_values=pixel_values)
        features = output.last_hidden_state[:, 0].float()
        return F.normalize(features, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            执行模型的前向传播。
        
        输入参数：
            self (Any)：当前实例。
            pixel_values (torch.Tensor)：经过预处理的批量图像，形状为 [B, 3, H, W]。
        
        返回值：
            torch.Tensor：L2 归一化的 DINOv2 CLS 特征，形状为 [B, D_dino]。
        """
        return self.encode(pixel_values)


class DinoRetrievalModel(nn.Module):
    """
    方法作用：
        在 DINOv2 特征之上构建可训练的检索与分类模型。
    
    输入参数：
        dino_model (nn.Module)：DINOv2 模型实例。
        num_classes (int)：初始化实例所需的 num_classes 参数。
        embedding_dim (int)：初始化实例所需的 embedding_dim 参数。
        dropout (float)：初始化实例所需的 dropout 参数。
    
    返回值：
        DinoRetrievalModel：初始化后的类实例。
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
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            dino_model (nn.Module)：DINOv2 模型实例。
            num_classes (int)：方法所需的 num_classes 参数。
            embedding_dim (int)：方法所需的 embedding_dim 参数。
            dropout (float)：方法所需的 dropout 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        super().__init__()
        self.dino = dino_model
        hidden_dim = int(dino_model.config.hidden_size)
        self.projection = nn.Linear(hidden_dim, embedding_dim, bias=False)
        if hidden_dim == embedding_dim:
            nn.init.eye_(self.projection.weight)
        else:
            nn.init.xavier_uniform_(self.projection.weight)
        self.dropout = nn.Dropout(dropout)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self._backbone_parameters: list[nn.Parameter] = []
        self._backbone_parameter_names: set[str] = set()
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)

    def configure_backbone(self, last_n_blocks: int, enabled: bool) -> None:
        """
        方法作用：
            选择需要微调的主干网络末端层并设置其训练状态。
        
        输入参数：
            self (Any)：当前实例。
            last_n_blocks (int)：方法所需的 last_n_blocks 参数。
            enabled (bool)：是否启用对应功能或参数训练。
        
        返回值：
            None：方法直接完成相应操作。
        """
        layers = _dino_encoder_layers(self.dino)
        if not 0 <= last_n_blocks <= len(layers):
            raise ValueError(
                f"last_n_blocks={last_n_blocks}, but DINOv2 has {len(layers)} blocks"
            )
        target_modules: list[nn.Module] = (
            list(layers[-last_n_blocks:]) if last_n_blocks else []
        )
        if last_n_blocks:
            target_modules.append(self.dino.layernorm)
        target_ids = {
            id(parameter)
            for module in target_modules
            for parameter in module.parameters()
        }
        self._backbone_parameters = []
        self._backbone_parameter_names = set()
        for name, parameter in self.named_parameters():
            if id(parameter) in target_ids:
                parameter.requires_grad_(enabled)
                self._backbone_parameters.append(parameter)
                self._backbone_parameter_names.add(name)

    def set_backbone_enabled(self, enabled: bool) -> None:
        """
        方法作用：
            启用或冻结已选中的主干网络参数。
        
        输入参数：
            self (Any)：当前实例。
            enabled (bool)：是否启用对应功能或参数训练。
        
        返回值：
            None：方法直接完成相应操作。
        """
        for parameter in self._backbone_parameters:
            parameter.requires_grad_(enabled)

    def backbone_parameters(self) -> list[nn.Parameter]:
        """
        方法作用：
            获取参与微调的主干网络参数。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            list[nn.Parameter]：方法执行得到的结果。
        """
        return list(self._backbone_parameters)

    def head_parameters(self) -> list[nn.Parameter]:
        """
        方法作用：
            获取检索投影头、归一化层和分类头的参数。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            list[nn.Parameter]：方法执行得到的结果。
        """
        modules = (self.projection, self.bnneck, self.classifier)
        return [parameter for module in modules for parameter in module.parameters()]

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            将输入图像编码为归一化的检索特征向量。
        
        输入参数：
            self (Any)：当前实例。
            pixel_values (torch.Tensor)：经过预处理的批量图像，形状为 [B, 3, H, W]。
        
        返回值：
            torch.Tensor：投影并归一化后的检索特征，形状为 [B, D]。
        """
        output = self.dino(pixel_values=pixel_values)
        base = output.last_hidden_state[:, 0].float()
        projected = self.projection(self.dropout(base))
        neck = self.bnneck(projected)
        return F.normalize(neck, dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        方法作用：
            执行模型的前向传播。
        
        输入参数：
            self (Any)：当前实例。
            pixel_values (torch.Tensor)：经过预处理的批量图像，形状为 [B, 3, H, W]。
        
        返回值：
            tuple[torch.Tensor, torch.Tensor]：依次返回检索特征 [B, D] 和
            基础类别分类 logits [B, C]。
        """
        embedding = self.encode(pixel_values)
        logits = self.classifier(embedding)
        return embedding, logits

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """
        方法作用：
            导出适配器及已解冻主干层的参数状态。
        
        输入参数：
            self (Any)：当前实例。
        
        返回值：
            dict[str, torch.Tensor]：方法执行得到的结果。
        """
        state = self.state_dict()
        prefixes = ("projection.", "bnneck.", "classifier.")
        return {
            name: tensor.detach().cpu()
            for name, tensor in state.items()
            if name.startswith(prefixes) or name in self._backbone_parameter_names
        }
