from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_feature_tensor(output: Any, modality: str) -> torch.Tensor:
    """
    方法作用：
        从模型输出中提取指定模态的特征张量。
    
    输入参数：
        output (Any)：模型原始输出；其中目标特征通常为 [B, D]，
        `last_hidden_state` 则通常为 [B, L, D]。
        modality (str)：方法所需的 modality 参数。
    
    返回值：
        torch.Tensor：提取后的批量特征，形状为 [B, D]。
    """
    if torch.is_tensor(output):
        return output
    preferred = (
        ("image_embeds", "pooler_output", "last_hidden_state")
        if modality == "image"
        else ("text_embeds", "pooler_output", "last_hidden_state")
    )
    for name in preferred:
        value = getattr(output, name, None)
        if value is not None:
            return value[:, 0] if name == "last_hidden_state" and value.ndim == 3 else value
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0][:, 0] if output[0].ndim == 3 else output[0]
    raise TypeError(f"Cannot extract {modality} tensor from {type(output).__name__}")


def load_local_clip(model_dir: Path):
    """
    方法作用：
        从本地目录加载 CLIP 模型与处理器。
    
    输入参数：
        model_dir (Path)：方法所需的 model_dir 参数。
    
    返回值：
        Any：方法执行得到的结果。
    """
    from transformers import CLIPModel, CLIPProcessor

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local CLIP model directory does not exist: {model_dir}")
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    return model, processor


def _vision_encoder_layers(clip_model: nn.Module) -> nn.ModuleList:
    """
    方法作用：
        获取 CLIP 视觉编码器中的 Transformer 层。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
    
    返回值：
        nn.ModuleList：方法执行得到的结果。
    """
    vision = clip_model.vision_model
    if hasattr(vision, "vision_model"):
        vision = vision.vision_model
    layers = vision.encoder.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Unsupported CLIP vision encoder layer layout")
    return layers


def _vision_post_layer_norm(clip_model: nn.Module) -> nn.Module | None:
    """
    方法作用：
        获取 CLIP 视觉编码器末端的归一化层。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
    
    返回值：
        nn.Module | None：方法执行得到的结果。
    """
    vision = clip_model.vision_model
    if hasattr(vision, "vision_model"):
        vision = vision.vision_model
    return getattr(vision, "post_layernorm", None)


def _text_token_embedding(clip_model: nn.Module) -> nn.Module:
    """
    方法作用：
        获取 CLIP 文本编码器的词元嵌入层。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
    
    返回值：
        nn.Module：方法执行得到的结果。
    """
    text = clip_model.text_model
    if hasattr(text, "text_model"):
        text = text.text_model
    return text.embeddings.token_embedding


class FrozenClipEncoder(nn.Module):
    """
    方法作用：
        冻结 CLIP 参数并将其作为归一化图像特征提取器。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
    
    返回值：
        FrozenClipEncoder：初始化后的类实例。
    """
    def __init__(self, clip_model: nn.Module) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            clip_model (nn.Module)：CLIP 模型实例。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        super().__init__()
        self.clip = clip_model
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        方法作用：
            执行模型的前向传播。
        
        输入参数：
            self (Any)：当前实例。
            pixel_values (torch.Tensor)：经过预处理的批量图像，形状为 [B, 3, H, W]。
        
        返回值：
            torch.Tensor：L2 归一化的 CLIP 图像特征，形状为 [B, D_clip]。
        """
        output = self.clip.get_image_features(pixel_values=pixel_values)
        features = extract_feature_tensor(output, "image").float()
        return F.normalize(features, dim=-1)


class RetrievalModel(nn.Module):
    """
    方法作用：
        在 CLIP 图像特征之上构建投影层、BN Neck 和分类头。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
        num_classes (int)：初始化实例所需的 num_classes 参数。
        embedding_dim (int)：初始化实例所需的 embedding_dim 参数。
        dropout (float)：初始化实例所需的 dropout 参数。
    
    返回值：
        RetrievalModel：初始化后的类实例。
    """
    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            clip_model (nn.Module)：CLIP 模型实例。
            num_classes (int)：方法所需的 num_classes 参数。
            embedding_dim (int)：方法所需的 embedding_dim 参数。
            dropout (float)：方法所需的 dropout 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        super().__init__()
        self.clip = clip_model
        clip_dim = int(clip_model.config.projection_dim)
        self.projection = nn.Linear(clip_dim, embedding_dim, bias=False)
        if clip_dim == embedding_dim:
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
        for parameter in self.clip.parameters():
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
        layers = _vision_encoder_layers(self.clip)
        if not 0 <= last_n_blocks <= len(layers):
            raise ValueError(
                f"last_n_blocks={last_n_blocks}, but CLIP has {len(layers)} blocks"
            )
        target_modules: list[nn.Module] = list(layers[-last_n_blocks:]) if last_n_blocks else []
        post_norm = _vision_post_layer_norm(self.clip)
        if last_n_blocks and post_norm is not None:
            target_modules.append(post_norm)
        target_ids = {id(parameter) for module in target_modules for parameter in module.parameters()}
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
        output = self.clip.get_image_features(pixel_values=pixel_values)
        base = extract_feature_tensor(output, "image").float()
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
            基础类别分类 logits [B, C]。B 为批量大小，D 为嵌入维度，
            C 为当前折的基础类别数。
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


class TextAnchorPrompt(nn.Module):
    """
    方法作用：
        构造可学习的类别文本上下文，为 B2 提供文本锚点。
    
    输入参数：
        clip_model (nn.Module)：CLIP 模型实例。
        tokenizer (Any)：初始化实例所需的 tokenizer 参数。
        num_classes (int)：初始化实例所需的 num_classes 参数。
        context_tokens (int)：初始化实例所需的 context_tokens 参数。
        prefix (str)：初始化实例所需的 prefix 参数。
        suffix (str)：初始化实例所需的 suffix 参数。
        init_std (float)：初始化实例所需的 init_std 参数。
    
    返回值：
        TextAnchorPrompt：初始化后的类实例。
    """

    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        num_classes: int,
        context_tokens: int,
        prefix: str,
        suffix: str,
        init_std: float,
    ) -> None:
        """
        方法作用：
            初始化当前对象及其运行所需的状态。
        
        输入参数：
            self (Any)：当前实例。
            clip_model (nn.Module)：CLIP 模型实例。
            tokenizer (Any)：方法所需的 tokenizer 参数。
            num_classes (int)：方法所需的 num_classes 参数。
            context_tokens (int)：方法所需的 context_tokens 参数。
            prefix (str)：方法所需的 prefix 参数。
            suffix (str)：方法所需的 suffix 参数。
            init_std (float)：方法所需的 init_std 参数。
        
        返回值：
            None：仅完成实例初始化，不返回数据。
        """
        super().__init__()
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        token_embedding = _text_token_embedding(clip_model)
        hidden_dim = int(token_embedding.embedding_dim)
        self.context = nn.Parameter(
            torch.empty(num_classes, context_tokens, hidden_dim)
        )
        nn.init.normal_(self.context, std=init_std)

        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        if prefix_ids and isinstance(prefix_ids[0], list):
            prefix_ids = prefix_ids[0]
        if suffix_ids and isinstance(suffix_ids[0], list):
            suffix_ids = suffix_ids[0]
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        if bos_id is None or eos_id is None:
            raise ValueError("CLIP tokenizer must provide BOS and EOS token IDs")
        max_length = min(int(getattr(tokenizer, "model_max_length", 77)), 77)
        context_start = 1 + len(prefix_ids)
        eos_position = context_start + context_tokens + len(suffix_ids)
        if eos_position >= max_length:
            raise ValueError("Prompt is longer than the CLIP text context")
        filler_id = prefix_ids[-1] if prefix_ids else bos_id
        ids = [bos_id] + list(prefix_ids) + [filler_id] * context_tokens + list(suffix_ids) + [eos_id]
        attention = [1] * len(ids)
        ids.extend([pad_id] * (max_length - len(ids)))
        attention.extend([0] * (max_length - len(attention)))
        self.register_buffer(
            "input_ids", torch.tensor(ids, dtype=torch.long).repeat(num_classes, 1)
        )
        self.register_buffer(
            "attention_mask",
            torch.tensor(attention, dtype=torch.long).repeat(num_classes, 1),
        )
        self.context_start = context_start
        self.context_tokens = context_tokens
        self.template = f"{prefix} {{learnable_tokens}} {suffix}"

    @contextmanager
    def _inject_context(self, token_embedding: nn.Module) -> Iterator[None]:
        """
        方法作用：
            临时将可学习上下文注入 CLIP 的词元嵌入过程。
        
        输入参数：
            self (Any)：当前实例。
            token_embedding (nn.Module)：方法所需的 token_embedding 参数。
        
        返回值：
            Iterator[None]：方法执行得到的结果。
        """
        start = self.context_start
        stop = start + self.context_tokens

        def hook(_module, _inputs, output):
            """
            方法作用：
                执行 hook 对应的处理流程。
            
            输入参数：
                _module (Any)：方法所需的 _module 参数。
                _inputs (Any)：方法所需的 _inputs 参数。
                output (Any)：文本词元嵌入，形状为 [C, L, D_text]。
            
            返回值：
                torch.Tensor：注入可学习上下文后的词元嵌入，形状保持 [C, L, D_text]。
            """
            if output.shape[0] != self.context.shape[0]:
                raise ValueError("Prompt batch does not match number of class anchors")
            replaced = output.clone()
            replaced[:, start:stop, :] = self.context.to(dtype=output.dtype)
            return replaced

        handle = token_embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, clip_model: nn.Module) -> torch.Tensor:
        """
        方法作用：
            执行模型的前向传播。
        
        输入参数：
            self (Any)：当前实例。
            clip_model (nn.Module)：CLIP 模型实例。
        
        返回值：
            torch.Tensor：每个类别的归一化文本锚点，形状为 [C, D]；
            C 为类别数，D 为 CLIP 图文共享特征维度。
        """
        token_embedding = _text_token_embedding(clip_model)
        with self._inject_context(token_embedding):
            output = clip_model.get_text_features(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
            )
        features = extract_feature_tensor(output, "text").float()
        return F.normalize(features, dim=-1)


def trainable_parameter_count(module: nn.Module) -> int:
    """
    方法作用：
        统计模块中需要梯度更新的参数数量。
    
    输入参数：
        module (nn.Module)：方法所需的 module 参数。
    
    返回值：
        int：方法执行得到的结果。
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
