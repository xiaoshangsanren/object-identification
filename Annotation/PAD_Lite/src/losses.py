from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    方法作用：
        使用 batch-hard 方式为归一化后的样本嵌入计算三元组损失，
        约束同类样本更靠近、不同类样本更远离。

    输入参数：
        embeddings (torch.Tensor):
            当前批次经过 L2 归一化的样本嵌入，形状为 [B, D]。
        labels (torch.Tensor):
            与嵌入逐一对应的类别标签，形状为 [B]。
        margin (float):
            三元组损失的间隔阈值。

    返回值：
        torch.Tensor:
            标量三元组损失，形状为 [B,D]。其中 B 为批量大小，D 为嵌入维度。
    """
    # 检查输入的嵌入向量是否为二维张量，如果不是则抛出异常。
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [batch, dimension]")
    # 计算当前批次里所有样本两两之间的欧氏距离。
    # 此处得到的是 [batch_size, batch_size] 的距离矩阵，其中每个元素表示两个样本之间的距离。
    distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    # 生成一个“是否属于同一类别”的布尔矩阵，形状为 [batch_size, batch_size]。
    same_class = labels[:, None].eq(labels[None, :])
    # 生成一个单位矩阵，大小是当前 batch 的样本数。
    # 用于在后面排除自己与自己的对比
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    # 构造“正样本掩码”。
    positive_mask = same_class & ~eye
    # 构造“负样本掩码”。
    negative_mask = ~same_class
    # 检查每个样本是否至少有一个正样本和一个负样本，如果没有则抛出异常。
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every sample needs a positive partner; use the PK sampler")
    # 检查每个样本是否都至少有一个负样本。
    if not negative_mask.any(dim=1).all():
        raise ValueError("Every sample needs a negative class; use at least P=2")
    # 对于每个样本，找到最难的正样本（距离最远的正样本）。
    hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    # 对于每个样本，找到最难的负样本（距离最近的负样本）。
    hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    # 计算三元组损失，使用 ReLU 负值归0，并返回平均损失。
    # 核心思想是：
        # 让正样本距离尽量小；
        # 让负样本距离尽量大；
        # 还要至少超过 margin 这个间隔。
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def text_anchor_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_anchors: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    方法作用：
        通过图像嵌入与文本锚点嵌入之间的对比学习，
        强化同类样本在文本锚点空间中的对齐效果。

    输入参数：
        image_embeddings (torch.Tensor):
            图像编码得到的归一化嵌入，形状为 [B, D]。
        text_anchors (torch.Tensor):
            每个类别对应的归一化文本锚点，形状为 [C, D]。
        labels (torch.Tensor):
            图像所属类别标签，形状为 [B]，取值范围为 [0, C-1]。
        temperature (float):
            对比损失中的温度系数。

    返回值：
        torch.Tensor:
            标量图像-文本双向对比损失，形状为 []。其中 B 为批量大小，
            C 为类别数，D 为图文共享特征维度。
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # 计算图像嵌入与文本锚点嵌入之间的相似度得分，并进行温度缩放。
    logits_i2t = image_embeddings @ text_anchors.t() / temperature
    # 计算图像到文本的交叉熵损失，作为图像-文本对比损失的一部分。
    image_to_text = F.cross_entropy(logits_i2t, labels)

    # 计算文本到图像的对比损失。
    logits_t2i = logits_i2t.t()
    # 生成一个类别索引张量，形状为 [C]，用于标识每个类别。
    class_ids = torch.arange(text_anchors.shape[0], device=labels.device)
    # 生成一个布尔掩码，标识哪些文本锚点与图像标签匹配。
    positive_mask = class_ids[:, None].eq(labels[None, :])
    # 检查每个文本锚点是否至少有一个对应的图像样本，如果没有则抛出异常。
    present = positive_mask.any(dim=1)
    # 计算文本到图像的对比损失。
    positive_logits = logits_t2i.masked_fill(~positive_mask, float("-inf"))
    # 计算每个文本锚点的正样本对数和，以及所有图像样本的对数和。
    log_positive = torch.logsumexp(positive_logits[present], dim=1)
    # 计算所有图像样本的对数和。
    log_all = torch.logsumexp(logits_t2i[present], dim=1)
    # 计算文本到图像的对比损失，取平均值。
    text_to_image = -(log_positive - log_all).mean()
    # 返回图像到文本和文本到图像的对比损失的平均值，作为最终的双向对比损失。
    return 0.5 * (image_to_text + text_to_image)
