from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from PAD_Lite.frozen_backbone_comparison import (
    FrozenConvNextEncoder,
    _parse_folds,
    load_comparison_config,
)


class _FakeOutput:
    """
    方法作用：
        为冻结 ConvNeXt 编码器单元测试提供 pooler_output。

    输入参数：
        value (torch.Tensor)：形状 [B,D] 的模拟池化特征。

    返回值：
        _FakeOutput：包含模拟特征的对象。
    """

    def __init__(self, value: torch.Tensor) -> None:
        """
        方法作用：
            保存模拟全局池化特征。

        输入参数：
            value (torch.Tensor)：形状 [B,D] 的模拟特征。

        返回值：
            None：仅初始化对象。
        """

        self.pooler_output = value


class _FakeConvNext(nn.Module):
    """
    方法作用：
        模拟带可训练参数和全局池化输出的 ConvNeXt 主干。

    输入参数：
        无。

    返回值：
        _FakeConvNext：单元测试模型。
    """

    def __init__(self) -> None:
        """
        方法作用：
            创建模拟线性参数。

        输入参数：
            无。

        返回值：
            None：仅初始化模块。
        """

        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pixel_values: torch.Tensor) -> _FakeOutput:
        """
        方法作用：
            对 [B,3,H,W] 输入生成 [B,3] 模拟全局特征。

        输入参数：
            pixel_values (torch.Tensor)：图像批次，形状 [B,3,H,W]。

        返回值：
            _FakeOutput：pooler_output 形状 [B,3]。
        """

        return _FakeOutput(pixel_values.mean(dim=(2, 3)) * self.scale)


def test_frozen_convnext_has_no_trainable_parameters() -> None:
    """
    方法作用：
        验证包装后的 ConvNeXt 参数全部冻结且输出完成归一化。

    输入参数：
        无。

    返回值：
        None：通过断言表达测试结果。
    """

    encoder = FrozenConvNextEncoder(_FakeConvNext())
    output = encoder.encode(torch.ones(2, 3, 4, 4))
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    assert output.shape == (2, 3)
    assert torch.allclose(output.norm(dim=1), torch.ones(2))


def test_parse_folds() -> None:
    """
    方法作用：
        验证 all 与逗号分隔折号解析及越界拒绝逻辑。

    输入参数：
        无。

    返回值：
        None：通过断言表达测试结果。
    """

    assert _parse_folds("all") == [1, 2, 3, 4, 5]
    assert _parse_folds("3,1,3") == [1, 3]
    with pytest.raises(ValueError):
        _parse_folds("0")


def test_config_rejects_training_section(tmp_path: Path) -> None:
    """
    方法作用：
        验证冻结实验配置出现 training 字段时立即拒绝执行。

    输入参数：
        tmp_path (Path)：pytest 临时目录。

    返回值：
        None：通过异常断言表达测试结果。
    """

    payload = {
        "schema_version": 1,
        "paths": {
            key: "."
            for key in (
                "crop_root",
                "splits_root",
                "dino_model",
                "convnext_model",
                "p2b_results_root",
                "output_root",
            )
        },
        "data": {"fold_count": 5},
        "retrieval": {"prototype_top_k": 3},
        "training": {},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbids training"):
        load_comparison_config(path)
