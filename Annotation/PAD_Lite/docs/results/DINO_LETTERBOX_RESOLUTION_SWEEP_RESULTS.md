# DINOv2 Letterbox输入分辨率五折消融

## 实验设置

- 输入分辨率：`224`、`336`、`448`、`518`
- 空间预处理：等比例缩放完整车辆并居中填充（Letterbox）
- 视觉主干：`DINOv2-small`，全部冻结
- 训练参数：检索头和可学习文本提示
- 损失：`ID Loss (1.0) + Batch-Hard Triplet Loss (1.0) + Text-anchor Contrastive Loss (0.5)`
- 训练轮数：每折30轮
- 评测：相同五折、Support/Query、`prototype_top_k=3`
- 随机种子：`2026 + fold`

除输入尺寸和不影响特征结果的评估Batch Size外，其余实验条件保持一致。224直接复用已完成的Letterbox实验；336、448和518使用GPU 4、6并行训练。

## Rank-1结果

| 输入尺寸 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 224 | 97.00% | 67.38% | 96.92% | 72.09% | 91.80% | 85.04% | 12.72% |
| 336 | **97.85%** | **71.67%** | 95.90% | 72.56% | 94.14% | 86.42% | **11.75%** |
| 448 | 97.42% | 67.38% | **98.97%** | 75.35% | **94.53%** | **86.73%** | 12.88% |
| 518 | **97.85%** | 68.67% | 96.41% | **75.81%** | 94.14% | 86.58% | 11.98% |

相对224的五折均值变化：

| 输入尺寸 | Rank-1均值变化 | 平均训练时间/折 | 相对224训练时间 |
|---:|---:|---:|---:|
| 224 | — | 119.1秒 | 1.00倍 |
| 336 | +1.39 pp | 143.6秒 | 1.21倍 |
| 448 | +1.69 pp | 231.5秒 | 1.94倍 |
| 518 | +1.54 pp | 290.8秒 | 2.44倍 |

这里的时间只统计30轮训练循环，不包含模型加载、Support/Query特征提取和结果写入时间。

## 结论

1. 提高分辨率确实能够带来有限收益，但不是越高越好。五折平均在448达到最高值86.73%，518回落到86.58%。两者只差0.15个百分点，在单随机种子下不能视为显著差异。
2. Fold 2在336达到最高71.67%，继续提高到448/518反而下降；Fold 4则在448/518达到75%左右。因此不同细粒度类别对具有不同的有效视觉尺度。
3. 即使使用每折最佳尺寸，Fold 2和Fold 4仍明显低于其他折。输入分辨率是次要影响因素，不是困难折的根本原因。
4. 若只追求本次五折平均Rank-1，可选择448；若考虑速度、训练成本和Fold 2表现，336是更合理的工程折中。
5. 下一步不建议继续提高到560或672。应优先对336和448补充多随机种子验证，随后考察CLS全局特征与局部Patch Token融合。

## 输出位置

- 224：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_text_anchor/dino/b2/`
- 336：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_text_anchor_336/dino/b2/`
- 448：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_text_anchor_448/dino/b2/`
- 518：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_text_anchor_518/dino/b2/`
- 日志：`PAD_Lite/logs/dino_letterbox_resolution_sweep/`
