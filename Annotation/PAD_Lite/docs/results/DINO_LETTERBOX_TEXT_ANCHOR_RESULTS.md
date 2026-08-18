# DINOv2 Letterbox + 文本锚点五折实验

## 实验目的

验证当前 `Resize(256) + CenterCrop(224)` 是否因裁掉车辆两端而造成细粒度识别性能下降。

## 实验配置

- 视觉主干：本地 `DINOv2-small`
- 输入尺寸：`224 x 224`
- 输入方式：保持宽高比缩放完整车辆，再使用 ImageNet 均值颜色居中填充为正方形
- 主干解冻层数：`0`
- 训练参数：检索头与可学习文本提示
- 损失：`ID Loss (1.0) + Batch-Hard Triplet Loss (1.0) + Text-anchor Contrastive Loss (0.5)`
- 评测：相同五折、Support/Query 和 `prototype_top_k=3`
- 随机种子：`2026 + fold`

训练阶段不再使用会删除车辆局部的 `RandomResizedCrop`、随机透视和随机擦除；保留水平翻转、颜色扰动和轻度高斯模糊。推理阶段同样使用 Letterbox，保证训练与测试的空间预处理一致。

配置文件：`PAD_Lite/configs/dino_letterbox_text_anchor.json`

## Rank-1 结果

| Fold | 未见类别 | CenterCrop B2 | Letterbox B2 | 变化 |
|---:|---|---:|---:|---:|
| 1 | MT-LB / T-64 | 97.00% | 97.00% | +0.00 pp |
| 2 | T-72 / T-80 | 69.53% | 67.38% | -2.15 pp |
| 3 | BMD-2 / BTR-70 | 97.95% | 96.92% | -1.03 pp |
| 4 | BMP-1 / BMP-2 | 74.42% | 72.09% | -2.33 pp |
| 5 | BM-21 / BTR-80 | 92.19% | 91.80% | -0.39 pp |
| **五折平均** | — | **86.22%** | **85.04%** | **-1.18 pp** |

Letterbox逐类Rank-1：

| Fold | 类别1 | Rank-1 | 类别2 | Rank-1 |
|---:|---|---:|---|---:|
| 1 | MT-LB | 96.46% | T-64 | 97.50% |
| 2 | T-72 | 52.21% | T-80 | 81.67% |
| 3 | BMD-2 | 94.12% | BTR-70 | 100.00% |
| 4 | BMP-1 | 77.27% | BMP-2 | 66.67% |
| 5 | BM-21 | 100.00% | BTR-80 | 82.50% |

## 结论

当前单次固定种子实验没有显示 `224 x 224 Letterbox` 优于原 CenterCrop。Fold 2 和 Fold 4 仍明显较差，因此“车辆两端被中心裁掉”不是这两个折性能低的充分解释。

Letterbox在保留完整车身的同时，会让细长车辆在 `224 x 224` 画布中占据更少的纵向像素。它可能解决了内容截断，却进一步降低了炮塔、负重轮等局部部件的有效分辨率。下一项建议是在保持Letterbox的条件下测试 `336 x 336` 和 `448 x 448`，将“完整内容”和“局部细节分辨率”同时保留。

由于本实验只有一个随机种子，1~2个百分点的差异不应解释为Letterbox必然更差；能够确认的是，目前没有获得支持Letterbox改进效果的证据。

## 输出位置

- 汇总：`PAD_Lite/outputs/dino_letterbox_text_anchor/dino/b2/summary.json`
- 每折指标：`PAD_Lite/outputs/dino_letterbox_text_anchor/dino/b2/fold_*/metrics.json`
- 训练日志：`PAD_Lite/logs/dino_letterbox_text_anchor/`
