# DINOv2 Letterbox语义锚点消融

## 实验目的

在相同Letterbox预处理、输入分辨率、五折划分和冻结DINOv2主干条件下，对比：

- 无语义锚点（B1）：`ID Loss + Batch-Hard Triplet Loss`
- 有语义锚点（B2）：`ID Loss + Batch-Hard Triplet Loss + Text-anchor Contrastive Loss`

无文本实验中没有加载CLIP文本编码器，没有创建文本提示，`anchor_weight=0`、`prompt_lr=0`，30轮历史中的`anchor_loss`均为0。

## 无文本锚点Rank-1

| 输入尺寸 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 |
|---:|---:|---:|---:|---:|---:|---:|
| 224 | 97.42% | 67.81% | 94.36% | 68.84% | 92.58% | 84.20% |
| 336 | **97.85%** | **73.39%** | 94.36% | 73.49% | 94.14% | **86.65%** |
| 448 | 97.00% | 70.82% | 95.90% | 72.09% | **94.92%** | 86.14% |
| 518 | 96.57% | 68.67% | **96.92%** | **73.95%** | 94.14% | 86.05% |

## 文本锚点的净变化

下表为同一分辨率下`B2文本锚点 - B1无文本`的五折平均Rank-1变化：

| 输入尺寸 | 无文本B1 | 文本锚点B2 | 文本锚点变化 |
|---:|---:|---:|---:|
| 224 | 84.20% | 85.04% | +0.84 pp |
| 336 | **86.65%** | 86.42% | -0.22 pp |
| 448 | 86.14% | **86.73%** | +0.59 pp |
| 518 | 86.05% | 86.58% | +0.53 pp |

跨四种分辨率平均后，文本锚点对各Fold的影响为：

| Fold | 未见类别 | 文本锚点平均变化 |
|---:|---|---:|
| 1 | MT-LB / T-64 | +0.32 pp |
| 2 | T-72 / T-80 | **-1.39 pp** |
| 3 | BMD-2 / BTR-70 | **+1.67 pp** |
| 4 | BMP-1 / BMP-2 | **+1.86 pp** |
| 5 | BM-21 / BTR-80 | -0.29 pp |

全部20组同尺寸、同Fold配对结果的平均文本增益为`+0.43 pp`。

## 结论

1. 文本锚点整体只有较小平均收益，且高度依赖类别。它稳定帮助Fold 3/4，却稳定伤害Fold 2，对Fold 1/5影响较小。
2. 分别选择最佳分辨率时，无文本最佳为336的86.65%，有文本最佳为448的86.73%，二者只差0.09个百分点。单随机种子下不能据此证明语义锚点带来总体提升。
3. Fold 2在无文本336下达到73.39%，高于所有带文本配置；说明当前文本锚点可能压缩或扭曲T-72/T-80需要的细粒度视觉边界，存在负迁移。
4. Fold 3/4在带文本的高分辨率配置中提升明显，说明语义约束并非无效，而是当前类别无关提示或统一权重缺乏适应性。
5. 下一步不应简单增大统一`anchor_weight`。建议固定336和448，进行多随机种子复验，并尝试较小权重、延迟启用文本损失或按视觉冲突程度自适应加权。

## 输出位置

- 无文本224：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_no_text_224/dino/b1/`
- 无文本336：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_no_text_336/dino/b1/`
- 无文本448：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_no_text_448/dino/b1/`
- 无文本518：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_no_text_518/dino/b1/`
- 无文本日志：`PAD_Lite/logs/dino_letterbox_no_text_resolution_sweep/`
- 带文本原实验：`PAD_Lite/docs/results/DINO_LETTERBOX_RESOLUTION_SWEEP_RESULTS.md`
