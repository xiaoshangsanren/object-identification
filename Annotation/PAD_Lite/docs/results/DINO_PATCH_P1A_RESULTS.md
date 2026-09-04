# DINOv2 Patch-Safe P1a 五折实验结果

## 1. 实验定义

P1a是Patch-Safe路线的第一个表示消融：

```text
冻结DINOv2-Small
  ↓
提取576个Patch Token
  ↓
使用Letterbox有效区域Mask排除Padding Token
  ↓
对剩余Patch做均匀平均
  ↓
训练Patch Projection + BNNeck + ID Head
  ↓
只用Patch特征完成Novel Support/Query检索
```

P1a明确不使用：

- CLS Token；
- CLS/Patch融合；
- Weighted Pooling；
- Text Anchor；
- DINOv2主干微调。

训练损失为：

\[
L=L_{ID}+L_{Triplet}
\]

其他条件：

- 输入：Letterbox 336；
- Patch Size：14；
- Patch网格：24×24，共576个Token；
- 每批4类×4实例；
- 每折30轮；
- `prototype_top_k=3`；
- 随机种子：`2026 + fold`。

P1a只训练201,728个参数。五折检查点均只含：

- Patch Projection；
- Patch BNNeck；
- Patch ID Classifier。

## 2. Rank-1结果

| 方法 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0：CLS-B2 | **97.85%** | **71.67%** | **95.90%** | **72.56%** | **94.14%** | **86.42%** | 11.75% |
| P1a：Masked Patch Average only | 94.42% | 63.09% | 95.38% | 63.72% | 89.06% | 81.14% | 14.64% |
| P1a − P0 | -3.43 pp | -8.58 pp | -0.51 pp | -8.84 pp | -5.08 pp | **-5.29 pp** | — |

容易折Fold 1/3/5平均：

```text
P0  = 95.96%
P1a = 92.96%
变化 = -3.01 pp
```

困难折Fold 2/4平均：

```text
P0  = 72.12%
P1a = 63.41%
变化 = -8.71 pp
```

## 3. P1a各类别结果

| Fold | 类别1 Rank-1 | 类别2 Rank-1 |
|---:|---:|---:|
| 1 | MT-LB：90.27% | T-64：98.33% |
| 2 | T-72：41.59% | T-80：83.33% |
| 3 | BMD-2：92.16% | BTR-70：98.92% |
| 4 | BMP-1：67.27% | BMP-2：60.00% |
| 5 | BM-21：100.00% | BTR-80：76.67% |

Fold 2的主要退化来自T-72，Fold 5的主要退化来自BTR-80；Fold 4的两个相似BMP类别都没有得到改善。

## 4. CLS与Patch错误互补性

| Fold | CLS正确/Patch正确 | CLS正确/Patch错误 | **CLS错误/Patch正确** | CLS错误/Patch错误 | 净纠错 |
|---:|---:|---:|---:|---:|---:|
| 1 | 219 | 9 | 1 | 4 | -8 |
| 2 | 123 | 44 | **24** | 42 | -20 |
| 3 | 184 | 3 | 2 | 6 | -1 |
| 4 | 108 | 48 | **29** | 30 | -19 |
| 5 | 225 | 16 | 3 | 12 | -13 |

`CLS错误/Patch正确`表示Patch具有纠错潜力；`CLS正确/Patch错误`表示统一使用Patch会造成的伤害。

关键现象：

- Fold 2中Patch纠正24个CLS错误，但损坏44个CLS正确样本；
- Fold 4中Patch纠正29个CLS错误，但损坏48个CLS正确样本；
- 两个困难折合计存在53个`CLS错误/Patch正确`样本，说明局部Patch并非完全没有互补信息；
- 但均匀平均无法选择真正的判别区域，最终伤害大于纠错。

CLS与Patch真实类别Margin的相关系数为：

| Fold | Pearson相关系数 |
|---:|---:|
| 1 | 0.767 |
| 2 | 0.588 |
| 3 | 0.879 |
| 4 | 0.427 |
| 5 | 0.821 |

Fold 2/4相关性相对较低，也支持局部表示与CLS存在一定互补性，而不是完全复制同一种表示。

## 5. 结论

1. P1a失败了：Masked Patch Average不能替代CLS，五折平均下降5.29个百分点。
2. 简单排除Padding仍不足以解决背景和非判别部件被均匀聚合的问题。
3. P1a对Fold 2/4的下降最大，证明“把所有有效Patch平均起来”不会自动提高细粒度车型区分能力。
4. 但是Fold 2/4存在53个`CLS错误/Patch正确`样本，说明Patch包含可利用的互补信号，只是P1a没有能力筛选这些信号。
5. 下一步应进入P2a：Masked Weighted Patch only。只有当动态权重能明显减少`CLS正确/Patch错误`并保持或增加`CLS错误/Patch正确`时，才继续P2b和Top-K融合。
6. P1a结果不能用来直接否定Patch Token；它否定的是均匀平均策略，这与参考论文的消融结论一致。

## 6. 完整性检查

- 五折均完成30轮；
- 五折`summary.json`显示`completed_folds=5`；
- DINOv2可训练参数为0；
- 每折Patch Head可训练参数为201,728；
- 检查点不包含DINO权重、CLS Head或Prompt；
- `text_anchor_used=false`；
- `cls_feature_used=false`；
- 日志未发现Traceback、CUDA OOM或NaN；
- 9项单元测试全部通过。

## 7. 文件位置

- 模型：`PAD_Lite/src/dino_patch_models.py`
- 训练与评测：`PAD_Lite/src/dino_patch_engine.py`
- CLI：`PAD_Lite/src/dino_patch_cli.py`
- 配置：`configs/dino_patch_safe_letterbox_336.json`
- 测试：`PAD_Lite/tests/test_dino_patch_p1a.py`
- 五折输出：`outputs/03_p2b_patch_reranking/dino_patch_safe_letterbox_336/p1a_patch_average_only/`
- 日志：`logs/dino_patch_safe_letterbox_336/p1a_gpu4.log`
