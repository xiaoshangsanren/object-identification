# DINOv2 Patch-Safe P2a 五折实验结果

## 1. 实验定义

P2a把P1a的均匀Patch平均替换为逐图、内容相关的动态加权聚合：

```text
冻结DINOv2-Small
  ↓
提取24×24个Patch Token
  ↓
使用Letterbox有效区域Mask排除Padding Token
  ↓
共享Linear(384, 1)对每个有效Patch动态打分
  ↓
Masked Softmax Weighted Pooling
  ↓
Patch Projection + BNNeck + ID Head
  ↓
只用Weighted Patch特征完成Novel Support/Query检索
```

P2a固定条件：

- DINOv2-Small主干完全冻结；
- Letterbox 336，Patch Size为14；
- 只训练Patch Scorer、Projection、BNNeck和Classifier；
- 训练损失为`ID + Batch-Hard Triplet`；
- 不使用CLS Token、CLS/Patch融合或文本锚点；
- 每批4类×4实例，每折30轮；
- `prototype_top_k=3`；
- 与P0/P1a/P1b使用相同五折、Support/Query和种子；
- Scorer使用全零初始化，因此训练起点严格等价于P1a的Masked Average。

P2a共有202,113个可训练参数，其中Patch Scorer只有385个参数；DINOv2可训练参数为0。

## 2. Rank-1结果

| 方法 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0：CLS-B2 | **97.85%** | **71.67%** | 95.90% | **72.56%** | **94.14%** | **86.42%** | 11.75% |
| P1a：Patch Average only | 94.42% | **63.09%** | 95.38% | 63.72% | 89.06% | 81.14% | 14.64% |
| P1b：CLS + Patch Average | **97.85%** | 68.24% | **96.41%** | **72.56%** | 93.36% | 85.68% | 12.64% |
| P2a：Weighted Patch only | **97.85%** | 62.66% | 94.87% | **72.56%** | 93.36% | 84.26% | 14.03% |
| P2a − P0 | 0.00 pp | -9.01 pp | -1.03 pp | 0.00 pp | -0.78 pp | **-2.16 pp** | — |
| P2a − P1a | +3.43 pp | -0.43 pp | -0.51 pp | +8.84 pp | +4.30 pp | **+3.13 pp** | — |

容易折Fold 1/3/5：

```text
P0  = 95.96%
P2a = 95.36%
变化 = -0.60 pp
```

困难折Fold 2/4：

```text
P0  = 72.12%
P2a = 67.61%
变化 = -4.51 pp
```

动态加权相对P1a提升3.13个百分点，说明学习Patch权重比均匀平均有效；但它没有达到相对P0的安全提升目标，尤其没有改善困难折。

## 3. P2a各类别Rank-1

| Fold | 类别1 Rank-1 | 类别2 Rank-1 |
|---:|---:|---:|
| 1 | MT-LB：96.46% | T-64：99.17% |
| 2 | T-72：53.10% | T-80：71.67% |
| 3 | BMD-2：92.16% | BTR-70：97.85% |
| 4 | BMP-1：86.36% | BMP-2：58.10% |
| 5 | BM-21：100.00% | BTR-80：85.83% |

Fold 4虽然总Rank-1与P0相同，但并不是逐样本预测不变：P2a明显偏向BMP-1，BMP-1得到提升的同时BMP-2受到等量伤害。这是候选车型之间发生方向性偏置，而不是安全保护容易样本。

## 4. 相对P0的纠错与伤害

| Fold | CLS正确/Weighted Patch正确 | CLS正确/Weighted Patch错误 | **CLS错误/Weighted Patch正确** | 两者都错 | 净纠错 |
|---:|---:|---:|---:|---:|---:|
| 1 | 226 | 2 | 2 | 3 | 0 |
| 2 | 125 | **42** | **21** | 45 | -21 |
| 3 | 182 | 5 | 3 | 5 | -2 |
| 4 | 128 | **28** | **28** | 31 | 0 |
| 5 | 237 | 4 | 2 | 13 | -2 |
| **总计** | **898** | **81** | **56** | **97** | **-25** |

与P1a相比：

```text
P1a：rescued=59，harmed=120，net=-61
P2a：rescued=56，harmed=81， net=-25
```

动态权重只少损失3次纠错机会，却减少了39次对P0正确结果的破坏。因此P2a相对P1a的提升是真实的，但仍有81个P0正确样本被破坏，尚不足以单独替代CLS。

Fold 2最明显：P2a救回21张P0错图，同时破坏42张P0正确图。Fold 4内部则表现为：

```text
BMP-1：净纠错 +8
BMP-2：净纠错 -8
总计：       0
```

这证明同一个无条件共享Scorer无法稳定决定相似车型之间“应该看哪个部件”。

## 5. 注意力与Mask诊断

| Fold | Query平均有效Patch数 | 加权后的有效Patch数 | 归一化熵 | Padding注意力最大值 |
|---:|---:|---:|---:|---:|
| 1 | 445.18 | 170.32 | 0.837 | 0 |
| 2 | 449.41 | 155.85 | 0.816 | 0 |
| 3 | 438.15 | 225.01 | 0.889 | 0 |
| 4 | 456.45 | 226.31 | 0.883 | 0 |
| 5 | 445.13 | 223.57 | 0.886 | 0 |

可确认：

1. 五折Support和Query的Padding注意力质量均严格为0，退化不是Mask实现错误；
2. Scorer确实从均匀初始化学出了选择性，等效Patch数由约438至456个降到约156至226个；
3. Fold 2的注意力最集中，但Rank-1最低，说明“更尖锐”不等于“更正确”；
4. 热力图大多覆盖车辆本体，但常集中于炮管末端、单一炮塔区域、局部涂装/标识或人员附近；
5. 当前Scorer学习的是训练类别上的通用显著性，不是针对当前CLS Top-K候选关系的差异部件。

人工抽查覆盖了Fold 2/4的`both_correct`、`both_wrong`、`cls_wrong_patch_correct`和`cls_correct_patch_wrong`四组样本。热力图位于每折的`attention_samples/`目录，索引记录在`index.json`。

## 6. 结论与后续决策

1. **P2a相对P1a有效**：动态Weighted Pooling使五折平均提高3.13个百分点，显著减少均匀聚合造成的伤害。
2. **P2a不能替代P0**：平均仍低2.16个百分点，困难折平均低4.51个百分点。
3. **P2a没有达到Patch-Safe目标**：五折累计`rescued=56 < harmed=81`。
4. **Fold 2/4确有互补局部信息**：两折共有49个`CLS错误/Patch正确`样本，但P2a不能识别何时应当相信该信息。
5. **不应在Novel Query上扫描权重或阈值**，否则会产生评测泄漏。
6. P2a代码和预设应保留为独立可复现实验，但不设为Annotation默认识别路径。
7. 如继续P2b，它只能作为固定融合的诊断实验；更有希望的最终方向仍是用独立开发集训练/标定的`CLS Top-K → 条件式Patch重排`，让Patch只在歧义Query上参与，而不是无条件替换CLS。

## 7. 完整性检查

- 五折均完成30轮，Managed Run状态为`completed`；
- 五折均使用独立最佳验证检查点评测Novel Query；
- DINOv2可训练参数为0；
- 每折检查点均包含`patch_scorer.weight`与`patch_scorer.bias`；
- `text_anchor_used=false`，`cls_feature_used=false`；
- 所有训练和验证轮次的Padding注意力质量均为0；
- 日志和结构化结果中未发现Traceback、CUDA OOM或NaN；
- 每折均保存Query权重、互补性统计和11至16张注意力图；
- PAD_Lite全量23项单元测试通过，源码`compileall`通过；
- 当次Managed Run源码快照包含55个文件。

## 8. 文件位置

- 模型：`PAD_Lite/src/dino_patch_weighted_models.py`
- 训练、评测与热力图：`PAD_Lite/src/dino_patch_weighted_engine.py`
- 统一入口：`PAD_Lite/src/experiment_cli.py`
- 兼容入口：`PAD_Lite/src/dino_patch_cli.py`
- 配置：`configs/dino_patch_p2a_weighted_letterbox_336.json`
- 预设：`experiments/dino_s_p2a_weighted_patch_letterbox_336.json`
- 测试：`PAD_Lite/tests/test_dino_patch_p2a.py`
- Managed Run：`outputs/03_p2b_patch_reranking/experiment_runs/dino_s_p2a_weighted_patch_letterbox_336/fivefold_seed2026_v1/`
