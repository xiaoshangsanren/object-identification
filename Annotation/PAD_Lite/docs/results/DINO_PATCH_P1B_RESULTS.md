# DINOv2 Patch-Safe P1b 五折实验结果

## 1. 实验定义

P1b是P1a之后的Global/Local简单融合消融：

```text
P0：冻结DINOv2-Small + CLS-B2 Head
                    ├── 512维 z_cls
同一批图像          │
                    └── P1a Masked Patch Average Head
                                      └── 512维 z_patch
                                              ↓
       [sqrt(0.5) * z_cls, sqrt(0.5) * z_patch]
                                              ↓
                         1024维联合L2归一化特征
                                              ↓
                       prototype_top_k=3 检索
```

固定条件：

- P0来源：DINOv2-Small、Letterbox 336、训练期文本锚点、冻结主干；
- Patch来源：已经完成的P1a Masked Patch Average；
- CLS权重：0.5；
- Patch权重：0.5；
- 不使用Calibration；
- 不使用门控；
- 不使用Novel Query标签选择权重；
- P1b执行阶段不加载文本编码器；
- P1b没有新增训练参数，也不重复进行DINO前向。

等权拼接后的余弦相似度严格等于两个归一化分支余弦相似度的等权和，因此P1b只回答：

> 在不做动态Patch筛选、门控和标定时，CLS与Masked Patch Average的简单等权组合是否互补？

## 2. Rank-1结果

| 方法 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 |
|---|---:|---:|---:|---:|---:|---:|
| P0：CLS-B2 | 97.85% | **71.67%** | 95.90% | 72.56% | **94.14%** | **86.42%** |
| P1a：Patch Average only | 94.42% | 63.09% | 95.38% | 63.72% | 89.06% | 81.14% |
| P1b：CLS + Patch Average | **97.85%** | 68.24% | **96.41%** | **72.56%** | 93.36% | 85.68% |
| P1b − P0 | 0.00 pp | -3.43 pp | +0.51 pp | 0.00 pp | -0.78 pp | **-0.74 pp** |
| P1b − P1a | +3.43 pp | +5.15 pp | +1.03 pp | +8.84 pp | +4.30 pp | **+4.55 pp** |

P1b五折标准差为12.64%。

容易折Fold 1/3/5：

```text
P0  = 95.96%
P1b = 95.87%
变化 = -0.09 pp
```

困难折Fold 2/4：

```text
P0  = 72.12%
P1b = 70.40%
变化 = -1.72 pp
```

P1b几乎恢复了容易折的CLS性能，但没有改善困难折。

## 3. 各类别Rank-1

| Fold | 类别1 Rank-1 | 类别2 Rank-1 |
|---:|---:|---:|
| 1 | MT-LB：97.35% | T-64：98.33% |
| 2 | T-72：46.90% | T-80：88.33% |
| 3 | BMD-2：93.14% | BTR-70：100.00% |
| 4 | BMP-1：79.09% | BMP-2：65.71% |
| 5 | BM-21：100.00% | BTR-80：85.83% |

Fold 2的主要退化仍来自T-72。Patch分支虽然改善了部分T-80样本，但对T-72造成的伤害更大。

## 4. 相对CLS的纠错与伤害

| Fold | CLS错误→P1b正确（rescued） | CLS正确→P1b错误（harmed） | 净纠错 |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 0 |
| 2 | 18 | 26 | -8 |
| 3 | 2 | 1 | +1 |
| 4 | 18 | 18 | 0 |
| 5 | 1 | 3 | -2 |
| **总计** | **40** | **49** | **-9** |

解释：

- Fold 2/4确实存在较多Patch可纠正的CLS错误，证明两条分支含有互补信息；
- 但无门控等权融合无法识别哪些Query应该相信Patch；
- Fold 2中救回18个样本的同时破坏26个原本正确的CLS判断；
- Fold 4的18次救回被18次伤害完全抵消；
- 五折总计`rescued < harmed`，不满足安全融合标准。

## 5. 结论

1. P1b没有超过P0：五折平均低0.74个百分点。
2. P1b比P1a高4.55个百分点，说明CLS全局路径能够显著保护均匀Patch造成的性能损失。
3. 简单等权融合不能改善Fold 2/4，困难折平均反而下降1.72个百分点。
4. P1b进一步证明Patch Average包含互补信号，但其错误率过高，不能对所有Query无条件参与决策。
5. 不应该根据当前Novel Query结果继续扫描融合权重，否则会造成评测泄漏。
6. 下一步应按既定实验矩阵进入P2a：使用动态Masked Weighted Patch Pooling，只评测Weighted Patch本身；P2a证明有效后再进入P2b和门控融合。

## 6. 完整性检查

- 五折均完成；
- 使用与P0/P1a完全一致的Support/Query顺序和标签；
- 每折校验P0/P1a类别顺序、标签和Query元数据；
- 每折保存P0/P1a源特征文件路径与SHA-256；
- 固定`cls_weight=0.5`、`patch_weight=0.5`；
- `novel_query_tuning=false`；
- `calibration_used=false`；
- `training_performed=false`；
- 18项单元测试全部通过；
- Managed Run事件状态为`completed`；
- 当次源码快照共51个文件。

## 7. 文件位置

- P1b实现：`PAD_Lite/src/dino_patch_fusion.py`
- 统一入口：`PAD_Lite/src/experiment_cli.py`
- 兼容入口：`PAD_Lite/src/dino_patch_cli.py`
- 配置：`configs/dino_patch_p1b_equal_fusion_336.json`
- 预设：`experiments/dino_s_p1b_cls_patch_average_equal_fusion_336.json`
- 测试：`PAD_Lite/tests/test_dino_patch_p1b.py`
- Managed Run：`outputs/experiment_runs/dino_s_p1b_cls_patch_average_equal_fusion_336/fivefold_equal_v1/`
