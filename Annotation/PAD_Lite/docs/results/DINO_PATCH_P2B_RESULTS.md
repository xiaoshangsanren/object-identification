# DINOv2 Patch-Safe P2b 五折实验结果

## 1. 实验定义

P2b是P2a之后的固定Global/Local融合消融：

```text
P0：冻结DINOv2-Small + 文本增强训练得到的CLS-B2 Head
                                      └── 512维 z_cls
同一图像
P2a：Dynamic Masked Weighted Patch Head
                                      └── 512维 z_patch
                                               ↓
                 [sqrt(0.5)·z_cls, sqrt(0.5)·z_patch]
                                               ↓
                                1024维L2归一化特征
                                               ↓
                         全部候选类别prototype_top_k=3检索
```

固定条件：

- CLS来源：P0 DINOv2-Small、Letterbox 336、训练期文本锚点；
- Patch来源：P2a Dynamic Masked Weighted Patch；
- CLS/Patch权重固定为0.5/0.5；
- 对全部候选类别进行融合，不使用Top-K候选门控；
- 不使用Calibration、Prototype Gate或Query Margin Gate；
- 不使用Novel Query标签选择权重或阈值；
- P2b阶段不加载文本编码器、不训练参数、不重复执行DINO前向；
- P0与P2a缓存均通过类别、标签、Query顺序和逐图元数据校验。

P2b只回答：

> 动态Weighted Patch与CLS做最简单的无条件等权融合时，能否证明局部信息具有净互补收益，并测量这种融合的潜在伤害？

## 2. Rank-1结果

| 方法 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0：CLS-B2 | 97.85% | 71.67% | 95.90% | 72.56% | 94.14% | 86.42% | 11.75% |
| P1a：Patch Average only | 94.42% | 63.09% | 95.38% | 63.72% | 89.06% | 81.14% | 14.64% |
| P1b：CLS + Patch Average | 97.85% | 68.24% | **96.41%** | 72.56% | 93.36% | 85.68% | 12.64% |
| P2a：Weighted Patch only | 97.85% | 62.66% | 94.87% | 72.56% | 93.36% | 84.26% | 14.03% |
| **P2b：CLS + Weighted Patch** | **98.28%** | **72.53%** | 95.90% | **73.95%** | **94.14%** | **86.96%** | **11.29%** |
| P2b − P0 | +0.43 pp | +0.86 pp | 0.00 pp | +1.40 pp | 0.00 pp | **+0.54 pp** | — |
| P2b − P1b | +0.43 pp | +4.29 pp | -0.51 pp | +1.40 pp | +0.78 pp | **+1.28 pp** | — |
| P2b − P2a | +0.43 pp | +9.87 pp | +1.03 pp | +1.40 pp | +0.78 pp | **+2.70 pp** | — |

容易折Fold 1/3/5：

```text
P0  = 95.96%
P2b = 96.11%
变化 = +0.14 pp
```

困难折Fold 2/4：

```text
P0  = 72.12%
P2b = 73.24%
变化 = +1.13 pp
```

P2b是当前Patch-Safe实验中第一个五折平均超过P0、同时不降低任何一折总体Rank-1的方法。

## 3. 各类别Rank-1与方向性偏置

| Fold | 类别 | P0 | P2b | 变化 |
|---:|---|---:|---:|---:|
| 1 | MT-LB | 97.35% | 97.35% | 0.00 pp |
| 1 | T-64 | 98.33% | 99.17% | +0.83 pp |
| 2 | T-72 | 59.29% | 58.41% | -0.88 pp |
| 2 | T-80 | 83.33% | 85.83% | +2.50 pp |
| 3 | BMD-2 | 92.16% | 92.16% | 0.00 pp |
| 3 | BTR-70 | 100.00% | 100.00% | 0.00 pp |
| 4 | BMP-1 | 79.09% | 89.09% | **+10.00 pp** |
| 4 | BMP-2 | 65.71% | 58.10% | **-7.62 pp** |
| 5 | BM-21 | 100.00% | 100.00% | 0.00 pp |
| 5 | BTR-80 | 87.50% | 87.50% | 0.00 pp |

P2b虽然提高了Fold 2/4的总体Rank-1，但仍存在明显类别方向性偏置：

- Fold 2的收益来自T-80，T-72略微下降；
- Fold 4对BMP-1提升很大，却明显损害BMP-2；
- 因此P2b已经证明Weighted Patch具有互补价值，但还不是类别公平、安全的最终融合策略。

## 4. 相对P0的纠错与伤害

| Fold | P0错误→P2b正确 | P0正确→P2b错误 | 净纠错 |
|---:|---:|---:|---:|
| 1 | 1 | 0 | +1 |
| 2 | 13 | 11 | +2 |
| 3 | 2 | 2 | 0 |
| 4 | 16 | 13 | +3 |
| 5 | 1 | 1 | 0 |
| **总计** | **33** | **27** | **+6** |

与两个Patch分支的无条件使用相比：

```text
P1a Patch Average only：rescued=59，harmed=120，net=-61
P2a Weighted Patch only：rescued=56，harmed=81， net=-25
P2b CLS+Weighted Patch： rescued=33，harmed=27， net=+6
```

CLS路径显著抑制了Weighted Patch的伤害，使`rescued > harmed`首次成立。

P2b相对P1b累计救回62张、损害47张，净增加15张正确结果。说明P2b的提升不是单纯来自CLS保护，而是Weighted Patch确实比Patch Average提供了更有效的局部表示。

## 5. 结果解释

1. P2a单独使用时低于P0，但其错误与CLS并不完全重合；
2. P2b把两种表示放入同一个联合余弦空间后，CLS保护全局结构，Weighted Patch补充炮塔、车体和炮管等局部信息；
3. Fold 2/4均获得净提升，说明局部互补信号确实存在于最困难的相似车型中；
4. 50/50无条件融合仍会破坏27张P0正确图，特别是在BMP-2上出现系统性损害；
5. 当前两类别Novel Fold只能验证“全候选融合”，不能证明最终N分类中的CLS Top-K候选生成和重排能力。

## 6. 结论与后续决策

1. **P2b达到本阶段目标**：五折平均比P0高0.54个百分点，困难折平均高1.13个百分点。
2. **Weighted Patch优于Average Patch**：P2b比P1b高1.28个百分点，困难折提升最明显。
3. **容易折得到保护**：Fold 1/3/5均未下降，但目前只有一个训练种子，不能宣称已经获得稳定统计优势。
4. **P2b仍不直接替代默认流程**：总体收益只有6张图，且Fold 4存在BMP-1/BMP-2方向性偏置。
5. 结果支持继续P3：保持P0为主路径，使用CLS候选关系决定是否允许Weighted Patch参与；所有门控阈值必须在独立开发协议中选择。
6. P3/P4最终必须进入5-way及8/10-way开发评测，报告CLS Recall@K；不能只凭当前二分类五折决定N分类部署参数。

## 7. 完整性检查

- 五折缓存评测全部完成，Managed Run状态为`completed`；
- 固定`cls_weight=0.5`、`patch_weight=0.5`；
- `candidate_scope=all_classes`；
- `calibration_used=false`、`gating_used=false`；
- `novel_query_tuning=false`、`training_performed=false`；
- P2b执行阶段未加载文本编码器；
- 每折验证P2a来源`variant=p2a_masked_weighted_patch_only`；
- 每折保存P0/P2a特征及P2a配置文件的SHA-256；
- P0/P2a类别顺序、标签、Query元数据全部一致；
- 28项单元测试和源码`compileall`通过；
- 日志和结构化结果中未发现Traceback、CUDA OOM或NaN；
- 当次Managed Run源码快照包含58个文件。

## 8. 文件位置

- P2b实现：`PAD_Lite/src/dino_patch_weighted_fusion.py`
- 统一入口：`PAD_Lite/src/experiment_cli.py`
- 兼容入口：`PAD_Lite/src/dino_patch_cli.py`
- 配置：`configs/dino_patch_p2b_equal_fusion_336.json`
- 预设：`experiments/dino_s_p2b_cls_weighted_patch_equal_fusion_336.json`
- 测试：`PAD_Lite/tests/test_dino_patch_p2b.py`
- Managed Run：`outputs/experiment_runs/dino_s_p2b_cls_weighted_patch_equal_fusion_336/fivefold_equal_v1/`
