# DINOv2-Base + Letterbox 336 B2 五折实验

## 实验设置

- 视觉主干：`DINOv2-Base (ViT-B/14)`，86,580,480 参数，输出维度 768
- 对照主干：`DINOv2-Small (ViT-S/14)`，22,056,576 参数，输出维度 384
- 输入：`336 x 336`
- 空间预处理：Letterbox，完整车辆等比例缩放后居中填充
- 主干训练状态：完全冻结，`unfreeze_last_blocks=0`
- 训练参数：768→512 检索投影头、BNNeck、ID 分类头和可学习文本提示
- B2 损失：`ID Loss (1.0) + Batch-Hard Triplet Loss (1.0) + Text-anchor Contrastive Loss (0.5)`
- 训练轮数：每折 30 轮
- 评测：相同五折、Support/Query、`prototype_top_k=3`
- 随机种子：`2026 + fold`

除 DINOv2 主干及其输入维度外，Base 与此前 Small 336 B2 实验设置相同。

## Rank-1 结果

| 主干 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2-Small | 97.85% | **71.67%** | 95.90% | 72.56% | 94.14% | **86.425%** | 11.75% |
| DINOv2-Base | **98.71%** | 66.52% | **97.44%** | 72.56% | **96.88%** | 86.421% | 13.93% |

标准差与程序的 `summary.json` 一致，使用五折 Rank-1 的总体标准差。

Base 相对 Small 的逐折变化：

| Fold | Rank-1 变化 |
|---:|---:|
| 1 | +0.86 pp |
| 2 | **-5.15 pp** |
| 3 | +1.54 pp |
| 4 | +0.00 pp |
| 5 | +2.73 pp |
| 五折平均 | **-0.004 pp** |

## 结论

1. Base 与 Small 的五折平均 Rank-1 实质相同：86.421% 对 86.425%，差值只有 -0.004 个百分点。
2. Base 提高了 Fold 1、3、5，但 Fold 2 下降了 5.15 个百分点；五折标准差由 11.75% 增加到 13.93%，跨类别稳定性反而变差。
3. 更大的冻结视觉主干没有解决 Fold 2/4。当前困难折的瓶颈更可能来自类别本身、Support/Query 分布、局部判别部件或检索表示，而不是 DINOv2-Small 参数量不足。
4. 在当前数据和冻结主干设置下，没有理由直接用 Base 替换 Small：平均精度没有收益，模型参数量约为 Small 的 3.93 倍。
5. 如果继续验证 Base，下一步应做“Base 解冻最后 1 个 Block”或“CLS 全局特征 + Patch Token 局部特征融合”的单变量实验；不建议仅继续扩大到 DINOv2-Large。

## 训练时间与资源

- Base 五折训练循环平均：145.0 秒/折
- 训练时单卡显存约：2.4 GB
- 使用 GPU 4、6 并行完成

训练时间仅累计 30 轮训练循环，不包括模型加载、验证后 Support/Query 特征提取和结果写入。

## 文件位置

- 配置：`PAD_Lite/configs/dino_base_letterbox_text_anchor_336.json`
- Base 模型：`Resource/models/dinov2-base/`
- 五折输出：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_base_letterbox_text_anchor_336/dino/b2/`
- 日志：`PAD_Lite/logs/dino_base_letterbox_text_anchor_336/`

五折均存在 `metrics.json` 和 30 轮 `history.json`；结果与运行配置均记录为 `dinov2-base`，主干解冻参数量为 0，日志未发现 Traceback、CUDA OOM 或 NaN。
