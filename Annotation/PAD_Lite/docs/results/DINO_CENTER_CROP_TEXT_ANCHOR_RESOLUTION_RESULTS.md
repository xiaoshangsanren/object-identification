# DINOv2 CenterCrop 输入分辨率 B2 五折实验

## 实验设置

- 输入分辨率：`224`、`336`、`448`、`518`
- 测试预处理：先按短边等比例缩放，再从图像中心裁成正方形
- 短边缩放尺寸：`256`、`384`、`512`、`592`，始终保持 `resize_short_edge / image_size = 8/7`
- 训练预处理：`RandomResizedCrop(image_size)` 及原 CenterCrop 训练增强流程
- 视觉主干：`DINOv2-small`，全部冻结（`unfreeze_last_blocks=0`）
- 训练参数：检索头与可学习文本提示
- B2 损失：`ID Loss (1.0) + Batch-Hard Triplet Loss (1.0) + Text-anchor Contrastive Loss (0.5)`
- 训练轮数：每折 30 轮
- 评测：相同五折、Support/Query、`prototype_top_k=3`
- 随机种子：`2026 + fold`

224 复用此前相同 CenterCrop、B2 和冻结主干设置的实验结果；336、448、518 使用 GPU 4、6 完成。除输入尺寸、短边缩放尺寸和不影响特征结果的评估 Batch Size 外，其余条件一致。

## CenterCrop Rank-1 结果

| 输入尺寸 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折平均 | 标准差 | 平均训练时间/折 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 224 | **97.00%** | **69.53%** | 97.95% | **74.42%** | 92.19% | **86.22%** | **11.89%** | 122.0 秒 |
| 336 | 96.57% | 69.10% | **98.46%** | 72.09% | 91.02% | 85.45% | 12.41% | 162.0 秒 |
| 448 | 96.57% | 64.38% | 97.44% | 71.63% | 91.41% | 84.28% | 13.65% | 259.1 秒 |
| 518 | **97.00%** | 66.95% | 97.44% | 72.56% | **92.58%** | 85.30% | 12.93% | 331.7 秒 |

标准差与程序生成的 `summary.json` 一致，使用五折 Rank-1 的总体标准差。训练时间只累计 30 轮训练循环，不包括模型加载、Support/Query 特征提取和结果写入。

## 与同尺寸 Letterbox B2 对照

下表为 `CenterCrop Rank-1 - Letterbox Rank-1`，正数表示 CenterCrop 更好。

| 输入尺寸 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | 五折均值差 |
|---:|---:|---:|---:|---:|---:|---:|
| 224 | +0.00 pp | +2.15 pp | +1.03 pp | +2.33 pp | +0.39 pp | **+1.18 pp** |
| 336 | -1.29 pp | -2.58 pp | +2.56 pp | -0.47 pp | -3.13 pp | **-0.98 pp** |
| 448 | -0.86 pp | -3.00 pp | -1.54 pp | -3.72 pp | -3.13 pp | **-2.45 pp** |
| 518 | -0.86 pp | -1.72 pp | +1.03 pp | -3.26 pp | -1.56 pp | **-1.27 pp** |

对应的五折均值：

| 输入尺寸 | CenterCrop | Letterbox |
|---:|---:|---:|
| 224 | **86.22%** | 85.04% |
| 336 | 85.45% | **86.42%** |
| 448 | 84.28% | **86.73%** |
| 518 | 85.30% | **86.58%** |

需要注意，这不是只替换测试时裁剪方式的单变量消融。现有 CenterCrop 配方在训练阶段使用 `RandomResizedCrop`，Letterbox 配方在训练和测试阶段都保留完整车辆并填充。因此结果比较的是两套完整空间预处理配方。

## 结论

1. CenterCrop 下增大输入分辨率没有提高五折平均 Rank-1。224 的 86.22% 最高；336、448、518 分别下降 0.77、1.93、0.91 个百分点。
2. 448 的结果最差且波动最大，说明“输入越大，细粒度识别越好”在 CenterCrop 配方下不成立。增大分辨率只提高保留下来的中心区域的 Token 密度，不能恢复已经被裁掉的车头、车尾或上下文。
3. Fold 2/4 仍是困难折。它们在 CenterCrop 448 上分别只有 64.38% 和 71.63%，说明困难折的主要问题不是 224 输入像素不足。
4. 除 224 外，Letterbox 在各尺寸的五折均值都优于 CenterCrop；448 的优势达到 2.45 个百分点。当前应保留 Letterbox 作为主预处理方案。
5. CenterCrop 224 高于 Letterbox 224，说明中心区域对低分辨率特征仍有价值，但这个优势没有随分辨率增长。若继续优化，应研究“完整车辆全局特征 + 局部 Patch Token/部件特征”的融合，而不是继续单独提高 CenterCrop 尺寸。

## 输出位置

- 224：`PAD_Lite/outputs/01_baseline_and_finetuning/dino_text_anchor_backbone_sweep/unfreeze_00/dino/b2/`
- 336：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_center_crop_text_anchor_336/dino/b2/`
- 448：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_center_crop_text_anchor_448/dino/b2/`
- 518：`PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_center_crop_text_anchor_518/dino/b2/`
- 日志：`PAD_Lite/logs/dino_center_crop_text_anchor_resolution_sweep/`

所有 20 个折次均通过以下结果检查：每折存在 `metrics.json`，训练历史为 30 轮，模型为 DINOv2-small B2，主干解冻数为 0，损失权重为 `1.0 / 1.0 / 0.5`。本次新增的 15 个折次日志未发现 Traceback、CUDA OOM 或 NaN。
