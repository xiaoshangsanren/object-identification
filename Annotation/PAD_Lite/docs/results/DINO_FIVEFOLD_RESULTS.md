# DINOv2 B0 / B1 / B2 五折实验结果

实验完成日期：2026-08-09

## 实验设置

- 数据：Russian-Military-Vehicles 的 10 个有效类别目标裁剪图。
- 划分：沿用原 PAD-Lite 五折类别留出划分；每折 8 个基础类参与训练，2 个未见类只用于 Support/Query 检索评测。
- Gallery：本次五折实验不读取 Annotation Gallery。
- 视觉主干：Annotation 本地 `DINOv2-small`，384 维 CLS 特征。
- B0：冻结原始 DINOv2，直接进行原型检索。
- B1：DINOv2 最后 2 个 Transformer Block、投影头、BNNeck 和分类头，使用 ID Loss + Batch-Hard Triplet Loss。
- B2：在 B1 基础上加入冻结 CLIP 文本编码器和类别独立可学习 Prompt，使用文本锚点对比损失。Prompt 模板不包含车型名称；测试时删除文本分支，只输入图像。
- 选择模型：依据基础类验证集的非 tiny 分类准确率保存每折 `best.pt`。

B2 使用的固定模板：

```text
A photo of a {learnable_tokens} military vehicle.
```

## DINOv2 五折结果

下表为未见类别 Query 的 Rank-1：

| 折 | 留出类别 | B0 | B1 | B2 |
|---|---|---:|---:|---:|
| 1 | mt-lb, t-64 | 96.14% | 97.00% | 96.57% |
| 2 | t-72, t-80 | 65.67% | 60.09% | 60.52% |
| 3 | bmd-2, btr-70 | 97.95% | 98.97% | 98.46% |
| 4 | bmp-1, bmp-2 | 61.86% | 63.26% | 68.84% |
| 5 | bm-21, btr-80 | 94.53% | 91.80% | 92.58% |
| **五折平均** | — | **83.23%** | **82.22%** | **83.39%** |

五折汇总指标：

| 方法 | Rank-1 | class mAP | retrieval mAP |
|---|---:|---:|---:|
| DINO-B0 | 83.23 ± 15.98% | 76.95 ± 16.36% | 75.47 ± 13.16% |
| DINO-B1 | 82.22 ± 16.97% | 76.64 ± 20.06% | 78.22 ± 17.26% |
| DINO-B2 | **83.39 ± 15.62%** | **77.22 ± 19.42%** | **78.67 ± 16.68%** |

## 与原 CLIP 实验对比

| 方法 | CLIP Rank-1 | DINOv2 Rank-1 | 变化 | CLIP retrieval mAP | DINOv2 retrieval mAP | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 74.45% | 83.23% | +8.78 pp | 64.80% | 75.47% | +10.67 pp |
| B1 | 79.23% | 82.22% | +2.99 pp | 75.71% | 78.22% | +2.51 pp |
| B2 | 76.95% | 83.39% | +6.45 pp | 74.75% | 78.67% | +3.92 pp |

## 结论

1. DINOv2-small 明显强于当前 CLIP ViT-B/32，尤其冻结 B0 的 Rank-1 提升 8.78 个百分点，说明 DINOv2 的纯视觉细粒度表征更适合这批数据。
2. DINO-B2 的平均 Rank-1、class mAP 和 retrieval mAP 均为三组最高，但相对 DINO-B0 的 Rank-1 仅提高 0.16 个百分点；语义训练当前最明确的收益体现在 retrieval mAP，提高 3.19 个百分点。
3. 改进并不稳定：B2 在 bmp-1/bmp-2 折比 B0 高 6.98 个百分点，但在 t-72/t-80 折低 5.15 个百分点。当前文本锚点尚不能稳定解决高度近似车型之间的区别。
4. 因此，现阶段不能只凭均值认定 B2 已全面优于冻结 DINOv2。进入完整 Annotation 流程前，应同时保留 DINO-B0 与 DINO-B2，并重点分析第 2 折的混淆样本和模型注意区域。

## 复现实验

在 `Annotation` 根目录执行：

```bash
conda activate cc_PAD_Lite
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.dino_cli b0 --fold all --device cuda:0
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.dino_cli b1 --fold all --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=6 python -m PAD_Lite.dino_cli b2 --fold all --device cuda:0 --resume
```

配置文件：`PAD_Lite/configs/dino_default.json`

正式输出：`PAD_Lite/outputs/dino/{b0,b1,b2}`

训练日志：`PAD_Lite/logs/dino/`

