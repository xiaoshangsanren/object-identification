# Annotation 检测器与细粒度识别器对比实验计划

## 1. 研究目标

本计划将两个模块分别比较，避免检测误差和分类误差相互混淆：

1. 实验1：在原始整图上比较通用预训练 YOLO 与 DETR 的车辆候选框能力。
2. 实验2：在 XML 真实框生成的 `russian_pad` 裁剪图上比较 P2B 与 ConvNeXt-Tiny 的细粒度检索能力。
3. 两项独立实验完成后，才进行四种组合的端到端确认。

```text
原始整图
   ↓
YOLO / DETR                         ← 实验1
   ↓ 预测框裁剪
P2B / ConvNeXt-Tiny                 ← 实验2先用GT裁剪单独比较
   ↓
Gallery retrieval
```

## 2. 实验1：YOLO vs DETR

### 2.1 第一阶段：Zero-shot直接替换实验

这是当前首先执行的阶段，两套模型都不训练、不反向传播、不更新参数：

| 模型 | 权重 | 输入 |
|---|---|---|
| YOLO | `Resource/models/yolo/yolo26n.pt` | Russian原始整图 |
| DETR | `Resource/models/detr-resnet-50` | 完全相同的原始整图 |

数据来自：

```text
datasets/Russian-Military-Vehicles/train/*.jpg
datasets/Russian-Military-Vehicles/train/*.xml
```

XML只用于评估，不用于更新模型。项目中另一个已经微调过的 `custom_yolo_best.pt` 不进入本阶段。

为复现当前 Annotation YOLO 候选框规则，两模型只保留COCO车辆类别：

```text
car, motorcycle, bus, truck
```

YOLO沿用当前640输入；DETR沿用本地模型自带的默认预处理。因而本阶段回答的是“保持各自开箱设置时，DETR能否直接替换当前YOLO”，不是严格的相同计算量比较。

默认运行点置信度为0.30；计算AP时从0.001开始保留预测。指标包括：

- AP50、AP75、mAP50:95；
- 置信度0.30下的Precision、Recall、F1；
- 每图误检数；
- 小目标Recall；
- 按原始车型统计的Recall；
- 单图推理耗时。

先运行CPU冒烟验证：

```bash
bash Annotation/PAD_Lite/scripts/Run_Detector_ZeroShot_Smoke.sh
```

GPU空闲后，在物理GPU 4执行全量993张图片：

```bash
bash Annotation/PAD_Lite/scripts/Run_Experiment1_Stage1_ZeroShot.sh 4
```

脚本要求该卡至少有8000 MiB空闲显存，不满足时直接退出而不会抢占现有任务。GPU参数可换成任意空闲卡号。结果保存到：

```text
Annotation/PAD_Lite/outputs/detector_zero_shot/full_<UTC时间>/
```

### 2.2 第二阶段：相同数据的领域微调

本阶段回答“在相同Russian监督下哪种检测架构更强”。YOLO和DETR同时从各自通用预训练权重开始，10种车型统一为单一粗类别 `military_vehicle`。

固定数据位于：

```text
PAD_Lite/evaluation_data/detector_finetune_v2/
```

划分种子为2026。每张原图及其全部目标只能位于一个分区；五个外层Test互不重叠且合计覆盖全部993张图。每折约为：

```text
Train       693～698张
Validation  100张
Test        195～200张
```

训练预算：

| 模型 | 最大Epoch | 早停Patience | Batch | 输入 |
|---|---:|---:|---:|---|
| YOLO26n | 100 | 20 | 16 | 640 |
| DETR-R50 | 30 | 8 | 4 | 模型默认短边800、最长边1333 |

DETR使用AdamW，检测Transformer部分学习率为 `1e-4`，ResNet backbone为 `1e-5`。两模型均只用Validation选择最佳权重，Test只在训练完成后评估。

GPU 4和6均空闲后，一条命令并行完成两模型的五折训练、Test推理和汇总：

```bash
bash Annotation/PAD_Lite/scripts/Run_Experiment1_Stage2_FineTune.sh 4 6
```

脚本要求两张不同GPU均至少有18000 MiB空闲显存。第三个可选参数是Run ID；若中断，可使用同一ID重跑并跳过已有 `result.json` 的完整折：

```bash
bash Annotation/PAD_Lite/scripts/Run_Experiment1_Stage2_FineTune.sh 4 6 20260820T010000Z
```

结果位于 `PAD_Lite/outputs/detector_finetune/<Run ID>/`。阶段2不能与阶段1混写结论：阶段1比较开箱即用能力，阶段2比较相同领域监督后的能力。

## 3. 实验2：P2B vs ConvNeXt-Tiny

### 3.1 数据边界

使用：

```text
datasets/processed/russian_pad/
```

这些目标图由XML真实框生成，不使用YOLO或DETR预测框，因此实验2测到的是识别器本身的能力上限。任何来自同一原图的多个crop必须始终处于同一个Train、Validation、Gallery或Query分区，禁止源图泄漏。

### 3.2 主协议

主实验采用10类已知车型的图像级五折检索：

- Train：训练特征模型；
- Validation：早停和选择设置；
- Gallery：不参与梯度，用于构建10类参考特征；
- Query：只进行最终Rank-1评估。

现有“3折训练、2折测试、4-way未见类”协议保留为补充压力测试，不替代10类主结果。

### 3.3 两层比较

系统级比较：

```text
当前完整P2B（DINOv2 + CLS候选 + Patch Scorer + rerank）
vs
ConvNeXt-Tiny + Projection + BNNeck + Gallery retrieval
```

该结果比较完整方案，不能解释为纯backbone优劣。

如果需要隔离backbone差异，再增加受控比较：

```text
DINOv2 + Global Pool + 相同Retrieval Head
ConvNeXt-Tiny + Global Pool + 相同Retrieval Head
```

两者统一336 Letterbox、512维embedding、ID Loss、Batch-Hard Triplet、Gallery原型和余弦相似度。

主要指标：每折Rank-1、平均Rank-1、Macro Rank-1、Micro Rank-1、逐车型Rank-1、耗时和显存。

## 4. 后续端到端确认

独立实验完成后运行：

| 检测器 | 识别器 |
|---|---|
| YOLO | P2B |
| DETR | P2B |
| YOLO | ConvNeXt-Tiny |
| DETR | ConvNeXt-Tiny |

Query从原始整图开始，以IoU不低于0.5进行预测框与GT的一对一匹配。端到端主指标为：

```text
Joint Rank-1 = 成功检测且车型Rank-1正确的GT数量 / 全部GT数量
```

同时报告Detection Recall和在成功检测目标上的Conditional Rank-1，以定位性能损失来自检测还是识别。

## 5. 固定实验规则

1. 所有结果必须保存模型哈希、配置、逐图预测和运行设备。
2. Query/Test不得用于阈值选择或训练。
3. 新实验必须新增入口和输出目录，不覆盖历史B0/B1/B2/P1/P2代码与结果。
4. 冒烟结果只能说明程序连通，不能作为模型能力结论。
