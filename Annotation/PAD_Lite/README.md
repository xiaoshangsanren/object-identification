# Annotation PAD-Lite：B0 / B1 / B2

本目录是 Russian-Military-Vehicles 五折实验和 PAD-Lite 识别模型训练的固定程序目录。

## 目录结构

```text
PAD_Lite/
├── src/           # 可执行 Python 实现；模块名仍为 PAD_Lite.<module>
├── scripts/       # Windows/Linux 启动脚本
├── configs/       # 训练与评测配置
├── experiments/   # 可切换的命名实验预设
├── tests/         # 单元测试
├── evaluation_data/ # PAD-Lite 自有的固定评测数据划分
├── docs/
│   ├── design/    # 实验设计、评审和管理约定
│   └── results/   # 已完成实验的结果报告
├── environment/   # Conda 与 pip 环境文件
├── outputs/       # 模型、指标、预测和特征等运行产物
└── logs/          # 运行日志
```

根目录的 `__init__.py` 和 `__main__.py` 仅用于兼容原有命令；实际实现统一位于 `src/`。因此现有 `python -m PAD_Lite...` 命令不需要改变。

## 当前代码边界

当前可执行代码保留到纯视觉 P2B 实验及其既有训练、评测工具。P2B之后新增的外部知识库、图文知识重排、where文本定位、知识MIL和知识BBox实验代码已移除；对应历史输出与研究报告仅作归档。B2的训练期文本锚点属于P2B基线组成，继续保留，推理阶段仍为纯视觉输入。

## YOLO / DETR 零样本检测对比

使用 Russian-Military-Vehicles 原始整图和 XML，只评估当前通用预训练权重，不进行训练：

```bash
# CPU少量图片连通性检查
bash PAD_Lite/scripts/Run_Detector_ZeroShot_Smoke.sh

# 实验1阶段1：GPU空闲后进行零样本全量评测
bash PAD_Lite/scripts/Run_Experiment1_Stage1_ZeroShot.sh 4

# 实验1阶段2：GPU 4训练YOLO、GPU 6训练DETR，完成相同五折微调
bash PAD_Lite/scripts/Run_Experiment1_Stage2_FineTune.sh 4 6
```

阶段1配置位于 `configs/detector_zero_shot_yolo_detr.json`，阶段2配置位于 `configs/detector_finetune_yolo_detr.json`。实验1和实验2的完整设计见 `docs/design/DETECTOR_AND_RECOGNIZER_COMPARISON_PLAN.md`。

## 推荐：使用命名实验预设

现在所有历史 B0/B1/B2、文本锚点开关、输入策略、分辨率和主干解冻深度都通过独立实验名切换。统一入口不会改写旧配置，每次训练默认创建新的时间戳 Run，并保存当次源码与配置快照：

```bash
python -m PAD_Lite.experiment_cli list
python -m PAD_Lite.experiment_cli show dino_s_b2_text_anchor_letterbox_336_u00
python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_letterbox_336_u00 \
  --fold all --device cuda:0
```

不使用文本锚点时，只需切换实验名：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b1_no_text_letterbox_336_u00 \
  --fold all --device cuda:0
```

完整的防覆盖、恢复训练、源码快照和新增实验约定见 `docs/design/EXPERIMENT_PRESETS.md`。下文所有旧入口继续兼容，不会删除。

## 实验边界

- 参数训练只读取 Russian-Military-Vehicles 的目标裁剪图。
- 每折 8 个基础类训练，2 个未见类只用于 Support/Query 检索。
- Annotation Gallery 不会被本模块读取。
- B2 Prompt 模板不包含车型名称，避免 label leakage。
- 第一版不包含 EMA。

## 三组实验

| 实验 | 图像模型 | 训练损失 | 文本分支 |
|---|---|---|---|
| B0 | 原始冻结 CLIP ViT-B/32 | 无训练 | 无 |
| B1 | CLIP最后2个Block＋检索头 | ID＋Batch-Hard Triplet | 无 |
| B2 | CLIP最后2个Block＋检索头 | ID＋Triplet＋文本锚点对比 | 冻结文本编码器＋可学习TA-Prompt |

B2 的固定模板为：

```text
A photo of a {learnable_tokens} military vehicle.
```

每个基础类拥有4个独立可学习Token，但车型名称不会进入模板。推理时只保留视觉编码器和检索头。

## 输入

默认配置读取：

```text
../datasets/processed/russian_pad/crop_manifest.json
../datasets/Russian-Military-Vehicles/pad_lite_splits_v1/fold_01.json ... fold_05.json
Resource/models/clip-vit-base-patch32/
```

训练默认排除 `tiny=true` 样本；Support也排除tiny，Query保留全部样本并分别报告全部、core和tiny结果。

## Windows运行

在 `Annotation` 根目录执行：

```bat
runtime\python.exe -m PAD_Lite b0 --fold all
runtime\python.exe -m PAD_Lite b1 --fold all --resume
runtime\python.exe -m PAD_Lite b2 --fold all --resume
```

也可以直接双击或调用：

```text
PAD_Lite\scripts\Run_B0.cmd
PAD_Lite\scripts\Run_B1.cmd
PAD_Lite\scripts\Run_B2.cmd
```

只运行单折：

```bat
runtime\python.exe -m PAD_Lite b0 --fold 1
runtime\python.exe -m PAD_Lite b1 --fold 1 --resume
runtime\python.exe -m PAD_Lite b2 --fold 1 --resume
```

一轮快速连通性测试：

```bat
runtime\python.exe -m PAD_Lite b1 --fold 1 --epochs 1 --workers 0 --output-root PAD_Lite\smoke_outputs
runtime\python.exe -m PAD_Lite b2 --fold 1 --epochs 1 --workers 0 --output-root PAD_Lite\smoke_outputs
```

## Linux研发环境运行

服务器已经创建独立环境：

```bash
conda activate cc_PAD_Lite
```

环境位置为：

```text
/home/NCUT/25/cc/.conda/envs/cc_PAD_Lite
```

进入 `Annotation` 后运行：

```bash
python -m PAD_Lite b0 --fold all --device cuda
python -m PAD_Lite b1 --fold all --device cuda --resume
python -m PAD_Lite b2 --fold all --device cuda --resume
```

服务器有多张显卡时，先通过 `nvidia-smi` 选择空闲卡。例如使用物理GPU 6：

```bash
CUDA_VISIBLE_DEVICES=6 python -m PAD_Lite b1 --fold all --device cuda:0 --resume
```

其中程序内的 `cuda:0` 表示 `CUDA_VISIBLE_DEVICES` 暴露后的第一张卡。

## DINOv2 五折实验

以 Annotation 自带的本地 DINOv2-small 替换 CLIP 视觉主干，同时保持原有五折划分、Support/Query 评测和 B0/B1/B2 损失定义：

```bash
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.dino_cli b0 --fold all --device cuda:0
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.dino_cli b1 --fold all --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=6 python -m PAD_Lite.dino_cli b2 --fold all --device cuda:0 --resume
```

DINO-B0 使用冻结 DINOv2；DINO-B1 解冻最后 2 个 Block 并使用 ID + Triplet；DINO-B2 在训练阶段额外使用冻结 CLIP 文本编码器和不含车型名的可学习文本锚点，推理仍为纯视觉。

其独立配置为 `configs/dino_default.json`，输出位于 `outputs/01_baseline_and_finetuning/dino/`，不会覆盖原 CLIP 结果。完整指标和对比结论见 `docs/results/DINO_FIVEFOLD_RESULTS.md`。

## CLIP Backbone 解冻深度消融

单次训练可以通过 `--unfreeze-last-blocks` 设置解冻的最后 `n` 个 CLIP 视觉 Transformer Block；`n=0` 表示冻结 CLIP、只训练 retrieval head：

```bash
python -m PAD_Lite b1 --unfreeze-last-blocks 0 --output-root PAD_Lite/outputs/example_n0
python -m PAD_Lite b1 --unfreeze-last-blocks 2 --output-root PAD_Lite/outputs/example_n2
```

执行完整的 `n=0..12` 五折 Rank-1 消融：

```bash
python -m PAD_Lite.clip_backbone_sweep --blocks all --fold all --device cuda:0 --resume
```

也可以用 `--blocks 0,1,2` 或 `--blocks 0-6` 运行子集。该实验固定使用 B1 的 ID Loss + Batch-Hard Triplet Loss，只汇总 `rank1_all`；默认在每折评测后删除临时训练检查点以节省磁盘。结果位于 `outputs/01_baseline_and_finetuning/clip_backbone_sweep/RANK1_RESULTS.md`。

DINOv2-small 支持同样的参数和消融入口：

```bash
python -m PAD_Lite.dino_cli b1 --unfreeze-last-blocks 0 --output-root PAD_Lite/outputs/dino_example_n0
python -m PAD_Lite.dino_backbone_sweep --blocks all --fold all --device cuda:0 --resume
```

DINOv2 消融结果位于 `outputs/01_baseline_and_finetuning/dino_backbone_sweep/RANK1_RESULTS.md`，同样只汇总 `rank1_all`。

### DINOv2 Letterbox + 文本锚点实验

为避免中心裁剪删除细长车辆的车头和车尾，可使用保持宽高比、完整缩放并填充为正方形的 Letterbox 配置：

```bash
python -m PAD_Lite.dino_cli b2 \
  --config PAD_Lite/configs/dino_letterbox_text_anchor.json \
  --fold all \
  --device cuda:0
```

该配置冻结 DINOv2 主干，仅训练检索头和文本提示，损失为 ID、Batch-Hard Triplet 与文本锚点对比损失。五折结果与 CenterCrop 对照见 `docs/results/DINO_LETTERBOX_TEXT_ANCHOR_RESULTS.md`。

可以通过 `--image-size` 执行 DINOv2 输入分辨率实验；尺寸必须是 Patch Size 14 的整数倍。例如：

```bash
python -m PAD_Lite.dino_cli b2 \
  --config PAD_Lite/configs/dino_letterbox_text_anchor.json \
  --image-size 448 \
  --eval-batch-size 16 \
  --output-root PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_text_anchor_448 \
  --fold all \
  --device cuda:0
```

`224/336/448/518`四档输入尺寸的完整五折对比见 `docs/results/DINO_LETTERBOX_RESOLUTION_SWEEP_RESULTS.md`。

不使用文本锚点、只训练ID与Batch-Hard Triplet损失的DINO-B1对照可使用：

```bash
python -m PAD_Lite.dino_cli b1 \
  --config PAD_Lite/configs/dino_letterbox_no_text.json \
  --image-size 448 \
  --eval-batch-size 16 \
  --output-root PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_letterbox_no_text_448 \
  --fold all \
  --device cuda:0
```

无文本B1与文本锚点B2在`224/336/448/518`四档分辨率上的完整配对结果见 `docs/results/DINO_LETTERBOX_TEXT_ANCHOR_ABLATION_RESULTS.md`。

使用原`Resize + CenterCrop`策略运行B2分辨率实验时，建议保持测试短边与输入尺寸的`8/7`比例：

```bash
python -m PAD_Lite.dino_cli b2 \
  --config PAD_Lite/configs/dino_center_crop_text_anchor.json \
  --image-size 448 \
  --resize-short-edge 512 \
  --eval-batch-size 16 \
  --output-root PAD_Lite/outputs/02_dino_resolution_and_preprocess/dino_center_crop_text_anchor_448 \
  --fold all \
  --device cuda:0
```

`224/336/448/518`四档 CenterCrop B2 的完整五折结果及其与 Letterbox 的逐折对照见 `docs/results/DINO_CENTER_CROP_TEXT_ANCHOR_RESOLUTION_RESULTS.md`。

使用 DINOv2-Base 复现 Letterbox 336 B2 五折实验：

```bash
python -m PAD_Lite.dino_cli b2 \
  --config PAD_Lite/configs/dino_base_letterbox_text_anchor_336.json \
  --fold all \
  --device cuda:0
```

Base 与 Small 的逐折 Rank-1 对照见 `docs/results/DINO_BASE_LETTERBOX_TEXT_ANCHOR_336_RESULTS.md`。

参考 CVPRW 2026 工业物体检索论文，将 CLS 全局特征与 Patch 局部特征安全融合、并重点改善困难类别的实现方案见 `docs/design/DINO_PATCH_SAFE_RERANKER_PLAN.md`。

P1a（Masked Patch Average only）使用冻结DINOv2-Small，不使用CLS融合和文本锚点，只训练Patch投影、BNNeck与ID头：

```bash
python -m PAD_Lite.dino_patch_cli p1a \
  --config PAD_Lite/configs/dino_patch_safe_letterbox_336.json \
  --fold all \
  --device cuda:0
```

P1a的五折Rank-1和CLS/Patch错误互补分析见 `docs/results/DINO_PATCH_P1A_RESULTS.md`。

P1b在不重新训练DINO或Patch Head的前提下，复用P0 CLS-B2与P1a缓存特征，构造固定50/50的联合余弦表示。融合权重不使用Novel Query标签选择，也不提前引入P4的Calibration：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_p1b_cls_patch_average_equal_fusion_336 \
  --run-id fivefold_equal_v1 \
  --fold all --device cpu
```

也可以使用兼容入口：

```bash
python -m PAD_Lite.dino_patch_cli p1b --fold all --device cpu
```

P1b只进行缓存特征融合与检索评测，因此无需占用GPU，也没有可恢复的训练检查点。

P1b完整五折结果及其相对P0的纠错/伤害分析见 `docs/results/DINO_PATCH_P1B_RESULTS.md`。

P2a把P1a的均匀Patch Average替换为逐图内容相关的Masked Softmax Weighted Pooling；DINO继续冻结，只训练Patch Scorer、Projection、BNNeck和ID Head：

```bash
CUDA_VISIBLE_DEVICES=6 python -m PAD_Lite.experiment_cli run \
  dino_s_p2a_weighted_patch_letterbox_336 \
  --run-id fivefold_seed2026_v1 \
  --fold all --device cuda:0
```

P2a是Patch-only消融，不使用CLS融合或文本锚点。每折额外保存Query Patch权重、Padding质量检查、P0错误互补统计和注意力热力图。

P2a完整五折结果、P0/P1a/P1b对照和注意力诊断见 `docs/results/DINO_PATCH_P2A_RESULTS.md`。

P2b复用P0 CLS-B2与P2a Weighted Patch缓存，执行固定50/50、无门控、无标定的全候选融合：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_p2b_cls_weighted_patch_equal_fusion_336 \
  --run-id fivefold_equal_v1 \
  --fold all --device cpu
```

P2b不进行训练，也不根据Novel Query结果选择融合权重；其作用是测量Weighted Patch与CLS无条件融合的上限和潜在伤害。

P2b完整五折Rank-1、相对P0/P1b/P2a的纠错与伤害分析见 `docs/results/DINO_PATCH_P2B_RESULTS.md`。

更严格的3折训练、2折测试协议把原5个类别折中的3折（6类）用于训练，剩余2折（4类）合并为未见类别Support/Query；默认枚举全部10种测试折组合：

```bash
python -m PAD_Lite.p2b_three_two_protocol build-splits
python -m PAD_Lite.p2b_three_two_protocol p0 --episodes all --device cuda:0
python -m PAD_Lite.p2b_three_two_protocol p2a --episodes all --device cuda:0
python -m PAD_Lite.p2b_three_two_protocol p2b --episodes all --device cpu
python -m PAD_Lite.p2b_three_two_protocol summary
```

该协议使用独立配置`configs/p2b_three_train_two_test_4way.json`和独立输出目录，不改变原二类别五折实验。

该协议的完整10-Episode实验结果、逐类别统计和完整性检查见 `docs/results/P2B_THREE_TRAIN_TWO_TEST_4WAY_RESULTS.md`。

导出某个未见测试类在所有相关Episode中的P2a Patch热力图，并同时标注P0、P2a、P2b预测结果：

```bash
python -m PAD_Lite.export_class_attention_heatmaps --class-name t-72
```

默认输出到该实验目录下的`t-72_attention_heatmaps_gallery_reference/`。每张图左侧是P2b融合特征在当前Episode Gallery支持集中余弦相似度最高的Top-1参考图，右侧是当前Query的P2a Patch热力图；`index.html`可直接浏览全部结果，`index.csv`和`index.json`保存逐图预测、参考图类别及相似度。P2b分类使用Top-3类别原型聚合，因此分类结果可能与单张Gallery Top-1参考图的类别不同。

加入训练期文本锚点的 DINO-B2 解冻深度消融：

```bash
python -m PAD_Lite.dino_backbone_sweep --variant b2 --blocks all --fold all --device cuda:0 --resume
```

其结果位于 `outputs/01_baseline_and_finetuning/dino_text_anchor_backbone_sweep/`；`RANK1_RESULTS.md` 是逐折结果，`TEXT_ANCHOR_EFFECT.md` 是相同解冻层数下 B2 相对无文本 B1 的 Rank-1 净变化。

环境复现文件：

```text
PAD_Lite/environment/environment-server.yml
PAD_Lite/environment/requirements-server.txt
```

默认配置在 `configs/default.json`。路径相对于 `Annotation` 根目录解析，因此从其他工作目录启动也不会改变数据位置。

## 输出

```text
PAD_Lite/outputs/
├── b0/
│   ├── fold_01/
│   │   ├── metrics.json
│   │   ├── predictions.json
│   │   └── features.pt
│   └── summary.json
├── b1/
│   ├── fold_01/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   ├── history.json
│   │   ├── metrics.json
│   │   └── predictions.json
│   └── summary.json
└── b2/
    └── ...
```

每折主要报告：

- `rank1_all`：全部Query；
- `rank1_core`：非tiny Query；
- `rank1_tiny`：tiny Query；
- `class_map`：按类别分数计算的mAP；
- `retrieval_map`：Query对Support原型的检索mAP。

每个类别折只有2个候选类，因此不报告没有意义的Rank-5。

## 训练策略

B1前3个Epoch、B2前5个Epoch只训练投影层、BNNeck、ID头以及B2 Prompt；之后解冻CLIP视觉编码器最后2个Transformer Block。冻结的CLIP文本编码器只在B2训练期间提供语义空间，部署时删除。

每个Batch默认采样4类、每类4张，保证Batch-Hard Triplet存在正负样本。模型按基础类验证集的非tiny分类准确率保存 `best.pt`，随后使用该权重评测未见类别。

## 全量 Final 模型与完整流程

五折实验用于选择方法和观察稳定性。部署模型不选择任意一折，而是在固定30轮预算下使用 Russian-Military-Vehicles 的10类、全部1254个有效裁剪重新训练：

```bash
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.final_train b1 --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=6 python -m PAD_Lite.final_train b2 --device cuda:0 --resume
```

输出为：

```text
PAD_Lite/outputs/01_baseline_and_finetuning/final/b1/final.pt
PAD_Lite/outputs/01_baseline_and_finetuning/final/b2/final.pt
```

B0保持原始冻结CLIP，不产生训练权重。完整Annotation流程新增三个Gallery识别后端：`pad_lite_b0`、`pad_lite_b1`、`pad_lite_b2`。三者仍使用YOLO产生候选框，区别只在候选裁剪的视觉编码器；B2推理阶段不加载文本Prompt。

固定10类Gallery测试划分使用每类约2/3图片作为Support、约1/3作为Test：

```bash
python -m PAD_Lite.build_gallery_10class_split
python -m PAD_Lite.evaluate_gallery_10class b0 --device cuda:0
python -m PAD_Lite.evaluate_gallery_10class b1 --device cuda:0
python -m PAD_Lite.evaluate_gallery_10class b2 --device cuda:0
```

原始`Resource/clip_gallery`不会被修改。Support Gallery位于`Resource/clip_gallery_eval_10class_v1`，Test与固定划分清单位于`PAD_Lite/evaluation_data/gallery_10class_eval_v1`，结果位于`PAD_Lite/outputs/04_gallery_and_recognizer_evaluation/gallery_10class`。
