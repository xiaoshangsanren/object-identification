# Annotation PAD-Lite 训练流程

## 1. 方案结论

本项目采用接近 PAD（Prompt-Anchored Vision-Text Distillation）原论文的轻量方案：

> 使用 Russian-Military-Vehicles 训练视觉检索编码器，通过可学习 TA-Prompt 将视觉特征锚定到冻结的 CLIP 文本空间，同时使用 ID Loss 和 Triplet Loss 增强判别能力；可选使用 EMA 视觉蒸馏提高增强条件下的稳定性。模型训练完成后完全冻结，再使用 Annotation Gallery 建立离线参考特征库。

第一版不构建炮塔、履带、炮管等显式概念资料，也不需要逐图文本描述。其语义信息来自 CLIP 冻结文本空间中的隐式语义锚定。

## 2. 任务定义

### 2.1 训练任务

- 训练数据：`Russian-Military-Vehicles`。
- 训练目标：学习能够迁移到未见军用车辆类别的视觉距离空间。
- 训练标签：Russian 数据集已有的 10 个类别标签。
- 文本输入：统一模板与可学习 TA-Prompt Token。
- 训练输出：纯视觉图像编码器与检索投影层。

### 2.2 最终检索任务

- 参考库：`Resource/clip_gallery/prototypes/`。
- 查询数据：与 Gallery 原型图片不重合的坦克图片或视频目标框。
- 输出：Gallery 已知车型、`unknown` 或背景拒绝。
- 推理阶段不使用文本编码器、TA-Prompt、ID 分类头或 Russian 数据集。

### 2.3 研究任务的准确名称

该实验主要属于 Base-to-Novel 少样本检索：

```text
Russian 的 10 个基础类别 → 参数训练
Gallery 的 10 个新类别   → 只提供少量视觉原型
测试视频中的目标         → 与 Gallery 原型进行检索
```

这不是让模型凭空识别未知车型。模型虽然没有用 Gallery 类别更新参数，但离线推理时能够看到这些类别的参考图片。

## 3. 严格的数据边界

### 3.1 允许进入训练的数据

```text
datasets/Russian-Military-Vehicles/
```

训练阶段可以使用：

- 原始图片；
- Pascal VOC XML 检测框；
- Russian 的类别标签；
- 由训练图片产生的数据增强版本。

### 3.2 禁止进入训练的数据

```text
Annotation/Resource/clip_gallery/prototypes/
Annotation/Resource/clip_gallery/unknown_tanks/
最终测试视频
最终 Query 测试图片
Gallery 十类的类别名称和文本提示
```

禁止的数据不得用于：

- 梯度更新；
- Prompt 学习；
- 超参数选择；
- Early Stopping；
- 检索阈值调节。

模型完全训练并冻结后，才允许读取 Gallery 原型图进行离线建库。

## 4. 总体流程

```text
Russian 图片和 VOC XML
          ↓
      车辆框裁剪
          ↓
类别均衡采样与图像增强
          ↓
   PAD-Lite 阶段一
  TA-Prompt 和投影层预热
          ↓
   PAD-Lite 阶段二
视觉微调与度量学习
          ↓
   可选 PAD-Lite 阶段三
      EMA 视觉蒸馏
          ↓
 Russian 类别留出验证
          ↓
使用全部 Russian 类别重新训练
          ↓
删除训练专用模块并冻结模型
          ↓
首次读取 Annotation Gallery
          ↓
生成离线 Gallery 特征库
          ↓
YOLO → PAD-Lite → Gallery 检索 → 时间融合
```

## 5. Russian 数据预处理

### 5.1 车辆裁剪

根据每张图片对应的 VOC XML：

1. 读取 `object/name` 和 `bndbox`。
2. 在检测框周围保留约 5%～10% 的上下文。
3. 将裁剪结果保存到对应类别目录。
4. 丢弃无效框；无目标图片单独保存为背景样本，不参与 ID 分类。

建议输出结构：

```text
datasets/processed/russian_pad/
├── images/
│   ├── bm-21/
│   ├── bmd-2/
│   ├── bmp-1/
│   ├── bmp-2/
│   ├── btr-70/
│   ├── btr-80/
│   ├── mt-lb/
│   ├── t-64/
│   ├── t-72/
│   └── t-80/
└── metadata.csv
```

`metadata.csv` 至少记录：

```text
image_path,class_id,source_image,source_split,bbox
```

### 5.2 图像增强

建议使用：

- 水平翻转；
- 轻微透视和尺度变化；
- 随机检测框边界偏移；
- 亮度、对比度和色温变化；
- 高斯模糊与轻度运动模糊；
- JPEG 压缩和分辨率退化；
- Random Erasing 模拟局部遮挡。

禁止使用明显不符合真实车辆成像的垂直翻转或过强几何扭曲。

## 6. PAD-Lite 模型结构

### 6.1 图像分支

第一版使用项目已有的 CLIP 图像编码器：

```text
车辆裁剪图
    ↓
CLIP ViT-B/32 图像编码器
    ↓
检索投影层 / BNNeck
    ↓
L2 归一化特征 z
```

训练期间增加临时 ID 分类头；部署时删除分类头。

### 6.2 文本锚点分支

冻结 CLIP 文本编码器，为每个 Russian 类别学习一组 TA-Prompt Token：

```text
A photo of a [V1][V2][V3][V4] military vehicle.
```

其中 `[V1]`～`[V4]` 是可学习 Token，不是人工属性描述。类别标签只负责将图片与对应的 Prompt 关联。

该方案不需要：

- 图片级文本描述；
- 车型百科资料；
- 部件 Mask；
- 概念概率；
- 最终 Gallery 类别文本。

## 7. 损失函数

第一版完整目标为：

\[
L =
\lambda_s L_{\mathrm{SupCon}}
+ \lambda_i L_{\mathrm{ID}}
+ \lambda_t L_{\mathrm{Triplet}}
+ \lambda_e L_{\mathrm{EMA}}
\]

各损失的作用如下：

| 损失 | 作用 |
|---|---|
| `L_SupCon` | 让图像特征与同类别 TA-Prompt 文本特征对齐 |
| `L_ID` | 提供 Russian 10 类的监督分类信号 |
| `L_Triplet` | 学习适合 Gallery retrieval 的类内紧凑、类间分离特征 |
| `L_EMA` | 可选；保持强弱增强视图之间的视觉特征与预测稳定 |

初始配置建议：

```yaml
supcon_weight: 0.5
id_weight: 1.0
triplet_weight: 1.0
triplet_margin: 0.3
ema_weight: 0.0
```

先令 `ema_weight: 0.0` 跑通前三项；EMA 作为独立消融实验加入。

## 8. 分阶段训练

### 8.1 阶段一：Prompt 与投影层预热

冻结：

- CLIP 文本编码器；
- CLIP 图像主干。

训练：

- TA-Prompt；
- 检索投影层；
- BNNeck；
- ID 分类头。

使用损失：

```text
L_SupCon + L_ID
```

第一轮可训练 5～10 个 Epoch，确认图文对比损失和 ID Loss 能够稳定下降。

### 8.2 阶段二：视觉度量学习

保持文本编码器冻结，解冻 CLIP 图像编码器最后 1～2 个 Transformer Block，同时继续训练投影层、TA-Prompt 和 ID 分类头。

使用损失：

```text
L_SupCon + L_ID + L_Triplet
```

采用类别均衡采样。例如每个 Batch 选择 5 个类别，每类选择 4～8 张图片，使同一 Batch 内同时存在正样本和困难负样本。

### 8.3 阶段三：可选 EMA 视觉蒸馏

阶段二稳定后，复制当前学生模型初始化 EMA 教师：

\[
\theta_T \leftarrow 0.999\theta_T + 0.001\theta_S
\]

同一图片生成两个视图：

```text
弱增强图片 → EMA 教师
强增强图片 → 学生模型
```

学生对齐教师的视觉特征与软分类分布。建议从 `ema_weight: 0.2` 开始独立验证，不要与基础版本同时首次上线。

## 9. Russian 内部 Base-to-Novel 验证

在最终 Gallery 参与之前，先用 Russian 类别模拟新类别检索。

建议进行 5 折类别留出：

```text
每折：
8 个 Russian 类别 → 训练
2 个 Russian 类别 → 完全不参与参数训练
```

对于留出的类别：

```text
每类少量图片 → 临时 Gallery
其余图片     → Query
```

评测指标：

- mAP；
- Rank-1；
- Rank-5；
- 类内平均距离；
- 最近类间距离；
- 模糊、遮挡和低分辨率条件下的检索稳定性。

由于每折只有两个未见类别，绝对准确率仅作为内部参考；主要比较不同训练方法在完全相同划分下的相对提升。

## 10. 必须完成的消融实验

| 实验 | 配置 |
|---|---|
| B0 | 原始冻结 CLIP，直接做临时 Gallery 检索 |
| B1 | `ID + Triplet` 微调 |
| B2 | `TA-Prompt SupCon + ID + Triplet`，即主要 PAD-Lite |
| B3 | B2 加 EMA 视觉蒸馏 |

判断规则：

```text
B2 > B1 > B0
```

只有当 B2 稳定优于 B1，才能认为文本锚定带来了额外价值。只有当 B3 在模糊、遮挡和低分辨率测试中稳定优于 B2，才保留 EMA 模块。

## 11. 最终训练与模型导出

完成类别留出实验和超参数选择后：

1. 使用全部 Russian 10 类重新训练最终模型。
2. 删除 ID 分类头。
3. 删除文本编码器和 TA-Prompt 分支。
4. 删除 EMA 教师。
5. 冻结图像编码器与检索投影层。
6. 导出纯视觉权重和预处理配置。

建议输出：

```text
Resource/models/pad_lite/
├── pad_lite_clip.pt
├── model_config.json
└── preprocessing.json
```

## 12. 接入 Annotation Gallery

模型冻结后，首次读取：

```text
Resource/clip_gallery/prototypes/
```

使用 PAD-Lite 图像编码器重新计算全部 Gallery 原型特征：

```text
104 张 Gallery 原型图
        ↓
冻结的 PAD-Lite 图像编码器
        ↓
L2 归一化特征
        ↓
离线 Gallery 特征缓存
```

原有 Gallery 分类、背景拒绝、未知拒识和时间融合逻辑继续复用。

## 13. 最终离线运行流程

```text
输入视频
    ↓
抽帧
    ↓
YOLO 检测车辆候选框
    ↓
裁剪车辆目标
    ↓
PAD-Lite 纯视觉编码器
    ↓
与 Gallery 多原型计算余弦相似度
    ↓
known / unknown / rejected_background
    ↓
IOU 关联与时间融合
    ↓
输出标注视频和 JSON
```

最终离线包需要：

- YOLO 权重；
- PAD-Lite 图像编码器；
- PAD-Lite 投影层；
- Gallery 图片或预计算特征；
- 检索和拒识阈值；
- Annotation 现有视频处理依赖。

最终离线包不需要：

- CLIP 文本编码器；
- TA-Prompt；
- EMA 教师；
- Russian 数据集；
- 网络连接。

## 14. 第一项实施任务

第一项实际开发工作是完成 Russian 数据预处理和固定实验划分：

1. 解析 VOC XML 并裁剪车辆。
2. 生成统一的 `metadata.csv`。
3. 固定 5 折类别留出配置并保存为配置文件。
4. 检查类别数量、无效框和可能的数据重复。
5. 在完全相同的划分上依次运行 B0、B1、B2、B3。

在 B2 尚未稳定超过 B1 前，不修改 Annotation 的正式离线识别分支。
