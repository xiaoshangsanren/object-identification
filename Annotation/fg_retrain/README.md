# fg_retrain：训练期语言增强的纯视觉细粒度检索

## 0. 统一实验环境

从 2026-09-03 起，本项目后续的数据处理、训练、评测和模型导出统一使用 Conda 环境：

```bash
conda activate cc_object_identification
```

- 环境名：`cc_object_identification`
- Conda入口：`/home/NCUT/25/cc/.conda/envs/cc_object_identification`
- 物理位置：`/mnt/sata_ssd/cc-2025/conda_envs/cc_object_identification`
- Python：3.11.15
- PyTorch：2.7.1+cu118
- torchvision：0.22.1+cu118
- 已验证GPU：8张 NVIDIA GeForce RTX 4090
- 已安装：Transformers、timm、datasets、PyArrow、Pandas、scikit-learn、Pillow、SciPy等基础依赖

除非某项复现实验明确要求隔离环境，否则不得再使用 `base` 或 `cc_PAD_Lite` 运行本阶段实验。运行实验前应记录 `sys.executable`、PyTorch版本、CUDA版本和实际可见GPU。

## 1. 本阶段的研究问题

本阶段不以复现或超越 FG-CLIP、FG-CLIP 2 等依靠海量图文数据预训练得到的基础模型为目标，也不把“更换一个更强的预训练模型”当作方法贡献。

本阶段研究一个范围更窄、变量更可控的问题：

> 在 Stanford Cars-196 这一特定细粒度领域中，能否只在训练阶段加入类别语义和类别间差异知识，使一个普通 ImageNet 预训练 ViT 学到更有判别力、更具泛化性的纯视觉检索特征？

该问题的研究层级更接近 LaFG：语言是训练监督信号，不是测试输入。最终任务仍是 image-to-image Gallery retrieval。

## 2. 固定的基础模型

### 2.1 视觉端：实际被训练和部署的模型

- 模型：`google/vit-base-patch16-224`
- 架构：ViT-Base/16
- 参数量：约 86.4M（不计原始 ImageNet 分类头）
- 预训练：ImageNet-21k 预训练，再在 ImageNet-1k 上微调
- 输入：`pixel_values`，形状为 `(B, 3, 224, 224)`
- Token 输出：`(B, 197, 768)`，其中包含 1 个 CLS token 和 196 个图像 patch token
- 项目路径：`Annotation/Resource/models/vit-base-patch16-224-imagenet`

ViT 的 ImageNet 分类头不用于 Gallery 检索。模型取 CLS/聚合后的视觉特征，经可学习的 Retrieval Projection 映射为检索向量。

### 2.2 语言端：仅训练期使用的监督工具

- 模型：`openai/clip-vit-base-patch32`
- 实际使用部分：CLIP tokenizer、Text Transformer 和 text projection
- 不使用：CLIP image encoder
- 文本特征输出：`(B_text, 512)`
- 项目路径：`Annotation/Resource/models/clip-vit-base-patch32`

第一版冻结 CLIP 文本端。类别描述和类别差异描述先经过 CLIP 文本端编码成固定语义锚点，再监督 ViT 的视觉表示。

ViT 的原始 768 维视觉空间与 CLIP 的 512 维文本空间不是天然对齐的，禁止直接计算二者相似度。必须使用可学习的视觉投影头把 ViT 表示映射到共享的 512 维空间，并通过训练损失建立对齐。

## 3. 数据集

- 数据集：Stanford Cars-196
- 类别数：196
- 官方训练图像：8,144
- 官方测试图像：8,041
- 总图像数：16,185
- 项目路径：`datasets/Stanford-Cars-196`
- 物理路径：`/mnt/sata_ssd/cc-2025/annotation_datasets/Stanford-Cars-196`
- 当前图像存储：`parquet/train/*.parquet` 和 `parquet/test/*.parquet`
- 官方类别名称与边界框标注：`annotations/`

类别名称、语义文本和类别对差异文本只允许由训练协议可见的信息生成。测试 Query 图像不能用于生成或修订训练文本。

## 4. 训练数据流

```text
训练图像 I
  -> ImageNet预训练ViT
  -> 768维视觉表示
  -> Retrieval Projection（768 -> 512）
  -> L2归一化视觉检索向量 z

训练期类别/差异文本 T
  -> CLIP Tokenizer
  -> 冻结的CLIP Text Encoder
  -> 512维文本锚点 e

z与图像标签
  -> 图像-图像检索损失（主任务）

z与e
  -> 训练期语言辅助损失

联合损失
  -> 只更新ViT允许解冻的层、Retrieval Projection和训练期辅助头
```

主任务始终是图像检索。语言损失只能作为辅助约束，不能取代图像-图像检索损失。

建议第一版总损失为：

```text
L_total = L_retrieval + lambda_id * L_id
                        + lambda_sem * L_class_semantic
                        + lambda_diff * L_pair_difference
```

其中：

- `L_retrieval`：SupCon/Triplet 等与 Gallery 检索直接一致的视觉度量学习损失；
- `L_id`：仅训练期使用的类别分类辅助损失；
- `L_class_semantic`：视觉特征与本类别语义锚点的对齐损失；
- `L_pair_difference`：困难类别对之间的视觉关系与有方向差异文本的对齐损失。

## 5. 测试和部署数据流

测试阶段必须删除或禁用：

- CLIP tokenizer；
- CLIP text encoder；
- 类别语义文本和差异文本；
- 文本特征缓存；
- ID 分类头和仅训练期关系头。

只导出：

```text
ImageNet预训练并经过任务训练的ViT
  + Retrieval Projection
  + L2 Normalize
```

最终检索流程：

```text
Gallery图像 -> 纯视觉模型 -> Gallery特征库（离线预计算）
Query图像   -> 纯视觉模型 -> Query特征
Query特征 x Gallery特征 -> 余弦相似度 -> Top-K结果/类别结果
```

因此，项目最终证明的应当是“训练期语言监督改善了纯视觉索引”，而不是“测试时图文模型读取文本后完成分类”。

## 6. 最小对照实验

所有实验必须从同一份 ImageNet ViT 初始化权重独立开始：

| 实验 | 视觉检索训练 | 类别语义 | 类别对差异文本 | 目的 |
|---|---:|---:|---:|---|
| E0 | 否 | 否 | 否 | ImageNet ViT 原始检索能力 |
| E1 | 是 | 否 | 否 | 纯视觉训练基线 |
| E2 | 是 | 是 | 否 | 类别语义是否带来增益 |
| E3 | 是 | 是 | 是 | 类别差异知识是否进一步增益 |
| E3-Shuffle | 是 | 打乱 | 打乱 | 排除普通正则化或额外损失造成的假增益 |
| E3-Reverse | 是 | 是 | 交换方向 | 验证模型是否真正使用差异方向 |

E1 与 E2/E3 的差异才是本研究的核心结论。FG-CLIP 2 等大型细粒度图文模型最多作为外部性能参考，不作为要求公平击败的同资源基线。

## 7. 当前明确不纳入主线的内容

- 不以 FG-CLIP 2 作为待训练主干；
- 不继承 P2B、旧知识库、where 定位、旧 BBox 或旧热力图流程；
- 不在推理阶段输入任何文本；
- 不让最终测试 Gallery 参与反向传播；
- 不把海量额外图文预训练数据纳入第一阶段；
- 第一阶段不引入 DiT 教师或额外生成模型。

## 8. 第一阶段代码：E0 纯视觉 ViT 基线

当前已实现的 E0 只测量 ImageNet 预训练 ViT 的原始细粒度检索能力：

```text
Stanford Cars-196 官方 test（8,041 张）
  -> ViT Base/16 图像预处理（224 × 224）
  -> ViT 最后一层 CLS token（768 维）
  -> L2 Normalize
  -> 测试集 leave-one-out 全库检索
  -> Rank-1 / Recall@K / mAP
```

E0 中不进行训练，不使用随机初始化的 Retrieval Projection，也不加载 CLIP 或任何文本。否则测得的就不再是 ImageNet ViT 的原始视觉能力。

### 8.1 Query 与 Gallery 的严格定义

设官方测试集为：

```text
D_test = {(x_i, y_i)}，i = 1, ..., 8041
```

当 `x_i` 作为 Query 时，它自己的 Gallery 为：

```text
G_i = D_test - {(x_i, y_i)}
```

因此每个 Query 对应 8,040 张 Gallery 图像。程序会把相似度矩阵对角线置为负无穷，确保 Query 本身不能成为检索结果；Rank-1 类别取余弦相似度最高的另一张测试图的标签。

这套协议属于“测试集 leave-one-out 检索”：它使用其余带类别身份的测试图作为 Gallery，但没有 Query 自匹配泄漏。它不同于“训练集作 Gallery、测试集作 Query”的归纳式协议，论文和表格中必须单独标明，后续各对照实验也必须使用完全相同的协议。

### 8.2 代码阅读顺序

1. `configs/e0_vit_test_leave_one_out.json`：实验参数和资源路径；
2. `src/fg_retrain/evaluate_vit_leave_one_out.py`：完整实验入口；
3. `src/fg_retrain/data.py`：读取测试 parquet、保持原始行号、组织 ViT 输入；
4. `src/fg_retrain/modeling.py`：加载纯视觉 ViT、提取并归一化 CLS 特征；
5. `src/fg_retrain/retrieval.py`：屏蔽自身并完成 leave-one-out 全库检索；
6. `tests/test_retrieval.py`：协议正确性的最小单元测试。

以上代码只在 `Annotation/fg_retrain/src/fg_retrain` 包内互相导入，不调用 PAD_Lite 或 Annotation 中其他实验代码。数据集和预训练权重仅作为只读资源使用。

### 8.3 启动命令

在仓库任意目录执行：

```bash
bash Annotation/fg_retrain/scripts/run_e0_vit_test_leave_one_out.sh
```

指定物理 GPU 时，例如使用 4 号卡：

```bash
CUDA_VISIBLE_DEVICES=4 bash Annotation/fg_retrain/scripts/run_e0_vit_test_leave_one_out.sh
```

脚本固定通过 `cc_object_identification` 环境运行。每次执行都会创建新的时间戳目录，避免覆盖历史结果。

### 8.4 输出文件

结果默认写入：

```text
Annotation/fg_retrain/outputs/e0_vit_test_leave_one_out/<运行时间>/
```

- `metrics.json`：Micro/Macro Rank-1、Recall@1/5/10、mAP 和各类别结果；
- `per_query.jsonl`：每张 Query 的 Top-K 邻居、相似度和是否命中；
- `test_embeddings.pt`：8,041 张测试图的纯视觉特征及标签；
- `config_resolved.json`：本次运行的完整参数和绝对路径；
- `environment.json`：Python、PyTorch、CUDA、GPU 等复现信息。

`metrics.json` 中的 `gallery_size_per_query` 必须为 8,040；如果不是，则说明评测数据或协议发生了变化。

## 9. 第二阶段代码：E1 ViT + 纯视觉 LaFG Laux

E1 严格实现 `others/LaFG_Laux_execution_method.md` 定义的视觉损失：

```text
官方 train 图像
  -> 每批随机选择 64 类 × 每类严格 2 张
  -> ImageNet ViT Base/16
  -> L2 归一化 CLS 特征
  -> 两两平方欧氏距离（等价于 2 - 2 cosine）
  -> 每个 anchor 的唯一同类图为 positive
  -> 批内其余所有异类图为 negatives
  -> Laux
  -> 完整微调 ViT
```

本实验不加载 CLIP、LLM、类别文本或差异文本。测试流程与 E0 完全相同：官方 test 的每张图作为 Query，剩余 8,040 张作为 Gallery。

正式配置采用 SGD、初始学习率 `1e-5`、momentum `0.9`、weight decay `1e-4`、200 epochs，并每5轮将学习率乘以 `0.9`。论文没有明确给出可复现的 temperature，也没有解释 batch size 900 与“每类2张”的矛盾，因此本地配置明确记录：`temperature=0.1`、`64类×2张=128张/batch`。这些属于工程选择，不能写成论文原始超参数。

运行：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU编号> \
bash Annotation/fg_retrain/scripts/run_e1_vit_vision_laux.sh
```

若指定多个可见 GPU，代码使用 DataParallel，在全部图像特征汇总后统一计算全局 Laux；中断后会从 `checkpoint_last.pt` 自动续训。

默认结果写入：

```text
Annotation/fg_retrain/outputs/E1_VIT_VisionL/
```

其中包括训练历史、恢复检查点、最终纯视觉模型、测试特征、逐 Query 结果和总指标。
需要与历史结果隔离时，可传入`--run-name <name>`，新产物会写入
`Annotation/fg_retrain/outputs/E1_VIT_VisionL/<name>/`；同名目录已存在时会自动
添加数字后缀。

## 10. E0/E1共有的类别对困难度指标

E0和E1在常规全Gallery评测后，都会额外遍历196类构成的19,110个无向类别对。对有向关系`A -> B`：

```text
Query：每一张A类测试图
Pairwise Gallery：其他A类测试图 + 全部B类测试图
Positive：其他A类图
Negative：B类图
```

记录首张Positive出现的位置，并分别计算`Recall@1、@2、@3、@4、@5`。类别对还包含：

- `a_to_b`与`b_to_a`：两个方向各自的Recall@1～5；
- `balanced_recall_at_k`：两个方向Recall@K的等权平均；
- `micro_recall_at_k`：按两个类别Query数量加权的Recall@K；
- `difficulty_score_r1_to_r5`：`1-balanced Recall@K`在K=1～5上的平均值，越大越难；
- `mean_similarity_margin`：最佳同类相似度减最佳对方类别相似度，负值表示对方更近；
- `global_top1_confusion_count`：在完整196类Gallery中实际被对方夺走Top-1的次数；
- `prototype_cosine_similarity`：两个类别平均视觉原型的余弦相似度。

每次实验自动输出：

```text
pairwise_difficulty_summary.json
pairwise_difficulty_all_pairs.json
pairwise_difficulty_all_pairs.csv
pairwise_difficulty_per_class.json
top1_confusion_per_class.json
top1_confusion_all_directions.csv
```

其中全类别对JSON/CSV包含19,110对，逐类别困难度文件为每种车型列出
Top-20综合困难邻居。`top1_confusion_per_class.json`则对每类列出与其余
195类的完整Top-1错分次数/比例，并单独给出按实际错分次数排序的
`top5_error_classifications`；同等方向性统计也保存为方便排序的CSV。已有embedding
可以通过`fg_retrain.analyze_pairwise_difficulty`补算，无需重新训练。
