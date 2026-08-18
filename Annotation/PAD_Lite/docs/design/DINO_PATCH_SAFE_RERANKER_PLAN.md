# DINOv2 Patch-Safe Reranker V2：面向N分类的CLS Top-K与Patch重排方案

## 1. 文档定位

本文档是在初版 Patch-Safe Reranker 方案和 `PAD_Lite/docs/design/DINO_PATCH_SAFE_RERANKER_REVIEW.md` 的基础上形成的正式实施版本。

最重要的修订是：

> 不再把方法定义为两个类别之间的二次判断，而是把 CLS 定义为面向全部 N 个类别的高召回候选生成器，再由 Patch 特征对 CLS Top-K 类别进行细粒度重排。

当前 Russian 五折中每个 Novel Fold 只有两个类别，这是数据量不足下的开发性妥协，不代表最终 Annotation 系统是二分类。最终方法从接口、训练验证和部署逻辑上都必须支持任意类别数 N。

方法的最终目标为：

```text
N类Gallery
   ↓
CLS对全部N类快速检索
   ↓
保留Top-K类别候选
   ↓
歧义门控
   ↓
Patch局部特征只对Top-K重排
   ↓
输出最终Top-1类别
```

---

## 2. 对Review建议的采纳决定

| Review建议 | 决定 | 纳入方式 |
|---|---|---|
| P1/P2增加Patch-only消融 | 采纳 | 增加P1a/P1b和P2a/P2b |
| Weighted Pooling不是最终上限 | 采纳 | 第一版使用Weighted Pooling，P5再条件式研究Patch-to-Patch |
| 两类别rerank改为CLS Top-K → Patch rerank | **强制采纳** | V2核心架构，所有接口按N分类设计 |
| Prototype Gate + Query Margin Gate | 采纳 | 推广为N分类Top-K候选集合上的双门控 |
| 禁止直接使用0.7阈值 | 采纳 | 所有阈值只在Russian开发集内部选择 |
| 融合前进行分数标定 | 采纳 | CLS/Patch分别标定后再进行凸组合 |
| 冻结DINO与原CLS路径 | 采纳 | 第一阶段只训练Patch轻量分支 |
| Patch第一版不使用文本锚点 | 采纳 | Patch只使用ID和Triplet，CLS保留原文本增强能力 |
| 使用动态内容权重 | 采纳 | `Linear(hidden_dim, 1)`逐图生成权重，并屏蔽Padding |
| Error Complementarity分析 | 采纳 | 作为方法是否继续的必要诊断 |
| Patch热力图作为必要检查 | 采纳 | 每折必须生成并人工抽查 |
| Russian五折改为开发Benchmark | 采纳 | 不再宣称Russian五折是最终无偏测试 |
| 当前Gallery作为untouched final test | **修正后采纳** | 当前Gallery已经被检查和评测过，不能再称严格untouched；正式最终测试需新建并锁定数据 |

### 2.1 对Patch-to-Patch的保留意见

论文指出，直接使用Chamfer等局部匹配进行重排可能因背景Patch而降低整体性能。因此P5不是默认下一步，只在以下条件同时成立时启动：

1. Weighted Patch在困难类别上确有互补性；
2. 仍有较多`CLS错误 / Patch正确`潜力未被单向量聚合利用；
3. Patch热力图已确认主要关注车辆区域；
4. P2/P4提升受单向量压缩限制，而不是训练或标定失败。

P5也必须使用有效区域Mask，并限制在CLS Top-K内部，不能对全Gallery执行无约束的局部匹配。

---

## 3. 当前证据与问题定义

当前基线为：

- DINOv2-Small；
- Letterbox 336；
- 冻结视觉主干；
- B2：ID + Batch-Hard Triplet + Text Anchor；
- 每折2个Novel类别；
- Gallery Support原型检索。

### 3.1 当前五折Rank-1

| Fold | Novel类别 | Rank-1 |
|---:|---|---:|
| 1 | MT-LB / T-64 | 97.85% |
| 2 | T-72 / T-80 | 71.67% |
| 3 | BMD-2 / BTR-70 | 95.90% |
| 4 | BMP-1 / BMP-2 | 72.56% |
| 5 | BM-21 / BTR-80 | 94.14% |

### 3.2 CLS空间诊断

| Fold | 两个Support类别原型余弦相似度 | Query Top-1/Top-2中位间隔 |
|---:|---:|---:|
| 1 | -0.370 | 0.691 |
| 2 | **0.929** | **0.026** |
| 3 | -0.008 | 0.578 |
| 4 | **0.826** | **0.049** |
| 5 | 0.423 | 0.303 |

Fold 2/4 的类别在 CLS 空间高度重叠，符合“整体外观相似但局部结构不同”的细粒度问题。Patch Token可能提供炮塔、车体比例、履带、轮组和炮管连接等局部几何信息。

但是当前两类别五折只能证明局部特征能否改善一对相似类别，不能证明Top-K候选生成和N分类重排有效。因此V2必须增加独立的N-way开发实验。

---

## 4. N分类总体架构

设最终Gallery共有N个类别：

\[
\mathcal C=\{c_1,c_2,\ldots,c_N\}
\]

每个Gallery图像通过一次冻结DINOv2前向，同时得到：

- CLS Token：全局结构；
- Patch Tokens：局部结构。

```text
输入车辆裁剪
      │
      ▼
冻结DINOv2（只运行一次）
      │
      ├── CLS Token ── 原B2 CLS Head ── z_cls
      │
      └── Patch Tokens + Valid Mask
                     └── Weighted Pooling
                              └── Patch Head ── z_patch
```

Gallery离线缓存每个类别的双原型：

```text
P_cls(c)
P_patch(c)
```

在线或离线视频推理时分成两阶段。

### 4.1 阶段A：CLS对N类进行候选生成

Query与所有N个CLS类别原型计算相似度：

\[
S_{cls}(q,c)=\cos(z_{cls}(q),P_{cls}(c))
\]

按照分数排序，选出Top-K类别集合：

\[
\mathcal C_K(q)=\operatorname{TopK}_{c\in\mathcal C}S_{cls}(q,c)
\]

必须满足：

```text
K < N
```

才能体现候选筛选的意义。若`K=N`，仍可验证Patch重排，但不能证明CLS候选生成器降低了搜索范围。

### 4.2 阶段B：Patch只重排Top-K类别

Patch分数只在`C_K(q)`内部计算：

\[
S_{patch}(q,c)=\cos(z_{patch}(q),P_{patch}(c)),\quad c\in\mathcal C_K(q)
\]

候选集合外的类别不参与Patch重排，也不能重新进入最终结果。

因此真实类别能够被Patch救回的必要条件是：

\[
y_q\in\mathcal C_K(q)
\]

这意味着CLS的`Recall@K`是整个系统的理论上界之一，必须在N分类开发阶段记录。

### 4.3 最终输出

若门控不触发：

\[
\hat y=\arg\max_{c\in\mathcal C} S_{cls}(q,c)
\]

若门控触发：

\[
\hat y=\arg\max_{c\in\mathcal C_K(q)} S_{final}(q,c)
\]

Patch分支不会改变候选集合外类别的排序。

---

## 5. CLS与Patch特征设计

### 5.1 冻结CLS主路径

当前CLS提取方式：

```python
hidden = dino(pixel_values).last_hidden_state
cls_token = hidden[:, 0]
patch_tokens = hidden[:, 1:]
```

第一阶段冻结：

- DINOv2-Small主干；
- 原CLS Projection；
- 原CLS BNNeck；
- 原ID Head；
- 原文本Prompt；
- 当前B2权重。

这样可以确保新增Patch分支不会直接改变CLS特征。

### 5.2 Letterbox有效区域Mask

336×336输入、Patch Size 14，对应24×24，共576个Patch Token。

Letterbox填充区域不能参与Patch聚合。预处理必须返回：

```text
valid_patch_mask: [576]
```

其中：

- `True`：由车辆裁剪图缩放得到的有效区域；
- `False`：Letterbox Padding。

Mask必须由Letterbox几何参数准确生成，不能依赖像素颜色猜测。

### 5.3 P1：Masked Patch Average

\[
z_{avg}=\frac{\sum_iM_ip_i}{\sum_iM_i}
\]

Masked Average只作为弱基线，用来判断Patch中是否存在可用信号。论文已表明无条件Patch Average可能因背景噪声明显退化。

### 5.4 P2：Masked Weighted Patch Pooling

使用逐图、依赖内容的共享评分层：

\[
a_i=Wp_i+b
\]

```python
scores = patch_scorer(patch_tokens).squeeze(-1)
scores = scores.masked_fill(~valid_patch_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
pooled = (weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
```

然后通过独立Patch Head：

```text
Patch Projection → Patch BNNeck → L2 Normalize
```

得到512维`z_patch`。

不使用固定空间位置权重。车辆角度、位置和部件投影会变化，权重必须根据每张图的Token内容动态生成。

---

## 6. Patch分支训练

### 6.1 第一版可训练参数

只训练：

- `patch_scorer`；
- `patch_projection`；
- `patch_bnneck`；
- `patch_classifier`。

### 6.2 损失

\[
L_{patch}=L_{ID}+L_{Triplet}
\]

默认：

```text
ID Loss                 = 1.0
Batch-Hard Triplet Loss = 1.0
```

第一版Patch分支不加入Text Anchor。原CLS-B2已经保留训练期语义增强；Patch分支需要学习与全局语义互补的局部几何信息。过早施加类别级文本约束可能使Patch再次收缩到CLS相似方向。

### 6.3 推理要求

训练结束后Patch分支和CLS分支都只使用视觉特征：

- 不加载CLIP Text Encoder；
- 不加载Prompt；
- 不需要网络；
- 支持纯离线部署。

---

## 7. N分类双门控

### 7.1 Prototype Gate的N分类定义

令CLS Top-1类别为`c1`，Top-K中的其余类别为候选竞争者。计算：

\[
A(q)=\max_{c\in\mathcal C_K(q),c\neq c_1}
\cos(P_{cls}(c_1),P_{cls}(c))
\]

若：

\[
A(q)>\tau_{proto}
\]

说明CLS Top-1类别在候选集中存在结构高度相似的竞争类别。

可同时记录Top-K候选原型的最大两两相似度作为诊断，但默认门控应围绕当前Top-1及其竞争者，避免两个无关的低排名候选触发重排。

### 7.2 Query Margin Gate的N分类定义

\[
\Delta(q)=S_{cls}^{(1)}(q)-S_{cls}^{(2)}(q)
\]

若：

\[
\Delta(q)<\tau_{margin}
\]

说明具体Query的CLS判断不确定。

类别数较多时可附加记录Top-K Softmax熵，但第一版仍使用Top-1/Top-2 Margin，保持参数少且易解释。

### 7.3 双门控规则

```python
use_patch = (
    prototype_ambiguity > tau_proto
    and cls_top1_top2_margin < tau_margin
)
```

只有两个条件同时满足才进入Patch重排。

强样本直接返回原CLS结果，因此只要门控对Fold 1/3/5不触发，就可以保持其逐Query预测不变。

### 7.4 K的边界处理

```text
K_effective = min(K, N)
```

- N=2的旧五折中，`K_effective=2`，Top-K等于全部类别，只能验证重排，不能验证候选召回；
- N>2的开发/部署环境中，必须单独评估CLS Recall@K；
- 最终K优先考虑3或5，但必须由N-way开发验证决定。

---

## 8. 分数标定与融合

CLS和Patch都使用余弦相似度，不代表二者具有相同分布和尺度。融合前分别在开发验证集上得到标定分数：

\[
\tilde S_{cls}=Cal_{cls}(S_{cls})
\]

\[
\tilde S_{patch}=Cal_{patch}(S_{patch})
\]

第一版推荐Logistic Calibration。标定器只使用Russian开发验证样本拟合。

门控触发后的融合：

\[
S_{final}=(1-\lambda)\tilde S_{cls}+\lambda\tilde S_{patch},
\quad0\leq\lambda\leq1
\]

这样`λ`可以解释为局部信息在融合分数中的权重，而不是同时补偿两个分支的数值尺度。

`Cal_cls`、`Cal_patch`、`λ`、`τ_proto`、`τ_margin`和K都必须在开发验证阶段锁定。

---

## 9. 两套开发验证协议

当前数据无法只靠一个协议同时回答“未见类别泛化”和“N分类Top-K系统”两个问题，因此需要并行使用两套开发协议。

### 9.1 协议A：原Russian五折，两类Novel开发实验

用途：

- 与现有P0基线严格对比；
- 验证Patch是否改善Fold 2/4；
- 验证Fold 1/3/5保护机制；
- 验证未见类别局部特征迁移。

限制：

- 每折只有2个Novel类别；
- Recall@2恒为100%，不能证明Top-K候选生成有效；
- 不能用它选择N分类中的K=3或K=5。

### 9.2 协议B：Russian N-way开发检索

用途：

- 验证N分类完整链路；
- 测量CLS Recall@K；
- 选择K；
- 验证真实类别位于CLS第3/第5名时能否被Patch救回；
- 测量Top-K重排的收益和伤害。

建议包含两种互补设置。

#### B1：重复的多类Pseudo-Novel Episode

在Russian 10类开发数据内重复构建：

```text
5类：训练Patch Head
5类：Pseudo-Novel Support + Query
```

轮换训练类和Pseudo-Novel类，使每个类别多次作为未见类别出现。该设置可以验证类别泛化，并支持：

```text
K ∈ {2, 3, 5}
```

当K=5时等于对全部Pseudo-Novel类别重排，可作为“无候选裁剪”对照，不代表高效Top-K配置。

#### B2：8或10类系统级检索

按图像而非类别拆分Train/Validation/Support/Query，在更多类别同时存在时评估：

- CLS Rank-1；
- CLS Recall@3/5；
- Top-3/5 Patch rerank后的Rank-1；
- 运行时间和门控触发率。

B2能检验真实N-way系统行为，但类别在Patch训练中可见，因此不能替代B1的未见类别泛化结论。

### 9.3 参数选择原则

优先使用B1选择类别泛化相关参数，再用B2检查N-way工程行为：

1. B1选择Patch checkpoint和主要融合方向；
2. B1/B2共同选择最小可行K；
3. B2选择满足延迟约束的门控配置；
4. 所有参数锁定后才进入外部最终数据。

不再使用“8个Base中6类训练、2类验证”作为唯一内部协议，因为它仍然只能验证二分类，无法为N分类Top-K提供充分依据。

---

## 10. 实验矩阵

固定第一阶段条件：

- DINOv2-Small；
- Letterbox 336；
- 冻结DINO和原CLS-B2；
- Patch Head训练30 Epoch；
- 相同数据划分和种子；
- 最终主指标Rank-1。

### 10.1 表示消融

| 实验 | 表示与检索方式 | 目的 |
|---|---|---|
| P0 | CLS only | 当前基线 |
| P1a | Masked Patch Average only | Patch平均特征本身是否有效 |
| P1b | CLS + Masked Patch Average | 简单Global/Local是否互补 |
| P2a | Weighted Patch only | 学习权重是否优于平均 |
| P2b | CLS + Weighted Patch，全候选融合 | 融合上限和潜在伤害 |

Patch-only消融必须保留。否则即使融合提升，也无法判断Patch是否真正提供了CLS没有的局部信息。

### 10.2 Top-K与门控消融

| 实验 | 候选与门控 | 目的 |
|---|---|---|
| P3 | CLS Top-K + Weighted Patch + Prototype Gate | 验证类别歧义门控 |
| P4 | CLS Top-K + Prototype Gate + Query Margin Gate + 标定融合 | 推荐最终方法 |
| P5 | CLS Top-K + Masked Patch-to-Patch Matching | 仅在P2/P4证明Patch互补但提升受限时启动 |

### 10.3 K消融

在N-way协议中至少比较：

```text
K = 2, 3, 5
```

选择K时先看CLS Recall@K，再看Patch重排Rank-1和延迟。推荐选择满足目标Recall@K的最小K，而不是直接选择最大K。

---

## 11. 必要诊断

### 11.1 Candidate Generator诊断

必须报告：

- CLS Rank-1；
- CLS Recall@2/3/5；
- 各K下真实类别未进入候选集的数量；
- Patch重排的理论可救回样本数。

如果CLS Recall@K过低，Patch重排无法解决，应先改善候选生成器或提高K。

### 11.2 Error Complementarity

记录：

| CLS | Patch | 样本数 |
|---|---|---:|
| 正确 | 正确 | — |
| 正确 | 错误 | — |
| **错误** | **正确** | **核心纠错潜力** |
| 错误 | 错误 | — |

同时记录：

- CLS错误、融合后修正的数量 `rescued`；
- CLS正确、融合后变错的数量 `harmed`；
- `net_rescue = rescued - harmed`；
- CLS/Patch分数相关系数；
- 每个类别的纠错与伤害数量。

只有`rescued > harmed`且多种子稳定，Patch重排才具有实际意义。

### 11.3 门控诊断

记录：

- Prototype Gate触发率；
- Query Margin Gate触发率；
- 双门控最终触发率；
- 困难类别和容易类别的触发率；
- 不触发时预测是否与P0逐样本一致。

### 11.4 Patch热力图

Weighted Pooling每折必须生成24×24权重热力图。人工抽查至少包括：

- 正确样本；
- CLS错/Patch对样本；
- CLS对/Patch错样本；
- Tiny、Crowded、Clipped样本；
- 不同视角和背景。

合理关注区域：

- 炮塔；
- 车体；
- 履带和轮组；
- 炮管连接处；
- 主体轮廓及比例。

异常关注区域：

- Padding或Letterbox边缘；
- 草地、道路、天空；
- 水印、文字或数据源标记；
- 长期固定的绝对位置。

若异常区域长期占据主要权重，应停止性能调参，先修正Mask、数据增强或Patch Head。

---

## 12. 防止测试泄漏与实验定位

### 12.1 Russian五折的正式定位

Russian五折已经用于：

- CLIP/DINO比较；
- B0/B1/B2比较；
- 解冻深度；
- 输入分辨率；
- CenterCrop/Letterbox；
- Small/Base；
- Patch方案设计。

因此它应正式标记为：

```text
Development / Model-Selection Benchmark
```

它仍可用于方法对比，但不能再被描述为完全未触碰的最终测试集。

### 12.2 当前Annotation Gallery的定位修正

当前Gallery及其部分划分已经被检查并用于过模型测试，因此也不能严格称为“从未使用的Final Test”。

正式最终验收建议二选一：

1. 新收集或下载一批从未参与设计的Gallery/Query，建立`gallery_final_v2`，在全部参数锁定前不运行；
2. 如果无法获得新数据，则如实把当前Gallery称为`held-out evaluation`，并声明它曾用于早期系统检查，不能提供严格无偏的最终泛化估计。

禁止根据现有Gallery结果继续选择K、阈值、λ、标定器或Patch checkpoint。

### 12.3 严格参数流

```text
Russian开发数据
  ├── 训练Patch Head
  ├── N-way pseudo-novel验证
  ├── 选择K、阈值、λ和Calibration
  └── 多种子确认
        ↓
锁定代码、权重和全部参数
        ↓
新建且未使用的final_v2数据
        ↓
只运行一次最终评测
```

---

## 13. 验收标准

### 13.1 原两类五折开发标准

1. Fold 1/3/5各自Rank-1下降不超过0.5个百分点；
2. Fold 2/4平均Rank-1至少提高3个百分点；
3. 五折平均Rank-1高于P0；
4. 至少3个随机种子方向一致。

### 13.2 N分类Top-K标准

1. 在N-way开发协议中报告CLS Recall@K；
2. 选择满足候选召回目标的最小K；
3. Patch rerank后的Rank-1高于同一CLS候选基线；
4. `rescued > harmed`；
5. 非门控样本与P0结果逐样本一致；
6. 多类别中不能只依靠一两个类别产生全部增益；
7. 推理延迟和显存满足离线Annotation流程要求。

候选召回目标不应预先拍脑袋固定。可先以CLS Recall@K达到98%作为工程候选线，再根据数据规模、错误成本和延迟在开发集上确认。

### 13.3 Patch有效性标准

1. 保留Patch-only结果；
2. Weighted Patch整体优于Masked Average；
3. 存在稳定的`CLS错误 / Patch正确`样本；
4. CLS与Patch不是近乎完全重复的信息；
5. Patch权重主要集中于车辆主体；
6. Padding区域聚合权重严格为零。

---

## 14. 代码改造建议

建议新增独立实现，不覆盖当前B2基线：

```text
PAD_Lite/
├── dino_patch_models.py
├── dino_patch_engine.py
├── dino_patch_cli.py
├── patch_calibration.py
├── patch_nway_eval.py
├── patch_visualize.py
└── configs/
    └── dino_patch_safe_letterbox_336.json
```

职责：

- `PAD_Lite/src/dino_patch_models.py`：Mask、Average/Weighted Pooling、Patch Head；
- `PAD_Lite/src/dino_patch_engine.py`：Patch训练、双特征提取、五折评测；
- `PAD_Lite/src/dino_patch_cli.py`：P1a～P5实验入口；
- `PAD_Lite/src/patch_calibration.py`：分支标定、λ和双门控参数；
- `PAD_Lite/src/patch_nway_eval.py`：N-way候选生成、Recall@K、Top-K rerank；
- `PAD_Lite/src/patch_visualize.py`：24×24权重热力图和错误互补样本导出；
- 配置文件：保存K、阈值、融合和输出路径。

输出必须独立保存：

```text
PAD_Lite/outputs/dino_patch_safe_letterbox_336/
├── p1a_patch_average_only/
├── p1b_cls_patch_average/
├── p2a_weighted_patch_only/
├── p2b_cls_weighted_patch/
├── p3_topk_prototype_gate/
├── p4_topk_dual_gate/
└── p5_local_matching/
```

不得覆盖：

```text
PAD_Lite/outputs/dino_letterbox_text_anchor_336/
```

---

## 15. 最终Annotation推理流程

对于任意N类Gallery：

```text
YOLO候选框
    ↓
车辆裁剪 + Letterbox 336 + Valid Patch Mask
    ↓
DINOv2一次前向
    ├── CLS特征
    └── Patch Tokens
    ↓
CLS与全部N个类别原型计算相似度
    ↓
取得Top-K类别候选
    ↓
Prototype Gate + Query Margin Gate
    ├── 不触发：直接输出CLS Top-1
    └── 触发：计算/读取Patch特征
              ↓
              标定CLS与Patch分数
              ↓
              只在Top-K类别内融合重排
              ↓
              输出最终Top-1
```

Gallery提前离线缓存：

```text
gallery_cls_features.pt
gallery_patch_features.pt
gallery_cls_prototypes.pt
gallery_patch_prototypes.pt
reranker_calibration.json
```

因为CLS与Patch来自同一次DINO前向，主要新增成本是Patch聚合、Top-K类别相似度和门控，而不是第二次运行视觉主干。

---

## 16. 实施顺序

1. 扩展Letterbox变换，准确返回576维有效Patch Mask；
2. 扩展DINO接口，一次前向返回CLS和Patch Tokens；
3. 加载并冻结Small Letterbox 336 B2五折权重；
4. 实现P1a/P1b，完成Patch Average only与融合消融；
5. 实现动态Masked Weighted Pooling；
6. 实现P2a/P2b并生成Patch热力图；
7. 做Error Complementarity分析，确认Patch具有真实纠错潜力；
8. 建立Russian 5-way pseudo-novel和8/10-way系统开发评测；
9. 统计CLS Recall@2/3/5，选择最小可行K；
10. 在开发数据上拟合CLS/Patch标定器；
11. 实现N分类Prototype Gate和Query Margin Gate；
12. 完成P3/P4五折及N-way实验；
13. 使用至少3个随机种子复验；
14. 若Patch互补明显但Weighted Pooling仍受限，再决定是否启动P5；
15. 锁定代码、权重、K、阈值、λ和标定器；
16. 新建并封存`gallery_final_v2`；
17. 只在参数全部锁定后执行最终N分类评测。

---

## 17. 最终建议

推荐正式实施：

\[
\boxed{
\text{CLS N-way Candidate Generator}
+
\text{Conditional Patch Top-K Reranker}
}
\]

它不是二分类补丁，而是标准的两阶段N分类检索系统：

- CLS负责在全部类别中保持高召回；
- Patch负责在少量相似候选中识别局部几何差异；
- 双门控保护容易样本；
- Calibration保证Global/Local分数可解释地融合；
- Patch-only、互补错误和热力图证明增益来源；
- Russian承担开发与选择，新的封存数据承担最终评测。

当前两类五折仍有价值，但只用于验证“Patch能否区分相似未见类别”和“强折能否保持”，不能独立支撑最终N分类结论。

## 18. 参考文件

- Review：`PAD_Lite/docs/design/DINO_PATCH_SAFE_RERANKER_REVIEW.md`
- 论文：`../../papers/Efficient_Fine-grained_Image_Retrieval_CVPRW_2026.pdf`
- CLS Letterbox分辨率实验：`PAD_Lite/docs/results/DINO_LETTERBOX_RESOLUTION_SWEEP_RESULTS.md`
- DINOv2-Base/Small对照：`PAD_Lite/docs/results/DINO_BASE_LETTERBOX_TEXT_ANCHOR_336_RESULTS.md`
