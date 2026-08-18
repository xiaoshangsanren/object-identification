# DINOv2 Patch-Safe Reranker 改进建议

## 1. 总体评价

当前方案整体方向合理，可以实施。

其核心思路是：

- 保留当前已经表现较强的 CLS 全局检索路径；
- DINOv2 主干继续冻结，避免破坏预训练视觉表示；
- Patch 分支只承担局部细粒度信息补充；
- 对 CLS 无法可靠区分的候选类别启用 Patch 重排；
- 所有门控阈值、融合权重和模型选择均只在 Base 内部验证，避免 Novel Query 泄漏；
- 最终部署仍保持纯视觉、离线 Gallery 检索。

推荐继续保留 **Patch-Safe Reranker** 这一总体设计，但在正式实现前对实验结构和门控逻辑做以下调整。

---

## 2. 建议一：P1 / P2 必须增加 Patch-only 消融

当前方案：

```text
P0：原 CLS-B2
P1：CLS + Masked Patch Average
P2：CLS + Masked Weighted Patch
P3：CLS主路径 + Weighted Patch + 类别级门控
P4：CLS主路径 + Weighted Patch + 类别级 + Query级门控
```

存在一个问题：

如果 P1 或 P2 提升，无法判断提升究竟来自：

- Patch 表示本身包含额外信息；
- 还是融合策略产生了偶然收益。

因此建议改为：

```text
P0
CLS baseline

P1a
Patch Average ONLY

P1b
CLS + Patch Average

P2a
Weighted Patch ONLY

P2b
CLS + Weighted Patch

P3
CLS Top-K candidate generation
+ Weighted Patch reranking
+ Prototype gate

P4
CLS Top-K
+ Prototype gate
+ Query margin gate
+ Calibrated score fusion
```

这样能够明确回答三个问题：

1. Patch Token 自身是否包含 CLS 没有充分利用的细粒度信息；
2. Weighted Pooling 是否优于简单 Average Pooling；
3. Global 与 Local 是否真正具有互补性。

理想结果例如：

| 方法 | Fold 2 | Fold 4 |
|---|---:|---:|
| CLS | 71.7 | 72.6 |
| Patch Avg only | 73.0 | 74.0 |
| Weighted Patch only | 77.0 | 78.0 |
| CLS + Weighted Patch | **79.0** | **80.0** |

如果出现这种趋势，则可以较有力地证明：

```text
DINO Patch Tokens 中确实保留了额外的局部判别信息，
而 Weighted Pooling 能够更有效地利用这些信息。
```

---

## 3. 建议二：Weighted Pooling 作为第一版即可，但不要把它当作最终上限

当前方案将 576 个 Patch Token：

```text
24 × 24 = 576 Patch Tokens
```

通过：

```text
patch_scorer
→ softmax weights
→ weighted sum
```

最终仍压缩为一个 Patch embedding：

```text
576 个局部 Token
        ↓
Weighted Pooling
        ↓
1 个 512-D Local Embedding
```

这一设计适合作为第一阶段验证，因为：

- 实现简单；
- 参数少；
- DINO backbone 可以完全冻结；
- 推理开销很小；
- 容易与当前 CLS 分支对照。

但它仍然存在一个潜在问题：

> 多个局部判别信息最终再次被压缩成单一向量。

因此，如果 P2 已经比 CLS 有提升，但 Fold 2 / 4 仍然提升有限，不应该立即得出“Patch Token 没有用”的结论。

下一阶段建议增加：

## P5：Patch-to-Patch Local Matching

对于 Query：

\[
P_q=\{q_1,q_2,\ldots,q_N\}
\]

对于 Gallery：

\[
P_g=\{g_1,g_2,\ldots,g_M\}
\]

直接构造局部相似度矩阵：

\[
S_{ij}=\cos(q_i,g_j)
\]

然后计算局部最佳对应关系，例如：

```text
Query 炮塔 Patch
      ↕
Gallery 炮塔 Patch

Query 负重轮 Patch
      ↕
Gallery 负重轮 Patch
```

这种方法可以避免要求 Query 与 Gallery 的局部结构严格出现在相同空间位置，也避免所有局部信息被再次平均到一个向量中。

因此推荐阶段关系：

```text
Weighted Pooling
      ↓
验证 Patch 是否有互补信息
      ↓
如果有效但提升有限
      ↓
Patch-to-Patch Matching
```

---

## 4. 建议三：最终系统应采用 Top-K 候选重排，而不是只考虑两个类别

当前五折实验每个 Novel Fold 只有两个类别，因此：

```text
Top-1
vs
Top-2
```

就基本等价于所有 Novel 类。

但最终 Annotation Gallery 是多类别环境，因此实际部署应改为：

```text
Query
  ↓
CLS 全局检索
  ↓
获得 Top-K 候选
  ↓
判断是否歧义
  ↓
必要时使用 Patch 分支进行 Top-K 重排
  ↓
最终结果
```

推荐候选：

```text
K = 3 或 5
```

具体值必须在 Base 内部验证选择。

原因是，如果真实类别在 CLS 排名中位于第三名：

```text
1. T-80
2. T-64
3. T-72   ← 正确类别
```

只对 Top-1 / Top-2 重排时，Patch 永远无法把 T-72 救回来。

而 Top-K reranking 可以把 CLS 看作：

```text
高召回 Candidate Generator
```

Patch 分支则负责：

```text
细粒度 Candidate Reranker
```

这比“两个类别之间的二次判断”更符合真实 Gallery 系统。

---

## 5. 建议四：保留 Prototype Gate + Query Margin Gate 双门控

当前方案定义：

### 类别原型歧义

\[
A(c_i,c_j)=\cos(P_{cls}(c_i),P_{cls}(c_j))
\]

若：

\[
A(c_i,c_j)>\tau_{proto}
\]

说明两个类别在 CLS 空间中本身比较接近。

### Query 置信间隔

\[
\Delta(q)
=
S_{cls}^{top1}(q)-S_{cls}^{top2}(q)
\]

若：

\[
\Delta(q)<\tau_{margin}
\]

说明具体 Query 的 CLS 判断也不确定。

推荐继续使用：

```text
Prototype 高歧义
AND
Query Margin 小
        ↓
启用 Patch Reranker
```

而不是只使用其中一个条件。

原因是：

```text
Prototype Gate
```

回答：

> 这两个类别本身是否容易混淆？

而：

```text
Query Margin Gate
```

回答：

> 这一张具体 Query 是否真的存在判断困难？

两者结合可以显著减少 Patch 分支对 Fold 1 / 3 / 5 等容易样本的无意义干扰。

---

## 6. 建议五：不要直接使用 0.7 作为最终 Prototype 阈值

当前 Fold 的诊断结果已经表明：

```text
Fold 1：Prototype cosine = -0.370
Fold 2：Prototype cosine = 0.929
Fold 3：Prototype cosine = -0.008
Fold 4：Prototype cosine = 0.826
Fold 5：Prototype cosine = 0.423
```

因此：

```text
τ_proto = 0.7
```

会刚好选择 Fold 2 / 4。

但由于我们已经知道 Fold 2 / 4 是困难 Fold，因此不能直接根据这些最终 Novel 结果确定：

```text
τ_proto = 0.7
```

否则会形成开发集泄漏。

正确方法应为：

## Base 内 pseudo-Novel Episodic Validation

对于某个外层 Fold：

```text
8 个 Base Classes
```

继续内部拆分：

```text
6 类 → Patch Head Training
2 类 → Pseudo-Novel Validation
```

Pseudo-Novel 两类再划分：

```text
Support
+
Query
```

在这个内部任务上选择：

- `tau_proto`
- `tau_margin`
- `lambda`
- `Top-K`
- 分数标定参数
- Patch Head checkpoint

然后全部固定，再进入真正的外层 Novel Fold。

这样选择出的参数才是在模拟：

```text
Base Classes
    ↓
完全未见类别
    ↓
Few-shot Gallery Retrieval
```

---

## 7. 建议六：融合前必须进行分数标定

CLS 与 Patch 都使用 cosine similarity，并不意味着它们的数值尺度相同。

例如可能出现：

```text
CLS:
positive ≈ 0.80
negative ≈ 0.60

Patch:
positive ≈ 0.45
negative ≈ 0.15
```

如果直接：

\[
S_{final}=S_{cls}+\lambda S_{patch}
\]

则 `lambda` 同时受到：

- Local / Global 信息权重；
- 两个分支原始数值尺度；

两个因素影响，不容易解释。

因此建议先得到：

\[
\tilde S_{cls}
\]

\[
\tilde S_{patch}
\]

推荐在 Base pseudo-Novel validation 上进行 Logistic Calibration。

然后再：

\[
S_{final}
=
(1-\lambda)\tilde S_{cls}
+
\lambda\tilde S_{patch}
\]

其中：

\[
0\le\lambda\le1
\]

这样：

```text
lambda = 0.2
```

就可以明确解释为：

```text
80% Global
+
20% Local
```

参数更直观，也更容易分析。

---

## 8. 建议七：继续冻结 DINO 与 CLS 主路径

这一点不建议修改。

根据现有实验：

```text
DINOv2-Small
Frozen Backbone
+ Head
+ Text Anchor
```

已经在 Fold 1 / 3 / 5 上取得很强性能。

而大规模解冻 DINO 后：

- Fold 2 容易持续退化；
- 五折平均通常下降；
- Novel 泛化变得不稳定。

因此 Patch 阶段应保持：

### 冻结

- DINOv2-Small backbone；
- CLS Projection；
- CLS BNNeck；
- 当前 CLS-B2 权重；
- 原 Text Prompt；
- 原 ID Head。

### 只训练

- Patch Scorer；
- Patch Projection；
- Patch BNNeck；
- Patch ID Classifier。

这样实验变量能够被严格控制为：

```text
“Patch 分支是否带来新的局部信息”
```

而不是：

```text
“DINO 又被重新训练了一遍”
```

---

## 9. 建议八：Patch 分支第一版不要加入 Text Anchor

这一点当前方案是合理的。

CLS 分支当前的 Text Anchor 已经承担：

```text
全局视觉表示
        ↓
训练期语义约束
```

Patch 分支主要希望补充：

- 炮塔几何；
- 车体比例；
- 轮组结构；
- 履带；
- 炮管连接区域；
- 其他局部差异。

这些局部视觉信息并不一定能由当前 class-level Text Prompt 准确描述。

因此第一版推荐：

\[
L_{patch}
=
L_{ID}
+
L_{Triplet}
\]

暂时不加入：

\[
L_{TextAnchor}
\]

避免 Patch 分支又被迫向与 CLS 相似的全局语义方向收缩，降低 Global / Local 的互补性。

如果 Patch 分支验证成功，后续再单独研究：

```text
Local semantic supervision
```

是否有额外价值。

---

## 10. 建议九：Weighted Pooling 使用动态内容权重，不要固定位置权重

推荐：

\[
a_i=f(p_i)
\]

例如：

```python
patch_scorer = Linear(hidden_dim, 1)
```

每张图片根据其 Patch 内容动态生成：

```text
a1 ... aN
```

再：

```python
scores = scores.masked_fill(~valid_mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

最后：

\[
z_{patch}
=
\sum_i w_i p_i
\]

不建议使用：

```text
第 35 个 Patch 永远重要
第 40 个 Patch 永远不重要
```

这种固定空间位置权重。

原因包括：

- 不同车辆视角不同；
- 目标在 Letterbox 后的位置可能略有变化；
- 炮塔和车体的空间位置随视角变化；
- Query / Gallery 可能存在域变化。

动态内容权重更适合当前任务。

---

## 11. 建议十：增加 Error Complementarity 分析

仅看：

```text
CLS Rank-1
Patch Rank-1
Fusion Rank-1
```

还不够。

真正需要确认的是 Patch 是否在“CLS 错误的样本”上提供额外信息。

建议记录：

| CLS | Patch | 数量 |
|---|---|---:|
| 正确 | 正确 | — |
| 正确 | 错误 | — |
| **错误** | **正确** | **核心指标** |
| 错误 | 错误 | — |

最重要的是：

```text
CLS 错 / Patch 对
```

的比例。

例如 Fold 2：

```text
CLS 错误：66 个
其中 Patch 单独正确：31 个
```

说明 Patch 有很强的纠错潜力。

反过来：

```text
CLS 错误：66 个
Patch 也错误：64 个
```

说明两条分支基本学习到了相同的信息，复杂融合意义有限。

同时建议记录：

\[
corr(S_{cls},S_{patch})
\]

如果相关性长期接近：

\[
0.95\sim1.0
\]

说明 Patch 与 CLS 信息高度重复。

理想状态应该是：

```text
Global 与 Local 整体相关，
但在困难样本上具有明显互补性。
```

---

## 12. 建议十一：Patch 热力图作为必要诊断，而不是可选可视化

Weighted Pooling 训练后，应把：

\[
w_i
\]

还原到：

```text
24 × 24
```

Patch 网格。

重点检查：

### 正常情况

权重集中于：

- 炮塔；
- 车体；
- 履带；
- 轮组；
- 炮管连接区域；
- 车辆主体边界。

### 异常情况

权重集中于：

- Letterbox 边缘；
- 固定 Padding 区域；
- 草地；
- 道路；
- 天空；
- 水印；
- 数据源标记。

如果长期关注后一类区域，则说明 Patch Head 正在重新学习错误关联，需要立即调整：

- Mask；
- 数据增强；
- Head 容量；
- 训练策略。

---

## 13. 建议十二：将当前 Russian 五折重新定义为开发 / 模型选择 Benchmark

目前已经多次根据五折结果进行了：

- DINO / CLIP 比较；
- Small / Base 比较；
- 输入分辨率选择；
- CenterCrop / Letterbox 比较；
- Text Anchor 选择；
- Patch 方法设计。

因此严格意义上，这五折已经不再是完全 untouched test set。

建议实验定义改成：

```text
Russian 五折
=
Development / Model-Selection Benchmark
```

用于：

- 方法设计；
- 消融；
- 超参数；
- Patch 门控；
- 融合权重；
- 模型选择。

然后：

```text
Final Annotation Gallery
=
真正 Untouched Final Test
```

只有在所有方法和超参数完全锁定后，才第一次评测。

这样能够保持最终实验结论的可信性。

---

## 14. 推荐最终实验流程

```text
当前最佳 CLS-B2
Frozen DINOv2-Small
Letterbox 336
        ↓
固定并冻结
        ↓
P0：CLS Baseline
        ↓
P1a：Patch Average Only
        ↓
P1b：CLS + Patch Average
        ↓
P2a：Weighted Patch Only
        ↓
P2b：CLS + Weighted Patch
        ↓
检查：
Patch是否提供互补信息？
Weighted是否优于Average？
        ↓
Base内部 Pseudo-Novel Validation
        ↓
选择：
Top-K
τ_proto
τ_margin
λ
Calibration
        ↓
P3：
CLS Top-K
+
Prototype Gate
+
Weighted Patch Reranking
        ↓
P4：
CLS Top-K
+
Prototype Gate
+
Query Margin Gate
+
Calibrated Global/Local Fusion
        ↓
3个随机种子验证
        ↓
若提升有限但Patch有互补性
        ↓
P5：Patch-to-Patch Matching
        ↓
锁死最终模型和所有参数
        ↓
首次进入 Final Annotation Gallery
```

---

## 15. 推荐验收标准

建议继续使用当前方案中的基本标准，并增加 Patch-specific 标准。

### 主要性能标准

1. Fold 1 / 3 / 5 各自 Rank-1 下降不超过 0.5 pp；
2. Fold 2 / 4 平均 Rank-1 至少提高 3 pp；
3. 五折平均 Rank-1 高于当前 CLS-B2；
4. 至少 3 个随机种子方向一致。

### Patch 有效性标准

5. `Patch-only` 在 Fold 2 / 4 至少一折优于随机 / 无意义基线；
6. Weighted Patch 优于 Patch Average；
7. 存在明显 `CLS错误 / Patch正确` 样本；
8. Patch score 与 CLS score 不是完全高度重复；
9. Patch 权重主要集中于车辆主体，而不是背景 / Padding。

### 泛化与部署标准

10. DINO backbone 保持冻结；
11. 推理阶段不需要 Text Encoder；
12. 所有门控、融合、Calibration 参数均未使用外层 Novel Query 标签；
13. 最终 Gallery 在所有方法和超参数锁定后才第一次评测。

---

## 16. 最终建议

当前 Patch-Safe Reranker 的总体技术方向值得继续实施。

最重要的设计原则应保持为：

```text
强 Global CLS 主路径
        +
轻量 Local Patch 补充分支
        +
只对歧义候选进行条件式重排
```

不建议：

```text
Patch 完全替换 CLS
```

也不建议：

```text
重新大幅微调 DINO Backbone
```

推荐最终目标是：

\[
\boxed{
\text{Global Stable Retrieval}
+
\text{Conditional Local Fine-Grained Reranking}
}
\]

即：

> 在保持 Fold 1 / 3 / 5 当前强全局检索能力的同时，只对 CLS 空间中高度重叠、Query 置信度较低的候选类别调用 Patch 局部几何特征进行二阶段重排。

正式实施前最重要的三项修改为：

1. **P1 / P2 增加 Patch-only 消融；**
2. **从双类别重排扩展为 CLS Top-K → Patch Reranking；**
3. **所有门控和融合参数使用 Base 内 pseudo-Novel episodic validation 选择。**

完成这三项后，整个实验逻辑会更加完整，也更容易证明 Patch 分支的真实增益来源。
