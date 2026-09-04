# LaFG 中 \(L_{\text{aux}}\) 阶段的执行方法

> 依据：LaFG（Language-driven Fine-grained Retrieval, CVPR 2026）论文第 3.4 节与实验设置。  
> 本文档只讨论 **\(L_{\text{aux}}\)**，不展开 \(L_{\text{ali}}\)、LLM、属性库或 Linguistic Prototype。

---

## 1. \(L_{\text{aux}}\) 是什么

LaFG 中的 \(L_{\text{aux}}\) 是一个纯视觉的监督式对比学习损失，用来建立最终的图像检索空间。

它只使用：

- 图像；
- 图像类别标签；
- Retrieval Model \(F\)。

不使用：

- LLM；
- CLIP 文本编码器；
- 属性库；
- Linguistic Prototype；
- 文本监督。

目标非常直接：

\[
\boxed{
\text{同类别图像拉近，不同类别图像推远}
}
\]

---

# 2. Batch 的采样方式

论文第 3.4 节写的是：

> During training, we sample \(N\) categories with two instances per class, resulting in a batch size of \(K=2N\).

因此一个 batch 的基本结构是：

\[
N \text{ 个类别} \times 2 \text{ 张图片/类}
\]

例如：

```text
Class A: A1, A2
Class B: B1, B2
Class C: C1, C2
Class D: D1, D2
```

如果：

\[
N=4
\]

则：

\[
K=8
\]

整个 batch：

```text
[A1, A2, B1, B2, C1, C2, D1, D2]
```

这本质上是一个：

\[
\boxed{P\times K\text{ 的 PK Sampling}}
\]

其中 LaFG 的方法描述对应：

\[
K=2
\]

即每类固定 2 张。

---

# 3. 每张图片都经过同一个 ViT Retrieval Model

对于 batch 中每张图片：

\[
X_i
\]

经过视觉检索模型：

\[
z_i=F(X_i)
\]

得到视觉 embedding。

假设 embedding dimension 为 \(d\)，则：

\[
Z=
[z_1,z_2,\ldots,z_K]
\in \mathbb R^{K\times d}
\]

这里的 \(z_i\) 就是最终 image-image retrieval 使用的视觉表示。

---

# 4. 先对 embedding 做 L2 Normalization

LaFG 的距离定义为：

\[
D(z_i,z_j)
=
\left\|
\frac{z_i}{\|z_i\|_2}
-
\frac{z_j}{\|z_j\|_2}
\right\|_2^2
\]

因此先定义：

\[
\hat z_i=
\frac{z_i}{\|z_i\|_2}
\]

得到单位向量。

在代码中相当于：

```python
z = F(images)                  # [B, D]
z = F.normalize(z, dim=1)      # [B, D]
```

---

# 5. 计算整个 Batch 内的两两距离

对于归一化后的特征：

\[
\hat z_i,\hat z_j
\]

LaFG 使用：

\[
D(z_i,z_j)
=
\|\hat z_i-\hat z_j\|_2^2
\]

论文进一步推导：

\[
D(z_i,z_j)
=
2-2
\frac{
\langle z_i,z_j\rangle
}{
\|z_i\|_2\|z_j\|_2
}
\]

也就是：

\[
\boxed{
D(z_i,z_j)=2-2\cos(z_i,z_j)
}
\]

因此：

\[
D\downarrow
\Longleftrightarrow
\cos(z_i,z_j)\uparrow
\]

实际上可以直接构造 cosine similarity matrix：

\[
S=ZZ^\top
\]

如果 batch size 是 \(K\)，则：

\[
S\in\mathbb R^{K\times K}
\]

例如：

```text
         A1    A2    B1    B2    C1    C2
A1        1    .84   .31   .28   .22   .19
A2       .84    1    .29   .25   .20   .18
B1       .31   .29    1    .88   .21   .25
B2       .28   .25   .88    1    .22   .20
C1       .22   .20   .21   .22    1    .86
C2       .19   .18   .25   .20   .86    1
```

---

# 6. Positive 是怎么确定的

LaFG 每类只采两张，因此对于任意 anchor，**同类别的另一张图片就是唯一 positive**。

例如：

```text
A1 的 positive = A2
A2 的 positive = A1

B1 的 positive = B2
B2 的 positive = B1
```

对于 anchor \(z_i\)，论文记其同类别 positive 为：

\[
z_j
\]

所以：

\[
(z_i,z_j)
\]

是一个正样本对。

---

# 7. Negative 是怎么确定的

除了：

- anchor 自己；
- 唯一同类 positive；

之外，batch 中所有其他图片都是 negative。

例如对：

\[
A_1
\]

而言：

Positive：

\[
A_2
\]

Negatives：

\[
B_1,B_2,C_1,C_2,D_1,D_2,\ldots
\]

因此如果 batch size：

\[
K=2N
\]

每个 anchor 有：

\[
1\text{ 个 positive}
\]

以及：

\[
K-2
\]

个 negatives。

LaFG 没有在 \(L_{\text{aux}}\) 里额外规定：

- hardest negative；
- semi-hard negative；
- Top-k negative；
- 人工 triplet。

**batch 中所有异类样本都会作为 negative。**

---

# 8. 单个 Anchor 的 \(L_{\text{aux}}\)

论文定义：

\[
\mathcal L_{\text{aux}}(z_i)
=
-\log
\frac{
\exp(-D(z_i,z_j)/\tau)
}{
\sum_{k=1,k\neq i}^{K}
\exp(-D(z_i,z_k)/\tau)
}
\]

其中：

- \(z_i\)：anchor；
- \(z_j\)：唯一同类 positive；
- \(z_k\)：batch 中除 anchor 自身之外的全部图片；
- \(\tau\)：temperature。

注意：

\[
\sum_{k\neq i}
\]

中 **包含 positive \(z_j\)**。

所以这是一个：

\[
\boxed{
\text{“在 batch 内所有其他图片中，把唯一同类图片选出来”}
}
\]

的 softmax 对比目标。

---

# 9. 用 Cosine 形式理解更直观

由于：

\[
D=2-2\cos
\]

则：

\[
-D=-2+2\cos
\]

Softmax 中常数 \(-2\) 会抵消，因此本质上可以理解成：

\[
\mathcal L_{\text{aux}}(z_i)
=
-\log
\frac{
\exp(\cos(z_i,z_j)/\tau')
}{
\sum_{k\neq i}
\exp(\cos(z_i,z_k)/\tau')
}
\]

其中 \(\tau'\) 吸收了系数 2。

因此本质就是：

\[
\boxed{
\text{让 positive cosine 最大，
让所有 negative cosine 相对变小}
}
\]

---

# 10. 一个具体例子

假设：

```text
Batch:
A1 A2 B1 B2 C1 C2
```

对于 anchor：

\[
A_1
\]

假设 cosine similarity：

```text
A1-A2 = 0.80  ← positive
A1-B1 = 0.60
A1-B2 = 0.55
A1-C1 = 0.10
A1-C2 = 0.12
```

则：

\[
A_2
\]

必须在所有候选中获得最大的 softmax probability。

训练会：

- 提高 \(A_1\leftrightarrow A_2\) 的相似度；
- 降低 \(A_1\leftrightarrow B_1/B_2\)；
- 降低 \(A_1\leftrightarrow C_1/C_2\)。

但由于 B 与 A 当前更接近：

\[
0.60,0.55
\]

而 C 较远：

\[
0.10,0.12
\]

softmax 中 B 类样本的贡献会明显更大。

因此 LaFG 虽然没有显式 hard-negative mining，但会自然形成：

\[
\boxed{
\text{implicit hard-negative weighting}
}
\]

即当前最容易混淆的异类样本会产生更强梯度。

---

# 11. 每张图都要轮流当 Anchor

整个 batch 中：

```text
A1
A2
B1
B2
C1
C2
...
```

每张图片都计算一次：

\[
L_{\text{aux}}(z_i)
\]

最终 batch loss 一般理解为：

\[
L_{\text{aux}}
=
\frac{1}{K}
\sum_{i=1}^{K}
L_{\text{aux}}(z_i)
\]

论文公式展示的是单 anchor 形式，batch 训练自然需要对所有 anchor 聚合。

---

# 12. 完整数据流

```text
Step 1
Sampler
│
├── 随机选择 N 个类别
│
└── 每类随机选择 2 张图片
       ↓

Step 2
Batch
[A1,A2,B1,B2,...]
       ↓

Step 3
ViT Retrieval Model F
       ↓

Step 4
Embeddings
[zA1,zA2,zB1,zB2,...]
       ↓

Step 5
L2 Normalize
       ↓

Step 6
Batch-wise Pairwise Distance / Similarity Matrix
       ↓

Step 7
对于每个 Anchor：
    同类另一张 = Positive
    所有异类图片 = Negatives
       ↓

Step 8
Softmax Contrastive Loss
       ↓

Step 9
对所有 Anchor 求平均
       ↓

Step 10
Laux
       ↓

Step 11
Backpropagation
       ↓

Step 12
Update ViT Retrieval Model F
```

---

# 13. PyTorch 风格伪代码

```python
# --------------------------------------------------
# 1. PK sampler:
#    N classes × 2 images per class
# --------------------------------------------------

images, labels = batch
# images: [B, C, H, W]
# labels: [B]
# B = 2N


# --------------------------------------------------
# 2. Visual embedding
# --------------------------------------------------

z = model(images)               # [B, D]


# --------------------------------------------------
# 3. L2 normalization
# --------------------------------------------------

z = F.normalize(z, p=2, dim=1)  # [B, D]


# --------------------------------------------------
# 4. Cosine similarity matrix
# --------------------------------------------------

sim = z @ z.T                   # [B, B]


# --------------------------------------------------
# 5. Remove self-comparison
# --------------------------------------------------

self_mask = torch.eye(B, dtype=torch.bool, device=z.device)


# --------------------------------------------------
# 6. Construct positive mask
# --------------------------------------------------

same_class = labels[:, None] == labels[None, :]

positive_mask = same_class & (~self_mask)

# Under LaFG sampling:
# every row should contain exactly one True positive.


# --------------------------------------------------
# 7. All non-self samples participate in denominator
# --------------------------------------------------

logits = sim / temperature
logits = logits.masked_fill(self_mask, float("-inf"))


# --------------------------------------------------
# 8. Positive logit
# --------------------------------------------------

positive_logits = logits[positive_mask]
# exactly B positive logits total


# --------------------------------------------------
# 9. log denominator
# --------------------------------------------------

log_den = torch.logsumexp(logits, dim=1)


# --------------------------------------------------
# 10. per-anchor loss
# --------------------------------------------------

pos_per_anchor = (logits * positive_mask).sum(dim=1)

loss_per_anchor = -(pos_per_anchor - log_den)


# --------------------------------------------------
# 11. batch loss
# --------------------------------------------------

L_aux = loss_per_anchor.mean()
```

工程实现时需避免 `-inf * 0` 导致数值问题，可通过 index/gather 获取 positive logit。

---

# 14. 更稳妥的 Positive Index 实现

因为 LaFG 明确：

\[
\text{每类恰好 2 张}
\]

所以可以预先生成：

```text
positive_index[A1] = A2
positive_index[A2] = A1
positive_index[B1] = B2
positive_index[B2] = B1
...
```

于是：

```python
logits = sim / tau
logits.fill_diagonal_(-float("inf"))

pos_logits = logits[
    torch.arange(B, device=z.device),
    positive_index
]

log_den = torch.logsumexp(logits, dim=1)

L_aux = -(pos_logits - log_den).mean()
```

这样和论文公式最直接对应。

---

# 15. \(L_{\text{aux}}\) 的训练作用

\(L_{\text{aux}}\) 只告诉模型：

```text
A1 和 A2 应该接近
A 和 B 应该远离
A 和 C 应该远离
...
```

它完全不知道：

```text
A/B 为什么不同？
```

也不知道：

```text
是车灯不同？
是车轮不同？
是背景不同？
```

因此：

\[
\boxed{
L_{\text{aux}}
\text{负责建立基础视觉检索空间}
}
\]

而 LaFG 的：

\[
L_{\text{ali}}
\]

才负责额外加入语言属性语义。

---

# 16. LaFG 完整模型中它如何组合

完整 LaFG：

\[
L
=
L_{\text{aux}}
+
\beta L_{\text{ali}}
\]

其中：

### \(L_{\text{aux}}\)

```text
Same Class → Pull Together
Different Class → Push Apart
```

### \(L_{\text{ali}}\)

```text
让视觉表示进一步接受 linguistic prototype 的属性结构约束
```

因此：

\[
\boxed{
L_{\text{aux}}
\text{是视觉 retrieval 主任务}
}
\]

\[
\boxed{
L_{\text{ali}}
\text{是语言辅助监督}
}
\]

---

# 17. 纯 \(L_{\text{aux}}\) 的结果

论文 CUB 消融：

\[
\boxed{
\text{ViT + }L_{\text{aux}}
=
82.6\%~R@1
}
\]

完整 LaFG：

\[
\boxed{
L_{\text{aux}}+L_{\text{ali}}
=
87.2\%~R@1
}
\]

因此纯视觉 \(L_{\text{aux}}\) 就是 LaFG 最直接的 visual-only retrieval baseline。

---

# 18. 论文中一个没有解释清楚的地方

方法部分明确写：

\[
N\text{ classes}\times2\text{ instances/class}
\]

所以：

\[
B=2N
\]

但实验设置又报告：

\[
batch~size=900
\]

CUB 训练只有 100 个训练类别，因此如果严格每类只取 2 张：

\[
B_{\max}=200
\]

和 900 存在表面矛盾。

论文正文没有解释 900 是否指：

- 多 GPU global batch；
- effective batch；
- gradient accumulation；
- 或实际代码中的采样方式与正文有所差异。

因此复现 \(L_{\text{aux}}\) 时，最可靠的是遵循其**明确的正负样本定义**：

\[
\boxed{
N\text{ classes}\times2\text{ images/class}
}
\]

而不是机械照搬 batch size 900。

---

# 19. 如果用于当前项目，最直接的对齐方式

如果希望视觉训练流程与 LaFG 看齐，可以直接采用：

\[
\boxed{
L_{\text{visual}}=L_{\text{aux}}
}
\]

训练：

```text
N classes × 2 images/class
       ↓
ViT
       ↓
Normalized embeddings
       ↓
Batch-wise supervised contrastive loss
       ↓
Same class close / Different classes far
```

然后新的差异知识监督作为独立附加项：

\[
L_{\text{total}}
=
L_{\text{aux}}
+
\lambda L_{\text{diff}}
\]

这样：

- 基础视觉检索部分与 LaFG 的范式一致；
- 新增创新只集中在 \(L_{\text{diff}}\)；
- 更容易做公平消融。

---

# 20. 一句话总结 \(L_{\text{aux}}\)

\[
\boxed{
\text{LaFG 的 }L_{\text{aux}}
=
\text{每类采 2 张图片，在 batch 内让唯一同类图片成为 positive，}
}
\]

\[
\boxed{
\text{其余全部异类图片成为 negatives，并通过 softmax 对比学习训练 ViT 的 cosine retrieval space。}
}
\]
