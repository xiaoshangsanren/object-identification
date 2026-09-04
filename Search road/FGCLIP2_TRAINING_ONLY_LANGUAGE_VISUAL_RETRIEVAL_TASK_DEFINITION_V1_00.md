# 训练期差异语言监督、测试期纯视觉检索：研究任务定义

> 版本：1.00  
> 日期：2026-09-01  
> 项目代号：FG-CLIP2 Training-only Language Supervision  
> 计划代码目录：`Annotation/fg_retrain`  
> 用途：将本文完整提供给 GPT、SciSpace、Google Scholar 等工具，用于检索并分析该任务的国内外研究进展。

---

## 1. 一句话任务定义

研究如何在细粒度车辆图像检索中，利用**仅在训练阶段提供的类别语义与类别间差异文本**监督 FG-CLIP2 的视觉表征学习，使模型在测试阶段**不输入任何文本，只使用图像编码器完成 Query—Gallery 图像检索**，并提高同一已知车型在视角、环境、尺度和成像条件变化下的识别泛化能力。

---

## 2. 研究背景

当前任务是细粒度图—图检索：给定一张 Query 车辆图像，需要从 Gallery 图像中找到同一车型的参考图像，并以最近邻参考图像的类别作为预测结果。

仅依赖类别标签训练视觉模型时，模型只知道哪些图片属于同类，却不知道不同车型之间真正具有判别性的视觉差异是什么。因此，模型可能依赖背景、颜色、拍摄环境、水印或数据集风格等虚假关联，而没有学习车灯、格栅、车身轮廓、车窗、车尾等稳定结构特征。

自然语言能够明确表达这些类别语义和类别间细粒度差异。本研究希望把语言信息当作一种**训练期辅助监督或特权信息**，将其携带的判别知识迁移到图像编码器中；部署时去掉文本数据流，避免推理阶段必须准备文本、运行文本编码器或访问外部知识库。

---

## 3. 与历史任务的边界

本任务是一个新的独立研究阶段，需遵守以下边界：

1. 不以 P2B 为基础，不使用 P2B 的 Patch Scorer、Patch Pooling 或重排序流程；
2. 不使用此前构建的军事车辆知识库、`where` 文本、属性可见性标注、文本热力图或 BBox 文本定位实验；
3. 不研究推理时临时调用知识库解决 Top-2 犹豫的方案；
4. 不要求模型识别训练中从未出现过的车型；
5. 不把旧实验结论直接视为本任务的实验结果；
6. 新的数据、文本、代码、模型权重、训练记录和评价结果均在新阶段单独建立和管理。

---

## 4. 本研究中“泛化”的准确含义

本任务主要研究**闭集细粒度识别中的条件泛化**：

- 训练和测试涉及相同的车型类别集合；
- 测试图片中的具体实例、背景、视角、光照、遮挡、尺度和成像条件可与训练集不同；
- 目标是让同一车型在不同条件下仍然靠近，让不同但相似的车型仍然可分。

本阶段不把下列任务设为主要目标：

- 未见类别识别；
- 开放集识别；
- 零样本类别迁移；
- 用家用车训练后直接识别从未见过的坦克型号；
- 依赖测试期文本提示完成分类。

上述能力可以作为未来扩展，但不能与当前实验中的泛化概念混为一谈。

---

## 5. 研究目标

### 5.1 总目标

构建一个以 FG-CLIP2 为基础的训练框架，使训练期差异语言知识进入视觉表征，而测试期形成一个可独立部署的纯视觉细粒度检索模型。

### 5.2 具体目标

1. 设计适合细粒度车辆识别的类别语义文本和类别对差异文本；
2. 设计能够把“文本差异方向”转化为“视觉差异方向”的训练目标；
3. 避免模型仅记忆类别名称、固定模板或数据集背景；
4. 研究冻结文本编码器、解冻视觉编码器不同层数时的训练稳定性；
5. 验证训练期文本是否真正提升测试期纯视觉图—图检索，而不是依赖推理时文本；
6. 检验方法是否在跨视角、跨背景和难类别对上比纯视觉基线更稳定；
7. 控制推理成本，使最终模型能够嵌入 Annotation 的离线 Gallery 检索阶段。

---

## 6. 任务形式化

### 6.1 数据定义

设训练类别集合为：

```text
C = {c1, c2, ..., cK}
```

每张训练图像表示为：

```text
(I_i, y_i),  y_i ∈ C
```

为每个类别准备类别语义文本：

```text
T_cls(c)
```

为容易混淆的有序类别对准备差异文本：

```text
T_diff(a → b)
```

差异文本描述类别 `a` 相对于类别 `b` 的可观察视觉差异。方向反转时，文本中的比较关系也必须反转。

训练样本可组织为图像对或三元组：

```text
Pair:    (I_q, I_g, y_q, y_g, T_cls, T_diff)
Triplet: (I_anchor, I_positive, I_negative, T_diff(y_anchor → y_negative))
```

其中所有文本只参与训练监督。

### 6.2 训练阶段

训练阶段允许使用：

- Query 图像；
- 正/负 Gallery 图像；
- 图像类别标签；
- 类别自身的语义描述；
- 类别之间的细粒度差异描述。

基础模型为一个 FG-CLIP2 图文模型，包含图像编码器和文本编码器：

```text
z_img = F_img(I)
z_txt = F_txt(T)
```

第一版计划冻结文本编码器，用它把语义文本映射到稳定的语言特征空间；训练视觉编码器的最后若干层及轻量训练头，让视觉特征接受类别语义和差异语义监督。

文本不是测试输入，也不应成为推理必需组件。它的作用类似训练阶段的辅助教师信号或特权模态。

### 6.3 测试阶段

严格主实验的测试输入只有：

```text
Query image: I_q
Gallery: {(I_g1, y_g1), ..., (I_gN, y_gN)}
```

测试阶段禁止输入：

- 类别语义文本；
- 类别差异文本；
- 文本知识库；
- 文本编码器输出；
- 测试样本的人工属性或位置标注。

图像编码器分别提取 Query 和 Gallery 的视觉特征：

```text
v_q  = normalize(F_img(I_q))
v_gi = normalize(F_img(I_gi))
score(q, gi) = cosine(v_q, v_gi)
```

按照相似度从高到低排序，Rank-1 Gallery 图像的标签作为 Query 的预测类别。

### 6.4 输出定义

每张 Query 的输出包括：

```text
predicted_class
ranked_gallery_items
similarity_scores
inference_time
```

数据集级输出至少包括：

```text
Micro Rank-1
Macro Rank-1
Rank-5
mAP
每张 Query 平均视觉检索耗时
```

其中主结论优先使用 Macro Rank-1 与 Micro Rank-1，并同时报告难类别对结果。

---

## 7. 预期模型框架

### 7.1 单模型原则

从预训练 FG-CLIP2 出发，不额外训练一个独立视觉学生模型。训练和测试使用的是同一个模型的视觉分支：

```text
训练：FG-CLIP2 图像分支 + 冻结的 FG-CLIP2 文本分支
测试：训练完成后的 FG-CLIP2 图像分支
```

因此，本质上是利用一个图文对齐模型自身的语言空间指导其视觉分支学习，而不是先训练图文模型、再蒸馏到第二个纯视觉模型。

### 7.2 建议的训练分支

在图像特征之后可以加入训练期轻量头：

```text
z_img ──→ H_cls  ──→ 类别相关视觉特征
      └─→ H_nuis ──→ 图像私有/干扰特征（可选）
```

`H_cls` 用于图—图检索、类别判别和语义对齐；`H_nuis` 用于尝试分离背景、光照、纹理噪声等非类别因素。是否需要显式解耦应通过消融实验验证，而不是预设其一定有效。

### 7.3 候选训练目标

总损失的研究起点可以写为：

```text
L_total =
    L_visual_retrieval
  + λ_cls  · L_class_semantic
  + λ_diff · L_difference_alignment
  + λ_dec  · L_disentanglement
```

各项含义如下：

1. `L_visual_retrieval`：纯视觉检索主任务损失，可由 ID Loss、Triplet Loss、SupCon 或其组合构成；
2. `L_class_semantic`：使图像特征与正确类别语义接近、与错误类别语义分离；
3. `L_difference_alignment`：使图像对形成的视觉关系向量与对应类别对差异文本的方向对齐；
4. `L_disentanglement`：可选的类别特征/干扰特征解耦正则。

差异对齐不能只比较单张图像与一段“二者差异文本”的相似度，因为差异文本表达的是一种关系。需要重点调研以下视觉关系表达：

```text
v_a - v_b
|v_a - v_b|
v_a ⊙ v_b
concat(v_a, v_b, v_a-v_b, |v_a-v_b|)
可学习的 pair/relation encoder
```

并研究如何保持有序差异 `a → b` 的方向性。

---

## 8. 当前模型与数据资源

### 8.1 基础模型

当前选定的基础模型是：

```text
qihoo360/fg-clip2-base
```

本地模型目录：

```text
Annotation/Resource/models/fg-clip2-base
```

选择它的原因是本任务需要同一个图文模型同时提供视觉表示和语言监督空间，而最终可以单独保留其视觉分支做图—图检索。

### 8.2 主数据集候选

目前最适合作为方法研发主数据的是 Car-1000：

```text
datasets/Car-1000
```

它包含约 14 万张真实车辆图片和 1000 个车型类别，并提供品牌—车型层级信息及训练/验证/测试划分。相比当前约一千张有效图像的 Russian-Military-Vehicles，它更适合训练和评价细粒度视觉表征。

第一阶段建议从 Car-1000 中构建可控子集 `Car-1000-FG50`：

- 约 50 个车型类别；
- 按 5 个品牌或相近车型组组织，每组约 10 类；
- 优先选择每类样本数充足、标签可靠、外观相近但可区分的类别；
- 先去重和近重复清理，再划分训练/验证/测试；
- 保证同一张图及其缩放、水印或轻微编辑版本不跨集合；
- 测试集必须包含视角、背景和成像条件变化。

使用 50 类而不是直接处理 1000 类，是为了把文本构建规模控制在可审核范围，同时保留真正的多类别细粒度难度。

### 8.3 后续军事车辆数据

Russian-Military-Vehicles 暂不作为新方法的大规模主训练集。它可以在 Car-1000 上完成方法验证后，用作军事车辆小样本迁移或外部验证数据。

这一步不能被解释为“仅用家用车训练就应识别未知坦克型号”。军事车辆类别仍需进入后续训练或适配阶段。

---

## 9. 文本数据的设计要求

### 9.1 类别语义文本

每个车型建立若干条描述该类别自身外观的文本，建议覆盖：

- 整体车身比例与轮廓；
- 前脸、格栅与车灯；
- 车侧、车窗、车门和轮拱；
- 车尾、尾灯和后窗；
- 车型类别与明显结构特征；
- 在不同视角下仍可观察的稳定线索。

文本只写图像可能观察到的内容，不写价格、发动机参数、销量、年代背景等不可见信息。

### 9.2 类别对差异文本

不为全部 1000 类建立全组合文本。对 `Car-1000-FG50` 中的同品牌、同车型组或视觉上容易混淆的类别对建立差异描述。

若按 5 组、每组 10 类组织，则组内无向类别对为：

```text
5 × C(10, 2) = 225 pairs
```

每个类别对可准备 2～3 个等价改写，并保存正反两个有序方向。文本应记录：

```text
class_a
class_b
shared_visual_properties
differences[]:
  region
  class_a_description
  class_b_description
  observable_views
  confidence
source_or_evidence
```

### 9.3 文本质量约束

1. 差异必须可由普通车辆图片观察，不能依赖车辆铭牌或隐藏参数；
2. 不把类别名本身当作唯一判别信息；
3. 不写数据集背景、水印、拍摄网站等捷径信息；
4. 区分可靠事实、推测和不可确认内容；
5. 同一类别的描述长度和模板数量尽量平衡；
6. 文本生成过程需要保留来源、模型版本、提示词和审核状态；
7. 自动生成文本需要抽样审核，不能默认所有生成内容正确。

---

## 10. 必须建立的实验基线

### E0：预训练 FG-CLIP2 纯视觉零微调

直接使用原始 FG-CLIP2 图像编码器做 Gallery 检索，作为绝对起点。

### E1：只有视觉损失

在相同训练集上用 `L_visual_retrieval` 微调 FG-CLIP2 图像分支，不使用任何文本监督。这是最重要的公平基线。

### E2：类别语义监督

在 E1 基础上加入 `L_class_semantic`，检验普通类别描述是否有效。

### E3：类别语义 + 差异语言监督

加入 `L_difference_alignment`，检验类别对差异文本能否进一步提高纯视觉检索。

### E4：加入解耦正则

在 E3 基础上加入可选的类别/干扰特征解耦，验证它是否减少虚假关联。

### E5：文本破坏负对照

至少进行以下负对照：

- 随机打乱类别语义与图像的对应关系；
- 随机打乱差异文本对应的类别对；
- 交换 `a → b` 的差异方向但不交换文本内容；
- 用等长度的无关车辆文本代替真实差异文本。

若正确文本和破坏文本结果相近，说明模型没有真正利用语言监督。

### E6：严格纯视觉推理核验

删除或禁用文本输入和文本编码器，重新执行测试，验证主结果不是由缓存文本特征、候选类别 Prompt 或代码泄漏产生。

所有实验必须使用相同的数据划分、Gallery、图像分辨率、训练轮数、优化器设置和随机种子集合。

---

## 11. 评价问题

除整体检索分数外，需要回答：

1. 文本监督是否提高 Macro Rank-1 和 Micro Rank-1？
2. 提升是否集中在少数大类，还是覆盖多数类别？
3. 最易混淆的同品牌/相近车型对是否得到改善？
4. 跨视角、跨背景、跨光照和低分辨率子集是否改善？
5. 文本负对照是否显著变差？
6. 测试期完全禁用文本后，提升是否仍然存在？
7. 图像特征是否减少了对背景、水印和颜色等捷径的依赖？
8. 不同文本模板、文本编码方式和差异关系表达对结果有何影响？
9. 解冻视觉 Transformer 最后不同层数时，收益和过拟合如何变化？
10. 方法带来的训练成本、推理耗时和显存变化是多少？

---

## 12. 希望调研的关键科研问题

请围绕以下问题检索现有工作，而不仅仅搜索 FG-CLIP2：

### 12.1 训练期语言、测试期纯视觉

- 是否已有方法把文本作为训练期特权模态，而测试时仅使用视觉模态？
- 这类工作通常被归入哪些术语：Learning Using Privileged Information、privileged modality、modality distillation、language-guided representation learning，还是其他方向？
- 如何证明语言知识已经进入视觉编码器，而不是只在跨模态头中起作用？

### 12.2 差异文本监督

- 是否有工作使用“类别 A 相比类别 B 有什么不同”的自然语言训练视觉模型？
- 对比性/相对性文本应如何编码为有方向的关系向量？
- 单图描述、图像对差异描述、relative attributes 和 comparative caption 各自有什么优缺点？
- 如何处理差异文本与当前图片视角不匹配的问题？

### 12.3 细粒度图像检索

- 训练期语义监督对 fine-grained image retrieval、vehicle re-identification 和 industrial object retrieval 有哪些已验证方法？
- 全局图像特征是否足够，还是必须引入 part-aware/local feature 机制？
- 如果局部结构确有必要，怎样避免重新引入复杂且不稳定的文本定位流程？

### 12.4 虚假关联与特征解耦

- 如何分离类别判别特征与背景、光照、拍摄域、纹理噪声？
- 正交约束、子空间分解、因果表示、domain generalization 和 DecAlign 类方法分别适用于什么条件？
- 如何用实验而不是可视化直觉证明虚假关联减少？

### 12.5 文本质量与自动构建

- 在缺少人工专家标注时，如何用大模型或公开资料生成可靠的类别差异描述？
- 如何自动验证文本是否视觉可观察、事实正确且不含数据泄漏？
- 文本数量、粒度、模板多样性和噪声对训练结果有何影响？

### 12.6 训练稳定性

- FG-CLIP/CLIP 类模型在小规模细粒度数据上微调时，冻结哪些模块更合理？
- 如何避免文本空间被小数据破坏，同时让视觉空间真正吸收监督？
- LoRA、adapter、partial fine-tuning、prompt tuning 与完整微调谁更适合本任务？

---

## 13. 建议检索关键词

### 13.1 英文核心检索式

```text
training-time language supervision test-time visual-only retrieval
language-guided visual representation learning without text at inference
text as privileged information for image recognition
privileged modality distillation vision language model
cross-modal knowledge transfer to visual encoder
language supervision for fine-grained image retrieval
comparative text supervision fine-grained recognition
pairwise difference caption visual representation learning
relative attribute learning image retrieval
class difference descriptions fine-grained classification
vision-language model visual-only inference
semantic auxiliary supervision image retrieval
language-guided vehicle re-identification
fine-grained vehicle retrieval vision-language
disentangled representation spurious correlation fine-grained recognition
modality dropout missing text modality inference
CLIP fine-tuning image-image retrieval
FG-CLIP fine-grained retrieval
FG-CLIP2 fine-tuning visual encoder
```

### 13.2 相关概念词

```text
Learning Using Privileged Information (LUPI)
generalized distillation
privileged modality
cross-modal distillation
modality hallucination
teacher-free multimodal supervision
comparative captioning
relative attributes
relation embedding
difference representation
fine-grained contrastive learning
semantic regularization
feature disentanglement
spurious correlation mitigation
domain generalization
```

### 13.3 中文关键词

```text
训练期文本监督 测试期纯视觉
语言作为特权信息 图像识别
跨模态知识迁移 视觉编码器
差异文本监督 细粒度识别
相对属性 图像检索
类别对差异描述 视觉关系学习
细粒度车辆检索 图文模型
虚假关联 特征解耦 细粒度识别
```

---

## 14. 文献检索的纳入与排除标准

### 14.1 优先纳入

- 训练阶段使用文本或其他模态、测试阶段可只用图像的工作；
- 细粒度分类、检索、ReID 或工业对象识别工作；
- 使用类别描述、比较描述、差异文本或 relative attributes 的方法；
- 能迁移到 CLIP、FG-CLIP、SigLIP 或其他视觉语言基础模型的方法；
- 具有公开论文、代码、数据或可复现实验细节的工作；
- 顶级会议/期刊论文，以及真正相关的高质量预印本；
- 经典 LUPI、跨模态蒸馏和相对属性工作，即使发表年份较早。

### 14.2 单独标记，不能直接当作同一任务

- 测试阶段仍必须输入文本 Prompt 的方法；
- 纯零样本分类或开放词汇识别；
- 仅做文本到图像检索，而不是图像到图像检索；
- 仅用类别名做普通 CLIP 对齐、没有训练期到纯视觉推理转移；
- 需要人工部件框、分割掩码或密集属性标注的方法；
- 依赖超大规模私有训练数据、无法在两张 RTX 4090 上验证的方法；
- 只展示注意力图但没有检索性能或负对照证据的方法。

---

## 15. 希望文献调研最终回答的内容

请把检索结果组织成表格，每篇论文至少记录：

```text
论文题目
发表会议/期刊与年份
可验证的论文链接
官方代码链接
任务类型
训练阶段输入模态
测试阶段输入模态
是否支持纯视觉推理
是否使用类别/差异文本
是否需要人工局部标注
视觉关系或差异的建模方法
主要损失函数
使用的数据集
核心指标
计算资源
与本任务的直接相关性
可移植到 FG-CLIP2 的模块
主要风险或不适用原因
```

并在汇总后明确回答：

1. 该任务在科研界最接近的标准名称是什么？
2. 哪些论文真正满足“训练有文本、测试无文本”？
3. 哪些论文真正建模了成对差异，而不是普通类别 Prompt？
4. 当前最主流的技术路线有哪些？
5. 哪条路线最适合 FG-CLIP2 Base、Car-1000-FG50 和两张 RTX 4090？
6. 本文提出的框架中哪些部分已有成熟先例，哪些部分可能构成研究创新？
7. 最小可行实验应该怎样设计，才能有力证明文本知识被视觉分支吸收？
8. 应优先复现哪些 3～5 篇论文？

---

## 16. 可直接交给 GPT 的检索指令

```text
请把下面的研究任务定义视为一个全新的课题，不要自动关联任何历史方案。

任务目标：在细粒度车辆图—图检索中，以 FG-CLIP2 为单一基础模型，训练阶段使用类别语义文本和类别对差异文本作为辅助监督；测试阶段彻底移除文本输入与文本编码器，只使用训练后的图像编码器完成 Query—Gallery 检索。这里的泛化是已知车型在新视角、新背景、新光照、尺度和成像条件下的闭集泛化，不是未见车型的零样本识别。

请联网检索并系统梳理该任务截至当前的研究进展。重点覆盖：
1. training-time language supervision / test-time visual-only inference；
2. Learning Using Privileged Information、跨模态蒸馏和模态缺失推理；
3. 比较文本、类别对差异描述、relative attributes 与视觉关系学习；
4. 细粒度图像检索、车辆检索和图文基础模型微调；
5. 虚假关联抑制与类别相关/图像私有特征解耦；
6. 无大规模人工局部标注条件下的文本数据构建与质量控制。

请区分：
- 真正测试期纯视觉的方法；
- 测试时仍使用类别 Prompt 或缓存文本向量的方法；
- 零样本分类；
- 文本—图像检索；
- 图像—图像检索。

不要因为标题或摘要含有 language-guided 就判断相关。必须检查方法和测试数据流。优先使用论文原文、项目主页和官方代码作为证据，不要杜撰论文、会议、年份或实验结果。对 CVPR 2026 等较新工作需要核验其真实发表状态。

最终请输出：
- 该任务最准确的学术定位；
- 按技术路线组织的研究进展；
- 论文对比表；
- 最相关的 3～5 个可复现基线；
- 对 FG-CLIP2 Base + Car-1000-FG50 + 两张 RTX 4090 的适配建议；
- 当前方案已有先例的部分、仍有创新空间的部分及主要失败风险；
- 一个从最小验证到完整实验的实施顺序。
```

---

## 17. 成功判据

本阶段不能仅凭“加入了文本”或“损失下降”判断成功。至少需要同时满足：

1. 测试代码在完全没有文本输入和文本特征的条件下可独立运行；
2. 相比相同设置的视觉微调基线，Macro/Micro Rank-1 有稳定提升；
3. 多个随机种子或交叉划分中结果具有一致性，而不是单次偶然提升；
4. 难类别对、跨背景或跨视角子集有可解释的改善；
5. 真实文本优于打乱、反向或无关文本负对照；
6. 不依赖测试数据泄漏、重复图像或类别名称捷径；
7. 最终视觉分支的推理时间和显存满足 Annotation 离线 Gallery 检索要求。

如果真实文本监督没有优于严格视觉基线和文本破坏负对照，则应得出“当前文本设计或知识迁移机制无效”的结论，而不是默认增加模态必然提高性能。

