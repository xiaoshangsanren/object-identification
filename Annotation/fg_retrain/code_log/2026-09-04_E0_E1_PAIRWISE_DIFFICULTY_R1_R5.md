# 代码修改记录：E0/E1类别对困难度Recall@1～5

## 修改时间

2026-09-04

## 修改目的

在E0纯ViT与E1 ViT+视觉Laux中增加细粒度类别对分析，使每个车型都能明确看到与哪些其他车型最难区分。指标不能只依赖Rank-1，必须覆盖Recall@1、@2、@3、@4、@5。

## 修改范围

新增成对难度计算与历史结果回填模块；扩展E0/E1配置和入口；新增单元测试与README说明。没有修改ViT特征、Laux训练损失、优化器或原有全Gallery检索指标。

## 修改文件及具体内容

### 新增 `src/fg_retrain/pairwise_difficulty.py`

- 定义双类别受限Gallery：A类Query只与其他A类和全部B类图像比较；
- 计算A到B、B到A的Recall@1～5；
- 计算双向Balanced/Micro Recall@1～5；
- 计算最佳同类与最佳对方类别的相似度margin；
- 统计完整196类Gallery中的方向性Top-1混淆；
- 计算类别原型余弦相似度；
- 为19,110个无向类别对生成JSON和CSV；
- 为每个类别生成Top-20困难邻居。

### 新增 `src/fg_retrain/analyze_pairwise_difficulty.py`

- 支持直接从已有`test_embeddings.pt`补算类别对指标；
- 支持将紧凑摘要和产物索引追加到已有`metrics.json`；
- 不需要重新提取特征或重新训练模型。

### 修改 `src/fg_retrain/evaluate_vit_leave_one_out.py`

- E0完成原评测后自动计算并保存成对Recall@1～5。

### 修改 `src/fg_retrain/train_laux.py`

- E1完成原评测后自动计算并保存成对Recall@1～5。

### 修改 `src/fg_retrain/config.py` 和 `src/fg_retrain/laux_config.py`

- 增加成对计算分块大小、固定Recall@1～5和逐类别Top-N设置；
- 强制成对指标必须覆盖1～5，避免未来配置遗漏Rank-5。

### 修改两个实验JSON配置

- 增加`pairwise_query_chunk_size=256`；
- 增加`pairwise_recall_ks=[1,2,3,4,5]`；
- 增加`pairwise_top_n_per_class=20`。

### 新增 `tests/test_pairwise_difficulty.py`

- 构造A/B困难、C独立的合成特征；
- 验证能得到`R@1=0、R@2=0.5、R@3～5=1`而不是重复Rank-1；
- 验证全部JSON/CSV产物包含Rank-1和Rank-5字段。

### 修改 `README.md`

- 说明类别对Gallery、方向性、Recall@1～5、困难分数及输出文件。

## 是否影响旧实验

原E0/E1模型、训练、特征和既有总体指标不变。今后重跑会多生成四个分析文件；已有结果已使用原始embedding离线补算。`metrics.json`仅新增`pairwise_difficulty`字段。

## 验证方法与结果

- `Annotation/fg_retrain/tests`：7项测试全部通过；
- 全部新增类与方法包含“作用/参数/返回值”注释；
- E0和E1均成功处理8,041张测试图、196类和19,110个无向类别对；
- E0最困难对为Audi A5 Coupe 2012与Audi S5 Coupe 2012；
- E1最困难对为Chevrolet Silverado 1500 Extended Cab 2012与Hybrid Crew Cab 2012。

## 对应运行命令

E0历史结果回填：

```bash
PYTHONPATH=Annotation/fg_retrain/src \
python -m fg_retrain.analyze_pairwise_difficulty \
  --embeddings Annotation/fg_retrain/outputs/E0_only_Vit/e0_vit_test_leave_one_out/20260903_162507/test_embeddings.pt \
  --output-dir Annotation/fg_retrain/outputs/E0_only_Vit/e0_vit_test_leave_one_out/20260903_162507 \
  --device cpu --update-metrics
```

E1历史结果回填：

```bash
PYTHONPATH=Annotation/fg_retrain/src \
python -m fg_retrain.analyze_pairwise_difficulty \
  --embeddings Annotation/fg_retrain/outputs/E1_VIT_VisionL/test_embeddings.pt \
  --output-dir Annotation/fg_retrain/outputs/E1_VIT_VisionL \
  --device cpu --update-metrics
```

## 输出位置

```text
Annotation/fg_retrain/outputs/E0_only_Vit/e0_vit_test_leave_one_out/20260903_162507/
Annotation/fg_retrain/outputs/E1_VIT_VisionL/
```

## 已知假设或限制

- 当前协议使用官方test中相同类别的其他图片作为positive；
- `difficulty_score_r1_to_r5`是便于总体排序的汇总量，正式分析仍应同时查看五个Recall；
- 成对受限检索衡量A/B本身的可分性，不等于完整196类Gallery中的最终分类正确率；
- temperature、训练数据或模型改变后必须重新提取embedding或重新补算。
