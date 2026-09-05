# 代码修改记录：逐类Top-5错分与E1隔离运行目录

## 修改时间

2026-09-04 22:00（Asia/Shanghai）

## 修改目的

为E0/E1输出每一个真实类别对其余195类的完整Top-1错分次数和比例，
并明确给出按实际错分次数排序的Top-5错分类别。同时允许E1使用独立运行名，
防止新实验与旧检查点、指标混合。

## 修改范围

- 扩展共享类别对分析和文件保存模块；
- 扩展E1命令行输出目录策略；
- 补充单元测试和README产物说明。

## 修改文件及具体内容

- `src/fg_retrain/pairwise_difficulty.py`：新增逐真实类别的正确/错误Top-1总数，
  对其余所有类别的方向性错分统计和Top-5错分列表；新增JSON/CSV产物。
- `src/fg_retrain/train_laux.py`：新增`--run-name`，运行名已存在时自动添加数字后缀。
- `tests/test_pairwise_difficulty.py`：验证Top-5错分内容、完整其他类别数和新产物。
- `README.md`：记录新增的错分JSON/CSV及字段语义。

## 是否影响旧实验

不改变特征提取、Laux训练、原有检索指标或综合困难度定义。E1未传
`--run-name`时保持原固定目录/断点续训行为；传入后使用新隔离子目录。

## 验证方法与结果

验证结果：

- `Annotation/fg_retrain/tests`：7项全部通过；
- E0/E1都完整处理8,041个Query和196个类别；
- 每个逐类JSON都含196类，每类含195个其他类；
- 每个方向性CSV都含38,220条数据行；
- 逐类`correct_top1_count + error_top1_count == num_queries`全部成立；
- E1历史含200轮，两个实验均以退出码0完成；
- E0 Micro Rank-1为0.3137669563，mAP为0.0908499435；
- E1 Micro Rank-1为0.3887576163，mAP为0.1295168251。

## 对应运行命令

```bash
CUDA_VISIBLE_DEVICES=4 bash Annotation/fg_retrain/scripts/run_e0_vit_test_leave_one_out.sh \
  --run-name top5_confusion_20260904_213057
CUDA_VISIBLE_DEVICES=4 bash Annotation/fg_retrain/scripts/run_e1_vit_vision_laux.sh \
  --run-name top5_confusion_20260904_213057 --no-resume
```

## 输出位置

```text
Annotation/fg_retrain/outputs/e0_vit_test_leave_one_out/top5_confusion_20260904_213057/
Annotation/fg_retrain/outputs/E1_VIT_VisionL/top5_confusion_20260904_213057/
```

## 已知假设或限制

- 错分统计按完整196类Gallery中的Top-1预测定义；
- Query自身被排除，同类的其他测试图作为正例；
- Top-5只保留错分次数大于0的类别，因此极端情况可少于5项。
