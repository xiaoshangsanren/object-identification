# PAD-Lite Verification Record

更新时间：2026-08-08

## 1. 数据与协议

- Russian-Military-Vehicles 有效目标裁剪：1,254 张，10 类。
- Final 训练使用全部 1,254 张裁剪；没有使用 Gallery 图片训练、选轮次或调参。
- B0 是冻结的预训练 CLIP 基线，因此不存在 B0 训练过程。
- B1 和 B2 的训练轮次在查看 Gallery 结果前固定为 30 epoch。
- 原始 Gallery 未修改：`Resource/clip_gallery`。
- 版本化评测 Gallery：`Resource/clip_gallery_eval_10class_v1`。
- 版本化测试集：`PAD_Lite/evaluation_data/gallery_10class_eval_v1/test`。
- 10 类 Gallery 图片共 104 张，按类别约 2/3 : 1/3 划分为 70 张参考图和 34 张测试图。
- 固定随机种子：`20260808`。
- 完全相同文件哈希的图片被绑定在同一侧；参考集与测试集的哈希交集为 0。
- 划分清单 SHA-256：`12e4e682db5b5d2e09d3e4b617a497390e7470e59bbf0a96852acc425b632390`（目录整理后仅路径字段改变，图片划分未变）。

## 2. Final 模型

| 模型 | 训练数据 | epoch | 最终训练准确率 | 最终训练损失 | 文本锚点损失 |
|---|---:|---:|---:|---:|---:|
| B0 | 不训练 | - | - | - | - |
| B1 | 1,254 | 30 | 0.6994 | 1.8952 | - |
| B2 | 1,254 | 30 | 0.8370 | 2.0876 | 0.1486 |

权重：

- `PAD_Lite/outputs/final/b1/final.pt`
- `PAD_Lite/outputs/final/b2/final.pt`

B1 使用 GPU 4、B2 使用 GPU 6 完成训练。B2 的文本只参与训练；部署推理路径只加载视觉编码器和投影头，不加载文本编码器或文本提示。

## 3. Gallery 10 类测试

测试通过项目原有的 `GalleryTemporalMatcher` 构造参考原型；每类最多使用 3 个参考原型。测试图片不经过 YOLO，因为该测试的目标是单独衡量识别/检索模型，而不是把检测误差混入分类误差。

| 模型 | Rank-1 | 正确数 | Class mAP | Retrieval mAP | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.3529 | 12/34 | 0.2496 | 0.2602 | 0.3494 |
| B1 | 0.2941 | 10/34 | 0.2907 | 0.2650 | 0.2611 |
| B2 | 0.3529 | 12/34 | 0.3079 | 0.2629 | 0.3460 |

相对 B0：

- B1 的 Class mAP 提高 4.11 个百分点，但 Rank-1 降低 5.88 个百分点。
- B2 的 Class mAP 提高 5.83 个百分点，Rank-1 与 B0 持平。
- B2 相对 B0 改对 4 张，同时改错 4 张，因此当前不能声称它提升了 10 类 Top-1 准确率。
- `M-26`、`Tiger_2`、`Panzer_IV_H` 在三个模型上的 Rank-1 都是 0，是后续误差分析的首要类别。

完整结果：`PAD_Lite/outputs/gallery_10class/summary.json`。

## 4. Annotation 完整流程接入

完整程序现支持以下识别后端：

- `clip`
- `dinov2`
- `pad_lite_b0`
- `pad_lite_b1`
- `pad_lite_b2`

三个 PAD-Lite 后端均已通过真实视频冒烟测试：

1. YOLO 检测视频帧中的目标；
2. PAD-Lite 对目标裁剪提取视觉特征；
3. Gallery 匹配器完成类别检索；
4. 时序逻辑生成检测结果；
5. 成功输出 MP4 和 JSON。

冒烟测试输入为 `Resource/videos/Test_1.mp4`，三个输出视频时长均为 4.9 秒且文件非空。该测试只证明接口和完整运行链路正常，不代表分类准确率。

## 5. 已执行的技术检查

- Python 编译检查通过。
- B1/B2 Final checkpoint 能正常加载，元数据均为 10 类、1,254 个训练样本、30 epoch。
- Gallery 划分复现检查通过，参考集与测试集无文件哈希泄漏。
- B0/B1/B2 静态 10 类评测完成。
- B0/B1/B2 完整视频流程执行完成。
- Final 训练日志未发现 traceback、CUDA OOM 或 NaN。

## 6. 当前结论与限制

- 全量 Russian 训练协议比只训练两个 held-out 类更符合当前目标，也避免 Gallery 泄漏。
- B2 学到了更好的类别排序信息，但尚未转化为更高的 Rank-1；B1 当前不优于 B0。
- Gallery 只有 34 张测试图片，单次划分的不确定性较大，不能据此做强泛化结论。
- Russian 与 Gallery 的车型类别不重合，且图像域差异明显；当前结果衡量的是跨类别、跨域的表征迁移，而不是同车型跨环境识别。
- 完整流程中的已知/未知阈值仍沿用原有 CLIP 标定。B1/B2 的部署阈值必须使用独立验证集重新标定，不能用这 34 张测试图调阈值。
