# PAD_Lite 可回档实验管理

## 目标

PAD_Lite 的实验从现在起使用“命名预设 + 独立 Run + 源码快照”的方式管理：

- B0、B1、B2、带文本锚点、不带文本锚点不再依靠临时改代码切换。
- CenterCrop、Letterbox、输入分辨率、主干解冻层数和 P1a 都有独立名称。
- 每次新训练默认创建一个新目录，不复用也不覆盖旧结果。
- 每个 Run 保存完整的 Python 源码、配置和预设快照。
- 原有 `python -m PAD_Lite`、`PAD_Lite/src/dino_cli.py`、`PAD_Lite/src/dino_patch_cli.py` 和 sweep 入口继续保留。

## 统一入口

在 `Annotation` 根目录执行：

```bash
conda activate cc_PAD_Lite
python -m PAD_Lite.experiment_cli list
```

只看不带文本锚点的实验：

```bash
python -m PAD_Lite.experiment_cli list --contains no-text
```

只看带文本锚点的实验：

```bash
python -m PAD_Lite.experiment_cli list --contains text-anchor
```

查看某个模式的完整配置，但不训练：

```bash
python -m PAD_Lite.experiment_cli show \
  dino_s_b2_text_anchor_letterbox_336_u00
```

## B1/B2 随时切换

DINOv2-Small、Letterbox 336、不使用文本锚点的 B1：

```bash
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.experiment_cli run \
  dino_s_b1_no_text_letterbox_336_u00 \
  --fold all --device cuda:0
```

只把实验名切换成下面这个名字，即可运行相同视觉设置、加入训练期文本锚点的 B2：

```bash
CUDA_VISIBLE_DEVICES=4 python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_letterbox_336_u00 \
  --fold all --device cuda:0
```

`u00` 表示解冻最后 0 个 Transformer Block，只训练检索头；`u01` 到 `u12` 表示解冻最后 1 到 12 个 Block。

例如切换到 DINOv2-Small、CenterCrop 224、带文本锚点、解冻最后 2 层：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_center_crop_224_u02 \
  --fold all --device cuda:0
```

## Run 与防覆盖规则

不传 `--run-id` 时，程序自动生成 UTC 时间戳作为 Run ID：

```text
PAD_Lite/outputs/experiment_runs/<实验名>/<UTC时间戳>/
```

也可以指定便于识别的 Run ID：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_letterbox_336_u00 \
  --run-id reproduce_v1 --fold all --device cuda:0
```

如果 `reproduce_v1` 已有文件，普通训练会直接拒绝执行，不会覆盖。恢复中断训练必须显式指定原 Run：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_letterbox_336_u00 \
  --run-id reproduce_v1 --fold all --device cuda:0 --resume
```

只评估已有检查点：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b2_text_anchor_letterbox_336_u00 \
  --run-id reproduce_v1 --fold all --device cuda:0 --eval-only
```

列出该模式已经创建的所有 Run：

```bash
python -m PAD_Lite.experiment_cli runs \
  dino_s_b2_text_anchor_letterbox_336_u00
```

正式执行前可使用 `--dry-run` 检查选择是否正确。该参数不创建目录、不加载模型：

```bash
python -m PAD_Lite.experiment_cli run \
  dino_s_b1_no_text_letterbox_336_u00 \
  --fold 1 --device cuda:0 --dry-run
```

## 每个 Run 保存的内容

```text
<run_root>/
├── experiment_manifest.json       # 实验名、B版本、文本开关、预设覆盖项
├── resolved_config.json           # 训练引擎实际收到的完整配置
├── events/                        # 每次 train/resume/eval 的状态与结果
├── source_snapshot/
│   ├── snapshot_manifest.json     # 快照文件 SHA-256
│   └── PAD_Lite/                  # 当次 src、兼容入口、config、preset 文件
└── dino/b2/fold_01/...            # 原训练引擎输出；不同 family 略有差异
```

因此即使以后实现 P1b、P2 或修改当前训练模块，也可以从旧 Run 的 `source_snapshot/PAD_Lite/` 找回当时的完整代码和实验定义。新 Run 的实现文件保存在快照的 `PAD_Lite/src/`；旧 Run 仍保持其创建时的原始目录。回档前应先备份当前源码，再将所需快照复制回项目；不要删除旧 Run。

## 新实验的强制约定

1. 不修改或删除已有 `PAD_Lite/experiments/*.json` 的实验含义。
2. 新实验在 `PAD_Lite/experiments/` 中新增一个不同名称的 JSON 预设。
3. 新算法优先新增模型/引擎模块；若必须扩展共用模块，旧分支和旧参数行为必须保留。
4. 新预设不得设置 `paths.output_root`，输出位置由统一入口隔离。
5. 实验名必须明确包含关键差异，例如 `b1/b2`、`text_anchor/no_text`、输入策略、尺寸和 `uXX`。
6. 先运行 `list`、`show`、`--dry-run` 和测试，再启动完整五折训练。

## 预设文件格式

单一实验示例：

```json
{
  "schema_version": 1,
  "name": "dino_s_b2_text_anchor_letterbox_336_u00",
  "family": "dino",
  "variant": "b2",
  "base_config": "PAD_Lite/configs/dino_letterbox_text_anchor.json",
  "description": "...",
  "text_anchor": true,
  "overrides": {
    "data.image_size": 336,
    "model.unfreeze_last_blocks": 0
  },
  "tags": ["dino-small", "b2", "text-anchor"]
}
```

`family` 当前支持 `clip`、`dino`、`dino_patch`；对应的 `variant` 分别为 `b0/b1/b2` 和 `p1a/p1b/p2a/p2b`。P1b/P2b继承P0中训练期文本增强得到的CLS分支，但融合执行和最终推理均不加载文本编码器；P1a/P2a均为不使用文本锚点的Patch-only消融。
