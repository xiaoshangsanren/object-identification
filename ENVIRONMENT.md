# Annotation 项目统一实验环境

从 2026-09-03 起，本项目后续的数据处理、模型训练、消融实验、评测和模型导出统一使用：

```bash
source /home/anaconda3/etc/profile.d/conda.sh
conda activate cc_object_identification
```

环境信息：

- 名称：`cc_object_identification`
- Conda入口：`/home/NCUT/25/cc/.conda/envs/cc_object_identification`
- 物理位置：`/mnt/sata_ssd/cc-2025/conda_envs/cc_object_identification`
- Python：3.11.15
- PyTorch：2.7.1+cu118
- torchvision：0.22.1+cu118
- GPU验证：8张 NVIDIA GeForce RTX 4090，CUDA张量运算通过

已配置当前项目需要的核心组件，包括 PyTorch、torchvision、Transformers、timm、Hugging Face datasets、PyArrow、Pandas、scikit-learn、Pillow、SciPy、OpenCV 和 Ultralytics 等。

运行实验前可执行：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

除非某项第三方复现实验明确要求单独隔离，否则不要再使用 `base` 或 `cc_PAD_Lite` 运行本项目后续实验。

