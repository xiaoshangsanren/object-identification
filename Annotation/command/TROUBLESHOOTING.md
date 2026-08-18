### Windows 交付包可能报错与排查

本文面向甲方 Windows 离线设备，用于排查便携 Python、NVIDIA 驱动、四类模型、NVENC、端口、浏览器和文件权限问题。Windows 包不使用 Docker 或 WSL2，不要求安装系统 Python、CUDA Toolkit，也不会联网下载模型；甲方主机仍须具备兼容的 NVIDIA Windows 驱动和浏览器。

甲方 Windows 驱动版本目前尚未获得确认。此前提供的 525.85.05 是甲方 Linux 环境信息，不能直接当作 Windows 驱动事实。Windows 是否可用，应以 `Check_Environment.cmd` 的完整 JSON 结果为准。

### 首先确定失败阶段

在交付包根目录按以下顺序双击，或在命令提示符中执行：

```bat
Verify_Package.cmd
Check_Environment.cmd
Start_Annotation.cmd
```

各阶段含义如下：

| 阶段 | 通过后能够证明 | 不能证明 |
| --- | --- | --- |
| `Verify_Package.cmd` | 必要程序、模型和资源路径存在 | 文件内容未损坏、GPU 可用 |
| `Check_Environment.cmd` | 便携 Python、CUDA、四类模型、Gallery、NVENC、RAR、端口实测通过 | 完整视频效果符合预期 |
| `Start_Annotation.cmd` | Web 服务能够启动 | 局域网防火墙已经放行 |

`Run_Sample_Test.cmd` 会执行完整样例视频，不是环境安装或基础检查步骤。网页正在运行完整任务时不要同时执行它。

### 包结构缺失、复制中断或安全软件隔离文件

典型报错包括：

```text
必要文件缺失
找不到指定的模块
The system cannot find the path specified
```

先运行 `Verify_Package.cmd`，根据输出确认缺失路径。常见原因是复制中断、只复制了部分子目录，或安全软件隔离了 `python.exe`、DLL、FFmpeg、7-Zip 等可执行文件。

当前包按要求不使用全包哈希。结构检查只能证明文件存在；文件被截断但仍存在时，可能在 `Check_Environment.cmd` 的实际模型加载阶段才暴露。处理方法是从正式交付包重新复制整个受影响目录，不要从互联网临时下载同名文件替换。

若安全软件有拦截记录，应由甲方管理员核验来源后恢复被隔离的包内文件，并将本交付目录加入甲方允许范围。不要关闭整机安全防护作为长期方案。

### VC++ 运行库缺失

典型报错包括：

```text
VCRUNTIME140.dll was not found
MSVCP140.dll was not found
WinError 126
PyTorch加载失败
```

运行包内安装程序：

```bat
prerequisites\VC_redist.x64.exe
```

安装完成后重新运行：

```bat
Check_Environment.cmd
```

VC++ 运行库属于 Windows 主机依赖，不应通过替换 PyTorch、模型文件或系统 Python 处理。

### NVIDIA 驱动或 CUDA 无法使用

典型的 `Check_Environment.cmd` 失败项包括：

```text
nvidia-smi不可用，未检测到Windows NVIDIA驱动
NVIDIA Windows驱动低于CUDA 11.8兼容下限452.39
PyTorch无法访问CUDA GPU
CUDA driver version is insufficient for CUDA runtime version
```

本包携带 PyTorch 2.7.1+cu118 及 CUDA 11.8 用户态运行库，但不携带 NVIDIA Windows 驱动。处理顺序：

1. 在命令提示符执行 `nvidia-smi`，确认 Windows 能识别 NVIDIA GPU。
2. 查看 `Check_Environment.cmd` JSON 中的 `driver_version`、`cuda_available` 和 `visible_cuda_devices`。
3. 若驱动缺失、过旧或损坏，由甲方管理员安装适用于其 Windows 版本和 A40 的 NVIDIA 数据中心驱动。
4. 不需要安装系统 CUDA Toolkit，也不要用系统 Python 替代包内运行环境。

如果 `torch_version` 或 `torch_cuda_version` 与包清单不符，通常说明没有从交付包根目录启动，或运行目录被修改。应重新使用根目录中的 CMD 入口。

### GPU 编号错误或显存不足

默认使用 GPU 0。若 GPU 0 忙碌，可在新的命令提示符中执行：

```bat
set GPU_ID=1
Check_Environment.cmd
Start_Annotation.cmd
```

甲方有多张 GPU 时，编号以 `nvidia-smi` 显示为准。`CUDA out of memory` 表示所选 GPU 显存不足；可切换空闲 GPU，或在网页中降低 YOLO、CLIP、DINO 批大小。关闭启动窗口或按 `Ctrl+C` 会停止当前 Web 服务。

### 模型或 Gallery 离线加载失败

典型失败项包括：

```text
YOLO模型加载失败
CLIP模型单图离线推理失败
DINOv2模型单图离线推理失败
Grounding DINO模型单图离线推理失败
Gallery原型库未就绪或类别数不是10
```

本包运行时已设置离线模式，不会从 Hugging Face 或其他网站补文件。处理方法：

1. 查看 JSON 中的 `missing_files` 和对应模型失败信息。
2. 从正式交付包重新复制整个受影响模型目录，例如 `Resource\models\grounding-dino-tiny\`。
3. Gallery 报错时整体重新复制 `Resource\clip_gallery\`，不要只复制图片而遗漏 `classes.json`。
4. 不要安装另一套 Python、PyTorch 或 Transformers 覆盖包内运行目录。

若多个模型同时出现 `DLL load failed`，优先检查 VC++ 运行库和 NVIDIA 驱动；若只有一个模型报文件解析错误，优先判断该模型目录复制不完整。

### NVENC 编码失败

典型失败项：

```text
FFmpeg未提供h264_nvenc编码器
NVENC实际编码失败
Cannot load nvcuda.dll
No capable devices found
```

`Check_Environment.cmd` 会使用包内 FFmpeg 实际编码一帧。编码失败可能来自驱动、GPU 可见性或 NVENC 能力，而不是输入视频损坏。

若 GPU 推理正常而仅 NVENC 失败，可以在网页中关闭“使用 FFmpeg NVENC 写视频”，改用 CPU 编码继续功能测试；但此时环境不能记为“NVENC 验收通过”。

### 7-Zip 或训练压缩包无法读取

典型失败项：

```text
未找到包内7z.exe
包内7-Zip未提供RAR/RAR5读取后端
```

重新复制 `tools\7zip\`。此问题主要影响 ZIP、RAR 训练数据上传，不影响已提供模型的视频检测。不要依赖甲方系统中另外安装的 7-Zip，因为交付包已经固定使用包内后端。

### 端口被占用

典型失败项：

```text
端口7860已被占用
Only one usage of each socket address is normally permitted
```

如果 Web 服务已经启动，再运行环境检查，7860 被占用是正常现象。若没有服务在运行，则在新的命令提示符中改用其他端口：

```bat
set PORT=7861
Check_Environment.cmd
Start_Annotation.cmd
```

访问地址相应改为 `http://127.0.0.1:7861/`。

### 本机可访问但局域网无法访问

这通常是 Windows 防火墙或甲方网络隔离策略造成，不是模型损坏。首次启动时如出现 Windows 防火墙提示，应由甲方按其安全规范允许专用网络访问。脚本不会自动添加防火墙规则。

局域网访问格式为：

```text
http://甲方Windows设备IP:端口/
```

若本机 `http://127.0.0.1:端口/` 也无法访问，应先查看 `Start_Annotation.cmd` 控制台是否仍在运行及最后一段错误，而不是先修改防火墙。

### 目录不可写、磁盘不足或直接从 U 盘运行

典型失败项包括：

```text
目录不可写
PermissionError
Access is denied
There is not enough space on the disk
```

建议先把整个交付包复制到甲方本机可写磁盘，再从包根目录启动。不要直接在只读 U 盘、受保护系统目录或无写权限的网络共享中运行。

Windows 包约 8.3 GiB，运行时还会写入 `outputs`、`logs`、`tmp`、`datasets` 和 `runs`。建议至少准备 12 GiB 可用空间；长视频输出和训练任务需要另行增加余量。

包已按相对路径设计，可放在不同盘符以及包含中文或空格的目录中。但为减少第三方组件的路径限制，建议目录不要嵌套过深，例如：

```text
D:\Annotation_Windows
```

### 服务窗口、浏览器和任务冲突

`Start_Annotation.cmd` 的控制台必须保持打开。关闭窗口或按 `Ctrl+C` 会停止服务。浏览器没有自动打开不代表服务失败，可手动访问：

```text
http://127.0.0.1:7860/
```

若网页可打开但单次检测失败：

1. 保留启动窗口中的完整 Python 错误。
2. 检查所选算法、输入视频、模型和参数。
3. 同一时刻只运行一个完整视频任务，避免显存和输出文件竞争。
4. 不要在网页任务运行期间再启动 `Run_Sample_Test.cmd`。
5. 输出位于包内 `outputs\`，日志位于 `logs\`。

### 需要提交的排查信息

Windows 首要证据是 `Check_Environment.cmd` 输出的完整 JSON，尤其是 `status`、`failures`、`driver_version`、`cuda_available`、`visible_cuda_devices`、四类模型状态、`nvenc_encode_returncode` 和 `port_available`。

同时提交以下信息：

```bat
Verify_Package.cmd
Check_Environment.cmd
nvidia-smi
```

此外应提供：

- 执行的准确 CMD 文件和执行顺序；
- 使用的 `GPU_ID` 与 `PORT`；
- `Start_Annotation.cmd` 控制台最后一段完整文本；
- 交付包所在路径和磁盘剩余空间；
- 问题发生于启动、模型加载、网页访问还是视频处理阶段。

能够复制文本时不要只发送截图。完整 JSON 和原始错误能区分驱动、VC++、端口、模型复制和权限问题，避免无依据地重装整个系统。
