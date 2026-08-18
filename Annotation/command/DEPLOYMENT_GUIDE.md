### Windows原生离线交付包部署指引

本文用于系统部署者在甲方完全断网的64位Windows设备上部署`Annotation_Windows`。本包采用便携Python原生运行，不使用Docker或WSL2，不要求安装系统Python或CUDA Toolkit。

### 当前包与适配边界

| 项目 | 当前事实 |
|---|---|
| 交付包 | `Annotation_Windows`，约8.317 GiB |
| 运行时 | CPython 3.11.9 x64便携运行时 |
| PyTorch | 2.7.1+cu118，CUDA 11.8用户态运行库随包提供 |
| 算法 | `CLIP`、`clip_gallery_temporal`、`dino_gallery_temporal`、`text_grounding_temporal` |
| 模型 | YOLO、CLIP、DINOv2-small、Grounding DINO Tiny |
| 视频 | `Test_Video.mp4`、`Test_1.mp4`、`Test_2.mp4`、`Test_3.mp4` |
| 附带工具 | FFmpeg/NVENC、7-Zip、Microsoft Visual C++ Redistributable |

当前包已在本机Windows 10、GTX 1650、驱动591.59环境中通过CUDA张量、YOLO、CLIP、DINOv2、Grounding DINO、10类Gallery、NVENC和RAR/RAR5读取检查，并在含中文与空格的路径下成功启动网页。

甲方提供的Docker 28.3.2、A40、驱动525.85.05、CUDA 12.0和NVIDIA Container Toolkit 1.17.8属于Linux设备信息，不能据此推断甲方Windows驱动状态。Windows包是否可运行，必须以甲方Windows执行`Check_Environment.cmd`的实际结果为准。

CUDA 11.8在Windows上的次版本兼容最低驱动为452.39；本包的环境脚本据此检查驱动版本。但“版本达到下限”仍不能替代CUDA张量、模型加载和NVENC实测。兼容性依据可参考NVIDIA的[CUDA次版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。该链接仅作为部署前依据，甲方离线执行不依赖网络。

### 甲方Windows必须预先具备的条件

- 64位Windows，处理器架构为`AMD64/x86_64`。
- 已安装支持目标NVIDIA GPU的Windows驱动，且`nvidia-smi`能够正常运行。
- 本机浏览器。
- 本地磁盘建议至少保留12 GiB可用空间，并为后续输出视频预留额外空间。

本包已经包含Python、PyTorch CUDA 11.8运行库、全部模型、FFmpeg和7-Zip。不要另外安装Python或CUDA Toolkit，也不要在断网设备上执行`pip install`。

### 放置交付包

建议先把整个`Annotation_Windows`目录从U盘复制到甲方本地磁盘，再执行。包可以位于任意盘符，也可以位于含中文或空格的路径。不能只复制`runtime/`、`app/`或`Resource/`中的单个目录。

双击入口文件即可运行。若需要查看完整输出或改变GPU、端口，应打开CMD并进入包根目录：

```bat
cd /d D:\实际路径\Annotation_Windows
```

### 执行顺序与作用

| 顺序 | 文件 | 是否必须 | 作用 |
|---:|---|---|---|
| 1 | `Verify_Package.cmd` | 必须 | 秒级检查64位Windows、便携运行时、应用、模型、视频、参考图、Gallery、FFmpeg和7-Zip是否存在；不执行全量哈希。 |
| 2 | `Check_Environment.cmd` | 必须 | 检查Windows驱动、CUDA、可见GPU、目录写入、端口、四类模型单图离线推理、Gallery、NVENC和RAR后端。 |
| 3 | `Start_Annotation.cmd` | 必须 | 设置离线路径和缓存目录，启动网页服务并打开浏览器。控制台必须保持运行。 |
| 4 | 浏览器访问 | 必须 | 本机访问`http://127.0.0.1:7860/`；局域网访问使用该Windows主机的IPv4地址。 |
| 5 | `Run_Sample_Test.cmd` | 可选 | 使用标准CLIP参数完整处理`Test_Video.mp4`。不能与网页中的另一项检测同时运行。 |
| 6 | `Ctrl+C`或关闭启动窗口 | 停止时执行 | 停止网页服务，不删除模型和输出。 |

`command/set_environment.cmd`、`verify_package.ps1`、`check_environment.py`和`launch_ui.py`是上述入口调用的内部文件，正常部署时不需要单独运行。`command/copy_to_usb.cmd`和`copy_to_usb.ps1`只供系统部署者在自己的Windows电脑上复制整包到U盘。

### GPU与端口

默认使用GPU 0和端口7860。在同一个CMD窗口中临时修改：

```bat
set GPU_ID=1
set PORT=7861
Check_Environment.cmd
Start_Annotation.cmd
```

`Check_Environment.cmd`会检查所选端口尚未被占用，因此应在启动服务之前执行。如果服务已经运行，再执行环境检查会报告端口被占用，这不代表模型损坏。

### 常见边界

- 若提示缺少VC++运行库或DLL，运行`prerequisites\VC_redist.x64.exe`，完成后重新执行环境检查。
- 若`cuda_available`为`false`，应修复或更新甲方Windows NVIDIA驱动；不要通过安装系统CUDA Toolkit解决。
- 若NVENC失败但模型加载成功，网页中可以关闭NVENC后改用CPU编码，但标准验收仍应记录该环境差异。
- 首次启动出现Windows防火墙提示时，只允许甲方认可的网络范围。脚本不会自动修改防火墙。
- 输出写入`outputs/`，日志写入`logs/`，缓存写入`tmp/`；这些路径都位于交付包内部。

### 执行命令与预期输出（节选）

结构检查：

```bat
Verify_Package.cmd
```

预期输出：

```text
结构检查通过：必要程序、模型和运行资源均存在，训练数据与Gallery评价标注未进入交付包。
```

环境验收：

```bat
Check_Environment.cmd
```

预期JSON至少包含：

```json
{
  "cuda_available": true,
  "yolo_models": "loaded",
  "clip_model": "single_image_offline_inference_ok",
  "dinov2_model": "single_image_offline_inference_ok",
  "grounding_dino_model": "single_image_offline_inference_ok",
  "nvenc_encode_returncode": 0,
  "seven_zip_rar_supported": true,
  "port_available": true,
  "status": "ok",
  "failures": []
}
```

甲方GPU名称应显示实际设备，例如`NVIDIA A40`。只有`status`为`ok`且`failures`为空，才进入启动步骤。

启动服务：

```bat
Start_Annotation.cmd
```

预期输出至少包括：

```text
Package: <实际包路径>
GPU: 0
Local URL: http://127.0.0.1:7860/
LAN URL: http://<本机IPv4地址>:7860/
```

控制台保持运行后，在浏览器打开`http://127.0.0.1:7860/`。停止时在该窗口按`Ctrl+C`。

可选完整样例：

```bat
Run_Sample_Test.cmd
```

预期行为是完整处理`Resource\videos\Test_Video.mp4`，并在`outputs\`生成标注视频和JSON。该命令没有固定耗时，不应作为结构检查或启动前置步骤。
