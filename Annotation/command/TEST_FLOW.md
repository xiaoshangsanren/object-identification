# Windows离线测试流程

本包使用便携Python原生运行，不依赖Docker、WSL2、系统Python或CUDA Toolkit。甲方Windows主机只需已有可用的64位NVIDIA驱动和浏览器；运行时不会联网下载模型。

## 1. 结构与环境检查

依次双击：

```text
Verify_Package.cmd
Check_Environment.cmd
```

环境检查应显示CUDA可用，并依次完成YOLO、CLIP、DINOv2、Grounding DINO单图离线推理、10类Gallery检查、NVENC编码和7-Zip RAR读取检查。若提示VC++运行库缺失，运行`prerequisites/VC_redist.x64.exe`后重新检查。

## 2. 启动网页

双击`Start_Annotation.cmd`。默认地址为`http://127.0.0.1:7860/`；局域网使用`http://本机IPv4地址:7860/`。Windows防火墙提示时允许专用网络访问。

临时改变GPU或端口：

```bat
set GPU_ID=1
set PORT=7861
Start_Annotation.cmd
```

## 3. 四种算法

- `CLIP`：YOLO产生候选框，单张参考图进行相似度筛选；标准测试使用此算法。
- `clip_gallery_temporal`：YOLO候选框加CLIP多原型库与时序稳定，不要求上传目标图。
- `dino_gallery_temporal`：YOLO候选框加DINOv2多原型库与可选Temporal Hysteresis，不要求上传目标图。
- `text_grounding_temporal`：Grounding DINO直接按英文文本定位，不加载YOLO、CLIP或DINOv2，不要求上传目标图。

## 4. 标准CLIP测试

1. 上传`Resource/videos/Test_Video.mp4`。
2. 上传`Resource/reference_images/IS-2.webp`。
3. 选择`CLIP`和`Resource/models/yolo/custom_yolo_best.pt`，点击“填入推荐类别ID”，确认类别ID为0。
4. 设置目标框`60,80,640,420`、检测尺寸640、候选框置信度0.40、相似度阈值0.24、红框阈值0.40、采样FPS 15、YOLO批大小16、CLIP批大小128。
5. FP16关闭、NVENC开启，开始检测。结果写入`outputs/`。

`yolo26n.pt`仅作为通用COCO模型对照。完整命令行样例可双击`Run_Sample_Test.cmd`，但该命令会处理完整视频，构建和验收阶段不会自动运行。

## 5. YOLO新类别训练数据

网页训练入口支持标准YOLO/data.yaml、LabelMe/X-AnyLabeling JSON、COCO JSON、Pascal VOC XML和扁平YOLO TXT/classes.txt。普通数据中没有标注的图片视为背景图；全部图片均无标注时拒绝训练。压缩包名为`tank_only_train`时启用坦克裁剪图特例：未标注图片按整图坦克框转换，并自动填入类别`tank`。

关闭服务时，在启动窗口按`Ctrl+C`或直接关闭窗口。
