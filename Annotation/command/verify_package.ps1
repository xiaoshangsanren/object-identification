[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "仅支持64位Windows。"
}
if (Test-Path -LiteralPath (Join-Path $packageRoot "Resource\train_data")) {
    throw "包中不应存在Resource\train_data。"
}
if (Test-Path -LiteralPath (Join-Path $packageRoot "Resource\clip_gallery\evaluation")) {
    throw "包中不应存在Gallery评价标注目录。"
}

$required = @(
    "runtime\python.exe",
    "runtime\python311.dll",
    "app\app.py",
    "app\detect_video.py",
    "app\target_video_search\dino_embedder.py",
    "app\target_video_search\grounding_dino.py",
    "app\target_video_search\temporal_hysteresis.py",
    "Resource\models\yolo\custom_yolo_best.pt",
    "Resource\models\yolo\custom_yolo_best.json",
    "Resource\models\yolo\yolo26n.pt",
    "Resource\models\clip-vit-base-patch32\pytorch_model.bin",
    "Resource\models\dinov2-small\config.json",
    "Resource\models\dinov2-small\model.safetensors",
    "Resource\models\dinov2-small\preprocessor_config.json",
    "Resource\models\grounding-dino-tiny\config.json",
    "Resource\models\grounding-dino-tiny\model.safetensors",
    "Resource\models\grounding-dino-tiny\preprocessor_config.json",
    "Resource\models\grounding-dino-tiny\tokenizer.json",
    "Resource\clip_gallery\classes.json",
    "Resource\videos\Test_Video.mp4",
    "Resource\reference_images\IS-2.webp",
    "tools\ffmpeg\ffmpeg.exe",
    "tools\7zip\7z.exe",
    "tools\7zip\7z.dll",
    "PACKAGE_INFO.txt"
)
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $packageRoot $relative) -PathType Leaf)) {
        throw "缺少文件：$relative"
    }
}

foreach ($directory in @("prototypes", "negatives", "unknown_tanks")) {
    $galleryPath = Join-Path $packageRoot "Resource\clip_gallery\$directory"
    if (-not (Test-Path -LiteralPath $galleryPath -PathType Container)) {
        throw "缺少Gallery目录：$directory"
    }
    if (-not (Get-ChildItem -LiteralPath $galleryPath -Recurse -File | Select-Object -First 1)) {
        throw "Gallery目录为空：$directory"
    }
}

Write-Host "结构检查通过：必要程序、模型和运行资源均存在，训练数据与Gallery评价标注未进入交付包。"
