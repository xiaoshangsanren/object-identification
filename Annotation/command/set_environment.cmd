@echo off
for %%I in ("%~dp0..") do set "PACKAGE_ROOT=%%~fI"

if not defined GPU_ID set "GPU_ID=0"
if not defined PORT set "PORT=7860"

set "CUDA_VISIBLE_DEVICES=%GPU_ID%"
set "TVS_MODEL_ROOT=%PACKAGE_ROOT%\Resource\models"
set "TVS_YOLO_MODEL_DIR=%PACKAGE_ROOT%\Resource\models\yolo"
set "TVS_DETECTOR_MODEL=%PACKAGE_ROOT%\Resource\models\yolo\custom_yolo_best.pt"
set "TVS_CLIP_BACKEND=transformers"
set "TVS_CLIP_LOCAL_DIR=%PACKAGE_ROOT%\Resource\models\clip-vit-base-patch32"
set "TVS_CLIP_PRETRAINED=%PACKAGE_ROOT%\Resource\models\clip-vit-base-patch32"
set "TVS_DINO_LOCAL_DIR=%PACKAGE_ROOT%\Resource\models\dinov2-small"
set "TVS_PAD_LITE_CHECKPOINT_ROOT=%PACKAGE_ROOT%\PAD_Lite\outputs\final"
set "TVS_GROUNDING_MODEL_DIR=%PACKAGE_ROOT%\Resource\models\grounding-dino-tiny"
set "TVS_CLIP_GALLERY_ROOT=%PACKAGE_ROOT%\Resource\clip_gallery"
set "TVS_INPUT_ROOT=%PACKAGE_ROOT%\Resource\videos"
set "TVS_REFERENCE_ROOT=%PACKAGE_ROOT%\Resource\reference_images"
set "TVS_TRAIN_DATA_ROOT=%PACKAGE_ROOT%\Resource\train_data"
set "TVS_OUTPUT_ROOT=%PACKAGE_ROOT%\outputs"
set "TVS_DATASET_ROOT=%PACKAGE_ROOT%\datasets"
set "TVS_RUNS_ROOT=%PACKAGE_ROOT%\runs"
set "LOG_DIR=%PACKAGE_ROOT%\logs"
set "GRADIO_TEMP_DIR=%PACKAGE_ROOT%\tmp\gradio"
set "TMP=%PACKAGE_ROOT%\tmp"
set "TEMP=%PACKAGE_ROOT%\tmp"
set "TORCH_HOME=%PACKAGE_ROOT%\tmp\torch"
set "HF_HOME=%PACKAGE_ROOT%\tmp\huggingface"
set "YOLO_CONFIG_DIR=%PACKAGE_ROOT%\tmp\ultralytics"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "GRADIO_ANALYTICS_ENABLED=False"
set "DO_NOT_TRACK=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "NO_PROXY=localhost,127.0.0.1,::1"
set "PATH=%PACKAGE_ROOT%\tools\ffmpeg;%PACKAGE_ROOT%\tools\7zip;%PATH%"

for %%D in (outputs logs tmp datasets runs) do if not exist "%PACKAGE_ROOT%\%%D" mkdir "%PACKAGE_ROOT%\%%D" >nul
if not exist "%GRADIO_TEMP_DIR%" mkdir "%GRADIO_TEMP_DIR%" >nul
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%" >nul
if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%" >nul

cd /d "%PACKAGE_ROOT%"
