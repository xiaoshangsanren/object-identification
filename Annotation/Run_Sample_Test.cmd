@echo off
setlocal
call "%~dp0command\set_environment.cmd"
echo This command processes the complete sample video. Do not run it together with the web task.
"%PACKAGE_ROOT%\runtime\python.exe" "%PACKAGE_ROOT%\app\detect_video.py" ^
  --algorithm CLIP ^
  --video "%PACKAGE_ROOT%\Resource\videos\Test_Video.mp4" ^
  --target-image "%PACKAGE_ROOT%\Resource\reference_images\IS-2.webp" ^
  --detector-model "%PACKAGE_ROOT%\Resource\models\yolo\custom_yolo_best.pt" ^
  --classes 0 ^
  --target-box 60,80,640,420 ^
  --detector-imgsz 640 ^
  --proposal-conf 0.40 ^
  --similarity-threshold 0.24 ^
  --highlight-similarity-threshold 0.40 ^
  --target-sample-fps 15 ^
  --frame-stride 0 ^
  --yolo-batch-size 16 ^
  --clip-batch-size 128 ^
  --clip-backend transformers ^
  --clip-local-dir "%PACKAGE_ROOT%\Resource\models\clip-vit-base-patch32" ^
  --clip-pretrained "%PACKAGE_ROOT%\Resource\models\clip-vit-base-patch32" ^
  --output-dir "%PACKAGE_ROOT%\outputs" ^
  --device auto ^
  --no-half
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
