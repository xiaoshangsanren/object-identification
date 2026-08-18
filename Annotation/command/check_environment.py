from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in value.strip().split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main() -> None:
    root = Path(os.environ["PACKAGE_ROOT"]).resolve()
    failures: list[str] = []
    report: dict[str, object] = {
        "package_root": str(root),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "selected_gpu": os.environ.get("GPU_ID", "0"),
    }

    if platform.system() != "Windows":
        failures.append("当前系统不是Windows。")
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        failures.append("当前系统不是x86_64/amd64。")

    required = [
        root / "Resource" / "models" / "yolo" / "custom_yolo_best.pt",
        root / "Resource" / "models" / "yolo" / "custom_yolo_best.json",
        root / "Resource" / "models" / "yolo" / "yolo26n.pt",
        root / "Resource" / "models" / "clip-vit-base-patch32" / "config.json",
        root / "Resource" / "models" / "clip-vit-base-patch32" / "pytorch_model.bin",
        root / "Resource" / "models" / "dinov2-small" / "config.json",
        root / "Resource" / "models" / "dinov2-small" / "model.safetensors",
        root / "Resource" / "models" / "dinov2-small" / "preprocessor_config.json",
        root / "Resource" / "models" / "grounding-dino-tiny" / "config.json",
        root / "Resource" / "models" / "grounding-dino-tiny" / "model.safetensors",
        root / "Resource" / "models" / "grounding-dino-tiny" / "preprocessor_config.json",
        root / "Resource" / "models" / "grounding-dino-tiny" / "tokenizer.json",
        root / "Resource" / "clip_gallery" / "classes.json",
        root / "PAD_Lite" / "outputs" / "final" / "b1" / "final.pt",
        root / "PAD_Lite" / "outputs" / "final" / "b2" / "final.pt",
        root / "Resource" / "videos" / "Test_Video.mp4",
        root / "Resource" / "reference_images" / "IS-2.webp",
        root / "tools" / "ffmpeg" / "ffmpeg.exe",
        root / "tools" / "7zip" / "7z.exe",
        root / "tools" / "7zip" / "7z.dll",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    report["missing_files"] = missing
    if missing:
        failures.append("必要文件缺失。")

    writable: dict[str, bool] = {}
    for name in ("outputs", "logs", "tmp", "datasets", "runs"):
        directory = root / name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable[name] = True
        except OSError:
            writable[name] = False
            failures.append(f"目录不可写：{directory}")
    report["writable_directories"] = writable

    smi = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    report["nvidia_smi"] = smi.stdout.strip().splitlines()
    if smi.returncode != 0 or not smi.stdout.strip():
        failures.append("nvidia-smi不可用，未检测到Windows NVIDIA驱动。")
    else:
        first_driver = smi.stdout.splitlines()[0].split(",")[2].strip()
        report["driver_version"] = first_driver
        if version_tuple(first_driver) < (452, 39):
            failures.append("NVIDIA Windows驱动低于CUDA 11.8兼容下限452.39。")

    try:
        import torch

        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            failures.append("PyTorch无法访问CUDA GPU。")
        else:
            report["visible_cuda_devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            torch.ones(1, device="cuda")
        if not str(torch.version.cuda or "").startswith("11.8"):
            failures.append("便携PyTorch不是计划中的CUDA 11.8构建。")
    except (ImportError, OSError, RuntimeError) as exc:
        failures.append(f"PyTorch加载失败：{exc}。可尝试运行prerequisites/VC_redist.x64.exe。")

    if not missing and "torch" in locals():
        try:
            import torch
            from ultralytics import YOLO

            YOLO(str(root / "Resource" / "models" / "yolo" / "custom_yolo_best.pt"))
            YOLO(str(root / "Resource" / "models" / "yolo" / "yolo26n.pt"))
            report["yolo_models"] = "loaded"
        except Exception as exc:
            failures.append(f"YOLO模型加载失败：{exc}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            image = Image.new("RGB", (96, 96), color=(96, 112, 80))
            clip_dir = root / "Resource" / "models" / "clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)
            model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True).to(device).eval()
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                features = model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            if not torch.is_tensor(features):
                features = next(
                    (
                        getattr(features, name)
                        for name in ("image_embeds", "pooler_output", "last_hidden_state")
                        if getattr(features, name, None) is not None
                    ),
                    None,
                )
                if features is not None and features.ndim == 3:
                    features = features[:, 0]
            if features is None:
                raise RuntimeError("无法从CLIP输出提取图像特征")
            if features.shape[-1] <= 0:
                raise RuntimeError("CLIP输出维度无效")
            del features, inputs, model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            report["clip_model"] = "single_image_offline_inference_ok"
        except Exception as exc:
            failures.append(f"CLIP模型单图离线推理失败：{exc}")

        try:
            from transformers import AutoImageProcessor, Dinov2Model

            dino_dir = root / "Resource" / "models" / "dinov2-small"
            processor = AutoImageProcessor.from_pretrained(str(dino_dir), local_files_only=True)
            model = Dinov2Model.from_pretrained(
                str(dino_dir), local_files_only=True, use_safetensors=True
            ).to(device).eval()
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                output = model(pixel_values=inputs["pixel_values"].to(device))
            if output.last_hidden_state.shape[-1] <= 0:
                raise RuntimeError("DINOv2输出维度无效")
            del output, inputs, model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            report["dinov2_model"] = "single_image_offline_inference_ok"
        except Exception as exc:
            failures.append(f"DINOv2模型单图离线推理失败：{exc}")

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            grounding_dir = root / "Resource" / "models" / "grounding-dino-tiny"
            processor = AutoProcessor.from_pretrained(str(grounding_dir), local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(grounding_dir), local_files_only=True, use_safetensors=True
            ).to(device).eval()
            inputs = processor(images=image, text="tank.", return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            if output.pred_boxes.shape[-1] != 4:
                raise RuntimeError("Grounding DINO输出结构无效")
            del output, inputs, model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            report["grounding_dino_model"] = "single_image_offline_inference_ok"
        except Exception as exc:
            failures.append(f"Grounding DINO模型单图离线推理失败：{exc}")

        try:
            sys.path.insert(0, str(root / "app"))
            from target_video_search.algorithms import inspect_gallery_resources

            gallery = inspect_gallery_resources(root / "Resource" / "clip_gallery")
            report["clip_gallery"] = gallery
            if not gallery.get("ready") or int(gallery.get("class_count", 0)) != 10:
                failures.append("Gallery原型库未就绪或类别数不是10。")
        except Exception as exc:
            failures.append(f"Gallery原型库检查失败：{exc}")

    ffmpeg = shutil.which("ffmpeg")
    report["ffmpeg"] = ffmpeg
    if not ffmpeg:
        failures.append("未找到包内ffmpeg.exe。")
    else:
        encoders = run([ffmpeg, "-hide_banner", "-encoders"])
        report["nvenc_listed"] = "h264_nvenc" in (encoders.stdout + encoders.stderr)
        if not report["nvenc_listed"]:
            failures.append("FFmpeg未提供h264_nvenc编码器。")
        else:
            encode = run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=64x64:rate=1",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "h264_nvenc",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=60,
            )
            report["nvenc_encode_returncode"] = encode.returncode
            if encode.returncode != 0:
                failures.append("NVENC实际编码失败：" + encode.stderr.strip())

    seven_zip = shutil.which("7z")
    report["seven_zip"] = seven_zip
    if not seven_zip:
        failures.append("未找到包内7z.exe。")
    else:
        formats = run([seven_zip, "i"])
        format_text = formats.stdout + formats.stderr
        report["seven_zip_rar_supported"] = " Rar " in format_text and " Rar5 " in format_text
        if formats.returncode != 0 or not report["seven_zip_rar_supported"]:
            failures.append("包内7-Zip未提供RAR/RAR5读取后端。")

    port_text = os.environ.get("PORT", "7860")
    if port_text.isdigit() and 1 <= int(port_text) <= 65535:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", int(port_text)))
                report["port_available"] = True
            except OSError:
                report["port_available"] = False
                failures.append(f"端口{port_text}已被占用。")
    else:
        failures.append("PORT必须是1到65535之间的整数。")

    report["status"] = "ok" if not failures else "failed"
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
