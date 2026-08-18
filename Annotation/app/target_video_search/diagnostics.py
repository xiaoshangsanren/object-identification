from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip()
    return output if output else f"exit code {result.returncode}"


def _resource_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
    }
    if path.is_file():
        result["type"] = "file"
        result["size_bytes"] = path.stat().st_size
    elif path.is_dir():
        result["type"] = "directory"
    return result


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value) if value else default


def collect_runtime_diagnostics() -> dict[str, Any]:
    package_root = _env_path("PACKAGE_ROOT", Path.cwd())
    model_root = _env_path("TVS_MODEL_ROOT", Path("/app/models"))
    input_root = _env_path("TVS_INPUT_ROOT", Path("/app/inputs"))
    reference_root = _env_path("TVS_REFERENCE_ROOT", Path("/app/reference_images"))
    train_data_root = _env_path("TVS_TRAIN_DATA_ROOT", Path("/app/train_data"))
    gallery_root = _env_path("TVS_CLIP_GALLERY_ROOT", Path("/app/clip_gallery"))
    detector_model = Path(
        os.getenv("TVS_DETECTOR_MODEL", str(model_root / "yolo" / "yolo26n.pt"))
    )
    clip_dir = Path(
        os.getenv(
            "TVS_CLIP_LOCAL_DIR",
            str(model_root / "clip-vit-base-patch32"),
        )
    )
    dino_dir = _env_path("TVS_DINO_LOCAL_DIR", model_root / "dinov2-small")
    grounding_dir = _env_path(
        "TVS_GROUNDING_MODEL_DIR", model_root / "grounding-dino-tiny"
    )
    pad_lite_root = _env_path(
        "TVS_PAD_LITE_CHECKPOINT_ROOT",
        package_root / "PAD_Lite" / "outputs" / "final",
    )

    required_resources = {
        "default_detector": _resource_status(detector_model),
        "custom_yolo": _resource_status(model_root / "yolo" / "custom_yolo_best.pt"),
        "custom_yolo_metadata": _resource_status(
            model_root / "yolo" / "custom_yolo_best.json"
        ),
        "clip_pytorch": _resource_status(clip_dir / "pytorch_model.bin"),
        "dinov2_model": _resource_status(dino_dir / "model.safetensors"),
        "dinov2_config": _resource_status(dino_dir / "config.json"),
        "grounding_model": _resource_status(grounding_dir / "model.safetensors"),
        "grounding_config": _resource_status(grounding_dir / "config.json"),
        "gallery_classes": _resource_status(gallery_root / "classes.json"),
        "gallery_prototypes": _resource_status(gallery_root / "prototypes"),
        "gallery_negatives": _resource_status(gallery_root / "negatives"),
        "gallery_unknown_tanks": _resource_status(gallery_root / "unknown_tanks"),
        "pad_lite_b1_final": _resource_status(pad_lite_root / "b1" / "final.pt"),
        "pad_lite_b2_final": _resource_status(pad_lite_root / "b2" / "final.pt"),
        "test_video": _resource_status(input_root / "Test_Video.mp4"),
        "reference_images": _resource_status(reference_root),
    }
    optional_resources = {"train_data": _resource_status(train_data_root)}
    resources = {**required_resources, **optional_resources}
    missing_resources = [
        name for name, item in required_resources.items() if not bool(item["exists"])
    ]

    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )

    ffmpeg_version = _command_output(["ffmpeg", "-version"]).splitlines()[0]
    ffmpeg_encoders = _command_output(["ffmpeg", "-hide_banner", "-encoders"])
    nvidia_smi = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )

    if missing_resources:
        status = "资源不完整"
    elif not cuda_available:
        status = "GPU不可用"
    else:
        status = "可运行"

    return {
        "status": status,
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "container": Path("/.dockerenv").exists(),
            "working_directory": str(Path.cwd()),
            "package_root": str(package_root),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "torchvision",
                "ultralytics",
                "transformers",
                "gradio",
                "opencv-python",
                "numpy",
            )
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "current_device": torch.cuda.current_device() if cuda_available else None,
            "devices": devices,
            "nvidia_smi": nvidia_smi.splitlines(),
        },
        "media": {
            "ffmpeg": ffmpeg_version,
            "nvenc_available": "h264_nvenc" in ffmpeg_encoders,
        },
        "resources": resources,
        "missing_resources": missing_resources,
        "writable_directories": {
            str(path): path.is_dir() and os.access(path, os.W_OK)
            for path in (
                _env_path("TVS_OUTPUT_ROOT", package_root / "outputs"),
                _env_path("LOG_DIR", package_root / "logs"),
                _env_path("GRADIO_TEMP_DIR", package_root / "tmp"),
                _env_path("TVS_DATASET_ROOT", package_root / "datasets"),
                _env_path("TVS_RUNS_ROOT", package_root / "runs"),
            )
        },
    }
