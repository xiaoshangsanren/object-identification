from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import DetectionConfig

REQUIRED_DINO_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)


def inspect_dino_model(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir)
    missing = [name for name in REQUIRED_DINO_FILES if not (root / name).is_file()]
    return {
        "root": str(root),
        "ready": root.is_dir() and not missing,
        "required_files": list(REQUIRED_DINO_FILES),
        "missing": missing,
    }


class DinoEmbedder:
    backend = "dinov2"

    def __init__(self, config: DetectionConfig) -> None:
        import torch
        from transformers import AutoImageProcessor, Dinov2Model

        self.torch = torch
        self.device = self._choose_device(config.device)
        self.model_dir = Path(config.dino_local_dir).expanduser()
        status = inspect_dino_model(self.model_dir)
        if not status["ready"]:
            missing = ", ".join(status["missing"]) or "模型目录不存在"
            raise RuntimeError(
                f"DINOv2 本地模型资源不完整: {self.model_dir} ({missing})"
            )
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        self.model = Dinov2Model.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def _choose_device(self, configured: str) -> str:
        if configured and configured.lower() != "auto":
            return configured
        return "cuda:0" if self.torch.cuda.is_available() else "cpu"

    def encode_images(self, images: list[Image.Image], batch_size: int) -> Any:
        if not images:
            return None
        features = []
        resolved_batch_size = max(1, int(batch_size))
        with self.torch.inference_mode():
            for start in range(0, len(images), resolved_batch_size):
                batch = images[start : start + resolved_batch_size]
                inputs = self.processor(
                    images=[image.convert("RGB") for image in batch],
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(pixel_values=inputs["pixel_values"])
                batch_features = outputs.last_hidden_state[:, 0].float()
                batch_features = batch_features / batch_features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-6)
                features.append(batch_features.cpu())
        return self.torch.cat(features, dim=0)

    def encode_text(self, text: str | None) -> None:
        del text
        return None
