from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PIL import Image

from .config import (
    PAD_LITE_B0_ALGORITHM,
    PAD_LITE_B1_ALGORITHM,
    PAD_LITE_B2_ALGORITHM,
    DetectionConfig,
)


ALGORITHM_TO_VARIANT = {
    PAD_LITE_B0_ALGORITHM: "b0",
    PAD_LITE_B1_ALGORITHM: "b1",
    PAD_LITE_B2_ALGORITHM: "b2",
}


def inspect_pad_lite_model(
    checkpoint_root: str | Path,
    algorithm: str,
) -> dict[str, Any]:
    variant = ALGORITHM_TO_VARIANT.get(str(algorithm))
    root = Path(checkpoint_root).expanduser()
    checkpoint = root / str(variant) / "final.pt" if variant in {"b1", "b2"} else None
    missing: list[str] = []
    if variant is None:
        missing.append(f"unsupported algorithm: {algorithm}")
    elif checkpoint is not None and not checkpoint.is_file():
        missing.append(str(checkpoint))
    return {
        "algorithm": algorithm,
        "variant": variant,
        "checkpoint_root": str(root),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "ready": not missing,
        "missing": missing,
    }


class PADLiteEmbedder:
    """Image-only PAD-Lite deployment backend used by the Gallery matcher."""

    def __init__(self, config: DetectionConfig) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.algorithm = str(config.algorithm)
        try:
            self.variant = ALGORITHM_TO_VARIANT[self.algorithm]
        except KeyError as exc:
            raise ValueError(f"Unsupported PAD-Lite algorithm: {self.algorithm}") from exc
        self.backend = f"pad_lite_{self.variant}"
        self.device = self._choose_device(config.device)
        self.half_cuda = bool(config.half and self.device.startswith("cuda"))
        self.model_dir = Path(config.clip_local_dir).expanduser()
        if not self.model_dir.is_dir():
            raise RuntimeError(f"PAD-Lite CLIP model directory is missing: {self.model_dir}")
        self.processor = CLIPProcessor.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )
        clip_model = CLIPModel.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )

        annotation_root = Path(__file__).resolve().parents[2]
        if str(annotation_root) not in sys.path:
            sys.path.insert(0, str(annotation_root))
        from PAD_Lite.models import FrozenClipEncoder, RetrievalModel

        self.checkpoint_path: Path | None = None
        self.class_names: list[str] = []
        if self.variant == "b0":
            self.model = FrozenClipEncoder(clip_model)
        else:
            status = inspect_pad_lite_model(
                config.pad_lite_checkpoint_root,
                self.algorithm,
            )
            if not status["ready"]:
                raise RuntimeError(
                    f"PAD-Lite {self.variant.upper()} Final checkpoint is missing: "
                    + ", ".join(status["missing"])
                )
            self.checkpoint_path = Path(str(status["checkpoint"]))
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("format") != "annotation_pad_lite_adapter_v1":
                raise RuntimeError(f"Unsupported PAD-Lite checkpoint: {self.checkpoint_path}")
            if checkpoint.get("variant") != self.variant:
                raise RuntimeError(
                    f"Checkpoint variant mismatch: expected {self.variant}, "
                    f"got {checkpoint.get('variant')}"
                )
            if checkpoint.get("training_scope") != "all_russian_classes":
                raise RuntimeError(
                    "Deployment requires a Final checkpoint trained on all Russian classes"
                )
            self.class_names = [str(name) for name in checkpoint["base_classes"]]
            projection = checkpoint["model_state"].get("projection.weight")
            if projection is None or projection.ndim != 2:
                raise RuntimeError("Checkpoint has no valid projection.weight")
            self.model = RetrievalModel(
                clip_model,
                num_classes=len(self.class_names),
                embedding_dim=int(projection.shape[0]),
                dropout=0.0,
            )
            self.model.configure_backbone(last_n_blocks=2, enabled=False)
            incompatible = self.model.load_state_dict(
                checkpoint["model_state"],
                strict=False,
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected PAD-Lite checkpoint keys: {incompatible.unexpected_keys}"
                )

        self.model.to(self.device).eval()
        if self.half_cuda:
            self.model.half()

    def _choose_device(self, configured: str) -> str:
        if configured and configured.lower() != "auto":
            return configured
        return "cuda:0" if self.torch.cuda.is_available() else "cpu"

    def _autocast(self):
        if not self.half_cuda:
            return nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=self.torch.float16)

    def encode_images(self, images: list[Image.Image], batch_size: int) -> Any:
        if not images:
            return None
        batches = []
        resolved_batch_size = max(1, int(batch_size))
        with self.torch.inference_mode():
            for start in range(0, len(images), resolved_batch_size):
                batch = images[start : start + resolved_batch_size]
                inputs = self.processor(
                    images=[image.convert("RGB") for image in batch],
                    return_tensors="pt",
                ).to(self.device)
                pixel_values = inputs["pixel_values"]
                if self.half_cuda:
                    pixel_values = pixel_values.half()
                with self._autocast():
                    if self.variant == "b0":
                        features = self.model(pixel_values)
                    else:
                        features = self.model.encode(pixel_values)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                batches.append(features.cpu())
        return self.torch.cat(batches, dim=0)

    def encode_text(self, text: str | None) -> None:
        del text
        return None

    def summary_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "variant": self.variant,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "training_classes": list(self.class_names),
            "text_branch_used_at_inference": False,
        }
