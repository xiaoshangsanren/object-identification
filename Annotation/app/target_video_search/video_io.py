from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class BaseVideoWriter:
    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVVideoWriter(BaseVideoWriter):
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"OpenCV cannot open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class FfmpegNvencWriter(BaseVideoWriter):
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "ll",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self._process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg/NVENC exited with code {return_code}")


def has_nvenc_encoder() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return False
    return "h264_nvenc" in (result.stdout + result.stderr)


def make_video_writer(
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
    use_nvenc: bool = True,
) -> BaseVideoWriter:
    if use_nvenc and has_nvenc_encoder():
        return FfmpegNvencWriter(output_path, fps, width, height)
    return OpenCVVideoWriter(output_path, fps, width, height)
