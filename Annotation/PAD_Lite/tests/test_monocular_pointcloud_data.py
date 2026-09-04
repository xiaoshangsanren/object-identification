from __future__ import annotations

from pathlib import Path

import numpy as np

from PAD_Lite.src.build_monocular_pointcloud_dataset import (
    depth_to_fixed_pointcloud,
    deterministic_pixel_indices,
    write_binary_ply,
)


def test_deterministic_sampler_is_unique_and_exact() -> None:
    first = deterministic_pixel_indices(37, 61, 512)
    second = deterministic_pixel_indices(37, 61, 512)
    assert len(first) == 512
    assert len(np.unique(first)) == 512
    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < 37 * 61


def test_pointcloud_padding_and_shapes_are_training_ready() -> None:
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(9, dtype=np.uint8)[None, :]
    depth = np.linspace(-2.0, 3.0, 72, dtype=np.float32).reshape(8, 9)
    cloud = depth_to_fixed_pointcloud(rgb, depth, 128)
    assert cloud["xyz"].shape == (128, 3)
    assert cloud["rgb"].shape == (128, 3)
    assert cloud["pixel_xy"].shape == (128, 2)
    assert cloud["valid_mask"].shape == (128,)
    assert int(cloud["valid_count"]) == 72
    assert int(cloud["valid_mask"].sum()) == 72
    assert np.isfinite(cloud["xyz"]).all()
    assert np.all(cloud["xyz"][72:] == 0)


def test_binary_ply_contains_only_valid_points(tmp_path: Path) -> None:
    xyz = np.asarray([[0, 0, 0], [1, 2, 3], [9, 9, 9]], dtype=np.float32)
    rgb = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    mask = np.asarray([True, True, False])
    path = tmp_path / "sample.ply"
    write_binary_ply(path, xyz, rgb, mask)
    raw = path.read_bytes()
    assert b"format binary_little_endian 1.0" in raw
    assert b"element vertex 2" in raw
    assert len(raw) > 2 * 15
