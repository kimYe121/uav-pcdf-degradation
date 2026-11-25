"""mosaic / HSV 增强的几何与通道正确性测试。"""

from __future__ import annotations

import numpy as np
import pytest

from pcdf.augment_obb import clip_polygon_rect, hsv_jitter_rgb, mosaic_canvas, polygon_area


def _square(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([
        [cx - w / 2, cy - h / 2],
        [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2],
        [cx - w / 2, cy + h / 2],
    ])


def test_polygon_area_of_unit_square() -> None:
    assert polygon_area(_square(0.5, 0.5, 1.0, 1.0)) == pytest.approx(1.0)


def test_clip_keeps_inner_polygon_untouched() -> None:
    poly = _square(5, 5, 2, 2)
    out = clip_polygon_rect(poly, 0, 0, 10, 10)
    assert out is not None
    assert polygon_area(out) == pytest.approx(4.0)


def test_clip_halves_a_straddling_polygon() -> None:
    """中心在 x=0 的 4×4 方块，跨过裁剪边界后只保留右半边（面积 8）。"""
    out = clip_polygon_rect(_square(0, 5, 4, 4), 0, 0, 10, 10)
    assert out is not None
    assert polygon_area(out) == pytest.approx(8.0)

    # 中心移到 x=-1：只剩 1 宽的条带
    out = clip_polygon_rect(_square(-1, 5, 4, 4), 0, 0, 10, 10)
    assert out is not None
    assert polygon_area(out) == pytest.approx(4.0)


def test_clip_outside_returns_none() -> None:
    assert clip_polygon_rect(_square(50, 50, 2, 2), 0, 0, 10, 10) is None


def test_clip_rotated_polygon_area_is_conserved_when_inside() -> None:
    ang = np.deg2rad(30)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    poly = _square(0, 0, 4, 2) @ rot.T + np.array([20.0, 20.0])
    out = clip_polygon_rect(poly, 0, 0, 40, 40)
    assert out is not None
    assert polygon_area(out) == pytest.approx(8.0, rel=1e-6)


def test_hsv_touches_only_rgb_channels() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (8, 8, 6), dtype=np.uint8)
    np.random.seed(0)
    out = hsv_jitter_rgb(img, h=0.5, s=0.5, v=0.5, p=1.0)
    assert out.shape == img.shape
    # IR 三通道必须一字不改
    assert np.array_equal(out[..., 3:], img[..., 3:])
    assert not np.array_equal(out[..., :3], img[..., :3])


def test_hsv_disabled_is_identity() -> None:
    img = np.zeros((4, 4, 6), dtype=np.uint8)
    assert hsv_jitter_rgb(img, 0.0, 0.0, 0.0, p=1.0) is img


def test_mosaic_geometry_and_labels() -> None:
    """4 张纯色图 + 每张一个居中方框，裁完必须仍在框内且类别顺序一致。"""
    H, W = 32, 40
    tiles = []
    for k in range(4):
        img = np.full((H, W, 6), 10 * (k + 1), dtype=np.uint8)
        poly = [_square(0.5, 0.5, 0.25, 0.25)]
        tiles.append((img, poly, np.array([float(k)])))

    rng = np.random.default_rng(0)
    crop, segs, cls = mosaic_canvas(tiles, (H, W), rng=rng)
    assert crop.shape == (H, W, 6)
    assert len(segs) == len(cls) <= 4
    for poly, c in zip(segs, cls):
        poly = np.asarray(poly)
        assert poly.min() >= -1e-6 and poly.max() <= 1 + 1e-6, "裁剪后坐标必须归一化到 [0,1]"
        assert polygon_area(poly) > 0
        # 类别与像素颜色必须对得上：取多边形中心处的像素，颜色 = 10*(c+1)
        cy, cx = int(poly[:, 1].mean() * H), int(poly[:, 0].mean() * W)
        cy = min(max(cy, 0), H - 1)
        cx = min(max(cx, 0), W - 1)
        assert crop[cy, cx, 0] == 10 * (int(c) + 1)


def test_mosaic_drops_targets_cut_away() -> None:
    """目标整体落在裁剪区外时必须被丢弃，不能留下一个空框。"""
    H, W = 32, 40
    tiles = []
    for k in range(4):
        img = np.full((H, W, 6), 10 * (k + 1), dtype=np.uint8)
        poly = [_square(0.02, 0.02, 0.02, 0.02)]  # 贴左上角的小框
        tiles.append((img, poly, np.array([float(k)])))
    for seed in range(20):
        _, segs, cls = mosaic_canvas(tiles, (H, W), rng=np.random.default_rng(seed))
        assert len(segs) <= 1, "贴角目标最多只剩一个"
        assert len(segs) == len(cls)


def test_mosaic_requires_four_tiles() -> None:
    with pytest.raises(ValueError, match="4 个 tile"):
        mosaic_canvas([(np.zeros((8, 8, 6), np.uint8), [], np.array([]))], (8, 8))