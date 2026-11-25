"""六通道（RGB+IR）配对数据的增强原语：HSV 抖动与 4 图拼贴（mosaic）。

为什么不用 Ultralytics 现成的 ``Mosaic``
----------------------------------------
现成实现强制方形画布、并且自己处理 ``xywhr``；我们的样本是 6 通道 RGB+IR
配对图，标签侧只有多边形（``segments``）与 AABB。直接复用会改坏 IR 通道或
标签几何，所以这里只提供纯函数原语，几何全部在多边形上做（不涉及角度
参数化），再交给 ``RandomPerspective(preserve_obb=True)`` 继续。

HSV 只作用于前 3 个通道：IR 是同一张灰度图复制成 3 份放在通道 3–5，如果一起
做 HSV，红外辐照度会被当成颜色改动，等于伪造模态信息。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["clip_polygon_rect", "polygon_area", "hsv_jitter_rgb", "mosaic_canvas"]


def polygon_area(poly: np.ndarray) -> float:
    """鞋带公式，返回绝对面积。``poly`` 为 ``(n, 2)``。"""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _clip_half_plane(poly: np.ndarray, axis: int, value: float, keep_greater: bool) -> np.ndarray:
    """Sutherland–Hodgman 的单条半平面裁剪。"""
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in = cur[axis] >= value if keep_greater else cur[axis] <= value
        nxt_in = nxt[axis] >= value if keep_greater else nxt[axis] <= value
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            denom = nxt[axis] - cur[axis]
            if abs(denom) > 1e-12:
                t = (value - cur[axis]) / denom
                out.append(cur + t * (nxt - cur))
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 2), dtype=np.float64)


def clip_polygon_rect(
    poly: np.ndarray, x0: float, y0: float, x1: float, y1: float
) -> np.ndarray | None:
    """把多边形裁到矩形 ``[x0, x1] × [y0, y1]``；完全在外或退化成线时返回 ``None``。"""
    out = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    for axis, value, keep_greater in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        out = _clip_half_plane(out, axis, value, keep_greater)
        if len(out) < 3:
            return None
    return out


def hsv_jitter_rgb(img: np.ndarray, h: float, s: float, v: float, p: float = 0.5) -> np.ndarray:
    """只对前 3 个通道（BGR）做 HSV 抖动，通道 3 及之后原样返回。

    ``img`` 为 ``(H, W, C)`` uint8，``C >= 3``；``h`` 是色相抖动比例（``0.015``
    对应 ±0.015×180°），``s``/``v`` 是乘性抖动幅度。
    """
    if max(h, s, v) <= 0 or np.random.rand() > p:
        return img
    out = np.asarray(img).copy()
    hsv = cv2.cvtColor(out[..., :3], cv2.COLOR_BGR2HSV)
    if h > 0:
        shift = np.random.uniform(-h, h) * 180.0  # OpenCV 8 位色相是 0–180
        hsv[..., 0] = np.mod(hsv[..., 0].astype(np.float32) + shift, 180.0)
    if s > 0:
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * np.random.uniform(1 - s, 1 + s), 0, 255)
    if v > 0:
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * np.random.uniform(1 - v, 1 + v), 0, 255)
    out[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def mosaic_canvas(
    tiles: list[tuple[np.ndarray, list[np.ndarray], np.ndarray]],
    out_hw: tuple[int, int],
    *,
    min_keep_ratio: float = 0.3,
    rng: np.random.Generator | None = None,
    tries: int = 3,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """4 图拼成 2×2 画布后随机裁一块 ``out_hw``，返回 ``(img, segments, cls)``。

    Args:
        tiles: 4 项 ``(img, polys, cls)``。``img`` 为 ``(h, w, C)`` uint8；
            ``polys`` 是归一化多边形列表（相对该 tile）；``cls`` 是该 tile 的
            类别数组（与 ``polys`` 一一对应）。
        out_hw: 目标 ``(H, W)``，即网络输入尺寸；画布为 ``(2H, 2W)``。
        min_keep_ratio: 裁剪后面积 / 裁剪前面积 低于该值的目标直接丢弃，
            避免把半截目标当完整目标教给网络。

    Returns:
        ``(crop, segments, cls)``：``crop`` 为 ``(H, W, C)``；``segments`` 是
        归一化到裁剪框内的多边形列表；``cls`` 为 ``(n,)`` float32。
    """
    if len(tiles) != 4:
        raise ValueError(f"mosaic 需要 4 个 tile，得到 {len(tiles)}")
    rng = rng or np.random.default_rng()
    H, W = int(out_hw[0]), int(out_hw[1])
    canvas = np.zeros((2 * H, 2 * W, tiles[0][0].shape[2]), dtype=np.uint8)
    offsets = [(0, 0), (W, 0), (0, H), (W, H)]

    polys_px: list[np.ndarray] = []
    cls_all: list[float] = []
    for (img, polys, cls), (dx, dy) in zip(tiles, offsets):
        if (img.shape[0], img.shape[1]) != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        canvas[dy : dy + H, dx : dx + W] = img
        cls = np.asarray(cls).reshape(-1)
        for poly, c in zip(polys, cls):
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2) * (W, H) + (dx, dy)
            polys_px.append(p)
            cls_all.append(float(c))

    x0 = 0.0
    y0 = 0.0
    crop = np.zeros((H, W, tiles[0][0].shape[2]), dtype=np.uint8)
    out_segs: list[np.ndarray] = []
    out_cls: list[float] = []
    # 目标都是"半张图大"的车辆，随机裁一次很容易把目标全裁掉。空标签样本
    # 太多会让模型偏向低召回，因此空结果时重抽裁剪位置（最多 tries 次）。
    for _ in range(max(1, tries)):
        x0 = float(rng.integers(0, W + 1))
        y0 = float(rng.integers(0, H + 1))
        crop = canvas[int(y0) : int(y0) + H, int(x0) : int(x0) + W].copy()
        out_segs, out_cls = [], []
        for poly, c in zip(polys_px, cls_all):
            area0 = polygon_area(poly)
            if area0 <= 1e-9:
                continue
            clipped = clip_polygon_rect(poly, x0, y0, x0 + W, y0 + H)
            if clipped is None:
                continue
            if polygon_area(clipped) / area0 < min_keep_ratio:
                continue
            out_segs.append((clipped - (x0, y0)) / (W, H))
            out_cls.append(c)
        if out_cls:
            break

    return crop, out_segs, np.asarray(out_cls, dtype=np.float32)