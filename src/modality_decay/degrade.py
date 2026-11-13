"""Modality degradation operators: P1 contamination, P2 zeroing, P3 misregistration."""

from __future__ import annotations

import cv2
import numpy as np

P1_KINDS = ("night", "fog", "blur", "noise")


def fuse(rgb: np.ndarray, ir: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Fixed-weight input-level fusion (pilot proxy for a two-stream baseline)."""
    ir3 = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR) if ir.ndim == 2 else ir
    return cv2.addWeighted(rgb, alpha, ir3, 1.0 - alpha, 0)


def _p1(img: np.ndarray, s: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    if s <= 0:
        return img.copy()
    x = img.astype(np.float32)

    if kind == "night":
        x = 255.0 * np.power(x / 255.0, 1.0 + 2.2 * s)
        x = x + rng.normal(0.0, 32.0 * s, x.shape)
    elif kind == "fog":
        t = 1.0 - 0.85 * s
        x = x * t + 215.0 * (1.0 - t)
        x = (x - 128.0) * (1.0 - 0.45 * s) + 128.0
    elif kind == "blur":
        sigma = 6.0 * s
        x = cv2.GaussianBlur(x, (0, 0), sigma)
    elif kind == "noise":
        x = x + rng.normal(0.0, 45.0 * s, x.shape)
    else:
        raise ValueError(f"unknown P1 kind: {kind}")

    return np.clip(x, 0, 255).astype(np.uint8)


def _p2(img: np.ndarray, s: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * (1.0 - s), 0, 255).astype(np.uint8)


def _p3(ir: np.ndarray, s: float, max_px: float = 20.0) -> np.ndarray:
    """Break RGB-IR registration by translating the IR frame (translation only).

    A pure translation keeps every single-modality image fully usable (its own
    quality is untouched) while destroying cross-modal correspondence.
    """
    if s <= 0:
        return ir.copy()
    h, w = ir.shape[:2]
    dx = max_px * s
    dy = 0.3 * max_px * s
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(ir, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def degrade_pair(
    rgb: np.ndarray,
    ir: np.ndarray,
    protocol: str,
    level: float,
    alpha: float = 0.5,
    p1_kind: str = "night",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return the three model inputs (rgb / ir / fused) for one protocol-level."""
    rng = np.random.default_rng(seed)
    rgb_out, ir_out = rgb.copy(), ir.copy()

    if protocol == "P1":
        rgb_out = _p1(rgb, level, p1_kind, rng)
    elif protocol == "P2":
        rgb_out = _p2(rgb, level)
    elif protocol == "P3":
        ir_out = _p3(ir, level)
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    return {
        "rgb": rgb_out,
        "ir": ir_out,
        "fused": fuse(rgb_out, ir_out, alpha),
    }
