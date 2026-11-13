"""Synthetic RGB-IR pairs with oriented boxes.

Purpose: exercise the full pipeline (train -> degrade -> evaluate -> curve)
before the real ATR-UMOD data is available. Results on synthetic data are a
code-correctness check, NOT evidence for the research claim.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .degrade import fuse

SPLITS = ("train", "val")
VIEWS = ("rgb", "ir", "fused")


def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = float(rng.integers(60, 110))
    ramp = np.linspace(0.0, 28.0, h)[:, None]
    ripple = 8.0 * np.sin(np.linspace(0.0, 6.28, w))[None, :]
    noise = rng.normal(0.0, 6.0, (h, w))
    gray = np.clip(base + ramp + ripple + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw_targets(rgb: np.ndarray, ir: np.ndarray, rng: np.random.Generator) -> list[str]:
    h, w = rgb.shape[:2]
    labels: list[str] = []
    for _ in range(int(rng.integers(2, 7))):
        cx = int(rng.integers(60, w - 60))
        cy = int(rng.integers(50, h - 50))
        bw = int(rng.integers(36, 100))
        bh = int(rng.integers(18, 48))
        angle = float(rng.uniform(-40, 40))
        box = cv2.boxPoints(((cx, cy), (bw, bh), angle)).astype(np.float32)

        color = rng.integers(40, 230, 3)
        if rng.random() < 0.2:
            color = (120 + (color - 120) * 0.6).astype(int)
        cv2.fillPoly(rgb, [box.astype(np.int32)], tuple(int(c) for c in color))
        cv2.polylines(rgb, [box.astype(np.int32)], True, (25, 25, 25), 1, cv2.LINE_AA)

        hot = int(rng.integers(180, 245)) if rng.random() < 0.4 else int(rng.integers(70, 110))
        cv2.fillPoly(ir, [box.astype(np.int32)], hot)

        xs = np.clip(box[:, 0] / w, 0.0, 1.0)
        ys = np.clip(box[:, 1] / h, 0.0, 1.0)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        labels.append(f"0 {coords}")
    return labels


def _make_pair(size: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    w, h = size
    rgb = _background(h, w, rng)
    ir = cv2.cvtColor(_background(h, w, rng), cv2.COLOR_BGR2GRAY)
    ir = (ir.astype(np.float32) * 0.45).astype(np.uint8)

    labels = _draw_targets(rgb, ir, rng)

    rgb = cv2.GaussianBlur(rgb, (3, 3), 0.6)
    ir = cv2.GaussianBlur(ir, (9, 9), 2.2)
    ir = np.clip(ir.astype(np.float32) + rng.normal(0, 10.0, ir.shape), 0, 255).astype(np.uint8)
    return rgb, ir, labels


def build_synthetic(root: str | Path, n_train: int, n_val: int, size: list[int], seed: int = 0) -> Path:
    """Create data/synth/{rgb,ir,fused}/{images,labels}/{train,val}."""
    root = Path(root)
    rng = np.random.default_rng(seed)

    for view in VIEWS:
        for split in SPLITS:
            (root / view / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / view / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": n_train, "val": n_val}
    for split, n in counts.items():
        for i in range(n):
            rgb, ir, labels = _make_pair((size[0], size[1]), rng)
            stem = f"{i:06d}"
            cv2.imwrite(str(root / "rgb" / "images" / split / f"{stem}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(root / "ir" / "images" / split / f"{stem}.jpg"), ir, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(
                str(root / "fused" / "images" / split / f"{stem}.jpg"),
                fuse(rgb, ir, 0.5),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            text = "\n".join(labels) + ("\n" if labels else "")
            for view in VIEWS:
                (root / view / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")

    return root
