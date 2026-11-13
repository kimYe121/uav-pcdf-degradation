"""Dataset views: YOLO-OBB yaml files and degraded evaluation splits."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .degrade import degrade_pair

VIEWS = ("rgb", "ir", "fused")


def write_data_yaml(root: str | Path, names: list[str] | None = None) -> Path:
    root = Path(root).resolve()
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names or ["vehicle"])},
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def build_degraded_view(
    src_root: str | Path,
    out_root: str | Path,
    split: str,
    protocol: str,
    level: float,
    alpha: float = 0.5,
    p1_kind: str = "night",
) -> Path:
    """Materialize one degraded split with all three views (rgb/ir/fused)."""
    src_root, out_root = Path(src_root), Path(out_root)
    img_dir = src_root / "rgb" / "images" / split
    lbl_dir = src_root / "rgb" / "labels" / split
    ir_dir = src_root / "ir" / "images" / split

    for view in VIEWS:
        (out_root / view / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / view / "labels" / split).mkdir(parents=True, exist_ok=True)

    file_ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    for i, stem in enumerate(file_ids):
        rgb = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_dir / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None:
            continue
        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        outs = degrade_pair(rgb, ir, protocol, level, alpha=alpha, p1_kind=p1_kind, seed=1000 + i)
        for view in VIEWS:
            cv2.imwrite(
                str(out_root / view / "images" / split / f"{stem}.jpg"),
                outs[view],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            label_src = lbl_dir / f"{stem}.txt"
            label_dst = out_root / view / "labels" / split / f"{stem}.txt"
            label_dst.write_text(label_src.read_text(encoding="utf-8") if label_src.exists() else "", encoding="utf-8")

    return out_root
