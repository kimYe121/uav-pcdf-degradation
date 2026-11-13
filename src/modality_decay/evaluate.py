"""Validation / evaluation wrappers."""

from __future__ import annotations

from pathlib import Path

from .datasets import build_degraded_view, write_data_yaml


def eval_weights(weights: str | Path, data_yaml: str | Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        batch=8,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
    )
    names = metrics.names if isinstance(metrics.names, dict) else {i: n for i, n in enumerate(metrics.names)}
    ap50 = metrics.box.ap50.tolist() if hasattr(metrics.box.ap50, "tolist") else list(metrics.box.ap50)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_ap50": {str(names[i]): float(v) for i, v in enumerate(ap50)},
    }


def prepare_and_eval(
    cfg,
    src_root: Path,
    protocol: str,
    level: float,
    weights: dict[str, Path],
    work_root: Path,
    split: str = "val",
) -> list[dict]:
    """Build one degraded split, then evaluate every model on its own view."""
    tag = f"{protocol}_s{int(round(level * 100)):03d}"
    out_root = work_root / tag
    build_degraded_view(
        src_root=src_root,
        out_root=out_root,
        split=split,
        protocol=protocol,
        level=level,
        alpha=cfg.fuse_alpha,
        p1_kind=cfg.p1_kind,
    )

    rows: list[dict] = []
    for view in ("rgb", "ir", "fused"):
        data_yaml = write_data_yaml(out_root / view)
        res = eval_weights(weights[view], data_yaml, cfg.imgsz, cfg.device)
        rows.append({"protocol": protocol, "level": level, "model": view, **res})
    return rows
