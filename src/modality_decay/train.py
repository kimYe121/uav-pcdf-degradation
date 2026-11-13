"""Train one detector per modality view."""

from __future__ import annotations

from pathlib import Path

from .core import Config


def train_view(cfg: Config, data_yaml: str | Path, tag: str, fallback_yaml: str = "yolo11n-obb.yaml") -> Path:
    from ultralytics import YOLO

    try:
        model = YOLO(cfg.model)
    except Exception as exc:  # noqa: BLE001 - offline fallback
        print(f"[warn] cannot load {cfg.model} ({exc}); training from scratch with {fallback_yaml}")
        model = YOLO(fallback_yaml)

    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        amp=True,
        seed=cfg.seed,
        deterministic=True,
        project=str(cfg.path(cfg.runs_dir)),
        name=f"{cfg.name}_{tag}",
        exist_ok=True,
        plots=False,
        verbose=False,
        val=True,
    )
    return Path(model.trainer.save_dir) / "weights" / "best.pt"
