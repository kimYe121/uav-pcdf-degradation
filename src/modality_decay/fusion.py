"""Six-channel early-fusion control, not PCDF or dual-stream feature fusion.

The input layout is [R, G, B, IR, IR, IR]. No B-channel replacement is used.
Ultralytics supplies the OBB model/loss/metrics; the paired data pipeline and
six-channel weight inflation are implemented here. This preliminary control
must not be reported as a reproduction of the PCDF architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb import OBBTrainer, OBBValidator
from ultralytics.utils import IterableSimpleNamespace, colorstr
from ultralytics.utils.torch_utils import unwrap_model


def _ir_path(rgb_path: str) -> Path:
    path = Path(rgb_path)
    parts = list(path.parts)
    try:
        index = next(i for i, part in enumerate(parts) if part.lower() == "rgb")
    except StopIteration as exc:
        raise ValueError(f"RGB path must contain an 'rgb' directory: {path}") from exc
    parts[index] = "ir"
    return Path(*parts)


class RGBIRDataset(YOLODataset):
    """Paired RGB/IR dataset returning six channels with synchronized transforms."""

    def _load_source(self, i: int) -> np.ndarray:
        rgb_path = Path(self.im_files[i])
        ir_path = _ir_path(str(rgb_path))
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or ir is None:
            raise FileNotFoundError(f"Missing RGB/IR pair: {rgb_path} | {ir_path}")
        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(f"Unaligned image sizes: {rgb_path} {rgb.shape} vs {ir_path} {ir.shape}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.concatenate((rgb, np.repeat(ir[..., None], 3, axis=2)), axis=2)

    def build_transforms(self, hyp: IterableSimpleNamespace):
        # The default Albumentations/HSV ops are RGB-only. Geometry, mosaic and
        # flips still operate on all six channels, preserving RGB/IR alignment.
        transforms = super().build_transforms(hyp)
        transforms.transforms = [
            transform for transform in transforms.transforms
            if transform.__class__.__name__ not in {"Albumentations", "RandomHSV"}
        ]
        return transforms


def build_rgbir_dataset(
    args: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str,
    rect: bool,
    stride: int,
) -> RGBIRDataset:
    data = dict(data)
    data["channels"] = 6
    return RGBIRDataset(
        img_path=img_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=rect,
        cache=False,
        single_cls=args.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task="obb",
        classes=args.classes,
        data=data,
        fraction=args.fraction,
        tile=False,
    )


class EarlyFusionTrainer(OBBTrainer):
    """OBB trainer with a paired six-channel data pipeline."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> RGBIRDataset:
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_rgbir_dataset(self.args, img_path, batch, self.data, mode, mode == "val", stride)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = super().get_model(cfg, weights, verbose)
        if weights is not None:
            source = weights.state_dict().get("model.0.conv.weight")
            target = model.state_dict().get("model.0.conv.weight")
            if source is not None and target is not None and source.shape[1] == 3 and target.shape[1] == 6:
                with torch.no_grad():
                    infrared_init = source.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                    model.model[0].conv.weight.copy_(torch.cat((source, infrared_init), dim=1))
                print("[init] inflated first convolution: RGB(3) + mean-RGB -> IR(3)", flush=True)
        return model


class EarlyFusionValidator(OBBValidator):
    """Standalone validator using the same six-channel paired input pipeline."""

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> RGBIRDataset:
        return build_rgbir_dataset(self.args, img_path, batch, self.data, mode, False, self.stride)


def write_early_fusion_yaml(data_root: str | Path) -> Path:
    """Create a data descriptor whose images remain in the existing RGB folder."""
    root = Path(data_root).resolve()
    source = yaml.safe_load((root / "rgb" / "data.yaml").read_text(encoding="utf-8"))
    out_dir = root / "early6"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(root / "rgb"),
        "train": "images/train",
        "val": "images/val",
        "names": source["names"],
        "channels": 6,
    }
    path = out_dir / "data.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def metrics_dict(validator) -> dict[str, Any]:
    metrics = validator.metrics
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_ap50": {str(names[i]): float(v) for i, v in enumerate(metrics.box.ap50)},
    }
