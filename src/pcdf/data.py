"""配对 RGB/IR 数据集与 DataLoader。

设计要点
--------
Ultralytics 的 ``YOLODataset`` 已经实现了 YOLO-OBB 标签解析、矩形 letterbox
和几何一致的增强。我们只覆盖三个位置：

* ``_load_source``：读取同一 stem 的 IR/RGB，拼成 ``[B, G, R, IR, IR, IR]``；
* ``build_transforms``：使用矩形 ``imgsz=(H, W)``（论文 640×512），并跳过
  mosaic 与 HSV——mosaic 会强制方形画布，HSV 会改动红外通道；
* ``__getitem__`` / ``collate_fn``：把条件 id 随样本带入 batch。

图像和标签都来自 IR 目录；RGB 从同级 ``rgb`` 目录读取。论文 Sec. 5.2 说明
多模态检测器使用 IR 标签训练和评估。

两个 DataLoader 工厂
--------------------
* :func:`build_loader` 新建数据集 + loader，用于一次性场景（评估、自检）；
* :func:`epoch_loader` 复用已有数据集、按 ``seed + epoch`` 重建 loader，
  用于每个 epoch 固定样本顺序——断点续训要靠它复现同一批次的顺序。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Compose, LetterBox, RandomFlip, RandomPerspective
from ultralytics.data.dataset import YOLODataset

from .config import DataConfig
from .conditions import ConditionTable


class PairedRGBIRDataset(YOLODataset):
    """返回六通道图像与条件 id 的 YOLO-OBB 数据集。"""

    def __init__(
        self,
        img_path: str | Path,
        table: ConditionTable,
        data: dict[str, Any],
        imgsz: tuple[int, int] | int,
        augment: bool,
        hyp,
        batch_size: int,
        stride: int,
        pair_dir: str = "rgb",
        modality_dir: str = "ir",
        fraction: float = 1.0,
    ) -> None:
        # 这些属性必须在 super().__init__ 之前设置：父类会在构造期间调用
        # build_transforms()，其中会读取增强参数。
        self.condition_table = table
        self.pair_dir = pair_dir
        self.modality_dir = modality_dir
        self.augment = augment
        self.degrees = float(hyp.degrees)
        self.translate = float(hyp.translate)
        self.scale = float(hyp.scale)
        self.shear = float(hyp.shear)
        self.perspective = float(hyp.perspective)
        self.flip_p = float(hyp.flip_p)
        self.flip_vertical = float(hyp.flip_vertical)
        super().__init__(
            img_path=str(img_path),
            imgsz=imgsz,
            data=data,
            task="obb",
            augment=augment,
            hyp=hyp,
            rect=False,
            cache=False,
            batch_size=batch_size,
            stride=stride,
            pad=0.5,
            fraction=fraction,
        )

    # ------------------------------------------------------------------ 图像
    def _pair_path(self, ir_path: Path) -> Path:
        """把 IR 路径中的最后一个 ``ir`` 目录替换为 ``rgb``。"""
        parts = list(ir_path.parts)
        matches = [i for i, part in enumerate(parts) if part == self.modality_dir]
        if not matches:
            raise ValueError(
                f"无法从 {ir_path} 推导 RGB 路径：路径中没有 {self.modality_dir!r} 目录"
            )
        parts[matches[-1]] = self.pair_dir
        return Path(*parts)

    def _load_source(self, i: int) -> np.ndarray:
        """读取一对已配准图像并拼成 6 通道（BGR + 复制的 IR）。"""
        ir_path = Path(self.im_files[i])
        rgb_path = self._pair_path(ir_path)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None:
            raise FileNotFoundError(f"缺少 RGB 图像：{rgb_path}")
        if ir is None:
            raise FileNotFoundError(f"缺少 IR 图像：{ir_path}")
        if ir.shape[:2] != rgb.shape[:2]:
            # 正式数据已经配准；这里只做保险，避免静默的尺寸错位。
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Ultralytics 全程使用 BGR，预训练 backbone 也按 BGR 训练。
        return np.concatenate([rgb, np.repeat(ir[..., None], 3, axis=2)], axis=2)

    # -------------------------------------------------------------- 数据增强
    def build_transforms(self, hyp):
        """构造矩形输入下的几何增强 + 最终 OBB 格式化。"""
        height, width = (self.imgsz, self.imgsz) if isinstance(self.imgsz, int) else tuple(self.imgsz)
        if self.augment:
            transforms = [
                RandomPerspective(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear,
                    perspective=self.perspective,
                    size=(height, width),
                    preserve_obb=True,
                )
            ]
            if self.flip_vertical > 0:
                transforms.append(RandomFlip(p=self.flip_vertical, direction="vertical"))
            if self.flip_p > 0:
                transforms.append(RandomFlip(p=self.flip_p, direction="horizontal"))
            transforms.append(
                LetterBox(
                    new_shape=(height, width),
                    auto=False,
                    scaleup=False,
                    stride=self.stride,
                    padding_value=114,
                )
            )
            compose = Compose(transforms)
        else:
            compose = Compose(
                [
                    LetterBox(
                        new_shape=(height, width),
                        auto=False,
                        scaleup=False,
                        stride=self.stride,
                        padding_value=114,
                    )
                ]
            )
        # 与 Ultralytics YOLODataset 相同的收尾：张量化、xywhr、归一化、batch_idx。
        compose.append(
            self.format_class(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=0.0,  # 保持 BGR，不做通道交换
            )
        )
        return compose

    # ---------------------------------------------------------------- 条件 id
    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        stem = Path(self.labels[index]["im_file"]).stem
        sample["cond"] = torch.from_numpy(self.condition_table.ids(stem)).long()
        return sample

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """先取出条件 id，再复用 Ultralytics 对 OBB 标签的拼接逻辑。"""
        cond = torch.stack([sample.pop("cond") for sample in batch], dim=0)
        out = YOLODataset.collate_fn(batch)
        out["cond"] = cond
        return out


def _augmentation_hyp(cfg: DataConfig):
    """把配置里的增强参数写成 Ultralytics 的 hyp 对象。

    只覆盖用到的字段；mosaic / mixup / copy-paste / HSV 一律关闭并写明原因，
    避免 Ultralytics 默认值在未来版本里悄悄生效。
    """
    hyp = get_cfg()
    hyp.degrees = cfg.degrees
    hyp.translate = cfg.translate
    hyp.scale = cfg.scale
    hyp.shear = cfg.shear
    hyp.perspective = cfg.perspective
    hyp.flip_p = cfg.flip_p
    hyp.flip_vertical = cfg.flip_vertical
    hyp.mosaic = 0.0  # 会强制方形画布，破坏 640×512 的矩形输入
    hyp.mixup = 0.0
    hyp.cutmix = 0.0
    hyp.copy_paste = 0.0
    hyp.hsv_h = hyp.hsv_s = hyp.hsv_v = 0.0  # 会改动复制成 3 通道的 IR
    return hyp


def build_dataset(
    cfg: DataConfig,
    table: ConditionTable,
    split: str,
    augment: bool,
    *,
    root: str | Path,
    batch_size: int | None = None,
    fraction: float | None = None,
) -> PairedRGBIRDataset:
    """按划分构建配对数据集。

    Args:
        cfg: 数据配置（增强、类别名、输入尺寸）。
        table: 条件索引。
        split: ``train`` / ``val`` / ``test``。
        augment: 是否启用几何增强。
        root: 已绝对化的数据根目录（由 :class:`pcdf.paths` 解析后传入，
            数据层本身不负责路径策略）。
        batch_size: 仅用于 Ultralytics 的矩形推理启发式，不决定实际 batch。
        fraction: 采样比例覆盖（诊断/冒烟用）。
    """
    data_root = Path(root)
    img_path = data_root / cfg.modality_dir / "images" / split
    if not img_path.is_dir():
        raise FileNotFoundError(f"未找到划分图像目录：{img_path}")

    data = {
        "path": str(data_root / cfg.modality_dir),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "names": {i: name for i, name in enumerate(cfg.names)},
        "nc": cfg.nc,
        "channels": 6,
    }
    return PairedRGBIRDataset(
        img_path=img_path,
        table=table,
        data=data,
        imgsz=tuple(cfg.imgsz),
        augment=augment,
        hyp=_augmentation_hyp(cfg),
        batch_size=batch_size or cfg.batch,
        stride=32,
        pair_dir=cfg.pair_dir,
        modality_dir=cfg.modality_dir,
        fraction=cfg.fraction if fraction is None else fraction,
    )


def _clamp_workers(workers: int) -> int:
    """worker 数不能超过 CPU 核数，否则进程互相抢核反而更慢。"""
    cpu_count = os.cpu_count() or 0
    if cpu_count and workers > cpu_count:
        print(f"[data] workers={workers} 超过 CPU 核数 {cpu_count}，降为 {cpu_count}", flush=True)
        return cpu_count
    return max(0, workers)


def _make_loader(
    dataset: PairedRGBIRDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: str | torch.device,
    seed: int | None,
    persistent: bool = False,
) -> DataLoader:
    """构造 DataLoader 的唯一入口，保证各处的 pin_memory/generator 语义一致。"""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    n_workers = _clamp_workers(workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        collate_fn=dataset.collate_fn,
        pin_memory="cuda" in str(device),
        drop_last=False,
        persistent_workers=persistent and n_workers > 0,
        generator=generator,
    )


def build_loader(
    cfg: DataConfig,
    table: ConditionTable,
    split: str,
    augment: bool,
    *,
    root: str | Path,
    batch_size: int | None = None,
    workers: int | None = None,
    shuffle: bool | None = None,
    fraction: float | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> tuple[PairedRGBIRDataset, DataLoader]:
    """构建数据集与 DataLoader（一次性场景：评估、自检、冒烟）。"""
    dataset = build_dataset(
        cfg, table, split, augment, root=root, batch_size=batch_size, fraction=fraction
    )
    return dataset, _make_loader(
        dataset,
        batch_size=batch_size or cfg.batch,
        workers=cfg.workers if workers is None else workers,
        shuffle=augment if shuffle is None else shuffle,
        device=device,
        seed=seed,
    )


def epoch_loader(
    dataset: PairedRGBIRDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: str | torch.device,
    seed: int | None = None,
) -> DataLoader:
    """在已有数据集上重建 DataLoader，用于逐 epoch 固定随机顺序。

    断点续训时用同一个 ``seed + epoch`` 重建 loader，再跳过已训练的 step，
    即可在 mid-epoch 抢占后继续同一批样本顺序。
    """
    return _make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=shuffle,
        device=device,
        seed=seed,
    )


def batch_to_device(batch: dict, device: torch.device | str) -> dict:
    """把张量搬到设备，保留 ``im_file`` 等元数据在 CPU。"""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
