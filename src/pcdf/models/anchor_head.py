"""论文式 8 的 anchor-based OBB 检测头（YOLOv5 风格）。

依据
----
* 式 8：``L_dc = Σ(L_cls + L_reg + L_obj)`` —— 有 objectness 分支，说明是
  anchor-based 头（Ultralytics 的 v8/v11 OBB 头是 anchor-free，没有 obj）；
* 一作公开 fork 了 YOLOv5 v7.0（零改动），骨架与训练框架来自 v5；
* Sec. 5.2：所有检测器统一接 OBB head，说明这个头是通用可插拔的。

与 v5 原生头的差异（均为必要适配）
--------------------------------
1. 回归量是 5 维 ``(x, y, w, h, θ)``，θ 用 ``(2σ−1)·π/2`` 覆盖 ``(−90°, 90°)``；
2. 框损失用 ProbIoU（CTIoU/KLD 的近似），一次管住位置与角度，避免角度周期
   跳变；这也是评估器使用的同一度量；
3. 目标分配沿用 v5 的"中心格 + 四邻域 + 宽高比阈值"，但宽高比按排序后的
   边长比较，避免旋转框 ``w/h`` 互换导致漏配。

输出布局与 Ultralytics 对齐：``(B, 4 + nc + 1, N)``，通道依次是
``x, y, w, h, cls_0..cls_{nc-1}, θ``，其中 ``cls`` 已乘 objectness——这样
``non_max_suppression(..., rotated=True)`` 与评估器无需改动。

可配置项（锚框、损失增益、匹配阈值）全部来自 ``ModelConfig``；本模块不再
内置"改一行才生效"的魔数。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from ultralytics.utils.metrics import probiou

from ..constants import DEFAULT_ANCHORS, STRIDES
from ..config import LossConfig

# 中心格 + 四邻域，与 YOLOv5 的正样本扩增一致。
NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class AnchorOBBHead(nn.Module):
    """三尺度 anchor-based 旋转框头。

    Args:
        nc: 类别数。
        ch: 三个尺度的输入通道数。
        strides: 特征金字塔步长，默认 :data:`pcdf.constants.STRIDES`。
        anchors: ``{stride: [(w, h), ...]}``；默认 k-means 结果。
    """

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int, int],
        strides: tuple[int, ...] = STRIDES,
        anchors: dict[int, list[tuple[int, int]]] | None = None,
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.strides = tuple(int(s) for s in strides)
        self.na = 3
        # 每格预测：tx, ty, tw, th, tθ, obj, cls...
        self.no = 5 + 1 + self.nc
        anchors = anchors or DEFAULT_ANCHORS
        flat = [wh for stride in self.strides for wh in anchors[stride]]
        if len(flat) != self.na * len(self.strides):
            raise ValueError(
                f"每个尺度必须给 {self.na} 个锚框，当前共 {len(flat)} 个"
            )
        self.register_buffer("anchor_grid", torch.tensor(flat, dtype=torch.float32))
        self.register_buffer("stride_grid", torch.tensor(self.strides, dtype=torch.float32))
        # 与 YOLOv5 的 Detect 一致：1×1 裸卷积（无 BN、有 bias），
        # 这样 objectness 先验才能写进 bias。
        self.convs = nn.ModuleList([nn.Conv2d(c, self.na * self.no, 1) for c in ch])
        self.init_bias()

    def init_bias(self) -> None:
        """objectness 先验压到 0.02，避免训练初期正负样本极度失衡。

        必须在模型级 ``initialize_weights`` 之后调用一次——Ultralytics 的
        初始化会把卷积 bias 清零，否则先验会被覆盖。
        """
        for conv in self.convs:
            with torch.no_grad():
                view = conv.bias.view(self.na, self.no)
                view[:, 4].fill_(-3.9)  # sigmoid(-3.9) ≈ 0.02
                view[:, 5:].fill_(0.0)

    @property
    def stride(self) -> Tensor:
        return self.stride_grid

    # ---------------------------------------------------------------- 前向
    def forward(self, feats: list[Tensor]) -> list[Tensor]:
        """逐尺度 1×1 卷积，返回三个原始输出张量。"""
        return [conv(f) for conv, f in zip(self.convs, feats)]

    def split(self, raw: Tensor, index: int) -> Tensor:
        """把 ``(B, na*no, H, W)`` 拆成 ``(B, na, H, W, no)``，便于按格取用。"""
        b, _, h, w = raw.shape
        return raw.view(b, self.na, self.no, h, w).permute(0, 1, 3, 4, 2)

    def _grid(self, height: int, width: int, stride: int, dtype: torch.dtype,
              device: torch.device) -> tuple[Tensor, Tensor]:
        """生成网格中心坐标与锚框，供逐格解码复用。"""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        scale_index = self.strides.index(stride)
        anchor = self.anchor_grid[
            scale_index * self.na : (scale_index + 1) * self.na
        ].to(dtype)
        return torch.stack([grid_x, grid_y], dim=-1), anchor

    def decode(self, raws: list[Tensor]) -> Tensor:
        """把各尺度原始输出拼成 ``(B, 4+nc+1, N)``，供 NMS / 评估使用。"""
        outs: list[Tensor] = []
        for index, (raw, stride) in enumerate(zip(raws, self.strides)):
            b, _, h, w = raw.shape
            p = self.split(raw, index)
            device, dtype = p.device, p.dtype
            grid, anchor = self._grid(h, w, stride, dtype, device)

            xy = (p[..., 0:2].sigmoid() * 2 - 0.5 + grid) * stride
            wh = (p[..., 2:4].sigmoid() * 2).pow(2) * anchor[None, :, None, None, :]
            obj = p[..., 4].sigmoid()
            cls = p[..., 5 : 5 + self.nc].sigmoid() * obj.unsqueeze(-1)
            theta = (p[..., 5 + self.nc].sigmoid() * 2 - 1) * (math.pi / 2)

            out = torch.cat([xy, wh, cls, theta.unsqueeze(-1)], dim=-1)  # (B,na,H,W,5+nc)
            # NMS 要求通道为预测维度：把 anchor 维并进空间维。
            outs.append(out.permute(0, 4, 1, 2, 3).reshape(b, 5 + self.nc, self.na * h * w))
        return torch.cat(outs, dim=2)

    def grid_boxes(self, raw: Tensor, index: int) -> Tensor:
        """逐格解码出 ``(B, na, H, W, 5)`` 的 ``xywhr``（不乘 objectness）。"""
        return self.boxes_from_logits(self.split(raw, index), index)

    def boxes_from_logits(self, logits: Tensor, index: int) -> Tensor:
        """在已拆分的 logits 上解码，避免重复 ``view/permute``。"""
        b, na, h, w, _ = logits.shape
        stride = self.strides[index]
        grid, anchor = self._grid(h, w, stride, logits.dtype, logits.device)
        xy = (logits[..., 0:2].sigmoid() * 2 - 0.5 + grid) * stride
        wh = (logits[..., 2:4].sigmoid() * 2).pow(2) * anchor[None, :, None, None, :]
        theta = (logits[..., 5 + self.nc].sigmoid() * 2 - 1) * (math.pi / 2)
        return torch.cat([xy, wh, theta.unsqueeze(-1)], dim=-1)


class AnchorOBBLoss(nn.Module):
    """式 8：``L_obj + L_cls + L_reg``（``L_reg`` 用 ``1 − ProbIoU``）。

    增益与匹配阈值来自 :class:`pcdf.config.LossConfig`，默认值与论文
    "anchor 头 + ProbIoU" 的量纲匹配。
    """

    def __init__(
        self,
        head: AnchorOBBHead,
        *,
        box_gain: float = 0.05,
        cls_gain: float = 0.5,
        obj_gain: float = 1.0,
        anchor_thr: float = 0.25,
    ) -> None:
        super().__init__()
        self.head = head
        self.box_gain = float(box_gain)
        self.cls_gain = float(cls_gain)
        self.obj_gain = float(obj_gain)
        self.anchor_thr = float(anchor_thr)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @classmethod
    def from_config(cls, head: AnchorOBBHead, loss: LossConfig) -> "AnchorOBBLoss":
        """按配置构造，避免调用方各自记默认值。"""
        return cls(
            head,
            box_gain=loss.anchor_box,
            cls_gain=loss.anchor_cls,
            obj_gain=loss.anchor_obj,
            anchor_thr=loss.anchor_thr,
        )

    def forward(self, preds, batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """计算损失。

        Args:
            preds: ``(decoded, raws)`` 或直接 ``raws``；训练循环传 head 原始输出。
            batch: Ultralytics 风格的 batch，``bboxes`` 为归一化 ``xywhr``。

        Returns:
            ``(loss_vector, items)``；``loss_vector`` 的三项已乘各自增益，
            训练循环只做 ``sum()``。
        """
        raws = preds[1] if isinstance(preds, tuple) else preds
        device = raws[0].device
        stride0 = self.head.strides[0]
        imgsz = (raws[0].shape[-2] * stride0, raws[0].shape[-1] * stride0)

        gt_box = batch["bboxes"].to(device).float().clone()  # 归一化 xywhr
        gt_cls = batch["cls"].reshape(-1).long().to(device)
        gt_b = batch["batch_idx"].reshape(-1).long().to(device)
        if gt_box.numel():
            gt_box[:, [0, 2]] *= imgsz[1]
            gt_box[:, [1, 3]] *= imgsz[0]

        l_obj = raws[0].new_zeros(())
        l_cls = raws[0].new_zeros(())
        l_box = raws[0].new_zeros(())
        n_pos = 0

        for index, raw in enumerate(raws):
            logits = self.head.split(raw, index)  # (B,na,H,W,no)
            b, na, h, w, _ = logits.shape
            obj_logit = logits[..., 4]
            obj_target = torch.zeros_like(obj_logit)
            cls_target = torch.zeros(
                (b, na, h, w, self.head.nc), device=device, dtype=logits.dtype
            )

            cells = self._assign(gt_box, gt_b, index, h, w)
            if cells:
                idx = torch.tensor(
                    [(c[0], c[1], c[2], c[3]) for c in cells], device=device, dtype=torch.long
                )
                gt_idx = torch.tensor([c[4] for c in cells], device=device, dtype=torch.long)
                obj_target[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = 1.0
                cls_target[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], gt_cls[gt_idx]] = 1.0

                boxes = self.head.boxes_from_logits(logits, index)[
                    idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
                ]
                l_box = l_box + (1.0 - probiou(boxes, gt_box[gt_idx])).mean()
                l_cls = l_cls + self.bce(
                    logits[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], 5 : 5 + self.head.nc],
                    cls_target[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]],
                ).mean()
                n_pos += idx.shape[0]

            l_obj = l_obj + self.bce(obj_logit, obj_target).mean()

        n_scales = len(raws)
        l_obj = l_obj / n_scales
        if n_pos:
            l_cls = l_cls / n_scales
            l_box = l_box / n_scales
        total = self.obj_gain * l_obj + self.cls_gain * l_cls + self.box_gain * l_box
        items = {
            "box_loss": float(l_box.detach()) * self.box_gain,
            "cls_loss": float(l_cls.detach()) * self.cls_gain,
            "obj_loss": float(l_obj.detach()) * self.obj_gain,
            "dfl_loss": 0.0,
            "angle_loss": 0.0,
        }
        return torch.stack([self.box_gain * l_box, self.cls_gain * l_cls, self.obj_gain * l_obj]), items

    def _assign(
        self,
        gt_box: Tensor,
        gt_batch: Tensor,
        index: int,
        height: int,
        width: int,
    ) -> list[tuple[int, int, int, int, int]]:
        """把 GT 分配到 ``(img, anchor, gy, gx, gt_idx)`` 正样本格。"""
        if not gt_box.numel():
            return []
        stride = self.head.strides[index]
        cells: list[tuple[int, int, int, int, int]] = []
        for g in range(gt_box.shape[0]):
            img = int(gt_batch[g])
            cx, cy, gw, gh, _theta = gt_box[g].tolist()
            gx, gy = int(math.floor(cx / stride)), int(math.floor(cy / stride))
            for anchor_index in self._match_anchors(gw, gh, index):
                for dx, dy in NEIGHBOUR_OFFSETS:
                    xi, yi = gx + dx, gy + dy
                    if 0 <= xi < width and 0 <= yi < height:
                        cells.append((img, anchor_index, yi, xi, g))
        return cells

    def _match_anchors(self, gw: float, gh: float, index: int) -> list[int]:
        """按排序后的边长比筛锚框（阈值默认 0.25，与 v5 的 ``1/4`` 一致）。"""
        small, large = min(gw, gh), max(gw, gh)
        picked: list[int] = []
        for anchor_index in range(self.head.na):
            aw, ah = self.head.anchor_grid[index * self.head.na + anchor_index].tolist()
            small_a, large_a = min(aw, ah), max(aw, ah)
            ratio = min(small / small_a, small_a / small) * min(large / large_a, large_a / large)
            if ratio > self.anchor_thr:
                picked.append(anchor_index)
        return picked
