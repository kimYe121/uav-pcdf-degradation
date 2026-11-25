"""OBB 评估：旋转 IoU、mAP50 与 mAP50-95。

Ultralytics 的 validator 与它自己的 trainer/model 包装耦合较紧。本复现使用
一个紧凑评估器，但保持相同约定：

* 旋转 NMS（``rotated=True``）；
* 按置信度贪心匹配 GT，同一 GT 只匹配一次；
* 101 点插值 AP；
* 旋转框 IoU 使用 probIoU。

数据集中的 GT 是归一化 ``xywhr``；预测是 letterbox 后输入尺度上的像素坐标。
评估只走视觉引导 ``F_s``，不需要条件 id——这是论文"测试不需要条件信息"的
部署设定。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import non_max_suppression

# 10 个 IoU 阈值：0.50:0.05:0.95，与 COCO / Ultralytics 的 mAP50-95 一致。
MAP5095_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@torch.inference_mode()
def evaluate_obb(
    model,
    loader: Iterable[dict],
    device: torch.device | str,
    nc: int,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    max_batches: int = 0,
    thresholds: tuple[float, ...] = (0.5,),
    names: list[str] | None = None,
    guidance: str = "visual",
    fuse: str | None = None,
) -> dict:
    """在给定 DataLoader 上计算 mAP。

    Args:
        model: ``PCDFNet`` 或 ``BaselineNet``；评估时只走视觉引导。
        loader: 验证/测试 DataLoader。
        device: 推理设备。
        nc: 类别数。
        conf/iou/max_det: NMS 参数；conf 默认 0.001 以保留完整 PR 曲线。
        max_batches: 限制评估 batch 数（调试用，0 表示全部）。
        thresholds: IoU 阈值；``(0.5,)`` 只算 mAP50。
        names: 类别名，用于逐类结果的可读输出。
        guidance: ``visual``（stage2/测试部署设定）或 ``text``（stage1 检查点：
            PCD 尚未训练，视觉引导通路是随机初始化，必须用文本引导评）。

    Returns:
        ``{"map50", "map50_95", "per_class_ap50", "n_images", "n_gt", "n_pred"}``。
    """
    model.eval()
    device = torch.device(device)

    per_image: list[dict[str, Tensor]] = []
    n_gt = np.zeros(nc, dtype=np.int64)
    n_pred = 0

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["img"].to(device, non_blocking=True)
        fwd = {"cond": batch.get("cond"), "guidance": guidance}
        if fuse is not None:
            fwd["fuse"] = fuse
        out = model(images, **fwd)
        preds = out["main"]
        if isinstance(preds, tuple):
            preds = preds[0]
        dets = non_max_suppression(
            preds, conf_thres=conf, iou_thres=iou, nc=nc, rotated=True,
            max_det=max_det,
            # 640×512 时三尺度共 80·64+40·32+20·16 = 6720 个候选，全量送进旋转
            # NMS 会触发 "NMS time limit 4.4s exceeded"，一次验证要好几个小时。
            # 按置信度先截前 3000 个候选（最终 max_det 只有 300），
            # mAP@0.5 不受影响，单图 NMS 从秒级降到毫秒级。
            max_nms=3000,
        )

        height, width = images.shape[-2:]
        # GT 的 xywh 是归一化坐标；角度已是弧度，不需要缩放。
        gt_boxes = batch["bboxes"].to(device).clone().float()
        if gt_boxes.numel():
            gt_boxes[:, [0, 2]] *= width
            gt_boxes[:, [1, 3]] *= height
        gt_cls = batch["cls"].reshape(-1).long().to(device)
        batch_idx = batch["batch_idx"].reshape(-1).long().to(device)

        for b in range(images.shape[0]):
            mask = batch_idx == b
            det = dets[b]
            item = {
                "pred_cls": det[:, 5].long(),
                "pred_conf": det[:, 4],
                # NMS 输出 [x, y, w, h, conf, cls, angle]；probIoU 需要 xywhr。
                "pred_box": torch.cat([det[:, :4], det[:, 6:7]], dim=1),
                "gt_cls": gt_cls[mask],
                "gt_box": gt_boxes[mask],
            }
            per_image.append(item)
            n_pred += int(item["pred_cls"].numel())
            if item["gt_cls"].numel():
                n_gt += np.bincount(item["gt_cls"].cpu().numpy(), minlength=nc)

    results: dict = {"n_images": len(per_image), "n_pred": n_pred, "n_gt": int(n_gt.sum())}
    aps: dict[float, np.ndarray] = {
        thr: _average_precision(per_image, nc, thr, n_gt=n_gt) for thr in thresholds
    }
    results["per_class_ap50"] = {
        (names[c] if names and c < len(names) else str(c)): (
            float(aps[0.5][c]) if 0.5 in aps else float("nan")
        )
        for c in range(nc)
    }
    results["map50"] = float(np.nanmean(aps[0.5])) if 0.5 in aps else float("nan")
    if len(thresholds) > 1:
        stacked = np.stack([aps[t] for t in thresholds], axis=0)
        results["map50_95"] = float(np.nanmean(stacked))
    else:
        results["map50_95"] = float("nan")
    return results


def _average_precision(
    per_image: list[dict[str, Tensor]],
    nc: int,
    threshold: float,
    n_gt: np.ndarray | None = None,
) -> np.ndarray:
    """单个 IoU 阈值下的逐类 AP。

    没有 GT 的类别返回 NaN（不参与 mAP）；有 GT 但完全没有预测的类别返回 0，
    否则漏检类别会被静默排除，导致 mAP 虚高。
    """
    tp: list[list[Tensor]] = [[] for _ in range(nc)]
    scores: list[list[Tensor]] = [[] for _ in range(nc)]
    if n_gt is None:
        n_gt = np.zeros(nc, dtype=np.int64)
        for item in per_image:
            if item["gt_cls"].numel():
                n_gt += np.bincount(item["gt_cls"].cpu().numpy(), minlength=nc)

    for item in per_image:
        gt_cls = item["gt_cls"]
        gt_box = item["gt_box"]
        pred_cls = item["pred_cls"]
        if pred_cls.numel() == 0:
            continue
        pred_box = item["pred_box"]
        pred_conf = item["pred_conf"]

        for c in torch.unique(pred_cls).tolist():
            p_mask = pred_cls == c
            p_box = pred_box[p_mask]
            p_conf = pred_conf[p_mask]
            order = torch.argsort(p_conf, descending=True)
            p_box, p_conf = p_box[order], p_conf[order]

            g_box = gt_box[gt_cls == c]
            if g_box.numel():
                iou = batch_probiou(p_box, g_box)
                matched = torch.zeros(g_box.shape[0], dtype=torch.bool, device=g_box.device)
            else:
                iou = None
                matched = None

            for i in range(p_box.shape[0]):
                hit = False
                if iou is not None:
                    best_iou, best_idx = iou[i].max(dim=0)
                    if float(best_iou) >= threshold and not bool(matched[best_idx]):
                        matched[best_idx] = True
                        hit = True
                tp[c].append(torch.tensor(1.0 if hit else 0.0))
                scores[c].append(p_conf[i].detach().cpu())

    ap = np.full(nc, np.nan, dtype=np.float64)
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        if not tp[c]:
            ap[c] = 0.0
            continue
        tp_c = torch.stack(tp[c]).numpy()
        conf_c = torch.stack(scores[c]).numpy()
        order = np.argsort(-conf_c)
        tp_c = tp_c[order]
        tp_cum = np.cumsum(tp_c)
        fp_cum = np.cumsum(1.0 - tp_c)
        recall = tp_cum / max(int(n_gt[c]), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        # 101 点插值，与 COCO / Ultralytics 的 AP 计算一致。
        grid = np.linspace(0.0, 1.0, 101)
        interp = np.array(
            [precision[recall >= r].max() if (recall >= r).any() else 0.0 for r in grid]
        )
        ap[c] = float(interp.mean())
    return ap
