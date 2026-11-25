"""定性结果可视化（论文 Fig.5 / Fig.15-20 风格）。

约定与论文一致：融合检测结果画在 **IR** 上（与监督标签同模态），置信度阈值 0.25，
漏检（FN）蓝虚线圈、误检（FP）红虚线圈。在此基础上加两样报告友好的东西：
条件字幕条（补充 B.1 提示句）与色盲安全的逐类色板（Paul Tol bright 系）。

坐标还原：数据管线把原图**直接 warp** 到 imgsz（无 letterbox 边），故预测框先转
四角点再做各向异性缩放回原图尺寸——旋转框在anisotropic缩放下不再是旋转框，
但四角点四边形是 warp 的精确逆，不能直接缩放 xywhr 的角度。
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywhr2xyxyxyxy

# 类别顺序 = constants/data.names：bus car crane excavator freight_car motorcycle
# suv tank_truck trailer truck van。Tol bright 系，暗底 IR 与照片上均可分辨。
CLASS_COLORS = [
    "#CCBB44", "#66CCEE", "#BBBBBB", "#FFEE33", "#EE7733", "#AA3377",
    "#EE6677", "#DDAA33", "#009988", "#4477AA", "#228833",
]
FN_COLOR = (255, 170, 0)     # BGR 蓝（OpenCV）
FP_COLOR = (60, 60, 230)     # BGR 红
BG = "#101418"
FG = "#E8EAED"
MUTED = "#9AA9B2"


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


@torch.inference_mode()
def predict_batch(model, batch, device, nc: int, conf: float, iou: float = 0.7):
    """单 batch 逐图预测/GT（imgsz 像素坐标），与 evaluate_obb 同一 NMS 口径。"""
    images = batch["img"].to(device)
    out = model(images, guidance="visual")
    preds = out["main"]
    if isinstance(preds, tuple):
        preds = preds[0]
    dets = non_max_suppression(
        preds, conf_thres=conf, iou_thres=iou, nc=nc, rotated=True,
        max_det=300, max_nms=3000,
    )
    h, w = images.shape[-2:]
    gt = batch["bboxes"].to(device).clone().float()
    if gt.numel():
        gt[:, [0, 2]] *= w
        gt[:, [1, 3]] *= h
    gt_cls = batch["cls"].reshape(-1).long().to(device)
    bidx = batch["batch_idx"].reshape(-1).long().to(device)
    items = []
    for b in range(images.shape[0]):
        m = bidx == b
        d = dets[b]
        items.append({
            "im_file": batch["im_file"][b],
            "pred_cls": d[:, 5].long().cpu(),
            "pred_conf": d[:, 4].cpu(),
            "pred_box": torch.cat([d[:, :4], d[:, 6:7]], dim=1).cpu(),
            "gt_cls": gt_cls[m].cpu(),
            "gt_box": gt[m].cpu(),
            "imgsz": (h, w),
        })
    return items


@torch.inference_mode()
def predict_per_image(model, loader, device, nc: int, conf: float, iou: float = 0.7):
    """逐图返回预测/GT；多模型对照时改用 predict_batch 单遍扫双模型。"""
    model.eval()
    device = torch.device(device)
    items = []
    for batch in loader:
        items.extend(predict_batch(model, batch, device, nc, conf, iou))
    return items


def _match_fn_fp(item, thr: float = 0.5):
    """按置信度贪心、同类匹配（与 AP 计算同口径），返回 FN/FP 掩码。"""
    pb, gb = item["pred_box"], item["gt_box"]
    fn = torch.ones(gb.shape[0], dtype=torch.bool)
    fp = torch.ones(pb.shape[0], dtype=torch.bool)
    if pb.numel() and gb.numel():
        iou = batch_probiou(pb, gb)
        same = item["pred_cls"][:, None] == item["gt_cls"][None, :]
        iou = iou * same
        order = item["pred_conf"].argsort(descending=True)
        for p in order.tolist():
            cand = (iou[p] >= thr) & fn
            if cand.any():
                g = int(cand.float().argmax())
                fn[g] = False
                fp[p] = False
    return fn, fp


def _corners_orig(box: torch.Tensor, imgsz, orig_hw) -> np.ndarray:
    """xywhr(imgsz 像素) -> 原图像素四角点（LetterBox 的精确逆）。

    管线几何 = 等比 r=min(ih/oh, iw/ow) + 居中灰边（训练 RandomPerspective 输出
    后再 LetterBox，评估直接 LetterBox，二者一致）。逆变换 = 减 pad 除 r。
    旋转框在等比缩放下仍是旋转框，但统一走四角点最稳。
    """
    if box.numel() == 0:
        return np.zeros((0, 4, 2))
    c = xywhr2xyxyxyxy(box).numpy()
    oh, ow = orig_hw
    ih, iw = imgsz
    r = min(ih / oh, iw / ow)
    pad_x, pad_y = (iw - ow * r) / 2, (ih - oh * r) / 2
    c[..., 0] = (c[..., 0] - pad_x) / r
    c[..., 1] = (c[..., 1] - pad_y) / r
    return c


def _dashed_circle(img, center, radius, color, thickness=1):
    pts = [
        (int(center[0] + radius * np.cos(t)), int(center[1] + radius * np.sin(t)))
        for t in np.linspace(0, 2 * np.pi, 25)[:-1]
    ]
    for a, b in zip(pts[::2], pts[1::2]):
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _draw(img, corners, classes, confs, names, fn_fp=None, gt_corners=None):
    placed: list[tuple[int, int, int, int]] = []
    for poly, cls, conf in zip(corners, classes.tolist(), confs.tolist()):
        color = _bgr(CLASS_COLORS[int(cls) % len(CLASS_COLORS)])
        p = poly.astype(np.int32)
        cv2.polylines(img, [p], True, color, 2, cv2.LINE_AA)
        label = f"{names[int(cls)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        x0, y0 = int(p[:, 0].min()), int(p[:, 1].min())
        y1 = int(p[:, 1].max())
        # 标签避碰：先框上方，撞了放框下方，再撞就只画框不画字。
        spot = None
        for y in (y0 - 5, y1 + th + 5):
            rect = (x0, y - th, x0 + tw, y)
            if not any(rect[0] < q[2] and q[0] < rect[2] and rect[1] < q[3] and q[1] < rect[3]
                       for q in placed):
                spot = (x0, y)
                placed.append(rect)
                break
        if spot is None:
            continue
        cv2.putText(img, label, spot, cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, spot, cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    if fn_fp is not None:
        fn, fp = fn_fp
        for poly, miss in zip(gt_corners, fn.tolist()):
            if miss:
                c = poly.mean(axis=0)
                r = max(12, int(np.linalg.norm(poly[0] - poly[2]) / 2) + 6)
                _dashed_circle(img, c, r, FN_COLOR)
        for poly, false in zip(corners, fp.tolist()):
            if false:
                c = poly.mean(axis=0)
                r = max(12, int(np.linalg.norm(poly[0] - poly[2]) / 2) + 6)
                _dashed_circle(img, c, r, FP_COLOR)
    return img


def render_figure(
    panels: list[tuple[str, np.ndarray]],
    title: str,
    caption: str,
    footer: str,
    out_path: Path,
) -> Path:
    """panels = [(面板标题, BGR 图), ...]；暗底报告风格。"""
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(5.0 * k, 5.4), facecolor=BG)
    if k == 1:
        axes = [axes]
    for ax, (ptitle, img) in zip(axes, panels):
        ax.imshow(img[..., ::-1])
        ax.set_title(ptitle, color=FG, fontsize=10, pad=6)
        ax.axis("off")
    fig.suptitle(title, color=FG, fontsize=12, y=0.98)
    fig.text(0.5, 0.945, textwrap.fill(caption, 96), color=MUTED, fontsize=7.5,
             family="monospace", ha="center", va="top", linespacing=1.5)
    fig.text(0.5, 0.02, footer, color=MUTED, fontsize=7.5, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG)
    plt.close(fig)
    return out_path


def _load_pair(item):
    ir_path = Path(item["im_file"])
    rgb_path = Path(str(ir_path).replace("/ir/", "/rgb/").replace("\\ir\\", "\\rgb\\"))
    return cv2.imread(str(ir_path)), cv2.imread(str(rgb_path))


def _make_panels(item, row_items, names, conf):
    """row_items=None 单模型；否则 [(tag, item), ...] 对照。返回 (panels, footer)。"""
    ir, rgb = _load_pair(item)
    oh, ow = ir.shape[:2]
    if row_items is None:
        fn, fp = _match_fn_fp(item)
        pc = _corners_orig(item["pred_box"], item["imgsz"], (oh, ow))
        gc = _corners_orig(item["gt_box"], item["imgsz"], (oh, ow))
        panels = [("RGB", rgb),
                  (f"IR · pred (conf≥{conf})",
                   _draw(ir.copy(), pc, item["pred_cls"], item["pred_conf"],
                         names, (fn, fp), gc))]
        return panels, f"pred={len(pc)}  gt={len(gc)}"
    panels = [("RGB", rgb)]
    foot = []
    for tag, it in row_items:
        fn, fp = _match_fn_fp(it)
        pc = _corners_orig(it["pred_box"], it["imgsz"], (oh, ow))
        gc = _corners_orig(it["gt_box"], it["imgsz"], (oh, ow))
        panels.append((f"IR · {tag}", _draw(ir.copy(), pc, it["pred_cls"],
                                           it["pred_conf"], names, (fn, fp), gc)))
        foot.append(f"{tag}: pred={len(pc)}")
    return panels, "  ".join(foot) + f"  gt={len(gc)}"


def visualize_samples(
    items: list[dict],
    names: list[str],
    table,
    split: str,
    out_dir: Path,
    models: list[tuple[str, list[dict]]] | None = None,
    conf: float = 0.25,
) -> list[Path]:
    """单模型：RGB | IR+pred；models 给两个时：RGB | IR+模型A | IR+模型B。"""
    outs = []
    for idx, item in enumerate(items):
        ir_path = Path(item["im_file"])
        oh, ow = cv2.imread(str(ir_path)).shape[:2]
        stem = ir_path.stem
        ids = table.ids_for(stem, subset=split)
        caption = table.prompt_from_ids(ids.tolist())
        row_items = [(tag, mitems[idx]) for tag, mitems in models] if models else None
        panels, footer = _make_panels(item, row_items, names, conf)
        footer += "  blue dashed=missed  red dashed=false positive"
        outs.append(render_figure(
            panels, f"{stem} · {split}", caption, footer,
            out_dir / f"{stem}.png",
        ))
    return outs


def render_grid(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
    suptitle: str,
    footer: str,
    out_path: Path,
) -> Path:
    """报告级网格图：每行一个条件取值，行内 RGB | IR(+对照)。论文 Fig.15-20 组织。"""
    n_rows = len(rows)
    k = max(len(p) for _, p in rows)
    fig, axes = plt.subplots(n_rows, k, figsize=(4.6 * k, 3.9 * n_rows),
                             facecolor=BG, squeeze=False)
    for r, (label, panels) in enumerate(rows):
        for c in range(k):
            ax = axes[r][c]
            if c < len(panels):
                ptitle, img = panels[c]
                ax.imshow(img[..., ::-1])
                ax.set_title(f"{label} · {ptitle}" if c == 0 else ptitle,
                             color=FG, fontsize=9, pad=4)
            ax.axis("off")
    fig.suptitle(suptitle, color=FG, fontsize=13, y=0.995)
    fig.text(0.5, 0.005, footer, color=MUTED, fontsize=8, ha="center")
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    return out_path


def visualize_grid(
    rows: list[tuple[str, list[dict]]],
    names: list[str],
    out_dir: Path,
    tag: str,
    models: list[tuple[str, list[list[dict]]]] | None = None,
    conf: float = 0.25,
) -> Path:
    """rows = [(条件取值标签, 该行的样本 items), ...]；models 按行对齐给对照。"""
    grid_rows = []
    for r, (label, items) in enumerate(rows):
        panels = None
        for item in items[:1]:
            row_items = [(t, m[r][0]) for t, m in models] if models else None
            panels, _ = _make_panels(item, row_items, names, conf)
        grid_rows.append((label, panels or []))
    return render_grid(
        grid_rows, f"{tag} · qualitative results (conf≥{conf})",
        "blue dashed=missed  red dashed=false positive",
        out_dir / f"grid_{tag}.png",
    )
