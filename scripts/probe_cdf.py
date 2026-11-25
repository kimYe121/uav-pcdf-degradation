"""CDF / SCPL 机制探针：权重动态性、推理期干预、按条件切片 AP。

不训练，只做前向与评估，用来回答"机制到底有没有在起作用"。

用法（仓库根目录执行）::

    python .workbuddy/cdf_probe.py stats --ckpt runs/pcdf_s2_server/last.pt --split test
    python .workbuddy/cdf_probe.py eval  --ckpt runs/pcdf_s2_server/last.pt --split test --mode normal
    python .workbuddy/cdf_probe.py slice --ckpt runs/pcdf_s2_server/last.pt --split test \
        --out .workbuddy/slice_pcdf_test.json
    python .workbuddy/cdf_probe.py gate --ckpt runs/pcdf_s2_server/last.pt --split test \
        --out .workbuddy/gate_pcdf_test.json

``eval --mode`` 的推理期干预：

* ``normal``：原样；
* ``uniform``：强制 W_rgb = W_ir = 0.5（通道/样本都不再自适应）；
* ``guide_zero``：引导向量置零（权重只由 F_v 的池化特征决定）；
* ``guide_shuffle``：把 batch 内引导打乱（破坏样本-权重对应）；
* ``ir_only`` / ``rgb_only``：只保留一个模态。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcdf.cli.common import config_from_checkpoint, load_table  # noqa: E402
from pcdf.cli.evaluate import build_model, infer_arch  # noqa: E402
from pcdf.config import load_config  # noqa: E402
from pcdf.data import build_loader  # noqa: E402
from pcdf.evaluate import _average_precision  # noqa: E402
from pcdf.models import blocks  # noqa: E402

MODE = "normal"
GUIDANCE = "visual"
STATIC_W = None


def _patched_forward(self, guidance, fv_rgb, fv_ir):
    """带干预开关的 CDF 前向；与 ``ChannelNormSoftGate.forward`` 同构。"""
    if MODE == "normal":
        return _orig_forward(self, guidance, fv_rgb, fv_ir)

    if MODE == "static":
        ws = STATIC_W
        fused, weights, weighted_rgb, weighted_ir = [], [], [], []
        for s_i, (xr, xi) in enumerate(zip(fv_rgb, fv_ir)):
            b, c = xr.shape[:2]
            wi = torch.as_tensor(ws[s_i], device=xr.device, dtype=xr.dtype).view(1, c, 1, 1)
            wr = 1.0 - wi
            weighted_rgb.append(wr * xr)
            weighted_ir.append(wi * xi)
            fused.append(torch.cat([wr * xr, wi * xi], dim=1))
            weights.append(wi.expand(b, c, 1, 1)[:, :, 0, 0])
        return fused, weights, weighted_rgb, weighted_ir

    if MODE == "swap_all":
        perm = torch.randperm(guidance.shape[0], device=guidance.device)
        guidance = guidance[perm]
        fv_rgb = [x[perm] for x in fv_rgb]
        fv_ir = [x[perm] for x in fv_ir]

    if MODE == "guide_zero":
        guidance = torch.zeros_like(guidance)
    elif MODE == "guide_shuffle":
        guidance = guidance[torch.randperm(guidance.shape[0], device=guidance.device)]

    fused, weights, weighted_rgb, weighted_ir = [], [], [], []
    for proj, xr, xi in zip(self.projections, fv_rgb, fv_ir):
        b, c = xr.shape[:2]
        if MODE in ("uniform", "ir_only", "rgb_only", "concat_v"):
            if MODE == "uniform":
                w = torch.full((b, 2, c, 1, 1), 0.5, device=xr.device, dtype=xr.dtype)
            elif MODE == "concat_v":
                w = torch.ones((b, 2, c, 1, 1), device=xr.device, dtype=xr.dtype)
            elif MODE == "ir_only":
                w = torch.zeros((b, 2, c, 1, 1), device=xr.device, dtype=xr.dtype)
                w[:, 1] = 1.0
            else:
                w = torch.zeros((b, 2, c, 1, 1), device=xr.device, dtype=xr.dtype)
                w[:, 0] = 1.0
        else:
            pool_r = xr.mean(dim=(2, 3))
            pool_i = xi.mean(dim=(2, 3))
            if MODE == "guide_only":
                pool_r = torch.zeros_like(pool_r)
                pool_i = torch.zeros_like(pool_i)
            logits = proj(torch.cat([guidance, pool_r, pool_i], dim=1)).view(b, 2, c, 1, 1)
            w = logits.softmax(dim=1)
        wr, wi = w[:, 0] * xr, w[:, 1] * xi
        weighted_rgb.append(wr)
        weighted_ir.append(wi)
        fused.append(torch.cat([wr, wi], dim=1))
        weights.append(w[:, 1, :, 0, 0])
    return fused, weights, weighted_rgb, weighted_ir


_orig_forward = blocks.ChannelNormSoftGate.forward
blocks.ChannelNormSoftGate.forward = _patched_forward


def load_model(ckpt: str, device: str):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    env = load_config(None)
    cfg = config_from_checkpoint(payload["cfg"], env) if payload.get("cfg") else env
    # 条件码本、预训练权重都是仓库内相对路径；检查点里存的是训练机绝对路径。
    cfg.model.condition_spec = env.model.condition_spec
    cfg.model.text_encoder = env.model.text_encoder
    cfg.model.encoder_weights = env.model.encoder_weights
    table = load_table(cfg)
    dev = torch.device(device)
    model = build_model(cfg, infer_arch(payload), table, dev)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return cfg, table, model, dev, infer_arch(payload)


def make_loader(cfg, table, split, batch, device):
    _, loader = build_loader(
        cfg.data, table, split=split, augment=False, root=cfg.paths.data_root,
        batch_size=batch, workers=0, shuffle=False, device=device,
    )
    return loader


@torch.inference_mode()
def run_stats(cfg, table, model, dev, loader, out_path, limit):
    rec = {
        "w_ir_mean": [], "ch_std": [], "guide_norm": [],
        "cond": [], "stems": [], "guide": [],
        "w_full": {0: [], 1: [], 2: []},
        "scale_mean": {0: [], 1: [], 2: []},
    }
    n = 0
    for batch in loader:
        if limit and n >= limit:
            break
        imgs = batch["img"].to(dev)
        out = model(imgs, cond=batch["cond"].to(dev), guidance=GUIDANCE)
        ws = out["cdf_ir_weights"]  # 每层 (B, C)
        for s, w in enumerate(ws):
            rec["scale_mean"][s].append(w.mean(1).float().cpu().numpy())
            rec["w_full"][s].append(w.float().cpu().numpy())
            if s == 2:
                rec["w_ir_mean"].append(w.mean(1).float().cpu().numpy())
                ch = w.float()
                # 每个样本在通道维的标准差，取均值 = 通道选择性
                rec["ch_std"].append(ch.std(1).cpu().numpy())
        rec["guide_norm"].append(out["guidance"].float().norm(dim=1).cpu().numpy())
        rec["guide"].append(out["guidance"].float().cpu().numpy())
        rec["cond"].append(batch["cond"].numpy())
        rec["stems"].extend(Path(p).stem for p in batch["im_file"])
        n += imgs.shape[0]

    cond = np.concatenate(rec["cond"])
    stems = rec["stems"]
    mean_all = np.concatenate(rec["w_ir_mean"])
    ch_std = np.concatenate(rec["ch_std"])
    gn = np.concatenate(rec["guide_norm"])
    scales = {s: np.concatenate(v) for s, v in rec["scale_mean"].items()}

    keys = table.keys
    summary = {
        "n": int(mean_all.size),
        "w_ir_overall": float(mean_all.mean()),
        "w_ir_between_image_std": float(mean_all.std()),
        "w_ir_channel_std_mean": float(ch_std.mean()),
        "guide_norm_mean": float(gn.mean()),
        "by_scale": {f"P{i+3}": float(scales[i].mean()) for i in range(3)},
        "by_attr": {},
    }
    for i, key in enumerate(keys):
        vals = cond[:, i]
        names = table.specs[i].names
        groups = {}
        for v in np.unique(vals):
            sel = vals == v
            if sel.sum() < 20:
                continue
            groups[names[int(v)] if int(v) < len(names) else str(int(v))] = {
                "n": int(sel.sum()),
                "w_ir": float(mean_all[sel].mean()),
            }
        summary["by_attr"][key] = groups

    payload = {"summary": summary,
               "per_image": {"stems": stems, "w_ir": mean_all.tolist(),
                             "cond": cond.tolist(),
                             "guide": np.concatenate(rec["guide"]).tolist(),
                             "w_full": {str(k): np.concatenate(v).tolist()
                                        for k, v in rec["w_full"].items()}}}
    Path(out_path).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


@torch.inference_mode()
def run_eval(cfg, table, model, dev, loader, out_path, limit, mode):
    global MODE
    MODE = mode
    if mode != "normal":
        torch.manual_seed(0)
    nc = cfg.data.nc
    per_image, n_gt, stems = [], np.zeros(nc, dtype=np.int64), []
    n_pred, n = 0, 0
    from ultralytics.utils.nms import non_max_suppression
    for batch in loader:
        if limit and n >= limit:
            break
        imgs = batch["img"].to(dev)
        out = model(imgs, cond=batch.get("cond", None) if GUIDANCE=="text" else None, guidance=GUIDANCE)
        preds = out["main"][0] if isinstance(out["main"], tuple) else out["main"]
        dets = non_max_suppression(preds, conf_thres=cfg.data.eval_conf,
                                   iou_thres=cfg.data.eval_iou, nc=nc, rotated=True,
                                   max_det=cfg.data.eval_max_det, max_nms=3000)
        h, w = imgs.shape[-2:]
        gt = batch["bboxes"].to(dev).clone().float()
        if gt.numel():
            gt[:, [0, 2]] *= w
            gt[:, [1, 3]] *= h
        gcls = batch["cls"].reshape(-1).long().to(dev)
        bidx = batch["batch_idx"].reshape(-1).long().to(dev)
        for b in range(imgs.shape[0]):
            m = bidx == b
            det = dets[b]
            item = {"pred_cls": det[:, 5].long(), "pred_conf": det[:, 4],
                    "pred_box": torch.cat([det[:, :4], det[:, 6:7]], dim=1),
                    "gt_cls": gcls[m], "gt_box": gt[m]}
            per_image.append(item)
            stems.append(Path(batch["im_file"][b]).stem)
            n_pred += int(det.shape[0])
            if item["gt_cls"].numel():
                n_gt += np.bincount(item["gt_cls"].cpu().numpy(), minlength=nc)
        n += imgs.shape[0]
    ap = _average_precision(per_image, nc, 0.5, n_gt=n_gt)
    names = cfg.data.names
    res = {"mode": mode, "split": cfg.data.val_split, "n_images": len(per_image),
           "n_pred": n_pred, "n_gt": int(n_gt.sum()),
           "map50": float(np.nanmean(ap)),
           "per_class_ap50": {names[c]: float(ap[c]) for c in range(nc)},
           "_per_image": per_image, "_stems": stems, "_gt": n_gt.tolist()}
    Path(out_path).write_text(json.dumps({k: v for k, v in res.items()
                                          if not k.startswith("_")}, indent=1), encoding="utf-8")
    print(f"[{mode}] mAP50={res['map50']:.4f} n={len(per_image)}")
    return res


@torch.inference_mode()
def run_slice(cfg, table, model, dev, loader, out_path, limit, split_name, tag):
    res = run_eval(cfg, table, model, dev, loader, out_path + ".full.json", limit, "normal")
    per_image = res["_per_image"]
    stems = res["_stems"]
    nc = cfg.data.nc
    # stem -> 条件 id（同一 stem 在不同 split 条件不同，用 ids_by_split_stem）
    key = "ids_by_split_stem"
    idx = getattr(table, key)
    cond_by_stem = {s: idx.get((split_name, s)) for s in stems}

    summary = {"tag": tag, "overall": float(res["map50"]), "slices": {}}
    for i, attr in enumerate(table.specs):
        groups = {}
        for v, name in enumerate(attr.names):
            sel = [j for j, s in enumerate(stems)
                   if cond_by_stem[s] is not None and cond_by_stem[s][i] == v]
            if len(sel) < 20:
                continue
            sub = [per_image[j] for j in sel]
            n_gt = np.zeros(nc, dtype=np.int64)
            for it in sub:
                if it["gt_cls"].numel():
                    n_gt += np.bincount(it["gt_cls"].cpu().numpy(), minlength=nc)
            ap = _average_precision(sub, nc, 0.5, n_gt=n_gt)
            groups[name] = {"n_images": len(sel),
                            "map50": float(np.nanmean(ap)),
                            "per_class": {cfg.data.names[c]: float(ap[c])
                                          for c in range(nc) if n_gt[c] > 0}}
        summary["slices"][attr.key] = groups
    Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: round(vv["map50"], 4) for kk, vv in vv_.items()}
                      for k, vv_ in summary["slices"].items()}, indent=1))


@torch.inference_mode()
def run_gate(cfg, table, model, dev, loader, out_path, limit, split_name):
    """SCPL 门控统计（文本引导，需要 cond）。"""
    n = 0
    kepts, conds, probs_all, stems = [], [], [], []
    for batch in loader:
        if limit and n >= limit:
            break
        imgs = batch["img"].to(dev)
        out = model(imgs, cond=batch["cond"].to(dev), guidance="text")
        kepts.append(out["kept"].cpu().numpy())
        probs_all.append(out["gate_probs"].float().cpu().numpy())
        conds.append(batch["cond"].numpy())
        stems.extend(Path(p).stem for p in batch["im_file"])
        n += imgs.shape[0]
    kept = np.concatenate(kepts)
    cond = np.concatenate(conds)
    probs = np.concatenate(probs_all)
    key = "ids_by_split_stem"
    idx = getattr(table, key)
    summary = {"n": int(kept.size), "kept_mean": float(kept.mean()),
               "kept_hist": {int(k): int((kept == k).sum()) for k in np.unique(kept)},
               "by_attr": {}}
    for i, attr in enumerate(table.specs):
        groups = {}
        for v, name in enumerate(attr.names):
            sel = cond[:, i] == v
            if sel.sum() < 20:
                continue
            groups[name] = {"n": int(sel.sum()),
                            "kept": float(kept[sel].mean()),
                            "prob": float(probs[sel, i].mean())}
        summary["by_attr"][attr.key] = groups
    summary["prob_by_attr"] = {k: float(probs[:, i].mean())
                               for i, k in enumerate(table.keys)}
    Path(out_path).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stats", "eval", "slice", "gate"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="前 N 张图，0=全部")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", dest="intervention", default="normal",
                    choices=("normal", "uniform", "guide_zero", "guide_shuffle",
                             "ir_only", "rgb_only", "static", "concat_v", "swap_all", "guide_only"))
    ap.add_argument("--static-path", default=None)
    ap.add_argument("--guidance", default="visual", choices=("visual", "text"))
    args = ap.parse_args()

    global GUIDANCE, STATIC_W
    GUIDANCE = args.guidance
    if args.intervention == "static":
        stats = json.loads(Path(args.static_path).read_text(encoding="utf-8"))
        wf = stats["per_image"]["w_full"]
        STATIC_W = {int(k): np.asarray(v, dtype=np.float32).mean(0) for k, v in wf.items()}
        print("static W means:", {k: float(v.mean()) for k, v in STATIC_W.items()})

    cfg, table, model, dev, arch = load_model(args.ckpt, args.device)
    loader = make_loader(cfg, table, args.split, args.batch, dev)
    out = args.out or f".workbuddy/probe_{args.mode}_{args.split}.json"
    if args.mode == "stats":
        run_stats(cfg, table, model, dev, loader, out, args.limit)
    elif args.mode == "eval":
        run_eval(cfg, table, model, dev, loader, out, args.limit, args.intervention)
    elif args.mode == "slice":
        run_slice(cfg, table, model, dev, loader, out, args.limit, args.split, arch)
    else:
        run_gate(cfg, table, model, dev, loader, out, args.limit, args.split)


if __name__ == "__main__":
    main()