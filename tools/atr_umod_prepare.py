#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ATR-UMOD 数据准备脚手架（RTX 4060 Laptop 8GB 适配版）

三步用法（务必按顺序）：

  1) 先侦察，不要急着转：
     python atr_umod_prepare.py check --xml-dir /path/to/Annotations

     重点看输出的：类别名列表、每类实例数、坐标点个数、属性字段名。
     然后按实际结果修改下面的 CLASSES 和 ATTR_TAGS。

  2) 转换 + 生成条件索引表：
     python atr_umod_prepare.py convert \
         --xml-dir /path/to/Annotations \
         --img-dir /path/to/RGB \
         --out-dir ./dataset \
         --vis 50

     会生成 dataset/labels_obb/ 和 dataset/condition_index.csv

  3) 生成退化测试集（用于模态缺失/退化实验）：
     python atr_umod_prepare.py degrade \
         --img-dir ./dataset/images_rgb_512 \
         --out-dir ./dataset/degraded \
         --severity 2

设计说明：
- 本脚本不假设 XML 的具体结构，用启发式定位 bbox/polygon 与数值坐标，
  因此对绝大多数 VOC 变体都能跑，但**必须先用 check 验证**。
- OBB 点序是最大雷区。脚本会强制统一绕向并输出面积比诊断，
  面积比中位数应 > 0.6，否则说明点序或坐标有问题。
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------- 配置区
# 按 check 的输出修正这里。顺序即 class_id。
CLASSES = [
    "car", "suv", "van", "bus", "freight_car",
    "truck", "motorcycle", "trailer", "excavator", "crane", "tank_truck",
]

# check 打印出的属性字段名，填到这里（大小写不敏感匹配）
ATTR_TAGS = {
    "altitude": ["altitude", "height", "alt"],
    "angle":    ["angle", "pitch", "camera_angle"],
    "time":     ["time", "shooting_time", "period"],
    "weather":  ["weather", "climate"],
    "illum":    ["illumination", "light", "lighting", "illum"],
    "scenario": ["scenario", "scene", "scene_type"],
}

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


# ---------------------------------------------------------------- 几何工具
def poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def unify_winding(pts: np.ndarray) -> np.ndarray:
    """统一为顺时针，并从最靠左上角的点起始，保证 OBB 点序一致。

    这个函数是防『框自交』的关键：不同的标注工具给出的点序可能是
    顺时针/逆时针/任意起点，任何不一致都会让训练目标变成非法多边形。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return pts
    # 只取前 4 点（8 点时通常是 4 个角点重复一次或含中点的变体，见 check 输出）
    if len(pts) > 4:
        pts = pts[:4]
    # 保证顺时针（signed area < 0）
    x, y = pts[:, 0], pts[:, 1]
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if signed < 0:
        pts = pts[::-1]
    # 从左上角附近起始
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def to_yolo_obb(pts: np.ndarray, w: float, h: float) -> np.ndarray:
    p = unify_winding(pts)
    p[:, 0] = np.clip(p[:, 0] / w, 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1] / h, 0.0, 1.0)
    return p.reshape(-1)


# ---------------------------------------------------------------- XML 解析
def find_numeric_points(elem, min_len=8):
    """在任意元素里找一段 >=min_len 的数值序列，返回点列表。"""
    cands = []
    for node in elem.iter():
        txt = (node.text or "").strip()
        if not txt:
            continue
        nums = NUM_RE.findall(txt)
        if len(nums) >= min_len:
            vals = [float(v) for v in nums[: 8 * (len(nums) // 8)]]
            cands.append(vals)
    if not cands:
        return None
    best = max(cands, key=len)
    return np.array(best[:8], dtype=np.float64).reshape(-1, 2)


def parse_size(root):
    for node in root.iter():
        tag = node.tag.lower()
        if tag in ("width", "w") and node.text:
            w = float(node.text)
            continue
    w = h = None
    for node in root.iter():
        tag = node.tag.lower()
        if tag == "width" and node.text:
            w = float(NUM_RE.findall(node.text)[0])
        elif tag == "height" and node.text:
            h = float(NUM_RE.findall(node.text)[0])
    return w, h


def parse_attrs(root):
    """按 ATTR_TAGS 抓条件属性，抓不到就返回空。"""
    flat = {}
    for node in root.iter():
        txt = (node.text or "").strip()
        if txt:
            flat[node.tag.lower()] = txt
    out = {}
    for key, aliases in ATTR_TAGS.items():
        for tag, val in flat.items():
            if any(a in tag for a in aliases):
                out[key] = val
                break
    return out


def parse_objects(root):
    """返回 [(cls_name, pts, w, h, area_ratio)]"""
    w, h = parse_size(root)
    objs = []
    for node in root.iter():
        tag = node.tag.lower()
        has_child = any(c.tag.lower() in ("bndbox", "polygon", "points", "segmented")
                        for c in node.iter())
        if tag not in ("object", "target", "annotation") and not has_child:
            continue
        name = None
        for child in node:
            if child.tag.lower() in ("name", "label", "class", "type") and child.text:
                name = child.text.strip()
                break
        pts = find_numeric_points(node)
        if pts is None or name is None:
            continue
        area = poly_area(pts)
        if area <= 0:
            continue
        # 面积比：多边形面积 / 其轴对齐外接矩形面积
        bw = pts[:, 0].max() - pts[:, 0].min()
        bh = pts[:, 1].max() - pts[:, 1].min()
        ratio = area / max(bw * bh, 1e-6)
        objs.append((name, pts, w, h, ratio))
    return objs


# ---------------------------------------------------------------- check
def cmd_check(args):
    xmls = sorted(Path(args.xml_dir).rglob("*.xml"))
    if not xmls:
        sys.exit(f"未找到 XML：{args.xml_dir}")
    print(f"XML 文件数：{len(xmls)}\n")

    cls_counter = Counter()
    ratios = []
    unknown = Counter()
    attr_keys = Counter()
    pt_lens = Counter()
    no_size = 0
    per_file_objs = []

    for p in xmls[: args.limit]:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError as e:
            unknown[f"PARSE_FAIL:{e}"] += 1
            continue
        w, h = parse_size(root)
        if not w or not h:
            no_size += 1
        objs = parse_objects(root)
        per_file_objs.append(len(objs))
        for name, pts, _, _, ratio in objs:
            cls_counter[name] += 1
            ratios.append(ratio)
            pt_lens[len(pts)] += 1
            if name.lower() not in [c.lower() for c in CLASSES]:
                unknown[name] += 1
        for k in parse_attrs(root):
            attr_keys[k] += 1

    print("=" * 64)
    print("类别统计（实例数）")
    print("=" * 64)
    for name, n in cls_counter.most_common():
        flag = "" if name.lower() in [c.lower() for c in CLASSES] else "   <-- 不在 CLASSES 里"
        print(f"  {name:<18} {n:>7}{flag}")
    if unknown:
        print("\n  [注意] 未匹配的类别名，请把它们加入 CLASSES，否则会被跳过：")
        for u, n in unknown.most_common():
            print(f"    {u}  x{n}")

    print(f"\n文件数：{len(xmls)}   已解析：{len(per_file_objs)}")
    print(f"平均每图实例：{np.mean(per_file_objs):.1f}   最多：{max(per_file_objs)}")
    print(f"缺少 width/height 的文件：{no_size}")

    print("\n" + "=" * 64)
    print("OBB 点序健康度（最关键）")
    print("=" * 64)
    if ratios:
        r = np.array(ratios)
        print(f"  面积比 中位数 {np.median(r):.3f}  |  10 分位 {np.percentile(r,10):.3f}")
        print(f"  <0.30 的比例（疑似点序错误）：{(r < 0.30).mean()*100:.2f}%")
        verdict = "正常" if np.median(r) > 0.6 and (r < 0.3).mean() < 0.02 else "异常，检查点序/坐标"
        print(f"  判定：{verdict}")
        print("  说明：水平框的面积比接近 1.0，倾斜框在 0.5-0.9 之间；")
        print("        若大量 <0.3，几乎一定是 8 点顺序不是 4 个角点连续排列。")
    print(f"  坐标点个数分布：{dict(pt_lens)}")
    print("  提示：若出现 8 点，用 check 抽一张图打印原始坐标确认是『4角点+重复』还是『含边中点』。")

    print("\n" + "=" * 64)
    print("条件属性字段")
    print("=" * 64)
    if attr_keys:
        for k, n in attr_keys.items():
            print(f"  {k:<12} 命中 {n} 个文件")
    else:
        print("  未按 ATTR_TAGS 命中。下面列出文件里所有叶子字段名，请据此修改 ATTR_TAGS：")
        tags = Counter()
        for p in xmls[:50]:
            try:
                for node in ET.parse(p).getroot().iter():
                    if node.text and node.text.strip():
                        tags[node.tag] += 1
            except Exception:
                pass
        for t, n in tags.most_common(40):
            print(f"    {t:<24} {n}")

    print("\n下一步：把上面确认的类别名和属性字段填进脚本顶部，再运行 convert。")


# ---------------------------------------------------------------- convert
def cmd_convert(args):
    xmls = sorted(Path(args.xml_dir).rglob("*.xml"))
    out = Path(args.out_dir)
    (out / "labels_obb").mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = defaultdict(int)
    ratios_all = []
    cls_lower = {c.lower(): i for i, c in enumerate(CLASSES)}

    for p in xmls:
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            skipped["parse_fail"] += 1
            continue
        w, h = parse_size(root)
        if not w or not h:
            skipped["no_size"] += 1
            continue
        objs = parse_objects(root)
        if not objs:
            skipped["no_obj"] += 1
            continue

        stem = p.stem
        lines = []
        for name, pts, _, _, ratio in objs:
            cid = cls_lower.get(name.lower())
            if cid is None:
                skipped[f"unknown_class:{name}"] += 1
                continue
            flat = to_yolo_obb(pts, w, h)
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in flat))
            ratios_all.append(ratio)

        if not lines:
            skipped["no_valid_obj"] += 1
            continue

        (out / "labels_obb" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        attrs = parse_attrs(root)
        rows.append({
            "image_id": stem,
            "n_obj": len(lines),
            "width": int(w), "height": int(h),
            **{k: attrs.get(k, "") for k in ATTR_TAGS},
        })

    with open(out / "condition_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)

    # 目录结构对照检查
    if args.img_dir:
        img_dir = Path(args.img_dir)
        imgs = {p.stem for p in img_dir.rglob("*.jpg")} | {p.stem for p in img_dir.rglob("*.png")}
        if imgs:
            labeled = {r["image_id"] for r in rows}
            miss_lbl = imgs - labeled
            print(f"\n有图无标签：{len(miss_lbl)}")
            if miss_lbl:
                print("  示例：", list(sorted(miss_lbl))[:5])

    print(f"\n转换完成：{len(rows)} 个标签文件 → {out/'labels_obb'}")
    print(f"条件索引表 → {out/'condition_index.csv'}")
    if skipped:
        print("\n跳过统计：")
        for k, v in skipped.items():
            print(f"  {k}: {v}")

    if ratios_all:
        r = np.array(ratios_all)
        print(f"\n面积比中位数 {np.median(r):.3f}（应 > 0.6）；<0.3 占 {(r<0.3).mean()*100:.2f}%")

    if args.vis and cv2 is not None and args.img_dir:
        vis_dir = out / "_vis"
        vis_dir.mkdir(exist_ok=True)
        img_dir = Path(args.img_dir)
        files = sorted(img_dir.rglob("*.jpg"))[: args.vis]
        for i, ip in enumerate(files):
            lab = out / "labels_obb" / f"{ip.stem}.txt"
            if not lab.exists():
                continue
            img = cv2.imread(str(ip))
            if img is None:
                continue
            H, W = img.shape[:2]
            for line in lab.read_text().splitlines():
                v = list(map(float, line.split()))
                cid, pts = int(v[0]), np.array(v[1:]).reshape(-1, 2)
                pts[:, 0] *= W
                pts[:, 1] *= H
                cv2.polylines(img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(img, CLASSES[cid], tuple(pts[0].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(vis_dir / f"vis_{i:03d}.jpg"), img)
        print(f"可视化 {args.vis} 张 → {vis_dir}（务必人工看一遍）")


# ---------------------------------------------------------------- degrade
def _motion_blur(img, sev):
    if cv2 is None:
        return img
    k = {1: 5, 2: 11, 3: 21, 4: 31}[sev]
    ang = np.random.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _jitter(img, sev):
    if cv2 is None:
        return img
    H, W = img.shape[:2]
    amp = {1: 0.004, 2: 0.008, 3: 0.014, 4: 0.022}[sev]
    rot = {1: 0.3, 2: 0.7, 3: 1.2, 4: 1.8}[sev]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), np.random.uniform(-rot, rot), 1.0)
    M[0, 2] += np.random.uniform(-amp, amp) * W
    M[1, 2] += np.random.uniform(-amp, amp) * H
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _rolling_shutter(img, sev):
    H, W = img.shape[:2]
    shift = {1: 2, 2: 4, 3: 8, 4: 14}[sev]
    out = np.empty_like(img)
    for y in range(H):
        out[y] = np.roll(img[y], int(shift * (y / H - 0.5) * 2), axis=0)
    return out


def _fog(img, sev, beta=None):
    """大气散射简化：I = J*t + A*(1-t)，t = exp(-beta*d)，d 用垂直梯度近似。"""
    H, W = img.shape[:2]
    if beta is None:
        beta = {1: 0.008, 2: 0.018, 3: 0.032, 4: 0.050}[sev]
    yy = np.linspace(0, 1, H)[:, None]
    depth = 0.35 + 0.65 * yy
    t = np.exp(-beta * depth * 100)
    A = 235.0
    out = img.astype(np.float32) * t[..., None] + A * (1 - t[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _lowlight(img, sev):
    gain = {1: 0.75, 2: 0.55, 3: 0.38, 4: 0.25}[sev]
    noise_sigma = {1: 3, 2: 7, 3: 14, 4: 24}[sev]
    out = img.astype(np.float32) * gain
    out += np.random.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


MODES = {
    "jitter": _jitter,
    "motion_blur": _motion_blur,
    "rolling_shutter": _rolling_shutter,
    "fog": _fog,
    "lowlight": _lowlight,
}


def cmd_degrade(args):
    if cv2 is None:
        sys.exit("需要 opencv-python：pip install opencv-python")
    src = Path(args.img_dir)
    imgs = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".png")])
    if not imgs:
        sys.exit(f"未找到图像：{src}")
    np.random.seed(args.seed)
    modes = args.modes.split(",")
    for m in modes:
        if m not in MODES:
            sys.exit(f"未知模式 {m}，可选：{list(MODES)}")
    out = Path(args.out_dir) / f"sev{args.severity}"
    out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        for m in modes:
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{p.stem}__{m}{p.suffix}"
            cv2.imwrite(str(dst), MODES[m](img, args.severity))
    print(f"生成 {len(imgs)*len(modes)} 张 → {out}")
    print("提醒：标签不变，可直接用原 labels_obb 评测；但必须先确认退化没有裁掉目标。")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="ATR-UMOD 数据准备脚手架")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="侦察 XML 结构（先跑这个）")
    c.add_argument("--xml-dir", required=True)
    c.add_argument("--limit", type=int, default=200)
    c.set_defaults(func=cmd_check)

    v = sub.add_parser("convert", help="转为 Ultralytics OBB 标签 + 条件索引表")
    v.add_argument("--xml-dir", required=True)
    v.add_argument("--img-dir", default=None)
    v.add_argument("--out-dir", required=True)
    v.add_argument("--vis", type=int, default=0, help="可视化前 N 张用于人工校验")
    v.set_defaults(func=cmd_convert)

    d = sub.add_parser("degrade", help="生成退化测试集")
    d.add_argument("--img-dir", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--severity", type=int, default=2, choices=[1, 2, 3, 4])
    d.add_argument("--modes", default="jitter,motion_blur,rolling_shutter,fog,lowlight")
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_degrade)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
