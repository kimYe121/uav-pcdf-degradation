"""生成投稿级架构图（原生 PPT 形状，可编辑）：baseline 双流 concat。

风格对齐论文 Fig.4 的视觉语言：梯形 encoder、三维特征块、圆形 concat 节点、
带箭头连接符、白底、图注在下方。所有元素为 PowerPoint 原生对象，可直接拖改。
生成后用 PowerPoint COM 导出 PNG 供目检。

用法：python scripts/make_arch_figure.py
产物：output/figures/baseline_architecture.{pptx,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = RGBColor(0x22, 0x22, 0x22)
LINE = RGBColor(0x44, 0x44, 0x44)
ARROW = RGBColor(0x55, 0x55, 0x55)
MUTED = RGBColor(0x66, 0x66, 0x66)
RGB_C = (0x44, 0x77, 0xAA)
IR_C = (0xEE, 0x77, 0x33)
FU_C = (0xAA, 0x33, 0x77)
FONT = "Arial"


def shade(c, f):
    return RGBColor(*[min(255, int(v * f)) for v in c])


def tint(c, f):
    return RGBColor(*[int(v + (255 - v) * f) for v in c])


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank


def box(x, y, w, h):
    return Inches(x), Inches(y), Inches(w), Inches(h)


def text(x, y, w, h, parts, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """parts = [(字符串, 字号, 粗体, 颜色, 下标, 斜体), ...]"""
    tb = slide.shapes.add_textbox(*box(x, y, w, h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    for s, size, bold, color, sub, ital in parts:
        lines = s.split("\n")
        for i, line in enumerate(lines):
            if i:
                p = tf.add_paragraph()
                p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = ital
            r.font.color.rgb = color
            if sub:
                r._r.get_or_add_rPr().set("baseline", "-25000")
    return tb


def label(x, y, w, h, s, size=9, bold=False, color=INK, align=PP_ALIGN.CENTER):
    return text(x, y, w, h, [(s, size, bold, color, False, False)], align=align)


def shape(kind, x, y, w, h, fill, line=LINE, lw=0.75, rot=0):
    sp = slide.shapes.add_shape(kind, *box(x, y, w, h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    sp.line.color.rgb = line
    sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    if rot:
        sp.rotation = rot
    return sp


def cuboid(x, y, w, h, c, d=0.13):
    """三维特征块：正面+顶面+侧面（freeform，论文特征图语言）。"""
    front = shape(MSO_SHAPE.RECTANGLE, x, y, w, h, shade(c, 0.85))
    top = slide.shapes.build_freeform(Emu(int(Inches(x))), Emu(int(Inches(y))))
    top.add_line_segments([
        (Inches(x + d), Inches(y - d)), (Inches(x + w + d), Inches(y - d)),
        (Inches(x + w), Inches(y)),
    ], close=True)
    tsp = top.convert_to_shape()
    tsp.fill.solid(); tsp.fill.fore_color.rgb = tint(c, 0.45)
    tsp.line.color.rgb = LINE; tsp.line.width = Pt(0.75); tsp.shadow.inherit = False
    side = slide.shapes.build_freeform(Inches(x + w), Inches(y))
    side.add_line_segments([
        (Inches(x + w + d), Inches(y - d)), (Inches(x + w + d), Inches(y + h - d)),
        (Inches(x + w), Inches(y + h)),
    ], close=True)
    ssp = side.convert_to_shape()
    ssp.fill.solid(); ssp.fill.fore_color.rgb = shade(c, 0.6)
    ssp.line.color.rgb = LINE; ssp.line.width = Pt(0.75); ssp.shadow.inherit = False
    return front


def arrow(x1, y1, x2, y2, w=1.0, color=ARROW, head=True):
    cn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cn.line.color.rgb = color
    cn.line.width = Pt(w)
    if head:
        ln = cn.line._get_or_add_ln()
        from pptx.oxml.ns import qn
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)
    return cn


def plain(x1, y1, x2, y2, w=1.0, color=ARROW):
    return arrow(x1, y1, x2, y2, w, color, head=False)


# ---------------------------------------------------------------- 输入图像
tmp = ROOT / ".workbuddy" / "arch_assets"
tmp.mkdir(parents=True, exist_ok=True)
ir0 = cv2.imread(str(ROOT / "data/atr_umod/ir/images/test/00211.jpg"))
rgb0 = cv2.imread(str(ROOT / "data/atr_umod/rgb/images/test/00211.jpg"))
cv2.imwrite(str(tmp / "in_rgb.jpg"), rgb0)
cv2.imwrite(str(tmp / "in_ir.jpg"), ir0)
diag = cv2.imread(str(ROOT / ".workbuddy/coord_diag/00211.png"))
ow = ir0.shape[1]
cv2.imwrite(str(tmp / "result.jpg"), diag[:, 2 * ow:3 * ow])  # 第三 panel = IR+pred
slide.shapes.add_picture(str(tmp / "in_rgb.jpg"), *box(0.30, 1.30, 1.35, 1.08))
slide.shapes.add_picture(str(tmp / "in_ir.jpg"), *box(0.30, 4.30, 1.35, 1.08))
label(0.30, 2.42, 1.35, 0.25, "RGB image", 9)
label(0.30, 5.42, 1.35, 0.25, "IR image", 9)

# ---------------------------------------------------------------- encoder
shape(MSO_SHAPE.TRAPEZOID, 1.95, 1.35, 1.5, 1.1, tint(RGB_C, 0.75), rot=90)
label(2.15, 1.85, 1.1, 0.5, "RGB\nEncoder", 9.5, True)
shape(MSO_SHAPE.TRAPEZOID, 1.95, 3.95, 1.5, 1.1, tint(IR_C, 0.75), rot=90)
label(2.15, 4.45, 1.1, 0.5, "IR\nEncoder", 9.5, True)
arrow(1.65, 1.84, 2.15, 1.90)
arrow(1.65, 4.84, 2.15, 4.50)

# ---------------------------------------------------------------- 三条尺度 lane
LANES = [
    ("P3 /8", 1.30, 0.85, 0.80, "128×80×64"),
    ("P4 /16", 2.90, 0.65, 0.62, "256×40×32"),
    ("P5 /32", 4.50, 0.50, 0.48, "256×20×16"),
]
# encoder -> 各 lane 的竖直 bus + 水平 stub（避免跨 lane 交叉线）
plain(3.30, 1.70, 3.30, 4.90)
plain(4.79, 1.70, 4.79, 5.35)
plain(3.05, 2.10, 3.05, 2.45)
arrow(3.05, 2.45, 3.30, 2.45)
plain(3.05, 4.70, 3.05, 5.35)
plain(3.05, 5.35, 4.79, 5.35)
for name, y, w, h, dims in LANES:
    yc = y + h / 2
    arrow(3.30, yc, 3.75, yc)
    arrow(4.79, yc, 4.90, yc)
    cuboid(3.75, y, w, h, RGB_C)
    cuboid(4.90, y, w, h, IR_C)
    label(3.60, y - 0.50, 1.6, 0.24, name, 8.5, True, MUTED)
    label(3.55, y + h + 0.03, 1.5, 0.22, dims, 7.5, False, MUTED)
    # concat 节点 F
    shape(MSO_SHAPE.OVAL, 6.00, yc - 0.16, 0.32, 0.32, RGBColor(0x33, 0x33, 0x33), line=LINE)
    label(6.00, yc - 0.13, 0.32, 0.26, "F", 9, True, RGBColor(0xFF, 0xFF, 0xFF))
    # RGB -> F 走特征块上方 bypass，避免穿过 IR 块
    plain(3.75 + w + 0.13, y - 0.13, 3.75 + w + 0.13, y - 0.22)
    plain(3.75 + w + 0.13, y - 0.22, 6.16, y - 0.22)
    arrow(6.16, y - 0.22, 6.16, yc - 0.16)
    arrow(4.90 + w + 0.13, yc + 0.06, 6.00, yc + 0.06)
    # 融合特征 2C
    cuboid(6.50, y, w, h, FU_C)
    label(6.40, y + h + 0.03, 1.3, 0.22, "2C concat", 7.5, False, MUTED)
    arrow(6.32, yc, 6.50, yc)
    arrow(6.50 + w + 0.13, yc, 7.60, yc)

# ---------------------------------------------------------------- neck / head
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 7.60, 1.15, 1.15, 4.35,
      RGBColor(0xF2, 0xF2, 0xF2), line=LINE)
label(7.60, 1.30, 1.15, 0.3, "PAN Neck", 10, True)
text(7.70, 1.75, 0.95, 1.2, [
    ("1×1 reduce", 7.5, False, MUTED, False, False),
], align=PP_ALIGN.CENTER)
text(7.70, 2.05, 0.95, 0.6, [("2C→C", 7.5, False, MUTED, False, False)], align=PP_ALIGN.CENTER)
text(7.70, 3.30, 0.95, 0.6, [("top-down", 7.5, False, MUTED, False, False)], align=PP_ALIGN.CENTER)
text(7.70, 4.30, 0.95, 0.6, [("bottom-up", 7.5, False, MUTED, False, False)], align=PP_ALIGN.CENTER)
arrow(8.17, 2.60, 8.17, 3.20)   # top-down
arrow(8.17, 4.20, 8.17, 3.60)   # bottom-up
for name, y, w, h, dims in LANES:
    yc = y + h / 2
    shape(MSO_SHAPE.RECTANGLE, 9.10, y + 0.02, 0.95, h - 0.04,
          RGBColor(0xFF, 0xFF, 0xFF), line=LINE)
    label(9.10, yc - 0.20, 0.95, 0.24, "OBB head", 8.5, True)
    label(9.10, yc + 0.02, 0.95, 0.22, name.split()[1], 7.5, False, MUTED)
    arrow(8.75, yc, 9.10, yc)
    arrow(10.05, yc, 10.40, 2.60 if y < 2 else (2.90 if y < 4 else 3.20))
label(8.95, 5.10, 1.3, 0.4, "anchor-free\nDFL + angle", 7.5, False, MUTED)

# ---------------------------------------------------------------- 结果与损失
slide.shapes.add_picture(str(tmp / "result.jpg"), *box(10.40, 1.50, 2.70, 2.16))
label(10.40, 3.70, 2.70, 0.25, "Detection results (OBB on IR)", 9)
text(10.40, 4.15, 2.75, 1.0, [
    ("L", 9.5, False, INK, False, True), ("det", 7.5, False, INK, True, False),
    (" = L", 9.5, False, INK, False, True), ("box", 7.5, False, INK, True, False),
    (" + L", 9.5, False, INK, False, True), ("cls", 7.5, False, INK, True, False),
    (" + L", 9.5, False, INK, False, True), ("DFL", 7.5, False, INK, True, False),
    (" + L", 9.5, False, INK, False, True), ("angle", 7.5, False, INK, True, False),
], align=PP_ALIGN.LEFT)
text(10.40, 4.55, 2.75, 0.5, [
    ("TAL top-k = 10 · supervised with IR labels", 7.5, False, MUTED, False, False),
])

# ---------------------------------------------------------------- 图例与图注
for i, (c, s) in enumerate([(RGB_C, "RGB stream"), (IR_C, "IR stream"), (FU_C, "concat fused")]):
    shape(MSO_SHAPE.RECTANGLE, 0.30 + i * 1.85, 6.15, 0.22, 0.16, shade(c, 0.85))
    label(0.58 + i * 1.85, 6.12, 1.3, 0.24, s, 8, False, INK, align=PP_ALIGN.LEFT)
arrow(6.30, 6.23, 6.80, 6.23)
label(6.90, 6.12, 1.6, 0.24, "feature flow", 8, False, INK, align=PP_ALIGN.LEFT)
shape(MSO_SHAPE.OVAL, 8.60, 6.13, 0.20, 0.20, RGBColor(0x33, 0x33, 0x33))
label(8.60, 6.15, 0.20, 0.18, "F", 7, True, RGBColor(0xFF, 0xFF, 0xFF))
label(8.88, 6.12, 2.6, 0.24, "channel concatenation", 8, False, INK, align=PP_ALIGN.LEFT)
text(0.30, 6.60, 12.73, 0.75, [
    ("Figure 1. ", 9, True, INK, False, False),
    ("Overview of the dual-stream concatenation baseline (Stage 0). RGB and IR images are "
     "encoded by two independent YOLOv5s backbones; per-scale features are concatenated (F) "
     "and fused by a PAN neck before an anchor-free OBB head. Trained with IR labels; "
     "56.10 mAP50 on the ATR-UMOD test split.", 9, False, INK, False, False),
])

pptx_path = OUT / "baseline_architecture.pptx"
prs.save(str(pptx_path))
print("saved", pptx_path)

# ---------------------------------------------------------------- COM 导出 PNG
import win32com.client  # noqa: E402
app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(str(pptx_path), WithWindow=False)
png_path = OUT / "baseline_architecture.png"
pres.Slides(1).Export(str(png_path), "PNG", 2000, 1125)
pres.Close()
app.Quit()
print("saved", png_path)
