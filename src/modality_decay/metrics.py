"""Negative-fusion metrics: RA, NF, crossing point, slope."""

from __future__ import annotations

from typing import Sequence


def ra_curve(values: Sequence[float]) -> list[float]:
    base = values[0]
    if base <= 0:
        return [float("nan")] * len(values)
    return [v / base for v in values]


def nf_curve(fusion: Sequence[float], rgb: Sequence[float], ir: Sequence[float]) -> list[float]:
    """NF(s) = mAP_fusion - max(mAP_rgb_only, mAP_ir_only). Negative => negative fusion."""
    return [f - max(r, i) for f, r, i in zip(fusion, rgb, ir)]


def crossing_point(levels: Sequence[float], nf: Sequence[float]) -> float | None:
    """First s where NF changes sign, by linear interpolation. None if never."""
    for i in range(1, len(nf)):
        a, b = nf[i - 1], nf[i]
        if a == 0:
            return float(levels[i - 1])
        if a * b < 0:
            x0, x1 = levels[i - 1], levels[i]
            return float(x0 + (0 - a) * (x1 - x0) / (b - a))
    return None


def slope(levels: Sequence[float], values: Sequence[float], normalize: bool = True) -> float:
    """Least-squares slope of mAP vs s (per unit s)."""
    xs = list(levels)
    ys = list(values)
    if len(xs) < 2:
        return float("nan")
    if normalize and ys[0] > 0:
        ys = [y / ys[0] for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom)


def summarize(df) -> dict:
    """Compute per-protocol NF curve, crossing point and slopes from a tidy dataframe."""
    out: dict = {}
    for protocol, g in df.groupby("protocol", sort=False):
        pivot = g.pivot_table(index="level", columns="model", values="map50", aggfunc="mean").sort_index()
        levels = [float(x) for x in pivot.index]
        fusion = pivot["fused"].tolist()
        rgb = pivot["rgb"].tolist()
        ir = pivot["ir"].tolist()
        nf = nf_curve(fusion, rgb, ir)
        out[protocol] = {
            "levels": levels,
            "fusion": fusion,
            "rgb": rgb,
            "ir": ir,
            "nf": nf,
            "s_star": crossing_point(levels, nf),
            "slope_fusion": slope(levels, fusion),
            "slope_rgb": slope(levels, rgb),
            "slope_ir": slope(levels, ir),
            "min_nf": min(nf),
        }
    return out
