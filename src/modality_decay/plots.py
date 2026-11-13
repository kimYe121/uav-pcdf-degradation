"""Plots for availability-performance curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"fused": "#A32D2D", "rgb": "#185FA5", "ir": "#0F6E56"}
LABELS = {"fused": "fusion (channel-concat)", "rgb": "RGB-only", "ir": "IR-only"}


def plot_curves(summary: dict, out_png: str | Path, title: str = "Availability-performance") -> Path:
    protocols = list(summary.keys())
    n = len(protocols)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), squeeze=False)
    for ax, protocol in zip(axes[0], protocols):
        s = summary[protocol]
        for key in ("fused", "rgb", "ir"):
            ax.plot(s["levels"], s[key], marker="o", ms=4, lw=1.8, color=COLORS[key], label=LABELS[key])
        if s["s_star"] is not None:
            ax.axvline(s["s_star"], color="#888780", ls="--", lw=1)
            ax.annotate(f"s*={s['s_star']:.2f}", xy=(s["s_star"], ax.get_ylim()[1]),
                        xytext=(4, -10), textcoords="offset points", fontsize=9, color="#5F5E5A")
        ax.set_title(protocol, fontsize=11)
        ax.set_xlabel("degradation level s")
        ax.set_ylabel("mAP@0.5 (OBB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
