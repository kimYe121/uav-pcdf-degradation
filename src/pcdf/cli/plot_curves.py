"""``pcdf-plot``：把训练记录画成可用的曲线，并生成 markdown 摘要。

输入是训练器写出的 ``metrics.json``（优先）或 ``log.csv``。
输出：``curves/`` 下的 loss、mAP、阶段 2 损失、lr/kept 四张图，以及
``report.md``。

示例::

    python -m pcdf.cli.plot_curves --run-dir runs/pcdf_stage2
    python -m pcdf.cli.plot_curves --run-dir runs/pcdf_stage1 --out paper/figs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 —— 必须先设置后端


def load_history(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """读取 ``metrics.json``；没有时退回 ``log.csv``。"""
    meta = _read_json(run_dir / "meta.json")
    summary = _read_json(run_dir / "summary.json")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        return list(_read_json(metrics_path).get("history", [])), meta, summary

    csv_path = run_dir / "log.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 metrics.json 或 log.csv：{run_dir}")
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows, meta, summary


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """取某指标的 (epoch, value)，跳过缺失与非有限值。"""
    xs, ys = [], []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        ys.append(value)
        xs.append(int(row.get("epoch", len(xs) + 1)))
    return xs, ys


def plot_loss(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in ("loss", "box", "cls", "dfl", "angle"):
        xs, ys = series(rows, key)
        if ys and any(abs(v) > 1e-12 for v in ys):
            ax.plot(xs, ys, label=key, linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=200)
    plt.close(fig)


def plot_map(rows: list[dict[str, Any]], out: Path) -> None:
    xs, ys = series(rows, "map50")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if xs:
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label="mAP50")
        best_i = max(range(len(ys)), key=lambda i: ys[i])
        ax.scatter([xs[best_i]], [ys[best_i]], color="tab:red", zorder=3)
        ax.annotate(
            f"best {ys[best_i]:.4f} @ {xs[best_i]}",
            (xs[best_i], ys[best_i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP50")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "map50.png", dpi=200)
    plt.close(fig)


def plot_stage2_terms(rows: list[dict[str, Any]], out: Path) -> None:
    keys = ("l_dt", "l_irr", "l_dc")
    # 阶段 0/1 这三项恒为 0；不要生成一张全零的图。
    if not any(any(abs(value) > 0 for value in series(rows, key)[1]) for key in keys):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key in keys:
        xs, ys = series(rows, key)
        if ys:
            ax.plot(xs, ys, label=key, linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("auxiliary loss")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stage2_losses.png", dpi=200)
    plt.close(fig)


def plot_lr_kept(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    xs, ys = series(rows, "lr")
    if xs:
        ax1.plot(xs, ys, color="tab:blue", label="lr")
        ax1.set_ylabel("learning rate", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    xs2, ys2 = series(rows, "kept")
    if xs2:
        ax2.plot(xs2, ys2, color="tab:orange", label="kept attributes", linewidth=1.2)
        ax2.set_ylabel("kept attributes", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0, 6.5)
    ax1.set_xlabel("epoch")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lr_kept.png", dpi=200)
    plt.close(fig)


def write_report(
    run_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any],
    summary: dict[str, Any], out: Path,
) -> None:
    last = rows[-1] if rows else {}
    lines = [
        f"# {run_dir.name}",
        "",
        f"- stage: {meta.get('stage', summary.get('stage', '?'))}",
        f"- profile: {meta.get('profile', summary.get('profile', '')) or 'base'}",
        f"- status: {summary.get('status', 'unknown')}",
        f"- best mAP50: {summary.get('best_map50', last.get('map50'))}",
        f"- best epoch: {summary.get('best_epoch')}",
        f"- epochs: {summary.get('epochs_done', len(rows))}/{summary.get('epochs_total', '?')}",
        f"- batch: {meta.get('batch', '?')}  imgsz: {meta.get('imgsz', '?')}",
        f"- trainable params: {meta.get('trainable_params', '?')}",
        f"- GPU: {meta.get('gpu', '?')}  peak mem: {summary.get('peak_mem_gb', '?')} GB",
        f"- git: {meta.get('git', {})}",
        "",
        "| epoch | loss | box | cls | map50 | lr | kept |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {loss} | {box} | {cls} | {map50} | {lr} | {kept} |".format(
                epoch=row.get("epoch", ""),
                loss=_fmt(row.get("loss")),
                box=_fmt(row.get("box")),
                cls=_fmt(row.get("cls")),
                map50=_fmt(row.get("map50"), 4),
                lr=_fmt(row.get("lr"), 6),
                kept=_fmt(row.get("kept"), 2),
            )
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pcdf-plot",
        description="把训练记录画成曲线并生成 markdown 摘要",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-dir", required=True, help="包含 metrics.json / log.csv 的运行目录")
    parser.add_argument("--out", default=None, help="输出目录；默认 <run-dir>/curves")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "curves"
    try:
        rows, meta, summary = load_history(run_dir)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"[error] {run_dir} 没有可绘制的 epoch 记录", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    plot_loss(rows, out)
    plot_map(rows, out)
    plot_stage2_terms(rows, out)
    plot_lr_kept(rows, out)
    write_report(run_dir, rows, meta, summary, out)

    print(f"已生成：{out}")
    for name in sorted(p.name for p in out.iterdir()):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
