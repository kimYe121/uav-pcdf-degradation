"""运行目录巡检：供 shell 脚本与人工排查使用。

为什么要一个独立模块
--------------------
早期 ``status.sh`` / ``common.sh`` 里各写了一段 ``python - <<'PY' ... torch.load``
的 heredoc，用来读检查点里的 ``next_epoch``。同一份逻辑存在三处，而且每次
巡检都要 import torch、反序列化 59 MB 检查点。

现在统一读**纯 JSON**：训练器每个 epoch 都会原子覆盖 ``metrics.json``，里面
已有 ``epochs_done`` / ``epochs_total`` / ``best_map50`` / ``updated_at``。
巡检不需要 torch，也就能在 CPU 环境、甚至没装 torch 的跳板机上运行。

命令行::

    python -m pcdf.report --runs-dir runs                 # 人类可读
    python -m pcdf.report --runs-dir runs --json          # 机器可读
    python -m pcdf.report --runs-dir runs --require-stage 1   # 未跑满则退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import STAGE_NAMES
from .paths import stage_dir

# 阶段完成判定：epochs_done >= epochs_total 且没有 interrupted 标记。
UNKNOWN = "unknown"


@dataclass
class StageStatus:
    """单个阶段的巡检结果（全部来自 JSON，不依赖 torch）。"""

    stage: int
    name: str
    run_dir: str
    exists: bool = False
    epochs_done: int = 0
    epochs_total: int = 0
    best_map50: float | None = None
    best_epoch: int | None = None
    status: str = "not_started"
    updated_at: str | None = None
    has_last: bool = False
    has_best: bool = False

    @property
    def complete(self) -> bool:
        """是否跑满：epochs_total > 0 且已完成轮数达到总数。"""
        return self.epochs_total > 0 and self.epochs_done >= self.epochs_total

    @property
    def progress(self) -> str:
        if not self.exists:
            return "未开始"
        if self.complete:
            return f"{self.epochs_done}/{self.epochs_total}"
        return f"{self.epochs_done}/{self.epochs_total or '?'}（未完成）"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe_stage(runs_dir: str | Path, run_name: str, stage: int) -> StageStatus:
    """读取某阶段的 ``metrics.json`` / ``summary.json`` 生成状态。"""
    run_dir = stage_dir(runs_dir, run_name, stage)
    status = StageStatus(
        stage=stage,
        name=STAGE_NAMES.get(stage, str(stage)),
        run_dir=str(run_dir),
        exists=run_dir.is_dir(),
        has_last=(run_dir / "last.pt").is_file(),
        has_best=(run_dir / "best.pt").is_file(),
    )
    metrics = _read_json(run_dir / "metrics.json")
    summary = _read_json(run_dir / "summary.json")

    if metrics:
        history = metrics.get("history") or []
        status.epochs_done = len(history)
        status.epochs_total = int(metrics.get("epochs_total") or 0)
        best = metrics.get("best_map50")
        status.best_map50 = float(best) if best is not None else None
        status.best_epoch = metrics.get("best_epoch")
        status.updated_at = metrics.get("updated_at")
        status.status = "running"
    if summary:
        status.status = str(summary.get("status", status.status))
        status.epochs_done = int(summary.get("epochs_done", status.epochs_done))
        status.epochs_total = int(summary.get("epochs_total", status.epochs_total))
        best = summary.get("best_map50")
        if best is not None:
            status.best_map50 = float(best)
        status.best_epoch = summary.get("best_epoch", status.best_epoch)
        status.updated_at = summary.get("finished_at") or status.updated_at
    if status.exists and not metrics and not summary:
        status.status = "checkpoint_only" if status.has_last else "empty"
    return status


def describe_all(runs_dir: str | Path, run_name: str) -> list[StageStatus]:
    """按阶段顺序返回 0/1/2 的状态。"""
    return [describe_stage(runs_dir, run_name, stage) for stage in (0, 1, 2)]


def format_table(statuses: list[StageStatus]) -> str:
    """人类可读表格；列宽固定，便于在终端与日志里对齐。"""
    lines = [
        f"{'stage':<10}{'进度':<18}{'best mAP50':<14}{'状态':<14}{'更新时间':<21}目录",
        "-" * 110,
    ]
    for item in statuses:
        best = f"{item.best_map50:.4f}" if item.best_map50 is not None else "-"
        lines.append(
            f"{item.name:<10}{item.progress:<18}{best:<14}{item.status:<14}"
            f"{(item.updated_at or '-'):<21}{item.run_dir}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练运行目录巡检（只读 JSON，不需要 torch）")
    parser.add_argument("--runs-dir", default=None, help="训练输出根目录；默认取配置")
    parser.add_argument("--run-name", default=None, help="运行名前缀；默认取配置")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--profile", default=None, help="配置档位")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument(
        "--stage-dir",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="只打印某阶段的运行目录（供 shell 复用命名规则），不读状态",
    )
    parser.add_argument(
        "--require-stage",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="检查某阶段是否跑满；未跑满时退出码为 1",
    )
    args = parser.parse_args(argv)

    runs_dir, run_name = args.runs_dir, args.run_name
    if runs_dir is None or run_name is None:
        # 只有缺参时才加载配置（避免为一次巡检去解析整份 YAML）。
        from .config import load_config

        cfg = load_config(args.config, profile=args.profile)
        runs_dir = runs_dir or cfg.paths.runs_dir
        run_name = run_name or cfg.train.run_name

    if args.stage_dir is not None:
        # 运行目录命名只在这里定义一遍，shell 通过本命令复用。
        print(stage_dir(runs_dir, run_name, args.stage_dir))
        return 0

    statuses = describe_all(runs_dir, run_name)

    if args.require_stage is not None:
        target = statuses[args.require_stage]
        if target.complete:
            print(f"[report] {target.name} 已跑满 {target.epochs_done}/{target.epochs_total}")
            return 0
        if target.status in ("error", "interrupted"):
            print(f"[report] {target.name} 状态 {target.status}：{target.progress}")
            return 1
        print(f"[report] {target.name} 未跑满：{target.progress}")
        return 1

    if args.json:
        print(json.dumps([asdict(item) | {"complete": item.complete} for item in statuses],
                         ensure_ascii=False, indent=2))
    else:
        print(format_table(statuses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
