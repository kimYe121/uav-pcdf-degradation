"""运行巡检：状态读取、完成判定与 --require-stage 退出码。"""

from __future__ import annotations

import json
from pathlib import Path

from pcdf.report import describe_all, describe_stage, format_table, main


def _write_run(root: Path, run_name: str, stage: int, **payload) -> Path:
    run_dir = root / f"{run_name}_stage{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "last.pt").write_bytes(b"stub")
    return run_dir


def test_not_started(tmp_path: Path) -> None:
    status = describe_stage(tmp_path, "pcdf", 0)
    assert status.exists is False
    assert status.complete is False
    assert status.progress == "未开始"


def test_running_and_complete(tmp_path: Path) -> None:
    _write_run(tmp_path, "pcdf", 0, history=[{}, {}], epochs_total=4, best_map50=0.5)
    running = describe_stage(tmp_path, "pcdf", 0)
    assert running.epochs_done == 2 and running.epochs_total == 4
    assert running.complete is False
    assert "未完成" in running.progress

    _write_run(tmp_path, "pcdf", 1, history=[{}, {}], epochs_total=2, best_map50=0.61)
    done = describe_stage(tmp_path, "pcdf", 1)
    assert done.complete is True
    assert done.best_map50 == 0.61


def test_summary_overrides_status(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "pcdf", 2, history=[{}], epochs_total=4)
    (run_dir / "summary.json").write_text(
        json.dumps({"status": "error", "epochs_done": 1, "epochs_total": 4}),
        encoding="utf-8",
    )
    status = describe_stage(tmp_path, "pcdf", 2)
    assert status.status == "error"
    assert status.complete is False


def test_describe_all_covers_three_stages(tmp_path: Path) -> None:
    statuses = describe_all(tmp_path, "pcdf")
    assert [s.stage for s in statuses] == [0, 1, 2]
    assert "baseline" in format_table(statuses)


def test_require_stage_exit_code(tmp_path: Path) -> None:
    _write_run(tmp_path, "pcdf", 1, history=[{}], epochs_total=2)
    assert main(["--runs-dir", str(tmp_path), "--run-name", "pcdf", "--require-stage", "1"]) == 1
    assert main(["--runs-dir", str(tmp_path), "--run-name", "pcdf", "--require-stage", "0"]) == 1

    _write_run(tmp_path, "pcdf", 1, history=[{}, {}], epochs_total=2)
    assert main(["--runs-dir", str(tmp_path), "--run-name", "pcdf", "--require-stage", "1"]) == 0
