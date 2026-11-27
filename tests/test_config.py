"""配置层测试：默认值、档位合并、点号覆盖、交叉校验与路径归一。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcdf.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "pcdf.yaml"


@pytest.fixture(autouse=True)
def _clear_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """路径环境变量会覆盖配置，测试里清掉以免受开发机影响。"""
    for name in (
        "PCDF_HOME", "PCDF_DATA_ROOT", "PCDF_CONDITION_INDEX",
        "PCDF_WEIGHTS_DIR", "PCDF_CLIP_DIR", "PCDF_RUNS_DIR", "PCDF_LOGS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_base_config_is_paper_defaults() -> None:
    cfg = load_config(BASE)
    assert cfg.data.batch == 16
    assert cfg.train.lr == pytest.approx(0.01)
    assert cfg.train.warmup_epochs == pytest.approx(0.0)
    assert cfg.train.profile == ""
    assert cfg.model.backbone_yaml == "yolov5.yaml"
    assert cfg.model.backbone_scale == "s"


def test_profile_overrides_only_listed_fields() -> None:
    cfg = load_config(BASE, profile="48g")
    assert cfg.data.batch == 32
    assert cfg.data.workers == 12
    assert cfg.train.lr == pytest.approx(0.02)
    assert cfg.train.warmup_epochs == pytest.approx(1.0)
    assert cfg.train.profile == "48g"
    # 未在档位里出现的字段保持基础配置
    assert cfg.data.imgsz == (512, 640)
    assert cfg.model.widths == (128, 256, 256)


def test_profile_does_not_mutate_base() -> None:
    load_config(BASE, profile="48g")
    cfg = load_config(BASE)
    assert cfg.data.batch == 16
    assert cfg.train.lr == pytest.approx(0.01)


def test_cpu_profile_turns_off_amp() -> None:
    """CPU 档必须显式关闭 AMP，否则日志显示"已启用"但实际不生效。"""
    cfg = load_config(BASE, profile="cpu")
    assert cfg.train.amp is False
    assert cfg.data.workers == 0
    assert cfg.data.imgsz == (256, 320)


def test_unknown_profile_reports_available_names() -> None:
    with pytest.raises(FileNotFoundError, match="48g"):
        load_config(BASE, profile="does-not-exist")


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="未知配置键"):
        load_config(BASE, overrides={"train.lrr": 0.1})


def test_dotted_override_and_type_coercion() -> None:
    """--set 传进来的都是字符串，必须按声明类型转换。"""
    cfg = load_config(BASE, overrides={
        "train.lr": "0.005",
        "train.amp": "false",
        "data.imgsz": "320,256",
        "model.lambda_dt": "0.1",
    })
    assert cfg.train.lr == pytest.approx(0.005)
    assert cfg.train.amp is False
    assert cfg.data.imgsz == (320, 256)
    assert cfg.model.lambda_dt == pytest.approx(0.1)


def test_validate_rejects_inconsistent_guidance_dim() -> None:
    """guidance_dim 必须等于 P5 通道数；旧实现要到建模时才报错。"""
    with pytest.raises(ConfigError, match="guidance_dim"):
        load_config(BASE, overrides={"model.guidance_dim": 128})


def test_validate_rejects_bad_imgsz() -> None:
    with pytest.raises(ConfigError, match="32 的正整数倍"):
        load_config(BASE, overrides={"data.imgsz": "300,200"})


def test_validate_rejects_non_v5_backbone() -> None:
    """只支持 YOLOv5 骨架；写别的 yaml 必须报错，避免"配置切了权重没切"。"""
    with pytest.raises(ConfigError, match="只支持 YOLOv5"):
        load_config(BASE, overrides={"model.backbone_yaml": "custom_backbone.yaml"})


def test_paths_are_absolute_and_derived() -> None:
    cfg = load_config(BASE)
    for value in (
        cfg.paths.data_root, cfg.paths.condition_index, cfg.paths.weights_dir,
        cfg.paths.runs_dir, cfg.model.encoder_weights, cfg.model.text_encoder,
        cfg.model.condition_spec,
    ):
        assert Path(value).is_absolute(), value
    # condition_index 留空时由 data_root 推导
    assert Path(cfg.paths.condition_index) == Path(cfg.paths.data_root) / "condition_index.csv"


def test_env_overrides_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PCDF_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("PCDF_WEIGHTS_DIR", str(tmp_path / "w"))
    cfg = load_config(BASE)
    assert Path(cfg.paths.data_root) == (tmp_path / "data").resolve()
    # 权重文件名保持配置里的选择，只换目录
    assert Path(cfg.model.encoder_weights) == (tmp_path / "w" / "yolov5su.pt").resolve()
    assert Path(cfg.paths.condition_index) == (tmp_path / "data" / "condition_index.csv").resolve()


def test_to_dict_roundtrip() -> None:
    """checkpoint 里存的就是 to_dict()，必须能原样重建。"""
    cfg = load_config(BASE)
    raw = cfg.to_dict()
    assert isinstance(raw["data"]["imgsz"], list)
    rebuilt = Config.from_dict(raw)
    rebuilt.resolve_paths()
    rebuilt.validate()
    assert rebuilt.data.imgsz == cfg.data.imgsz
    assert rebuilt.model.neck == cfg.model.neck


def test_no_crlf_in_source_and_scripts() -> None:
    """源码与 shell 脚本必须是 LF：CRLF 会让 ``set -euo pipefail`` 变成非法选项。

    Windows 上用 Python 直接改写文件容易引入 ``\\r\\n``，这条测试把问题挡在打包前。
    """
    bad = []
    for pattern in ("scripts/**/*.sh", "scripts/*.py", "src/**/*.py", "tests/**/*.py"):
        for path in ROOT.glob(pattern):
            if path.is_file() and b"\r\n" in path.read_bytes():
                bad.append(str(path.relative_to(ROOT)))
    assert not bad, f"以下文件含 CRLF：{bad}"

def test_checkpoint_cfg_adopts_local_runtime_paths() -> None:
    """Windows 训练、Linux 评估：运行时路径取本机，结构字段保持检查点。"""
    ckpt = load_config().to_dict()
    ckpt["model"]["condition_spec"] = r"D:\old\configs\conditions.yaml"
    ckpt["model"]["backbone_yaml"] = r"D:\old\yolov5.yaml"
    ckpt["model"]["encoder_weights"] = r"D:\old\weights\yolov5su.pt"
    ckpt["model"]["text_encoder"] = r"D:\old\weights\clip-vit-base-patch16"

    cfg = Config.from_dict(ckpt)
    env = load_config()
    cfg.adopt_runtime_paths(env)

    for key in ("condition_spec", "backbone_yaml", "encoder_weights", "text_encoder"):
        assert getattr(cfg.model, key) == getattr(env.model, key), key
    assert cfg.model.widths == env.model.widths

