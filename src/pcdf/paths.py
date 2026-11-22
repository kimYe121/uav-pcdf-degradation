"""仓库路径约定：项目根、资源目录、运行目录命名的唯一来源。

设计目标
--------
远端开发时，代码、数据、权重、运行产物经常不在同一块盘上（容器系统盘会被
重建）。因此**所有路径解析只在这里做一次**，其他模块只接收已经绝对化的
``str``/``Path``，不再拼接相对路径，也不依赖当前工作目录。

优先级
------
1. 环境变量（部署时注入，见 :data:`PATH_ENV_VARS`）；
2. YAML 配置（``paths:`` 段）；
3. 仓库默认值。

运行目录命名 ``<run_name>_stage<N>`` 也只在这里定义：Python 训练器、shell
脚本、巡检工具都调用 :func:`stage_dir`，避免三处各写一遍字符串。
"""

from __future__ import annotations

import os
from pathlib import Path

# 环境变量名：集中声明，避免字符串散落在 shell / Python 两侧。
ENV_HOME = "PCDF_HOME"
ENV_DATA_ROOT = "PCDF_DATA_ROOT"
ENV_CONDITION_INDEX = "PCDF_CONDITION_INDEX"
ENV_WEIGHTS_DIR = "PCDF_WEIGHTS_DIR"
ENV_CLIP_DIR = "PCDF_CLIP_DIR"
ENV_RUNS_DIR = "PCDF_RUNS_DIR"
ENV_LOGS_DIR = "PCDF_LOGS_DIR"
ENV_DEVICE = "PCDF_DEVICE"

# 环境变量 -> 配置键（点号路径）。``paths.py`` 只声明映射，
# 真正写入配置的动作在 :mod:`pcdf.config` 中完成。
PATH_ENV_VARS: dict[str, str] = {
    ENV_DATA_ROOT: "paths.data_root",
    ENV_CONDITION_INDEX: "paths.condition_index",
    ENV_WEIGHTS_DIR: "paths.weights_dir",
    ENV_RUNS_DIR: "paths.runs_dir",
    ENV_LOGS_DIR: "paths.logs_dir",
}


def _discover_project_root() -> Path:
    """定位仓库根目录。

    依次尝试：``PCDF_HOME``、从本文件向上查找含 ``configs/pcdf.yaml`` 的目录、
    最后回退到 ``src`` 的上一级。这样 editable 安装、直接解压、以及被复制到
    其它目录运行都能找到同一份 ``configs/``。
    """
    env_home = os.environ.get(ENV_HOME, "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "pcdf.yaml").is_file():
            return parent
    # 兜底：src/pcdf/paths.py -> src -> <root>
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _discover_project_root()

# 默认配置文件与档位目录。CLI 的 ``--config`` / ``--profile`` 都以它为基准。
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pcdf.yaml"
PROFILES_DIR = PROJECT_ROOT / "configs" / "profiles"


def resolve(path: str | Path, base: str | Path | None = None) -> Path:
    """把路径转成绝对路径；相对路径相对 ``base``（默认仓库根）解析。

    ``~`` 会被展开。函数不检查路径是否存在——存在性检查属于 preflight。
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    anchor = Path(base).expanduser() if base is not None else PROJECT_ROOT
    return (anchor / candidate).resolve()


def stage_dir(runs_dir: str | Path, run_name: str, stage: int) -> Path:
    """返回某阶段的运行目录 ``<runs_dir>/<run_name>_stage<N>``。

    这是运行目录命名的**唯一**实现；shell 侧通过 ``pcdf.report`` 间接复用，
    不再各自拼字符串。
    """
    if stage not in (0, 1, 2):
        raise ValueError(f"stage 必须是 0/1/2，收到 {stage!r}")
    return Path(runs_dir) / f"{run_name}_stage{stage}"


def display_path(path: str | Path) -> str:
    """把绝对路径转成相对仓库根的短路径，仅用于日志展示。

    路径不在仓库内时原样返回，避免打印出误导性的 ``../../..``。
    """
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)
