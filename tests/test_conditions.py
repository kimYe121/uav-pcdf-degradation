"""条件合并、提示构造、索引访问与配置校验测试。"""

from __future__ import annotations

import numpy as np
import pytest

from pcdf.conditions import AttributeSpec, ConditionTable
from pcdf.config import ConfigError, load_config


@pytest.fixture(scope="module")
def table() -> ConditionTable:
    cfg = load_config()
    return ConditionTable.load(cfg.paths.condition_index, cfg.model.condition_spec)


def test_attribute_thresholds() -> None:
    alt = AttributeSpec(
        key="altitude", source="UAVheight", prefix="at a altitude of", kind="threshold",
        threshold=120, low="low", high="high",
    )
    assert alt.resolve("80") == 0
    assert alt.resolve("120") == 0
    assert alt.resolve("120.1") == 1
    ang = AttributeSpec(
        key="angle", source="UAVangle", prefix="at an angle of", kind="threshold",
        threshold=30, low="low", high="high",
    )
    assert ang.resolve("0") == 0
    assert ang.resolve("30") == 0
    assert ang.resolve("45") == 1


def test_identity_groups_merge() -> None:
    spec = AttributeSpec(
        key="weather", source="weather", prefix="in", kind="identity",
        codes=[0, 1, 2, 3, 4], groups=[0, 0, 1, 1, 2],
        names=["sunny", "cloudy", "foggy"],
    )
    assert spec.resolve("0") == 0
    assert spec.resolve("1") == 0
    assert spec.resolve("2") == 1
    assert spec.resolve("4") == 2
    assert spec.n_classes == 3


def test_time_bins() -> None:
    spec = AttributeSpec(
        key="time", source="time", prefix="at", kind="time",
        bins=[
            {"start": 7, "end": 14, "name": "morning"},
            {"start": 14, "end": 19, "name": "afternoon"},
            {"start": 19, "end": 7, "name": "night"},
        ],
    )
    assert spec.resolve("2022-6-5T07:00") == 0
    assert spec.resolve("2022-6-5T13:59") == 0
    assert spec.resolve("2022-6-5T14:00") == 1
    assert spec.resolve("2022-6-5T18:59") == 1
    assert spec.resolve("2022-6-5T19:00") == 2
    assert spec.resolve("2022-6-5T03:00") == 2


def test_prompt_template(table: ConditionTable) -> None:
    ids = [0, 1, 0, 0, 0, 0]
    prompt = table.prompt_from_ids(ids)
    assert prompt.startswith("An aerial image, over scenario 0,")
    assert prompt.endswith(".")
    assert "at a altitude of high altitude" in prompt
    assert "at an angle of low angle" in prompt
    assert "at morning" in prompt
    assert "in weather type 0" in prompt
    assert "in illumination level 0" in prompt


def test_mask_drops_blocks(table: ConditionTable) -> None:
    ids = [0, 1, 0, 0, 0, 0]
    prompt = table.prompt_from_ids(ids, [True, False, True, False, True, False])
    assert "scenario 0" in prompt
    assert "altitude" not in prompt
    assert "angle" in prompt
    assert "morning" not in prompt
    with pytest.raises(ValueError):
        table.prompt_from_ids(ids, [False] * 6)


def test_ids_match_index(table: ConditionTable) -> None:
    stem = next(iter(table.ids_by_stem))
    ids = table.ids(stem)
    assert ids.shape == (table.n_attributes,)
    assert np.all(ids >= 0)
    assert np.all(ids < np.array(table.class_counts))


def test_stems_by_subset(table: ConditionTable) -> None:
    """诊断脚本按划分挑样本；这里保证接口与索引一致。"""
    train = table.stems("train")
    assert train and all(table.subset_of(stem) == "train" for stem in train)
    assert set(train) <= set(table.stems())
    assert table.subset_of("not-a-real-stem") == ""


def test_row_by_stem_keeps_raw_columns(table: ConditionTable) -> None:
    """原始行要保留非条件列（n_objects），过拟合诊断靠它筛掉空标签图。"""
    stem = table.stems("train")[0]
    row = table.row_by_stem[stem]
    assert row["stem"] == stem
    assert "n_objects" in row
    assert set(table.raw_by_stem[stem]) == {spec.source for spec in table.specs}


def test_unknown_config_key() -> None:
    with pytest.raises(ConfigError):
        load_config("configs/conditions.yaml")  # 不是 PCDF 配置
