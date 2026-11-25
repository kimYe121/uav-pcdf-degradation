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
    """提示必须用官方码本的自然语言，而不是匿名占位。"""
    ids = [0, 1, 0, 0, 0, 0]
    prompt = table.prompt_from_ids(ids)
    assert prompt.startswith("An aerial image, over road in the urban villages,")
    assert prompt.endswith(".")
    assert "at a altitude of high altitude" in prompt
    assert "at an angle of low angle" in prompt
    assert "at morning" in prompt
    assert "in sunny" in prompt
    assert "in normal illumination" in prompt


def test_official_codebook_and_b2_merges(table: ConditionTable) -> None:
    """码本 = supercpy/ATR-UMOD 的 attribute_details.txt + 补充材料 B.2 合并。

    weather: 0 Sunny/1 Cloudy/2 Rainy/3 After Rain/4 Foggy/5 Snowy/
             6 Sunny_Night/7 Cloudy_Night/8 Sunny_Near_Night/9 Cloudy_Near_Night
             -> 5 类（Rainy+After Rain 并入 Cloudy，四个夜间码并入 Night）
    illumination: 0 Overexposure/1 Normal/2 Pre-Twilight/3 Post-Twilight/
             4 Dim/5 Near Night/6 Night -> 4 类
    """
    counts = dict(zip(table.keys, table.class_counts))
    assert counts["weather"] == 5
    assert counts["illumination"] == 4
    assert counts["scenario"] == 11
    assert counts["altitude"] == 2 and counts["angle"] == 2 and counts["time"] == 3

    weather = table.specs[table.keys.index("weather")]
    assert weather.resolve("0") != weather.resolve("1")           # sunny vs cloudy
    assert weather.resolve("2") == weather.resolve("1")           # rainy -> cloudy
    assert weather.resolve("3") == weather.resolve("1")           # after rain -> cloudy
    assert weather.resolve("6") == weather.resolve("9")           # night 变体合并
    illum = table.specs[table.keys.index("illumination")]
    assert illum.resolve("0") == illum.resolve("1")               # overexposure -> normal
    assert illum.resolve("2") == illum.resolve("4")               # twilight -> dim
    assert illum.resolve("6") != illum.resolve("5")               # night != near night

    # 统计必须覆盖 CSV 全部行（按 stem 折叠会少掉整个 test 划分的 1503 行）
    for key, dist in table.class_distribution().items():
        assert sum(dist.values()) == len(table.all_ids), key


def test_stem_is_only_unique_within_subset(table: ConditionTable) -> None:
    """ATR-UMOD 里同一 stem 可同时出现在 val 与 test（条件完全不同）。

    只用 stem 索引会让后读到的行覆盖先读到的，测试集条件被静默替换。
    """
    dup = [s for s in table.ids_by_stem if ("val", s) in table.ids_by_split_stem
           and ("test", s) in table.ids_by_split_stem]
    assert dup, "索引里应存在同时属于 val 与 test 的 stem"
    stem = dup[0]
    assert not np.array_equal(table.ids_for(stem, "val"), table.ids_for(stem, "test"))


def test_mask_drops_blocks(table: ConditionTable) -> None:
    ids = [0, 1, 0, 0, 0, 0]
    prompt = table.prompt_from_ids(ids, [True, False, True, False, True, False])
    assert "road in the urban villages" in prompt
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
