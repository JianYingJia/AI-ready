from pathlib import Path

from ai_ready.config import CleaningConfig
from ai_ready.ops.cleaning import clean_units

RECIPE = Path(__file__).resolve().parents[1] / "configs/data_juicer/l1_clean.yaml"


def test_actual_dj_cleaning_merges_sources_and_preserves_numbers(unit_factory):
    records = [
        unit_factory("a", "  提前30日申请。  "),
        unit_factory("b", "提前30日申请。"),
        unit_factory("c", "提前15日申请。"),
        unit_factory("d", "不得提前30日申请。"),
        unit_factory("e", " "),
    ]
    cleaned, audit = clean_units(records, CleaningConfig(recipe=RECIPE, near_duplicates=False))
    assert len(cleaned) == 3
    canonical = next(unit for unit in cleaned if unit.id == "a")
    assert canonical.raw_text == "  提前30日申请。  "
    assert canonical.text == "提前30日申请。"
    assert canonical.duplicate_ids == ["b"]
    assert len(canonical.sources) == 2
    assert any(row["id"] == "e" and row["action"] == "filtered" for row in audit)
    assert records[0].text == "  提前30日申请。  "


def test_heading_context_prevents_unsafe_merge(unit_factory):
    records = [unit_factory("a", "最低金额10000元", ["商业贷款"]), unit_factory("b", "最低金额10000元", ["公积金贷款"])]
    cleaned, _ = clean_units(records, CleaningConfig(recipe=RECIPE, near_duplicates=False))
    assert len(cleaned) == 2


def test_minhash_flags_without_dropping_numeric_difference(unit_factory):
    common = "".join(f"第{i}条：借款人提交第{i}项申请材料，业务机构核验第{i}项材料并保存审查记录。" for i in range(30))
    records = [unit_factory("a", common + "提前30日申请"), unit_factory("b", common + "提前15日申请")]
    cleaned, audit = clean_units(records, CleaningConfig(recipe=RECIPE, minhash_threshold=0.8))
    assert len(cleaned) == 2
    assert any(unit.near_duplicate_candidate for unit in cleaned)
    assert any(row["action"] == "near_duplicate_flagged" for row in audit)


def test_short_records_are_not_minhash_duplicates(unit_factory):
    cleaned, _ = clean_units([unit_factory("a", "甲"), unit_factory("b", "乙")], CleaningConfig(recipe=RECIPE))
    assert len(cleaned) == 2
    assert not any(unit.near_duplicate_candidate for unit in cleaned)


def test_empty_input():
    assert clean_units([], CleaningConfig(recipe=RECIPE)) == ([], [])
