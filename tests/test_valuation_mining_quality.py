import pytest

from ai_ready.valuation.mining import (
    ActivityRule,
    build_coverage_obligations,
    fixed_chunk_baseline,
    mine_semantic_units,
)
from ai_ready.valuation.quality import (
    SFTSample,
    UnitDisposition,
    validate_full_traversal,
)


def test_mining_tags_only_source_spans_and_skips_window_copies(unit_factory):
    source = unit_factory("rule", "若患者无症状，可以观察等待；但是出现进展时不得继续等待。", ["治疗指征"])
    window = unit_factory("window", "若患者无症状，可以观察等待。")
    window.is_window_copy = True
    units = mine_semantic_units(
        [source, window],
        batch_id="B01",
        activity_rules=[ActivityRule(activity_id="treatment_indication", keywords=["观察等待"])],
    )
    assert len(units) == 1
    assert units[0].content == source.text
    assert units[0].condition_spans
    assert units[0].exception_spans
    assert units[0].activity_ids == ["treatment_indication"]
    obligations = build_coverage_obligations(units)
    assert {row.kind for row in obligations} >= {"rule", "condition", "exception", "topic"}


def test_fixed_chunk_baseline_uses_same_source_but_drops_explicit_rule_links(unit_factory):
    source = unit_factory("rule", "若患者无症状，可以观察等待；但是出现进展时不得继续等待。", ["治疗指征"])
    chunks = fixed_chunk_baseline(
        [source],
        batch_id="B01",
        activity_rules=[ActivityRule(activity_id="treatment_indication", keywords=["观察等待"])],
        chunk_characters=12,
    )

    assert len(chunks) > 1
    assert "".join(row.content for row in chunks) == source.text
    assert all(row.evidence_unit_ids == [source.id] for row in chunks)
    assert all(not row.condition_spans and not row.exception_spans for row in chunks)
    assert len({row.semantic_unit_id for row in chunks}) == len(chunks)


def test_full_traversal_requires_disposition_and_traceability():
    sample = SFTSample(
        sample_id="s1",
        batch_id="B01",
        semantic_unit_ids=["u1"],
        instruction="何时可以观察等待？",
        response="仅在材料所述适用条件成立时。",
        evidence_spans=["若患者无症状，可以观察等待。"],
        source_ids=["source-1"],
        generator="test",
        generator_revision="v1",
    )
    valid = validate_full_traversal(
        ["u1", "u2"],
        [
            UnitDisposition(semantic_unit_id="u1", status="accepted", sample_ids=["s1"]),
            UnitDisposition(semantic_unit_id="u2", status="rejected", reason_codes=["unsupported_answer"]),
        ],
        [sample],
    )
    assert valid.valid
    assert valid.summary["accepted_unit_yield"] == 0.5
    missing = validate_full_traversal(
        ["u1", "u2"],
        [UnitDisposition(semantic_unit_id="u1", status="accepted", sample_ids=["s1"])],
        [sample],
    )
    assert not missing.valid
    assert any("no accepted/rejected/failed" in error for error in missing.errors)


def test_rejected_unit_cannot_hitchhike_in_an_accepted_sample():
    sample = SFTSample(
        sample_id="s1",
        batch_id="B01",
        semantic_unit_ids=["accepted", "rejected"],
        instruction="综合两条规则回答。",
        response="answer",
        evidence_spans=["evidence"],
        source_ids=["source-1"],
        generator="test",
        generator_revision="v1",
    )
    report = validate_full_traversal(
        ["accepted", "rejected"],
        [
            UnitDisposition(semantic_unit_id="accepted", status="accepted", sample_ids=["s1"]),
            UnitDisposition(semantic_unit_id="rejected", status="rejected", reason_codes=["medical_error"]),
        ],
        [sample],
    )
    assert not report.valid
    assert any("references non-accepted unit rejected" in error for error in report.errors)


def test_rejected_disposition_cannot_publish_samples():
    with pytest.raises(ValueError, match="cannot contribute SFT samples"):
        UnitDisposition(
            semantic_unit_id="u1",
            status="rejected",
            sample_ids=["s1"],
            reason_codes=["medical_error"],
        )


def test_full_traversal_rejects_duplicated_supervision_to_fill_budget():
    samples = [
        SFTSample(
            sample_id=sample_id,
            batch_id="B01",
            semantic_unit_ids=[unit_id],
            instruction="相同问题",
            response="相同答案",
            evidence_spans=["source evidence"],
            source_ids=["source-1"],
            generator="test",
            generator_revision="v1",
        )
        for sample_id, unit_id in (("s1", "u1"), ("s2", "u2"))
    ]
    report = validate_full_traversal(
        ["u1", "u2"],
        [
            UnitDisposition(semantic_unit_id="u1", status="accepted", sample_ids=["s1"]),
            UnitDisposition(semantic_unit_id="u2", status="accepted", sample_ids=["s2"]),
        ],
        samples,
    )
    assert not report.valid
    assert any("Duplicate SFT supervision" in error for error in report.errors)
