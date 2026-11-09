import pytest

from ai_ready.valuation.conditional_value import (
    ConditionalValueUnit,
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeObservation,
    ScoreInterval,
    UnitCellMapping,
    build_value_profile,
    compare_value_profiles,
    reconstruct_value_profile,
)


def _observation(cell_id, weight):
    return ProbeObservation(
        cell_id=cell_id,
        capability_id=cell_id,
        operation_id="decision",
        difficulty_id="medium",
        benchmark_weight=weight,
        point_score=0.8,
        score_interval=ScoreInterval(lower=0.7, upper=0.9),
        probe_count=20,
        probabilities=MasteryProbabilities.one_hot("mastered"),
        stratum_state="mastered",
        scoring_method="rubric",
    )


def _profile():
    return ModelMasteryProfile(
        profile_id="weighted-blueprint",
        benchmark_id="benchmark-a",
        model_id="base",
        model_revision="r1",
        checkpoint_sha256="a" * 64,
        probe_set_id="probes",
        probe_set_sha256="b" * 64,
        scoring_protocol_sha256="c" * 64,
        observations=[_observation("diagnosis", 4.0), _observation("treatment", 1.0)],
    )


def _unit(identifier, cell, probability=1.0):
    return ConditionalValueUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="d" * 64,
        dependency_group_id=identifier,
        cell_mappings=[UnitCellMapping(cell_id=cell, weight=1.0)],
        data_attributes={"knowledge_type": "rule"},
        inclusion_probability=probability,
    )


def test_registered_benchmark_weights_drive_default_task_sensitive_distance():
    model = _profile()
    full = build_value_profile(
        [_unit("d", "diagnosis"), _unit("t", "treatment")],
        model,
        profile_id="full",
    )
    diagnosis_only = reconstruct_value_profile(
        [_unit("d", "diagnosis", probability=0.5)],
        model,
        profile_id="diagnosis-only",
    )

    comparison = compare_value_profiles(full, diagnosis_only)

    assert full.benchmark_cell_weights == {"diagnosis": 4.0, "treatment": 1.0}
    assert comparison.joint_total_variation == pytest.approx(0.5)
    assert comparison.critical_weighted_total_variation == pytest.approx(0.2)
