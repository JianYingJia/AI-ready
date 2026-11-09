import pytest
from pydantic import ValidationError

from ai_ready.valuation.conditional_value import (
    FidelityThresholds,
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeObservation,
    ScoreInterval,
    build_value_profile,
    conditional_value_unit_from_sampling_record,
)
from ai_ready.valuation.unit_mastery import (
    UnitCellMasteryObservation,
    build_unit_cell_mastery_profile,
    validate_unit_cell_mastery_profile,
)
from ai_ready.valuation.value_sampling import ValueSamplingUnit
from ai_ready.valuation.value_sweep import (
    ValueFidelitySweepConfig,
    run_value_fidelity_sweep,
)


def _parent_profile():
    return ModelMasteryProfile(
        profile_id="m0-parent",
        benchmark_id="medical-benchmark",
        model_id="base-model",
        model_revision="revision-1",
        checkpoint_sha256="a" * 64,
        probe_set_id="capability-prior",
        probe_set_sha256="b" * 64,
        scoring_protocol_sha256="c" * 64,
        observations=[
            ProbeObservation(
                cell_id="diagnosis",
                capability_id="diagnosis",
                operation_id="__absent__",
                difficulty_id="__absent__",
                probe_count=0,
                probabilities=MasteryProbabilities.one_hot("unknown"),
                stratum_state="unknown",
                scoring_method="pending",
            )
        ],
    )


def _unit(identifier):
    return ValueSamplingUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="d" * 64,
        dependency_group_id=identifier,
        capability_ids=["diagnosis"],
        model_states={"diagnosis": "unknown"},
        data_attributes={"knowledge_type": "rule", "generation_status": "accepted"},
    )


def _measurement(identifier, state="mastered"):
    probabilities = MasteryProbabilities.one_hot(state)
    point_score, lower, upper = {
        "mastered": (0.9, 0.8, 1.0),
        "uncertain": (0.5, 0.4, 0.6),
        "unmastered": (0.1, 0.0, 0.2),
    }[state]
    return UnitCellMasteryObservation(
        unit_id=identifier,
        cell_id="diagnosis",
        point_score=point_score,
        score_interval=ScoreInterval(lower=lower, upper=upper),
        probe_count=2,
        probe_ids=[f"{identifier}-p1", f"{identifier}-p2"],
        probabilities=probabilities,
        stratum_state=state,
    )


def test_unit_profile_binds_every_eligible_unit_cell_to_parent_and_population():
    parent = _parent_profile()
    units = [_unit("u1"), _unit("u2")]
    profile = build_unit_cell_mastery_profile(
        profile_id="unit-profile",
        probe_set_id="unit-probes",
        scoring_protocol_sha256="e" * 64,
        parent=parent,
        units=units,
        observations=[_measurement("u1"), _measurement("u2", "uncertain")],
    )

    validate_unit_cell_mastery_profile(profile, parent, units)
    assert profile.measurement_granularity == "unit_cell"
    assert len(profile.by_unit_cell) == 2


def test_unit_profile_rejects_missing_unit_cell_measurement():
    with pytest.raises(ValueError, match="missing unit-cell observations"):
        build_unit_cell_mastery_profile(
            profile_id="unit-profile",
            probe_set_id="unit-probes",
            scoring_protocol_sha256="e" * 64,
            parent=_parent_profile(),
            units=[_unit("u1"), _unit("u2")],
            observations=[_measurement("u1")],
        )


def test_unit_measurement_rejects_locked_benchmark_item():
    with pytest.raises(ValidationError, match="must not use locked benchmark"):
        _measurement("u1").model_copy(update={"used_locked_test_items": True}).__class__.model_validate(
            {**_measurement("u1").model_dump(), "used_locked_test_items": True}
        )


def test_unit_cell_probabilities_override_unknown_capability_prior_in_profile_and_sweep():
    parent = _parent_profile()
    units = []
    for identifier, state in (("u1", "mastered"), ("u2", "unmastered")):
        unit = _unit(identifier).model_copy(
            update={
                "model_states": {"diagnosis": state},
                "model_state_probabilities": {"diagnosis": MasteryProbabilities.one_hot(state).model_dump(mode="json")},
                "model_state_granularity": "unit_cell",
            }
        )
        units.append(ValueSamplingUnit.model_validate(unit.model_dump()))

    conditional = [conditional_value_unit_from_sampling_record(unit) for unit in units]
    profile = build_value_profile(conditional, parent, profile_id="unit-level-full")
    assert profile.unmeasured_cell_ids == []
    assert profile.unknown_mastery_mass == 0
    assert profile.overridden_unit_count == 2

    result = run_value_fidelity_sweep(
        units,
        parent,
        FidelityThresholds(
            maximum_joint_total_variation=1,
            maximum_critical_weighted_total_variation=1,
            maximum_cell_marginal_total_variation=1,
            maximum_cell_proportion_error=1,
            maximum_mastery_marginal_total_variation=1,
            maximum_attribute_marginal_total_variation=1,
            maximum_ht_total_relative_error=1,
            minimum_effective_sample_size=1,
        ),
        ValueFidelitySweepConfig(
            sampling_fractions=[1],
            seeds=[7],
            design_stratum_attributes=["knowledge_type"],
            audit_only_attributes=["generation_status"],
        ),
    )
    assert result["model_measurement_status"] == "complete"
    assert result["model_dimension_evaluated"] is True
    assert result["model_measurement_granularity"] == "unit_cell"
