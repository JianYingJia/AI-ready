import pytest

from ai_ready.valuation.conditional_value import (
    FidelityThresholds,
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeObservation,
    ScoreInterval,
)
from ai_ready.valuation.value_sampling import ValueSamplingUnit
from ai_ready.valuation.value_sweep import (
    ValueFidelitySweepConfig,
    run_value_fidelity_sweep,
    value_population_units_digest,
)


def _profile() -> ModelMasteryProfile:
    observations = []
    for capability_id, state, score in (("diagnosis", "mastered", 0.9), ("treatment", "unmastered", 0.2)):
        observations.append(
            ProbeObservation(
                cell_id=capability_id,
                capability_id=capability_id,
                operation_id="decision",
                difficulty_id="medium",
                point_score=score,
                score_interval=ScoreInterval(lower=score - 0.05, upper=score + 0.05),
                probe_count=20,
                probabilities=MasteryProbabilities.one_hot(state),
                stratum_state=state,
                scoring_method="test-rubric",
            )
        )
    return ModelMasteryProfile(
        profile_id="sweep-model",
        benchmark_id="medical-benchmark",
        model_id="base-model",
        model_revision="revision-1",
        checkpoint_sha256="a" * 64,
        probe_set_id="probe-set",
        probe_set_sha256="b" * 64,
        scoring_protocol_sha256="c" * 64,
        observations=observations,
    )


def _units() -> list[ValueSamplingUnit]:
    rows = []
    for index in range(4):
        capability_id = "diagnosis" if index < 2 else "treatment"
        state = "mastered" if capability_id == "diagnosis" else "unmastered"
        rows.append(
            ValueSamplingUnit(
                unit_id=f"u{index}",
                batch_id="B01",
                batch_content_sha256="d" * 64,
                dependency_group_id=f"g{index}",
                capability_ids=[capability_id],
                model_states={capability_id: state},
                data_attributes={
                    "knowledge_type": "rule",
                    "quality_bucket": "clean",
                    "generation_status": "rejected" if index == 0 else "accepted",
                },
            )
        )
    return rows


def _thresholds() -> FidelityThresholds:
    return FidelityThresholds(
        maximum_joint_total_variation=0.01,
        maximum_critical_weighted_total_variation=0.01,
        maximum_cell_marginal_total_variation=0.01,
        maximum_cell_proportion_error=0.01,
        maximum_mastery_marginal_total_variation=0.01,
        maximum_attribute_marginal_total_variation=0.01,
        minimum_effective_sample_size=3,
    )


def test_sweep_finds_smallest_budget_that_passes_every_seed():
    result = run_value_fidelity_sweep(
        _units(),
        _profile(),
        _thresholds(),
        ValueFidelitySweepConfig(
            sampling_fractions=[0.5, 1.0],
            seeds=[3, 7],
            design_stratum_attributes=["knowledge_type", "quality_bucket"],
            audit_only_attributes=["generation_status"],
        ),
    )

    assert result["schema_version"] == "conditional-value-fidelity-sweep-v2"
    assert result["claim_status"] == "complete"
    assert result["model_measurement_status"] == "complete"
    assert result["fidelity_status"] == "passed"
    assert result["model_dimension_evaluated"] is True
    assert result["input_unit_count"] == 4
    assert len(result["population_units_sha256"]) == 64
    assert len(result["model_profile_sha256"]) == 64
    assert result["population_units_sha256"] == value_population_units_digest(list(reversed(_units())))
    assert result["attribute_roles"] == {
        "design_stratum_attributes": ["knowledge_type", "quality_bucket"],
        "audit_only_attributes": ["generation_status"],
        "additional_profile_audit_attributes": [],
        "certificate_audited_attributes": [
            "generation_status",
            "knowledge_type",
            "quality_bucket",
        ],
    }
    assert result["batch_summary"][0]["minimum_grid_fraction_all_seeds_pass"] == 1.0
    assert result["batch_summary"][0]["fidelity_status"] == "passed"
    assert result["cells"][0]["all_seeds_pass"] is False
    assert result["cells"][0]["audit_only_attributes_all_seeds_pass"] is False
    assert result["cells"][0]["maximum_audit_only_attribute_total_variation"] == pytest.approx(0.25)
    assert result["cells"][0]["trials"][0]["attribute_total_variation_by_role"]["audit_only_attributes"] == {
        "generation_status": pytest.approx(0.25)
    }
    assert result["cells"][1]["all_seeds_pass"] is True
    assert result["cells"][1]["audit_only_attributes_all_seeds_pass"] is True


def test_sweep_rejects_model_states_not_bound_to_profile():
    units = _units()
    units[0] = units[0].model_copy(update={"model_states": {"diagnosis": "unmastered"}})
    with pytest.raises(ValueError, match="stale model_states"):
        run_value_fidelity_sweep(
            units,
            _profile(),
            _thresholds(),
            ValueFidelitySweepConfig(
                sampling_fractions=[1.0],
                seeds=[3],
                stratum_attributes=["knowledge_type"],
            ),
        )


def test_sweep_rejects_attribute_used_for_both_design_and_holdout():
    with pytest.raises(ValueError, match="held out from the sampling strata"):
        ValueFidelitySweepConfig(
            sampling_fractions=[1.0],
            seeds=[3],
            design_stratum_attributes=["knowledge_type"],
            audit_only_attributes=["knowledge_type"],
        )


def test_sweep_requires_holdout_attribute_in_population_profile():
    units = [
        unit.model_copy(
            update={
                "data_attributes": {
                    key: value for key, value in unit.data_attributes.items() if key != "generation_status"
                }
            }
        )
        for unit in _units()
    ]
    with pytest.raises(ValueError, match="missing registered sweep attributes"):
        run_value_fidelity_sweep(
            units,
            _profile(),
            _thresholds(),
            ValueFidelitySweepConfig(
                sampling_fractions=[1.0],
                seeds=[3],
                design_stratum_attributes=["knowledge_type"],
                audit_only_attributes=["generation_status"],
            ),
        )


def test_sweep_reports_no_qualifying_budget_separately_from_model_measurement():
    result = run_value_fidelity_sweep(
        _units(),
        _profile(),
        _thresholds(),
        ValueFidelitySweepConfig(
            sampling_fractions=[0.5],
            seeds=[3, 7],
            design_stratum_attributes=["knowledge_type", "quality_bucket"],
            audit_only_attributes=["generation_status"],
        ),
    )

    assert result["model_measurement_status"] == "complete"
    assert result["fidelity_status"] == "no_qualifying_budget"
    assert result["claim_status"] == "no_qualifying_budget"
    assert result["batch_summary"][0]["fidelity_status"] == "no_qualifying_budget"
    assert result["batch_summary"][0]["minimum_grid_fraction_all_seeds_pass"] is None


def test_sweep_excludes_whole_mixed_eligibility_dependency_group_from_population():
    units = _units()
    units[0] = units[0].model_copy(update={"dependency_group_id": "mixed"})
    units.append(
        ValueSamplingUnit(
            unit_id="ineligible-partner",
            batch_id="B01",
            batch_content_sha256="d" * 64,
            dependency_group_id="mixed",
            capability_ids=["diagnosis"],
            model_states={"diagnosis": "mastered"},
            data_attributes={
                "knowledge_type": "rule",
                "quality_bucket": "clean",
                "generation_status": "accepted",
            },
            eligible=False,
        )
    )

    result = run_value_fidelity_sweep(
        units,
        _profile(),
        _thresholds(),
        ValueFidelitySweepConfig(
            sampling_fractions=[1.0],
            seeds=[3],
            design_stratum_attributes=["knowledge_type", "quality_bucket"],
            audit_only_attributes=["generation_status"],
        ),
    )

    assert result["batch_summary"][0]["population_group_count"] == 3
    assert result["batch_summary"][0]["fidelity_status"] == "passed"
    assert result["cells"][0]["trials"][0]["realized_group_count"] == 3
