import math

import pytest
from pydantic import ValidationError

from ai_ready.valuation.conditional_value import (
    BenchmarkCellDefinition,
    ConditionalValueUnit,
    FidelityThresholds,
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeAggregationConfig,
    ProbeObservation,
    ScoredProbe,
    ScoreInterval,
    UnitCellMapping,
    aggregate_probe_observations,
    build_fidelity_certificate,
    build_raw_sample_profile,
    build_value_profile,
    compare_value_profiles,
    conditional_value_unit_from_sampling_record,
    reconstruct_value_profile,
    verify_fidelity_certificate,
)

SHA = "a" * 64


def _observation(cell_id, probabilities=None):
    return ProbeObservation(
        cell_id=cell_id,
        capability_id=cell_id,
        operation_id="clinical_decision",
        difficulty_id="medium",
        point_score=0.6,
        score_interval=ScoreInterval(lower=0.5, upper=0.7),
        probe_count=20,
        probabilities=probabilities or MasteryProbabilities(mastered=0.25, uncertain=0.25, unmastered=0.5, unknown=0),
        stratum_state="uncertain",
        scoring_method="registered-rubric-v1",
    )


def _profile(observations=None):
    return ModelMasteryProfile(
        profile_id="base-model-v1",
        benchmark_id="medical-benchmark",
        model_id="base-model",
        model_revision="revision-1",
        checkpoint_sha256="b" * 64,
        probe_set_id="parallel-probes-v1",
        probe_set_sha256="c" * 64,
        scoring_protocol_sha256="d" * 64,
        observations=observations or [_observation("diagnosis"), _observation("treatment")],
    )


def _unit(identifier, cell, quality, probability=1.0, override=None):
    return ConditionalValueUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256=SHA,
        dependency_group_id=identifier,
        cell_mappings=[UnitCellMapping(cell_id=cell, weight=1.0)],
        data_attributes={"knowledge_type": "rule", "quality": quality, "source": "guideline"},
        inclusion_probability=probability,
        model_state_overrides={cell: override} if override is not None else {},
    )


def _thresholds():
    return FidelityThresholds(
        maximum_joint_total_variation=0.01,
        maximum_critical_weighted_total_variation=0.01,
        maximum_cell_marginal_total_variation=0.01,
        maximum_cell_proportion_error=0.01,
        maximum_mastery_marginal_total_variation=0.01,
        maximum_attribute_marginal_total_variation=0.01,
        minimum_effective_sample_size=1,
    )


def test_probe_observation_requires_unknown_for_unmeasured_cell():
    observation = ProbeObservation(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
        probe_count=0,
        probabilities=MasteryProbabilities.one_hot("unknown"),
        stratum_state="unknown",
        scoring_method="pending",
    )
    assert observation.point_score is None
    assert observation.probabilities.unknown == 1

    with pytest.raises(ValidationError, match="must have probability 1 on unknown"):
        ProbeObservation(
            cell_id="diagnosis",
            capability_id="diagnosis",
            operation_id="decision",
            difficulty_id="medium",
            probe_count=0,
            probabilities=MasteryProbabilities.one_hot("unmastered"),
            stratum_state="unknown",
            scoring_method="pending",
        )


def test_locked_benchmark_items_cannot_be_used_for_model_profile():
    with pytest.raises(ValidationError, match="must not use locked benchmark test items"):
        _observation("diagnosis").model_copy(
            update={"used_locked_test_items": True}, deep=True
        ).__class__.model_validate({**_observation("diagnosis").model_dump(), "used_locked_test_items": True})


def test_seeded_probe_bootstrap_builds_soft_probabilities_and_ci_state():
    cells = [
        BenchmarkCellDefinition(
            cell_id="diagnosis",
            capability_id="diagnosis",
            operation_id="decision",
            difficulty_id="medium",
        ),
        BenchmarkCellDefinition(
            cell_id="treatment",
            capability_id="treatment",
            operation_id="selection",
            difficulty_id="hard",
        ),
    ]
    probes = [
        ScoredProbe(probe_id=f"d-{index}", item_id=f"item-{index}", cell_id="diagnosis", score=score)
        for index, score in enumerate([0.8, 0.9, 0.7, 1.0, 0.8, 0.9])
    ]
    config = ProbeAggregationConfig(
        tau_low=0.4,
        tau_high=0.7,
        confidence_level=0.9,
        bootstrap_iterations=200,
        seed=17,
        minimum_probe_count=5,
        scoring_method="rubric-v1",
    )
    first = aggregate_probe_observations(cells, probes, config)
    second = aggregate_probe_observations(list(reversed(cells)), probes, config)

    assert first == second
    diagnosis, treatment = first
    assert diagnosis.stratum_state == "mastered"
    assert diagnosis.probabilities.unknown == 0
    assert sum(diagnosis.probabilities.as_dict().values()) == pytest.approx(1)
    assert treatment.stratum_state == "unknown"
    assert treatment.probe_count == 0
    assert treatment.point_score is None


def test_probe_aggregation_rejects_locked_test_items():
    cell = BenchmarkCellDefinition(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
    )
    with pytest.raises(ValidationError, match="Locked benchmark test items"):
        ScoredProbe(
            probe_id="locked",
            item_id="locked-item",
            cell_id=cell.cell_id,
            score=1.0,
            is_locked_test_item=True,
        )


def test_prompt_variants_do_not_inflate_probe_item_count():
    cell = BenchmarkCellDefinition(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
    )
    probes = [
        ScoredProbe(
            probe_id=f"probe-{variant}",
            item_id="same-item",
            variant_id=f"variant-{variant}",
            cell_id="diagnosis",
            score=score,
        )
        for variant, score in enumerate([0.8, 0.9, 1.0])
    ]
    config = ProbeAggregationConfig(
        tau_low=0.4,
        tau_high=0.7,
        bootstrap_iterations=100,
        seed=3,
        minimum_probe_count=2,
        scoring_method="rubric-v1",
    )

    observation = aggregate_probe_observations([cell], probes, config)[0]
    assert observation.probe_count == 1
    assert observation.stratum_state == "unknown"
    assert observation.point_score is None


def test_probe_aggregation_uses_inclusive_tau_high_for_soft_and_hard_states():
    cell = BenchmarkCellDefinition(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
    )
    probes = [
        ScoredProbe(
            probe_id=f"boundary-{index}",
            item_id=f"item-{index}",
            cell_id="diagnosis",
            score=0.75,
        )
        for index in range(3)
    ]
    config = ProbeAggregationConfig(
        tau_low=0.4,
        tau_high=0.75,
        bootstrap_iterations=100,
        seed=3,
        minimum_probe_count=3,
        scoring_method="rubric-v1",
    )

    observation = aggregate_probe_observations([cell], probes, config)[0]

    assert observation.score_interval.lower == pytest.approx(config.tau_high)
    assert observation.probabilities.mastered == 1.0
    assert observation.stratum_state == "mastered"


def test_probe_aggregation_rejects_duplicate_item_variant_pairs():
    cell = BenchmarkCellDefinition(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
    )
    probes = [
        ScoredProbe(
            probe_id=probe_id,
            item_id="same-item",
            variant_id="same-variant",
            cell_id="diagnosis",
            score=score,
        )
        for probe_id, score in (("first", 0.6), ("second", 0.8))
    ]
    config = ProbeAggregationConfig(
        tau_low=0.4,
        tau_high=0.7,
        bootstrap_iterations=100,
        seed=3,
        minimum_probe_count=1,
        scoring_method="rubric-v1",
    )

    with pytest.raises(ValueError, match=r"unique \(item_id, variant_id\) pairs"):
        aggregate_probe_observations([cell], probes, config)


def test_full_population_profile_softly_splits_cell_mastery_mass():
    profile = _profile()
    units = [_unit("d1", "diagnosis", "clean"), _unit("t1", "treatment", "usable_noisy")]
    result = build_value_profile(units, profile, profile_id="population")

    assert result.total_mass == pytest.approx(2.0)
    assert sum(result.distribution.values()) == pytest.approx(1.0)
    mastery = {}
    for stratum in result.strata:
        mastery[stratum.key.mastery_state] = mastery.get(stratum.key.mastery_state, 0) + stratum.proportion
    assert mastery == pytest.approx({"mastered": 0.25, "uncertain": 0.25, "unmastered": 0.5})
    assert result.unknown_mastery_mass == 0


def test_ht_reconstruction_recovers_population_while_raw_sample_is_biased():
    profile = _profile()
    population = [
        _unit("d1", "diagnosis", "clean"),
        _unit("d2", "diagnosis", "clean"),
        _unit("t1", "treatment", "clean"),
    ]
    selected = [
        _unit("d1", "diagnosis", "clean", probability=0.5),
        _unit("t1", "treatment", "clean", probability=1.0),
    ]
    full = build_value_profile(population, profile, profile_id="population")
    raw = build_raw_sample_profile(selected, profile, profile_id="raw")
    reconstructed = reconstruct_value_profile(selected, profile, profile_id="ht")

    assert compare_value_profiles(full, raw).joint_total_variation == pytest.approx(1 / 6)
    assert compare_value_profiles(full, reconstructed).joint_total_variation == pytest.approx(0)
    assert reconstructed.effective_sample_size == pytest.approx(9 / 5)


def test_sampling_adapter_preserves_multiple_capabilities_and_design_probability():
    unit = conditional_value_unit_from_sampling_record(
        {
            "unit_id": "u1",
            "batch_id": "B01",
            "batch_content_sha256": SHA,
            "dependency_group_id": "g1",
            "capability_ids": ["treatment", "diagnosis"],
            "model_states": {"diagnosis": "mastered", "treatment": "uncertain"},
            "data_attributes": {"quality": "clean"},
            "population_weight": 2.0,
            "eligible": True,
            "inclusion_probability": 0.25,
            "design_weight": 4.0,
            "stratum_id": "sampler-stratum",
        },
        use_model_state_overrides=True,
    )

    assert [(mapping.cell_id, mapping.weight) for mapping in unit.cell_mappings] == [
        ("diagnosis", 0.5),
        ("treatment", 0.5),
    ]
    assert unit.model_state_overrides == {
        "diagnosis": MasteryProbabilities.one_hot("mastered"),
        "treatment": MasteryProbabilities.one_hot("uncertain"),
    }
    assert unit.inclusion_probability == 0.25

    with pytest.raises(ValueError, match="design_weight must equal"):
        conditional_value_unit_from_sampling_record(
            {
                **unit.model_dump(exclude={"cell_mappings", "model_state_overrides"}),
                "capability_ids": ["diagnosis"],
                "model_states": {"diagnosis": "uncertain"},
                "design_weight": 99,
            }
        )


def test_certificate_reports_raw_and_ht_fidelity_and_verifies_digests():
    profile = _profile()
    population = [
        _unit("d1", "diagnosis", "clean"),
        _unit("d2", "diagnosis", "clean"),
        _unit("t1", "treatment", "clean"),
    ]
    selected = [
        _unit("d1", "diagnosis", "clean", probability=0.5),
        _unit("t1", "treatment", "clean", probability=1.0),
    ]
    full = build_value_profile(population, profile, profile_id="population")
    raw = build_raw_sample_profile(selected, profile, profile_id="raw")
    ht = reconstruct_value_profile(selected, profile, profile_id="ht")
    certificate = build_fidelity_certificate(
        full,
        raw,
        ht,
        certificate_id="certificate-1",
        thresholds=_thresholds(),
        critical_cell_weights={"diagnosis": 2.0, "treatment": 1.0},
    )

    assert certificate.raw_sample_comparison.joint_total_variation == pytest.approx(1 / 6)
    assert certificate.weighted_comparison.joint_total_variation == pytest.approx(0)
    assert certificate.passed
    assert certificate.complete_three_dimensional_claim
    verify_fidelity_certificate(certificate, full, raw, ht)


def test_missing_micro_stratum_is_reported_but_governed_by_registered_distance_bounds():
    profile = _profile()
    population = [
        _unit("d-clean", "diagnosis", "clean"),
        _unit("d-noisy", "diagnosis", "usable_noisy"),
    ]
    selected = [_unit("d-clean", "diagnosis", "clean", probability=0.5)]
    full = build_value_profile(population, profile, profile_id="population-micro")
    raw = build_raw_sample_profile(selected, profile, profile_id="raw-micro")
    ht = reconstruct_value_profile(selected, profile, profile_id="ht-micro")
    permissive = FidelityThresholds(
        maximum_joint_total_variation=1.0,
        maximum_critical_weighted_total_variation=1.0,
        maximum_cell_marginal_total_variation=1.0,
        maximum_cell_proportion_error=1.0,
        maximum_mastery_marginal_total_variation=1.0,
        maximum_attribute_marginal_total_variation=1.0,
        minimum_effective_sample_size=1.0,
    )
    certificate = build_fidelity_certificate(
        full,
        raw,
        ht,
        certificate_id="certificate-micro",
        thresholds=permissive,
        critical_cell_weights={"diagnosis": 2.0},
    )

    assert certificate.weighted_comparison.critical_missing_strata
    assert certificate.distribution_fidelity_passed
    assert "critical_strata_present" not in certificate.checks


def test_effective_sample_requirement_is_capped_by_full_population_information():
    profile = _profile()
    population = [_unit("d1", "diagnosis", "clean"), _unit("d2", "diagnosis", "clean")]
    full = build_value_profile(population, profile, profile_id="small-population")
    raw = build_raw_sample_profile(population, profile, profile_id="small-raw")
    ht = reconstruct_value_profile(population, profile, profile_id="small-ht")
    thresholds = _thresholds().model_copy(update={"minimum_effective_sample_size": 10.0})
    certificate = build_fidelity_certificate(
        full,
        raw,
        ht,
        certificate_id="small-certificate",
        thresholds=thresholds,
    )

    assert certificate.required_effective_sample_size == pytest.approx(2.0)
    assert certificate.checks["effective_sample_size"]


def test_unknown_model_mass_makes_certificate_partial_even_with_perfect_profile_match():
    unknown = ProbeObservation(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
        probe_count=0,
        probabilities=MasteryProbabilities.one_hot("unknown"),
        stratum_state="unknown",
        scoring_method="pending",
    )
    profile = _profile([unknown])
    population = [_unit("d1", "diagnosis", "clean")]
    selected = [_unit("d1", "diagnosis", "clean", probability=1.0)]
    full = build_value_profile(population, profile, profile_id="population")
    raw = build_raw_sample_profile(selected, profile, profile_id="raw")
    ht = reconstruct_value_profile(selected, profile, profile_id="ht")
    certificate = build_fidelity_certificate(
        full,
        raw,
        ht,
        certificate_id="certificate-pending",
        thresholds=_thresholds(),
    )

    assert full.unmeasured_cell_ids == ["diagnosis"]
    assert math.isclose(full.unknown_mastery_mass, 1.0)
    assert certificate.distribution_fidelity_passed
    assert not certificate.passed
    assert not certificate.complete_three_dimensional_claim
    assert certificate.status == "partial_pending_model_measurement"


def test_unknown_mastery_mass_is_clamped_against_floating_point_overflow():
    unknown = ProbeObservation(
        cell_id="diagnosis",
        capability_id="diagnosis",
        operation_id="decision",
        difficulty_id="medium",
        probe_count=0,
        probabilities=MasteryProbabilities.one_hot("unknown"),
        stratum_state="unknown",
        scoring_method="pending",
    )
    profile = _profile([unknown])
    units = [_unit(f"d{index}", "diagnosis", f"quality-{index}") for index in range(11)]

    result = build_value_profile(units, profile, profile_id="unknown-population")
    assert result.unknown_mastery_mass == 1.0


def test_profiles_from_different_model_contexts_cannot_be_compared():
    unit = _unit("d1", "diagnosis", "clean")
    left = build_value_profile([unit], _profile(), profile_id="left")
    other_profile = _profile().model_copy(update={"profile_id": "other-model-profile"})
    right = build_value_profile([unit], other_profile, profile_id="right")

    with pytest.raises(ValueError, match="incompatible conditioning context"):
        compare_value_profiles(left, right)


def test_critical_weights_must_reference_cells_in_reference_profile():
    profile = _profile()
    full = build_value_profile(
        [_unit("d1", "diagnosis", "clean")],
        profile,
        profile_id="reference",
    )
    candidate = build_value_profile(
        [_unit("d1", "diagnosis", "clean")],
        profile,
        profile_id="candidate",
    )

    with pytest.raises(ValueError, match="absent from the reference profile"):
        compare_value_profiles(
            full,
            candidate,
            critical_cell_weights={"nonexistent-cell": 2.0},
        )


def _grouped_unit(identifier, group_id, *, probability=1.0, population_weight=1.0):
    return ConditionalValueUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256=SHA,
        dependency_group_id=group_id,
        cell_mappings=[UnitCellMapping(cell_id="diagnosis", weight=1.0)],
        data_attributes={"knowledge_type": "rule", "quality": "clean", "source": "guideline"},
        population_weight=population_weight,
        inclusion_probability=probability,
    )


def test_effective_sample_size_uses_dependency_groups_instead_of_semantic_rows():
    population = [
        _grouped_unit("rule", "rule-with-exception"),
        _grouped_unit("exception", "rule-with-exception"),
        _grouped_unit("independent", "independent"),
    ]

    value_profile = build_value_profile(population, _profile(), profile_id="group-ess")

    assert value_profile.unit_count == 3
    assert value_profile.dependency_group_count == 2
    assert value_profile.effective_sample_size == pytest.approx(9 / 5)
    assert [group.unit_ids for group in value_profile.dependency_groups] == [
        ["independent"],
        ["exception", "rule"],
    ]


def test_profile_rejects_different_probabilities_inside_one_dependency_group():
    sample = [
        _grouped_unit("rule", "clinical-rule", probability=0.5),
        _grouped_unit("exception", "clinical-rule", probability=0.25),
    ]

    with pytest.raises(ValueError, match="one shared inclusion_probability"):
        reconstruct_value_profile(sample, _profile(), profile_id="invalid-group-design")


def test_certificate_requires_complete_atomic_population_groups():
    population = [
        _grouped_unit("rule", "clinical-rule"),
        _grouped_unit("exception", "clinical-rule"),
        _grouped_unit("independent", "independent"),
    ]
    incomplete_sample = [
        _grouped_unit("rule", "clinical-rule", probability=0.5),
        _grouped_unit("independent", "independent", probability=1.0),
    ]
    profile = _profile()
    full = build_value_profile(population, profile, profile_id="atomic-full")
    raw = build_raw_sample_profile(incomplete_sample, profile, profile_id="atomic-raw")
    ht = reconstruct_value_profile(incomplete_sample, profile, profile_id="atomic-ht")

    with pytest.raises(ValueError, match="not an atomic population group"):
        build_fidelity_certificate(
            full,
            raw,
            ht,
            certificate_id="atomic-certificate",
            thresholds=_thresholds(),
        )


def test_certificate_requires_raw_and_ht_profiles_to_bind_the_same_pi():
    population = [_grouped_unit("one", "one"), _grouped_unit("two", "two")]
    raw_units = [_grouped_unit("one", "one", probability=0.5)]
    tampered_ht_units = [_grouped_unit("one", "one", probability=0.25)]
    profile = _profile()
    full = build_value_profile(population, profile, profile_id="pi-full")
    raw = build_raw_sample_profile(raw_units, profile, profile_id="pi-raw")
    ht = reconstruct_value_profile(tampered_ht_units, profile, profile_id="pi-ht")

    with pytest.raises(ValueError, match="different inclusion probabilities"):
        build_fidelity_certificate(
            full,
            raw,
            ht,
            certificate_id="pi-certificate",
            thresholds=_thresholds(),
        )


def test_ht_total_check_detects_uniform_pi_tampering_that_preserves_shape_and_ess():
    population = [_grouped_unit("one", "one"), _grouped_unit("two", "two")]
    tampered_sample = [
        _grouped_unit("one", "one", probability=0.5),
        _grouped_unit("two", "two", probability=0.5),
    ]
    profile = _profile()
    full = build_value_profile(population, profile, profile_id="total-full")
    raw = build_raw_sample_profile(tampered_sample, profile, profile_id="total-raw")
    ht = reconstruct_value_profile(tampered_sample, profile, profile_id="total-ht")
    certificate = build_fidelity_certificate(
        full,
        raw,
        ht,
        certificate_id="total-certificate",
        thresholds=_thresholds(),
    )

    assert certificate.weighted_comparison.joint_total_variation == pytest.approx(0)
    assert ht.effective_sample_size == pytest.approx(2)
    assert certificate.ht_total_relative_error == pytest.approx(1)
    assert not certificate.checks["ht_total_relative_error"]
    assert not certificate.distribution_fidelity_passed
