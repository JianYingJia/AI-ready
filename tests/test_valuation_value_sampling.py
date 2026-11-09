import pytest

from ai_ready.valuation.value_sampling import (
    RareStratumQuota,
    ValueSamplingConfig,
    ValueSamplingUnit,
    sample_value_distribution,
)


def _unit(
    identifier,
    state,
    *,
    group=None,
    capability_ids=None,
    quality="clean",
    knowledge_type="rule",
    eligible=True,
):
    capabilities = capability_ids or ["diagnosis"]
    model_states = state if isinstance(state, dict) else {capability_id: state for capability_id in capabilities}
    return ValueSamplingUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="a" * 64,
        dependency_group_id=group or identifier,
        capability_ids=capabilities,
        model_states=model_states,
        data_attributes={"quality": quality, "knowledge_type": knowledge_type},
        eligible=eligible,
    )


def test_sampler_preserves_all_four_model_states_and_is_deterministic():
    units = []
    for state in ("mastered", "uncertain", "unmastered", "unknown"):
        for index in range(4):
            units.append(_unit(f"{state}-{index}", state))
    config = ValueSamplingConfig(
        sampling_fraction=0.5,
        seed=17,
        stratum_attributes=["quality", "knowledge_type"],
    )

    first = sample_value_distribution(units, config)
    second = sample_value_distribution(units, config)

    assert first == second
    assert first["realized_group_count"] == 8
    assert {state for record in first["selected_unit_records"] for state in record["model_states"].values()} == {
        "mastered",
        "uncertain",
        "unmastered",
        "unknown",
    }
    assert {record["inclusion_probability"] for record in first["selected_unit_records"]} == {0.5}
    assert {record["design_weight"] for record in first["selected_unit_records"]} == {2.0}
    assert first["weighted_total_variation"] == pytest.approx(0.0)


def test_dependency_group_is_atomic_even_when_members_have_different_cells():
    units = [
        _unit("rule", "unmastered", group="clinical-rule", knowledge_type="rule"),
        _unit("exception", "uncertain", group="clinical-rule", knowledge_type="exception"),
        _unit("other-rule", "mastered", group="other", knowledge_type="rule"),
        _unit("other-condition", "mastered", group="other", knowledge_type="condition"),
    ]
    result = sample_value_distribution(
        units,
        ValueSamplingConfig(target_group_count=1, seed=3, stratum_attributes=["knowledge_type"]),
    )

    # Two distinct group compositions require positive support, so the sampler
    # transparently expands the requested one-group sample to both groups.
    assert result["realized_group_count"] == 2
    assert result["size_expansion_audit"] == [{"reason": "joint_stratum_support_or_rare_quota"}]
    selected_ids = set(result["selected_unit_ids"])
    assert {"rule", "exception"} <= selected_ids
    assert {"other-rule", "other-condition"} <= selected_ids


def test_rare_quota_oversamples_but_design_weights_recover_population_profile():
    units = [
        *[_unit(f"known-{index}", "mastered") for index in range(8)],
        *[
            _unit(
                f"rare-{index}",
                "unmastered",
                knowledge_type="exception",
                capability_ids=["treatment", "safety"],
            )
            for index in range(2)
        ],
    ]
    config = ValueSamplingConfig(
        target_group_count=5,
        seed=9,
        stratum_attributes=["knowledge_type"],
        rare_stratum_quotas=[
            RareStratumQuota(
                quota_id="unmastered-exceptions",
                minimum_groups=2,
                capability_id="safety",
                model_state="unmastered",
                data_attributes={"knowledge_type": "exception"},
            )
        ],
    )
    result = sample_value_distribution(units, config)

    rare_records = [
        record for record in result["selected_unit_records"] if "unmastered" in record["model_states"].values()
    ]
    known_records = [
        record for record in result["selected_unit_records"] if set(record["model_states"].values()) == {"mastered"}
    ]
    assert len(rare_records) == 2
    assert len(known_records) == 3
    assert {record["inclusion_probability"] for record in rare_records} == {1.0}
    assert {record["inclusion_probability"] for record in known_records} == {3 / 8}
    assert result["rare_quota_audit"][0]["satisfied"] is True
    assert result["model_state_totals"]["mastered"]["sample_weighted_weight"] == pytest.approx(8.0)
    assert result["model_state_totals"]["unmastered"]["sample_weighted_weight"] == pytest.approx(2.0)
    assert result["weighted_total_variation"] == pytest.approx(0.0)


def test_ineligible_member_excludes_complete_dependency_group_and_audits_rows():
    units = [
        _unit("valid-rule", "unmastered", group="broken"),
        _unit("invalid-exception", "unmastered", group="broken", eligible=False),
        _unit("independent", "mastered"),
    ]
    result = sample_value_distribution(units, ValueSamplingConfig(sampling_fraction=1.0))

    assert result["selected_unit_ids"] == ["independent"]
    assert result["ineligible_unit_ids"] == ["invalid-exception"]
    assert result["excluded_dependency_group_ids"] == ["broken"]
    assert result["dependency_units_excluded"] == 1


def test_impossible_quota_fails_instead_of_silently_substituting_another_stratum():
    units = [_unit("known", "mastered"), _unit("unknown", "unknown")]
    config = ValueSamplingConfig(
        sampling_fraction=1.0,
        rare_stratum_quotas=[RareStratumQuota(quota_id="missing", minimum_groups=1, model_state="unmastered")],
    )

    with pytest.raises(ValueError, match="Rare-stratum quotas exceed the eligible population"):
        sample_value_distribution(units, config)


def test_target_too_small_can_be_strictly_rejected():
    units = [_unit("known", "mastered"), _unit("unknown", "unknown")]
    config = ValueSamplingConfig(target_group_count=1, allow_target_expansion=False)

    with pytest.raises(ValueError, match="at least 2 are required"):
        sample_value_distribution(units, config)


def test_manifest_keeps_multi_capability_unit_once():
    units = [
        _unit(
            "multi",
            {"safety": "unmastered", "treatment": "mastered"},
            capability_ids=["safety", "treatment"],
        )
    ]
    result = sample_value_distribution(units, ValueSamplingConfig(sampling_fraction=1.0))

    assert result["selected_unit_ids"] == ["multi"]
    assert result["selected_unit_records"][0]["capability_ids"] == ["safety", "treatment"]
    assert result["selected_unit_records"][0]["model_states"] == {
        "safety": "unmastered",
        "treatment": "mastered",
    }
    assert len(result["selected_unit_records"]) == 1
    assert {row["capability_id"]: row["population_weight"] for row in result["joint_profile"]} == {
        "safety": 0.5,
        "treatment": 0.5,
    }


def test_model_state_keys_must_exactly_cover_capabilities():
    with pytest.raises(ValueError, match="must exactly match capability_ids"):
        _unit(
            "mismatch",
            {"diagnosis": "mastered"},
            capability_ids=["diagnosis", "treatment"],
        )


def test_missing_configured_data_attribute_is_rejected():
    units = [_unit("one", "mastered")]
    with pytest.raises(ValueError, match="missing configured stratum attributes"):
        sample_value_distribution(
            units,
            ValueSamplingConfig(sampling_fraction=1.0, stratum_attributes=["source"]),
        )


def test_quota_cannot_match_an_attribute_missing_from_sampling_strata():
    with pytest.raises(ValueError, match="not part of the sampling strata"):
        ValueSamplingConfig(
            sampling_fraction=0.5,
            rare_stratum_quotas=[
                RareStratumQuota(
                    quota_id="exception",
                    minimum_groups=1,
                    data_attributes={"knowledge_type": "exception"},
                )
            ],
        )
