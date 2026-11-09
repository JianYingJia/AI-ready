import pytest

from ai_ready.valuation.selection import CoverageObligation, ProxySelectionConfig, ProxyUnit, select_proxy


def _unit(
    identifier,
    group,
    vector,
    quality="clean",
    tokens=10,
    valid=True,
    relevance=0.5,
    sources=None,
):
    return ProxyUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="a" * 64,
        group_id=group,
        source_ids=sources or [f"source-{group}"],
        estimated_tokens=tokens,
        embedding=vector,
        quality_bucket=quality,
        benchmark_relevance=relevance,
        medically_valid=valid,
    )


def test_dependency_group_is_indivisible_and_budget_is_respected():
    units = [
        _unit("rule", "g-rule", [1.0, 0.0], tokens=8),
        _unit("exception", "g-rule", [1.0, 0.0], tokens=7),
        _unit("other", "g-other", [0.0, 1.0], tokens=10),
    ]
    obligations = [
        CoverageObligation(obligation_id="rule-with-exception", kind="exception", unit_ids=["rule", "exception"])
    ]
    result = select_proxy(units, obligations, ProxySelectionConfig(token_budget=15, obligation_weight=2.0))
    assert result["used_tokens"] <= 15
    assert set(result["selected_unit_ids"]) == {"rule", "exception"}


def test_quality_fidelity_does_not_select_only_clean_head():
    units = [
        _unit("clean-1", "clean-1", [1.0, 0.0], quality="clean"),
        _unit("clean-2", "clean-2", [0.9, 0.1], quality="clean"),
        _unit("usable-noisy", "noisy", [0.0, 1.0], quality="usable_noisy"),
    ]
    config = ProxySelectionConfig(
        token_budget=20,
        facility_weight=0.0,
        obligation_weight=0.0,
        quality_fidelity_weight=1.0,
        source_coverage_weight=0.0,
    )
    result = select_proxy(units, [], config)
    assert "usable-noisy" in result["selected_unit_ids"]
    assert result["components"]["quality_fidelity"] > 0.8


def test_medically_invalid_units_are_excluded_and_counted_as_yield():
    units = [
        _unit("valid", "valid", [1.0, 0.0]),
        _unit("invalid", "invalid", [0.0, 1.0], valid=False),
    ]
    result = select_proxy(units, [], ProxySelectionConfig(token_budget=20))
    assert result["selected_unit_ids"] == ["valid"]
    assert result["invalid_units_excluded"] == 1
    assert result["effective_unit_yield"] == 0.5


def test_invalid_dependency_member_excludes_the_complete_group():
    units = [
        _unit("rule", "rule-group", [1.0, 0.0]),
        _unit("invalid-exception", "rule-group", [1.0, 0.0], valid=False),
        _unit("independent", "independent", [0.0, 1.0]),
    ]
    result = select_proxy(units, [], ProxySelectionConfig(token_budget=30))

    assert result["selected_unit_ids"] == ["independent"]
    assert result["excluded_group_ids"] == ["rule-group"]
    assert result["invalid_units_excluded"] == 1
    assert result["dependency_units_excluded"] == 1
    assert result["effective_unit_yield"] == 1 / 3


def test_selection_binds_frozen_batch_and_complete_config():
    units = [
        _unit("one", "one", [1.0, 0.0]),
        _unit("two", "two", [0.0, 1.0]),
    ]
    config = ProxySelectionConfig(token_budget=10, random_state=7)
    result = select_proxy(units, [], config)
    assert result["batch_content_sha256"] == "a" * 64
    assert result["selection_config"] == config.model_dump(mode="json")
    assert len(result["selection_config_sha256"]) == 64

    units[1].batch_content_sha256 = "b" * 64
    try:
        select_proxy(units, [], config)
    except ValueError as error:
        assert "same frozen batch content" in str(error)
    else:
        raise AssertionError("mixed batch content hashes must be rejected")


def test_diversity_first_pick_is_invariant_to_relevance():
    def units_with_relevance(center_relevance, edge_relevance):
        return [
            _unit("left", "left", [1.0, 0.0], relevance=edge_relevance),
            _unit("center", "center", [1.0, 1.0], relevance=center_relevance),
            _unit("right", "right", [0.0, 1.0], relevance=edge_relevance),
        ]

    config = ProxySelectionConfig(method="diversity", token_budget=10)
    low_center = select_proxy(units_with_relevance(0.0, 1.0), [], config)
    high_center = select_proxy(units_with_relevance(1.0, 0.0), [], config)

    assert low_center["selected_group_ids"] == ["center"]
    assert high_center["selected_group_ids"] == low_center["selected_group_ids"]


def test_stratified_random_is_deterministic_and_covers_available_strata():
    units = []
    for source in ("source-a", "source-b"):
        for quality in ("clean", "usable_noisy"):
            for replica in (1, 2):
                identifier = f"{source}-{quality}-{replica}"
                units.append(
                    _unit(
                        identifier,
                        identifier,
                        [float(replica == 1), float(replica == 2)],
                        quality=quality,
                        sources=[source],
                    )
                )

    config = ProxySelectionConfig(method="stratified_random", token_budget=40, random_state=19)
    first = select_proxy(units, [], config)
    second = select_proxy(units, [], config)

    assert first["selected_group_ids"] == second["selected_group_ids"]
    assert first["trace"] == second["trace"]
    available = {(item["source_id"], item["quality_bucket"]) for item in first["available_strata"]}
    selected = {(item["source_id"], item["quality_bucket"]) for item in first["selected_strata"]}
    assert (
        selected
        == available
        == {
            ("source-a", "clean"),
            ("source-a", "usable_noisy"),
            ("source-b", "clean"),
            ("source-b", "usable_noisy"),
        }
    )
    assert all("selection_stratum" in item for item in first["trace"])


def test_coverage_only_forces_quality_weight_to_zero_and_ignores_quality_labels():
    config = ProxySelectionConfig(
        method="coverage_only",
        token_budget=10,
        facility_weight=1.0,
        obligation_weight=0.0,
        quality_fidelity_weight=999.0,
        source_coverage_weight=0.0,
    )
    assert config.quality_fidelity_weight == 0.0

    original = [
        _unit("left", "left", [1.0, 0.0], quality="clean"),
        _unit("center", "center", [1.0, 1.0], quality="usable_noisy"),
        _unit("right", "right", [0.0, 1.0], quality="clean"),
    ]
    relabeled = [unit.model_copy(update={"quality_bucket": "other"}) for unit in original]

    first = select_proxy(original, [], config)
    second = select_proxy(relabeled, [], config)
    assert first["selection_config"]["quality_fidelity_weight"] == 0.0
    assert first["selected_group_ids"] == second["selected_group_ids"]


def test_coverage_only_rejects_an_objective_with_no_enabled_component():
    with pytest.raises(ValueError, match="coverage_only requires at least one positive enabled objective weight"):
        ProxySelectionConfig(
            method="coverage_only",
            token_budget=10,
            facility_weight=0.0,
            obligation_weight=0.0,
            quality_fidelity_weight=10.0,
            source_coverage_weight=0.0,
        )


@pytest.mark.parametrize(
    "method",
    ["coverage_quality", "coverage_only", "random", "stratified_random", "relevance", "diversity"],
)
def test_every_selection_method_respects_group_token_budget(method):
    units = [
        _unit("paired-rule", "paired", [1.0, 0.0], tokens=7, sources=["source-a"]),
        _unit("paired-exception", "paired", [1.0, 0.0], tokens=6, sources=["source-a"]),
        _unit("small", "small", [0.0, 1.0], tokens=9, sources=["source-b"]),
        _unit("too-large", "too-large", [1.0, 1.0], tokens=20, sources=["source-c"]),
    ]
    result = select_proxy(units, [], ProxySelectionConfig(method=method, token_budget=15, random_state=3))

    assert result["used_tokens"] <= 15
    paired_ids = {"paired-rule", "paired-exception"} & set(result["selected_unit_ids"])
    assert paired_ids in (set(), {"paired-rule", "paired-exception"})
