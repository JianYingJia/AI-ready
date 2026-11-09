import hashlib

from ai_ready.valuation.conditional_value import (
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeAggregationConfig,
    ProbeObservation,
    ScoredProbe,
    ScoreInterval,
)
from ai_ready.valuation.model_probe import FourChoiceProbe
from ai_ready.valuation.unit_mastery import (
    UnitCellMasteryObservation,
    build_unit_cell_mastery_profile,
)
from ai_ready.valuation.unit_prior_experiment import (
    PriorManipulationConfig,
    ProbeHoldoutConfig,
    ProbeUnitBinding,
    analyse_prior_manipulation,
    bind_probes_to_semantic_units,
    build_sparse_unit_observations,
    split_bindings_for_probe_holdout,
)
from ai_ready.valuation.value_sampling import ValueSamplingUnit


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _probe(evidence: str) -> FourChoiceProbe:
    block = f"Context. {evidence} More context."
    return FourChoiceProbe.model_validate(
        {
            "schema_version": "cll-independent-capability-probe-v1",
            "probe_id": "probe-1",
            "item_id": "probe-1",
            "variant_id": "base",
            "cell_id": "diagnosis",
            "question": "What threshold applies?",
            "options": ["One", "Five", "Ten", "Twenty"],
            "answer_index": 1,
            "answer_index_base": 0,
            "answer_text": "Five",
            "rationale": "The evidence states five.",
            "source_evidence": evidence,
            "source_evidence_sha256": _sha(evidence),
            "source_file": "data/public.md",
            "source_sha256": "a" * 64,
            "evidence_location": "data/public.md#block=0",
            "source_location": {
                "source_id": "public",
                "title": "Public",
                "canonical_url": "https://example.org/public",
                "version": "v1",
                "knowledge_cutoff": "2026-01-01",
                "processed_path": "data/public.md",
                "processed_sha256": "a" * 64,
                "blocks_path": "data/public.blocks.jsonl",
                "blocks_sha256": "b" * 64,
                "block_index_base": 0,
                "block_index": 0,
                "section": "Definition",
                "source_block_text_sha256": _sha(block),
            },
            "temporal_risk": "low",
            "locked_test": False,
            "test_items_included": False,
            "is_locked_test_item": False,
        }
    )


def _unit(identifier: str, state: str = "unknown") -> ValueSamplingUnit:
    probabilities = MasteryProbabilities.one_hot(state).model_dump(mode="json")
    return ValueSamplingUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="c" * 64,
        dependency_group_id=identifier,
        capability_ids=["diagnosis"],
        model_states={"diagnosis": state},
        model_state_probabilities={"diagnosis": probabilities},
        data_attributes={"knowledge_type": "rule"},
    )


def _parent() -> ModelMasteryProfile:
    return ModelMasteryProfile(
        profile_id="parent",
        benchmark_id="benchmark",
        model_id="model",
        model_revision="revision",
        checkpoint_sha256="d" * 64,
        probe_set_id="capability-probes",
        probe_set_sha256="e" * 64,
        scoring_protocol_sha256="f" * 64,
        observations=[
            ProbeObservation(
                cell_id="diagnosis",
                capability_id="diagnosis",
                operation_id="__absent__",
                difficulty_id="__absent__",
                point_score=0.5,
                score_interval=ScoreInterval(lower=0.4, upper=0.6),
                probe_count=10,
                probabilities=MasteryProbabilities.one_hot("uncertain"),
                stratum_state="uncertain",
                scoring_method="test",
            )
        ],
    )


def test_exact_evidence_binding_respects_population_capability():
    evidence = "At least 5 × 10^9/L clonal B cells"
    unit = _unit("u1")
    bindings = bind_probes_to_semantic_units(
        [_probe(evidence)],
        [
            {
                "semantic_unit_id": "u1",
                "batch_id": "B01",
                "content": f"Definition: {evidence} [ 3 ].",
                "content_sha256": _sha(f"Definition: {evidence} [ 3 ]."),
            }
        ],
        [unit],
    )
    assert [(row.probe_id, row.unit_id, row.cell_id) for row in bindings] == [("probe-1", "u1", "diagnosis")]


def test_sparse_observations_keep_unmeasured_pairs_unknown():
    units = [_unit("u1"), _unit("u2")]
    binding = {
        "probe_id": "probe-1",
        "item_id": "probe-1",
        "unit_id": "u1",
        "batch_id": "B01",
        "cell_id": "diagnosis",
        "evidence_sha256": "a" * 64,
        "semantic_content_sha256": "b" * 64,
        "match_method": "normalized_exact_evidence",
    }
    scores = [
        ScoredProbe(
            probe_id="probe-1:v1",
            item_id="probe-1",
            variant_id="v1",
            cell_id="diagnosis",
            score=1,
        ),
        ScoredProbe(
            probe_id="probe-1:v2",
            item_id="probe-1",
            variant_id="v2",
            cell_id="diagnosis",
            score=0,
        ),
    ]
    observations = build_sparse_unit_observations(
        units,
        [binding],
        scores,
        ProbeAggregationConfig(
            tau_low=0.35,
            tau_high=0.75,
            bootstrap_iterations=100,
            minimum_probe_count=1,
            scoring_method="screening",
        ),
    )
    by_unit = {row.unit_id: row for row in observations}
    assert by_unit["u1"].point_score == 0.5
    assert by_unit["u1"].stratum_state == "uncertain"
    assert by_unit["u1"].probe_count == 1
    assert by_unit["u2"].stratum_state == "unknown"
    assert by_unit["u2"].probe_count == 0


def test_manipulation_check_detects_rule_level_membership_change():
    parent = _parent()
    units = [_unit(f"u{index}") for index in range(4)]
    observations = []
    for index, unit in enumerate(units):
        state = "mastered" if index < 2 else "unmastered"
        point = 1.0 if state == "mastered" else 0.0
        observations.append(
            UnitCellMasteryObservation(
                unit_id=unit.unit_id,
                cell_id="diagnosis",
                point_score=point,
                score_interval=ScoreInterval(lower=point, upper=point),
                probe_count=1,
                probe_ids=[f"probe-{index}"],
                probabilities=MasteryProbabilities.one_hot(state),
                stratum_state=state,
            )
        )
    profile = build_unit_cell_mastery_profile(
        profile_id="unit-profile",
        probe_set_id="rule-probes",
        scoring_protocol_sha256="1" * 64,
        parent=parent,
        units=units,
        observations=observations,
    )
    result = analyse_prior_manipulation(
        units,
        parent,
        profile,
        PriorManipulationConfig(
            sampling_fractions=[0.25],
            seeds=[17],
            minimum_capabilities_with_variance=1,
            maximum_unchanged_jaccard=0.95,
            minimum_changed_trial_fraction=1,
        ),
    )
    assert result["status"] == "identifiable_for_screening"
    assert result["selection_changed_trial_fraction"] == 1
    assert result["trials"][0]["aware_realized_fraction"] == 0.5
    assert result["trials"][0]["measured_state_tv_advantage"] > 0


def _binding(unit_id: str, item_id: str, probe_id: str) -> ProbeUnitBinding:
    return ProbeUnitBinding(
        probe_id=probe_id,
        item_id=item_id,
        unit_id=unit_id,
        batch_id="B01",
        cell_id="diagnosis",
        evidence_sha256="a" * 64,
        semantic_content_sha256="b" * 64,
    )


def test_within_pair_holdout_keeps_design_and_audit_items_disjoint():
    bindings = [
        _binding("u1", "item-1", "probe-1"),
        _binding("u1", "item-2", "probe-2"),
        _binding("u2", "item-3", "probe-3"),
    ]
    design, audit, manifest = split_bindings_for_probe_holdout(
        bindings,
        ProbeHoldoutConfig(split_mode="within_pair_items", seed=17),
    )
    assert {row.item_id for row in design}.isdisjoint({row.item_id for row in audit})
    assert {row.unit_id for row in design} == {"u1"}
    assert {row.unit_id for row in audit} == {"u1"}
    assert manifest["eligible_pair_count"] == 1
    assert manifest["excluded_pair_count"] == 1


def test_disjoint_unit_holdout_never_leaks_one_pair_between_panels():
    bindings = [
        _binding("u1", "item-1", "probe-1"),
        _binding("u1", "item-2", "probe-2"),
        _binding("u2", "item-3", "probe-3"),
        _binding("u3", "item-4", "probe-4"),
        _binding("u4", "item-5", "probe-5"),
    ]
    design, audit, manifest = split_bindings_for_probe_holdout(
        bindings,
        ProbeHoldoutConfig(split_mode="disjoint_unit_cells", seed=17),
    )
    assert {row.unit_id for row in design}.isdisjoint({row.unit_id for row in audit})
    assert {row.unit_id for row in design} | {row.unit_id for row in audit} == {"u1", "u2", "u3", "u4"}
    assert manifest["design_pair_count"] == 2
    assert manifest["audit_pair_count"] == 2
