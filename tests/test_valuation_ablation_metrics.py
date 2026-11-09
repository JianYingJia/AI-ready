import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.valuation.ablation import ablation_protocol_digest, load_ablation_protocol
from ai_ready.valuation.ablation_metrics import (
    METRIC_REGISTRY,
    PairedPredictionAudit,
    RankingUtilityAudit,
    StaticSelectionAudit,
    collect_prediction_observations,
    collect_ranking_observations,
    collect_selection_observations,
    compute_leave_one_lineage_out_metrics,
    compute_prediction_metric,
    compute_prediction_metrics,
    compute_ranking_metrics,
    compute_selection_metric,
    compute_selection_metrics,
    protocol_metric_coverage,
    validate_metric_value,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility" / "stage_ablation_protocol.yaml"


def _static_payload() -> dict:
    return {
        "schema_version": "ablation-static-selection-v1",
        "batch_id": "B01",
        "lineage_group": "L1",
        "capability_ids": ["c1", "c2"],
        "quality_buckets": ["good", "bad"],
        "units": [
            {"unit_id": "u1", "estimated_tokens": 100, "quality_bucket": "good"},
            {"unit_id": "u2", "estimated_tokens": 100, "quality_bucket": "bad"},
            {"unit_id": "u3", "estimated_tokens": 100, "quality_bucket": "bad"},
        ],
        "obligations": [
            {
                "obligation_id": "o1",
                "kind": "topic",
                "unit_ids": ["u1"],
                "weight": 3.0,
                "capability_ids": ["c1"],
            },
            {
                "obligation_id": "o2",
                "kind": "condition",
                "unit_ids": ["u2"],
                "weight": 1.0,
                "capability_ids": ["c1"],
            },
            {
                "obligation_id": "o3",
                "kind": "exception",
                "unit_ids": ["u3"],
                "weight": 2.0,
                "capability_ids": ["c2"],
            },
        ],
        "blueprint": [
            {"capability_id": "c1", "weight": 0.75},
            {"capability_id": "c2", "weight": 0.25},
        ],
        "selected_unit_ids": ["u1", "u2"],
        "mining_gold_elements": [
            {
                "element_id": "e1",
                "kind": "condition",
                "annotator_id": "expert-1",
                "source_span_id": "span-1",
            },
            {
                "element_id": "e2",
                "kind": "exception",
                "annotator_id": "expert-1",
                "source_span_id": "span-2",
            },
            {
                "element_id": "e3",
                "kind": "fact",
                "annotator_id": "expert-1",
                "source_span_id": "span-3",
            },
        ],
        "mining_gold_dependency_groups": [
            {
                "dependency_group_id": "gold-group-1",
                "required_element_ids": ["e1", "e2"],
                "annotator_id": "expert-1",
            }
        ],
        "extracted_mining_groups": [
            {"extracted_group_id": "g1", "matched_gold_element_ids": ["e1", "e3"]},
            {"extracted_group_id": "g2", "matched_gold_element_ids": ["e2"]},
        ],
        "assertion_adjudications": [
            {
                "assertion_id": "a1",
                "extracted_group_id": "g1",
                "support_label": "supported",
                "annotator_id": "expert-2",
                "cited_source_span_ids": ["span-1"],
            },
            {
                "assertion_id": "a2",
                "extracted_group_id": "g2",
                "support_label": "unsupported",
                "annotator_id": "expert-2",
                "cited_source_span_ids": [],
            },
        ],
        "capability_mapping_annotations": [
            {
                "unit_id": "u1",
                "gold_capability_ids": ["c1"],
                "predicted_capability_ids": ["c1"],
                "annotator_id": "expert-3",
            },
            {
                "unit_id": "u2",
                "gold_capability_ids": ["c2"],
                "predicted_capability_ids": ["c1"],
                "annotator_id": "expert-3",
            },
            {
                "unit_id": "u3",
                "gold_capability_ids": ["c2"],
                "predicted_capability_ids": ["c2"],
                "annotator_id": "expert-3",
            },
        ],
        "facility_targets": [
            {
                "target_id": "t1",
                "weight": 2.0,
                "similarity_by_unit_id": {"u1": 0.9, "u2": 0.4, "u3": 0.1},
            },
            {
                "target_id": "t2",
                "weight": 1.0,
                "similarity_by_unit_id": {"u1": 0.2, "u2": 0.8, "u3": 0.9},
            },
        ],
        "generation_attempts": [
            {"attempt_id": "ga1", "unit_id": "u1", "accepted": True},
            {
                "attempt_id": "ga2",
                "unit_id": "u1",
                "accepted": False,
                "rejection_reason": "unsupported",
            },
            {"attempt_id": "ga3", "unit_id": "u2", "accepted": True},
            {
                "attempt_id": "ga4",
                "unit_id": "u3",
                "accepted": False,
                "rejection_reason": "invalid",
            },
        ],
        "prior_capability_gaps": [
            {"capability_id": "c1", "weight": 0.8},
            {"capability_id": "c2", "weight": 0.2},
        ],
    }


def _prediction_payload() -> dict:
    items = [
        {
            "item_id": "i1",
            "gold_label": "yes",
            "capability_id": "c1",
            "target_label": "target",
            "probe_form": "cross_form",
        },
        {
            "item_id": "i2",
            "gold_label": "yes",
            "capability_id": "c1",
            "target_label": "nontarget",
            "probe_form": "trained_form",
        },
        {
            "item_id": "i3",
            "gold_label": "yes",
            "capability_id": "c2",
            "target_label": "target",
            "probe_form": "cross_form",
        },
        {
            "item_id": "i4",
            "gold_label": "yes",
            "capability_id": "c2",
            "target_label": "nontarget",
            "probe_form": "cross_form",
        },
    ]
    return {
        "schema_version": "ablation-paired-predictions-v1",
        "batch_id": "B01",
        "lineage_group": "L1",
        "seed": 17,
        "catalog": {"capability_ids": ["c1", "c2"], "items": items},
        "base": {
            "run_id": "base-17",
            "predictions": [
                {"item_id": "i1", "predicted_label": "no"},
                {"item_id": "i2", "predicted_label": "yes"},
                {"item_id": "i3", "predicted_label": "no"},
                {"item_id": "i4", "predicted_label": "yes"},
            ],
        },
        "tuned": {
            "run_id": "tuned-17",
            "predictions": [
                {"item_id": "i1", "predicted_label": "yes"},
                {"item_id": "i2", "predicted_label": "yes"},
                {"item_id": "i3", "predicted_label": "yes"},
                {"item_id": "i4", "predicted_label": "no"},
            ],
        },
    }


def _ranking_payload() -> dict:
    return {
        "schema_version": "ablation-ranking-utilities-v1",
        "selection_k": 1,
        "low_value_threshold": 0.25,
        "indifference_margin": 0.0,
        "rows": [
            {"batch_id": "B01", "lineage_group": "L1", "proxy_utility": 0.9, "fom_utility": 0.8},
            {"batch_id": "B02", "lineage_group": "L2", "proxy_utility": 0.8, "fom_utility": 0.7},
            {"batch_id": "B03", "lineage_group": "L3", "proxy_utility": 0.2, "fom_utility": 0.2},
            {"batch_id": "B04", "lineage_group": "L4", "proxy_utility": 0.1, "fom_utility": 0.1},
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_audit_manifest(path: Path, artifact_path: Path, study_id: str, cell_id: str) -> None:
    protocol = load_ablation_protocol(PROTOCOL_PATH)
    _write_json(
        path,
        {
            "schema_version": "stage-ablation-audit-manifest-v1",
            "execution_id": "unit-test-execution",
            "protocol_sha256": ablation_protocol_digest(protocol),
            "study_id": study_id,
            "cell_id": cell_id,
            "artifact_uri": artifact_path.name,
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        },
    )


def test_repository_protocol_metrics_all_have_raw_artifact_calculators():
    protocol = load_ablation_protocol(PROTOCOL_PATH)
    coverage = protocol_metric_coverage(protocol)

    assert len(coverage["required"]) == 19
    assert coverage["required"] == coverage["implemented"]
    assert set(coverage["required"]) <= set(METRIC_REGISTRY)

    protocol.studies[0].cell_gates[0].metric = "invented_final_scalar"
    with pytest.raises(ValueError, match="without raw-artifact calculators"):
        protocol_metric_coverage(protocol)


def test_static_metrics_are_recomputed_from_raw_annotations_and_membership():
    artifact = StaticSelectionAudit.model_validate(_static_payload())
    values = compute_selection_metrics(artifact)

    assert values.condition_exception_recall == pytest.approx(1.0)
    assert values.unsupported_medical_assertion_rate == pytest.approx(0.5)
    assert values.dependency_group_recall == pytest.approx(0.0)
    assert values.capability_mapping_macro_f1 == pytest.approx(2 / 3)
    assert values.weighted_obligation_coverage == pytest.approx(4 / 6)
    assert values.condition_exception_coverage == pytest.approx(0.5)
    assert values.quality_distribution_similarity == pytest.approx(5 / 6)
    assert values.blueprint_alignment == pytest.approx(0.75)
    assert values.semantic_facility == pytest.approx(2.6 / 3)
    assert values.acceptance_rate_absolute_error == pytest.approx(1 / 6)
    assert values.prior_gap_selection_alignment == pytest.approx(1.0)


def test_static_artifact_cannot_smuggle_a_precomputed_endpoint():
    payload = _static_payload()
    payload["weighted_obligation_coverage"] = 1.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StaticSelectionAudit.model_validate(payload)


def test_static_metric_rejects_an_unidentifiable_quality_distribution():
    payload = _static_payload()
    payload["quality_buckets"] = ["same"]
    for unit in payload["units"]:
        unit["quality_bucket"] = "same"
    artifact = StaticSelectionAudit.model_validate(payload)

    with pytest.raises(ValueError, match="not identifiable"):
        compute_selection_metric(artifact, "quality_distribution_similarity")


def test_prediction_metrics_use_fully_paired_item_level_outputs():
    artifact = PairedPredictionAudit.model_validate(_prediction_payload())
    values = compute_prediction_metrics(artifact)

    assert values.training_utility == pytest.approx(0.25)
    assert values.cross_form_probe_gain == pytest.approx(1 / 3)
    assert values.target_capability_probe_gain == pytest.approx(1.0)
    assert values.target_minus_nontarget_probe_gain == pytest.approx(1.5)
    assert compute_prediction_metric(artifact, "raw_capability_gain", capability_id="c1") == pytest.approx(0.5)
    assert compute_prediction_metric(artifact, "raw_capability_gain", capability_id="c2") == pytest.approx(0.0)


def test_prediction_artifact_requires_exactly_the_locked_item_ids():
    payload = _prediction_payload()
    payload["tuned"]["predictions"].pop()

    with pytest.raises(ValidationError, match="do not match the item catalog"):
        PairedPredictionAudit.model_validate(payload)


def test_ranking_metrics_cover_order_decision_regret_and_false_promotion():
    payload = _ranking_payload()
    payload["rows"] = [
        {"batch_id": "B01", "lineage_group": "L1", "proxy_utility": 0.9, "fom_utility": 0.1},
        {"batch_id": "B02", "lineage_group": "L2", "proxy_utility": 0.7, "fom_utility": 0.3},
        {"batch_id": "B03", "lineage_group": "L3", "proxy_utility": 0.3, "fom_utility": 0.7},
        {"batch_id": "B04", "lineage_group": "L4", "proxy_utility": 0.1, "fom_utility": 0.9},
    ]
    values = compute_ranking_metrics(RankingUtilityAudit.model_validate(payload))

    assert values.spearman == pytest.approx(-1.0)
    assert values.pairwise_accuracy == pytest.approx(0.0)
    assert values.top1_regret == pytest.approx(0.8)
    assert values.low_value_false_promotion_rate == pytest.approx(1.0)


def test_leave_one_lineage_out_recomputes_each_fold():
    artifact = RankingUtilityAudit.model_validate(_ranking_payload())
    folds = compute_leave_one_lineage_out_metrics(artifact)

    assert [row.excluded_lineage_group for row in folds] == ["L1", "L2", "L3", "L4"]
    assert all(row.metrics.spearman == pytest.approx(1.0) for row in folds)
    assert all(row.metrics.pairwise_accuracy == pytest.approx(1.0) for row in folds)
    assert all(row.metrics.top1_regret == pytest.approx(0.0) for row in folds)


def test_collector_binds_real_artifact_and_execution_manifest_hashes(tmp_path):
    artifact_path = tmp_path / "static_audit.json"
    manifest_path = tmp_path / "audit_manifest.json"
    _write_json(artifact_path, _static_payload())
    _write_audit_manifest(manifest_path, artifact_path, "A3_subset_selection", "coverage_quality_25")

    observations = collect_selection_observations(
        PROTOCOL_PATH,
        artifact_path,
        manifest_path,
        study_id="A3_subset_selection",
        cell_id="coverage_quality_25",
    )

    assert {row.metric for row in observations} == {
        "weighted_obligation_coverage",
        "condition_exception_coverage",
        "semantic_facility",
    }
    assert {row.artifact_uri for row in observations} == {str(artifact_path.resolve())}
    assert {row.artifact_sha256 for row in observations} == {hashlib.sha256(artifact_path.read_bytes()).hexdigest()}
    assert {row.execution_manifest_sha256 for row in observations} == {
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    }


def test_prediction_collector_emits_capability_rows_only_when_protocol_pairs_them(tmp_path):
    artifact_path = tmp_path / "predictions.json"
    manifest_path = tmp_path / "audit_manifest.json"
    _write_json(artifact_path, _prediction_payload())
    _write_audit_manifest(manifest_path, artifact_path, "A5_supervision_organization", "mixed_grounded")

    observations = collect_prediction_observations(
        PROTOCOL_PATH,
        artifact_path,
        manifest_path,
        study_id="A5_supervision_organization",
        cell_id="mixed_grounded",
    )
    assert len(observations) == 2
    assert {row.capability_id for row in observations} == {"c1", "c2"}
    assert {row.metric for row in observations} == {"cross_form_probe_gain"}

    target_manifest = tmp_path / "target_manifest.json"
    _write_audit_manifest(target_manifest, artifact_path, "A2_blueprint_conditioning", "task_blueprint")
    target = collect_prediction_observations(
        PROTOCOL_PATH,
        artifact_path,
        target_manifest,
        study_id="A2_blueprint_conditioning",
        cell_id="task_blueprint",
        metrics=["target_capability_probe_gain"],
    )
    assert len(target) == 1
    assert target[0].capability_id is None
    assert target[0].value == pytest.approx(1.0)


def test_ranking_collector_emits_all_leave_one_lineage_out_protocol_metrics(tmp_path):
    artifact_path = tmp_path / "ranking.json"
    manifest_path = tmp_path / "audit_manifest.json"
    _write_json(artifact_path, _ranking_payload())
    _write_audit_manifest(manifest_path, artifact_path, "A9_operational_fom_prediction", "coverage_proxy")

    observations = collect_ranking_observations(
        PROTOCOL_PATH,
        artifact_path,
        manifest_path,
        study_id="A9_operational_fom_prediction",
        cell_id="coverage_proxy",
    )

    assert len(observations) == 16
    assert len({row.lineage_group for row in observations}) == 4
    assert {row.metric for row in observations} == {
        "leave_one_lineage_out_spearman_to_fom",
        "leave_one_lineage_out_pairwise_accuracy_to_fom",
        "leave_one_lineage_out_top1_regret_to_fom",
        "leave_one_lineage_out_low_value_promotion_rate_to_fom",
    }


def test_manifest_must_bind_the_exact_artifact(tmp_path):
    artifact_path = tmp_path / "static_audit.json"
    manifest_path = tmp_path / "audit_manifest.json"
    _write_json(artifact_path, _static_payload())
    _write_audit_manifest(manifest_path, artifact_path, "A3_subset_selection", "coverage_quality_25")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "0" * 64
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="artifact_sha256 does not match"):
        collect_selection_observations(
            PROTOCOL_PATH,
            artifact_path,
            manifest_path,
            study_id="A3_subset_selection",
            cell_id="coverage_quality_25",
            metrics=["weighted_obligation_coverage"],
        )


def test_metric_registry_rejects_unknown_and_out_of_range_values():
    with pytest.raises(ValueError, match="Unknown or uncomputed"):
        validate_metric_value("caller_supplied_score", 0.5)
    with pytest.raises(ValueError, match="outside its registered range"):
        validate_metric_value("weighted_obligation_coverage", 1.01)
