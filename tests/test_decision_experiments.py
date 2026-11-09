from pathlib import Path

import pytest

from ai_ready.valuation.contracts import load_batch_suite, suite_digest
from ai_ready.valuation.decision_experiments import (
    CostAnalysisConfig,
    CostLedgerEvent,
    CrossTaskBatchValue,
    CrossTaskConfig,
    NaturalBatchAuditConfig,
    ReferenceBatchValue,
    StaticBaselineConfig,
    StaticBatchFeatures,
    TrainableProxyBatchValue,
    analyze_cost_ledger,
    analyze_cross_task_values,
    analyze_static_baselines,
    audit_natural_batches,
    evaluate_rankings,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "valuation_cll_public_feasibility" / "batch_suite.frozen.yaml"


def test_decision_ranking_metrics_recover_order_and_regret():
    report = evaluate_rankings(
        {"A": 3.0, "B": 2.0, "C": 1.0},
        {"A": 0.3, "B": 0.2, "C": 0.1},
        top_ks=[1, 2],
        indifference_margin=0.0,
    )
    assert report["spearman"] == pytest.approx(1.0)
    assert report["pairwise_order_accuracy"] == pytest.approx(1.0)
    assert report["top_k"]["2"]["recall"] == pytest.approx(1.0)
    assert report["regret"]["1"] == pytest.approx(0.0)
    assert report["regret"]["2"] is None


def test_static_baselines_are_evaluated_without_reference_fitting():
    suite = load_batch_suite(SUITE_PATH)
    suite_sha = suite_digest(suite)
    features = []
    references = []
    proxy_values = []
    for index, batch in enumerate(suite.batches, start=1):
        features.append(
            StaticBatchFeatures(
                suite_sha256=suite_sha,
                batch_id=batch.batch_id,
                batch_content_sha256=batch.content_sha256,
                lineage_group=batch.lineage_group,
                document_count=index,
                token_count=index * 100,
                unique_token_count=index * 80,
                semantic_unit_count=index * 2,
                mean_quality=index / 12,
                information_density=index / 12,
                benchmark_relevance=index / 12,
                redundancy_rate=(12 - index) / 12,
            )
        )
        references.append(
            ReferenceBatchValue(
                task_id="task-a",
                suite_sha256=suite_sha,
                benchmark_sha256="a" * 64,
                base_model_id="base-model",
                training_protocol_sha256="b" * 64,
                seeds=[17, 29, 43],
                batch_id=batch.batch_id,
                batch_content_sha256=batch.content_sha256,
                lineage_group=batch.lineage_group,
                utility=float(index),
            )
        )
        proxy_values.append(
            TrainableProxyBatchValue(
                task_id="task-a",
                suite_sha256=suite_sha,
                benchmark_sha256="a" * 64,
                base_model_id="base-model",
                training_protocol_sha256="b" * 64,
                seeds=[17, 29, 43],
                proxy_method="coverage_quality",
                budget_id="proxy-100k",
                batch_id=batch.batch_id,
                batch_content_sha256=batch.content_sha256,
                lineage_group=batch.lineage_group,
                utility=float(index),
            )
        )
    config = StaticBaselineConfig(
        enabled_baselines=["token_count", "intrinsic_static", "task_aware_static"],
        named_composites={
            "intrinsic_static": {
                "unique_token_count": 1.0,
                "mean_quality": 1.0,
                "information_density": 1.0,
                "low_redundancy": 1.0,
            },
            "task_aware_static": {
                "unique_token_count": 1.0,
                "mean_quality": 1.0,
                "information_density": 1.0,
                "benchmark_relevance": 1.0,
                "low_redundancy": 1.0,
            },
        },
        top_ks=[1, 3],
        hard_pair_baseline="task_aware_static",
        hard_pair_count=4,
    )
    report = analyze_static_baselines(suite, features, references, config, proxy_values=proxy_values)
    assert report["fitted_to_reference"] is False
    assert report["baselines"]["token_count"]["global"]["spearman"] == pytest.approx(1.0)
    assert report["baselines"]["intrinsic_static"]["global"]["top_k"]["1"]["recall"] == 1.0
    assert report["baselines"]["task_aware_static"]["global"]["top_k"]["1"]["recall"] == 1.0
    assert "benchmark_relevance" not in report["named_composite_definitions"]["intrinsic_static"]
    assert "benchmark_relevance" in report["named_composite_definitions"]["task_aware_static"]
    assert report["trainable_proxy_vs_static"]["task_aware_static"]["spearman"] == pytest.approx(0.0)
    assert report["hard_pair_analysis"]["selected_pair_count"] == 4
    assert report["baselines"]["token_count"]["leave_one_lineage_out"]["folds"]


def test_cross_task_analysis_detects_task_specific_reversal():
    rows = []
    first = {"A": 3.0, "B": 2.0, "C": 1.0}
    second = {"A": 1.0, "B": 2.0, "C": 3.0}
    for task_id, values, benchmark_hash in (
        ("task-a", first, "a" * 64),
        ("task-b", second, "b" * 64),
    ):
        for batch_id, utility in values.items():
            for arm in ("proxy", "reference"):
                rows.append(
                    CrossTaskBatchValue(
                        task_id=task_id,
                        suite_sha256="0" * 64,
                        benchmark_sha256=benchmark_hash,
                        base_model_id="base-model",
                        training_protocol_sha256="c" * 64,
                        seeds=[17, 29, 43],
                        batch_id=batch_id,
                        batch_content_sha256={"A": "d", "B": "e", "C": "f"}[batch_id] * 64,
                        lineage_group=f"lineage-{batch_id}",
                        arm=arm,
                        utility=utility,
                    )
                )
    report = analyze_cross_task_values(rows, CrossTaskConfig(top_ks=[1]))
    assert report["within_task_proxy_fidelity"]["task-a"]["spearman"] == pytest.approx(1.0)
    comparison = report["between_task_ranking_change"][0]
    assert comparison["reference"]["spearman"] == pytest.approx(-1.0)
    assert comparison["reference"]["ordering_inversions"] == 3
    assert comparison["reference_task_conditioning_observed"] is True


def test_cost_ledger_charges_failed_and_shared_events():
    suite = load_batch_suite(SUITE_PATH)
    events = []
    for batch in suite.batches:
        for branch, gpu_seconds in (("proxy", 10.0), ("reference", 100.0)):
            for stage in ("training", "evaluation"):
                events.append(
                    CostLedgerEvent(
                        event_id=f"{batch.batch_id}-{branch}-{stage}",
                        task_id="task-a",
                        batch_id=batch.batch_id,
                        batch_content_sha256=batch.content_sha256,
                        branch=branch,
                        stage=stage,
                        status="success",
                        gpu_seconds=gpu_seconds if stage == "training" else 0.0,
                    )
                )
    events.append(
        CostLedgerEvent(
            event_id="shared-failed-proxy-attempt",
            task_id="task-a",
            branch="proxy",
            stage="orchestration",
            status="failed",
            wall_seconds=12.0,
        )
    )
    config = CostAnalysisConfig(
        proxy_required_stages=["training", "evaluation"],
        reference_required_stages=["training", "evaluation"],
    )
    report = analyze_cost_ledger(suite, events, config)
    assert report["all_statuses_charged"] is True
    assert report["status_totals"]["proxy"]["failed"]["wall_seconds"] == pytest.approx(12.0)
    assert report["branch_totals"]["proxy"]["gpu_seconds"] == pytest.approx(120.0)
    assert report["proxy_to_reference_ratio"]["gpu_seconds"] == pytest.approx(0.1)


def test_natural_batch_audit_rejects_dependent_four_batch_check():
    suite = load_batch_suite(SUITE_PATH)
    report = audit_natural_batches(suite, NaturalBatchAuditConfig())
    assert report["valid"] is False
    assert report["natural_batch_count"] == 4
    assert report["independent_lineage_count"] == 3
    assert report["overlapping_sources"]["nci_cll_professional"] == ["B09", "B10"]
    assert any("paired transformations" in error for error in report["errors"])
