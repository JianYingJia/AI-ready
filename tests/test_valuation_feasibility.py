import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.valuation.contracts import contract_digest, load_batch_suite, load_production_contract, suite_digest
from ai_ready.valuation.feasibility import assess_feasibility
from ai_ready.valuation.schema import FeasibilityProtocol, TrialScore

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "valuation_cll"


def _suite_and_contract():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    for batch in suite.batches:
        batch.content_sha256 = hashlib.sha256(f"feasibility:{batch.batch_id}".encode()).hexdigest()
    return suite, load_production_contract(EXAMPLE_DIR / "production_contract.yaml")


def _protocol(**criteria_overrides):
    target_decision = {
        "min_spearman": 0.8,
        "min_kendall_tau_b": 0.7,
        "min_pairwise_order_accuracy": 0.8,
        "max_top1_regret": 0.01,
        "min_spearman_advantage_over_each_baseline": 0.1,
        "max_top1_regret_excess_over_each_baseline": 0.0,
    }
    target_decision.update(criteria_overrides.pop("target_decision", {}))
    reference_identifiability = {
        "min_utility_range": 0.05,
        "min_between_batch_sd": 0.02,
        "max_within_to_between_sd_ratio": 0.2,
        "min_pairwise_seed_rank_spearman_median": 0.9,
        "positive_margin": 0.0,
        "min_clearly_positive_batch_count": 2,
        "ci_separation_margin": 0.0,
        "min_distinguishable_ci_pair_fraction": 0.1,
    }
    reference_identifiability.update(criteria_overrides.pop("reference_identifiability", {}))
    payload = {
        "schema_version": "pilot-feasibility-protocol-v2",
        "protocol_id": "unit-test-pilot",
        "inference_scope": "descriptive_pilot_no_confirmatory_inference",
        "expected_benchmark_item_count": 46,
        "target": {
            "arm": "proxy",
            "label": "coverage_quality",
            "method": "coverage_quality",
            "budget_id": "small",
            "max_effective_train_tokens": 100_000,
            "selection_seed": 7,
            "selection_config_uri": "selectors/coverage-quality.yaml",
            "selection_config_sha256": "c" * 64,
        },
        "baselines": [
            {
                "arm": "proxy",
                "label": "uniform_random",
                "method": "uniform_random_seed_101",
                "budget_id": "small",
                "max_effective_train_tokens": 100_000,
                "selection_seed": 101,
                "selection_config_uri": "selectors/random-101.yaml",
                "selection_config_sha256": "d" * 64,
            }
        ],
        "reference": {
            "arm": "reference",
            "label": "full_reference",
            "method": "fully_open_meditron",
            "budget_id": "full",
            "max_effective_train_tokens": 1_000_000,
            "selection_seed": None,
            "selection_config_uri": None,
            "selection_config_sha256": None,
        },
        "reference_identifiability": reference_identifiability,
        "target_decision": target_decision,
        "indifference_margin": 0.0,
        "minimum_independent_lineage_count_for_warning": 8,
    }
    payload.update(criteria_overrides)
    return FeasibilityProtocol.model_validate(payload)


def _trial_rows(target="aligned", baseline="reverse", reference="separated"):
    suite, contract = _suite_and_contract()
    batch_ids = [batch.batch_id for batch in suite.batches]
    reference_values = {
        batch_id: (0.05 if reference == "constant" else 0.01 * (len(batch_ids) - index))
        for index, batch_id in enumerate(batch_ids)
    }

    def arm_values(kind):
        if kind == "aligned":
            return dict(reference_values)
        if kind == "reverse":
            return {batch_id: reference_values[batch_ids[-index - 1]] for index, batch_id in enumerate(batch_ids)}
        if kind == "constant":
            return {batch_id: 0.05 for batch_id in batch_ids}
        raise AssertionError(kind)

    target_values, baseline_values = arm_values(target), arm_values(baseline)
    suite_sha, contract_sha = suite_digest(suite), contract_digest(contract)
    baseline_by_seed = {17: 0.41, 29: 0.47, 43: 0.44}
    rows = []
    for batch in suite.batches:
        for seed in contract.training.seeds:
            base = baseline_by_seed[seed]
            for arm, method, budget_id, values, tokens in (
                ("proxy", "coverage_quality", "small", target_values, 100_000),
                ("proxy", "uniform_random_seed_101", "small", baseline_values, 100_000),
                (
                    "reference",
                    "fully_open_meditron",
                    "full",
                    reference_values,
                    contract.training.effective_train_tokens,
                ),
            ):
                rows.append(
                    TrialScore(
                        run_id=f"{method}-{batch.batch_id}-{seed}",
                        suite_sha256=suite_sha,
                        batch_id=batch.batch_id,
                        batch_content_sha256=batch.content_sha256,
                        arm=arm,
                        method=method,
                        budget_id=budget_id,
                        seed=seed,
                        benchmark_score=base + values[batch.batch_id],
                        baseline_score=base,
                        benchmark_sha256=contract.benchmark.test_manifest_sha256,
                        contract_sha256=contract_sha,
                        effective_train_tokens=tokens,
                    )
                )
    return suite, contract, rows


def test_feasible_when_target_passes_absolute_and_baseline_gates():
    suite, contract, rows = _trial_rows(target="aligned", baseline="reverse")
    report = assess_feasibility(rows, suite, contract, _protocol())

    assert report["verdict"] == "feasible"
    assert report["reason_codes"] == ["ALL_PREREGISTERED_GATES_PASSED"]
    assert len(report["protocol_sha256"]) == 64
    assert report["reference_identifiability"]["identified"] is True
    assert report["target"]["report"]["spearman"] == pytest.approx(1.0)
    assert all(row["target_spearman"] == pytest.approx(1.0) for row in report["leave_one_training_seed_out"])
    assert any("item-sampling uncertainty" in warning for warning in report["warnings"])
    assert any("lineage count is low" in warning for warning in report["warnings"])
    assert report["baseline_comparisons"][0]["target_advantages"]["spearman"] == pytest.approx(2.0)
    assert report["training_exposure_audit"]["registered_proxy_caps_identical"] is True
    assert (
        report["training_exposure_audit"]["target_vs_baseline_realized_exposure"]["uniform_random"][
            "all_batch_seed_cells_exactly_matched"
        ]
        is True
    )
    assert report["lineage_sensitivity"]["bootstrap_replicates_requested"] == 2000
    assert report["lineage_sensitivity"]["leave_one_lineage_out"]
    json.dumps(report, allow_nan=False)


def test_infeasible_requires_consistent_leave_one_seed_failure():
    suite, contract, rows = _trial_rows(target="reverse", baseline="aligned")
    report = assess_feasibility(rows, suite, contract, _protocol())

    assert report["verdict"] == "infeasible"
    assert "TARGET_FAILED_A_GATE_IN_EVERY_LEAVE_ONE_SEED_OUT_RUN" in report["reason_codes"]
    assert "TARGET_CONSISTENTLY_BASELINE_DOMINATED" in report["reason_codes"]
    assert all(row["target_clearly_failed_gates"] for row in report["leave_one_training_seed_out"])


def test_identifiable_but_undefined_target_ranking_is_inconclusive():
    suite, contract, rows = _trial_rows(target="constant", baseline="reverse")
    protocol = _protocol(
        target_decision={
            "min_kendall_tau_b": -1.0,
            "min_pairwise_order_accuracy": 0.0,
            "max_top1_regret": 1.0,
        }
    )
    report = assess_feasibility(rows, suite, contract, protocol)

    assert report["reference_identifiability"]["identified"] is True
    assert report["verdict"] == "inconclusive"
    assert report["reason_codes"] == ["TARGET_RESULT_NOT_DECISIVE_ACROSS_TRAINING_SEEDS"]


def test_incomplete_baseline_arm_is_rejected():
    suite, contract, rows = _trial_rows()
    rows = [
        row
        for row in rows
        if not (row.method == "uniform_random_seed_101" and row.batch_id == "B12" and row.seed == 43)
    ]
    with pytest.raises(ValueError, match="paired seeds"):
        assess_feasibility(rows, suite, contract, _protocol())


def test_per_arm_budget_violation_is_rejected():
    suite, contract, rows = _trial_rows()
    target_row = next(row for row in rows if row.method == "coverage_quality")
    target_row.effective_train_tokens = 100_001
    with pytest.raises(ValueError, match="preregistered token cap for arm coverage_quality"):
        assess_feasibility(rows, suite, contract, _protocol())


def test_reference_gate_failure_has_priority_over_strong_failure_signal():
    suite, contract, rows = _trial_rows(target="reverse", baseline="aligned")
    protocol = _protocol(reference_identifiability={"min_utility_range": 1.0})
    report = assess_feasibility(rows, suite, contract, protocol)

    assert all(row["target_clearly_failed_gates"] for row in report["leave_one_training_seed_out"])
    assert report["verdict"] == "inconclusive"
    assert report["reason_codes"][0] == "REFERENCE_NOT_IDENTIFIABLE"
    assert "REFERENCE_UTILITY_RANGE_BELOW_MIN" in report["reason_codes"]
    assert not any(code.startswith("TARGET_") for code in report["reason_codes"])


def test_seed_sensitive_target_cannot_receive_feasible_verdict():
    suite, contract, rows = _trial_rows(target="aligned", baseline="reverse")
    batch_ids = [batch.batch_id for batch in suite.batches]
    reversed_values = {batch_id: 0.01 * (index + 1) for index, batch_id in enumerate(batch_ids)}
    for row in rows:
        if row.method == "coverage_quality" and row.seed == 43:
            row.benchmark_score = row.baseline_score + reversed_values[row.batch_id]

    report = assess_feasibility(rows, suite, contract, _protocol())

    assert report["target"]["absolute_gates_passed"] is True
    assert report["leave_one_training_seed_out_stable"] is False
    assert report["verdict"] == "inconclusive"


def test_proxy_arms_must_use_equal_preregistered_token_budgets():
    protocol = _protocol().model_dump(mode="json")
    protocol["baselines"][0]["max_effective_train_tokens"] = 90_000
    with pytest.raises(ValidationError, match="same maximum effective training-token budget"):
        FeasibilityProtocol.model_validate(protocol)


def test_leave_one_seed_metrics_honor_reference_indifference_margin():
    suite, contract, rows = _trial_rows(target="aligned", baseline="reverse")
    for row in rows:
        if row.method != "coverage_quality":
            continue
        if row.batch_id == "B01":
            row.benchmark_score = row.baseline_score + 0.11
        elif row.batch_id == "B02":
            row.benchmark_score = row.baseline_score + 0.12

    protocol_payload = _protocol().model_dump(mode="json")
    protocol_payload["indifference_margin"] = 0.015
    with_margin = assess_feasibility(rows, suite, contract, FeasibilityProtocol.model_validate(protocol_payload))
    protocol_payload["indifference_margin"] = 0.0
    without_margin = assess_feasibility(rows, suite, contract, FeasibilityProtocol.model_validate(protocol_payload))

    assert with_margin["target"]["report"]["pairwise_order_accuracy"] == pytest.approx(1.0)
    assert all(
        row["target_metrics"]["pairwise_order_accuracy"] == pytest.approx(1.0)
        for row in with_margin["leave_one_training_seed_out"]
    )
    assert without_margin["target"]["report"]["pairwise_order_accuracy"] < 1.0


def test_direct_assessment_requires_full_reference_token_cap():
    suite, contract, rows = _trial_rows()
    payload = _protocol().model_dump(mode="json")
    payload["reference"]["max_effective_train_tokens"] = contract.training.effective_train_tokens - 1
    protocol = FeasibilityProtocol.model_validate(payload)

    with pytest.raises(ValueError, match="full-data reference"):
        assess_feasibility(rows, suite, contract, protocol)
