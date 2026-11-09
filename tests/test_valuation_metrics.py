import hashlib
from itertools import combinations as enumerate_combinations
from pathlib import Path

import pytest

from ai_ready.valuation.contracts import contract_digest, load_batch_suite, load_production_contract, suite_digest
from ai_ready.valuation.metrics import aggregate_trials, evaluate_proxy, validate_trial_completeness
from ai_ready.valuation.schema import BatchUtility, CombinationUtility, TrialScore

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "valuation_cll"


def _freeze_test_suite(suite):
    for batch in suite.batches:
        batch.content_sha256 = hashlib.sha256(f"frozen:{batch.batch_id}".encode()).hexdigest()
    return suite


def _trials(proxy_values=None, reference_values=None):
    suite = _freeze_test_suite(load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml"))
    contract = load_production_contract(EXAMPLE_DIR / "production_contract.yaml")
    proxy_values = proxy_values or {batch.batch_id: 0.01 * (13 - index) for index, batch in enumerate(suite.batches, 1)}
    reference_values = reference_values or {
        batch.batch_id: 0.02 * (13 - index) for index, batch in enumerate(suite.batches, 1)
    }
    rows = []
    suite_sha = suite_digest(suite)
    baseline_by_seed = {17: 0.41, 29: 0.47, 43: 0.44}
    for batch in suite.batches:
        for seed in contract.training.seeds:
            baseline = baseline_by_seed[seed]
            rows.extend(
                [
                    TrialScore(
                        run_id=f"proxy-{batch.batch_id}-{seed}",
                        suite_sha256=suite_sha,
                        batch_id=batch.batch_id,
                        batch_content_sha256=batch.content_sha256,
                        arm="proxy",
                        method="coverage_quality",
                        budget_id="low",
                        seed=seed,
                        benchmark_score=baseline + proxy_values[batch.batch_id],
                        baseline_score=baseline,
                        benchmark_sha256=contract.benchmark.test_manifest_sha256,
                        contract_sha256=contract_digest(contract),
                        effective_train_tokens=100000,
                    ),
                    TrialScore(
                        run_id=f"reference-{batch.batch_id}-{seed}",
                        suite_sha256=suite_sha,
                        batch_id=batch.batch_id,
                        batch_content_sha256=batch.content_sha256,
                        arm="reference",
                        method="fully_open_meditron",
                        budget_id="high",
                        seed=seed,
                        benchmark_score=baseline + reference_values[batch.batch_id],
                        baseline_score=baseline,
                        benchmark_sha256=contract.benchmark.test_manifest_sha256,
                        contract_sha256=contract_digest(contract),
                        effective_train_tokens=contract.training.effective_train_tokens,
                    ),
                ]
            )
    return suite, contract, rows


def _report(rows):
    contract = load_production_contract(EXAMPLE_DIR / "production_contract.yaml")
    suite = _freeze_test_suite(load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml"))
    utilities = aggregate_trials(rows, contract=contract)
    return evaluate_proxy(
        utilities,
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[1, 3],
    )


def test_paired_seed_aggregation_and_perfect_ranking():
    suite, contract, rows = _trials()
    validate_trial_completeness(
        rows,
        suite,
        contract,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
    )
    report = _report(rows)
    assert report["spearman"] == pytest.approx(1.0)
    assert report["kendall_tau_b"] == pytest.approx(1.0)
    assert report["regret"]["1"] == pytest.approx(0.0)
    assert report["regret"]["3"] is None
    assert any("joint-training" in warning for warning in report["warnings"])


def test_reverse_ranking_and_top1_regret():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    proxy = {batch.batch_id: float(index) for index, batch in enumerate(suite.batches)}
    reference = {batch.batch_id: float(len(suite.batches) - index) for index, batch in enumerate(suite.batches)}
    _, _, rows = _trials(proxy, reference)
    report = _report(rows)
    assert report["spearman"] == pytest.approx(-1.0)
    assert report["kendall_tau_b"] == pytest.approx(-1.0)
    assert report["regret"]["1"] == pytest.approx(11.0)


def test_all_ties_are_structured_none_not_nan():
    suite = _freeze_test_suite(load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml"))
    utility_rows = [
        BatchUtility(
            suite_sha256=suite_digest(suite),
            batch_id=batch_id,
            batch_content_sha256=next(batch.content_sha256 for batch in suite.batches if batch.batch_id == batch_id),
            arm=arm,
            method="proxy" if arm == "proxy" else "reference",
            budget_id="low" if arm == "proxy" else "high",
            benchmark_sha256="a" * 64,
            contract_sha256="b" * 64,
            mean_utility=0.0,
            std_utility=0.0,
            ci95_low=0.0,
            ci95_high=0.0,
            seeds=[1, 2, 3],
            mean_score=0.5,
            mean_baseline_score=0.5,
            mean_effective_train_tokens=10,
        )
        for batch_id in (batch.batch_id for batch in suite.batches)
        for arm in ("proxy", "reference")
    ]
    report = evaluate_proxy(
        utility_rows,
        suite=suite,
        proxy_method="proxy",
        proxy_budget_id="low",
        reference_method="reference",
        reference_budget_id="high",
    )
    assert report["spearman"] is None
    assert report["kendall_tau_b"] is None
    assert report["top_k"]["3"]["linear_ndcg"] == 1.0


def test_missing_reference_batch_is_rejected():
    suite, contract, rows = _trials()
    rows = [row for row in rows if not (row.arm == "reference" and row.batch_id == "B12")]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_trial_completeness(
            rows,
            suite,
            contract,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
        )


def test_inconsistent_paired_baseline_is_rejected():
    _, contract, rows = _trials()
    rows[1].baseline_score += 0.01
    with pytest.raises(ValueError, match="inconsistent paired baseline"):
        aggregate_trials(rows, contract=contract)


def test_aggregation_rejects_wrong_contract_and_benchmark_identity():
    _, contract, rows = _trials()
    rows[0].contract_sha256 = "f" * 64
    with pytest.raises(ValueError, match="supplied production contract"):
        aggregate_trials(rows, contract=contract)


def test_batch_content_is_frozen_and_bound_to_every_trial():
    raw_suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    _, contract, rows = _trials()
    with pytest.raises(ValueError, match="frozen content_sha256"):
        validate_trial_completeness(
            rows,
            raw_suite,
            contract,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
        )

    suite, contract, rows = _trials()
    rows[0].batch_content_sha256 = "f" * 64
    with pytest.raises(ValueError, match="frozen content of batch"):
        validate_trial_completeness(
            rows,
            suite,
            contract,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
        )


def test_aggregation_rejects_same_batch_with_different_content():
    _, contract, rows = _trials()
    rows[0].batch_content_sha256 = "f" * 64
    with pytest.raises(ValueError, match="inconsistent content hashes"):
        aggregate_trials(rows, contract=contract)
    _, contract, rows = _trials()
    rows[0].benchmark_sha256 = "f" * 64
    with pytest.raises(ValueError, match="supplied locked benchmark"):
        aggregate_trials(rows, contract=contract)


def test_lower_is_better_direction_is_explicitly_normalized():
    _, contract, all_rows = _trials()
    contract.benchmark.higher_is_better = False
    rows = [row for row in all_rows if row.arm == "reference" and row.batch_id == "B01"]
    for row in rows:
        row.contract_sha256 = contract_digest(contract)
        row.benchmark_score = row.baseline_score - 0.2
    utility = aggregate_trials(rows, contract=contract)[0]
    assert utility.mean_utility == pytest.approx(0.2)


def test_small_seed_confidence_interval_uses_student_t():
    _, contract, all_rows = _trials()
    rows = [row for row in all_rows if row.arm == "reference" and row.batch_id == "B01"]
    for row, delta in zip(rows, (0.0, 1.0, 2.0)):
        row.benchmark_score = row.baseline_score + delta
    utility = aggregate_trials(rows, contract=contract)[0]
    assert utility.mean_utility == pytest.approx(1.0)
    assert utility.std_utility == pytest.approx(1.0)
    assert utility.ci95_high == pytest.approx(3.4841377117)
    assert utility.ci95_low == pytest.approx(-1.4841377117)


def test_low_yield_reference_may_be_below_cap_but_never_above_it():
    suite, contract, rows = _trials()
    low_yield = next(row for row in rows if row.arm == "reference")
    low_yield.effective_train_tokens = contract.budget.training_token_cap - 1
    validate_trial_completeness(
        rows,
        suite,
        contract,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
    )
    low_yield.effective_train_tokens = contract.budget.training_token_cap + 1
    with pytest.raises(ValueError, match="exceeds the frozen training-token cap"):
        validate_trial_completeness(
            rows,
            suite,
            contract,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
        )


def test_combination_regret_requires_matching_observed_joint_training_identity():
    suite, contract, rows = _trials()
    utilities = aggregate_trials(rows, contract=contract)
    reference = {
        row.batch_id: row
        for row in utilities
        if row.arm == "reference" and row.method == "fully_open_meditron"
    }

    def combination(batch_ids, value):
        return CombinationUtility(
            suite_sha256=suite_digest(suite),
            batch_ids=batch_ids,
            batch_content_sha256={batch_id: reference[batch_id].batch_content_sha256 for batch_id in batch_ids},
            reference_method="fully_open_meditron",
            budget_id="high",
            benchmark_sha256=contract.benchmark.test_manifest_sha256,
            contract_sha256=contract_digest(contract),
            seeds=contract.training.seeds,
            mean_utility=value,
        )

    selected = combination(["B01", "B02"], 0.3)
    complete_combinations = [
        combination(list(batch_ids), 0.5 if batch_ids == ("B01", "B03") else 0.3)
        for batch_ids in enumerate_combinations(sorted(reference), 2)
    ]
    report = evaluate_proxy(
        utilities,
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[2],
        combination_utilities=complete_combinations,
    )
    assert report["regret"]["2"] == pytest.approx(0.2)

    with pytest.raises(ValueError, match="Duplicate observed combination"):
        evaluate_proxy(
            utilities,
            suite=suite,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
            top_ks=[2],
            combination_utilities=[selected, selected.model_copy(deep=True)],
        )

    wrong_content = selected.model_copy(deep=True)
    wrong_content.batch_content_sha256["B01"] = "f" * 64
    with pytest.raises(ValueError, match="frozen content of batch B01"):
        evaluate_proxy(
            utilities,
            suite=suite,
            proxy_method="coverage_quality",
            proxy_budget_id="low",
            reference_method="fully_open_meditron",
            reference_budget_id="high",
            top_ks=[2],
            combination_utilities=[
                wrong_content if row.batch_ids == selected.batch_ids else row for row in complete_combinations
            ],
        )


def test_combination_regret_is_not_reported_for_partial_joint_training():
    suite, contract, rows = _trials()
    utilities = aggregate_trials(rows, contract=contract)
    reference = {
        row.batch_id: row
        for row in utilities
        if row.arm == "reference" and row.method == "fully_open_meditron"
    }
    partial = CombinationUtility(
        suite_sha256=suite_digest(suite),
        batch_ids=["B01", "B02"],
        batch_content_sha256={batch_id: reference[batch_id].batch_content_sha256 for batch_id in ("B01", "B02")},
        reference_method="fully_open_meditron",
        budget_id="high",
        benchmark_sha256=contract.benchmark.test_manifest_sha256,
        contract_sha256=contract_digest(contract),
        seeds=contract.training.seeds,
        mean_utility=0.3,
    )
    report = evaluate_proxy(
        utilities,
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[2],
        combination_utilities=[partial],
    )
    assert report["regret"]["2"] is None
    assert any("incomplete" in warning for warning in report["warnings"])


def test_indifference_margin_does_not_reward_or_penalize_near_tie_order():
    raw_suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    batch_ids = [batch.batch_id for batch in raw_suite.batches]
    reference = {
        batch_id: (0.10 if index == 0 else 0.11 if index == 1 else 0.16 + 0.02 * index)
        for index, batch_id in enumerate(batch_ids)
    }
    aligned = dict(reference)
    swapped = dict(reference)
    swapped[batch_ids[0]], swapped[batch_ids[1]] = swapped[batch_ids[1]], swapped[batch_ids[0]]

    suite, contract, aligned_rows = _trials(aligned, reference)
    aligned_report = evaluate_proxy(
        aggregate_trials(aligned_rows, contract=contract),
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[1],
        indifference_margin=0.02,
    )
    suite, contract, swapped_rows = _trials(swapped, reference)
    swapped_report = evaluate_proxy(
        aggregate_trials(swapped_rows, contract=contract),
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[1],
        indifference_margin=0.02,
    )
    no_margin_report = evaluate_proxy(
        aggregate_trials(swapped_rows, contract=contract),
        suite=suite,
        proxy_method="coverage_quality",
        proxy_budget_id="low",
        reference_method="fully_open_meditron",
        reference_budget_id="high",
        top_ks=[1],
        indifference_margin=0.0,
    )

    assert swapped_report["pairwise_order_accuracy"] == pytest.approx(1.0)
    assert swapped_report["spearman"] == pytest.approx(aligned_report["spearman"])
    assert swapped_report["kendall_tau_b"] == pytest.approx(aligned_report["kendall_tau_b"])
    assert swapped_report["raw_spearman"] < aligned_report["raw_spearman"]
    assert no_margin_report["pairwise_order_accuracy"] < 1.0
