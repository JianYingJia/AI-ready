"""Preregistered, descriptive feasibility decisions for proxy SFT data.

This module deliberately stops short of confirmatory inference.  It validates
one target proxy, one or more cheap baselines, and a full-data reference on the
same frozen batches and training seeds.  The reference must first demonstrate
that the experiment can distinguish batch utility at all; a weak reference can
only produce an ``inconclusive`` result, never evidence against the proxy.
"""

import math
import random
import statistics
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import yaml

from ai_ready.valuation.contracts import canonical_digest
from ai_ready.valuation.metrics import aggregate_trials, evaluate_proxy, validate_trial_completeness
from ai_ready.valuation.schema import (
    BatchSuite,
    BatchUtility,
    FeasibilityArmSpec,
    FeasibilityProtocol,
    ProductionContract,
    TrialScore,
)


def load_feasibility_protocol(path: str | Path) -> FeasibilityProtocol:
    """Load a strict YAML protocol; unknown fields are rejected by Pydantic."""

    protocol_path = Path(path)
    with protocol_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Feasibility protocol must contain a YAML mapping: {protocol_path}")
    return FeasibilityProtocol.model_validate(payload)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = ((position + 1) + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average
        position = end
    return ranks


def _margin_aware_ranks(values: Sequence[float], margin: float) -> list[float]:
    """Match the practical-equivalence ranking used by ``evaluate_proxy``."""

    if margin <= 0:
        return _average_ranks(values)
    ranks = []
    for value in values:
        definitely_below = sum(value - other > margin for other in values)
        indistinguishable = sum(abs(value - other) <= margin for other in values)
        ranks.append(float(definitely_below + (indistinguishable + 1) / 2.0))
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_delta) * sum(value * value for value in right_delta))
    if denominator <= 1e-15:
        return None
    return float(sum(a * b for a, b in zip(left_delta, right_delta)) / denominator)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau_b(
    left: Sequence[float],
    right: Sequence[float],
    *,
    right_indifference_margin: float = 0.0,
) -> float | None:
    concordant = discordant = tied_left = tied_right = 0
    for first, second in combinations(range(len(left)), 2):
        left_sign = (left[first] > left[second]) - (left[first] < left[second])
        right_delta = right[first] - right[second]
        right_sign = 0 if abs(right_delta) <= right_indifference_margin else (right_delta > 0) - (right_delta < 0)
        if left_sign == 0 and right_sign == 0:
            continue
        if left_sign == 0:
            tied_left += 1
        elif right_sign == 0:
            tied_right += 1
        elif left_sign == right_sign:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + tied_left) * (concordant + discordant + tied_right))
    if denominator == 0:
        return None
    return float((concordant - discordant) / denominator)


def _ranking_metrics(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    indifference_margin: float,
) -> dict[str, float | None]:
    """Metrics used for leave-one-training-seed-out stability checks."""

    batch_ids = sorted(reference)
    candidate_values = [candidate[batch_id] for batch_id in batch_ids]
    reference_values = [reference[batch_id] for batch_id in batch_ids]
    pairwise = []
    for first, second in combinations(range(len(batch_ids)), 2):
        reference_delta = reference_values[first] - reference_values[second]
        reference_sign = (
            0 if abs(reference_delta) <= indifference_margin else (reference_delta > 0) - (reference_delta < 0)
        )
        if reference_sign == 0:
            continue
        candidate_sign = (candidate_values[first] > candidate_values[second]) - (
            candidate_values[first] < candidate_values[second]
        )
        pairwise.append(0.5 if candidate_sign == 0 else float(candidate_sign == reference_sign))
    predicted_top = sorted(candidate, key=lambda batch_id: (-candidate[batch_id], batch_id))[0]
    top1_regret = max(reference.values()) - reference[predicted_top]
    return {
        "spearman": _pearson(
            _average_ranks(candidate_values),
            _margin_aware_ranks(reference_values, indifference_margin),
        ),
        "kendall_tau_b": _kendall_tau_b(
            candidate_values,
            reference_values,
            right_indifference_margin=indifference_margin,
        ),
        "pairwise_order_accuracy": _mean(pairwise) if pairwise else None,
        "top1_regret": float(top1_regret),
    }


def _arm_identity(arm: FeasibilityArmSpec) -> tuple[str, str, str]:
    return arm.arm, arm.method, arm.budget_id


def _validate_and_select_trials(
    trials: Sequence[TrialScore],
    suite: BatchSuite,
    contract: ProductionContract,
    protocol: FeasibilityProtocol,
) -> tuple[list[TrialScore], int]:
    arms = [protocol.target, *protocol.baselines, protocol.reference]
    if any(arm.max_effective_train_tokens > contract.budget.training_token_cap for arm in arms):
        raise ValueError("Every protocol arm token cap must be at most the production-contract training-token cap")
    if protocol.reference.max_effective_train_tokens != contract.training.effective_train_tokens:
        raise ValueError("The full-data reference must use the frozen production-contract training-token cap")

    by_identity = {_arm_identity(arm): arm for arm in arms}
    selected = [row for row in trials if (row.arm, row.method, row.budget_id) in by_identity]
    ignored_count = len(trials) - len(selected)

    # Reuse the frozen suite/contract checks for each proxy arm.  Doing this
    # separately makes every arm × batch × training-seed cell mandatory.
    for proxy_arm in [protocol.target, *protocol.baselines]:
        validate_trial_completeness(
            selected,
            suite,
            contract,
            proxy_method=proxy_arm.method,
            proxy_budget_id=proxy_arm.budget_id,
            reference_method=protocol.reference.method,
            reference_budget_id=protocol.reference.budget_id,
        )

    for row in selected:
        arm = by_identity[(row.arm, row.method, row.budget_id)]
        if row.effective_train_tokens > arm.max_effective_train_tokens:
            raise ValueError(
                f"Trial {row.run_id} exceeds preregistered token cap for arm {arm.label}: "
                f"{row.effective_train_tokens} > {arm.max_effective_train_tokens}"
            )

    # Aggregation also rejects duplicate run IDs/cells and enforces exact seed
    # sets, so a duplicated observation cannot silently receive extra weight.
    aggregate_trials(selected, contract=contract)
    return selected, ignored_count


def _reference_identifiability(
    trials: Sequence[TrialScore],
    utilities: Sequence[BatchUtility],
    contract: ProductionContract,
    protocol: FeasibilityProtocol,
) -> dict:
    reference_arm = protocol.reference
    reference_rows = [
        row
        for row in utilities
        if row.arm == "reference" and row.method == reference_arm.method and row.budget_id == reference_arm.budget_id
    ]
    reference_by_batch = {row.batch_id: row for row in reference_rows}
    batch_ids = sorted(reference_by_batch)
    means = [reference_by_batch[batch_id].mean_utility for batch_id in batch_ids]
    utility_range = float(max(means) - min(means))
    between_batch_sd = float(statistics.stdev(means)) if len(means) > 1 else 0.0

    degrees_of_freedom = sum(len(row.seeds) - 1 for row in reference_rows)
    pooled_variance = (
        sum((len(row.seeds) - 1) * row.std_utility**2 for row in reference_rows) / degrees_of_freedom
        if degrees_of_freedom > 0
        else 0.0
    )
    pooled_within_batch_sd = float(math.sqrt(max(pooled_variance, 0.0)))
    within_to_between_sd_ratio = float(pooled_within_batch_sd / between_batch_sd) if between_batch_sd > 1e-15 else None

    direction = 1.0 if contract.benchmark.higher_is_better else -1.0
    raw_reference = [
        row
        for row in trials
        if row.arm == "reference" and row.method == reference_arm.method and row.budget_id == reference_arm.budget_id
    ]
    utility_by_seed_batch = {(row.seed, row.batch_id): direction * row.utility for row in raw_reference}
    seed_rank_correlations = []
    undefined_seed_pairs = 0
    for first_seed, second_seed in combinations(contract.training.seeds, 2):
        correlation = _spearman(
            [utility_by_seed_batch[(first_seed, batch_id)] for batch_id in batch_ids],
            [utility_by_seed_batch[(second_seed, batch_id)] for batch_id in batch_ids],
        )
        if correlation is None:
            undefined_seed_pairs += 1
        else:
            seed_rank_correlations.append(correlation)
    # A median over only the well-behaved seed pairs could hide one completely
    # tied run.  Treat any undefined pair as an unidentified stability metric.
    seed_rank_median = (
        float(statistics.median(seed_rank_correlations))
        if seed_rank_correlations and undefined_seed_pairs == 0
        else None
    )

    criteria = protocol.reference_identifiability
    clearly_positive = sorted(
        batch_id for batch_id, row in reference_by_batch.items() if row.ci95_low > criteria.positive_margin
    )
    distinguishable_pairs = []
    total_pairs = math.comb(len(batch_ids), 2)
    for first, second in combinations(batch_ids, 2):
        first_row, second_row = reference_by_batch[first], reference_by_batch[second]
        separated = (
            first_row.ci95_low > second_row.ci95_high + criteria.ci_separation_margin
            or second_row.ci95_low > first_row.ci95_high + criteria.ci_separation_margin
        )
        if separated:
            distinguishable_pairs.append([first, second])
    distinguishable_fraction = float(len(distinguishable_pairs) / total_pairs) if total_pairs else None

    gate_rows = {
        "utility_range": {
            "observed": utility_range,
            "operator": ">=",
            "threshold": criteria.min_utility_range,
            "passed": utility_range >= criteria.min_utility_range,
            "reason_code": "REFERENCE_UTILITY_RANGE_BELOW_MIN",
        },
        "between_batch_sd": {
            "observed": between_batch_sd,
            "operator": ">=",
            "threshold": criteria.min_between_batch_sd,
            "passed": between_batch_sd >= criteria.min_between_batch_sd,
            "reason_code": "REFERENCE_BETWEEN_BATCH_SD_BELOW_MIN",
        },
        "within_to_between_sd_ratio": {
            "observed": within_to_between_sd_ratio,
            "operator": "<=",
            "threshold": criteria.max_within_to_between_sd_ratio,
            "passed": within_to_between_sd_ratio is not None
            and within_to_between_sd_ratio <= criteria.max_within_to_between_sd_ratio,
            "reason_code": (
                "REFERENCE_NOISE_RATIO_UNDEFINED"
                if within_to_between_sd_ratio is None
                else "REFERENCE_NOISE_RATIO_ABOVE_MAX"
            ),
        },
        "pairwise_seed_rank_spearman_median": {
            "observed": seed_rank_median,
            "operator": ">=",
            "threshold": criteria.min_pairwise_seed_rank_spearman_median,
            "passed": seed_rank_median is not None
            and seed_rank_median >= criteria.min_pairwise_seed_rank_spearman_median,
            "reason_code": (
                "REFERENCE_SEED_RANK_STABILITY_UNDEFINED"
                if seed_rank_median is None
                else "REFERENCE_SEED_RANK_STABILITY_BELOW_MIN"
            ),
        },
        "clearly_positive_batch_count": {
            "observed": len(clearly_positive),
            "operator": ">=",
            "threshold": criteria.min_clearly_positive_batch_count,
            "passed": len(clearly_positive) >= criteria.min_clearly_positive_batch_count,
            "reason_code": "REFERENCE_TOO_FEW_CLEARLY_POSITIVE_BATCHES",
        },
        "distinguishable_ci_pair_fraction": {
            "observed": distinguishable_fraction,
            "operator": ">=",
            "threshold": criteria.min_distinguishable_ci_pair_fraction,
            "passed": distinguishable_fraction is not None
            and distinguishable_fraction >= criteria.min_distinguishable_ci_pair_fraction,
            "reason_code": "REFERENCE_TOO_FEW_DISTINGUISHABLE_PAIRS",
        },
    }
    failure_codes = sorted({row["reason_code"] for row in gate_rows.values() if not row["passed"]})
    return {
        "identified": not failure_codes,
        "utility_range": utility_range,
        "between_batch_sd": between_batch_sd,
        "pooled_within_batch_sd": pooled_within_batch_sd,
        "within_to_between_sd_ratio": within_to_between_sd_ratio,
        "pairwise_seed_rank_spearman_median": seed_rank_median,
        "pairwise_seed_rank_spearman_values": seed_rank_correlations,
        "undefined_seed_rank_pair_count": undefined_seed_pairs,
        "clearly_positive_batch_ids": clearly_positive,
        "positive_margin": criteria.positive_margin,
        "distinguishable_ci_pairs": distinguishable_pairs,
        "distinguishable_ci_pair_fraction": distinguishable_fraction,
        "ci_separation_margin": criteria.ci_separation_margin,
        "gates": gate_rows,
        "failure_reason_codes": failure_codes,
    }


def _absolute_gates(report: dict, protocol: FeasibilityProtocol) -> dict:
    criteria = protocol.target_decision
    observed = {
        "spearman": report["spearman"],
        "kendall_tau_b": report["kendall_tau_b"],
        "pairwise_order_accuracy": report["pairwise_order_accuracy"],
        "top1_regret": report["regret"]["1"],
    }
    definitions = {
        "spearman": (">=", criteria.min_spearman),
        "kendall_tau_b": (">=", criteria.min_kendall_tau_b),
        "pairwise_order_accuracy": (">=", criteria.min_pairwise_order_accuracy),
        "top1_regret": ("<=", criteria.max_top1_regret),
    }
    gates = {}
    for name, value in observed.items():
        operator, threshold = definitions[name]
        passed = value is not None and (value >= threshold if operator == ">=" else value <= threshold)
        gates[name] = {"observed": value, "operator": operator, "threshold": threshold, "passed": passed}
    return gates


def _comparison(
    target_report: dict,
    baseline_report: dict,
    baseline: FeasibilityArmSpec,
    protocol: FeasibilityProtocol,
) -> dict:
    criteria = protocol.target_decision
    target_spearman = target_report["spearman"]
    baseline_spearman = baseline_report["spearman"]
    target_regret = target_report["regret"]["1"]
    baseline_regret = baseline_report["regret"]["1"]
    spearman_delta = (
        float(target_spearman - baseline_spearman)
        if target_spearman is not None and baseline_spearman is not None
        else None
    )
    regret_excess = (
        float(target_regret - baseline_regret) if target_regret is not None and baseline_regret is not None else None
    )
    spearman_passed = (
        spearman_delta is not None and spearman_delta >= criteria.min_spearman_advantage_over_each_baseline
    )
    regret_passed = regret_excess is not None and regret_excess <= criteria.max_top1_regret_excess_over_each_baseline
    # Positive values below always favor the proposed target.  Reporting the
    # direct differences prevents two overlapping per-method intervals from
    # being mistaken for evidence about the method contrast itself.
    target_advantages = {
        "spearman": spearman_delta,
        "kendall_tau_b": _optional_difference(target_report["kendall_tau_b"], baseline_report["kendall_tau_b"]),
        "pairwise_order_accuracy": _optional_difference(
            target_report["pairwise_order_accuracy"], baseline_report["pairwise_order_accuracy"]
        ),
        "top1_recovery": _optional_difference(
            target_report["top_k"]["1"]["recall"], baseline_report["top_k"]["1"]["recall"]
        ),
        "top1_regret_reduction": _optional_difference(baseline_regret, target_regret),
    }
    return {
        "baseline_label": baseline.label,
        "selection_seed": baseline.selection_seed,
        "spearman_advantage": {
            "observed": spearman_delta,
            "operator": ">=",
            "threshold": criteria.min_spearman_advantage_over_each_baseline,
            "passed": spearman_passed,
        },
        "top1_regret_excess": {
            "observed": regret_excess,
            "operator": "<=",
            "threshold": criteria.max_top1_regret_excess_over_each_baseline,
            "passed": regret_passed,
        },
        "target_advantages": target_advantages,
        "baseline_dominates": (
            spearman_delta is not None
            and spearman_delta < criteria.min_spearman_advantage_over_each_baseline
            and regret_excess is not None
            and regret_excess > criteria.max_top1_regret_excess_over_each_baseline
        ),
        "passed": spearman_passed and regret_passed,
    }


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return float(left - right) if left is not None and right is not None else None


def _arm_values(
    utilities: Sequence[BatchUtility],
    arms: Sequence[FeasibilityArmSpec],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for arm in arms:
        selected = {
            row.batch_id: row.mean_utility
            for row in utilities
            if (row.arm, row.method, row.budget_id) == _arm_identity(arm)
        }
        values[arm.label] = selected
    return values


def _metric_advantages(
    target: dict[str, float],
    baseline: dict[str, float],
    reference: dict[str, float],
    *,
    indifference_margin: float,
) -> dict[str, float | None]:
    target_metrics = _ranking_metrics(target, reference, indifference_margin=indifference_margin)
    baseline_metrics = _ranking_metrics(baseline, reference, indifference_margin=indifference_margin)
    target_top = sorted(target, key=lambda batch_id: (-target[batch_id], batch_id))[0]
    baseline_top = sorted(baseline, key=lambda batch_id: (-baseline[batch_id], batch_id))[0]
    reference_top = sorted(reference, key=lambda batch_id: (-reference[batch_id], batch_id))[0]
    return {
        "spearman": _optional_difference(target_metrics["spearman"], baseline_metrics["spearman"]),
        "kendall_tau_b": _optional_difference(target_metrics["kendall_tau_b"], baseline_metrics["kendall_tau_b"]),
        "pairwise_order_accuracy": _optional_difference(
            target_metrics["pairwise_order_accuracy"], baseline_metrics["pairwise_order_accuracy"]
        ),
        "top1_recovery": float(target_top == reference_top) - float(baseline_top == reference_top),
        "top1_regret_reduction": _optional_difference(baseline_metrics["top1_regret"], target_metrics["top1_regret"]),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile from an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _lineage_sensitivity(
    utilities: Sequence[BatchUtility],
    suite: BatchSuite,
    protocol: FeasibilityProtocol,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261022,
) -> dict:
    """Estimate target-vs-baseline contrasts at the independent-lineage level.

    The interval is a descriptive cluster bootstrap.  It resamples lineage
    groups rather than treating related constructed batches as independent.
    With few lineages it is deliberately reported as a sensitivity interval,
    not a confirmatory confidence interval.
    """

    arms = [protocol.target, *protocol.baselines, protocol.reference]
    values = _arm_values(utilities, arms)
    lineage_to_batches: dict[str, list[str]] = {}
    for batch in suite.batches:
        lineage_to_batches.setdefault(batch.lineage_group, []).append(batch.batch_id)
    lineages = sorted(lineage_to_batches)

    leave_one_out = []
    for omitted in lineages:
        retained = [
            batch_id
            for lineage, batch_ids in lineage_to_batches.items()
            if lineage != omitted
            for batch_id in batch_ids
        ]
        if len(retained) < 2:
            continue
        reference = {batch_id: values[protocol.reference.label][batch_id] for batch_id in retained}
        leave_one_out.append(
            {
                "omitted_lineage_group": omitted,
                "retained_batch_count": len(retained),
                "target_advantages": {
                    baseline.label: _metric_advantages(
                        {batch_id: values[protocol.target.label][batch_id] for batch_id in retained},
                        {batch_id: values[baseline.label][batch_id] for batch_id in retained},
                        reference,
                        indifference_margin=protocol.indifference_margin,
                    )
                    for baseline in protocol.baselines
                },
            }
        )

    random_generator = random.Random(bootstrap_seed)
    draws: dict[str, dict[str, list[float]]] = {
        baseline.label: {
            metric: []
            for metric in (
                "spearman",
                "kendall_tau_b",
                "pairwise_order_accuracy",
                "top1_recovery",
                "top1_regret_reduction",
            )
        }
        for baseline in protocol.baselines
    }
    for _ in range(bootstrap_replicates):
        sampled_lineages = [random_generator.choice(lineages) for _ in lineages]
        sampled_ids = [
            (f"draw-{draw_index:02d}-{batch_index:03d}-{batch_id}", batch_id)
            for draw_index, lineage in enumerate(sampled_lineages)
            for batch_index, batch_id in enumerate(lineage_to_batches[lineage])
        ]
        if len(sampled_ids) < 2:
            continue
        reference = {draw_id: values[protocol.reference.label][batch_id] for draw_id, batch_id in sampled_ids}
        target = {draw_id: values[protocol.target.label][batch_id] for draw_id, batch_id in sampled_ids}
        for baseline in protocol.baselines:
            baseline_values = {draw_id: values[baseline.label][batch_id] for draw_id, batch_id in sampled_ids}
            advantages = _metric_advantages(
                target,
                baseline_values,
                reference,
                indifference_margin=protocol.indifference_margin,
            )
            for metric, value in advantages.items():
                if value is not None and math.isfinite(value):
                    draws[baseline.label][metric].append(value)

    intervals = {}
    for baseline in protocol.baselines:
        intervals[baseline.label] = {}
        for metric, metric_draws in draws[baseline.label].items():
            intervals[baseline.label][metric] = {
                "valid_replicates": len(metric_draws),
                "percentile_95_interval": (
                    [_percentile(metric_draws, 0.025), _percentile(metric_draws, 0.975)] if metric_draws else None
                ),
                "fraction_target_advantage_positive": (
                    sum(value > 0 for value in metric_draws) / len(metric_draws) if metric_draws else None
                ),
            }
    return {
        "inference_scope": "descriptive_lineage_cluster_sensitivity",
        "independent_lineage_count": len(lineages),
        "bootstrap_replicates_requested": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "leave_one_lineage_out": leave_one_out,
        "cluster_bootstrap_target_advantage_intervals": intervals,
    }


def _training_exposure_audit(
    trials: Sequence[TrialScore],
    protocol: FeasibilityProtocol,
) -> dict:
    proxy_arms = [protocol.target, *protocol.baselines]
    all_arms = [*proxy_arms, protocol.reference]
    by_label = {
        arm.label: [row for row in trials if (row.arm, row.method, row.budget_id) == _arm_identity(arm)]
        for arm in all_arms
    }
    summaries = {}
    for arm in all_arms:
        tokens = [row.effective_train_tokens for row in by_label[arm.label]]
        generator_identities = sorted(
            {
                (
                    getattr(row, "generator", None),
                    getattr(row, "generator_revision", None),
                    getattr(row, "generation_recipe_sha256", None),
                )
                for row in by_label[arm.label]
                if getattr(row, "generator", None) is not None
            }
        )
        summaries[arm.label] = {
            "registered_max_effective_train_tokens": arm.max_effective_train_tokens,
            "observed_trial_count": len(tokens),
            "observed_mean_effective_train_tokens": _mean(tokens),
            "observed_min_effective_train_tokens": min(tokens),
            "observed_max_effective_train_tokens": max(tokens),
            "generator_identities": [list(identity) for identity in generator_identities],
        }

    target_cells = {(row.batch_id, row.seed): row.effective_train_tokens for row in by_label[protocol.target.label]}
    realized_comparisons = {}
    for baseline in protocol.baselines:
        baseline_cells = {(row.batch_id, row.seed): row.effective_train_tokens for row in by_label[baseline.label]}
        differences = [target_cells[key] - baseline_cells[key] for key in sorted(target_cells)]
        realized_comparisons[baseline.label] = {
            "mean_target_minus_baseline_tokens": _mean(differences),
            "mean_absolute_token_difference": _mean([abs(value) for value in differences]),
            "max_absolute_token_difference": max(abs(value) for value in differences),
            "all_batch_seed_cells_exactly_matched": all(value == 0 for value in differences),
        }

    proxy_generator_identities = {
        identity
        for arm in proxy_arms
        for identity in (tuple(row) for row in summaries[arm.label]["generator_identities"])
    }
    provenance_available = all(summaries[arm.label]["generator_identities"] for arm in proxy_arms)
    return {
        "registered_proxy_caps_identical": len({arm.max_effective_train_tokens for arm in proxy_arms}) == 1,
        "registered_proxy_selection_budgets_identical": len({arm.selection_token_budget for arm in proxy_arms}) == 1,
        "generator_provenance_available": provenance_available,
        "same_proxy_generator_revision_and_recipe": (
            len(proxy_generator_identities) == 1 if provenance_available else None
        ),
        "arm_summaries": summaries,
        "target_vs_baseline_realized_exposure": realized_comparisons,
    }


def _leave_one_seed_out(
    trials: Sequence[TrialScore],
    contract: ProductionContract,
    protocol: FeasibilityProtocol,
) -> list[dict]:
    arms = [protocol.target, *protocol.baselines, protocol.reference]
    direction = 1.0 if contract.benchmark.higher_is_better else -1.0
    batch_ids = sorted({row.batch_id for row in trials})
    utilities = {
        (_arm_identity(arm), row.batch_id, row.seed): direction * row.utility
        for arm in arms
        for row in trials
        if (row.arm, row.method, row.budget_id) == _arm_identity(arm)
    }
    rows = []
    criteria = protocol.target_decision
    for omitted_seed in sorted(contract.training.seeds):
        remaining_seeds = [seed for seed in contract.training.seeds if seed != omitted_seed]
        means_by_arm = {
            arm.label: {
                batch_id: _mean([utilities[(_arm_identity(arm), batch_id, seed)] for seed in remaining_seeds])
                for batch_id in batch_ids
            }
            for arm in arms
        }
        reference = means_by_arm[protocol.reference.label]
        arm_metrics = {
            arm.label: _ranking_metrics(
                means_by_arm[arm.label],
                reference,
                indifference_margin=protocol.indifference_margin,
            )
            for arm in [protocol.target, *protocol.baselines]
        }
        target_metrics = arm_metrics[protocol.target.label]
        target_failures = []
        if target_metrics["spearman"] is not None and target_metrics["spearman"] < criteria.min_spearman:
            target_failures.append("spearman")
        if target_metrics["kendall_tau_b"] is not None and target_metrics["kendall_tau_b"] < criteria.min_kendall_tau_b:
            target_failures.append("kendall_tau_b")
        if (
            target_metrics["pairwise_order_accuracy"] is not None
            and target_metrics["pairwise_order_accuracy"] < criteria.min_pairwise_order_accuracy
        ):
            target_failures.append("pairwise_order_accuracy")
        if target_metrics["top1_regret"] is not None and target_metrics["top1_regret"] > criteria.max_top1_regret:
            target_failures.append("top1_regret")

        baseline_comparisons = []
        for baseline in protocol.baselines:
            baseline_metrics = arm_metrics[baseline.label]
            target_spearman, baseline_spearman = target_metrics["spearman"], baseline_metrics["spearman"]
            target_regret, baseline_regret = target_metrics["top1_regret"], baseline_metrics["top1_regret"]
            spearman_delta = (
                float(target_spearman - baseline_spearman)
                if target_spearman is not None and baseline_spearman is not None
                else None
            )
            regret_excess = (
                float(target_regret - baseline_regret)
                if target_regret is not None and baseline_regret is not None
                else None
            )
            baseline_comparisons.append(
                {
                    "baseline_label": baseline.label,
                    "spearman_advantage": spearman_delta,
                    "top1_regret_excess": regret_excess,
                    "spearman_passed": (
                        spearman_delta is not None
                        and spearman_delta >= criteria.min_spearman_advantage_over_each_baseline
                    ),
                    "top1_regret_passed": (
                        regret_excess is not None
                        and regret_excess <= criteria.max_top1_regret_excess_over_each_baseline
                    ),
                    # Requiring failure on both axes is a conservative meaning
                    # of domination; one noisy metric cannot force infeasible.
                    "baseline_dominates": (
                        spearman_delta is not None
                        and spearman_delta < criteria.min_spearman_advantage_over_each_baseline
                        and regret_excess is not None
                        and regret_excess > criteria.max_top1_regret_excess_over_each_baseline
                    ),
                }
            )
        rows.append(
            {
                "omitted_training_seed": omitted_seed,
                "remaining_training_seeds": remaining_seeds,
                "target_spearman": target_metrics["spearman"],
                "target_metrics": target_metrics,
                "target_clearly_failed_gates": target_failures,
                "target_absolute_gates_passed": (
                    target_metrics["spearman"] is not None
                    and target_metrics["spearman"] >= criteria.min_spearman
                    and target_metrics["kendall_tau_b"] is not None
                    and target_metrics["kendall_tau_b"] >= criteria.min_kendall_tau_b
                    and target_metrics["pairwise_order_accuracy"] is not None
                    and target_metrics["pairwise_order_accuracy"] >= criteria.min_pairwise_order_accuracy
                    and target_metrics["top1_regret"] is not None
                    and target_metrics["top1_regret"] <= criteria.max_top1_regret
                ),
                "baseline_metrics": {baseline.label: arm_metrics[baseline.label] for baseline in protocol.baselines},
                "baseline_comparisons": baseline_comparisons,
            }
        )
    return rows


def assess_feasibility(
    trials: Sequence[TrialScore],
    suite: BatchSuite,
    contract: ProductionContract,
    protocol: FeasibilityProtocol,
) -> dict:
    """Assess a proxy pilot against preregistered reference and baseline gates.

    The return value contains only finite JSON numbers or ``None``.  It is a
    descriptive pilot decision based on batch-level scores and training-seed
    variation; item-level benchmark uncertainty is intentionally not inferred.
    """

    selected, ignored_count = _validate_and_select_trials(list(trials), suite, contract, protocol)
    utilities = aggregate_trials(selected, contract=contract)

    arms = [protocol.target, *protocol.baselines]
    reports = {
        arm.label: evaluate_proxy(
            utilities,
            suite=suite,
            proxy_method=arm.method,
            proxy_budget_id=arm.budget_id,
            reference_method=protocol.reference.method,
            reference_budget_id=protocol.reference.budget_id,
            top_ks=(1,),
            indifference_margin=protocol.indifference_margin,
        )
        for arm in arms
    }
    target_report = reports[protocol.target.label]
    target_gates = _absolute_gates(target_report, protocol)
    comparisons = [
        _comparison(target_report, reports[baseline.label], baseline, protocol) for baseline in protocol.baselines
    ]
    reference = _reference_identifiability(selected, utilities, contract, protocol)
    leave_one_out = _leave_one_seed_out(selected, contract, protocol)
    lineage_sensitivity = _lineage_sensitivity(utilities, suite, protocol)
    exposure_audit = _training_exposure_audit(selected, protocol)

    warnings = {
        "This is a descriptive pilot; it does not support confirmatory inference.",
        "No item-level benchmark outcomes were supplied; uncertainty omits item-sampling uncertainty.",
    }
    for report in reports.values():
        warnings.update(report["warnings"])
    lineage_count = len({batch.lineage_group for batch in suite.batches})
    if lineage_count < protocol.minimum_independent_lineage_count_for_warning:
        warnings.add(
            f"Independent lineage count is low ({lineage_count} observed; "
            f"{protocol.minimum_independent_lineage_count_for_warning} preregistered); "
            "batches sharing lineage do not provide independent evidence."
        )
    if ignored_count:
        warnings.add(f"Ignored {ignored_count} trial rows whose arm identity was outside the frozen protocol.")
    stochastic_methods = {"random", "stratified_random"}
    missing_stochastic_seed = [
        arm.label
        for arm in [protocol.target, *protocol.baselines]
        if arm.method in stochastic_methods and arm.selection_seed is None
    ]
    if missing_stochastic_seed:
        warnings.add(
            "Stochastic proxy arms are missing preregistered selection seeds: "
            + ", ".join(sorted(missing_stochastic_seed))
        )
    stochastic_draw_count = {
        method: sum(arm.method == method for arm in protocol.baselines)
        for method in stochastic_methods
        if any(arm.method == method for arm in protocol.baselines)
    }
    if any(count < 2 for count in stochastic_draw_count.values()):
        warnings.add(
            "Random-selector baselines use one preregistered selection draw per method; "
            "the pilot does not estimate selector-sampling variance."
        )
    if exposure_audit["same_proxy_generator_revision_and_recipe"] is False:
        warnings.add(
            "Proxy arms do not share one generator revision and generation recipe; "
            "method contrasts are confounded by generation differences."
        )
    if exposure_audit["generator_provenance_available"] is False:
        warnings.add(
            "Generator provenance is unavailable in lightweight TrialScore rows; "
            "use ExperimentTrialScore for a formal same-generator claim."
        )
    if not all(
        row["all_batch_seed_cells_exactly_matched"]
        for row in exposure_audit["target_vs_baseline_realized_exposure"].values()
    ):
        warnings.add(
            "Proxy arms share preregistered token caps but realized effective training tokens differ in some "
            "batch-seed cells; interpret the exposure audit alongside method effects."
        )
    if lineage_sensitivity["independent_lineage_count"] < 8:
        warnings.add(
            "Lineage-cluster bootstrap intervals use fewer than eight independent lineages and are descriptive "
            "sensitivity intervals, not confirmatory confidence intervals."
        )

    absolute_passed = all(gate["passed"] for gate in target_gates.values())
    comparison_passed = all(comparison["passed"] for comparison in comparisons)
    all_loso_clearly_fail = bool(leave_one_out) and all(row["target_clearly_failed_gates"] for row in leave_one_out)
    loso_stable = bool(leave_one_out) and all(
        row["target_absolute_gates_passed"]
        and all(
            comparison["spearman_passed"] and comparison["top1_regret_passed"]
            for comparison in row["baseline_comparisons"]
        )
        for row in leave_one_out
    )
    consistently_dominating_baselines = [
        baseline.label
        for baseline, point_comparison in zip(protocol.baselines, comparisons)
        if point_comparison["baseline_dominates"]
        and all(
            next(
                comparison
                for comparison in row["baseline_comparisons"]
                if comparison["baseline_label"] == baseline.label
            )["baseline_dominates"]
            for row in leave_one_out
        )
    ]

    # Decision priority is intentional: an unidentifiable reference cannot
    # establish either success or failure, even when the proxy looks poor.
    if not reference["identified"]:
        verdict = "inconclusive"
        reason_codes = ["REFERENCE_NOT_IDENTIFIABLE", *reference["failure_reason_codes"]]
    elif (
        absolute_passed
        and comparison_passed
        and (loso_stable or not protocol.target_decision.require_leave_one_seed_out_stability)
    ):
        verdict = "feasible"
        reason_codes = ["ALL_PREREGISTERED_GATES_PASSED"]
    elif (not absolute_passed and all_loso_clearly_fail) or consistently_dominating_baselines:
        verdict = "infeasible"
        reason_codes = []
        if not absolute_passed and all_loso_clearly_fail:
            reason_codes.append("TARGET_FAILED_A_GATE_IN_EVERY_LEAVE_ONE_SEED_OUT_RUN")
        if consistently_dominating_baselines:
            reason_codes.append("TARGET_CONSISTENTLY_BASELINE_DOMINATED")
    else:
        verdict = "inconclusive"
        reason_codes = ["TARGET_RESULT_NOT_DECISIVE_ACROSS_TRAINING_SEEDS"]

    return {
        "schema_version": "pilot-feasibility-report-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": canonical_digest(protocol),
        "inference_scope": protocol.inference_scope,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "reference_identifiability": reference,
        "target": {
            "label": protocol.target.label,
            "selection_seed": protocol.target.selection_seed,
            "report": target_report,
            "absolute_gates": target_gates,
            "absolute_gates_passed": absolute_passed,
        },
        "baselines": [
            {
                "label": baseline.label,
                "selection_seed": baseline.selection_seed,
                "report": reports[baseline.label],
            }
            for baseline in protocol.baselines
        ],
        "baseline_comparisons": comparisons,
        "training_exposure_audit": exposure_audit,
        "lineage_sensitivity": lineage_sensitivity,
        "all_baseline_comparisons_passed": comparison_passed,
        "leave_one_training_seed_out": leave_one_out,
        "leave_one_training_seed_out_stable": loso_stable,
        "consistently_dominating_baseline_labels": consistently_dominating_baselines,
        "batch_count": len(suite.batches),
        "independent_lineage_count": lineage_count,
        "ignored_trial_count": ignored_count,
        "warnings": sorted(warnings),
    }
