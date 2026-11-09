"""Paired utility aggregation and decision-oriented valuation metrics."""

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

from scipy.stats import t as student_t

from ai_ready.valuation.contracts import contract_digest, suite_digest
from ai_ready.valuation.schema import BatchSuite, BatchUtility, CombinationUtility, ProductionContract, TrialScore


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based ascending ranks, assigning the mean rank to ties."""

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
    """Return ranks for a partial order with a practical-equivalence margin.

    A value receives credit only for values it exceeds by more than ``margin``;
    values inside the margin share half-credit.  At margin zero this is exactly
    the usual average rank.  Unlike binning, the calculation has no arbitrary
    origin, and unlike transitive clustering, a chain of small gaps cannot hide
    a clearly large end-to-end difference.
    """

    if margin <= 0:
        return _average_ranks(values)
    ranks = []
    for value in values:
        definitely_below = sum(value - other > margin for other in values)
        indistinguishable = sum(abs(value - other) <= margin for other in values)
        ranks.append(float(definitely_below + (indistinguishable + 1) / 2.0))
    return ranks


def _kendall_tau_b(
    left: Sequence[float],
    right: Sequence[float],
    *,
    right_indifference_margin: float = 0.0,
) -> float | None:
    concordant = discordant = tied_left = tied_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
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


def aggregate_trials(trials: Iterable[TrialScore], *, contract: ProductionContract) -> list[BatchUtility]:
    """Aggregate paired seed deltas; never select a batch's best seed.

    Every observation carries the score of the baseline run with the same seed.
    Pairing before averaging removes part of the seed-specific model/evaluation
    noise.  Treating seeds as independent data batches would inflate evidence.
    The production contract supplies the metric direction and expected artifact
    identities: utility is always normalized so that a positive value means the
    candidate improved.
    """

    rows = list(trials)
    if not rows:
        raise ValueError("At least one trial is required")
    expected_contract_sha = contract_digest(contract)
    expected_benchmark_sha = contract.benchmark.test_manifest_sha256
    for row in rows:
        if row.contract_sha256 != expected_contract_sha:
            raise ValueError(f"Trial {row.run_id} does not use the supplied production contract")
        if row.benchmark_sha256 != expected_benchmark_sha:
            raise ValueError(f"Trial {row.run_id} does not use the supplied locked benchmark manifest")
    identities = {(row.suite_sha256, row.contract_sha256, row.benchmark_sha256) for row in rows}
    if len(identities) != 1:
        raise ValueError("All trials in one comparison must use one suite, contract and benchmark")

    baseline_by_seed: dict[int, float] = {}
    batch_hashes: dict[str, str] = {}
    seen_runs: set[str] = set()
    seen_keys: set[tuple[str, str, str, str, int]] = set()
    for row in rows:
        if row.run_id in seen_runs:
            raise ValueError(f"Duplicate run_id: {row.run_id}")
        seen_runs.add(row.run_id)
        previous_batch_hash = batch_hashes.setdefault(row.batch_id, row.batch_content_sha256)
        if previous_batch_hash != row.batch_content_sha256:
            raise ValueError(f"Batch {row.batch_id} has inconsistent content hashes across trials")
        key = (row.batch_id, row.arm, row.method, row.budget_id, row.seed)
        if key in seen_keys:
            raise ValueError(f"Duplicate trial for batch/method/budget/seed: {key}")
        seen_keys.add(key)
        previous = baseline_by_seed.setdefault(row.seed, row.baseline_score)
        if not math.isclose(previous, row.baseline_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Seed {row.seed} has inconsistent paired baseline scores")

    grouped: dict[tuple[str, str, str, str], list[TrialScore]] = defaultdict(list)
    for row in rows:
        grouped[(row.batch_id, row.arm, row.method, row.budget_id)].append(row)

    suite_sha = rows[0].suite_sha256
    benchmark_sha, contract_sha = rows[0].benchmark_sha256, rows[0].contract_sha256
    aggregated = []
    for (batch_id, arm, method, budget_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row.seed)
        actual_seeds = {row.seed for row in group}
        expected_seeds = set(contract.training.seeds)
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"Batch {batch_id} in {(arm, method, budget_id)} must use seeds "
                f"{sorted(expected_seeds)}, got {sorted(actual_seeds)}"
            )
        direction = 1.0 if contract.benchmark.higher_is_better else -1.0
        utilities = [direction * row.utility for row in group]
        mean_utility = _mean(utilities)
        std_utility = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
        # Reference runs normally use only three to five seeds.  A normal 1.96
        # multiplier materially understates uncertainty at those sample sizes,
        # so use the two-sided Student-t critical value for the observed df.
        critical = float(student_t.ppf(0.975, df=len(utilities) - 1)) if len(utilities) > 1 else 0.0
        half_width = critical * std_utility / math.sqrt(len(utilities)) if len(utilities) > 1 else 0.0
        costs = [row.cost for row in group if row.cost is not None]
        if costs and len(costs) != len(group):
            raise ValueError(f"Batch {batch_id} has only partially recorded trial costs")
        aggregated.append(
            BatchUtility(
                suite_sha256=suite_sha,
                batch_id=batch_id,
                batch_content_sha256=batch_hashes[batch_id],
                arm=arm,
                method=method,
                budget_id=budget_id,
                benchmark_sha256=benchmark_sha,
                contract_sha256=contract_sha,
                mean_utility=mean_utility,
                std_utility=std_utility,
                ci95_low=mean_utility - half_width,
                ci95_high=mean_utility + half_width,
                seeds=[row.seed for row in group],
                mean_score=_mean([row.benchmark_score for row in group]),
                mean_baseline_score=_mean([row.baseline_score for row in group]),
                mean_effective_train_tokens=_mean([float(row.effective_train_tokens) for row in group]),
                mean_cost=_mean(costs) if costs else None,
            )
        )
    return aggregated


def validate_trial_completeness(
    trials: Sequence[TrialScore],
    suite: BatchSuite,
    contract: ProductionContract,
    *,
    proxy_method: str,
    proxy_budget_id: str,
    reference_method: str,
    reference_budget_id: str,
) -> None:
    """Reject partial/cherry-picked comparisons before calculating rankings."""

    if suite.benchmark_id != contract.benchmark.benchmark_id:
        raise ValueError("Batch suite and production contract use different benchmark IDs")
    if suite.capability_blueprint_sha256 != contract.benchmark.capability_blueprint_sha256:
        raise ValueError("Batch suite and production contract use different capability blueprints")
    if reference_method != contract.generation.reference.generator:
        raise ValueError("Reference method must match the frozen reference generator")
    expected_batches = {batch.batch_id for batch in suite.batches}
    if any(batch.content_sha256 is None for batch in suite.batches):
        raise ValueError("Every batch must have a frozen content_sha256 before trial comparison")
    expected_batch_hashes = {batch.batch_id: batch.content_sha256 for batch in suite.batches}
    expected_seeds = set(contract.training.seeds)
    expected_suite = suite_digest(suite)
    expected_contract = contract_digest(contract)
    for row in trials:
        if row.suite_sha256 != expected_suite:
            raise ValueError(f"Trial {row.run_id} does not use the frozen batch suite")
        if row.contract_sha256 != expected_contract:
            raise ValueError(f"Trial {row.run_id} does not use the frozen production contract")
        if row.benchmark_sha256 != contract.benchmark.test_manifest_sha256:
            raise ValueError(f"Trial {row.run_id} does not use the locked benchmark manifest")
        if row.batch_id not in expected_batches:
            raise ValueError(f"Trial {row.run_id} references unregistered batch {row.batch_id}")
        if row.batch_content_sha256 != expected_batch_hashes[row.batch_id]:
            raise ValueError(f"Trial {row.run_id} does not use the frozen content of batch {row.batch_id}")
        if row.effective_train_tokens > contract.budget.training_token_cap:
            # The same global ceiling applies to all arms. The proxy may use a
            # lower preregistered budget, but can never exceed the reference cap.
            raise ValueError(f"Trial {row.run_id} exceeds the frozen training-token cap")

    requested = {
        ("proxy", proxy_method, proxy_budget_id): defaultdict(set),
        ("reference", reference_method, reference_budget_id): defaultdict(set),
    }
    for row in trials:
        key = (row.arm, row.method, row.budget_id)
        if key in requested:
            requested[key][row.batch_id].add(row.seed)
    for key, by_batch in requested.items():
        if set(by_batch) != expected_batches:
            missing = sorted(expected_batches - set(by_batch))
            extra = sorted(set(by_batch) - expected_batches)
            raise ValueError(f"Incomplete {key} observations; missing={missing}, extra={extra}")
        for batch_id, seeds in by_batch.items():
            if seeds != expected_seeds:
                raise ValueError(
                    f"Batch {batch_id} in {key} must use paired seeds {sorted(expected_seeds)}, got {sorted(seeds)}"
                )


def _unique_by_batch(rows: Sequence[BatchUtility], description: str) -> dict[str, BatchUtility]:
    result = {}
    for row in rows:
        if row.batch_id in result:
            raise ValueError(f"{description} contains duplicate batch utility: {row.batch_id}")
        result[row.batch_id] = row
    return result


def _sorted_batches(values: dict[str, float]) -> list[str]:
    # Batch ID is the pre-registered deterministic tie break; it prevents an
    # evaluator from resolving tied proxy scores after seeing reference utility.
    return sorted(values, key=lambda batch_id: (-values[batch_id], batch_id))


def _linear_ndcg(predicted_order: Sequence[str], reference_values: dict[str, float], k: int) -> float:
    minimum = min(reference_values.values())
    relevance = {batch_id: value - minimum for batch_id, value in reference_values.items()}
    ideal_order = _sorted_batches(reference_values)

    def dcg(order: Sequence[str]) -> float:
        return sum(relevance[batch_id] / math.log2(rank + 2) for rank, batch_id in enumerate(order[:k]))

    ideal = dcg(ideal_order)
    return 1.0 if ideal == 0 else float(dcg(predicted_order) / ideal)


def _label(value: float, margin: float) -> str:
    if value > margin:
        return "positive"
    if value < -margin:
        return "negative"
    return "indistinguishable"


def _combination_regret(
    predicted_order: Sequence[str],
    combinations: Sequence[CombinationUtility],
    k: int,
) -> tuple[float | None, str | None]:
    candidates = {tuple(row.batch_ids): row.mean_utility for row in combinations if len(row.batch_ids) == k}
    if not candidates:
        return None, f"No observed joint-training utilities were supplied for k={k}"
    expected_count = math.comb(len(predicted_order), k)
    if len(candidates) != expected_count:
        return (
            None,
            f"Joint-training coverage for k={k} is incomplete: observed {len(candidates)} of "
            f"{expected_count} registered combinations",
        )
    selected = tuple(sorted(predicted_order[:k]))
    if selected not in candidates:
        return None, f"The proxy-selected k={k} combination was not actually trained"
    # 单批效用不能相加：联合训练可能存在重复、冲突或协同。这里只比较真实观测到的组合训练效用 Y(T)。
    return float(max(candidates.values()) - candidates[selected]), None


def _validate_combination_utilities(
    combinations: Sequence[CombinationUtility],
    reference: dict[str, BatchUtility],
    *,
    reference_method: str,
    reference_budget_id: str,
) -> None:
    """Bind observed joint-training utilities to the same reference experiment."""

    if not combinations:
        return
    reference_seed_sets = {tuple(row.seeds) for row in reference.values()}
    if len(reference_seed_sets) != 1:
        raise ValueError("Reference batch utilities use inconsistent seed sets")
    expected_seeds = next(iter(reference_seed_sets))
    first_reference = next(iter(reference.values()))
    seen: set[tuple[str, ...]] = set()
    for combination in combinations:
        key = tuple(combination.batch_ids)
        if key in seen:
            raise ValueError(f"Duplicate observed combination utility: {list(key)}")
        seen.add(key)
        if unknown := set(combination.batch_ids) - set(reference):
            raise ValueError(f"Combination utility references unknown batches: {sorted(unknown)}")
        if combination.reference_method != reference_method or combination.budget_id != reference_budget_id:
            raise ValueError("Combination utility uses a different reference method or budget")
        if (
            combination.suite_sha256 != first_reference.suite_sha256
            or combination.contract_sha256 != first_reference.contract_sha256
            or combination.benchmark_sha256 != first_reference.benchmark_sha256
        ):
            raise ValueError("Combination utility uses a different suite, contract or benchmark")
        if tuple(combination.seeds) != expected_seeds:
            raise ValueError("Combination utility must use the same paired seeds as the reference batches")
        for batch_id in combination.batch_ids:
            if combination.batch_content_sha256[batch_id] != reference[batch_id].batch_content_sha256:
                raise ValueError(f"Combination utility does not use the frozen content of batch {batch_id}")


def evaluate_proxy(
    utilities: Sequence[BatchUtility],
    *,
    suite: BatchSuite,
    proxy_method: str,
    proxy_budget_id: str,
    reference_method: str,
    reference_budget_id: str,
    top_ks: Sequence[int] = (1, 3),
    indifference_margin: float = 0.0,
    combination_utilities: Sequence[CombinationUtility] = (),
) -> dict:
    if not math.isfinite(indifference_margin) or indifference_margin < 0:
        raise ValueError("indifference_margin must be a finite, non-negative number")
    proxy_rows = [
        row
        for row in utilities
        if row.arm == "proxy" and row.method == proxy_method and row.budget_id == proxy_budget_id
    ]
    reference_rows = [
        row
        for row in utilities
        if row.arm == "reference" and row.method == reference_method and row.budget_id == reference_budget_id
    ]
    proxy = _unique_by_batch(proxy_rows, "Proxy selection")
    reference = _unique_by_batch(reference_rows, "Reference selection")
    if not proxy or not reference:
        raise ValueError("Both proxy and reference utilities are required")
    if set(proxy) != set(reference):
        missing_proxy = sorted(set(reference) - set(proxy))
        missing_reference = sorted(set(proxy) - set(reference))
        raise ValueError(
            f"Proxy and reference must cover the same registered batches; "
            f"missing_proxy={missing_proxy}, missing_reference={missing_reference}"
        )
    expected_batches = {batch.batch_id for batch in suite.batches}
    if set(proxy) != expected_batches:
        missing = sorted(expected_batches - set(proxy))
        extra = sorted(set(proxy) - expected_batches)
        raise ValueError(f"Utilities must cover the registered batch suite; missing={missing}, extra={extra}")
    if any(batch.content_sha256 is None for batch in suite.batches):
        raise ValueError("Every batch must have a frozen content_sha256 before valuation")
    suite_hashes = {batch.batch_id: batch.content_sha256 for batch in suite.batches}
    identities = {
        (row.suite_sha256, row.contract_sha256, row.benchmark_sha256) for row in [*proxy.values(), *reference.values()]
    }
    if len(identities) != 1:
        raise ValueError("Proxy and reference utilities use different suites, contracts or benchmarks")
    expected_suite_sha = suite_digest(suite)
    if next(iter(identities))[0] != expected_suite_sha:
        raise ValueError("Utilities do not use the supplied frozen batch suite")
    for batch_id in proxy:
        if proxy[batch_id].batch_content_sha256 != reference[batch_id].batch_content_sha256:
            raise ValueError(f"Proxy and reference use different content for batch {batch_id}")
        if proxy[batch_id].batch_content_sha256 != suite_hashes[batch_id]:
            raise ValueError(f"Utilities do not use the frozen content of batch {batch_id}")
        if proxy[batch_id].seeds != reference[batch_id].seeds:
            raise ValueError(f"Proxy and reference use different paired seeds for batch {batch_id}")
    _validate_combination_utilities(
        combination_utilities,
        reference,
        reference_method=reference_method,
        reference_budget_id=reference_budget_id,
    )

    batch_ids = sorted(proxy)
    proxy_values = [proxy[batch_id].mean_utility for batch_id in batch_ids]
    reference_values_list = [reference[batch_id].mean_utility for batch_id in batch_ids]
    proxy_values_by_batch = dict(zip(batch_ids, proxy_values))
    reference_values = dict(zip(batch_ids, reference_values_list))
    predicted_order = _sorted_batches(proxy_values_by_batch)
    reference_order = _sorted_batches(reference_values)
    pairwise_scores = []
    for first in range(len(batch_ids)):
        for second in range(first + 1, len(batch_ids)):
            reference_delta = reference_values_list[first] - reference_values_list[second]
            reference_sign = (
                0 if abs(reference_delta) <= indifference_margin else (reference_delta > 0) - (reference_delta < 0)
            )
            proxy_sign = (proxy_values[first] > proxy_values[second]) - (proxy_values[first] < proxy_values[second])
            if reference_sign == 0:
                continue
            pairwise_scores.append(0.5 if proxy_sign == 0 else float(proxy_sign == reference_sign))

    if not top_ks:
        raise ValueError("At least one top-k value is required")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > len(batch_ids) for k in top_ks):
        raise ValueError(f"Each top-k value must be an integer between 1 and {len(batch_ids)}")
    requested_top_ks = sorted(set(top_ks))
    warnings = []
    lineage_group_count = len({batch.lineage_group for batch in suite.batches})
    if lineage_group_count < len(batch_ids):
        warnings.append(
            f"The {len(batch_ids)} batches contain only {lineage_group_count} lineage groups; "
            "correlation values are descriptive point estimates, not independent-sample inference"
        )
    if suite.intended_claim == "feasibility_and_method_screening":
        warnings.append("This suite is registered for feasibility and method screening, not confirmatory claims")
    top_k = {}
    combination_regret = {}
    for k in requested_top_ks:
        predicted = predicted_order[:k]
        actual = reference_order[:k]
        overlap = len(set(predicted) & set(actual))
        top_k[str(k)] = {
            "predicted_batch_ids": predicted,
            "reference_batch_ids": actual,
            "overlap": overlap,
            "precision": overlap / k,
            "recall": overlap / k,
            "linear_ndcg": _linear_ndcg(predicted_order, reference_values, k),
        }
        if k == 1:
            combination_regret[str(k)] = max(reference_values.values()) - reference_values[predicted[0]]
        else:
            regret, warning = _combination_regret(predicted_order, combination_utilities, k)
            combination_regret[str(k)] = regret
            if warning:
                warnings.append(warning)

    mean_absolute_error = _mean(
        [
            abs(proxy_value - reference_value)
            for proxy_value, reference_value in zip(proxy_values, reference_values_list)
        ]
    )
    root_mean_squared_error = math.sqrt(
        _mean(
            [
                (proxy_value - reference_value) ** 2
                for proxy_value, reference_value in zip(proxy_values, reference_values_list)
            ]
        )
    )
    raw_spearman = _pearson(_average_ranks(proxy_values), _average_ranks(reference_values_list))
    raw_kendall = _kendall_tau_b(proxy_values, reference_values_list)
    return {
        "schema_version": "valuation-report-v1",
        "proxy": {"method": proxy_method, "budget_id": proxy_budget_id},
        "reference": {"method": reference_method, "budget_id": reference_budget_id},
        "batch_count": len(batch_ids),
        "lineage_group_count": lineage_group_count,
        "intended_claim": suite.intended_claim,
        "inference_scope": "descriptive_point_estimates",
        "batch_ids": batch_ids,
        # These are the decision metrics.  The reference rank treats utility
        # gaps inside the preregistered practical-equivalence margin as ties.
        "spearman": _pearson(
            _average_ranks(proxy_values),
            _margin_aware_ranks(reference_values_list, indifference_margin),
        ),
        "kendall_tau_b": _kendall_tau_b(
            proxy_values,
            reference_values_list,
            right_indifference_margin=indifference_margin,
        ),
        "raw_spearman": raw_spearman,
        "raw_kendall_tau_b": raw_kendall,
        "ranking_indifference_margin": indifference_margin,
        "pearson": _pearson(proxy_values, reference_values_list),
        "pairwise_order_accuracy": _mean(pairwise_scores) if pairwise_scores else None,
        "direction_agreement": _mean(
            [
                float(_label(proxy_value, indifference_margin) == _label(reference_value, indifference_margin))
                for proxy_value, reference_value in zip(proxy_values, reference_values_list)
            ]
        ),
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "top_k": top_k,
        "regret": combination_regret,
        "proxy_order": predicted_order,
        "reference_order": reference_order,
        "warnings": sorted(set(warnings)),
    }
