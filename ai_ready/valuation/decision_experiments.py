"""Auditable analyses for four decision-oriented valuation experiments.

The module deliberately consumes frozen, batch-level artifacts.  It never
fits a baseline to the expensive reference utilities and it never silently
drops failed cost events.  This keeps the added experiments aligned with the
paper's central claim: low-cost signals are useful only when they preserve the
decisions induced by complete processing and training.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ai_ready.valuation.contracts import suite_digest
from ai_ready.valuation.schema import BatchSuite, StrictModel


def _sha256(value: str | None, field_name: str, *, optional: bool = False) -> str | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{field_name} is required")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean from an empty sequence")
    return float(sum(values) / len(values))


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _margin_aware_ranks(values: list[float], margin: float) -> list[float]:
    ranks = []
    for value in values:
        definitely_below = sum(value - other > margin for other in values)
        indistinguishable = sum(abs(value - other) <= margin for other in values)
        ranks.append(float(definitely_below + (indistinguishable + 1) / 2.0))
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_delta) * sum(value * value for value in right_delta))
    if denominator == 0:
        return None
    return float(sum(a * b for a, b in zip(left_delta, right_delta)) / denominator)


def _kendall_tau_b(
    left: list[float],
    right: list[float],
    *,
    left_margin: float = 0.0,
    right_margin: float = 0.0,
) -> float | None:
    concordant = discordant = tied_left = tied_right = 0
    for first, second in combinations(range(len(left)), 2):
        left_delta = left[first] - left[second]
        right_delta = right[first] - right[second]
        left_sign = 0 if abs(left_delta) <= left_margin else (1 if left_delta > 0 else -1)
        right_sign = 0 if abs(right_delta) <= right_margin else (1 if right_delta > 0 else -1)
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


def _ordered(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda batch_id: (-values[batch_id], batch_id))


def _linear_ndcg(predicted_order: list[str], reference: dict[str, float], k: int) -> float:
    minimum = min(reference.values())
    relevance = {batch_id: value - minimum for batch_id, value in reference.items()}

    def dcg(order: list[str]) -> float:
        return sum(relevance[batch_id] / math.log2(rank + 2) for rank, batch_id in enumerate(order[:k]))

    ideal = dcg(_ordered(reference))
    return 1.0 if ideal == 0 else float(dcg(predicted_order) / ideal)


def evaluate_rankings(
    predicted: dict[str, float],
    reference: dict[str, float],
    *,
    top_ks: list[int],
    indifference_margin: float,
) -> dict:
    """Compare two batch-level value vectors using decision metrics."""

    if set(predicted) != set(reference):
        raise ValueError("Predicted and reference values must cover exactly the same batches")
    if len(predicted) < 2:
        raise ValueError("Ranking evaluation requires at least two batches")
    if not math.isfinite(indifference_margin) or indifference_margin < 0:
        raise ValueError("indifference_margin must be finite and non-negative")
    if not top_ks or any(isinstance(k, bool) or k < 1 or k > len(predicted) for k in top_ks):
        raise ValueError(f"Each top-k value must be between 1 and {len(predicted)}")

    batch_ids = sorted(predicted)
    predicted_values = [predicted[batch_id] for batch_id in batch_ids]
    reference_values = [reference[batch_id] for batch_id in batch_ids]
    predicted_order = _ordered(predicted)
    reference_order = _ordered(reference)
    pairwise_scores = []
    for first, second in combinations(range(len(batch_ids)), 2):
        reference_delta = reference_values[first] - reference_values[second]
        if abs(reference_delta) <= indifference_margin:
            continue
        reference_sign = 1 if reference_delta > 0 else -1
        predicted_delta = predicted_values[first] - predicted_values[second]
        predicted_sign = 1 if predicted_delta > 0 else -1 if predicted_delta < 0 else 0
        pairwise_scores.append(0.5 if predicted_sign == 0 else float(predicted_sign == reference_sign))

    top_k = {}
    regret = {}
    for k in sorted(set(top_ks)):
        predicted_ids = predicted_order[:k]
        reference_ids = reference_order[:k]
        overlap = len(set(predicted_ids) & set(reference_ids))
        top_k[str(k)] = {
            "predicted_batch_ids": predicted_ids,
            "reference_batch_ids": reference_ids,
            "overlap": overlap,
            "precision": overlap / k,
            "recall": overlap / k,
            "linear_ndcg": _linear_ndcg(predicted_order, reference, k),
        }
        # Additive multi-batch utility is not assumed.  Only top-1 regret can
        # be identified from individual-batch reference utilities.
        regret[str(k)] = max(reference.values()) - reference[predicted_ids[0]] if k == 1 else None

    return {
        "batch_count": len(batch_ids),
        "batch_ids": batch_ids,
        "spearman": _pearson(
            _average_ranks(predicted_values),
            _margin_aware_ranks(reference_values, indifference_margin),
        ),
        "kendall_tau_b": _kendall_tau_b(
            predicted_values,
            reference_values,
            right_margin=indifference_margin,
        ),
        "raw_spearman": _pearson(_average_ranks(predicted_values), _average_ranks(reference_values)),
        "raw_kendall_tau_b": _kendall_tau_b(predicted_values, reference_values),
        "pairwise_order_accuracy": _mean(pairwise_scores) if pairwise_scores else None,
        "ranking_indifference_margin": indifference_margin,
        "top_k": top_k,
        "regret": regret,
        "predicted_order": predicted_order,
        "reference_order": reference_order,
    }


class StaticBatchFeatures(StrictModel):
    schema_version: Literal["static-batch-features-v1"] = "static-batch-features-v1"
    suite_sha256: str
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    lineage_group: str = Field(min_length=1)
    document_count: int = Field(ge=1)
    token_count: int = Field(ge=1)
    unique_token_count: int = Field(ge=1)
    semantic_unit_count: int = Field(ge=0)
    mean_quality: float
    information_density: float
    benchmark_relevance: float
    redundancy_rate: float = Field(ge=0.0, le=1.0)
    llm_judge_score: float | None = None

    @field_validator("suite_sha256", "batch_content_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _sha256(value, info.field_name) or ""

    @model_validator(mode="after")
    def validate_counts(self):
        if self.unique_token_count > self.token_count:
            raise ValueError("unique_token_count cannot exceed token_count")
        return self


class ReferenceBatchValue(StrictModel):
    schema_version: Literal["reference-batch-value-v1"] = "reference-batch-value-v1"
    task_id: str = Field(min_length=1)
    suite_sha256: str
    benchmark_sha256: str
    base_model_id: str = Field(min_length=1)
    training_protocol_sha256: str
    seeds: list[int] = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    lineage_group: str = Field(min_length=1)
    utility: float

    @field_validator(
        "suite_sha256",
        "benchmark_sha256",
        "training_protocol_sha256",
        "batch_content_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _sha256(value, info.field_name) or ""

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("seeds must be unique")
        return sorted(value)


class TrainableProxyBatchValue(StrictModel):
    """Frozen batch utility measured by low-budget proxy training."""

    schema_version: Literal["trainable-proxy-batch-value-v1"] = "trainable-proxy-batch-value-v1"
    task_id: str = Field(min_length=1)
    suite_sha256: str
    benchmark_sha256: str
    base_model_id: str = Field(min_length=1)
    training_protocol_sha256: str
    seeds: list[int] = Field(min_length=1)
    proxy_method: str = Field(min_length=1)
    budget_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    lineage_group: str = Field(min_length=1)
    utility: float

    @field_validator(
        "suite_sha256",
        "benchmark_sha256",
        "training_protocol_sha256",
        "batch_content_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _sha256(value, info.field_name) or ""

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("seeds must be unique")
        return sorted(value)


_STATIC_BASELINES = {
    "document_count",
    "token_count",
    "unique_token_count",
    "semantic_unit_count",
    "mean_quality",
    "information_density",
    "benchmark_relevance",
    "low_redundancy",
    "llm_judge_score",
    "composite",
}
_COMPOSITE_FEATURES = {
    "unique_token_count",
    "mean_quality",
    "information_density",
    "benchmark_relevance",
    "low_redundancy",
    "llm_judge_score",
}


class StaticBaselineConfig(StrictModel):
    schema_version: Literal["static-baseline-config-v1", "static-baseline-config-v2"] = "static-baseline-config-v2"
    enabled_baselines: list[str] = Field(
        default_factory=lambda: [
            "token_count",
            "unique_token_count",
            "information_density",
            "benchmark_relevance",
            "low_redundancy",
            "composite",
        ]
    )
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "unique_token_count": 1.0,
            "mean_quality": 1.0,
            "information_density": 1.0,
            "benchmark_relevance": 1.0,
            "low_redundancy": 1.0,
        }
    )
    # Named, frozen composites make the reviewer-facing comparison explicit:
    # an intrinsic score may use only task-independent corpus properties,
    # whereas a task-aware score may additionally use benchmark alignment.
    # The analyzer never fits either set of weights to reference utility.
    named_composites: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_ks: list[int] = Field(default_factory=lambda: [1, 3], min_length=1)
    indifference_margin: float = Field(default=0.0, ge=0.0)
    hard_pair_baseline: str | None = None
    hard_pair_count: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def validate_baselines(self):
        if len(self.enabled_baselines) != len(set(self.enabled_baselines)):
            raise ValueError("enabled_baselines must be unique")
        if collisions := set(self.named_composites) & _STATIC_BASELINES:
            raise ValueError(f"Named composites collide with built-in baselines: {sorted(collisions)}")
        if unknown := set(self.enabled_baselines) - _STATIC_BASELINES - set(self.named_composites):
            raise ValueError(f"Unknown static baselines: {sorted(unknown)}")
        if unknown := set(self.composite_weights) - _COMPOSITE_FEATURES:
            raise ValueError(f"Unknown composite features: {sorted(unknown)}")
        if not self.composite_weights or any(weight < 0 for weight in self.composite_weights.values()):
            raise ValueError("composite_weights must be non-negative and non-empty")
        if sum(self.composite_weights.values()) <= 0:
            raise ValueError("At least one composite weight must be positive")
        for name, weights in self.named_composites.items():
            if not name:
                raise ValueError("Named composite names must be non-empty")
            if unknown := set(weights) - _COMPOSITE_FEATURES:
                raise ValueError(f"Unknown features in named composite {name}: {sorted(unknown)}")
            if not weights or any(weight < 0 for weight in weights.values()):
                raise ValueError(f"Named composite {name} must have non-negative, non-empty weights")
            if sum(weights.values()) <= 0:
                raise ValueError(f"Named composite {name} must have at least one positive weight")
        if self.hard_pair_baseline is not None and self.hard_pair_baseline not in self.enabled_baselines:
            raise ValueError("hard_pair_baseline must be one of the enabled baselines")
        return self


def _validate_suite_rows(
    suite: BatchSuite,
    rows: list[StaticBatchFeatures] | list[ReferenceBatchValue],
    *,
    description: str,
) -> None:
    expected_suite_sha = suite_digest(suite)
    registered = {batch.batch_id: batch for batch in suite.batches}
    if any(batch.content_sha256 is None for batch in suite.batches):
        raise ValueError("The batch suite must be frozen before analysis")
    if len(rows) != len(registered) or {row.batch_id for row in rows} != set(registered):
        raise ValueError(f"{description} must contain exactly one row for every registered batch")
    if len({row.batch_id for row in rows}) != len(rows):
        raise ValueError(f"{description} contains duplicate batch IDs")
    for row in rows:
        batch = registered[row.batch_id]
        if row.suite_sha256 != expected_suite_sha:
            raise ValueError(f"{description} row {row.batch_id} uses a different batch suite")
        if row.batch_content_sha256 != batch.content_sha256:
            raise ValueError(f"{description} row {row.batch_id} uses different batch content")
        if row.lineage_group != batch.lineage_group:
            raise ValueError(f"{description} row {row.batch_id} uses a different lineage group")


def _feature_value(row: StaticBatchFeatures, name: str) -> float:
    if name == "low_redundancy":
        return -row.redundancy_rate
    value = getattr(row, name)
    if value is None:
        raise ValueError(f"Baseline {name} is missing for batch {row.batch_id}")
    return float(value)


def _minmax(values: dict[str, float]) -> dict[str, float]:
    minimum, maximum = min(values.values()), max(values.values())
    if minimum == maximum:
        return {batch_id: 0.5 for batch_id in values}
    return {batch_id: (value - minimum) / (maximum - minimum) for batch_id, value in values.items()}


def _leave_one_lineage_out(
    predicted: dict[str, float],
    reference: dict[str, float],
    lineage_by_batch: dict[str, str],
    config: StaticBaselineConfig,
) -> dict:
    folds = []
    for lineage in sorted(set(lineage_by_batch.values())):
        retained = [batch_id for batch_id in predicted if lineage_by_batch[batch_id] != lineage]
        if len(retained) < 2:
            continue
        valid_top_ks = [k for k in config.top_ks if k <= len(retained)]
        fold = evaluate_rankings(
            {batch_id: predicted[batch_id] for batch_id in retained},
            {batch_id: reference[batch_id] for batch_id in retained},
            top_ks=valid_top_ks,
            indifference_margin=config.indifference_margin,
        )
        folds.append({"excluded_lineage_group": lineage, "metrics": fold})
    finite_spearman = [fold["metrics"]["spearman"] for fold in folds if fold["metrics"]["spearman"] is not None]
    return {
        "folds": folds,
        "minimum_spearman": min(finite_spearman) if finite_spearman else None,
    }


def analyze_static_baselines(
    suite: BatchSuite,
    features: list[StaticBatchFeatures],
    references: list[ReferenceBatchValue],
    config: StaticBaselineConfig,
    proxy_values: list[TrainableProxyBatchValue] | None = None,
) -> dict:
    """Evaluate preregistered static scores without tuning them on reference utility."""

    _validate_suite_rows(suite, features, description="Static features")
    _validate_suite_rows(suite, references, description="Reference utilities")
    if proxy_values is not None:
        _validate_suite_rows(suite, proxy_values, description="Trainable proxy utilities")
    reference_identities = {
        (
            row.task_id,
            row.benchmark_sha256,
            row.base_model_id,
            row.training_protocol_sha256,
            tuple(row.seeds),
        )
        for row in references
    }
    if len(reference_identities) != 1:
        raise ValueError(
            "Reference utilities must use one task, benchmark, base model, training protocol, and seed set"
        )
    if proxy_values is not None:
        proxy_identities = {
            (
                row.task_id,
                row.benchmark_sha256,
                row.base_model_id,
                row.training_protocol_sha256,
                tuple(row.seeds),
                row.proxy_method,
                row.budget_id,
            )
            for row in proxy_values
        }
        if len(proxy_identities) != 1:
            raise ValueError(
                "Trainable proxy utilities must use one task, benchmark, base model, training protocol, "
                "seed set, method, and budget"
            )
        proxy_identity = next(iter(proxy_identities))
        reference_identity = next(iter(reference_identities))
        if proxy_identity[:5] != reference_identity:
            raise ValueError(
                "Trainable proxy and reference utilities must share task, benchmark, base model, "
                "training protocol, and paired seeds"
            )
    if any(k > len(suite.batches) for k in config.top_ks):
        raise ValueError("A requested top-k exceeds the number of registered batches")

    by_batch = {row.batch_id: row for row in features}
    reference = {row.batch_id: row.utility for row in references}
    lineage_by_batch = {batch.batch_id: batch.lineage_group for batch in suite.batches}
    score_vectors: dict[str, dict[str, float]] = {}
    for baseline in config.enabled_baselines:
        weights = config.composite_weights if baseline == "composite" else config.named_composites.get(baseline)
        if weights is not None:
            normalized = {
                feature_name: _minmax(
                    {batch_id: _feature_value(row, feature_name) for batch_id, row in by_batch.items()}
                )
                for feature_name in weights
            }
            total_weight = sum(weights.values())
            score_vectors[baseline] = {
                batch_id: sum(weights[feature_name] * normalized[feature_name][batch_id] for feature_name in weights)
                / total_weight
                for batch_id in by_batch
            }
        else:
            score_vectors[baseline] = {batch_id: _feature_value(row, baseline) for batch_id, row in by_batch.items()}

    results = {}
    for baseline, predicted in score_vectors.items():
        results[baseline] = {
            "global": evaluate_rankings(
                predicted,
                reference,
                top_ks=config.top_ks,
                indifference_margin=config.indifference_margin,
            ),
            "leave_one_lineage_out": _leave_one_lineage_out(
                predicted,
                reference,
                lineage_by_batch,
                config,
            ),
        }
    proxy_comparisons = None
    hard_pair_analysis = None
    if proxy_values is not None:
        proxy_scores = {row.batch_id: row.utility for row in proxy_values}
        results["trainable_proxy"] = {
            "global": evaluate_rankings(
                proxy_scores,
                reference,
                top_ks=config.top_ks,
                indifference_margin=config.indifference_margin,
            ),
            "leave_one_lineage_out": _leave_one_lineage_out(
                proxy_scores,
                reference,
                lineage_by_batch,
                config,
            ),
        }
        proxy_global = results["trainable_proxy"]["global"]
        proxy_comparisons = {
            baseline: _ranking_advantage(proxy_global, result["global"])
            for baseline, result in results.items()
            if baseline != "trainable_proxy"
        }
        if config.hard_pair_baseline is not None:
            hard_pair_analysis = _hard_pair_analysis(
                score_vectors[config.hard_pair_baseline],
                proxy_scores,
                reference,
                pair_count=config.hard_pair_count,
                indifference_margin=config.indifference_margin,
            )
        score_vectors["trainable_proxy"] = proxy_scores
    score_rows = [
        {
            "schema_version": "static-baseline-score-v1",
            "baseline": baseline,
            "batch_id": batch_id,
            "score": score,
            "reference_utility": reference[batch_id],
        }
        for baseline, scores in sorted(score_vectors.items())
        for batch_id, score in sorted(scores.items())
    ]
    first_reference = references[0]
    return {
        "schema_version": "static-baseline-report-v1",
        "suite_sha256": suite_digest(suite),
        "task_id": first_reference.task_id,
        "benchmark_sha256": first_reference.benchmark_sha256,
        "base_model_id": first_reference.base_model_id,
        "training_protocol_sha256": first_reference.training_protocol_sha256,
        "seeds": first_reference.seeds,
        "batch_count": len(suite.batches),
        "lineage_group_count": len(set(lineage_by_batch.values())),
        "fitted_to_reference": False,
        "named_composite_definitions": config.named_composites,
        "trainable_proxy_vs_static": proxy_comparisons,
        "hard_pair_analysis": hard_pair_analysis,
        "config": config.model_dump(mode="json"),
        "baselines": results,
        "score_rows": score_rows,
    }


def _ranking_advantage(target: dict, baseline: dict) -> dict[str, float | None]:
    """Return target-minus-baseline differences, with regret sign reversed."""

    return {
        "spearman": _difference(target["spearman"], baseline["spearman"]),
        "kendall_tau_b": _difference(target["kendall_tau_b"], baseline["kendall_tau_b"]),
        "pairwise_order_accuracy": _difference(target["pairwise_order_accuracy"], baseline["pairwise_order_accuracy"]),
        "top1_recovery": _difference(target["top_k"]["1"]["recall"], baseline["top_k"]["1"]["recall"]),
        "top1_regret_reduction": _difference(baseline["regret"]["1"], target["regret"]["1"]),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    return float(left - right) if left is not None and right is not None else None


def _hard_pair_analysis(
    static_scores: dict[str, float],
    proxy_scores: dict[str, float],
    reference: dict[str, float],
    *,
    pair_count: int,
    indifference_margin: float,
) -> dict:
    """Evaluate pairs selected solely because the static score is uncertain."""

    candidate_pairs = sorted(
        combinations(sorted(static_scores), 2),
        key=lambda pair: (abs(static_scores[pair[0]] - static_scores[pair[1]]), pair),
    )[:pair_count]
    rows = []
    static_correct = []
    proxy_correct = []
    for first, second in candidate_pairs:
        reference_delta = reference[first] - reference[second]
        distinguishable = abs(reference_delta) > indifference_margin
        reference_sign = 0 if not distinguishable else (1 if reference_delta > 0 else -1)
        static_delta = static_scores[first] - static_scores[second]
        proxy_delta = proxy_scores[first] - proxy_scores[second]
        static_sign = 1 if static_delta > 0 else -1 if static_delta < 0 else 0
        proxy_sign = 1 if proxy_delta > 0 else -1 if proxy_delta < 0 else 0
        if distinguishable:
            static_correct.append(0.5 if static_sign == 0 else float(static_sign == reference_sign))
            proxy_correct.append(0.5 if proxy_sign == 0 else float(proxy_sign == reference_sign))
        rows.append(
            {
                "batch_ids": [first, second],
                "static_score_gap": abs(static_delta),
                "reference_utility_gap": abs(reference_delta),
                "reference_distinguishable": distinguishable,
                "static_order_correct": None if not distinguishable else static_correct[-1],
                "proxy_order_correct": None if not distinguishable else proxy_correct[-1],
            }
        )
    return {
        "selection_rule": "smallest_absolute_static_score_gap_before_viewing_reference_order",
        "requested_pair_count": pair_count,
        "selected_pair_count": len(rows),
        "distinguishable_pair_count": len(static_correct),
        "static_pairwise_order_accuracy": _mean(static_correct) if static_correct else None,
        "proxy_pairwise_order_accuracy": _mean(proxy_correct) if proxy_correct else None,
        "proxy_accuracy_advantage": (_mean(proxy_correct) - _mean(static_correct) if static_correct else None),
        "pairs": rows,
    }


class CrossTaskBatchValue(StrictModel):
    schema_version: Literal["cross-task-batch-value-v1"] = "cross-task-batch-value-v1"
    task_id: str = Field(min_length=1)
    suite_sha256: str
    benchmark_sha256: str
    base_model_id: str = Field(min_length=1)
    training_protocol_sha256: str
    seeds: list[int] = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    lineage_group: str = Field(min_length=1)
    arm: Literal["proxy", "reference"]
    utility: float

    @field_validator(
        "suite_sha256",
        "benchmark_sha256",
        "training_protocol_sha256",
        "batch_content_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _sha256(value, info.field_name) or ""

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("seeds must be unique")
        return sorted(value)


class CrossTaskConfig(StrictModel):
    schema_version: Literal["cross-task-config-v1"] = "cross-task-config-v1"
    top_ks: list[int] = Field(default_factory=lambda: [1, 3], min_length=1)
    indifference_margin_by_task: dict[str, float] = Field(default_factory=dict)

    @field_validator("indifference_margin_by_task")
    @classmethod
    def validate_margins(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(margin) or margin < 0 for margin in value.values()):
            raise ValueError("Task indifference margins must be finite and non-negative")
        return value


def _task_pair_comparison(
    first_task: str,
    second_task: str,
    first: dict[str, float],
    second: dict[str, float],
    margins: dict[str, float],
) -> dict:
    batch_ids = sorted(first)
    first_values = [first[batch_id] for batch_id in batch_ids]
    second_values = [second[batch_id] for batch_id in batch_ids]
    first_margin = margins.get(first_task, 0.0)
    second_margin = margins.get(second_task, 0.0)
    comparable = inversions = 0
    for left, right in combinations(batch_ids, 2):
        first_delta = first[left] - first[right]
        second_delta = second[left] - second[right]
        first_sign = 0 if abs(first_delta) <= first_margin else (1 if first_delta > 0 else -1)
        second_sign = 0 if abs(second_delta) <= second_margin else (1 if second_delta > 0 else -1)
        if first_sign == 0 or second_sign == 0:
            continue
        comparable += 1
        inversions += int(first_sign != second_sign)
    spearman = _pearson(
        _margin_aware_ranks(first_values, first_margin),
        _margin_aware_ranks(second_values, second_margin),
    )
    return {
        "first_task_id": first_task,
        "second_task_id": second_task,
        "spearman": spearman,
        "kendall_tau_b": _kendall_tau_b(
            first_values,
            second_values,
            left_margin=first_margin,
            right_margin=second_margin,
        ),
        "comparable_batch_pairs": comparable,
        "ordering_inversions": inversions,
        "ordering_inversion_rate": inversions / comparable if comparable else None,
        "same_top1": _ordered(first)[0] == _ordered(second)[0],
        "first_order": _ordered(first),
        "second_order": _ordered(second),
    }


def analyze_cross_task_values(rows: list[CrossTaskBatchValue], config: CrossTaskConfig) -> dict:
    """Test whether proxy rankings track task-specific full-training rankings."""

    if not rows:
        raise ValueError("Cross-task analysis requires observations")
    task_ids = sorted({row.task_id for row in rows})
    if len(task_ids) < 2:
        raise ValueError("Cross-task analysis requires at least two downstream tasks")
    if unknown := set(config.indifference_margin_by_task) - set(task_ids):
        raise ValueError(f"Margins reference unknown tasks: {sorted(unknown)}")
    if len({row.base_model_id for row in rows}) != 1:
        raise ValueError("All task comparisons must use the same base model")
    if len({row.suite_sha256 for row in rows}) != 1:
        raise ValueError("All task comparisons must use the same frozen batch suite")
    if len({row.training_protocol_sha256 for row in rows}) != 1:
        raise ValueError("All task comparisons must use the same training protocol")
    if len({tuple(row.seeds) for row in rows}) != 1:
        raise ValueError("All task comparisons must use the same paired seed set")

    grouped: dict[tuple[str, str], dict[str, CrossTaskBatchValue]] = defaultdict(dict)
    for row in rows:
        key = (row.task_id, row.arm)
        if row.batch_id in grouped[key]:
            raise ValueError(f"Duplicate cross-task observation: {key} / {row.batch_id}")
        grouped[key][row.batch_id] = row
    expected_batches = set(grouped[(task_ids[0], "reference")])
    if len(expected_batches) < 2:
        raise ValueError("Each task requires at least two batches")
    content_by_batch: dict[str, str] = {}
    lineage_by_batch: dict[str, str] = {}
    benchmark_by_task: dict[str, str] = {}
    for task_id in task_ids:
        for arm in ("proxy", "reference"):
            observed = grouped[(task_id, arm)]
            if set(observed) != expected_batches:
                raise ValueError(f"Task {task_id} arm {arm} does not cover the common batch set")
            for batch_id, row in observed.items():
                previous_hash = content_by_batch.setdefault(batch_id, row.batch_content_sha256)
                previous_lineage = lineage_by_batch.setdefault(batch_id, row.lineage_group)
                if previous_hash != row.batch_content_sha256 or previous_lineage != row.lineage_group:
                    raise ValueError(f"Batch identity changed across tasks: {batch_id}")
                previous_benchmark = benchmark_by_task.setdefault(task_id, row.benchmark_sha256)
                if previous_benchmark != row.benchmark_sha256:
                    raise ValueError(f"Task {task_id} uses more than one benchmark manifest")
    if any(k > len(expected_batches) for k in config.top_ks):
        raise ValueError("A requested top-k exceeds the common batch count")

    values = {
        arm: {
            task_id: {batch_id: grouped[(task_id, arm)][batch_id].utility for batch_id in expected_batches}
            for task_id in task_ids
        }
        for arm in ("proxy", "reference")
    }
    within_task = {
        task_id: evaluate_rankings(
            values["proxy"][task_id],
            values["reference"][task_id],
            top_ks=config.top_ks,
            indifference_margin=config.indifference_margin_by_task.get(task_id, 0.0),
        )
        for task_id in task_ids
    }
    task_pairs = []
    for first_task, second_task in combinations(task_ids, 2):
        reference_comparison = _task_pair_comparison(
            first_task,
            second_task,
            values["reference"][first_task],
            values["reference"][second_task],
            config.indifference_margin_by_task,
        )
        proxy_comparison = _task_pair_comparison(
            first_task,
            second_task,
            values["proxy"][first_task],
            values["proxy"][second_task],
            config.indifference_margin_by_task,
        )
        task_pairs.append(
            {
                "first_task_id": first_task,
                "second_task_id": second_task,
                "reference": reference_comparison,
                "proxy": proxy_comparison,
                "reference_task_conditioning_observed": (
                    reference_comparison["ordering_inversions"] > 0 or not reference_comparison["same_top1"]
                ),
            }
        )
    return {
        "schema_version": "cross-task-report-v1",
        "task_ids": task_ids,
        "batch_ids": sorted(expected_batches),
        "batch_count": len(expected_batches),
        "suite_sha256": rows[0].suite_sha256,
        "base_model_id": rows[0].base_model_id,
        "training_protocol_sha256": rows[0].training_protocol_sha256,
        "seeds": rows[0].seeds,
        "benchmark_sha256_by_task": benchmark_by_task,
        "within_task_proxy_fidelity": within_task,
        "between_task_ranking_change": task_pairs,
    }


CostStage = Literal[
    "source_scan",
    "parsing",
    "semantic_extraction",
    "proxy_selection",
    "generation",
    "quality_control",
    "training",
    "evaluation",
    "orchestration",
]


class CostLedgerEvent(StrictModel):
    schema_version: Literal["valuation-cost-event-v1"] = "valuation-cost-event-v1"
    event_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    batch_id: str | None = None
    batch_content_sha256: str | None = None
    branch: Literal["proxy", "reference"]
    stage: CostStage
    status: Literal["success", "rejected", "failed"]
    wall_seconds: float = Field(default=0.0, ge=0.0)
    cpu_seconds: float = Field(default=0.0, ge=0.0)
    gpu_seconds: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    train_tokens: int = Field(default=0, ge=0)
    notes: str | None = None

    @field_validator("batch_content_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _sha256(value, "batch_content_sha256", optional=True)

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.batch_id is None) != (self.batch_content_sha256 is None):
            raise ValueError("Batch-scoped events require both batch_id and batch_content_sha256")
        return self


class CostAnalysisConfig(StrictModel):
    schema_version: Literal["cost-analysis-config-v1"] = "cost-analysis-config-v1"
    proxy_required_stages: list[CostStage] = Field(
        default_factory=lambda: [
            "source_scan",
            "parsing",
            "semantic_extraction",
            "proxy_selection",
            "generation",
            "quality_control",
            "training",
            "evaluation",
        ]
    )
    reference_required_stages: list[CostStage] = Field(
        default_factory=lambda: [
            "source_scan",
            "parsing",
            "semantic_extraction",
            "generation",
            "quality_control",
            "training",
            "evaluation",
        ]
    )
    shared_event_allocation: Literal["equal_across_batches"] = "equal_across_batches"


_COST_FIELDS = [
    "wall_seconds",
    "cpu_seconds",
    "gpu_seconds",
    "input_tokens",
    "output_tokens",
    "train_tokens",
]


def _zero_costs() -> dict[str, float]:
    return {field: 0.0 for field in _COST_FIELDS}


def _add_costs(target: dict[str, float], event: CostLedgerEvent, fraction: float) -> None:
    for field in _COST_FIELDS:
        target[field] += float(getattr(event, field)) * fraction


def _cost_ratio(proxy: float, reference: float) -> float | None:
    return proxy / reference if reference > 0 else None


def analyze_cost_ledger(
    suite: BatchSuite,
    events: list[CostLedgerEvent],
    config: CostAnalysisConfig,
) -> dict:
    """Aggregate every incurred event into an end-to-end proxy/reference ledger."""

    if not events:
        raise ValueError("Cost analysis requires at least one ledger event")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("Cost event IDs must be unique")
    if len({event.task_id for event in events}) != 1:
        raise ValueError("One cost report may describe only one downstream task")
    registered = {batch.batch_id: batch for batch in suite.batches}
    if any(batch.content_sha256 is None for batch in suite.batches):
        raise ValueError("The batch suite must be frozen before cost analysis")
    for event in events:
        if event.batch_id is None:
            continue
        if event.batch_id not in registered:
            raise ValueError(f"Cost event references an unknown batch: {event.batch_id}")
        if event.batch_content_sha256 != registered[event.batch_id].content_sha256:
            raise ValueError(f"Cost event uses different content for batch {event.batch_id}")

    batch_ids = sorted(registered)
    totals = {(batch_id, branch): _zero_costs() for batch_id in batch_ids for branch in ("proxy", "reference")}
    stages = {
        (batch_id, branch, stage): _zero_costs()
        for batch_id in batch_ids
        for branch in ("proxy", "reference")
        for stage in CostStage.__args__
    }
    statuses = {
        (batch_id, branch, status): _zero_costs()
        for batch_id in batch_ids
        for branch in ("proxy", "reference")
        for status in ("success", "rejected", "failed")
    }
    observed_stages: dict[tuple[str, str], set[str]] = defaultdict(set)
    for event in events:
        allocations = (
            [(event.batch_id, 1.0)] if event.batch_id else [(batch_id, 1.0 / len(batch_ids)) for batch_id in batch_ids]
        )
        for batch_id, fraction in allocations:
            _add_costs(totals[(batch_id, event.branch)], event, fraction)
            _add_costs(stages[(batch_id, event.branch, event.stage)], event, fraction)
            _add_costs(statuses[(batch_id, event.branch, event.status)], event, fraction)
            observed_stages[(batch_id, event.branch)].add(event.stage)

    required = {
        "proxy": set(config.proxy_required_stages),
        "reference": set(config.reference_required_stages),
    }
    missing_stages = []
    for batch_id in batch_ids:
        for branch in ("proxy", "reference"):
            missing = sorted(required[branch] - observed_stages[(batch_id, branch)])
            if missing:
                missing_stages.append({"batch_id": batch_id, "branch": branch, "missing_stages": missing})
    if missing_stages:
        raise ValueError(f"End-to-end cost ledger is incomplete: {missing_stages}")

    batch_rows = [
        {
            "schema_version": "valuation-batch-cost-v1",
            "batch_id": batch_id,
            "branch": branch,
            **totals[(batch_id, branch)],
        }
        for batch_id in batch_ids
        for branch in ("proxy", "reference")
    ]
    stage_rows = [
        {
            "schema_version": "valuation-stage-cost-v1",
            "batch_id": batch_id,
            "branch": branch,
            "stage": stage,
            **stages[(batch_id, branch, stage)],
        }
        for batch_id in batch_ids
        for branch in ("proxy", "reference")
        for stage in CostStage.__args__
        if stage in observed_stages[(batch_id, branch)]
    ]
    branch_totals = {branch: _zero_costs() for branch in ("proxy", "reference")}
    for batch_id in batch_ids:
        for branch in ("proxy", "reference"):
            for field in _COST_FIELDS:
                branch_totals[branch][field] += totals[(batch_id, branch)][field]
    ratios = {
        field: _cost_ratio(branch_totals["proxy"][field], branch_totals["reference"][field]) for field in _COST_FIELDS
    }
    savings = {field: (1.0 - ratio if ratio is not None else None) for field, ratio in ratios.items()}
    status_totals = {
        branch: {
            status: {
                field: sum(statuses[(batch_id, branch, status)][field] for batch_id in batch_ids)
                for field in _COST_FIELDS
            }
            for status in ("success", "rejected", "failed")
        }
        for branch in ("proxy", "reference")
    }
    return {
        "schema_version": "valuation-cost-report-v1",
        "task_id": events[0].task_id,
        "suite_sha256": suite_digest(suite),
        "batch_count": len(batch_ids),
        "event_count": len(events),
        "all_statuses_charged": True,
        "shared_event_allocation": config.shared_event_allocation,
        "branch_totals": branch_totals,
        "proxy_to_reference_ratio": ratios,
        "proxy_savings_fraction": savings,
        "status_totals": status_totals,
        "batch_rows": batch_rows,
        "stage_rows": stage_rows,
    }


class NaturalBatchAuditConfig(StrictModel):
    schema_version: Literal["natural-batch-audit-config-v1"] = "natural-batch-audit-config-v1"
    minimum_natural_batch_count: int = Field(default=8, ge=2)
    minimum_independent_lineage_count: int = Field(default=8, ge=2)
    maximum_batches_per_lineage: int = Field(default=1, ge=1)
    forbid_source_overlap_between_natural_batches: bool = True
    forbid_lineage_overlap_with_constructed_batches: bool = True
    forbid_pairing_between_natural_batches: bool = True


def audit_natural_batches(
    suite: BatchSuite,
    config: NaturalBatchAuditConfig,
    *,
    suite_path: Path | None = None,
    check_paths: bool = False,
) -> dict:
    """Audit whether natural-check batches provide genuinely independent evidence."""

    natural = [batch for batch in suite.batches if batch.kind == "natural_check"]
    constructed = [batch for batch in suite.batches if batch.kind != "natural_check"]
    errors: list[str] = []
    warnings: list[str] = []
    if len(natural) < config.minimum_natural_batch_count:
        errors.append(
            f"Natural batch count is {len(natural)}; at least {config.minimum_natural_batch_count} are required"
        )
    lineage_counts: dict[str, int] = defaultdict(int)
    for batch in natural:
        lineage_counts[batch.lineage_group] += 1
    if len(lineage_counts) < config.minimum_independent_lineage_count:
        errors.append(
            f"Independent natural lineage count is {len(lineage_counts)}; "
            f"at least {config.minimum_independent_lineage_count} are required"
        )
    crowded = {
        lineage: count for lineage, count in lineage_counts.items() if count > config.maximum_batches_per_lineage
    }
    if crowded:
        errors.append(f"Natural lineages exceed maximum_batches_per_lineage: {crowded}")

    source_to_batches: dict[str, list[str]] = defaultdict(list)
    for batch in natural:
        for source_id in batch.source_ids:
            source_to_batches[source_id].append(batch.batch_id)
    overlapping_sources = {
        source_id: sorted(batch_ids) for source_id, batch_ids in source_to_batches.items() if len(batch_ids) > 1
    }
    if config.forbid_source_overlap_between_natural_batches and overlapping_sources:
        errors.append(f"Natural batches reuse source identities: {overlapping_sources}")
    constructed_lineages = {batch.lineage_group for batch in constructed}
    lineage_overlap = sorted(set(lineage_counts) & constructed_lineages)
    if config.forbid_lineage_overlap_with_constructed_batches and lineage_overlap:
        errors.append(f"Natural and constructed batches reuse lineages: {lineage_overlap}")
    paired_natural = {batch.batch_id: batch.paired_with for batch in natural if batch.paired_with}
    if config.forbid_pairing_between_natural_batches and paired_natural:
        errors.append(f"Independent natural batches cannot be paired transformations: {paired_natural}")
    missing_hashes = [batch.batch_id for batch in natural if batch.content_sha256 is None]
    if missing_hashes:
        errors.append(f"Natural batches are not content-frozen: {missing_hashes}")
    hashes = [batch.content_sha256 for batch in natural if batch.content_sha256 is not None]
    if len(hashes) != len(set(hashes)):
        errors.append("Natural batches contain duplicate content hashes")

    path_checks = []
    if check_paths:
        if suite_path is None:
            raise ValueError("suite_path is required when check_paths=True")
        root = suite_path.resolve().parent
        for batch in natural:
            if "://" in batch.input_uri:
                path_checks.append({"batch_id": batch.batch_id, "status": "remote_uri_not_checked"})
                warnings.append(f"Remote input URI was not opened for natural batch {batch.batch_id}")
                continue
            path = Path(batch.input_uri)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            exists = resolved.exists()
            path_checks.append({"batch_id": batch.batch_id, "path": str(resolved), "exists": exists})
            if not exists:
                errors.append(f"Natural batch input does not exist: {batch.batch_id} -> {resolved}")

    registry_rows = [
        {
            "schema_version": "natural-batch-registry-row-v1",
            "suite_sha256": suite_digest(suite),
            "batch_id": batch.batch_id,
            "lineage_group": batch.lineage_group,
            "source_ids": batch.source_ids,
            "input_uri": batch.input_uri,
            "content_sha256": batch.content_sha256,
            "split": batch.split,
            "paired_with": batch.paired_with,
        }
        for batch in sorted(natural, key=lambda item: item.batch_id)
    ]
    return {
        "schema_version": "natural-batch-independence-report-v1",
        "suite_sha256": suite_digest(suite),
        "valid": not errors,
        "natural_batch_count": len(natural),
        "independent_lineage_count": len(lineage_counts),
        "lineage_counts": dict(sorted(lineage_counts.items())),
        "overlapping_sources": overlapping_sources,
        "lineage_overlap_with_constructed_batches": lineage_overlap,
        "path_checks": path_checks,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "config": config.model_dump(mode="json"),
        "registry_rows": registry_rows,
    }
