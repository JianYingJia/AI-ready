"""Aggregate the local proxy-construction screening run.

Accuracy is the pre-specified endpoint.  A correct-option log-likelihood
margin is also reported as an explicitly exploratory sensitivity endpoint,
because the locked local benchmark has only 46 items and may not resolve the
small updates produced by a one-epoch LoRA screen.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_ready.valuation.decision_experiments import evaluate_rankings

REFERENCE_METHOD = "same_generator_full_pool"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _margin(row: dict[str, Any]) -> float:
    scores = row["option_log_likelihoods"]
    answer = int(row["answer_index"])
    return float(scores[answer] - max(score for index, score in enumerate(scores) if index != answer))


def _discover(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    bases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("trial_results.jsonl")):
        trials.extend(_read_jsonl(path))
        prediction_path = path.with_name("item_predictions.jsonl")
        if prediction_path.exists():
            predictions.extend(_read_jsonl(prediction_path))
        base_path = path.with_name("base_model_result.json")
        if base_path.exists():
            base = _read_json(base_path)
            for trial in _read_jsonl(path):
                bases.setdefault(trial["batch_id"], base)
    return trials, predictions, bases


def _range(values: list[float]) -> float:
    return max(values) - min(values)


def _safe_ranking(
    predicted: dict[str, float | None], reference: dict[str, float], top_ks: list[int]
) -> dict[str, Any] | None:
    if any(value is None or not math.isfinite(float(value)) for value in predicted.values()):
        return None
    values = {batch_id: float(value) for batch_id, value in predicted.items() if value is not None}
    if len(set(values.values())) <= 1 or len(set(reference.values())) <= 1:
        return None
    return evaluate_rankings(values, reference, top_ks=top_ks, indifference_margin=0.0)


def _same_construction_no_training(
    enriched_trials: list[dict[str, Any]], reference: dict[str, float]
) -> dict[str, Any]:
    cq_trials = [row for row in enriched_trials if row["method"] == "coverage_quality"]
    content_static = {
        row["batch_id"]: row["selection"].get("content_static_score") for row in cq_trials
    }
    frozen_model = {row["batch_id"]: row["training"].get("frozen_model_score") for row in cq_trials}
    trial_training = {row["batch_id"]: row["utility"] for row in cq_trials}
    return {
        "content_static_values": content_static,
        "frozen_model_values": frozen_model,
        "trial_training_values": trial_training,
        "content_static_ranking": _safe_ranking(content_static, reference, [1, 3]),
        "frozen_model_ranking": _safe_ranking(frozen_model, reference, [1, 3]),
        "trial_training_ranking": _safe_ranking(trial_training, reference, [1, 3]),
        "interpretation_guard": (
            "All three estimators use the exact CQ-selected content. Content-static and frozen-model scores "
            "perform no parameter update; trial-training utility is the only intervention arm."
        ),
    }


def analyze(root: Path, expected_batches: int, expected_methods: int) -> dict[str, Any]:
    trials, predictions, bases = _discover(root)
    by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for trial in trials:
        key = (trial["batch_id"], trial["method"])
        if key in by_cell:
            raise ValueError(f"Duplicate completed trial cell: {key}")
        if trial["status"] != "completed":
            raise ValueError(f"Non-completed trial cell: {key}")
        by_cell[key] = trial

    batches = sorted({batch_id for batch_id, _ in by_cell})
    methods = sorted({method for _, method in by_cell})
    expected_cells = expected_batches * expected_methods
    if len(batches) != expected_batches or len(methods) != expected_methods or len(by_cell) != expected_cells:
        raise ValueError(
            f"Incomplete matrix: batches={len(batches)}/{expected_batches}, "
            f"methods={len(methods)}/{expected_methods}, cells={len(by_cell)}/{expected_cells}"
        )
    if REFERENCE_METHOD not in methods:
        raise ValueError(f"Missing reference method: {REFERENCE_METHOD}")

    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[row["trial_id"]].append(row)

    base_margins: dict[str, float] = {}
    for batch_id, base in bases.items():
        base_margins[batch_id] = _mean([_margin(row) for row in base["predictions"]])

    enriched_trials = []
    for key in sorted(by_cell):
        trial = dict(by_cell[key])
        rows = prediction_groups[trial["trial_id"]]
        if len(rows) != trial["benchmark"]["item_count"]:
            raise ValueError(f"Prediction count mismatch for {trial['trial_id']}")
        mean_margin = _mean([_margin(row) for row in rows])
        trial["exploratory_mean_correct_option_margin"] = mean_margin
        trial["exploratory_margin_utility"] = mean_margin - base_margins[trial["batch_id"]]
        enriched_trials.append(trial)

    accuracy_reference = {batch_id: by_cell[(batch_id, REFERENCE_METHOD)]["utility"] for batch_id in batches}
    margin_reference = {
        trial["batch_id"]: trial["exploratory_margin_utility"]
        for trial in enriched_trials
        if trial["method"] == REFERENCE_METHOD
    }
    primary_identifiable = len(set(accuracy_reference.values())) > 1
    exploratory_identifiable = len(set(margin_reference.values())) > 1
    accuracy_resolution = 1.0 / next(iter(by_cell.values()))["benchmark"]["item_count"]
    nonzero_reference_batches = sum(not math.isclose(value, 0.0) for value in accuracy_reference.values())
    weak_primary_identifiability = primary_identifiable and nonzero_reference_batches <= 1

    method_summaries: dict[str, dict[str, Any]] = {}
    primary_ranking_summaries: dict[str, dict[str, Any] | None] = {}
    ranking_summaries: dict[str, dict[str, Any] | None] = {}
    for method in methods:
        method_trials = [trial for trial in enriched_trials if trial["method"] == method]
        accuracies = [trial["benchmark"]["accuracy"] for trial in method_trials]
        utilities = [trial["utility"] for trial in method_trials]
        margin_utilities = [trial["exploratory_margin_utility"] for trial in method_trials]
        token_counts = [trial["training"]["effective_supervised_tokens"] for trial in method_trials]
        sample_counts = [trial["selection"]["accepted_sample_count"] for trial in method_trials]
        method_summaries[method] = {
            "mean_accuracy": _mean(accuracies),
            "accuracy_range": _range(accuracies),
            "mean_utility": _mean(utilities),
            "mean_exploratory_margin_utility": _mean(margin_utilities),
            "mean_effective_supervised_tokens": _mean(token_counts),
            "effective_supervised_token_range": [min(token_counts), max(token_counts)],
            "mean_accepted_samples": _mean(sample_counts),
            "zero_sample_batch_count": sum(count == 0 for count in sample_counts),
            "changed_answer_batch_count": sum(not math.isclose(value, 0.0) for value in utilities),
        }
        if method == REFERENCE_METHOD or not primary_identifiable:
            primary_ranking_summaries[method] = None
        else:
            predicted_accuracy_utilities = {trial["batch_id"]: trial["utility"] for trial in method_trials}
            primary_ranking_summaries[method] = evaluate_rankings(
                predicted_accuracy_utilities,
                accuracy_reference,
                top_ks=[1, 3],
                indifference_margin=0.0,
            )
        if method == REFERENCE_METHOD or not exploratory_identifiable:
            ranking_summaries[method] = None
        else:
            predicted = {trial["batch_id"]: trial["exploratory_margin_utility"] for trial in method_trials}
            ranking_summaries[method] = evaluate_rankings(
                predicted,
                margin_reference,
                top_ks=[1, 3],
                indifference_margin=0.0,
            )

    cq = method_summaries.get("coverage_quality")
    direct_differences = {}
    if cq is not None:
        for baseline in ("coverage_only", "random", "stratified_random", "relevance", "diversity"):
            if baseline not in method_summaries:
                continue
            comparison = method_summaries[baseline]
            cq_ranking = ranking_summaries.get("coverage_quality")
            baseline_ranking = ranking_summaries.get(baseline)
            direct_differences[baseline] = {
                "mean_accuracy_difference": cq["mean_accuracy"] - comparison["mean_accuracy"],
                "mean_margin_utility_difference": cq["mean_exploratory_margin_utility"]
                - comparison["mean_exploratory_margin_utility"],
                "exploratory_spearman_difference": (
                    None
                    if cq_ranking is None
                    or baseline_ranking is None
                    or cq_ranking["spearman"] is None
                    or baseline_ranking["spearman"] is None
                    else cq_ranking["spearman"] - baseline_ranking["spearman"]
                ),
            }

    return {
        "schema_version": "local-proxy-screening-summary-v1",
        "artifact_role": "single_seed_local_screening_not_formal_confirmation",
        "batch_count": len(batches),
        "method_count": len(methods),
        "trial_count": len(enriched_trials),
        "benchmark_item_count": next(iter(by_cell.values()))["benchmark"]["item_count"],
        "primary_accuracy_endpoint": {
            "identifiable": primary_identifiable,
            "weakly_identifiable": weak_primary_identifiability,
            "full_pool_reference_values": accuracy_reference,
            "reference_range": _range(list(accuracy_reference.values())),
            "single_item_resolution": accuracy_resolution,
            "nonzero_reference_batch_count": nonzero_reference_batches,
            "interpretation": (
                "Weakly identifiable: all reference variation comes from one item changing in one batch."
                if weak_primary_identifiability
                else (
                    "The primary endpoint identifies between-batch reference variation."
                    if primary_identifiable
                    else "Inconclusive: the full-pool reference has no between-batch variation on accuracy, "
                    "so ranking preservation cannot be evaluated."
                )
            ),
        },
        "exploratory_margin_endpoint": {
            "post_hoc": True,
            "identifiable": exploratory_identifiable,
            "full_pool_reference_values": margin_reference,
            "reference_range": _range(list(margin_reference.values())),
            "warning": "Exploratory sensitivity analysis; it is not a replacement for the accuracy endpoint.",
        },
        "method_summaries": method_summaries,
        "primary_accuracy_ranking_summaries": primary_ranking_summaries,
        "exploratory_ranking_summaries": ranking_summaries,
        "coverage_quality_minus_baseline": direct_differences,
        "same_construction_no_training": _same_construction_no_training(enriched_trials, accuracy_reference),
        "trials": enriched_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int, default=12)
    parser.add_argument("--expected-methods", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input_dir / "local_screening_summary.json"
    summary = analyze(args.input_dir, args.expected_batches, args.expected_methods)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("batch_count", "method_count", "trial_count")}, indent=2))
    print(summary["primary_accuracy_endpoint"]["interpretation"])
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
