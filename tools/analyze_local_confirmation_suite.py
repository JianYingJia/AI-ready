"""Audit and summarize the local matched-exposure confirmation suite."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ai_ready.valuation.decision_experiments import evaluate_rankings

REFERENCE_METHOD = "same_generator_full_pool"
TARGET_METHOD = "coverage_quality"
PROXY_METHODS = (
    "coverage_quality",
    "coverage_only",
    "random",
    "stratified_random",
    "relevance",
    "diversity",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _interval(values: list[float]) -> dict[str, Any]:
    return {
        "valid_replicates": len(values),
        "median": _percentile(values, 0.5),
        "percentile_95_interval": [_percentile(values, 0.025), _percentile(values, 0.975)] if values else None,
        "fraction_positive": sum(value > 0 for value in values) / len(values) if values else None,
    }


def _ranking(
    predicted: dict[str, float],
    reference: dict[str, float],
    indifference_margin: float,
) -> dict[str, Any] | None:
    if len(set(predicted.values())) <= 1 or len(set(reference.values())) <= 1:
        return None
    return evaluate_rankings(
        predicted,
        reference,
        top_ks=[1, 3],
        indifference_margin=indifference_margin,
    )


def _normalized_correct_log_likelihood(prediction: dict[str, Any]) -> float:
    values = [float(value) for value in prediction["option_log_likelihoods"]]
    maximum = max(values)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return values[int(prediction["answer_index"])] - log_normalizer


def _attach_continuous_utilities(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict]],
    bases: dict[str, list[dict]],
) -> None:
    for row in rows:
        trial_predictions = predictions[row["trial_id"]]
        base_predictions = bases[row["_run_directory"]]
        if len(trial_predictions) != 46 or len(base_predictions) != 46:
            raise ValueError(f"Trial {row['trial_id']} does not contain exactly 46 predictions")
        trial_score = sum(_normalized_correct_log_likelihood(value) for value in trial_predictions) / 46
        base_score = sum(_normalized_correct_log_likelihood(value) for value in base_predictions) / 46
        row["continuous_utility"] = trial_score - base_score


def _mean_vectors(rows: list[dict[str, Any]], value_key: str = "utility") -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value: Any = row
        for part in value_key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            grouped[(row["method"], row["batch_id"])].append(float(value))
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (method, batch_id), values in grouped.items():
        output[method][batch_id] = sum(values) / len(values)
    return dict(output)


def _static_batch_profiles(source_root: Path, batch_ids: list[str]) -> dict[str, dict[str, Any]]:
    profiles = {}
    for batch_id in batch_ids:
        units = _read_jsonl(source_root / batch_id / "proxy_units.jsonl")
        obligations = _read_jsonl(source_root / batch_id / "coverage_obligations.jsonl")
        total_tokens = sum(int(row["estimated_tokens"]) for row in units)
        relevance = (
            sum(float(row["benchmark_relevance"]) * int(row["estimated_tokens"]) for row in units) / total_tokens
        )
        quality_tokens: dict[str, int] = defaultdict(int)
        for row in units:
            quality_tokens[str(row["quality_bucket"])] += int(row["estimated_tokens"])
        embedding_signatures = {tuple(round(float(value), 6) for value in row["embedding"]) for row in units}
        profiles[batch_id] = {
            "size_tokens": total_tokens,
            "relevance": relevance,
            "quality_distribution": {key: value / total_tokens for key, value in sorted(quality_tokens.items())},
            "redundancy": 1 - len(embedding_signatures) / len(units),
            "unique_embedding_count": len(embedding_signatures),
            "information_density": len(embedding_signatures) / total_tokens,
            "generic_quality": 1 - quality_tokens.get("parser_noisy", 0) / total_tokens,
            "semantic_coverage": len(obligations),
        }
    return profiles


def _intrinsic_static_scores(profiles: dict[str, dict[str, Any]]) -> dict[str, float]:
    features = {
        batch_id: {
            "log_size": math.log1p(float(profile["size_tokens"])),
            "low_redundancy": 1 - float(profile["redundancy"]),
            "information_density": float(profile["information_density"]),
            "generic_quality": float(profile["generic_quality"]),
        }
        for batch_id, profile in profiles.items()
    }
    keys = ("log_size", "low_redundancy", "information_density", "generic_quality")
    bounds = {
        key: (
            min(values := [row[key] for row in features.values()]),
            max(values),
        )
        for key in keys
    }
    scores = {}
    for batch_id, row in features.items():
        normalized = []
        for key in keys:
            low, high = bounds[key]
            normalized.append((row[key] - low) / (high - low) if high > low else 0.5)
        scores[batch_id] = sum(normalized) / len(normalized)
    return scores


def _static_pair_distances(profiles: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
    scalar_keys = ("size_tokens", "relevance", "redundancy", "semantic_coverage")
    scales = {}
    for key in scalar_keys:
        values = [float(profile[key]) for profile in profiles.values()]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scales[key] = max(math.sqrt(variance), 1e-12)
    distances = {}
    batch_ids = sorted(profiles)
    for index, left in enumerate(batch_ids):
        for right in batch_ids[index + 1 :]:
            squared = sum(
                ((float(profiles[left][key]) - float(profiles[right][key])) / scales[key]) ** 2 for key in scalar_keys
            )
            left_quality = profiles[left]["quality_distribution"]
            right_quality = profiles[right]["quality_distribution"]
            quality_keys = set(left_quality) | set(right_quality)
            quality_total_variation = 0.5 * sum(
                abs(left_quality.get(key, 0.0) - right_quality.get(key, 0.0)) for key in quality_keys
            )
            distances[(left, right)] = math.sqrt(squared + quality_total_variation**2)
    return distances


def _load_suite(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict]], dict[str, list[dict]]]:
    trials = []
    predictions: dict[str, list[dict]] = defaultdict(list)
    base_predictions: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("trial_results.jsonl")):
        directory = path.parent
        local_trials = _read_jsonl(path)
        manifest = _read_json(directory / "run_manifest.json")
        base = _read_json(directory / "base_model_result.json")["predictions"]
        base_predictions[str(directory)] = base
        for row in local_trials:
            row["_run_directory"] = str(directory)
            row["_manifest_training_seed"] = manifest["training_seed"]
            row["_manifest_selection_seed"] = manifest["selection_seed"]
            trials.append(row)
        for row in _read_jsonl(directory / "item_predictions.jsonl"):
            predictions[row["trial_id"]].append(row)
    return trials, predictions, base_predictions


def _audit_matrix(trials: list[dict[str, Any]], batch_ids: list[str]) -> dict[str, Any]:
    core = [row for row in trials if row["_manifest_selection_seed"] == 101]
    expected_core = {
        (batch_id, method, training_seed)
        for batch_id in batch_ids
        for method in (*PROXY_METHODS, REFERENCE_METHOD)
        for training_seed in (17, 29, 43)
    }
    observed_core = {(row["batch_id"], row["method"], row["training_seed"]) for row in core}
    random_rows = [
        row for row in trials if row["method"] in {"random", "stratified_random"} and row["training_seed"] == 17
    ]
    expected_random = {
        (batch_id, method, selection_seed)
        for batch_id in batch_ids
        for method in ("random", "stratified_random")
        for selection_seed in (101, 102, 103, 104, 105)
    }
    observed_random = {(row["batch_id"], row["method"], row["selection_seed"]) for row in random_rows}
    return {
        "core_expected": len(expected_core),
        "core_observed": len(observed_core),
        "core_missing": [list(value) for value in sorted(expected_core - observed_core)],
        "construction_seed_expected": len(expected_random),
        "construction_seed_observed": len(observed_random),
        "construction_seed_missing": [list(value) for value in sorted(expected_random - observed_random)],
        "complete": expected_core == observed_core and expected_random == observed_random,
    }


def _construction_identifiability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["batch_id"], row["method"], row["training_seed"]): row for row in rows}
    batch_ids = sorted({row["batch_id"] for row in rows})
    output = {}
    for batch_id in batch_ids:
        comparisons = {}
        for method in PROXY_METHODS[1:]:
            selection_equal = True
            training_equal = True
            for seed in (17, 29, 43):
                target = by_cell[(batch_id, TARGET_METHOD, seed)]
                baseline = by_cell[(batch_id, method, seed)]
                selection_keys = (
                    "token_budget",
                    "used_source_tokens",
                    "selected_unit_count",
                    "accepted_sample_count",
                    "selection_components",
                    "content_static_score",
                )
                selection_equal &= all(
                    target["selection"].get(key) == baseline["selection"].get(key) for key in selection_keys
                )
                training_equal &= (
                    target["training"]["encoded_sample_count"] == baseline["training"]["encoded_sample_count"]
                    and target["training"]["effective_supervised_tokens"]
                    == baseline["training"]["effective_supervised_tokens"]
                    and math.isclose(
                        float(target["training"]["frozen_model_score"]),
                        float(baseline["training"]["frozen_model_score"]),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
            comparisons[method] = {
                "selection_summary_identical": selection_equal,
                "training_exposure_fingerprint_identical": training_equal,
                "contrast_identifiable": not (selection_equal and training_equal),
            }
        output[batch_id] = comparisons
    return output


def _hard_pairs(
    content_values: dict[str, float],
    static_profiles: dict[str, dict[str, Any]],
    reference: dict[str, float],
    trial: dict[str, float],
    indifference_margin: float,
) -> list[dict]:
    static_distances = _static_pair_distances(static_profiles)
    pairs = []
    batch_ids = sorted(content_values)
    for index, left in enumerate(batch_ids):
        for right in batch_ids[index + 1 :]:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "static_profile_distance": static_distances[(left, right)],
                    "static_gap": abs(content_values[left] - content_values[right]),
                    "reference_gap": reference[left] - reference[right],
                    "trial_gap": trial[left] - trial[right],
                }
            )
    selected = sorted(
        pairs,
        key=lambda row: (row["static_profile_distance"], row["static_gap"], row["left"], row["right"]),
    )[:5]
    for row in selected:
        row["left_static_profile"] = static_profiles[row["left"]]
        row["right_static_profile"] = static_profiles[row["right"]]
        row["reference_distinguishable"] = abs(row["reference_gap"]) > indifference_margin
        row["content_static_correct"] = (
            row["reference_gap"] * (content_values[row["left"]] - content_values[row["right"]]) > 0
        )
        row["trial_training_correct"] = row["reference_gap"] * row["trial_gap"] > 0
    return selected


def _hard_pair_bootstrap(
    selected_pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict]],
    replicates: int,
    score_kind: str,
) -> list[dict[str, Any]]:
    by_cell = {
        (row["method"], row["batch_id"], int(row["training_seed"])): predictions[row["trial_id"]]
        for row in rows
        if row["method"] in {REFERENCE_METHOD, TARGET_METHOD}
    }
    seeds = sorted({int(row["training_seed"]) for row in rows})
    output = []
    for pair_index, pair in enumerate(selected_pairs):
        rng = random.Random(20261025 + pair_index)
        reference_gaps = []
        trial_gaps = []
        paired_ordering = []
        for _ in range(replicates):
            sampled_seeds = [rng.choice(seeds) for _ in seeds]
            sampled_items = [rng.randrange(46) for _ in range(46)]

            def score(method: str, batch_id: str) -> float:
                values = []
                for seed in sampled_seeds:
                    cell = by_cell[(method, batch_id, seed)]
                    if score_kind == "accuracy":
                        values.extend(float(bool(cell[index]["correct"])) for index in sampled_items)
                    elif score_kind == "normalized_mcq_log_likelihood":
                        values.extend(_normalized_correct_log_likelihood(cell[index]) for index in sampled_items)
                    else:
                        raise ValueError(f"Unknown score kind: {score_kind}")
                return sum(values) / len(values)

            reference_gap = score(REFERENCE_METHOD, pair["left"]) - score(REFERENCE_METHOD, pair["right"])
            trial_gap = score(TARGET_METHOD, pair["left"]) - score(TARGET_METHOD, pair["right"])
            reference_gaps.append(reference_gap)
            trial_gaps.append(trial_gap)
            paired_ordering.append(float(reference_gap * trial_gap > 0))
        output.append(
            {
                "left": pair["left"],
                "right": pair["right"],
                "reference_gap": _interval(reference_gaps),
                "trial_training_gap": _interval(trial_gaps),
                "trial_matches_reference_fraction": sum(paired_ordering) / len(paired_ordering),
            }
        )
    return output


def _bootstrap_items(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[dict]],
    bases: dict[str, list[dict]],
    intrinsic_values: dict[str, float],
    content_values: dict[str, float],
    replicates: int,
    score_kind: str,
    indifference_margin: float,
) -> dict[str, Any]:
    rng = random.Random(20261025)
    comparisons = {name: [] for name in ("intrinsic_static", "content_static", "frozen_model", *PROXY_METHODS[1:])}
    for _ in range(replicates):
        sampled = [rng.randrange(46) for _ in range(46)]
        utilities: dict[tuple[str, str], list[float]] = defaultdict(list)
        frozen: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            trial_predictions = predictions[row["trial_id"]]
            base = bases[row["_run_directory"]]
            if score_kind == "accuracy":
                trial_score = sum(bool(trial_predictions[index]["correct"]) for index in sampled) / len(sampled)
                base_score = sum(bool(base[index]["correct"]) for index in sampled) / len(sampled)
            elif score_kind == "normalized_mcq_log_likelihood":
                trial_score = sum(
                    _normalized_correct_log_likelihood(trial_predictions[index]) for index in sampled
                ) / len(sampled)
                base_score = sum(_normalized_correct_log_likelihood(base[index]) for index in sampled) / len(sampled)
            else:
                raise ValueError(f"Unknown score kind: {score_kind}")
            utilities[(row["method"], row["batch_id"])].append(trial_score - base_score)
            if row["method"] == TARGET_METHOD and row["training"].get("frozen_model_score") is not None:
                frozen[row["batch_id"]].append(float(row["training"]["frozen_model_score"]))
        vectors: dict[str, dict[str, float]] = defaultdict(dict)
        for (method, batch_id), values in utilities.items():
            vectors[method][batch_id] = sum(values) / len(values)
        reference = vectors.get(REFERENCE_METHOD, {})
        target_metrics = _ranking(
            vectors.get(TARGET_METHOD, {}),
            reference,
            indifference_margin,
        )
        if target_metrics is None or target_metrics["spearman"] is None:
            continue
        baselines = {"intrinsic_static": intrinsic_values, "content_static": content_values}
        baselines["frozen_model"] = {key: sum(value) / len(value) for key, value in frozen.items()}
        baselines.update({method: vectors.get(method, {}) for method in PROXY_METHODS[1:]})
        for name, values in baselines.items():
            metrics = _ranking(values, reference, indifference_margin) if set(values) == set(reference) else None
            if metrics is not None and metrics["spearman"] is not None:
                comparisons[name].append(target_metrics["spearman"] - metrics["spearman"])
    return {name: _interval(values) for name, values in comparisons.items()}


def _lineage_bootstrap(
    vectors: dict[str, dict[str, float]],
    lineages: dict[str, list[str]],
    replicates: int,
    indifference_margin: float,
) -> dict[str, Any]:
    rng = random.Random(20261025)
    outputs = {method: [] for method in ("intrinsic_static", "content_static", "frozen_model", *PROXY_METHODS[1:])}
    names = sorted(lineages)
    for _ in range(replicates):
        sampled = [rng.choice(names) for _ in names]
        reference = {}
        target = {}
        baselines = {name: {} for name in outputs}
        for draw_index, lineage in enumerate(sampled):
            for batch_index, batch_id in enumerate(lineages[lineage]):
                draw_id = f"{draw_index}-{batch_index}-{batch_id}"
                reference[draw_id] = vectors[REFERENCE_METHOD][batch_id]
                target[draw_id] = vectors[TARGET_METHOD][batch_id]
                for name in baselines:
                    baselines[name][draw_id] = vectors[name][batch_id]
        target_metrics = _ranking(target, reference, indifference_margin)
        if target_metrics is None or target_metrics["spearman"] is None:
            continue
        for name, values in baselines.items():
            metrics = _ranking(values, reference, indifference_margin)
            if metrics is not None and metrics["spearman"] is not None:
                outputs[name].append(target_metrics["spearman"] - metrics["spearman"])
    return {name: _interval(values) for name, values in outputs.items()}


def analyze(root: Path, batch_suite: Path, bootstrap_replicates: int) -> dict[str, Any]:
    suite = yaml.safe_load(batch_suite.read_text(encoding="utf-8"))
    batch_ids = [row["batch_id"] for row in suite["batches"]]
    source_root = batch_suite.resolve().parents[2] / "outputs" / "cll_local_sft_qwen2_5_1_5b_full_20260921"
    static_profiles = _static_batch_profiles(source_root, batch_ids)
    intrinsic_values = _intrinsic_static_scores(static_profiles)
    lineages: dict[str, list[str]] = defaultdict(list)
    for row in suite["batches"]:
        lineages[row["lineage_group"]].append(row["batch_id"])
    trials, predictions, bases = _load_suite(root)
    audit = _audit_matrix(trials, batch_ids)
    if not audit["complete"]:
        return {"schema_version": "local-confirmation-report-v1", "matrix_audit": audit, "status": "incomplete"}

    _attach_continuous_utilities(trials, predictions, bases)
    core = [row for row in trials if row["_manifest_selection_seed"] == 101]
    content_values = _mean_vectors(core, "selection.content_static_score")[TARGET_METHOD]
    frozen_values = _mean_vectors(core, "training.frozen_model_score")[TARGET_METHOD]

    def endpoint_report(value_key: str, score_kind: str, margin: float) -> dict[str, Any]:
        vectors = _mean_vectors(core, value_key)
        reference = vectors[REFERENCE_METHOD]
        rankings = {
            method: _ranking(values, reference, margin)
            for method, values in vectors.items()
            if method != REFERENCE_METHOD
        }
        vectors["intrinsic_static"] = intrinsic_values
        vectors["content_static"] = content_values
        vectors["frozen_model"] = frozen_values
        hard_pairs = _hard_pairs(
            content_values,
            static_profiles,
            reference,
            vectors[TARGET_METHOD],
            margin,
        )
        return {
            "indifference_margin": margin,
            "reference_values": reference,
            "rankings": rankings,
            "three_layer_comparison": {
                "intrinsic_static": _ranking(intrinsic_values, reference, margin),
                "task_aware_static": _ranking(content_values, reference, margin),
                "frozen_model": _ranking(frozen_values, reference, margin),
                "trainable_proxy": rankings[TARGET_METHOD],
            },
            "hard_pairs": hard_pairs,
            "hard_pair_seed_item_bootstrap": _hard_pair_bootstrap(
                hard_pairs,
                core,
                predictions,
                bootstrap_replicates,
                score_kind,
            ),
            "item_bootstrap_spearman_advantage": _bootstrap_items(
                core,
                predictions,
                bases,
                intrinsic_values,
                content_values,
                bootstrap_replicates,
                score_kind,
                margin,
            ),
            "lineage_bootstrap_spearman_advantage": _lineage_bootstrap(
                vectors,
                dict(lineages),
                bootstrap_replicates,
                margin,
            ),
        }

    accuracy_report = endpoint_report("utility", "accuracy", 1 / 46)
    continuous_report = endpoint_report(
        "continuous_utility",
        "normalized_mcq_log_likelihood",
        0.0,
    )

    random_rows = [row for row in trials if row["training_seed"] == 17]

    def construction_seed_summary(value_key: str) -> dict[str, Any]:
        values_by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in random_rows:
            if row["method"] in {"random", "stratified_random"}:
                values_by_method[row["method"]][row["batch_id"]].append(float(row[value_key]))
        return {
            method: {
                batch_id: {"values": values, "range": max(values) - min(values)} for batch_id, values in batches.items()
            }
            for method, batches in values_by_method.items()
        }

    return {
        "schema_version": "local-confirmation-report-v1",
        "status": "complete",
        "matrix_audit": audit,
        "primary_endpoint": "accuracy_delta",
        "sensitivity_endpoint": "normalized_mcq_log_likelihood_delta",
        "static_score_definitions": {
            "intrinsic_static": (
                "Equal-weight mean of min-max-normalized log source tokens, low embedding redundancy, "
                "unique-embedding information density, and non-parser-noisy token share; no task relevance, "
                "coverage obligation, model score, or reference utility is used."
            ),
            "task_aware_static": (
                "Frozen weighted objective over coverage-quality-selected content: facility 1.0, task "
                "obligation coverage 1.5, quality fidelity 1.0, and source coverage 0.5."
            ),
        },
        "endpoints": {
            "accuracy_delta": accuracy_report,
            "normalized_mcq_log_likelihood_delta": continuous_report,
        },
        "construction_seed_stability": {
            "accuracy_delta": construction_seed_summary("utility"),
            "normalized_mcq_log_likelihood_delta": construction_seed_summary("continuous_utility"),
        },
        "construction_contrast_identifiability": _construction_identifiability(core),
        "inference_guard": (
            "Four lineage groups permit descriptive sensitivity analysis only. "
            "Seed/item bootstrap intervals quantify only the observed training and benchmark variation; "
            "they are not population-general confidence guarantees. The continuous endpoint is a "
            "prespecified sensitivity analysis with zero tie margin, not a replacement primary endpoint."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--batch-suite", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.input_dir, args.batch_suite, args.bootstrap_replicates)
    output = args.output or args.input_dir / "confirmation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["matrix_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
