"""Compare direct raw-chunk trials with the existing local CQ/full-pool matrix.

The output is an exploratory 12-batch/46-item result. It cannot be pooled with
the rev1 paper's 16-batch/200-item confirmatory Suite C or operational Suite O.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_ready.valuation.contracts import file_sha256, load_batch_suite
from ai_ready.valuation.decision_experiments import evaluate_rankings

METHODS = ("coverage_quality", "raw_chunk_random", "raw_chunk_relevance", "same_generator_full_pool")
SEEDS = (17, 29, 43)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _correct_log_likelihood(prediction: dict[str, Any]) -> float:
    values = [float(value) for value in prediction["option_log_likelihoods"]]
    maximum = max(values)
    normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return values[int(prediction["answer_index"])] - normalizer


def _load_trials(root: Path, allowed: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    for path in sorted(root.glob("training_seed_*/trial_results.jsonl")):
        directory = path.parent
        manifest = _json(directory / "run_manifest.json")
        manifests.append(manifest)
        base = _json(directory / "base_model_result.json")["predictions"]
        base_by_id = {row["benchmark_item_id"]: row for row in base}
        predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for prediction in _jsonl(directory / "item_predictions.jsonl"):
            predictions[prediction["trial_id"]].append(prediction)
        for row in _jsonl(path):
            if row["method"] not in allowed:
                continue
            trial_predictions = predictions[row["trial_id"]]
            if len(trial_predictions) != 46 or set(base_by_id) != {
                item["benchmark_item_id"] for item in trial_predictions
            }:
                raise ValueError(f"Prediction set mismatch: {row['trial_id']}")
            for item in trial_predictions:
                if item["answer_index"] != base_by_id[item["benchmark_item_id"]]["answer_index"]:
                    raise ValueError(f"Answer-key mismatch: {row['trial_id']}")
            trial_score = sum(_correct_log_likelihood(item) for item in trial_predictions) / 46
            base_score = sum(_correct_log_likelihood(item) for item in base) / 46
            record = dict(row)
            record["continuous_utility"] = trial_score - base_score
            record["source_directory"] = str(directory)
            rows.append(record)
    return rows, manifests


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rank(predicted: dict[str, float], reference: dict[str, float], margin: float) -> dict[str, Any] | None:
    if len(set(predicted.values())) <= 1 or len(set(reference.values())) <= 1:
        return None
    return evaluate_rankings(predicted, reference, top_ks=[1, 3], indifference_margin=margin)


def _metric_table(rows: list[dict[str, Any]], batch_ids: list[str], field: str, margin: float) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for method in METHODS:
        for batch_id in batch_ids:
            selected = [row for row in rows if row["method"] == method and row["batch_id"] == batch_id]
            values[method][batch_id] = _mean([float(row[field]) for row in selected])
    reference = values["same_generator_full_pool"]
    return {
        "reference_values": reference,
        "methods": {
            method: {"batch_utilities": values[method], "ranking": _rank(values[method], reference, margin)}
            for method in METHODS
        },
    }


def _generation_costs(root: Path, batch_ids: list[str]) -> dict[str, Any]:
    result = {}
    for method in ("raw_chunk_random", "raw_chunk_relevance"):
        manifests = [_json(root / batch_id / method / "generation_manifest.json") for batch_id in batch_ids]
        result[method] = {
            "all_batches_reached_target": all(row["target_reached"] for row in manifests),
            "accepted_samples": sum(row["accepted_sample_count"] for row in manifests),
            "selected_raw_chunks": sum(row["selected_chunk_count"] for row in manifests),
            "failed_or_rejected_chunks": sum(row["failure_count"] for row in manifests),
            "estimated_source_tokens": sum(row["used_source_tokens_estimated"] for row in manifests),
            "teacher_calls": sum(row["teacher_usage"]["teacher_calls"] for row in manifests),
            "teacher_cache_hits": sum(row["teacher_usage"]["cache_hits"] for row in manifests),
            "teacher_prompt_tokens": sum(row["teacher_usage"]["prompt_tokens"] for row in manifests),
            "teacher_output_tokens": sum(row["teacher_usage"]["output_tokens"] for row in manifests),
            "teacher_api_duration_seconds": sum(row["teacher_usage"]["api_duration_ns"] for row in manifests) / 1e9,
            "generation_wall_seconds": sum(row["method_seconds"] for row in manifests),
            "shared_parse_seconds": sum(row["parse_seconds_shared"] for row in manifests) / 2,
        }
    return result


def _lineage_utilities(metric: dict[str, Any], lineage_by_batch: dict[str, str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for batch_id, lineage in lineage_by_batch.items():
        groups[lineage].append(batch_id)
    return {
        method: {
            lineage: _mean([details["batch_utilities"][batch_id] for batch_id in batch_ids])
            for lineage, batch_ids in sorted(groups.items())
        }
        for method, details in metric["methods"].items()
    }


def _lineage_rankings(values: dict[str, dict[str, float]], margin: float) -> dict[str, Any]:
    reference = values["same_generator_full_pool"]
    return {method: _rank(utilities, reference, margin) for method, utilities in values.items()}


def _display_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "无法计算（效用无变异）"


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_batch_suite(args.suite.resolve())
    batch_ids = sorted(batch.batch_id for batch in suite.batches)
    old, old_manifests = _load_trials(
        args.confirmation_root.resolve(), {"coverage_quality", "same_generator_full_pool"}
    )
    new, new_manifests = _load_trials(args.raw_trials_root.resolve(), {"raw_chunk_random", "raw_chunk_relevance"})
    rows = old + new
    observed = [(row["batch_id"], row["method"], int(row["training_seed"])) for row in rows]
    expected = [(batch_id, method, seed) for batch_id in batch_ids for method in METHODS for seed in SEEDS]
    missing = sorted(set(expected) - set(observed))
    duplicate = len(observed) != len(set(observed))
    if missing or duplicate or len(observed) != len(expected):
        raise ValueError(
            f"Incomplete or duplicated trial matrix: {len(observed)}/{len(expected)}, missing={missing[:5]}"
        )
    model_revisions = {manifest["model_revision"] for manifest in [*old_manifests, *new_manifests]}
    if len(model_revisions) != 1:
        raise ValueError(f"Base model revision differs across trial matrices: {model_revisions}")
    if any(
        row["training"]["effective_supervised_tokens"] != 256 or row["training"]["optimizer_steps"] != 4
        for row in rows
        if row["method"] != "same_generator_full_pool"
    ):
        raise ValueError("Proxy training exposure is not matched at 256 tokens and four updates")
    accuracy = _metric_table(rows, batch_ids, "utility", 1 / 46)
    continuous = _metric_table(rows, batch_ids, "continuous_utility", 0.0)
    lineage_by_batch = {batch.batch_id: batch.lineage_group for batch in suite.batches}
    lineage_accuracy = _lineage_utilities(accuracy, lineage_by_batch)
    lineage_continuous = _lineage_utilities(continuous, lineage_by_batch)
    seed_accuracy = {
        str(seed): _metric_table([row for row in rows if row["training_seed"] == seed], batch_ids, "utility", 1 / 46)
        for seed in SEEDS
    }
    generation = _generation_costs(args.raw_sft_root.resolve(), batch_ids)
    training = {
        method: {
            "trial_count": sum(row["method"] == method for row in rows),
            "sum_training_wall_seconds": sum(
                row["training"]["wall_seconds"] for row in rows if row["method"] == method
            ),
        }
        for method in METHODS
    }
    result = {
        "schema_version": "raw-chunk-followup-analysis-v1",
        "scope": "12 local batches, 46 Task A items, Qwen2.5-1.5B, same-generator full-pool reference",
        "rev1_results_modified": False,
        "confirmatory_suite_c_comparison": False,
        "batch_suite_sha256": file_sha256(args.suite.resolve()),
        "model_revision": next(iter(model_revisions)),
        "batch_count": len(batch_ids),
        "effective_lineage_count": len({batch.lineage_group for batch in suite.batches}),
        "training_seeds": list(SEEDS),
        "trial_count": len(rows),
        "accuracy_primary": accuracy,
        "accuracy_by_source_lineage": lineage_accuracy,
        "accuracy_source_lineage_rankings": _lineage_rankings(lineage_accuracy, 1 / 46),
        "accuracy_by_training_seed": seed_accuracy,
        "log_likelihood_sensitivity": continuous,
        "log_likelihood_by_source_lineage": lineage_continuous,
        "log_likelihood_source_lineage_rankings": _lineage_rankings(lineage_continuous, 0.0),
        "raw_generation_costs": generation,
        "training_time_observed": training,
        "full_cq_mining_cost_comparison_available": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# 本地 raw-chunk 补充实验",
        "",
        f"范围：{result['scope']}。共有 {len(rows)} 个完整训练试验，{result['effective_lineage_count']} 个有效来源谱系。rev1 数值保持原样。",
        "",
        "| 方法 | Accuracy Spearman | Accuracy Regret@1 | Continuous Spearman | Continuous Regret@1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        a = accuracy["methods"][method]["ranking"]
        c = continuous["methods"][method]["ranking"]
        lines.append(
            f"| {method} | {_display_metric(a['raw_spearman'] if a else None)} | {_display_metric(a['regret']['1'] if a else None)} | {_display_metric(c['raw_spearman'] if c else None)} | {_display_metric(c['regret']['1'] if c else None)} |"
        )
    lines.extend(
        [
            "",
            "按四个来源谱系等权合并后的描述性结果：",
            "",
            "| 方法 | Accuracy Spearman | Accuracy Regret@1 | Continuous Spearman | Continuous Regret@1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in METHODS:
        a = result["accuracy_source_lineage_rankings"][method]
        c = result["log_likelihood_source_lineage_rankings"][method]
        lines.append(
            f"| {method} | {_display_metric(a['raw_spearman'] if a else None)} | {_display_metric(a['regret']['1'] if a else None)} | {_display_metric(c['raw_spearman'] if c else None)} | {_display_metric(c['regret']['1'] if c else None)} |"
        )
    lines.extend(
        [
            "",
            "Accuracy 只有 46 题，存在离散步长与并列；连续指标仅作敏感性分析。四个有效来源谱系不足以支持稳健的总体置信区间。",
            "",
            "Raw-chunk 的生成 token、调用与时间已单独计量；既有 CQ 工件没有同边界的完整 mining/API 成本，故本次不能声称等总成本优势。详见 analysis.json。",
        ]
    )
    (args.output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--raw-trials-root", type=Path, required=True)
    parser.add_argument("--raw-sft-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    print(
        json.dumps(
            {key: result[key] for key in ("scope", "batch_count", "trial_count", "effective_lineage_count")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
