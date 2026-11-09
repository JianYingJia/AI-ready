"""Analyze matched-exposure component ablations against the full-pool reference."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from analyze_local_confirmation_suite import (
    _attach_continuous_utilities,
    _load_suite,
    _mean_vectors,
    _percentile,
)

from ai_ready.valuation.decision_experiments import evaluate_rankings

METHODS = ("coverage_quality", "no_dependency_grouping", "no_source_coverage")
REFERENCE_METHOD = "same_generator_full_pool"


def _interval(values: list[float]) -> dict[str, Any]:
    return {
        "median": _percentile(values, 0.5),
        "percentile_95_interval": [_percentile(values, 0.025), _percentile(values, 0.975)],
        "fraction_positive": sum(value > 0 for value in values) / len(values),
    }


def _paired_bootstrap(
    target: dict[tuple[str, int], float],
    baseline: dict[tuple[str, int], float],
    replicates: int,
) -> dict[str, Any]:
    batch_ids = sorted({batch_id for batch_id, _ in target})
    seeds = sorted({seed for _, seed in target})
    rng = random.Random(20261026)
    values = []
    for _ in range(replicates):
        sampled_batches = [rng.choice(batch_ids) for _ in batch_ids]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        differences = [
            target[(batch_id, seed)] - baseline[(batch_id, seed)]
            for batch_id in sampled_batches
            for seed in sampled_seeds
        ]
        values.append(sum(differences) / len(differences))
    return _interval(values)


def analyze(core_root: Path, ablation_root: Path, replicates: int) -> dict[str, Any]:
    core, core_predictions, core_bases = _load_suite(core_root)
    ablations, ablation_predictions, ablation_bases = _load_suite(ablation_root)
    _attach_continuous_utilities(core, core_predictions, core_bases)
    _attach_continuous_utilities(ablations, ablation_predictions, ablation_bases)

    core = [row for row in core if row["_manifest_selection_seed"] == 101]
    rows = [row for row in core if row["method"] in {"coverage_quality", REFERENCE_METHOD}]
    rows.extend(ablations)
    batch_ids = sorted({row["batch_id"] for row in rows})
    seeds = (17, 29, 43)
    expected = {
        (batch_id, method, seed) for batch_id in batch_ids for method in (*METHODS, REFERENCE_METHOD) for seed in seeds
    }
    observed = {(row["batch_id"], row["method"], row["training_seed"]) for row in rows}
    if missing := expected - observed:
        return {"status": "incomplete", "missing": [list(value) for value in sorted(missing)]}

    output: dict[str, Any] = {
        "schema_version": "component-ablation-report-v1",
        "status": "complete",
        "matrix": {"expected": len(expected), "observed": len(observed), "missing": []},
        "endpoints": {},
    }
    for endpoint in ("utility", "continuous_utility"):
        means = _mean_vectors(rows, endpoint)
        reference = means[REFERENCE_METHOD]
        rankings = {
            method: evaluate_rankings(means[method], reference, top_ks=[1, 3], indifference_margin=0.0)
            for method in METHODS
        }
        cell_values = {
            method: {
                (row["batch_id"], row["training_seed"]): float(row[endpoint]) for row in rows if row["method"] == method
            }
            for method in METHODS
        }
        output["endpoints"][endpoint] = {
            "rankings": rankings,
            "paired_difference_vs_coverage_quality": {
                method: _paired_bootstrap(cell_values[method], cell_values["coverage_quality"], replicates)
                for method in METHODS[1:]
            },
        }

    by_cell = {(row["batch_id"], row["method"], row["training_seed"]): row for row in rows}
    identifiability = {}
    for method in METHODS[1:]:
        changed = []
        for batch_id in batch_ids:
            per_seed = []
            for seed in seeds:
                target = by_cell[(batch_id, "coverage_quality", seed)]
                ablation = by_cell[(batch_id, method, seed)]
                per_seed.append(
                    target["selection"].get("selection_sha256") != ablation["selection"].get("selection_sha256")
                    and not (
                        target["training"]["encoded_sample_count"] == ablation["training"]["encoded_sample_count"]
                        and target["training"]["frozen_model_score"] == ablation["training"]["frozen_model_score"]
                    )
                )
            changed.append({"batch_id": batch_id, "identifiable_all_seeds": all(per_seed)})
        identifiability[method] = {
            "identifiable_batch_count": sum(row["identifiable_all_seeds"] for row in changed),
            "batches": changed,
        }
    output["identifiability"] = identifiability
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    report = analyze(args.core_root, args.ablation_root, args.bootstrap_replicates)
    output = args.ablation_root / "component_ablation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
