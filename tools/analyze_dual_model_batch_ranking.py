"""Compare batch-value rankings across two base models.

Both model roots must contain the same 12 batches, the coverage-quality proxy,
the same-generator full-pool reference, three training seeds, and item-level
predictions produced by ``run_local_proxy_screening.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_local_confirmation_suite import (
    REFERENCE_METHOD,
    TARGET_METHOD,
    _attach_continuous_utilities,
    _load_suite,
    _mean_vectors,
    _ranking,
)


def _model_report(root: Path, batch_ids: list[str]) -> dict[str, Any]:
    trials, predictions, bases = _load_suite(root)
    rows = [
        row
        for row in trials
        if row["_manifest_selection_seed"] == 101 and row["method"] in {TARGET_METHOD, REFERENCE_METHOD}
    ]
    expected = {
        (batch_id, method, seed)
        for batch_id in batch_ids
        for method in (TARGET_METHOD, REFERENCE_METHOD)
        for seed in (17, 29, 43)
    }
    observed = {(row["batch_id"], row["method"], int(row["training_seed"])) for row in rows}
    audit = {
        "expected": len(expected),
        "observed": len(observed),
        "missing": [list(value) for value in sorted(expected - observed)],
        "complete": expected == observed,
    }
    if not audit["complete"]:
        return {"audit": audit}

    _attach_continuous_utilities(rows, predictions, bases)
    accuracy = _mean_vectors(rows, "utility")
    continuous = _mean_vectors(rows, "continuous_utility")
    content_static = _mean_vectors(rows, "selection.content_static_score")[TARGET_METHOD]
    frozen = _mean_vectors(rows, "training.frozen_model_score")[TARGET_METHOD]
    return {
        "audit": audit,
        "accuracy": {
            "proxy": accuracy[TARGET_METHOD],
            "reference": accuracy[REFERENCE_METHOD],
            "proxy_ranking": _ranking(accuracy[TARGET_METHOD], accuracy[REFERENCE_METHOD], 1 / 46),
            "task_aware_static_ranking": _ranking(content_static, accuracy[REFERENCE_METHOD], 1 / 46),
            "frozen_model_ranking": _ranking(frozen, accuracy[REFERENCE_METHOD], 1 / 46),
        },
        "normalized_mcq_log_likelihood": {
            "proxy": continuous[TARGET_METHOD],
            "reference": continuous[REFERENCE_METHOD],
            "proxy_ranking": _ranking(continuous[TARGET_METHOD], continuous[REFERENCE_METHOD], 0.0),
            "task_aware_static_ranking": _ranking(content_static, continuous[REFERENCE_METHOD], 0.0),
            "frozen_model_ranking": _ranking(frozen, continuous[REFERENCE_METHOD], 0.0),
        },
        "task_aware_static": content_static,
        "frozen_model": frozen,
    }


def analyze(model_a_root: Path, model_b_root: Path, batch_ids: list[str]) -> dict[str, Any]:
    model_a = _model_report(model_a_root, batch_ids)
    model_b = _model_report(model_b_root, batch_ids)
    if not model_a["audit"]["complete"] or not model_b["audit"]["complete"]:
        return {
            "schema_version": "dual-model-batch-ranking-v1",
            "status": "incomplete",
            "model_a": model_a,
            "model_b": model_b,
        }

    reversals = []
    for index, left in enumerate(batch_ids):
        for right in batch_ids[index + 1 :]:
            reference_gap_a = (
                model_a["normalized_mcq_log_likelihood"]["reference"][left]
                - model_a["normalized_mcq_log_likelihood"]["reference"][right]
            )
            reference_gap_b = (
                model_b["normalized_mcq_log_likelihood"]["reference"][left]
                - model_b["normalized_mcq_log_likelihood"]["reference"][right]
            )
            if reference_gap_a * reference_gap_b >= 0:
                continue
            proxy_gap_a = (
                model_a["normalized_mcq_log_likelihood"]["proxy"][left]
                - model_a["normalized_mcq_log_likelihood"]["proxy"][right]
            )
            proxy_gap_b = (
                model_b["normalized_mcq_log_likelihood"]["proxy"][left]
                - model_b["normalized_mcq_log_likelihood"]["proxy"][right]
            )
            static_gap = model_a["task_aware_static"][left] - model_a["task_aware_static"][right]
            reversals.append(
                {
                    "left": left,
                    "right": right,
                    "reference_gap_model_a": reference_gap_a,
                    "reference_gap_model_b": reference_gap_b,
                    "proxy_gap_model_a": proxy_gap_a,
                    "proxy_gap_model_b": proxy_gap_b,
                    "task_aware_static_gap": static_gap,
                    "proxy_recovers_both_directions": (
                        proxy_gap_a * reference_gap_a > 0 and proxy_gap_b * reference_gap_b > 0
                    ),
                    "static_can_express_reversal": False,
                }
            )

    return {
        "schema_version": "dual-model-batch-ranking-v1",
        "status": "complete",
        "model_a": model_a,
        "model_b": model_b,
        "model_conditioned_reference_reversals": reversals,
        "reference_reversal_count": len(reversals),
        "proxy_recovered_reversal_count": sum(row["proxy_recovers_both_directions"] for row in reversals),
        "interpretation_guard": (
            "Preference reversals are descriptive over 12 batches and two Qwen2.5 scales. "
            "They establish model conditioning only when the full-reference directions reverse; "
            "proxy recovery is assessed against both model-specific references."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-root", type=Path, required=True)
    parser.add_argument("--model-b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batches",
        nargs="+",
        default=[f"B{index:02d}" for index in range(1, 13)],
    )
    args = parser.parse_args()
    report = analyze(args.model_a_root, args.model_b_root, args.batches)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
