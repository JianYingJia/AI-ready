"""Run the complete local matched-exposure confirmation suite.

The core matrix contains 12 batches, seven construction/reference arms, and
three training seeds. Random and stratified-random selectors receive four
additional construction seeds at training seed 17. Each sub-run is resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _completed_cells(path: Path) -> set[tuple[str, str, int, int | None]]:
    if not path.is_file():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        (row["batch_id"], row["method"], int(row["training_seed"]), row.get("selection_seed"))
        for row in rows
        if row.get("status") == "completed"
    }


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--max-trials", type=int)
    args = parser.parse_args()

    runner = Path(__file__).with_name("run_local_proxy_screening.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_ids = args.batches or [f"B{index:02d}" for index in range(1, 13)]
    core_methods = (
        "coverage_quality",
        "coverage_only",
        "random",
        "stratified_random",
        "relevance",
        "diversity",
        "same_generator_full_pool",
    )
    for training_seed in args.training_seeds:
        output = args.output_dir / f"training_seed_{training_seed}_selection_seed_{args.selection_seeds[0]}"
        for batch_group in _chunks(batch_ids, 1):
            completed = _completed_cells(output / "trial_results.jsonl")
            expected = {
                (
                    batch_id,
                    method,
                    training_seed,
                    args.selection_seeds[0] if method in {"random", "stratified_random"} else None,
                )
                for batch_id in batch_group
                for method in core_methods
            }
            if expected <= completed:
                continue
            command = [
                sys.executable,
                str(runner),
                "--config",
                str(args.config),
                "--output-dir",
                str(output),
                "--training-seed",
                str(training_seed),
                "--selection-seed",
                str(args.selection_seeds[0]),
                "--batches",
                *batch_group,
            ]
            if args.max_trials is not None:
                command.extend(["--max-trials", str(args.max_trials)])
            _run(command)

    for selection_seed in args.selection_seeds[1:]:
        output = args.output_dir / f"training_seed_{args.training_seeds[0]}_selection_seed_{selection_seed}"
        for batch_group in _chunks(batch_ids, 1):
            completed = _completed_cells(output / "trial_results.jsonl")
            expected = {
                (batch_id, method, args.training_seeds[0], selection_seed)
                for batch_id in batch_group
                for method in ("random", "stratified_random")
            }
            if expected <= completed:
                continue
            command = [
                sys.executable,
                str(runner),
                "--config",
                str(args.config),
                "--output-dir",
                str(output),
                "--training-seed",
                str(args.training_seeds[0]),
                "--selection-seed",
                str(selection_seed),
                "--methods",
                "random",
                "stratified_random",
                "--batches",
                *batch_group,
            ]
            if args.max_trials is not None:
                command.extend(["--max-trials", str(args.max_trials)])
            _run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
