"""Prepare annotations and build a generic benchmark knowledge--capability blueprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_ready.valuation.benchmark_blueprint import (
    ConsensusConfig,
    build_benchmark_blueprint,
    prepare_annotation_prompts,
    run_automated_blueprint_pipeline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Render backend-neutral JSONL prompts for public benchmark items")
    prepare.add_argument("--items", type=Path, required=True)
    prepare.add_argument("--ontology", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build", help="Validate annotations, form consensus, and freeze the blueprint")
    build.add_argument("--items", type=Path, required=True)
    build.add_argument("--ontology", type=Path, required=True)
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--minimum-annotations", type=int, default=2)
    build.add_argument("--minimum-agreement", type=float, default=2 / 3)
    build.add_argument("--minimum-mean-confidence", type=float, default=0.60)
    build.add_argument("--require-structure-consensus", action="store_true")

    automatic = subparsers.add_parser(
        "run-ollama", help="Run multi-LLM annotation, automatic adjudication, and blueprint freezing"
    )
    automatic.add_argument("--items", type=Path, required=True)
    automatic.add_argument("--ontology", type=Path, required=True)
    automatic.add_argument("--output-dir", type=Path, required=True)
    automatic.add_argument("--annotator-model", action="append", required=True)
    automatic.add_argument("--adjudicator-model", action="append")
    automatic.add_argument("--runs-per-model", type=int, default=3)
    automatic.add_argument("--adjudication-runs-per-model", type=int, default=3)
    automatic.add_argument("--minimum-annotations", type=int, default=3)
    automatic.add_argument("--minimum-agreement", type=float, default=2 / 3)
    automatic.add_argument("--minimum-mean-confidence", type=float, default=0.60)
    automatic.add_argument("--require-structure-consensus", action="store_true")
    automatic.add_argument("--base-seed", type=int, default=17)
    automatic.add_argument("--temperature", type=float, default=0.0)
    automatic.add_argument("--adjudication-temperature", type=float, default=0.0)
    automatic.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_annotation_prompts(args.items, args.ontology, args.output)
    elif args.command == "build":
        config = ConsensusConfig(
            minimum_annotations=args.minimum_annotations,
            minimum_agreement=args.minimum_agreement,
            minimum_mean_confidence=args.minimum_mean_confidence,
            require_structure_consensus=args.require_structure_consensus,
        )
        result = build_benchmark_blueprint(
            items_path=args.items,
            ontology_path=args.ontology,
            annotations_path=args.annotations,
            output_dir=args.output_dir,
            config=config,
        )
    else:
        config = ConsensusConfig(
            minimum_annotations=args.minimum_annotations,
            minimum_agreement=args.minimum_agreement,
            minimum_mean_confidence=args.minimum_mean_confidence,
            require_structure_consensus=args.require_structure_consensus,
        )
        result = run_automated_blueprint_pipeline(
            items_path=args.items,
            ontology_path=args.ontology,
            output_dir=args.output_dir,
            annotator_models=args.annotator_model,
            adjudicator_models=args.adjudicator_model,
            runs_per_model=args.runs_per_model,
            adjudication_runs_per_model=args.adjudication_runs_per_model,
            config=config,
            base_seed=args.base_seed,
            temperature=args.temperature,
            adjudication_temperature=args.adjudication_temperature,
            ollama_url=args.ollama_url,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
