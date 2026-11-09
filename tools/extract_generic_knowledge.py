"""Extract domain-general, evidence-bound knowledge units with local LLM panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_ready.valuation.generic_knowledge import (
    ExtractionConfig,
    run_generic_knowledge_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="EvidenceUnit JSONL or Parquet produced by L1")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extraction-model", action="append", required=True)
    parser.add_argument("--consolidation-model", required=True)
    parser.add_argument("--verification-model", action="append", required=True)
    parser.add_argument("--adjudicator-model", action="append")
    parser.add_argument("--extraction-runs-per-model", type=int, default=2)
    parser.add_argument("--verification-runs-per-model", type=int, default=2)
    parser.add_argument("--adjudication-runs-per-model", type=int, default=3)
    parser.add_argument("--minimum-verifier-support", type=float, default=2 / 3)
    parser.add_argument("--minimum-verifier-confidence", type=float, default=0.60)
    parser.add_argument("--base-seed", type=int, default=23)
    parser.add_argument("--extraction-temperature", type=float, default=0.1)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    config = ExtractionConfig(
        extraction_runs_per_model=args.extraction_runs_per_model,
        verification_runs_per_model=args.verification_runs_per_model,
        adjudication_runs_per_model=args.adjudication_runs_per_model,
        minimum_verifier_support=args.minimum_verifier_support,
        minimum_verifier_confidence=args.minimum_verifier_confidence,
        base_seed=args.base_seed,
        extraction_temperature=args.extraction_temperature,
    )
    result = run_generic_knowledge_pipeline(
        records_path=args.records,
        output_dir=args.output_dir,
        batch_id=args.batch_id,
        extraction_models=args.extraction_model,
        consolidation_model=args.consolidation_model,
        verification_models=args.verification_model,
        adjudicator_models=args.adjudicator_model,
        config=config,
        ollama_url=args.ollama_url,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
