"""Run the frozen independent four-choice probe set through a local Ollama model.

This runner exists for the model-necessity screening experiment.  It reuses the
same probe catalogue, prompt variants, strict A-D parser, and exact-index scorer
as the Transformers pipeline, while binding outputs to the Ollama model digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ai_ready.utils.io import write_json, write_jsonl
from ai_ready.valuation.contracts import file_sha256
from ai_ready.valuation.model_probe import (
    ProbePrediction,
    build_probe_prompts,
    canonical_sha256,
    load_probe_catalogue,
    parse_answer_choice,
    probe_catalogue_digest,
    prompt_protocol_digest,
    score_probe_predictions,
)


def _request_json(base_url: str, path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Ollama {path} returned a non-object response")
    return result


def _model_metadata(base_url: str, model: str) -> dict:
    tags = _request_json(base_url, "/api/tags")
    matches = [row for row in tags.get("models", []) if row.get("name") == model or row.get("model") == model]
    if len(matches) != 1:
        raise ValueError(f"Expected one installed Ollama model named {model!r}, found {len(matches)}")
    row = matches[0]
    digest = row.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Installed Ollama model has no full immutable digest")
    show = _request_json(base_url, "/api/show", {"model": model})
    version = _request_json(base_url, "/api/version")
    return {
        "name": model,
        "digest": digest,
        "size": row.get("size"),
        "modified_at": row.get("modified_at"),
        "details": row.get("details"),
        "template_sha256": hashlib.sha256(str(show.get("template", "")).encode("utf-8")).hexdigest(),
        "ollama_version": version.get("version"),
    }


def run(args: argparse.Namespace) -> None:
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    catalogue = load_probe_catalogue(args.catalogue)
    variants = ["clinical_mcq_letter_v1", "clinical_mcq_letter_v2"]
    prompts = build_probe_prompts(catalogue.probes, variants)
    probe_set_sha256 = probe_catalogue_digest(catalogue)
    model = _model_metadata(args.base_url, args.model)
    predictions = []
    response_audit = []
    for index, prompt in enumerate(prompts, start=1):
        payload = {
            "model": args.model,
            "messages": [message.model_dump(mode="json") for message in prompt.messages],
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": args.seed,
                "num_predict": args.max_new_tokens,
            },
        }
        response = _request_json(args.base_url, "/api/chat", payload, timeout=args.timeout)
        content = str(response.get("message", {}).get("content", ""))
        parsed = parse_answer_choice(content)
        original_index = None if parsed.answer_index is None else prompt.option_order[parsed.answer_index]
        prediction = ProbePrediction(
            prediction_id=f"{prompt.item_id}:{prompt.variant_id}",
            item_id=prompt.item_id,
            variant_id=prompt.variant_id,
            cell_id=prompt.cell_id,
            model_id=args.model,
            model_revision=model["digest"],
            checkpoint_sha256=model["digest"],
            probe_set_sha256=probe_set_sha256,
            prompt_sha256=canonical_sha256(payload["messages"]),
            option_order=prompt.option_order,
            raw_output=content,
            parse_status=parsed.status,
            parsed_answer_letter=parsed.answer_letter,
            parsed_display_index=parsed.answer_index,
            parsed_answer_index=original_index,
        )
        predictions.append(prediction)
        response_audit.append(
            {
                "prediction_id": prediction.prediction_id,
                "done_reason": response.get("done_reason"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "total_duration_ns": response.get("total_duration"),
            }
        )
        if index % 10 == 0 or index == len(prompts):
            print(f"completed {index}/{len(prompts)}", flush=True)
    scored = score_probe_predictions(catalogue.probes, predictions)
    predictions_path = destination / "raw_predictions.jsonl"
    scored_path = destination / "scored_probes.jsonl"
    audit_path = destination / "response_audit.jsonl"
    write_jsonl(predictions_path, (row.model_dump(mode="json") for row in predictions))
    write_jsonl(scored_path, (row.model_dump(mode="json") for row in scored))
    write_jsonl(audit_path, response_audit)
    counts = {
        status: sum(row.parse_status == status for row in predictions) for status in ("valid", "invalid", "ambiguous")
    }
    manifest = {
        "schema_version": "ollama-base-model-probe-run-v1",
        "purpose": "model_relevance_necessity_screening",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "probe_catalogue": str(args.catalogue.resolve()),
        "probe_catalogue_sha256": file_sha256(args.catalogue.resolve()),
        "probe_set_sha256": probe_set_sha256,
        "prompt_protocol_sha256": prompt_protocol_digest(variants),
        "prompt_variant_ids": variants,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(prompts),
        "parse_counts": counts,
        "correct_prediction_count": int(sum(row.score for row in scored)),
        "accuracy": sum(row.score for row in scored) / len(scored),
        "benchmark_test_items_accessed": False,
        "locked_test_items_used": False,
        "outputs": {
            "raw_predictions": {"path": predictions_path.name, "sha256": file_sha256(predictions_path)},
            "scored_probes": {"path": scored_path.name, "sha256": file_sha256(scored_path)},
            "response_audit": {"path": audit_path.name, "sha256": file_sha256(audit_path)},
        },
    }
    write_json(destination / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    run(parser.parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
