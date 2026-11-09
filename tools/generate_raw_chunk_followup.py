"""Build direct raw-chunk SFT controls without the semantic-unit inventory.

This is a local 12-batch follow-up, separate from the rev1 paper's Suite C.
Both arms read frozen source documents before semantic mining, use the same
teacher prompt and quality gate as the local CQ generator, and stop after
enough response-only supervision exists for the matched 256-token trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from ai_ready.valuation.batches import batch_content_fingerprint
from ai_ready.valuation.contracts import file_sha256, load_batch_suite
from ai_ready.valuation.local_sft import (
    PROMPT_SHA256,
    QUALITY_GATE_SHA256,
    RETRY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    OllamaTeacher,
    TeacherOutput,
    _candidate_quality_error,
    _estimated_tokens,
    _load_blueprint,
    _parse_artifacts,
    _parse_teacher_output,
    _requested_task_type,
    _resolve_local_batch_path,
    _task_requirement,
)
from ai_ready.valuation.quality import SFTSample

METHODS = ("raw_chunk_random", "raw_chunk_relevance")
CONDITION_CUE = re.compile(r"\b(?:if|when|unless|except|in patients with|for patients with)\b", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _available_training_tokens(tokenizer: Any, samples: list[dict[str, Any]], max_length: int) -> int:
    """Match the screening runner's response-only truncation rule."""

    def ids(value: Any) -> list[int]:
        if isinstance(value, dict) or hasattr(value, "keys"):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return [int(token_id) for token_id in value]

    total = 0
    for row in samples:
        prompt = [{"role": "user", "content": row["instruction"]}]
        full = [*prompt, {"role": "assistant", "content": row["response"]}]
        prompt_ids = ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        full_ids = ids(tokenizer.apply_chat_template(full, tokenize=True, add_generation_prompt=False))[:max_length]
        total += max(0, len(full_ids) - min(len(prompt_ids), len(full_ids)))
    return total


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _split_raw_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split only at source sentence/space boundaries; return source offsets."""

    if len(text) <= max_chars:
        return [(0, text)]
    pieces: list[tuple[int, str]] = []
    cursor = 0
    boundaries = sorted({match.end() for match in SENTENCE_END.finditer(text)} | {len(text)})
    while cursor < len(text):
        candidates = [position for position in boundaries if cursor < position <= cursor + max_chars]
        end = max(candidates) if candidates else min(len(text), cursor + max_chars)
        if end < len(text) and not candidates:
            space = text.rfind(" ", cursor + 1, end)
            if space > cursor:
                end = space + 1
        raw = text[cursor:end]
        stripped = raw.strip()
        if stripped:
            offset = cursor + len(raw) - len(raw.lstrip())
            pieces.append((offset, stripped))
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    return pieces


def _raw_candidates(evidence_units: list[Any], max_chars: int) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for evidence in evidence_units:
        for offset, content in _split_raw_text(evidence.text, max_chars):
            if len(content.split()) < 8:
                continue
            content_hash = _digest(content)
            if content_hash in seen:
                duplicate_count += 1
                continue
            seen.add(content_hash)
            location = evidence.sources[0].locations[0]
            candidates.append(
                {
                    "candidate_id": _digest(f"{evidence.id}|{offset}|{content_hash}")[:24],
                    "evidence_id": evidence.id,
                    "content_sha256": content_hash,
                    "content": content,
                    "headings": evidence.headings,
                    "source_ids": sorted({source.locations[0]["source_id"] for source in evidence.sources}),
                    "artifact_path": location["artifact_path"],
                    "artifact_sha256": location["artifact_sha256"],
                    "paragraph_offset": offset,
                    "estimated_source_tokens": _estimated_tokens(content),
                }
            )
    return candidates, duplicate_count


def _rank(candidates: list[dict[str, Any]], method: str, queries: list[str], seed: int) -> list[dict[str, Any]]:
    if method == "raw_chunk_random":
        import random

        result = list(candidates)
        random.Random(seed).shuffle(result)
        return result
    if method != "raw_chunk_relevance":
        raise ValueError(f"Unknown raw-chunk method: {method}")
    if not candidates:
        return []
    texts = [f"{' '.join(row['headings'])} {row['content']}" for row in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([*texts, *queries])
    scores = (matrix[: len(texts)] @ matrix[len(texts) :].T).toarray().max(axis=1)
    return [
        candidates[index]
        for index in sorted(
            range(len(candidates)), key=lambda index: (-scores[index], candidates[index]["candidate_id"])
        )
    ]


class MeteredTeacher(OllamaTeacher):
    last_usage: dict[str, int]

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = super()._request(path, payload)
        if path == "/api/generate" and isinstance(response, dict):
            self.last_usage = {
                key: int(response.get(key) or 0)
                for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "eval_duration")
            }
        return response


def _requested_type(candidate: dict[str, Any]) -> str:
    content = candidate["content"]
    cue = [content] if CONDITION_CUE.search(content) else []
    raw_unit = SimpleNamespace(
        content=content,
        headings=candidate["headings"],
        content_sha256=candidate["content_sha256"],
        condition_spans=cue,
        exception_spans=[],
        uncertainty_spans=[],
    )
    return _requested_task_type(raw_unit, 1)


def _generate_one(
    candidate: dict[str, Any],
    *,
    batch_id: str,
    method: str,
    teacher: MeteredTeacher,
    teacher_metadata: dict[str, str],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    max_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, int], str | None]:
    requested_type = _requested_type(candidate)
    prompt = USER_PROMPT_TEMPLATE.format(
        requested_task_type=requested_type,
        task_requirement=_task_requirement(requested_type),
        sample_index=1,
        headings_json=json.dumps(candidate["headings"], ensure_ascii=False),
        content_json=json.dumps(candidate["content"], ensure_ascii=False),
    )
    cache_key = _digest(
        json.dumps(
            {
                "teacher_digest": teacher_metadata["model_digest"],
                "prompt_sha256": PROMPT_SHA256,
                "content_sha256": candidate["content_sha256"],
                "headings": candidate["headings"],
                "requested_type": requested_type,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    usage = {"teacher_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "output_tokens": 0, "api_duration_ns": 0}
    if cache_key in cache:
        output = TeacherOutput.model_validate(cache[cache_key])
        usage["cache_hits"] = 1
    else:
        output = None
        last_error = "generation_failed"
        for attempt in range(1, max_retries + 1):
            teacher.last_usage = {
                key: 0
                for key in ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "eval_duration")
            }
            try:
                call_prompt = (
                    prompt
                    if attempt == 1
                    else prompt + RETRY_PROMPT_TEMPLATE.format(attempt=attempt, error_code=last_error)
                )
                output = _parse_teacher_output(teacher(SYSTEM_PROMPT, call_prompt))
                usage["teacher_calls"] += 1
                usage["prompt_tokens"] += teacher.last_usage["prompt_eval_count"]
                usage["output_tokens"] += teacher.last_usage["eval_count"]
                usage["api_duration_ns"] += teacher.last_usage["total_duration"]
                last_error = _candidate_quality_error(output, candidate["content"], requested_type) or ""
                if not last_error:
                    cache[cache_key] = output.model_dump(mode="json")
                    _append_jsonl(cache_path, {"cache_key": cache_key, "output": cache[cache_key]})
                    break
                output = None
            except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as error:
                usage["teacher_calls"] += 1
                usage["prompt_tokens"] += teacher.last_usage["prompt_eval_count"]
                usage["output_tokens"] += teacher.last_usage["eval_count"]
                usage["api_duration_ns"] += teacher.last_usage["total_duration"]
                last_error = type(error).__name__
        if output is None:
            return None, usage, last_error
    if _candidate_quality_error(output, candidate["content"], requested_type):
        return None, usage, "cached_quality_failure"
    sample = SFTSample(
        sample_id=_digest(f"{batch_id}|{method}|{candidate['candidate_id']}|{cache_key}")[:24],
        batch_id=batch_id,
        semantic_unit_ids=[candidate["candidate_id"]],
        instruction=output.instruction.strip(),
        response=output.evidence_span.strip(),
        evidence_spans=[output.evidence_span],
        source_ids=candidate["source_ids"],
        task_type=output.task_type,
        generator="raw_chunk_local_followup",
        generator_revision=file_sha256(Path(__file__)),
        teacher_model=teacher_metadata["model"],
        teacher_revision=teacher_metadata["model_digest"],
        prompt_sha256=PROMPT_SHA256,
        quality_gate_sha256=QUALITY_GATE_SHA256,
    )
    return sample.model_dump(mode="json"), usage, None


def _run(args: argparse.Namespace) -> None:
    suite_path = args.suite.resolve()
    suite = load_batch_suite(suite_path)
    blueprint = _load_blueprint(args.blueprint.resolve(), suite)
    queries = [f"{row['capability_id'].replace('_', ' ')} {row['description']}" for row in blueprint]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path.resolve(), local_files_only=True)
    teacher = MeteredTeacher(model=args.teacher_model, seed=args.teacher_seed)
    teacher_metadata = teacher.runtime_metadata()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_root / "teacher_cache.jsonl"
    cache = (
        {
            row["cache_key"]: row["output"]
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))
        }
        if cache_path.exists()
        else {}
    )
    selected_batches = set(args.batches or [batch.batch_id for batch in suite.batches])
    for batch in suite.batches:
        if batch.batch_id not in selected_batches:
            continue
        started = time.perf_counter()
        batch_dir = _resolve_local_batch_path(batch.input_uri, suite_path)
        if batch_content_fingerprint(suite, batch, batch_dir) != batch.content_sha256:
            raise ValueError(f"Frozen batch hash mismatch: {batch.batch_id}")
        evidence = _parse_artifacts(batch_dir, batch, suite)
        candidates, duplicates = _raw_candidates(evidence, args.max_chunk_chars)
        parse_seconds = time.perf_counter() - started
        candidate_digest = _digest(json.dumps(candidates, sort_keys=True, ensure_ascii=False))
        for method in args.methods:
            destination = args.output_root / batch.batch_id / method
            if (destination / "generation_manifest.json").exists():
                print(f"skip {batch.batch_id}/{method}: complete", flush=True)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            ordered = _rank(candidates, method, queries, args.selection_seed)
            samples: list[dict[str, Any]] = []
            selected: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            seen_supervision: set[tuple[str, str]] = set()
            stats = {"teacher_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "output_tokens": 0, "api_duration_ns": 0}
            used_source_tokens = 0
            method_started = time.perf_counter()
            for candidate in ordered[: args.max_chunks]:
                if used_source_tokens + candidate["estimated_source_tokens"] > args.max_source_tokens:
                    continue
                selected.append({key: value for key, value in candidate.items() if key != "content"})
                used_source_tokens += candidate["estimated_source_tokens"]
                sample, usage, error = _generate_one(
                    candidate,
                    batch_id=batch.batch_id,
                    method=method,
                    teacher=teacher,
                    teacher_metadata=teacher_metadata,
                    cache=cache,
                    cache_path=cache_path,
                    max_retries=args.max_retries,
                )
                for key in stats:
                    stats[key] += usage[key]
                if sample is None:
                    failures.append({"candidate_id": candidate["candidate_id"], "reason": str(error)})
                    continue
                signature = (sample["instruction"].casefold().strip(), sample["response"].casefold().strip())
                if signature in seen_supervision:
                    failures.append({"candidate_id": candidate["candidate_id"], "reason": "duplicate_supervision"})
                    continue
                seen_supervision.add(signature)
                samples.append(sample)
                if (
                    _available_training_tokens(tokenizer, samples, args.max_train_length)
                    >= args.target_supervised_tokens
                ):
                    break
            sft_path = destination / "accepted_sft.jsonl"
            with sft_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            _write_json(destination / "selected_raw_chunks.json", selected)
            _write_json(destination / "failures.json", failures)
            available_tokens = _available_training_tokens(tokenizer, samples, args.max_train_length)
            manifest = {
                "schema_version": "raw-chunk-followup-v1",
                "batch_id": batch.batch_id,
                "batch_content_sha256": batch.content_sha256,
                "method": method,
                "selection_seed": args.selection_seed if method == "raw_chunk_random" else None,
                "candidate_count": len(candidates),
                "duplicate_content_chunks_excluded": duplicates,
                "candidate_inventory_sha256": candidate_digest,
                "selected_chunk_count": len(selected),
                "used_source_tokens_estimated": used_source_tokens,
                "accepted_sample_count": len(samples),
                "available_supervised_tokens": available_tokens,
                "target_supervised_tokens": args.target_supervised_tokens,
                "target_reached": available_tokens >= args.target_supervised_tokens,
                "failure_count": len(failures),
                "teacher": teacher_metadata,
                "prompt_sha256": PROMPT_SHA256,
                "quality_gate_sha256": QUALITY_GATE_SHA256,
                "teacher_usage": stats,
                "parse_seconds_shared": parse_seconds,
                "method_seconds": time.perf_counter() - method_started,
                "accepted_sft_sha256": file_sha256(sft_path),
                "benchmark_items_accessed": False,
            }
            _write_json(destination / "generation_manifest.json", manifest)
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in (
                            "batch_id",
                            "method",
                            "selected_chunk_count",
                            "accepted_sample_count",
                            "available_supervised_tokens",
                            "target_reached",
                            "method_seconds",
                        )
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batches", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--selection-seed", type=int, default=101)
    parser.add_argument("--teacher-model", default="qwen2.5:1.5b-instruct")
    parser.add_argument("--teacher-seed", type=int, default=42)
    parser.add_argument("--target-supervised-tokens", type=int, default=256)
    parser.add_argument("--max-train-length", type=int, default=512)
    parser.add_argument("--max-source-tokens", type=int, default=100000)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=3)
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
