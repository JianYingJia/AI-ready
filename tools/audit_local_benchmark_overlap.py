"""Audit locked local benchmark identity and text overlap with source/SFT.

Scores are screening signals, not proof that a pretrained base model never saw
public MedMCQA items. The report omits benchmark question text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_ready.valuation.contracts import file_sha256, load_batch_suite
from ai_ready.valuation.local_sft import _parse_artifacts, _resolve_local_batch_path

TOKEN = re.compile(r"[a-z0-9]+", re.I)


def _tokens(value: str) -> list[str]:
    return TOKEN.findall(value.casefold())


def _shingles(value: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = _tokens(value)
    if len(tokens) < width:
        return set()
    return {tuple(tokens[start : start + width]) for start in range(len(tokens) - width + 1)}


def _overlap(question: str, candidate: str) -> tuple[float, bool]:
    question_tokens = _tokens(question)
    candidate_tokens = _tokens(candidate)
    if not question_tokens or not candidate_tokens:
        return 0.0, False
    exact = question_tokens == candidate_tokens
    qshingles = _shingles(question)
    cshingles = _shingles(candidate)
    score = 1.0 if exact else len(qshingles & cshingles) / len(qshingles) if qshingles else 0.0
    return score, exact


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_records(suite_path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    suite = load_batch_suite(suite_path)
    records = []
    counts: dict[str, int] = {}
    for batch in suite.batches:
        batch_dir = _resolve_local_batch_path(batch.input_uri, suite_path)
        units = _parse_artifacts(batch_dir, batch, suite)
        counts[batch.batch_id] = len(units)
        for unit in units:
            location = unit.sources[0].locations[0]
            records.append(
                {
                    "batch_id": batch.batch_id,
                    "path": str(batch_dir / location["artifact_path"]),
                    "text": unit.text,
                    "text_sha256": _hash(unit.text),
                }
            )
    return records, counts


def _sft_records(roots: list[Path]) -> tuple[list[dict[str, str]], dict[str, int]]:
    records = []
    counts: dict[str, int] = defaultdict(int)
    for root in roots:
        for path in root.glob("**/accepted_sft.jsonl"):
            for row in _jsonl(path):
                if not row.get("instruction"):
                    continue
                counts[str(root)] += 1
                records.append(
                    {
                        "batch_id": str(row.get("batch_id", "")),
                        "path": str(path),
                        "text": str(row["instruction"]),
                        "text_sha256": _hash(str(row["instruction"])),
                    }
                )
    return records, dict(counts)


def _top_matches(question: str, records: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    best = {"fivegram_recall": 0.0, "exact_question": False, "match": None}
    count_above = 0
    for row in records:
        score, exact = _overlap(question, row["text"])
        if score >= threshold or exact:
            count_above += 1
        if score > best["fivegram_recall"] or exact and not best["exact_question"]:
            best = {
                "fivegram_recall": round(score, 4),
                "exact_question": exact,
                "match": {key: row[key] for key in ("batch_id", "path", "text_sha256")},
            }
    best["count_at_or_above_threshold"] = count_above
    return best


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_path = (manifest_path.parent / manifest["payload_uri"]).resolve()
    rows = _jsonl(benchmark_path)
    ids = [str(row["benchmark_item_id"]) for row in rows]
    if len(rows) != manifest["item_count"] or len(ids) != len(set(ids)):
        raise ValueError("Benchmark count or item IDs violate the locked manifest")
    actual_hash = file_sha256(benchmark_path)
    if actual_hash != manifest["payload_sha256"]:
        raise ValueError("Locked benchmark payload hash mismatch")
    suite = load_batch_suite(args.suite.resolve())
    blueprint_hash = file_sha256(args.blueprint.resolve())
    if blueprint_hash != suite.capability_blueprint_sha256:
        raise ValueError("Blueprint hash does not match frozen batch suite")
    sources, source_counts = _source_records(args.suite.resolve())
    sft, sft_counts = _sft_records([path.resolve() for path in args.sft_roots])
    questions = [str(row["question"]) for row in rows]
    item_results = []
    for index, row in enumerate(rows):
        other_questions = [
            {"batch_id": "benchmark", "path": str(benchmark_path), "text": text, "text_sha256": _hash(text)}
            for other, text in enumerate(questions)
            if other != index
        ]
        item_results.append(
            {
                "benchmark_item_id": row["benchmark_item_id"],
                "question_sha256": _hash(questions[index]),
                "source_overlap": _top_matches(questions[index], sources, args.threshold),
                "sft_instruction_overlap": _top_matches(questions[index], sft, args.threshold),
                "within_benchmark_overlap": _top_matches(questions[index], other_questions, args.threshold),
            }
        )
    result = {
        "schema_version": "local-benchmark-overlap-audit-v1",
        "scope": "12-batch 46-item local follow-up; not rev1 Suite C 200-item audit",
        "benchmark_manifest": str(manifest_path),
        "benchmark_manifest_sha256": file_sha256(manifest_path),
        "benchmark_payload_sha256": actual_hash,
        "benchmark_item_count": len(rows),
        "short_question_count_below_five_tokens": sum(len(_tokens(question)) < 5 for question in questions),
        "benchmark_source_split": manifest.get("upstream", {}).get("source_split"),
        "blueprint_sha256": blueprint_hash,
        "source_chunk_count": len(sources),
        "source_chunks_by_batch": source_counts,
        "sft_instruction_count": len(sft),
        "sft_instructions_by_root": sft_counts,
        "fivegram_recall_threshold": args.threshold,
        "exact_question_in_source_count": sum(row["source_overlap"]["exact_question"] for row in item_results),
        "exact_question_in_sft_count": sum(row["sft_instruction_overlap"]["exact_question"] for row in item_results),
        "near_question_in_source_count": sum(
            row["source_overlap"]["fivegram_recall"] >= args.threshold for row in item_results
        ),
        "near_question_in_sft_count": sum(
            row["sft_instruction_overlap"]["fivegram_recall"] >= args.threshold for row in item_results
        ),
        "freeze_timeline_verified": False,
        "public_pretraining_contamination_excluded": False,
        "items": item_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    summary = [
        "# 本地 Benchmark 重合审计",
        "",
        f"- 范围：{result['scope']}",
        f"- 锁定题数：{len(rows)}；payload SHA-256 校验通过；蓝图 SHA-256 校验通过。",
        f"- 少于五个词的短题干：{result['short_question_count_below_five_tokens']}；不作 5-gram 近重复判定。",
        f"- 原文段落：{len(sources)}；SFT 问题：{len(sft)}。",
        f"- 题干精确出现在原文：{result['exact_question_in_source_count']}；出现在 SFT 问题：{result['exact_question_in_sft_count']}。",
        f"- 题干 5-gram recall ≥ {args.threshold:.2f}：原文 {result['near_question_in_source_count']}，SFT 问题 {result['near_question_in_sft_count']}。",
        "",
        "这是本地文本重合筛查。共享医学知识不等于题目泄漏；低重合也不能排除基础模型预训练见过公开 MedMCQA 题目。文件时间戳不能替代经签名或版本控制的冻结时间线，后者尚未核实。",
        "",
        "逐题哈希、最高相似来源和阈值计数见 audit.json；报告不输出题干正文。",
    ]
    (args.output_dir / "audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, required=True)
    parser.add_argument("--sft-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    result = run(args)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "benchmark_item_count",
                    "source_chunk_count",
                    "sft_instruction_count",
                    "exact_question_in_source_count",
                    "exact_question_in_sft_count",
                    "near_question_in_source_count",
                    "near_question_in_sft_count",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
