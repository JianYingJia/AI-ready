"""Validate the public-source CLL Base Model probe catalogue without test-set access.

The validator deliberately rejects paths that contain a locked-benchmark marker.
It verifies the catalogue, manifest, source files, source-block files, evidence
excerpts, per-item labels, and all declared count summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "inputs/conditional_value/cll_independent_capability_probes_v1.jsonl"
DEFAULT_MANIFEST = ROOT / "inputs/conditional_value/cll_independent_capability_probes_v1.manifest.json"
FORBIDDEN_PARTS = {"benchmark_locked", "locked_benchmark", "locked-test", "locked_test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _public_path(value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: expected a repository-relative path, got {value!r}")
    lowered = {part.lower() for part in relative.parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"{label}: locked-benchmark paths are prohibited")
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"{label}: file does not exist: {value}")
    return path


def validate(catalog_path: Path, manifest_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(catalog_path)

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(manifest.get("artifact_line_count") == len(rows), "artifact_line_count mismatch")
    check(manifest.get("artifact_sha256") == _sha256(catalog_path), "artifact_sha256 mismatch")
    check(manifest.get("answer_index_base") == 0, "answer_index_base must be zero")
    check(manifest.get("locked_test_isolation", {}).get("locked_payload_read_or_used") is False,
          "manifest must declare that locked payload was not read or used")
    check(manifest.get("locked_test_isolation", {}).get("test_items_included") is False,
          "manifest must declare that locked test items are excluded")

    ids = [row.get("probe_id") for row in rows]
    item_ids = [row.get("item_id") for row in rows]
    questions = [row.get("question") for row in rows]
    check(len(ids) == len(set(ids)), "probe_id values are not unique")
    check(len(item_ids) == len(set(item_ids)), "item_id values are not unique")
    check(len(questions) == len(set(questions)), "question texts are not unique")

    blocks_cache: dict[Path, list[dict[str, Any]]] = {}
    per_source = Counter()
    per_cell = Counter()
    per_answer = Counter()
    per_temporal = Counter()
    for index, row in enumerate(rows, start=1):
        probe_id = str(row.get("probe_id", f"row-{index}"))
        options = row.get("options")
        answer_index = row.get("answer_index")
        check(isinstance(options, list) and len(options) == 4 and len(set(options)) == 4,
              f"{probe_id}: expected four distinct options")
        check(isinstance(answer_index, int) and 0 <= answer_index < 4,
              f"{probe_id}: answer_index must be in [0, 3]")
        if isinstance(options, list) and isinstance(answer_index, int) and 0 <= answer_index < len(options):
            check(row.get("answer_text") == options[answer_index], f"{probe_id}: answer_text/index mismatch")
        check(row.get("answer_index_base") == 0, f"{probe_id}: answer_index_base must be zero")
        check(row.get("probe_id") == row.get("item_id"), f"{probe_id}: probe_id must equal item_id")
        for flag in ("locked_test", "is_locked_test_item", "test_items_included"):
            check(row.get(flag) is False, f"{probe_id}: {flag} must be false")

        evidence = row.get("source_evidence")
        check(isinstance(evidence, str) and bool(evidence.strip()), f"{probe_id}: source_evidence is empty")
        if isinstance(evidence, str):
            check(row.get("source_evidence_sha256") == _text_sha256(evidence),
                  f"{probe_id}: source_evidence_sha256 mismatch")

        location = row.get("source_location") or {}
        try:
            processed_path = _public_path(location["processed_path"], label=f"{probe_id} processed_path")
            blocks_path = _public_path(location["blocks_path"], label=f"{probe_id} blocks_path")
            check(row.get("source_file") == location["processed_path"], f"{probe_id}: source_file mismatch")
            processed_hash = _sha256(processed_path)
            blocks_hash = _sha256(blocks_path)
            check(processed_hash == row.get("source_sha256"), f"{probe_id}: source_sha256 mismatch")
            check(processed_hash == location.get("processed_sha256"),
                  f"{probe_id}: source_location processed_sha256 mismatch")
            check(blocks_hash == location.get("blocks_sha256"), f"{probe_id}: blocks_sha256 mismatch")
            if blocks_path not in blocks_cache:
                blocks_cache[blocks_path] = _read_jsonl(blocks_path)
            block_index = location.get("block_index")
            check(location.get("block_index_base") == 0, f"{probe_id}: block_index_base must be zero")
            if not isinstance(block_index, int) or not 0 <= block_index < len(blocks_cache[blocks_path]):
                errors.append(f"{probe_id}: block_index is out of range")
            else:
                block = blocks_cache[blocks_path][block_index]
                block_text = block.get("text")
                check(isinstance(block_text, str), f"{probe_id}: source block has no text")
                if isinstance(block_text, str):
                    check(location.get("source_block_text_sha256") == _text_sha256(block_text),
                          f"{probe_id}: source_block_text_sha256 mismatch")
                    if isinstance(evidence, str):
                        check(evidence in block_text, f"{probe_id}: evidence excerpt not found in source block")
                check(location.get("section") == block.get("section"), f"{probe_id}: source section mismatch")
        except (KeyError, ValueError) as exc:
            errors.append(f"{probe_id}: {exc}")

        source_id = location.get("source_id")
        per_source[str(source_id)] += 1
        per_cell[str(row.get("cell_id"))] += 1
        per_answer[str(answer_index)] += 1
        per_temporal[str(row.get("temporal_risk"))] += 1

    counts = manifest.get("counts", {})
    check(counts.get("total") == len(rows), "manifest counts.total mismatch")
    check(counts.get("by_cell") == dict(sorted(per_cell.items())), "manifest counts.by_cell mismatch")
    check(counts.get("by_answer_index") == dict(sorted(per_answer.items())),
          "manifest counts.by_answer_index mismatch")
    check(counts.get("by_source") == dict(sorted(per_source.items())), "manifest counts.by_source mismatch")
    check(counts.get("by_temporal_risk") == dict(sorted(per_temporal.items())),
          "manifest counts.by_temporal_risk mismatch")

    blueprint = manifest.get("blueprint", {})
    try:
        blueprint_path = _public_path(blueprint["path"], label="blueprint path")
        check(_sha256(blueprint_path) == blueprint.get("sha256"), "blueprint sha256 mismatch")
        check(blueprint.get("test_items_included") is False, "blueprint must exclude test items")
    except (KeyError, ValueError) as exc:
        errors.append(str(exc))

    lineage_by_source = {entry.get("source_id"): entry for entry in manifest.get("source_lineage", [])}
    check(set(lineage_by_source) == set(per_source), "source_lineage IDs do not match catalogue")
    for source_id, count in per_source.items():
        entry = lineage_by_source.get(source_id)
        if entry is None:
            continue
        check(entry.get("probe_count") == count, f"{source_id}: probe_count mismatch")
        for path_key, hash_key in (("processed_path", "processed_sha256"), ("blocks_path", "blocks_sha256")):
            try:
                source_path = _public_path(entry[path_key], label=f"{source_id} {path_key}")
                check(_sha256(source_path) == entry.get(hash_key), f"{source_id}: {hash_key} mismatch")
            except (KeyError, ValueError) as exc:
                errors.append(str(exc))

    if errors:
        raise ValueError("Probe catalogue validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "artifact_sha256": _sha256(catalog_path),
        "artifact_line_count": len(rows),
        "counts": {
            "by_cell": dict(sorted(per_cell.items())),
            "by_answer_index": dict(sorted(per_answer.items())),
            "by_source": dict(sorted(per_source.items())),
            "by_temporal_risk": dict(sorted(per_temporal.items())),
        },
        "checked_source_files": len(per_source),
        "checked_source_block_files": len(blocks_cache),
        "locked_benchmark_payload_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(validate(args.catalog.resolve(), args.manifest.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
