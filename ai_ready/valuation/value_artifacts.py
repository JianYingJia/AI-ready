"""Adapters from existing valuation artifacts to value-distribution records.

The adapter is deliberately descriptive.  Existing B01--B12 artifacts do not
contain an independent M0 probe result, so every emitted record is marked
``unknown``.  A later, separately audited probe import may replace that state;
content, relevance, or generation success must never be used as a proxy for
model mastery.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.contracts import file_sha256, load_batch_suite


UNMAPPED_CAPABILITY = "__unmapped__"


def _relevance_bucket(value: float) -> str:
    if value >= 2 / 3:
        return "high"
    if value >= 1 / 3:
        return "medium"
    return "low"


def _knowledge_type(semantic: dict[str, Any], obligation_kinds: set[str]) -> str:
    if semantic.get("exception_spans") or "exception" in obligation_kinds:
        return "exception"
    if semantic.get("condition_spans") or "condition" in obligation_kinds:
        return "condition"
    if semantic.get("temporal_spans"):
        return "temporal"
    if semantic.get("uncertainty_spans"):
        return "uncertainty"
    if obligation_kinds:
        return sorted(obligation_kinds)[0]
    return "rule"


def _require_artifact(batch_dir: Path, name: str) -> Path:
    path = batch_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {batch_dir.name} artifact: {path}")
    return path


def adapt_existing_value_population(
    *,
    suite_path: Path,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Normalize an existing B01--B12 generation run for value sampling.

    One source semantic unit may map to several blueprint capabilities.  Those
    mappings stay on one indivisible record; duplicate/dependency groups are
    then sampled atomically.  A separate map resolves selected records back to
    the original semantic-unit artifacts before SFT generation.
    """

    suite_path = suite_path.resolve()
    artifact_root = artifact_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path must be an empty directory: {output_dir}")

    suite = load_batch_suite(suite_path)
    expected_batches = {batch.batch_id: batch for batch in suite.batches}
    if not expected_batches:
        raise ValueError("Batch suite contains no batches")

    records: list[dict[str, Any]] = []
    membership_map: list[dict[str, str]] = []
    source_artifacts: dict[str, dict[str, str]] = {}
    batch_counts: dict[str, dict[str, int]] = {}

    for batch_id, batch in sorted(expected_batches.items()):
        batch_dir = artifact_root / batch_id
        semantic_path = _require_artifact(batch_dir, "semantic_units.jsonl")
        proxy_path = _require_artifact(batch_dir, "proxy_units.jsonl")
        obligations_path = _require_artifact(batch_dir, "coverage_obligations.jsonl")
        dispositions_path = _require_artifact(batch_dir, "dispositions.jsonl")
        generation_manifest_path = _require_artifact(batch_dir, "generation_manifest.json")

        generation_manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
        if generation_manifest.get("batch_id") != batch_id:
            raise ValueError(f"Generation manifest batch mismatch for {batch_id}")
        if generation_manifest.get("batch_content_sha256") != batch.content_sha256:
            raise ValueError(f"Generation manifest content hash mismatch for {batch_id}")

        semantic_rows = read_jsonl(semantic_path)
        proxy_rows = {row["unit_id"]: row for row in read_jsonl(proxy_path)}
        dispositions = {row["semantic_unit_id"]: row for row in read_jsonl(dispositions_path)}
        capabilities_by_unit: dict[str, set[str]] = defaultdict(set)
        kinds_by_unit: dict[str, set[str]] = defaultdict(set)
        for obligation in read_jsonl(obligations_path):
            for unit_id in obligation["unit_ids"]:
                capabilities_by_unit[unit_id].update(obligation.get("capability_ids", []))
                kinds_by_unit[unit_id].add(obligation["kind"])

        semantic_ids = [row["semantic_unit_id"] for row in semantic_rows]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError(f"Duplicate semantic unit IDs in {semantic_path}")
        if set(semantic_ids) != set(proxy_rows):
            missing_proxy = sorted(set(semantic_ids) - set(proxy_rows))
            extra_proxy = sorted(set(proxy_rows) - set(semantic_ids))
            raise ValueError(
                f"Semantic/proxy unit mismatch for {batch_id}: "
                f"missing_proxy={missing_proxy[:5]}, extra_proxy={extra_proxy[:5]}"
            )

        emitted = 0
        for semantic in sorted(semantic_rows, key=lambda row: row["semantic_unit_id"]):
            semantic_id = semantic["semantic_unit_id"]
            proxy = proxy_rows[semantic_id]
            disposition = dispositions.get(semantic_id)
            capability_ids = sorted(
                capabilities_by_unit.get(semantic_id) or semantic.get("activity_ids") or [UNMAPPED_CAPABILITY]
            )
            source_ids = sorted(proxy.get("source_ids", []))
            dependency_group_id = str(proxy.get("group_id") or semantic.get("group_id") or semantic_id)
            knowledge_type = _knowledge_type(semantic, kinds_by_unit.get(semantic_id, set()))
            status = disposition["status"] if disposition else "not_generated"
            eligible = bool(proxy.get("eligible", True) and proxy.get("medically_valid", True))

            records.append(
                {
                    "unit_id": semantic_id,
                    "batch_id": batch_id,
                    "batch_content_sha256": batch.content_sha256,
                    "dependency_group_id": dependency_group_id,
                    "capability_ids": capability_ids,
                    # No real M0 probes exist in these historical artifacts.
                    "model_states": {capability_id: "unknown" for capability_id in capability_ids},
                    "population_weight": 1.0,
                    "data_attributes": {
                        "knowledge_type": knowledge_type,
                        "quality_bucket": str(proxy.get("quality_bucket", "unknown")),
                        "benchmark_relevance": _relevance_bucket(float(proxy.get("benchmark_relevance", 0))),
                        "generation_status": status,
                        "source_scope": "+".join(source_ids) if source_ids else "unknown",
                        "batch_kind": str(batch.kind),
                        "coverage_scope": str(batch.factors.coverage_scope),
                        "target_relevance": str(batch.factors.target_relevance),
                        "redundancy": str(batch.factors.redundancy),
                        "parsing_quality": str(batch.factors.parsing_quality),
                    },
                    "eligible": eligible,
                }
            )
            membership_map.append(
                {
                    "membership_id": semantic_id,
                    "semantic_unit_id": semantic_id,
                    "batch_id": batch_id,
                    "capability_ids": capability_ids,
                    "dependency_group_id": dependency_group_id,
                }
            )
            emitted += 1

        source_artifacts[batch_id] = {
            name: file_sha256(path)
            for name, path in {
                "semantic_units.jsonl": semantic_path,
                "proxy_units.jsonl": proxy_path,
                "coverage_obligations.jsonl": obligations_path,
                "dispositions.jsonl": dispositions_path,
                "generation_manifest.json": generation_manifest_path,
            }.items()
        }
        batch_counts[batch_id] = {"semantic_units": len(semantic_rows), "membership_records": emitted}

    output_dir.mkdir(parents=True, exist_ok=False)
    population_path = output_dir / "value_population_units.jsonl"
    membership_path = output_dir / "membership_map.jsonl"
    write_jsonl(population_path, records)
    write_jsonl(membership_path, membership_map)
    manifest = {
        "schema_version": "value-artifact-adapter-v1",
        "suite_id": suite.suite_id,
        "suite_file_sha256": file_sha256(suite_path),
        "artifact_root": str(artifact_root),
        "batch_ids": sorted(expected_batches),
        "batch_counts": batch_counts,
        "source_artifacts": source_artifacts,
        "population_units": {
            "path": population_path.name,
            "sha256": file_sha256(population_path),
            "count": len(records),
        },
        "membership_map": {
            "path": membership_path.name,
            "sha256": file_sha256(membership_path),
            "count": len(membership_map),
        },
        "model_mastery_status": "pending",
        "default_model_state": "unknown",
        "probe_evidence_present": False,
        "claim_status": "partial_pending_model_measurement",
        "benchmark_test_items_accessed": False,
        "measurement_granularity": "capability",
        "cell_id_semantics": "cell_id_equals_capability_id",
        "operation_dimension": "absent",
        "probe_difficulty_dimension": "absent",
        "notes": [
            "Historical generation artifacts contain no independent M0 probe evidence.",
            "Unknown mastery states must be replaced only by an audited scored-probe import.",
            "Unit/capability memberships from one dependency group must be sampled atomically.",
        ],
    }
    write_json(output_dir / "adapter_manifest.json", manifest)
    return manifest


def materialize_selected_value_artifacts(
    *,
    sample_manifest: dict[str, Any],
    membership_map_path: Path,
    artifact_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Resolve selected membership records back to existing source artifacts.

    This only materializes the first-stage semantic-unit sample.  It does not
    call a teacher model and must not be described as generated SFT.
    """

    membership_map_path = membership_map_path.resolve()
    artifact_root = artifact_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path must be an empty directory: {output_dir}")

    selected_memberships = set(sample_manifest.get("selected_unit_ids", []))
    if not selected_memberships:
        raise ValueError("Sample manifest has no selected_unit_ids")
    membership_rows = read_jsonl(membership_map_path)
    membership_by_id = {row["membership_id"]: row for row in membership_rows}
    if len(membership_by_id) != len(membership_rows):
        raise ValueError("Membership map contains duplicate membership IDs")
    if unknown := selected_memberships - set(membership_by_id):
        raise ValueError(f"Sample manifest references unknown membership IDs: {sorted(unknown)[:5]}")

    selected_by_batch: dict[str, set[str]] = defaultdict(set)
    for membership_id in selected_memberships:
        row = membership_by_id[membership_id]
        selected_by_batch[row["batch_id"]].add(row["semantic_unit_id"])

    semantic_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    obligation_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for batch_id, selected_semantic_ids in sorted(selected_by_batch.items()):
        batch_dir = artifact_root / batch_id
        semantic_path = _require_artifact(batch_dir, "semantic_units.jsonl")
        proxy_path = _require_artifact(batch_dir, "proxy_units.jsonl")
        obligations_path = _require_artifact(batch_dir, "coverage_obligations.jsonl")

        batch_semantic = [
            row for row in read_jsonl(semantic_path) if row["semantic_unit_id"] in selected_semantic_ids
        ]
        batch_proxy = [row for row in read_jsonl(proxy_path) if row["unit_id"] in selected_semantic_ids]
        found_semantic = {row["semantic_unit_id"] for row in batch_semantic}
        found_proxy = {row["unit_id"] for row in batch_proxy}
        if found_semantic != selected_semantic_ids or found_proxy != selected_semantic_ids:
            raise ValueError(f"Selected semantic units are missing from {batch_id} source artifacts")
        semantic_rows.extend(batch_semantic)
        proxy_rows.extend(batch_proxy)

        for obligation in read_jsonl(obligations_path):
            selected_ids = [unit_id for unit_id in obligation["unit_ids"] if unit_id in selected_semantic_ids]
            if selected_ids:
                obligation_rows.append(
                    {
                        **obligation,
                        "unit_ids": selected_ids,
                    }
                )
        source_hashes[batch_id] = {
            "semantic_units.jsonl": file_sha256(semantic_path),
            "proxy_units.jsonl": file_sha256(proxy_path),
            "coverage_obligations.jsonl": file_sha256(obligations_path),
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    semantic_output = output_dir / "selected_semantic_units.jsonl"
    proxy_output = output_dir / "selected_proxy_units.jsonl"
    obligations_output = output_dir / "selected_coverage_obligations.jsonl"
    write_jsonl(semantic_output, semantic_rows)
    write_jsonl(proxy_output, proxy_rows)
    write_jsonl(obligations_output, obligation_rows)

    manifest = {
        "schema_version": "value-sample-artifacts-v1",
        "artifact_role": "first_stage_semantic_unit_sample",
        "claim_status": sample_manifest.get("claim_status", "partial_pending_model_measurement"),
        "benchmark_test_items_accessed": False,
        "teacher_model_called": False,
        "generated_sft": False,
        "selected_membership_count": len(selected_memberships),
        "selected_semantic_unit_count": len({row["semantic_unit_id"] for row in semantic_rows}),
        "selected_batch_ids": sorted(selected_by_batch),
        "source_artifacts": source_hashes,
        "sample_manifest_sha256": hashlib.sha256(
            json.dumps(sample_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "artifacts": {
            semantic_output.name: file_sha256(semantic_output),
            proxy_output.name: file_sha256(proxy_output),
            obligations_output.name: file_sha256(obligations_output),
        },
    }
    write_json(output_dir / "value_sample_artifact_manifest.json", manifest)
    return manifest


__all__ = [
    "UNMAPPED_CAPABILITY",
    "adapt_existing_value_population",
    "materialize_selected_value_artifacts",
]
