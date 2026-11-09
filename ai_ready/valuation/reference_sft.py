"""Import a complete Fully Open Meditron batch into the reference SFT contract.

The reference branch is intentionally independent from proxy selection.  It
accepts normalized Fully Open Meditron rows for every successfully generated
source unit, records an explicit failed disposition for units with no accepted
row, and emits the lineage artifacts required by the LLaMAFactory adapter.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_ready.core.schema import stable_id
from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.contracts import canonical_digest, contract_digest, file_sha256
from ai_ready.valuation.quality import SFTSample, UnitDisposition, validate_full_traversal
from ai_ready.valuation.schema import ProductionContract


class NormalizedFullyOpenMeditronRow(BaseModel):
    """Minimum normalized boundary accepted from Fully Open Meditron.

    Extra upstream fields are allowed because their exact bytes remain bound by
    ``fully_open_meditron_input_sha256`` in the generation manifest.  Only the
    source-grounded fields below enter the common SFT schema.
    """

    model_config = ConfigDict(extra="allow")

    semantic_unit_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    response: str = Field(min_length=1)
    evidence_spans: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    task_type: Literal["grounded_qa", "conditional_judgment", "clinical_vignette"]
    batch_id: str | None = Field(default=None, min_length=1)

    @field_validator("evidence_spans", "source_ids")
    @classmethod
    def require_nonempty_unique_strings(cls, values: list[str], info) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"{info.field_name} cannot contain empty strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must contain unique values")
        return normalized

    @field_validator("semantic_unit_id", "instruction", "response")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Required text fields cannot be blank")
        return normalized


class FullyOpenMeditronRunManifest(BaseModel):
    """Strict upstream receipt proving which frozen inputs produced FOM output.

    The importer does not infer these values or fill them from the local
    contract.  Fully Open Meditron must emit the receipt at generation time;
    the importer then binds every claim to the frozen contract and exact input
    and output bytes before accepting the reference branch.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["fully-open-meditron-run-manifest-v1"]
    generator: Literal["fully_open_meditron"]
    generator_revision: str = Field(min_length=1)
    tool_adapter: Literal["fully_open_meditron"]
    tool_repository: str = Field(min_length=1)
    tool_config_sha256: str
    teacher_model: str = Field(min_length=1)
    teacher_revision: str = Field(min_length=1)
    prompt_sha256: str
    quality_gate_sha256: str
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    eligible_units_sha256: str
    normalized_output_sha256: str
    benchmark_items_accessed: Literal[False]
    uses_proxy_selector: Literal[False]
    complete_input_traversal: Literal[True]
    formal_production_contract_used: Literal[True]
    contract_id: str = Field(min_length=1)
    contract_sha256: str

    @field_validator(
        "prompt_sha256",
        "quality_gate_sha256",
        "tool_config_sha256",
        "batch_content_sha256",
        "eligible_units_sha256",
        "normalized_output_sha256",
        "contract_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)


def _validate_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _load_eligible_units(path: Path, *, batch_id: str) -> tuple[list[dict[str, Any]], list[str], dict[str, set[str]]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Reference import requires at least one eligible unit: {path}")

    normalized_rows: list[dict[str, Any]] = []
    unit_ids: list[str] = []
    sources_by_unit: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Eligible-unit row {index} is not a JSON object: {path}")
        unit_id = row.get("semantic_unit_id")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ValueError(f"Eligible-unit row {index} has no non-empty semantic_unit_id: {path}")
        unit_id = unit_id.strip()
        row_batch_id = row.get("batch_id")
        if row_batch_id is not None and row_batch_id != batch_id:
            raise ValueError(
                f"Eligible-unit row {index} belongs to batch {row_batch_id!r}, expected {batch_id!r}: {path}"
            )
        raw_source_ids = row.get("source_ids")
        if (
            not isinstance(raw_source_ids, list)
            or not raw_source_ids
            or any(not isinstance(source_id, str) or not source_id.strip() for source_id in raw_source_ids)
        ):
            raise ValueError(f"Eligible-unit row {index} requires non-empty source_ids: {path}")
        else:
            source_ids = {source_id.strip() for source_id in raw_source_ids}
            if len(source_ids) != len(raw_source_ids):
                raise ValueError(f"Eligible-unit row {index} has duplicate source_ids: {path}")
        unit_ids.append(unit_id)
        sources_by_unit[unit_id] = source_ids
        normalized_rows.append(
            {
                **row,
                "semantic_unit_id": unit_id,
                "batch_id": batch_id,
                "source_ids": sorted(source_ids),
            }
        )

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError(f"Eligible semantic_unit_id values must be unique: {path}")
    return normalized_rows, unit_ids, sources_by_unit


def _reference_recipe_and_tool(contract: ProductionContract):
    if contract.reference_policy.traversal != "full" or contract.reference_policy.uses_proxy_selector:
        raise ValueError("Fully Open Meditron reference import requires full traversal without a proxy selector")
    recipe = contract.generation.reference
    if recipe.generator != "fully_open_meditron":
        raise ValueError("The reference generation recipe must identify Fully Open Meditron")
    tools = [tool for tool in contract.tools if tool.stage == "generate"]
    if len(tools) != 1 or tools[0].adapter != "fully_open_meditron":
        raise ValueError("The production contract must bind reference generation to the fully_open_meditron adapter")
    tool = tools[0]
    if tool.revision != recipe.revision:
        raise ValueError("Fully Open Meditron tool and reference recipe revisions must match")
    unresolved = {
        "generation.reference.revision": recipe.revision,
        "generation.reference.teacher_revision": recipe.teacher_revision,
        "tools[generate].revision": tool.revision,
    }
    placeholder_fields = [
        field_name
        for field_name, value in unresolved.items()
        if "example" in value.lower() or "placeholder" in value.lower()
    ]
    if placeholder_fields:
        raise ValueError(
            "Formal Fully Open Meditron import requires real pinned revisions; unresolved fields: "
            + ", ".join(placeholder_fields)
        )
    return recipe, tool


def _load_and_validate_run_manifest(
    path: Path,
    *,
    contract: ProductionContract,
    recipe,
    tool,
    batch_id: str,
    batch_content_sha256: str,
    eligible_units_sha256: str,
    normalized_output_sha256: str,
) -> FullyOpenMeditronRunManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Fully Open Meditron run manifest JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Fully Open Meditron run manifest must be a JSON object: {path}")
    receipt = FullyOpenMeditronRunManifest.model_validate(raw)

    # These comparisons turn the upstream receipt into checked provenance
    # rather than letting the importer self-assert that arbitrary JSONL came
    # from the contracted Fully Open Meditron run.
    expected = {
        "generator_revision": recipe.revision,
        "tool_adapter": tool.adapter,
        "tool_repository": tool.repository,
        "tool_config_sha256": tool.config_sha256,
        "teacher_model": recipe.teacher_model,
        "teacher_revision": recipe.teacher_revision,
        "prompt_sha256": recipe.prompt_sha256,
        "quality_gate_sha256": recipe.quality_gate_sha256,
        "batch_id": batch_id,
        "batch_content_sha256": batch_content_sha256,
        "eligible_units_sha256": eligible_units_sha256,
        "normalized_output_sha256": normalized_output_sha256,
        "contract_id": contract.contract_id,
        "contract_sha256": contract_digest(contract),
    }
    mismatches = [
        f"{field}: receipt={getattr(receipt, field)!r}, expected={value!r}"
        for field, value in expected.items()
        if getattr(receipt, field) != value
    ]
    if mismatches:
        raise ValueError("Fully Open Meditron run manifest does not match frozen inputs: " + "; ".join(mismatches))
    return receipt


def import_fully_open_meditron_reference(
    *,
    batch_id: str,
    batch_content_sha256: str,
    contract: ProductionContract,
    eligible_units_path: str | Path,
    fully_open_meditron_path: str | Path,
    run_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create audited full-reference SFT artifacts from normalized FOM JSONL.

    The function never selects units.  Every registered eligible unit receives
    exactly one disposition: accepted when at least one normalized FOM row is
    present, otherwise failed so low generation yield remains observable.
    """

    if not batch_id.strip():
        raise ValueError("batch_id cannot be blank")
    batch_id = batch_id.strip()
    batch_content_sha256 = _validate_sha256(batch_content_sha256, "batch_content_sha256")
    recipe, tool = _reference_recipe_and_tool(contract)

    eligible_source = Path(eligible_units_path).resolve()
    fom_source = Path(fully_open_meditron_path).resolve()
    run_manifest_source = Path(run_manifest_path).resolve()
    for label, path in (
        ("eligible units", eligible_source),
        ("normalized Fully Open Meditron JSONL", fom_source),
        ("Fully Open Meditron run manifest", run_manifest_source),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Reference SFT output already exists: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Incomplete reference SFT output already exists: {partial}")

    eligible_input_sha256 = file_sha256(eligible_source)
    fom_input_sha256 = file_sha256(fom_source)
    run_manifest_sha256 = file_sha256(run_manifest_source)
    _load_and_validate_run_manifest(
        run_manifest_source,
        contract=contract,
        recipe=recipe,
        tool=tool,
        batch_id=batch_id,
        batch_content_sha256=batch_content_sha256,
        eligible_units_sha256=eligible_input_sha256,
        normalized_output_sha256=fom_input_sha256,
    )
    eligible_rows, eligible_unit_ids, sources_by_unit = _load_eligible_units(eligible_source, batch_id=batch_id)
    known_units = set(eligible_unit_ids)
    raw_fom_rows = read_jsonl(fom_source)
    normalized_rows = [NormalizedFullyOpenMeditronRow.model_validate(row) for row in raw_fom_rows]

    samples: list[SFTSample] = []
    sample_ids_by_unit: dict[str, list[str]] = defaultdict(list)
    signatures: set[tuple[str, str]] = set()
    for index, row in enumerate(normalized_rows):
        if row.batch_id is not None and row.batch_id != batch_id:
            raise ValueError(
                f"Fully Open Meditron row {index} belongs to batch {row.batch_id!r}, expected {batch_id!r}"
            )
        if row.semantic_unit_id not in known_units:
            raise ValueError(f"Fully Open Meditron row {index} references non-eligible unit {row.semantic_unit_id!r}")
        registered_sources = sources_by_unit[row.semantic_unit_id]
        if registered_sources and not set(row.source_ids).issubset(registered_sources):
            unknown = sorted(set(row.source_ids) - registered_sources)
            raise ValueError(
                f"Fully Open Meditron row {index} references unregistered sources for "
                f"{row.semantic_unit_id}: {unknown}"
            )
        if row.task_type not in recipe.task_quotas:
            raise ValueError(
                f"Fully Open Meditron row {index} uses task type outside the reference recipe: {row.task_type}"
            )
        signature = (row.instruction, row.response)
        if signature in signatures:
            raise ValueError("Duplicate Fully Open Meditron instruction/response supervision is forbidden")
        signatures.add(signature)
        sample_id = stable_id(
            "fully-open-meditron-reference-sft-v1",
            batch_id,
            batch_content_sha256,
            fom_input_sha256,
            str(index),
            row.semantic_unit_id,
            row.instruction,
            row.response,
        )
        sample = SFTSample(
            sample_id=sample_id,
            batch_id=batch_id,
            semantic_unit_ids=[row.semantic_unit_id],
            instruction=row.instruction,
            response=row.response,
            evidence_spans=row.evidence_spans,
            source_ids=row.source_ids,
            task_type=row.task_type,
            generator=recipe.generator,
            generator_revision=recipe.revision,
            teacher_model=recipe.teacher_model,
            teacher_revision=recipe.teacher_revision,
            prompt_sha256=recipe.prompt_sha256,
            quality_gate_sha256=recipe.quality_gate_sha256,
        )
        samples.append(sample)
        sample_ids_by_unit[row.semantic_unit_id].append(sample_id)

    dispositions = [
        UnitDisposition(
            semantic_unit_id=unit_id,
            status="accepted" if sample_ids_by_unit[unit_id] else "failed",
            sample_ids=sample_ids_by_unit[unit_id],
            reason_codes=[] if sample_ids_by_unit[unit_id] else ["fully_open_meditron_no_accepted_output"],
        )
        for unit_id in eligible_unit_ids
    ]
    missing_units = [row.semantic_unit_id for row in dispositions if row.status == "failed"]
    if missing_units and contract.reference_policy.low_yield_policy == "fail_batch":
        raise ValueError(f"Fully Open Meditron produced no accepted output for eligible units: {missing_units}")

    traversal = validate_full_traversal(eligible_unit_ids, dispositions, samples)
    if not traversal.valid:
        raise ValueError(f"Fully Open Meditron reference traversal is invalid: {traversal.errors}")
    if (
        file_sha256(eligible_source) != eligible_input_sha256
        or file_sha256(fom_source) != fom_input_sha256
        or file_sha256(run_manifest_source) != run_manifest_sha256
    ):
        raise RuntimeError("Reference import inputs changed during validation")

    try:
        partial.mkdir(parents=True, exist_ok=False)
        accepted_path = partial / "accepted_sft.jsonl"
        dispositions_path = partial / "dispositions.jsonl"
        eligible_path = partial / "eligible_units.jsonl"
        traversal_path = partial / "full_traversal_report.json"
        manifest_path = partial / "generation_manifest.json"
        copied_run_manifest_path = partial / "fully_open_meditron_run_manifest.json"

        shutil.copyfile(run_manifest_source, copied_run_manifest_path)
        if file_sha256(copied_run_manifest_path) != run_manifest_sha256:
            raise RuntimeError("Fully Open Meditron run manifest changed while being copied")
        write_jsonl(accepted_path, (sample.model_dump(mode="json") for sample in samples))
        write_jsonl(dispositions_path, (row.model_dump(mode="json") for row in dispositions))
        write_jsonl(eligible_path, eligible_rows)
        report = {
            "schema_version": "full-traversal-report-v1",
            "batch_id": batch_id,
            "batch_content_sha256": batch_content_sha256,
            "eligible_units_sha256": file_sha256(eligible_path),
            "dispositions_sha256": file_sha256(dispositions_path),
            "source_sft_sha256": file_sha256(accepted_path),
            **traversal.model_dump(mode="json"),
        }
        write_json(traversal_path, report)

        generation_manifest = {
            "schema_version": "sft-generation-manifest-v2",
            "artifact_role": "reference",
            "input_scope": "full_batch",
            "batch_id": batch_id,
            "batch_content_sha256": batch_content_sha256,
            "generator": recipe.generator,
            "generator_revision": recipe.revision,
            "generation_recipe_sha256": canonical_digest(recipe),
            "source_sft_sha256": file_sha256(accepted_path),
            "full_traversal_report_sha256": file_sha256(traversal_path),
            "benchmark_items_accessed": False,
            "uses_proxy_selector": False,
            "complete_input_traversal": True,
            "formal_production_contract_used": True,
            "upstream_run_manifest_sha256": run_manifest_sha256,
            "selection_artifacts": [],
            "contract_id": contract.contract_id,
            "contract_sha256": contract_digest(contract),
            "tool_binding": {
                "adapter": tool.adapter,
                "repository": tool.repository,
                "revision": tool.revision,
                "config_sha256": tool.config_sha256,
            },
            "inputs": {
                "eligible_units_uri": str(eligible_source),
                "eligible_units_sha256": eligible_input_sha256,
                "fully_open_meditron_uri": str(fom_source),
                "fully_open_meditron_sha256": fom_input_sha256,
                "fully_open_meditron_run_manifest_uri": str(run_manifest_source),
                "fully_open_meditron_run_manifest_sha256": run_manifest_sha256,
            },
            "outputs": {
                "accepted_sft_sha256": file_sha256(accepted_path),
                "dispositions_sha256": file_sha256(dispositions_path),
                "eligible_units_sha256": file_sha256(eligible_path),
                "full_traversal_report_sha256": file_sha256(traversal_path),
                "fully_open_meditron_run_manifest_sha256": file_sha256(copied_run_manifest_path),
            },
            "counts": {
                "eligible_units": len(eligible_unit_ids),
                "normalized_fom_rows": len(normalized_rows),
                "accepted_samples": len(samples),
                "accepted_units": sum(row.status == "accepted" for row in dispositions),
                "failed_units": len(missing_units),
            },
        }
        write_json(manifest_path, generation_manifest)
        if (
            file_sha256(eligible_source) != eligible_input_sha256
            or file_sha256(fom_source) != fom_input_sha256
            or file_sha256(run_manifest_source) != run_manifest_sha256
        ):
            raise RuntimeError("Reference import inputs changed while artifacts were written")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise

    return {
        "output_dir": str(destination),
        "batch_id": batch_id,
        "accepted_samples": len(samples),
        "eligible_units": len(eligible_unit_ids),
        "failed_units": len(missing_units),
        "generation_manifest": str(destination / "generation_manifest.json"),
        "fully_open_meditron_run_manifest": str(destination / "fully_open_meditron_run_manifest.json"),
        "upstream_run_manifest_sha256": run_manifest_sha256,
    }


__all__ = [
    "FullyOpenMeditronRunManifest",
    "NormalizedFullyOpenMeditronRow",
    "import_fully_open_meditron_reference",
]
