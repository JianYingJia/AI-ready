"""Compile a frozen ablation protocol into an auditable execution matrix.

``ablation_data`` builds the data for one batch.  This module is the stricter
outer boundary used for a real experiment: it checks that an execution spec is
an exact implementation of the registered protocol, binds every batch to the
frozen suite, materializes all batches atomically, and proves that the emitted
jobs form the complete batch x cell x seed matrix.

The Fully Open Meditron arm is intentionally absent here.  It remains an
independently produced operational reference and is imported at analysis time.
Likewise, static ``no_sft`` cells are audited by their own stage and never
become training jobs.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.ablation import AblationProtocol, ablation_protocol_digest
from ai_ready.valuation.ablation_data import (
    AblationBatchInput,
    AblationCellExecution,
    AblationExecutionSpec,
    AblationJobRecord,
    batch_input_digest,
    execution_digest,
    materialize_ablation_batch,
)
from ai_ready.valuation.contracts import contract_digest, file_sha256, suite_digest
from ai_ready.valuation.schema import BatchSuite, ProductionContract

ModelT = TypeVar("ModelT", bound=BaseModel)

_SUPPORTED_GENERATOR = "coverage_guided_sft"
_SUPPORTED_MINING = "structured_rule_family_v1"

_SELECTOR_METHODS = {
    "coverage_quality": "coverage_quality",
    "coverage_only": "coverage_only",
    "stratified_random_seed_set": "stratified_random",
    "stratified_random_seed_101": "stratified_random",
    "random_seed_set": "random",
    "random_seed_101": "random",
    "full_candidate_pool": "coverage_quality",
    "coverage_quality_include_target": "coverage_quality",
    "coverage_quality_knockout_target": "coverage_quality",
    "coverage_quality_rescue_target": "coverage_quality",
    "coverage_quality_nested": "coverage_quality",
    "coverage_quality_prior_aware": "coverage_quality",
    "coverage_quality_no_prior": "coverage_quality",
}

_SELECTOR_INTERVENTIONS = {
    "coverage_quality_include_target": "include",
    "coverage_quality_knockout_target": "knockout",
    "coverage_quality_rescue_target": "rescue",
}

_ORGANIZATIONS = {
    "mixed_grounded_v1": "mixed",
    "direct_qa_only": "direct",
    "random_form_seed_101": "random",
}

_STAGE_KINDS = {
    "blueprint_conditioning": "selection",
    "subset_selection": "selection",
    "supervision_organization": "organization",
    "learnability_bridge": "learnability",
    "scale_repeat_noise": "robustness",
    "base_model_prior": "base_model_prior",
    "operational_external_validity": "selection",
}


def _load_mapping(path: str | Path) -> dict[str, Any]:
    """Load one JSON/YAML mapping without letting file formatting affect identity."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Ablation input does not exist: {resolved}")
    if resolved.suffix.lower() == ".json":
        value = json.loads(resolved.read_text(encoding="utf-8"))
    elif resolved.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Ablation inputs must be JSON or YAML: {resolved}")
    if not isinstance(value, dict):
        raise ValueError(f"Ablation input must contain one object: {resolved}")
    return value


def _load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(_load_mapping(path))


def load_ablation_execution_spec(path: str | Path) -> AblationExecutionSpec:
    """Load and strictly validate a JSON/YAML execution specification."""

    return _load_model(path, AblationExecutionSpec)


def load_ablation_batch_input(path: str | Path) -> AblationBatchInput:
    """Load and strictly validate one JSON/YAML frozen batch input."""

    return _load_model(path, AblationBatchInput)


def _required_protocol_cells(protocol: AblationProtocol) -> dict[tuple[str, str], tuple[Any, Any]]:
    """Return precisely the proxy-generated SFT cells this compiler owns."""

    required: dict[tuple[str, str], tuple[Any, Any]] = {}
    for study in protocol.studies:
        for cell in study.cells:
            factors = cell.factors
            if factors.training_mode != "sft" or factors.generator == "fully_open_meditron":
                continue
            key = (study.study_id, cell.cell_id)
            required[key] = (study, cell)
    return required


def _execution_value(cell: AblationCellExecution, field: str, implicit: Any) -> Any:
    """Use future explicit fields while keeping today's data schema compatible."""

    return getattr(cell, field) if hasattr(cell, field) else implicit


def _require_equal(key: tuple[str, str], field: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"Execution cell {key[0]}/{key[1]} has wrong {field}: " f"expected={expected!r}, observed={observed!r}"
        )


def _require_float_equal(key: tuple[str, str], field: str, observed: float, expected: float) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Execution cell {key[0]}/{key[1]} has wrong {field}: " f"expected={expected!r}, observed={observed!r}"
        )


def _validate_cell_binding(
    execution_cell: AblationCellExecution,
    *,
    study: Any,
    protocol_cell: Any,
    spec: AblationExecutionSpec,
    contract: ProductionContract,
) -> None:
    """Check every materialized control against its registered protocol factor."""

    key = (study.study_id, protocol_cell.cell_id)
    factors = protocol_cell.factors

    _require_equal(key, "mining", _execution_value(execution_cell, "mining", _SUPPORTED_MINING), factors.mining)
    _require_equal(
        key,
        "generator",
        _execution_value(execution_cell, "generator", _SUPPORTED_GENERATOR),
        factors.generator,
    )
    _require_equal(key, "capability_profile_id", execution_cell.capability_profile_id, factors.blueprint)

    expected_method = _SELECTOR_METHODS.get(factors.selector)
    if expected_method is None:
        raise ValueError(f"Protocol cell {key[0]}/{key[1]} uses an unsupported selector: {factors.selector}")
    _require_equal(key, "selector_config.method", execution_cell.selector_config.method, expected_method)

    expected_intervention = _SELECTOR_INTERVENTIONS.get(factors.selector, "none")
    _require_equal(key, "intervention_mode", execution_cell.intervention_mode, expected_intervention)
    if expected_intervention == "none" and execution_cell.intervention_id is not None:
        raise ValueError(f"Execution cell {key[0]}/{key[1]} binds an unregistered intervention")
    if expected_intervention != "none" and execution_cell.intervention_id is None:
        raise ValueError(f"Execution cell {key[0]}/{key[1]} requires a preregistered intervention_id")

    expected_selection_mode = "nested" if factors.selector == "coverage_quality_nested" else "direct"
    _require_equal(key, "selection_mode", execution_cell.selection_mode, expected_selection_mode)

    expected_organization = _ORGANIZATIONS.get(factors.organization)
    if expected_organization is None:
        raise ValueError(f"Protocol cell {key[0]}/{key[1]} uses an unsupported organization: {factors.organization}")
    _require_equal(key, "organization", execution_cell.organization, expected_organization)
    if factors.organization == "random_form_seed_101":
        _require_equal(key, "organization_seed", execution_cell.organization_seed, 101)

    _require_float_equal(
        key,
        "source_budget_fraction",
        execution_cell.source_budget_fraction,
        factors.source_budget_fraction,
    )
    _require_float_equal(
        key,
        "repetition_factor",
        execution_cell.repetition_factor,
        factors.repetition_factor,
    )
    _require_float_equal(key, "label_noise_rate", execution_cell.label_noise_rate, factors.noise_rate)
    observed_noise_kind = "none" if execution_cell.label_noise_rate == 0 else "label_corruption"
    _require_equal(key, "noise_kind", observed_noise_kind, factors.noise_kind)

    model_profiles = {profile.profile_id: profile for profile in spec.model_profiles}
    model_profile_id = execution_cell.model_profile_id or spec.default_model_profile_id
    model_profile = model_profiles[model_profile_id]
    _require_equal(key, "model_profile_id/prior_profile", model_profile_id, factors.prior_profile)
    _require_equal(key, "model_profile.model_id/base_model", model_profile.model_id, factors.base_model)
    _require_equal(key, "contract.training.base_model", contract.training.base_model, factors.base_model)

    expected_prior_aware = factors.selector == "coverage_quality_prior_aware"
    if factors.selector in {"coverage_quality_prior_aware", "coverage_quality_no_prior"}:
        _require_equal(key, "prior_aware", execution_cell.prior_aware, expected_prior_aware)
    elif execution_cell.prior_aware:
        raise ValueError(f"Execution cell {key[0]}/{key[1]} enables an unregistered model prior")

    expected_kind = "quality_fidelity" if study.study_id == "A4_quality_fidelity" else _STAGE_KINDS[study.stage]
    _require_equal(key, "kind", execution_cell.kind, expected_kind)


def validate_execution_against_protocol_suite_contract(
    spec: AblationExecutionSpec,
    protocol: AblationProtocol,
    suite: BatchSuite,
    contract: ProductionContract,
    batch_inputs: list[AblationBatchInput],
) -> dict[str, Any]:
    """Reject an execution that is not an exact implementation of frozen inputs.

    This validator intentionally uses semantic model digests rather than file
    hashes.  Reformatting YAML therefore cannot create a new protocol identity,
    while changing any registered value necessarily invalidates the binding.
    """

    expected_digests = {
        "protocol_sha256": ablation_protocol_digest(protocol),
        "suite_sha256": suite_digest(suite),
        "contract_sha256": contract_digest(contract),
    }
    for field, expected in expected_digests.items():
        observed = getattr(spec, field)
        if observed != expected:
            raise ValueError(f"Execution {field} mismatch: expected={expected}, observed={observed}")

    if spec.training_seeds != contract.training.seeds:
        raise ValueError(
            "Execution training_seeds must exactly match the frozen production contract: "
            f"expected={contract.training.seeds}, observed={spec.training_seeds}"
        )

    required = _required_protocol_cells(protocol)
    execution_cells = {(cell.study_id, cell.cell_id): cell for cell in spec.cells}
    missing = sorted(set(required) - set(execution_cells))
    extra = sorted(set(execution_cells) - set(required))
    if missing or extra:
        raise ValueError(
            f"Execution cells must exactly cover proxy SFT protocol cells; missing={missing}, extra={extra}"
        )
    for key, execution_cell in execution_cells.items():
        study, protocol_cell = required[key]
        _validate_cell_binding(
            execution_cell,
            study=study,
            protocol_cell=protocol_cell,
            spec=spec,
            contract=contract,
        )

    registered_batches = {batch.batch_id: batch for batch in suite.batches}
    supplied_batches = {batch.batch_id: batch for batch in batch_inputs}
    if len(supplied_batches) != len(batch_inputs):
        raise ValueError("Ablation batch inputs must have unique batch_id values")
    missing_batches = sorted(set(registered_batches) - set(supplied_batches))
    extra_batches = sorted(set(supplied_batches) - set(registered_batches))
    if missing_batches or extra_batches:
        raise ValueError(
            "Ablation batch inputs must exactly cover the frozen suite; "
            f"missing={missing_batches}, extra={extra_batches}"
        )
    for batch_id, batch_input in supplied_batches.items():
        registered = registered_batches[batch_id]
        if registered.content_sha256 is None:
            raise ValueError(f"Frozen suite batch {batch_id} has no content_sha256")
        if batch_input.batch_content_sha256 != registered.content_sha256:
            raise ValueError(
                f"Batch {batch_id} content hash mismatch: expected={registered.content_sha256}, "
                f"observed={batch_input.batch_content_sha256}"
            )
        if batch_input.lineage_group != registered.lineage_group:
            raise ValueError(
                f"Batch {batch_id} lineage mismatch: expected={registered.lineage_group!r}, "
                f"observed={batch_input.lineage_group!r}"
            )

    return {
        "schema_version": "stage-ablation-execution-validation-v1",
        "execution_id": spec.execution_id,
        "execution_sha256": execution_digest(spec),
        **expected_digests,
        "training_seeds": list(spec.training_seeds),
        "batch_ids": sorted(supplied_batches),
        "batch_count": len(supplied_batches),
        "executable_cell_count": len(execution_cells),
        "excluded_protocol_cells": [
            {
                "study_id": study.study_id,
                "cell_id": cell.cell_id,
                "reason": (
                    "static_no_sft" if cell.factors.training_mode == "no_sft" else "fully_open_meditron_external"
                ),
            }
            for study in protocol.studies
            for cell in study.cells
            if cell.factors.training_mode == "no_sft" or cell.factors.generator == "fully_open_meditron"
        ],
    }


def _safe_relative_uri(uri: str, *, label: str) -> PurePosixPath:
    normalized = uri.replace("\\", "/")
    path = PurePosixPath(normalized)
    if "://" in normalized or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be an unambiguous relative artifact path: {uri!r}")
    return path


def _prefix_job_paths(job: AblationJobRecord, batch_id: str) -> AblationJobRecord:
    updates = {}
    for field in ("train_uri", "lineage_uri", "cell_manifest_uri"):
        relative = _safe_relative_uri(getattr(job, field), label=field)
        updates[field] = (PurePosixPath("batches") / batch_id / relative).as_posix()
    return job.model_copy(update=updates)


def _verify_job_artifacts(root: Path, job: AblationJobRecord) -> None:
    for uri_field, hash_field in (
        ("train_uri", "train_sha256"),
        ("lineage_uri", "lineage_sha256"),
        ("cell_manifest_uri", "cell_manifest_sha256"),
    ):
        relative = _safe_relative_uri(getattr(job, uri_field), label=uri_field)
        path = root.joinpath(*relative.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Job artifact escapes the execution root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Job artifact does not exist: {path}")
        observed = file_sha256(path)
        expected = getattr(job, hash_field)
        if observed != expected:
            raise ValueError(f"Job artifact hash mismatch for {path}: expected={expected}, observed={observed}")


def _load_batch_materialization(batch_root: Path) -> tuple[dict[str, Any], list[AblationJobRecord]]:
    manifest_path = batch_root / "materialization_manifest.json"
    jobs_path = batch_root / "jobs.jsonl"
    if not manifest_path.is_file() or not jobs_path.is_file():
        raise FileNotFoundError(f"Batch materialization is incomplete: {batch_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Batch materialization manifest must be an object: {manifest_path}")
    jobs = [AblationJobRecord.model_validate(row) for row in read_jsonl(jobs_path)]
    return manifest, jobs


def _not_identifiable_exclusions(
    batch_root: Path,
    manifest: dict[str, Any],
    *,
    cells: dict[tuple[str, str], AblationCellExecution],
) -> list[dict[str, Any]]:
    exclusions = []
    for summary in manifest.get("cells", []):
        if summary.get("status") != "not_identifiable":
            continue
        key = (summary.get("study_id"), summary.get("cell_id"))
        cell = cells.get(key)
        if cell is None or cell.kind != "quality_fidelity":
            raise ValueError(f"Only registered quality-fidelity cells may be not_identifiable: {key}")
        relative = _safe_relative_uri(summary["cell_manifest_uri"], label="cell_manifest_uri")
        cell_manifest_path = batch_root.joinpath(*relative.parts)
        payload = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
        exclusions.append(
            {
                "batch_id": manifest["batch_id"],
                "study_id": key[0],
                "cell_id": key[1],
                "reason": payload.get("quality_fidelity", {}).get("reason") or "quality_fidelity_not_identifiable",
                "cell_manifest_uri": (PurePosixPath("batches") / manifest["batch_id"] / relative).as_posix(),
                "cell_manifest_sha256": file_sha256(cell_manifest_path),
            }
        )
    return exclusions


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        if path.name == "execution_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        artifacts.append({"uri": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    return artifacts


def materialize_ablation_execution(
    spec: AblationExecutionSpec,
    batch_inputs: list[AblationBatchInput],
    output_dir: str | Path,
    *,
    protocol: AblationProtocol,
    suite: BatchSuite,
    contract: ProductionContract,
    exclude_not_identifiable_a4: bool = False,
) -> dict[str, Any]:
    """Atomically materialize and audit the complete multi-batch job matrix.

    Set ``exclude_not_identifiable_a4`` only when the preregistered policy
    allows batches with one eligible quality bucket to omit A4 training jobs.
    Every such omission is recorded in ``execution_manifest.json``; no other
    missing batch/cell/seed combination is accepted.
    """

    validation = validate_execution_against_protocol_suite_contract(
        spec,
        protocol,
        suite,
        contract,
        batch_inputs,
    )
    output_root = Path(output_dir).resolve()
    partial_root = output_root.with_name(output_root.name + ".partial")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite ablation execution output: {output_root}")
    if partial_root.exists():
        raise FileExistsError(f"Incomplete prior ablation execution must be inspected: {partial_root}")

    batch_by_id = {batch.batch_id: batch for batch in batch_inputs}
    cells = {(cell.study_id, cell.cell_id): cell for cell in spec.cells}
    all_jobs: list[AblationJobRecord] = []
    batch_summaries = []
    exclusions: list[dict[str, Any]] = []
    partial_root.mkdir(parents=True, exist_ok=False)
    try:
        for batch_id in validation["batch_ids"]:
            batch = batch_by_id[batch_id]
            batch_root = partial_root / "batches" / batch_id
            materialize_ablation_batch(spec, batch, batch_root)
            batch_manifest, local_jobs = _load_batch_materialization(batch_root)
            if batch_manifest.get("batch_id") != batch_id:
                raise ValueError(f"Materializer returned the wrong batch_id for {batch_id}")
            if batch_manifest.get("batch_input_sha256") != batch_input_digest(batch):
                raise ValueError(f"Materializer returned the wrong batch input digest for {batch_id}")
            if batch_manifest.get("execution_sha256") != execution_digest(spec):
                raise ValueError(f"Materializer returned the wrong execution digest for {batch_id}")

            batch_exclusions = _not_identifiable_exclusions(batch_root, batch_manifest, cells=cells)
            if batch_exclusions and not exclude_not_identifiable_a4:
                identities = [(row["study_id"], row["cell_id"]) for row in batch_exclusions]
                raise ValueError(
                    f"Batch {batch_id} has not-identifiable A4 cells {identities}; "
                    "rerun only with an explicitly preregistered exclusion policy"
                )
            exclusions.extend(batch_exclusions)

            for local_job in local_jobs:
                if local_job.batch_id != batch_id:
                    raise ValueError(f"Batch {batch_id} emitted a job for {local_job.batch_id}")
                global_job = _prefix_job_paths(local_job, batch_id)
                _verify_job_artifacts(partial_root, global_job)
                all_jobs.append(global_job)

            materialization_manifest = batch_root / "materialization_manifest.json"
            batch_summaries.append(
                {
                    "batch_id": batch_id,
                    "lineage_group": batch.lineage_group,
                    "batch_content_sha256": batch.batch_content_sha256,
                    "batch_input_sha256": batch_input_digest(batch),
                    "root_uri": (PurePosixPath("batches") / batch_id).as_posix(),
                    "materialization_manifest_uri": (
                        PurePosixPath("batches") / batch_id / "materialization_manifest.json"
                    ).as_posix(),
                    "materialization_manifest_sha256": file_sha256(materialization_manifest),
                    "local_jobs_sha256": file_sha256(batch_root / "jobs.jsonl"),
                    "job_count": len(local_jobs),
                    "excluded_not_identifiable_count": len(batch_exclusions),
                }
            )

        observed_keys = [(job.batch_id, job.study_id, job.cell_id, job.seed) for job in all_jobs]
        if len(observed_keys) != len(set(observed_keys)):
            raise ValueError("Materialized execution contains duplicate batch/cell/seed jobs")
        excluded_keys = {(row["batch_id"], row["study_id"], row["cell_id"]) for row in exclusions}
        expected_keys = {
            (batch_id, study_id, cell_id, seed)
            for batch_id in validation["batch_ids"]
            for study_id, cell_id in cells
            for seed in spec.training_seeds
            if (batch_id, study_id, cell_id) not in excluded_keys
        }
        observed_key_set = set(observed_keys)
        if observed_key_set != expected_keys:
            missing = sorted(expected_keys - observed_key_set)
            extra = sorted(observed_key_set - expected_keys)
            raise ValueError(f"Incomplete ablation job matrix; missing={missing}, extra={extra}")

        all_jobs.sort(key=lambda row: (row.batch_id, row.study_id, row.cell_id, row.seed))
        jobs_path = partial_root / "jobs.jsonl"
        write_jsonl(jobs_path, (job.model_dump(mode="json") for job in all_jobs))
        inventory = _artifact_inventory(partial_root)
        execution_manifest = {
            "schema_version": "stage-ablation-execution-manifest-v1",
            "execution_id": spec.execution_id,
            "execution_sha256": execution_digest(spec),
            "protocol_sha256": spec.protocol_sha256,
            "suite_sha256": spec.suite_sha256,
            "contract_sha256": spec.contract_sha256,
            "training_seeds": list(spec.training_seeds),
            "batch_count": len(batch_inputs),
            "executable_cell_count": len(cells),
            "expected_job_count": len(expected_keys),
            "job_count": len(all_jobs),
            "complete_matrix": True,
            "exclude_not_identifiable_a4": exclude_not_identifiable_a4,
            "excluded_not_identifiable_count": len(exclusions),
            "excluded_not_identifiable": exclusions,
            "jobs_uri": "jobs.jsonl",
            "jobs_sha256": file_sha256(jobs_path),
            "batches": batch_summaries,
            "excluded_protocol_cells": validation["excluded_protocol_cells"],
            "artifact_count": len(inventory),
            "artifacts": inventory,
        }
        manifest_path = partial_root / "execution_manifest.json"
        write_json(manifest_path, execution_manifest)
        partial_root.replace(output_root)
    except Exception:
        if partial_root.exists():
            shutil.rmtree(partial_root)
        raise

    return {
        "output_dir": str(output_root),
        "execution_manifest": str(output_root / "execution_manifest.json"),
        "execution_manifest_sha256": file_sha256(output_root / "execution_manifest.json"),
        "jobs": str(output_root / "jobs.jsonl"),
        "job_count": len(all_jobs),
        "excluded_not_identifiable_count": len(exclusions),
    }


__all__ = [
    "load_ablation_batch_input",
    "load_ablation_execution_spec",
    "materialize_ablation_execution",
    "validate_execution_against_protocol_suite_contract",
]
