"""Validation and content freezing for candidate data batches."""

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from ai_ready.utils.io import atomic_text
from ai_ready.valuation.schema import (
    BatchSpec,
    BatchSuite,
    SourceSpec,
    ValidationReport,
)

BATCH_CONTENT_HASH_SCHEMA = "batch-content-v1"
FACTORIAL_COMBINATIONS = {
    (coverage, relevance, redundancy)
    for coverage in ("broad", "narrow")
    for relevance in ("high", "low")
    for redundancy in ("high", "low")
}


def _update_frame(digest: Any, value: bytes) -> None:
    """Add one unambiguous length-prefixed value to a streaming digest."""

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _construction_metadata(batch: BatchSpec, sources: list[SourceSpec]) -> dict:
    """Return only metadata that changes how this batch was constructed.

    The allowlist is deliberate. Experimental choices such as a selector,
    token budget, output directory, split, or display ID must never change the
    identity of the source batch whose value is being estimated.
    """

    registered = {source.source_id: source for source in sources}
    source_identity = []
    for source_id in sorted(batch.source_ids):
        if source_id not in registered:
            raise ValueError(f"Batch {batch.batch_id} references unknown source: {source_id}")
        source = registered[source_id]
        source_identity.append(
            {
                "source_id": source.source_id,
                "version": source.version,
                "content_sha256": source.content_sha256,
                "knowledge_cutoff": source.knowledge_cutoff,
            }
        )
    return {
        "kind": batch.kind,
        "sources": source_identity,
        "lineage_group": batch.lineage_group,
        "factors": batch.factors.model_dump(mode="json"),
        "construction_notes": sorted(batch.construction_notes),
    }


def content_fingerprint(path: str | Path, construction_metadata: Mapping[str, Any]) -> str:
    """Hash a file or directory using construction metadata, relative names and bytes.

    A directory may be copied to another machine or renamed without changing
    this digest. Files inside it retain their POSIX-style relative names in
    the hash, so a rename, added file, removed file, or byte change is visible.
    Symlinks are rejected because following an external target would make the
    apparent batch depend on state outside the registered directory.
    """

    requested_path = Path(path)
    if requested_path.is_symlink():
        raise ValueError(f"Batch input cannot be a symlink: {requested_path}")
    root = requested_path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Batch input does not exist: {root}")

    if root.is_file():
        files = [(unicodedata.normalize("NFC", root.name), root)]
    elif root.is_dir():
        files = []
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"Batch input cannot contain symlinks: {candidate}")
            if candidate.is_file():
                relative_name = unicodedata.normalize("NFC", candidate.relative_to(root).as_posix())
                files.append((relative_name, candidate))
        files.sort(key=lambda item: item[0].encode("utf-8"))
        if not files:
            raise ValueError(f"Batch input directory contains no files: {root}")
    else:
        raise ValueError(f"Batch input must be a regular file or directory: {root}")

    digest = hashlib.sha256()
    _update_frame(digest, BATCH_CONTENT_HASH_SCHEMA.encode("ascii"))
    metadata = json.dumps(
        dict(construction_metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _update_frame(digest, metadata)
    for relative_name, file_path in files:
        _update_frame(digest, relative_name.encode("utf-8"))
        expected_size = file_path.stat().st_size
        digest.update(expected_size.to_bytes(8, byteorder="big", signed=False))
        observed_size = 0
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                observed_size += len(chunk)
                digest.update(chunk)
        if observed_size != expected_size:
            raise RuntimeError(f"Batch input changed while it was being hashed: {file_path}")
    return digest.hexdigest()


def batch_content_fingerprint(suite: BatchSuite, batch: BatchSpec, input_path: str | Path) -> str:
    """Fingerprint one batch without including its location or experiment controls."""

    return content_fingerprint(input_path, _construction_metadata(batch, suite.sources))


def _local_path(input_uri: str, suite_path: Path | None) -> Path | None:
    if "://" in input_uri:
        parsed = urlparse(input_uri)
        if parsed.scheme != "file":
            return None
        if parsed.query or parsed.fragment:
            raise ValueError(f"Batch file URI cannot contain a query or fragment: {input_uri}")
        decoded_path = unquote(parsed.path)
        if os.name == "nt":
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                candidate = Path(f"//{parsed.netloc}{decoded_path}")
            else:
                # RFC 8089 drive paths use /C:/...; pathlib on Windows expects C:/....
                if re.match(r"^/[A-Za-z]:/", decoded_path):
                    decoded_path = decoded_path[1:]
                candidate = Path(decoded_path)
        elif parsed.netloc and parsed.netloc.lower() != "localhost":
            candidate = Path(f"//{parsed.netloc}{decoded_path}")
        else:
            candidate = Path(decoded_path)
    else:
        candidate = Path(input_uri)
    if not candidate.is_absolute() and suite_path is not None:
        candidate = suite_path.resolve().parent / candidate
    return candidate.resolve()


def _relocate_local_uri(input_path: Path, frozen_suite_path: Path) -> str:
    """Keep a relative input URI valid after the suite YAML moves."""

    try:
        relative = os.path.relpath(input_path, start=frozen_suite_path.parent)
    except ValueError:
        # Windows cannot express a relative path across drives. A file URI is
        # explicit and remains distinguishable from a remote data-service URI.
        return input_path.as_uri()
    return Path(relative).as_posix()


def freeze_batch_suite(source_path: str | Path, output_path: str | Path) -> dict:
    """Fill missing batch content hashes and write a new suite YAML.

    Existing local hashes are recomputed and must match; pinned remote hashes
    are preserved. The source YAML is never overwritten: a frozen suite is a
    new, reviewable artifact rather than a silent registry mutation.
    """

    from ai_ready.valuation.contracts import file_sha256, load_batch_suite, suite_digest

    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("Frozen batch suite must use a new output path; in-place overwrite is forbidden")
    if output_path.exists():
        raise FileExistsError(f"Frozen batch suite already exists: {output_path}")

    suite = load_batch_suite(source_path).model_copy(deep=True)
    blueprint_path = _local_path(suite.capability_blueprint_uri, source_path)
    if blueprint_path is not None:
        if not blueprint_path.is_file():
            raise FileNotFoundError(f"Capability blueprint does not exist: {blueprint_path}")
        observed_blueprint_hash = _file_sha256(blueprint_path)
        if observed_blueprint_hash != suite.capability_blueprint_sha256:
            raise ValueError(
                "Capability blueprint content does not match capability_blueprint_sha256; "
                f"expected={suite.capability_blueprint_sha256}, observed={observed_blueprint_hash}"
            )
        suite.capability_blueprint_uri = _relocate_local_uri(blueprint_path, output_path)
    newly_frozen: list[str] = []
    preserved: list[str] = []
    fingerprints: dict[str, str] = {}
    for batch in suite.batches:
        input_path = _local_path(batch.input_uri, source_path)
        if input_path is None:
            if batch.content_sha256 is None:
                raise ValueError(
                    f"Batch {batch.batch_id} uses a non-local input URI and has no content hash: {batch.input_uri}"
                )
            preserved.append(batch.batch_id)
            fingerprints[batch.batch_id] = batch.content_sha256
            continue

        observed = batch_content_fingerprint(suite, batch, input_path)
        if batch.content_sha256 is None:
            batch.content_sha256 = observed
            newly_frozen.append(batch.batch_id)
        elif batch.content_sha256 != observed:
            # A stale or invented digest must never be blessed by copying it to a file named "frozen".
            raise ValueError(
                f"Batch {batch.batch_id} content_sha256 does not match its local content; "
                f"expected={batch.content_sha256}, observed={observed}"
            )
        else:
            preserved.append(batch.batch_id)
        fingerprints[batch.batch_id] = batch.content_sha256
        batch.input_uri = _relocate_local_uri(input_path, output_path)
        batch.input_uri = _relocate_local_uri(input_path, output_path)

    payload = yaml.safe_dump(
        suite.model_dump(mode="json"),
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    atomic_text(output_path, payload)
    return {
        "schema_version": "frozen-batch-suite-v1",
        "hash_schema": BATCH_CONTENT_HASH_SCHEMA,
        "source_file": source_path.name,
        "frozen_file": str(output_path),
        "frozen_file_sha256": file_sha256(output_path),
        "suite_sha256": suite_digest(suite),
        "newly_frozen_batch_ids": newly_frozen,
        "preserved_batch_ids": preserved,
        "batch_content_sha256": fingerprints,
    }


def validate_feasibility_design(
    suite: BatchSuite,
    *,
    suite_path: str | Path | None = None,
    check_paths: bool = False,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if suite_path is not None:
        blueprint_path = _local_path(suite.capability_blueprint_uri, Path(suite_path))
        if blueprint_path is None:
            warnings.append("The remote capability blueprint cannot be re-hashed locally")
        elif not blueprint_path.is_file():
            errors.append(f"Capability blueprint does not exist: {blueprint_path}")
        else:
            observed_blueprint_hash = _file_sha256(blueprint_path)
            if observed_blueprint_hash != suite.capability_blueprint_sha256:
                errors.append(
                    "Capability blueprint content does not match capability_blueprint_sha256; "
                    f"expected={suite.capability_blueprint_sha256}, observed={observed_blueprint_hash}"
                )
            blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
            if not isinstance(blueprint, dict):
                errors.append("Capability blueprint must contain a mapping")
            else:
                if blueprint.get("benchmark_id") != suite.benchmark_id:
                    errors.append("Capability blueprint and batch suite use different benchmark IDs")
                # The selection-side blueprint must contain descriptions only.
                # A true value here would make benchmark-conditioned selection leak test items.
                if blueprint.get("test_items_included") is not False:
                    errors.append("Capability blueprint must explicitly declare test_items_included: false")
                capability_rows = blueprint.get("capabilities")
                if not isinstance(capability_rows, list) or any(not isinstance(row, dict) for row in capability_rows):
                    errors.append("Capability blueprint capabilities must be a list of mappings")
                else:
                    capability_ids = [row.get("capability_id") for row in capability_rows]
                    if capability_ids != suite.capability_blueprint:
                        errors.append("Capability IDs in the blueprint and batch suite do not match in order")

    factorial = [batch for batch in suite.batches if batch.kind == "factorial"]
    natural = [batch for batch in suite.batches if batch.kind == "natural_check"]

    actual_combinations = {
        (batch.factors.coverage_scope, batch.factors.target_relevance, batch.factors.redundancy) for batch in factorial
    }
    if len(suite.batches) > 12 and suite.intended_claim == "feasibility_and_method_screening":
        warnings.append("The feasibility design normally uses 8-12 batches; the suite contains more than 12")
    if len(factorial) != 8 or actual_combinations != FACTORIAL_COMBINATIONS:
        missing = sorted(FACTORIAL_COMBINATIONS - actual_combinations)
        repeated = len(factorial) - len(actual_combinations)
        errors.append(
            "The controlled arm must contain each coverage × relevance × redundancy combination exactly once; "
            f"missing={missing}, repeated={max(repeated, 0)}"
        )
    if len(natural) < 4:
        errors.append("At least four natural-check batches are required for the exploratory design")
    if len({batch.lineage_group for batch in natural}) < 2:
        warnings.append("Natural-check batches should include at least two independent lineage groups")
    if not any(batch.split == "evaluation" for batch in natural):
        warnings.append("At least one natural-check batch should be held out for evaluation")

    registered = {source.source_id: source for source in suite.sources}
    for batch in suite.batches:
        for source_id in batch.source_ids:
            source = registered[source_id]
            if source.license_status == "audit_only":
                errors.append(f"Batch {batch.batch_id} includes audit-only source {source_id} in training inputs")
            elif source.license_status == "unknown":
                warnings.append(f"Batch {batch.batch_id} includes source {source_id} with unknown training permission")
            elif "model-training" not in {use.casefold() for use in source.allowed_uses}:
                errors.append(f"Batch {batch.batch_id} source {source_id} is not registered for model-training use")
        if batch.content_sha256 is None:
            warnings.append(f"Batch {batch.batch_id} has no frozen content_sha256 yet")
        if check_paths:
            path = _local_path(batch.input_uri, Path(suite_path) if suite_path else None)
            if path is not None and not path.exists():
                errors.append(f"Batch {batch.batch_id} input does not exist: {path}")
            elif path is not None and batch.content_sha256 is not None:
                observed = batch_content_fingerprint(suite, batch, path)
                if observed != batch.content_sha256:
                    errors.append(
                        f"Batch {batch.batch_id} content_sha256 does not match its registered content; "
                        f"expected={batch.content_sha256}, observed={observed}"
                    )

    # B01-B08 are paired experimental conditions, so they cannot be counted as
    # independent suppliers when reporting confirmatory correlation.
    if suite.intended_claim == "confirmatory" and len({batch.lineage_group for batch in suite.batches}) < 8:
        errors.append("Confirmatory studies require more independent lineage groups than this suite provides")
    elif suite.intended_claim == "feasibility_and_method_screening":
        warnings.append("Results from 8-12 batches are exploratory and must not be reported as stable correlation")

    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=sorted(set(warnings)),
        summary={
            "batch_count": len(suite.batches),
            "factorial_batches": len(factorial),
            "natural_check_batches": len(natural),
            "lineage_groups": len({batch.lineage_group for batch in suite.batches}),
            "sources": len(suite.sources),
        },
    )
