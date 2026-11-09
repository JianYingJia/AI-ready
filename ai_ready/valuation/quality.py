"""Source-grounded SFT records and full-traversal production checks."""

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.valuation.schema import ValidationReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SFTSample(StrictModel):
    sample_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    semantic_unit_ids: list[str] = Field(min_length=1)
    instruction: str = Field(min_length=1)
    response: str = Field(min_length=1)
    evidence_spans: list[str] = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    task_type: Literal["grounded_qa", "conditional_judgment", "clinical_vignette"] | None = None
    generator: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    # These fields bind a supervision row to the frozen production recipe.
    # They remain optional at the generic schema boundary so historical audit
    # files can still be inspected; the formal LLaMAFactory converter requires
    # and checks every field before a row may enter a training bundle.
    teacher_model: str | None = Field(default=None, min_length=1)
    teacher_revision: str | None = Field(default=None, min_length=1)
    prompt_sha256: str | None = None
    quality_gate_sha256: str | None = None

    @field_validator("prompt_sha256", "quality_gate_sha256")
    @classmethod
    def validate_recipe_hashes(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"{info.field_name} must be a 64-character SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def require_unique_traceability(self):
        if len(self.semantic_unit_ids) != len(set(self.semantic_unit_ids)):
            raise ValueError("SFT sample semantic_unit_ids must be unique")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("SFT sample source_ids must be unique")
        return self


class UnitDisposition(StrictModel):
    semantic_unit_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected", "failed"]
    sample_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_payload(self):
        if self.status == "accepted" and not self.sample_ids:
            raise ValueError("Accepted units must identify at least one SFT sample")
        if self.status != "accepted" and not self.reason_codes:
            raise ValueError("Rejected or failed units must record a reason")
        if self.status != "accepted" and self.sample_ids:
            raise ValueError("Rejected or failed units cannot contribute SFT samples")
        return self


def validate_full_traversal(
    eligible_unit_ids: list[str],
    dispositions: list[UnitDisposition],
    samples: list[SFTSample],
) -> ValidationReport:
    """Require an explicit outcome for every source unit in the reference run."""

    errors, warnings = [], []
    if len(eligible_unit_ids) != len(set(eligible_unit_ids)):
        errors.append("eligible_unit_ids contains duplicates")
    disposition_ids = [row.semantic_unit_id for row in dispositions]
    duplicates = sorted(unit_id for unit_id, count in Counter(disposition_ids).items() if count > 1)
    if duplicates:
        errors.append(f"Multiple dispositions were recorded for units: {duplicates}")
    missing = sorted(set(eligible_unit_ids) - set(disposition_ids))
    unexpected = sorted(set(disposition_ids) - set(eligible_unit_ids))
    if missing:
        errors.append(f"Eligible units have no accepted/rejected/failed disposition: {missing}")
    if unexpected:
        errors.append(f"Dispositions reference non-eligible units: {unexpected}")

    disposition_by_id = {row.semantic_unit_id: row for row in dispositions}
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        errors.append("SFT sample IDs must be unique")
    sample_batches = {sample.batch_id for sample in samples}
    if len(sample_batches) > 1:
        errors.append(f"One production traversal cannot mix sample batches: {sorted(sample_batches)}")
    supervision_signatures: dict[tuple[str, str], list[str]] = {}
    for sample in samples:
        signature = (sample.instruction.strip(), sample.response.strip())
        supervision_signatures.setdefault(signature, []).append(sample.sample_id)
    duplicated_samples = [sample_ids for sample_ids in supervision_signatures.values() if len(sample_ids) > 1]
    if duplicated_samples:
        # Exact copies with new IDs would conceal a batch's low production yield
        # and inflate its nominal training-token count.
        errors.append(f"Duplicate SFT supervision records are forbidden: {duplicated_samples}")
    for disposition in dispositions:
        for sample_id in disposition.sample_ids:
            sample = sample_by_id.get(sample_id)
            if sample is None:
                errors.append(f"Disposition references unknown sample: {sample_id}")
            elif disposition.semantic_unit_id not in sample.semantic_unit_ids:
                errors.append(
                    f"Sample {sample_id} does not trace back to disposition unit {disposition.semantic_unit_id}"
                )
    orphan_samples = sorted(
        sample.sample_id
        for sample in samples
        if not any(sample.sample_id in disposition.sample_ids for disposition in dispositions)
    )
    if orphan_samples:
        errors.append(f"SFT samples have no accepted source disposition: {orphan_samples}")

    # A multi-unit sample is valid only when every contributing unit passed the
    # quality gate and explicitly points back to that sample. Otherwise a
    # rejected medical claim could hitchhike inside an accepted unit's answer.
    for sample in samples:
        for unit_id in sample.semantic_unit_ids:
            disposition = disposition_by_id.get(unit_id)
            if disposition is None:
                errors.append(f"Sample {sample.sample_id} references a unit with no disposition: {unit_id}")
            elif disposition.status != "accepted":
                errors.append(f"Sample {sample.sample_id} references non-accepted unit {unit_id}")
            elif sample.sample_id not in disposition.sample_ids:
                errors.append(f"Sample {sample.sample_id} is not registered by contributing disposition unit {unit_id}")

    counts = Counter(disposition.status for disposition in dispositions)
    total = len(eligible_unit_ids)
    accepted_yield = counts["accepted"] / total if total else 0.0
    if counts["failed"]:
        warnings.append("Generation failures are retained as production-cost evidence")
    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        summary={
            "eligible_units": total,
            "accepted_units": counts["accepted"],
            "rejected_units": counts["rejected"],
            "failed_units": counts["failed"],
            "accepted_unit_yield": accepted_yield,
            "accepted_samples": len(samples),
        },
    )
