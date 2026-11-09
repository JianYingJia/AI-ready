"""Deterministic data construction for stage-level valuation ablations.

The production proxy and Fully Open Meditron branches deliberately have a
strict, narrow contract.  Mechanism experiments need additional controlled
operations (paired supervision forms, rule knockout/rescue, repetition, and
label corruption) that must never be mistaken for production data.  This
module provides that separate execution layer and binds every derived training
row to a frozen candidate pool, execution specification, and model checkpoint.

Selection is completed from :class:`ProxyUnit` and
:class:`CoverageObligation` records before the candidate SFT bank is read.
Consequently, text generated for a mechanism experiment cannot influence which
source units enter the sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.core.schema import stable_id
from ai_ready.utils.io import write_json, write_jsonl
from ai_ready.valuation.ablation_selection import candidate_pool_digest
from ai_ready.valuation.contracts import file_sha256
from ai_ready.valuation.quality import SFTSample
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
    select_proxy,
    selection_config_digest,
)

TASK_TYPES = ("grounded_qa", "conditional_judgment", "clinical_vignette")
SUPPORTED_FRACTIONS = (0.25, 0.5, 1.0)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _validate_sha256(value: str, field_name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def _canonical_digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fraction_key(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _derived_seed(*parts: str | int) -> int:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class CapabilityWeightProfile(_StrictModel):
    """Task demand weights that are independent of source-unit mining."""

    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    weights: dict[str, float] = Field(min_length=1)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not capability_id.strip() for capability_id in value):
            raise ValueError("Capability IDs cannot be empty")
        if any(not math.isfinite(weight) or weight <= 0 for weight in value.values()):
            raise ValueError("Capability weights must be finite and positive")
        return dict(sorted(value.items()))


class ModelProfile(_StrictModel):
    """A real model state and its independently measured capability prior."""

    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    checkpoint_uri: str = Field(min_length=1)
    checkpoint_sha256: str
    calibration_uri: str = Field(min_length=1)
    calibration_sha256: str
    capability_scores: dict[str, float] = Field(min_length=1)
    minimum_gap: float = Field(default=0.05, gt=0, le=1)

    @field_validator("checkpoint_sha256", "calibration_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("capability_scores")
    @classmethod
    def validate_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not capability_id.strip() for capability_id in value):
            raise ValueError("Capability IDs cannot be empty")
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in value.values()):
            raise ValueError("Capability calibration scores must be finite values in [0, 1]")
        return dict(sorted(value.items()))

    @property
    def capability_gaps(self) -> dict[str, float]:
        return {
            capability_id: max(self.minimum_gap, 1.0 - score) for capability_id, score in self.capability_scores.items()
        }


class RuleInterventionSpec(_StrictModel):
    """Pre-registered, size-matched target/replacement dependency groups."""

    intervention_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    target_group_ids: list[str] = Field(min_length=1)
    replacement_group_ids: list[str] = Field(min_length=1)
    maximum_token_difference: int = Field(default=0, ge=0)

    @field_validator("target_group_ids", "replacement_group_ids")
    @classmethod
    def validate_group_ids(cls, value: list[str], info) -> list[str]:
        if any(not group_id.strip() for group_id in value):
            raise ValueError(f"{info.field_name} cannot contain empty IDs")
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must contain unique IDs")
        return value

    @model_validator(mode="after")
    def validate_pairs(self):
        if len(self.target_group_ids) != len(self.replacement_group_ids):
            raise ValueError("Target and replacement group lists must have the same length")
        if set(self.target_group_ids) & set(self.replacement_group_ids):
            raise ValueError("Target and replacement groups must be disjoint")
        return self


class AblationCellExecution(_StrictModel):
    """One executable cell bound to the factors needed to build its SFT."""

    study_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    kind: Literal[
        "selection",
        "quality_fidelity",
        "organization",
        "learnability",
        "robustness",
        "base_model_prior",
    ]
    capability_profile_id: str = Field(min_length=1)
    selector_config: ProxySelectionConfig
    source_budget_fraction: float
    selection_mode: Literal["direct", "nested"] = "direct"
    organization: Literal["direct", "mixed", "random"] = "mixed"
    organization_seed: int = 101
    intervention_id: str | None = None
    intervention_mode: Literal["none", "include", "knockout", "rescue"] = "none"
    repetition_factor: int = Field(default=1, ge=1)
    label_noise_rate: float = 0.0
    label_noise_seed: int = 101
    model_profile_id: str | None = None
    prior_aware: bool = False

    @field_validator("source_budget_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if not any(math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-12) for allowed in SUPPORTED_FRACTIONS):
            raise ValueError("source_budget_fraction must be one of 0.25, 0.5, or 1.0")
        return value

    @field_validator("label_noise_rate")
    @classmethod
    def validate_noise_rate(cls, value: float) -> float:
        if not any(math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-12) for allowed in (0.0, 0.1, 0.3)):
            raise ValueError("label_noise_rate must be one of 0.0, 0.1, or 0.3")
        return value

    @model_validator(mode="after")
    def validate_intervention(self):
        if (self.intervention_mode == "none") != (self.intervention_id is None):
            raise ValueError("intervention_id is required exactly when intervention_mode is not none")
        if self.kind == "learnability" and self.intervention_mode == "none":
            raise ValueError("Learnability cells require include, knockout, or rescue")
        if self.prior_aware and self.kind != "base_model_prior":
            raise ValueError("prior_aware is reserved for base_model_prior cells")
        return self


class AblationExecutionSpec(_StrictModel):
    """Frozen bindings that turn symbolic ablation cells into data jobs."""

    schema_version: Literal["stage-ablation-execution-v1"] = "stage-ablation-execution-v1"
    execution_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    training_seeds: list[int] = Field(min_length=1)
    capability_profiles: list[CapabilityWeightProfile] = Field(min_length=1)
    model_profiles: list[ModelProfile] = Field(min_length=1)
    default_model_profile_id: str = Field(min_length=1)
    cells: list[AblationCellExecution] = Field(min_length=1)

    @field_validator("protocol_sha256", "suite_sha256", "contract_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("training_seeds")
    @classmethod
    def validate_training_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("training_seeds must be unique")
        return value

    @model_validator(mode="after")
    def validate_references(self):
        capability_profiles = {profile.profile_id: profile for profile in self.capability_profiles}
        if len(capability_profiles) != len(self.capability_profiles):
            raise ValueError("Capability profile IDs must be unique")
        model_profiles = {profile.profile_id: profile for profile in self.model_profiles}
        if len(model_profiles) != len(self.model_profiles):
            raise ValueError("Model profile IDs must be unique")
        if self.default_model_profile_id not in model_profiles:
            raise ValueError("default_model_profile_id does not reference a model profile")
        cell_keys = [(cell.study_id, cell.cell_id) for cell in self.cells]
        if len(cell_keys) != len(set(cell_keys)):
            raise ValueError("Execution cells must have unique study_id + cell_id identities")
        for cell in self.cells:
            if cell.capability_profile_id not in capability_profiles:
                raise ValueError(f"Cell {cell.cell_id} references an unknown capability profile")
            model_profile_id = cell.model_profile_id or self.default_model_profile_id
            if model_profile_id not in model_profiles:
                raise ValueError(f"Cell {cell.cell_id} references an unknown model profile")
            if cell.prior_aware:
                required = set(capability_profiles[cell.capability_profile_id].weights)
                missing = required - set(model_profiles[model_profile_id].capability_scores)
                if missing:
                    raise ValueError(f"Prior-aware cell {cell.cell_id} has no calibrated scores for: {sorted(missing)}")
        return self


class AblationBatchInput(_StrictModel):
    """Frozen source-side candidates plus a separately generated SFT bank."""

    schema_version: Literal["stage-ablation-batch-input-v1"] = "stage-ablation-batch-input-v1"
    batch_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    batch_content_sha256: str
    proxy_units: list[ProxyUnit] = Field(min_length=1)
    obligations: list[CoverageObligation] = Field(default_factory=list)
    unit_capabilities: dict[str, list[str]] = Field(min_length=1)
    candidate_sft_bank: list[SFTSample] = Field(min_length=1)
    interventions: list[RuleInterventionSpec] = Field(default_factory=list)

    @field_validator("batch_content_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "batch_content_sha256")

    @model_validator(mode="after")
    def validate_inputs(self):
        unit_ids = [unit.unit_id for unit in self.proxy_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("proxy_units must have unique unit IDs")
        known_units = set(unit_ids)
        if any(unit.batch_id != self.batch_id for unit in self.proxy_units):
            raise ValueError("Every proxy unit must use the batch input batch_id")
        if any(unit.batch_content_sha256 != self.batch_content_sha256 for unit in self.proxy_units):
            raise ValueError("Every proxy unit must use the frozen batch content hash")
        if set(self.unit_capabilities) != known_units:
            raise ValueError("unit_capabilities must map every proxy unit exactly once")
        for unit_id, capability_ids in self.unit_capabilities.items():
            if not capability_ids or any(not value.strip() for value in capability_ids):
                raise ValueError(f"Unit {unit_id} requires at least one non-empty capability ID")
            if len(capability_ids) != len(set(capability_ids)):
                raise ValueError(f"Unit {unit_id} capability IDs must be unique")
        obligation_ids = [row.obligation_id for row in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("Coverage obligation IDs must be unique")
        for obligation in self.obligations:
            if unknown := set(obligation.unit_ids) - known_units:
                raise ValueError(f"Coverage obligation {obligation.obligation_id} references unknown units: {unknown}")
        sample_ids = [sample.sample_id for sample in self.candidate_sft_bank]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Candidate SFT sample IDs must be unique")
        for sample in self.candidate_sft_bank:
            if sample.batch_id != self.batch_id:
                raise ValueError("Every candidate SFT sample must use the batch input batch_id")
            if len(sample.semantic_unit_ids) != 1:
                raise ValueError("Paired mechanism SFT samples must bind exactly one semantic unit")
            if sample.semantic_unit_ids[0] not in known_units:
                raise ValueError(f"Candidate SFT sample {sample.sample_id} references an unknown unit")
            if sample.task_type is None:
                raise ValueError("Every candidate SFT sample requires a registered task_type")
        intervention_ids = [row.intervention_id for row in self.interventions]
        if len(intervention_ids) != len(set(intervention_ids)):
            raise ValueError("Intervention IDs must be unique")
        known_groups = {unit.group_id for unit in self.proxy_units}
        group_costs = _group_costs(self.proxy_units)
        for intervention in self.interventions:
            referenced = set(intervention.target_group_ids) | set(intervention.replacement_group_ids)
            if unknown := referenced - known_groups:
                raise ValueError(f"Intervention {intervention.intervention_id} references unknown groups: {unknown}")
            target_tokens = sum(group_costs[group_id] for group_id in intervention.target_group_ids)
            replacement_tokens = sum(group_costs[group_id] for group_id in intervention.replacement_group_ids)
            if abs(target_tokens - replacement_tokens) > intervention.maximum_token_difference:
                raise ValueError(
                    f"Intervention {intervention.intervention_id} exceeds its matched-token tolerance: "
                    f"target={target_tokens}, replacement={replacement_tokens}"
                )
        return self


class AblationCellManifest(_StrictModel):
    schema_version: Literal["stage-ablation-cell-artifact-v1"] = "stage-ablation-cell-artifact-v1"
    execution_id: str
    execution_sha256: str
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    batch_input_sha256: str
    batch_id: str
    lineage_group: str
    batch_content_sha256: str
    study_id: str
    cell_id: str
    kind: str
    status: Literal["ready", "not_identifiable"]
    factors: dict[str, Any]
    factors_sha256: str
    candidate_pool_sha256: str
    candidate_sft_bank_sha256: str
    capability_profile: dict[str, Any]
    model_profile: dict[str, Any]
    weighted_obligations_sha256: str
    selection: dict[str, Any]
    organization: dict[str, Any]
    intervention: dict[str, Any]
    label_noise: dict[str, Any]
    repetition: dict[str, Any]
    quality_fidelity: dict[str, Any]
    outputs: dict[str, Any]

    @field_validator(
        "execution_sha256",
        "protocol_sha256",
        "suite_sha256",
        "contract_sha256",
        "batch_input_sha256",
        "batch_content_sha256",
        "factors_sha256",
        "candidate_pool_sha256",
        "candidate_sft_bank_sha256",
        "weighted_obligations_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)


class AblationJobRecord(_StrictModel):
    schema_version: Literal["stage-ablation-job-v1"] = "stage-ablation-job-v1"
    job_id: str
    execution_id: str
    execution_sha256: str
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    batch_input_sha256: str
    batch_id: str
    batch_content_sha256: str
    lineage_group: str
    study_id: str
    cell_id: str
    seed: int
    model_profile_id: str
    model_id: str
    model_revision: str
    checkpoint_uri: str
    checkpoint_sha256: str
    calibration_sha256: str
    capability_profile_sha256: str
    selector_config_sha256: str
    selection_seed: int
    train_uri: str
    train_sha256: str
    lineage_uri: str
    lineage_sha256: str
    cell_manifest_uri: str
    cell_manifest_sha256: str
    train_row_count: int = Field(gt=0)

    @field_validator(
        "execution_sha256",
        "protocol_sha256",
        "suite_sha256",
        "contract_sha256",
        "batch_input_sha256",
        "batch_content_sha256",
        "checkpoint_sha256",
        "calibration_sha256",
        "capability_profile_sha256",
        "selector_config_sha256",
        "train_sha256",
        "lineage_sha256",
        "cell_manifest_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)


def execution_digest(spec: AblationExecutionSpec) -> str:
    return _canonical_digest(spec)


def batch_input_digest(batch: AblationBatchInput) -> str:
    payload = batch.model_dump(mode="json")
    payload["proxy_units"] = sorted(payload["proxy_units"], key=lambda row: row["unit_id"])
    payload["obligations"] = sorted(payload["obligations"], key=lambda row: row["obligation_id"])
    payload["candidate_sft_bank"] = sorted(payload["candidate_sft_bank"], key=lambda row: row["sample_id"])
    payload["interventions"] = sorted(payload["interventions"], key=lambda row: row["intervention_id"])
    return _canonical_digest(payload)


def _group_costs(units: list[ProxyUnit]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for unit in units:
        result[unit.group_id] += unit.estimated_tokens
    return dict(result)


def _eligible_groups(units: list[ProxyUnit]) -> dict[str, list[ProxyUnit]]:
    grouped: dict[str, list[ProxyUnit]] = defaultdict(list)
    for unit in units:
        grouped[unit.group_id].append(unit)
    return {
        group_id: sorted(group, key=lambda row: row.unit_id)
        for group_id, group in grouped.items()
        if all(unit.eligible and unit.medically_valid for unit in group)
    }


def weight_capability_obligations(
    obligations: list[CoverageObligation],
    *,
    unit_capabilities: dict[str, list[str]],
    capability_weights: dict[str, float],
    capability_gaps: dict[str, float] | None = None,
) -> list[CoverageObligation]:
    """Apply task weights, and optionally model gaps, without changing units.

    A multi-capability obligation uses the arithmetic mean of its capability
    weights.  This avoids rewarding broad, noisy multi-label mappings merely
    because they carry more labels.
    """

    result = []
    for obligation in obligations:
        capability_ids = list(obligation.capability_ids)
        if not capability_ids and obligation.kind != "source":
            capability_ids = sorted(
                {
                    capability_id
                    for unit_id in obligation.unit_ids
                    for capability_id in unit_capabilities.get(unit_id, [])
                }
            )
        if capability_ids:
            missing_weights = set(capability_ids) - set(capability_weights)
            if missing_weights:
                raise ValueError(
                    f"Obligation {obligation.obligation_id} has no task weights for: {sorted(missing_weights)}"
                )
            if capability_gaps is not None:
                missing_gaps = set(capability_ids) - set(capability_gaps)
                if missing_gaps:
                    raise ValueError(
                        f"Obligation {obligation.obligation_id} has no model gaps for: {sorted(missing_gaps)}"
                    )
            effective = [
                capability_weights[capability_id] * (1.0 if capability_gaps is None else capability_gaps[capability_id])
                for capability_id in capability_ids
            ]
            multiplier = sum(effective) / len(effective)
        else:
            multiplier = 1.0
        result.append(
            obligation.model_copy(
                update={
                    "capability_ids": capability_ids,
                    "weight": obligation.weight * multiplier,
                }
            )
        )
    return result


def build_nested_selections(
    units: list[ProxyUnit],
    obligations: list[CoverageObligation],
    config: ProxySelectionConfig,
    *,
    fractions: tuple[float, ...] = SUPPORTED_FRACTIONS,
) -> dict[str, dict[str, Any]]:
    """Create strict 25/50 percent prefixes and their common 100 percent parent.

    The selector is run once with the complete eligible-token budget.  Smaller
    cells are prefixes of that frozen order, so 25% is necessarily contained in
    50%, which is contained in 100%.
    """

    if not units:
        raise ValueError("At least one proxy unit is required")
    if not fractions:
        raise ValueError("At least one nested fraction is required")
    normalized_fractions = sorted(set(float(value) for value in fractions))
    if any(value <= 0 or value > 1 for value in normalized_fractions):
        raise ValueError("Nested selection fractions must be in (0, 1]")
    eligible_groups = _eligible_groups(units)
    if not eligible_groups:
        raise ValueError("No eligible dependency groups are available")
    group_cost = {group_id: sum(unit.estimated_tokens for unit in group) for group_id, group in eligible_groups.items()}
    candidate_tokens = sum(group_cost.values())
    full_payload = config.model_dump(mode="json")
    full_payload["token_budget"] = candidate_tokens
    full_config = ProxySelectionConfig.model_validate(full_payload)
    full = select_proxy(units, obligations, full_config)
    order = list(full["selected_group_ids"])
    if set(order) != set(eligible_groups) or len(order) != len(eligible_groups):
        raise RuntimeError("The full-budget selector did not traverse every eligible dependency group")
    order_sha256 = _canonical_digest(order)
    pool_sha256 = candidate_pool_digest(units, obligations)
    eligible_unit_ids = {unit.unit_id for group in eligible_groups.values() for unit in group}
    eligible_obligations = [obligation for obligation in obligations if set(obligation.unit_ids) & eligible_unit_ids]
    total_obligation_weight = sum(row.weight for row in eligible_obligations)
    condition_exception = [row for row in eligible_obligations if row.kind in {"condition", "exception"}]
    condition_exception_weight = sum(row.weight for row in condition_exception)
    results: dict[str, dict[str, Any]] = {}
    for fraction in normalized_fractions:
        budget = math.floor(candidate_tokens * fraction)
        selected_groups = []
        used_tokens = 0
        for group_id in order:
            if used_tokens + group_cost[group_id] > budget:
                break
            selected_groups.append(group_id)
            used_tokens += group_cost[group_id]
        if not selected_groups:
            raise ValueError(
                f"Nested budget fraction {fraction} cannot fit the first frozen dependency group "
                f"({group_cost[order[0]]} tokens)"
            )
        if fraction < 1 and len(selected_groups) == len(order):
            raise ValueError(f"Nested fraction {fraction} does not produce a strict subset")
        selected_unit_ids = [unit.unit_id for group_id in selected_groups for unit in eligible_groups[group_id]]
        selected_set = set(selected_unit_ids)
        covered = [row for row in eligible_obligations if set(row.unit_ids) & selected_set]
        covered_condition_exception = [row for row in condition_exception if set(row.unit_ids) & selected_set]
        trace = full["trace"][len(selected_groups) - 1]
        results[_fraction_key(fraction)] = {
            "schema_version": "nested-ablation-selection-v1",
            "fraction": fraction,
            "derived_token_budget": budget,
            "eligible_estimated_tokens": candidate_tokens,
            "used_tokens": used_tokens,
            "eligible_group_count": len(order),
            "selected_group_count": len(selected_groups),
            "selected_group_fraction": len(selected_groups) / len(order),
            "strict_subset_observed": len(selected_groups) < len(order),
            "selected_group_ids": selected_groups,
            "selected_unit_ids": selected_unit_ids,
            "candidate_pool_sha256": pool_sha256,
            "nested_order_sha256": order_sha256,
            "base_selection_config_sha256": selection_config_digest(config),
            "full_selection_config_sha256": full["selection_config_sha256"],
            "method": config.method,
            "objective": trace["objective"],
            "components": trace["components"],
            "weighted_obligation_coverage": (
                sum(row.weight for row in covered) / total_obligation_weight if total_obligation_weight else 1.0
            ),
            "condition_exception_coverage": (
                sum(row.weight for row in covered_condition_exception) / condition_exception_weight
                if condition_exception_weight
                else 1.0
            ),
            "uncovered_obligation_ids": sorted(row.obligation_id for row in eligible_obligations if row not in covered),
        }
    for smaller, larger in zip(normalized_fractions, normalized_fractions[1:]):
        smaller_groups = set(results[_fraction_key(smaller)]["selected_group_ids"])
        larger_groups = set(results[_fraction_key(larger)]["selected_group_ids"])
        if not smaller_groups <= larger_groups:
            raise RuntimeError("Nested selection construction produced non-nested group sets")
    return results


def quality_fidelity_identifiability(units: list[ProxyUnit]) -> dict[str, Any]:
    eligible = [unit for group in _eligible_groups(units).values() for unit in group]
    token_counts = Counter()
    unit_counts = Counter()
    for unit in eligible:
        token_counts[unit.quality_bucket] += unit.estimated_tokens
        unit_counts[unit.quality_bucket] += 1
    buckets = sorted(token_counts)
    return {
        "status": "identifiable" if len(buckets) >= 2 else "not_identifiable",
        "quality_buckets": buckets,
        "quality_bucket_count": len(buckets),
        "unit_counts": dict(sorted(unit_counts.items())),
        "token_counts": dict(sorted(token_counts.items())),
        "reason": (
            None
            if len(buckets) >= 2
            else "A quality-fidelity contrast requires at least two eligible quality buckets within the batch."
        ),
    }


def apply_rule_intervention(
    selected_group_ids: list[str],
    *,
    selection_order: list[str],
    intervention: RuleInterventionSpec,
    mode: Literal["include", "knockout", "rescue"],
) -> tuple[list[str], dict[str, Any]]:
    """Swap matched dependency groups while keeping the selected size fixed."""

    selected = set(selected_group_ids)
    for target, replacement in zip(
        intervention.target_group_ids,
        intervention.replacement_group_ids,
        strict=True,
    ):
        present = int(target in selected) + int(replacement in selected)
        if present != 1:
            raise ValueError(
                f"Intervention pair ({target}, {replacement}) requires exactly one member in the base selection"
            )
    remove = set(intervention.target_group_ids) | set(intervention.replacement_group_ids)
    add = set(intervention.replacement_group_ids) if mode == "knockout" else set(intervention.target_group_ids)
    transformed = (selected - remove) | add
    if len(transformed) != len(selected):
        raise RuntimeError("Matched rule intervention changed the selected dependency-group count")
    order_index = {group_id: index for index, group_id in enumerate(selection_order)}
    if unknown := transformed - set(order_index):
        raise ValueError(f"Intervention introduces groups absent from the frozen selection order: {sorted(unknown)}")
    result = sorted(transformed, key=lambda group_id: order_index[group_id])
    return result, {
        "intervention_id": intervention.intervention_id,
        "mode": mode,
        "target_group_ids": intervention.target_group_ids,
        "replacement_group_ids": intervention.replacement_group_ids,
        "removed_group_ids": sorted(selected - transformed),
        "added_group_ids": sorted(transformed - selected),
        "base_group_count": len(selected),
        "result_group_count": len(result),
        "matched_token_tolerance": intervention.maximum_token_difference,
    }


def organize_paired_samples(
    candidate_bank: list[SFTSample],
    selected_unit_ids: list[str],
    *,
    organization: Literal["direct", "mixed", "random"],
    seed: int,
    batch_id: str,
) -> tuple[list[SFTSample], dict[str, Any]]:
    """Select one supervision form per source unit from a paired SFT bank."""

    by_unit: dict[str, dict[str, list[SFTSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in candidate_bank:
        unit_id = sample.semantic_unit_ids[0]
        by_unit[unit_id][str(sample.task_type)].append(sample)
    selected = []
    assignments = []
    for index, unit_id in enumerate(selected_unit_ids):
        available = by_unit.get(unit_id, {})
        if organization == "direct":
            task_type = "grounded_qa"
            required_types = (task_type,)
        else:
            required_types = TASK_TYPES
            missing = set(required_types) - set(available)
            if missing:
                raise ValueError(f"Unit {unit_id} lacks paired organization forms: {sorted(missing)}")
            representatives = [sorted(available[kind], key=lambda row: row.sample_id)[0] for kind in required_types]
            signatures = {
                (sample.response.strip(), tuple(span.strip() for span in sample.evidence_spans))
                for sample in representatives
            }
            if len(signatures) != 1:
                raise ValueError(f"Unit {unit_id} organization forms do not share the same answer fact and evidence")
            if organization == "mixed":
                task_type = TASK_TYPES[index % len(TASK_TYPES)]
            else:
                rng = random.Random(_derived_seed("organization", seed, batch_id, unit_id))
                task_type = TASK_TYPES[rng.randrange(len(TASK_TYPES))]
        missing = set(required_types) - set(available)
        if missing:
            raise ValueError(f"Unit {unit_id} lacks required organization forms: {sorted(missing)}")
        sample = sorted(available[task_type], key=lambda row: row.sample_id)[0]
        selected.append(sample)
        assignments.append(
            {
                "semantic_unit_id": unit_id,
                "task_type": task_type,
                "sample_id": sample.sample_id,
            }
        )
    return selected, {
        "strategy": organization,
        "seed": seed,
        "content_unit_ids": selected_unit_ids,
        "content_unit_count": len(selected_unit_ids),
        "selected_sample_ids": [sample.sample_id for sample in selected],
        "task_type_counts": dict(sorted(Counter(sample.task_type for sample in selected).items())),
        "assignments": assignments,
    }


def nested_label_noise_plan(
    samples: list[SFTSample],
    *,
    rates: tuple[float, ...] = (0.0, 0.1, 0.3),
    seed: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Label-noise planning requires at least one sample")
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("Label-noise rates must be in [0, 1]")
    ordered = sorted(
        samples,
        key=lambda sample: (
            hashlib.sha256(f"{seed}\0{sample.sample_id}".encode("utf-8")).hexdigest(),
            sample.sample_id,
        ),
    )
    plans = {}
    for rate in sorted(set(rates)):
        count = math.floor(len(samples) * rate)
        plans[_fraction_key(rate)] = [sample.sample_id for sample in ordered[:count]]
    return {
        "seed": seed,
        "sample_order": [sample.sample_id for sample in ordered],
        "sample_order_sha256": _canonical_digest([sample.sample_id for sample in ordered]),
        "rates": plans,
    }


def apply_nested_label_noise(
    samples: list[SFTSample],
    *,
    rate: float,
    seed: int,
) -> tuple[list[SFTSample], dict[str, Any]]:
    plan = nested_label_noise_plan(samples, rates=(0.0, 0.1, 0.3, rate), seed=seed)
    corrupt_ids = set(plan["rates"][_fraction_key(rate)])
    order = plan["sample_order"]
    by_id = {sample.sample_id: sample for sample in samples}
    transformed = []
    records = []
    for sample_id in [sample.sample_id for sample in samples]:
        sample = by_id[sample_id]
        if sample_id not in corrupt_ids:
            transformed.append(sample)
            continue
        position = order.index(sample_id)
        donor = None
        for offset in range(1, len(order)):
            candidate = by_id[order[(position + offset) % len(order)]]
            if candidate.response != sample.response:
                donor = candidate
                break
        if donor is None:
            raise ValueError("Label corruption requires at least two distinct candidate responses")
        corrupted_id = stable_id(
            "stage-ablation-label-noise-v1",
            sample.sample_id,
            donor.sample_id,
            str(seed),
            _fraction_key(rate),
        )
        corrupted = sample.model_copy(update={"sample_id": corrupted_id, "response": donor.response})
        transformed.append(corrupted)
        records.append(
            {
                "sample_id": corrupted_id,
                "parent_sample_id": sample.sample_id,
                "donor_sample_id": donor.sample_id,
                "original_response_sha256": hashlib.sha256(sample.response.encode("utf-8")).hexdigest(),
                "corrupted_response_sha256": hashlib.sha256(donor.response.encode("utf-8")).hexdigest(),
            }
        )
    return transformed, {
        "kind": "none" if rate == 0 else "label_corruption",
        "rate": rate,
        "seed": seed,
        "sample_order_sha256": plan["sample_order_sha256"],
        "corrupted_parent_sample_ids": sorted(corrupt_ids),
        "corrupted_count": len(records),
        "records": records,
    }


def _verify_bound_artifact(uri: str, expected_sha256: str, description: str) -> None:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        raw = unquote(parsed.path)
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw)
    elif "://" not in uri:
        path = Path(uri)
    else:
        return
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bound {description} artifact does not exist: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"Bound {description} artifact hash mismatch: expected={expected_sha256}, observed={observed}")


def _selected_unit_ids(group_ids: list[str], units: list[ProxyUnit]) -> list[str]:
    by_group: dict[str, list[ProxyUnit]] = defaultdict(list)
    for unit in units:
        by_group[unit.group_id].append(unit)
    return [unit.unit_id for group_id in group_ids for unit in sorted(by_group[group_id], key=lambda row: row.unit_id)]


def _materialized_rows(
    samples: list[SFTSample],
    *,
    spec_sha256: str,
    batch: AblationBatchInput,
    cell: AblationCellExecution,
    repetition_factor: int,
    noise_manifest: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    corruption_by_sample = {row["sample_id"]: row for row in noise_manifest["records"]}
    train_rows = []
    lineage_rows = []
    for exposure_index in range(1, repetition_factor + 1):
        for sample in samples:
            corruption = corruption_by_sample.get(sample.sample_id)
            parent_sample_id = corruption["parent_sample_id"] if corruption else sample.sample_id
            row_id = stable_id(
                "stage-ablation-train-row-v1",
                spec_sha256,
                batch.batch_id,
                cell.study_id,
                cell.cell_id,
                sample.sample_id,
                str(exposure_index),
            )
            train_rows.append(
                {
                    "instruction": sample.instruction,
                    "input": "",
                    "output": sample.response,
                }
            )
            lineage_rows.append(
                {
                    "schema_version": "stage-ablation-row-lineage-v1",
                    "row_id": row_id,
                    "source_sample_id": parent_sample_id,
                    "materialized_sample_id": sample.sample_id,
                    "semantic_unit_ids": sample.semantic_unit_ids,
                    "source_ids": sample.source_ids,
                    "task_type": sample.task_type,
                    "exposure_index": exposure_index,
                    "repetition_factor": repetition_factor,
                    "controlled_duplicate": exposure_index > 1,
                    "label_noise": corruption,
                    "study_id": cell.study_id,
                    "cell_id": cell.cell_id,
                    "batch_id": batch.batch_id,
                }
            )
    return train_rows, lineage_rows


def materialize_ablation_batch(
    spec: AblationExecutionSpec,
    batch: AblationBatchInput,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize one batch across every registered cell and training seed.

    Each ready cell emits ``train.jsonl``, ``lineage.jsonl`` and
    ``cell_manifest.json``.  The root ``jobs.jsonl`` is the complete
    batch × cell × seed matrix, excluding cells that are explicitly marked
    ``not_identifiable`` by the quality-fidelity gate.
    """

    output_dir = Path(output_dir).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite ablation output: {output_dir}")
    if partial_dir.exists():
        raise FileExistsError(f"Incomplete prior ablation output must be inspected: {partial_dir}")
    spec_sha256 = execution_digest(spec)
    batch_sha256 = batch_input_digest(batch)
    capability_profiles = {profile.profile_id: profile for profile in spec.capability_profiles}
    model_profiles = {profile.profile_id: profile for profile in spec.model_profiles}
    interventions = {row.intervention_id: row for row in batch.interventions}
    candidate_bank_sha256 = _canonical_digest(
        [row.model_dump(mode="json") for row in sorted(batch.candidate_sft_bank, key=lambda sample: sample.sample_id)]
    )
    quality_audit = quality_fidelity_identifiability(batch.proxy_units)
    partial_dir.mkdir(parents=True, exist_ok=False)
    cell_summaries = []
    jobs: list[AblationJobRecord] = []
    try:
        for cell in sorted(spec.cells, key=lambda row: (row.study_id, row.cell_id)):
            capability_profile = capability_profiles[cell.capability_profile_id]
            model_profile_id = cell.model_profile_id or spec.default_model_profile_id
            model_profile = model_profiles[model_profile_id]
            _verify_bound_artifact(
                model_profile.checkpoint_uri,
                model_profile.checkpoint_sha256,
                f"model checkpoint for {model_profile.profile_id}",
            )
            _verify_bound_artifact(
                model_profile.calibration_uri,
                model_profile.calibration_sha256,
                f"model calibration for {model_profile.profile_id}",
            )
            weighted_obligations = weight_capability_obligations(
                batch.obligations,
                unit_capabilities=batch.unit_capabilities,
                capability_weights=capability_profile.weights,
                capability_gaps=model_profile.capability_gaps if cell.prior_aware else None,
            )
            weighted_obligations_sha256 = _canonical_digest(
                [
                    row.model_dump(mode="json")
                    for row in sorted(weighted_obligations, key=lambda obligation: obligation.obligation_id)
                ]
            )
            nested = build_nested_selections(
                batch.proxy_units,
                weighted_obligations,
                cell.selector_config,
            )
            selection = nested[_fraction_key(cell.source_budget_fraction)]
            base_groups = list(selection["selected_group_ids"])
            effective_groups = base_groups
            intervention_manifest: dict[str, Any] = {
                "intervention_id": None,
                "mode": "none",
                "base_group_count": len(base_groups),
                "result_group_count": len(base_groups),
                "removed_group_ids": [],
                "added_group_ids": [],
            }
            if cell.intervention_id is not None:
                intervention = interventions.get(cell.intervention_id)
                if intervention is None:
                    raise ValueError(
                        f"Cell {cell.cell_id} references unknown batch intervention {cell.intervention_id}"
                    )
                full_order = nested[_fraction_key(1.0)]["selected_group_ids"]
                effective_groups, intervention_manifest = apply_rule_intervention(
                    base_groups,
                    selection_order=full_order,
                    intervention=intervention,
                    mode=cell.intervention_mode,  # type: ignore[arg-type]
                )
            effective_unit_ids = _selected_unit_ids(effective_groups, batch.proxy_units)
            organized_samples, organization_manifest = organize_paired_samples(
                batch.candidate_sft_bank,
                effective_unit_ids,
                organization=cell.organization,
                seed=cell.organization_seed,
                batch_id=batch.batch_id,
            )
            noisy_samples, noise_manifest = apply_nested_label_noise(
                organized_samples,
                rate=cell.label_noise_rate,
                seed=cell.label_noise_seed,
            )
            train_rows, lineage_rows = _materialized_rows(
                noisy_samples,
                spec_sha256=spec_sha256,
                batch=batch,
                cell=cell,
                repetition_factor=cell.repetition_factor,
                noise_manifest=noise_manifest,
            )
            relative_dir = Path("cells") / cell.study_id / cell.cell_id / batch.batch_id
            cell_dir = partial_dir / relative_dir
            train_path = cell_dir / "train.jsonl"
            lineage_path = cell_dir / "lineage.jsonl"
            manifest_path = cell_dir / "cell_manifest.json"
            write_jsonl(train_path, train_rows)
            write_jsonl(lineage_path, lineage_rows)
            status = (
                "not_identifiable"
                if cell.kind == "quality_fidelity" and quality_audit["status"] == "not_identifiable"
                else "ready"
            )
            selection_manifest = {
                **selection,
                "base_selected_group_ids": base_groups,
                "effective_selected_group_ids": effective_groups,
                "effective_selected_unit_ids": effective_unit_ids,
                "effective_group_count": len(effective_groups),
                "effective_unit_count": len(effective_unit_ids),
            }
            cell_payload = AblationCellManifest(
                execution_id=spec.execution_id,
                execution_sha256=spec_sha256,
                protocol_sha256=spec.protocol_sha256,
                suite_sha256=spec.suite_sha256,
                contract_sha256=spec.contract_sha256,
                batch_input_sha256=batch_sha256,
                batch_id=batch.batch_id,
                lineage_group=batch.lineage_group,
                batch_content_sha256=batch.batch_content_sha256,
                study_id=cell.study_id,
                cell_id=cell.cell_id,
                kind=cell.kind,
                status=status,
                factors=cell.model_dump(mode="json"),
                factors_sha256=_canonical_digest(cell),
                candidate_pool_sha256=selection["candidate_pool_sha256"],
                candidate_sft_bank_sha256=candidate_bank_sha256,
                capability_profile={
                    **capability_profile.model_dump(mode="json"),
                    "profile_sha256": _canonical_digest(capability_profile),
                    "effective_weights": {
                        capability_id: capability_profile.weights[capability_id]
                        * (model_profile.capability_gaps[capability_id] if cell.prior_aware else 1.0)
                        for capability_id in capability_profile.weights
                    },
                    "prior_aware": cell.prior_aware,
                },
                model_profile={
                    **model_profile.model_dump(mode="json"),
                    "profile_sha256": _canonical_digest(model_profile),
                },
                weighted_obligations_sha256=weighted_obligations_sha256,
                selection=selection_manifest,
                organization=organization_manifest,
                intervention=intervention_manifest,
                label_noise=noise_manifest,
                repetition={
                    "factor": cell.repetition_factor,
                    "unique_sample_count": len(noisy_samples),
                    "materialized_row_count": len(train_rows),
                },
                quality_fidelity=quality_audit,
                outputs={
                    "train_uri": str(relative_dir / "train.jsonl"),
                    "train_sha256": file_sha256(train_path),
                    "train_row_count": len(train_rows),
                    "lineage_uri": str(relative_dir / "lineage.jsonl"),
                    "lineage_sha256": file_sha256(lineage_path),
                    "lineage_row_count": len(lineage_rows),
                },
            )
            write_json(manifest_path, cell_payload.model_dump(mode="json"))
            manifest_sha256 = file_sha256(manifest_path)
            cell_summaries.append(
                {
                    "study_id": cell.study_id,
                    "cell_id": cell.cell_id,
                    "status": status,
                    "cell_manifest_uri": str(relative_dir / "cell_manifest.json"),
                    "cell_manifest_sha256": manifest_sha256,
                }
            )
            if status == "ready":
                for training_seed in spec.training_seeds:
                    jobs.append(
                        AblationJobRecord(
                            job_id=stable_id(
                                "stage-ablation-job-v1",
                                spec_sha256,
                                batch_sha256,
                                cell.study_id,
                                cell.cell_id,
                                str(training_seed),
                            ),
                            execution_id=spec.execution_id,
                            execution_sha256=spec_sha256,
                            protocol_sha256=spec.protocol_sha256,
                            suite_sha256=spec.suite_sha256,
                            contract_sha256=spec.contract_sha256,
                            batch_input_sha256=batch_sha256,
                            batch_id=batch.batch_id,
                            batch_content_sha256=batch.batch_content_sha256,
                            lineage_group=batch.lineage_group,
                            study_id=cell.study_id,
                            cell_id=cell.cell_id,
                            seed=training_seed,
                            model_profile_id=model_profile.profile_id,
                            model_id=model_profile.model_id,
                            model_revision=model_profile.revision,
                            checkpoint_uri=model_profile.checkpoint_uri,
                            checkpoint_sha256=model_profile.checkpoint_sha256,
                            calibration_sha256=model_profile.calibration_sha256,
                            capability_profile_sha256=_canonical_digest(capability_profile),
                            selector_config_sha256=selection_config_digest(cell.selector_config),
                            selection_seed=cell.selector_config.random_state,
                            train_uri=str(relative_dir / "train.jsonl"),
                            train_sha256=file_sha256(train_path),
                            lineage_uri=str(relative_dir / "lineage.jsonl"),
                            lineage_sha256=file_sha256(lineage_path),
                            cell_manifest_uri=str(relative_dir / "cell_manifest.json"),
                            cell_manifest_sha256=manifest_sha256,
                            train_row_count=len(train_rows),
                        )
                    )
        jobs_path = partial_dir / "jobs.jsonl"
        write_jsonl(jobs_path, (job.model_dump(mode="json") for job in jobs))
        root_manifest = {
            "schema_version": "stage-ablation-materialization-v1",
            "execution_id": spec.execution_id,
            "execution_sha256": spec_sha256,
            "protocol_sha256": spec.protocol_sha256,
            "suite_sha256": spec.suite_sha256,
            "contract_sha256": spec.contract_sha256,
            "batch_id": batch.batch_id,
            "batch_content_sha256": batch.batch_content_sha256,
            "batch_input_sha256": batch_sha256,
            "cell_count": len(cell_summaries),
            "ready_cell_count": sum(row["status"] == "ready" for row in cell_summaries),
            "not_identifiable_cell_count": sum(row["status"] == "not_identifiable" for row in cell_summaries),
            "job_count": len(jobs),
            "jobs_uri": "jobs.jsonl",
            "jobs_sha256": file_sha256(jobs_path),
            "cells": cell_summaries,
        }
        write_json(partial_dir / "materialization_manifest.json", root_manifest)
        partial_dir.replace(output_dir)
    except Exception:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise
    return {
        "output_dir": str(output_dir),
        "materialization_manifest": str(output_dir / "materialization_manifest.json"),
        "jobs": str(output_dir / "jobs.jsonl"),
        "cell_count": len(cell_summaries),
        "job_count": len(jobs),
    }


__all__ = [
    "AblationBatchInput",
    "AblationCellExecution",
    "AblationCellManifest",
    "AblationExecutionSpec",
    "AblationJobRecord",
    "CapabilityWeightProfile",
    "ModelProfile",
    "RuleInterventionSpec",
    "apply_nested_label_noise",
    "apply_rule_intervention",
    "batch_input_digest",
    "build_nested_selections",
    "execution_digest",
    "materialize_ablation_batch",
    "nested_label_noise_plan",
    "organize_paired_samples",
    "quality_fidelity_identifiability",
    "weight_capability_obligations",
]
