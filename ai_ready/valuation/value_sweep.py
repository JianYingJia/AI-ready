"""Budget and seed stability sweep for conditional-value probability samples."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ai_ready.valuation.conditional_value import (
    FidelityThresholds,
    ModelMasteryProfile,
    build_fidelity_certificate,
    build_raw_sample_profile,
    build_value_profile,
    conditional_value_unit_from_sampling_record,
    reconstruct_value_profile,
)
from ai_ready.valuation.value_sampling import (
    RareStratumQuota,
    ValueSamplingConfig,
    ValueSamplingUnit,
    sample_value_distribution,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ValueFidelitySweepConfig(_StrictModel):
    """Registered exploratory grid for selecting a viable sampling budget."""

    sampling_fractions: list[float] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    design_stratum_attributes: list[str] = Field(
        min_length=1,
        validation_alias=AliasChoices("design_stratum_attributes", "stratum_attributes"),
    )
    audit_only_attributes: list[str] = Field(default_factory=list)
    rare_stratum_quotas: list[RareStratumQuota] = Field(default_factory=list)
    allow_target_expansion: bool = True

    @field_validator("sampling_fractions")
    @classmethod
    def validate_fractions(cls, values: list[float]) -> list[float]:
        normalized = [float(value) for value in values]
        if any(value <= 0 or value > 1 for value in normalized):
            raise ValueError("sampling_fractions must lie in (0, 1]")
        if len(normalized) != len(set(normalized)):
            raise ValueError("sampling_fractions must be unique")
        return sorted(normalized)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("seeds must be unique")
        return sorted(values)

    @field_validator("design_stratum_attributes", "audit_only_attributes")
    @classmethod
    def validate_attributes(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Attribute role lists cannot contain empty names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Attribute role lists must contain unique names")
        return sorted(normalized)

    @model_validator(mode="after")
    def validate_quota_ids(self):
        quota_ids = [quota.quota_id for quota in self.rare_stratum_quotas]
        if len(quota_ids) != len(set(quota_ids)):
            raise ValueError("rare_stratum_quotas must have unique quota_id values")
        overlap = set(self.design_stratum_attributes) & set(self.audit_only_attributes)
        if overlap:
            raise ValueError(
                "audit_only_attributes must be held out from the sampling strata; " f"overlap={sorted(overlap)}"
            )
        design_attributes = set(self.design_stratum_attributes)
        for quota in self.rare_stratum_quotas:
            if unstratified := set(quota.data_attributes) - design_attributes:
                raise ValueError(
                    f"Rare quota {quota.quota_id} uses attributes outside design_stratum_attributes: "
                    f"{sorted(unstratified)}"
                )
        return self


def _canonical_sha256(value: Any) -> str:
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


def value_fidelity_sweep_config_digest(config: ValueFidelitySweepConfig) -> str:
    return _canonical_sha256(config)


def value_population_units_digest(units: Sequence[ValueSamplingUnit]) -> str:
    """Hash the complete sweep input independently of JSONL row order."""

    rows = [unit.model_dump(mode="json") for unit in units]
    rows.sort(
        key=lambda row: json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return _canonical_sha256(rows)


def _require_profile_alignment(units: Sequence[ValueSamplingUnit], profile: ModelMasteryProfile) -> str:
    observations = profile.by_cell
    granularities = {unit.model_state_granularity for unit in units}
    if len(granularities) != 1:
        raise ValueError("A fidelity sweep cannot mix model-state measurement granularities")
    granularity = next(iter(granularities))
    for unit in units:
        if missing := set(unit.capability_ids) - set(observations):
            raise ValueError(f"Unit {unit.unit_id} references cells absent from model profile: {sorted(missing)}")
        expected = {capability_id: observations[capability_id].stratum_state for capability_id in unit.capability_ids}
        if granularity == "capability_prior" and unit.model_states != expected:
            raise ValueError(
                f"Unit {unit.unit_id} has stale model_states; expected states from profile {profile.profile_id}"
            )
    return granularity


def _eligible_population_units(units: Sequence[ValueSamplingUnit]) -> list[ValueSamplingUnit]:
    """Apply the sampler's atomic dependency-group eligibility rule."""

    groups: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in units:
        groups[unit.dependency_group_id].append(unit)
    return [
        unit
        for group_id in sorted(groups)
        if all(unit.eligible for unit in groups[group_id])
        for unit in groups[group_id]
    ]


def _require_attribute_roles(units: Sequence[ValueSamplingUnit], config: ValueFidelitySweepConfig) -> list[str]:
    """Ensure every registered design and holdout attribute is auditable."""

    registered = set(config.design_stratum_attributes) | set(config.audit_only_attributes)
    for unit in units:
        if missing := registered - set(unit.data_attributes):
            raise ValueError(f"Unit {unit.unit_id} is missing registered sweep attributes: {sorted(missing)}")
    return sorted({name for unit in units for name in unit.data_attributes})


def _attribute_tv_by_role(
    attribute_tv: Mapping[str, float], config: ValueFidelitySweepConfig
) -> dict[str, dict[str, float]]:
    design = {name: attribute_tv[name] for name in config.design_stratum_attributes if name in attribute_tv}
    audit_only = {name: attribute_tv[name] for name in config.audit_only_attributes if name in attribute_tv}
    registered = set(design) | set(audit_only)
    additional_audit = {name: value for name, value in sorted(attribute_tv.items()) if name not in registered}
    return {
        "design_stratum_attributes": design,
        "audit_only_attributes": audit_only,
        "additional_profile_audit_attributes": additional_audit,
    }


def _maximum(values: Mapping[str, float]) -> float:
    return max(values.values(), default=0.0)


def run_value_fidelity_sweep(
    units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    model_profile: ModelMasteryProfile,
    thresholds: FidelityThresholds,
    config: ValueFidelitySweepConfig,
    *,
    critical_cell_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate sample fidelity over a registered budget and seed grid.

    The sweep is a generation-stage design experiment.  It never trains a
    model, and a profile with unmeasured cells keeps the result explicitly
    partial even when the observed data/task distribution checks pass.
    """

    parsed = [unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit) for unit in units]
    if not parsed:
        raise ValueError("At least one value-sampling unit is required")
    model_state_granularity = _require_profile_alignment(parsed, model_profile)
    profile_attribute_dimensions = _require_attribute_roles(parsed, config)
    by_batch: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in parsed:
        by_batch[unit.batch_id].append(unit)

    weights = dict(sorted((critical_cell_weights or model_profile.benchmark_cell_weights).items()))
    if unknown_weight_cells := set(weights) - set(model_profile.by_cell):
        raise ValueError(
            "Critical cell weights reference cells absent from the model profile: " f"{sorted(unknown_weight_cells)}"
        )
    cells: list[dict[str, Any]] = []
    batch_summary: list[dict[str, Any]] = []
    model_measurement_complete = True
    for batch_id, batch_units in sorted(by_batch.items()):
        eligible_population = _eligible_population_units(batch_units)
        if not eligible_population:
            raise ValueError(f"Batch {batch_id} has no eligible dependency groups")
        full_units = [conditional_value_unit_from_sampling_record(unit) for unit in eligible_population]
        full = build_value_profile(
            full_units,
            model_profile,
            profile_id=f"{batch_id}-sweep-full",
        )
        represented_cells = {stratum.key.cell_id for stratum in full.strata}
        batch_weights = {cell_id: weight for cell_id, weight in weights.items() if cell_id in represented_cells}
        if missing := (set(config.design_stratum_attributes) | set(config.audit_only_attributes)) - set(
            full.attribute_dimensions
        ):
            raise AssertionError(f"The full profile for {batch_id} omitted registered attributes: {sorted(missing)}")
        model_measurement_complete &= not full.unmeasured_cell_ids and full.unknown_mastery_mass <= 1e-9
        batch_cells = []
        for fraction in config.sampling_fractions:
            trials = []
            for seed in config.seeds:
                sampling_config = ValueSamplingConfig(
                    sampling_fraction=fraction,
                    seed=seed,
                    stratum_attributes=config.design_stratum_attributes,
                    rare_stratum_quotas=config.rare_stratum_quotas,
                    allow_target_expansion=config.allow_target_expansion,
                )
                sample = sample_value_distribution(batch_units, sampling_config)
                selected = [
                    conditional_value_unit_from_sampling_record(record) for record in sample["selected_unit_records"]
                ]
                raw = build_raw_sample_profile(
                    selected,
                    model_profile,
                    profile_id=f"{batch_id}-{fraction:g}-{seed}-raw",
                )
                reconstruction = reconstruct_value_profile(
                    selected,
                    model_profile,
                    profile_id=f"{batch_id}-{fraction:g}-{seed}-ht",
                )
                certificate = build_fidelity_certificate(
                    full,
                    raw,
                    reconstruction,
                    certificate_id=f"{batch_id}-{fraction:g}-{seed}-certificate",
                    thresholds=thresholds,
                    critical_cell_weights=batch_weights,
                )
                comparison = certificate.weighted_comparison
                attribute_tv_by_role = _attribute_tv_by_role(
                    comparison.attribute_marginal_total_variation,
                    config,
                )
                design_attribute_tv = attribute_tv_by_role["design_stratum_attributes"]
                audit_only_attribute_tv = attribute_tv_by_role["audit_only_attributes"]
                additional_attribute_tv = attribute_tv_by_role["additional_profile_audit_attributes"]
                trials.append(
                    {
                        "seed": seed,
                        "requested_fraction": fraction,
                        "requested_group_count": sample["requested_group_count"],
                        "realized_group_count": sample["realized_group_count"],
                        "realized_fraction": sample["realized_sampling_fraction"],
                        "joint_total_variation": comparison.joint_total_variation,
                        "maximum_attribute_total_variation": max(
                            comparison.attribute_marginal_total_variation.values(),
                            default=0.0,
                        ),
                        "attribute_total_variation_by_role": attribute_tv_by_role,
                        "maximum_design_attribute_total_variation": _maximum(design_attribute_tv),
                        "maximum_audit_only_attribute_total_variation": _maximum(audit_only_attribute_tv),
                        "maximum_additional_profile_attribute_total_variation": _maximum(additional_attribute_tv),
                        "audit_only_attributes_passed": all(
                            value <= thresholds.maximum_attribute_marginal_total_variation
                            for value in audit_only_attribute_tv.values()
                        ),
                        "effective_sample_size": reconstruction.effective_sample_size,
                        "required_effective_sample_size": certificate.required_effective_sample_size,
                        "distribution_fidelity_passed": certificate.distribution_fidelity_passed,
                        "complete_three_dimensional_claim": certificate.complete_three_dimensional_claim,
                        "failed_distribution_checks": [
                            name
                            for name, passed in certificate.checks.items()
                            if not passed and name != "model_measurement_complete"
                        ],
                    }
                )
            cell = {
                "batch_id": batch_id,
                "requested_fraction": fraction,
                "pass_rate": sum(row["distribution_fidelity_passed"] for row in trials) / len(trials),
                "all_seeds_pass": all(row["distribution_fidelity_passed"] for row in trials),
                "realized_fraction_min": min(row["realized_fraction"] for row in trials),
                "realized_fraction_max": max(row["realized_fraction"] for row in trials),
                "joint_total_variation_median": statistics.median(row["joint_total_variation"] for row in trials),
                "joint_total_variation_max": max(row["joint_total_variation"] for row in trials),
                "maximum_attribute_total_variation": max(row["maximum_attribute_total_variation"] for row in trials),
                "maximum_design_attribute_total_variation": max(
                    row["maximum_design_attribute_total_variation"] for row in trials
                ),
                "maximum_audit_only_attribute_total_variation": max(
                    row["maximum_audit_only_attribute_total_variation"] for row in trials
                ),
                "audit_only_attributes_all_seeds_pass": all(row["audit_only_attributes_passed"] for row in trials),
                "effective_sample_size_min": min(row["effective_sample_size"] for row in trials),
                "required_effective_sample_size": trials[0]["required_effective_sample_size"],
                "trials": trials,
            }
            batch_cells.append(cell)
            cells.append(cell)

        qualifying = [cell for cell in batch_cells if cell["all_seeds_pass"]]
        minimum = min(qualifying, key=lambda cell: cell["requested_fraction"]) if qualifying else None
        batch_summary.append(
            {
                "batch_id": batch_id,
                "population_group_count": len({unit.dependency_group_id for unit in eligible_population}),
                "fidelity_status": "passed" if minimum else "no_qualifying_budget",
                "minimum_grid_fraction_all_seeds_pass": minimum["requested_fraction"] if minimum else None,
                "realized_fraction_range_at_minimum": (
                    [minimum["realized_fraction_min"], minimum["realized_fraction_max"]] if minimum else None
                ),
                "joint_total_variation_max_at_minimum": (minimum["joint_total_variation_max"] if minimum else None),
                "effective_sample_size_min_at_minimum": (minimum["effective_sample_size_min"] if minimum else None),
                "required_effective_sample_size": (minimum["required_effective_sample_size"] if minimum else None),
                "audit_only_total_variation_max_at_minimum": (
                    minimum["maximum_audit_only_attribute_total_variation"] if minimum else None
                ),
            }
        )

    fidelity_status = (
        "passed" if all(batch["fidelity_status"] == "passed" for batch in batch_summary) else "no_qualifying_budget"
    )
    model_measurement_status = "complete" if model_measurement_complete else "partial_pending_model_measurement"
    if not model_measurement_complete:
        claim_status = "partial_pending_model_measurement"
    elif fidelity_status == "passed":
        claim_status = "complete"
    else:
        claim_status = "no_qualifying_budget"

    return {
        "schema_version": "conditional-value-fidelity-sweep-v2",
        "purpose": "exploratory_generation_stage_budget_selection_not_training_utility",
        "claim_status": claim_status,
        "model_measurement_status": model_measurement_status,
        "fidelity_status": fidelity_status,
        "model_dimension_evaluated": model_measurement_complete,
        "input_unit_count": len(parsed),
        "population_units_sha256": value_population_units_digest(parsed),
        "model_profile_id": model_profile.profile_id,
        "model_profile_sha256": _canonical_sha256(model_profile),
        "model_measurement_granularity": model_state_granularity,
        "attribute_roles": {
            "design_stratum_attributes": config.design_stratum_attributes,
            "audit_only_attributes": config.audit_only_attributes,
            "additional_profile_audit_attributes": sorted(
                set(profile_attribute_dimensions)
                - set(config.design_stratum_attributes)
                - set(config.audit_only_attributes)
            ),
            "certificate_audited_attributes": profile_attribute_dimensions,
        },
        "sweep_config": config.model_dump(mode="json"),
        "sweep_config_sha256": value_fidelity_sweep_config_digest(config),
        "thresholds": thresholds.model_dump(mode="json"),
        "critical_cell_weights": weights,
        "batch_summary": batch_summary,
        "cells": cells,
    }


__all__ = [
    "ValueFidelitySweepConfig",
    "run_value_fidelity_sweep",
    "value_fidelity_sweep_config_digest",
    "value_population_units_digest",
]
