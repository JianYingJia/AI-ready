"""Probability sampling that preserves a batch's conditional-value distribution.

This module is deliberately separate from :mod:`ai_ready.valuation.selection`.
The coverage selector optimizes which units to train on.  The sampler below is
an estimation design: every eligible dependency group has a known, positive
inclusion probability, so the selected rows can be weighted back to the source
batch instead of being interpreted as an optimized training set.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelState = Literal["mastered", "uncertain", "unmastered", "unknown"]
ModelStateGranularity = Literal["capability_prior", "unit_cell"]
MODEL_STATES = {"mastered", "uncertain", "unmastered", "unknown"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ValueSamplingUnit(StrictModel):
    """One source-derived semantic unit before SFT question generation.

    ``dependency_group_id`` is the primary sampling unit.  A rule and its
    condition/exception rows therefore enter or leave the sample together.
    Multiple capabilities remain attached to one row; callers must not clone a
    row merely because it maps to more than one blueprint cell.
    """

    unit_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_group_id: str = Field(min_length=1)
    capability_ids: list[str] = Field(min_length=1)
    model_states: dict[str, ModelState] = Field(min_length=1)
    model_state_probabilities: dict[str, dict[ModelState, float]] = Field(default_factory=dict)
    model_state_granularity: ModelStateGranularity = "capability_prior"
    data_attributes: dict[str, str] = Field(default_factory=dict)
    population_weight: float = Field(default=1.0, gt=0)
    eligible: bool = True

    @field_validator("capability_ids")
    @classmethod
    def validate_capability_ids(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("capability_ids cannot contain empty IDs")
        if len(normalized) != len(set(normalized)):
            raise ValueError("capability_ids must be unique")
        return sorted(normalized)

    @field_validator("data_attributes")
    @classmethod
    def validate_data_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {str(key).strip(): str(item).strip() for key, item in value.items()}
        if any(not key or not item for key, item in normalized.items()):
            raise ValueError("data_attributes keys and values must be non-empty")
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def require_one_model_state_per_capability(self):
        if set(self.model_states) != set(self.capability_ids):
            raise ValueError("model_states keys must exactly match capability_ids")
        self.model_states = dict(sorted(self.model_states.items()))
        if self.model_state_probabilities:
            if set(self.model_state_probabilities) != set(self.capability_ids):
                raise ValueError("model_state_probabilities keys must exactly match capability_ids")
            normalized = {}
            for capability_id, probabilities in self.model_state_probabilities.items():
                if set(probabilities) != MODEL_STATES:
                    raise ValueError(
                        "Each model_state_probabilities entry must contain mastered, uncertain, "
                        "unmastered, and unknown"
                    )
                parsed = {state: float(value) for state, value in probabilities.items()}
                if any(value < 0 or value > 1 for value in parsed.values()):
                    raise ValueError("Model-state probabilities must lie in [0, 1]")
                if not math.isclose(sum(parsed.values()), 1.0, abs_tol=1e-6):
                    raise ValueError("Model-state probabilities must sum to 1")
                normalized[capability_id] = dict(sorted(parsed.items()))
            self.model_state_probabilities = dict(sorted(normalized.items()))
        if self.model_state_granularity == "unit_cell" and not self.model_state_probabilities:
            raise ValueError("unit_cell model states require model_state_probabilities")
        return self


class RareStratumQuota(StrictModel):
    """Minimum sampled dependency groups matching an important rare cell."""

    quota_id: str = Field(min_length=1)
    minimum_groups: int = Field(ge=1)
    capability_id: str | None = None
    model_state: ModelState | None = None
    data_attributes: dict[str, str] = Field(default_factory=dict)

    @field_validator("data_attributes")
    @classmethod
    def validate_data_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {str(key).strip(): str(item).strip() for key, item in value.items()}
        if any(not key or not item for key, item in normalized.items()):
            raise ValueError("quota data_attributes keys and values must be non-empty")
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def require_matcher(self):
        if self.capability_id is None and self.model_state is None and not self.data_attributes:
            raise ValueError("A rare-stratum quota must specify at least one matching field")
        return self


class ValueSamplingConfig(StrictModel):
    """Frozen probability-sampling design.

    Exactly one size control is required. ``sampling_fraction`` is interpreted
    as a lower bound: the realized sample may expand to keep positive support
    in every observed joint stratum or to satisfy preregistered rare quotas.
    The same expansion can be enabled for ``target_group_count`` through
    ``allow_target_expansion`` (enabled by default and always audited).
    """

    target_group_count: int | None = Field(default=None, ge=1)
    sampling_fraction: float | None = Field(default=None, gt=0, le=1)
    seed: int = 42
    stratum_attributes: list[str] = Field(default_factory=list)
    rare_stratum_quotas: list[RareStratumQuota] = Field(default_factory=list)
    allow_target_expansion: bool = True

    @field_validator("stratum_attributes")
    @classmethod
    def validate_stratum_attributes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("stratum_attributes cannot contain empty names")
        if len(normalized) != len(set(normalized)):
            raise ValueError("stratum_attributes must be unique")
        return sorted(normalized)

    @model_validator(mode="after")
    def validate_design(self):
        if (self.target_group_count is None) == (self.sampling_fraction is None):
            raise ValueError("Configure exactly one of target_group_count or sampling_fraction")
        quota_ids = [quota.quota_id for quota in self.rare_stratum_quotas]
        if len(quota_ids) != len(set(quota_ids)):
            raise ValueError("rare_stratum_quotas quota_id values must be unique")
        configured_attributes = set(self.stratum_attributes)
        for quota in self.rare_stratum_quotas:
            if unstratified := set(quota.data_attributes) - configured_attributes:
                raise ValueError(
                    f"Rare quota {quota.quota_id} matches attributes that are not part of the sampling strata: "
                    f"{sorted(unstratified)}"
                )
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sampling_config_digest(config: ValueSamplingConfig) -> str:
    """Hash normalized sampling semantics for a reproducible manifest."""

    return hashlib.sha256(_canonical_json(config.model_dump(mode="json")).encode("utf-8")).hexdigest()


def value_sampling_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return a formatting-independent digest of one complete sample manifest.

    The digest deliberately covers caller-added lineage fields as well as the
    sampler's core payload.  The provenance verifier below separately rebuilds
    the core payload from the frozen population frame; together these checks
    bind both the sampling design and the exact manifest that was certified.
    """

    return hashlib.sha256(_canonical_json(dict(manifest)).encode("utf-8")).hexdigest()


def _stratum_attributes(unit: ValueSamplingUnit, attribute_names: list[str]) -> dict[str, str]:
    missing = [name for name in attribute_names if name not in unit.data_attributes]
    if missing:
        raise ValueError(f"Unit {unit.unit_id} is missing configured stratum attributes: {missing}")
    return {name: unit.data_attributes[name] for name in attribute_names}


def _unit_signature_cell(unit: ValueSamplingUnit, attribute_names: list[str]) -> dict[str, Any]:
    return {
        "capability_model_state_pairs": [
            {"capability_id": capability_id, "model_state": unit.model_states[capability_id]}
            for capability_id in unit.capability_ids
        ],
        "data_attributes": _stratum_attributes(unit, attribute_names),
    }


def _unit_profile_cells(unit: ValueSamplingUnit, attribute_names: list[str]) -> list[tuple[dict[str, Any], float]]:
    """Return capability marginals without counting a multi-mapped unit K times."""

    share = 1.0 / len(unit.capability_ids)
    attributes = _stratum_attributes(unit, attribute_names)
    return [
        (
            {
                "capability_id": capability_id,
                "model_state": unit.model_states[capability_id],
                "data_attributes": attributes,
            },
            share,
        )
        for capability_id in unit.capability_ids
    ]


def _cell_id(cell: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(cell).encode("utf-8")).hexdigest()[:16]


def _group_signature(group: list[ValueSamplingUnit], attribute_names: list[str]) -> tuple[str, list[dict[str, Any]]]:
    counts: dict[str, dict[str, Any]] = {}
    for unit in group:
        cell = _unit_signature_cell(unit, attribute_names)
        canonical = _canonical_json(cell)
        if canonical not in counts:
            counts[canonical] = {"cell": cell, "unit_count": 0}
        counts[canonical]["unit_count"] += 1
    composition = [counts[key] for key in sorted(counts)]
    signature = hashlib.sha256(_canonical_json(composition).encode("utf-8")).hexdigest()[:16]
    return signature, composition


def _quota_matches_unit(quota: RareStratumQuota, unit: ValueSamplingUnit) -> bool:
    if quota.capability_id is not None and quota.capability_id not in unit.capability_ids:
        return False
    if quota.model_state is not None:
        if quota.capability_id is not None:
            if unit.model_states[quota.capability_id] != quota.model_state:
                return False
        elif quota.model_state not in unit.model_states.values():
            return False
    return all(unit.data_attributes.get(key) == value for key, value in quota.data_attributes.items())


def _allocate_groups(
    strata: dict[str, list[str]],
    stratum_quota_ids: dict[str, set[str]],
    quotas: list[RareStratumQuota],
    requested_size: int,
    allow_expansion: bool,
) -> tuple[dict[str, int], int, list[dict[str, Any]]]:
    """Create fixed per-stratum sample sizes before any random draw.

    Every nonempty joint stratum starts with one sampled group. This gives each
    eligible group a positive first-order inclusion probability. Rare quotas
    then raise lower bounds. Remaining slots follow proportional allocation.
    """

    allocation = {stratum_id: 1 for stratum_id in strata}
    quota_by_id = {quota.quota_id: quota for quota in quotas}

    impossible = []
    for quota in quotas:
        available = sum(
            len(strata[stratum_id]) for stratum_id in strata if quota.quota_id in stratum_quota_ids[stratum_id]
        )
        if available < quota.minimum_groups:
            impossible.append(
                f"{quota.quota_id}: minimum={quota.minimum_groups}, matching_population_groups={available}"
            )
    if impossible:
        raise ValueError("Rare-stratum quotas exceed the eligible population: " + "; ".join(impossible))

    def quota_count(quota_id: str) -> int:
        return sum(allocation[stratum_id] for stratum_id in strata if quota_id in stratum_quota_ids[stratum_id])

    # Increment the group-composition stratum that helps the largest number of
    # unsatisfied quotas. Ties are deterministic and favor remaining capacity.
    while True:
        deficits = {
            quota.quota_id: quota.minimum_groups - quota_count(quota.quota_id)
            for quota in quotas
            if quota_count(quota.quota_id) < quota.minimum_groups
        }
        if not deficits:
            break
        candidates = [
            stratum_id
            for stratum_id, group_ids in strata.items()
            if allocation[stratum_id] < len(group_ids)
            and any(quota_id in stratum_quota_ids[stratum_id] for quota_id in deficits)
        ]
        if not candidates:
            raise ValueError("Rare-stratum quota lower bounds are jointly infeasible")
        chosen = max(
            candidates,
            key=lambda stratum_id: (
                sum(deficits[quota_id] for quota_id in deficits if quota_id in stratum_quota_ids[stratum_id]),
                len(strata[stratum_id]) - allocation[stratum_id],
                stratum_id,
            ),
        )
        allocation[chosen] += 1

    required_size = sum(allocation.values())
    realized_target = requested_size
    expansion_reasons = []
    if required_size > requested_size:
        if not allow_expansion:
            raise ValueError(
                f"Requested {requested_size} groups cannot preserve positive joint-stratum support and quotas; "
                f"at least {required_size} are required"
            )
        realized_target = required_size
        expansion_reasons.append("joint_stratum_support_or_rare_quota")

    population_size = sum(len(group_ids) for group_ids in strata.values())
    if realized_target > population_size:
        raise ValueError(
            f"Requested sample size {realized_target} exceeds {population_size} eligible dependency groups"
        )

    # Largest-deficit proportional allocation, subject to the lower bounds.
    while sum(allocation.values()) < realized_target:
        current_size = sum(allocation.values())
        candidates = [stratum_id for stratum_id, group_ids in strata.items() if allocation[stratum_id] < len(group_ids)]
        chosen = max(
            candidates,
            key=lambda stratum_id: (
                realized_target * len(strata[stratum_id]) / population_size - allocation[stratum_id],
                len(strata[stratum_id]) - allocation[stratum_id],
                stratum_id,
            ),
        )
        allocation[chosen] += 1
        if sum(allocation.values()) <= current_size:  # defensive guard
            raise RuntimeError("Sampling allocation made no progress")

    quota_audit = []
    for quota_id in sorted(quota_by_id):
        quota = quota_by_id[quota_id]
        matching_population = sum(
            len(strata[stratum_id]) for stratum_id in strata if quota_id in stratum_quota_ids[stratum_id]
        )
        allocated = quota_count(quota_id)
        quota_audit.append(
            {
                "quota_id": quota_id,
                "minimum_groups": quota.minimum_groups,
                "matching_population_groups": matching_population,
                "allocated_sample_groups": allocated,
                "satisfied": allocated >= quota.minimum_groups,
            }
        )
    return allocation, realized_target, [{"reason": reason} for reason in expansion_reasons] + quota_audit


def _profile_rows(
    eligible_units: list[ValueSamplingUnit],
    selected_group_probabilities: dict[str, float],
    attribute_names: list[str],
) -> tuple[list[dict[str, Any]], float]:
    population: dict[str, float] = defaultdict(float)
    sample_raw: dict[str, float] = defaultdict(float)
    sample_weighted: dict[str, float] = defaultdict(float)
    cells: dict[str, dict[str, Any]] = {}

    for unit in eligible_units:
        for cell, share in _unit_profile_cells(unit, attribute_names):
            cell_key = _canonical_json(cell)
            cells[cell_key] = cell
            cell_weight = unit.population_weight * share
            population[cell_key] += cell_weight
            if unit.dependency_group_id in selected_group_probabilities:
                probability = selected_group_probabilities[unit.dependency_group_id]
                sample_raw[cell_key] += cell_weight
                sample_weighted[cell_key] += cell_weight / probability

    population_total = sum(population.values())
    raw_total = sum(sample_raw.values())
    weighted_total = sum(sample_weighted.values())
    rows = []
    total_variation = 0.0
    for cell_key in sorted(cells):
        population_proportion = population[cell_key] / population_total if population_total else 0.0
        raw_proportion = sample_raw[cell_key] / raw_total if raw_total else 0.0
        weighted_proportion = sample_weighted[cell_key] / weighted_total if weighted_total else 0.0
        absolute_error = abs(weighted_proportion - population_proportion)
        total_variation += absolute_error
        rows.append(
            {
                "cell_id": _cell_id(cells[cell_key]),
                **cells[cell_key],
                "population_weight": population[cell_key],
                "population_proportion": population_proportion,
                "sample_raw_weight": sample_raw[cell_key],
                "sample_raw_proportion": raw_proportion,
                "sample_weighted_weight": sample_weighted[cell_key],
                "sample_weighted_proportion": weighted_proportion,
                "weighted_absolute_error": absolute_error,
            }
        )
    return rows, 0.5 * total_variation


def sample_value_distribution(units: list[ValueSamplingUnit], config: ValueSamplingConfig) -> dict[str, Any]:
    """Draw a valuation-preserving probability sample and return its audit manifest.

    The function performs no relevance ranking, embedding optimization, or gap
    prioritization. Model state is a stratum to preserve, not a score to
    maximize. The returned Horvitz--Thompson weights are valid because the
    allocation is frozen before seeded simple random sampling within each
    group-composition stratum.
    """

    if not units:
        raise ValueError("At least one value-sampling unit is required")
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Value-sampling unit IDs must be unique")
    batch_ids = {unit.batch_id for unit in units}
    if len(batch_ids) != 1:
        raise ValueError("Value sampling must run independently for one batch at a time")
    batch_hashes = {unit.batch_content_sha256 for unit in units}
    if len(batch_hashes) != 1:
        raise ValueError("All units must come from the same frozen batch content")
    model_state_granularities = {unit.model_state_granularity for unit in units}
    if len(model_state_granularities) != 1:
        raise ValueError("One sampling run cannot mix capability_prior and unit_cell model states")
    model_state_granularity = next(iter(model_state_granularities))

    all_groups: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in units:
        all_groups[unit.dependency_group_id].append(unit)
    excluded_group_ids = {
        group_id for group_id, group in all_groups.items() if any(not unit.eligible for unit in group)
    }
    eligible_groups = {
        group_id: sorted(group, key=lambda unit: unit.unit_id)
        for group_id, group in all_groups.items()
        if group_id not in excluded_group_ids
    }
    if not eligible_groups:
        raise ValueError("No eligible dependency groups are available")

    strata: dict[str, list[str]] = defaultdict(list)
    stratum_composition: dict[str, list[dict[str, Any]]] = {}
    group_stratum: dict[str, str] = {}
    stratum_quota_ids: dict[str, set[str]] = defaultdict(set)
    for group_id in sorted(eligible_groups):
        stratum_id, composition = _group_signature(eligible_groups[group_id], config.stratum_attributes)
        strata[stratum_id].append(group_id)
        stratum_composition[stratum_id] = composition
        group_stratum[group_id] = stratum_id
        for quota in config.rare_stratum_quotas:
            if any(_quota_matches_unit(quota, unit) for unit in eligible_groups[group_id]):
                stratum_quota_ids[stratum_id].add(quota.quota_id)

    population_group_count = len(eligible_groups)
    if config.target_group_count is not None:
        requested_size = config.target_group_count
        requested_size_kind = "target_group_count"
    else:
        requested_size = math.ceil(population_group_count * float(config.sampling_fraction))
        requested_size_kind = "sampling_fraction_lower_bound"

    allocation, realized_target, allocation_audit = _allocate_groups(
        dict(strata),
        stratum_quota_ids,
        config.rare_stratum_quotas,
        requested_size,
        config.allow_target_expansion,
    )

    rng = random.Random(config.seed)
    selected_group_probabilities: dict[str, float] = {}
    group_records = []
    draw_trace = []
    for stratum_id in sorted(strata):
        population_groups = sorted(strata[stratum_id])
        draw_order = population_groups[:]
        rng.shuffle(draw_order)
        sample_count = allocation[stratum_id]
        selected = draw_order[:sample_count]
        probability = sample_count / len(population_groups)
        for rank, group_id in enumerate(selected, start=1):
            selected_group_probabilities[group_id] = probability
            draw_trace.append(
                {
                    "stratum_id": stratum_id,
                    "within_stratum_rank": rank,
                    "dependency_group_id": group_id,
                }
            )
        group_records.append(
            {
                "stratum_id": stratum_id,
                "composition": stratum_composition[stratum_id],
                "population_group_count": len(population_groups),
                "sample_group_count": sample_count,
                "inclusion_probability": probability,
                "design_weight": 1.0 / probability,
                "population_group_ids": population_groups,
                "selected_group_ids": selected,
                "matched_quota_ids": sorted(stratum_quota_ids[stratum_id]),
            }
        )

    selected_unit_records = []
    for group_id in sorted(selected_group_probabilities):
        probability = selected_group_probabilities[group_id]
        for unit in eligible_groups[group_id]:
            selected_unit_records.append(
                {
                    **unit.model_dump(mode="json"),
                    "inclusion_probability": probability,
                    "design_weight": 1.0 / probability,
                    "stratum_id": group_stratum[group_id],
                }
            )

    eligible_units = [unit for group in eligible_groups.values() for unit in group]
    profile, weighted_total_variation = _profile_rows(
        eligible_units,
        selected_group_probabilities,
        config.stratum_attributes,
    )
    model_state_population = defaultdict(float)
    model_state_sample_weighted = defaultdict(float)
    for row in profile:
        model_state_population[row["model_state"]] += row["population_weight"]
        model_state_sample_weighted[row["model_state"]] += row["sample_weighted_weight"]

    return {
        "schema_version": "value-distribution-sample-v2",
        "design": "fixed-allocation-stratified-srswor",
        "purpose": "conditional_value_distribution_estimation",
        "batch_id": next(iter(batch_ids)),
        "batch_content_sha256": next(iter(batch_hashes)),
        "sampling_config": config.model_dump(mode="json"),
        "sampling_config_sha256": value_sampling_config_digest(config),
        "seed": config.seed,
        "model_state_granularity": model_state_granularity,
        "stratification_basis": ["capability_model_state_pairs", *config.stratum_attributes],
        "requested_size_kind": requested_size_kind,
        "requested_group_count": requested_size,
        "realized_group_count": realized_target,
        "population_group_count": population_group_count,
        "realized_sampling_fraction": realized_target / population_group_count,
        "selected_group_ids": sorted(selected_group_probabilities),
        "selected_unit_ids": [record["unit_id"] for record in selected_unit_records],
        "selected_unit_records": selected_unit_records,
        "group_strata": group_records,
        "joint_profile": profile,
        "model_state_totals": {
            state: {
                "population_weight": model_state_population.get(state, 0.0),
                "sample_weighted_weight": model_state_sample_weighted.get(state, 0.0),
            }
            for state in ("mastered", "uncertain", "unmastered", "unknown")
        },
        "weighted_total_variation": weighted_total_variation,
        "rare_quota_audit": [item for item in allocation_audit if "quota_id" in item],
        "size_expansion_audit": [item for item in allocation_audit if "reason" in item],
        "ineligible_unit_ids": sorted(unit.unit_id for unit in units if not unit.eligible),
        "excluded_dependency_group_ids": sorted(excluded_group_ids),
        "dependency_units_excluded": sum(
            unit.eligible for group_id in excluded_group_ids for unit in all_groups[group_id]
        ),
        "draw_trace": draw_trace,
    }


def verify_value_sample_manifest(
    manifest: Mapping[str, Any],
    population_units: list[ValueSamplingUnit],
) -> dict[str, Any]:
    """Replay and verify a sample manifest against its complete population frame.

    A selected-row profile can internally assert any inclusion probability it
    wants.  It cannot, by itself, prove that ``pi`` came from the registered
    ``n_h / N_h`` design.  This verifier closes that gap by rebuilding the
    strata and deterministic draw from the full source population and frozen
    sampling config.  Any change to membership, stratum counts, selected-row
    probabilities/weights, seed, config hash, or batch identity is rejected.

    Caller-added lineage fields are allowed, but the returned manifest digest
    covers them too so a certificate can bind the exact serialized content.
    """

    payload = dict(manifest)
    if payload.get("schema_version") != "value-distribution-sample-v2":
        raise ValueError("Unsupported value sample manifest schema_version")
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Value sample manifest requires a non-empty batch_id")
    if not isinstance(payload.get("sampling_config"), dict):
        raise ValueError("Value sample manifest requires a sampling_config object")

    config = ValueSamplingConfig.model_validate(payload["sampling_config"])
    config_sha256 = value_sampling_config_digest(config)
    if payload.get("sampling_config_sha256") != config_sha256:
        raise ValueError("Value sample manifest sampling_config_sha256 does not match sampling_config")

    frame = [unit for unit in population_units if unit.batch_id == batch_id]
    if not frame:
        raise ValueError(f"Population frame contains no units for manifest batch {batch_id}")
    batch_hashes = {unit.batch_content_sha256 for unit in frame}
    if len(batch_hashes) != 1:
        raise ValueError("Population frame contains multiple batch content hashes")
    frame_batch_sha256 = next(iter(batch_hashes))
    if payload.get("batch_content_sha256") != frame_batch_sha256:
        raise ValueError("Value sample manifest batch_content_sha256 does not match population frame")

    expected = sample_value_distribution(frame, config)
    mismatches = [
        key for key, expected_value in expected.items() if key not in payload or payload[key] != expected_value
    ]
    if mismatches:
        raise ValueError(
            "Value sample manifest does not replay from the registered population design; "
            f"mismatched fields: {mismatches}"
        )

    frame_payload = [unit.model_dump(mode="json") for unit in sorted(frame, key=lambda row: row.unit_id)]
    grouped_frame: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in frame:
        grouped_frame[unit.dependency_group_id].append(unit)
    eligible_groups = {
        group_id: sorted(group, key=lambda row: row.unit_id)
        for group_id, group in grouped_frame.items()
        if all(unit.eligible for unit in group)
    }
    selected_probabilities = {
        group_id: row["inclusion_probability"]
        for row in expected["group_strata"]
        for group_id in row["selected_group_ids"]
    }

    def profile_group_design(group_ids: list[str], *, sampled: bool) -> list[dict[str, Any]]:
        return [
            {
                "dependency_group_id": group_id,
                "unit_ids": [unit.unit_id for unit in eligible_groups[group_id]],
                "inclusion_probability": selected_probabilities[group_id] if sampled else 1.0,
                "population_weight": sum(unit.population_weight for unit in eligible_groups[group_id]),
            }
            for group_id in sorted(group_ids)
        ]

    reference_group_design = profile_group_design(list(eligible_groups), sampled=False)
    sample_group_design = profile_group_design(expected["selected_group_ids"], sampled=True)
    reference_membership = [
        {
            "dependency_group_id": row["dependency_group_id"],
            "unit_ids": row["unit_ids"],
        }
        for row in reference_group_design
    ]
    sample_membership = [
        {
            "dependency_group_id": row["dependency_group_id"],
            "unit_ids": row["unit_ids"],
        }
        for row in sample_group_design
    ]
    stratum_audit = []
    for row in expected["group_strata"]:
        stratum_audit.append(
            {
                "stratum_id": row["stratum_id"],
                "population_group_count": row["population_group_count"],
                "sample_group_count": row["sample_group_count"],
                "inclusion_probability": row["inclusion_probability"],
                "design_weight": row["design_weight"],
                "population_group_ids_sha256": hashlib.sha256(
                    _canonical_json(row["population_group_ids"]).encode("utf-8")
                ).hexdigest(),
                "selected_group_ids_sha256": hashlib.sha256(
                    _canonical_json(row["selected_group_ids"]).encode("utf-8")
                ).hexdigest(),
            }
        )

    return {
        "schema_version": "value-sampling-provenance-verification-v1",
        "design_provenance_verified": True,
        "sampling_manifest_sha256": value_sampling_manifest_digest(payload),
        "population_frame_sha256": hashlib.sha256(_canonical_json(frame_payload).encode("utf-8")).hexdigest(),
        "sampling_config_sha256": config_sha256,
        "batch_id": batch_id,
        "batch_content_sha256": frame_batch_sha256,
        "reference_membership_sha256": hashlib.sha256(
            _canonical_json(reference_membership).encode("utf-8")
        ).hexdigest(),
        "sample_membership_sha256": hashlib.sha256(_canonical_json(sample_membership).encode("utf-8")).hexdigest(),
        "sample_design_sha256": hashlib.sha256(_canonical_json(sample_group_design).encode("utf-8")).hexdigest(),
        "population_unit_count": len(frame),
        "population_group_count": expected["population_group_count"],
        "selected_unit_count": len(expected["selected_unit_ids"]),
        "selected_group_count": expected["realized_group_count"],
        "stratum_count": len(expected["group_strata"]),
        "strata": stratum_audit,
        "checks": {
            "sampling_config_hash": True,
            "batch_content_hash": True,
            "population_membership": True,
            "selected_membership": True,
            "stratum_population_and_sample_counts": True,
            "inclusion_probability_equals_n_h_over_N_h": True,
            "design_weight_equals_inverse_probability": True,
            "deterministic_draw_replay": True,
        },
    }


__all__ = [
    "ModelState",
    "ModelStateGranularity",
    "RareStratumQuota",
    "ValueSamplingConfig",
    "ValueSamplingUnit",
    "sample_value_distribution",
    "value_sampling_config_digest",
    "value_sampling_manifest_digest",
    "verify_value_sample_manifest",
]
