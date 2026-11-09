"""Auditable unit-by-benchmark-cell Base Model measurements.

Capability-level probes are useful priors, but they cannot establish that a
model knows one rule and misses another rule inside the same capability.  This
module defines the optional finer measurement used when the valuation sample
must preserve the full batch's unit-level mastered/uncertain/unmastered mix.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.valuation.conditional_value import (
    MasteryProbabilities,
    MasteryState,
    ModelMasteryProfile,
    ScoreInterval,
)
from ai_ready.valuation.value_sampling import ValueSamplingUnit


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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


def _validate_sha256(value: str, field_name: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


def unit_population_digest(units: Sequence[ValueSamplingUnit | Mapping[str, Any]]) -> str:
    """Hash source membership and attributes while excluding measured states."""

    parsed = [unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit) for unit in units]
    payload = [
        {
            "unit_id": unit.unit_id,
            "batch_id": unit.batch_id,
            "batch_content_sha256": unit.batch_content_sha256,
            "dependency_group_id": unit.dependency_group_id,
            "capability_ids": unit.capability_ids,
            "data_attributes": unit.data_attributes,
            "population_weight": unit.population_weight,
            "eligible": unit.eligible,
        }
        for unit in sorted(parsed, key=lambda item: item.unit_id)
    ]
    return _canonical_digest(payload)


class UnitCellMasteryObservation(_StrictModel):
    """One Base Model measurement for one source unit and blueprint cell."""

    unit_id: str = Field(min_length=1)
    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    point_score: float | None = Field(default=None, ge=0, le=1)
    score_interval: ScoreInterval | None = None
    probe_count: int = Field(ge=0)
    probe_ids: list[str] = Field(default_factory=list)
    probabilities: MasteryProbabilities
    stratum_state: MasteryState
    used_locked_test_items: bool = False

    @field_validator("probe_ids")
    @classmethod
    def validate_probe_ids(cls, value: list[str]) -> list[str]:
        normalized = sorted(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("probe_ids cannot contain empty IDs")
        if len(normalized) != len(set(normalized)):
            raise ValueError("probe_ids must be unique within a unit-cell observation")
        return normalized

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.used_locked_test_items:
            raise ValueError("Unit-level model probes must not use locked benchmark test items")
        if self.probe_count == 0:
            if self.point_score is not None or self.score_interval is not None or self.probe_ids:
                raise ValueError("An unmeasured unit-cell cannot report scores or probe IDs")
            if self.probabilities != MasteryProbabilities.one_hot("unknown"):
                raise ValueError("An unmeasured unit-cell must have probability 1 on unknown")
            if self.stratum_state != "unknown":
                raise ValueError("An unmeasured unit-cell must use stratum_state=unknown")
        else:
            if self.point_score is None or self.score_interval is None or not self.probe_ids:
                raise ValueError("A measured unit-cell requires score, interval, and probe IDs")
            if not self.score_interval.lower <= self.point_score <= self.score_interval.upper:
                raise ValueError("point_score must lie inside score_interval")
            if self.stratum_state == "unknown" or self.probabilities.unknown > 1e-6:
                raise ValueError("A measured unit-cell cannot retain unknown probability mass")
        return self


class UnitCellMasteryProfile(_StrictModel):
    """Frozen unit-level refinement tied to one model and source population."""

    schema_version: Literal["unit-cell-mastery-profile-v1"] = "unit-cell-mastery-profile-v1"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    parent_model_profile_id: str = Field(min_length=1)
    parent_model_profile_sha256: str
    benchmark_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_sha256: str
    population_units_sha256: str
    probe_set_id: str = Field(min_length=1)
    probe_set_sha256: str
    scoring_protocol_sha256: str
    measurement_granularity: Literal["unit_cell"] = "unit_cell"
    observations: list[UnitCellMasteryObservation] = Field(min_length=1)

    @field_validator(
        "parent_model_profile_sha256",
        "checkpoint_sha256",
        "population_units_sha256",
        "probe_set_sha256",
        "scoring_protocol_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_observations(self):
        keys = [(item.unit_id, item.cell_id) for item in self.observations]
        if len(keys) != len(set(keys)):
            raise ValueError("Unit-cell observations must have unique (unit_id, cell_id) pairs")
        self.observations = sorted(self.observations, key=lambda item: (item.unit_id, item.cell_id))
        return self

    @property
    def by_unit_cell(self) -> dict[tuple[str, str], UnitCellMasteryObservation]:
        return {(item.unit_id, item.cell_id): item for item in self.observations}


def _eligible_population_units(units: Sequence[ValueSamplingUnit]) -> list[ValueSamplingUnit]:
    groups: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in units:
        groups[unit.dependency_group_id].append(unit)
    return [
        unit
        for _, group in sorted(groups.items())
        if all(item.eligible for item in group)
        for unit in sorted(group, key=lambda item: item.unit_id)
    ]


def validate_unit_cell_mastery_profile(
    profile: UnitCellMasteryProfile,
    parent: ModelMasteryProfile,
    units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
) -> None:
    """Reject a unit profile detached from its model or population frame."""

    parsed = [unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit) for unit in units]
    if profile.parent_model_profile_id != parent.profile_id:
        raise ValueError("Unit mastery profile references a different parent model profile")
    if profile.parent_model_profile_sha256 != _canonical_digest(parent):
        raise ValueError("Unit mastery parent model profile digest does not match")
    for field_name in ("benchmark_id", "model_id", "model_revision", "checkpoint_sha256"):
        if getattr(profile, field_name) != getattr(parent, field_name):
            raise ValueError(f"Unit mastery {field_name} does not match the parent model profile")
    if profile.population_units_sha256 != unit_population_digest(parsed):
        raise ValueError("Unit mastery profile is detached from the source population")

    eligible = _eligible_population_units(parsed)
    expected_pairs = {(unit.unit_id, capability_id) for unit in eligible for capability_id in unit.capability_ids}
    observed_pairs = set(profile.by_unit_cell)
    if missing := expected_pairs - observed_pairs:
        raise ValueError(f"Unit mastery profile is missing unit-cell observations: {sorted(missing)[:5]}")
    if extra := observed_pairs - expected_pairs:
        raise ValueError(f"Unit mastery profile contains observations outside the population: {sorted(extra)[:5]}")
    if unknown_cells := {cell_id for _, cell_id in observed_pairs} - set(parent.by_cell):
        raise ValueError(f"Unit mastery profile references unknown benchmark cells: {sorted(unknown_cells)}")


def build_unit_cell_mastery_profile(
    *,
    profile_id: str,
    probe_set_id: str,
    scoring_protocol_sha256: str,
    parent: ModelMasteryProfile,
    units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    observations: Sequence[UnitCellMasteryObservation | Mapping[str, Any]],
) -> UnitCellMasteryProfile:
    """Build and immediately validate a unit-level model measurement profile."""

    parsed_units = [
        unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit) for unit in units
    ]
    parsed_observations = [
        item if isinstance(item, UnitCellMasteryObservation) else UnitCellMasteryObservation.model_validate(item)
        for item in observations
    ]
    probe_set_sha256 = _canonical_digest(
        [
            item.model_dump(mode="json")
            for item in sorted(parsed_observations, key=lambda row: (row.unit_id, row.cell_id))
        ]
    )
    profile = UnitCellMasteryProfile(
        profile_id=profile_id,
        parent_model_profile_id=parent.profile_id,
        parent_model_profile_sha256=_canonical_digest(parent),
        benchmark_id=parent.benchmark_id,
        model_id=parent.model_id,
        model_revision=parent.model_revision,
        checkpoint_sha256=parent.checkpoint_sha256,
        population_units_sha256=unit_population_digest(parsed_units),
        probe_set_id=probe_set_id,
        probe_set_sha256=probe_set_sha256,
        scoring_protocol_sha256=scoring_protocol_sha256,
        observations=parsed_observations,
    )
    validate_unit_cell_mastery_profile(profile, parent, parsed_units)
    return profile


def unit_cell_mastery_profile_digest(profile: UnitCellMasteryProfile) -> str:
    return _canonical_digest(profile)


__all__ = [
    "UnitCellMasteryObservation",
    "UnitCellMasteryProfile",
    "build_unit_cell_mastery_profile",
    "unit_cell_mastery_profile_digest",
    "unit_population_digest",
    "validate_unit_cell_mastery_profile",
]
