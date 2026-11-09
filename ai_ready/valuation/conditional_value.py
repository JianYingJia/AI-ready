"""Conditional-value profiles for representative valuation samples.

The profile in this module is a *design-stage proxy*, not observed training
utility.  It describes how source-derived semantic units are distributed over
benchmark cells, independently measured base-model mastery states, and
pre-registered data attributes.  A sample is faithful when its
inclusion-probability-weighted profile reconstructs the corresponding full
population profile.

The implementation deliberately keeps the four mastery states probabilistic.
A borderline probe result must not be collapsed into a confident binary
``mastered``/``unmastered`` label, and missing probes remain ``unknown``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MasteryState = Literal["mastered", "uncertain", "unmastered", "unknown"]
PopulationKind = Literal["full_population", "raw_sample", "weighted_reconstruction"]
MASTERY_STATES: tuple[MasteryState, ...] = ("mastered", "uncertain", "unmastered", "unknown")
_PROBABILITY_TOLERANCE = 1e-6


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


class ScoreInterval(_StrictModel):
    """A calibrated interval for a benchmark-cell probe score."""

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.lower > self.upper:
            raise ValueError("Score interval lower bound cannot exceed its upper bound")
        return self


class MasteryProbabilities(_StrictModel):
    """Soft membership in the four non-overlapping model-mastery states."""

    mastered: float = Field(ge=0, le=1)
    uncertain: float = Field(ge=0, le=1)
    unmastered: float = Field(ge=0, le=1)
    unknown: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_probability_simplex(self):
        total = self.mastered + self.uncertain + self.unmastered + self.unknown
        if not math.isclose(total, 1.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise ValueError(f"Mastery probabilities must sum to 1; received {total}")
        return self

    def as_dict(self) -> dict[MasteryState, float]:
        return {state: float(getattr(self, state)) for state in MASTERY_STATES}

    @classmethod
    def one_hot(cls, state: MasteryState) -> "MasteryProbabilities":
        return cls(**{candidate: float(candidate == state) for candidate in MASTERY_STATES})


class ProbeObservation(_StrictModel):
    """Independent base-model calibration for one benchmark blueprint cell.

    ``point_score`` and ``score_interval`` describe observed performance.  The
    state probabilities come from the registered calibration procedure (for
    example bootstrap replicates); this module does not invent them from the
    point score.  With no valid probes, the observation must remain fully
    ``unknown``.
    """

    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    capability_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    difficulty_id: str = Field(min_length=1)
    benchmark_weight: float = Field(default=1.0, gt=0)
    point_score: float | None = Field(default=None, ge=0, le=1)
    score_interval: ScoreInterval | None = None
    probe_count: int = Field(ge=0)
    probabilities: MasteryProbabilities
    stratum_state: MasteryState
    scoring_method: str = Field(min_length=1)
    used_locked_test_items: bool = False

    @model_validator(mode="after")
    def validate_observation(self):
        if self.used_locked_test_items:
            raise ValueError("Model-mastery probes must not use locked benchmark test items")
        if self.point_score is None:
            if self.score_interval is not None:
                raise ValueError("An observation without a point score cannot report an interval")
            if self.probabilities != MasteryProbabilities.one_hot("unknown"):
                raise ValueError("An unmeasured observation must have probability 1 on unknown")
            if self.stratum_state != "unknown":
                raise ValueError("An unmeasured observation must use stratum_state=unknown")
        else:
            if self.probe_count == 0:
                raise ValueError("An observation with zero probes cannot report a score")
            if self.point_score is None or self.score_interval is None:
                raise ValueError("A probed benchmark cell requires both a point score and score interval")
            if not self.score_interval.lower <= self.point_score <= self.score_interval.upper:
                raise ValueError("point_score must lie inside score_interval")
            if self.stratum_state == "unknown" or self.probabilities.unknown > _PROBABILITY_TOLERANCE:
                raise ValueError("A measured observation cannot have an unknown state or unknown probability mass")
        return self


class BenchmarkCellDefinition(_StrictModel):
    """Registered blueprint metadata required to construct probe observations."""

    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    capability_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    difficulty_id: str = Field(min_length=1)
    benchmark_weight: float = Field(default=1.0, gt=0)


class ScoredProbe(_StrictModel):
    """One independently scored calibration item; locked test items are illegal."""

    probe_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    variant_id: str | None = Field(default=None, min_length=1)
    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    score: float = Field(ge=0, le=1)
    is_locked_test_item: bool = False

    @model_validator(mode="after")
    def reject_locked_test_item(self):
        if self.is_locked_test_item:
            raise ValueError("Locked benchmark test items cannot be used as model-mastery probes")
        return self


class ProbeAggregationConfig(_StrictModel):
    """Frozen bootstrap and mastery-threshold semantics."""

    tau_low: float = Field(ge=0, le=1)
    tau_high: float = Field(ge=0, le=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_iterations: int = Field(default=2000, ge=100)
    seed: int = 42
    minimum_probe_count: int = Field(default=10, ge=1)
    scoring_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_thresholds(self):
        if self.tau_low >= self.tau_high:
            raise ValueError("tau_low must be strictly less than tau_high")
        return self


def probe_aggregation_config_digest(config: ProbeAggregationConfig) -> str:
    return _canonical_digest(config)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(ordered[lower_index] * (1 - fraction) + ordered[upper_index] * fraction)


def aggregate_probe_observations(
    cells: Sequence[BenchmarkCellDefinition | Mapping[str, Any]],
    probes: Sequence[ScoredProbe | Mapping[str, Any]],
    config: ProbeAggregationConfig,
) -> list[ProbeObservation]:
    """Aggregate item scores into deterministic cell-level soft mastery states.

    Bootstrap replicate means below ``tau_low`` are unmastered, those at or
    above ``tau_high`` are mastered, and the interval between is uncertain.
    The hard confidence-interval state uses the same inclusive upper boundary.
    Missing or insufficient cells remain fully unknown.
    """

    parsed_cells = [
        cell if isinstance(cell, BenchmarkCellDefinition) else BenchmarkCellDefinition.model_validate(cell)
        for cell in cells
    ]
    parsed_probes = [probe if isinstance(probe, ScoredProbe) else ScoredProbe.model_validate(probe) for probe in probes]
    cell_ids = [cell.cell_id for cell in parsed_cells]
    if not cell_ids:
        raise ValueError("At least one registered benchmark cell is required")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Benchmark cell definitions must have unique cell_id values")
    probe_ids = [probe.probe_id for probe in parsed_probes]
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("Scored probes must have unique probe_id values")
    item_variants = [(probe.item_id, probe.variant_id) for probe in parsed_probes]
    if len(item_variants) != len(set(item_variants)):
        raise ValueError("Scored probes must have unique (item_id, variant_id) pairs")
    if unknown := {probe.cell_id for probe in parsed_probes} - set(cell_ids):
        raise ValueError(f"Scored probes reference unknown benchmark cells: {sorted(unknown)}")

    item_cells: dict[str, str] = {}
    scores_by_cell_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for probe in parsed_probes:
        previous_cell = item_cells.setdefault(probe.item_id, probe.cell_id)
        if previous_cell != probe.cell_id:
            raise ValueError(f"Probe item {probe.item_id} cannot belong to multiple benchmark cells")
        scores_by_cell_item[probe.cell_id][probe.item_id].append(probe.score)

    observations = []
    for cell in sorted(parsed_cells, key=lambda item: item.cell_id):
        # Prompt variants are repeated measurements of one item.  Averaging
        # them before resampling prevents prompt count from inflating the
        # effective item count or narrowing the interval spuriously.
        scores = [
            sum(variant_scores) / len(variant_scores)
            for _, variant_scores in sorted(scores_by_cell_item[cell.cell_id].items())
        ]
        if len(scores) < config.minimum_probe_count:
            observations.append(
                ProbeObservation(
                    cell_id=cell.cell_id,
                    capability_id=cell.capability_id,
                    operation_id=cell.operation_id,
                    difficulty_id=cell.difficulty_id,
                    benchmark_weight=cell.benchmark_weight,
                    point_score=None,
                    score_interval=None,
                    probe_count=len(scores),
                    probabilities=MasteryProbabilities.one_hot("unknown"),
                    stratum_state="unknown",
                    scoring_method=config.scoring_method,
                )
            )
            continue

        # A per-cell derived seed makes results invariant to cell input order.
        seed_material = f"{config.seed}\0{cell.cell_id}".encode("utf-8")
        cell_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(cell_seed)
        sample_size = len(scores)
        replicate_means = [
            sum(scores[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
            for _ in range(config.bootstrap_iterations)
        ]
        state_counts = {
            "mastered": sum(mean >= config.tau_high for mean in replicate_means),
            "uncertain": sum(config.tau_low <= mean < config.tau_high for mean in replicate_means),
            "unmastered": sum(mean < config.tau_low for mean in replicate_means),
            "unknown": 0,
        }
        alpha = 1.0 - config.confidence_level
        interval = ScoreInterval(
            lower=_percentile(replicate_means, alpha / 2),
            upper=_percentile(replicate_means, 1 - alpha / 2),
            confidence_level=config.confidence_level,
        )
        if interval.lower >= config.tau_high:
            stratum_state: MasteryState = "mastered"
        elif interval.upper < config.tau_low:
            stratum_state = "unmastered"
        else:
            stratum_state = "uncertain"
        observations.append(
            ProbeObservation(
                cell_id=cell.cell_id,
                capability_id=cell.capability_id,
                operation_id=cell.operation_id,
                difficulty_id=cell.difficulty_id,
                benchmark_weight=cell.benchmark_weight,
                point_score=sum(scores) / sample_size,
                score_interval=interval,
                probe_count=sample_size,
                probabilities=MasteryProbabilities(
                    **{state: state_counts[state] / config.bootstrap_iterations for state in MASTERY_STATES}
                ),
                stratum_state=stratum_state,
                scoring_method=config.scoring_method,
            )
        )
    return observations


class ModelMasteryProfile(_StrictModel):
    """Frozen model-initialization profile measured outside the locked test."""

    schema_version: Literal["model-mastery-profile-v1"] = "model-mastery-profile-v1"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    benchmark_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_sha256: str
    probe_set_id: str = Field(min_length=1)
    probe_set_sha256: str
    scoring_protocol_sha256: str
    observations: list[ProbeObservation] = Field(min_length=1)

    @field_validator("checkpoint_sha256", "probe_set_sha256", "scoring_protocol_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_cells(self):
        cell_ids = [observation.cell_id for observation in self.observations]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Model mastery observations must have unique cell_id values")
        self.observations = sorted(self.observations, key=lambda observation: observation.cell_id)
        return self

    @property
    def by_cell(self) -> dict[str, ProbeObservation]:
        return {observation.cell_id: observation for observation in self.observations}

    @property
    def benchmark_cell_weights(self) -> dict[str, float]:
        return {observation.cell_id: observation.benchmark_weight for observation in self.observations}


def build_model_mastery_profile(
    *,
    profile_id: str,
    benchmark_id: str,
    model_id: str,
    model_revision: str,
    checkpoint_sha256: str,
    probe_set_id: str,
    probe_set_sha256: str,
    scoring_protocol_sha256: str,
    observations: Sequence[ProbeObservation | Mapping[str, Any]],
) -> ModelMasteryProfile:
    """Build and validate a frozen model profile without inferring missing probes."""

    parsed = [
        observation if isinstance(observation, ProbeObservation) else ProbeObservation.model_validate(observation)
        for observation in observations
    ]
    return ModelMasteryProfile(
        profile_id=profile_id,
        benchmark_id=benchmark_id,
        model_id=model_id,
        model_revision=model_revision,
        checkpoint_sha256=checkpoint_sha256,
        probe_set_id=probe_set_id,
        probe_set_sha256=probe_set_sha256,
        scoring_protocol_sha256=scoring_protocol_sha256,
        observations=parsed,
    )


def build_model_mastery_profile_from_probes(
    *,
    profile_id: str,
    benchmark_id: str,
    model_id: str,
    model_revision: str,
    checkpoint_sha256: str,
    probe_set_id: str,
    probe_set_sha256: str,
    cells: Sequence[BenchmarkCellDefinition | Mapping[str, Any]],
    probes: Sequence[ScoredProbe | Mapping[str, Any]],
    aggregation_config: ProbeAggregationConfig,
    evidence_protocol_sha256: str | None = None,
) -> ModelMasteryProfile:
    """Build a frozen model profile directly from item-level scored probes.

    ``evidence_protocol_sha256`` lets a real inference run bind the model,
    prompts, raw responses, scorer, and aggregation configuration into one
    digest. Legacy imports that only contain pre-scored rows retain the
    aggregation-config digest so existing contracts remain reproducible.
    """

    observations = aggregate_probe_observations(cells, probes, aggregation_config)
    return build_model_mastery_profile(
        profile_id=profile_id,
        benchmark_id=benchmark_id,
        model_id=model_id,
        model_revision=model_revision,
        checkpoint_sha256=checkpoint_sha256,
        probe_set_id=probe_set_id,
        probe_set_sha256=probe_set_sha256,
        scoring_protocol_sha256=(evidence_protocol_sha256 or probe_aggregation_config_digest(aggregation_config)),
        observations=observations,
    )


class UnitCellMapping(_StrictModel):
    """Soft mapping of one semantic unit to a benchmark blueprint cell."""

    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    weight: float = Field(gt=0, le=1)


class ConditionalValueUnit(_StrictModel):
    """One source-derived semantic unit before task-example generation.

    ``population_weight`` describes the unit's mass in the frozen source
    population.  ``inclusion_probability`` is the probability produced by the
    sampling design; it is never replaced by an empirical sample fraction.
    """

    unit_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    dependency_group_id: str = Field(min_length=1)
    cell_mappings: list[UnitCellMapping] = Field(min_length=1)
    data_attributes: dict[str, str] = Field(min_length=1)
    population_weight: float = Field(default=1.0, gt=0)
    inclusion_probability: float = Field(default=1.0, gt=0, le=1)
    model_state_overrides: dict[str, MasteryProbabilities] = Field(default_factory=dict)
    eligible: bool = True

    @field_validator("batch_content_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "batch_content_sha256")

    @field_validator("data_attributes")
    @classmethod
    def validate_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not str(key).strip() or not str(item).strip() for key, item in value.items()):
            raise ValueError("data_attributes keys and values must be non-empty strings")
        return dict(sorted((str(key), str(item)) for key, item in value.items()))

    @model_validator(mode="after")
    def validate_mappings(self):
        cell_ids = [mapping.cell_id for mapping in self.cell_mappings]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("A unit cannot map to the same benchmark cell more than once")
        total = sum(mapping.weight for mapping in self.cell_mappings)
        if not math.isclose(total, 1.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise ValueError(f"Unit cell mapping weights must sum to 1; received {total}")
        self.cell_mappings = sorted(self.cell_mappings, key=lambda mapping: mapping.cell_id)
        unknown_override_cells = set(self.model_state_overrides) - set(cell_ids)
        if unknown_override_cells:
            raise ValueError(
                f"model_state_overrides reference cells absent from cell_mappings: {sorted(unknown_override_cells)}"
            )
        self.model_state_overrides = dict(sorted(self.model_state_overrides.items()))
        return self


def conditional_value_unit_from_sampling_record(
    record: Mapping[str, Any] | BaseModel,
    *,
    assume_unknown_when_missing: bool = False,
    use_model_state_overrides: bool = False,
) -> ConditionalValueUnit:
    """Adapt a stratified-sampler row without losing its design probability.

    The adapter accepts either ``cell_mappings`` or one or more capability IDs.
    Sampler ``model_states`` are audited but, by default, discarded so profile
    aggregation continues to use the independent soft
    :class:`ModelMasteryProfile`.  Callers may explicitly retain them as
    per-cell one-hot overrides.  Legacy data with no real probes may explicitly
    request an ``unknown`` override.
    """

    payload = record.model_dump(mode="python") if isinstance(record, BaseModel) else dict(record)
    if "cell_mappings" not in payload:
        capability_ids = payload.pop("capability_ids", None)
        cell_id = payload.pop("cell_id", None) or payload.pop("capability_id", None)
        if capability_ids is None and cell_id is not None:
            capability_ids = [cell_id]
        if not capability_ids:
            raise ValueError("Sampling record requires cell_mappings, cell_id, capability_id, or capability_ids")
        capability_ids = sorted(str(capability_id) for capability_id in capability_ids)
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability_ids must be unique")
        mapping_weight = 1.0 / len(capability_ids)
        payload["cell_mappings"] = [
            {"cell_id": capability_id, "weight": mapping_weight} for capability_id in capability_ids
        ]

    mapped_cell_ids = {mapping["cell_id"] for mapping in payload["cell_mappings"]}
    states = payload.pop("model_states", None)
    state_probabilities = payload.pop("model_state_probabilities", None)
    state_granularity = payload.pop("model_state_granularity", "capability_prior")
    if state_granularity not in {"capability_prior", "unit_cell"}:
        raise ValueError(f"Unknown model_state_granularity: {state_granularity}")
    legacy_state = payload.pop("model_state", None)
    if states is not None:
        if set(states) != mapped_cell_ids:
            raise ValueError("model_states keys must exactly equal mapped benchmark cells")
        if any(state not in MASTERY_STATES for state in states.values()):
            raise ValueError("model_states contains an unknown state")
    if legacy_state is not None:
        if len(mapped_cell_ids) != 1:
            raise ValueError("A singular model_state is valid only for a single mapped benchmark cell")
        if legacy_state not in MASTERY_STATES:
            raise ValueError(f"Unknown model_state: {legacy_state}")
        states = {next(iter(mapped_cell_ids)): legacy_state}
    if state_probabilities:
        if states is None:
            raise ValueError("model_state_probabilities require model_states")
        if set(state_probabilities) != mapped_cell_ids:
            raise ValueError("model_state_probabilities keys must exactly equal mapped benchmark cells")
        parsed_probabilities = {
            cell_id: MasteryProbabilities.model_validate(probabilities)
            for cell_id, probabilities in state_probabilities.items()
        }
    else:
        parsed_probabilities = None
    if state_granularity == "unit_cell" and (states is None or parsed_probabilities is None):
        raise ValueError("unit_cell records require hard states and soft model_state_probabilities")
    # Unit-level measurements must affect both the sampling strata and the
    # reconstructed value portrait.  Capability priors remain soft values from
    # ModelMasteryProfile unless callers explicitly request a hard override.
    use_record_overrides = use_model_state_overrides or state_granularity == "unit_cell"
    if use_record_overrides and states is not None:
        payload["model_state_overrides"] = parsed_probabilities or {
            cell_id: MasteryProbabilities.one_hot(state) for cell_id, state in states.items()
        }
    elif assume_unknown_when_missing and payload.get("model_state_overrides") is None:
        payload["model_state_overrides"] = {
            cell_id: MasteryProbabilities.one_hot("unknown") for cell_id in mapped_cell_ids
        }

    design_weight = payload.pop("design_weight", None)
    if design_weight is not None:
        probability = float(payload.get("inclusion_probability", 0.0))
        if probability <= 0 or not math.isclose(float(design_weight), 1.0 / probability, rel_tol=1e-6):
            raise ValueError("design_weight must equal 1 / inclusion_probability")

    # A sampler may keep convenience fields beside the registered joint
    # attributes.  They do not belong to the strict unit contract.
    payload.pop("selection_rank", None)
    payload.pop("stratum_id", None)
    return ConditionalValueUnit.model_validate(payload)


class StratumKey(_StrictModel):
    cell_id: str = Field(min_length=1)
    mastery_state: MasteryState
    data_attributes: dict[str, str] = Field(min_length=1)

    @field_validator("data_attributes")
    @classmethod
    def normalize_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        return dict(sorted(value.items()))

    @property
    def canonical_id(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class StratumMass(_StrictModel):
    key: StratumKey
    mass: float = Field(ge=0)
    proportion: float = Field(ge=0, le=1)


class DependencyGroupDesign(_StrictModel):
    """Auditable first-stage sampling unit embedded in a value profile.

    A semantic row is not an independent draw when several rows belong to one
    dependency group.  Keeping the complete group membership, first-order
    inclusion probability, and source-population mass in the profile lets a
    certificate bind the exact sample design instead of seeing only an
    aggregate distribution.
    """

    dependency_group_id: str = Field(min_length=1)
    unit_ids: list[str] = Field(min_length=1)
    inclusion_probability: float = Field(gt=0, le=1)
    population_weight: float = Field(gt=0)

    @field_validator("unit_ids")
    @classmethod
    def validate_unit_ids(cls, value: list[str]) -> list[str]:
        normalized = sorted(str(unit_id).strip() for unit_id in value)
        if any(not unit_id for unit_id in normalized):
            raise ValueError("Dependency-group unit IDs must be non-empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Dependency-group unit IDs must be unique")
        return normalized


class ValueProfile(_StrictModel):
    """Soft joint distribution under fixed benchmark and base-model state."""

    schema_version: Literal["conditional-value-profile-v3"] = "conditional-value-profile-v3"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    population_kind: PopulationKind
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    benchmark_id: str = Field(min_length=1)
    model_profile_id: str = Field(min_length=1)
    model_profile_sha256: str
    registered_cell_ids: list[str] = Field(min_length=1)
    benchmark_cell_weights: dict[str, float] = Field(min_length=1)
    unit_count: int = Field(gt=0)
    dependency_group_count: int = Field(gt=0)
    excluded_unit_count: int = Field(ge=0)
    overridden_unit_count: int = Field(ge=0)
    dependency_groups: list[DependencyGroupDesign] = Field(min_length=1)
    membership_sha256: str
    sampling_design_sha256: str
    unmeasured_cell_ids: list[str]
    unknown_mastery_mass: float = Field(ge=0, le=1)
    total_mass: float = Field(gt=0)
    effective_sample_size: float = Field(gt=0)
    attribute_dimensions: list[str] = Field(min_length=1)
    strata: list[StratumMass] = Field(min_length=1)

    @field_validator(
        "batch_content_sha256",
        "model_profile_sha256",
        "membership_sha256",
        "sampling_design_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_distribution(self):
        if len(self.registered_cell_ids) != len(set(self.registered_cell_ids)):
            raise ValueError("registered_cell_ids must be unique")
        self.registered_cell_ids = sorted(self.registered_cell_ids)
        if set(self.benchmark_cell_weights) != set(self.registered_cell_ids):
            raise ValueError("benchmark_cell_weights keys must exactly match registered_cell_ids")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.benchmark_cell_weights.values()):
            raise ValueError("benchmark_cell_weights must be finite and positive")
        self.benchmark_cell_weights = dict(sorted(self.benchmark_cell_weights.items()))
        group_ids = [group.dependency_group_id for group in self.dependency_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Dependency-group IDs must be unique within a value profile")
        self.dependency_groups = sorted(
            self.dependency_groups,
            key=lambda group: group.dependency_group_id,
        )
        if self.dependency_group_count != len(self.dependency_groups):
            raise ValueError("dependency_group_count must equal the embedded group design count")
        member_ids = [unit_id for group in self.dependency_groups for unit_id in group.unit_ids]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("A unit cannot belong to multiple dependency groups")
        if self.unit_count != len(member_ids):
            raise ValueError("unit_count must equal the embedded dependency-group membership")
        membership_payload = [
            {
                "dependency_group_id": group.dependency_group_id,
                "unit_ids": group.unit_ids,
            }
            for group in self.dependency_groups
        ]
        if self.membership_sha256 != _canonical_digest(membership_payload):
            raise ValueError("membership_sha256 does not match embedded dependency-group membership")
        dependency_group_payload = [group.model_dump(mode="json") for group in self.dependency_groups]
        if self.sampling_design_sha256 != _canonical_digest(dependency_group_payload):
            raise ValueError("sampling_design_sha256 does not match embedded group probabilities")
        if self.population_kind == "full_population" and any(
            not math.isclose(group.inclusion_probability, 1.0) for group in self.dependency_groups
        ):
            raise ValueError("A full-population profile must use group inclusion_probability=1")
        group_contributions = [
            (
                group.population_weight / group.inclusion_probability
                if self.population_kind == "weighted_reconstruction"
                else group.population_weight
            )
            for group in self.dependency_groups
        ]
        expected_total_mass = sum(group_contributions)
        if not math.isclose(expected_total_mass, self.total_mass, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Embedded dependency-group design does not reproduce total_mass")
        expected_ess = expected_total_mass**2 / sum(contribution * contribution for contribution in group_contributions)
        if not math.isclose(expected_ess, self.effective_sample_size, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("effective_sample_size must be calculated from dependency groups")
        if self.effective_sample_size > self.dependency_group_count + _PROBABILITY_TOLERANCE:
            raise ValueError("effective_sample_size cannot exceed included dependency_group_count")
        if len(self.attribute_dimensions) != len(set(self.attribute_dimensions)):
            raise ValueError("attribute_dimensions must be unique")
        expected_dimensions = set(self.attribute_dimensions)
        keys = [stratum.key.canonical_id for stratum in self.strata]
        if len(keys) != len(set(keys)):
            raise ValueError("Value profile strata must have unique keys")
        if any(set(stratum.key.data_attributes) != expected_dimensions for stratum in self.strata):
            raise ValueError("Every stratum must contain exactly the registered attribute dimensions")
        if unknown_cells := {stratum.key.cell_id for stratum in self.strata} - set(self.registered_cell_ids):
            raise ValueError(f"Profile strata reference unregistered benchmark cells: {sorted(unknown_cells)}")
        mass_total = sum(stratum.mass for stratum in self.strata)
        if not math.isclose(mass_total, self.total_mass, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Stratum masses must sum to total_mass")
        proportion_total = sum(stratum.proportion for stratum in self.strata)
        if not math.isclose(proportion_total, 1.0, abs_tol=1e-8):
            raise ValueError("Stratum proportions must sum to 1")
        self.attribute_dimensions = sorted(self.attribute_dimensions)
        if len(self.unmeasured_cell_ids) != len(set(self.unmeasured_cell_ids)):
            raise ValueError("unmeasured_cell_ids must be unique")
        self.unmeasured_cell_ids = sorted(self.unmeasured_cell_ids)
        self.strata = sorted(self.strata, key=lambda stratum: stratum.key.canonical_id)
        return self

    @property
    def distribution(self) -> dict[str, float]:
        return {stratum.key.canonical_id: stratum.proportion for stratum in self.strata}


def value_profile_digest(profile: ValueProfile) -> str:
    return _canonical_digest(profile)


def _parsed_units(units: Sequence[ConditionalValueUnit | Mapping[str, Any]]) -> list[ConditionalValueUnit]:
    return [
        unit if isinstance(unit, ConditionalValueUnit) else ConditionalValueUnit.model_validate(unit) for unit in units
    ]


def build_value_profile(
    units: Sequence[ConditionalValueUnit | Mapping[str, Any]],
    model_profile: ModelMasteryProfile,
    *,
    profile_id: str,
    population_kind: PopulationKind = "full_population",
) -> ValueProfile:
    """Aggregate a soft joint conditional-value profile.

    For a weighted reconstruction the Horvitz-Thompson contribution of unit
    ``i`` is ``population_weight_i / inclusion_probability_i``.  The state
    probability and cell-mapping weight split that contribution without
    changing its total mass.
    """

    parsed = _parsed_units(units)
    if not parsed:
        raise ValueError("At least one conditional-value unit is required")
    unit_ids = [unit.unit_id for unit in parsed]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Conditional-value unit IDs must be unique")
    batch_ids = {unit.batch_id for unit in parsed}
    batch_hashes = {unit.batch_content_sha256 for unit in parsed}
    if len(batch_ids) != 1 or len(batch_hashes) != 1:
        raise ValueError("A value profile must describe one frozen batch")
    attribute_dimensions = {tuple(unit.data_attributes) for unit in parsed}
    if len(attribute_dimensions) != 1:
        raise ValueError("All units must use the same registered data attribute dimensions")
    if population_kind == "full_population" and any(
        not math.isclose(unit.inclusion_probability, 1.0) for unit in parsed
    ):
        raise ValueError("Full-population aggregation requires inclusion_probability=1 for every unit")

    parsed_groups: dict[str, list[ConditionalValueUnit]] = defaultdict(list)
    for unit in parsed:
        parsed_groups[unit.dependency_group_id].append(unit)
    # A dependency group is the primary sampling unit.  If any member is
    # ineligible, the complete group leaves the population frame, matching the
    # probability sampler's atomic exclusion rule.
    eligible_groups = {
        group_id: sorted(group, key=lambda item: item.unit_id)
        for group_id, group in parsed_groups.items()
        if all(unit.eligible for unit in group)
    }
    eligible = [unit for group in eligible_groups.values() for unit in group]
    if not eligible:
        raise ValueError("No eligible units are available for profile aggregation")
    known_cells = model_profile.by_cell
    for unit in eligible:
        if unknown := {mapping.cell_id for mapping in unit.cell_mappings} - set(known_cells):
            raise ValueError(f"Unit {unit.unit_id} references cells absent from the model profile: {sorted(unknown)}")

    dependency_groups = []
    for group_id, group in sorted(eligible_groups.items()):
        probabilities = {unit.inclusion_probability for unit in group}
        if len(probabilities) != 1:
            raise ValueError(f"Dependency group {group_id} must have one shared inclusion_probability")
        dependency_groups.append(
            DependencyGroupDesign(
                dependency_group_id=group_id,
                unit_ids=[unit.unit_id for unit in group],
                inclusion_probability=next(iter(probabilities)),
                population_weight=sum(unit.population_weight for unit in group),
            )
        )

    masses: dict[str, float] = defaultdict(float)
    stratum_keys: dict[str, StratumKey] = {}
    overridden = 0
    unmeasured_effective_cells: set[str] = set()
    for unit in eligible:
        design_mass = unit.population_weight
        if population_kind == "weighted_reconstruction":
            design_mass /= unit.inclusion_probability
        if unit.model_state_overrides:
            overridden += 1
        for mapping in unit.cell_mappings:
            probabilities = (
                unit.model_state_overrides.get(mapping.cell_id) or known_cells[mapping.cell_id].probabilities
            )
            if probabilities.unknown > _PROBABILITY_TOLERANCE:
                unmeasured_effective_cells.add(mapping.cell_id)
            for state, probability in probabilities.as_dict().items():
                if probability <= 0:
                    continue
                key = StratumKey(
                    cell_id=mapping.cell_id,
                    mastery_state=state,
                    data_attributes=unit.data_attributes,
                )
                canonical_id = key.canonical_id
                stratum_keys[canonical_id] = key
                masses[canonical_id] += design_mass * mapping.weight * probability

    total_mass = sum(masses.values())
    if total_mass <= 0:
        raise ValueError("Profile aggregation produced zero mass")
    group_contributions = [
        (
            group.population_weight / group.inclusion_probability
            if population_kind == "weighted_reconstruction"
            else group.population_weight
        )
        for group in dependency_groups
    ]
    denominator = sum(weight * weight for weight in group_contributions)
    effective_sample_size = sum(group_contributions) ** 2 / denominator
    strata = [
        StratumMass(key=stratum_keys[key], mass=mass, proportion=mass / total_mass) for key, mass in masses.items()
    ]
    unmeasured_cell_ids = sorted(unmeasured_effective_cells)
    unknown_mastery_mass = min(
        1.0,
        max(0.0, sum(stratum.proportion for stratum in strata if stratum.key.mastery_state == "unknown")),
    )
    return ValueProfile(
        profile_id=profile_id,
        population_kind=population_kind,
        batch_id=next(iter(batch_ids)),
        batch_content_sha256=next(iter(batch_hashes)),
        benchmark_id=model_profile.benchmark_id,
        model_profile_id=model_profile.profile_id,
        model_profile_sha256=_canonical_digest(model_profile),
        registered_cell_ids=sorted(known_cells),
        benchmark_cell_weights=model_profile.benchmark_cell_weights,
        unit_count=len(eligible),
        dependency_group_count=len(dependency_groups),
        excluded_unit_count=len(parsed) - len(eligible),
        overridden_unit_count=overridden,
        dependency_groups=dependency_groups,
        membership_sha256=_canonical_digest(
            [
                {
                    "dependency_group_id": group.dependency_group_id,
                    "unit_ids": group.unit_ids,
                }
                for group in dependency_groups
            ]
        ),
        sampling_design_sha256=_canonical_digest([group.model_dump(mode="json") for group in dependency_groups]),
        unmeasured_cell_ids=unmeasured_cell_ids,
        unknown_mastery_mass=unknown_mastery_mass,
        total_mass=total_mass,
        effective_sample_size=effective_sample_size,
        attribute_dimensions=list(next(iter(attribute_dimensions))),
        strata=strata,
    )


def reconstruct_value_profile(
    sample_units: Sequence[ConditionalValueUnit | Mapping[str, Any]],
    model_profile: ModelMasteryProfile,
    *,
    profile_id: str,
) -> ValueProfile:
    """Reconstruct a full-batch profile from sampled units using HT weights."""

    return build_value_profile(
        sample_units,
        model_profile,
        profile_id=profile_id,
        population_kind="weighted_reconstruction",
    )


def build_raw_sample_profile(
    sample_units: Sequence[ConditionalValueUnit | Mapping[str, Any]],
    model_profile: ModelMasteryProfile,
    *,
    profile_id: str,
) -> ValueProfile:
    """Describe the observed sample composition before design-weight correction."""

    return build_value_profile(
        sample_units,
        model_profile,
        profile_id=profile_id,
        population_kind="raw_sample",
    )


class StratumDifference(_StrictModel):
    key: StratumKey
    reference_proportion: float = Field(ge=0, le=1)
    candidate_proportion: float = Field(ge=0, le=1)
    absolute_error: float = Field(ge=0, le=1)


class ProfileComparison(_StrictModel):
    schema_version: Literal["conditional-value-comparison-v1"] = "conditional-value-comparison-v1"
    reference_profile_id: str = Field(min_length=1)
    candidate_profile_id: str = Field(min_length=1)
    joint_total_variation: float = Field(ge=0, le=1)
    critical_weighted_total_variation: float = Field(ge=0, le=1)
    cell_marginal_total_variation: float = Field(ge=0, le=1)
    maximum_cell_proportion_error: float = Field(ge=0, le=1)
    mastery_marginal_total_variation: float = Field(ge=0, le=1)
    attribute_marginal_total_variation: dict[str, float]
    maximum_stratum_error: float = Field(ge=0, le=1)
    missing_strata: list[StratumKey]
    critical_missing_strata: list[StratumKey]
    stratum_differences: list[StratumDifference]

    @field_validator("attribute_marginal_total_variation")
    @classmethod
    def validate_attribute_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not 0 <= metric <= 1 for metric in value.values()):
            raise ValueError("Attribute marginal total variation values must lie in [0, 1]")
        return dict(sorted(value.items()))


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    # Sort the support before summation.  Iterating a set makes the final
    # floating-point bit pattern depend on Python's per-process hash seed,
    # which can make a serialized certificate fail verification in a later
    # process even though its inputs are unchanged.
    support = sorted(set(left) | set(right))
    distance = 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in support)
    return min(1.0, max(0.0, distance))


def _marginal(profile: ValueProfile, dimension: str) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for stratum in profile.strata:
        if dimension == "cell_id":
            value = stratum.key.cell_id
        elif dimension == "mastery_state":
            value = stratum.key.mastery_state
        else:
            value = stratum.key.data_attributes[dimension]
        result[value] += stratum.proportion
    return dict(result)


def _cell_weighted_distribution(profile: ValueProfile, weights: Mapping[str, float]) -> dict[str, float]:
    raw = {
        stratum.key.canonical_id: stratum.proportion * weights.get(stratum.key.cell_id, 1.0)
        for stratum in profile.strata
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def compare_value_profiles(
    reference: ValueProfile,
    candidate: ValueProfile,
    *,
    critical_cell_weights: Mapping[str, float] | None = None,
) -> ProfileComparison:
    """Compare joint and marginal fidelity for two profiles in one context."""

    context_fields = (
        "batch_id",
        "batch_content_sha256",
        "benchmark_id",
        "model_profile_id",
        "model_profile_sha256",
        "registered_cell_ids",
        "benchmark_cell_weights",
        "attribute_dimensions",
    )
    mismatched = [field for field in context_fields if getattr(reference, field) != getattr(candidate, field)]
    if mismatched:
        raise ValueError(f"Profiles have incompatible conditioning context: {mismatched}")
    weights = dict(critical_cell_weights or reference.benchmark_cell_weights)
    if any(not cell_id.strip() or not math.isfinite(weight) or weight <= 0 for cell_id, weight in weights.items()):
        raise ValueError("Critical cell weights must have non-empty IDs and finite positive values")
    if unknown_weight_cells := set(weights) - set(reference.registered_cell_ids):
        raise ValueError(
            "Critical cell weights reference cells absent from the reference profile's "
            "registered benchmark cells: "
            f"{sorted(unknown_weight_cells)}"
        )

    reference_by_key = {stratum.key.canonical_id: stratum for stratum in reference.strata}
    candidate_by_key = {stratum.key.canonical_id: stratum for stratum in candidate.strata}
    differences = []
    for key in sorted(set(reference_by_key) | set(candidate_by_key)):
        reference_stratum = reference_by_key.get(key)
        candidate_stratum = candidate_by_key.get(key)
        reference_proportion = reference_stratum.proportion if reference_stratum else 0.0
        candidate_proportion = candidate_stratum.proportion if candidate_stratum else 0.0
        stratum_key = (reference_stratum or candidate_stratum).key
        differences.append(
            StratumDifference(
                key=stratum_key,
                reference_proportion=reference_proportion,
                candidate_proportion=candidate_proportion,
                absolute_error=abs(reference_proportion - candidate_proportion),
            )
        )

    attribute_tv = {
        dimension: _total_variation(_marginal(reference, dimension), _marginal(candidate, dimension))
        for dimension in reference.attribute_dimensions
    }
    cell_reference = _marginal(reference, "cell_id")
    cell_candidate = _marginal(candidate, "cell_id")
    missing_strata = [
        difference.key
        for difference in differences
        if difference.reference_proportion > 0 and difference.candidate_proportion == 0
    ]
    critical_cells = set(weights) if weights else set(cell_reference)
    return ProfileComparison(
        reference_profile_id=reference.profile_id,
        candidate_profile_id=candidate.profile_id,
        joint_total_variation=_total_variation(reference.distribution, candidate.distribution),
        critical_weighted_total_variation=_total_variation(
            _cell_weighted_distribution(reference, weights),
            _cell_weighted_distribution(candidate, weights),
        ),
        cell_marginal_total_variation=_total_variation(cell_reference, cell_candidate),
        maximum_cell_proportion_error=max(
            (
                abs(cell_reference.get(cell_id, 0.0) - cell_candidate.get(cell_id, 0.0))
                for cell_id in set(cell_reference) | set(cell_candidate)
            ),
            default=0.0,
        ),
        mastery_marginal_total_variation=_total_variation(
            _marginal(reference, "mastery_state"), _marginal(candidate, "mastery_state")
        ),
        attribute_marginal_total_variation=attribute_tv,
        maximum_stratum_error=max((difference.absolute_error for difference in differences), default=0.0),
        missing_strata=missing_strata,
        critical_missing_strata=[key for key in missing_strata if key.cell_id in critical_cells],
        stratum_differences=differences,
    )


class FidelityThresholds(_StrictModel):
    """Pre-registered acceptance bounds for profile reconstruction."""

    maximum_joint_total_variation: float = Field(ge=0, le=1)
    maximum_critical_weighted_total_variation: float = Field(ge=0, le=1)
    maximum_cell_marginal_total_variation: float = Field(ge=0, le=1)
    maximum_cell_proportion_error: float = Field(ge=0, le=1)
    maximum_mastery_marginal_total_variation: float = Field(ge=0, le=1)
    maximum_attribute_marginal_total_variation: float = Field(ge=0, le=1)
    maximum_ht_total_relative_error: float = Field(default=0.1, ge=0, le=1)
    minimum_effective_sample_size: float = Field(gt=0)


class ProfileFidelityCertificate(_StrictModel):
    schema_version: Literal["conditional-value-fidelity-certificate-v2"] = "conditional-value-fidelity-certificate-v2"
    certificate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    reference_profile_id: str = Field(min_length=1)
    reference_profile_sha256: str
    raw_sample_profile_id: str = Field(min_length=1)
    raw_sample_profile_sha256: str
    reconstructed_profile_id: str = Field(min_length=1)
    reconstructed_profile_sha256: str
    reference_membership_sha256: str
    sample_membership_sha256: str
    sample_design_sha256: str
    sampling_manifest_sha256: str | None = None
    design_provenance_verified: bool = False
    reference_dependency_group_count: int = Field(gt=0)
    sample_dependency_group_count: int = Field(gt=0)
    ht_total_relative_error: float = Field(ge=0)
    thresholds: FidelityThresholds
    critical_cell_weights: dict[str, float]
    raw_sample_comparison: ProfileComparison
    weighted_comparison: ProfileComparison
    required_effective_sample_size: float = Field(gt=0)
    checks: dict[str, bool]
    distribution_fidelity_passed: bool
    passed: bool
    complete_three_dimensional_claim: bool
    status: Literal["passed", "failed", "partial_pending_model_measurement"]
    interpretation: Literal["design_profile_fidelity_not_training_utility"] = (
        "design_profile_fidelity_not_training_utility"
    )

    @field_validator(
        "reference_profile_sha256",
        "raw_sample_profile_sha256",
        "reconstructed_profile_sha256",
        "reference_membership_sha256",
        "sample_membership_sha256",
        "sample_design_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("sampling_manifest_sha256")
    @classmethod
    def validate_optional_manifest_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, "sampling_manifest_sha256")

    @model_validator(mode="after")
    def validate_decision(self):
        if not self.checks:
            raise ValueError("A fidelity certificate must contain acceptance checks")
        if self.design_provenance_verified != (self.sampling_manifest_sha256 is not None):
            raise ValueError("design_provenance_verified requires exactly one bound sampling_manifest_sha256")
        if self.design_provenance_verified and self.checks.get("design_provenance") is not True:
            raise ValueError("A provenance-bound certificate requires a passing design_provenance check")
        if not self.design_provenance_verified and "design_provenance" in self.checks:
            raise ValueError("An unbound certificate cannot contain a design_provenance check")
        if self.sample_dependency_group_count > self.reference_dependency_group_count:
            raise ValueError("Sample dependency-group count cannot exceed the population frame")
        distribution_checks = {
            name: passed for name, passed in self.checks.items() if name != "model_measurement_complete"
        }
        if self.distribution_fidelity_passed != all(distribution_checks.values()):
            raise ValueError("distribution_fidelity_passed must equal the conjunction of distribution checks")
        if self.passed != (self.distribution_fidelity_passed and self.checks.get("model_measurement_complete", False)):
            raise ValueError("Certificate passed flag must equal the conjunction of its checks")
        if any(
            comparison.reference_profile_id != self.reference_profile_id
            for comparison in (self.raw_sample_comparison, self.weighted_comparison)
        ):
            raise ValueError("Certificate reference profile does not match comparisons")
        if self.raw_sample_profile_id != self.raw_sample_comparison.candidate_profile_id:
            raise ValueError("Certificate raw sample profile does not match comparison")
        if self.reconstructed_profile_id != self.weighted_comparison.candidate_profile_id:
            raise ValueError("Certificate reconstructed profile does not match comparison")
        expected_status = (
            "partial_pending_model_measurement"
            if not self.complete_three_dimensional_claim
            else ("passed" if self.passed else "failed")
        )
        if self.status != expected_status:
            raise ValueError("Certificate status is inconsistent with claim completeness and checks")
        return self


def _validate_certificate_sample_design(
    reference: ValueProfile,
    raw_sample: ValueProfile,
    reconstruction: ValueProfile,
) -> None:
    """Require the two sample views to describe one atomic PSU design.

    The reference profile supplies the population-frame membership and source
    mass for every dependency group.  Raw and HT profiles must contain the
    same complete selected groups and identical first-order probabilities.
    This turns membership and ``pi`` into certificate inputs instead of
    leaving them as unaudited intermediates.
    """

    if raw_sample.membership_sha256 != reconstruction.membership_sha256:
        raise ValueError("Raw and reconstructed profiles use different sample membership")
    if raw_sample.sampling_design_sha256 != reconstruction.sampling_design_sha256:
        raise ValueError("Raw and reconstructed profiles use different inclusion probabilities")
    if raw_sample.dependency_groups != reconstruction.dependency_groups:
        raise ValueError("Raw and reconstructed profiles use different dependency-group designs")

    reference_groups = {group.dependency_group_id: group for group in reference.dependency_groups}
    for sample_group in raw_sample.dependency_groups:
        reference_group = reference_groups.get(sample_group.dependency_group_id)
        if reference_group is None:
            raise ValueError(
                f"Sample dependency group {sample_group.dependency_group_id} is absent from the population frame"
            )
        if sample_group.unit_ids != reference_group.unit_ids:
            raise ValueError(
                f"Sample dependency group {sample_group.dependency_group_id} is not an atomic population group"
            )
        if not math.isclose(
            sample_group.population_weight,
            reference_group.population_weight,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Sample dependency group {sample_group.dependency_group_id} changed population weight")


def build_fidelity_certificate(
    reference: ValueProfile,
    raw_sample: ValueProfile,
    reconstruction: ValueProfile,
    *,
    certificate_id: str,
    thresholds: FidelityThresholds,
    critical_cell_weights: Mapping[str, float] | None = None,
    sampling_manifest_sha256: str | None = None,
    design_provenance_verified: bool = False,
) -> ProfileFidelityCertificate:
    """Issue a design-fidelity certificate from pre-registered thresholds."""

    if reference.population_kind != "full_population":
        raise ValueError("Certificate reference must be a full_population profile")
    if raw_sample.population_kind != "raw_sample":
        raise ValueError("Certificate raw sample must be a raw_sample profile")
    if reconstruction.population_kind != "weighted_reconstruction":
        raise ValueError("Certificate candidate must be a weighted_reconstruction profile")
    _validate_certificate_sample_design(reference, raw_sample, reconstruction)
    # When no separate clinical-risk modifiers are supplied, the registered
    # Benchmark blueprint weights are the task-sensitive fidelity weights.
    weights = dict(sorted((critical_cell_weights or reference.benchmark_cell_weights).items()))
    raw_comparison = compare_value_profiles(
        reference,
        raw_sample,
        critical_cell_weights=weights,
    )
    comparison = compare_value_profiles(
        reference,
        reconstruction,
        critical_cell_weights=weights,
    )
    maximum_attribute_tv = max(comparison.attribute_marginal_total_variation.values(), default=0.0)
    # A fixed ESS floor cannot exceed the information available in the full
    # population itself.  This matters for deliberately tiny diagnostic
    # batches: requiring ESS=10 from a four-unit population is impossible even
    # under a census.  The registered floor is therefore capped by the full
    # population's own Kish ESS, and the realized requirement is recorded in
    # the certificate.
    required_effective_sample_size = min(
        thresholds.minimum_effective_sample_size,
        reference.effective_sample_size,
    )
    ht_total_relative_error = abs(reconstruction.total_mass - reference.total_mass) / reference.total_mass
    checks = {
        "joint_total_variation": (comparison.joint_total_variation <= thresholds.maximum_joint_total_variation),
        "critical_weighted_total_variation": (
            comparison.critical_weighted_total_variation <= thresholds.maximum_critical_weighted_total_variation
        ),
        "cell_marginal_total_variation": (
            comparison.cell_marginal_total_variation <= thresholds.maximum_cell_marginal_total_variation
        ),
        "maximum_cell_proportion_error": (
            comparison.maximum_cell_proportion_error <= thresholds.maximum_cell_proportion_error
        ),
        "mastery_marginal_total_variation": (
            comparison.mastery_marginal_total_variation <= thresholds.maximum_mastery_marginal_total_variation
        ),
        "attribute_marginal_total_variation": (
            maximum_attribute_tv <= thresholds.maximum_attribute_marginal_total_variation
        ),
        "ht_total_relative_error": (ht_total_relative_error <= thresholds.maximum_ht_total_relative_error),
        "effective_sample_size": reconstruction.effective_sample_size >= required_effective_sample_size,
    }
    if sampling_manifest_sha256 is not None:
        checks["design_provenance"] = design_provenance_verified
    # Exact joint-stratum absence remains a diagnostic in
    # ``critical_missing_strata`` rather than an unconditional pass/fail gate.
    # A practical sample may preserve a critical capability and knowledge type
    # while omitting one source/status micro-cell.  Hard medical requirements
    # belong in the probability design's preregistered rare-stratum quotas;
    # joint and critical-weighted TV bound the remaining distribution error.
    model_measurement_complete = (
        not reference.unmeasured_cell_ids and reference.unknown_mastery_mass <= _PROBABILITY_TOLERANCE
    )
    checks["model_measurement_complete"] = model_measurement_complete
    distribution_fidelity_passed = all(
        passed for name, passed in checks.items() if name != "model_measurement_complete"
    )
    passed = distribution_fidelity_passed and model_measurement_complete
    return ProfileFidelityCertificate(
        certificate_id=certificate_id,
        reference_profile_id=reference.profile_id,
        reference_profile_sha256=value_profile_digest(reference),
        raw_sample_profile_id=raw_sample.profile_id,
        raw_sample_profile_sha256=value_profile_digest(raw_sample),
        reconstructed_profile_id=reconstruction.profile_id,
        reconstructed_profile_sha256=value_profile_digest(reconstruction),
        reference_membership_sha256=reference.membership_sha256,
        sample_membership_sha256=raw_sample.membership_sha256,
        sample_design_sha256=raw_sample.sampling_design_sha256,
        sampling_manifest_sha256=sampling_manifest_sha256,
        design_provenance_verified=design_provenance_verified,
        reference_dependency_group_count=reference.dependency_group_count,
        sample_dependency_group_count=raw_sample.dependency_group_count,
        ht_total_relative_error=ht_total_relative_error,
        thresholds=thresholds,
        critical_cell_weights=weights,
        raw_sample_comparison=raw_comparison,
        weighted_comparison=comparison,
        required_effective_sample_size=required_effective_sample_size,
        checks=checks,
        distribution_fidelity_passed=distribution_fidelity_passed,
        passed=passed,
        complete_three_dimensional_claim=model_measurement_complete,
        status=(
            ("passed" if passed else "failed") if model_measurement_complete else "partial_pending_model_measurement"
        ),
    )


def verify_fidelity_certificate(
    certificate: ProfileFidelityCertificate,
    reference: ValueProfile,
    raw_sample: ValueProfile,
    reconstruction: ValueProfile,
    *,
    sampling_manifest_sha256: str | None = None,
) -> None:
    """Raise when a certificate is detached from or inconsistent with inputs."""

    if certificate.reference_profile_sha256 != value_profile_digest(reference):
        raise ValueError("Certificate reference profile digest does not match")
    if certificate.raw_sample_profile_sha256 != value_profile_digest(raw_sample):
        raise ValueError("Certificate raw sample profile digest does not match")
    if certificate.reconstructed_profile_sha256 != value_profile_digest(reconstruction):
        raise ValueError("Certificate reconstructed profile digest does not match")
    if certificate.design_provenance_verified:
        if sampling_manifest_sha256 is None:
            raise ValueError("A provenance-bound certificate requires its verified sampling manifest")
        if certificate.sampling_manifest_sha256 != sampling_manifest_sha256:
            raise ValueError("Certificate sampling manifest digest does not match")
    elif sampling_manifest_sha256 is not None:
        raise ValueError("Certificate is not bound to a sampling manifest")
    expected = build_fidelity_certificate(
        reference,
        raw_sample,
        reconstruction,
        certificate_id=certificate.certificate_id,
        thresholds=certificate.thresholds,
        critical_cell_weights=certificate.critical_cell_weights,
        sampling_manifest_sha256=certificate.sampling_manifest_sha256,
        design_provenance_verified=certificate.design_provenance_verified,
    )
    if certificate != expected:
        raise ValueError("Certificate contents do not match recomputed profile comparisons")


def fidelity_certificate_digest(certificate: ProfileFidelityCertificate) -> str:
    return _canonical_digest(certificate)
