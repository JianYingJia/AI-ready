"""Sparse rule-level Base Model profiling and pre-training manipulation checks.

Capability priors assign the same model state to every rule in one capability.
This module refines only source units supported by independent probe evidence,
keeps every other unit-cell explicitly unknown, and tests whether the refined
signal actually changes probability-sample membership before A8 training is
interpreted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.valuation.conditional_value import (
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeAggregationConfig,
    ScoredProbe,
    ScoreInterval,
)
from ai_ready.valuation.model_probe import FourChoiceProbe
from ai_ready.valuation.unit_mastery import (
    UnitCellMasteryObservation,
    UnitCellMasteryProfile,
    build_unit_cell_mastery_profile,
    validate_unit_cell_mastery_profile,
)
from ai_ready.valuation.value_sampling import (
    ValueSamplingConfig,
    ValueSamplingUnit,
    sample_value_distribution,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _canonical_digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CITATION_PATTERN = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_evidence_text(value: str) -> str:
    """Normalize public evidence and semantic text for strict containment.

    The normalization removes citation-number formatting and punctuation but
    does not use fuzzy matching. A binding is emitted only when the complete
    normalized evidence span occurs in the normalized semantic unit.
    """

    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = _CITATION_PATTERN.sub(" ", normalized)
    normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split())


class ProbeUnitBinding(_StrictModel):
    """Auditable exact binding from one independent probe to one source unit."""

    probe_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    match_method: Literal["normalized_exact_evidence"] = "normalized_exact_evidence"


def bind_probes_to_semantic_units(
    probes: Sequence[FourChoiceProbe | Mapping[str, Any]],
    semantic_units: Sequence[Mapping[str, Any]],
    population_units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    *,
    require_all: bool = True,
) -> list[ProbeUnitBinding]:
    """Bind probes to frozen units containing exact evidence in the same cell.

    ``require_all=False`` is reserved for an explicitly labelled screening
    profile.  It keeps the frozen population unchanged when an upstream
    capability label omits the probe's cell; callers must report exclusions.
    """

    parsed_probes = [
        probe if isinstance(probe, FourChoiceProbe) else FourChoiceProbe.model_validate(probe) for probe in probes
    ]
    parsed_population = [
        unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit)
        for unit in population_units
    ]
    population = {unit.unit_id: unit for unit in parsed_population}
    semantic_by_id = {}
    for row in semantic_units:
        unit_id = str(row.get("semantic_unit_id") or "")
        content = str(row.get("content") or "")
        batch_id = str(row.get("batch_id") or "")
        if not unit_id or not content or not batch_id:
            raise ValueError("Semantic rows require semantic_unit_id, batch_id, and content")
        if unit_id in semantic_by_id:
            raise ValueError(f"Duplicate semantic_unit_id: {unit_id}")
        semantic_by_id[unit_id] = {**dict(row), "_normalized_content": normalize_evidence_text(content)}
    if missing := set(population) - set(semantic_by_id):
        raise ValueError(f"Frozen population references missing semantic units: {sorted(missing)[:5]}")

    bindings = []
    bound_probe_ids = set()
    for probe in sorted(parsed_probes, key=lambda item: item.probe_id):
        evidence = normalize_evidence_text(probe.source_evidence)
        if not evidence:
            raise ValueError(f"Probe {probe.probe_id} has empty normalized evidence")
        for unit_id, unit in sorted(population.items()):
            if probe.cell_id not in unit.capability_ids:
                continue
            semantic = semantic_by_id[unit_id]
            if evidence not in semantic["_normalized_content"]:
                continue
            bindings.append(
                ProbeUnitBinding(
                    probe_id=probe.probe_id,
                    item_id=probe.item_id,
                    unit_id=unit_id,
                    batch_id=unit.batch_id,
                    cell_id=probe.cell_id,
                    evidence_sha256=probe.source_evidence_sha256,
                    semantic_content_sha256=str(
                        semantic.get("content_sha256")
                        or hashlib.sha256(str(semantic["content"]).encode("utf-8")).hexdigest()
                    ),
                )
            )
            bound_probe_ids.add(probe.probe_id)
    if require_all and (missing_probes := {probe.probe_id for probe in parsed_probes} - bound_probe_ids):
        raise ValueError(f"Independent probes have no exact semantic-unit binding: {sorted(missing_probes)}")
    return sorted(bindings, key=lambda row: (row.unit_id, row.cell_id, row.probe_id))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def build_sparse_unit_observations(
    population_units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    bindings: Sequence[ProbeUnitBinding | Mapping[str, Any]],
    scored_probes: Sequence[ScoredProbe | Mapping[str, Any]],
    config: ProbeAggregationConfig,
) -> list[UnitCellMasteryObservation]:
    """Aggregate bound probe items and emit unknown rows for every unmeasured pair."""

    units = [
        unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit)
        for unit in population_units
    ]
    groups: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in units:
        groups[unit.dependency_group_id].append(unit)
    eligible_unit_ids = {
        unit.unit_id for group in groups.values() if all(member.eligible for member in group) for unit in group
    }
    eligible_units = [unit for unit in units if unit.unit_id in eligible_unit_ids]
    parsed_bindings = [
        row if isinstance(row, ProbeUnitBinding) else ProbeUnitBinding.model_validate(row) for row in bindings
    ]
    scores = [row if isinstance(row, ScoredProbe) else ScoredProbe.model_validate(row) for row in scored_probes]
    score_by_item: dict[tuple[str, str], list[float]] = defaultdict(list)
    score_ids_by_item: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in scores:
        score_by_item[(row.item_id, row.cell_id)].append(row.score)
        score_ids_by_item[(row.item_id, row.cell_id)].append(row.probe_id)
    bound_items: dict[tuple[str, str], set[str]] = defaultdict(set)
    for binding in parsed_bindings:
        if binding.unit_id not in eligible_unit_ids:
            continue
        if (binding.item_id, binding.cell_id) not in score_by_item:
            raise ValueError(f"Binding {binding.probe_id} has no scored probe rows")
        bound_items[(binding.unit_id, binding.cell_id)].add(binding.item_id)

    observations = []
    for unit in sorted(eligible_units, key=lambda item: item.unit_id):
        for cell_id in unit.capability_ids:
            item_ids = sorted(bound_items.get((unit.unit_id, cell_id), set()))
            if len(item_ids) < config.minimum_probe_count:
                observations.append(
                    UnitCellMasteryObservation(
                        unit_id=unit.unit_id,
                        cell_id=cell_id,
                        probe_count=0,
                        probabilities=MasteryProbabilities.one_hot("unknown"),
                        stratum_state="unknown",
                    )
                )
                continue
            item_scores = [
                sum(score_by_item[(item_id, cell_id)]) / len(score_by_item[(item_id, cell_id)]) for item_id in item_ids
            ]
            seed_material = f"{config.seed}\0{unit.unit_id}\0{cell_id}".encode("utf-8")
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
            replicates = [
                sum(item_scores[rng.randrange(len(item_scores))] for _ in item_scores) / len(item_scores)
                for _ in range(config.bootstrap_iterations)
            ]
            counts = {
                "mastered": sum(value >= config.tau_high for value in replicates),
                "uncertain": sum(config.tau_low <= value < config.tau_high for value in replicates),
                "unmastered": sum(value < config.tau_low for value in replicates),
                "unknown": 0,
            }
            alpha = 1 - config.confidence_level
            interval = ScoreInterval(
                lower=_percentile(replicates, alpha / 2),
                upper=_percentile(replicates, 1 - alpha / 2),
                confidence_level=config.confidence_level,
            )
            if interval.lower >= config.tau_high:
                hard_state = "mastered"
            elif interval.upper < config.tau_low:
                hard_state = "unmastered"
            else:
                hard_state = "uncertain"
            observations.append(
                UnitCellMasteryObservation(
                    unit_id=unit.unit_id,
                    cell_id=cell_id,
                    point_score=sum(item_scores) / len(item_scores),
                    score_interval=interval,
                    probe_count=len(item_ids),
                    probe_ids=item_ids,
                    probabilities=MasteryProbabilities(
                        **{state: count / config.bootstrap_iterations for state, count in counts.items()}
                    ),
                    stratum_state=hard_state,
                )
            )
    return observations


def build_sparse_unit_profile(
    *,
    profile_id: str,
    probe_set_id: str,
    parent: ModelMasteryProfile,
    population_units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    bindings: Sequence[ProbeUnitBinding | Mapping[str, Any]],
    scored_probes: Sequence[ScoredProbe | Mapping[str, Any]],
    aggregation_config: ProbeAggregationConfig,
) -> tuple[UnitCellMasteryProfile, list[UnitCellMasteryObservation], dict[str, Any]]:
    """Build a partial unit-cell profile and its source-binding audit summary."""

    observations = build_sparse_unit_observations(population_units, bindings, scored_probes, aggregation_config)
    scoring_protocol_sha256 = _canonical_digest(
        {
            "schema_version": "sparse-unit-probe-scoring-v1",
            "aggregation_config": aggregation_config.model_dump(mode="json"),
            "bindings": [
                (row if isinstance(row, ProbeUnitBinding) else ProbeUnitBinding.model_validate(row)).model_dump(
                    mode="json"
                )
                for row in bindings
            ],
            "scored_probes": [
                (row if isinstance(row, ScoredProbe) else ScoredProbe.model_validate(row)).model_dump(mode="json")
                for row in scored_probes
            ],
        }
    )
    profile = build_unit_cell_mastery_profile(
        profile_id=profile_id,
        probe_set_id=probe_set_id,
        scoring_protocol_sha256=scoring_protocol_sha256,
        parent=parent,
        units=population_units,
        observations=observations,
    )
    measured = [row for row in observations if row.probe_count > 0]
    by_capability: dict[str, list[UnitCellMasteryObservation]] = defaultdict(list)
    for row in measured:
        by_capability[row.cell_id].append(row)
    capability_summary = {}
    for cell_id, rows in sorted(by_capability.items()):
        points = [float(row.point_score) for row in rows if row.point_score is not None]
        capability_summary[cell_id] = {
            "measured_unit_cell_count": len(rows),
            "point_score_mean": statistics.fmean(points),
            "point_score_variance": statistics.pvariance(points) if len(points) > 1 else 0.0,
            "observed_states": sorted({row.stratum_state for row in rows}),
        }
    manifest = {
        "schema_version": "sparse-unit-mastery-build-v1",
        "profile_id": profile.profile_id,
        "measurement_granularity": "sparse_unit_cell",
        "population_unit_cell_count": len(observations),
        "measured_unit_cell_count": len(measured),
        "unknown_unit_cell_count": len(observations) - len(measured),
        "measured_fraction": len(measured) / len(observations),
        "binding_count": len(bindings),
        "single_item_measurement_count": sum(row.probe_count == 1 for row in measured),
        "state_counts": {
            state: sum(row.stratum_state == state for row in measured)
            for state in ("mastered", "uncertain", "unmastered")
        },
        "capability_summary": capability_summary,
        "claim_status": "screening_partial_unit_measurement",
        "benchmark_test_items_accessed": False,
        "limitations": [
            "Unbound unit-cell pairs remain explicitly unknown.",
            "Single-item rule measurements are screening signals, not confirmatory mastery estimates.",
            "Repeated source content may inherit one independently measured rule result across batches.",
        ],
    }
    return profile, observations, manifest


class PriorManipulationConfig(_StrictModel):
    """Frozen generation-stage grid and screening identifiability thresholds."""

    sampling_fractions: list[float] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    stratum_attributes: list[str] = Field(default_factory=list)
    allow_target_expansion: bool = True
    minimum_capabilities_with_variance: int = Field(default=2, ge=1)
    maximum_unchanged_jaccard: float = Field(default=0.95, ge=0, le=1)
    minimum_changed_trial_fraction: float = Field(default=0.5, ge=0, le=1)

    @field_validator("sampling_fractions")
    @classmethod
    def validate_fractions(cls, value: list[float]) -> list[float]:
        parsed = sorted(float(item) for item in value)
        if any(item <= 0 or item > 1 for item in parsed) or len(parsed) != len(set(parsed)):
            raise ValueError("sampling_fractions must be unique and lie in (0, 1]")
        return parsed

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        parsed = sorted(value)
        if len(parsed) != len(set(parsed)):
            raise ValueError("seeds must be unique")
        return parsed

    @model_validator(mode="after")
    def validate_attributes(self):
        normalized = sorted(item.strip() for item in self.stratum_attributes)
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("stratum_attributes must be non-empty and unique")
        self.stratum_attributes = normalized
        return self


class ProbeHoldoutConfig(_StrictModel):
    """Deterministic probe holdout split and descriptive screening thresholds."""

    split_mode: Literal["within_pair_items", "disjoint_unit_cells"] = "within_pair_items"
    seed: int = 20260929
    minimum_distinct_items_per_pair: int = Field(default=2, ge=2)
    minimum_positive_advantage_fraction: float = Field(default=0.5, ge=0, le=1)
    minimum_median_advantage: float = Field(default=0.0, ge=0)


def split_bindings_for_probe_holdout(
    bindings: Sequence[ProbeUnitBinding | Mapping[str, Any]],
    config: ProbeHoldoutConfig,
) -> tuple[list[ProbeUnitBinding], list[ProbeUnitBinding], dict[str, Any]]:
    """Split distinct items within a unit-cell into design and audit panels."""

    parsed = [row if isinstance(row, ProbeUnitBinding) else ProbeUnitBinding.model_validate(row) for row in bindings]
    by_pair: dict[tuple[str, str], list[ProbeUnitBinding]] = defaultdict(list)
    for row in parsed:
        by_pair[(row.unit_id, row.cell_id)].append(row)
    design: list[ProbeUnitBinding] = []
    audit: list[ProbeUnitBinding] = []
    excluded_pairs = []
    pair_records = []
    if config.split_mode == "disjoint_unit_cells":
        strata: dict[tuple[str, str], list[tuple[tuple[str, str], list[ProbeUnitBinding]]]] = defaultdict(list)
        for pair, rows in by_pair.items():
            strata[(rows[0].batch_id, pair[1])].append((pair, rows))
        for stratum, pairs in sorted(strata.items()):
            ordered_pairs = sorted(
                pairs,
                key=lambda item: hashlib.sha256(f"{config.seed}\0{item[0][0]}\0{item[0][1]}".encode()).hexdigest(),
            )
            for index, ((unit_id, cell_id), rows) in enumerate(ordered_pairs):
                panel = "design" if index % 2 == 0 else "audit"
                if len(ordered_pairs) == 1:
                    digest = hashlib.sha256(f"{config.seed}\0{unit_id}\0{cell_id}\0singleton".encode()).digest()
                    panel = "design" if digest[0] % 2 == 0 else "audit"
                (design if panel == "design" else audit).extend(rows)
                pair_records.append(
                    {
                        "unit_id": unit_id,
                        "cell_id": cell_id,
                        "batch_id": stratum[0],
                        "panel": panel,
                        "item_ids": sorted({row.item_id for row in rows}),
                    }
                )
        manifest = {
            "schema_version": "unit-probe-holdout-split-v1",
            "split_mode": config.split_mode,
            "seed": config.seed,
            "eligible_pair_count": len(pair_records),
            "excluded_pair_count": 0,
            "design_pair_count": sum(row["panel"] == "design" for row in pair_records),
            "audit_pair_count": sum(row["panel"] == "audit" for row in pair_records),
            "design_binding_count": len(design),
            "audit_binding_count": len(audit),
            "pair_records": pair_records,
            "excluded_pairs": [],
        }
        return (
            sorted(design, key=lambda row: (row.unit_id, row.cell_id, row.probe_id)),
            sorted(audit, key=lambda row: (row.unit_id, row.cell_id, row.probe_id)),
            manifest,
        )
    for (unit_id, cell_id), rows in sorted(by_pair.items()):
        item_ids = sorted({row.item_id for row in rows})
        if len(item_ids) < config.minimum_distinct_items_per_pair:
            excluded_pairs.append({"unit_id": unit_id, "cell_id": cell_id, "distinct_item_count": len(item_ids)})
            continue
        ordered = sorted(
            item_ids,
            key=lambda item_id: hashlib.sha256(f"{config.seed}\0{unit_id}\0{cell_id}\0{item_id}".encode()).hexdigest(),
        )
        design_count = max(1, len(ordered) // 2)
        design_items = set(ordered[:design_count])
        audit_items = set(ordered[design_count:])
        if not audit_items:
            raise AssertionError("Holdout split must retain at least one audit item")
        design.extend(row for row in rows if row.item_id in design_items)
        audit.extend(row for row in rows if row.item_id in audit_items)
        pair_records.append(
            {
                "unit_id": unit_id,
                "cell_id": cell_id,
                "design_item_ids": sorted(design_items),
                "audit_item_ids": sorted(audit_items),
            }
        )
    manifest = {
        "schema_version": "unit-probe-holdout-split-v1",
        "split_mode": config.split_mode,
        "seed": config.seed,
        "minimum_distinct_items_per_pair": config.minimum_distinct_items_per_pair,
        "eligible_pair_count": len(pair_records),
        "excluded_pair_count": len(excluded_pairs),
        "design_binding_count": len(design),
        "audit_binding_count": len(audit),
        "pair_records": pair_records,
        "excluded_pairs": excluded_pairs,
    }
    return (
        sorted(design, key=lambda row: (row.unit_id, row.cell_id, row.probe_id)),
        sorted(audit, key=lambda row: (row.unit_id, row.cell_id, row.probe_id)),
        manifest,
    )


def _annotate_units(
    units: Sequence[ValueSamplingUnit],
    parent: ModelMasteryProfile,
    unit_profile: UnitCellMasteryProfile | None,
) -> list[ValueSamplingUnit]:
    parent_by_cell = parent.by_cell
    overrides = unit_profile.by_unit_cell if unit_profile else None
    annotated = []
    for unit in units:
        if overrides is None:
            states = {cell: parent_by_cell[cell].stratum_state for cell in unit.capability_ids}
            probabilities = {
                cell: parent_by_cell[cell].probabilities.model_dump(mode="json") for cell in unit.capability_ids
            }
            granularity = "capability_prior"
        else:
            rows = {cell: overrides.get((unit.unit_id, cell)) for cell in unit.capability_ids}
            states = {cell: row.stratum_state if row else "unknown" for cell, row in rows.items()}
            probabilities = {
                cell: (
                    row.probabilities.model_dump(mode="json")
                    if row
                    else MasteryProbabilities.one_hot("unknown").model_dump(mode="json")
                )
                for cell, row in rows.items()
            }
            granularity = "unit_cell"
        annotated.append(
            unit.model_copy(
                update={
                    "model_states": states,
                    "model_state_probabilities": probabilities,
                    "model_state_granularity": granularity,
                }
            )
        )
    return [ValueSamplingUnit.model_validate(unit.model_dump(mode="json")) for unit in annotated]


def _true_joint_state_tv(
    truth_units: Sequence[ValueSamplingUnit],
    sample_manifest: Mapping[str, Any],
    *,
    measured_only: bool,
) -> float | None:
    selected_probability = {}
    for group in sample_manifest["group_strata"]:
        for group_id in group["selected_group_ids"]:
            selected_probability[group_id] = float(group["inclusion_probability"])
    population: dict[tuple[str, str], float] = defaultdict(float)
    reconstruction: dict[tuple[str, str], float] = defaultdict(float)
    for unit in truth_units:
        share = unit.population_weight / len(unit.capability_ids)
        for cell_id in unit.capability_ids:
            state = unit.model_states[cell_id]
            if measured_only and state == "unknown":
                continue
            key = (cell_id, state)
            population[key] += share
            if unit.dependency_group_id in selected_probability:
                reconstruction[key] += share / selected_probability[unit.dependency_group_id]
    population_total = sum(population.values())
    reconstruction_total = sum(reconstruction.values())
    if population_total == 0 or reconstruction_total == 0:
        return None
    keys = set(population) | set(reconstruction)
    return 0.5 * sum(
        abs(population.get(key, 0.0) / population_total - reconstruction.get(key, 0.0) / reconstruction_total)
        for key in keys
    )


def analyse_prior_manipulation(
    population_units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    parent: ModelMasteryProfile,
    unit_profile: UnitCellMasteryProfile,
    config: PriorManipulationConfig,
) -> dict[str, Any]:
    """Compare prior-aware and prior-agnostic sample designs before training."""

    units = [
        unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit)
        for unit in population_units
    ]
    validate_unit_cell_mastery_profile(unit_profile, parent, units)
    aware_units = _annotate_units(units, parent, unit_profile)
    agnostic_units = _annotate_units(units, parent, None)
    by_batch_aware: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    by_batch_agnostic: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in aware_units:
        by_batch_aware[unit.batch_id].append(unit)
    for unit in agnostic_units:
        by_batch_agnostic[unit.batch_id].append(unit)

    capability_scores: dict[str, list[float]] = defaultdict(list)
    for observation in unit_profile.observations:
        if observation.point_score is not None:
            capability_scores[observation.cell_id].append(observation.point_score)
    capability_variance = {
        cell_id: statistics.pvariance(values) if len(values) > 1 else 0.0
        for cell_id, values in sorted(capability_scores.items())
    }
    capabilities_with_variance = sum(value > 0 for value in capability_variance.values())

    trials = []
    for batch_id in sorted(by_batch_aware):
        truth = by_batch_aware[batch_id]
        measured_pairs = sum(
            unit.model_states[cell_id] != "unknown" for unit in truth for cell_id in unit.capability_ids
        )
        for fraction in config.sampling_fractions:
            for seed in config.seeds:
                sampling = ValueSamplingConfig(
                    sampling_fraction=fraction,
                    seed=seed,
                    stratum_attributes=config.stratum_attributes,
                    allow_target_expansion=config.allow_target_expansion,
                )
                aware = sample_value_distribution(truth, sampling)
                agnostic = sample_value_distribution(by_batch_agnostic[batch_id], sampling)
                aware_ids = set(aware["selected_group_ids"])
                agnostic_ids = set(agnostic["selected_group_ids"])
                union = aware_ids | agnostic_ids
                jaccard = len(aware_ids & agnostic_ids) / len(union) if union else 1.0
                aware_measured_tv = _true_joint_state_tv(truth, aware, measured_only=True)
                agnostic_measured_tv = _true_joint_state_tv(truth, agnostic, measured_only=True)
                trials.append(
                    {
                        "batch_id": batch_id,
                        "requested_fraction": fraction,
                        "seed": seed,
                        "measured_unit_cell_count": measured_pairs,
                        "selection_jaccard": jaccard,
                        "selection_changed": jaccard < config.maximum_unchanged_jaccard,
                        "aware_realized_fraction": aware["realized_sampling_fraction"],
                        "agnostic_realized_fraction": agnostic["realized_sampling_fraction"],
                        "aware_expansion": aware["realized_sampling_fraction"] - fraction,
                        "agnostic_expansion": agnostic["realized_sampling_fraction"] - fraction,
                        "aware_all_state_tv": _true_joint_state_tv(truth, aware, measured_only=False),
                        "agnostic_all_state_tv": _true_joint_state_tv(truth, agnostic, measured_only=False),
                        "aware_measured_state_tv": aware_measured_tv,
                        "agnostic_measured_state_tv": agnostic_measured_tv,
                        "measured_state_tv_advantage": (
                            None
                            if aware_measured_tv is None or agnostic_measured_tv is None
                            else agnostic_measured_tv - aware_measured_tv
                        ),
                    }
                )
    informative = [row for row in trials if row["measured_unit_cell_count"] > 0]
    changed_trial_fraction = (
        sum(row["selection_changed"] for row in informative) / len(informative) if informative else 0.0
    )
    advantages = [
        row["measured_state_tv_advantage"] for row in informative if row["measured_state_tv_advantage"] is not None
    ]
    median_jaccard = statistics.median(row["selection_jaccard"] for row in informative) if informative else 1.0
    median_advantage = statistics.median(advantages) if advantages else None
    signal_passed = capabilities_with_variance >= config.minimum_capabilities_with_variance
    manipulation_passed = changed_trial_fraction >= config.minimum_changed_trial_fraction
    status = (
        "identifiable_for_screening"
        if signal_passed and manipulation_passed
        else "not_identifiable_at_current_granularity"
    )
    return {
        "schema_version": "unit-prior-manipulation-analysis-v1",
        "purpose": "generation_stage_model_prior_identifiability_not_training_utility",
        "status": status,
        "model_profile_id": parent.profile_id,
        "unit_model_profile_id": unit_profile.profile_id,
        "population_unit_count": len(units),
        "measured_unit_cell_count": sum(row.probe_count > 0 for row in unit_profile.observations),
        "population_unit_cell_count": len(unit_profile.observations),
        "capability_point_score_variance": capability_variance,
        "capabilities_with_nonzero_within_variance": capabilities_with_variance,
        "signal_variation_passed": signal_passed,
        "informative_trial_count": len(informative),
        "selection_changed_trial_fraction": changed_trial_fraction,
        "selection_manipulation_passed": manipulation_passed,
        "selection_jaccard_median": median_jaccard,
        "measured_state_tv_advantage_median": median_advantage,
        "config": config.model_dump(mode="json"),
        "trials": trials,
        "interpretation": (
            "The sparse rule-level signal changes sample membership and can proceed to A8 screening."
            if status == "identifiable_for_screening"
            else "The model-prior intervention is not identifiable; do not interpret downstream A8 utility."
        ),
    }


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = statistics.fmean(values_x)
    mean_y = statistics.fmean(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y, strict=True))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in values_x) * sum((y - mean_y) ** 2 for y in values_y))
    return numerator / denominator if denominator else None


def analyse_prior_holdout_fidelity(
    population_units: Sequence[ValueSamplingUnit | Mapping[str, Any]],
    parent: ModelMasteryProfile,
    design_profile: UnitCellMasteryProfile,
    audit_profile: UnitCellMasteryProfile,
    sampling_config: PriorManipulationConfig,
    holdout_config: ProbeHoldoutConfig,
) -> dict[str, Any]:
    """Evaluate design-prior sampling on independently held-out probe items."""

    units = [
        unit if isinstance(unit, ValueSamplingUnit) else ValueSamplingUnit.model_validate(unit)
        for unit in population_units
    ]
    validate_unit_cell_mastery_profile(design_profile, parent, units)
    validate_unit_cell_mastery_profile(audit_profile, parent, units)
    aware_units = _annotate_units(units, parent, design_profile)
    agnostic_units = _annotate_units(units, parent, None)
    audit_truth = _annotate_units(units, parent, audit_profile)
    aware_by_batch: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    agnostic_by_batch: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    audit_by_batch: dict[str, list[ValueSamplingUnit]] = defaultdict(list)
    for unit in aware_units:
        aware_by_batch[unit.batch_id].append(unit)
    for unit in agnostic_units:
        agnostic_by_batch[unit.batch_id].append(unit)
    for unit in audit_truth:
        audit_by_batch[unit.batch_id].append(unit)

    design_by_pair = design_profile.by_unit_cell
    audit_by_pair = audit_profile.by_unit_cell
    paired_scores = [
        (float(design.point_score), float(audit_by_pair[key].point_score))
        for key, design in sorted(design_by_pair.items())
        if design.point_score is not None and audit_by_pair[key].point_score is not None
    ]
    score_correlation = _pearson(
        [row[0] for row in paired_scores],
        [row[1] for row in paired_scores],
    )

    trials = []
    for batch_id in sorted(aware_by_batch):
        truth = audit_by_batch[batch_id]
        audit_group_ids = {
            unit.dependency_group_id
            for unit in truth
            if any(unit.model_states[cell_id] != "unknown" for cell_id in unit.capability_ids)
        }
        measured_pairs = sum(
            unit.model_states[cell_id] != "unknown" for unit in truth for cell_id in unit.capability_ids
        )
        for fraction in sampling_config.sampling_fractions:
            for seed in sampling_config.seeds:
                sampling = ValueSamplingConfig(
                    sampling_fraction=fraction,
                    seed=seed,
                    stratum_attributes=sampling_config.stratum_attributes,
                    allow_target_expansion=sampling_config.allow_target_expansion,
                )
                aware = sample_value_distribution(aware_by_batch[batch_id], sampling)
                agnostic = sample_value_distribution(agnostic_by_batch[batch_id], sampling)
                aware_tv = _true_joint_state_tv(truth, aware, measured_only=True)
                agnostic_tv = _true_joint_state_tv(truth, agnostic, measured_only=True)
                aware_ids = set(aware["selected_group_ids"])
                agnostic_ids = set(agnostic["selected_group_ids"])
                union = aware_ids | agnostic_ids
                trials.append(
                    {
                        "batch_id": batch_id,
                        "requested_fraction": fraction,
                        "seed": seed,
                        "audit_measured_unit_cell_count": measured_pairs,
                        "selection_jaccard": len(aware_ids & agnostic_ids) / len(union) if union else 1.0,
                        "aware_audit_group_selection_rate": (
                            len(aware_ids & audit_group_ids) / len(audit_group_ids) if audit_group_ids else None
                        ),
                        "agnostic_audit_group_selection_rate": (
                            len(agnostic_ids & audit_group_ids) / len(audit_group_ids) if audit_group_ids else None
                        ),
                        "aware_realized_fraction": aware["realized_sampling_fraction"],
                        "agnostic_realized_fraction": agnostic["realized_sampling_fraction"],
                        "aware_audit_state_tv": aware_tv,
                        "agnostic_audit_state_tv": agnostic_tv,
                        "audit_state_tv_advantage": (
                            None if aware_tv is None or agnostic_tv is None else agnostic_tv - aware_tv
                        ),
                    }
                )
    informative = [
        row
        for row in trials
        if row["audit_measured_unit_cell_count"] > 0 and row["audit_state_tv_advantage"] is not None
    ]
    advantages = [float(row["audit_state_tv_advantage"]) for row in informative]
    positive_fraction = sum(value > 0 for value in advantages) / len(advantages) if advantages else 0.0
    median_advantage = statistics.median(advantages) if advantages else None
    aware_audit_selection_rates = [
        float(row["aware_audit_group_selection_rate"])
        for row in informative
        if row["aware_audit_group_selection_rate"] is not None
    ]
    agnostic_audit_selection_rates = [
        float(row["agnostic_audit_group_selection_rate"])
        for row in informative
        if row["agnostic_audit_group_selection_rate"] is not None
    ]
    supports = bool(
        median_advantage is not None
        and median_advantage >= holdout_config.minimum_median_advantage
        and positive_fraction >= holdout_config.minimum_positive_advantage_fraction
    )
    unseen_unit_generalization = holdout_config.split_mode == "disjoint_unit_cells"
    if unseen_unit_generalization:
        status = "supports_unseen_unit_screening" if supports else "does_not_support_unseen_unit_screening"
    else:
        status = "supports_item_holdout_fidelity" if supports else "does_not_support_item_holdout_fidelity"
    return {
        "schema_version": "unit-prior-independent-holdout-analysis-v1",
        "purpose": "generation_stage_independent_probe_holdout_not_training_utility",
        "status": status,
        "split_mode": holdout_config.split_mode,
        "unseen_unit_generalization_evaluated": unseen_unit_generalization,
        "design_profile_id": design_profile.profile_id,
        "audit_profile_id": audit_profile.profile_id,
        "paired_unit_cell_count": len(paired_scores),
        "design_audit_point_score_correlation": score_correlation,
        "informative_trial_count": len(informative),
        "positive_advantage_trial_fraction": positive_fraction,
        "audit_state_tv_advantage_median": median_advantage,
        "aware_audit_group_selection_rate_median": (
            statistics.median(aware_audit_selection_rates) if aware_audit_selection_rates else None
        ),
        "agnostic_audit_group_selection_rate_median": (
            statistics.median(agnostic_audit_selection_rates) if agnostic_audit_selection_rates else None
        ),
        "holdout_supports_claim": supports,
        "sampling_config": sampling_config.model_dump(mode="json"),
        "holdout_config": holdout_config.model_dump(mode="json"),
        "trials": trials,
        "interpretation": (
            (
                "A design-prior sample better reconstructs disjoint held-out unit states in this screening panel."
                if supports
                else "The design-prior sample does not improve disjoint held-out unit-state reconstruction."
            )
            if unseen_unit_generalization
            else (
                "Independent items on the same measured units support measurement stability only; unseen-rule "
                "generalization was not evaluated."
            )
        ),
    }


__all__ = [
    "PriorManipulationConfig",
    "ProbeHoldoutConfig",
    "ProbeUnitBinding",
    "analyse_prior_holdout_fidelity",
    "analyse_prior_manipulation",
    "bind_probes_to_semantic_units",
    "build_sparse_unit_observations",
    "build_sparse_unit_profile",
    "normalize_evidence_text",
    "split_bindings_for_probe_holdout",
]
