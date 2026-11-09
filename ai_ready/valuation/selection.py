"""Token-budgeted proxy selection over evidence-bound semantic units.

This module is intentionally independent from the audit channel and from locked
benchmark items.  Its inputs contain only source-derived units, a capability
blueprint expressed as coverage obligations, and observable quality strata.
"""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProxyUnit(StrictModel):
    unit_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_id: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    estimated_tokens: int = Field(gt=0)
    embedding: list[float] = Field(min_length=1)
    quality_bucket: str = Field(min_length=1)
    benchmark_relevance: float = Field(default=0.0, ge=0, le=1)
    eligible: bool = True
    medically_valid: bool = True

    @model_validator(mode="after")
    def reject_nonfinite_embedding(self):
        if any(not math.isfinite(value) for value in self.embedding):
            raise ValueError("Embedding values must be finite")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("Proxy unit source_ids must be unique")
        return self


class CoverageObligation(StrictModel):
    obligation_id: str = Field(min_length=1)
    kind: Literal["topic", "rule", "condition", "exception", "long_tail", "source"]
    unit_ids: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    capability_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_references(self):
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError("Coverage obligation unit_ids must be unique")
        if len(self.capability_ids) != len(set(self.capability_ids)):
            raise ValueError("Coverage obligation capability_ids must be unique")
        return self


class ProxySelectionConfig(StrictModel):
    method: Literal[
        "coverage_quality",
        "coverage_only",
        "random",
        "stratified_random",
        "relevance",
        "diversity",
    ] = "coverage_quality"
    token_budget: int = Field(gt=0)
    random_state: int = 42
    facility_weight: float = Field(default=1.0, ge=0)
    obligation_weight: float = Field(default=1.0, ge=0)
    quality_fidelity_weight: float = Field(default=1.0, ge=0)
    source_coverage_weight: float = Field(default=0.5, ge=0)

    @model_validator(mode="after")
    def require_objective(self):
        # ``coverage_only`` is an ablation of the main selector.  Normalize the
        # serialized config as well as the objective so an accidentally copied
        # quality weight cannot leak into this baseline.
        if self.method == "coverage_only":
            self.quality_fidelity_weight = 0.0
        if self.method in {"coverage_quality", "coverage_only"} and not any(
            value > 0
            for value in (
                self.facility_weight,
                self.obligation_weight,
                self.source_coverage_weight,
                *(() if self.method == "coverage_only" else (self.quality_fidelity_weight,)),
            )
        ):
            raise ValueError(f"{self.method} requires at least one positive enabled objective weight")
        return self


def selection_config_digest(config: ProxySelectionConfig) -> str:
    """Hash the normalized selector semantics rather than YAML formatting."""

    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _distribution(counts: Counter, total: float) -> dict[str, float]:
    return {key: float(value / total) for key, value in sorted(counts.items())} if total else {}


def _quality_fidelity(selected: Counter, target: dict[str, float]) -> float:
    total = sum(selected.values())
    if total <= 0:
        return 0.0
    actual = _distribution(selected, total)
    keys = set(actual) | set(target)
    total_variation = 0.5 * sum(abs(actual.get(key, 0.0) - target.get(key, 0.0)) for key in keys)
    return float(max(0.0, 1.0 - total_variation))


def select_proxy(
    units: list[ProxyUnit],
    obligations: list[CoverageObligation],
    config: ProxySelectionConfig,
) -> dict:
    """Select complete dependency groups within a token budget.

    `group_id` is the indivisible unit: a rule, its conditions and its exception
    evidence must enter or leave together.  Invalid medical content is never
    converted into a supervision target; its exclusion is reported as yield.
    """

    if not units:
        raise ValueError("At least one proxy unit is required")
    batch_ids = {unit.batch_id for unit in units}
    if len(batch_ids) != 1:
        raise ValueError("Proxy selection must run independently for one batch at a time")
    batch_hashes = {unit.batch_content_sha256 for unit in units}
    if len(batch_hashes) != 1:
        raise ValueError("All proxy units must come from the same frozen batch content")
    dimensions = {len(unit.embedding) for unit in units}
    if len(dimensions) != 1:
        raise ValueError("All proxy embeddings must have the same dimension")
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Proxy unit IDs must be unique")
    known_units = set(unit_ids)
    obligation_ids = [obligation.obligation_id for obligation in obligations]
    if len(obligation_ids) != len(set(obligation_ids)):
        raise ValueError("Coverage obligation IDs must be unique")
    for obligation in obligations:
        if unknown := set(obligation.unit_ids) - known_units:
            raise ValueError(f"Coverage obligation {obligation.obligation_id} references unknown units: {unknown}")

    all_by_group: dict[str, list[ProxyUnit]] = defaultdict(list)
    for unit in units:
        all_by_group[unit.group_id].append(unit)

    # Eligibility is decided for the complete dependency group. If an exception
    # or prerequisite is invalid, keeping only the remaining rule would turn a
    # medically incomplete fragment into supervision and break the atomicity
    # promised by group_id.
    excluded_group_ids = {
        group_id
        for group_id, group in all_by_group.items()
        if any(not unit.eligible or not unit.medically_valid for unit in group)
    }
    by_group = {group_id: group for group_id, group in all_by_group.items() if group_id not in excluded_group_ids}
    eligible = [unit for group in by_group.values() for unit in group]
    if not by_group:
        raise ValueError("No eligible, medically valid units are available")
    group_ids = sorted(by_group)
    group_cost = {group_id: sum(unit.estimated_tokens for unit in group) for group_id, group in by_group.items()}
    group_vector = {
        group_id: _normalized(np.asarray([unit.embedding for unit in group], dtype=np.float64).mean(axis=0))
        for group_id, group in by_group.items()
    }
    target_matrix = np.vstack([group_vector[group_id] for group_id in group_ids])
    target_weights = np.asarray(
        [
            sum(unit.estimated_tokens * (1.0 + unit.benchmark_relevance) for unit in by_group[group_id])
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    obligation_by_group: dict[str, set[str]] = defaultdict(set)
    eligible_ids = {unit.unit_id for unit in eligible}
    obligation_weights = {}
    for obligation in obligations:
        supported_ids = set(obligation.unit_ids) & eligible_ids
        if not supported_ids:
            continue
        obligation_weights[obligation.obligation_id] = obligation.weight
        for group_id, group in by_group.items():
            if supported_ids & {unit.unit_id for unit in group}:
                obligation_by_group[group_id].add(obligation.obligation_id)

    full_quality_counts = Counter()
    all_sources = set()
    group_strata: dict[str, set[tuple[str, str]]] = {}
    groups_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit in eligible:
        full_quality_counts[unit.quality_bucket] += unit.estimated_tokens
        all_sources.update(unit.source_ids)
    for group_id, group in by_group.items():
        # These strata are registered before selection and use source evidence
        # plus the observable quality bucket only.  Benchmark relevance,
        # embeddings and obligations cannot affect stratum membership.
        strata = {(source_id, unit.quality_bucket) for unit in group for source_id in unit.source_ids}
        group_strata[group_id] = strata
        for stratum in strata:
            groups_by_stratum[stratum].append(group_id)
    target_quality = _distribution(full_quality_counts, sum(full_quality_counts.values()))
    total_objective_weight = max(
        config.facility_weight
        + config.obligation_weight
        + config.quality_fidelity_weight
        + config.source_coverage_weight,
        1e-12,
    )

    def components(selected_groups: list[str]) -> dict[str, float]:
        if not selected_groups:
            return {"facility": 0.0, "obligation": 0.0, "quality_fidelity": 0.0, "source_coverage": 0.0}
        selected_matrix = np.vstack([group_vector[group_id] for group_id in selected_groups])
        similarities = np.clip(target_matrix @ selected_matrix.T, 0.0, 1.0)
        facility = float(np.sum(target_weights * similarities.max(axis=1)) / np.sum(target_weights))
        covered_obligations = set().union(*(obligation_by_group[group_id] for group_id in selected_groups))
        obligation_total = sum(obligation_weights.values())
        obligation_score = (
            sum(obligation_weights[obligation_id] for obligation_id in covered_obligations) / obligation_total
            if obligation_total
            else 1.0
        )
        selected_quality = Counter()
        selected_sources = set()
        for group_id in selected_groups:
            for unit in by_group[group_id]:
                selected_quality[unit.quality_bucket] += unit.estimated_tokens
                selected_sources.update(unit.source_ids)
        return {
            "facility": facility,
            "obligation": float(obligation_score),
            # 估值代理要保持清洗后批次的质量结构，而不是只挑最干净的“甜桃”。
            "quality_fidelity": _quality_fidelity(selected_quality, target_quality),
            "source_coverage": len(selected_sources) / len(all_sources) if all_sources else 1.0,
        }

    def objective(selected_groups: list[str]) -> float:
        score = components(selected_groups)
        return float(
            (
                config.facility_weight * score["facility"]
                + config.obligation_weight * score["obligation"]
                + config.quality_fidelity_weight * score["quality_fidelity"]
                + config.source_coverage_weight * score["source_coverage"]
            )
            / total_objective_weight
        )

    selected: list[str] = []
    trace = []
    remaining = set(group_ids)
    used_tokens = 0
    rng = random.Random(config.random_state)
    if config.method == "random":
        ordered = group_ids[:]
        rng.shuffle(ordered)
    elif config.method == "stratified_random":
        # A seeded order is preregistered once, then the sampler visits the
        # currently least represented source/quality strata in round-robin
        # order.  Shuffling only breaks ties; it never changes the strata.
        stratum_order = sorted(groups_by_stratum)
        rng.shuffle(stratum_order)
        ordered_by_stratum = {}
        for stratum in stratum_order:
            stratum_groups = sorted(groups_by_stratum[stratum])
            rng.shuffle(stratum_groups)
            ordered_by_stratum[stratum] = stratum_groups
        stratum_cursor = 0
    elif config.method == "relevance":
        ordered = sorted(
            group_ids,
            key=lambda group_id: (
                -sum(unit.benchmark_relevance * unit.estimated_tokens for unit in by_group[group_id])
                / group_cost[group_id],
                group_id,
            ),
        )
    else:
        ordered = []

    while remaining:
        feasible = [group_id for group_id in remaining if used_tokens + group_cost[group_id] <= config.token_budget]
        if not feasible:
            break
        previous_score = objective(selected)
        if config.method in {"random", "relevance"}:
            chosen = next(group_id for group_id in ordered if group_id in feasible)
            selection_stratum = None
        elif config.method == "stratified_random":
            coverage_counts = {
                stratum: sum(stratum in group_strata[group_id] for group_id in selected) for stratum in stratum_order
            }
            active_strata = [
                stratum
                for stratum in stratum_order
                if any(group_id in feasible for group_id in ordered_by_stratum[stratum])
            ]
            minimum_count = min(coverage_counts[stratum] for stratum in active_strata)
            least_covered = {stratum for stratum in active_strata if coverage_counts[stratum] == minimum_count}
            rotated_strata = stratum_order[stratum_cursor:] + stratum_order[:stratum_cursor]
            selection_stratum = next(stratum for stratum in rotated_strata if stratum in least_covered)
            chosen = next(group_id for group_id in ordered_by_stratum[selection_stratum] if group_id in feasible)
            stratum_cursor = (stratum_order.index(selection_stratum) + 1) % len(stratum_order)
        elif config.method == "diversity":
            selection_stratum = None
            if not selected:
                # Start from the medoid-like group nearest the unweighted
                # embedding cloud.  This is deliberately independent of
                # benchmark relevance so diversity remains a valid baseline.
                chosen = max(
                    feasible,
                    key=lambda group_id: (
                        float(np.mean(np.clip(target_matrix @ group_vector[group_id], -1.0, 1.0))),
                        -group_cost[group_id],
                        group_id,
                    ),
                )
            else:
                chosen = max(
                    feasible,
                    key=lambda group_id: (
                        min(1.0 - float(group_vector[group_id] @ group_vector[current]) for current in selected),
                        -group_cost[group_id],
                        group_id,
                    ),
                )
        else:
            selection_stratum = None
            # The greedy step evaluates the complete normalized objective after
            # adding each group.  This keeps semantic coverage and distribution
            # fidelity on a comparable [0, 1] scale.
            chosen = max(
                feasible,
                key=lambda group_id: (objective([*selected, group_id]) - previous_score, group_id),
            )
        selected.append(chosen)
        remaining.remove(chosen)
        used_tokens += group_cost[chosen]
        new_score = objective(selected)
        trace_entry = {
            "rank": len(selected),
            "group_id": chosen,
            "group_tokens": group_cost[chosen],
            "cumulative_tokens": used_tokens,
            "marginal_gain": new_score - previous_score,
            "objective": new_score,
            "components": components(selected),
        }
        if selection_stratum is not None:
            trace_entry["selection_stratum"] = {
                "source_id": selection_stratum[0],
                "quality_bucket": selection_stratum[1],
            }
        trace.append(trace_entry)

    selected_unit_ids = [unit.unit_id for group_id in selected for unit in by_group[group_id]]
    covered_obligations = set().union(*(obligation_by_group[group_id] for group_id in selected)) if selected else set()
    invalid_units = [unit for unit in units if unit.eligible and not unit.medically_valid]
    dependency_units_excluded = [
        unit
        for group_id in excluded_group_ids
        for unit in all_by_group[group_id]
        if unit.eligible and unit.medically_valid
    ]
    available_strata = [
        {
            "source_id": source_id,
            "quality_bucket": quality_bucket,
            "eligible_group_ids": sorted(groups_by_stratum[(source_id, quality_bucket)]),
        }
        for source_id, quality_bucket in sorted(groups_by_stratum)
    ]
    selected_strata = [
        {
            "source_id": source_id,
            "quality_bucket": quality_bucket,
            "selected_group_ids": sorted(
                group_id for group_id in selected if (source_id, quality_bucket) in group_strata[group_id]
            ),
        }
        for source_id, quality_bucket in sorted(groups_by_stratum)
        if any((source_id, quality_bucket) in group_strata[group_id] for group_id in selected)
    ]
    return {
        "schema_version": "proxy-selection-v1",
        "batch_id": next(iter(batch_ids)),
        "batch_content_sha256": next(iter(batch_hashes)),
        "method": config.method,
        "selection_config": config.model_dump(mode="json"),
        "selection_config_sha256": selection_config_digest(config),
        "token_budget": config.token_budget,
        "used_tokens": used_tokens,
        "selected_group_ids": selected,
        "selected_unit_ids": selected_unit_ids,
        "objective": objective(selected),
        "components": components(selected),
        "uncovered_obligation_ids": sorted(set(obligation_weights) - covered_obligations),
        "full_quality_distribution": target_quality,
        "stratification_basis": ["source_ids", "quality_bucket"],
        "available_strata": available_strata,
        "selected_strata": selected_strata,
        "eligible_units": len(eligible),
        "invalid_units_excluded": len(invalid_units),
        "dependency_units_excluded": len(dependency_units_excluded),
        "excluded_group_ids": sorted(excluded_group_ids),
        "effective_unit_yield": len(eligible) / len(units),
        "trace": trace,
    }
