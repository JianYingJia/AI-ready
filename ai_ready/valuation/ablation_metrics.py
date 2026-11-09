"""Compute ablation observations from strict, auditable source artifacts.

This module is deliberately the only bridge between raw ablation artifacts and
``AblationObservation``.  Callers provide item-level predictions, selection
inputs, or paired batch utilities; they never provide an already-computed
metric value.  Observation provenance is therefore bound to the exact artifact
bytes and to the semantic digest of the supplied frozen protocol.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.valuation.ablation import (
    AblationObservation,
    AblationProtocol,
    ablation_protocol_digest,
    load_ablation_protocol,
)
from ai_ready.valuation.contracts import file_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


MetricFamily = Literal["selection", "prediction", "ranking"]


class MetricSpec(_StrictModel):
    """A registered metric and the closed range of values it may produce."""

    name: str = Field(min_length=1)
    family: MetricFamily
    minimum: float
    maximum: float | None
    capability_level: bool = False
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("Metric maximum must be at least its minimum")
        return self


def _spec(
    name: str,
    family: MetricFamily,
    minimum: float,
    maximum: float | None,
    description: str,
    *,
    capability_level: bool = False,
) -> MetricSpec:
    return MetricSpec(
        name=name,
        family=family,
        minimum=minimum,
        maximum=maximum,
        capability_level=capability_level,
        description=description,
    )


_METRIC_SPECS = [
    _spec(
        "condition_exception_recall",
        "selection",
        0.0,
        1.0,
        "Gold condition/exception elements recovered anywhere in the mined groups.",
    ),
    _spec(
        "unsupported_medical_assertion_rate",
        "selection",
        0.0,
        1.0,
        "Human-adjudicated unsupported assertions divided by all audited assertions.",
    ),
    _spec(
        "dependency_group_recall",
        "selection",
        0.0,
        1.0,
        "Gold dependency groups whose required elements co-occur in one mined group.",
    ),
    _spec(
        "capability_mapping_macro_f1",
        "selection",
        0.0,
        1.0,
        "Macro F1 of predicted capability assignments against item-level gold annotations.",
    ),
    _spec(
        "weighted_obligation_coverage",
        "selection",
        0.0,
        1.0,
        "Fraction of preregistered obligation weight covered by selected units.",
    ),
    _spec(
        "condition_exception_coverage",
        "selection",
        0.0,
        1.0,
        "Fraction of condition and exception obligations covered by selected units.",
    ),
    _spec(
        "quality_distribution_similarity",
        "selection",
        0.0,
        1.0,
        "One minus total-variation distance between selected and candidate quality mass.",
    ),
    _spec(
        "blueprint_alignment",
        "selection",
        0.0,
        1.0,
        "One minus total-variation distance between selected capability mass and the blueprint.",
    ),
    # The frozen v1 protocol uses this longer spelling.  It is computed by the
    # same function rather than copied from an input artifact.
    _spec(
        "blueprint_alignment_score",
        "selection",
        0.0,
        1.0,
        "Protocol alias for blueprint_alignment.",
    ),
    _spec(
        "semantic_facility",
        "selection",
        0.0,
        1.0,
        "Weighted target-to-selected maximum semantic similarity.",
    ),
    _spec(
        "acceptance_rate_absolute_error",
        "selection",
        0.0,
        1.0,
        "Absolute difference between selected-sample and full-pool generation acceptance rates.",
    ),
    _spec(
        "prior_gap_selection_alignment",
        "selection",
        -1.0,
        1.0,
        "Pearson alignment between independently measured capability gaps and selected capability mass.",
    ),
    _spec("training_utility", "prediction", -1.0, 1.0, "Tuned accuracy minus base accuracy."),
    _spec(
        "raw_capability_gain",
        "prediction",
        -1.0,
        1.0,
        "Tuned-minus-base accuracy for one registered capability.",
        capability_level=True,
    ),
    _spec(
        "cross_form_probe_gain",
        "prediction",
        -1.0,
        1.0,
        "Tuned-minus-base accuracy on held-out supervision forms.",
        capability_level=True,
    ),
    _spec(
        "target_capability_probe_gain",
        "prediction",
        -1.0,
        1.0,
        "Tuned-minus-base accuracy on the preregistered target-capability probes.",
    ),
    _spec(
        "target_minus_nontarget_probe_gain",
        "prediction",
        -2.0,
        2.0,
        "Target probe gain minus matched nontarget probe gain.",
        capability_level=True,
    ),
    _spec("spearman_to_fom", "ranking", -1.0, 1.0, "Proxy/FOM Spearman rank correlation."),
    _spec("pairwise_accuracy_to_fom", "ranking", 0.0, 1.0, "Accuracy on distinguishable FOM pairs."),
    _spec("top1_regret_to_fom", "ranking", 0.0, 1.0, "FOM utility lost by choosing the proxy top batch."),
    _spec(
        "low_value_false_promotion_rate",
        "ranking",
        0.0,
        1.0,
        "Fraction of FOM-low-value batches promoted into the proxy top-k set.",
    ),
    _spec(
        "low_value_promotion_rate",
        "ranking",
        0.0,
        1.0,
        "Protocol-compatible alias for the low-value false-promotion rate.",
    ),
    _spec(
        "leave_one_lineage_out_spearman_to_fom",
        "ranking",
        -1.0,
        1.0,
        "Proxy/FOM Spearman after excluding one lineage.",
    ),
    _spec(
        "leave_one_lineage_out_pairwise_accuracy_to_fom",
        "ranking",
        0.0,
        1.0,
        "Distinguishable-pair accuracy after excluding one lineage.",
    ),
    _spec(
        "leave_one_lineage_out_top1_regret_to_fom",
        "ranking",
        0.0,
        1.0,
        "FOM top-1 regret after excluding one lineage.",
    ),
    _spec(
        "leave_one_lineage_out_low_value_false_promotion_rate",
        "ranking",
        0.0,
        1.0,
        "Low-value false-promotion rate after excluding one lineage.",
    ),
    _spec(
        "leave_one_lineage_out_low_value_promotion_rate_to_fom",
        "ranking",
        0.0,
        1.0,
        "Protocol alias for the low-value false-promotion rate after excluding one lineage.",
    ),
]

METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType({spec.name: spec for spec in _METRIC_SPECS})


def metric_spec(metric: str) -> MetricSpec:
    """Return a registered metric or reject an unimplemented/free-form name."""

    try:
        return METRIC_REGISTRY[metric]
    except KeyError as error:
        raise ValueError(f"Unknown or uncomputed ablation metric: {metric}") from error


def validate_metric_value(metric: str, value: float) -> float:
    """Validate a computed value against its registered mathematical range."""

    spec = metric_spec(metric)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Metric {metric} must be finite")
    if numeric < spec.minimum or (spec.maximum is not None and numeric > spec.maximum):
        upper = "+inf" if spec.maximum is None else str(spec.maximum)
        raise ValueError(f"Metric {metric}={numeric} is outside its registered range [{spec.minimum}, {upper}]")
    # Remove harmless floating-point spill at closed probability bounds.
    if spec.minimum == 0.0 and numeric < 0.0:
        return 0.0
    if spec.maximum == 1.0 and numeric > 1.0:
        return 1.0
    return numeric


class StaticAuditUnit(_StrictModel):
    """One eligible unit in the candidate pool before selection."""

    unit_id: str = Field(min_length=1)
    estimated_tokens: int = Field(gt=0)
    quality_bucket: str = Field(min_length=1)


class StaticAuditObligation(_StrictModel):
    obligation_id: str = Field(min_length=1)
    kind: Literal["topic", "rule", "condition", "exception", "long_tail", "source"]
    unit_ids: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)
    capability_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_references(self):
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError(f"Obligation {self.obligation_id} contains duplicate unit IDs")
        if len(self.capability_ids) != len(set(self.capability_ids)):
            raise ValueError(f"Obligation {self.obligation_id} contains duplicate capability IDs")
        return self


class BlueprintWeight(_StrictModel):
    capability_id: str = Field(min_length=1)
    weight: float = Field(ge=0)


class MiningGoldElement(_StrictModel):
    """One expert-annotated source element; the label is never a metric."""

    element_id: str = Field(min_length=1)
    kind: Literal["fact", "condition", "exception"]
    annotator_id: str = Field(min_length=1)
    source_span_id: str = Field(min_length=1)


class MiningGoldDependencyGroup(_StrictModel):
    dependency_group_id: str = Field(min_length=1)
    required_element_ids: list[str] = Field(min_length=2)
    annotator_id: str = Field(min_length=1)

    @field_validator("required_element_ids")
    @classmethod
    def validate_required_elements(cls, value: list[str]) -> list[str]:
        _require_unique(value, "dependency-group element IDs")
        return value


class ExtractedMiningGroup(_StrictModel):
    extracted_group_id: str = Field(min_length=1)
    matched_gold_element_ids: list[str] = Field(default_factory=list)

    @field_validator("matched_gold_element_ids")
    @classmethod
    def validate_matches(cls, value: list[str]) -> list[str]:
        _require_unique(value, "matched gold element IDs")
        return value


class AssertionAdjudication(_StrictModel):
    """Expert support judgment for one material assertion emitted by mining."""

    assertion_id: str = Field(min_length=1)
    extracted_group_id: str = Field(min_length=1)
    support_label: Literal["supported", "unsupported"]
    annotator_id: str = Field(min_length=1)
    cited_source_span_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_support_evidence(self):
        _require_unique(self.cited_source_span_ids, f"cited spans for assertion {self.assertion_id}")
        if self.support_label == "supported" and not self.cited_source_span_ids:
            raise ValueError("A supported assertion must cite at least one source span")
        return self


class CapabilityMappingAnnotation(_StrictModel):
    unit_id: str = Field(min_length=1)
    gold_capability_ids: list[str] = Field(min_length=1)
    predicted_capability_ids: list[str] = Field(default_factory=list)
    annotator_id: str = Field(min_length=1)

    @field_validator("gold_capability_ids", "predicted_capability_ids")
    @classmethod
    def validate_capability_ids(cls, value: list[str], info) -> list[str]:
        _require_unique(value, info.field_name)
        return value


class FacilityTarget(_StrictModel):
    """Frozen target vector similarities to every eligible source unit."""

    target_id: str = Field(min_length=1)
    weight: float = Field(gt=0)
    similarity_by_unit_id: dict[str, float] = Field(min_length=1)

    @field_validator("similarity_by_unit_id")
    @classmethod
    def validate_similarities(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not unit_id.strip() for unit_id in value):
            raise ValueError("Facility similarity unit IDs cannot be empty")
        if any(not 0 <= similarity <= 1 for similarity in value.values()):
            raise ValueError("Facility similarities must be in [0, 1]")
        return dict(sorted(value.items()))


class GenerationAttempt(_StrictModel):
    """One attempted SFT generation, before aggregation into an acceptance rate."""

    attempt_id: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    accepted: bool
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_disposition(self):
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("Accepted generation attempts cannot have a rejection reason")
        if not self.accepted and not self.rejection_reason:
            raise ValueError("Rejected generation attempts require a rejection reason")
        return self


class StaticSelectionAudit(_StrictModel):
    """Raw static evidence used to recompute mining and selection endpoints.

    Sections may be omitted when they are unrelated to a requested metric.  A
    metric-specific computation refuses to run unless its entire raw section
    is present, so an omitted section can never silently turn into a score.
    """

    schema_version: Literal["ablation-static-selection-v1"] = "ablation-static-selection-v1"
    batch_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    capability_ids: list[str] = Field(default_factory=list)
    quality_buckets: list[str] = Field(default_factory=list)
    units: list[StaticAuditUnit] = Field(default_factory=list)
    obligations: list[StaticAuditObligation] = Field(default_factory=list)
    blueprint: list[BlueprintWeight] = Field(default_factory=list)
    selected_unit_ids: list[str] = Field(default_factory=list)
    mining_gold_elements: list[MiningGoldElement] = Field(default_factory=list)
    mining_gold_dependency_groups: list[MiningGoldDependencyGroup] = Field(default_factory=list)
    extracted_mining_groups: list[ExtractedMiningGroup] = Field(default_factory=list)
    assertion_adjudications: list[AssertionAdjudication] = Field(default_factory=list)
    capability_mapping_annotations: list[CapabilityMappingAnnotation] = Field(default_factory=list)
    facility_targets: list[FacilityTarget] = Field(default_factory=list)
    generation_attempts: list[GenerationAttempt] = Field(default_factory=list)
    prior_capability_gaps: list[BlueprintWeight] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_registry_and_references(self):
        _require_unique(self.capability_ids, "capability_ids")
        _require_unique(self.quality_buckets, "quality_buckets")
        unit_ids = [row.unit_id for row in self.units]
        _require_unique(unit_ids, "static audit unit IDs")
        obligation_ids = [row.obligation_id for row in self.obligations]
        _require_unique(obligation_ids, "static audit obligation IDs")
        blueprint_ids = [row.capability_id for row in self.blueprint]
        _require_unique(blueprint_ids, "blueprint capability IDs")
        _require_unique(self.selected_unit_ids, "selected_unit_ids")

        known_units = set(unit_ids)
        if unknown := set(self.selected_unit_ids) - known_units:
            raise ValueError(f"Selection references unknown or missing units: {sorted(unknown)}")
        known_capabilities = set(self.capability_ids)
        known_quality = set(self.quality_buckets)
        for unit in self.units:
            if unit.quality_bucket not in known_quality:
                raise ValueError(f"Unit {unit.unit_id} uses unknown quality bucket {unit.quality_bucket}")
        for obligation in self.obligations:
            if unknown := set(obligation.unit_ids) - known_units:
                raise ValueError(f"Obligation {obligation.obligation_id} references missing units: {sorted(unknown)}")
            if unknown := set(obligation.capability_ids) - known_capabilities:
                raise ValueError(
                    f"Obligation {obligation.obligation_id} references unknown capabilities: {sorted(unknown)}"
                )
        if self.blueprint and set(blueprint_ids) != known_capabilities:
            missing = sorted(known_capabilities - set(blueprint_ids))
            unknown = sorted(set(blueprint_ids) - known_capabilities)
            raise ValueError(
                f"Blueprint must cover the capability registry exactly; missing={missing}, unknown={unknown}"
            )
        if self.blueprint and sum(row.weight for row in self.blueprint) <= 0:
            raise ValueError("Blueprint must contain positive total weight")

        gold_element_ids = [row.element_id for row in self.mining_gold_elements]
        _require_unique(gold_element_ids, "mining gold element IDs")
        dependency_ids = [row.dependency_group_id for row in self.mining_gold_dependency_groups]
        _require_unique(dependency_ids, "mining gold dependency-group IDs")
        known_elements = set(gold_element_ids)
        for group in self.mining_gold_dependency_groups:
            if unknown := set(group.required_element_ids) - known_elements:
                raise ValueError(
                    f"Gold dependency group {group.dependency_group_id} references unknown elements: {sorted(unknown)}"
                )
        extracted_group_ids = [row.extracted_group_id for row in self.extracted_mining_groups]
        _require_unique(extracted_group_ids, "extracted mining group IDs")
        known_extracted_groups = set(extracted_group_ids)
        for group in self.extracted_mining_groups:
            if unknown := set(group.matched_gold_element_ids) - known_elements:
                raise ValueError(
                    f"Extracted group {group.extracted_group_id} references unknown gold elements: {sorted(unknown)}"
                )
        assertion_ids = [row.assertion_id for row in self.assertion_adjudications]
        _require_unique(assertion_ids, "assertion adjudication IDs")
        for assertion in self.assertion_adjudications:
            if assertion.extracted_group_id not in known_extracted_groups:
                raise ValueError(
                    f"Assertion {assertion.assertion_id} references unknown extracted group "
                    f"{assertion.extracted_group_id}"
                )

        mapping_unit_ids = [row.unit_id for row in self.capability_mapping_annotations]
        _require_unique(mapping_unit_ids, "capability mapping unit IDs")
        for annotation in self.capability_mapping_annotations:
            if annotation.unit_id not in known_units:
                raise ValueError(f"Capability mapping references unknown unit {annotation.unit_id}")
            mapped = set(annotation.gold_capability_ids) | set(annotation.predicted_capability_ids)
            if unknown := mapped - known_capabilities:
                raise ValueError(
                    f"Capability mapping for {annotation.unit_id} uses unknown capabilities: {sorted(unknown)}"
                )

        target_ids = [row.target_id for row in self.facility_targets]
        _require_unique(target_ids, "facility target IDs")
        for target in self.facility_targets:
            observed = set(target.similarity_by_unit_id)
            if observed != known_units:
                missing = sorted(known_units - observed)
                unknown = sorted(observed - known_units)
                raise ValueError(
                    f"Facility target {target.target_id} must cover every candidate unit; "
                    f"missing={missing}, unknown={unknown}"
                )

        attempt_ids = [row.attempt_id for row in self.generation_attempts]
        _require_unique(attempt_ids, "generation attempt IDs")
        for attempt in self.generation_attempts:
            if attempt.unit_id not in known_units:
                raise ValueError(f"Generation attempt {attempt.attempt_id} references unknown unit {attempt.unit_id}")

        prior_ids = [row.capability_id for row in self.prior_capability_gaps]
        _require_unique(prior_ids, "prior capability IDs")
        if self.prior_capability_gaps and set(prior_ids) != known_capabilities:
            missing = sorted(known_capabilities - set(prior_ids))
            unknown = sorted(set(prior_ids) - known_capabilities)
            raise ValueError(f"Capability gaps must cover the registry exactly; missing={missing}, unknown={unknown}")
        if self.prior_capability_gaps and sum(row.weight for row in self.prior_capability_gaps) <= 0:
            raise ValueError("Capability gaps must contain positive total mass")
        return self


TargetLabel = Literal["target", "nontarget"]
ProbeForm = Literal["trained_form", "cross_form"]


class ProbeItem(_StrictModel):
    item_id: str = Field(min_length=1)
    gold_label: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    target_label: TargetLabel
    probe_form: ProbeForm


class ProbeCatalog(_StrictModel):
    capability_ids: list[str] = Field(min_length=1)
    items: list[ProbeItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_items(self):
        _require_unique(self.capability_ids, "probe capability_ids")
        item_ids = [row.item_id for row in self.items]
        _require_unique(item_ids, "probe item IDs")
        known = set(self.capability_ids)
        for item in self.items:
            if item.capability_id not in known:
                raise ValueError(f"Probe item {item.item_id} uses unknown capability {item.capability_id}")
        return self


class ItemPrediction(_StrictModel):
    item_id: str = Field(min_length=1)
    predicted_label: str = Field(min_length=1)


class PredictionRun(_StrictModel):
    run_id: str = Field(min_length=1)
    predictions: list[ItemPrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prediction_ids(self):
        _require_unique([row.item_id for row in self.predictions], f"prediction item IDs in {self.run_id}")
        return self


class PairedPredictionAudit(_StrictModel):
    """A locked item catalog plus fully paired base and tuned predictions."""

    schema_version: Literal["ablation-paired-predictions-v1"] = "ablation-paired-predictions-v1"
    batch_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    seed: int
    catalog: ProbeCatalog
    base: PredictionRun
    tuned: PredictionRun

    @model_validator(mode="after")
    def validate_complete_pair(self):
        if self.base.run_id == self.tuned.run_id:
            raise ValueError("Base and tuned predictions require distinct run IDs")
        expected = {row.item_id for row in self.catalog.items}
        for role, run in (("base", self.base), ("tuned", self.tuned)):
            observed = {row.item_id for row in run.predictions}
            if observed != expected:
                missing = sorted(expected - observed)
                unknown = sorted(observed - expected)
                raise ValueError(
                    f"{role} predictions do not match the item catalog; missing={missing}, unknown={unknown}"
                )
        return self


class BatchUtilityPair(_StrictModel):
    batch_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    proxy_utility: float
    fom_utility: float


class RankingUtilityAudit(_StrictModel):
    """Paired, already seed-aggregated proxy and FOM utility by frozen batch."""

    schema_version: Literal["ablation-ranking-utilities-v1"] = "ablation-ranking-utilities-v1"
    rows: list[BatchUtilityPair] = Field(min_length=3)
    selection_k: int = Field(ge=1)
    low_value_threshold: float
    indifference_margin: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_rows(self):
        _require_unique([row.batch_id for row in self.rows], "ranking batch IDs")
        if self.selection_k >= len(self.rows):
            raise ValueError("selection_k must be smaller than the number of batches")
        if not any(row.fom_utility <= self.low_value_threshold for row in self.rows):
            raise ValueError("At least one FOM-low-value batch is required for false-promotion rate")
        return self


class SelectionMetricValues(_StrictModel):
    condition_exception_recall: float = Field(ge=0, le=1)
    unsupported_medical_assertion_rate: float = Field(ge=0, le=1)
    dependency_group_recall: float = Field(ge=0, le=1)
    capability_mapping_macro_f1: float = Field(ge=0, le=1)
    weighted_obligation_coverage: float = Field(ge=0, le=1)
    condition_exception_coverage: float = Field(ge=0, le=1)
    quality_distribution_similarity: float = Field(ge=0, le=1)
    blueprint_alignment: float = Field(ge=0, le=1)
    semantic_facility: float = Field(ge=0, le=1)
    acceptance_rate_absolute_error: float = Field(ge=0, le=1)
    prior_gap_selection_alignment: float = Field(ge=-1, le=1)


class PredictionMetricValues(_StrictModel):
    training_utility: float = Field(ge=-1, le=1)
    raw_capability_gain: float = Field(ge=-1, le=1)
    cross_form_probe_gain: float = Field(ge=-1, le=1)
    target_capability_probe_gain: float = Field(ge=-1, le=1)
    target_minus_nontarget_probe_gain: float = Field(ge=-2, le=2)


class RankingMetricValues(_StrictModel):
    spearman: float = Field(ge=-1, le=1)
    pairwise_accuracy: float = Field(ge=0, le=1)
    top1_regret: float = Field(ge=0)
    low_value_false_promotion_rate: float = Field(ge=0, le=1)


class LeaveOneLineageOutMetrics(_StrictModel):
    excluded_lineage_group: str = Field(min_length=1)
    metrics: RankingMetricValues


class AuditExecutionManifest(_StrictModel):
    """Dedicated manifest for static annotations or post-training evaluations."""

    schema_version: Literal["stage-ablation-audit-manifest-v1"] = "stage-ablation-audit-manifest-v1"
    execution_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    protocol_sha256: str
    study_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    artifact_uri: str = Field(min_length=1)
    artifact_sha256: str

    @field_validator("protocol_sha256", "artifact_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"{info.field_name} must be a 64-character SHA-256 digest")
        return normalized


class MaterializationExecutionManifest(_StrictModel):
    """Strict header and inventory emitted by ``materialize_ablation_batch``."""

    schema_version: Literal["stage-ablation-materialization-v1"] = "stage-ablation-materialization-v1"
    execution_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    execution_sha256: str
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    batch_input_sha256: str
    cell_count: int = Field(ge=0)
    ready_cell_count: int = Field(ge=0)
    not_identifiable_cell_count: int = Field(ge=0)
    job_count: int = Field(ge=0)
    jobs_uri: str = Field(min_length=1)
    jobs_sha256: str
    cells: list[dict]

    @field_validator(
        "execution_sha256",
        "protocol_sha256",
        "suite_sha256",
        "contract_sha256",
        "batch_content_sha256",
        "batch_input_sha256",
        "jobs_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"{info.field_name} must be a 64-character SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def validate_counts(self):
        if self.ready_cell_count + self.not_identifiable_cell_count != self.cell_count:
            raise ValueError("Materialization cell status counts must sum to cell_count")
        if len(self.cells) != self.cell_count:
            raise ValueError("Materialization cells must contain exactly cell_count records")
        return self


def _require_unique(values: Sequence[str], label: str) -> None:
    if len(values) != len(set(values)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        raise ValueError(f"{label} must be unique; duplicates={duplicates}")


def _covered_obligations(artifact: StaticSelectionAudit) -> list[StaticAuditObligation]:
    selected = set(artifact.selected_unit_ids)
    return [row for row in artifact.obligations if selected.intersection(row.unit_ids)]


def _similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("Distribution similarity requires positive mass on both sides")
    keys = set(left) | set(right)
    total_variation = 0.5 * sum(
        abs(left.get(key, 0.0) / left_total - right.get(key, 0.0) / right_total) for key in keys
    )
    return float(max(0.0, min(1.0, 1.0 - total_variation)))


def _require_nonempty(rows: Sequence[object], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} requires non-empty raw audit rows")


def _selected_capability_mass(artifact: StaticSelectionAudit) -> Counter[str]:
    _require_nonempty(artifact.obligations, "Selected capability mass")
    selected_capabilities: Counter[str] = Counter()
    for obligation in _covered_obligations(artifact):
        share = obligation.weight / len(obligation.capability_ids)
        for capability_id in obligation.capability_ids:
            selected_capabilities[capability_id] += share
    if not selected_capabilities:
        raise ValueError("Selected capability mass requires at least one covered obligation")
    return selected_capabilities


SelectionMetricName = Literal[
    "condition_exception_recall",
    "unsupported_medical_assertion_rate",
    "dependency_group_recall",
    "capability_mapping_macro_f1",
    "weighted_obligation_coverage",
    "condition_exception_coverage",
    "quality_distribution_similarity",
    "blueprint_alignment",
    "blueprint_alignment_score",
    "semantic_facility",
    "acceptance_rate_absolute_error",
    "prior_gap_selection_alignment",
]


def compute_selection_metric(artifact: StaticSelectionAudit, metric: SelectionMetricName) -> float:
    """Recompute one static endpoint from item-level or annotation-level evidence."""

    selected_ids = set(artifact.selected_unit_ids)
    if metric == "condition_exception_recall":
        gold = [row for row in artifact.mining_gold_elements if row.kind in {"condition", "exception"}]
        _require_nonempty(gold, metric)
        recovered = {
            element_id for group in artifact.extracted_mining_groups for element_id in group.matched_gold_element_ids
        }
        value = sum(row.element_id in recovered for row in gold) / len(gold)
    elif metric == "unsupported_medical_assertion_rate":
        _require_nonempty(artifact.assertion_adjudications, metric)
        value = sum(row.support_label == "unsupported" for row in artifact.assertion_adjudications) / len(
            artifact.assertion_adjudications
        )
    elif metric == "dependency_group_recall":
        _require_nonempty(artifact.mining_gold_dependency_groups, metric)
        extracted_sets = [set(row.matched_gold_element_ids) for row in artifact.extracted_mining_groups]
        value = sum(
            any(set(group.required_element_ids) <= extracted for extracted in extracted_sets)
            for group in artifact.mining_gold_dependency_groups
        ) / len(artifact.mining_gold_dependency_groups)
    elif metric == "capability_mapping_macro_f1":
        _require_nonempty(artifact.capability_mapping_annotations, metric)
        _require_nonempty(artifact.capability_ids, metric)
        per_capability = []
        for capability_id in artifact.capability_ids:
            true_positive = sum(
                capability_id in row.gold_capability_ids and capability_id in row.predicted_capability_ids
                for row in artifact.capability_mapping_annotations
            )
            false_positive = sum(
                capability_id not in row.gold_capability_ids and capability_id in row.predicted_capability_ids
                for row in artifact.capability_mapping_annotations
            )
            false_negative = sum(
                capability_id in row.gold_capability_ids and capability_id not in row.predicted_capability_ids
                for row in artifact.capability_mapping_annotations
            )
            if true_positive + false_negative == 0:
                raise ValueError(f"{metric} requires at least one gold-positive item for {capability_id}")
            denominator = 2 * true_positive + false_positive + false_negative
            per_capability.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        value = sum(per_capability) / len(per_capability)
    elif metric == "weighted_obligation_coverage":
        _require_nonempty(artifact.obligations, metric)
        covered_ids = {row.obligation_id for row in _covered_obligations(artifact)}
        total_weight = sum(row.weight for row in artifact.obligations)
        value = sum(row.weight for row in artifact.obligations if row.obligation_id in covered_ids) / total_weight
    elif metric == "condition_exception_coverage":
        relevant = [row for row in artifact.obligations if row.kind in {"condition", "exception"}]
        _require_nonempty(relevant, metric)
        covered_ids = {row.obligation_id for row in _covered_obligations(artifact)}
        value = sum(row.obligation_id in covered_ids for row in relevant) / len(relevant)
    elif metric == "quality_distribution_similarity":
        _require_nonempty(artifact.units, metric)
        if not selected_ids:
            raise ValueError(f"{metric} requires at least one selected unit")
        candidate_quality: Counter[str] = Counter()
        selected_quality: Counter[str] = Counter()
        for unit in artifact.units:
            candidate_quality[unit.quality_bucket] += unit.estimated_tokens
            if unit.unit_id in selected_ids:
                selected_quality[unit.quality_bucket] += unit.estimated_tokens
        represented_buckets = sum(mass > 0 for mass in candidate_quality.values())
        if represented_buckets < 2:
            raise ValueError(f"{metric} is not identifiable with fewer than two represented quality buckets")
        value = _similarity(candidate_quality, selected_quality)
    elif metric in {"blueprint_alignment", "blueprint_alignment_score"}:
        _require_nonempty(artifact.blueprint, metric)
        value = _similarity(
            _selected_capability_mass(artifact),
            {row.capability_id: row.weight for row in artifact.blueprint},
        )
    elif metric == "semantic_facility":
        _require_nonempty(artifact.facility_targets, metric)
        if not selected_ids:
            raise ValueError(f"{metric} requires at least one selected unit")
        numerator = sum(
            target.weight * max(target.similarity_by_unit_id[unit_id] for unit_id in selected_ids)
            for target in artifact.facility_targets
        )
        value = numerator / sum(target.weight for target in artifact.facility_targets)
    elif metric == "acceptance_rate_absolute_error":
        _require_nonempty(artifact.generation_attempts, metric)
        selected_attempts = [row for row in artifact.generation_attempts if row.unit_id in selected_ids]
        _require_nonempty(selected_attempts, f"{metric} selected stratum")
        full_rate = sum(row.accepted for row in artifact.generation_attempts) / len(artifact.generation_attempts)
        selected_rate = sum(row.accepted for row in selected_attempts) / len(selected_attempts)
        value = abs(selected_rate - full_rate)
    elif metric == "prior_gap_selection_alignment":
        _require_nonempty(artifact.prior_capability_gaps, metric)
        selected_mass = _selected_capability_mass(artifact)
        if len(artifact.capability_ids) < 2:
            raise ValueError(f"{metric} requires at least two capabilities")
        gaps = {row.capability_id: row.weight for row in artifact.prior_capability_gaps}
        value = _pearson(
            [gaps[capability_id] for capability_id in artifact.capability_ids],
            [selected_mass.get(capability_id, 0.0) for capability_id in artifact.capability_ids],
        )
    else:  # pragma: no cover - protected by the Literal and registry
        raise ValueError(f"Selection metric has no computation binding: {metric}")
    return validate_metric_value(metric, value)


def compute_selection_metrics(artifact: StaticSelectionAudit) -> SelectionMetricValues:
    """Recompute every static endpoint; useful for a deliberately complete audit."""

    values = SelectionMetricValues(
        condition_exception_recall=compute_selection_metric(artifact, "condition_exception_recall"),
        unsupported_medical_assertion_rate=compute_selection_metric(
            artifact,
            "unsupported_medical_assertion_rate",
        ),
        dependency_group_recall=compute_selection_metric(artifact, "dependency_group_recall"),
        capability_mapping_macro_f1=compute_selection_metric(artifact, "capability_mapping_macro_f1"),
        weighted_obligation_coverage=compute_selection_metric(artifact, "weighted_obligation_coverage"),
        condition_exception_coverage=compute_selection_metric(artifact, "condition_exception_coverage"),
        quality_distribution_similarity=compute_selection_metric(artifact, "quality_distribution_similarity"),
        blueprint_alignment=compute_selection_metric(artifact, "blueprint_alignment"),
        semantic_facility=compute_selection_metric(artifact, "semantic_facility"),
        acceptance_rate_absolute_error=compute_selection_metric(artifact, "acceptance_rate_absolute_error"),
        prior_gap_selection_alignment=compute_selection_metric(artifact, "prior_gap_selection_alignment"),
    )
    return values


def _accuracy(
    item_ids: set[str],
    catalog: Mapping[str, ProbeItem],
    predictions: Mapping[str, str],
    *,
    label: str,
) -> float:
    if not item_ids:
        raise ValueError(f"{label} requires at least one registered item")
    return sum(predictions[item_id] == catalog[item_id].gold_label for item_id in item_ids) / len(item_ids)


def _prediction_context(
    artifact: PairedPredictionAudit,
    *,
    capability_id: str | None,
) -> tuple[dict[str, ProbeItem], dict[str, str], dict[str, str], set[str]]:
    known_capabilities = set(artifact.catalog.capability_ids)
    if capability_id is not None and capability_id not in known_capabilities:
        raise ValueError(f"Unknown capability requested for prediction metrics: {capability_id}")
    catalog = {row.item_id: row for row in artifact.catalog.items}
    base = {row.item_id: row.predicted_label for row in artifact.base.predictions}
    tuned = {row.item_id: row.predicted_label for row in artifact.tuned.predictions}
    scoped = {
        item_id for item_id, item in catalog.items() if capability_id is None or item.capability_id == capability_id
    }
    if not scoped:
        label = "all capabilities" if capability_id is None else f"capability {capability_id}"
        raise ValueError(f"Prediction metrics require at least one item for {label}")
    return catalog, base, tuned, scoped


def _accuracy_gain(
    item_ids: set[str],
    catalog: Mapping[str, ProbeItem],
    base: Mapping[str, str],
    tuned: Mapping[str, str],
    *,
    label: str,
) -> float:
    return _accuracy(item_ids, catalog, tuned, label=label) - _accuracy(item_ids, catalog, base, label=label)


def compute_prediction_metric(
    artifact: PairedPredictionAudit,
    metric: Literal[
        "training_utility",
        "raw_capability_gain",
        "cross_form_probe_gain",
        "target_capability_probe_gain",
        "target_minus_nontarget_probe_gain",
    ],
    *,
    capability_id: str | None = None,
) -> float:
    """Compute one prediction endpoint without requiring unrelated probe strata."""

    catalog, base, tuned, scoped = _prediction_context(artifact, capability_id=capability_id)
    if metric in {"training_utility", "raw_capability_gain"}:
        value = _accuracy_gain(scoped, catalog, base, tuned, label=metric)
    elif metric == "cross_form_probe_gain":
        item_ids = {item_id for item_id in scoped if catalog[item_id].probe_form == "cross_form"}
        value = _accuracy_gain(item_ids, catalog, base, tuned, label=metric)
    elif metric == "target_capability_probe_gain":
        item_ids = {item_id for item_id in scoped if catalog[item_id].target_label == "target"}
        value = _accuracy_gain(item_ids, catalog, base, tuned, label=metric)
    elif metric == "target_minus_nontarget_probe_gain":
        target = {item_id for item_id in scoped if catalog[item_id].target_label == "target"}
        nontarget = {item_id for item_id in scoped if catalog[item_id].target_label == "nontarget"}
        target_gain = _accuracy_gain(
            target,
            catalog,
            base,
            tuned,
            label="target_minus_nontarget_probe_gain target stratum",
        )
        nontarget_gain = _accuracy_gain(
            nontarget,
            catalog,
            base,
            tuned,
            label="target_minus_nontarget_probe_gain nontarget stratum",
        )
        value = target_gain - nontarget_gain
    else:  # pragma: no cover - the Literal and registry guard public callers
        raise ValueError(f"Prediction metric has no computation binding: {metric}")
    return validate_metric_value(metric, value)


def compute_prediction_metrics(
    artifact: PairedPredictionAudit,
    *,
    capability_id: str | None = None,
) -> PredictionMetricValues:
    """Compute gains from complete base/tuned item predictions, optionally by capability."""

    values = PredictionMetricValues(
        training_utility=compute_prediction_metric(artifact, "training_utility", capability_id=capability_id),
        raw_capability_gain=compute_prediction_metric(artifact, "raw_capability_gain", capability_id=capability_id),
        cross_form_probe_gain=compute_prediction_metric(
            artifact,
            "cross_form_probe_gain",
            capability_id=capability_id,
        ),
        target_capability_probe_gain=compute_prediction_metric(
            artifact,
            "target_capability_probe_gain",
            capability_id=capability_id,
        ),
        target_minus_nontarget_probe_gain=compute_prediction_metric(
            artifact,
            "target_minus_nontarget_probe_gain",
            capability_id=capability_id,
        ),
    )
    return values


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0 + 1.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = rank
        position = end
    return ranks


def _margin_aware_ranks(values: Sequence[float], margin: float) -> list[float]:
    if margin <= 0:
        return _average_ranks(values)
    # Connected components on the ordered scale make the tie relation
    # deterministic even when a chain of adjacent gaps lies inside the margin.
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] - values[order[end - 1]] <= margin:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for ordered_index in order[start:end]:
            ranks[ordered_index] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation requires at least two paired values")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered) * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        raise ValueError("Spearman is undefined when proxy or FOM ranks are constant")
    return float(sum(a * b for a, b in zip(left_centered, right_centered)) / denominator)


def _compute_ranking_rows(
    rows: Sequence[BatchUtilityPair],
    *,
    selection_k: int,
    low_value_threshold: float,
    indifference_margin: float,
) -> RankingMetricValues:
    if len(rows) < 2:
        raise ValueError("Ranking metrics require at least two paired batches")
    if selection_k < 1 or selection_k >= len(rows):
        raise ValueError("selection_k must be between one and one less than the evaluated batch count")
    ordered_rows = sorted(rows, key=lambda row: row.batch_id)
    proxy = [row.proxy_utility for row in ordered_rows]
    fom = [row.fom_utility for row in ordered_rows]
    spearman = _pearson(_average_ranks(proxy), _margin_aware_ranks(fom, indifference_margin))

    pairwise_scores: list[float] = []
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            fom_delta = fom[first] - fom[second]
            if abs(fom_delta) <= indifference_margin:
                continue
            fom_sign = 1 if fom_delta > 0 else -1
            proxy_delta = proxy[first] - proxy[second]
            proxy_sign = 1 if proxy_delta > 0 else -1 if proxy_delta < 0 else 0
            pairwise_scores.append(0.5 if proxy_sign == 0 else float(proxy_sign == fom_sign))
    if not pairwise_scores:
        raise ValueError("Pairwise accuracy requires at least one FOM pair outside the indifference margin")

    proxy_order = sorted(ordered_rows, key=lambda row: (-row.proxy_utility, row.batch_id))
    selected_ids = {row.batch_id for row in proxy_order[:selection_k]}
    proxy_top = proxy_order[0]
    top1_regret = max(row.fom_utility for row in rows) - proxy_top.fom_utility
    low_value_ids = {row.batch_id for row in rows if row.fom_utility <= low_value_threshold}
    if not low_value_ids:
        raise ValueError("False-promotion rate requires at least one FOM-low-value batch")
    # This is a false-discovery proportion: among batches the proxy would buy,
    # how many are low-value under the expensive FOM reference.  Dividing by
    # all low-value batches would perversely make the metric improve merely by
    # adding more bad batches to the candidate pool.
    false_promotion_rate = len(selected_ids & low_value_ids) / selection_k
    values = RankingMetricValues(
        spearman=spearman,
        pairwise_accuracy=sum(pairwise_scores) / len(pairwise_scores),
        top1_regret=max(0.0, top1_regret),
        low_value_false_promotion_rate=false_promotion_rate,
    )
    validate_metric_value("spearman_to_fom", values.spearman)
    validate_metric_value("pairwise_accuracy_to_fom", values.pairwise_accuracy)
    validate_metric_value("top1_regret_to_fom", values.top1_regret)
    validate_metric_value("low_value_false_promotion_rate", values.low_value_false_promotion_rate)
    return values


def compute_ranking_metrics(artifact: RankingUtilityAudit) -> RankingMetricValues:
    """Compute decision metrics from paired proxy/FOM batch utilities."""

    return _compute_ranking_rows(
        artifact.rows,
        selection_k=artifact.selection_k,
        low_value_threshold=artifact.low_value_threshold,
        indifference_margin=artifact.indifference_margin,
    )


def compute_leave_one_lineage_out_metrics(artifact: RankingUtilityAudit) -> list[LeaveOneLineageOutMetrics]:
    """Recompute ranking metrics after excluding each lineage in turn."""

    lineages = sorted({row.lineage_group for row in artifact.rows})
    if len(lineages) < 2:
        raise ValueError("Leave-one-lineage-out metrics require at least two lineage groups")
    results = []
    for lineage in lineages:
        remaining = [row for row in artifact.rows if row.lineage_group != lineage]
        results.append(
            LeaveOneLineageOutMetrics(
                excluded_lineage_group=lineage,
                metrics=_compute_ranking_rows(
                    remaining,
                    selection_k=artifact.selection_k,
                    low_value_threshold=artifact.low_value_threshold,
                    indifference_margin=artifact.indifference_margin,
                ),
            )
        )
    return results


ArtifactT = TypeVar("ArtifactT", bound=BaseModel)


def _load_json_artifact(path: str | Path, model_type: type[ArtifactT]) -> tuple[Path, ArtifactT]:
    artifact_path = Path(path).resolve()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Artifact must contain valid JSON: {artifact_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {artifact_path}")
    return artifact_path, model_type.model_validate(payload)


def load_static_selection_audit(path: str | Path) -> StaticSelectionAudit:
    return _load_json_artifact(path, StaticSelectionAudit)[1]


def load_paired_prediction_audit(path: str | Path) -> PairedPredictionAudit:
    return _load_json_artifact(path, PairedPredictionAudit)[1]


def load_ranking_utility_audit(path: str | Path) -> RankingUtilityAudit:
    return _load_json_artifact(path, RankingUtilityAudit)[1]


_COMPUTED_METRICS = {
    "condition_exception_recall",
    "unsupported_medical_assertion_rate",
    "dependency_group_recall",
    "capability_mapping_macro_f1",
    "weighted_obligation_coverage",
    "condition_exception_coverage",
    "quality_distribution_similarity",
    "blueprint_alignment",
    "blueprint_alignment_score",
    "semantic_facility",
    "acceptance_rate_absolute_error",
    "prior_gap_selection_alignment",
    "training_utility",
    "raw_capability_gain",
    "cross_form_probe_gain",
    "target_capability_probe_gain",
    "target_minus_nontarget_probe_gain",
    "spearman_to_fom",
    "pairwise_accuracy_to_fom",
    "top1_regret_to_fom",
    "low_value_false_promotion_rate",
    "low_value_promotion_rate",
    "leave_one_lineage_out_spearman_to_fom",
    "leave_one_lineage_out_pairwise_accuracy_to_fom",
    "leave_one_lineage_out_top1_regret_to_fom",
    "leave_one_lineage_out_low_value_false_promotion_rate",
    "leave_one_lineage_out_low_value_promotion_rate_to_fom",
}


def protocol_metric_coverage(protocol: AblationProtocol) -> dict[str, list[str]]:
    """Report and enforce that every frozen endpoint has a raw-data calculator."""

    required = sorted(
        {
            endpoint.metric
            for study in protocol.studies
            for endpoint in [*study.cell_gates, *study.contrasts, *study.interactions]
        }
    )
    unregistered = sorted(set(required) - set(METRIC_REGISTRY))
    unimplemented = sorted(set(required) - _COMPUTED_METRICS)
    if unregistered or unimplemented:
        raise ValueError(
            "Frozen ablation protocol contains metrics without raw-artifact calculators; "
            f"unregistered={unregistered}, unimplemented={unimplemented}"
        )
    return {"required": required, "implemented": required, "unregistered": [], "unimplemented": []}


def _load_execution_manifest(
    path: str | Path,
    *,
    protocol_sha256: str,
    artifact_path: Path,
    artifact_sha256: str,
    study_id: str,
    cell_id: str,
) -> tuple[Path, str]:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Execution manifest must contain valid JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Execution manifest must contain a JSON object: {manifest_path}")
    schema_version = payload.get("schema_version")
    if schema_version == "stage-ablation-audit-manifest-v1":
        manifest = AuditExecutionManifest.model_validate(payload)
        if manifest.study_id != study_id or manifest.cell_id != cell_id:
            raise ValueError(
                "Audit execution manifest does not match the requested cell: "
                f"expected={study_id}/{cell_id}, observed={manifest.study_id}/{manifest.cell_id}"
            )
        referenced_artifact = Path(manifest.artifact_uri)
        if not referenced_artifact.is_absolute():
            referenced_artifact = manifest_path.parent / referenced_artifact
        if referenced_artifact.resolve() != artifact_path:
            raise ValueError("Audit execution manifest artifact_uri does not resolve to the supplied artifact")
        if manifest.artifact_sha256 != artifact_sha256:
            raise ValueError("Audit execution manifest artifact_sha256 does not match the supplied artifact")
    elif schema_version == "stage-ablation-materialization-v1":
        manifest = MaterializationExecutionManifest.model_validate(payload)
        jobs_path = Path(manifest.jobs_uri)
        if not jobs_path.is_absolute():
            jobs_path = manifest_path.parent / jobs_path
        if not jobs_path.is_file() or file_sha256(jobs_path) != manifest.jobs_sha256:
            raise ValueError("Materialization execution manifest jobs artifact is missing or hash-mismatched")
    else:
        raise ValueError(f"Unsupported execution manifest schema_version: {schema_version!r}")
    if manifest.protocol_sha256 != protocol_sha256:
        raise ValueError("Execution manifest protocol_sha256 does not match the supplied frozen protocol")
    return manifest_path, file_sha256(manifest_path)


def _registered_metrics_for_cell(protocol: AblationProtocol, study_id: str, cell_id: str) -> set[str]:
    studies = {study.study_id: study for study in protocol.studies}
    try:
        study = studies[study_id]
    except KeyError as error:
        raise ValueError(f"Unknown ablation study: {study_id}") from error
    if cell_id not in {cell.cell_id for cell in study.cells}:
        raise ValueError(f"Unknown ablation cell: {study_id}/{cell_id}")
    metrics = {gate.metric for gate in study.cell_gates if gate.cell_id == cell_id}
    for endpoint in study.contrasts:
        if cell_id in {endpoint.treatment_cell_id, endpoint.control_cell_id}:
            metrics.add(endpoint.metric)
    for endpoint in study.interactions:
        if cell_id in {
            endpoint.first_treatment_cell_id,
            endpoint.first_control_cell_id,
            endpoint.second_treatment_cell_id,
            endpoint.second_control_cell_id,
        }:
            metrics.add(endpoint.metric)
    return metrics


def _prepare_collection(
    protocol_path: str | Path,
    artifact_path: str | Path,
    execution_manifest_path: str | Path,
    model_type: type[ArtifactT],
    *,
    study_id: str,
    cell_id: str,
    requested_metrics: Sequence[str] | None,
    family: MetricFamily,
) -> tuple[AblationProtocol, ArtifactT, Path, str, str, list[str]]:
    protocol = load_ablation_protocol(protocol_path)
    protocol_metric_coverage(protocol)
    resolved_path, artifact = _load_json_artifact(artifact_path, model_type)
    artifact_digest = file_sha256(resolved_path)
    _, execution_manifest_digest = _load_execution_manifest(
        execution_manifest_path,
        protocol_sha256=ablation_protocol_digest(protocol),
        artifact_path=resolved_path,
        artifact_sha256=artifact_digest,
        study_id=study_id,
        cell_id=cell_id,
    )
    registered = _registered_metrics_for_cell(protocol, study_id, cell_id)
    supported = {name for name, spec in METRIC_REGISTRY.items() if spec.family == family}
    metrics = sorted(registered & supported) if requested_metrics is None else list(requested_metrics)
    if not metrics:
        raise ValueError(f"No computed {family} metrics are registered for {study_id}/{cell_id}")
    if len(metrics) != len(set(metrics)):
        raise ValueError("Requested metrics must be unique")
    for metric in metrics:
        spec = metric_spec(metric)
        if spec.family != family:
            raise ValueError(f"Metric {metric} is not a {family} metric")
        if metric not in registered:
            raise ValueError(f"Metric {metric} is not preregistered for {study_id}/{cell_id}")
    return protocol, artifact, resolved_path, artifact_digest, execution_manifest_digest, metrics


def _observation(
    *,
    protocol: AblationProtocol,
    artifact_path: Path,
    artifact_sha256: str,
    execution_manifest_sha256: str,
    study_id: str,
    cell_id: str,
    metric: str,
    batch_id: str,
    lineage_group: str,
    value: float,
    seed: int | None = None,
    capability_id: str | None = None,
    unit_id: str | None = None,
) -> AblationObservation:
    return AblationObservation(
        protocol_sha256=ablation_protocol_digest(protocol),
        study_id=study_id,
        cell_id=cell_id,
        metric=metric,
        batch_id=batch_id,
        lineage_group=lineage_group,
        seed=seed,
        capability_id=capability_id,
        unit_id=unit_id,
        value=validate_metric_value(metric, value),
        artifact_sha256=artifact_sha256,
        artifact_uri=str(artifact_path),
        execution_manifest_sha256=execution_manifest_sha256,
    )


def collect_selection_observations(
    protocol_path: str | Path,
    artifact_path: str | Path,
    execution_manifest_path: str | Path,
    *,
    study_id: str,
    cell_id: str,
    metrics: Sequence[str] | None = None,
) -> list[AblationObservation]:
    """Load and hash a static artifact, then emit only recomputed metrics."""

    protocol, artifact, resolved_artifact_path, artifact_digest, execution_digest, requested = _prepare_collection(
        protocol_path,
        artifact_path,
        execution_manifest_path,
        StaticSelectionAudit,
        study_id=study_id,
        cell_id=cell_id,
        requested_metrics=metrics,
        family="selection",
    )
    return [
        _observation(
            protocol=protocol,
            artifact_path=resolved_artifact_path,
            artifact_sha256=artifact_digest,
            execution_manifest_sha256=execution_digest,
            study_id=study_id,
            cell_id=cell_id,
            metric=metric,
            batch_id=artifact.batch_id,
            lineage_group=artifact.lineage_group,
            value=compute_selection_metric(artifact, metric),
        )
        for metric in requested
    ]


def collect_prediction_observations(
    protocol_path: str | Path,
    artifact_path: str | Path,
    execution_manifest_path: str | Path,
    *,
    study_id: str,
    cell_id: str,
    metrics: Sequence[str] | None = None,
) -> list[AblationObservation]:
    """Emit protocol-bound overall or per-capability prediction endpoints."""

    protocol, artifact, resolved_artifact_path, artifact_digest, execution_digest, requested = _prepare_collection(
        protocol_path,
        artifact_path,
        execution_manifest_path,
        PairedPredictionAudit,
        study_id=study_id,
        cell_id=cell_id,
        requested_metrics=metrics,
        family="prediction",
    )
    observations = []
    for metric in requested:
        spec = metric_spec(metric)
        capability_ids: list[str | None] = artifact.catalog.capability_ids if spec.capability_level else [None]
        for capability_id in capability_ids:
            observations.append(
                _observation(
                    protocol=protocol,
                    artifact_path=resolved_artifact_path,
                    artifact_sha256=artifact_digest,
                    execution_manifest_sha256=execution_digest,
                    study_id=study_id,
                    cell_id=cell_id,
                    metric=metric,
                    batch_id=artifact.batch_id,
                    lineage_group=artifact.lineage_group,
                    seed=artifact.seed,
                    capability_id=capability_id,
                    value=compute_prediction_metric(artifact, metric, capability_id=capability_id),
                )
            )
    return observations


_LOO_RANKING_FIELDS = {
    "leave_one_lineage_out_spearman_to_fom": "spearman",
    "leave_one_lineage_out_pairwise_accuracy_to_fom": "pairwise_accuracy",
    "leave_one_lineage_out_top1_regret_to_fom": "top1_regret",
    "leave_one_lineage_out_low_value_false_promotion_rate": "low_value_false_promotion_rate",
    "leave_one_lineage_out_low_value_promotion_rate_to_fom": "low_value_false_promotion_rate",
}
_GLOBAL_RANKING_FIELDS = {
    "spearman_to_fom": "spearman",
    "pairwise_accuracy_to_fom": "pairwise_accuracy",
    "top1_regret_to_fom": "top1_regret",
    "low_value_false_promotion_rate": "low_value_false_promotion_rate",
    "low_value_promotion_rate": "low_value_false_promotion_rate",
}


def collect_ranking_observations(
    protocol_path: str | Path,
    artifact_path: str | Path,
    execution_manifest_path: str | Path,
    *,
    study_id: str,
    cell_id: str,
    metrics: Sequence[str] | None = None,
) -> list[AblationObservation]:
    """Emit global or leave-one-lineage-out proxy/FOM decision metrics."""

    protocol, artifact, resolved_artifact_path, artifact_digest, execution_digest, requested = _prepare_collection(
        protocol_path,
        artifact_path,
        execution_manifest_path,
        RankingUtilityAudit,
        study_id=study_id,
        cell_id=cell_id,
        requested_metrics=metrics,
        family="ranking",
    )
    observations = []
    global_values: RankingMetricValues | None = None
    loo_values: list[LeaveOneLineageOutMetrics] | None = None
    for metric in requested:
        if metric in _GLOBAL_RANKING_FIELDS:
            if global_values is None:
                global_values = compute_ranking_metrics(artifact)
            observations.append(
                _observation(
                    protocol=protocol,
                    artifact_path=resolved_artifact_path,
                    artifact_sha256=artifact_digest,
                    execution_manifest_sha256=execution_digest,
                    study_id=study_id,
                    cell_id=cell_id,
                    metric=metric,
                    batch_id="all-batches",
                    lineage_group="all-lineages",
                    value=getattr(global_values, _GLOBAL_RANKING_FIELDS[metric]),
                )
            )
        elif metric in _LOO_RANKING_FIELDS:
            if loo_values is None:
                loo_values = compute_leave_one_lineage_out_metrics(artifact)
            for fold in loo_values:
                observations.append(
                    _observation(
                        protocol=protocol,
                        artifact_path=resolved_artifact_path,
                        artifact_sha256=artifact_digest,
                        execution_manifest_sha256=execution_digest,
                        study_id=study_id,
                        cell_id=cell_id,
                        metric=metric,
                        batch_id=f"leave-out:{fold.excluded_lineage_group}",
                        lineage_group=fold.excluded_lineage_group,
                        unit_id=f"excluded_lineage:{fold.excluded_lineage_group}",
                        value=getattr(fold.metrics, _LOO_RANKING_FIELDS[metric]),
                    )
                )
        else:  # guarded by the registry, retained as a defensive invariant
            raise ValueError(f"Ranking metric has no computation binding: {metric}")
    return observations


__all__ = [
    "AssertionAdjudication",
    "AuditExecutionManifest",
    "BatchUtilityPair",
    "BlueprintWeight",
    "CapabilityMappingAnnotation",
    "ExtractedMiningGroup",
    "FacilityTarget",
    "GenerationAttempt",
    "ItemPrediction",
    "LeaveOneLineageOutMetrics",
    "METRIC_REGISTRY",
    "MaterializationExecutionManifest",
    "MetricSpec",
    "MiningGoldDependencyGroup",
    "MiningGoldElement",
    "PairedPredictionAudit",
    "PredictionMetricValues",
    "PredictionRun",
    "ProbeCatalog",
    "ProbeItem",
    "RankingMetricValues",
    "RankingUtilityAudit",
    "SelectionMetricValues",
    "StaticAuditObligation",
    "StaticAuditUnit",
    "StaticSelectionAudit",
    "collect_prediction_observations",
    "collect_ranking_observations",
    "collect_selection_observations",
    "compute_leave_one_lineage_out_metrics",
    "compute_prediction_metric",
    "compute_prediction_metrics",
    "compute_ranking_metrics",
    "compute_selection_metric",
    "compute_selection_metrics",
    "load_paired_prediction_audit",
    "load_ranking_utility_audit",
    "load_static_selection_audit",
    "metric_spec",
    "protocol_metric_coverage",
    "validate_metric_value",
]
