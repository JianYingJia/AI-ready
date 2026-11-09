"""Preregister and analyse stage-level ablations for the proxy-data mechanism.

The operational proxy-versus-Fully-Open-Meditron experiment answers whether a
cheap pipeline predicts an expensive one.  It cannot, by itself, attribute the
agreement to mining, blueprint conditioning, selection, supervision form, or
model learning.  This module keeps those claims separate by requiring paired
cells that differ only in preregistered factors and by clustering uncertainty
at the source-lineage (or batch) level.

The module does not run a generator or a trainer.  It freezes the comparisons
those systems must execute and rejects incomplete or unpaired result tables.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.stats import t as student_t

from ai_ready.valuation.contracts import canonical_digest, file_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


FactorName = Literal[
    "mining",
    "blueprint",
    "selector",
    "organization",
    "source_budget_fraction",
    "repetition_factor",
    "noise_kind",
    "noise_rate",
    "base_model",
    "prior_profile",
    "training_mode",
    "generator",
]

PairingKey = Literal["batch_id", "seed", "capability_id", "unit_id"]

_METRIC_RANGES: dict[str, tuple[float, float]] = {
    "condition_exception_recall": (0.0, 1.0),
    "unsupported_medical_assertion_rate": (0.0, 1.0),
    "dependency_group_recall": (0.0, 1.0),
    "capability_mapping_macro_f1": (0.0, 1.0),
    "blueprint_alignment_score": (-1.0, 1.0),
    "weighted_obligation_coverage": (0.0, 1.0),
    "condition_exception_coverage": (0.0, 1.0),
    "semantic_facility": (0.0, 1.0),
    "quality_distribution_similarity": (0.0, 1.0),
    "acceptance_rate_absolute_error": (0.0, 1.0),
    "training_utility": (-1.0, 1.0),
    "cross_form_probe_gain": (-1.0, 1.0),
    "target_minus_nontarget_probe_gain": (-2.0, 2.0),
    "prior_gap_selection_alignment": (-1.0, 1.0),
    "raw_capability_gain": (-1.0, 1.0),
    "target_capability_probe_gain": (-1.0, 1.0),
    "leave_one_lineage_out_spearman_to_fom": (-1.0, 1.0),
    "leave_one_lineage_out_pairwise_accuracy_to_fom": (0.0, 1.0),
    "leave_one_lineage_out_top1_regret_to_fom": (0.0, 1.0),
    "leave_one_lineage_out_low_value_promotion_rate_to_fom": (0.0, 1.0),
    "low_value_promotion_rate": (0.0, 1.0),
}


class PipelineFactors(_StrictModel):
    """The factors whose changes are allowed to support a mechanism claim."""

    mining: str = Field(min_length=1)
    blueprint: str = Field(min_length=1)
    selector: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    source_budget_fraction: float = Field(gt=0, le=1)
    repetition_factor: float = Field(gt=0)
    noise_kind: str = Field(min_length=1)
    noise_rate: float = Field(ge=0, le=1)
    base_model: str = Field(min_length=1)
    prior_profile: str = Field(min_length=1)
    training_mode: Literal["no_sft", "sft"]
    generator: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_noise_pair(self):
        if self.noise_kind == "none" and self.noise_rate != 0:
            raise ValueError("noise_kind=none requires noise_rate=0")
        if self.noise_kind != "none" and self.noise_rate <= 0:
            raise ValueError("A non-none noise_kind requires a positive noise_rate")
        return self


class AblationCell(_StrictModel):
    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = Field(min_length=1)
    factors: PipelineFactors


class CellGate(_StrictModel):
    """An absolute target that one cell must meet with a cluster-level CI."""

    gate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    cell_id: str = Field(min_length=1)
    metric: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    direction: Literal["at_least", "at_most"]
    threshold: float
    independence_key: Literal["lineage_group", "batch_id"] = "lineage_group"
    minimum_observations: int = Field(ge=2)
    minimum_independent_units: int = Field(ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    multiplicity_family: str = Field(default="primary", min_length=1)
    primary: bool = True
    rationale: str = Field(min_length=1)


class PairedContrast(_StrictModel):
    """A paired treatment-minus-control estimand."""

    contrast_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    treatment_cell_id: str = Field(min_length=1)
    control_cell_id: str = Field(min_length=1)
    metric: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    pairing_keys: list[PairingKey] = Field(min_length=1)
    independence_key: Literal["lineage_group", "batch_id"] = "lineage_group"
    direction: Literal["higher", "lower", "equivalent", "noninferior", "different"]
    minimum_effect: float = Field(default=0.0, ge=0)
    equivalence_margin: float | None = Field(default=None, gt=0)
    noninferiority_margin: float | None = Field(default=None, gt=0)
    minimum_pairs: int = Field(ge=2)
    minimum_independent_units: int = Field(ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    multiplicity_family: str = Field(default="primary", min_length=1)
    primary: bool = True
    rationale: str = Field(min_length=1)

    @field_validator("pairing_keys")
    @classmethod
    def unique_pairing_keys(cls, value: list[PairingKey]) -> list[PairingKey]:
        if len(value) != len(set(value)):
            raise ValueError("pairing_keys must be unique")
        if "batch_id" not in value:
            raise ValueError("pairing_keys must include batch_id")
        return value

    @model_validator(mode="after")
    def validate_direction_parameters(self):
        if self.treatment_cell_id == self.control_cell_id:
            raise ValueError("A paired contrast requires different treatment and control cells")
        if self.direction == "equivalent":
            if self.equivalence_margin is None:
                raise ValueError("An equivalence contrast requires equivalence_margin")
            if self.minimum_effect != 0:
                raise ValueError("An equivalence contrast cannot also require minimum_effect")
            if self.noninferiority_margin is not None:
                raise ValueError("An equivalence contrast cannot also require noninferiority_margin")
        elif self.direction == "noninferior":
            if self.noninferiority_margin is None:
                raise ValueError("A noninferiority contrast requires noninferiority_margin")
            if self.minimum_effect != 0:
                raise ValueError("A noninferiority contrast cannot also require minimum_effect")
            if self.equivalence_margin is not None:
                raise ValueError("A noninferiority contrast cannot also require equivalence_margin")
        elif self.equivalence_margin is not None or self.noninferiority_margin is not None:
            raise ValueError(
                "equivalence_margin is only valid for direction=equivalent and "
                "noninferiority_margin is only valid for direction=noninferior"
            )
        return self


class InteractionContrast(_StrictModel):
    """Difference in paired treatment effects across two preregistered contexts."""

    contrast_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    first_treatment_cell_id: str = Field(min_length=1)
    first_control_cell_id: str = Field(min_length=1)
    second_treatment_cell_id: str = Field(min_length=1)
    second_control_cell_id: str = Field(min_length=1)
    metric: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    pairing_keys: list[PairingKey] = Field(min_length=1)
    independence_key: Literal["lineage_group", "batch_id"] = "lineage_group"
    direction: Literal["higher", "lower", "equivalent", "different"]
    minimum_effect: float = Field(default=0.0, ge=0)
    equivalence_margin: float | None = Field(default=None, gt=0)
    minimum_pairs: int = Field(ge=2)
    minimum_independent_units: int = Field(ge=2)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    multiplicity_family: str = Field(default="primary", min_length=1)
    primary: bool = True
    rationale: str = Field(min_length=1)

    @field_validator("pairing_keys")
    @classmethod
    def unique_pairing_keys(cls, value: list[PairingKey]) -> list[PairingKey]:
        if len(value) != len(set(value)):
            raise ValueError("pairing_keys must be unique")
        if "batch_id" not in value:
            raise ValueError("pairing_keys must include batch_id")
        return value

    @model_validator(mode="after")
    def validate_interaction(self):
        cell_ids = {
            self.first_treatment_cell_id,
            self.first_control_cell_id,
            self.second_treatment_cell_id,
            self.second_control_cell_id,
        }
        if len(cell_ids) != 4:
            raise ValueError("An interaction contrast requires four distinct cells")
        if self.direction == "equivalent":
            if self.equivalence_margin is None:
                raise ValueError("An equivalent interaction requires equivalence_margin")
            if self.minimum_effect != 0:
                raise ValueError("An equivalent interaction cannot also require minimum_effect")
        elif self.equivalence_margin is not None:
            raise ValueError("equivalence_margin is only valid for direction=equivalent")
        return self


class AblationStudy(_StrictModel):
    study_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    stage: Literal[
        "semantic_mining",
        "blueprint_conditioning",
        "subset_selection",
        "supervision_organization",
        "learnability_bridge",
        "scale_repeat_noise",
        "base_model_prior",
        "operational_external_validity",
    ]
    phase: Literal["screening", "confirmation"]
    evidence_role: Literal["static_mechanism", "training_mechanism", "robustness", "operational_prediction"]
    target: str = Field(min_length=1)
    causal_interpretation: bool
    isolated_factors: list[FactorName] = Field(min_length=1)
    cells: list[AblationCell] = Field(min_length=1)
    cell_gates: list[CellGate] = Field(default_factory=list)
    contrasts: list[PairedContrast] = Field(default_factory=list)
    interactions: list[InteractionContrast] = Field(default_factory=list)

    @field_validator("isolated_factors")
    @classmethod
    def unique_isolated_factors(cls, value: list[FactorName]) -> list[FactorName]:
        if len(value) != len(set(value)):
            raise ValueError("isolated_factors must be unique")
        return value

    @model_validator(mode="after")
    def validate_study_cells_and_endpoints(self):
        cells = {cell.cell_id: cell for cell in self.cells}
        if len(cells) != len(self.cells):
            raise ValueError(f"Study {self.study_id} contains duplicate cell IDs")
        endpoint_ids = (
            [gate.gate_id for gate in self.cell_gates]
            + [row.contrast_id for row in self.contrasts]
            + [row.contrast_id for row in self.interactions]
        )
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError(f"Study {self.study_id} contains duplicate endpoint IDs")
        if not self.cell_gates and not self.contrasts and not self.interactions:
            raise ValueError(f"Study {self.study_id} must register at least one endpoint")
        if self.causal_interpretation and not self.contrasts and not self.interactions:
            raise ValueError(f"Causal study {self.study_id} requires at least one paired contrast or interaction")
        if not any(row.primary for row in [*self.cell_gates, *self.contrasts, *self.interactions]):
            raise ValueError(f"Study {self.study_id} must register at least one primary endpoint")
        for gate in self.cell_gates:
            if gate.cell_id not in cells:
                raise ValueError(f"Gate {gate.gate_id} references unknown cell {gate.cell_id}")
        factor_fields = set(PipelineFactors.model_fields)
        for contrast in self.contrasts:
            if contrast.treatment_cell_id not in cells or contrast.control_cell_id not in cells:
                raise ValueError(f"Contrast {contrast.contrast_id} references an unknown cell")
            treatment = cells[contrast.treatment_cell_id].factors.model_dump(mode="json")
            control = cells[contrast.control_cell_id].factors.model_dump(mode="json")
            changed = {field for field in factor_fields if treatment[field] != control[field]}
            if not changed:
                raise ValueError(f"Contrast {contrast.contrast_id} compares identical factor settings")
            unexpected = changed - set(self.isolated_factors)
            if unexpected:
                raise ValueError(f"Contrast {contrast.contrast_id} changes unregistered factors: {sorted(unexpected)}")
            if self.causal_interpretation and treatment["generator"] != control["generator"]:
                raise ValueError(f"Causal contrast {contrast.contrast_id} cannot compare different SFT generators")
        for interaction in self.interactions:
            interaction_cell_ids = {
                interaction.first_treatment_cell_id,
                interaction.first_control_cell_id,
                interaction.second_treatment_cell_id,
                interaction.second_control_cell_id,
            }
            if unknown := interaction_cell_ids - set(cells):
                raise ValueError(f"Interaction {interaction.contrast_id} references unknown cells: {sorted(unknown)}")
            factor_rows = [cells[cell_id].factors.model_dump(mode="json") for cell_id in interaction_cell_ids]
            changed = {
                field
                for field in factor_fields
                if len({str(row[field]) for row in factor_rows}) > 1
            }
            unexpected = changed - set(self.isolated_factors)
            if unexpected:
                raise ValueError(
                    f"Interaction {interaction.contrast_id} changes unregistered factors: {sorted(unexpected)}"
                )
            if self.causal_interpretation and len({row["generator"] for row in factor_rows}) != 1:
                raise ValueError(
                    f"Causal interaction {interaction.contrast_id} cannot compare different SFT generators"
                )
        if self.causal_interpretation and any(cell.factors.generator == "fully_open_meditron" for cell in self.cells):
            raise ValueError("Fully Open Meditron is an operational reference, not a causal ablation cell")
        if self.stage == "operational_external_validity":
            if self.causal_interpretation:
                raise ValueError("The cross-pipeline operational study cannot claim causal interpretation")
            if self.evidence_role != "operational_prediction":
                raise ValueError("The operational stage must use evidence_role=operational_prediction")
        return self


class AblationProtocol(_StrictModel):
    schema_version: Literal["stage-ablation-protocol-v1"] = "stage-ablation-protocol-v1"
    protocol_id: str = Field(min_length=1)
    inference_scope: Literal["descriptive_mechanism_screening", "held_out_mechanism_confirmation"]
    locked_before_results: Literal[True] = True
    benchmark_test_items_accessible_to_generation: Literal[False] = False
    fully_open_meditron_role: Literal["operational_external_reference_only"] = "operational_external_reference_only"
    multiplicity_correction: Literal["bonferroni"] = "bonferroni"
    studies: list[AblationStudy] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_protocol_coverage(self):
        study_ids = [study.study_id for study in self.studies]
        if len(study_ids) != len(set(study_ids)):
            raise ValueError("Ablation study IDs must be unique")
        required = {
            "semantic_mining",
            "blueprint_conditioning",
            "subset_selection",
            "supervision_organization",
            "learnability_bridge",
            "scale_repeat_noise",
            "base_model_prior",
            "operational_external_validity",
        }
        observed = {study.stage for study in self.studies}
        if missing := required - observed:
            raise ValueError(f"Ablation protocol is missing required mechanism stages: {sorted(missing)}")
        return self


class AblationObservation(_StrictModel):
    """One metric observation bound to a frozen protocol and source artifact."""

    schema_version: Literal["stage-ablation-observation-v1"] = "stage-ablation-observation-v1"
    protocol_sha256: str
    study_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    seed: int | None = None
    capability_id: str | None = None
    unit_id: str | None = None
    value: float
    artifact_sha256: str
    artifact_uri: str | None = None
    execution_manifest_sha256: str | None = None

    @field_validator("protocol_sha256", "artifact_sha256", "execution_manifest_sha256")
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"{info.field_name} must be a 64-character SHA-256 digest")
        return normalized

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Ablation observation values must be finite")
        return value

    @model_validator(mode="after")
    def validate_registered_metric_and_artifact(self):
        if bounds := _METRIC_RANGES.get(self.metric):
            lower, upper = bounds
            if not lower <= self.value <= upper:
                raise ValueError(f"Metric {self.metric} must be within [{lower}, {upper}]")
        if self.artifact_uri is not None:
            artifact_path = Path(self.artifact_uri).resolve()
            if not artifact_path.is_file():
                raise ValueError(f"Ablation observation artifact does not exist: {artifact_path}")
            observed = file_sha256(artifact_path)
            if observed != self.artifact_sha256:
                raise ValueError(
                    f"Ablation observation artifact hash mismatch: expected={self.artifact_sha256}, "
                    f"observed={observed}"
                )
        return self


def load_ablation_protocol(path: str | Path) -> AblationProtocol:
    protocol_path = Path(path)
    payload = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ablation protocol must contain a YAML mapping: {protocol_path}")
    return AblationProtocol.model_validate(payload)


def ablation_protocol_digest(protocol: AblationProtocol) -> str:
    return canonical_digest(protocol)


def ablation_matrix(protocol: AblationProtocol) -> list[dict]:
    """Return a review-friendly flat matrix without inventing run identities."""

    rows = []
    for study in protocol.studies:
        for cell in study.cells:
            rows.append(
                {
                    "study_id": study.study_id,
                    "stage": study.stage,
                    "phase": study.phase,
                    "evidence_role": study.evidence_role,
                    "causal_interpretation": study.causal_interpretation,
                    "isolated_factors": study.isolated_factors,
                    "cell_id": cell.cell_id,
                    "description": cell.description,
                    **cell.factors.model_dump(mode="json"),
                }
            )
    return rows


def endpoint_catalog(protocol: AblationProtocol) -> list[dict]:
    """List every required metric and its frozen decision rule for server runners."""

    rows = []
    for study in protocol.studies:
        for gate in study.cell_gates:
            rows.append(
                {
                    "study_id": study.study_id,
                    "stage": study.stage,
                    "endpoint_type": "cell_gate",
                    "endpoint_id": gate.gate_id,
                    "metric": gate.metric,
                    "required_cell_ids": [gate.cell_id],
                    "pairing_keys": [],
                    "independence_key": gate.independence_key,
                    "direction": gate.direction,
                    "decision_value": gate.threshold,
                    "minimum_observations": gate.minimum_observations,
                    "minimum_independent_units": gate.minimum_independent_units,
                    "primary": gate.primary,
                }
            )
        for contrast in study.contrasts:
            rows.append(
                {
                    "study_id": study.study_id,
                    "stage": study.stage,
                    "endpoint_type": "paired_contrast",
                    "endpoint_id": contrast.contrast_id,
                    "metric": contrast.metric,
                    "required_cell_ids": [contrast.treatment_cell_id, contrast.control_cell_id],
                    "pairing_keys": contrast.pairing_keys,
                    "independence_key": contrast.independence_key,
                    "direction": contrast.direction,
                    "decision_value": (
                        contrast.equivalence_margin
                        if contrast.direction == "equivalent"
                        else (
                            contrast.noninferiority_margin
                            if contrast.direction == "noninferior"
                            else contrast.minimum_effect
                        )
                    ),
                    "minimum_observations": contrast.minimum_pairs * 2,
                    "minimum_independent_units": contrast.minimum_independent_units,
                    "primary": contrast.primary,
                }
            )
        for interaction in study.interactions:
            rows.append(
                {
                    "study_id": study.study_id,
                    "stage": study.stage,
                    "endpoint_type": "interaction_contrast",
                    "endpoint_id": interaction.contrast_id,
                    "metric": interaction.metric,
                    "required_cell_ids": [
                        interaction.first_treatment_cell_id,
                        interaction.first_control_cell_id,
                        interaction.second_treatment_cell_id,
                        interaction.second_control_cell_id,
                    ],
                    "pairing_keys": interaction.pairing_keys,
                    "independence_key": interaction.independence_key,
                    "direction": interaction.direction,
                    "decision_value": (
                        interaction.equivalence_margin
                        if interaction.direction == "equivalent"
                        else interaction.minimum_effect
                    ),
                    "minimum_observations": interaction.minimum_pairs * 4,
                    "minimum_independent_units": interaction.minimum_independent_units,
                    "primary": interaction.primary,
                }
            )
    return rows


def protocol_summary(protocol: AblationProtocol) -> dict:
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": ablation_protocol_digest(protocol),
        "inference_scope": protocol.inference_scope,
        "study_count": len(protocol.studies),
        "cell_count": sum(len(study.cells) for study in protocol.studies),
        "endpoint_count": sum(
            len(study.cell_gates) + len(study.contrasts) + len(study.interactions) for study in protocol.studies
        ),
        "primary_endpoint_count": sum(
            sum(gate.primary for gate in study.cell_gates)
            + sum(row.primary for row in study.contrasts)
            + sum(row.primary for row in study.interactions)
            for study in protocol.studies
        ),
        "stages": sorted({study.stage for study in protocol.studies}),
        "causal_studies": [study.study_id for study in protocol.studies if study.causal_interpretation],
        "operational_studies": [
            study.study_id for study in protocol.studies if study.evidence_role == "operational_prediction"
        ],
    }


def _cluster_interval(values: list[float], alpha: float) -> dict:
    if len(values) < 2:
        raise ValueError("At least two independent units are required for an interval")
    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values))
    half_width = float(student_t.ppf(1 - alpha / 2, df=len(values) - 1) * std / math.sqrt(len(values)))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "independent_unit_count": len(values),
    }


def _adjusted_alpha(endpoint, family_sizes: dict[str, int]) -> float:
    return float(endpoint.alpha / family_sizes[endpoint.multiplicity_family])


def _cluster_values(rows: list[AblationObservation], independence_key: str) -> list[float]:
    clustered: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clustered[str(getattr(row, independence_key))].append(row.value)
    return [float(statistics.mean(values)) for _, values in sorted(clustered.items())]


def _positive_effect_status(ci_low: float, ci_high: float, minimum_effect: float) -> str:
    """Decide a directional positive effect without treating exact zero as success."""

    if minimum_effect == 0:
        return "passed" if ci_low > 0 else "failed" if ci_high <= 0 else "insufficient"
    return "passed" if ci_low >= minimum_effect else "failed" if ci_high < minimum_effect else "insufficient"


def _negative_effect_status(ci_low: float, ci_high: float, minimum_effect: float) -> str:
    """Decide a directional negative effect without treating exact zero as success."""

    if minimum_effect == 0:
        return "passed" if ci_high < 0 else "failed" if ci_low >= 0 else "insufficient"
    return "passed" if ci_high <= -minimum_effect else "failed" if ci_low > -minimum_effect else "insufficient"


def _different_effect_status(ci_low: float, ci_high: float, minimum_effect: float) -> str:
    """Require an interval wholly outside the no-effect region."""

    if minimum_effect == 0:
        return (
            "passed"
            if ci_low > 0 or ci_high < 0
            else "failed" if ci_low == 0 and ci_high == 0 else "insufficient"
        )
    return (
        "passed"
        if ci_low >= minimum_effect or ci_high <= -minimum_effect
        else (
            "failed"
            if ci_low >= -minimum_effect and ci_high <= minimum_effect
            else "insufficient"
        )
    )


def _analyse_gate(
    study: AblationStudy,
    gate: CellGate,
    observations: list[AblationObservation],
    adjusted_alpha: float,
) -> dict:
    rows = [
        row
        for row in observations
        if row.study_id == study.study_id and row.cell_id == gate.cell_id and row.metric == gate.metric
    ]
    if len(rows) < gate.minimum_observations:
        raise ValueError(f"Gate {gate.gate_id} has {len(rows)} observations; requires {gate.minimum_observations}")
    cluster_values = _cluster_values(rows, gate.independence_key)
    if len(cluster_values) < gate.minimum_independent_units:
        raise ValueError(
            f"Gate {gate.gate_id} has {len(cluster_values)} independent units; "
            f"requires {gate.minimum_independent_units}"
        )
    interval = _cluster_interval(cluster_values, adjusted_alpha)
    if gate.direction == "at_least":
        status = (
            "passed"
            if interval["ci_low"] >= gate.threshold
            else "failed" if interval["ci_high"] < gate.threshold else "insufficient"
        )
    else:
        status = (
            "passed"
            if interval["ci_high"] <= gate.threshold
            else "failed" if interval["ci_low"] > gate.threshold else "insufficient"
        )
    return {
        "endpoint_type": "cell_gate",
        "endpoint_id": gate.gate_id,
        "primary": gate.primary,
        "cell_id": gate.cell_id,
        "metric": gate.metric,
        "direction": gate.direction,
        "threshold": gate.threshold,
        "observation_count": len(rows),
        "alpha": gate.alpha,
        "multiplicity_adjusted_alpha": adjusted_alpha,
        **interval,
        "status": status,
        "passed": status == "passed",
    }


def _pair_key(row: AblationObservation, keys: list[PairingKey]) -> tuple:
    values = tuple(getattr(row, key) for key in keys)
    if any(value is None for value in values):
        raise ValueError(f"Observation {row.study_id}/{row.cell_id} has a null registered pairing key")
    return values


def _analyse_contrast(
    study: AblationStudy,
    contrast: PairedContrast,
    observations: list[AblationObservation],
    adjusted_alpha: float,
) -> dict:
    treatment_rows = [
        row
        for row in observations
        if row.study_id == study.study_id
        and row.cell_id == contrast.treatment_cell_id
        and row.metric == contrast.metric
    ]
    control_rows = [
        row
        for row in observations
        if row.study_id == study.study_id and row.cell_id == contrast.control_cell_id and row.metric == contrast.metric
    ]

    def indexed(rows: list[AblationObservation], cell_id: str) -> dict[tuple, AblationObservation]:
        result = {}
        for row in rows:
            key = _pair_key(row, contrast.pairing_keys)
            if key in result:
                raise ValueError(
                    f"Contrast {contrast.contrast_id} has duplicate {cell_id} observation for pairing key {key}"
                )
            result[key] = row
        return result

    treatment = indexed(treatment_rows, contrast.treatment_cell_id)
    control = indexed(control_rows, contrast.control_cell_id)
    if set(treatment) != set(control):
        missing_treatment = sorted(set(control) - set(treatment), key=str)
        missing_control = sorted(set(treatment) - set(control), key=str)
        raise ValueError(
            f"Contrast {contrast.contrast_id} is not fully paired; "
            f"missing_treatment={missing_treatment}, missing_control={missing_control}"
        )
    if len(treatment) < contrast.minimum_pairs:
        raise ValueError(
            f"Contrast {contrast.contrast_id} has {len(treatment)} pairs; requires {contrast.minimum_pairs}"
        )
    clustered: dict[str, list[float]] = defaultdict(list)
    for key in sorted(treatment, key=str):
        treated, untreated = treatment[key], control[key]
        if treated.batch_id != untreated.batch_id or treated.lineage_group != untreated.lineage_group:
            raise ValueError(f"Contrast {contrast.contrast_id} pairs observations from different source contexts")
        cluster = str(getattr(treated, contrast.independence_key))
        clustered[cluster].append(treated.value - untreated.value)
    cluster_values = [float(statistics.mean(values)) for _, values in sorted(clustered.items())]
    if len(cluster_values) < contrast.minimum_independent_units:
        raise ValueError(
            f"Contrast {contrast.contrast_id} has {len(cluster_values)} independent units; "
            f"requires {contrast.minimum_independent_units}"
        )
    interval = _cluster_interval(cluster_values, adjusted_alpha)
    if contrast.direction == "higher":
        status = _positive_effect_status(interval["ci_low"], interval["ci_high"], contrast.minimum_effect)
    elif contrast.direction == "lower":
        status = _negative_effect_status(interval["ci_low"], interval["ci_high"], contrast.minimum_effect)
    elif contrast.direction == "equivalent":
        margin = float(contrast.equivalence_margin)
        status = (
            "passed"
            if interval["ci_low"] >= -margin and interval["ci_high"] <= margin
            else "failed" if interval["ci_high"] < -margin or interval["ci_low"] > margin else "insufficient"
        )
    elif contrast.direction == "noninferior":
        margin = float(contrast.noninferiority_margin)
        status = (
            "passed"
            if interval["ci_low"] >= -margin
            else "failed" if interval["ci_high"] < -margin else "insufficient"
        )
    else:
        status = _different_effect_status(interval["ci_low"], interval["ci_high"], contrast.minimum_effect)
    return {
        "endpoint_type": "paired_contrast",
        "endpoint_id": contrast.contrast_id,
        "primary": contrast.primary,
        "treatment_cell_id": contrast.treatment_cell_id,
        "control_cell_id": contrast.control_cell_id,
        "metric": contrast.metric,
        "direction": contrast.direction,
        "minimum_effect": contrast.minimum_effect,
        "equivalence_margin": contrast.equivalence_margin,
        "noninferiority_margin": contrast.noninferiority_margin,
        "pair_count": len(treatment),
        "alpha": contrast.alpha,
        "multiplicity_adjusted_alpha": adjusted_alpha,
        **interval,
        "status": status,
        "passed": status == "passed",
    }


def _analyse_interaction(
    study: AblationStudy,
    interaction: InteractionContrast,
    observations: list[AblationObservation],
    adjusted_alpha: float,
) -> dict:
    cell_ids = (
        interaction.first_treatment_cell_id,
        interaction.first_control_cell_id,
        interaction.second_treatment_cell_id,
        interaction.second_control_cell_id,
    )

    def indexed(cell_id: str) -> dict[tuple, AblationObservation]:
        rows = [
            row
            for row in observations
            if row.study_id == study.study_id and row.cell_id == cell_id and row.metric == interaction.metric
        ]
        result = {}
        for row in rows:
            key = _pair_key(row, interaction.pairing_keys)
            if key in result:
                raise ValueError(
                    f"Interaction {interaction.contrast_id} has duplicate {cell_id} observation "
                    f"for pairing key {key}"
                )
            result[key] = row
        return result

    indexed_cells = {cell_id: indexed(cell_id) for cell_id in cell_ids}
    key_sets = {cell_id: set(rows) for cell_id, rows in indexed_cells.items()}
    expected_keys = key_sets[cell_ids[0]]
    if any(keys != expected_keys for keys in key_sets.values()):
        missing = {
            cell_id: sorted(expected_keys - keys, key=str)
            for cell_id, keys in key_sets.items()
            if keys != expected_keys
        }
        raise ValueError(f"Interaction {interaction.contrast_id} is not fully paired; missing={missing}")
    if len(expected_keys) < interaction.minimum_pairs:
        raise ValueError(
            f"Interaction {interaction.contrast_id} has {len(expected_keys)} pairs; "
            f"requires {interaction.minimum_pairs}"
        )

    clustered: dict[str, list[float]] = defaultdict(list)
    first_effects: list[float] = []
    second_effects: list[float] = []
    for key in sorted(expected_keys, key=str):
        first_treatment = indexed_cells[interaction.first_treatment_cell_id][key]
        first_control = indexed_cells[interaction.first_control_cell_id][key]
        second_treatment = indexed_cells[interaction.second_treatment_cell_id][key]
        second_control = indexed_cells[interaction.second_control_cell_id][key]
        source_contexts = {
            (row.batch_id, row.lineage_group)
            for row in (first_treatment, first_control, second_treatment, second_control)
        }
        if len(source_contexts) != 1:
            raise ValueError(f"Interaction {interaction.contrast_id} pairs different source contexts")
        first_effect = first_treatment.value - first_control.value
        second_effect = second_treatment.value - second_control.value
        first_effects.append(first_effect)
        second_effects.append(second_effect)
        cluster = str(getattr(first_treatment, interaction.independence_key))
        clustered[cluster].append(first_effect - second_effect)

    cluster_values = [float(statistics.mean(values)) for _, values in sorted(clustered.items())]
    if len(cluster_values) < interaction.minimum_independent_units:
        raise ValueError(
            f"Interaction {interaction.contrast_id} has {len(cluster_values)} independent units; "
            f"requires {interaction.minimum_independent_units}"
        )
    interval = _cluster_interval(cluster_values, adjusted_alpha)
    if interaction.direction == "higher":
        status = _positive_effect_status(interval["ci_low"], interval["ci_high"], interaction.minimum_effect)
    elif interaction.direction == "lower":
        status = _negative_effect_status(interval["ci_low"], interval["ci_high"], interaction.minimum_effect)
    elif interaction.direction == "equivalent":
        margin = float(interaction.equivalence_margin)
        status = (
            "passed"
            if interval["ci_low"] >= -margin and interval["ci_high"] <= margin
            else "failed" if interval["ci_high"] < -margin or interval["ci_low"] > margin else "insufficient"
        )
    else:
        status = _different_effect_status(interval["ci_low"], interval["ci_high"], interaction.minimum_effect)
    return {
        "endpoint_type": "interaction_contrast",
        "endpoint_id": interaction.contrast_id,
        "primary": interaction.primary,
        "first_treatment_cell_id": interaction.first_treatment_cell_id,
        "first_control_cell_id": interaction.first_control_cell_id,
        "second_treatment_cell_id": interaction.second_treatment_cell_id,
        "second_control_cell_id": interaction.second_control_cell_id,
        "metric": interaction.metric,
        "direction": interaction.direction,
        "minimum_effect": interaction.minimum_effect,
        "equivalence_margin": interaction.equivalence_margin,
        "pair_count": len(expected_keys),
        "first_effect_mean": float(statistics.mean(first_effects)),
        "second_effect_mean": float(statistics.mean(second_effects)),
        "alpha": interaction.alpha,
        "multiplicity_adjusted_alpha": adjusted_alpha,
        **interval,
        "status": status,
        "passed": status == "passed",
    }


def analyse_ablations(
    protocol: AblationProtocol,
    observations: list[AblationObservation],
    *,
    study_ids: list[str] | None = None,
) -> dict:
    """Audit completeness, compute cluster-level intervals, and apply frozen gates."""

    expected_digest = ablation_protocol_digest(protocol)
    if not observations:
        raise ValueError("At least one ablation observation is required")
    if any(row.protocol_sha256 != expected_digest for row in observations):
        raise ValueError("Every ablation observation must use the supplied protocol SHA-256")
    known_studies = {study.study_id: study for study in protocol.studies}
    selected_ids = list(known_studies) if study_ids is None else study_ids
    if not selected_ids:
        raise ValueError("At least one ablation study must be selected")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected ablation study IDs must be unique")
    if unknown := set(selected_ids) - set(known_studies):
        raise ValueError(f"Unknown selected ablation studies: {sorted(unknown)}")
    selected_studies = [known_studies[study_id] for study_id in selected_ids]
    selected_id_set = set(selected_ids)
    observation_keys = set()
    for row in observations:
        study = known_studies.get(row.study_id)
        if study is None:
            raise ValueError(f"Observation references unknown study {row.study_id}")
        if row.cell_id not in {cell.cell_id for cell in study.cells}:
            raise ValueError(f"Observation references unknown cell {row.study_id}/{row.cell_id}")
        registered_metrics = {
            endpoint.metric for endpoint in [*study.cell_gates, *study.contrasts, *study.interactions]
        }
        if row.metric not in registered_metrics:
            raise ValueError(f"Observation references unregistered metric {row.study_id}/{row.metric}")
        observation_key = (
            row.study_id,
            row.cell_id,
            row.metric,
            row.batch_id,
            row.lineage_group,
            row.seed,
            row.capability_id,
            row.unit_id,
        )
        if observation_key in observation_keys:
            raise ValueError(f"Duplicate ablation observation identity: {observation_key}")
        observation_keys.add(observation_key)
    selected_observations = [row for row in observations if row.study_id in selected_id_set]

    primary_endpoints = [
        endpoint
        for study in selected_studies
        for endpoint in [*study.cell_gates, *study.contrasts, *study.interactions]
        if endpoint.primary
    ]
    family_sizes = defaultdict(int)
    for endpoint in primary_endpoints:
        family_sizes[endpoint.multiplicity_family] += 1
    for study in selected_studies:
        for endpoint in [*study.cell_gates, *study.contrasts, *study.interactions]:
            if not endpoint.primary:
                family_sizes.setdefault(endpoint.multiplicity_family, 1)

    study_results = []
    for study in selected_studies:
        endpoints = []
        for gate in study.cell_gates:
            endpoints.append(_analyse_gate(study, gate, selected_observations, _adjusted_alpha(gate, family_sizes)))
        for contrast in study.contrasts:
            endpoints.append(
                _analyse_contrast(study, contrast, selected_observations, _adjusted_alpha(contrast, family_sizes))
            )
        for interaction in study.interactions:
            endpoints.append(
                _analyse_interaction(
                    study,
                    interaction,
                    selected_observations,
                    _adjusted_alpha(interaction, family_sizes),
                )
            )
        primary = [row for row in endpoints if row["primary"]]
        primary_statuses = [row["status"] for row in primary]
        verdict = (
            "passed"
            if all(status == "passed" for status in primary_statuses)
            else "failed" if any(status == "failed" for status in primary_statuses) else "insufficient"
        )
        study_results.append(
            {
                "study_id": study.study_id,
                "stage": study.stage,
                "phase": study.phase,
                "causal_interpretation": study.causal_interpretation,
                "verdict": verdict,
                "passed": verdict == "passed",
                "endpoints": endpoints,
            }
        )
    causal_primary = [row for row in study_results if row["causal_interpretation"]]
    operational_primary = [
        row for row in study_results if known_studies[row["study_id"]].evidence_role == "operational_prediction"
    ]

    def combined_verdict(rows: list[dict]) -> str:
        if not rows:
            return "not_assessed"
        verdicts = [row["verdict"] for row in rows]
        if all(verdict == "passed" for verdict in verdicts):
            return "passed"
        if any(verdict == "failed" for verdict in verdicts):
            return "failed"
        return "insufficient"

    return {
        "schema_version": "stage-ablation-report-v1",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": expected_digest,
        "inference_scope": protocol.inference_scope,
        "supplied_observation_count": len(observations),
        "observation_count": len(selected_observations),
        "selected_study_ids": selected_ids,
        "mechanism_verdict": combined_verdict(causal_primary),
        "operational_verdict": combined_verdict(operational_primary),
        "all_primary_endpoints_passed": all(row["passed"] for row in study_results),
        "studies": study_results,
        "interpretation": {
            "mechanism": "Only studies marked causal_interpretation=true support stage-level attribution.",
            "operational": "The Fully Open Meditron comparison supports cross-pipeline prediction only.",
        },
    }
