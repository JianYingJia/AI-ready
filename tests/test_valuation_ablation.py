import hashlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.valuation.ablation import (
    AblationObservation,
    AblationProtocol,
    AblationStudy,
    PipelineFactors,
    ablation_protocol_digest,
    analyse_ablations,
    load_ablation_protocol,
    protocol_summary,
)
from ai_ready.valuation.cli import main as valuation_main

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCREENING_PROTOCOL = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility" / "stage_ablation_protocol.yaml"


def _factors(**updates):
    payload = {
        "mining": "structured",
        "blueprint": "target",
        "selector": "coverage",
        "organization": "mixed",
        "source_budget_fraction": 0.25,
        "repetition_factor": 1.0,
        "noise_kind": "none",
        "noise_rate": 0.0,
        "base_model": "base",
        "prior_profile": "native",
        "training_mode": "sft",
        "generator": "coverage_guided_sft",
    }
    payload.update(updates)
    return payload


def _study(stage: str, index: int, *, paired: bool = False, operational: bool = False):
    treatment = {
        "cell_id": f"treatment_{index}",
        "description": "treatment",
        "factors": _factors(),
    }
    cells = [treatment]
    contrasts = []
    isolated = ["selector"]
    if paired:
        cells.append(
            {
                "cell_id": f"control_{index}",
                "description": "control",
                "factors": _factors(selector="control"),
            }
        )
        contrasts.append(
            {
                "contrast_id": f"contrast_{index}",
                "treatment_cell_id": f"treatment_{index}",
                "control_cell_id": f"control_{index}",
                "metric": "score",
                "pairing_keys": ["batch_id"],
                "independence_key": "lineage_group",
                "direction": "higher",
                "minimum_effect": 0.25,
                "minimum_pairs": 2,
                "minimum_independent_units": 2,
                "multiplicity_family": f"family_{index}",
                "primary": True,
                "rationale": "The treatment must improve the paired score.",
            }
        )
    return {
        "study_id": f"study_{index}",
        "stage": stage,
        "phase": "screening",
        "evidence_role": "operational_prediction" if operational else "training_mechanism",
        "target": "test target",
        "causal_interpretation": not operational,
        "isolated_factors": isolated,
        "cells": cells,
        "cell_gates": [
            {
                "gate_id": f"gate_{index}",
                "cell_id": f"treatment_{index}",
                "metric": "score",
                "direction": "at_least",
                "threshold": 0.5,
                "independence_key": "lineage_group",
                "minimum_observations": 2,
                "minimum_independent_units": 2,
                "multiplicity_family": f"family_{index}",
                "primary": True,
                "rationale": "The treatment must reach its absolute target.",
            }
        ],
        "contrasts": contrasts,
    }


def _protocol() -> AblationProtocol:
    stages = [
        "semantic_mining",
        "blueprint_conditioning",
        "subset_selection",
        "supervision_organization",
        "learnability_bridge",
        "scale_repeat_noise",
        "base_model_prior",
        "operational_external_validity",
    ]
    return AblationProtocol.model_validate(
        {
            "schema_version": "stage-ablation-protocol-v1",
            "protocol_id": "unit-test-ablation",
            "inference_scope": "descriptive_mechanism_screening",
            "locked_before_results": True,
            "benchmark_test_items_accessible_to_generation": False,
            "fully_open_meditron_role": "operational_external_reference_only",
            "multiplicity_correction": "bonferroni",
            "studies": [
                _study(
                    stage,
                    index,
                    paired=stage != "operational_external_validity",
                    operational=stage == "operational_external_validity",
                )
                for index, stage in enumerate(stages)
            ],
        }
    )


def _observations(protocol: AblationProtocol) -> list[AblationObservation]:
    digest = ablation_protocol_digest(protocol)
    rows = []
    for study in protocol.studies:
        for batch_id, lineage in (("B01", "L1"), ("B02", "L2")):
            for cell in study.cells:
                value = 0.0 if cell.cell_id.startswith("control") else 1.0
                artifact = hashlib.sha256(f"{study.study_id}:{cell.cell_id}:{batch_id}".encode()).hexdigest()
                rows.append(
                    AblationObservation(
                        protocol_sha256=digest,
                        study_id=study.study_id,
                        cell_id=cell.cell_id,
                        metric="score",
                        batch_id=batch_id,
                        lineage_group=lineage,
                        value=value,
                        artifact_sha256=artifact,
                    )
                )
    return rows


def test_repository_screening_protocol_covers_every_required_stage():
    protocol = load_ablation_protocol(SCREENING_PROTOCOL)
    summary = protocol_summary(protocol)

    assert summary["study_count"] == 11
    assert summary["cell_count"] == 35
    assert summary["endpoint_count"] == 35
    assert "A9_operational_fom_prediction" in summary["operational_studies"]
    assert "A3_subset_selection" in summary["causal_studies"]


def test_plan_ablations_cli_writes_reviewable_frozen_matrix(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "plan-ablations",
            "--protocol",
            str(SCREENING_PROTOCOL),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    matrix = json.loads((output_dir / "ablation_matrix.json").read_text(encoding="utf-8"))
    endpoints = json.loads((output_dir / "ablation_endpoints.json").read_text(encoding="utf-8"))
    observation_schema = json.loads((output_dir / "ablation_observation_schema.json").read_text(encoding="utf-8"))
    frozen = json.loads((output_dir / "ablation_protocol.json").read_text(encoding="utf-8"))
    assert response["study_count"] == 11
    assert response["cell_count"] == 35
    assert response["endpoint_count"] == 35
    assert len(matrix) == 35
    assert len(endpoints) == 35
    assert observation_schema["title"] == "AblationObservation"
    assert frozen["protocol_id"] == "cll-small-sample-stage-ablation-screening-v2"


def test_causal_contrast_rejects_generator_change():
    payload = _study("subset_selection", 1, paired=True)
    payload["cells"][1]["factors"]["generator"] = "fully_open_meditron"
    payload["isolated_factors"].append("generator")

    with pytest.raises(ValidationError, match="cannot compare different SFT generators"):
        AblationStudy.model_validate(payload)


def test_contrast_rejects_unregistered_factor_change():
    payload = _study("subset_selection", 1, paired=True)
    payload["cells"][1]["factors"]["organization"] = "direct_qa"

    with pytest.raises(ValidationError, match="unregistered factors"):
        AblationStudy.model_validate(payload)


def test_analysis_applies_absolute_and_paired_gates_at_lineage_level():
    protocol = _protocol()
    report = analyse_ablations(protocol, _observations(protocol))

    assert report["mechanism_verdict"] == "passed"
    assert report["operational_verdict"] == "passed"
    assert report["all_primary_endpoints_passed"] is True
    first = report["studies"][0]
    contrast = next(row for row in first["endpoints"] if row["endpoint_type"] == "paired_contrast")
    assert contrast["pair_count"] == 2
    assert contrast["independent_unit_count"] == 2
    assert contrast["mean"] == pytest.approx(1.0)
    assert contrast["passed"] is True


def test_analysis_can_apply_sequential_gate_to_one_study():
    protocol = _protocol()
    observations = _observations(protocol)
    report = analyse_ablations(protocol, observations, study_ids=["study_0"])

    assert report["selected_study_ids"] == ["study_0"]
    assert report["supplied_observation_count"] == len(observations)
    assert report["observation_count"] == 4
    assert report["mechanism_verdict"] == "passed"
    assert report["operational_verdict"] == "not_assessed"
    assert [row["study_id"] for row in report["studies"]] == ["study_0"]


def test_analysis_rejects_an_unpaired_result_table():
    protocol = _protocol()
    observations = _observations(protocol)
    observations = [
        row
        for row in observations
        if not (row.study_id == "study_0" and row.cell_id == "control_0" and row.batch_id == "B02")
    ]

    with pytest.raises(ValueError, match="not fully paired"):
        analyse_ablations(protocol, observations)


def test_wide_interval_is_insufficient_instead_of_a_component_failure():
    protocol = _protocol()
    observations = _observations(protocol)
    for row in observations:
        if row.study_id == "study_0" and row.cell_id == "treatment_0" and row.batch_id == "B01":
            row.value = 0.4

    report = analyse_ablations(protocol, observations)
    first = report["studies"][0]
    assert first["verdict"] == "insufficient"
    assert report["mechanism_verdict"] == "insufficient"


def test_analysis_rejects_observations_from_another_protocol():
    protocol = _protocol()
    observations = _observations(protocol)
    observations[0].protocol_sha256 = "0" * 64

    with pytest.raises(ValueError, match="supplied protocol SHA-256"):
        analyse_ablations(protocol, observations)


def test_analysis_rejects_unregistered_and_duplicate_observations():
    protocol = _protocol()
    observations = _observations(protocol)
    unknown = observations[0].model_copy(update={"metric": "invented_metric"})
    with pytest.raises(ValueError, match="unregistered metric"):
        analyse_ablations(protocol, [unknown, *observations[1:]])

    with pytest.raises(ValueError, match="Duplicate ablation observation"):
        analyse_ablations(protocol, [*observations, observations[0].model_copy()])


def test_noise_kind_and_rate_must_be_consistent():
    with pytest.raises(ValidationError, match="noise_kind=none"):
        PipelineFactors.model_validate(_factors(noise_rate=0.1))


def test_observation_validates_known_metric_ranges_and_real_artifact_hash(tmp_path):
    artifact = tmp_path / "predictions.jsonl"
    artifact.write_text('{"item_id":"i1","correct":true}\n', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    row = AblationObservation(
        protocol_sha256="a" * 64,
        study_id="A3",
        cell_id="coverage",
        metric="weighted_obligation_coverage",
        batch_id="B01",
        lineage_group="L1",
        value=0.9,
        artifact_sha256=digest,
        artifact_uri=str(artifact),
    )
    assert row.value == pytest.approx(0.9)

    with pytest.raises(ValidationError, match="within"):
        AblationObservation(
            protocol_sha256="a" * 64,
            study_id="A3",
            cell_id="coverage",
            metric="weighted_obligation_coverage",
            batch_id="B01",
            lineage_group="L1",
            value=1.1,
            artifact_sha256=digest,
            artifact_uri=str(artifact),
        )

    with pytest.raises(ValidationError, match="hash mismatch"):
        AblationObservation(
            protocol_sha256="a" * 64,
            study_id="A3",
            cell_id="coverage",
            metric="weighted_obligation_coverage",
            batch_id="B01",
            lineage_group="L1",
            value=0.9,
            artifact_sha256="b" * 64,
            artifact_uri=str(artifact),
        )


def test_zero_difference_does_not_pass_a_directional_contrast():
    protocol = _protocol()
    payload = protocol.model_dump(mode="json")
    payload["studies"][0]["contrasts"][0]["minimum_effect"] = 0.0
    protocol = AblationProtocol.model_validate(payload)
    observations = _observations(protocol)
    for row in observations:
        if row.study_id == "study_0" and row.cell_id == "control_0":
            row.value = 1.0

    report = analyse_ablations(protocol, observations, study_ids=["study_0"])
    contrast = next(
        endpoint for endpoint in report["studies"][0]["endpoints"] if endpoint["endpoint_type"] == "paired_contrast"
    )
    assert contrast["mean"] == pytest.approx(0.0)
    assert contrast["status"] == "failed"


def test_noninferiority_uses_a_one_sided_shortfall_margin():
    payload = _protocol().model_dump(mode="json")
    contrast = payload["studies"][0]["contrasts"][0]
    contrast.update(
        {
            "direction": "noninferior",
            "minimum_effect": 0.0,
            "equivalence_margin": None,
            "noninferiority_margin": 0.05,
        }
    )
    protocol = AblationProtocol.model_validate(payload)
    observations = _observations(protocol)
    for row in observations:
        if row.study_id == "study_0" and row.cell_id == "treatment_0":
            row.value = 0.98
        if row.study_id == "study_0" and row.cell_id == "control_0":
            row.value = 1.0

    report = analyse_ablations(protocol, observations, study_ids=["study_0"])
    contrast_result = next(
        endpoint for endpoint in report["studies"][0]["endpoints"] if endpoint["endpoint_type"] == "paired_contrast"
    )
    assert contrast_result["mean"] == pytest.approx(-0.02)
    assert contrast_result["status"] == "passed"


def test_interaction_computes_difference_between_two_paired_effects():
    payload = _protocol().model_dump(mode="json")
    study = payload["studies"][0]
    study["cells"].extend(
        [
            {
                "cell_id": "second_treatment",
                "description": "second treatment",
                "factors": _factors(selector="second-treatment"),
            },
            {
                "cell_id": "second_control",
                "description": "second control",
                "factors": _factors(selector="second-control"),
            },
        ]
    )
    study["interactions"] = [
        {
            "contrast_id": "state_interaction",
            "first_treatment_cell_id": "treatment_0",
            "first_control_cell_id": "control_0",
            "second_treatment_cell_id": "second_treatment",
            "second_control_cell_id": "second_control",
            "metric": "score",
            "pairing_keys": ["batch_id"],
            "independence_key": "lineage_group",
            "direction": "higher",
            "minimum_effect": 0.1,
            "minimum_pairs": 2,
            "minimum_independent_units": 2,
            "multiplicity_family": "interaction_primary",
            "primary": True,
            "rationale": "The first treatment effect should be larger.",
        }
    ]
    protocol = AblationProtocol.model_validate(payload)
    digest = ablation_protocol_digest(protocol)
    values = {
        "treatment_0": 1.0,
        "control_0": 0.0,
        "second_treatment": 0.4,
        "second_control": 0.2,
    }
    observations = []
    for batch_id, lineage in (("B01", "L1"), ("B02", "L2")):
        for cell_id, value in values.items():
            observations.append(
                AblationObservation(
                    protocol_sha256=digest,
                    study_id="study_0",
                    cell_id=cell_id,
                    metric="score",
                    batch_id=batch_id,
                    lineage_group=lineage,
                    value=value,
                    artifact_sha256=hashlib.sha256(f"{batch_id}:{cell_id}".encode()).hexdigest(),
                )
            )

    report = analyse_ablations(protocol, observations, study_ids=["study_0"])
    interaction = next(
        endpoint
        for endpoint in report["studies"][0]["endpoints"]
        if endpoint["endpoint_type"] == "interaction_contrast"
    )
    assert interaction["first_effect_mean"] == pytest.approx(1.0)
    assert interaction["second_effect_mean"] == pytest.approx(0.2)
    assert interaction["mean"] == pytest.approx(0.8)
    assert interaction["status"] == "passed"
