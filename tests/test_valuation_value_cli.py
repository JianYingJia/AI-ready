import json
import sys
from pathlib import Path

import yaml

from ai_ready.utils.io import read_jsonl
from ai_ready.valuation.cli import _annotate_sampling_units
from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.conditional_value import (
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeObservation,
    ScoreInterval,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"
ARTIFACT_ROOT = PROJECT_DIR / "outputs" / "cll_local_sft_qwen2_5_1_5b_full_20260921"
SHA = "a" * 64


def _run(monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["ai-ready-value", *arguments])
    assert valuation_main() == 0
    return json.loads(capsys.readouterr().out)


def test_annotation_keeps_distinct_model_state_for_each_capability():
    profile = ModelMasteryProfile(
        profile_id="two-cell-model",
        benchmark_id="benchmark",
        model_id="model",
        model_revision="revision",
        checkpoint_sha256=SHA,
        probe_set_id="probes",
        probe_set_sha256=SHA,
        scoring_protocol_sha256=SHA,
        observations=[
            ProbeObservation(
                cell_id="cap-a",
                capability_id="cap-a",
                operation_id="__absent__",
                difficulty_id="__absent__",
                point_score=0.9,
                score_interval=ScoreInterval(lower=0.8, upper=1.0),
                probe_count=10,
                probabilities=MasteryProbabilities(mastered=1, uncertain=0, unmastered=0, unknown=0),
                stratum_state="mastered",
                scoring_method="rubric",
            ),
            ProbeObservation(
                cell_id="cap-b",
                capability_id="cap-b",
                operation_id="__absent__",
                difficulty_id="__absent__",
                point_score=0.1,
                score_interval=ScoreInterval(lower=0.0, upper=0.2),
                probe_count=10,
                probabilities=MasteryProbabilities(mastered=0, uncertain=0, unmastered=1, unknown=0),
                stratum_state="unmastered",
                scoring_method="rubric",
            ),
        ],
    )
    rows = [
        {
            "unit_id": "unit-1",
            "batch_id": "B01",
            "batch_content_sha256": SHA,
            "dependency_group_id": "group-1",
            "capability_ids": ["cap-a", "cap-b"],
            "model_states": {"cap-a": "unknown", "cap-b": "unknown"},
            "data_attributes": {"quality": "high"},
        }
    ]

    units, status = _annotate_sampling_units(rows, profile)

    assert units[0].model_states == {"cap-a": "mastered", "cap-b": "unmastered"}
    assert units[0].model_state_probabilities == {
        "cap-a": {"mastered": 1.0, "uncertain": 0.0, "unmastered": 0.0, "unknown": 0.0},
        "cap-b": {"mastered": 0.0, "uncertain": 0.0, "unmastered": 1.0, "unknown": 0.0},
    }
    assert status == "complete"


def test_cli_runs_unknown_model_value_sampling_and_partial_certificate(tmp_path, monkeypatch, capsys):
    adapted = tmp_path / "adapted"
    _run(
        monkeypatch,
        capsys,
        "adapt-value-population",
        "--batches",
        str(CONFIG_DIR / "batch_suite.frozen.yaml"),
        "--artifact-root",
        str(ARTIFACT_ROOT),
        "--output-dir",
        str(adapted),
    )

    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "profile_id": "m0-pending",
                "benchmark_id": "medmcqa-cll-proxy-v1",
                "model_id": "test-model",
                "model_revision": "test-revision",
                "checkpoint_sha256": SHA,
                "probe_set_id": "pending-probes",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    aggregation = tmp_path / "aggregation.yaml"
    aggregation.write_text(
        yaml.safe_dump(
            {
                "tau_low": 0.35,
                "tau_high": 0.75,
                "confidence_level": 0.95,
                "bootstrap_iterations": 100,
                "seed": 42,
                "minimum_probe_count": 2,
                "scoring_method": "test-rubric",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    probes.write_text("", encoding="utf-8")
    model_profile = tmp_path / "model_profile.json"
    model_response = _run(
        monkeypatch,
        capsys,
        "build-model-profile",
        "--metadata",
        str(metadata),
        "--cells",
        str(CONFIG_DIR / "conditional_value" / "benchmark_cells.capability_only.jsonl"),
        "--probes",
        str(probes),
        "--aggregation-config",
        str(aggregation),
        "--output",
        str(model_profile),
    )
    assert model_response["claim_status"] == "partial_pending_model_measurement"
    assert len(model_response["unmeasured_cell_ids"]) == 6

    sampling = tmp_path / "sampling.yaml"
    sampling.write_text(
        yaml.safe_dump(
            {
                "sampling_fraction": 0.5,
                "seed": 17,
                "stratum_attributes": ["knowledge_type", "quality_bucket"],
                "rare_stratum_quotas": [],
                "allow_target_expansion": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sample_dir = tmp_path / "sample"
    sample_response = _run(
        monkeypatch,
        capsys,
        "sample-value-proxy",
        "--units",
        str(adapted / "value_population_units.jsonl"),
        "--model-profile",
        str(model_profile),
        "--config",
        str(sampling),
        "--batch-id",
        "B01",
        "--membership-map",
        str(adapted / "membership_map.jsonl"),
        "--artifact-root",
        str(ARTIFACT_ROOT),
        "--output-dir",
        str(sample_dir),
    )
    assert sample_response["claim_status"] == "partial_pending_model_measurement"
    assert sample_response["source_sample_materialized"] is True
    selected = read_jsonl(sample_dir / "selected_value_units.jsonl")
    assert selected
    assert all(set(row["model_states"].values()) == {"unknown"} for row in selected)
    selected_semantic = read_jsonl(sample_dir / "source_sample" / "selected_semantic_units.jsonl")
    assert len({row["semantic_unit_id"] for row in selected_semantic}) == len(selected_semantic)

    profiles = {}
    for kind, units_path in (
        ("full_population", adapted / "value_population_units.jsonl"),
        ("raw_sample", sample_dir / "selected_value_units.jsonl"),
        ("weighted_reconstruction", sample_dir / "selected_value_units.jsonl"),
    ):
        output = tmp_path / f"{kind}.json"
        response = _run(
            monkeypatch,
            capsys,
            "build-value-profile",
            "--units",
            str(units_path),
            "--model-profile",
            str(model_profile),
            "--batch-id",
            "B01",
            "--profile-id",
            f"B01-{kind}",
            "--kind",
            kind,
            "--output",
            str(output),
        )
        assert response["claim_status"] == "partial_pending_model_measurement"
        profiles[kind] = output

    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(
        yaml.safe_dump(
            {
                "maximum_joint_total_variation": 1.0,
                "maximum_critical_weighted_total_variation": 1.0,
                "maximum_cell_marginal_total_variation": 1.0,
                "maximum_cell_proportion_error": 1.0,
                "maximum_mastery_marginal_total_variation": 1.0,
                "maximum_attribute_marginal_total_variation": 1.0,
                "minimum_effective_sample_size": 1.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    certificate = tmp_path / "certificate.json"
    certificate_response = _run(
        monkeypatch,
        capsys,
        "build-value-certificate",
        "--reference-profile",
        str(profiles["full_population"]),
        "--raw-sample-profile",
        str(profiles["raw_sample"]),
        "--reconstructed-profile",
        str(profiles["weighted_reconstruction"]),
        "--thresholds",
        str(thresholds),
        "--certificate-id",
        "B01-partial",
        "--output",
        str(certificate),
    )
    assert certificate_response["claim_status"] == "partial_pending_model_measurement"
    verify_response = _run(
        monkeypatch,
        capsys,
        "verify-value-certificate",
        "--certificate",
        str(certificate),
        "--reference-profile",
        str(profiles["full_population"]),
        "--raw-sample-profile",
        str(profiles["raw_sample"]),
        "--reconstructed-profile",
        str(profiles["weighted_reconstruction"]),
    )
    assert verify_response["valid"] is True
    assert verify_response["claim_status"] == "partial_pending_model_measurement"

    sweep_config = tmp_path / "sweep.yaml"
    sweep_config.write_text(
        yaml.safe_dump(
            {
                "sampling_fractions": [1.0],
                "seeds": [3],
                "design_stratum_attributes": ["knowledge_type", "quality_bucket"],
                "audit_only_attributes": ["generation_status"],
                "rare_stratum_quotas": [],
                "allow_target_expansion": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sweep_output = tmp_path / "sweep.json"
    sweep_response = _run(
        monkeypatch,
        capsys,
        "sweep-value-fidelity",
        "--units",
        str(adapted / "value_population_units.jsonl"),
        "--model-profile",
        str(model_profile),
        "--config",
        str(sweep_config),
        "--thresholds",
        str(thresholds),
        "--batch-id",
        "B07",
        "--output",
        str(sweep_output),
    )
    assert sweep_response["claim_status"] == "partial_pending_model_measurement"
    assert sweep_response["batch_count"] == 1
    assert sweep_response["attribute_roles"]["design_stratum_attributes"] == [
        "knowledge_type",
        "quality_bucket",
    ]
    assert sweep_response["attribute_roles"]["audit_only_attributes"] == ["generation_status"]
    sweep = json.loads(sweep_output.read_text(encoding="utf-8"))
    assert sweep["batch_summary"][0]["batch_id"] == "B07"
    assert sweep["cells"][0]["trials"][0]["attribute_total_variation_by_role"]["audit_only_attributes"] == {
        "generation_status": 0.0
    }
