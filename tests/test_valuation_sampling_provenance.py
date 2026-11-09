import copy
import json
import sys

import pytest

from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.conditional_value import (
    FidelityThresholds,
    MasteryProbabilities,
    ModelMasteryProfile,
    ProbeObservation,
    ScoreInterval,
    build_fidelity_certificate,
    build_raw_sample_profile,
    build_value_profile,
    conditional_value_unit_from_sampling_record,
    reconstruct_value_profile,
    verify_fidelity_certificate,
)
from ai_ready.valuation.value_sampling import (
    ValueSamplingConfig,
    ValueSamplingUnit,
    sample_value_distribution,
    value_sampling_manifest_digest,
    verify_value_sample_manifest,
)

SHA = "a" * 64


def _units():
    return [
        ValueSamplingUnit(
            unit_id=f"unit-{index}",
            batch_id="B01",
            batch_content_sha256=SHA,
            dependency_group_id=f"group-{index}",
            capability_ids=["diagnosis"],
            model_states={"diagnosis": state},
            data_attributes={"knowledge_type": knowledge_type},
        )
        for index, (state, knowledge_type) in enumerate(
            [
                ("mastered", "rule"),
                ("mastered", "rule"),
                ("unmastered", "exception"),
                ("unmastered", "exception"),
                ("uncertain", "condition"),
                ("uncertain", "condition"),
            ]
        )
    ]


def _manifest():
    return sample_value_distribution(
        _units(),
        ValueSamplingConfig(
            sampling_fraction=0.5,
            seed=17,
            stratum_attributes=["knowledge_type"],
        ),
    )


def test_sampling_provenance_replays_full_stratified_design():
    manifest = _manifest()

    audit = verify_value_sample_manifest(manifest, _units())

    assert audit["design_provenance_verified"] is True
    assert audit["sampling_manifest_sha256"] == value_sampling_manifest_digest(manifest)
    assert audit["population_group_count"] == 6
    assert audit["selected_group_count"] == 3
    assert audit["stratum_count"] == 3
    assert all(audit["checks"].values())
    assert {row["inclusion_probability"] for row in audit["strata"]} == {0.5}
    assert {row["design_weight"] for row in audit["strata"]} == {2.0}


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda manifest: manifest["selected_unit_records"][0].update(
                {"inclusion_probability": 0.9, "design_weight": 1 / 0.9}
            ),
            "selected_unit_records",
        ),
        (
            lambda manifest: manifest["group_strata"][0]["population_group_ids"].append("forged-group"),
            "group_strata",
        ),
        (
            lambda manifest: manifest["group_strata"][0].update(
                {"population_group_count": manifest["group_strata"][0]["population_group_count"] + 1}
            ),
            "group_strata",
        ),
    ],
)
def test_sampling_provenance_rejects_tampered_probability_membership_or_stratum_count(
    mutation,
    field,
):
    manifest = copy.deepcopy(_manifest())
    mutation(manifest)

    with pytest.raises(ValueError, match=field):
        verify_value_sample_manifest(manifest, _units())


def test_sampling_provenance_rejects_config_or_batch_hash_tampering():
    bad_config = copy.deepcopy(_manifest())
    bad_config["sampling_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sampling_config_sha256"):
        verify_value_sample_manifest(bad_config, _units())

    bad_batch = copy.deepcopy(_manifest())
    bad_batch["batch_content_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="batch_content_sha256"):
        verify_value_sample_manifest(bad_batch, _units())


def test_sampling_provenance_cli_writes_replay_audit(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.json"
    population_path = tmp_path / "population.jsonl"
    output_path = tmp_path / "audit.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    population_path.write_text(
        "".join(json.dumps(unit.model_dump(mode="json")) + "\n" for unit in _units()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "verify-value-sample-manifest",
            "--manifest",
            str(manifest_path),
            "--population-units",
            str(population_path),
            "--output",
            str(output_path),
        ],
    )

    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    audit = json.loads(output_path.read_text(encoding="utf-8"))
    assert response["design_provenance_verified"] is True
    assert audit["sampling_manifest_sha256"] == response["sampling_manifest_sha256"]


def test_certificate_binds_verified_sampling_manifest_digest():
    sampling_units = _units()
    manifest = _manifest()
    manifest_sha256 = value_sampling_manifest_digest(manifest)
    model_profile = ModelMasteryProfile(
        profile_id="m0",
        benchmark_id="benchmark",
        model_id="model",
        model_revision="revision",
        checkpoint_sha256="b" * 64,
        probe_set_id="probe-set",
        probe_set_sha256="c" * 64,
        scoring_protocol_sha256="d" * 64,
        observations=[
            ProbeObservation(
                cell_id="diagnosis",
                capability_id="diagnosis",
                operation_id="decision",
                difficulty_id="medium",
                point_score=0.8,
                score_interval=ScoreInterval(lower=0.7, upper=0.9),
                probe_count=10,
                probabilities=MasteryProbabilities(mastered=1, uncertain=0, unmastered=0, unknown=0),
                stratum_state="mastered",
                scoring_method="rubric",
            )
        ],
    )
    full_units = [conditional_value_unit_from_sampling_record(unit) for unit in sampling_units]
    selected_units = [conditional_value_unit_from_sampling_record(row) for row in manifest["selected_unit_records"]]
    reference = build_value_profile(full_units, model_profile, profile_id="full")
    raw = build_raw_sample_profile(selected_units, model_profile, profile_id="raw")
    reconstructed = reconstruct_value_profile(selected_units, model_profile, profile_id="weighted")
    provenance = verify_value_sample_manifest(manifest, sampling_units)
    assert provenance["reference_membership_sha256"] == reference.membership_sha256
    assert provenance["sample_membership_sha256"] == raw.membership_sha256
    assert provenance["sample_design_sha256"] == raw.sampling_design_sha256
    thresholds = FidelityThresholds(
        maximum_joint_total_variation=1,
        maximum_critical_weighted_total_variation=1,
        maximum_cell_marginal_total_variation=1,
        maximum_cell_proportion_error=1,
        maximum_mastery_marginal_total_variation=1,
        maximum_attribute_marginal_total_variation=1,
        minimum_effective_sample_size=1,
    )

    certificate = build_fidelity_certificate(
        reference,
        raw,
        reconstructed,
        certificate_id="bound",
        thresholds=thresholds,
        sampling_manifest_sha256=manifest_sha256,
        design_provenance_verified=True,
    )

    assert certificate.design_provenance_verified is True
    assert certificate.sampling_manifest_sha256 == manifest_sha256
    verify_fidelity_certificate(
        certificate,
        reference,
        raw,
        reconstructed,
        sampling_manifest_sha256=manifest_sha256,
    )
    with pytest.raises(ValueError, match="requires its verified sampling manifest"):
        verify_fidelity_certificate(certificate, reference, raw, reconstructed)
    with pytest.raises(ValueError, match="sampling manifest digest does not match"):
        verify_fidelity_certificate(
            certificate,
            reference,
            raw,
            reconstructed,
            sampling_manifest_sha256="f" * 64,
        )
