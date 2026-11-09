import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.ablation import ablation_protocol_digest, load_ablation_protocol
from ai_ready.valuation.ablation_data import (
    AblationBatchInput,
    AblationCellExecution,
    AblationExecutionSpec,
    AblationJobRecord,
    CapabilityWeightProfile,
    ModelProfile,
    batch_input_digest,
    execution_digest,
)
from ai_ready.valuation.ablation_execution import (
    load_ablation_batch_input,
    load_ablation_execution_spec,
    materialize_ablation_execution,
    validate_execution_against_protocol_suite_contract,
)
from ai_ready.valuation.contracts import (
    contract_digest,
    file_sha256,
    load_batch_suite,
    load_production_contract,
    suite_digest,
)
from ai_ready.valuation.quality import SFTSample
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"
PROTOCOL_PATH = CONFIG_DIR / "stage_ablation_protocol.yaml"
SUITE_PATH = CONFIG_DIR / "batch_suite.frozen.yaml"
CONTRACT_PATH = CONFIG_DIR / "production_contract.yaml"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _inputs():
    return (
        load_ablation_protocol(PROTOCOL_PATH),
        load_batch_suite(SUITE_PATH),
        load_production_contract(CONTRACT_PATH),
    )


def _execution_spec(protocol, suite, contract) -> AblationExecutionSpec:
    capability_ids = suite.capability_blueprint
    capability_profiles = [
        CapabilityWeightProfile(profile_id=profile_id, weights={capability_id: 1.0 for capability_id in capability_ids})
        for profile_id in ("task_weighted_v1", "uniform_weights", "shuffled_weights_seed_101")
    ]
    model_profiles = [
        ModelProfile(
            profile_id=profile_id,
            model_id=contract.training.base_model,
            revision=contract.training.base_model_revision,
            checkpoint_uri=f"https://example.invalid/{profile_id}/checkpoint",
            checkpoint_sha256=_sha(f"checkpoint:{profile_id}"),
            calibration_uri=f"https://example.invalid/{profile_id}/calibration.json",
            calibration_sha256=_sha(f"calibration:{profile_id}"),
            capability_scores={capability_id: 0.5 for capability_id in capability_ids},
        )
        for profile_id in ("native", "controlled_low_prior", "controlled_high_prior")
    ]
    selector_methods = {
        "coverage_quality": "coverage_quality",
        "coverage_only": "coverage_only",
        "stratified_random_seed_set": "stratified_random",
        "stratified_random_seed_101": "stratified_random",
        "random_seed_set": "random",
        "random_seed_101": "random",
        "full_candidate_pool": "coverage_quality",
        "coverage_quality_include_target": "coverage_quality",
        "coverage_quality_knockout_target": "coverage_quality",
        "coverage_quality_rescue_target": "coverage_quality",
        "coverage_quality_nested": "coverage_quality",
        "coverage_quality_prior_aware": "coverage_quality",
        "coverage_quality_no_prior": "coverage_quality",
    }
    organizations = {
        "mixed_grounded_v1": "mixed",
        "direct_qa_only": "direct",
        "random_form_seed_101": "random",
    }
    kinds = {
        "blueprint_conditioning": "selection",
        "subset_selection": "selection",
        "supervision_organization": "organization",
        "learnability_bridge": "learnability",
        "scale_repeat_noise": "robustness",
        "base_model_prior": "base_model_prior",
        "operational_external_validity": "selection",
    }
    interventions = {
        "coverage_quality_include_target": "include",
        "coverage_quality_knockout_target": "knockout",
        "coverage_quality_rescue_target": "rescue",
    }
    cells = []
    for study in protocol.studies:
        for protocol_cell in study.cells:
            factors = protocol_cell.factors
            if factors.training_mode != "sft" or factors.generator == "fully_open_meditron":
                continue
            intervention_mode = interventions.get(factors.selector, "none")
            cells.append(
                AblationCellExecution(
                    study_id=study.study_id,
                    cell_id=protocol_cell.cell_id,
                    kind="quality_fidelity" if study.study_id == "A4_quality_fidelity" else kinds[study.stage],
                    capability_profile_id=factors.blueprint,
                    selector_config=ProxySelectionConfig(
                        method=selector_methods[factors.selector],
                        token_budget=100,
                        random_state=101,
                        facility_weight=1.0,
                        obligation_weight=1.0,
                        quality_fidelity_weight=1.0,
                        source_coverage_weight=1.0,
                    ),
                    source_budget_fraction=factors.source_budget_fraction,
                    selection_mode="nested" if factors.selector == "coverage_quality_nested" else "direct",
                    organization=organizations[factors.organization],
                    organization_seed=101,
                    intervention_id="target-rule-v1" if intervention_mode != "none" else None,
                    intervention_mode=intervention_mode,
                    repetition_factor=int(factors.repetition_factor),
                    label_noise_rate=factors.noise_rate,
                    label_noise_seed=101,
                    model_profile_id=None if factors.prior_profile == "native" else factors.prior_profile,
                    prior_aware=factors.selector == "coverage_quality_prior_aware",
                )
            )
    return AblationExecutionSpec(
        execution_id="test-execution",
        protocol_sha256=ablation_protocol_digest(protocol),
        suite_sha256=suite_digest(suite),
        contract_sha256=contract_digest(contract),
        training_seeds=contract.training.seeds,
        capability_profiles=capability_profiles,
        model_profiles=model_profiles,
        default_model_profile_id="native",
        cells=cells,
    )


def _batch_input(batch, capability_id: str) -> AblationBatchInput:
    unit_id = f"{batch.batch_id}-unit"
    proxy_unit = ProxyUnit(
        unit_id=unit_id,
        batch_id=batch.batch_id,
        batch_content_sha256=batch.content_sha256,
        group_id=f"{batch.batch_id}-group",
        source_ids=[batch.source_ids[0]],
        estimated_tokens=10,
        embedding=[1.0],
        quality_bucket="clean",
        benchmark_relevance=1.0,
    )
    samples = [
        SFTSample(
            sample_id=f"{unit_id}-{task_type}",
            batch_id=batch.batch_id,
            semantic_unit_ids=[unit_id],
            instruction=f"Question for {unit_id}",
            response=f"Answer for {unit_id}",
            evidence_spans=[f"Evidence for {unit_id}"],
            source_ids=[batch.source_ids[0]],
            task_type=task_type,
            generator="test-bank",
            generator_revision="v1",
        )
        for task_type in ("grounded_qa", "conditional_judgment", "clinical_vignette")
    ]
    return AblationBatchInput(
        batch_id=batch.batch_id,
        lineage_group=batch.lineage_group,
        batch_content_sha256=batch.content_sha256,
        proxy_units=[proxy_unit],
        obligations=[
            CoverageObligation(
                obligation_id=f"{batch.batch_id}-obligation",
                kind="rule",
                unit_ids=[unit_id],
                capability_ids=[capability_id],
                weight=1.0,
            )
        ],
        unit_capabilities={unit_id: [capability_id]},
        candidate_sft_bank=samples,
    )


def _batch_inputs(suite) -> list[AblationBatchInput]:
    return [_batch_input(batch, suite.capability_blueprint[0]) for batch in suite.batches]


def test_loaders_accept_json_and_yaml_and_preserve_semantic_identity(tmp_path: Path):
    protocol, suite, contract = _inputs()
    spec = _execution_spec(protocol, suite, contract)
    batch = _batch_inputs(suite)[0]
    spec_path = tmp_path / "execution.json"
    batch_path = tmp_path / "batch.yaml"
    spec_path.write_text(json.dumps(spec.model_dump(mode="json")), encoding="utf-8")
    batch_path.write_text(yaml.safe_dump(batch.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    assert execution_digest(load_ablation_execution_spec(spec_path)) == execution_digest(spec)
    assert batch_input_digest(load_ablation_batch_input(batch_path)) == batch_input_digest(batch)


def test_validation_rejects_fake_digest_wrong_factor_missing_cell_and_batch_mismatch():
    protocol, suite, contract = _inputs()
    spec = _execution_spec(protocol, suite, contract)
    batches = _batch_inputs(suite)
    report = validate_execution_against_protocol_suite_contract(spec, protocol, suite, contract, batches)
    assert report["executable_cell_count"] == 32
    assert {row["reason"] for row in report["excluded_protocol_cells"]} == {
        "static_no_sft",
        "fully_open_meditron_external",
    }

    with pytest.raises(ValueError, match="protocol_sha256 mismatch"):
        validate_execution_against_protocol_suite_contract(
            spec.model_copy(update={"protocol_sha256": "0" * 64}), protocol, suite, contract, batches
        )

    bad_cell = spec.cells[0].model_copy(
        update={"selector_config": spec.cells[0].selector_config.model_copy(update={"method": "random"})}
    )
    with pytest.raises(ValueError, match="selector_config.method"):
        validate_execution_against_protocol_suite_contract(
            spec.model_copy(update={"cells": [bad_cell, *spec.cells[1:]]}), protocol, suite, contract, batches
        )

    wrong_selection_mode = spec.cells[0].model_copy(update={"selection_mode": "nested"})
    with pytest.raises(ValueError, match="wrong selection_mode"):
        validate_execution_against_protocol_suite_contract(
            spec.model_copy(update={"cells": [wrong_selection_mode, *spec.cells[1:]]}),
            protocol,
            suite,
            contract,
            batches,
        )

    with pytest.raises(ValueError, match="missing="):
        validate_execution_against_protocol_suite_contract(
            spec.model_copy(update={"cells": spec.cells[:-1]}), protocol, suite, contract, batches
        )

    mismatched_batch = batches[0].model_copy(update={"lineage_group": "wrong-lineage"})
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_execution_against_protocol_suite_contract(
            spec, protocol, suite, contract, [mismatched_batch, *batches[1:]]
        )


def _fake_batch_materializer(spec, batch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    model_profiles = {profile.profile_id: profile for profile in spec.model_profiles}
    jobs = []
    cells = []
    for cell in spec.cells:
        relative_dir = Path("cells") / cell.study_id / cell.cell_id / batch.batch_id
        cell_dir = output_dir / relative_dir
        train_path = cell_dir / "train.jsonl"
        lineage_path = cell_dir / "lineage.jsonl"
        cell_manifest_path = cell_dir / "cell_manifest.json"
        write_jsonl(train_path, [{"instruction": "i", "input": "", "output": "o"}])
        write_jsonl(lineage_path, [{"batch_id": batch.batch_id, "cell_id": cell.cell_id}])
        write_json(cell_manifest_path, {"status": "ready"})
        profile = model_profiles[cell.model_profile_id or spec.default_model_profile_id]
        for seed in spec.training_seeds:
            jobs.append(
                AblationJobRecord(
                    job_id=f"{batch.batch_id}-{cell.study_id}-{cell.cell_id}-{seed}",
                    execution_id=spec.execution_id,
                    execution_sha256=execution_digest(spec),
                    protocol_sha256=spec.protocol_sha256,
                    suite_sha256=spec.suite_sha256,
                    contract_sha256=spec.contract_sha256,
                    batch_input_sha256=batch_input_digest(batch),
                    batch_id=batch.batch_id,
                    batch_content_sha256=batch.batch_content_sha256,
                    lineage_group=batch.lineage_group,
                    study_id=cell.study_id,
                    cell_id=cell.cell_id,
                    seed=seed,
                    model_profile_id=profile.profile_id,
                    model_id=profile.model_id,
                    model_revision=profile.revision,
                    checkpoint_uri=profile.checkpoint_uri,
                    checkpoint_sha256=profile.checkpoint_sha256,
                    calibration_sha256=profile.calibration_sha256,
                    capability_profile_sha256=_sha(cell.capability_profile_id),
                    selector_config_sha256=_sha(str(cell.selector_config.model_dump(mode="json"))),
                    selection_seed=cell.selector_config.random_state,
                    train_uri=str(relative_dir / "train.jsonl"),
                    train_sha256=file_sha256(train_path),
                    lineage_uri=str(relative_dir / "lineage.jsonl"),
                    lineage_sha256=file_sha256(lineage_path),
                    cell_manifest_uri=str(relative_dir / "cell_manifest.json"),
                    cell_manifest_sha256=file_sha256(cell_manifest_path),
                    train_row_count=1,
                )
            )
        cells.append(
            {
                "study_id": cell.study_id,
                "cell_id": cell.cell_id,
                "status": "ready",
                "cell_manifest_uri": str(relative_dir / "cell_manifest.json"),
                "cell_manifest_sha256": file_sha256(cell_manifest_path),
            }
        )
    write_jsonl(output_dir / "jobs.jsonl", (job.model_dump(mode="json") for job in jobs))
    write_json(
        output_dir / "materialization_manifest.json",
        {
            "batch_id": batch.batch_id,
            "batch_input_sha256": batch_input_digest(batch),
            "execution_sha256": execution_digest(spec),
            "cells": cells,
        },
    )


def _tiny_inputs():
    protocol, suite, contract = _inputs()
    blueprint_study = protocol.studies[1]
    tiny_study = blueprint_study.model_copy(update={"cells": [blueprint_study.cells[0]]})
    protocol = protocol.model_copy(update={"studies": [tiny_study]})
    suite = suite.model_copy(update={"batches": [suite.batches[0]]})
    spec = _execution_spec(protocol, suite, contract)
    batches = _batch_inputs(suite)
    return protocol, suite, contract, spec, batches


def test_multi_batch_compiler_writes_complete_hashed_matrix_atomically(tmp_path: Path, monkeypatch):
    protocol, suite, contract, spec, batches = _tiny_inputs()
    monkeypatch.setattr(
        "ai_ready.valuation.ablation_execution.materialize_ablation_batch",
        _fake_batch_materializer,
    )
    output = tmp_path / "execution"

    result = materialize_ablation_execution(
        spec,
        batches,
        output,
        protocol=protocol,
        suite=suite,
        contract=contract,
    )

    assert output.is_dir()
    assert not output.with_name("execution.partial").exists()
    manifest = json.loads((output / "execution_manifest.json").read_text(encoding="utf-8"))
    jobs = read_jsonl(output / "jobs.jsonl")
    assert manifest["complete_matrix"] is True
    assert manifest["job_count"] == len(contract.training.seeds)
    assert len(jobs) == len(contract.training.seeds)
    assert all(row["train_uri"].startswith("batches/B01/") for row in jobs)
    assert all((output / row["uri"]).is_file() for row in manifest["artifacts"])
    assert all(file_sha256(output / row["uri"]) == row["sha256"] for row in manifest["artifacts"])
    assert result["execution_manifest_sha256"] == file_sha256(output / "execution_manifest.json")


def test_multi_batch_compiler_removes_partial_tree_after_failure(tmp_path: Path, monkeypatch):
    protocol, suite, contract, spec, batches = _tiny_inputs()

    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("synthetic materializer failure")

    monkeypatch.setattr(
        "ai_ready.valuation.ablation_execution.materialize_ablation_batch",
        fail_materialization,
    )
    output = tmp_path / "failed-execution"
    with pytest.raises(RuntimeError, match="synthetic materializer failure"):
        materialize_ablation_execution(
            spec,
            batches,
            output,
            protocol=protocol,
            suite=suite,
            contract=contract,
        )
    assert not output.exists()
    assert not output.with_name("failed-execution.partial").exists()
