import json
from pathlib import Path

import pytest
import yaml

from ai_ready.utils.io import write_json, write_jsonl
from ai_ready.valuation.contracts import canonical_digest, file_sha256, load_batch_suite, load_production_contract
from ai_ready.valuation.experiment import build_experiment_manifest
from ai_ready.valuation.feasibility import load_feasibility_protocol
from ai_ready.valuation.llamafactory import (
    SFTArtifactEntry,
    _validate_generation_manifest,
    _validate_generation_provenance,
    write_sft_registry_template,
)
from ai_ready.valuation.quality import SFTSample

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"


def _inputs():
    suite = load_batch_suite(CONFIG_DIR / "batch_suite.frozen.yaml")
    contract = load_production_contract(CONFIG_DIR / "production_contract.yaml")
    protocol = load_feasibility_protocol(CONFIG_DIR / "feasibility_protocol.yaml")
    manifest = build_experiment_manifest(suite, contract, protocol)
    proxy = next(row for row in manifest.trials if row.arm == "proxy" and row.batch_id == "B01")
    reference = next(row for row in manifest.trials if row.arm == "reference" and row.batch_id == "B01")
    return contract, proxy, reference


def _sample(recipe, *, sample_id: str, semantic_unit_id: str) -> SFTSample:
    return SFTSample(
        sample_id=sample_id,
        batch_id="B01",
        semantic_unit_ids=[semantic_unit_id],
        instruction=f"What is supported by {semantic_unit_id}?",
        response=f"Evidence for {semantic_unit_id}.",
        evidence_spans=[f"Evidence for {semantic_unit_id}."],
        source_ids=[f"source-{semantic_unit_id}"],
        task_type="grounded_qa",
        generator=recipe.generator,
        generator_revision=recipe.revision,
        teacher_model=recipe.teacher_model,
        teacher_revision=recipe.teacher_revision,
        prompt_sha256=recipe.prompt_sha256,
        quality_gate_sha256=recipe.quality_gate_sha256,
    )


def test_row_provenance_uses_different_recipe_for_each_branch(tmp_path: Path):
    contract, proxy_trial, reference_trial = _inputs()
    proxy_sample = _sample(contract.generation.proxy, sample_id="proxy-sample", semantic_unit_id="proxy-unit")
    reference_sample = _sample(
        contract.generation.reference,
        sample_id="reference-sample",
        semantic_unit_id="reference-unit",
    )

    _validate_generation_provenance([proxy_sample], contract, proxy_trial, tmp_path / "proxy.jsonl")
    _validate_generation_provenance([reference_sample], contract, reference_trial, tmp_path / "reference.jsonl")

    # The two branches may use completely different generated sample and
    # semantic-unit identities.  Their common identity is the frozen source
    # batch hash, not a subset relation between the SFT rows.
    assert set(proxy_sample.semantic_unit_ids).isdisjoint(reference_sample.semantic_unit_ids)
    assert proxy_trial.batch_content_sha256 == reference_trial.batch_content_sha256
    assert proxy_trial.generation_recipe_sha256 != reference_trial.generation_recipe_sha256


@pytest.mark.parametrize("wrong_arm", ["proxy", "reference"])
def test_row_provenance_rejects_recipe_swaps_and_mixed_rows(tmp_path: Path, wrong_arm: str):
    contract, proxy_trial, reference_trial = _inputs()
    trial = proxy_trial if wrong_arm == "proxy" else reference_trial
    wrong_recipe = contract.generation.reference if wrong_arm == "proxy" else contract.generation.proxy
    correct_recipe = contract.generation.recipe_for_arm(wrong_arm)
    wrong = _sample(wrong_recipe, sample_id="wrong", semantic_unit_id="u-wrong")

    with pytest.raises(ValueError, match="frozen generation recipe"):
        _validate_generation_provenance([wrong], contract, trial, tmp_path / f"{wrong_arm}.jsonl")

    correct = _sample(correct_recipe, sample_id="correct", semantic_unit_id="u-correct")
    with pytest.raises(ValueError, match="frozen generation recipe"):
        _validate_generation_provenance([correct, wrong], contract, trial, tmp_path / "mixed.jsonl")


def _generation_entry(
    tmp_path: Path,
    *,
    contract,
    trial,
    manifest_overrides: dict | None = None,
) -> tuple[SFTArtifactEntry, Path]:
    recipe = contract.generation.recipe_for_arm(trial.arm)
    source_path = tmp_path / f"{trial.arm}-accepted.jsonl"
    write_jsonl(
        source_path,
        [
            _sample(recipe, sample_id=f"{trial.arm}-sample", semantic_unit_id=f"{trial.arm}-unit").model_dump(
                mode="json"
            )
        ],
    )
    lineage_path = tmp_path / ("selection.json" if trial.arm == "proxy" else "traversal.json")
    write_json(lineage_path, {"branch": trial.arm})
    proxy_units_path = tmp_path / "proxy-units.jsonl"
    upstream_run_path = tmp_path / "fully_open_meditron_run_manifest.json"
    if trial.arm == "proxy":
        write_jsonl(proxy_units_path, [{"unit_id": "proxy-unit"}])
    else:
        tool = next(row for row in contract.tools if row.stage == "generate")
        write_json(
            upstream_run_path,
            {
                "schema_version": "fully-open-meditron-run-manifest-v1",
                "generator": "fully_open_meditron",
                "generator_revision": recipe.revision,
                "tool_adapter": tool.adapter,
                "tool_repository": tool.repository,
                "tool_config_sha256": tool.config_sha256,
                "teacher_model": recipe.teacher_model,
                "teacher_revision": recipe.teacher_revision,
                "prompt_sha256": recipe.prompt_sha256,
                "quality_gate_sha256": recipe.quality_gate_sha256,
                "batch_id": trial.batch_id,
                "batch_content_sha256": trial.batch_content_sha256,
                "eligible_units_sha256": "e" * 64,
                "normalized_output_sha256": "f" * 64,
                "benchmark_items_accessed": False,
                "uses_proxy_selector": False,
                "complete_input_traversal": True,
                "formal_production_contract_used": True,
                "contract_id": contract.contract_id,
                "contract_sha256": trial.contract_sha256,
            },
        )
    generation_path = tmp_path / f"{trial.arm}-generation.json"
    payload = {
        "schema_version": "sft-generation-manifest-v2",
        "artifact_role": trial.arm,
        "input_scope": "selected_semantic_units" if trial.arm == "proxy" else "full_batch",
        "batch_id": trial.batch_id,
        "batch_content_sha256": trial.batch_content_sha256,
        "generator": recipe.generator,
        "generator_revision": recipe.revision,
        "implementation_sha256": recipe.implementation_sha256 if trial.arm == "proxy" else None,
        "generation_recipe_sha256": canonical_digest(recipe),
        "source_sft_sha256": file_sha256(source_path),
        "contract_id": contract.contract_id,
        "contract_sha256": trial.contract_sha256,
        "selection_manifest_sha256": file_sha256(lineage_path) if trial.arm == "proxy" else None,
        "full_traversal_report_sha256": file_sha256(lineage_path) if trial.arm == "reference" else None,
        "benchmark_items_accessed": False,
        "formal_production_contract_used": True,
        "selection_before_generation": True if trial.arm == "proxy" else None,
        "candidate_pool_complete": True if trial.arm == "proxy" else None,
        "candidate_pool_sha256": file_sha256(proxy_units_path) if trial.arm == "proxy" else None,
        "max_units_applied": False if trial.arm == "proxy" else None,
        "uses_proxy_selector": False if trial.arm == "reference" else None,
        "complete_input_traversal": True if trial.arm == "reference" else None,
        "upstream_run_manifest_sha256": file_sha256(upstream_run_path) if trial.arm == "reference" else None,
        "inputs": (
            {"eligible_units_sha256": "e" * 64, "fully_open_meditron_sha256": "f" * 64}
            if trial.arm == "reference"
            else {}
        ),
    }
    payload.update(manifest_overrides or {})
    write_json(generation_path, payload)
    common = {
        "batch_id": trial.batch_id,
        "arm_id": trial.arm_id,
        "arm_label": trial.arm_label,
        "source_sft_uri": str(source_path),
        "generation_manifest_uri": str(generation_path),
        "dispositions_uri": str(tmp_path / f"{trial.arm}-dispositions.jsonl"),
    }
    if trial.arm == "proxy":
        entry = SFTArtifactEntry(
            **common,
            selection_manifest_uri=str(lineage_path),
            proxy_units_uri=str(proxy_units_path),
            coverage_obligations_uri=str(tmp_path / "obligations.jsonl"),
        )
    else:
        entry = SFTArtifactEntry(
            **common,
            eligible_units_uri=str(tmp_path / "eligible.jsonl"),
            upstream_run_manifest_uri=str(upstream_run_path),
            full_traversal_report_uri=str(lineage_path),
        )
    return entry, source_path


@pytest.mark.parametrize("arm", ["proxy", "reference"])
def test_generation_manifest_binds_branch_recipe_and_source_batch(tmp_path: Path, arm: str):
    contract, proxy_trial, reference_trial = _inputs()
    trial = proxy_trial if arm == "proxy" else reference_trial
    entry, source_path = _generation_entry(tmp_path, contract=contract, trial=trial)

    _, manifest, _ = _validate_generation_manifest(
        entry=entry,
        trial=trial,
        contract=contract,
        source_path=source_path,
    )

    assert manifest.artifact_role == arm
    assert manifest.generator == contract.generation.recipe_for_arm(arm).generator
    assert manifest.batch_content_sha256 == trial.batch_content_sha256


def test_generation_manifest_rejects_source_batch_substitution(tmp_path: Path):
    contract, proxy_trial, _ = _inputs()
    entry, source_path = _generation_entry(
        tmp_path,
        contract=contract,
        trial=proxy_trial,
        manifest_overrides={"batch_content_sha256": "f" * 64},
    )

    with pytest.raises(ValueError, match="Generation manifest does not match trial"):
        _validate_generation_manifest(
            entry=entry,
            trial=proxy_trial,
            contract=contract,
            source_path=source_path,
        )


def test_generation_manifest_rejects_truncated_proxy_candidate_pool(tmp_path: Path):
    contract, proxy_trial, _ = _inputs()
    entry, source_path = _generation_entry(tmp_path, contract=contract, trial=proxy_trial)

    write_jsonl(Path(entry.proxy_units_uri), [{"unit_id": "different-or-truncated-unit"}])

    with pytest.raises(ValueError, match="candidate_pool_sha256"):
        _validate_generation_manifest(
            entry=entry,
            trial=proxy_trial,
            contract=contract,
            source_path=source_path,
        )


def test_generation_manifest_rejects_changed_fom_run_receipt(tmp_path: Path):
    contract, _, reference_trial = _inputs()
    entry, source_path = _generation_entry(tmp_path, contract=contract, trial=reference_trial)

    receipt = Path(entry.generation_manifest_uri).with_name("fully_open_meditron_run_manifest.json")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["teacher_model"] = "tampered-teacher"
    write_json(receipt, receipt_payload)

    with pytest.raises(ValueError, match="upstream_run_manifest_sha256"):
        _validate_generation_manifest(
            entry=entry,
            trial=reference_trial,
            contract=contract,
            source_path=source_path,
        )


def test_generation_manifest_rejects_self_consistent_but_wrong_fom_receipt(tmp_path: Path):
    contract, _, reference_trial = _inputs()
    entry, source_path = _generation_entry(tmp_path, contract=contract, trial=reference_trial)

    receipt = Path(entry.upstream_run_manifest_uri)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["teacher_model"] = "uncontracted-teacher"
    write_json(receipt, receipt_payload)
    generation_path = Path(entry.generation_manifest_uri)
    generation_payload = json.loads(generation_path.read_text(encoding="utf-8"))
    generation_payload["upstream_run_manifest_sha256"] = file_sha256(receipt)
    write_json(generation_path, generation_payload)

    with pytest.raises(ValueError, match="upstream_run_manifest.teacher_model"):
        _validate_generation_manifest(
            entry=entry,
            trial=reference_trial,
            contract=contract,
            source_path=source_path,
        )


def test_registry_entry_requires_a_branch_generation_manifest():
    with pytest.raises(ValueError, match="generation_manifest_uri"):
        SFTArtifactEntry.model_validate(
            {
                "batch_id": "B01",
                "arm_id": "arm",
                "arm_label": "proxy",
                "source_sft_uri": "accepted.jsonl",
                "selection_manifest_uri": "selection.json",
                "proxy_units_uri": "units.jsonl",
                "coverage_obligations_uri": "obligations.jsonl",
                "dispositions_uri": "dispositions.jsonl",
            }
        )


def test_registry_template_matches_arm_then_batch_generator_layout(tmp_path: Path):
    suite = load_batch_suite(CONFIG_DIR / "batch_suite.frozen.yaml")
    contract = load_production_contract(CONFIG_DIR / "production_contract.yaml")
    protocol = load_feasibility_protocol(CONFIG_DIR / "feasibility_protocol.yaml")
    manifest = build_experiment_manifest(suite, contract, protocol)
    registry_path = tmp_path / "registry.yaml"
    sft_root = tmp_path / "sft"

    write_sft_registry_template(manifest, registry_path, sft_root)

    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "llamafactory-sft-registry-v3"
    proxy = next(
        row for row in payload["entries"] if row["batch_id"] == "B01" and row["arm_label"] == "coverage_quality"
    )
    reference = next(
        row for row in payload["entries"] if row["batch_id"] == "B01" and row["arm_label"] == "operational_full"
    )
    assert Path(proxy["source_sft_uri"]).parent == (sft_root / "coverage_quality" / "B01").resolve()
    assert Path(reference["source_sft_uri"]).parent == (sft_root / "operational_full" / "B01").resolve()
    assert reference["upstream_run_manifest_uri"].endswith("fully_open_meditron_run_manifest.json")
