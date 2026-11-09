import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai_ready.valuation.contracts import load_batch_suite, load_production_contract
from ai_ready.valuation.experiment import audit_experiment_manifest, build_experiment_manifest
from ai_ready.valuation.schema import ExperimentTrialScore

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "valuation_cll"


def _inputs():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    contract = load_production_contract(EXAMPLE_DIR / "production_contract.yaml")
    suite = suite.model_copy(deep=True)
    for batch in suite.batches:
        batch.content_sha256 = hashlib.sha256(batch.batch_id.encode("utf-8")).hexdigest()
    protocol = {
        "protocol_id": "unit-test-experiment",
        "expected_benchmark_item_count": 46,
        "target": {
            "label": "coverage_quality",
            "method": "blueprint_coverage_quality",
            "budget_id": "proxy-100k",
            "max_effective_train_tokens": 100_000,
            "selection_seed": None,
            "selection_config_uri": "selectors/coverage-quality.yaml",
            "selection_config_sha256": "a" * 64,
        },
        "baselines": [
            {
                "label": "random-101",
                "method": "random",
                "budget_id": "proxy-100k",
                "max_effective_train_tokens": 100_000,
                "selection_seed": 101,
                "selection_config_uri": "selectors/random.yaml",
                "selection_config_sha256": "b" * 64,
            },
            {
                "label": "quality_only",
                "method": "quality_only",
                "budget_id": "proxy-100k",
                "max_effective_train_tokens": 100_000,
                "selection_seed": None,
                "selection_config_uri": "selectors/quality-only.yaml",
                "selection_config_sha256": "c" * 64,
            },
        ],
        "reference": {
            "label": "operational_full",
            "method": contract.generation.reference.generator,
            "budget_id": "full",
            "max_effective_train_tokens": contract.training.effective_train_tokens,
            "selection_seed": None,
            "selection_config_uri": None,
            "selection_config_sha256": None,
        },
        "decision": {"min_spearman": 0.6},
    }
    return suite, contract, protocol


def test_manifest_is_full_batch_arm_seed_cartesian_product():
    suite, contract, protocol = _inputs()
    manifest = build_experiment_manifest(suite, contract, protocol)

    expected_count = len(suite.batches) * 4 * len(contract.training.seeds)
    assert manifest.expected_trial_count == expected_count
    assert len(manifest.trials) == expected_count
    assert len({row.run_id for row in manifest.trials}) == expected_count
    assert {arm.role for arm in manifest.arms} == {"target", "baseline", "reference"}
    reference = next(arm for arm in manifest.arms if arm.role == "reference")
    assert reference.input_scope == "full_batch"
    assert reference.generator == "fully_open_meditron"
    assert all(arm.generator == "coverage_guided_sft" for arm in manifest.arms if arm.role != "reference")
    assert all(arm.input_scope == "selected_subset" for arm in manifest.arms if arm.role != "reference")
    assert {(row.batch_id, row.arm_id, row.seed) for row in manifest.trials} == {
        (batch.batch_id, arm.arm_id, seed)
        for batch in suite.batches
        for arm in manifest.arms
        for seed in contract.training.seeds
    }


def test_ids_are_deterministic_and_bound_to_all_frozen_hashes():
    suite, contract, protocol = _inputs()
    first = build_experiment_manifest(suite, contract, protocol)
    second = build_experiment_manifest(suite, contract, protocol)
    assert first == second

    row = first.trials[0]
    assert row.suite_sha256 == first.suite_sha256
    assert row.contract_sha256 == first.contract_sha256
    assert row.benchmark_sha256 == contract.benchmark.test_manifest_sha256
    assert row.batch_content_sha256 == first.batch_content_sha256[row.batch_id]

    changed_suite = suite.model_copy(deep=True)
    changed_suite.batches[0].content_sha256 = "a" * 64
    changed = build_experiment_manifest(changed_suite, contract, protocol)
    assert changed.experiment_id != first.experiment_id
    assert {trial.run_id for trial in changed.trials}.isdisjoint(trial.run_id for trial in first.trials)


def test_decision_gate_changes_protocol_and_experiment_identity():
    suite, contract, protocol = _inputs()
    first = build_experiment_manifest(suite, contract, protocol)
    changed_protocol = deepcopy(protocol)
    changed_protocol["decision"]["min_spearman"] = 0.7
    changed = build_experiment_manifest(suite, contract, changed_protocol)

    assert changed.protocol_sha256 != first.protocol_sha256
    assert changed.experiment_id != first.experiment_id
    assert {trial.run_id for trial in changed.trials}.isdisjoint(trial.run_id for trial in first.trials)


def test_manifest_writer_refuses_to_overwrite(tmp_path):
    suite, contract, protocol = _inputs()
    manifest = build_experiment_manifest(suite, contract, protocol)
    output = tmp_path / "experiment.json"

    manifest.write_json(output)
    assert json.loads(output.read_text(encoding="utf-8"))["experiment_id"] == manifest.experiment_id
    with pytest.raises(FileExistsError):
        manifest.write_json(output)


@pytest.mark.parametrize("bad_hash", [None, "not-a-sha256"])
def test_manifest_requires_valid_frozen_batch_hashes(bad_hash):
    suite, contract, protocol = _inputs()
    suite.batches[0].content_sha256 = bad_hash
    message = "run freeze-batches" if bad_hash is None else "64-character SHA-256"
    with pytest.raises(ValueError, match=message):
        build_experiment_manifest(suite, contract, protocol)


def test_audit_exposes_missing_and_duplicate_trials():
    suite, contract, protocol = _inputs()
    manifest = build_experiment_manifest(suite, contract, protocol)
    expected = manifest.trials[0]
    score = ExperimentTrialScore(
        run_id=expected.run_id,
        suite_sha256=expected.suite_sha256,
        batch_id=expected.batch_id,
        batch_content_sha256=expected.batch_content_sha256,
        arm=expected.arm,
        method=expected.method,
        budget_id=expected.budget_id,
        seed=expected.seed,
        benchmark_score=25 / 46,
        baseline_score=23 / 46,
        benchmark_sha256=expected.benchmark_sha256,
        contract_sha256=expected.contract_sha256,
        effective_train_tokens=expected.max_effective_train_tokens,
        generator=expected.generator,
        generator_revision=expected.generator_revision,
        generation_recipe_sha256=expected.generation_recipe_sha256,
        generation_manifest_sha256="c" * 64,
        selection_seed=expected.selection_seed,
        selection_config_sha256=expected.selection_config_sha256,
        selection_manifest_sha256="d" * 64,
        full_traversal_report_sha256=None,
        accepted_sft_sha256="e" * 64,
        checkpoint_manifest_sha256="f" * 64,
        predictions_sha256="1" * 64,
        evaluated_item_count=46,
        correct_item_count=25,
        baseline_correct_item_count=23,
    )

    audit = audit_experiment_manifest(manifest, [score, score])
    assert not audit.complete
    assert audit.duplicate_run_ids == [score.run_id]
    assert len(audit.missing_run_ids) == manifest.expected_trial_count - 1
