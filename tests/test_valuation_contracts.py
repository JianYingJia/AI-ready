import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.valuation.contracts import (
    contract_digest,
    freeze_contract,
    load_batch_suite,
    load_production_contract,
    suite_digest,
    validate_contract_against_suite,
)
from ai_ready.valuation.planning import build_reference_plan
from ai_ready.valuation.schema import ProductionContract

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "valuation_cll"


def _models():
    return (
        load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml"),
        load_production_contract(EXAMPLE_DIR / "production_contract.yaml"),
    )


def _with_content_hashes(suite):
    suite = suite.model_copy(deep=True)
    for batch in suite.batches:
        batch.content_sha256 = hashlib.sha256(batch.batch_id.encode("utf-8")).hexdigest()
    return suite


def test_contract_hash_is_semantic_and_changes_with_prompt():
    _, contract = _models()
    same = ProductionContract.model_validate(contract.model_dump(mode="json"))
    assert contract_digest(contract) == contract_digest(same)
    changed = contract.model_copy(deep=True)
    changed.generation.proxy.prompt_sha256 = "a" * 64
    assert contract_digest(contract) != contract_digest(changed)


def test_contract_freezes_independent_proxy_and_reference_generators():
    _, contract = _models()

    assert contract.generation.proxy.generator == "coverage_guided_sft"
    assert contract.generation.reference.generator == "fully_open_meditron"
    assert contract.generation.proxy.generator != contract.generation.reference.generator

    data = contract.model_dump(mode="json")
    data["generation"]["proxy"]["generator"] = data["generation"]["reference"]["generator"]
    with pytest.raises(ValidationError, match="independent generators"):
        ProductionContract.model_validate(data)


def test_suite_and_contract_must_bind_the_same_capability_blueprint():
    suite, contract = _models()
    suite.capability_blueprint_sha256 = "a" * 64
    report = validate_contract_against_suite(suite, contract)
    assert not report.valid
    assert any("different capability blueprints" in error for error in report.errors)


def test_contract_validation_rejects_another_proxy_implementation():
    suite, contract = _models()
    contract.generation.proxy.implementation_sha256 = "f" * 64

    report = validate_contract_against_suite(suite, contract)

    assert not report.valid
    assert any("local_sft.py implementation" in error for error in report.errors)


def test_reference_contract_rejects_proxy_selection_and_test_exposure():
    _, contract = _models()
    data = contract.model_dump(mode="json")
    data["reference_policy"]["uses_proxy_selector"] = True
    with pytest.raises(ValidationError, match="proxy selector"):
        ProductionContract.model_validate(data)
    data = contract.model_dump(mode="json")
    data["benchmark"]["generation_can_access_test_items"] = True
    with pytest.raises(ValidationError, match="cannot be exposed"):
        ProductionContract.model_validate(data)


def test_reference_contract_requires_three_unique_seeds():
    _, contract = _models()
    data = contract.model_dump(mode="json")
    data["training"]["seeds"] = [1, 2]
    with pytest.raises(ValidationError):
        ProductionContract.model_validate(data)
    data["training"]["seeds"] = [1, 1, 2]
    with pytest.raises(ValidationError, match="unique"):
        ProductionContract.model_validate(data)


def test_plan_covers_every_batch_and_paired_seed(tmp_path):
    suite, contract = _models()
    suite = _with_content_hashes(suite)
    report = validate_contract_against_suite(suite, contract)
    assert report.valid
    plan = build_reference_plan(suite, contract, tmp_path / "run")
    assert plan.suite_sha256 == suite_digest(suite)
    assert plan.contract_sha256 == contract_digest(contract)
    assert set(plan.batch_ids) == {batch.batch_id for batch in suite.batches}
    assert len(plan.stages) == 108  # 12 × (3 batch stages + 2 trial stages × 3 seeds)
    for batch in suite.batches:
        train_seeds = {
            stage.seed for stage in plan.stages if stage.batch_id == batch.batch_id and stage.stage == "train"
        }
        assert train_seeds == set(contract.training.seeds)
        assert sum(stage.stage == "generate" and stage.batch_id == batch.batch_id for stage in plan.stages) == 1
        assert all(
            stage.suite_sha256 == plan.suite_sha256
            and stage.batch_content_sha256 == batch.content_sha256
            and stage.contract_sha256 == plan.contract_sha256
            for stage in plan.stages
            if stage.batch_id == batch.batch_id
        )


def test_plan_requires_frozen_batch_content_and_binds_run_id_to_it(tmp_path):
    suite, contract = _models()
    with pytest.raises(ValueError, match="run freeze-batches"):
        build_reference_plan(suite, contract, tmp_path / "unfrozen")

    first_suite = _with_content_hashes(suite)
    first_plan = build_reference_plan(first_suite, contract, tmp_path / "first")
    second_suite = first_suite.model_copy(deep=True)
    second_suite.batches[0].content_sha256 = "a" * 64
    second_plan = build_reference_plan(second_suite, contract, tmp_path / "second")
    first_run = next(stage.run_id for stage in first_plan.stages if stage.batch_id == "B01")
    second_run = next(stage.run_id for stage in second_plan.stages if stage.batch_id == "B01")
    assert first_run != second_run


def test_production_contract_requires_all_five_stages_with_fixed_scopes():
    _, contract = _models()
    missing_gate = contract.model_dump(mode="json")
    missing_gate["tools"] = [tool for tool in missing_gate["tools"] if tool["stage"] != "quality_gate"]
    with pytest.raises(ValidationError, match="at least 5 items|missing tool stages"):
        ProductionContract.model_validate(missing_gate)

    wrong_scope = contract.model_dump(mode="json")
    next(tool for tool in wrong_scope["tools"] if tool["stage"] == "train")["scope"] = "batch"
    with pytest.raises(ValidationError, match="train must use trial scope"):
        ProductionContract.model_validate(wrong_scope)

    wrong_order = contract.model_dump(mode="json")
    wrong_order["tools"][0], wrong_order["tools"][1] = wrong_order["tools"][1], wrong_order["tools"][0]
    with pytest.raises(ValidationError, match="dependency order"):
        ProductionContract.model_validate(wrong_order)


def test_freeze_contract_is_idempotent(tmp_path):
    _, contract = _models()
    first = freeze_contract(contract, tmp_path)
    second = freeze_contract(contract, tmp_path)
    assert first == second
    assert (tmp_path / first["frozen_file"]).is_file()
