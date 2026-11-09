import json
from pathlib import Path

import pytest

from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.contracts import canonical_digest, contract_digest, file_sha256
from ai_ready.valuation.reference_sft import import_fully_open_meditron_reference
from ai_ready.valuation.schema import ProductionContract


def _contract(*, low_yield_policy: str = "record_and_continue") -> ProductionContract:
    digest = "a" * 64
    reference_revision = "65d23a58df15cffdc151fd366083cbe4e49615be"
    recipe_common = {
        "teacher_model": "openai/gpt-oss-120b",
        "teacher_revision": "teacher-revision-1",
        "prompt_sha256": digest,
        "quality_gate_sha256": "b" * 64,
        "temperature": 0.7,
        "max_retries": 8,
        "task_quotas": {
            "grounded_qa": 1,
            "conditional_judgment": 1,
            "clinical_vignette": 1,
        },
        "preserve_source_evidence": True,
    }
    return ProductionContract.model_validate(
        {
            "schema_version": "production-contract-v2",
            "contract_id": "reference-import-test",
            "version": "v1",
            "reference_policy": {
                "traversal": "full",
                "uses_proxy_selector": False,
                "run_all_batches": True,
                "low_yield_policy": low_yield_policy,
                "allow_sample_duplication": False,
                "allow_batch_specific_quality_thresholds": False,
            },
            "generation": {
                "proxy": {
                    **recipe_common,
                    "generator": "coverage_guided_sft",
                    "revision": "proxy-v1",
                    "implementation_sha256": "9" * 64,
                },
                "reference": {
                    **recipe_common,
                    "generator": "fully_open_meditron",
                    "revision": reference_revision,
                },
            },
            "training": {
                "objective": "sft",
                "update_method": "lora",
                "trainer": "llama_factory",
                "trainer_revision": "v0.9.5",
                "base_model": "Qwen/Qwen2.5-7B-Instruct",
                "base_model_revision": "model-revision",
                "tokenizer_revision": "tokenizer-revision",
                "chat_template_sha256": "c" * 64,
                "effective_train_tokens": 1_000_000,
                "response_only_loss": True,
                "seeds": [17, 29, 43],
                "hyperparameters": {"learning_rate": 0.00002},
                "background_mix_ratio": 0.0,
            },
            "benchmark": {
                "benchmark_id": "benchmark",
                "version": "v1",
                "capability_blueprint_sha256": "d" * 64,
                "test_manifest_sha256": "e" * 64,
                "evaluation_engine": "opencompass",
                "evaluation_engine_revision": "revision",
                "primary_metric": "accuracy",
                "higher_is_better": True,
                "generation_can_access_test_items": False,
                "decoding": {"temperature": 0.0},
            },
            "budget": {
                "value_target": "fixed_budget",
                "generation_token_cap": 5_000_000,
                "training_token_cap": 1_000_000,
                "cost_cap": 100.0,
                "stop_rule": "stop",
            },
            "tools": [
                {
                    "stage": "preprocess",
                    "adapter": "data_juicer",
                    "repository": "https://example.test/data-juicer",
                    "revision": "v1",
                    "config_sha256": digest,
                    "scope": "batch",
                    "argv": [],
                },
                {
                    "stage": "generate",
                    "adapter": "fully_open_meditron",
                    "repository": "https://github.com/EPFLiGHT/FullyOpenMeditron",
                    "revision": reference_revision,
                    "config_sha256": "f" * 64,
                    "scope": "batch",
                    "argv": [],
                },
                {
                    "stage": "quality_gate",
                    "adapter": "local_medical_quality_gate",
                    "repository": "local://quality-gate",
                    "revision": "v1",
                    "config_sha256": "1" * 64,
                    "scope": "batch",
                    "argv": [],
                },
                {
                    "stage": "train",
                    "adapter": "llama_factory",
                    "repository": "https://example.test/llamafactory",
                    "revision": "v0.9.5",
                    "config_sha256": "2" * 64,
                    "scope": "trial",
                    "argv": [],
                },
                {
                    "stage": "evaluate",
                    "adapter": "opencompass",
                    "repository": "https://example.test/opencompass",
                    "revision": "v1",
                    "config_sha256": "3" * 64,
                    "scope": "trial",
                    "argv": [],
                },
            ],
        }
    )


def _write_inputs(tmp_path: Path, *, include_unknown: bool = False) -> tuple[Path, Path]:
    eligible = tmp_path / "eligible.jsonl"
    fom = tmp_path / "fom.jsonl"
    write_jsonl(
        eligible,
        [
            {"semantic_unit_id": "u1", "batch_id": "B01", "source_ids": ["doc-1"]},
            {"semantic_unit_id": "u2", "batch_id": "B01", "source_ids": ["doc-2"]},
            {"semantic_unit_id": "u3", "batch_id": "B01", "source_ids": ["doc-3"]},
        ],
    )
    rows = [
        {
            "semantic_unit_id": "u1",
            "instruction": "What is stated for unit one?",
            "response": "Unit one response.",
            "evidence_spans": ["Unit one evidence."],
            "source_ids": ["doc-1"],
            "task_type": "grounded_qa",
            "upstream_metadata": {"kept": "by-input-hash"},
        },
        {
            "semantic_unit_id": "unknown" if include_unknown else "u2",
            "instruction": "Under the stated condition, what follows?",
            "response": "Unit two response.",
            "evidence_spans": ["Unit two evidence."],
            "source_ids": ["doc-2"],
            "task_type": "conditional_judgment",
        },
    ]
    write_jsonl(fom, rows)
    return eligible, fom


def _write_run_manifest(
    tmp_path: Path,
    *,
    contract: ProductionContract,
    eligible: Path,
    fom: Path,
    overrides: dict | None = None,
) -> Path:
    recipe = contract.generation.reference
    tool = next(row for row in contract.tools if row.stage == "generate")
    receipt = {
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
        "batch_id": "B01",
        "batch_content_sha256": "4" * 64,
        "eligible_units_sha256": file_sha256(eligible),
        "normalized_output_sha256": file_sha256(fom),
        "benchmark_items_accessed": False,
        "uses_proxy_selector": False,
        "complete_input_traversal": True,
        "formal_production_contract_used": True,
        "contract_id": contract.contract_id,
        "contract_sha256": contract_digest(contract),
    }
    receipt.update(overrides or {})
    path = tmp_path / "fully_open_meditron_run_manifest.json"
    write_json(path, receipt)
    return path


def test_import_fom_reference_records_full_traversal_without_selection(tmp_path: Path, monkeypatch):
    # A hard failure here proves the reference importer does not call the proxy
    # selector even when that module is already loaded in the process.
    import ai_ready.valuation.selection as selection

    monkeypatch.setattr(
        selection,
        "select_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference import called proxy selector")),
    )
    contract = _contract()
    eligible, fom = _write_inputs(tmp_path)
    run_manifest = _write_run_manifest(tmp_path, contract=contract, eligible=eligible, fom=fom)
    output = tmp_path / "reference"

    result = import_fully_open_meditron_reference(
        batch_id="B01",
        batch_content_sha256="4" * 64,
        contract=contract,
        eligible_units_path=eligible,
        fully_open_meditron_path=fom,
        run_manifest_path=run_manifest,
        output_dir=output,
    )

    assert result["accepted_samples"] == 2
    assert result["eligible_units"] == 3
    assert result["failed_units"] == 1
    assert {path.name for path in output.iterdir()} == {
        "accepted_sft.jsonl",
        "dispositions.jsonl",
        "eligible_units.jsonl",
        "full_traversal_report.json",
        "generation_manifest.json",
        "fully_open_meditron_run_manifest.json",
    }
    assert not (output / "selection_manifest.json").exists()
    assert not (output / "proxy_units.jsonl").exists()
    assert not (output / "coverage_obligations.jsonl").exists()

    samples = read_jsonl(output / "accepted_sft.jsonl")
    recipe = contract.generation.reference
    assert {row["generator"] for row in samples} == {recipe.generator}
    assert {row["generator_revision"] for row in samples} == {recipe.revision}
    assert {row["teacher_model"] for row in samples} == {recipe.teacher_model}
    assert {row["prompt_sha256"] for row in samples} == {recipe.prompt_sha256}
    assert {row["quality_gate_sha256"] for row in samples} == {recipe.quality_gate_sha256}

    dispositions = {row["semantic_unit_id"]: row for row in read_jsonl(output / "dispositions.jsonl")}
    assert dispositions["u1"]["status"] == "accepted"
    assert dispositions["u2"]["status"] == "accepted"
    assert dispositions["u3"] == {
        "semantic_unit_id": "u3",
        "status": "failed",
        "sample_ids": [],
        "reason_codes": ["fully_open_meditron_no_accepted_output"],
    }

    report = json.loads((output / "full_traversal_report.json").read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["summary"]["eligible_units"] == 3
    assert report["summary"]["accepted_units"] == 2
    assert report["summary"]["failed_units"] == 1
    assert report["source_sft_sha256"] == file_sha256(output / "accepted_sft.jsonl")

    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "sft-generation-manifest-v2"
    assert manifest["artifact_role"] == "reference"
    assert manifest["input_scope"] == "full_batch"
    assert manifest["batch_id"] == "B01"
    assert manifest["batch_content_sha256"] == "4" * 64
    assert manifest["generator"] == recipe.generator
    assert manifest["generator_revision"] == recipe.revision
    assert manifest["generation_recipe_sha256"] == canonical_digest(recipe)
    assert manifest["source_sft_sha256"] == file_sha256(output / "accepted_sft.jsonl")
    assert manifest["full_traversal_report_sha256"] == file_sha256(output / "full_traversal_report.json")
    assert manifest["benchmark_items_accessed"] is False
    assert manifest["uses_proxy_selector"] is False
    assert manifest["complete_input_traversal"] is True
    assert manifest["formal_production_contract_used"] is True
    assert manifest["upstream_run_manifest_sha256"] == file_sha256(run_manifest)
    assert manifest["selection_artifacts"] == []
    assert manifest["inputs"]["eligible_units_sha256"] == file_sha256(eligible)
    assert manifest["inputs"]["fully_open_meditron_sha256"] == file_sha256(fom)
    assert manifest["inputs"]["fully_open_meditron_run_manifest_sha256"] == file_sha256(run_manifest)
    copied_receipt = output / "fully_open_meditron_run_manifest.json"
    assert file_sha256(copied_receipt) == file_sha256(run_manifest)
    assert result["upstream_run_manifest_sha256"] == file_sha256(run_manifest)


def test_import_fom_reference_rejects_noneligible_rows_without_partial_output(tmp_path: Path):
    eligible, fom = _write_inputs(tmp_path, include_unknown=True)
    contract = _contract()
    run_manifest = _write_run_manifest(tmp_path, contract=contract, eligible=eligible, fom=fom)
    output = tmp_path / "reference"

    with pytest.raises(ValueError, match="non-eligible unit"):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=run_manifest,
            output_dir=output,
        )

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()


def test_import_fom_reference_honors_fail_batch_low_yield_policy(tmp_path: Path):
    eligible, fom = _write_inputs(tmp_path)
    contract = _contract(low_yield_policy="fail_batch")
    run_manifest = _write_run_manifest(tmp_path, contract=contract, eligible=eligible, fom=fom)
    output = tmp_path / "reference"

    with pytest.raises(ValueError, match="no accepted output"):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=run_manifest,
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "fully-open-meditron-run-manifest-v0"),
        ("generator", "self_reported_generator"),
        ("generator_revision", "wrong-revision"),
        ("tool_adapter", "wrong-adapter"),
        ("tool_repository", "https://example.test/wrong-repository"),
        ("tool_config_sha256", "a" * 64),
        ("teacher_model", "wrong-teacher"),
        ("teacher_revision", "wrong-teacher-revision"),
        ("prompt_sha256", "5" * 64),
        ("quality_gate_sha256", "6" * 64),
        ("batch_id", "B02"),
        ("batch_content_sha256", "7" * 64),
        ("eligible_units_sha256", "8" * 64),
        ("normalized_output_sha256", "9" * 64),
        ("benchmark_items_accessed", True),
        ("uses_proxy_selector", True),
        ("complete_input_traversal", False),
        ("formal_production_contract_used", False),
        ("contract_id", "wrong-contract"),
        ("contract_sha256", "0" * 64),
    ],
)
def test_import_fom_reference_rejects_untrusted_run_manifest_claims(tmp_path: Path, field: str, bad_value: object):
    contract = _contract()
    eligible, fom = _write_inputs(tmp_path)
    run_manifest = _write_run_manifest(
        tmp_path,
        contract=contract,
        eligible=eligible,
        fom=fom,
        overrides={field: bad_value},
    )

    with pytest.raises(ValueError):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=run_manifest,
            output_dir=tmp_path / "reference",
        )


def test_import_fom_reference_requires_external_run_manifest(tmp_path: Path):
    contract = _contract()
    eligible, fom = _write_inputs(tmp_path)

    with pytest.raises(FileNotFoundError, match="run manifest"):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=tmp_path / "missing-run-manifest.json",
            output_dir=tmp_path / "reference",
        )


def test_import_fom_reference_rejects_placeholder_revisions(tmp_path: Path):
    contract_payload = _contract().model_dump(mode="json")
    placeholder = "example-unpinned-fom-revision"
    contract_payload["generation"]["reference"]["revision"] = placeholder
    next(row for row in contract_payload["tools"] if row["stage"] == "generate")["revision"] = placeholder
    contract = ProductionContract.model_validate(contract_payload)
    eligible, fom = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="requires real pinned revisions"):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=tmp_path / "not-used.json",
            output_dir=tmp_path / "reference",
        )


def test_import_fom_reference_rejects_input_changed_during_import(tmp_path: Path, monkeypatch):
    import ai_ready.valuation.reference_sft as reference_sft

    contract = _contract()
    eligible, fom = _write_inputs(tmp_path)
    run_manifest = _write_run_manifest(tmp_path, contract=contract, eligible=eligible, fom=fom)
    real_file_sha256 = file_sha256
    fom_calls = 0

    def mutating_file_sha256(path: Path) -> str:
        nonlocal fom_calls
        resolved = Path(path).resolve()
        if resolved == fom.resolve():
            fom_calls += 1
            if fom_calls == 2:
                with fom.open("a", encoding="utf-8") as handle:
                    handle.write("{}\n")
        return real_file_sha256(resolved)

    monkeypatch.setattr(reference_sft, "file_sha256", mutating_file_sha256)

    with pytest.raises(RuntimeError, match="changed during validation"):
        import_fully_open_meditron_reference(
            batch_id="B01",
            batch_content_sha256="4" * 64,
            contract=contract,
            eligible_units_path=eligible,
            fully_open_meditron_path=fom,
            run_manifest_path=run_manifest,
            output_dir=tmp_path / "reference",
        )
