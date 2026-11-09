import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.utils.io import read_jsonl, write_jsonl
from ai_ready.valuation.conditional_value import ScoredProbe
from ai_ready.valuation.contracts import file_sha256
from ai_ready.valuation.model_probe import (
    BackendGeneration,
    BackendRuntime,
    CheckpointManifest,
    FourChoiceProbe,
    ProbeInferenceConfig,
    ProbeRunManifest,
    build_checkpoint_manifest,
    build_probe_prompts,
    load_probe_catalogue,
    parse_answer_choice,
    run_model_probe_pipeline,
    verify_production_contract_binding,
)

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _probe_row(tmp_path: Path) -> dict:
    source_dir = tmp_path / "data"
    source_dir.mkdir(exist_ok=True)
    processed = source_dir / "public.md"
    processed.write_text("# Public source\n", encoding="utf-8")
    evidence = "The registered threshold is five."
    block_text = f"Context. {evidence} More context."
    blocks = source_dir / "public.blocks.jsonl"
    write_jsonl(blocks, [{"kind": "paragraph", "section": "Definition", "text": block_text}])
    return {
        "schema_version": "cll-independent-capability-probe-v1",
        "probe_id": "probe-001",
        "item_id": "probe-001",
        "variant_id": "base",
        "cell_id": "diagnosis_and_differential",
        "question": "What is the registered threshold?",
        "options": ["One", "Five", "Ten", "Twenty"],
        "answer_index": 1,
        "answer_index_base": 0,
        "answer_text": "Five",
        "rationale": "The public evidence explicitly states five.",
        "source_evidence": evidence,
        "source_evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        "source_file": "data/public.md",
        "source_sha256": file_sha256(processed),
        "evidence_location": "data/public.md#block=0",
        "source_location": {
            "source_id": "public-source",
            "title": "Public source",
            "canonical_url": "https://example.org/public",
            "version": "v1",
            "knowledge_cutoff": "2026-01-01",
            "processed_path": "data/public.md",
            "processed_sha256": file_sha256(processed),
            "blocks_path": "data/public.blocks.jsonl",
            "blocks_sha256": file_sha256(blocks),
            "block_index_base": 0,
            "block_index": 0,
            "section": "Definition",
            "source_block_text_sha256": hashlib.sha256(block_text.encode("utf-8")).hexdigest(),
        },
        "temporal_risk": "low",
        "locked_test": False,
        "test_items_included": False,
        "is_locked_test_item": False,
    }


class _FakeBackend:
    def __init__(self):
        self._runtime = BackendRuntime(
            backend="fake-test-backend",
            python_version="3.11-test",
            transformers_version="test",
            torch_version="test",
            cuda_available=False,
            model_class="FakeModel",
            tokenizer_class="FakeTokenizer",
            chat_template_sha256="a" * 64,
            requested_dtype="float32",
            parameter_dtype="torch.float32",
            model_devices=["cpu"],
        )

    @property
    def runtime(self):
        return self._runtime

    def generate(self, prompts, *, batch_size, max_new_tokens, seed):
        del batch_size, max_new_tokens, seed
        generations = []
        for prompt in prompts:
            # The helper catalogue's correct original option is index 1.  Map
            # it back to the displayed letter after the v2 option rotation.
            displayed_index = prompt.option_order.index(1)
            rendered = json.dumps(
                [message.model_dump(mode="json") for message in prompt.messages],
                ensure_ascii=False,
                sort_keys=True,
            )
            generations.append(BackendGeneration(rendered_prompt=rendered, raw_output="ABCD"[displayed_index]))
        return generations


def test_four_choice_schema_rejects_locked_and_derived_items(tmp_path):
    payload = _probe_row(tmp_path)
    with pytest.raises(ValidationError, match="Locked benchmark"):
        FourChoiceProbe.model_validate({**payload, "locked_test": True})
    with pytest.raises(ValidationError, match="Locked benchmark"):
        FourChoiceProbe.model_validate({**payload, "derived_from_locked_test_item": True})
    with pytest.raises(ValidationError, match="answer_text"):
        FourChoiceProbe.model_validate({**payload, "answer_text": "One"})


def test_locked_catalogue_path_is_rejected_before_read(tmp_path):
    forbidden = tmp_path / "data" / "private" / "benchmark_locked" / "items.jsonl"
    with pytest.raises(ValueError, match="file was not read"):
        load_probe_catalogue(forbidden)


@pytest.mark.parametrize(
    ("raw", "status", "letter"),
    [
        ("A", "valid", "A"),
        ("答案：Ｃ。", "valid", "C"),
        ("(D)", "valid", "D"),
        ("A or B", "ambiguous", None),
        ("A because it is correct", "invalid", None),
        ("I cannot tell", "invalid", None),
    ],
)
def test_choice_parser_is_strict_and_deterministic(raw, status, letter):
    result = parse_answer_choice(raw)
    assert result.status == status
    assert result.answer_letter == letter


def test_second_prompt_variant_deterministically_rotates_options(tmp_path):
    probe = FourChoiceProbe.model_validate(_probe_row(tmp_path))
    first = build_probe_prompts([probe], ["clinical_mcq_letter_v1", "clinical_mcq_letter_v2"])
    second = build_probe_prompts([probe], ["clinical_mcq_letter_v1", "clinical_mcq_letter_v2"])
    assert first == second
    assert first[0].option_order == [0, 1, 2, 3]
    assert first[1].option_order != [0, 1, 2, 3]
    assert sorted(first[1].option_order) == [0, 1, 2, 3]


def test_checkpoint_manifest_binds_revision_and_all_stable_files(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    cache = snapshot / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "mutable.lock").write_text("ignored", encoding="utf-8")

    first = build_checkpoint_manifest(snapshot, model_id="org/model", model_revision=REVISION)
    assert [item.relative_path for item in first.files] == ["config.json", "model.safetensors"]
    CheckpointManifest.model_validate(first.model_dump(mode="json"))

    (snapshot / "model.safetensors").write_bytes(b"changed weights")
    second = build_checkpoint_manifest(snapshot, model_id="org/model", model_revision=REVISION)
    assert first.checkpoint_sha256 != second.checkpoint_sha256


def test_production_contract_binding_checks_hash_and_frozen_model_identity():
    contract = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility" / "production_contract.yaml"
    config = ProbeInferenceConfig(
        run_id="contract-test",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_revision=REVISION,
        snapshot_path=PROJECT_DIR,
        production_contract_path=contract,
        production_contract_sha256=file_sha256(contract),
    )

    binding = verify_production_contract_binding(config)
    assert binding == {"path": str(contract.resolve()), "sha256": file_sha256(contract)}
    with pytest.raises(ValueError, match="model_id"):
        verify_production_contract_binding(config.model_copy(update={"model_id": "wrong/model"}))


def test_pipeline_writes_profile_compatible_scores_and_auditable_hash_chain(tmp_path):
    catalogue_path = tmp_path / "inputs" / "probes.jsonl"
    catalogue_path.parent.mkdir()
    write_jsonl(catalogue_path, [_probe_row(tmp_path)])
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    config = ProbeInferenceConfig(
        run_id="unit-test-run",
        model_id="org/model",
        model_revision=REVISION,
        snapshot_path=snapshot,
        source_root=tmp_path,
        dtype="float32",
        device_map="cpu",
        batch_size=2,
    )

    artifacts = run_model_probe_pipeline(
        catalogue_path=catalogue_path,
        output_dir=tmp_path / "output",
        config=config,
        backend=_FakeBackend(),
    )

    manifest = ProbeRunManifest.model_validate(json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8")))
    assert manifest.prompt_count == 2
    assert manifest.probe_count == 1
    assert manifest.valid_prediction_count == 2
    assert manifest.predictions_sha256 == file_sha256(artifacts.predictions_path)
    assert manifest.scored_probes_sha256 == file_sha256(artifacts.scored_probes_path)
    assert manifest.checkpoint_manifest_sha256 == file_sha256(artifacts.checkpoint_manifest_path)
    assert artifacts.run_manifest_sha256 == file_sha256(artifacts.run_manifest_path)
    assert manifest.benchmark_test_items_accessed is False
    scored = [ScoredProbe.model_validate(row) for row in read_jsonl(artifacts.scored_probes_path)]
    assert len(scored) == 2
    assert {row.item_id for row in scored} == {"probe-001"}
    assert {row.variant_id for row in scored} == {
        "clinical_mcq_letter_v1",
        "clinical_mcq_letter_v2",
    }
    assert all(row.score == 1.0 for row in scored)
