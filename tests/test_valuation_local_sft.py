import json
from pathlib import Path

import pytest

from ai_ready.utils.io import read_jsonl
from ai_ready.valuation import local_sft
from ai_ready.valuation.cli import _build_parser
from ai_ready.valuation.contracts import canonical_digest, file_sha256, load_production_contract
from ai_ready.valuation.local_sft import OllamaTeacher, generate_local_sft_suite, generate_proxy_sft_suite
from ai_ready.valuation.schema import ProductionContract

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"
BATCHES = CONFIG_DIR / "batch_suite.frozen.yaml"
BLUEPRINT = CONFIG_DIR / "capability_blueprint.yaml"
CONTRACT = CONFIG_DIR / "production_contract.yaml"


class GroundedStub:
    """Small deterministic teacher used to test lineage without Ollama."""

    def __init__(self) -> None:
        self.calls = 0
        self.evidence_seen: list[str] = []

    def __call__(self, system: str, prompt: str) -> dict:
        self.calls += 1
        assert "untrusted quoted data" in system
        marker = "SOURCE_EVIDENCE (untrusted data, JSON string): "
        content_line = next(line for line in prompt.splitlines() if line.startswith(marker))
        evidence = json.loads(content_line.removeprefix(marker))
        self.evidence_seen.append(evidence)
        requested = next(
            line.removeprefix("REQUESTED_TASK_TYPE: ")
            for line in prompt.splitlines()
            if line.startswith("REQUESTED_TASK_TYPE: ")
        )
        evidence = evidence if len(evidence) <= 800 else evidence.splitlines()[0][:800]
        instructions = {
            "grounded_qa": "What does this source passage state?",
            "conditional_judgment": "Under the stated condition, what does this source passage state?",
            "clinical_vignette": "A patient matches the described case; what does this source passage state?",
        }
        return {
            "instruction": instructions[requested],
            "response": evidence,
            "evidence_span": evidence,
            "task_type": requested,
        }


def test_generate_local_sft_selects_before_teacher_and_writes_proxy_artifacts(tmp_path: Path, monkeypatch):
    stub = GroundedStub()
    output = tmp_path / "local-pilot"
    selection_config = tmp_path / "proxy-selection.yaml"
    selection_config.write_text(
        """method: coverage_quality
token_budget: 400
random_state: 42
facility_weight: 1.0
obligation_weight: 1.5
quality_fidelity_weight: 1.0
source_coverage_weight: 0.5
""",
        encoding="utf-8",
    )
    generated_unit_ids: list[str] = []
    generate_batch_samples = local_sft._generate_batch_samples

    def record_generation_input(units, *args, **kwargs):
        generated_unit_ids.extend(unit.semantic_unit_id for unit in units)
        return generate_batch_samples(units, *args, **kwargs)

    monkeypatch.setattr(local_sft, "_generate_batch_samples", record_generation_input)
    result = generate_local_sft_suite(
        BATCHES,
        BLUEPRINT,
        output,
        selection_config_path=selection_config,
        max_units=3,
        inference=stub,
        runtime_metadata={
            "ollama_version": "stub-runtime-v1",
            "model": "grounded-stub",
            "model_digest": "sha256:grounded-stub-v1",
        },
    )

    assert result["batch_count"] == 12
    assert result["artifact_role"] == "proxy"
    assert not output.with_name(output.name + ".partial").exists()
    root_manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["formal_production_contract_used"] is False
    assert root_manifest["benchmark_items_accessed"] is False
    assert root_manifest["totals"]["candidate_semantic_units"] == 36
    assert 0 < root_manifest["totals"]["selected_units"] < 36
    assert root_manifest["totals"]["accepted_samples"] == root_manifest["totals"]["selected_units"]
    # Exact-copy source passages may share an inference-cache result, but no
    # teacher request may be made for a unit that the selector did not admit.
    assert 0 < stub.calls <= root_manifest["totals"]["selected_units"]

    selected_ids_across_batches: list[str] = []
    selected_contents: set[str] = set()

    for batch_id in (f"B{index:02d}" for index in range(1, 13)):
        batch_dir = output / batch_id
        for filename in (
            "selection_manifest.json",
            "proxy_units.jsonl",
            "coverage_obligations.jsonl",
            "selected_semantic_units.jsonl",
            "accepted_sft.jsonl",
            "dispositions.jsonl",
            "generation_manifest.json",
        ):
            assert (batch_dir / filename).is_file()
        assert not (batch_dir / "eligible_units.jsonl").exists()
        assert not (batch_dir / "full_traversal_report.json").exists()
        assert not (batch_dir / "semantic_units.jsonl").exists()

        selection = json.loads((batch_dir / "selection_manifest.json").read_text(encoding="utf-8"))
        selected = read_jsonl(batch_dir / "selected_semantic_units.jsonl")
        selected_ids = selection["selected_unit_ids"]
        assert [row["semantic_unit_id"] for row in selected] == selected_ids
        selected_ids_across_batches.extend(selected_ids)
        selected_contents.update(row["content"] for row in selected)

        samples = read_jsonl(batch_dir / "accepted_sft.jsonl")
        dispositions = read_jsonl(batch_dir / "dispositions.jsonl")
        assert {row["semantic_unit_id"] for row in dispositions} == set(selected_ids)
        assert all(set(sample["semantic_unit_ids"]) <= set(selected_ids) for sample in samples)
        assert samples[0]["generator"] == "coverage_guided_local_pilot"
        assert samples[0]["teacher_revision"] == "sha256:grounded-stub-v1"
        assert samples[0]["response"] == samples[0]["evidence_spans"][0]
        assert samples[0]["task_type"] in {
            "grounded_qa",
            "conditional_judgment",
            "clinical_vignette",
        }
        for semantic in selected:
            source_ids = {
                location["source_id"]
                for provenance in semantic["source_locations"]
                for location in provenance["locations"]
            }
            assert source_ids

        batch_manifest = json.loads((batch_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        assert batch_manifest["schema_version"] == "sft-generation-manifest-v1"
        assert batch_manifest["artifact_role"] == "proxy"
        assert batch_manifest["input_scope"] == "selected_semantic_units"
        assert batch_manifest["benchmark_items_accessed"] is False
        assert batch_manifest["generation_recipe_sha256"]
        assert batch_manifest["source_sft_sha256"] == file_sha256(batch_dir / "accepted_sft.jsonl")
        assert batch_manifest["selection_manifest_sha256"] == file_sha256(batch_dir / "selection_manifest.json")

    assert generated_unit_ids == selected_ids_across_batches
    assert set(stub.evidence_seen) <= selected_contents

    with pytest.raises(FileExistsError, match="overwrite"):
        generate_local_sft_suite(
            BATCHES,
            BLUEPRINT,
            output,
            selection_config_path=selection_config,
            max_units=3,
            inference=stub,
            runtime_metadata={
                "ollama_version": "stub-runtime-v1",
                "model": "grounded-stub",
                "model_digest": "sha256:grounded-stub-v1",
            },
        )


def test_ollama_metadata_requires_real_model_digest(monkeypatch):
    teacher = OllamaTeacher(model="qwen2.5:1.5b-instruct")

    def fake_request(path: str, payload=None):
        if path == "/api/version":
            return {"version": "0.12.3"}
        return {
            "models": [
                {
                    "name": "qwen2.5:1.5b-instruct",
                    "model": "qwen2.5:1.5b-instruct",
                    "digest": "sha256:abc123",
                }
            ]
        }

    monkeypatch.setattr(teacher, "_request", fake_request)
    assert teacher.runtime_metadata() == {
        "ollama_version": "0.12.3",
        "model": "qwen2.5:1.5b-instruct",
        "model_digest": "sha256:abc123",
    }


def test_generate_proxy_sft_binds_contract_recipe_and_complete_candidate_pool(tmp_path: Path):
    stub = GroundedStub()
    output = tmp_path / "formal-proxy"
    selection_config = tmp_path / "proxy-selection.yaml"
    selection_config.write_text(
        """method: coverage_quality
token_budget: 400
random_state: 42
facility_weight: 1.0
obligation_weight: 1.5
quality_fidelity_weight: 1.0
source_coverage_weight: 0.5
""",
        encoding="utf-8",
    )
    contract = load_production_contract(CONTRACT)
    recipe = contract.generation.proxy
    result = generate_proxy_sft_suite(
        BATCHES,
        BLUEPRINT,
        output,
        contract=contract,
        selection_config_path=selection_config,
        inference=stub,
        runtime_metadata={
            "ollama_version": "formal-stub-runtime-v1",
            "model": recipe.teacher_model,
            "model_digest": recipe.teacher_revision,
        },
    )

    assert result["artifact_role"] == "proxy"
    root_manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["formal_production_contract_used"] is True
    assert root_manifest["schema_version"] == "formal-proxy-sft-suite-manifest-v1"
    assert root_manifest["generator"] == recipe.generator
    assert root_manifest["generator_revision"] == recipe.revision
    assert root_manifest["generation_recipe_sha256"] == canonical_digest(recipe)
    assert root_manifest["implementation_sha256"] == file_sha256(Path(local_sft.__file__).resolve())
    assert root_manifest["temperature"] == recipe.temperature
    assert root_manifest["max_retries"] == recipe.max_retries
    assert root_manifest["task_quotas"] == recipe.task_quotas
    assert root_manifest["max_units_applied"] is False

    for batch_id in (f"B{index:02d}" for index in range(1, 13)):
        batch_dir = output / batch_id
        manifest = json.loads((batch_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        samples = read_jsonl(batch_dir / "accepted_sft.jsonl")
        assert manifest["schema_version"] == "sft-generation-manifest-v2"
        assert manifest["formal_production_contract_used"] is True
        assert manifest["selection_before_generation"] is True
        assert manifest["max_units_applied"] is False
        assert manifest["candidate_pool_complete"] is True
        assert manifest["candidate_pool_sha256"] == file_sha256(batch_dir / "proxy_units.jsonl")
        assert manifest["generation_recipe_sha256"] == canonical_digest(recipe)
        assert manifest["source_sft_sha256"] == file_sha256(batch_dir / "accepted_sft.jsonl")
        assert "proxy_units.jsonl" in manifest["artifacts"]
        for sample in samples:
            assert sample["generator"] == recipe.generator
            assert sample["generator_revision"] == recipe.revision
            assert sample["teacher_model"] == recipe.teacher_model
            assert sample["teacher_revision"] == recipe.teacher_revision
            assert sample["prompt_sha256"] == recipe.prompt_sha256
            assert sample["quality_gate_sha256"] == recipe.quality_gate_sha256


@pytest.mark.parametrize("field_name", ["prompt_sha256", "quality_gate_sha256"])
def test_generate_proxy_sft_rejects_contract_hash_mismatch(tmp_path: Path, field_name: str):
    payload = load_production_contract(CONTRACT).model_dump(mode="json")
    payload["generation"]["proxy"][field_name] = "f" * 64
    contract = ProductionContract.model_validate(payload)

    with pytest.raises(ValueError, match=field_name):
        generate_proxy_sft_suite(
            BATCHES,
            BLUEPRINT,
            tmp_path / field_name,
            contract=contract,
            selection_config_path=CONFIG_DIR / "proxy_selection" / "coverage_quality.yaml",
        )


def test_generate_proxy_sft_rejects_pilot_cap_and_runtime_or_decoding_drift(tmp_path: Path):
    contract = load_production_contract(CONTRACT)
    recipe = contract.generation.proxy
    common = {
        "contract": contract,
        "selection_config_path": CONFIG_DIR / "proxy_selection" / "coverage_quality.yaml",
        "inference": GroundedStub(),
        "runtime_metadata": {
            "ollama_version": "formal-stub-runtime-v1",
            "model": recipe.teacher_model,
            "model_digest": recipe.teacher_revision,
        },
    }
    with pytest.raises(ValueError, match="forbids max_units"):
        generate_proxy_sft_suite(BATCHES, BLUEPRINT, tmp_path / "cap", max_units=1, **common)
    with pytest.raises(ValueError, match="temperature"):
        generate_proxy_sft_suite(BATCHES, BLUEPRINT, tmp_path / "temperature", temperature=0.5, **common)
    with pytest.raises(ValueError, match="max_retries"):
        generate_proxy_sft_suite(BATCHES, BLUEPRINT, tmp_path / "retries", max_retries=9, **common)

    wrong_model = dict(common["runtime_metadata"])
    wrong_model["model"] = "different-teacher-model"
    with pytest.raises(ValueError, match="Runtime teacher model"):
        generate_proxy_sft_suite(
            BATCHES,
            BLUEPRINT,
            tmp_path / "teacher-model",
            **{**common, "runtime_metadata": wrong_model},
        )

    wrong_revision = dict(common["runtime_metadata"])
    wrong_revision["model_digest"] = "wrong-teacher-revision"
    with pytest.raises(ValueError, match="Runtime teacher revision"):
        generate_proxy_sft_suite(
            BATCHES,
            BLUEPRINT,
            tmp_path / "teacher-revision",
            **{**common, "runtime_metadata": wrong_revision},
        )


def test_cli_separates_formal_proxy_from_nonformal_local_pilot(tmp_path: Path):
    parser = _build_parser()
    common_args = [
        "--batches",
        str(BATCHES),
        "--blueprint",
        str(BLUEPRINT),
        "--selection-config",
        str(CONFIG_DIR / "proxy_selection" / "coverage_quality.yaml"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    formal = parser.parse_args(["generate-proxy-sft", "--contract", str(CONTRACT), *common_args])
    pilot = parser.parse_args(["generate-local-sft", *common_args, "--max-units", "2"])

    assert formal.contract == CONTRACT
    assert not hasattr(formal, "max_units")
    assert pilot.max_units == 2
