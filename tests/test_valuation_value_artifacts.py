import json
import sys
from pathlib import Path

from ai_ready.utils.io import read_jsonl
from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.value_artifacts import adapt_existing_value_population


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"
ARTIFACT_ROOT = PROJECT_DIR / "outputs" / "cll_local_sft_qwen2_5_1_5b_full_20260921"


def test_adapt_existing_b01_b12_artifacts_marks_missing_mastery_unknown(tmp_path):
    output_dir = tmp_path / "adapted"
    manifest = adapt_existing_value_population(
        suite_path=CONFIG_DIR / "batch_suite.frozen.yaml",
        artifact_root=ARTIFACT_ROOT,
        output_dir=output_dir,
    )

    rows = read_jsonl(output_dir / "value_population_units.jsonl")
    mapping = read_jsonl(output_dir / "membership_map.jsonl")
    persisted = json.loads((output_dir / "adapter_manifest.json").read_text(encoding="utf-8"))

    assert manifest == persisted
    assert manifest["batch_ids"] == [f"B{index:02d}" for index in range(1, 13)]
    assert manifest["model_mastery_status"] == "pending"
    assert manifest["probe_evidence_present"] is False
    assert rows
    assert len(rows) == len(mapping)
    assert all(set(row["model_states"]) == set(row["capability_ids"]) for row in rows)
    assert all(set(row["model_states"].values()) == {"unknown"} for row in rows)
    assert {row["batch_id"] for row in rows} == set(manifest["batch_ids"])
    assert all(row["capability_ids"] for row in rows)
    assert all(row["dependency_group_id"] for row in rows)
    assert all(row["population_weight"] == 1.0 for row in rows)
    assert len(rows) == sum(counts["semantic_units"] for counts in manifest["batch_counts"].values())


def test_adapter_refuses_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "adapted"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("do not overwrite", encoding="utf-8")

    try:
        adapt_existing_value_population(
            suite_path=CONFIG_DIR / "batch_suite.frozen.yaml",
            artifact_root=ARTIFACT_ROOT,
            output_dir=output_dir,
        )
    except FileExistsError as error:
        assert "empty directory" in str(error)
    else:
        raise AssertionError("Expected a nonempty output directory to be rejected")


def test_adapt_value_population_cli(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "cli-adapted"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "adapt-value-population",
            "--batches",
            str(CONFIG_DIR / "batch_suite.frozen.yaml"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["claim_status"] == "partial_pending_model_measurement"
    assert response["measurement_granularity"] == "capability"
    assert (output_dir / "value_population_units.jsonl").is_file()
