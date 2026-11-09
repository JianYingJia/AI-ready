import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from ai_ready.valuation.batches import (
    BATCH_CONTENT_HASH_SCHEMA,
    _local_path,
    batch_content_fingerprint,
    freeze_batch_suite,
    validate_feasibility_design,
)
from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.contracts import load_batch_suite

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "valuation_cll"


def test_cll_suite_has_complete_factorial_design_and_natural_checks():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    report = validate_feasibility_design(suite)
    assert report.valid
    assert report.summary["batch_count"] == 12
    assert report.summary["factorial_batches"] == 8
    assert report.summary["natural_check_batches"] == 4
    assert any("exploratory" in warning for warning in report.warnings)


def test_training_batch_cannot_use_audit_only_source():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    suite.sources[0].license_status = "audit_only"
    report = validate_feasibility_design(suite)
    assert not report.valid
    assert any("audit-only" in error for error in report.errors)


def test_allowed_source_must_explicitly_allow_model_training():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    suite.sources[0].allowed_uses = ["research"]
    report = validate_feasibility_design(suite)
    assert not report.valid
    assert any("not registered for model-training" in error for error in report.errors)


def test_missing_factorial_combination_is_rejected():
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    suite.batches[0].factors = suite.batches[1].factors.model_copy(deep=True)
    report = validate_feasibility_design(suite)
    assert not report.valid
    assert any("exactly once" in error for error in report.errors)


def _write_local_suite(tmp_path: Path) -> Path:
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    (tmp_path / "capability_blueprint.yaml").write_bytes((EXAMPLE_DIR / "capability_blueprint.yaml").read_bytes())
    for batch in suite.batches:
        batch_dir = tmp_path / "inputs" / batch.batch_id
        (batch_dir / "nested").mkdir(parents=True)
        (batch_dir / "source.md").write_bytes(b"same registered source bytes\n")
        (batch_dir / "nested" / "table.csv").write_bytes(b"rule,condition\nA,B\n")
        batch.input_uri = f"inputs/{batch.batch_id}"
        batch.content_sha256 = None
    suite.batches[-1].content_sha256 = batch_content_fingerprint(
        suite,
        suite.batches[-1],
        tmp_path / "inputs" / suite.batches[-1].batch_id,
    )
    source_path = tmp_path / "batch_suite.yaml"
    source_path.write_text(
        yaml.safe_dump(suite.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return source_path


def test_batch_fingerprint_is_independent_of_root_path_and_file_creation_order(tmp_path):
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    batch = suite.batches[0]
    first = tmp_path / "first-root"
    second = tmp_path / "renamed-root"
    (first / "nested").mkdir(parents=True)
    (second / "nested").mkdir(parents=True)
    (first / "a.txt").write_bytes(b"alpha")
    (first / "nested" / "b.txt").write_bytes(b"beta")
    (second / "nested" / "b.txt").write_bytes(b"beta")
    (second / "a.txt").write_bytes(b"alpha")

    assert batch_content_fingerprint(suite, batch, first) == batch_content_fingerprint(suite, batch, second)

    (second / "nested" / "b.txt").rename(second / "nested" / "renamed.txt")
    assert batch_content_fingerprint(suite, batch, first) != batch_content_fingerprint(suite, batch, second)


def test_single_file_fingerprint_ignores_parent_path_but_keeps_filename(tmp_path):
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    batch = suite.batches[0]
    first = tmp_path / "one" / "source.md"
    second = tmp_path / "two" / "source.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")

    assert batch_content_fingerprint(suite, batch, first) == batch_content_fingerprint(suite, batch, second)

    renamed = second.with_name("renamed.md")
    second.rename(renamed)
    assert batch_content_fingerprint(suite, batch, first) != batch_content_fingerprint(suite, batch, renamed)


def test_batch_fingerprint_includes_construction_metadata_but_not_run_controls(tmp_path):
    content = tmp_path / "content"
    content.write_bytes(b"identical medical source")
    suite = load_batch_suite(EXAMPLE_DIR / "batch_suite.yaml")
    batch = suite.batches[0]

    run_variant = batch.model_copy(deep=True)
    run_variant.batch_id = "DISPLAY-NAME"
    run_variant.split = "evaluation"
    run_variant.input_uri = "a/different/output/location"
    run_variant.paired_with = ["B02"]
    assert batch_content_fingerprint(suite, batch, content) == batch_content_fingerprint(suite, run_variant, content)

    construction_variant = batch.model_copy(deep=True)
    construction_variant.factors.redundancy = "high"
    assert batch_content_fingerprint(suite, batch, content) != batch_content_fingerprint(
        suite, construction_variant, content
    )

    notes_variant = batch.model_copy(deep=True)
    notes_variant.construction_notes.append("Keep duplicated complete rule groups")
    assert batch_content_fingerprint(suite, batch, content) != batch_content_fingerprint(suite, notes_variant, content)

    source_variant = suite.model_copy(deep=True)
    source_variant.sources[0].version = "a-new-source-version"
    assert batch_content_fingerprint(suite, batch, content) != batch_content_fingerprint(
        source_variant, source_variant.batches[0], content
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows file URI rules")
def test_local_path_handles_windows_drive_and_unc_file_uris():
    drive_path = _local_path("file:///C:/medical/a%20b.md", None)
    unc_path = _local_path("file://server/share/medical/a.md", None)

    assert drive_path == Path("C:/medical/a b.md").resolve()
    assert str(unc_path).lower().replace("/", "\\").startswith("\\\\server\\share\\medical")


def test_freeze_batch_suite_writes_new_yaml_and_preserves_existing_hashes(tmp_path):
    source_path = _write_local_suite(tmp_path)
    original_text = source_path.read_text(encoding="utf-8")
    original_b12_hash = next(
        batch.content_sha256 for batch in load_batch_suite(source_path).batches if batch.batch_id == "B12"
    )
    output_path = tmp_path / "frozen" / "batch_suite.frozen.yaml"

    manifest = freeze_batch_suite(source_path, output_path)
    frozen = load_batch_suite(output_path)

    assert source_path.read_text(encoding="utf-8") == original_text
    assert output_path.is_file()
    assert manifest["hash_schema"] == BATCH_CONTENT_HASH_SCHEMA
    assert len(manifest["suite_sha256"]) == 64
    assert manifest["preserved_batch_ids"] == ["B12"]
    assert len(manifest["newly_frozen_batch_ids"]) == 11
    assert all(batch.content_sha256 is not None for batch in frozen.batches)
    assert next(batch for batch in frozen.batches if batch.batch_id == "B12").content_sha256 == original_b12_hash
    assert (
        next(batch for batch in frozen.batches if batch.batch_id == "B01").content_sha256
        != next(batch for batch in frozen.batches if batch.batch_id == "B02").content_sha256
    )
    # The frozen YAML lives in a new directory, so its relative input URIs must
    # be rebased rather than accidentally pointing under that output directory.
    path_report = validate_feasibility_design(frozen, suite_path=output_path, check_paths=True)
    assert path_report.valid


def test_freeze_batch_suite_refuses_in_place_or_existing_output(tmp_path):
    source_path = _write_local_suite(tmp_path)
    with pytest.raises(ValueError, match="in-place overwrite is forbidden"):
        freeze_batch_suite(source_path, source_path)

    output_path = tmp_path / "already-exists.yaml"
    output_path.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_batch_suite(source_path, output_path)
    assert output_path.read_text(encoding="utf-8") == "do not replace"


def test_freeze_batch_suite_rejects_stale_existing_local_hash(tmp_path):
    source_path = _write_local_suite(tmp_path)
    suite = load_batch_suite(source_path)
    suite.batches[0].content_sha256 = "f" * 64
    source_path.write_text(
        yaml.safe_dump(suite.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="B01 content_sha256 does not match"):
        freeze_batch_suite(source_path, tmp_path / "frozen.yaml")
    assert not (tmp_path / "frozen.yaml").exists()


def test_freeze_batches_cli_writes_the_requested_new_file(tmp_path, monkeypatch, capsys):
    source_path = _write_local_suite(tmp_path)
    output_path = tmp_path / "cli-frozen.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ai-ready-value", "freeze-batches", "--batches", str(source_path), "--output", str(output_path)],
    )

    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["frozen_file"] == str(output_path.resolve())
    assert output_path.is_file()
    assert all(batch.content_sha256 for batch in load_batch_suite(output_path).batches)


def test_frozen_suite_can_plan_all_reference_stages_from_cli(tmp_path, monkeypatch, capsys):
    source_path = _write_local_suite(tmp_path)
    frozen_path = tmp_path / "registration" / "batch_suite.frozen.yaml"
    freeze_batch_suite(source_path, frozen_path)
    output_dir = tmp_path / "reference-plan"
    contract_path = EXAMPLE_DIR / "production_contract.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "plan-reference",
            "--batches",
            str(frozen_path),
            "--contract",
            str(contract_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["planned_stages"] == 108
    assert (output_dir / "reference_plan.json").is_file()
    assert (output_dir / "contract" / "contract_manifest.json").is_file()


def test_check_paths_detects_content_changed_after_freezing(tmp_path):
    source_path = _write_local_suite(tmp_path)
    # This case checks every batch, so remove the deliberately pre-frozen B12 value.
    source_suite = load_batch_suite(source_path)
    source_suite.batches[-1].content_sha256 = None
    source_path.write_text(
        yaml.safe_dump(source_suite.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    frozen_path = tmp_path / "frozen.yaml"
    freeze_batch_suite(source_path, frozen_path)

    changed_file = tmp_path / "inputs" / "B01" / "source.md"
    changed_file.write_bytes(b"content changed after registration")
    report = validate_feasibility_design(
        load_batch_suite(frozen_path),
        suite_path=frozen_path,
        check_paths=True,
    )

    assert not report.valid
    assert any("B01 content_sha256 does not match" in error for error in report.errors)


def test_capability_blueprint_cannot_include_locked_test_items(tmp_path):
    source_path = _write_local_suite(tmp_path)
    blueprint_path = tmp_path / "capability_blueprint.yaml"
    blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    blueprint["test_items_included"] = True
    blueprint_path.write_text(yaml.safe_dump(blueprint, allow_unicode=True), encoding="utf-8")
    suite = load_batch_suite(source_path)
    suite.capability_blueprint_sha256 = hashlib.sha256(blueprint_path.read_bytes()).hexdigest()

    report = validate_feasibility_design(suite, suite_path=source_path)
    assert not report.valid
    assert any("test_items_included: false" in error for error in report.errors)
