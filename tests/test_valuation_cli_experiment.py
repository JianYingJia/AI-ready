import hashlib
import json
import sys
from pathlib import Path

import yaml

from ai_ready.utils.io import write_jsonl
from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.experiment import ExperimentManifest
from ai_ready.valuation.selection import ProxySelectionConfig, selection_config_digest

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs" / "valuation_cll_public_feasibility"
BATCHES = CONFIG_DIR / "batch_suite.frozen.yaml"
CONTRACT = CONFIG_DIR / "production_contract.yaml"


def _write_protocol(path: Path) -> Path:
    target_config = ProxySelectionConfig(method="coverage_quality", token_budget=100_000)
    random_config = ProxySelectionConfig(method="random", token_budget=100_000, random_state=101)
    for name, config in (("coverage-quality.yaml", target_config), ("random.yaml", random_config)):
        (path.parent / name).write_text(
            yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    payload = {
        "schema_version": "pilot-feasibility-protocol-v2",
        "protocol_id": "cli-feasibility-test",
        "inference_scope": "descriptive_pilot_no_confirmatory_inference",
        "expected_benchmark_item_count": 46,
        "target": {
            "arm": "proxy",
            "label": "coverage_quality",
            "method": "coverage_quality",
            "budget_id": "small",
            "max_effective_train_tokens": 100_000,
            "selection_seed": None,
            "selection_config_uri": "coverage-quality.yaml",
            "selection_config_sha256": selection_config_digest(target_config),
        },
        "baselines": [
            {
                "arm": "proxy",
                "label": "uniform_random",
                "method": "random",
                "budget_id": "small",
                "max_effective_train_tokens": 100_000,
                "selection_seed": 101,
                "selection_config_uri": "random.yaml",
                "selection_config_sha256": selection_config_digest(random_config),
            }
        ],
        "reference": {
            "arm": "reference",
            "label": "operational_full",
            "method": "fully_open_meditron",
            "budget_id": "full",
            "max_effective_train_tokens": 1_000_000,
            "selection_seed": None,
            "selection_config_uri": None,
            "selection_config_sha256": None,
        },
        "reference_identifiability": {
            "min_utility_range": 0.05,
            "min_between_batch_sd": 0.02,
            "max_within_to_between_sd_ratio": 0.2,
            "min_pairwise_seed_rank_spearman_median": 0.9,
            "positive_margin": 0.0,
            "min_clearly_positive_batch_count": 2,
            "ci_separation_margin": 0.0,
            "min_distinguishable_ci_pair_fraction": 0.1,
        },
        "target_decision": {
            "min_spearman": 0.8,
            "min_kendall_tau_b": 0.7,
            "min_pairwise_order_accuracy": 0.8,
            "max_top1_regret": 0.01,
            "min_spearman_advantage_over_each_baseline": 0.1,
            "max_top1_regret_excess_over_each_baseline": 0.0,
            "require_leave_one_seed_out_stability": True,
        },
        "indifference_margin": 0.0,
        "minimum_independent_lineage_count_for_warning": 8,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _plan(tmp_path: Path, monkeypatch, capsys) -> tuple[Path, ExperimentManifest]:
    protocol = _write_protocol(tmp_path / "protocol.yaml")
    output_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "plan-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(protocol),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert valuation_main() == 0
    response = json.loads(capsys.readouterr().out)
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = ExperimentManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert response["experiment_id"] == manifest.experiment_id
    return protocol, manifest


def _complete_trials(manifest: ExperimentManifest) -> list[dict]:
    ordered_batches = list(manifest.batch_content_sha256)
    improvement_by_batch = {batch_id: len(ordered_batches) - index for index, batch_id in enumerate(ordered_batches)}
    reversed_improvement = {
        batch_id: improvement_by_batch[ordered_batches[-index - 1]] for index, batch_id in enumerate(ordered_batches)
    }

    def digest(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    rows = []
    for expected in manifest.trials:
        if expected.role == "baseline":
            improvement = reversed_improvement[expected.batch_id]
        else:
            improvement = improvement_by_batch[expected.batch_id]
        baseline_correct = {17: 10, 29: 11, 43: 12}[expected.seed]
        correct = baseline_correct + improvement
        rows.append(
            {
                "run_id": expected.run_id,
                "suite_sha256": expected.suite_sha256,
                "batch_id": expected.batch_id,
                "batch_content_sha256": expected.batch_content_sha256,
                "arm": expected.arm,
                "method": expected.method,
                "budget_id": expected.budget_id,
                "seed": expected.seed,
                "benchmark_score": correct / 46,
                "baseline_score": baseline_correct / 46,
                "benchmark_sha256": expected.benchmark_sha256,
                "contract_sha256": expected.contract_sha256,
                "effective_train_tokens": expected.max_effective_train_tokens,
                "evaluation_metric": "accuracy",
                "generator": expected.generator,
                "generator_revision": expected.generator_revision,
                "generation_recipe_sha256": expected.generation_recipe_sha256,
                "generation_manifest_sha256": digest(f"generation:{expected.batch_id}:{expected.arm_id}"),
                "selection_seed": expected.selection_seed,
                "selection_config_sha256": expected.selection_config_sha256,
                "selection_manifest_sha256": (
                    digest(f"selection:{expected.batch_id}:{expected.arm_id}") if expected.arm == "proxy" else None
                ),
                "full_traversal_report_sha256": (
                    digest(f"traversal:{expected.batch_id}") if expected.arm == "reference" else None
                ),
                "accepted_sft_sha256": digest(f"sft:{expected.batch_id}:{expected.arm_id}"),
                "checkpoint_manifest_sha256": digest(f"checkpoint:{expected.run_id}"),
                "predictions_sha256": digest(f"predictions:{expected.run_id}"),
                "evaluated_item_count": 46,
                "correct_item_count": correct,
                "baseline_correct_item_count": baseline_correct,
            }
        )
    return rows


def test_plan_experiment_writes_strict_complete_manifest(tmp_path, monkeypatch, capsys):
    _, manifest = _plan(tmp_path, monkeypatch, capsys)

    assert len(manifest.batch_content_sha256) == 12
    assert len(manifest.arms) == 3
    assert len(manifest.seeds) == 3
    assert manifest.expected_trial_count == 108
    assert len({row.run_id for row in manifest.trials}) == 108


def test_plan_experiment_rejects_changed_selector_config(tmp_path, monkeypatch, capsys):
    protocol = _write_protocol(tmp_path / "protocol.yaml")
    selector_path = tmp_path / "coverage-quality.yaml"
    selector = yaml.safe_load(selector_path.read_text(encoding="utf-8"))
    selector["obligation_weight"] = 9.0
    selector_path.write_text(yaml.safe_dump(selector, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "plan-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(protocol),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 1
    assert "Selection config hash does not match arm coverage_quality" in capsys.readouterr().err
    assert not output_dir.exists()


def test_analyze_experiment_audits_and_writes_outputs(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl(trials_path, _complete_trials(manifest))
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "analyze-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(protocol),
            "--manifest",
            str(tmp_path / "plan" / "experiment_manifest.json"),
            "--trials",
            str(trials_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 0
    stdout_report = json.loads(capsys.readouterr().out)
    saved_report = json.loads((output_dir / "feasibility_report.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "manifest_audit.json").read_text(encoding="utf-8"))
    audited_trials = [
        json.loads(line) for line in (output_dir / "audited_trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    utility_rows = [
        json.loads(line) for line in (output_dir / "batch_utilities.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert stdout_report == saved_report
    assert saved_report["verdict"] == "feasible"
    assert audit["complete"] is True
    assert audit["matched_count"] == manifest.expected_trial_count
    assert len(audited_trials) == manifest.expected_trial_count
    assert all(row["predictions_sha256"] for row in audited_trials)
    assert len(utility_rows) == len(manifest.batch_content_sha256) * len(manifest.arms)


def test_analyze_experiment_rejects_incomplete_trials_before_writing(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    trials_path = tmp_path / "incomplete.jsonl"
    write_jsonl(trials_path, _complete_trials(manifest)[:-1])
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "analyze-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(protocol),
            "--manifest",
            str(tmp_path / "plan" / "experiment_manifest.json"),
            "--trials",
            str(trials_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 1
    captured = capsys.readouterr()
    assert "Experiment trial audit is incomplete" in captured.err
    assert not output_dir.exists()


def test_analyze_experiment_rejects_protocol_changed_after_planning(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    changed_payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    changed_payload["target_decision"]["min_spearman"] = 0.9
    changed_protocol = tmp_path / "changed-protocol.yaml"
    changed_protocol.write_text(
        yaml.safe_dump(changed_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    trials_path = tmp_path / "trials.jsonl"
    write_jsonl(trials_path, _complete_trials(manifest))
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "analyze-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(changed_protocol),
            "--manifest",
            str(tmp_path / "plan" / "experiment_manifest.json"),
            "--trials",
            str(trials_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert valuation_main() == 1
    captured = capsys.readouterr()
    assert "does not match the supplied batch suite, contract, and protocol" in captured.err
    assert not output_dir.exists()


def _run_analysis_with_rows(tmp_path, monkeypatch, capsys, protocol, rows):
    trials_path = tmp_path / "tampered-trials.jsonl"
    write_jsonl(trials_path, rows)
    output_dir = tmp_path / "tampered-analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "analyze-experiment",
            "--batches",
            str(BATCHES),
            "--contract",
            str(CONTRACT),
            "--protocol",
            str(protocol),
            "--manifest",
            str(tmp_path / "plan" / "experiment_manifest.json"),
            "--trials",
            str(trials_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    exit_code = valuation_main()
    captured = capsys.readouterr()
    return exit_code, captured, output_dir


def test_analyze_experiment_rejects_missing_artifact_lineage(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    rows = _complete_trials(manifest)
    del rows[0]["predictions_sha256"]

    exit_code, captured, output_dir = _run_analysis_with_rows(tmp_path, monkeypatch, capsys, protocol, rows)

    assert exit_code == 1
    assert "predictions_sha256" in captured.err
    assert not output_dir.exists()


def test_analyze_experiment_rejects_partial_benchmark_item_count(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    rows = _complete_trials(manifest)
    row = rows[0]
    row["evaluated_item_count"] = 45
    row["benchmark_score"] = row["correct_item_count"] / 45
    row["baseline_score"] = row["baseline_correct_item_count"] / 45

    exit_code, captured, output_dir = _run_analysis_with_rows(tmp_path, monkeypatch, capsys, protocol, rows)

    assert exit_code == 1
    assert "evaluated_item_count" in captured.err
    assert "Experiment trial audit is incomplete" in captured.err
    assert not output_dir.exists()


def test_analyze_experiment_rejects_score_count_mismatch(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    rows = _complete_trials(manifest)
    rows[0]["benchmark_score"] += 0.001

    exit_code, captured, output_dir = _run_analysis_with_rows(tmp_path, monkeypatch, capsys, protocol, rows)

    assert exit_code == 1
    assert "benchmark_score must equal" in captured.err
    assert not output_dir.exists()


def test_analyze_experiment_rejects_training_data_drift_across_seeds(tmp_path, monkeypatch, capsys):
    protocol, manifest = _plan(tmp_path, monkeypatch, capsys)
    rows = _complete_trials(manifest)
    target = rows[0]
    drifted = next(
        row
        for row in rows
        if row["batch_id"] == target["batch_id"] and row["method"] == target["method"] and row["seed"] != target["seed"]
    )
    drifted["accepted_sft_sha256"] = "9" * 64

    exit_code, captured, output_dir = _run_analysis_with_rows(tmp_path, monkeypatch, capsys, protocol, rows)

    assert exit_code == 1
    assert "accepted_sft_sha256" in captured.err
    assert not output_dir.exists()
