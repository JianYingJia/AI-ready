import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.utils.io import read_jsonl
from ai_ready.valuation.ablation_data import (
    AblationBatchInput,
    AblationCellExecution,
    AblationExecutionSpec,
    CapabilityWeightProfile,
    ModelProfile,
    RuleInterventionSpec,
    apply_nested_label_noise,
    apply_rule_intervention,
    build_nested_selections,
    materialize_ablation_batch,
    nested_label_noise_plan,
    organize_paired_samples,
    quality_fidelity_identifiability,
    weight_capability_obligations,
)
from ai_ready.valuation.quality import SFTSample
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unit(index: int, *, quality: str = "clean") -> ProxyUnit:
    capability_index = index % 2
    return ProxyUnit(
        unit_id=f"u{index:02d}",
        batch_id="B01",
        batch_content_sha256=_sha(b"batch"),
        group_id=f"g{index:02d}",
        source_ids=[f"source-{index % 3}"],
        estimated_tokens=10,
        embedding=[float(capability_index == 0), float(capability_index == 1)],
        quality_bucket=quality,
        benchmark_relevance=0.5,
    )


def _obligation(index: int) -> CoverageObligation:
    return CoverageObligation(
        obligation_id=f"o{index:02d}",
        kind="condition" if index % 3 == 0 else "rule",
        unit_ids=[f"u{index:02d}"],
        capability_ids=["cap_a" if index % 2 == 0 else "cap_b"],
        weight=1.0,
    )


def _sample(index: int, task_type: str) -> SFTSample:
    unit_id = f"u{index:02d}"
    instructions = {
        "grounded_qa": f"What fact is supported for {unit_id}?",
        "conditional_judgment": f"If the condition holds, what fact is supported for {unit_id}?",
        "clinical_vignette": f"A patient matches {unit_id}; what fact is supported?",
    }
    return SFTSample(
        sample_id=f"s-{unit_id}-{task_type}",
        batch_id="B01",
        semantic_unit_ids=[unit_id],
        instruction=instructions[task_type],
        response=f"fact-{unit_id}",
        evidence_spans=[f"fact-{unit_id}"],
        source_ids=[f"source-{index % 3}"],
        task_type=task_type,
        generator="paired_test_bank",
        generator_revision="v1",
    )


def _bank(count: int) -> list[SFTSample]:
    return [
        _sample(index, task_type)
        for index in range(count)
        for task_type in ("grounded_qa", "conditional_judgment", "clinical_vignette")
    ]


def _selector(method: str = "coverage_quality", *, seed: int = 7) -> ProxySelectionConfig:
    return ProxySelectionConfig(
        method=method,
        token_budget=999,
        random_state=seed,
        facility_weight=0.0,
        obligation_weight=1.0,
        quality_fidelity_weight=0.0,
        source_coverage_weight=0.0,
    )


def _batch(count: int = 8, *, mixed_quality: bool = True) -> AblationBatchInput:
    units = [
        _unit(index, quality="usable_noisy" if mixed_quality and index % 4 == 0 else "clean") for index in range(count)
    ]
    return AblationBatchInput(
        batch_id="B01",
        lineage_group="lineage-1",
        batch_content_sha256=_sha(b"batch"),
        proxy_units=units,
        obligations=[_obligation(index) for index in range(count)],
        unit_capabilities={f"u{index:02d}": ["cap_a" if index % 2 == 0 else "cap_b"] for index in range(count)},
        candidate_sft_bank=_bank(count),
    )


def _model_profile(tmp_path: Path, *, profile_id: str = "model-low") -> ModelProfile:
    checkpoint = tmp_path / f"{profile_id}.checkpoint"
    calibration = tmp_path / f"{profile_id}.calibration.json"
    checkpoint.write_bytes(f"checkpoint:{profile_id}".encode())
    calibration.write_bytes(f"calibration:{profile_id}".encode())
    return ModelProfile(
        profile_id=profile_id,
        model_id="example/model",
        revision="revision-1",
        checkpoint_uri=str(checkpoint),
        checkpoint_sha256=_sha(checkpoint.read_bytes()),
        calibration_uri=str(calibration),
        calibration_sha256=_sha(calibration.read_bytes()),
        capability_scores={"cap_a": 0.1, "cap_b": 0.9},
    )


def test_capability_and_prior_weights_change_obligations_without_mutating_inputs():
    obligations = [_obligation(0), _obligation(1)]
    weighted = weight_capability_obligations(
        obligations,
        unit_capabilities={"u00": ["cap_a"], "u01": ["cap_b"]},
        capability_weights={"cap_a": 2.0, "cap_b": 1.0},
        capability_gaps={"cap_a": 0.8, "cap_b": 0.2},
    )

    assert [row.weight for row in weighted] == pytest.approx([1.6, 0.2])
    assert [row.weight for row in obligations] == [1.0, 1.0]


def test_nested_selection_is_deterministic_and_25_is_contained_in_50_and_100():
    batch = _batch(8)
    first = build_nested_selections(batch.proxy_units, batch.obligations, _selector())
    second = build_nested_selections(list(reversed(batch.proxy_units)), batch.obligations, _selector())

    groups_25 = set(first["0.25"]["selected_group_ids"])
    groups_50 = set(first["0.5"]["selected_group_ids"])
    groups_100 = set(first["1"]["selected_group_ids"])
    assert groups_25 < groups_50 < groups_100
    assert len(groups_100) == 8
    assert first["0.25"]["strict_subset_observed"] is True
    assert first["1"]["strict_subset_observed"] is False
    assert first["0.25"]["selected_group_ids"] == second["0.25"]["selected_group_ids"]
    assert first["0.25"]["candidate_pool_sha256"] == second["0.25"]["candidate_pool_sha256"]


def test_organization_arms_keep_identical_content_and_random_is_reproducible():
    bank = _bank(6)
    unit_ids = [f"u{index:02d}" for index in range(6)]

    direct, direct_manifest = organize_paired_samples(bank, unit_ids, organization="direct", seed=19, batch_id="B01")
    mixed, mixed_manifest = organize_paired_samples(bank, unit_ids, organization="mixed", seed=19, batch_id="B01")
    random_first, random_manifest = organize_paired_samples(
        bank, unit_ids, organization="random", seed=19, batch_id="B01"
    )
    random_second, _ = organize_paired_samples(
        list(reversed(bank)), unit_ids, organization="random", seed=19, batch_id="B01"
    )

    assert [row.semantic_unit_ids for row in direct] == [row.semantic_unit_ids for row in mixed]
    assert [row.response for row in direct] == [row.response for row in mixed]
    assert {row.task_type for row in direct} == {"grounded_qa"}
    assert {row.task_type for row in mixed} == {
        "grounded_qa",
        "conditional_judgment",
        "clinical_vignette",
    }
    assert [row.sample_id for row in random_first] == [row.sample_id for row in random_second]
    assert direct_manifest["content_unit_ids"] == mixed_manifest["content_unit_ids"]
    assert random_manifest["content_unit_count"] == 6


def test_organization_rejects_forms_that_change_the_answer_fact():
    bank = _bank(2)
    changed = bank[1].model_copy(update={"response": "a different answer"})
    bank[1] = changed

    with pytest.raises(ValueError, match="same answer fact"):
        organize_paired_samples(
            bank,
            ["u00", "u01"],
            organization="mixed",
            seed=1,
            batch_id="B01",
        )


def test_include_knockout_rescue_swap_only_the_registered_group_pair():
    intervention = RuleInterventionSpec(
        intervention_id="rule-k",
        target_group_ids=["g-target"],
        replacement_group_ids=["g-replacement"],
    )
    order = ["g-target", "g-replacement", "g-other"]

    include, _ = apply_rule_intervention(
        ["g-target", "g-other"],
        selection_order=order,
        intervention=intervention,
        mode="include",
    )
    knockout, knockout_manifest = apply_rule_intervention(
        ["g-target", "g-other"],
        selection_order=order,
        intervention=intervention,
        mode="knockout",
    )
    rescue, _ = apply_rule_intervention(
        knockout,
        selection_order=order,
        intervention=intervention,
        mode="rescue",
    )

    assert include == rescue == ["g-target", "g-other"]
    assert knockout == ["g-replacement", "g-other"]
    assert knockout_manifest["removed_group_ids"] == ["g-target"]
    assert knockout_manifest["added_group_ids"] == ["g-replacement"]


def test_label_noise_sets_are_nested_and_corruption_is_deterministic():
    samples = [_sample(index, "grounded_qa") for index in range(10)]
    plan = nested_label_noise_plan(samples, seed=31)
    noisy_10, manifest_10 = apply_nested_label_noise(samples, rate=0.1, seed=31)
    noisy_30, manifest_30 = apply_nested_label_noise(samples, rate=0.3, seed=31)
    noisy_30_repeat, manifest_30_repeat = apply_nested_label_noise(list(samples), rate=0.3, seed=31)

    assert set(plan["rates"]["0.1"]) < set(plan["rates"]["0.3"])
    assert set(manifest_10["corrupted_parent_sample_ids"]) < set(manifest_30["corrupted_parent_sample_ids"])
    assert [row.model_dump() for row in noisy_30] == [row.model_dump() for row in noisy_30_repeat]
    assert manifest_30 == manifest_30_repeat
    assert sum(first.response != second.response for first, second in zip(samples, noisy_10)) == 1


def test_quality_fidelity_with_one_bucket_is_not_identifiable():
    audit = quality_fidelity_identifiability([_unit(index) for index in range(8)])

    assert audit["status"] == "not_identifiable"
    assert audit["quality_buckets"] == ["clean"]


def test_prior_aware_execution_requires_a_calibrated_checkpoint_profile(tmp_path: Path):
    profile = _model_profile(tmp_path)
    cell = AblationCellExecution(
        study_id="A8",
        cell_id="aware",
        kind="base_model_prior",
        capability_profile_id="task",
        selector_config=_selector(),
        source_budget_fraction=0.25,
        organization="direct",
        prior_aware=True,
        model_profile_id=profile.profile_id,
    )
    spec = AblationExecutionSpec(
        execution_id="prior-test",
        protocol_sha256=_sha(b"protocol"),
        suite_sha256=_sha(b"suite"),
        contract_sha256=_sha(b"contract"),
        training_seeds=[17],
        capability_profiles=[CapabilityWeightProfile(profile_id="task", weights={"cap_a": 1.0, "cap_b": 1.0})],
        model_profiles=[profile],
        default_model_profile_id=profile.profile_id,
        cells=[cell],
    )

    assert spec.cells[0].prior_aware is True
    with pytest.raises(ValidationError, match="64-character SHA-256"):
        ModelProfile(
            profile_id="bad",
            model_id="example/model",
            revision="revision-1",
            checkpoint_uri="hf://example/model",
            checkpoint_sha256="not-a-checkpoint-hash",
            calibration_uri="artifact://calibration",
            calibration_sha256=_sha(b"calibration"),
            capability_scores={"cap_a": 0.5},
        )


def test_materialize_writes_hash_bound_cells_and_batch_cell_seed_jobs(tmp_path: Path):
    model = _model_profile(tmp_path)
    capability = CapabilityWeightProfile(profile_id="task", weights={"cap_a": 2.0, "cap_b": 1.0})
    cells = [
        AblationCellExecution(
            study_id="A5",
            cell_id="mixed",
            kind="organization",
            capability_profile_id="task",
            selector_config=_selector(),
            source_budget_fraction=0.25,
            organization="mixed",
            repetition_factor=2,
        ),
        AblationCellExecution(
            study_id="A4",
            cell_id="quality-on",
            kind="quality_fidelity",
            capability_profile_id="task",
            selector_config=_selector(),
            source_budget_fraction=0.25,
            organization="direct",
        ),
        AblationCellExecution(
            study_id="A8",
            cell_id="prior-aware",
            kind="base_model_prior",
            capability_profile_id="task",
            selector_config=_selector(),
            source_budget_fraction=0.25,
            organization="direct",
            prior_aware=True,
        ),
    ]
    spec = AblationExecutionSpec(
        execution_id="materialize-test",
        protocol_sha256=_sha(b"protocol"),
        suite_sha256=_sha(b"suite"),
        contract_sha256=_sha(b"contract"),
        training_seeds=[17, 29],
        capability_profiles=[capability],
        model_profiles=[model],
        default_model_profile_id=model.profile_id,
        cells=cells,
    )
    output = tmp_path / "ablation-output"
    result = materialize_ablation_batch(spec, _batch(8, mixed_quality=False), output)

    root = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
    jobs = read_jsonl(output / "jobs.jsonl")
    mixed_dir = output / "cells" / "A5" / "mixed" / "B01"
    mixed_manifest = json.loads((mixed_dir / "cell_manifest.json").read_text(encoding="utf-8"))
    mixed_train = read_jsonl(mixed_dir / "train.jsonl")
    mixed_lineage = read_jsonl(mixed_dir / "lineage.jsonl")
    quality_manifest = json.loads(
        (output / "cells" / "A4" / "quality-on" / "B01" / "cell_manifest.json").read_text(encoding="utf-8")
    )
    prior_manifest = json.loads(
        (output / "cells" / "A8" / "prior-aware" / "B01" / "cell_manifest.json").read_text(encoding="utf-8")
    )

    assert result["cell_count"] == root["cell_count"] == 3
    assert root["ready_cell_count"] == 2
    assert root["not_identifiable_cell_count"] == 1
    assert len(jobs) == 2 * 2
    assert {(row["cell_id"], row["seed"]) for row in jobs} == {
        ("mixed", 17),
        ("mixed", 29),
        ("prior-aware", 17),
        ("prior-aware", 29),
    }
    assert all(row["checkpoint_sha256"] == model.checkpoint_sha256 for row in jobs)
    assert all(set(row) == {"instruction", "input", "output"} for row in mixed_train)
    assert len(mixed_train) == len(mixed_lineage) == 4
    assert sum(row["controlled_duplicate"] for row in mixed_lineage) == 2
    assert mixed_manifest["outputs"]["train_sha256"] == _sha((mixed_dir / "train.jsonl").read_bytes())
    assert quality_manifest["status"] == "not_identifiable"
    assert prior_manifest["capability_profile"]["prior_aware"] is True
    assert prior_manifest["model_profile"]["checkpoint_sha256"] == model.checkpoint_sha256
    assert prior_manifest["weighted_obligations_sha256"] != mixed_manifest["weighted_obligations_sha256"]


def test_materialize_verifies_local_checkpoint_bytes(tmp_path: Path):
    model = _model_profile(tmp_path)
    spec = AblationExecutionSpec(
        execution_id="checkpoint-test",
        protocol_sha256=_sha(b"protocol"),
        suite_sha256=_sha(b"suite"),
        contract_sha256=_sha(b"contract"),
        training_seeds=[17],
        capability_profiles=[CapabilityWeightProfile(profile_id="task", weights={"cap_a": 1.0, "cap_b": 1.0})],
        model_profiles=[model],
        default_model_profile_id=model.profile_id,
        cells=[
            AblationCellExecution(
                study_id="A3",
                cell_id="coverage",
                kind="selection",
                capability_profile_id="task",
                selector_config=_selector(),
                source_budget_fraction=0.25,
                organization="direct",
            )
        ],
    )
    Path(model.checkpoint_uri).write_bytes(b"changed-after-freeze")

    with pytest.raises(ValueError, match="checkpoint.*hash mismatch"):
        materialize_ablation_batch(spec, _batch(8), tmp_path / "bad-output")
    assert not (tmp_path / "bad-output.partial").exists()
