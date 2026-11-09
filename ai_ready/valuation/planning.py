"""Build a deterministic, reviewable plan for every high-budget reference run."""

import string
from pathlib import Path

from ai_ready.core.schema import stable_id
from ai_ready.utils.io import write_json, write_jsonl
from ai_ready.valuation.contracts import contract_digest, suite_digest
from ai_ready.valuation.schema import BatchSuite, PlannedStage, ProductionContract, ReferencePlan

EXPECTED_OUTPUTS = {
    "preprocess": ["cleaned_docs.jsonl", "preprocess_manifest.json"],
    "generate": [
        "eligible_units.jsonl",
        "normalized_fom.jsonl",
        "fully_open_meditron_run_manifest.json",
    ],
    "quality_gate": [
        "accepted_sft.jsonl",
        "eligible_units.jsonl",
        "dispositions.jsonl",
        "full_traversal_report.json",
        "generation_manifest.json",
        "fully_open_meditron_run_manifest.json",
    ],
    "train": ["checkpoint_manifest.json", "training_metrics.json"],
    "evaluate": ["predictions.jsonl", "metrics.json"],
}

ALLOWED_TEMPLATE_FIELDS = {
    "batch_id",
    "input_uri",
    "run_dir",
    "batch_dir",
    "seed",
    "suite_sha256",
    "batch_content_sha256",
    "contract_sha256",
}


def _render_argv(values: list[str], context: dict[str, str | int]) -> list[str]:
    rendered = []
    formatter = string.Formatter()
    for value in values:
        fields = {field for _, field, _, _ in formatter.parse(value) if field}
        if unknown := fields - ALLOWED_TEMPLATE_FIELDS:
            raise ValueError(f"Unknown command template fields: {sorted(unknown)}")
        if "seed" in fields and context.get("seed") == "":
            raise ValueError("A batch-scoped command cannot use the seed placeholder")
        rendered.append(value.format_map(context))
    return rendered


def build_reference_plan(
    suite: BatchSuite,
    contract: ProductionContract,
    output_dir: str | Path,
) -> ReferencePlan:
    output_dir = Path(output_dir).resolve()
    contract_sha = contract_digest(contract)
    suite_sha = suite_digest(suite)
    stages: list[PlannedStage] = []
    for batch in suite.batches:
        if batch.content_sha256 is None:
            raise ValueError(f"Batch {batch.batch_id} has no content_sha256; run freeze-batches before planning")
        batch_dir = output_dir / batch.batch_id
        for tool in contract.tools:
            seeds: list[int | None] = contract.training.seeds if tool.scope == "trial" else [None]
            for seed in seeds:
                run_dir = batch_dir / (f"seed-{seed}" if seed is not None else "shared") / tool.stage
                # Bind every run to both registration context and exact batch content. Otherwise an old
                # checkpoint could be mistaken for a run over newly constructed data under the same batch ID.
                run_id = stable_id(
                    "reference-stage-v2",
                    suite_sha,
                    batch.content_sha256,
                    contract_sha,
                    batch.batch_id,
                    tool.stage,
                    str(seed),
                )
                context: dict[str, str | int] = {
                    "batch_id": batch.batch_id,
                    "input_uri": batch.input_uri,
                    "run_dir": str(run_dir),
                    "batch_dir": str(batch_dir),
                    "seed": "" if seed is None else seed,
                    "suite_sha256": suite_sha,
                    "batch_content_sha256": batch.content_sha256,
                    "contract_sha256": contract_sha,
                }
                stages.append(
                    PlannedStage(
                        stage=tool.stage,
                        adapter=tool.adapter,
                        scope=tool.scope,
                        batch_id=batch.batch_id,
                        suite_sha256=suite_sha,
                        batch_content_sha256=batch.content_sha256,
                        contract_sha256=contract_sha,
                        seed=seed,
                        run_id=run_id,
                        run_dir=str(run_dir),
                        # Commands are rendered as argv arrays and are never passed through a shell.
                        argv=_render_argv(tool.argv, context),
                        expected_outputs=EXPECTED_OUTPUTS[tool.stage],
                    )
                )
    plan = ReferencePlan(
        suite_id=suite.suite_id,
        suite_sha256=suite_sha,
        contract_id=contract.contract_id,
        contract_sha256=contract_sha,
        batch_ids=[batch.batch_id for batch in suite.batches],
        batch_content_sha256={batch.batch_id: batch.content_sha256 for batch in suite.batches},
        seeds=contract.training.seeds,
        stages=stages,
    )
    _validate_plan_completeness(plan, contract)
    return plan


def _validate_plan_completeness(plan: ReferencePlan, contract: ProductionContract) -> None:
    expected_batch_stages = {tool.stage for tool in contract.tools if tool.scope == "batch"}
    expected_trial_stages = {tool.stage for tool in contract.tools if tool.scope == "trial"}
    for batch_id in plan.batch_ids:
        batch_stages = {stage.stage for stage in plan.stages if stage.batch_id == batch_id and stage.scope == "batch"}
        if batch_stages != expected_batch_stages:
            raise ValueError(f"Incomplete batch-scoped plan for {batch_id}")
        for seed in plan.seeds:
            trial_stages = {
                stage.stage
                for stage in plan.stages
                if stage.batch_id == batch_id and stage.scope == "trial" and stage.seed == seed
            }
            if trial_stages != expected_trial_stages:
                raise ValueError(f"Incomplete trial-scoped plan for {batch_id}, seed={seed}")


def write_reference_plan(plan: ReferencePlan, output_dir: str | Path) -> None:
    output_dir = Path(output_dir).resolve()
    write_json(output_dir / "reference_plan.json", plan.model_dump(mode="json"))
    write_jsonl(output_dir / "reference_stages.jsonl", (stage.model_dump(mode="json") for stage in plan.stages))
