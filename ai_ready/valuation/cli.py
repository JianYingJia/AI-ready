"""Command-line entry points for planning and evaluating data valuation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ai_ready.core.schema import EvidenceUnit
from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.ablation import (
    AblationObservation,
    ablation_matrix,
    analyse_ablations,
    endpoint_catalog,
    load_ablation_protocol,
    protocol_summary,
)
from ai_ready.valuation.ablation_selection import FractionalBudgetPolicy, select_fractional_proxy
from ai_ready.valuation.adapters import adapter_catalog
from ai_ready.valuation.batches import freeze_batch_suite, validate_feasibility_design
from ai_ready.valuation.conditional_value import (
    BenchmarkCellDefinition,
    FidelityThresholds,
    ModelMasteryProfile,
    ProbeAggregationConfig,
    ProfileFidelityCertificate,
    ScoredProbe,
    ValueProfile,
    build_fidelity_certificate,
    build_model_mastery_profile_from_probes,
    build_raw_sample_profile,
    build_value_profile,
    conditional_value_unit_from_sampling_record,
    reconstruct_value_profile,
    verify_fidelity_certificate,
)
from ai_ready.valuation.contracts import (
    file_sha256,
    freeze_contract,
    load_batch_suite,
    load_production_contract,
    validate_contract_against_suite,
)
from ai_ready.valuation.decision_experiments import (
    CostAnalysisConfig,
    CostLedgerEvent,
    CrossTaskBatchValue,
    CrossTaskConfig,
    NaturalBatchAuditConfig,
    ReferenceBatchValue,
    StaticBaselineConfig,
    StaticBatchFeatures,
    TrainableProxyBatchValue,
    analyze_cost_ledger,
    analyze_cross_task_values,
    analyze_static_baselines,
    audit_natural_batches,
)
from ai_ready.valuation.experiment import (
    ExperimentManifest,
    audit_experiment_manifest,
    build_experiment_manifest,
    write_experiment_manifest,
)
from ai_ready.valuation.feasibility import assess_feasibility, load_feasibility_protocol
from ai_ready.valuation.llamafactory import (
    prepare_llamafactory_jobs,
    profile_summary,
    run_llamafactory_job,
    summarize_llamafactory_jobs,
    write_sft_registry_template,
)
from ai_ready.valuation.local_sft import generate_local_sft_suite, generate_proxy_sft_suite
from ai_ready.valuation.metrics import aggregate_trials, evaluate_proxy, validate_trial_completeness
from ai_ready.valuation.mining import (
    ActivityRule,
    build_coverage_obligations,
    fixed_chunk_baseline,
    mine_semantic_units,
)
from ai_ready.valuation.model_probe import (
    ProbeInferenceConfig,
    ProbeRunManifest,
    load_probe_catalogue,
    run_model_probe_pipeline,
)
from ai_ready.valuation.planning import build_reference_plan, write_reference_plan
from ai_ready.valuation.quality import SFTSample, UnitDisposition, validate_full_traversal
from ai_ready.valuation.reference_sft import import_fully_open_meditron_reference
from ai_ready.valuation.schema import CombinationUtility, ExperimentTrialScore, TrialScore
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
    select_proxy,
    selection_config_digest,
)
from ai_ready.valuation.unit_mastery import (
    UnitCellMasteryObservation,
    UnitCellMasteryProfile,
    build_unit_cell_mastery_profile,
    unit_cell_mastery_profile_digest,
    validate_unit_cell_mastery_profile,
)
from ai_ready.valuation.unit_prior_experiment import (
    PriorManipulationConfig,
    ProbeHoldoutConfig,
    ProbeUnitBinding,
    analyse_prior_holdout_fidelity,
    analyse_prior_manipulation,
    bind_probes_to_semantic_units,
    build_sparse_unit_profile,
    split_bindings_for_probe_holdout,
)
from ai_ready.valuation.value_artifacts import adapt_existing_value_population, materialize_selected_value_artifacts
from ai_ready.valuation.value_sampling import (
    ValueSamplingConfig,
    ValueSamplingUnit,
    sample_value_distribution,
    verify_value_sample_manifest,
)
from ai_ready.valuation.value_sweep import ValueFidelitySweepConfig, run_value_fidelity_sweep


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _prepare_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _combined_validation(args, *, check_paths: bool | None = None) -> tuple[dict, bool]:
    suite = load_batch_suite(args.batches)
    contract = load_production_contract(args.contract)
    should_check_paths = args.check_paths if check_paths is None else check_paths
    design = validate_feasibility_design(suite, suite_path=args.batches, check_paths=should_check_paths)
    protocol = validate_contract_against_suite(suite, contract)
    report = {
        "valid": design.valid and protocol.valid,
        "design": design.model_dump(mode="json"),
        "production_contract": protocol.model_dump(mode="json"),
    }
    return report, report["valid"]


def _cmd_validate(args) -> int:
    report, valid = _combined_validation(args)
    _print(report)
    return 0 if valid else 2


def _cmd_freeze_batches(args) -> int:
    manifest = freeze_batch_suite(args.batches, args.output)
    _print(manifest)
    return 0


def _cmd_plan(args) -> int:
    # A reference plan is a formal execution artifact, so local content hashes
    # are always rechecked even when the caller omitted --check-paths.
    report, valid = _combined_validation(args, check_paths=True)
    if not valid:
        _print(report)
        return 2
    suite = load_batch_suite(args.batches)
    contract = load_production_contract(args.contract)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output path must be an empty directory: {output_dir}")
    # Build first. A missing batch hash or invalid command template must not
    # leave a half-frozen contract in an output directory that looks runnable.
    plan = build_reference_plan(suite, contract, output_dir)
    _prepare_output_dir(output_dir)
    freeze_contract(contract, output_dir / "contract")
    write_reference_plan(plan, output_dir)
    write_json(output_dir / "validation_report.json", report)
    _print(
        {
            "output_dir": str(output_dir),
            "contract_sha256": plan.contract_sha256,
            "batches": len(plan.batch_ids),
            "seeds": len(plan.seeds),
            "planned_stages": len(plan.stages),
        }
    )
    return 0


def _cmd_plan_experiment(args) -> int:
    """Freeze the complete batch × arm × training-seed experiment matrix."""

    # As with the reference planner, an experiment manifest is an execution
    # artifact.  Re-hash local source inputs before registering any run IDs.
    report, valid = _combined_validation(args, check_paths=True)
    if not valid:
        _print(report)
        return 2

    suite = load_batch_suite(args.batches)
    contract = load_production_contract(args.contract)
    protocol = load_feasibility_protocol(args.protocol)
    _validate_protocol_selection_configs(protocol, args.protocol)
    manifest = build_experiment_manifest(suite, contract, protocol)

    output_dir = _prepare_output_dir(args.output_dir)
    write_experiment_manifest(manifest, output_dir / "experiment_manifest.json")
    _print(
        {
            "output_dir": str(output_dir),
            "experiment_id": manifest.experiment_id,
            "protocol_sha256": manifest.protocol_sha256,
            "batches": len(manifest.batch_content_sha256),
            "arms": len(manifest.arms),
            "seeds": len(manifest.seeds),
            "expected_trials": manifest.expected_trial_count,
        }
    )
    return 0


def _cmd_plan_ablations(args) -> int:
    """Validate and freeze the stage-level mechanism ablation matrix."""

    protocol = load_ablation_protocol(args.protocol)
    summary = protocol_summary(protocol)
    output_dir = _prepare_output_dir(args.output_dir)
    write_json(output_dir / "ablation_protocol.json", protocol.model_dump(mode="json"))
    write_json(output_dir / "ablation_matrix.json", ablation_matrix(protocol))
    write_json(output_dir / "ablation_endpoints.json", endpoint_catalog(protocol))
    write_json(output_dir / "ablation_observation_schema.json", AblationObservation.model_json_schema())
    write_json(output_dir / "ablation_summary.json", summary)
    _print({"output_dir": str(output_dir), **summary})
    return 0


def _cmd_analyze_ablations(args) -> int:
    """Apply frozen stage gates to a complete ablation observation table."""

    protocol = load_ablation_protocol(args.protocol)
    observations = [AblationObservation.model_validate(row) for row in read_jsonl(args.observations)]
    report = analyse_ablations(protocol, observations, study_ids=args.study)
    output_dir = _prepare_output_dir(args.output_dir)
    write_jsonl(
        output_dir / "audited_ablation_observations.jsonl",
        (row.model_dump(mode="json") for row in observations),
    )
    write_json(output_dir / "ablation_report.json", report)
    _print(report)
    return 0


def _cmd_scaffold_llamafactory(args) -> int:
    """Write the expected 84 batch × arm data registrations.

    The registry is intentionally separate from the 252 training jobs: the
    three training seeds for one batch/arm must consume byte-identical SFT.
    """

    manifest = _load_experiment_manifest(args.manifest)
    registry_path = args.output.resolve()
    sft_root = args.sft_root.resolve()
    write_sft_registry_template(manifest, registry_path, sft_root)
    _print(
        {
            "experiment_id": manifest.experiment_id,
            "registry": str(registry_path),
            "sft_root": str(sft_root),
            "expected_entries": len(manifest.batch_content_sha256) * len(manifest.arms),
        }
    )
    return 0


def _cmd_inspect_llamafactory_profile(args) -> int:
    _print(profile_summary(args.profile))
    return 0


def _cmd_prepare_llamafactory(args) -> int:
    """Materialize audited LLaMAFactory data views and frozen jobs."""

    manifest_path = args.manifest.resolve()
    manifest = _load_experiment_manifest(manifest_path)
    contract = load_production_contract(args.contract)
    output_dir = args.output_dir.resolve()
    plan = prepare_llamafactory_jobs(
        experiment_manifest_path=manifest_path,
        experiment_manifest=manifest,
        contract=contract,
        registry_path=args.registry,
        profile_path=args.profile,
        llamafactory_root=args.llamafactory_root,
        output_dir=output_dir,
        discard_incomplete=args.discard_incomplete,
    )
    _print(
        {
            "experiment_id": plan.experiment_id,
            "output_dir": str(output_dir),
            "job_manifest": str(output_dir / "llamafactory_jobs.json"),
            "evaluation_requests": str(output_dir / "evaluation_requests.jsonl"),
            "slurm_script": str(output_dir / "run_slurm_array.sh"),
            "dataset_groups": len({(job.batch_id, job.arm_id) for job in plan.jobs}),
            "expected_jobs": plan.expected_job_count,
            "train_jobs": sum(job.execution_mode == "train" for job in plan.jobs),
            "no_op_jobs": sum(job.execution_mode == "no_op" for job in plan.jobs),
            "slurm_array": f"0-{plan.expected_job_count - 1}",
            "llamafactory_revision": plan.llamafactory_revision,
            "llamafactory_commit": plan.llamafactory_commit,
            "profile_sha256": plan.profile_sha256,
            "template_sha256": plan.template_sha256,
        }
    )
    return 0


def _cmd_run_llamafactory(args) -> int:
    result = run_llamafactory_job(
        args.jobs,
        run_id=args.run_id,
        job_index=args.job_index,
        resume=args.resume,
        reclaim_stale_lock=args.reclaim_stale_lock,
        restart_incomplete=args.restart_incomplete,
    )
    _print(result)
    return 0


def _cmd_llamafactory_status(args) -> int:
    _print(summarize_llamafactory_jobs(args.jobs))
    return 0


def _load_experiment_manifest(path: Path) -> ExperimentManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment manifest must contain a JSON object: {path.resolve()}")
    return ExperimentManifest.model_validate(payload)


def _validate_protocol_selection_configs(protocol, protocol_path: Path) -> None:
    """Resolve and verify every proxy selector config before assigning run IDs."""

    protocol_dir = protocol_path.resolve().parent
    for arm in [protocol.target, *protocol.baselines]:
        config_path = Path(arm.selection_config_uri)
        if not config_path.is_absolute():
            config_path = (protocol_dir / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Selection config for arm {arm.label} does not exist: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Selection config must contain a YAML mapping: {config_path}")
        config = ProxySelectionConfig.model_validate(payload)
        if config.method != arm.method:
            raise ValueError(f"Selection config method does not match arm {arm.label}: {config.method} != {arm.method}")
        if config.token_budget != arm.selection_token_budget:
            raise ValueError(
                f"Selection config source-token budget does not match arm {arm.label}: "
                f"{config.token_budget} != {arm.selection_token_budget}"
            )
        if config.method in {"random", "stratified_random"} and config.random_state != arm.selection_seed:
            raise ValueError(
                f"Selection config random_state does not match arm {arm.label} selection_seed: "
                f"{config.random_state} != {arm.selection_seed}"
            )
        observed_digest = selection_config_digest(config)
        if observed_digest != arm.selection_config_sha256:
            raise ValueError(
                f"Selection config hash does not match arm {arm.label}: "
                f"{observed_digest} != {arm.selection_config_sha256}"
            )


def _cmd_analyze_experiment(args) -> int:
    """Audit a complete frozen experiment before computing its pilot verdict."""

    report, valid = _combined_validation(args, check_paths=False)
    if not valid:
        _print(report)
        return 2

    suite = load_batch_suite(args.batches)
    contract = load_production_contract(args.contract)
    protocol = load_feasibility_protocol(args.protocol)
    _validate_protocol_selection_configs(protocol, args.protocol)
    manifest = _load_experiment_manifest(args.manifest)

    # Rebuilding from the supplied frozen inputs prevents a valid but unrelated
    # manifest from being paired with another protocol, suite, or contract.
    expected_manifest = build_experiment_manifest(suite, contract, protocol)
    if manifest != expected_manifest:
        raise ValueError("Experiment manifest does not match the supplied batch suite, contract, and protocol")

    trials = [ExperimentTrialScore.model_validate(row) for row in read_jsonl(args.trials)]
    audit = audit_experiment_manifest(manifest, trials)
    if not audit.complete:
        details = json.dumps(audit.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
        raise ValueError(f"Experiment trial audit is incomplete: {details}")

    feasibility_report = assess_feasibility(trials, suite, contract, protocol)
    utilities = aggregate_trials(trials, contract=contract)
    output_dir = _prepare_output_dir(args.output_dir)
    write_jsonl(output_dir / "audited_trials.jsonl", (row.model_dump(mode="json") for row in trials))
    write_jsonl(output_dir / "batch_utilities.jsonl", (row.model_dump(mode="json") for row in utilities))
    write_json(output_dir / "manifest_audit.json", audit.model_dump(mode="json"))
    write_json(output_dir / "feasibility_report.json", feasibility_report)
    _print(feasibility_report)
    return 0


def _cmd_metrics(args) -> int:
    suite = load_batch_suite(args.batches)
    contract = load_production_contract(args.contract)
    trials = [TrialScore.model_validate(row) for row in read_jsonl(args.trials)]
    validate_trial_completeness(
        trials,
        suite,
        contract,
        proxy_method=args.proxy_method,
        proxy_budget_id=args.proxy_budget,
        reference_method=args.reference_method,
        reference_budget_id=args.reference_budget,
    )
    utilities = aggregate_trials(trials, contract=contract)
    combinations = (
        [CombinationUtility.model_validate(row) for row in read_jsonl(args.combination_utilities)]
        if args.combination_utilities
        else []
    )
    report = evaluate_proxy(
        utilities,
        suite=suite,
        proxy_method=args.proxy_method,
        proxy_budget_id=args.proxy_budget,
        reference_method=args.reference_method,
        reference_budget_id=args.reference_budget,
        top_ks=args.top_k,
        indifference_margin=args.indifference_margin,
        combination_utilities=combinations,
    )
    output_dir = _prepare_output_dir(args.output_dir)
    write_jsonl(output_dir / "batch_utilities.jsonl", (row.model_dump(mode="json") for row in utilities))
    write_json(output_dir / "valuation_report.json", report)
    _print(report)
    return 0


def _cmd_analyze_static_baselines(args) -> int:
    suite = load_batch_suite(args.batches)
    features = [StaticBatchFeatures.model_validate(row) for row in read_jsonl(args.features)]
    references = [ReferenceBatchValue.model_validate(row) for row in read_jsonl(args.reference_values)]
    proxy_values = (
        [TrainableProxyBatchValue.model_validate(row) for row in read_jsonl(args.proxy_values)]
        if args.proxy_values
        else None
    )
    config = StaticBaselineConfig.model_validate(_load_mapping(args.config))
    report = analyze_static_baselines(suite, features, references, config, proxy_values=proxy_values)
    output_dir = _prepare_output_dir(args.output_dir)
    score_rows = report.pop("score_rows")
    write_jsonl(output_dir / "static_baseline_scores.jsonl", score_rows)
    write_json(output_dir / "static_baseline_report.json", report)
    _print(report)
    return 0


def _cmd_analyze_cross_task(args) -> int:
    rows = [CrossTaskBatchValue.model_validate(row) for row in read_jsonl(args.values)]
    config = CrossTaskConfig.model_validate(_load_mapping(args.config))
    report = analyze_cross_task_values(rows, config)
    output_dir = _prepare_output_dir(args.output_dir)
    write_json(output_dir / "cross_task_report.json", report)
    _print(report)
    return 0


def _cmd_analyze_costs(args) -> int:
    suite = load_batch_suite(args.batches)
    events = [CostLedgerEvent.model_validate(row) for row in read_jsonl(args.ledger)]
    config = CostAnalysisConfig.model_validate(_load_mapping(args.config))
    report = analyze_cost_ledger(suite, events, config)
    output_dir = _prepare_output_dir(args.output_dir)
    batch_rows = report.pop("batch_rows")
    stage_rows = report.pop("stage_rows")
    write_jsonl(output_dir / "batch_costs.jsonl", batch_rows)
    write_jsonl(output_dir / "stage_costs.jsonl", stage_rows)
    write_json(output_dir / "cost_report.json", report)
    _print(report)
    return 0


def _cmd_audit_natural_batches(args) -> int:
    suite = load_batch_suite(args.batches)
    config = NaturalBatchAuditConfig.model_validate(_load_mapping(args.config))
    report = audit_natural_batches(
        suite,
        config,
        suite_path=args.batches,
        check_paths=args.check_paths,
    )
    output_dir = _prepare_output_dir(args.output_dir)
    registry_rows = report.pop("registry_rows")
    write_jsonl(output_dir / "natural_batch_registry.jsonl", registry_rows)
    write_json(output_dir / "natural_batch_independence_report.json", report)
    _print(report)
    return 0 if report["valid"] else 2


def _cmd_mine(args) -> int:
    rows = [EvidenceUnit.model_validate(row) for row in read_jsonl(args.records)]
    data = yaml.safe_load(args.activities.read_text(encoding="utf-8"))
    raw_rules = data.get("activities", []) if isinstance(data, dict) else []
    rules = [ActivityRule.model_validate(row) for row in raw_rules]
    units = (
        mine_semantic_units(rows, batch_id=args.batch_id, activity_rules=rules)
        if args.mode == "structured"
        else fixed_chunk_baseline(
            rows,
            batch_id=args.batch_id,
            activity_rules=rules,
            chunk_characters=args.chunk_characters,
        )
    )
    obligations = build_coverage_obligations(units)
    output_dir = _prepare_output_dir(args.output_dir)
    write_jsonl(output_dir / "semantic_units.jsonl", (row.model_dump(mode="json") for row in units))
    write_jsonl(output_dir / "coverage_obligations.jsonl", (row.model_dump(mode="json") for row in obligations))
    _print({"semantic_units": len(units), "coverage_obligations": len(obligations)})
    return 0


def _cmd_select(args) -> int:
    units = [ProxyUnit.model_validate(row) for row in read_jsonl(args.units)]
    obligations = [CoverageObligation.model_validate(row) for row in read_jsonl(args.obligations)]
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = ProxySelectionConfig.model_validate(data)
    result = select_proxy(units, obligations, config)
    if args.output.exists():
        raise FileExistsError(f"Selection output already exists: {args.output.resolve()}")
    write_json(args.output, result)
    _print(result)
    return 0


def _cmd_select_ablation_fraction(args) -> int:
    """Select one batch at a relative budget without changing production semantics."""

    units = [ProxyUnit.model_validate(row) for row in read_jsonl(args.units)]
    obligations = [CoverageObligation.model_validate(row) for row in read_jsonl(args.obligations)]
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = ProxySelectionConfig.model_validate(data)
    policy = FractionalBudgetPolicy(
        budget_fraction=args.budget_fraction,
        require_strict_subset=args.require_strict_subset,
        minimum_selected_groups=args.minimum_selected_groups,
        maximum_selected_group_fraction=args.maximum_selected_group_fraction,
    )
    result = select_fractional_proxy(units, obligations, config, policy)
    if args.output.exists():
        raise FileExistsError(f"Selection output already exists: {args.output.resolve()}")
    write_json(args.output, result)
    _print(result)
    return 0


def _cmd_validate_production(args) -> int:
    eligible_rows = read_jsonl(args.eligible_units)
    eligible = [row["semantic_unit_id"] for row in eligible_rows]
    dispositions = [UnitDisposition.model_validate(row) for row in read_jsonl(args.dispositions)]
    samples = [SFTSample.model_validate(row) for row in read_jsonl(args.samples)]
    if args.batch_id and any(row.get("batch_id") != args.batch_id for row in eligible_rows):
        raise ValueError(f"Eligible semantic units contain a row outside --batch-id {args.batch_id}")
    if args.batch_id and any(sample.batch_id != args.batch_id for sample in samples):
        raise ValueError(f"Accepted SFT contains a sample outside --batch-id {args.batch_id}")
    report = validate_full_traversal(eligible, dispositions, samples)
    payload = report.model_dump(mode="json")
    lineage_values = (args.batch_id, args.batch_content_sha256, args.output)
    if any(value is not None for value in lineage_values) and not all(value is not None for value in lineage_values):
        raise ValueError(
            "Use --batch-id, --batch-content-sha256 and --output together when writing a full-traversal artifact"
        )
    if args.output:
        batch_content_sha256 = str(args.batch_content_sha256)
        if len(batch_content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in batch_content_sha256
        ):
            raise ValueError("--batch-content-sha256 must be a lowercase SHA-256 digest")
        if args.output.exists():
            raise FileExistsError(f"Full-traversal report already exists: {args.output.resolve()}")
        payload = {
            "schema_version": "full-traversal-report-v1",
            "batch_id": args.batch_id,
            "batch_content_sha256": batch_content_sha256,
            "eligible_units_sha256": file_sha256(args.eligible_units.resolve()),
            "dispositions_sha256": file_sha256(args.dispositions.resolve()),
            "source_sft_sha256": file_sha256(args.samples.resolve()),
            **payload,
        }
        write_json(args.output.resolve(), payload)
    _print(payload)
    return 0 if report.valid else 2


def _cmd_generate_proxy_sft(args) -> int:
    """Generate the formal proxy with provenance frozen by the contract."""

    result = generate_proxy_sft_suite(
        args.batches,
        args.blueprint,
        args.output_dir,
        contract=load_production_contract(args.contract),
        selection_config_path=args.selection_config,
        model=args.model,
        ollama_url=args.ollama_url,
        max_retries=args.max_retries,
        temperature=args.temperature,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
    )
    _print(result)
    return 0


def _cmd_generate_local_sft(args) -> int:
    """Generate source-grounded, explicitly non-formal local pilot data."""

    result = generate_local_sft_suite(
        args.batches,
        args.blueprint,
        args.output_dir,
        selection_config_path=args.selection_config,
        model=args.model,
        ollama_url=args.ollama_url,
        samples_per_unit=args.samples_per_unit,
        max_units=args.max_units,
        max_retries=args.max_retries,
        temperature=args.temperature,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
    )
    _print(result)
    return 0


def _cmd_import_fully_open_meditron(args) -> int:
    """Normalize and audit one independent full-reference generation batch."""

    result = import_fully_open_meditron_reference(
        batch_id=args.batch_id,
        batch_content_sha256=args.batch_content_sha256,
        contract=load_production_contract(args.contract),
        eligible_units_path=args.eligible_units,
        fully_open_meditron_path=args.fully_open_meditron_output,
        run_manifest_path=args.fully_open_meditron_run_manifest,
        output_dir=args.output_dir,
    )
    _print(result)
    return 0


def _cmd_adapt_value_population(args) -> int:
    """Adapt existing B01--B12 artifacts without inventing M0 mastery."""

    result = adapt_existing_value_population(
        suite_path=args.batches,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    _print(result)
    return 0


def _load_mapping(path: Path) -> dict[str, Any]:
    """Load one strict JSON/YAML mapping used by value-profile commands."""

    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _canonical_digest(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_profile_claim_status(profile: ModelMasteryProfile, capability_ids: set[str] | None = None) -> str:
    observations = profile.observations
    if capability_ids is not None:
        observations = [observation for observation in observations if observation.cell_id in capability_ids]
        missing = capability_ids - {observation.cell_id for observation in observations}
        if missing:
            raise ValueError(f"Model profile has no observation for benchmark cells: {sorted(missing)}")
    complete = all(
        observation.probe_count > 0 and observation.probabilities.unknown <= 1e-6 for observation in observations
    )
    return "complete" if complete else "partial_pending_model_measurement"


def _annotate_sampling_units(
    rows: list[dict[str, Any]],
    model_profile: ModelMasteryProfile,
    unit_model_profile: UnitCellMasteryProfile | None = None,
) -> tuple[list[ValueSamplingUnit], str]:
    observations = model_profile.by_cell
    capability_ids = {capability_id for row in rows for capability_id in row.get("capability_ids", [])}
    if missing := capability_ids - set(observations):
        raise ValueError(f"Model profile has no observation for benchmark cells: {sorted(missing)}")
    source_units = [ValueSamplingUnit.model_validate(row) for row in rows]
    unit_observations = None
    if unit_model_profile is not None:
        validate_unit_cell_mastery_profile(unit_model_profile, model_profile, source_units)
        unit_observations = unit_model_profile.by_unit_cell
    units = []
    for row in rows:
        payload = dict(row)
        if unit_observations is None:
            payload["model_states"] = {
                capability_id: observations[capability_id].stratum_state for capability_id in payload["capability_ids"]
            }
            # Keep the measured soft capability prior on every sampled row.
            # The hard CI state is used only by the discrete sampling strata;
            # value-profile aggregation continues to use these full
            # probabilities from the independently measured model profile.
            payload["model_state_probabilities"] = {
                capability_id: observations[capability_id].probabilities.model_dump(mode="json")
                for capability_id in payload["capability_ids"]
            }
            payload["model_state_granularity"] = "capability_prior"
        else:
            measured = {
                capability_id: unit_observations[(payload["unit_id"], capability_id)]
                for capability_id in payload["capability_ids"]
            }
            payload["model_states"] = {
                capability_id: observation.stratum_state for capability_id, observation in measured.items()
            }
            payload["model_state_probabilities"] = {
                capability_id: observation.probabilities.model_dump(mode="json")
                for capability_id, observation in measured.items()
            }
            payload["model_state_granularity"] = "unit_cell"
        units.append(ValueSamplingUnit.model_validate(payload))
    if unit_observations is None:
        status = _model_profile_claim_status(model_profile, capability_ids)
    else:
        status = (
            "complete"
            if all(
                observation.probe_count > 0 and observation.probabilities.unknown <= 1e-6
                for observation in unit_observations.values()
            )
            else "partial_pending_model_measurement"
        )
    return units, status


def _cmd_build_model_profile(args) -> int:
    """Freeze pre-scored independent M0 probes as a model-mastery profile."""

    metadata = _load_mapping(args.metadata)
    forbidden = {"observations", "probe_set_sha256", "scoring_protocol_sha256"} & set(metadata)
    if forbidden:
        raise ValueError(f"Model-profile metadata contains derived fields: {sorted(forbidden)}")
    cells = [BenchmarkCellDefinition.model_validate(row) for row in read_jsonl(args.cells)]
    probes = [ScoredProbe.model_validate(row) for row in read_jsonl(args.probes)]
    aggregation = ProbeAggregationConfig.model_validate(_load_mapping(args.aggregation_config))
    probe_set_sha256 = _canonical_digest(
        {
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "probes": [probe.model_dump(mode="json") for probe in probes],
        }
    )
    evidence_protocol_sha256 = None
    run_manifest = None
    run_manifest_sha256 = None
    if args.probe_run_manifest is not None:
        run_manifest = ProbeRunManifest.model_validate(_load_mapping(args.probe_run_manifest))
        run_manifest_sha256 = file_sha256(args.probe_run_manifest.resolve())
        if file_sha256(args.probes.resolve()) != run_manifest.scored_probes_sha256:
            raise ValueError("Scored probes do not match the bound Base Model probe run")
        for field_name in ("model_id", "model_revision", "checkpoint_sha256"):
            if metadata.get(field_name) != getattr(run_manifest, field_name):
                raise ValueError(f"Model-profile metadata {field_name} does not match the probe run")
        if len(probes) != run_manifest.prompt_count:
            raise ValueError("Scored-probe row count does not match the probe run prompt count")
        # A real run binds the source-grounded catalogue rather than deriving
        # its identity from model answers. Prompt variants and scores remain
        # bound through the evidence-protocol digest below.
        probe_set_sha256 = run_manifest.probe_set_sha256
        evidence_protocol_sha256 = _canonical_digest(
            {
                "schema_version": "model-profile-evidence-protocol-v1",
                "aggregation_config": aggregation.model_dump(mode="json"),
                "probe_run_manifest_sha256": run_manifest_sha256,
                "inference_config_sha256": run_manifest.inference_config_sha256,
                "prompt_protocol_sha256": run_manifest.prompt_protocol_sha256,
                "scoring_protocol_sha256": run_manifest.scoring_protocol_sha256,
                "predictions_sha256": run_manifest.predictions_sha256,
                "scored_probes_sha256": run_manifest.scored_probes_sha256,
            }
        )
    profile = build_model_mastery_profile_from_probes(
        **metadata,
        probe_set_sha256=probe_set_sha256,
        cells=cells,
        probes=probes,
        aggregation_config=aggregation,
        evidence_protocol_sha256=evidence_protocol_sha256,
    )
    observations = profile.observations
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, profile.model_dump(mode="json"))
    status = _model_profile_claim_status(profile)
    manifest = {
        "schema_version": "model-mastery-profile-build-v1",
        "profile_id": profile.profile_id,
        "profile_sha256": _canonical_digest(profile),
        "claim_status": status,
        "measurement_granularity": "capability",
        "operation_dimension": (
            "absent" if all(row.operation_id == "__absent__" for row in observations) else "measured"
        ),
        "difficulty_dimension": (
            "absent" if all(row.difficulty_id == "__absent__" for row in observations) else "measured"
        ),
        "benchmark_test_items_accessed": False,
        "registered_cell_count": len(cells),
        "scored_probe_count": len(probes),
        "observation_count": len(observations),
        "unmeasured_cell_ids": sorted(row.cell_id for row in observations if row.stratum_state == "unknown"),
    }
    if run_manifest is not None:
        manifest.update(
            {
                "schema_version": "model-mastery-profile-build-v2",
                "probe_run_manifest": str(args.probe_run_manifest.resolve()),
                "probe_run_manifest_sha256": run_manifest_sha256,
                "raw_predictions_sha256": run_manifest.predictions_sha256,
                "scored_probes_sha256": run_manifest.scored_probes_sha256,
                "benchmark_test_items_accessed": run_manifest.benchmark_test_items_accessed,
                "locked_test_items_used": run_manifest.locked_test_items_used,
                "evidence_protocol_sha256": evidence_protocol_sha256,
            }
        )
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    write_json(manifest_path, manifest)
    _print({"output": str(output), "manifest": str(manifest_path), **manifest})
    return 0


def _cmd_run_model_probes(args) -> int:
    """Run a frozen Base Model on independent source-grounded probes."""

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    config = ProbeInferenceConfig.model_validate(_load_mapping(args.config))
    artifacts = run_model_probe_pipeline(
        catalogue_path=args.catalogue,
        output_dir=output_dir,
        config=config,
    )
    manifest = ProbeRunManifest.model_validate(_load_mapping(artifacts.run_manifest_path))
    _print(
        {
            "output_dir": str(artifacts.output_dir),
            "checkpoint_sha256": artifacts.checkpoint_sha256,
            "probe_set_sha256": artifacts.probe_set_sha256,
            "run_manifest": str(artifacts.run_manifest_path),
            "run_manifest_sha256": artifacts.run_manifest_sha256,
            "scored_probes": str(artifacts.scored_probes_path),
            "prompt_count": manifest.prompt_count,
            "valid_prediction_count": manifest.valid_prediction_count,
            "invalid_prediction_count": manifest.invalid_prediction_count,
            "ambiguous_prediction_count": manifest.ambiguous_prediction_count,
        }
    )
    return 0


def _cmd_build_unit_model_profile(args) -> int:
    """Bind unit-by-cell probe results to one model and source population."""

    metadata = _load_mapping(args.metadata)
    allowed = {"profile_id", "probe_set_id", "scoring_protocol_sha256"}
    if unknown := set(metadata) - allowed:
        raise ValueError(f"Unit-model metadata contains unsupported or derived fields: {sorted(unknown)}")
    if missing := allowed - set(metadata):
        raise ValueError(f"Unit-model metadata is missing required fields: {sorted(missing)}")
    parent = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    units = [ValueSamplingUnit.model_validate(row) for row in read_jsonl(args.units)]
    observations = [UnitCellMasteryObservation.model_validate(row) for row in read_jsonl(args.observations)]
    profile = build_unit_cell_mastery_profile(
        **metadata,
        parent=parent,
        units=units,
        observations=observations,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, profile.model_dump(mode="json"))
    unknown_count = sum(item.stratum_state == "unknown" for item in profile.observations)
    manifest = {
        "schema_version": "unit-cell-mastery-profile-build-v1",
        "profile_id": profile.profile_id,
        "profile_sha256": unit_cell_mastery_profile_digest(profile),
        "parent_model_profile_id": profile.parent_model_profile_id,
        "population_units_sha256": profile.population_units_sha256,
        "measurement_granularity": "unit_cell",
        "observation_count": len(profile.observations),
        "unknown_observation_count": unknown_count,
        "claim_status": "complete" if unknown_count == 0 else "partial_pending_model_measurement",
        "benchmark_test_items_accessed": False,
    }
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    write_json(manifest_path, manifest)
    _print({"output": str(output), "manifest": str(manifest_path), **manifest})
    return 0


def _cmd_build_sparse_unit_model_profile(args) -> int:
    """Bind existing independent probes to source rules and build a sparse profile."""

    output_dir = _prepare_output_dir(args.output_dir)
    parent = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    population = [ValueSamplingUnit.model_validate(row) for row in read_jsonl(args.units)]
    catalogue = load_probe_catalogue(args.catalogue)
    scored = [ScoredProbe.model_validate(row) for row in read_jsonl(args.scored_probes)]
    aggregation = ProbeAggregationConfig.model_validate(_load_mapping(args.aggregation_config))
    semantic_rows = []
    for path in sorted(args.artifact_root.resolve().glob("B*/semantic_units.jsonl")):
        semantic_rows.extend(read_jsonl(path))
    if not semantic_rows:
        raise FileNotFoundError(f"No B*/semantic_units.jsonl files found under {args.artifact_root}")
    bindings = bind_probes_to_semantic_units(
        catalogue.probes,
        semantic_rows,
        population,
        require_all=False,
    )
    bound_probe_ids = {row.probe_id for row in bindings}
    unbound_probe_ids = sorted({row.probe_id for row in catalogue.probes} - bound_probe_ids)
    profile, observations, manifest = build_sparse_unit_profile(
        profile_id=args.profile_id,
        probe_set_id=args.probe_set_id,
        parent=parent,
        population_units=population,
        bindings=bindings,
        scored_probes=scored,
        aggregation_config=aggregation,
    )
    bindings_path = output_dir / "probe_unit_bindings.jsonl"
    observations_path = output_dir / "unit_cell_observations.jsonl"
    profile_path = output_dir / "unit_model_profile.json"
    write_jsonl(bindings_path, (row.model_dump(mode="json") for row in bindings))
    write_jsonl(observations_path, (row.model_dump(mode="json") for row in observations))
    write_json(profile_path, profile.model_dump(mode="json"))
    manifest.update(
        {
            "probe_count": len(catalogue.probes),
            "bound_probe_count": len(bound_probe_ids),
            "unbound_probe_ids": unbound_probe_ids,
            "probe_binding_completeness": len(bound_probe_ids) / len(catalogue.probes),
            "catalogue_sha256": file_sha256(args.catalogue.resolve()),
            "scored_probes_sha256": file_sha256(args.scored_probes.resolve()),
            "parent_model_profile_sha256": _canonical_digest(parent),
            "population_units_sha256": profile.population_units_sha256,
            "bindings_sha256": file_sha256(bindings_path),
            "observations_sha256": file_sha256(observations_path),
            "unit_model_profile_sha256": unit_cell_mastery_profile_digest(profile),
        }
    )
    if unbound_probe_ids:
        manifest["limitations"].append(
            "Some probes were excluded because the frozen semantic-unit capability labels did not "
            "contain the probe cell; the population was not relabelled after freezing."
        )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    _print({"output_dir": str(output_dir), "profile": str(profile_path), "manifest": str(manifest_path), **manifest})
    return 0


def _cmd_analyse_prior_manipulation(args) -> int:
    """Check whether sparse unit-level model states actually change sampling."""

    parent = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    unit_profile = UnitCellMasteryProfile.model_validate(_load_mapping(args.unit_model_profile))
    population = [ValueSamplingUnit.model_validate(row) for row in read_jsonl(args.units)]
    config = PriorManipulationConfig.model_validate(_load_mapping(args.config))
    result = analyse_prior_manipulation(population, parent, unit_profile, config)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, result)
    _print({"output": str(output), **{key: value for key, value in result.items() if key != "trials"}})
    return 0


def _cmd_analyse_prior_holdout(args) -> int:
    """Split independent items into design/audit panels and test held-out fidelity."""

    output_dir = _prepare_output_dir(args.output_dir)
    parent = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    population = [ValueSamplingUnit.model_validate(row) for row in read_jsonl(args.units)]
    bindings = [ProbeUnitBinding.model_validate(row) for row in read_jsonl(args.bindings)]
    scored = [ScoredProbe.model_validate(row) for row in read_jsonl(args.scored_probes)]
    aggregation = ProbeAggregationConfig.model_validate(_load_mapping(args.aggregation_config))
    sampling = PriorManipulationConfig.model_validate(_load_mapping(args.sampling_config))
    holdout = ProbeHoldoutConfig.model_validate(_load_mapping(args.holdout_config))
    design_bindings, audit_bindings, split_manifest = split_bindings_for_probe_holdout(bindings, holdout)
    design_profile, design_observations, design_manifest = build_sparse_unit_profile(
        profile_id=args.design_profile_id,
        probe_set_id=f"{args.probe_set_id}-design",
        parent=parent,
        population_units=population,
        bindings=design_bindings,
        scored_probes=scored,
        aggregation_config=aggregation,
    )
    audit_profile, audit_observations, audit_manifest = build_sparse_unit_profile(
        profile_id=args.audit_profile_id,
        probe_set_id=f"{args.probe_set_id}-audit",
        parent=parent,
        population_units=population,
        bindings=audit_bindings,
        scored_probes=scored,
        aggregation_config=aggregation,
    )
    analysis = analyse_prior_holdout_fidelity(
        population,
        parent,
        design_profile,
        audit_profile,
        sampling,
        holdout,
    )
    paths = {
        "design_bindings": output_dir / "design_bindings.jsonl",
        "audit_bindings": output_dir / "audit_bindings.jsonl",
        "design_observations": output_dir / "design_observations.jsonl",
        "audit_observations": output_dir / "audit_observations.jsonl",
        "design_profile": output_dir / "design_unit_model_profile.json",
        "audit_profile": output_dir / "audit_unit_model_profile.json",
        "analysis": output_dir / "holdout_analysis.json",
    }
    write_jsonl(paths["design_bindings"], (row.model_dump(mode="json") for row in design_bindings))
    write_jsonl(paths["audit_bindings"], (row.model_dump(mode="json") for row in audit_bindings))
    write_jsonl(paths["design_observations"], (row.model_dump(mode="json") for row in design_observations))
    write_jsonl(paths["audit_observations"], (row.model_dump(mode="json") for row in audit_observations))
    write_json(paths["design_profile"], design_profile.model_dump(mode="json"))
    write_json(paths["audit_profile"], audit_profile.model_dump(mode="json"))
    write_json(paths["analysis"], analysis)
    manifest = {
        "schema_version": "unit-prior-holdout-experiment-v1",
        "split": split_manifest,
        "design_profile": design_manifest,
        "audit_profile": audit_manifest,
        "analysis_status": analysis["status"],
        "outputs": {name: str(path.relative_to(output_dir)) for name, path in paths.items()},
        "output_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    _print(
        {
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            **{key: value for key, value in analysis.items() if key != "trials"},
        }
    )
    return 0


def _cmd_sample_value_proxy(args) -> int:
    """Draw a probability sample that preserves data × model × task strata."""

    model_profile = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    rows = [row for row in read_jsonl(args.units) if row.get("batch_id") == args.batch_id]
    if not rows:
        raise ValueError(f"No value-population units found for batch {args.batch_id}")
    unit_model_profile = (
        UnitCellMasteryProfile.model_validate(_load_mapping(args.unit_model_profile))
        if args.unit_model_profile
        else None
    )
    units, claim_status = _annotate_sampling_units(rows, model_profile, unit_model_profile)
    config = ValueSamplingConfig.model_validate(_load_mapping(args.config))
    sample_manifest = sample_value_distribution(units, config)
    sample_manifest.update(
        {
            "claim_status": claim_status,
            "model_profile_id": model_profile.profile_id,
            "model_profile_sha256": _canonical_digest(model_profile),
            "model_measurement_granularity": units[0].model_state_granularity,
            "unit_model_profile_id": unit_model_profile.profile_id if unit_model_profile else None,
            "unit_model_profile_sha256": (
                unit_cell_mastery_profile_digest(unit_model_profile) if unit_model_profile else None
            ),
            "benchmark_test_items_accessed": False,
        }
    )

    if (args.artifact_root is None) != (args.membership_map is None):
        raise ValueError("--artifact-root and --membership-map must be supplied together")
    output_dir = _prepare_output_dir(args.output_dir)
    manifest_path = output_dir / "value_sample_manifest.json"
    write_json(manifest_path, sample_manifest)
    write_jsonl(output_dir / "selected_value_units.jsonl", sample_manifest["selected_unit_records"])
    population_frame_path = output_dir / "annotated_population_units.jsonl"
    write_jsonl(
        population_frame_path,
        [unit.model_dump(mode="json") for unit in sorted(units, key=lambda row: row.unit_id)],
    )

    source_sample = None
    if args.artifact_root is not None:
        source_sample = materialize_selected_value_artifacts(
            sample_manifest=sample_manifest,
            membership_map_path=args.membership_map,
            artifact_root=args.artifact_root,
            output_dir=output_dir / "source_sample",
        )
    response = {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "annotated_population_units": str(population_frame_path),
        "batch_id": args.batch_id,
        "claim_status": claim_status,
        "model_measurement_granularity": units[0].model_state_granularity,
        "population_group_count": sample_manifest["population_group_count"],
        "selected_group_count": sample_manifest["realized_group_count"],
        "selected_unit_count": len(sample_manifest["selected_unit_ids"]),
        "weighted_total_variation": sample_manifest["weighted_total_variation"],
        "source_sample_materialized": source_sample is not None,
    }
    _print(response)
    return 0


def _verify_value_sampling_provenance(
    manifest_path: Path,
    population_units_path: Path,
) -> dict[str, Any]:
    """Load and replay one sampling manifest against an annotated frame."""

    manifest = _load_mapping(manifest_path)
    units = [ValueSamplingUnit.model_validate(row) for row in read_jsonl(population_units_path)]
    return verify_value_sample_manifest(manifest, units)


def _require_provenance_matches_profiles(
    provenance: dict[str, Any],
    reference: ValueProfile,
    raw_sample: ValueProfile,
    reconstruction: ValueProfile,
) -> None:
    """Bind a successfully replayed draw to the profiles being certified."""

    expected = {
        "batch_id": reference.batch_id,
        "batch_content_sha256": reference.batch_content_sha256,
        "reference_membership_sha256": reference.membership_sha256,
        "sample_membership_sha256": raw_sample.membership_sha256,
        "sample_design_sha256": raw_sample.sampling_design_sha256,
    }
    mismatches = [name for name, value in expected.items() if provenance.get(name) != value]
    if reconstruction.membership_sha256 != raw_sample.membership_sha256:
        mismatches.append("reconstructed_sample_membership_sha256")
    if reconstruction.sampling_design_sha256 != raw_sample.sampling_design_sha256:
        mismatches.append("reconstructed_sample_design_sha256")
    if mismatches:
        raise ValueError(
            "Verified sampling manifest does not describe the value profiles being certified; "
            f"mismatched fields: {mismatches}"
        )


def _cmd_verify_value_sample_manifest(args) -> int:
    """Prove selected membership and pi by replaying the registered design."""

    audit = _verify_value_sampling_provenance(args.manifest, args.population_units)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        write_json(output, audit)
    _print({**audit, "output": str(args.output.resolve()) if args.output is not None else None})
    return 0


def _cmd_sweep_value_fidelity(args) -> int:
    """Find stable generation-stage sample budgets across batches and seeds."""

    model_profile = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    rows = read_jsonl(args.units)
    if args.batch_id:
        selected_batches = set(args.batch_id)
        rows = [row for row in rows if row.get("batch_id") in selected_batches]
        missing = selected_batches - {row.get("batch_id") for row in rows}
        if missing:
            raise ValueError(f"No value-population units found for batches: {sorted(missing)}")
    if not rows:
        raise ValueError("No value-population units are available for the sweep")
    unit_model_profile = (
        UnitCellMasteryProfile.model_validate(_load_mapping(args.unit_model_profile))
        if args.unit_model_profile
        else None
    )
    units, claim_status = _annotate_sampling_units(rows, model_profile, unit_model_profile)
    config = ValueFidelitySweepConfig.model_validate(_load_mapping(args.config))
    thresholds = FidelityThresholds.model_validate(_load_mapping(args.thresholds))
    critical_weights = _load_mapping(args.critical_cell_weights) if args.critical_cell_weights else None
    result = run_value_fidelity_sweep(
        units,
        model_profile,
        thresholds,
        config,
        critical_cell_weights=critical_weights,
    )
    result["model_measurement_granularity"] = units[0].model_state_granularity
    result["unit_model_profile_id"] = unit_model_profile.profile_id if unit_model_profile else None
    result["unit_model_profile_sha256"] = (
        unit_cell_mastery_profile_digest(unit_model_profile) if unit_model_profile else None
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, result)
    _print(
        {
            "output": str(output),
            "claim_status": claim_status,
            "model_dimension_evaluated": result["model_dimension_evaluated"],
            "model_measurement_granularity": units[0].model_state_granularity,
            "attribute_roles": result["attribute_roles"],
            "batch_count": len(result["batch_summary"]),
            "batch_summary": result["batch_summary"],
        }
    )
    return 0


def _cmd_build_value_profile(args) -> int:
    """Aggregate one full, raw-sample, or HT-reconstructed value profile."""

    model_profile = ModelMasteryProfile.model_validate(_load_mapping(args.model_profile))
    rows = [row for row in read_jsonl(args.units) if row.get("batch_id") == args.batch_id]
    if not rows:
        raise ValueError(f"No units found for batch {args.batch_id}")
    if args.unit_model_profile:
        if args.kind != "full_population":
            raise ValueError(
                "--unit-model-profile is used to annotate the full population; sampled records must "
                "already carry their frozen unit-level states"
            )
        unit_model_profile = UnitCellMasteryProfile.model_validate(_load_mapping(args.unit_model_profile))
        annotated, _ = _annotate_sampling_units(rows, model_profile, unit_model_profile)
        units = [conditional_value_unit_from_sampling_record(row) for row in annotated]
    else:
        units = [conditional_value_unit_from_sampling_record(row) for row in rows]
    if args.kind == "full_population":
        profile = build_value_profile(units, model_profile, profile_id=args.profile_id)
    elif args.kind == "raw_sample":
        profile = build_raw_sample_profile(units, model_profile, profile_id=args.profile_id)
    else:
        profile = reconstruct_value_profile(units, model_profile, profile_id=args.profile_id)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, profile.model_dump(mode="json"))
    claim_status = (
        "complete"
        if not profile.unmeasured_cell_ids and profile.unknown_mastery_mass <= 1e-6
        else "partial_pending_model_measurement"
    )
    _print(
        {
            "output": str(output),
            "profile_id": profile.profile_id,
            "population_kind": profile.population_kind,
            "claim_status": claim_status,
            "unknown_mastery_mass": profile.unknown_mastery_mass,
            "unmeasured_cell_ids": profile.unmeasured_cell_ids,
        }
    )
    return 0


def _cmd_build_value_certificate(args) -> int:
    """Build and immediately replay-verify one profile-fidelity certificate."""

    reference = ValueProfile.model_validate(_load_mapping(args.reference_profile))
    raw_sample = ValueProfile.model_validate(_load_mapping(args.raw_sample_profile))
    reconstruction = ValueProfile.model_validate(_load_mapping(args.reconstructed_profile))
    thresholds = FidelityThresholds.model_validate(_load_mapping(args.thresholds))
    critical_weights = _load_mapping(args.critical_cell_weights) if args.critical_cell_weights else None
    if (args.sampling_manifest is None) != (args.sampling_population_units is None):
        raise ValueError("--sampling-manifest and --sampling-population-units must be supplied together")
    provenance = None
    if args.sampling_manifest is not None:
        provenance = _verify_value_sampling_provenance(
            args.sampling_manifest,
            args.sampling_population_units,
        )
        _require_provenance_matches_profiles(
            provenance,
            reference,
            raw_sample,
            reconstruction,
        )
    certificate = build_fidelity_certificate(
        reference,
        raw_sample,
        reconstruction,
        certificate_id=args.certificate_id,
        thresholds=thresholds,
        critical_cell_weights=critical_weights,
        sampling_manifest_sha256=(provenance["sampling_manifest_sha256"] if provenance else None),
        design_provenance_verified=provenance is not None,
    )
    verify_fidelity_certificate(
        certificate,
        reference,
        raw_sample,
        reconstruction,
        sampling_manifest_sha256=(provenance["sampling_manifest_sha256"] if provenance else None),
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    write_json(output, certificate.model_dump(mode="json"))
    claim_status = (
        "partial_pending_model_measurement"
        if not certificate.complete_three_dimensional_claim
        else ("complete" if certificate.passed else "invalid")
    )
    _print(
        {
            "output": str(output),
            "certificate_id": certificate.certificate_id,
            "certificate_sha256": _canonical_digest(certificate),
            "claim_status": claim_status,
            "fidelity_passed": certificate.passed,
            "distribution_fidelity_passed": certificate.distribution_fidelity_passed,
            "design_provenance_verified": certificate.design_provenance_verified,
            "sampling_manifest_sha256": certificate.sampling_manifest_sha256,
        }
    )
    return 0


def _cmd_verify_value_certificate(args) -> int:
    """Replay a saved certificate against the three frozen profiles."""

    certificate = ProfileFidelityCertificate.model_validate(_load_mapping(args.certificate))
    reference = ValueProfile.model_validate(_load_mapping(args.reference_profile))
    raw_sample = ValueProfile.model_validate(_load_mapping(args.raw_sample_profile))
    reconstruction = ValueProfile.model_validate(_load_mapping(args.reconstructed_profile))
    if (args.sampling_manifest is None) != (args.sampling_population_units is None):
        raise ValueError("--sampling-manifest and --sampling-population-units must be supplied together")
    provenance = None
    if args.sampling_manifest is not None:
        provenance = _verify_value_sampling_provenance(
            args.sampling_manifest,
            args.sampling_population_units,
        )
        _require_provenance_matches_profiles(
            provenance,
            reference,
            raw_sample,
            reconstruction,
        )
    verify_fidelity_certificate(
        certificate,
        reference,
        raw_sample,
        reconstruction,
        sampling_manifest_sha256=(provenance["sampling_manifest_sha256"] if provenance else None),
    )
    claim_status = (
        "partial_pending_model_measurement"
        if not certificate.complete_three_dimensional_claim
        else ("complete" if certificate.passed else "invalid")
    )
    _print(
        {
            "valid": True,
            "certificate_id": certificate.certificate_id,
            "certificate_sha256": _canonical_digest(certificate),
            "claim_status": claim_status,
            "fidelity_passed": certificate.passed,
            "distribution_fidelity_passed": certificate.distribution_fidelity_passed,
            "design_provenance_verified": certificate.design_provenance_verified,
            "sampling_manifest_sha256": certificate.sampling_manifest_sha256,
        }
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark-conditioned document-data valuation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the batch design and production contract")
    validate.add_argument("--batches", type=Path, required=True)
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--check-paths", action="store_true")
    validate.set_defaults(handler=_cmd_validate)

    freeze_batches = subparsers.add_parser(
        "freeze-batches", help="Fingerprint local batch contents and write a new suite YAML"
    )
    freeze_batches.add_argument("--batches", type=Path, required=True)
    freeze_batches.add_argument("--output", type=Path, required=True)
    freeze_batches.set_defaults(handler=_cmd_freeze_batches)

    plan = subparsers.add_parser("plan-reference", help="Freeze and plan every high-budget reference run")
    plan.add_argument("--batches", type=Path, required=True)
    plan.add_argument("--contract", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--check-paths", action="store_true")
    plan.set_defaults(handler=_cmd_plan)

    plan_experiment = subparsers.add_parser(
        "plan-experiment", help="Freeze every target, baseline and full-reference trial before execution"
    )
    plan_experiment.add_argument("--batches", type=Path, required=True)
    plan_experiment.add_argument("--contract", type=Path, required=True)
    plan_experiment.add_argument("--protocol", type=Path, required=True)
    plan_experiment.add_argument("--output-dir", type=Path, required=True)
    plan_experiment.set_defaults(handler=_cmd_plan_experiment)

    plan_ablations = subparsers.add_parser(
        "plan-ablations",
        help="Validate and freeze the stage-level small-sample mechanism ablation matrix",
    )
    plan_ablations.add_argument("--protocol", type=Path, required=True)
    plan_ablations.add_argument("--output-dir", type=Path, required=True)
    plan_ablations.set_defaults(handler=_cmd_plan_ablations)

    analyze_ablations = subparsers.add_parser(
        "analyze-ablations",
        help="Audit paired stage observations and apply preregistered ablation gates",
    )
    analyze_ablations.add_argument("--protocol", type=Path, required=True)
    analyze_ablations.add_argument("--observations", type=Path, required=True)
    analyze_ablations.add_argument(
        "--study",
        action="append",
        help="Analyze only this study ID; repeat for multiple sequential-gate studies",
    )
    analyze_ablations.add_argument("--output-dir", type=Path, required=True)
    analyze_ablations.set_defaults(handler=_cmd_analyze_ablations)

    scaffold_lf = subparsers.add_parser(
        "scaffold-llamafactory",
        help="Write the expected batch × arm SFT artifact registry template",
    )
    scaffold_lf.add_argument("--manifest", type=Path, required=True)
    scaffold_lf.add_argument("--sft-root", type=Path, required=True)
    scaffold_lf.add_argument("--output", type=Path, required=True)
    scaffold_lf.set_defaults(handler=_cmd_scaffold_llamafactory)

    inspect_lf = subparsers.add_parser(
        "inspect-llamafactory-profile",
        help="Summarize a pinned LLaMAFactory server profile without starting training",
    )
    inspect_lf.add_argument("--profile", type=Path, required=True)
    inspect_lf.set_defaults(handler=_cmd_inspect_llamafactory_profile)

    prepare_lf = subparsers.add_parser(
        "prepare-llamafactory",
        help="Materialize audited datasets and one frozen LLaMAFactory job per experiment trial",
    )
    prepare_lf.add_argument("--manifest", type=Path, required=True)
    prepare_lf.add_argument("--contract", type=Path, required=True)
    prepare_lf.add_argument("--registry", type=Path, required=True)
    prepare_lf.add_argument("--profile", type=Path, required=True)
    prepare_lf.add_argument("--llamafactory-root", type=Path, required=True)
    prepare_lf.add_argument("--output-dir", type=Path, required=True)
    prepare_lf.add_argument(
        "--discard-incomplete",
        action="store_true",
        help="Discard only a prior bundle carrying PREPARING.json and no BUNDLE_READY.json",
    )
    prepare_lf.set_defaults(handler=_cmd_prepare_llamafactory)

    run_lf = subparsers.add_parser("run-llamafactory", help="Run exactly one frozen LLaMAFactory job")
    run_lf.add_argument("--jobs", type=Path, required=True)
    run_selector = run_lf.add_mutually_exclusive_group(required=True)
    run_selector.add_argument("--run-id")
    run_selector.add_argument("--job-index", type=int, help="Zero-based index from llamafactory_jobs.json")
    run_lf.add_argument("--resume", action="store_true")
    run_lf.add_argument(
        "--reclaim-stale-lock",
        action="store_true",
        help="Archive a stale lock after refusing any live process on the current host",
    )
    run_lf.add_argument(
        "--restart-incomplete",
        action="store_true",
        help="Archive a non-resumable adapter directory and restart from the frozen base model",
    )
    run_lf.set_defaults(handler=_cmd_run_llamafactory)

    lf_status = subparsers.add_parser(
        "llamafactory-status",
        help="Summarize training states from a frozen LLaMAFactory job manifest",
    )
    lf_status.add_argument("--jobs", type=Path, required=True)
    lf_status.set_defaults(handler=_cmd_llamafactory_status)

    analyze_experiment = subparsers.add_parser(
        "analyze-experiment", help="Audit completed experiment trials and issue a descriptive pilot verdict"
    )
    analyze_experiment.add_argument("--batches", type=Path, required=True)
    analyze_experiment.add_argument("--contract", type=Path, required=True)
    analyze_experiment.add_argument("--protocol", type=Path, required=True)
    analyze_experiment.add_argument("--manifest", type=Path, required=True)
    analyze_experiment.add_argument("--trials", type=Path, required=True)
    analyze_experiment.add_argument("--output-dir", type=Path, required=True)
    analyze_experiment.set_defaults(handler=_cmd_analyze_experiment)

    metrics = subparsers.add_parser("metrics", help="Compare proxy trials with high-budget reference trials")
    metrics.add_argument("--batches", type=Path, required=True)
    metrics.add_argument("--contract", type=Path, required=True)
    metrics.add_argument("--trials", type=Path, required=True)
    metrics.add_argument("--proxy-method", required=True)
    metrics.add_argument("--proxy-budget", required=True)
    metrics.add_argument("--reference-method", required=True)
    metrics.add_argument("--reference-budget", required=True)
    metrics.add_argument("--top-k", nargs="+", type=int, default=[1, 3])
    metrics.add_argument("--indifference-margin", type=float, default=0.0)
    metrics.add_argument("--combination-utilities", type=Path)
    metrics.add_argument("--output-dir", type=Path, required=True)
    metrics.set_defaults(handler=_cmd_metrics)

    static_baselines = subparsers.add_parser(
        "analyze-static-baselines",
        help="Compare preregistered static batch scores with full-training value",
    )
    static_baselines.add_argument("--batches", type=Path, required=True)
    static_baselines.add_argument("--features", type=Path, required=True)
    static_baselines.add_argument("--reference-values", type=Path, required=True)
    static_baselines.add_argument(
        "--proxy-values",
        type=Path,
        help="Optional frozen trainable-proxy batch utilities for direct proxy-minus-static contrasts",
    )
    static_baselines.add_argument("--config", type=Path, required=True)
    static_baselines.add_argument("--output-dir", type=Path, required=True)
    static_baselines.set_defaults(handler=_cmd_analyze_static_baselines)

    cross_task = subparsers.add_parser(
        "analyze-cross-task",
        help="Test proxy fidelity and reference-ranking changes across downstream tasks",
    )
    cross_task.add_argument("--values", type=Path, required=True)
    cross_task.add_argument("--config", type=Path, required=True)
    cross_task.add_argument("--output-dir", type=Path, required=True)
    cross_task.set_defaults(handler=_cmd_analyze_cross_task)

    costs = subparsers.add_parser(
        "analyze-costs",
        help="Aggregate a complete proxy/reference event ledger without dropping failed work",
    )
    costs.add_argument("--batches", type=Path, required=True)
    costs.add_argument("--ledger", type=Path, required=True)
    costs.add_argument("--config", type=Path, required=True)
    costs.add_argument("--output-dir", type=Path, required=True)
    costs.set_defaults(handler=_cmd_analyze_costs)

    natural_batches = subparsers.add_parser(
        "audit-natural-batches",
        help="Audit natural-check batch independence and emit a frozen registry",
    )
    natural_batches.add_argument("--batches", type=Path, required=True)
    natural_batches.add_argument("--config", type=Path, required=True)
    natural_batches.add_argument("--check-paths", action="store_true")
    natural_batches.add_argument("--output-dir", type=Path, required=True)
    natural_batches.set_defaults(handler=_cmd_audit_natural_batches)

    mine = subparsers.add_parser("mine", help="Mine source-bound semantic units without generating QA")
    mine.add_argument("--records", type=Path, required=True)
    mine.add_argument("--activities", type=Path, required=True)
    mine.add_argument("--batch-id", required=True)
    mine.add_argument(
        "--mode",
        choices=["structured", "fixed-chunks"],
        default="structured",
        help="Use objective structured mining or the A1 fixed-chunk baseline",
    )
    mine.add_argument("--chunk-characters", type=int, default=512)
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.set_defaults(handler=_cmd_mine)

    proxy_sft = subparsers.add_parser(
        "generate-proxy-sft",
        help="Generate a formal coverage proxy whose provenance is frozen by a production contract",
    )
    proxy_sft.add_argument("--batches", type=Path, required=True)
    proxy_sft.add_argument("--contract", type=Path, required=True)
    proxy_sft.add_argument("--blueprint", type=Path, required=True)
    proxy_sft.add_argument(
        "--selection-config",
        type=Path,
        required=True,
        help="One ProxySelectionConfig YAML applied independently before generation in every batch",
    )
    proxy_sft.add_argument("--output-dir", type=Path, required=True)
    proxy_sft.add_argument("--model", help="Optional assertion; must equal the contract teacher_model")
    proxy_sft.add_argument("--ollama-url", default="http://localhost:11434")
    proxy_sft.add_argument(
        "--max-retries",
        type=int,
        help="Optional assertion; must equal the contract max_retries",
    )
    proxy_sft.add_argument(
        "--temperature",
        type=float,
        help="Optional assertion; must equal the contract temperature",
    )
    proxy_sft.add_argument("--seed", type=int, default=42)
    proxy_sft.add_argument("--max-output-tokens", type=int, default=512)
    proxy_sft.set_defaults(handler=_cmd_generate_proxy_sft)

    local_sft = subparsers.add_parser(
        "generate-local-sft",
        help="Generate a non-formal local pilot; provenance is not bound to the production contract",
    )
    local_sft.add_argument("--batches", type=Path, required=True)
    local_sft.add_argument("--blueprint", type=Path, required=True)
    local_sft.add_argument("--selection-config", type=Path, required=True)
    local_sft.add_argument("--output-dir", type=Path, required=True)
    local_sft.add_argument("--model", default="qwen2.5:1.5b-instruct")
    local_sft.add_argument("--ollama-url", default="http://localhost:11434")
    local_sft.add_argument("--samples-per-unit", type=int, default=1)
    local_sft.add_argument(
        "--max-units",
        type=int,
        help="Deterministic per-batch candidate cap applied before proxy selection",
    )
    local_sft.add_argument("--max-retries", type=int, default=3)
    local_sft.add_argument("--temperature", type=float, default=0.0)
    local_sft.add_argument("--seed", type=int, default=42)
    local_sft.add_argument("--max-output-tokens", type=int, default=512)
    local_sft.set_defaults(handler=_cmd_generate_local_sft)

    import_fom = subparsers.add_parser(
        "import-fully-open-meditron",
        help="Import a normalized Fully Open Meditron batch as the independent full-reference SFT",
    )
    import_fom.add_argument("--contract", type=Path, required=True)
    import_fom.add_argument("--batch-id", required=True)
    import_fom.add_argument("--batch-content-sha256", required=True)
    import_fom.add_argument("--eligible-units", type=Path, required=True)
    import_fom.add_argument("--fully-open-meditron-output", type=Path, required=True)
    import_fom.add_argument("--fully-open-meditron-run-manifest", type=Path, required=True)
    import_fom.add_argument("--output-dir", type=Path, required=True)
    import_fom.set_defaults(handler=_cmd_import_fully_open_meditron)

    adapt_value = subparsers.add_parser(
        "adapt-value-population",
        help="Adapt existing semantic/proxy artifacts into value-distribution records",
    )
    adapt_value.add_argument("--batches", type=Path, required=True)
    adapt_value.add_argument("--artifact-root", type=Path, required=True)
    adapt_value.add_argument("--output-dir", type=Path, required=True)
    adapt_value.set_defaults(handler=_cmd_adapt_value_population)

    model_profile = subparsers.add_parser(
        "build-model-profile",
        help="Freeze pre-scored independent M0 probes; missing probes remain unknown",
    )
    model_profile.add_argument("--metadata", type=Path, required=True)
    model_profile.add_argument("--cells", type=Path, required=True)
    model_profile.add_argument("--probes", type=Path, required=True)
    model_profile.add_argument("--aggregation-config", type=Path, required=True)
    model_profile.add_argument(
        "--probe-run-manifest",
        type=Path,
        help="Bind the real inference run, responses, scorer, and checkpoint to the profile",
    )
    model_profile.add_argument("--output", type=Path, required=True)
    model_profile.set_defaults(handler=_cmd_build_model_profile)

    run_model_probes = subparsers.add_parser(
        "run-model-probes",
        help="Measure the frozen Base Model with independent source-grounded capability probes",
    )
    run_model_probes.add_argument("--catalogue", type=Path, required=True)
    run_model_probes.add_argument("--config", type=Path, required=True)
    run_model_probes.add_argument("--output-dir", type=Path, required=True)
    run_model_probes.set_defaults(handler=_cmd_run_model_probes)

    unit_model_profile = subparsers.add_parser(
        "build-unit-model-profile",
        help="Bind unit-by-cell M0 probe results to a frozen model and population",
    )
    unit_model_profile.add_argument("--metadata", type=Path, required=True)
    unit_model_profile.add_argument("--model-profile", type=Path, required=True)
    unit_model_profile.add_argument("--units", type=Path, required=True)
    unit_model_profile.add_argument("--observations", type=Path, required=True)
    unit_model_profile.add_argument("--output", type=Path, required=True)
    unit_model_profile.set_defaults(handler=_cmd_build_unit_model_profile)

    sparse_unit_profile = subparsers.add_parser(
        "build-sparse-unit-model-profile",
        help="Bind source-grounded probes to semantic rules and build a partial unit-cell M0 profile",
    )
    sparse_unit_profile.add_argument("--catalogue", type=Path, required=True)
    sparse_unit_profile.add_argument("--scored-probes", type=Path, required=True)
    sparse_unit_profile.add_argument("--model-profile", type=Path, required=True)
    sparse_unit_profile.add_argument("--units", type=Path, required=True)
    sparse_unit_profile.add_argument("--artifact-root", type=Path, required=True)
    sparse_unit_profile.add_argument("--aggregation-config", type=Path, required=True)
    sparse_unit_profile.add_argument("--profile-id", required=True)
    sparse_unit_profile.add_argument("--probe-set-id", required=True)
    sparse_unit_profile.add_argument("--output-dir", type=Path, required=True)
    sparse_unit_profile.set_defaults(handler=_cmd_build_sparse_unit_model_profile)

    prior_manipulation = subparsers.add_parser(
        "analyze-prior-manipulation",
        help="Compare prior-aware and prior-agnostic sample membership before A8 training",
    )
    prior_manipulation.add_argument("--units", type=Path, required=True)
    prior_manipulation.add_argument("--model-profile", type=Path, required=True)
    prior_manipulation.add_argument("--unit-model-profile", type=Path, required=True)
    prior_manipulation.add_argument("--config", type=Path, required=True)
    prior_manipulation.add_argument("--output", type=Path, required=True)
    prior_manipulation.set_defaults(handler=_cmd_analyse_prior_manipulation)

    prior_holdout = subparsers.add_parser(
        "analyze-prior-holdout",
        help="Build a model-aware sample from design probes and audit it on held-out probes",
    )
    prior_holdout.add_argument("--bindings", type=Path, required=True)
    prior_holdout.add_argument("--scored-probes", type=Path, required=True)
    prior_holdout.add_argument("--model-profile", type=Path, required=True)
    prior_holdout.add_argument("--units", type=Path, required=True)
    prior_holdout.add_argument("--aggregation-config", type=Path, required=True)
    prior_holdout.add_argument("--sampling-config", type=Path, required=True)
    prior_holdout.add_argument("--holdout-config", type=Path, required=True)
    prior_holdout.add_argument("--probe-set-id", required=True)
    prior_holdout.add_argument("--design-profile-id", required=True)
    prior_holdout.add_argument("--audit-profile-id", required=True)
    prior_holdout.add_argument("--output-dir", type=Path, required=True)
    prior_holdout.set_defaults(handler=_cmd_analyse_prior_holdout)

    sample_value = subparsers.add_parser(
        "sample-value-proxy",
        help="Sample one batch while preserving data × model × benchmark strata",
    )
    sample_value.add_argument("--units", type=Path, required=True)
    sample_value.add_argument("--model-profile", type=Path, required=True)
    sample_value.add_argument("--unit-model-profile", type=Path)
    sample_value.add_argument("--config", type=Path, required=True)
    sample_value.add_argument("--batch-id", required=True)
    sample_value.add_argument("--membership-map", type=Path)
    sample_value.add_argument("--artifact-root", type=Path)
    sample_value.add_argument("--output-dir", type=Path, required=True)
    sample_value.set_defaults(handler=_cmd_sample_value_proxy)

    verify_value_sample = subparsers.add_parser(
        "verify-value-sample-manifest",
        help="Replay a value-sample manifest against its complete annotated population frame",
    )
    verify_value_sample.add_argument("--manifest", type=Path, required=True)
    verify_value_sample.add_argument("--population-units", type=Path, required=True)
    verify_value_sample.add_argument("--output", type=Path)
    verify_value_sample.set_defaults(handler=_cmd_verify_value_sample_manifest)

    sweep_value = subparsers.add_parser(
        "sweep-value-fidelity",
        help="Sweep sample fractions and seeds to find stable generation-stage budgets",
    )
    sweep_value.add_argument("--units", type=Path, required=True)
    sweep_value.add_argument("--model-profile", type=Path, required=True)
    sweep_value.add_argument("--unit-model-profile", type=Path)
    sweep_value.add_argument("--config", type=Path, required=True)
    sweep_value.add_argument("--thresholds", type=Path, required=True)
    sweep_value.add_argument("--critical-cell-weights", type=Path)
    sweep_value.add_argument(
        "--batch-id",
        action="append",
        help="Limit the sweep to this batch; repeat for multiple batches",
    )
    sweep_value.add_argument("--output", type=Path, required=True)
    sweep_value.set_defaults(handler=_cmd_sweep_value_fidelity)

    value_profile = subparsers.add_parser(
        "build-value-profile",
        help="Build a full, raw-sample, or weighted-reconstruction conditional value profile",
    )
    value_profile.add_argument("--units", type=Path, required=True)
    value_profile.add_argument("--model-profile", type=Path, required=True)
    value_profile.add_argument("--unit-model-profile", type=Path)
    value_profile.add_argument("--batch-id", required=True)
    value_profile.add_argument("--profile-id", required=True)
    value_profile.add_argument(
        "--kind",
        choices=["full_population", "raw_sample", "weighted_reconstruction"],
        required=True,
    )
    value_profile.add_argument("--output", type=Path, required=True)
    value_profile.set_defaults(handler=_cmd_build_value_profile)

    build_certificate = subparsers.add_parser(
        "build-value-certificate",
        help="Build and replay-verify a value-profile fidelity certificate",
    )
    build_certificate.add_argument("--reference-profile", type=Path, required=True)
    build_certificate.add_argument("--raw-sample-profile", type=Path, required=True)
    build_certificate.add_argument("--reconstructed-profile", type=Path, required=True)
    build_certificate.add_argument("--thresholds", type=Path, required=True)
    build_certificate.add_argument("--critical-cell-weights", type=Path)
    build_certificate.add_argument("--sampling-manifest", type=Path)
    build_certificate.add_argument("--sampling-population-units", type=Path)
    build_certificate.add_argument("--certificate-id", required=True)
    build_certificate.add_argument("--output", type=Path, required=True)
    build_certificate.set_defaults(handler=_cmd_build_value_certificate)

    verify_certificate = subparsers.add_parser(
        "verify-value-certificate",
        help="Replay a saved fidelity certificate against its three input profiles",
    )
    verify_certificate.add_argument("--certificate", type=Path, required=True)
    verify_certificate.add_argument("--reference-profile", type=Path, required=True)
    verify_certificate.add_argument("--raw-sample-profile", type=Path, required=True)
    verify_certificate.add_argument("--reconstructed-profile", type=Path, required=True)
    verify_certificate.add_argument("--sampling-manifest", type=Path)
    verify_certificate.add_argument("--sampling-population-units", type=Path)
    verify_certificate.set_defaults(handler=_cmd_verify_value_certificate)

    select = subparsers.add_parser("select", help="Select an indivisible, token-budgeted proxy set")
    select.add_argument("--units", type=Path, required=True)
    select.add_argument("--obligations", type=Path, required=True)
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(handler=_cmd_select)

    select_ablation = subparsers.add_parser(
        "select-ablation-fraction",
        help="Run a selector at a frozen fraction of one batch's eligible candidate tokens",
    )
    select_ablation.add_argument("--units", type=Path, required=True)
    select_ablation.add_argument("--obligations", type=Path, required=True)
    select_ablation.add_argument("--config", type=Path, required=True)
    select_ablation.add_argument("--budget-fraction", type=float, required=True)
    select_ablation.add_argument(
        "--allow-full-candidate-pool",
        action="store_false",
        dest="require_strict_subset",
        help="Allow the 100%% same-generator full-candidate comparator; strict subsets are the default",
    )
    select_ablation.set_defaults(require_strict_subset=True)
    select_ablation.add_argument("--minimum-selected-groups", type=int, default=1)
    select_ablation.add_argument("--maximum-selected-group-fraction", type=float)
    select_ablation.add_argument("--output", type=Path, required=True)
    select_ablation.set_defaults(handler=_cmd_select_ablation_fraction)

    production = subparsers.add_parser(
        "validate-production", help="Check that full traversal recorded every eligible source unit"
    )
    production.add_argument("--eligible-units", type=Path, required=True)
    production.add_argument("--dispositions", type=Path, required=True)
    production.add_argument("--samples", type=Path, required=True)
    production.add_argument("--batch-id")
    production.add_argument("--batch-content-sha256")
    production.add_argument("--output", type=Path)
    production.set_defaults(handler=_cmd_validate_production)

    catalog = subparsers.add_parser("catalog", help="Show supported external-tool artifact contracts")
    catalog.set_defaults(handler=lambda args: (_print(adapter_catalog()) or 0))
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
