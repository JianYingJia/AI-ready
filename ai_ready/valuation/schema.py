"""Strict, auditable records for the data-valuation experiment.

The classes here deliberately separate source batches, production contracts,
training observations and derived utilities.  Mixing those identities is the
easiest way to accidentally tune a reference run to a proxy result.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    # JSON manifests cannot represent NaN or infinity portably. Rejecting them
    # at the schema boundary prevents a late write failure or silent metric loss.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _validate_sha256(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


class SourceSpec(StrictModel):
    """A source is registered once and may feed one or more candidate batches."""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    version: str = Field(min_length=1)
    language: str = Field(min_length=1)
    license_status: Literal["allowed", "audit_only", "unknown"]
    allowed_uses: list[str] = Field(default_factory=list)
    content_sha256: str | None = None
    knowledge_cutoff: str | None = None

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "content_sha256")


class BatchFactors(StrictModel):
    """Observable factors used to describe a batch without using test scores."""

    coverage_scope: Literal["broad", "narrow", "natural"]
    target_relevance: Literal["high", "low", "unmeasured"]
    redundancy: Literal["high", "low", "unmeasured"]
    parsing_quality: Literal["clean", "noisy", "natural"] = "natural"


class BatchSpec(StrictModel):
    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    kind: Literal["factorial", "natural_check"]
    split: Literal["development", "calibration", "evaluation"]
    input_uri: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    lineage_group: str = Field(min_length=1)
    content_sha256: str | None = None
    factors: BatchFactors
    paired_with: list[str] = Field(default_factory=list)
    construction_notes: list[str] = Field(default_factory=list)

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "content_sha256")

    @field_validator("source_ids", "paired_with")
    @classmethod
    def validate_unique_references(cls, value: list[str], info) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError(f"{info.field_name} cannot contain empty IDs")
        if len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must contain unique IDs")
        return value


class BatchSuite(StrictModel):
    schema_version: Literal["batch-suite-v1"] = "batch-suite-v1"
    suite_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    intended_claim: Literal["feasibility_and_method_screening", "confirmatory"]
    statistical_unit: Literal["batch"] = "batch"
    capability_blueprint_uri: str = Field(min_length=1)
    capability_blueprint_sha256: str
    capability_blueprint: list[str] = Field(min_length=1)
    sources: list[SourceSpec] = Field(min_length=1)
    batches: list[BatchSpec] = Field(min_length=8, max_length=1000)

    @field_validator("capability_blueprint_sha256")
    @classmethod
    def validate_blueprint_hash(cls, value: str) -> str:
        return _validate_sha256(value, "capability_blueprint_sha256") or ""

    @model_validator(mode="after")
    def validate_identity_graph(self):
        if any(not capability_id.strip() for capability_id in self.capability_blueprint):
            raise ValueError("capability_blueprint cannot contain empty IDs")
        if len(self.capability_blueprint) != len(set(self.capability_blueprint)):
            raise ValueError("capability_blueprint IDs must be unique")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("batch_id values must be unique")
        known_sources = set(source_ids)
        known_batches = set(batch_ids)
        registered_batches = {batch.batch_id: batch for batch in self.batches}
        for batch in self.batches:
            if unknown := set(batch.source_ids) - known_sources:
                raise ValueError(f"Batch {batch.batch_id} references unknown sources: {sorted(unknown)}")
            if batch.batch_id in batch.paired_with:
                raise ValueError(f"Batch {batch.batch_id} cannot be paired with itself")
            if unknown := set(batch.paired_with) - known_batches:
                raise ValueError(f"Batch {batch.batch_id} references unknown paired batches: {sorted(unknown)}")
            for paired_id in batch.paired_with:
                paired = registered_batches[paired_id]
                if batch.batch_id not in paired.paired_with:
                    raise ValueError(f"Batch pairing must be reciprocal: {batch.batch_id} and {paired_id}")
                if batch.lineage_group != paired.lineage_group:
                    raise ValueError(f"Paired batches must share a lineage group: {batch.batch_id} and {paired_id}")
        return self


class ReferencePolicy(StrictModel):
    traversal: Literal["full", "pre_registered_stratified"] = "full"
    uses_proxy_selector: bool = False
    run_all_batches: bool = True
    low_yield_policy: Literal["record_and_continue", "fail_batch"] = "record_and_continue"
    allow_sample_duplication: bool = False
    allow_batch_specific_quality_thresholds: bool = False

    @model_validator(mode="after")
    def prevent_circular_reference(self):
        # 高预算参考不能调用待验证的覆盖选择器；否则只会证明同一算法在两个预算下自洽。
        if self.uses_proxy_selector:
            raise ValueError("High-budget reference cannot use the proxy selector")
        if not self.run_all_batches:
            raise ValueError("High-budget reference must run all registered evaluation batches")
        # 低有效产率本身是批次价值的一部分，不能复制样本或单独降低质量门来填满预算。
        if self.allow_sample_duplication:
            raise ValueError("Reference production cannot duplicate samples to hide low yield")
        if self.allow_batch_specific_quality_thresholds:
            raise ValueError("Reference production cannot lower quality thresholds for individual batches")
        return self


class GenerationRecipe(StrictModel):
    """Frozen provenance for one independently generated SFT branch."""

    generator: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    implementation_sha256: str | None = None
    teacher_model: str = Field(min_length=1)
    teacher_revision: str = Field(min_length=1)
    prompt_sha256: str
    quality_gate_sha256: str
    temperature: float = Field(ge=0, le=2)
    max_retries: int = Field(ge=1, le=20)
    task_quotas: dict[str, int] = Field(min_length=1)
    preserve_source_evidence: bool = True

    @field_validator("implementation_sha256", "prompt_sha256", "quality_gate_sha256")
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_generation_recipe(self):
        if any(value <= 0 for value in self.task_quotas.values()):
            raise ValueError("All task quotas must be positive")
        if not self.preserve_source_evidence:
            raise ValueError("Medical SFT generation must preserve source evidence")
        return self


class GenerationContract(StrictModel):
    """Independent recipes for the coverage proxy and the operational full reference.

    The proxy is generated only after source-side coverage selection.  The
    reference is generated separately from the complete registered input with
    Fully Open Meditron.  Keeping two recipes in the frozen contract prevents a
    proxy artifact from being passed off as a subset of the reference SFT.
    """

    proxy: GenerationRecipe
    reference: GenerationRecipe

    @model_validator(mode="after")
    def require_independent_branches(self):
        if self.proxy.generator == self.reference.generator:
            raise ValueError("Proxy and reference SFT must use independent generators")
        if self.proxy.generator != "coverage_guided_sft":
            raise ValueError("Proxy SFT must use the coverage_guided_sft generator")
        if self.proxy.implementation_sha256 is None:
            raise ValueError("Proxy SFT must bind the exact coverage generator implementation SHA-256")
        if self.reference.generator != "fully_open_meditron":
            raise ValueError("Reference SFT must use the fully_open_meditron generator")
        return self

    def recipe_for_arm(self, arm: Literal["proxy", "reference"]) -> GenerationRecipe:
        return self.proxy if arm == "proxy" else self.reference


class TrainingContract(StrictModel):
    objective: Literal["sft"] = "sft"
    update_method: Literal["lora", "qlora", "full"] = "lora"
    trainer: str = Field(min_length=1)
    trainer_revision: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    base_model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    chat_template_sha256: str
    effective_train_tokens: int = Field(gt=0)
    response_only_loss: bool = True
    seeds: list[int] = Field(min_length=3)
    hyperparameters: dict[str, Any] = Field(min_length=1)
    background_data_sha256: str | None = None
    background_mix_ratio: float = Field(default=0.0, ge=0, lt=1)

    @field_validator("chat_template_sha256", "background_data_sha256")
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_training_contract(self):
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Training seeds must be unique")
        if self.background_mix_ratio > 0 and not self.background_data_sha256:
            raise ValueError("background_data_sha256 is required when background_mix_ratio is positive")
        return self


class BenchmarkContract(StrictModel):
    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    capability_blueprint_sha256: str
    test_manifest_sha256: str
    evaluation_engine: str = Field(min_length=1)
    evaluation_engine_revision: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    higher_is_better: bool = True
    generation_can_access_test_items: bool = False
    decoding: dict[str, Any] = Field(min_length=1)

    @field_validator("capability_blueprint_sha256", "test_manifest_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @model_validator(mode="after")
    def enforce_test_lock(self):
        if self.generation_can_access_test_items:
            raise ValueError("The locked test items cannot be exposed to generation or selection")
        return self


class BudgetContract(StrictModel):
    value_target: Literal["fixed_budget", "full_package"] = "fixed_budget"
    generation_token_cap: int = Field(gt=0)
    training_token_cap: int = Field(gt=0)
    cost_cap: float | None = Field(default=None, gt=0)
    stop_rule: str = Field(min_length=1)


class ToolBinding(StrictModel):
    """Pinned external component metadata; the project does not vendor these tools."""

    stage: Literal["preprocess", "generate", "quality_gate", "train", "evaluate"]
    adapter: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    config_sha256: str
    scope: Literal["batch", "trial"]
    argv: list[str] = Field(default_factory=list)

    @field_validator("config_sha256")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        return _validate_sha256(value, "config_sha256") or ""

    @field_validator("revision")
    @classmethod
    def reject_moving_revisions(cls, value: str) -> str:
        if value.strip().lower() in {"main", "master", "head", "latest"}:
            raise ValueError("External tool revisions must be immutable tags or commit hashes")
        return value


class ProductionContract(StrictModel):
    schema_version: Literal["production-contract-v2"] = "production-contract-v2"
    contract_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    reference_policy: ReferencePolicy
    generation: GenerationContract
    training: TrainingContract
    benchmark: BenchmarkContract
    budget: BudgetContract
    tools: list[ToolBinding] = Field(min_length=5)

    @model_validator(mode="after")
    def validate_tool_chain(self):
        stages = [tool.stage for tool in self.tools]
        ordered_stages = ["preprocess", "generate", "quality_gate", "train", "evaluate"]
        required = set(ordered_stages)
        if missing := required - set(stages):
            raise ValueError(f"Production contract is missing tool stages: {sorted(missing)}")
        if len(stages) != len(set(stages)):
            raise ValueError("Configure one primary tool per production stage")
        if stages != ordered_stages:
            raise ValueError(f"Production tools must follow dependency order: {ordered_stages}")
        expected_scopes = {
            "preprocess": "batch",
            "generate": "batch",
            "quality_gate": "batch",
            "train": "trial",
            "evaluate": "trial",
        }
        for tool in self.tools:
            if tool.scope != expected_scopes[tool.stage]:
                raise ValueError(f"Production stage {tool.stage} must use {expected_scopes[tool.stage]} scope")
        reference_generate = next(tool for tool in self.tools if tool.stage == "generate")
        if reference_generate.adapter != self.generation.reference.generator:
            raise ValueError("Generate tool must bind the frozen reference generator")
        if reference_generate.revision != self.generation.reference.revision:
            raise ValueError("Generate tool revision must match the frozen reference recipe")
        if self.training.effective_train_tokens != self.budget.training_token_cap:
            raise ValueError("Training and budget token caps must match")
        return self


class ValidationReport(StrictModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class PlannedStage(StrictModel):
    stage: Literal["preprocess", "generate", "quality_gate", "train", "evaluate"]
    adapter: str
    scope: Literal["batch", "trial"]
    batch_id: str
    suite_sha256: str
    batch_content_sha256: str
    contract_sha256: str
    seed: int | None = None
    run_id: str
    run_dir: str
    argv: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)

    @field_validator("suite_sha256", "batch_content_sha256", "contract_sha256")
    @classmethod
    def validate_context_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""


class ReferencePlan(StrictModel):
    schema_version: Literal["reference-plan-v1"] = "reference-plan-v1"
    suite_id: str
    suite_sha256: str
    contract_id: str
    contract_sha256: str
    batch_ids: list[str]
    batch_content_sha256: dict[str, str]
    seeds: list[int]
    stages: list[PlannedStage]

    @field_validator("suite_sha256", "contract_sha256")
    @classmethod
    def validate_plan_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @model_validator(mode="after")
    def validate_plan_identity(self):
        if len(self.batch_ids) != len(set(self.batch_ids)):
            raise ValueError("Reference plan batch_ids must be unique")
        if set(self.batch_content_sha256) != set(self.batch_ids):
            raise ValueError("Reference plan batch-content hashes must match batch_ids exactly")
        self.batch_content_sha256 = {
            batch_id: _validate_sha256(digest, f"batch_content_sha256[{batch_id}]") or ""
            for batch_id, digest in sorted(self.batch_content_sha256.items())
        }
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Reference plan seeds must be unique")
        if any(stage.batch_id not in set(self.batch_ids) for stage in self.stages):
            raise ValueError("Reference plan stages must reference registered batch_ids")
        if any(stage.suite_sha256 != self.suite_sha256 for stage in self.stages):
            raise ValueError("Reference plan stages must use the plan suite_sha256")
        if any(stage.contract_sha256 != self.contract_sha256 for stage in self.stages):
            raise ValueError("Reference plan stages must use the plan contract_sha256")
        if any(stage.batch_content_sha256 != self.batch_content_sha256[stage.batch_id] for stage in self.stages):
            raise ValueError("Reference plan stages must use the registered batch-content hash")
        return self


class TrialScore(StrictModel):
    """One paired training observation for one batch, method and random seed."""

    run_id: str = Field(min_length=1)
    suite_sha256: str
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    arm: Literal["proxy", "reference"]
    method: str = Field(min_length=1)
    budget_id: str = Field(min_length=1)
    seed: int
    benchmark_score: float
    baseline_score: float
    benchmark_sha256: str
    contract_sha256: str
    # A quality gate can legitimately reject every generated example.  Such a
    # registered trial evaluates the unchanged base model and records zero
    # utility; forbidding zero here would hide low-yield batches from analysis.
    effective_train_tokens: int = Field(ge=0)
    cost: float | None = Field(default=None, ge=0)

    @field_validator("suite_sha256", "batch_content_sha256", "benchmark_sha256", "contract_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @field_validator("benchmark_score", "baseline_score")
    @classmethod
    def reject_nonfinite_scores(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Scores must be finite")
        return value

    @property
    def utility(self) -> float:
        return self.benchmark_score - self.baseline_score


class ExperimentTrialScore(TrialScore):
    """A trial score with the artifact lineage required by formal experiment audit.

    The legacy :class:`TrialScore` remains intentionally lightweight for the
    single-method ``metrics`` command.  The multi-arm experiment command uses
    this stricter record so a scalar score cannot be detached from the selected
    subset, accepted SFT data, checkpoint, or locked-item predictions.
    """

    evaluation_metric: Literal["accuracy"] = "accuracy"
    generator: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    generation_recipe_sha256: str
    generation_manifest_sha256: str
    selection_seed: int | None = None
    selection_config_sha256: str | None = None
    selection_manifest_sha256: str | None = None
    full_traversal_report_sha256: str | None = None
    accepted_sft_sha256: str
    checkpoint_manifest_sha256: str
    predictions_sha256: str
    evaluated_item_count: int = Field(gt=0)
    correct_item_count: int = Field(ge=0)
    baseline_correct_item_count: int = Field(ge=0)

    @field_validator(
        "selection_config_sha256",
        "generation_recipe_sha256",
        "generation_manifest_sha256",
        "selection_manifest_sha256",
        "full_traversal_report_sha256",
        "accepted_sft_sha256",
        "checkpoint_manifest_sha256",
        "predictions_sha256",
    )
    @classmethod
    def validate_artifact_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_accuracy_and_lineage(self):
        if self.correct_item_count > self.evaluated_item_count:
            raise ValueError("correct_item_count cannot exceed evaluated_item_count")
        if self.baseline_correct_item_count > self.evaluated_item_count:
            raise ValueError("baseline_correct_item_count cannot exceed evaluated_item_count")
        expected_score = self.correct_item_count / self.evaluated_item_count
        expected_baseline = self.baseline_correct_item_count / self.evaluated_item_count
        if not math.isclose(self.benchmark_score, expected_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("benchmark_score must equal correct_item_count / evaluated_item_count")
        if not math.isclose(self.baseline_score, expected_baseline, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("baseline_score must equal baseline_correct_item_count / evaluated_item_count")
        if self.effective_train_tokens == 0 and (
            self.correct_item_count != self.baseline_correct_item_count
            or not math.isclose(self.benchmark_score, self.baseline_score, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("Zero-token no-op trials must reproduce the registered base-model score")

        if self.arm == "proxy":
            if self.selection_seed is None and self.method in {"random", "stratified_random"}:
                raise ValueError("Random proxy trials require a selection_seed")
            if self.selection_config_sha256 is None or self.selection_manifest_sha256 is None:
                raise ValueError("Proxy trials require selection config and selection manifest hashes")
            if self.full_traversal_report_sha256 is not None:
                raise ValueError("Proxy trials cannot claim a full-traversal report")
        else:
            if any(
                value is not None
                for value in (self.selection_seed, self.selection_config_sha256, self.selection_manifest_sha256)
            ):
                raise ValueError("Reference trials cannot carry proxy-selection metadata")
            if self.full_traversal_report_sha256 is None:
                raise ValueError("Reference trials require a full-traversal report hash")
        return self


class BatchUtility(StrictModel):
    suite_sha256: str
    batch_id: str
    batch_content_sha256: str
    arm: Literal["proxy", "reference"]
    method: str
    budget_id: str
    benchmark_sha256: str
    contract_sha256: str
    mean_utility: float
    std_utility: float
    ci95_low: float
    ci95_high: float
    seeds: list[int] = Field(min_length=1)
    mean_score: float
    mean_baseline_score: float
    mean_effective_train_tokens: float
    mean_cost: float | None = None

    @field_validator("suite_sha256", "batch_content_sha256", "benchmark_sha256", "contract_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @field_validator("seeds")
    @classmethod
    def validate_unique_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("Batch utility seeds must be unique")
        return sorted(value)


class CombinationUtility(StrictModel):
    suite_sha256: str
    batch_ids: list[str] = Field(min_length=1)
    batch_content_sha256: dict[str, str] = Field(min_length=1)
    reference_method: str = Field(min_length=1)
    budget_id: str = Field(min_length=1)
    benchmark_sha256: str
    contract_sha256: str
    seeds: list[int] = Field(min_length=1)
    mean_utility: float

    @field_validator("batch_ids")
    @classmethod
    def canonicalize_batch_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Combination batch_ids must be unique")
        return sorted(value)

    @field_validator("suite_sha256", "benchmark_sha256", "contract_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @field_validator("seeds")
    @classmethod
    def validate_unique_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("Combination utility seeds must be unique")
        return sorted(value)

    @field_validator("mean_utility")
    @classmethod
    def validate_finite_utility(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Combination utility must be finite")
        return value

    @model_validator(mode="after")
    def validate_batch_hashes(self):
        if set(self.batch_content_sha256) != set(self.batch_ids):
            raise ValueError("Combination batch-content hashes must match batch_ids exactly")
        self.batch_content_sha256 = {
            batch_id: _validate_sha256(digest, f"batch_content_sha256[{batch_id}]") or ""
            for batch_id, digest in sorted(self.batch_content_sha256.items())
        }
        return self


class FeasibilityArmSpec(StrictModel):
    """One preregistered arm in a pilot feasibility comparison."""

    arm: Literal["proxy", "reference"]
    label: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    method: str = Field(min_length=1)
    budget_id: str = Field(min_length=1)
    # Source-side selector cost. ``ProxyUnit.estimated_tokens`` is measured
    # before SFT generation and therefore is not interchangeable with the
    # assistant-label token count used by ``max_effective_train_tokens``.
    selection_token_budget: int | None = Field(default=None, gt=0)
    max_effective_train_tokens: int = Field(gt=0)
    # This is the seed used to construct a random/stratified sample. It is
    # deliberately distinct from TrialScore.seed, which belongs to training.
    selection_seed: int | None = None
    selection_config_uri: str | None = Field(default=None, min_length=1)
    selection_config_sha256: str | None = None

    @field_validator("selection_config_sha256")
    @classmethod
    def validate_selection_config_hash(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "selection_config_sha256")

    @model_validator(mode="after")
    def validate_budget_roles(self):
        if self.arm == "reference":
            if self.selection_token_budget is not None:
                raise ValueError("The full-data reference cannot have a source-selection token budget")
        elif self.selection_token_budget is None:
            # Preserve compatibility for callers that construct an older
            # in-memory protocol while ensuring every normalized proxy arm has
            # an explicit selector budget in the resulting manifest.
            self.selection_token_budget = self.max_effective_train_tokens
        return self


class ReferenceIdentifiabilityCriteria(StrictModel):
    """Preregistered gates for deciding whether the full-data reference separates batches."""

    min_utility_range: float = Field(ge=0)
    min_between_batch_sd: float = Field(ge=0)
    max_within_to_between_sd_ratio: float = Field(ge=0)
    min_pairwise_seed_rank_spearman_median: float = Field(ge=-1, le=1)
    positive_margin: float = Field(ge=0)
    min_clearly_positive_batch_count: int = Field(ge=0)
    ci_separation_margin: float = Field(ge=0)
    min_distinguishable_ci_pair_fraction: float = Field(ge=0, le=1)


class TargetDecisionCriteria(StrictModel):
    """Absolute and baseline-relative gates for the target proxy arm."""

    min_spearman: float = Field(ge=-1, le=1)
    min_kendall_tau_b: float = Field(ge=-1, le=1)
    min_pairwise_order_accuracy: float = Field(ge=0, le=1)
    max_top1_regret: float = Field(ge=0)
    # Positive values demand an advantage; negative values encode a
    # preregistered non-inferiority margin.
    min_spearman_advantage_over_each_baseline: float = Field(ge=-2, le=2)
    max_top1_regret_excess_over_each_baseline: float = Field(ge=0)
    require_leave_one_seed_out_stability: bool = True


class FeasibilityProtocol(StrictModel):
    """Frozen, YAML-serializable protocol for one descriptive pilot."""

    schema_version: Literal["pilot-feasibility-protocol-v2"] = "pilot-feasibility-protocol-v2"
    protocol_id: str = Field(min_length=1)
    expected_benchmark_item_count: int = Field(gt=0)
    inference_scope: Literal["descriptive_pilot_no_confirmatory_inference"] = (
        "descriptive_pilot_no_confirmatory_inference"
    )
    target: FeasibilityArmSpec
    baselines: list[FeasibilityArmSpec] = Field(min_length=1)
    reference: FeasibilityArmSpec
    reference_identifiability: ReferenceIdentifiabilityCriteria
    target_decision: TargetDecisionCriteria
    indifference_margin: float = Field(ge=0)
    minimum_independent_lineage_count_for_warning: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_arm_roles_and_identities(self):
        if self.target.arm != "proxy":
            raise ValueError("The target feasibility arm must be a proxy arm")
        if any(arm.arm != "proxy" for arm in self.baselines):
            raise ValueError("Every baseline feasibility arm must be a proxy arm")
        if self.reference.arm != "reference":
            raise ValueError("The feasibility reference must be a reference arm")
        if self.reference.selection_seed is not None:
            raise ValueError("The full-data reference cannot have a selection_seed")
        if self.reference.selection_config_sha256 is not None:
            raise ValueError("The full-data reference cannot have a selection_config_sha256")
        if self.reference.selection_config_uri is not None:
            raise ValueError("The full-data reference cannot have a selection_config_uri")
        if any(arm.selection_config_sha256 is None for arm in [self.target, *self.baselines]):
            raise ValueError("Every proxy feasibility arm must freeze a selection_config_sha256")
        if any(not arm.selection_config_uri for arm in [self.target, *self.baselines]):
            raise ValueError("Every proxy feasibility arm must register a selection_config_uri")
        for arm in [self.target, *self.baselines]:
            if arm.method in {"random", "stratified_random"} and arm.selection_seed is None:
                raise ValueError(f"Stochastic proxy arm {arm.label} must freeze a selection_seed")

        arms = [self.target, *self.baselines, self.reference]
        labels = [arm.label for arm in arms]
        if len(labels) != len(set(labels)):
            raise ValueError("Feasibility arm labels must be unique")
        identities = [(arm.arm, arm.method, arm.budget_id) for arm in arms]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "Feasibility arm/method/budget identities must be unique; TrialScore does not carry selection_seed, "
                "so repeated selectors must use distinct method IDs"
            )
        proxy_token_caps = {arm.max_effective_train_tokens for arm in [self.target, *self.baselines]}
        if len(proxy_token_caps) != 1:
            raise ValueError("Target and baseline proxy arms must use the same maximum effective training-token budget")
        proxy_selection_budgets = {arm.selection_token_budget for arm in [self.target, *self.baselines]}
        if len(proxy_selection_budgets) != 1:
            raise ValueError("Target and baseline proxy arms must use the same source-selection token budget")
        return self
