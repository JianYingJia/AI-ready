"""Build an auditable matrix of the model-training trials an experiment owes.

The planner deliberately stops at identities and expected outputs.  It does not
invoke a generator, trainer, or evaluator; external adapters can consume this
manifest and return strict experiment-trial records with artifact hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.core.schema import stable_id
from ai_ready.valuation.contracts import canonical_digest, contract_digest, suite_digest
from ai_ready.valuation.schema import BatchSuite, ExperimentTrialScore, GenerationRecipe, ProductionContract


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalized


class ExperimentArm(_StrictModel):
    """One method/budget condition registered before scores are observed."""

    arm_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    role: Literal["target", "baseline", "reference"]
    arm: Literal["proxy", "reference"]
    input_scope: Literal["selected_subset", "full_batch"]
    method: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    generation_recipe_sha256: str
    budget_id: str = Field(min_length=1)
    selection_token_budget: int | None = Field(default=None, gt=0)
    max_effective_train_tokens: int = Field(gt=0)
    selection_seed: int | None = None
    selection_config_uri: str | None = None
    selection_config_sha256: str | None = None

    @field_validator("selection_config_sha256", "generation_recipe_sha256")
    @classmethod
    def validate_selection_hash(cls, value: str | None, info) -> str | None:
        return None if value is None else _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_role(self):
        if self.role == "reference":
            if self.arm != "reference" or self.input_scope != "full_batch":
                raise ValueError("The reference arm must consume the full batch")
            if self.selection_seed is not None:
                raise ValueError("The full reference arm cannot have a selection seed")
            if self.selection_config_sha256 is not None:
                raise ValueError("The full reference arm cannot have a selection config")
            if self.selection_config_uri is not None:
                raise ValueError("The full reference arm cannot have a selection config URI")
            if self.selection_token_budget is not None:
                raise ValueError("The full reference arm cannot have a source-selection token budget")
        elif self.arm != "proxy" or self.input_scope != "selected_subset":
            raise ValueError("Target and baseline arms must consume selected subsets")
        elif self.selection_token_budget is None:
            raise ValueError("Target and baseline arms must freeze a source-selection token budget")
        elif self.selection_config_sha256 is None:
            raise ValueError("Target and baseline arms must freeze a selection config")
        elif not self.selection_config_uri:
            raise ValueError("Target and baseline arms must register a selection config URI")
        return self


class ExpectedTrial(_StrictModel):
    """The immutable identity and upper token bound for one expected score row."""

    run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    benchmark_sha256: str
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    arm_id: str = Field(min_length=1)
    arm_label: str = Field(min_length=1)
    role: Literal["target", "baseline", "reference"]
    arm: Literal["proxy", "reference"]
    method: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    generation_recipe_sha256: str
    budget_id: str = Field(min_length=1)
    seed: int
    selection_token_budget: int | None = Field(default=None, gt=0)
    max_effective_train_tokens: int = Field(gt=0)
    selection_seed: int | None = None
    selection_config_uri: str | None = None
    selection_config_sha256: str | None = None
    expected_benchmark_item_count: int = Field(gt=0)

    @field_validator(
        "protocol_sha256",
        "suite_sha256",
        "contract_sha256",
        "benchmark_sha256",
        "batch_content_sha256",
        "selection_config_sha256",
        "generation_recipe_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info) -> str | None:
        return None if value is None else _sha256(value, info.field_name)


class ExperimentManifest(_StrictModel):
    """A complete Cartesian product of batches, registered arms, and seeds."""

    schema_version: Literal["valuation-experiment-manifest-v3"] = "valuation-experiment-manifest-v3"
    experiment_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    protocol_sha256: str
    suite_id: str = Field(min_length=1)
    suite_sha256: str
    contract_id: str = Field(min_length=1)
    contract_sha256: str
    benchmark_id: str = Field(min_length=1)
    benchmark_sha256: str
    expected_benchmark_item_count: int = Field(gt=0)
    batch_content_sha256: dict[str, str] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    arms: list[ExperimentArm] = Field(min_length=2)
    expected_trial_count: int = Field(gt=0)
    trials: list[ExpectedTrial] = Field(min_length=1)

    @field_validator("protocol_sha256", "suite_sha256", "contract_sha256", "benchmark_sha256")
    @classmethod
    def validate_hash(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)

    @field_validator("batch_content_sha256")
    @classmethod
    def validate_batch_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return {batch_id: _sha256(digest, f"batch_content_sha256[{batch_id}]") for batch_id, digest in value.items()}

    @model_validator(mode="after")
    def validate_cartesian_product(self):
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Manifest seeds must be unique")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("Manifest arm_id values must be unique")
        registered_arms = {arm.arm_id: arm for arm in self.arms}
        expected_count = len(self.batch_content_sha256) * len(self.arms) * len(self.seeds)
        if self.expected_trial_count != expected_count or len(self.trials) != expected_count:
            raise ValueError(
                "Manifest trials must contain every batch × arm × seed combination exactly once; "
                f"expected={expected_count}, declared={self.expected_trial_count}, observed={len(self.trials)}"
            )
        expected_keys = {
            (batch_id, arm.arm_id, seed)
            for batch_id in self.batch_content_sha256
            for arm in self.arms
            for seed in self.seeds
        }
        observed_keys = {(trial.batch_id, trial.arm_id, trial.seed) for trial in self.trials}
        if observed_keys != expected_keys or len(observed_keys) != len(self.trials):
            raise ValueError("Manifest contains missing or duplicate batch × arm × seed combinations")
        if len({trial.run_id for trial in self.trials}) != len(self.trials):
            raise ValueError("Manifest run_id values must be unique")
        for trial in self.trials:
            arm = registered_arms[trial.arm_id]
            if trial.experiment_id != self.experiment_id:
                raise ValueError("Every trial must use the manifest experiment_id")
            if trial.protocol_sha256 != self.protocol_sha256:
                raise ValueError("Every trial must use the manifest protocol_sha256")
            if trial.suite_sha256 != self.suite_sha256 or trial.contract_sha256 != self.contract_sha256:
                raise ValueError("Every trial must use the manifest suite and contract hashes")
            if trial.benchmark_sha256 != self.benchmark_sha256:
                raise ValueError("Every trial must use the manifest benchmark hash")
            if trial.batch_content_sha256 != self.batch_content_sha256[trial.batch_id]:
                raise ValueError("Every trial must use its registered batch-content hash")
            if (
                trial.role != arm.role
                or trial.arm_label != arm.label
                or trial.arm != arm.arm
                or trial.method != arm.method
                or trial.generator != arm.generator
                or trial.generator_revision != arm.generator_revision
                or trial.generation_recipe_sha256 != arm.generation_recipe_sha256
                or trial.budget_id != arm.budget_id
                or trial.selection_token_budget != arm.selection_token_budget
                or trial.max_effective_train_tokens != arm.max_effective_train_tokens
                or trial.selection_seed != arm.selection_seed
                or trial.selection_config_uri != arm.selection_config_uri
                or trial.selection_config_sha256 != arm.selection_config_sha256
                or trial.expected_benchmark_item_count != self.expected_benchmark_item_count
            ):
                raise ValueError("Every trial must use its registered arm configuration")
        return self

    def write_json(self, path: str | Path) -> None:
        """Write once, refusing to replace a previously reviewed manifest."""

        write_experiment_manifest(self, path)


class ManifestAudit(_StrictModel):
    """Completeness report for score rows returned by external runners."""

    complete: bool
    expected_count: int
    observed_count: int
    matched_count: int
    missing_run_ids: list[str] = Field(default_factory=list)
    unexpected_run_ids: list[str] = Field(default_factory=list)
    duplicate_run_ids: list[str] = Field(default_factory=list)
    identity_mismatches: dict[str, list[str]] = Field(default_factory=dict)


_MISSING = object()


def _read_field(value: Any, name: str) -> Any:
    result = value.get(name, _MISSING) if isinstance(value, Mapping) else getattr(value, name, _MISSING)
    if result is _MISSING:
        raise ValueError(f"Experiment protocol is missing {name!r}")
    return result


def _read_optional_field(value: Any, name: str, default: Any = None) -> Any:
    """Read an optional field from mappings or objects without guessing truthiness."""

    result = value.get(name, _MISSING) if isinstance(value, Mapping) else getattr(value, name, _MISSING)
    return default if result is _MISSING else result


def _normalize_arm(
    value: Any,
    role: Literal["target", "baseline", "reference"],
    recipe: GenerationRecipe,
) -> ExperimentArm:
    method = str(_read_field(value, "method")).strip()
    budget_id = str(_read_field(value, "budget_id")).strip()
    try:
        max_tokens = int(_read_field(value, "max_effective_train_tokens"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{role} max_effective_train_tokens must be an integer") from error
    if not method or not budget_id:
        raise ValueError(f"{role} method and budget_id cannot be empty")
    label = str(_read_optional_field(value, "label", f"{role}-{method}-{budget_id}")).strip()
    if not label:
        raise ValueError(f"{role} label cannot be empty")
    raw_selection_seed = _read_optional_field(value, "selection_seed")
    selection_seed = int(raw_selection_seed) if raw_selection_seed is not None else None
    raw_selection_uri = _read_optional_field(value, "selection_config_uri")
    selection_config_uri = str(raw_selection_uri).strip() if raw_selection_uri is not None else None
    if raw_selection_uri is not None and not selection_config_uri:
        raise ValueError(f"{role} selection_config_uri cannot be empty")
    raw_selection_hash = _read_optional_field(value, "selection_config_sha256")
    selection_config_sha256 = (
        _sha256(str(raw_selection_hash), "selection_config_sha256") if raw_selection_hash is not None else None
    )
    arm = "reference" if role == "reference" else "proxy"
    input_scope = "full_batch" if role == "reference" else "selected_subset"
    raw_selection_budget = _read_optional_field(value, "selection_token_budget")
    if role == "reference":
        selection_token_budget = None
        if raw_selection_budget is not None:
            raise ValueError("reference selection_token_budget must be null")
    else:
        # Older programmatic callers used one numeric field for two different
        # token domains. Keep those callers readable, while all formal YAML
        # protocols now state both budgets explicitly.
        try:
            selection_token_budget = int(raw_selection_budget if raw_selection_budget is not None else max_tokens)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{role} selection_token_budget must be an integer") from error
        if selection_token_budget <= 0:
            raise ValueError(f"{role} selection_token_budget must be positive")
    arm_id = stable_id(
        "valuation-experiment-arm-v3",
        role,
        arm,
        label,
        method,
        recipe.generator,
        recipe.revision,
        canonical_digest(recipe),
        budget_id,
        str(selection_token_budget),
        str(max_tokens),
        str(selection_seed),
        str(selection_config_uri),
        str(selection_config_sha256),
    )
    return ExperimentArm(
        arm_id=arm_id,
        label=label,
        role=role,
        arm=arm,
        input_scope=input_scope,
        method=method,
        generator=recipe.generator,
        generator_revision=recipe.revision,
        generation_recipe_sha256=canonical_digest(recipe),
        budget_id=budget_id,
        selection_token_budget=selection_token_budget,
        max_effective_train_tokens=max_tokens,
        selection_seed=selection_seed,
        selection_config_uri=selection_config_uri,
        selection_config_sha256=selection_config_sha256,
    )


def _protocol_arms(protocol: Any, contract: ProductionContract) -> list[ExperimentArm]:
    target = _normalize_arm(_read_field(protocol, "target"), "target", contract.generation.proxy)
    baseline_values = _read_field(protocol, "baselines")
    if isinstance(baseline_values, (str, bytes)) or not isinstance(baseline_values, Sequence):
        raise ValueError("Experiment protocol baselines must be a sequence")
    if not baseline_values:
        raise ValueError("Experiment protocol requires at least one baseline arm")
    baselines = sorted(
        (_normalize_arm(value, "baseline", contract.generation.proxy) for value in baseline_values),
        key=lambda arm: (arm.method, arm.budget_id, arm.selection_token_budget or 0, arm.max_effective_train_tokens),
    )
    reference = _normalize_arm(_read_field(protocol, "reference"), "reference", contract.generation.reference)
    arms = [target, *baselines, reference]

    identities = [(arm.arm, arm.method, arm.budget_id) for arm in arms]
    if len(identities) != len(set(identities)):
        raise ValueError("Each proxy/reference method and budget_id pair must identify one arm")
    token_cap = contract.training.effective_train_tokens
    oversized = [arm.arm_id for arm in arms if arm.max_effective_train_tokens > token_cap]
    if oversized:
        raise ValueError(f"Arm token caps cannot exceed the frozen training cap ({token_cap}): {oversized}")
    if reference.max_effective_train_tokens != token_cap:
        raise ValueError("The full reference arm must use the frozen training token cap")
    if reference.method != contract.generation.reference.generator:
        raise ValueError("The full reference method must match the production contract reference generator")
    return arms


def build_experiment_manifest(
    suite: BatchSuite,
    contract: ProductionContract,
    protocol: Any,
) -> ExperimentManifest:
    """Expand a frozen protocol into every expected batch/arm/seed trial.

    ``protocol`` is intentionally duck typed.  It may be a Pydantic model, a
    dataclass, or a mapping with ``target``, ``baselines`` and ``reference``;
    each arm must expose ``method``, ``budget_id`` and
    ``max_effective_train_tokens``. Proxy arms should also expose
    ``selection_token_budget``; legacy in-memory callers fall back to the
    training-token cap, but the two quantities remain distinct in manifests.
    """

    if suite.benchmark_id != contract.benchmark.benchmark_id:
        raise ValueError("Batch suite and production contract use different benchmark IDs")
    if suite.capability_blueprint_sha256 != contract.benchmark.capability_blueprint_sha256:
        raise ValueError("Batch suite and production contract use different capability blueprints")
    if contract.benchmark.primary_metric.lower() != "accuracy":
        raise ValueError("The strict experiment artifact contract currently supports accuracy benchmarks only")
    batch_hashes: dict[str, str] = {}
    for batch in suite.batches:
        if batch.content_sha256 is None:
            raise ValueError(f"Batch {batch.batch_id} has no content_sha256; run freeze-batches before planning")
        batch_hashes[batch.batch_id] = _sha256(batch.content_sha256, f"batch {batch.batch_id} content_sha256")

    benchmark_sha = _sha256(contract.benchmark.test_manifest_sha256, "benchmark test_manifest_sha256")
    suite_sha = _sha256(suite_digest(suite), "suite_sha256")
    contract_sha = _sha256(contract_digest(contract), "contract_sha256")
    arms = _protocol_arms(protocol, contract)
    # Hash the complete preregistration, including decision gates.  Hashing only
    # the arm matrix would allow thresholds to change after scores were seen
    # while retaining the same experiment identity.
    if hasattr(protocol, "model_dump"):
        protocol_payload = protocol.model_dump(mode="json")
    elif isinstance(protocol, Mapping):
        protocol_payload = dict(protocol)
    else:
        protocol_payload = {
            "target": _read_field(protocol, "target"),
            "baselines": list(_read_field(protocol, "baselines")),
            "reference": _read_field(protocol, "reference"),
        }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    protocol_id = str(_read_optional_field(protocol, "protocol_id", "unregistered-protocol")).strip()
    if not protocol_id:
        raise ValueError("Experiment protocol_id cannot be empty")
    try:
        expected_benchmark_item_count = int(_read_field(protocol, "expected_benchmark_item_count"))
    except (TypeError, ValueError) as error:
        raise ValueError("expected_benchmark_item_count must be a positive integer") from error
    if expected_benchmark_item_count <= 0:
        raise ValueError("expected_benchmark_item_count must be a positive integer")
    experiment_id = stable_id(
        "valuation-experiment-v2",
        suite_sha,
        contract_sha,
        benchmark_sha,
        protocol_sha,
    )
    trials: list[ExpectedTrial] = []
    for batch in suite.batches:
        batch_sha = batch_hashes[batch.batch_id]
        for arm in arms:
            for seed in contract.training.seeds:
                run_id = stable_id(
                    "valuation-trial-v2",
                    experiment_id,
                    suite_sha,
                    contract_sha,
                    benchmark_sha,
                    batch.batch_id,
                    batch_sha,
                    arm.arm_id,
                    str(seed),
                )
                trials.append(
                    ExpectedTrial(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        protocol_sha256=protocol_sha,
                        suite_sha256=suite_sha,
                        contract_sha256=contract_sha,
                        benchmark_sha256=benchmark_sha,
                        batch_id=batch.batch_id,
                        batch_content_sha256=batch_sha,
                        arm_id=arm.arm_id,
                        arm_label=arm.label,
                        role=arm.role,
                        arm=arm.arm,
                        method=arm.method,
                        generator=arm.generator,
                        generator_revision=arm.generator_revision,
                        generation_recipe_sha256=arm.generation_recipe_sha256,
                        budget_id=arm.budget_id,
                        seed=seed,
                        selection_token_budget=arm.selection_token_budget,
                        max_effective_train_tokens=arm.max_effective_train_tokens,
                        selection_seed=arm.selection_seed,
                        selection_config_uri=arm.selection_config_uri,
                        selection_config_sha256=arm.selection_config_sha256,
                        expected_benchmark_item_count=expected_benchmark_item_count,
                    )
                )
    return ExperimentManifest(
        experiment_id=experiment_id,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha,
        suite_id=suite.suite_id,
        suite_sha256=suite_sha,
        contract_id=contract.contract_id,
        contract_sha256=contract_sha,
        benchmark_id=contract.benchmark.benchmark_id,
        benchmark_sha256=benchmark_sha,
        expected_benchmark_item_count=expected_benchmark_item_count,
        batch_content_sha256=batch_hashes,
        seeds=list(contract.training.seeds),
        arms=arms,
        expected_trial_count=len(trials),
        trials=trials,
    )


def write_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> None:
    """Create a JSON manifest without ever replacing an existing file."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output_path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def audit_experiment_manifest(
    manifest: ExperimentManifest,
    observations: Sequence[ExperimentTrialScore],
) -> ManifestAudit:
    """Report missing, duplicate, unexpected, and identity-inconsistent scores."""

    expected = {trial.run_id: trial for trial in manifest.trials}
    counts = Counter(row.run_id for row in observations)
    observed_ids = set(counts)
    missing = sorted(set(expected) - observed_ids)
    unexpected = sorted(observed_ids - set(expected))
    duplicates = sorted(run_id for run_id, count in counts.items() if count > 1)
    mismatches: dict[str, list[str]] = {}
    fields = (
        "suite_sha256",
        "batch_id",
        "batch_content_sha256",
        "arm",
        "method",
        "generator",
        "generator_revision",
        "generation_recipe_sha256",
        "budget_id",
        "seed",
        "benchmark_sha256",
        "contract_sha256",
        "selection_seed",
        "selection_config_sha256",
    )
    for row in observations:
        planned = expected.get(row.run_id)
        if planned is None:
            continue
        errors = [name for name in fields if getattr(row, name) != getattr(planned, name)]
        if row.effective_train_tokens > planned.max_effective_train_tokens:
            errors.append("effective_train_tokens")
        if row.evaluated_item_count != manifest.expected_benchmark_item_count:
            errors.append("evaluated_item_count")
        if errors:
            mismatches[row.run_id] = sorted(set(errors))
    # Selection, full traversal, and accepted SFT are prepared once per
    # batch/arm, then reused across training seeds.  Allowing these hashes to
    # drift would confound data variance with training-seed variance.
    by_batch_arm: dict[tuple[str, str, str, str, int | None], list[ExperimentTrialScore]] = {}
    for row in observations:
        key = (row.batch_id, row.arm, row.method, row.budget_id, row.selection_seed)
        by_batch_arm.setdefault(key, []).append(row)
    for group in by_batch_arm.values():
        for field in (
            "generation_manifest_sha256",
            "selection_manifest_sha256",
            "full_traversal_report_sha256",
            "accepted_sft_sha256",
        ):
            if len({getattr(row, field) for row in group}) > 1:
                for row in group:
                    mismatches.setdefault(row.run_id, []).append(field)
    mismatches = {run_id: sorted(set(fields)) for run_id, fields in mismatches.items()}
    matched = sum(run_id in expected and run_id not in mismatches for run_id in observed_ids)
    complete = not missing and not unexpected and not duplicates and not mismatches
    return ManifestAudit(
        complete=complete,
        expected_count=manifest.expected_trial_count,
        observed_count=len(observations),
        matched_count=matched,
        missing_run_ids=missing,
        unexpected_run_ids=unexpected,
        duplicate_run_ids=duplicates,
        identity_mismatches=mismatches,
    )
