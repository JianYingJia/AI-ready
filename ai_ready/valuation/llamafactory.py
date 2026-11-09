"""Prepare and run auditable LLaMAFactory LoRA-SFT jobs on a training server.

The server environment installs LLaMAFactory from a pinned editable checkout
and exposes this project with ``--no-deps``.  The adapter writes immutable data
views, one training YAML per experiment trial, and a job manifest.  It then
invokes ``python -m llamafactory.cli`` without a shell and records hashes for
the resulting LoRA adapter.  The imported package path and dependency versions
must remain identical for all trials.

The token budget is measured after applying the pinned Qwen template and the
same cutoff rule used by LLaMAFactory v0.9.5.  Because ``train_on_prompt`` is
false, ``effective_train_tokens`` means non-masked assistant-label tokens.  We
also retain input and packed token counts so training cost is not confused
with supervised signal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.utils.io import atomic_text, read_jsonl, write_json, write_jsonl
from ai_ready.valuation.contracts import contract_digest, file_sha256
from ai_ready.valuation.experiment import ExpectedTrial, ExperimentManifest
from ai_ready.valuation.quality import SFTSample, UnitDisposition, validate_full_traversal
from ai_ready.valuation.reference_sft import FullyOpenMeditronRunManifest
from ai_ready.valuation.schema import ProductionContract
from ai_ready.valuation.selection import CoverageObligation, ProxySelectionConfig, ProxyUnit, select_proxy

QWEN_TEMPLATE_SPEC = {
    "name": "qwen",
    "format_system": "<|im_start|>system\n{{content}}<|im_end|>\n",
    "format_user": "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n",
    "format_assistant": "{{content}}<|im_end|>\n",
    "default_system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "stop_words": ["<|im_end|>"],
    "replace_eos": True,
}
# Bind a release tag to its complete Git object identity.  A short SHA is
# convenient for display, but it is not a sufficiently strong execution
# boundary for a long-lived experiment bundle.
LLAMAFACTORY_RELEASE_COMMITS = {"v0.9.5": "7af909522a951e3ad9f022ea6f88b6755257eaa5"}
LLAMAFACTORY_REPOSITORY = "https://github.com/hiyouga/LlamaFactory"
PROJECT_CODE_RELATIVE_PATHS = (
    "tools/value_data.py",
    "ai_ready/valuation/cli.py",
    "ai_ready/valuation/llamafactory.py",
    "ai_ready/valuation/schema.py",
    "ai_ready/valuation/contracts.py",
    "ai_ready/valuation/experiment.py",
    "ai_ready/valuation/local_sft.py",
    "ai_ready/valuation/quality.py",
    "ai_ready/valuation/reference_sft.py",
    "ai_ready/valuation/selection.py",
    "ai_ready/utils/io.py",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class ProjectCodeSnapshot(_StrictModel):
    """Content identity for the ai-ready code that prepares and runs a job.

    Git metadata is diagnostic because the project can legitimately live inside
    an untracked parent checkout.  The listed file hashes are the enforced
    boundary and are rechecked immediately before every training invocation.
    """

    schema_version: Literal["ai-ready-project-code-v1"] = "ai-ready-project-code-v1"
    project_root: str = Field(min_length=1)
    files_sha256: dict[str, str]
    git_root: str | None = None
    git_head: str | None = None
    git_remote: str | None = None
    git_dirty: bool | None = None

    @model_validator(mode="after")
    def validate_required_files(self):
        if set(self.files_sha256) != set(PROJECT_CODE_RELATIVE_PATHS):
            missing = sorted(set(PROJECT_CODE_RELATIVE_PATHS) - set(self.files_sha256))
            unexpected = sorted(set(self.files_sha256) - set(PROJECT_CODE_RELATIVE_PATHS))
            raise ValueError(f"Project code snapshot file set is invalid; missing={missing}, unexpected={unexpected}")
        for relative_path, digest in self.files_sha256.items():
            _validate_sha256(digest, f"files_sha256[{relative_path!r}]")
        if not Path(self.project_root).is_absolute():
            raise ValueError("Project code snapshot root must be absolute")
        return self


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_text(payload)


def _selector_replay_mismatch(expected: Any, observed: Any, path: str = "$") -> str | None:
    """Return the first structural replay mismatch, tolerating only float roundoff.

    Selector output can pass through a different NumPy/BLAS build before the
    server replays it. Keys, value types, and list order remain exact; only
    finite floating-point values receive a 1e-12 numerical tolerance.
    """

    if type(expected) is not type(observed):
        return f"{path}: type {type(observed).__name__}, expected {type(expected).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return (
                f"{path}: missing keys={sorted(expected_keys - observed_keys)}, "
                f"extra keys={sorted(observed_keys - expected_keys)}"
            )
        for key in sorted(expected_keys):
            mismatch = _selector_replay_mismatch(expected[key], observed[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: length {len(observed)}, expected {len(expected)}"
        for index, (expected_value, observed_value) in enumerate(zip(expected, observed, strict=True)):
            mismatch = _selector_replay_mismatch(expected_value, observed_value, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, float):
        if not math.isfinite(expected) or not math.isfinite(observed):
            return None if expected == observed else f"{path}: non-finite float mismatch"
        if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
            return f"{path}: {observed!r}, expected {expected!r}"
        return None
    return None if expected == observed else f"{path}: {observed!r}, expected {expected!r}"


def _validate_sha256(value: str | None, field_name: str) -> str | None:
    if value is not None and not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def qwen_template_digest() -> str:
    """Hash the exact v0.9.5 Qwen formatting rules used by token audit."""

    return _canonical_digest(QWEN_TEMPLATE_SPEC)


def server_profile_digest(profile: "LlamaFactoryServerProfile") -> str:
    """Hash profile semantics so Git line-ending conversion cannot change identity."""

    return _canonical_digest(profile.model_dump(mode="json"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not normalized:
        raise ValueError(f"Value cannot be converted to a safe path component: {value!r}")
    return normalized


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


class LlamaFactoryServerProfile(_StrictModel):
    """Hardware/runtime choices shared by every trial in one experiment."""

    schema_version: Literal["llamafactory-server-profile-v1"] = "llamafactory-server-profile-v1"
    llamafactory_revision: str = Field(min_length=1)
    template: Literal["qwen"] = "qwen"
    trust_remote_code: bool = True
    world_size: int = Field(default=1, gt=0)
    per_device_train_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    preprocessing_num_workers: Literal[1] = 1
    preprocessing_batch_size: int = Field(default=1_000_000, gt=0)
    dataloader_num_workers: int = Field(default=4, ge=0)
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    flash_attn: Literal["auto", "fa2", "sdpa", "disabled"] = "auto"
    logging_steps: int = Field(default=1, gt=0)
    save_steps: int = Field(default=50, gt=0)
    save_total_limit: int = Field(default=2, gt=0)
    report_to: Literal["none", "tensorboard", "wandb", "swanlab", "mlflow"] = "none"
    deepspeed_config: str | None = None

    @field_validator("llamafactory_revision")
    @classmethod
    def reject_moving_revision(cls, value: str) -> str:
        if value.strip().lower() in {"main", "master", "head", "latest"}:
            raise ValueError("LLaMAFactory revision must be an immutable tag or commit")
        if value != "v0.9.5":
            raise ValueError("This adapter mirrors LLaMAFactory v0.9.5 preprocessing and supports only v0.9.5")
        return value

    @model_validator(mode="after")
    def validate_precision_and_parallelism(self):
        if self.bf16 == self.fp16:
            raise ValueError("Exactly one of bf16 and fp16 must be enabled")
        if self.deepspeed_config and self.world_size == 1:
            raise ValueError("DeepSpeed is reserved for distributed jobs; omit it for world_size=1")
        return self

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.per_device_train_batch_size * self.gradient_accumulation_steps


class LlamaFactoryHyperparameters(_StrictModel):
    """Strict subset of LLaMAFactory settings that defines this experiment."""

    learning_rate: float = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: float = Field(gt=0)
    lora_dropout: float = Field(ge=0, lt=1)
    lora_target: Literal["all"] = "all"
    max_length: int = Field(gt=1)
    packing: bool
    neat_packing: bool
    num_train_epochs: float
    lr_scheduler_type: str = Field(min_length=1)
    warmup_ratio: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_fixed_budget_semantics(self):
        if self.num_train_epochs != 1.0:
            raise ValueError("The fixed-token experiment requires num_train_epochs=1.0")
        if self.neat_packing and not self.packing:
            raise ValueError("neat_packing=true requires packing=true")
        return self


class SFTArtifactEntry(_StrictModel):
    """Source SFT and lineage artifact for one batch × experiment arm."""

    batch_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)
    arm_label: str = Field(min_length=1)
    source_sft_uri: str = Field(min_length=1)
    generation_manifest_uri: str = Field(min_length=1)
    selection_manifest_uri: str | None = None
    proxy_units_uri: str | None = None
    coverage_obligations_uri: str | None = None
    eligible_units_uri: str | None = None
    upstream_run_manifest_uri: str | None = None
    dispositions_uri: str | None = None
    full_traversal_report_uri: str | None = None

    @model_validator(mode="after")
    def validate_lineage_inputs(self):
        has_selection = self.selection_manifest_uri is not None
        has_traversal = self.full_traversal_report_uri is not None
        if has_selection == has_traversal:
            raise ValueError("Exactly one selection manifest or full-traversal report is required")
        if self.dispositions_uri is None:
            raise ValueError("Every SFT entry requires unit dispositions for production replay")
        proxy_replay_inputs = [self.proxy_units_uri, self.coverage_obligations_uri]
        if has_selection and any(value is None for value in proxy_replay_inputs):
            raise ValueError(
                "Proxy entries require proxy units, coverage obligations, and unit dispositions for replay"
            )
        if has_selection and self.eligible_units_uri is not None:
            raise ValueError("Proxy entries cannot register reference eligible units")
        if has_selection and self.upstream_run_manifest_uri is not None:
            raise ValueError("Proxy entries cannot register an upstream Fully Open Meditron run manifest")
        if has_traversal and any(value is not None for value in proxy_replay_inputs):
            raise ValueError("Reference entries cannot register proxy selector inputs")
        if has_traversal and self.eligible_units_uri is None:
            raise ValueError("Reference entries require eligible units for full-traversal replay")
        if has_traversal and self.upstream_run_manifest_uri is None:
            raise ValueError("Reference entries require the Fully Open Meditron run manifest")
        return self


class SFTArtifactRegistry(_StrictModel):
    schema_version: Literal["llamafactory-sft-registry-v3"] = "llamafactory-sft-registry-v3"
    experiment_id: str = Field(min_length=1)
    entries: list[SFTArtifactEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_batch_arm_entries(self):
        keys = [(entry.batch_id, entry.arm_id) for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("SFT registry contains duplicate batch_id + arm_id entries")
        return self


class SFTGenerationManifest(_StrictModel):
    """Arm-specific proof that SFT was produced by the registered branch."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    schema_version: Literal["sft-generation-manifest-v2"] = "sft-generation-manifest-v2"
    artifact_role: Literal["proxy", "reference"]
    input_scope: Literal["selected_semantic_units", "full_batch"]
    batch_id: str = Field(min_length=1)
    batch_content_sha256: str
    generator: str = Field(min_length=1)
    generator_revision: str = Field(min_length=1)
    implementation_sha256: str | None = None
    generation_recipe_sha256: str
    source_sft_sha256: str
    contract_id: str = Field(min_length=1)
    contract_sha256: str
    selection_manifest_sha256: str | None = None
    full_traversal_report_sha256: str | None = None
    benchmark_items_accessed: Literal[False] = False
    formal_production_contract_used: Literal[True] = True
    selection_before_generation: Literal[True] | None = None
    candidate_pool_complete: Literal[True] | None = None
    candidate_pool_sha256: str | None = None
    max_units_applied: Literal[False] | None = None
    uses_proxy_selector: Literal[False] | None = None
    complete_input_traversal: Literal[True] | None = None
    upstream_run_manifest_sha256: str | None = None

    @field_validator(
        "batch_content_sha256",
        "generation_recipe_sha256",
        "implementation_sha256",
        "source_sft_sha256",
        "contract_sha256",
        "selection_manifest_sha256",
        "full_traversal_report_sha256",
        "candidate_pool_sha256",
        "upstream_run_manifest_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_branch(self):
        if self.artifact_role == "proxy":
            if self.input_scope != "selected_semantic_units" or self.selection_manifest_sha256 is None:
                raise ValueError("Proxy generation must bind selected semantic units and a selection manifest")
            if self.full_traversal_report_sha256 is not None:
                raise ValueError("Proxy generation cannot claim a full-traversal report")
            if (
                self.selection_before_generation is not True
                or self.candidate_pool_complete is not True
                or self.candidate_pool_sha256 is None
                or self.max_units_applied is not False
                or self.implementation_sha256 is None
            ):
                raise ValueError(
                    "Formal proxy generation must prove selection-before-generation over the complete candidate pool"
                )
            if (
                self.uses_proxy_selector is not None
                or self.complete_input_traversal is not None
                or self.upstream_run_manifest_sha256 is not None
            ):
                raise ValueError("Proxy generation cannot claim Fully Open Meditron reference provenance")
        else:
            if self.input_scope != "full_batch" or self.full_traversal_report_sha256 is None:
                raise ValueError("Reference generation must bind the full batch and a full-traversal report")
            if self.selection_manifest_sha256 is not None:
                raise ValueError("Reference generation cannot bind a proxy selection manifest")
            if (
                self.uses_proxy_selector is not False
                or self.complete_input_traversal is not True
                or self.upstream_run_manifest_sha256 is None
            ):
                raise ValueError(
                    "Formal reference generation must bind an upstream run manifest and prove selector-free traversal"
                )
            if any(
                value is not None
                for value in (
                    self.selection_before_generation,
                    self.candidate_pool_complete,
                    self.candidate_pool_sha256,
                    self.max_units_applied,
                    self.implementation_sha256,
                )
            ):
                raise ValueError("Reference generation cannot claim proxy candidate-pool provenance")
        return self


class TrainingDataAudit(_StrictModel):
    schema_version: Literal["llamafactory-training-data-audit-v3"] = "llamafactory-training-data-audit-v3"
    experiment_id: str
    batch_id: str
    arm_id: str
    arm_label: str
    source_sft_uri: str
    source_sft_sha256: str
    generation_manifest_uri: str
    generation_manifest_sha256: str
    generator: str
    generator_revision: str
    generation_recipe_sha256: str
    training_sft_uri: str
    training_sft_sha256: str
    dataset_info_uri: str
    dataset_info_sha256: str
    provenance_uri: str
    provenance_sha256: str
    lineage_artifact_uri: str
    lineage_artifact_sha256: str
    proxy_units_uri: str | None = None
    proxy_units_sha256: str | None = None
    coverage_obligations_uri: str | None = None
    coverage_obligations_sha256: str | None = None
    eligible_units_uri: str | None = None
    eligible_units_sha256: str | None = None
    upstream_run_manifest_uri: str | None = None
    upstream_run_manifest_sha256: str | None = None
    dispositions_uri: str | None = None
    dispositions_sha256: str | None = None
    sample_count: int = Field(ge=0)
    truncated_sample_count: int = Field(ge=0)
    input_token_count: int = Field(ge=0)
    supervised_token_count: int = Field(ge=0)
    effective_train_tokens: int = Field(ge=0)
    packed_token_count: int = Field(ge=0)
    configured_cutoff_len: int = Field(gt=1)
    processing_cutoff_len: int = Field(gt=0)
    preprocessing_num_workers: int = Field(gt=0)
    preprocessing_batch_size: int = Field(gt=0)
    num_train_epochs: float = Field(gt=0)
    tokenizer_name_or_path: str
    tokenizer_revision: str
    template: str
    template_sha256: str

    @field_validator(
        "source_sft_sha256",
        "generation_manifest_sha256",
        "generation_recipe_sha256",
        "training_sft_sha256",
        "dataset_info_sha256",
        "provenance_sha256",
        "lineage_artifact_sha256",
        "proxy_units_sha256",
        "coverage_obligations_sha256",
        "eligible_units_sha256",
        "upstream_run_manifest_sha256",
        "dispositions_sha256",
        "template_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_selector_replay_artifacts(self):
        units_pair = (self.proxy_units_uri, self.proxy_units_sha256)
        obligations_pair = (self.coverage_obligations_uri, self.coverage_obligations_sha256)
        eligible_pair = (self.eligible_units_uri, self.eligible_units_sha256)
        upstream_pair = (self.upstream_run_manifest_uri, self.upstream_run_manifest_sha256)
        dispositions_pair = (self.dispositions_uri, self.dispositions_sha256)
        if (units_pair[0] is None) != (units_pair[1] is None):
            raise ValueError("Proxy-unit URI and hash must be set together")
        if (obligations_pair[0] is None) != (obligations_pair[1] is None):
            raise ValueError("Coverage-obligation URI and hash must be set together")
        if (eligible_pair[0] is None) != (eligible_pair[1] is None):
            raise ValueError("Eligible-unit URI and hash must be set together")
        if (upstream_pair[0] is None) != (upstream_pair[1] is None):
            raise ValueError("Upstream run-manifest URI and hash must be set together")
        if (dispositions_pair[0] is None) != (dispositions_pair[1] is None):
            raise ValueError("Disposition URI and hash must be set together")
        if dispositions_pair[0] is None:
            raise ValueError("Every training-data audit must bundle unit dispositions")
        has_units = units_pair[0] is not None
        has_obligations = obligations_pair[0] is not None
        if has_units != has_obligations:
            raise ValueError("Proxy units and coverage obligations must be bundled together")
        has_eligible = eligible_pair[0] is not None
        if has_units == has_eligible:
            raise ValueError("Audit must bundle exactly one proxy replay input set or reference eligible units")
        if self.generator == "fully_open_meditron" and upstream_pair[0] is None:
            raise ValueError("Fully Open Meditron audits require the copied upstream run manifest")
        if self.generator != "fully_open_meditron" and upstream_pair[0] is not None:
            raise ValueError("Only Fully Open Meditron audits can bind an upstream run manifest")
        return self


class LlamaFactoryJob(_StrictModel):
    schema_version: Literal["llamafactory-job-v2"] = "llamafactory-job-v2"
    job_index: int = Field(ge=0)
    run_id: str
    experiment_id: str
    protocol_sha256: str
    suite_sha256: str
    contract_sha256: str
    benchmark_sha256: str
    batch_id: str
    batch_content_sha256: str
    arm_id: str
    arm_label: str
    role: Literal["target", "baseline", "reference"]
    arm: Literal["proxy", "reference"]
    method: str
    generator: str
    generator_revision: str
    generation_recipe_sha256: str
    budget_id: str
    seed: int
    selection_token_budget: int | None = Field(default=None, gt=0)
    selection_seed: int | None
    selection_config_sha256: str | None
    execution_mode: Literal["train", "no_op"]
    data_audit_uri: str
    data_audit_sha256: str
    generation_manifest_sha256: str
    accepted_sft_sha256: str
    selection_manifest_sha256: str | None
    proxy_units_sha256: str | None = None
    coverage_obligations_sha256: str | None = None
    eligible_units_sha256: str | None = None
    dispositions_sha256: str | None = None
    full_traversal_report_sha256: str | None
    effective_train_tokens: int = Field(ge=0)
    input_token_count: int = Field(ge=0)
    packed_token_count: int = Field(ge=0)
    max_effective_train_tokens: int = Field(gt=0)
    train_config_uri: str | None
    train_config_sha256: str | None
    run_dir: str
    adapter_dir: str | None
    base_model: str
    base_model_revision: str
    llamafactory_root: str
    llamafactory_revision: str
    llamafactory_commit: str
    python_executable: str
    llamafactory_package_uri: str
    runtime_packages: dict[str, str]
    project_code_sha256: str
    world_size: int = Field(gt=0)
    global_batch_size: int = Field(gt=0)
    deepspeed_config_uri: str | None = None
    deepspeed_config_sha256: str | None = None
    argv: list[str]

    @field_validator(
        "protocol_sha256",
        "suite_sha256",
        "contract_sha256",
        "benchmark_sha256",
        "batch_content_sha256",
        "selection_config_sha256",
        "data_audit_sha256",
        "generation_manifest_sha256",
        "generation_recipe_sha256",
        "accepted_sft_sha256",
        "selection_manifest_sha256",
        "proxy_units_sha256",
        "coverage_obligations_sha256",
        "eligible_units_sha256",
        "dispositions_sha256",
        "full_traversal_report_sha256",
        "train_config_sha256",
        "deepspeed_config_sha256",
        "project_code_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info) -> str | None:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_execution_payload(self):
        if self.effective_train_tokens > self.max_effective_train_tokens:
            raise ValueError("Job exceeds its registered supervised-token budget")
        if self.execution_mode == "train":
            if not self.argv or not self.train_config_uri or not self.train_config_sha256 or not self.adapter_dir:
                raise ValueError("Train jobs require argv, config, and adapter directory")
        else:
            if (
                self.effective_train_tokens != 0
                or self.argv
                or self.train_config_uri
                or self.train_config_sha256
                or self.adapter_dir
            ):
                raise ValueError("No-op jobs must have zero tokens and no training command")
        if (self.deepspeed_config_uri is None) != (self.deepspeed_config_sha256 is None):
            raise ValueError("DeepSpeed config URI and hash must be set together")
        if self.arm == "proxy":
            if self.selection_token_budget is None:
                raise ValueError("Proxy jobs require a frozen source-selection token budget")
            if (
                self.selection_manifest_sha256 is None
                or self.proxy_units_sha256 is None
                or self.coverage_obligations_sha256 is None
                or self.dispositions_sha256 is None
            ):
                raise ValueError("Proxy jobs require the selection manifest and every proxy-production input hash")
            if self.full_traversal_report_sha256 is not None:
                raise ValueError("Proxy jobs cannot use a full-traversal report")
            if self.eligible_units_sha256 is not None:
                raise ValueError("Proxy jobs cannot bind reference eligible units")
        else:
            if self.selection_token_budget is not None:
                raise ValueError("Reference jobs cannot have a source-selection token budget")
            if (
                self.full_traversal_report_sha256 is None
                or self.eligible_units_sha256 is None
                or self.dispositions_sha256 is None
            ):
                raise ValueError("Reference jobs require a full-traversal report, eligible units, and dispositions")
            if any(
                value is not None
                for value in (
                    self.selection_manifest_sha256,
                    self.proxy_units_sha256,
                    self.coverage_obligations_sha256,
                )
            ):
                raise ValueError("Reference jobs cannot bind proxy selector artifacts")
        return self


class LlamaFactoryJobManifest(_StrictModel):
    schema_version: Literal["llamafactory-job-manifest-v2"] = "llamafactory-job-manifest-v2"
    experiment_id: str
    experiment_manifest_sha256: str
    contract_sha256: str
    profile_sha256: str
    llamafactory_revision: str
    llamafactory_commit: str
    runtime_packages: dict[str, str]
    project_code: ProjectCodeSnapshot
    project_code_sha256: str
    template_sha256: str
    expected_job_count: int = Field(gt=0)
    jobs: list[LlamaFactoryJob] = Field(min_length=1)

    @field_validator(
        "experiment_manifest_sha256",
        "contract_sha256",
        "profile_sha256",
        "project_code_sha256",
        "template_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name) or ""

    @model_validator(mode="after")
    def validate_complete_job_list(self):
        if len(self.jobs) != self.expected_job_count:
            raise ValueError("LLaMAFactory job count does not match expected_job_count")
        run_ids = [job.run_id for job in self.jobs]
        indexes = [job.job_index for job in self.jobs]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("LLaMAFactory run IDs must be unique")
        if indexes != list(range(len(self.jobs))):
            raise ValueError("LLaMAFactory job indexes must be contiguous and ordered")
        if any(job.experiment_id != self.experiment_id for job in self.jobs):
            raise ValueError("Every job must belong to the manifest experiment")
        if any(job.contract_sha256 != self.contract_sha256 for job in self.jobs):
            raise ValueError("Every job must use the manifest production contract")
        if any(
            job.llamafactory_revision != self.llamafactory_revision
            or job.llamafactory_commit != self.llamafactory_commit
            for job in self.jobs
        ):
            raise ValueError("Every job must use the manifest LLaMAFactory revision and commit")
        runtimes = {(job.python_executable, job.llamafactory_package_uri) for job in self.jobs}
        if len(runtimes) != 1:
            raise ValueError("Every job must use the same pinned Python and LLaMAFactory package")
        if any(job.runtime_packages != self.runtime_packages for job in self.jobs):
            raise ValueError("Every job must use the manifest runtime and CUDA fingerprint")
        if _project_code_digest(self.project_code) != self.project_code_sha256:
            raise ValueError("Project code snapshot does not match project_code_sha256")
        if any(job.project_code_sha256 != self.project_code_sha256 for job in self.jobs):
            raise ValueError("Every job must use the manifest project code snapshot")
        return self


def load_server_profile(path: str | Path) -> LlamaFactoryServerProfile:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Server profile must contain a YAML mapping: {path}")
    return LlamaFactoryServerProfile.model_validate(value)


def load_sft_registry(path: str | Path) -> SFTArtifactRegistry:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"SFT registry must contain a YAML mapping: {path}")
    return SFTArtifactRegistry.model_validate(value)


def load_job_manifest(path: str | Path) -> LlamaFactoryJobManifest:
    path = Path(path).resolve()
    plan = LlamaFactoryJobManifest.model_validate(_load_json_object(path))
    ready_path = path.parent / "BUNDLE_READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"Training bundle has no BUNDLE_READY.json receipt: {path.parent}")
    ready = _load_json_object(ready_path)
    if ready.get("schema_version") != "llamafactory-bundle-ready-v1":
        raise ValueError(f"Training bundle has an unsupported ready receipt: {ready_path}")
    if ready.get("experiment_id") != plan.experiment_id or ready.get("job_manifest_sha256") != file_sha256(path):
        raise ValueError(f"Training bundle receipt does not match its job manifest: {path.parent}")
    if ready.get("project_code_sha256") != plan.project_code_sha256:
        raise ValueError(f"Training bundle receipt does not match its project code snapshot: {path.parent}")
    frozen_artifacts = {
        "job_jsonl_sha256": path.parent / "llamafactory_jobs.jsonl",
        "evaluation_requests_sha256": path.parent / "evaluation_requests.jsonl",
        "slurm_script_sha256": path.parent / "run_slurm_array.sh",
    }
    for field_name, artifact_path in frozen_artifacts.items():
        if not artifact_path.is_file() or ready.get(field_name) != file_sha256(artifact_path):
            raise ValueError(f"Training bundle artifact changed after preparation: {artifact_path}")
    return plan


def write_sft_registry_template(manifest: ExperimentManifest, path: str | Path, sft_root: str | Path) -> None:
    """Write the 84 expected batch × arm entries without claiming files exist."""

    output_path = Path(path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Registry output already exists: {output_path}")
    # Store an absolute root.  Registry-relative paths are supported when a
    # human writes them deliberately, but a generated template must not change
    # meaning merely because it is moved to another directory.
    root = Path(sft_root).resolve()
    entries = []
    for batch_id in sorted(manifest.batch_content_sha256):
        for arm in manifest.arms:
            # Each proxy command generates every batch for one selection
            # method, so the on-disk layout is arm/batch.  The reference
            # importer writes one batch at a time into operational_full/batch.
            # Matching those native outputs avoids a manual copy/relabel step.
            directory = root / arm.label / batch_id
            entry = {
                "batch_id": batch_id,
                "arm_id": arm.arm_id,
                "arm_label": arm.label,
                "source_sft_uri": str(directory / "accepted_sft.jsonl"),
                "generation_manifest_uri": str(directory / "generation_manifest.json"),
                "selection_manifest_uri": str(directory / "selection_manifest.json") if arm.arm == "proxy" else None,
                "proxy_units_uri": str(directory / "proxy_units.jsonl") if arm.arm == "proxy" else None,
                "coverage_obligations_uri": (
                    str(directory / "coverage_obligations.jsonl") if arm.arm == "proxy" else None
                ),
                "eligible_units_uri": str(directory / "eligible_units.jsonl") if arm.arm == "reference" else None,
                "upstream_run_manifest_uri": (
                    str(directory / "fully_open_meditron_run_manifest.json") if arm.arm == "reference" else None
                ),
                "dispositions_uri": str(directory / "dispositions.jsonl"),
                "full_traversal_report_uri": (
                    str(directory / "full_traversal_report.json") if arm.arm == "reference" else None
                ),
            }
            entries.append(entry)
    payload = {
        "schema_version": "llamafactory-sft-registry-v3",
        "experiment_id": manifest.experiment_id,
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(output_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def _resolve_registry_path(registry_path: Path, uri: str) -> Path:
    path = Path(uri)
    if not path.is_absolute():
        path = registry_path.parent / path
    return path.resolve()


def _optional_git_output(repository: Path, *arguments: str) -> str | None:
    """Return Git output when available without making Git a project-code requirement."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _capture_project_code_snapshot() -> ProjectCodeSnapshot:
    """Hash the exact project entry points and valuation logic used by a bundle."""

    # Resolving source files is intentional: unlike sys.executable these are
    # not virtual-environment entry points whose symlink identity must survive.
    project_root = Path(__file__).resolve().parents[2]
    files_sha256: dict[str, str] = {}
    for relative_path in PROJECT_CODE_RELATIVE_PATHS:
        source_path = project_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Required ai-ready project code file is missing: {source_path}")
        files_sha256[relative_path] = file_sha256(source_path)

    git_root_text = _optional_git_output(project_root, "rev-parse", "--show-toplevel")
    git_root = str(Path(git_root_text).resolve()) if git_root_text else None
    git_head = _optional_git_output(project_root, "rev-parse", "HEAD") if git_root else None
    git_remote = _optional_git_output(project_root, "remote", "get-url", "origin") if git_root else None
    if git_remote:
        # Remote URLs occasionally contain HTTPS credentials.  The host/path
        # identify the source sufficiently; credentials must not enter a bundle.
        git_remote = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)[^/@]+@", r"\1<redacted>@", git_remote)
    status = (
        _optional_git_output(project_root, "status", "--porcelain", "--untracked-files=normal", "--", ".")
        if git_root
        else None
    )
    return ProjectCodeSnapshot(
        project_root=str(project_root),
        files_sha256=files_sha256,
        git_root=git_root,
        git_head=git_head or None,
        git_remote=git_remote or None,
        git_dirty=None if status is None else bool(status),
    )


def _project_code_digest(snapshot: ProjectCodeSnapshot) -> str:
    return _canonical_digest(snapshot.model_dump(mode="json"))


def _verify_project_code_snapshot(snapshot: ProjectCodeSnapshot) -> None:
    """Require the running module and critical files to match the frozen bundle."""

    project_root = Path(snapshot.project_root)
    expected_module = (project_root / "ai_ready/valuation/llamafactory.py").resolve()
    if Path(__file__).resolve() != expected_module:
        raise ValueError(
            f"Running ai-ready code comes from {Path(__file__).resolve()}, expected frozen project {expected_module}"
        )
    observed: dict[str, str] = {}
    for relative_path in PROJECT_CODE_RELATIVE_PATHS:
        source_path = project_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Frozen ai-ready project code file is missing: {source_path}")
        observed[relative_path] = file_sha256(source_path)
    if observed != snapshot.files_sha256:
        changed = {
            relative_path: {
                "expected": snapshot.files_sha256.get(relative_path),
                "observed": observed.get(relative_path),
            }
            for relative_path in PROJECT_CODE_RELATIVE_PATHS
            if observed.get(relative_path) != snapshot.files_sha256.get(relative_path)
        }
        raise ValueError(f"ai-ready project code changed after bundle preparation: {changed}")


def _resolve_git_commit(repository: Path, revision: str) -> str:
    if not (repository / ".git").exists():
        raise ValueError(f"LLaMAFactory root is not a Git checkout: {repository}")
    expected = subprocess.run(
        ["git", "-C", str(repository), "rev-list", "-n", "1", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    actual = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not expected or actual != expected:
        raise ValueError(f"LLaMAFactory checkout is {actual}, expected {revision} ({expected})")
    release_commit = LLAMAFACTORY_RELEASE_COMMITS.get(revision)
    if release_commit is None or actual != release_commit:
        raise ValueError(
            f"LLaMAFactory checkout commit {actual} is not the registered immutable release commit for {revision}"
        )
    if dirty:
        raise ValueError("LLaMAFactory checkout has uncommitted changes; use a clean pinned checkout")
    return actual


def _resolve_installed_llamafactory(repository: Path, python_executable: str) -> Path:
    """Bind the training process to the editable package from ``repository``.

    Checking only the Git checkout is insufficient: a ``llamafactory-cli``
    found on PATH could import a different wheel.  Every job therefore invokes
    the same Python interpreter that prepared the bundle, and this probe
    requires that interpreter to import ``llamafactory`` from the pinned
    checkout's ``src/llamafactory`` directory.
    """

    probe = subprocess.run(
        [
            python_executable,
            "-c",
            "from pathlib import Path; import llamafactory; print(Path(llamafactory.__file__).resolve())",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    package_file = Path(probe.stdout.strip()).resolve()
    expected_root = (repository / "src" / "llamafactory").resolve()
    try:
        package_file.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(
            f"{python_executable} imports LLaMAFactory from {package_file}, expected an editable install from "
            f"{expected_root}"
        ) from error
    return package_file


def _runtime_packages(python_executable: str) -> dict[str, str]:
    """Fingerprint packages plus the CUDA kernel-selection environment."""

    script = """
import importlib.metadata as metadata
import importlib.util
import json
import platform
import subprocess

import torch

names = ["llamafactory", "torch", "transformers", "tokenizers", "datasets", "accelerate", "peft", "trl"]
payload = {"python": platform.python_version(), **{name: metadata.version(name) for name in names}}
cuda_available = torch.cuda.is_available()
cudnn_version = torch.backends.cudnn.version()
flash_attn_present = importlib.util.find_spec("flash_attn") is not None
try:
    flash_attn_version = metadata.version("flash-attn")
except metadata.PackageNotFoundError:
    flash_attn_version = "none"
payload.update(
    {
        "torch.version.cuda": torch.version.cuda or "none",
        "torch.backends.cudnn.version": str(cudnn_version) if cudnn_version is not None else "none",
        "torch.cuda.is_available": str(cuda_available).lower(),
        "torch.cuda.device_name.0": torch.cuda.get_device_name(0) if cuda_available else "none",
        "torch.cuda.device_capability.0": (
            ".".join(str(value) for value in torch.cuda.get_device_capability(0)) if cuda_available else "none"
        ),
        "flash_attn.present": str(flash_attn_present).lower(),
        "flash_attn.version": flash_attn_version,
    }
)
try:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    driver_versions = sorted({line.strip() for line in driver.stdout.splitlines() if line.strip()})
    payload["nvidia.driver_versions"] = (
        ",".join(driver_versions) if driver.returncode == 0 and driver_versions else "unavailable"
    )
except (OSError, subprocess.SubprocessError):
    payload["nvidia.driver_versions"] = "unavailable"
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [python_executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("Could not fingerprint the LLaMAFactory runtime packages")
    return payload


def _verify_preparation_devices(profile: LlamaFactoryServerProfile, runtime_packages: dict[str, str]) -> None:
    """Ensure preparation observes the same kind of GPU allocation required at run time."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = [value.strip() for value in (visible or "").split(",") if value.strip() and value.strip() != "-1"]
    if len(device_ids) != profile.world_size or runtime_packages.get("torch.cuda.is_available") != "true":
        raise RuntimeError(
            "Prepare the LLaMAFactory bundle inside a GPU compute allocation with CUDA_VISIBLE_DEVICES "
            f"set to exactly world_size={profile.world_size}; observed CUDA_VISIBLE_DEVICES={visible!r}, "
            f"torch.cuda.is_available={runtime_packages.get('torch.cuda.is_available')!r}"
        )
    if runtime_packages.get("torch.cuda.device_name.0") in {None, "none"} or runtime_packages.get(
        "torch.cuda.device_capability.0"
    ) in {None, "none"}:
        raise RuntimeError("Could not fingerprint the visible GPU name and compute capability during preparation")


def _validate_generation_manifest(
    *,
    entry: SFTArtifactEntry,
    trial: ExpectedTrial,
    contract: ProductionContract,
    source_path: Path,
) -> tuple[Path, SFTGenerationManifest, str]:
    """Bind one SFT file to the independent generator registered for its arm."""

    manifest_path = Path(entry.generation_manifest_uri).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing generation manifest: {manifest_path}")
    manifest_sha = file_sha256(manifest_path)
    manifest = SFTGenerationManifest.model_validate(_load_json_object(manifest_path))
    recipe = contract.generation.recipe_for_arm(trial.arm)
    recipe_sha = _canonical_digest(recipe.model_dump(mode="json"))
    if (
        trial.generator != recipe.generator
        or trial.generator_revision != recipe.revision
        or trial.generation_recipe_sha256 != recipe_sha
    ):
        raise ValueError(f"Trial {trial.arm_id} does not bind the frozen {trial.arm} generation recipe")
    expected = {
        "artifact_role": trial.arm,
        "input_scope": "selected_semantic_units" if trial.arm == "proxy" else "full_batch",
        "batch_id": trial.batch_id,
        "batch_content_sha256": trial.batch_content_sha256,
        "generator": trial.generator,
        "generator_revision": trial.generator_revision,
        "implementation_sha256": recipe.implementation_sha256 if trial.arm == "proxy" else None,
        "generation_recipe_sha256": trial.generation_recipe_sha256,
        "source_sft_sha256": file_sha256(source_path),
        "contract_id": contract.contract_id,
        "contract_sha256": contract_digest(contract),
    }
    mismatches = {
        name: {"expected": expected_value, "observed": getattr(manifest, name)}
        for name, expected_value in expected.items()
        if getattr(manifest, name) != expected_value
    }
    lineage_path = Path(entry.selection_manifest_uri or entry.full_traversal_report_uri or "").resolve()
    lineage_sha = file_sha256(lineage_path)
    observed_lineage_sha = (
        manifest.selection_manifest_sha256 if trial.arm == "proxy" else manifest.full_traversal_report_sha256
    )
    if observed_lineage_sha != lineage_sha:
        mismatches["lineage_sha256"] = {"expected": lineage_sha, "observed": observed_lineage_sha}
    if trial.arm == "proxy":
        # The selector is replayed later, but it is only meaningful when its
        # registered input is the complete pool mined from the frozen batch.
        # Bind that pool here so deleting hard units invalidates provenance.
        candidate_pool_path = Path(entry.proxy_units_uri or "").resolve()
        if not candidate_pool_path.is_file():
            raise FileNotFoundError(f"Missing complete proxy candidate pool: {candidate_pool_path}")
        candidate_pool_sha = file_sha256(candidate_pool_path)
        if manifest.candidate_pool_sha256 != candidate_pool_sha:
            mismatches["candidate_pool_sha256"] = {
                "expected": candidate_pool_sha,
                "observed": manifest.candidate_pool_sha256,
            }
    else:
        # The importer copies the external FOM run receipt beside the common
        # generation manifest.  Its digest prevents a normalized JSONL file
        # from acquiring FOM provenance merely through self-declared labels.
        upstream_run_path = Path(entry.upstream_run_manifest_uri or "").resolve()
        if not upstream_run_path.is_file():
            raise FileNotFoundError(f"Missing Fully Open Meditron run manifest: {upstream_run_path}")
        upstream_run_sha = file_sha256(upstream_run_path)
        if manifest.upstream_run_manifest_sha256 != upstream_run_sha:
            mismatches["upstream_run_manifest_sha256"] = {
                "expected": upstream_run_sha,
                "observed": manifest.upstream_run_manifest_sha256,
            }
        receipt = FullyOpenMeditronRunManifest.model_validate(_load_json_object(upstream_run_path))
        generation_tools = [tool for tool in contract.tools if tool.stage == "generate"]
        if len(generation_tools) != 1:
            raise ValueError("Production contract must bind exactly one reference generation tool")
        tool = generation_tools[0]
        receipt_expected = {
            "generator_revision": recipe.revision,
            "tool_adapter": tool.adapter,
            "tool_repository": tool.repository,
            "tool_config_sha256": tool.config_sha256,
            "teacher_model": recipe.teacher_model,
            "teacher_revision": recipe.teacher_revision,
            "prompt_sha256": recipe.prompt_sha256,
            "quality_gate_sha256": recipe.quality_gate_sha256,
            "batch_id": trial.batch_id,
            "batch_content_sha256": trial.batch_content_sha256,
            "contract_id": contract.contract_id,
            "contract_sha256": contract_digest(contract),
        }
        for field_name, expected_value in receipt_expected.items():
            observed_value = getattr(receipt, field_name)
            if observed_value != expected_value:
                mismatches[f"upstream_run_manifest.{field_name}"] = {
                    "expected": expected_value,
                    "observed": observed_value,
                }
        manifest_inputs = (manifest.model_extra or {}).get("inputs")
        if not isinstance(manifest_inputs, dict):
            mismatches["inputs"] = {"expected": "generation input hashes", "observed": manifest_inputs}
        else:
            receipt_input_bindings = {
                "eligible_units_sha256": receipt.eligible_units_sha256,
                "fully_open_meditron_sha256": receipt.normalized_output_sha256,
            }
            for field_name, expected_value in receipt_input_bindings.items():
                observed_value = manifest_inputs.get(field_name)
                if observed_value != expected_value:
                    mismatches[f"inputs.{field_name}"] = {
                        "expected": expected_value,
                        "observed": observed_value,
                    }
    if mismatches:
        raise ValueError(f"Generation manifest does not match trial {trial.arm_id}: {mismatches}")
    if file_sha256(manifest_path) != manifest_sha:
        raise RuntimeError("Generation manifest changed while provenance validation was running")
    return manifest_path, manifest, manifest_sha


def _validate_lineage_artifact(
    *,
    entry: SFTArtifactEntry,
    trial: ExpectedTrial,
    source_path: Path,
    source_rows: list[SFTSample],
) -> tuple[Path, Path | None, Path | None, Path | None, Path, dict[str, str]]:
    """Prove that the registered SFT rows belong to the intended experiment arm."""

    lineage_path = Path(entry.selection_manifest_uri or entry.full_traversal_report_uri or "").resolve()
    input_hashes = {
        "lineage": file_sha256(lineage_path),
        "source_sft": file_sha256(source_path),
    }
    payload = _load_json_object(lineage_path)
    if trial.arm == "proxy":
        if payload.get("schema_version") != "proxy-selection-v1":
            raise ValueError(f"Proxy lineage is not a proxy-selection-v1 manifest: {lineage_path}")
        if trial.selection_token_budget is None:
            raise ValueError(f"Proxy trial does not freeze its selector budget: {trial.arm_id}")
        config_payload = payload.get("selection_config")
        if not isinstance(config_payload, dict):
            raise ValueError(f"Proxy lineage has no valid selection_config object: {lineage_path}")
        config = ProxySelectionConfig.model_validate(config_payload)
        expected_fields = {
            "batch_id": trial.batch_id,
            "batch_content_sha256": trial.batch_content_sha256,
            "method": trial.method,
            "selection_config_sha256": trial.selection_config_sha256,
            "token_budget": trial.selection_token_budget,
        }
        mismatches = {
            name: {"expected": expected, "observed": payload.get(name)}
            for name, expected in expected_fields.items()
            if payload.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"Proxy lineage does not match trial {trial.arm_id}: {mismatches}")
        if config.method in {"random", "stratified_random"}:
            if trial.selection_seed is None or config.random_state != trial.selection_seed:
                raise ValueError(
                    f"Stochastic proxy selector random_state={config.random_state} does not match registered "
                    f"seed {trial.selection_seed}: {lineage_path}"
                )

        units_path = Path(entry.proxy_units_uri or "").resolve()
        obligations_path = Path(entry.coverage_obligations_uri or "").resolve()
        dispositions_path = Path(entry.dispositions_uri or "").resolve()
        for artifact_name, artifact_path in (
            ("proxy units", units_path),
            ("coverage obligations", obligations_path),
            ("unit dispositions", dispositions_path),
        ):
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing {artifact_name} required for proxy replay: {artifact_path}")
        input_hashes.update(
            {
                "proxy_units": file_sha256(units_path),
                "coverage_obligations": file_sha256(obligations_path),
                "dispositions": file_sha256(dispositions_path),
            }
        )
        units = [ProxyUnit.model_validate(row) for row in read_jsonl(units_path)]
        obligations = [CoverageObligation.model_validate(row) for row in read_jsonl(obligations_path)]
        dispositions = [UnitDisposition.model_validate(row) for row in read_jsonl(dispositions_path)]

        # A saved manifest is only a claim. Re-run the registered selector from
        # its frozen raw inputs. Structure, discrete values, and list order are
        # exact; a tiny tolerance is allowed only for BLAS-dependent floats.
        replayed = select_proxy(units, obligations, config)
        replay_mismatch = _selector_replay_mismatch(replayed, payload)
        if replay_mismatch:
            raise ValueError(
                f"Proxy selection manifest cannot be reproduced from its registered inputs at "
                f"{replay_mismatch}: {lineage_path}"
            )
        selected_unit_ids = payload.get("selected_unit_ids")
        if not isinstance(selected_unit_ids, list) or any(not isinstance(value, str) for value in selected_unit_ids):
            raise ValueError(f"Proxy lineage has invalid selected_unit_ids: {lineage_path}")
        selected = set(selected_unit_ids)
        outside = {
            sample.sample_id: sorted(set(sample.semantic_unit_ids) - selected)
            for sample in source_rows
            if set(sample.semantic_unit_ids) - selected
        }
        if outside:
            raise ValueError(f"SFT samples use semantic units outside the selected proxy subset: {outside}")
        traversal = validate_full_traversal(selected_unit_ids, dispositions, source_rows)
        if not traversal.valid:
            raise ValueError(
                "Proxy production does not account for every selected unit and accepted SFT row: " f"{traversal.errors}"
            )
        current_hashes = {
            "lineage": file_sha256(lineage_path),
            "source_sft": file_sha256(source_path),
            "proxy_units": file_sha256(units_path),
            "coverage_obligations": file_sha256(obligations_path),
            "dispositions": file_sha256(dispositions_path),
        }
        if current_hashes != input_hashes:
            raise RuntimeError("Proxy production inputs changed while selector replay was running")
        return lineage_path, units_path, obligations_path, None, dispositions_path, input_hashes
    else:
        if payload.get("schema_version") != "full-traversal-report-v1":
            raise ValueError(f"Reference lineage is not a full-traversal-report-v1 artifact: {lineage_path}")

        eligible_path = Path(entry.eligible_units_uri or "").resolve()
        dispositions_path = Path(entry.dispositions_uri or "").resolve()
        for artifact_name, artifact_path in (
            ("eligible units", eligible_path),
            ("unit dispositions", dispositions_path),
        ):
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing {artifact_name} required for reference replay: {artifact_path}")
        input_hashes.update(
            {
                "eligible_units": file_sha256(eligible_path),
                "dispositions": file_sha256(dispositions_path),
            }
        )

        eligible_rows = read_jsonl(eligible_path)
        eligible_unit_ids: list[str] = []
        for row_index, row in enumerate(eligible_rows):
            if not isinstance(row, dict):
                raise ValueError(f"Eligible-unit row {row_index} is not a JSON object: {eligible_path}")
            semantic_unit_id = row.get("semantic_unit_id")
            batch_id = row.get("batch_id")
            if not isinstance(semantic_unit_id, str) or not semantic_unit_id:
                raise ValueError(f"Eligible-unit row {row_index} has no non-empty semantic_unit_id: {eligible_path}")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError(f"Eligible-unit row {row_index} has no non-empty batch_id: {eligible_path}")
            if batch_id != trial.batch_id:
                raise ValueError(
                    f"Eligible-unit row {row_index} belongs to batch {batch_id!r}, expected {trial.batch_id!r}: "
                    f"{eligible_path}"
                )
            eligible_unit_ids.append(semantic_unit_id)
        dispositions = [UnitDisposition.model_validate(row) for row in read_jsonl(dispositions_path)]
        traversal = validate_full_traversal(eligible_unit_ids, dispositions, source_rows)
        if not traversal.valid:
            raise ValueError(
                "Reference production does not account for every eligible unit and accepted SFT row: "
                f"{traversal.errors}"
            )

        # The stored report is a cache of this deterministic conservation
        # check. Rebuild the complete report from the registered inputs so a
        # hand-edited `valid: true` or stale input hash cannot authorize data.
        reconstructed = {
            "schema_version": "full-traversal-report-v1",
            "batch_id": trial.batch_id,
            "batch_content_sha256": trial.batch_content_sha256,
            "eligible_units_sha256": input_hashes["eligible_units"],
            "dispositions_sha256": input_hashes["dispositions"],
            "source_sft_sha256": input_hashes["source_sft"],
            **traversal.model_dump(mode="json"),
        }
        replay_mismatch = _selector_replay_mismatch(reconstructed, payload)
        if replay_mismatch:
            raise ValueError(
                f"Full-traversal report cannot be reproduced from its registered inputs at "
                f"{replay_mismatch}: {lineage_path}"
            )
        current_hashes = {
            "lineage": file_sha256(lineage_path),
            "eligible_units": file_sha256(eligible_path),
            "dispositions": file_sha256(dispositions_path),
            "source_sft": file_sha256(source_path),
        }
        if current_hashes != input_hashes:
            raise RuntimeError("Reference production inputs changed while full-traversal replay was running")
        return lineage_path, None, None, eligible_path, dispositions_path, input_hashes


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Mirror LLaMAFactory v0.9.5 ``processor_utils.infer_seqlen``."""

    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    return min(max_source_len, source_len), new_target_len


def _greedy_knapsack_count(lengths: list[int], capacity: int) -> int:
    """Count packed examples using LLaMAFactory's largest-fit-first algorithm."""

    remaining = sorted(lengths)
    bins = 0
    while remaining:
        available = capacity
        while True:
            index = -1
            for candidate in range(len(remaining) - 1, -1, -1):
                if remaining[candidate] <= available:
                    index = candidate
                    break
            if index < 0:
                break
            available -= remaining.pop(index)
        bins += 1
    return bins


def _qwen_token_counts(tokenizer, instruction: str, response: str, cutoff_len: int) -> tuple[int, int, bool]:
    system = QWEN_TEMPLATE_SPEC["default_system"]
    source = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
    target = f"{response}<|im_end|>\n"
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), cutoff_len)
    truncated = source_len != len(source_ids) or target_len != len(target_ids)
    return source_len + target_len, target_len, truncated


def _validate_generation_provenance(
    samples: list[SFTSample],
    contract: ProductionContract,
    trial: ExpectedTrial,
    source_path: Path,
) -> None:
    """Reject rows that do not match the frozen recipe for this branch."""

    generation = contract.generation.recipe_for_arm(trial.arm)
    expected = {
        "generator": generation.generator,
        "generator_revision": generation.revision,
        "teacher_model": generation.teacher_model,
        "teacher_revision": generation.teacher_revision,
        "prompt_sha256": generation.prompt_sha256,
        "quality_gate_sha256": generation.quality_gate_sha256,
    }
    mismatches: dict[str, dict[str, dict[str, str | None]]] = {}
    for sample in samples:
        row_mismatches = {
            field_name: {"expected": expected_value, "observed": getattr(sample, field_name)}
            for field_name, expected_value in expected.items()
            if getattr(sample, field_name) != expected_value
        }
        if row_mismatches:
            mismatches[sample.sample_id] = row_mismatches
    if mismatches:
        preview = dict(list(sorted(mismatches.items()))[:20])
        suffix = "" if len(mismatches) <= 20 else f"; plus {len(mismatches) - 20} more rows"
        raise ValueError(f"SFT rows in {source_path} do not match the frozen generation recipe: {preview}{suffix}")


def _convert_training_data(
    *,
    entry: SFTArtifactEntry,
    source_path: Path,
    destination_dir: Path,
    tokenizer,
    contract: ProductionContract,
    trial: ExpectedTrial,
    num_train_epochs: float,
    preprocessing_num_workers: int,
    preprocessing_batch_size: int,
) -> TrainingDataAudit:
    source_rows = [SFTSample.model_validate(row) for row in read_jsonl(source_path)]
    if any(row.batch_id != entry.batch_id for row in source_rows):
        raise ValueError(f"SFT file for {entry.batch_id}/{entry.arm_label} contains another batch")
    _validate_generation_provenance(source_rows, contract, trial, source_path)
    sample_ids = [row.sample_id for row in source_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_id in {source_path}")
    supervision = [(row.instruction.strip(), row.response.strip()) for row in source_rows]
    duplicated = [pair for pair, count in Counter(supervision).items() if count > 1]
    if duplicated:
        raise ValueError(f"Duplicate instruction/response supervision in {source_path}: {len(duplicated)} pairs")
    if preprocessing_num_workers != 1:
        raise ValueError("Exact packed-row accounting requires preprocessing_num_workers=1")
    if preprocessing_batch_size < max(len(source_rows), 1):
        raise ValueError(
            f"preprocessing_batch_size={preprocessing_batch_size} would split {len(source_rows)} rows into "
            "multiple independent packing batches"
        )

    generation_manifest_source, generation_manifest, generation_manifest_sha = _validate_generation_manifest(
        entry=entry,
        trial=trial,
        contract=contract,
        source_path=source_path,
    )

    # Validate lineage before creating the immutable bundle directory.  This
    # prevents a valid SFT file from being attached to the wrong selector,
    # batch, or full-traversal audit merely by editing the registry path.
    (
        lineage_source,
        proxy_units_source,
        obligations_source,
        eligible_units_source,
        dispositions_source,
        lineage_input_hashes,
    ) = _validate_lineage_artifact(
        entry=entry,
        trial=trial,
        source_path=source_path,
        source_rows=source_rows,
    )

    destination_dir.mkdir(parents=True, exist_ok=False)
    training_path = destination_dir / "accepted_sft.jsonl"
    provenance_path = destination_dir / "provenance.jsonl"
    dataset_info_path = destination_dir / "dataset_info.json"
    generation_manifest_path = destination_dir / "generation_manifest.json"
    shutil.copyfile(generation_manifest_source, generation_manifest_path)
    if file_sha256(generation_manifest_path) != generation_manifest_sha:
        raise RuntimeError("Generation manifest changed between validation and immutable bundle copy")
    upstream_run_manifest_path: Path | None = None
    if trial.arm == "reference":
        upstream_run_source = Path(entry.upstream_run_manifest_uri or "").resolve()
        upstream_run_manifest_path = destination_dir / "fully_open_meditron_run_manifest.json"
        shutil.copyfile(upstream_run_source, upstream_run_manifest_path)
        if file_sha256(upstream_run_manifest_path) != generation_manifest.upstream_run_manifest_sha256:
            raise RuntimeError("Fully Open Meditron run manifest changed between validation and immutable bundle copy")
    lineage_name = "selection_manifest.json" if entry.selection_manifest_uri else "full_traversal_report.json"
    lineage_path = destination_dir / lineage_name
    shutil.copyfile(lineage_source, lineage_path)
    if file_sha256(lineage_path) != lineage_input_hashes["lineage"]:
        raise RuntimeError("Lineage artifact changed between validation and immutable bundle copy")
    proxy_units_path: Path | None = None
    obligations_path: Path | None = None
    eligible_units_path: Path | None = None
    dispositions_path = destination_dir / "dispositions.jsonl"
    shutil.copyfile(dispositions_source, dispositions_path)
    copied_hashes = {"dispositions": file_sha256(dispositions_path)}
    if proxy_units_source is not None and obligations_source is not None:
        proxy_units_path = destination_dir / "proxy_units.jsonl"
        obligations_path = destination_dir / "coverage_obligations.jsonl"
        shutil.copyfile(proxy_units_source, proxy_units_path)
        shutil.copyfile(obligations_source, obligations_path)
        copied_hashes.update(
            {
                "proxy_units": file_sha256(proxy_units_path),
                "coverage_obligations": file_sha256(obligations_path),
            }
        )
    elif eligible_units_source is not None:
        eligible_units_path = destination_dir / "eligible_units.jsonl"
        shutil.copyfile(eligible_units_source, eligible_units_path)
        copied_hashes["eligible_units"] = file_sha256(eligible_units_path)
    else:
        raise RuntimeError("Lineage replay returned neither proxy inputs nor reference eligible units")
    expected_hashes = {name: lineage_input_hashes[name] for name in copied_hashes}
    if copied_hashes != expected_hashes:
        raise RuntimeError("Production inputs changed between lineage replay and immutable bundle copy")
    if file_sha256(source_path) != lineage_input_hashes["source_sft"]:
        raise RuntimeError("Source SFT changed after production replay")

    training_rows = []
    provenance_rows = []
    input_lengths: list[int] = []
    supervised_tokens = 0
    truncated = 0
    hp = _required_hyperparameters(contract)
    configured_cutoff_len = hp.max_length
    # LLaMAFactory v0.9.5 DataArguments.__post_init__ subtracts one when
    # packing is enabled, then the packed processor pads each row to
    # processing_cutoff + 1.  Both stages matter for exact budget accounting.
    processing_cutoff_len = configured_cutoff_len - 1 if hp.packing else configured_cutoff_len
    generation_recipe_sha256 = _canonical_digest(contract.generation.recipe_for_arm(trial.arm).model_dump(mode="json"))
    for index, sample in enumerate(sorted(source_rows, key=lambda row: row.sample_id)):
        input_count, label_count, was_truncated = _qwen_token_counts(
            tokenizer, sample.instruction, sample.response, processing_cutoff_len
        )
        if label_count <= 0:
            raise ValueError(f"Sample {sample.sample_id} has no supervised tokens after truncation")
        input_lengths.append(input_count)
        supervised_tokens += label_count
        truncated += int(was_truncated)
        training_rows.append(
            {
                "messages": [
                    {"role": "system", "content": QWEN_TEMPLATE_SPEC["default_system"]},
                    {"role": "user", "content": sample.instruction},
                    {"role": "assistant", "content": sample.response},
                ],
                "sample_id": sample.sample_id,
                "semantic_unit_ids": sample.semantic_unit_ids,
                "source_ids": sample.source_ids,
            }
        )
        provenance_rows.append(
            {
                "row_index": index,
                "sample_id": sample.sample_id,
                "semantic_unit_ids": sample.semantic_unit_ids,
                "source_ids": sample.source_ids,
                "evidence_spans": sample.evidence_spans,
                "generator": sample.generator,
                "generator_revision": sample.generator_revision,
                "teacher_model": sample.teacher_model,
                "teacher_revision": sample.teacher_revision,
                "prompt_sha256": sample.prompt_sha256,
                "quality_gate_sha256": sample.quality_gate_sha256,
                "generation_arm": trial.arm,
                "generation_recipe_sha256": generation_recipe_sha256,
                "generation_manifest_sha256": generation_manifest_sha,
                "input_tokens_after_cutoff": input_count,
                "supervised_tokens_after_cutoff": label_count,
                "truncated": was_truncated,
            }
        )
    write_jsonl(training_path, training_rows)
    write_jsonl(provenance_path, provenance_rows)
    dataset_info = {
        "ai_ready_sft": {
            "file_name": training_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    write_json(dataset_info_path, dataset_info)

    effective_tokens = int(round(supervised_tokens * num_train_epochs))
    if not float(num_train_epochs).is_integer():
        raise ValueError("This experiment requires an integer epoch count for exact token accounting")
    if effective_tokens > trial.max_effective_train_tokens:
        raise ValueError(
            f"{entry.batch_id}/{entry.arm_label} has {effective_tokens} supervised tokens, "
            f"above its registered cap {trial.max_effective_train_tokens}"
        )
    packed_tokens = (
        _greedy_knapsack_count(input_lengths, processing_cutoff_len) * (processing_cutoff_len + 1)
        if hp.packing and input_lengths
        else sum(input_lengths)
    )
    if file_sha256(source_path) != lineage_input_hashes["source_sft"]:
        raise RuntimeError("Source SFT changed while the immutable training view was being built")
    return TrainingDataAudit(
        experiment_id=trial.experiment_id,
        batch_id=entry.batch_id,
        arm_id=entry.arm_id,
        arm_label=entry.arm_label,
        source_sft_uri=str(source_path),
        source_sft_sha256=lineage_input_hashes["source_sft"],
        generation_manifest_uri=str(generation_manifest_path),
        generation_manifest_sha256=generation_manifest_sha,
        generator=generation_manifest.generator,
        generator_revision=generation_manifest.generator_revision,
        generation_recipe_sha256=generation_manifest.generation_recipe_sha256,
        training_sft_uri=str(training_path),
        training_sft_sha256=file_sha256(training_path),
        dataset_info_uri=str(dataset_info_path),
        dataset_info_sha256=file_sha256(dataset_info_path),
        provenance_uri=str(provenance_path),
        provenance_sha256=file_sha256(provenance_path),
        lineage_artifact_uri=str(lineage_path),
        lineage_artifact_sha256=file_sha256(lineage_path),
        proxy_units_uri=str(proxy_units_path) if proxy_units_path else None,
        proxy_units_sha256=file_sha256(proxy_units_path) if proxy_units_path else None,
        coverage_obligations_uri=str(obligations_path) if obligations_path else None,
        coverage_obligations_sha256=file_sha256(obligations_path) if obligations_path else None,
        eligible_units_uri=str(eligible_units_path) if eligible_units_path else None,
        eligible_units_sha256=file_sha256(eligible_units_path) if eligible_units_path else None,
        upstream_run_manifest_uri=str(upstream_run_manifest_path) if upstream_run_manifest_path else None,
        upstream_run_manifest_sha256=(file_sha256(upstream_run_manifest_path) if upstream_run_manifest_path else None),
        dispositions_uri=str(dispositions_path),
        dispositions_sha256=file_sha256(dispositions_path),
        sample_count=len(training_rows),
        truncated_sample_count=truncated,
        input_token_count=sum(input_lengths),
        supervised_token_count=supervised_tokens,
        effective_train_tokens=effective_tokens,
        packed_token_count=packed_tokens,
        configured_cutoff_len=configured_cutoff_len,
        processing_cutoff_len=processing_cutoff_len,
        preprocessing_num_workers=preprocessing_num_workers,
        preprocessing_batch_size=preprocessing_batch_size,
        num_train_epochs=num_train_epochs,
        tokenizer_name_or_path=contract.training.base_model,
        tokenizer_revision=contract.training.tokenizer_revision,
        template="qwen",
        template_sha256=qwen_template_digest(),
    )


def _load_tokenizer(contract: ProductionContract, profile: LlamaFactoryServerProfile):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required on the server to audit LLaMAFactory token budgets") from error
    if contract.training.base_model_revision != contract.training.tokenizer_revision:
        raise ValueError("LLaMAFactory requires the base-model and tokenizer revisions to be identical")
    return AutoTokenizer.from_pretrained(
        contract.training.base_model,
        revision=contract.training.tokenizer_revision,
        trust_remote_code=profile.trust_remote_code,
    )


def _required_hyperparameters(contract: ProductionContract) -> LlamaFactoryHyperparameters:
    required = {
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target",
        "max_length",
        "packing",
        "neat_packing",
        "num_train_epochs",
        "lr_scheduler_type",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
    }
    missing = required - set(contract.training.hyperparameters)
    if missing:
        raise ValueError(f"Production contract is missing LLaMAFactory hyperparameters: {sorted(missing)}")
    if not contract.training.response_only_loss:
        raise ValueError("This adapter requires response_only_loss=true")
    return LlamaFactoryHyperparameters.model_validate(contract.training.hyperparameters)


def _training_config(
    *,
    trial: ExpectedTrial,
    contract: ProductionContract,
    profile: LlamaFactoryServerProfile,
    dataset_dir: Path,
    adapter_dir: Path,
) -> dict[str, Any]:
    hp = _required_hyperparameters(contract)
    config: dict[str, Any] = {
        "model_name_or_path": contract.training.base_model,
        "model_revision": contract.training.base_model_revision,
        "trust_remote_code": profile.trust_remote_code,
        "flash_attn": profile.flash_attn,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": hp.lora_target,
        "lora_rank": hp.lora_rank,
        "lora_alpha": hp.lora_alpha,
        "lora_dropout": hp.lora_dropout,
        "dataset": "ai_ready_sft",
        "dataset_dir": str(dataset_dir),
        "template": profile.template,
        "cutoff_len": hp.max_length,
        "train_on_prompt": False,
        "packing": hp.packing,
        "neat_packing": hp.neat_packing,
        "overwrite_cache": False,
        "preprocessing_batch_size": profile.preprocessing_batch_size,
        "preprocessing_num_workers": profile.preprocessing_num_workers,
        "dataloader_num_workers": profile.dataloader_num_workers,
        "output_dir": str(adapter_dir),
        "logging_steps": profile.logging_steps,
        "save_strategy": "steps",
        "save_steps": profile.save_steps,
        "save_total_limit": profile.save_total_limit,
        "save_safetensors": True,
        "save_only_model": False,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": profile.report_to,
        "per_device_train_batch_size": profile.per_device_train_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "learning_rate": hp.learning_rate,
        "num_train_epochs": hp.num_train_epochs,
        "lr_scheduler_type": hp.lr_scheduler_type,
        "warmup_ratio": hp.warmup_ratio,
        "weight_decay": hp.weight_decay,
        "max_grad_norm": hp.max_grad_norm,
        "bf16": profile.bf16,
        "fp16": profile.fp16,
        "gradient_checkpointing": profile.gradient_checkpointing,
        "seed": trial.seed,
        "data_seed": trial.seed,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
        "include_num_input_tokens_seen": True,
        "val_size": 0.0,
    }
    if profile.deepspeed_config:
        config["deepspeed"] = str(Path(profile.deepspeed_config).resolve())
    return config


def prepare_llamafactory_jobs(
    *,
    experiment_manifest_path: str | Path,
    experiment_manifest: ExperimentManifest,
    contract: ProductionContract,
    registry_path: str | Path,
    profile_path: str | Path,
    llamafactory_root: str | Path,
    output_dir: str | Path,
    discard_incomplete: bool = False,
) -> LlamaFactoryJobManifest:
    """Materialize 84 immutable datasets and 252 server-side train jobs."""

    experiment_manifest_path = Path(experiment_manifest_path).resolve()
    registry_path = Path(registry_path).resolve()
    profile_path = Path(profile_path).resolve()
    llamafactory_root = Path(llamafactory_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"LLaMAFactory output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        incomplete_marker = output_dir / "PREPARING.json"
        ready_marker = output_dir / "BUNDLE_READY.json"
        if discard_incomplete and incomplete_marker.is_file() and not ready_marker.exists():
            # The caller opted in explicitly.  A prepared bundle or a directory
            # without our marker is never removed by this recovery path.
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"LLaMAFactory output directory is not empty: {output_dir}. "
                "Use --discard-incomplete only when PREPARING.json exists and BUNDLE_READY.json does not."
            )
    manifest_from_disk = load_experiment_manifest(experiment_manifest_path)
    if manifest_from_disk != experiment_manifest:
        raise ValueError("experiment_manifest_path does not contain the supplied experiment manifest")
    if experiment_manifest.contract_sha256 != contract_digest(contract):
        raise ValueError("Experiment manifest does not match the supplied production contract")
    if contract.training.trainer != "llama_factory":
        raise ValueError("Production contract training.trainer must be llama_factory")
    if contract.training.update_method != "lora":
        raise ValueError("The current server adapter supports the preregistered LoRA arm only")
    if contract.training.chat_template_sha256 != qwen_template_digest():
        raise ValueError("Production contract chat_template_sha256 does not match the pinned Qwen template")

    profile = load_server_profile(profile_path)
    if profile.llamafactory_revision != contract.training.trainer_revision:
        raise ValueError("Server profile and production contract use different LLaMAFactory revisions")
    train_tools = [tool for tool in contract.tools if tool.stage == "train"]
    if len(train_tools) != 1 or train_tools[0].adapter != "llama_factory":
        raise ValueError("Production contract must bind the train stage to the llama_factory adapter")
    train_tool = train_tools[0]
    if train_tool.repository.rstrip("/") != LLAMAFACTORY_REPOSITORY:
        raise ValueError("Production contract train tool must reference the official LlamaFactory repository")
    if train_tool.revision != profile.llamafactory_revision:
        raise ValueError("Train tool and server profile use different LLaMAFactory revisions")
    if train_tool.config_sha256 != server_profile_digest(profile):
        raise ValueError("Train tool config_sha256 does not match the supplied server profile semantics")
    hp = _required_hyperparameters(contract)
    commit = _resolve_git_commit(llamafactory_root, profile.llamafactory_revision)
    # Preserve the virtual-environment entry point.  On Linux it is commonly a
    # symlink to the system binary; Path.resolve() would discard the venv path
    # and the later subprocess could lose the environment's site-packages.
    python_executable = os.path.abspath(sys.executable)
    package_file = _resolve_installed_llamafactory(llamafactory_root, python_executable)
    runtime_packages = _runtime_packages(python_executable)
    _verify_preparation_devices(profile, runtime_packages)
    project_code = _capture_project_code_snapshot()
    project_code_sha256 = _project_code_digest(project_code)
    deepspeed_path: Path | None = None
    deepspeed_sha: str | None = None
    source_deepspeed: Path | None = None
    if profile.deepspeed_config:
        source_deepspeed = Path(profile.deepspeed_config)
        if not source_deepspeed.is_absolute():
            source_deepspeed = profile_path.parent / source_deepspeed
        source_deepspeed = source_deepspeed.resolve()
        if not source_deepspeed.is_file():
            raise FileNotFoundError(f"DeepSpeed config does not exist: {source_deepspeed}")
    registry = load_sft_registry(registry_path)
    if registry.experiment_id != experiment_manifest.experiment_id:
        raise ValueError("SFT registry belongs to another experiment")

    trial_by_group: dict[tuple[str, str], ExpectedTrial] = {}
    for trial in experiment_manifest.trials:
        trial_by_group.setdefault((trial.batch_id, trial.arm_id), trial)
    entries = {(entry.batch_id, entry.arm_id): entry for entry in registry.entries}
    if set(entries) != set(trial_by_group):
        missing = sorted(set(trial_by_group) - set(entries))
        unexpected = sorted(set(entries) - set(trial_by_group))
        raise ValueError(f"SFT registry groups do not match the experiment; missing={missing}, unexpected={unexpected}")

    # Resolve registry-relative paths once.  The copied bundle then becomes the
    # only training input, so later edits to generation directories cannot
    # silently alter a planned trial.
    resolved_entries: dict[tuple[str, str], SFTArtifactEntry] = {}
    for key, entry in entries.items():
        source = _resolve_registry_path(registry_path, entry.source_sft_uri)
        generation_manifest = _resolve_registry_path(registry_path, entry.generation_manifest_uri)
        selection = (
            str(_resolve_registry_path(registry_path, entry.selection_manifest_uri))
            if entry.selection_manifest_uri
            else None
        )
        proxy_units = (
            str(_resolve_registry_path(registry_path, entry.proxy_units_uri)) if entry.proxy_units_uri else None
        )
        obligations = (
            str(_resolve_registry_path(registry_path, entry.coverage_obligations_uri))
            if entry.coverage_obligations_uri
            else None
        )
        eligible_units = (
            str(_resolve_registry_path(registry_path, entry.eligible_units_uri)) if entry.eligible_units_uri else None
        )
        upstream_run_manifest = (
            str(_resolve_registry_path(registry_path, entry.upstream_run_manifest_uri))
            if entry.upstream_run_manifest_uri
            else None
        )
        dispositions = (
            str(_resolve_registry_path(registry_path, entry.dispositions_uri)) if entry.dispositions_uri else None
        )
        traversal = (
            str(_resolve_registry_path(registry_path, entry.full_traversal_report_uri))
            if entry.full_traversal_report_uri
            else None
        )
        resolved_entries[key] = entry.model_copy(
            update={
                "source_sft_uri": str(source),
                "generation_manifest_uri": str(generation_manifest),
                "selection_manifest_uri": selection,
                "proxy_units_uri": proxy_units,
                "coverage_obligations_uri": obligations,
                "eligible_units_uri": eligible_units,
                "upstream_run_manifest_uri": upstream_run_manifest,
                "dispositions_uri": dispositions,
                "full_traversal_report_uri": traversal,
            }
        )
    for key, entry in resolved_entries.items():
        if not Path(entry.source_sft_uri).is_file():
            raise FileNotFoundError(f"Missing accepted SFT for {key}: {entry.source_sft_uri}")
        if not Path(entry.generation_manifest_uri).is_file():
            raise FileNotFoundError(f"Missing generation manifest for {key}: {entry.generation_manifest_uri}")
        lineage = entry.selection_manifest_uri or entry.full_traversal_report_uri
        if lineage is None or not Path(lineage).is_file():
            raise FileNotFoundError(f"Missing lineage artifact for {key}: {lineage}")
        expected = trial_by_group[key]
        if entry.arm_label != expected.arm_label:
            raise ValueError(
                f"SFT registry arm label does not match experiment for {key}: "
                f"{entry.arm_label} != {expected.arm_label}"
            )
        if expected.arm == "proxy" and not entry.selection_manifest_uri:
            raise ValueError(f"Proxy data group {key} requires a selection manifest")
        if expected.arm == "proxy":
            proxy_inputs = {
                "proxy units": entry.proxy_units_uri,
                "coverage obligations": entry.coverage_obligations_uri,
                "unit dispositions": entry.dispositions_uri,
            }
            missing_proxy_inputs = [name for name, uri in proxy_inputs.items() if not uri or not Path(uri).is_file()]
            if missing_proxy_inputs:
                raise FileNotFoundError(
                    f"Proxy data group {key} is missing replay/production inputs: {missing_proxy_inputs}"
                )
        if expected.arm == "reference" and not entry.full_traversal_report_uri:
            raise ValueError(f"Reference data group {key} requires a full-traversal report")
        if expected.arm == "reference":
            reference_inputs = {
                "eligible units": entry.eligible_units_uri,
                "Fully Open Meditron run manifest": entry.upstream_run_manifest_uri,
                "unit dispositions": entry.dispositions_uri,
            }
            missing_reference_inputs = [
                name for name, uri in reference_inputs.items() if not uri or not Path(uri).is_file()
            ]
            if missing_reference_inputs:
                raise FileNotFoundError(
                    f"Reference data group {key} is missing traversal replay inputs: {missing_reference_inputs}"
                )

    # The two branches start from the same frozen raw batch, but their SFT is
    # produced independently.  A shared path would recreate the old
    # "proxy is a filtered reference SFT" implementation.  Equal non-empty
    # bytes are also rejected because they provide no evidence of independent
    # branch production; two genuinely empty low-yield files remain valid.
    for batch_id in experiment_manifest.batch_content_sha256:
        reference_entries = [
            entry
            for key, entry in resolved_entries.items()
            if key[0] == batch_id and trial_by_group[key].arm == "reference"
        ]
        if len(reference_entries) != 1:
            raise ValueError(f"Batch {batch_id} must register exactly one full reference data group")
        reference_path = Path(reference_entries[0].source_sft_uri).resolve()
        reference_sha = file_sha256(reference_path)
        for key, proxy_entry in resolved_entries.items():
            if key[0] != batch_id or trial_by_group[key].arm != "proxy":
                continue
            proxy_path = Path(proxy_entry.source_sft_uri).resolve()
            if proxy_path == reference_path:
                raise ValueError(f"Proxy and reference branches share the same SFT file for batch {batch_id}")
            if proxy_path.stat().st_size > 0 and reference_path.stat().st_size > 0:
                if file_sha256(proxy_path) == reference_sha:
                    raise ValueError(f"Proxy and reference branches have identical non-empty SFT for batch {batch_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "PREPARING.json",
        {
            "experiment_id": experiment_manifest.experiment_id,
            "started_at": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        },
    )
    if source_deepspeed is not None:
        runtime_dir = output_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        deepspeed_path = runtime_dir / source_deepspeed.name
        shutil.copyfile(source_deepspeed, deepspeed_path)
        deepspeed_sha = file_sha256(deepspeed_path)
        if deepspeed_sha != file_sha256(source_deepspeed):
            raise OSError("Copied DeepSpeed config does not match its source")
        profile = profile.model_copy(update={"deepspeed_config": str(deepspeed_path)})
    datasets_root = output_dir / "datasets"
    jobs_root = output_dir / "runs"
    tokenizer = _load_tokenizer(contract, profile)
    num_epochs = hp.num_train_epochs
    audits: dict[tuple[str, str], tuple[TrainingDataAudit, Path]] = {}
    for key in sorted(resolved_entries):
        entry = resolved_entries[key]
        trial = trial_by_group[key]
        dataset_dir = datasets_root / _safe_component(entry.batch_id) / _safe_component(entry.arm_id)
        audit = _convert_training_data(
            entry=entry,
            source_path=Path(entry.source_sft_uri),
            destination_dir=dataset_dir,
            tokenizer=tokenizer,
            contract=contract,
            trial=trial,
            num_train_epochs=num_epochs,
            preprocessing_num_workers=profile.preprocessing_num_workers,
            preprocessing_batch_size=profile.preprocessing_batch_size,
        )
        audit_path = dataset_dir / "data_audit.json"
        write_json(audit_path, audit.model_dump(mode="json"))
        audits[key] = (audit, audit_path)

    jobs: list[LlamaFactoryJob] = []
    for job_index, trial in enumerate(experiment_manifest.trials):
        audit, audit_path = audits[(trial.batch_id, trial.arm_id)]
        run_dir = jobs_root / trial.run_id
        config_dir = run_dir / "config"
        adapter_dir = run_dir / "adapter"
        config_path = config_dir / "train.yaml"
        no_op = audit.effective_train_tokens == 0
        config_sha: str | None = None
        argv: list[str] = []
        if not no_op:
            config_dir.mkdir(parents=True, exist_ok=False)
            config = _training_config(
                trial=trial,
                contract=contract,
                profile=profile,
                dataset_dir=Path(audit.training_sft_uri).parent,
                adapter_dir=adapter_dir,
            )
            atomic_text(config_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            config_sha = file_sha256(config_path)
            argv = [python_executable, "-m", "llamafactory.cli", "train", str(config_path)]
        selection_sha = audit.lineage_artifact_sha256 if trial.arm == "proxy" else None
        traversal_sha = audit.lineage_artifact_sha256 if trial.arm == "reference" else None
        jobs.append(
            LlamaFactoryJob(
                job_index=job_index,
                run_id=trial.run_id,
                experiment_id=trial.experiment_id,
                protocol_sha256=trial.protocol_sha256,
                suite_sha256=trial.suite_sha256,
                contract_sha256=trial.contract_sha256,
                benchmark_sha256=trial.benchmark_sha256,
                batch_id=trial.batch_id,
                batch_content_sha256=trial.batch_content_sha256,
                arm_id=trial.arm_id,
                arm_label=trial.arm_label,
                role=trial.role,
                arm=trial.arm,
                method=trial.method,
                generator=trial.generator,
                generator_revision=trial.generator_revision,
                generation_recipe_sha256=trial.generation_recipe_sha256,
                budget_id=trial.budget_id,
                seed=trial.seed,
                selection_token_budget=trial.selection_token_budget,
                selection_seed=trial.selection_seed,
                selection_config_sha256=trial.selection_config_sha256,
                execution_mode="no_op" if no_op else "train",
                data_audit_uri=str(audit_path),
                data_audit_sha256=file_sha256(audit_path),
                generation_manifest_sha256=audit.generation_manifest_sha256,
                accepted_sft_sha256=audit.training_sft_sha256,
                selection_manifest_sha256=selection_sha,
                proxy_units_sha256=audit.proxy_units_sha256,
                coverage_obligations_sha256=audit.coverage_obligations_sha256,
                eligible_units_sha256=audit.eligible_units_sha256,
                dispositions_sha256=audit.dispositions_sha256,
                full_traversal_report_sha256=traversal_sha,
                effective_train_tokens=audit.effective_train_tokens,
                input_token_count=audit.input_token_count,
                packed_token_count=audit.packed_token_count,
                max_effective_train_tokens=trial.max_effective_train_tokens,
                train_config_uri=None if no_op else str(config_path),
                train_config_sha256=config_sha,
                run_dir=str(run_dir),
                adapter_dir=None if no_op else str(adapter_dir),
                base_model=contract.training.base_model,
                base_model_revision=contract.training.base_model_revision,
                llamafactory_root=str(llamafactory_root),
                llamafactory_revision=profile.llamafactory_revision,
                llamafactory_commit=commit,
                python_executable=python_executable,
                llamafactory_package_uri=str(package_file),
                runtime_packages=runtime_packages,
                project_code_sha256=project_code_sha256,
                world_size=profile.world_size,
                global_batch_size=profile.global_batch_size,
                deepspeed_config_uri=str(deepspeed_path) if deepspeed_path else None,
                deepspeed_config_sha256=deepspeed_sha,
                argv=argv,
            )
        )

    plan = LlamaFactoryJobManifest(
        experiment_id=experiment_manifest.experiment_id,
        experiment_manifest_sha256=file_sha256(experiment_manifest_path),
        contract_sha256=contract_digest(contract),
        profile_sha256=server_profile_digest(load_server_profile(profile_path)),
        llamafactory_revision=profile.llamafactory_revision,
        llamafactory_commit=commit,
        runtime_packages=runtime_packages,
        project_code=project_code,
        project_code_sha256=project_code_sha256,
        template_sha256=qwen_template_digest(),
        expected_job_count=experiment_manifest.expected_trial_count,
        jobs=jobs,
    )
    write_json(output_dir / "llamafactory_jobs.json", plan.model_dump(mode="json"))
    write_jsonl(output_dir / "llamafactory_jobs.jsonl", (job.model_dump(mode="json") for job in jobs))
    write_jsonl(
        output_dir / "evaluation_requests.jsonl",
        (
            {
                "run_id": job.run_id,
                "job_index": job.job_index,
                "experiment_id": job.experiment_id,
                "protocol_sha256": job.protocol_sha256,
                "suite_sha256": job.suite_sha256,
                "contract_sha256": job.contract_sha256,
                "project_code_sha256": job.project_code_sha256,
                "batch_id": job.batch_id,
                "batch_content_sha256": job.batch_content_sha256,
                "arm_id": job.arm_id,
                "arm_label": job.arm_label,
                "role": job.role,
                "arm": job.arm,
                "method": job.method,
                "generator": job.generator,
                "generator_revision": job.generator_revision,
                "generation_recipe_sha256": job.generation_recipe_sha256,
                "generation_manifest_sha256": job.generation_manifest_sha256,
                "budget_id": job.budget_id,
                "seed": job.seed,
                "selection_token_budget": job.selection_token_budget,
                "max_effective_train_tokens": job.max_effective_train_tokens,
                "selection_seed": job.selection_seed,
                "selection_config_sha256": job.selection_config_sha256,
                "selection_manifest_sha256": job.selection_manifest_sha256,
                "proxy_units_sha256": job.proxy_units_sha256,
                "coverage_obligations_sha256": job.coverage_obligations_sha256,
                "eligible_units_sha256": job.eligible_units_sha256,
                "dispositions_sha256": job.dispositions_sha256,
                "full_traversal_report_sha256": job.full_traversal_report_sha256,
                "accepted_sft_sha256": job.accepted_sft_sha256,
                "data_audit_sha256": job.data_audit_sha256,
                "effective_train_tokens": job.effective_train_tokens,
                "execution_mode": job.execution_mode,
                "model_name_or_path": contract.training.base_model,
                "model_revision": contract.training.base_model_revision,
                "adapter_name_or_path": job.adapter_dir,
                "checkpoint_manifest_uri": str(Path(job.run_dir) / "checkpoint_manifest.json"),
                "training_completion_uri": str(Path(job.run_dir) / "TRAINING_COMPLETE.json"),
                "template": profile.template,
                "benchmark_id": contract.benchmark.benchmark_id,
                "benchmark_version": contract.benchmark.version,
                "benchmark_sha256": job.benchmark_sha256,
                "evaluation_engine": contract.benchmark.evaluation_engine,
                "evaluation_engine_revision": contract.benchmark.evaluation_engine_revision,
                "primary_metric": contract.benchmark.primary_metric,
                "decoding": contract.benchmark.decoding,
                "expected_benchmark_item_count": next(
                    trial.expected_benchmark_item_count
                    for trial in experiment_manifest.trials
                    if trial.run_id == job.run_id
                ),
                "no_op_uses_base_model": job.execution_mode == "no_op",
                "request_state": "planned_waiting_for_training_receipt",
            }
            for job in jobs
        ),
    )
    script = """#!/usr/bin/env bash
set -euo pipefail
: "${AI_READY_ROOT:?Set AI_READY_ROOT to the ai-ready-benchmark checkout}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${AI_READY_ROOT}/tools/value_data.py" run-llamafactory \
  --jobs "${BUNDLE_DIR}/llamafactory_jobs.json" \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --resume
"""
    atomic_text(output_dir / "run_slurm_array.sh", script)
    (output_dir / "run_slurm_array.sh").chmod(0o755)
    ready = {
        "schema_version": "llamafactory-bundle-ready-v1",
        "experiment_id": plan.experiment_id,
        "job_manifest_sha256": file_sha256(output_dir / "llamafactory_jobs.json"),
        "job_jsonl_sha256": file_sha256(output_dir / "llamafactory_jobs.jsonl"),
        "evaluation_requests_sha256": file_sha256(output_dir / "evaluation_requests.jsonl"),
        "slurm_script_sha256": file_sha256(output_dir / "run_slurm_array.sh"),
        "project_code_sha256": plan.project_code_sha256,
        "prepared_at": _utc_now(),
    }
    write_json(output_dir / "BUNDLE_READY.json", ready)
    (output_dir / "PREPARING.json").unlink()
    return plan


def _hash_directory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return files


def _latest_checkpoint(adapter_dir: Path, world_size: int) -> Path | None:
    checkpoints = []
    for path in adapter_dir.glob("checkpoint-*"):
        if not path.is_dir() or not (path / "trainer_state.json").is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        # Optimizer/scheduler state is necessary for a real resume.  A
        # model-only checkpoint must never be presented as an exact restart.
        has_optimizer = (path / "optimizer.pt").is_file() or any(path.rglob("*optim*.pt"))
        has_scheduler = (path / "scheduler.pt").is_file() or any(path.rglob("*scheduler*.pt"))
        has_adapter_config = (path / "adapter_config.json").is_file()
        has_adapter_weights = any(path.glob("adapter_model*.safetensors")) or any(path.glob("adapter_model*.bin"))
        rng_files = list(path.glob("rng_state*.pth"))
        has_rng_state = (path / "rng_state.pth").is_file() if world_size == 1 else len(rng_files) >= world_size
        if has_optimizer and has_scheduler and has_adapter_config and has_adapter_weights and has_rng_state:
            checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def _write_status(run_dir: Path, state: str, **details: Any) -> None:
    write_json(
        run_dir / "status.json",
        {
            "state": state,
            "updated_at": _utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **details,
        },
    )


def _verify_job_inputs(job: LlamaFactoryJob, plan: LlamaFactoryJobManifest) -> None:
    if job.project_code_sha256 != plan.project_code_sha256:
        raise ValueError(f"Job does not use the bundle project code snapshot: {job.run_id}")
    if _project_code_digest(plan.project_code) != plan.project_code_sha256:
        raise ValueError(f"Bundle project code snapshot digest is invalid: {job.run_id}")
    _verify_project_code_snapshot(plan.project_code)
    if file_sha256(Path(job.data_audit_uri)) != job.data_audit_sha256:
        raise ValueError(f"Data audit changed after planning: {job.run_id}")
    audit = TrainingDataAudit.model_validate(_load_json_object(Path(job.data_audit_uri)))
    expected_identity = (job.experiment_id, job.batch_id, job.arm_id, job.arm_label)
    observed_identity = (audit.experiment_id, audit.batch_id, audit.arm_id, audit.arm_label)
    if observed_identity != expected_identity:
        raise ValueError(f"Data audit identity does not match job {job.run_id}: {observed_identity}")
    if audit.template_sha256 != plan.template_sha256:
        raise ValueError(f"Data audit template does not match job manifest: {job.run_id}")
    if audit.effective_train_tokens != job.effective_train_tokens:
        raise ValueError(f"Data audit token count does not match job: {job.run_id}")
    if (
        audit.generator != job.generator
        or audit.generator_revision != job.generator_revision
        or audit.generation_recipe_sha256 != job.generation_recipe_sha256
        or audit.generation_manifest_sha256 != job.generation_manifest_sha256
    ):
        raise ValueError(f"Data audit generation branch does not match job: {job.run_id}")
    if file_sha256(Path(audit.generation_manifest_uri)) != job.generation_manifest_sha256:
        raise ValueError(f"Generation manifest changed after planning: {job.run_id}")
    if audit.upstream_run_manifest_uri is not None:
        if file_sha256(Path(audit.upstream_run_manifest_uri)) != audit.upstream_run_manifest_sha256:
            raise ValueError(f"Fully Open Meditron run manifest changed after planning: {job.run_id}")
    if file_sha256(Path(audit.training_sft_uri)) != job.accepted_sft_sha256:
        raise ValueError(f"Training SFT changed after planning: {job.run_id}")
    if file_sha256(Path(audit.dataset_info_uri)) != audit.dataset_info_sha256:
        raise ValueError(f"dataset_info.json changed after planning: {job.run_id}")
    dataset_info = _load_json_object(Path(audit.dataset_info_uri))
    dataset = dataset_info.get("ai_ready_sft")
    if not isinstance(dataset, dict) or dataset.get("file_name") != Path(audit.training_sft_uri).name:
        raise ValueError(f"dataset_info.json no longer points to the audited SFT: {job.run_id}")
    dataset_target = (Path(audit.dataset_info_uri).parent / str(dataset["file_name"])).resolve()
    if dataset_target != Path(audit.training_sft_uri).resolve():
        raise ValueError(f"dataset_info.json resolves to another training file: {job.run_id}")
    if file_sha256(Path(audit.provenance_uri)) != audit.provenance_sha256:
        raise ValueError(f"Training provenance changed after planning: {job.run_id}")
    if file_sha256(Path(audit.lineage_artifact_uri)) != audit.lineage_artifact_sha256:
        raise ValueError(f"Lineage artifact changed after planning: {job.run_id}")
    expected_lineage_sha = job.selection_manifest_sha256 if job.arm == "proxy" else job.full_traversal_report_sha256
    if audit.lineage_artifact_sha256 != expected_lineage_sha:
        raise ValueError(f"Data audit lineage hash does not match the planned job: {job.run_id}")
    production_inputs = (
        ("proxy units", audit.proxy_units_uri, audit.proxy_units_sha256, job.proxy_units_sha256),
        (
            "coverage obligations",
            audit.coverage_obligations_uri,
            audit.coverage_obligations_sha256,
            job.coverage_obligations_sha256,
        ),
        ("eligible units", audit.eligible_units_uri, audit.eligible_units_sha256, job.eligible_units_sha256),
        ("unit dispositions", audit.dispositions_uri, audit.dispositions_sha256, job.dispositions_sha256),
    )
    for artifact_name, artifact_uri, audit_sha, job_sha in production_inputs:
        if audit_sha != job_sha:
            raise ValueError(f"Audited {artifact_name} hash does not match planned job: {job.run_id}")
        if artifact_uri is not None and file_sha256(Path(artifact_uri)) != audit_sha:
            raise ValueError(f"Bundled {artifact_name} changed after planning: {job.run_id}")
    if job.execution_mode == "train" and file_sha256(Path(job.train_config_uri or "")) != job.train_config_sha256:
        raise ValueError(f"Training config changed after planning: {job.run_id}")
    if job.deepspeed_config_uri and file_sha256(Path(job.deepspeed_config_uri)) != job.deepspeed_config_sha256:
        raise ValueError(f"DeepSpeed config changed after planning: {job.run_id}")
    if _resolve_git_commit(Path(job.llamafactory_root), job.llamafactory_revision) != job.llamafactory_commit:
        raise ValueError(f"LLaMAFactory checkout changed after planning: {job.run_id}")
    package_file = _resolve_installed_llamafactory(Path(job.llamafactory_root), job.python_executable)
    if package_file != Path(job.llamafactory_package_uri).resolve():
        raise ValueError(f"Python imports a different LLaMAFactory package than the planned job: {job.run_id}")
    if _runtime_packages(job.python_executable) != job.runtime_packages:
        raise ValueError(f"Training runtime or CUDA fingerprint changed after bundle preparation: {job.run_id}")


def _verify_visible_devices(job: LlamaFactoryJob, env: dict[str, str]) -> None:
    """Keep world size and therefore global batch fixed across all trials."""

    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before launching a training job")
    device_ids = [value.strip() for value in visible.split(",") if value.strip() and value.strip() != "-1"]
    if len(device_ids) != job.world_size:
        raise RuntimeError(
            f"Run {job.run_id} expects world_size={job.world_size}, but CUDA_VISIBLE_DEVICES exposes {device_ids}"
        )


def _reject_alternate_launchers(env: dict[str, str]) -> None:
    """Refuse ambient switches that replace the preregistered v0.9.5 trainer path."""

    false_values = {"", "0", "false", "no", "off"}
    active = {
        name: env[name]
        for name in ("USE_RAY", "USE_MCA", "USE_KT", "USE_V1")
        if name in env and env[name].strip().lower() not in false_values
    }
    if active:
        raise RuntimeError(f"Unset alternate LLaMAFactory launcher switches before training: {active}")


def _collect_checkpoint_manifest(job: LlamaFactoryJob, invocation: list[str]) -> dict[str, Any]:
    run_dir = Path(job.run_dir)
    identity = {
        "experiment_id": job.experiment_id,
        "protocol_sha256": job.protocol_sha256,
        "suite_sha256": job.suite_sha256,
        "contract_sha256": job.contract_sha256,
        "benchmark_sha256": job.benchmark_sha256,
        "batch_id": job.batch_id,
        "batch_content_sha256": job.batch_content_sha256,
        "arm_id": job.arm_id,
        "arm_label": job.arm_label,
        "role": job.role,
        "arm": job.arm,
        "method": job.method,
        "generator": job.generator,
        "generator_revision": job.generator_revision,
        "generation_recipe_sha256": job.generation_recipe_sha256,
        "generation_manifest_sha256": job.generation_manifest_sha256,
        "budget_id": job.budget_id,
        "seed": job.seed,
        "selection_token_budget": job.selection_token_budget,
        "max_effective_train_tokens": job.max_effective_train_tokens,
        "selection_seed": job.selection_seed,
        "selection_config_sha256": job.selection_config_sha256,
        "base_model": job.base_model,
        "base_model_revision": job.base_model_revision,
        "data_audit_sha256": job.data_audit_sha256,
        "selection_manifest_sha256": job.selection_manifest_sha256,
        "proxy_units_sha256": job.proxy_units_sha256,
        "coverage_obligations_sha256": job.coverage_obligations_sha256,
        "eligible_units_sha256": job.eligible_units_sha256,
        "dispositions_sha256": job.dispositions_sha256,
        "full_traversal_report_sha256": job.full_traversal_report_sha256,
        "llamafactory_revision": job.llamafactory_revision,
        "llamafactory_commit": job.llamafactory_commit,
        "python_executable": job.python_executable,
        "llamafactory_package_uri": job.llamafactory_package_uri,
        "runtime_packages": job.runtime_packages,
        "project_code_sha256": job.project_code_sha256,
        "world_size": job.world_size,
        "global_batch_size": job.global_batch_size,
    }
    if job.execution_mode == "no_op":
        payload = {
            "schema_version": "llamafactory-checkpoint-manifest-v1",
            "run_id": job.run_id,
            "execution_mode": "no_op",
            "reason": "quality gate produced zero supervised tokens; evaluate the unchanged base model",
            "base_model_unchanged": True,
            "accepted_sft_sha256": job.accepted_sft_sha256,
            "effective_train_tokens": 0,
            **identity,
            "files": [],
            "created_at": _utc_now(),
        }
    else:
        adapter_dir = Path(job.adapter_dir or "")
        required_config = adapter_dir / "adapter_config.json"
        weights = list(adapter_dir.glob("adapter_model*.safetensors")) + list(adapter_dir.glob("adapter_model*.bin"))
        if not required_config.is_file() or not weights:
            raise FileNotFoundError(f"LLaMAFactory did not produce a complete LoRA adapter: {adapter_dir}")
        adapter_config = _load_json_object(required_config)
        if str(adapter_config.get("peft_type", "")).upper() != "LORA":
            raise ValueError(f"Adapter output is not a LoRA checkpoint: {adapter_dir}")
        if adapter_config.get("base_model_name_or_path") != job.base_model:
            raise ValueError(f"Adapter base model does not match the frozen contract: {adapter_dir}")
        train_config_path = Path(job.train_config_uri or "")
        train_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
        if not isinstance(train_config, dict):
            raise ValueError(f"Frozen training config is not a YAML mapping: {train_config_path}")
        expected_lora = {
            "r": int(train_config["lora_rank"]),
            "lora_alpha": float(train_config["lora_alpha"]),
            "lora_dropout": float(train_config["lora_dropout"]),
        }
        observed_lora = {
            "r": int(adapter_config.get("r", -1)),
            "lora_alpha": float(adapter_config.get("lora_alpha", -1)),
            "lora_dropout": float(adapter_config.get("lora_dropout", -1)),
        }
        if observed_lora != expected_lora:
            raise ValueError(
                f"Adapter LoRA parameters do not match the frozen training config: "
                f"expected={expected_lora}, observed={observed_lora}"
            )
        trainer_state_path = adapter_dir / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"LLaMAFactory output has no final trainer_state.json: {adapter_dir}")
        trainer_state = _load_json_object(trainer_state_path)
        if int(trainer_state.get("global_step", 0)) <= 0 or float(trainer_state.get("epoch", 0.0)) < 1.0 - 1e-6:
            raise ValueError(f"Training did not reach the preregistered one epoch: {adapter_dir}")
        payload = {
            "schema_version": "llamafactory-checkpoint-manifest-v1",
            "run_id": job.run_id,
            "execution_mode": "train",
            "train_config_sha256": job.train_config_sha256,
            "accepted_sft_sha256": job.accepted_sft_sha256,
            "effective_train_tokens": job.effective_train_tokens,
            "input_token_count": job.input_token_count,
            "packed_token_count": job.packed_token_count,
            **identity,
            "invocation": invocation,
            "trainer_global_step": int(trainer_state["global_step"]),
            "trainer_epoch": float(trainer_state["epoch"]),
            "files": _hash_directory(adapter_dir),
            "created_at": _utc_now(),
        }
    manifest_path = run_dir / "checkpoint_manifest.json"
    write_json(manifest_path, payload)
    return {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "payload": payload}


def _verify_checkpoint_manifest(job: LlamaFactoryJob, manifest_path: Path) -> dict[str, Any]:
    """Re-hash a completed adapter before reporting a training job as ready."""

    payload = _load_json_object(manifest_path)
    expected = {
        "schema_version": "llamafactory-checkpoint-manifest-v1",
        "run_id": job.run_id,
        "experiment_id": job.experiment_id,
        "protocol_sha256": job.protocol_sha256,
        "suite_sha256": job.suite_sha256,
        "contract_sha256": job.contract_sha256,
        "benchmark_sha256": job.benchmark_sha256,
        "batch_id": job.batch_id,
        "batch_content_sha256": job.batch_content_sha256,
        "arm_id": job.arm_id,
        "arm_label": job.arm_label,
        "role": job.role,
        "arm": job.arm,
        "method": job.method,
        "generator": job.generator,
        "generator_revision": job.generator_revision,
        "generation_recipe_sha256": job.generation_recipe_sha256,
        "generation_manifest_sha256": job.generation_manifest_sha256,
        "budget_id": job.budget_id,
        "seed": job.seed,
        "selection_token_budget": job.selection_token_budget,
        "max_effective_train_tokens": job.max_effective_train_tokens,
        "selection_seed": job.selection_seed,
        "selection_config_sha256": job.selection_config_sha256,
        "execution_mode": job.execution_mode,
        "base_model": job.base_model,
        "base_model_revision": job.base_model_revision,
        "data_audit_sha256": job.data_audit_sha256,
        "accepted_sft_sha256": job.accepted_sft_sha256,
        "selection_manifest_sha256": job.selection_manifest_sha256,
        "proxy_units_sha256": job.proxy_units_sha256,
        "coverage_obligations_sha256": job.coverage_obligations_sha256,
        "eligible_units_sha256": job.eligible_units_sha256,
        "dispositions_sha256": job.dispositions_sha256,
        "full_traversal_report_sha256": job.full_traversal_report_sha256,
        "effective_train_tokens": job.effective_train_tokens,
        "llamafactory_revision": job.llamafactory_revision,
        "llamafactory_commit": job.llamafactory_commit,
        "python_executable": job.python_executable,
        "llamafactory_package_uri": job.llamafactory_package_uri,
        "runtime_packages": job.runtime_packages,
        "project_code_sha256": job.project_code_sha256,
        "world_size": job.world_size,
        "global_batch_size": job.global_batch_size,
    }
    if job.execution_mode == "train":
        expected.update(
            {
                "train_config_sha256": job.train_config_sha256,
                "input_token_count": job.input_token_count,
                "packed_token_count": job.packed_token_count,
            }
        )
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint manifest identity mismatch for {job.run_id}: {mismatches}")
    if job.execution_mode == "train":
        adapter_dir = Path(job.adapter_dir or "")
        if payload.get("files") != _hash_directory(adapter_dir):
            raise ValueError(f"LoRA adapter files changed after checkpoint manifest creation: {job.run_id}")
    elif payload.get("files") != [] or payload.get("base_model_unchanged") is not True:
        raise ValueError(f"No-op checkpoint manifest is invalid: {job.run_id}")
    return payload


def _verify_training_receipt(job: LlamaFactoryJob, receipt: dict[str, Any], checkpoint_path: Path) -> None:
    """Bind the terminal receipt to one job and one verified checkpoint manifest."""

    expected = {
        "schema_version": "llamafactory-training-completion-v1",
        "run_id": job.run_id,
        "execution_mode": job.execution_mode,
        "checkpoint_manifest_sha256": file_sha256(checkpoint_path),
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Training receipt identity mismatch for {job.run_id}: {mismatches}")
    if not isinstance(receipt.get("completed_at"), str) or not receipt["completed_at"].strip():
        raise ValueError(f"Training receipt has no completion timestamp: {job.run_id}")


def run_llamafactory_job(
    jobs_path: str | Path,
    *,
    run_id: str | None = None,
    job_index: int | None = None,
    resume: bool = False,
    reclaim_stale_lock: bool = False,
    restart_incomplete: bool = False,
) -> dict[str, Any]:
    """Run exactly one job selected by immutable run ID or zero-based index."""

    if (run_id is None) == (job_index is None):
        raise ValueError("Specify exactly one of run_id and job_index")
    plan = load_job_manifest(jobs_path)
    if run_id is not None:
        matching = [job for job in plan.jobs if job.run_id == run_id]
        if not matching:
            raise KeyError(f"Unknown run_id: {run_id}")
        job = matching[0]
    else:
        if job_index is None or not 0 <= job_index < len(plan.jobs):
            raise IndexError(f"job_index must be between 0 and {len(plan.jobs) - 1}")
        job = plan.jobs[job_index]
    run_dir = Path(job.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "TRAINING_COMPLETE.json"
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    lock_path = run_dir / ".run.lock"
    if lock_path.exists():
        if not reclaim_stale_lock:
            raise RuntimeError(f"Run is already claimed or has a stale lock: {job.run_id}")
        lock = _load_json_object(lock_path)
        lock_host = str(lock.get("host", ""))
        lock_pid = int(lock.get("pid", -1))
        if lock_host == socket.gethostname() and lock_pid > 0:
            try:
                os.kill(lock_pid, 0)
            except ProcessLookupError:
                pass
            except PermissionError as error:
                raise RuntimeError(f"Cannot prove that lock PID {lock_pid} has stopped") from error
            else:
                raise RuntimeError(f"Refusing to reclaim a live local process lock: pid={lock_pid}")
        reclaimed = run_dir / f"run.lock.reclaimed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        os.replace(lock_path, reclaimed)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"Run is already claimed or has a stale lock: {job.run_id}") from error
    invocation = list(job.argv)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "claimed_at": _utc_now()}))
        # Input verification and terminal-receipt recovery happen while the
        # exclusive run lock is held. This makes corrupt-input failures visible
        # in status.json and prevents two schedulers from recovering one receipt.
        _verify_job_inputs(job, plan)
        if complete_path.is_file() != checkpoint_path.is_file():
            if checkpoint_path.is_file() and not complete_path.exists():
                # A crash between manifest verification and receipt creation is
                # recoverable without another optimizer step.
                checkpoint_payload = _verify_checkpoint_manifest(job, checkpoint_path)
                receipt = {
                    "schema_version": "llamafactory-training-completion-v1",
                    "run_id": job.run_id,
                    "completed_at": checkpoint_payload["created_at"],
                    "checkpoint_manifest_sha256": file_sha256(checkpoint_path),
                    "execution_mode": job.execution_mode,
                    "recovered_receipt": True,
                }
                _verify_training_receipt(job, receipt, checkpoint_path)
                write_json(complete_path, receipt)
                _write_status(run_dir, "TRAINED", **receipt)
                return {"state": "TRAINED", **receipt}
            raise ValueError(f"Training receipt and checkpoint manifest are incomplete for {job.run_id}")
        if complete_path.is_file() and checkpoint_path.is_file():
            receipt = _load_json_object(complete_path)
            _verify_training_receipt(job, receipt, checkpoint_path)
            _verify_checkpoint_manifest(job, checkpoint_path)
            return {**receipt, "run_id": job.run_id, "state": "TRAINED", "skipped": True}
        _write_status(run_dir, "TRAINING" if job.execution_mode == "train" else "NO_OP")
        if job.execution_mode == "train":
            adapter_dir = Path(job.adapter_dir or "")
            if adapter_dir.exists() and any(adapter_dir.iterdir()):
                checkpoint = _latest_checkpoint(adapter_dir, job.world_size) if resume else None
                if checkpoint is None:
                    if not restart_incomplete:
                        raise FileExistsError(
                            f"Adapter directory is non-empty and has no resumable checkpoint: {adapter_dir}. "
                            "Use --restart-incomplete to archive it and restart from the base model."
                        )
                    abandoned = adapter_dir.parent / (
                        f"{adapter_dir.name}.abandoned-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    os.replace(adapter_dir, abandoned)
                    _write_status(run_dir, "RESTARTING", abandoned_adapter_dir=str(abandoned))
                else:
                    invocation.append(f"resume_from_checkpoint={checkpoint}")
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / "train.log"
            env = os.environ.copy()
            _verify_visible_devices(job, env)
            _reject_alternate_launchers(env)
            if job.world_size > 1:
                env["FORCE_TORCHRUN"] = "1"
            else:
                # A login-shell override must not silently turn a frozen
                # single-GPU trial into a torchrun launch.
                env.pop("FORCE_TORCHRUN", None)
            python_bin = str(Path(job.python_executable).parent)
            env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
            env["NNODES"] = "1"
            env["NPROC_PER_NODE"] = str(job.world_size)
            env["NODE_RANK"] = "0"
            # Python reads this only at interpreter startup, so set it on the
            # LLaMAFactory subprocess rather than mutating the parent process.
            env["PYTHONHASHSEED"] = str(job.seed)
            # Dataset preprocessing caches are trial-local.  Reusing an
            # arbitrary host cache would weaken the data/config hash boundary.
            env["HF_DATASETS_CACHE"] = str(run_dir / "hf_datasets_cache")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    invocation,
                    cwd=job.llamafactory_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    check=False,
                )
            if result.returncode != 0:
                _write_status(run_dir, "FAILED", exit_code=result.returncode, invocation=invocation)
                raise RuntimeError(f"LLaMAFactory failed for {job.run_id} with exit code {result.returncode}")
        checkpoint = _collect_checkpoint_manifest(job, invocation)
        receipt = {
            "schema_version": "llamafactory-training-completion-v1",
            "run_id": job.run_id,
            "completed_at": _utc_now(),
            "checkpoint_manifest_sha256": checkpoint["sha256"],
            "execution_mode": job.execution_mode,
        }
        _verify_checkpoint_manifest(job, Path(checkpoint["path"]))
        _verify_training_receipt(job, receipt, Path(checkpoint["path"]))
        write_json(complete_path, receipt)
        _write_status(run_dir, "TRAINED", **receipt)
        return {"state": "TRAINED", **receipt}
    except Exception as error:
        status_path = run_dir / "status.json"
        existing_state = None
        if status_path.is_file():
            try:
                existing_state = _load_json_object(status_path).get("state")
            except Exception:
                existing_state = None
        # Preserve the subprocess exit code already written above.  Failures
        # in CUDA checks, resume handling, or checkpoint auditing still receive
        # an explicit terminal status instead of looking perpetually TRAINING.
        if existing_state != "FAILED":
            _write_status(
                run_dir,
                "FAILED",
                error_type=type(error).__name__,
                error_message=str(error),
                invocation=invocation,
            )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def summarize_llamafactory_jobs(jobs_path: str | Path) -> dict[str, Any]:
    plan = load_job_manifest(jobs_path)
    states = Counter()
    rows = []
    for job in plan.jobs:
        run_dir = Path(job.run_dir)
        completion = run_dir / "TRAINING_COMPLETE.json"
        checkpoint_manifest = run_dir / "checkpoint_manifest.json"
        if completion.is_file() and checkpoint_manifest.is_file():
            try:
                receipt = _load_json_object(completion)
                _verify_training_receipt(job, receipt, checkpoint_manifest)
                _verify_checkpoint_manifest(job, checkpoint_manifest)
                state = "TRAINED"
            except Exception:
                state = "CORRUPT_TRAINING_ARTIFACT"
        elif checkpoint_manifest.is_file():
            try:
                _verify_checkpoint_manifest(job, checkpoint_manifest)
                state = "RECOVERABLE_RECEIPT"
            except Exception:
                state = "CORRUPT_TRAINING_ARTIFACT"
        elif completion.is_file():
            state = "CORRUPT_TRAINING_ARTIFACT"
        elif (run_dir / ".run.lock").is_file():
            state = "RUNNING_OR_STALE_LOCK"
        elif (run_dir / "status.json").is_file():
            try:
                recorded_state = str(_load_json_object(run_dir / "status.json").get("state", "UNKNOWN"))
                if recorded_state == "TRAINED":
                    state = "CORRUPT_TRAINING_ARTIFACT"
                elif recorded_state in {"TRAINING", "NO_OP", "RESTARTING"}:
                    state = "INTERRUPTED"
                else:
                    state = recorded_state
            except Exception:
                state = "CORRUPT_STATUS"
        else:
            state = "PENDING"
        states[state] += 1
        rows.append({"job_index": job.job_index, "run_id": job.run_id, "state": state})
    return {
        "experiment_id": plan.experiment_id,
        "expected_jobs": plan.expected_job_count,
        "state_counts": dict(sorted(states.items())),
        "jobs": rows,
    }


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    return ExperimentManifest.model_validate(_load_json_object(Path(path).resolve()))


def profile_summary(path: str | Path) -> dict[str, Any]:
    profile = load_server_profile(path)
    return {
        **profile.model_dump(mode="json"),
        "global_batch_size": profile.global_batch_size,
        "profile_sha256": server_profile_digest(profile),
        "profile_file_sha256": file_sha256(Path(path).resolve()),
        "qwen_template_sha256": qwen_template_digest(),
        "python": sys.version,
    }
