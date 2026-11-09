"""Auditable, locked-test-independent Base Model capability probes.

This module measures the frozen Base Model before any SFT run.  It deliberately
keeps the probe catalogue separate from the locked benchmark and records enough
information to reproduce the exact model, prompts, responses, and scores used
to construct a capability-level model profile.

The public entry point is :func:`run_model_probe_pipeline`.  Transformers is
imported lazily so catalogue validation, scoring, and unit tests do not require
loading model weights.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.utils.io import read_jsonl, write_json, write_jsonl
from ai_ready.valuation.conditional_value import ScoredProbe
from ai_ready.valuation.contracts import file_sha256, load_production_contract

SHA256_PATTERN = r"^[0-9a-f]{64}$"
CHOICE_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
FORBIDDEN_LOCKED_PATH_PARTS = {"benchmark_locked", "locked_benchmark", "locked-test", "locked_test"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def canonical_sha256(value: Any) -> str:
    """Return a portable SHA-256 over canonical UTF-8 JSON."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_locked_path_marker(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(part in FORBIDDEN_LOCKED_PATH_PARTS for part in normalized.split("/") if part)


def _reject_locked_catalog_path(path: Path) -> None:
    # This check intentionally runs before opening the file.  A caller cannot
    # accidentally turn the locked benchmark itself into a probe catalogue.
    if _contains_locked_path_marker(str(path.resolve())):
        raise ValueError("Probe catalogue path points at a locked benchmark location; file was not read")


class ProbeSourceLocation(_StrictModel):
    """Frozen public-source location supporting one independent probe."""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    canonical_url: str = Field(min_length=1)
    version: str = Field(min_length=1)
    knowledge_cutoff: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    processed_sha256: str = Field(pattern=SHA256_PATTERN)
    blocks_path: str = Field(min_length=1)
    blocks_sha256: str = Field(pattern=SHA256_PATTERN)
    block_index_base: Literal[0]
    block_index: int = Field(ge=0)
    section: str = Field(min_length=1)
    source_block_text_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def reject_locked_sources(self):
        for value in (self.canonical_url, self.processed_path, self.blocks_path):
            if _contains_locked_path_marker(value):
                raise ValueError("Probe source locations cannot reference the locked benchmark")
        return self


class FourChoiceProbe(_StrictModel):
    """One source-grounded four-choice item outside the locked benchmark."""

    schema_version: Literal["cll-independent-capability-probe-v1"]
    probe_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    item_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    variant_id: Literal["base"]
    cell_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    answer_index_base: Literal[0]
    answer_text: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_evidence: str = Field(min_length=1)
    source_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_location: str = Field(min_length=1)
    source_location: ProbeSourceLocation
    temporal_risk: Literal["low", "moderate", "high"]
    locked_test: bool = False
    test_items_included: bool = False
    is_locked_test_item: bool = False
    derived_from_locked_test_item: bool = False

    @field_validator("question", "answer_text", "rationale", "source_evidence")
    @classmethod
    def strip_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields cannot be blank")
        return normalized

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str]) -> list[str]:
        normalized = [option.strip() for option in value]
        if any(not option for option in normalized):
            raise ValueError("Probe options cannot be blank")
        if len(set(normalized)) != 4:
            raise ValueError("Probe options must contain four distinct choices")
        return normalized

    @model_validator(mode="after")
    def validate_gold_and_independence(self):
        if (
            self.locked_test
            or self.test_items_included
            or self.is_locked_test_item
            or self.derived_from_locked_test_item
        ):
            raise ValueError("Locked benchmark items or content cannot be used as Base Model probes")
        if self.item_id != self.probe_id:
            raise ValueError("Independent catalogue item_id must equal probe_id")
        if self.answer_text != self.options[self.answer_index]:
            raise ValueError("answer_text must exactly equal options[answer_index]")
        if self.source_evidence_sha256 != _text_sha256(self.source_evidence):
            raise ValueError("source_evidence_sha256 does not match source_evidence")
        if self.source_sha256 != self.source_location.processed_sha256:
            raise ValueError("source_sha256 must equal source_location.processed_sha256")
        if self.source_file.replace("\\", "/") != self.source_location.processed_path.replace("\\", "/"):
            raise ValueError("source_file must equal source_location.processed_path")
        for value in (self.source_file, self.evidence_location):
            if _contains_locked_path_marker(value):
                raise ValueError("Probe evidence cannot reference the locked benchmark")
        return self


class ProbeCatalogue(_StrictModel):
    """Canonical in-memory probe set with uniqueness and lock checks."""

    schema_version: Literal["base-model-probe-catalogue-v1"] = "base-model-probe-catalogue-v1"
    probes: list[FourChoiceProbe] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_probe_ids(self):
        identifiers = [probe.probe_id for probe in self.probes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Probe catalogue probe_id values must be unique")
        self.probes = sorted(self.probes, key=lambda probe: probe.probe_id)
        return self


class CheckpointFile(_StrictModel):
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or normalized.startswith("./"):
            raise ValueError("checkpoint relative_path must be a normalized relative path")
        return normalized


class CheckpointManifest(_StrictModel):
    schema_version: Literal["base-model-checkpoint-manifest-v1"] = "base-model-checkpoint-manifest-v1"
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    files: list[CheckpointFile] = Field(min_length=1)
    total_size_bytes: int = Field(gt=0)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_file_inventory(self):
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Checkpoint manifest paths must be unique")
        self.files = sorted(self.files, key=lambda item: item.relative_path)
        if self.total_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("Checkpoint manifest total_size_bytes does not match file inventory")
        expected = checkpoint_content_digest(self.model_id, self.model_revision, self.files)
        if self.checkpoint_sha256 != expected:
            raise ValueError("checkpoint_sha256 does not match model identity and file inventory")
        return self


def checkpoint_content_digest(
    model_id: str, model_revision: str, files: Sequence[CheckpointFile | Mapping[str, Any]]
) -> str:
    parsed = [item if isinstance(item, CheckpointFile) else CheckpointFile.model_validate(item) for item in files]
    return canonical_sha256(
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "files": [item.model_dump(mode="json") for item in sorted(parsed, key=lambda row: row.relative_path)],
        }
    )


def build_checkpoint_manifest(snapshot_path: str | Path, *, model_id: str, model_revision: str) -> CheckpointManifest:
    """Hash every stable file in a local Hugging Face snapshot.

    Hugging Face ``local_dir`` downloads create a mutable ``.cache`` directory;
    it contains locks and transport metadata rather than model inputs, so it is
    excluded.  All other regular files, including every weight shard, are bound
    into ``checkpoint_sha256``.
    """

    root = Path(snapshot_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Local model snapshot does not exist: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0].lower() == ".cache":
            continue
        records.append(
            CheckpointFile(
                relative_path=relative.as_posix(),
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
            )
        )
    if not records:
        raise ValueError(f"Local model snapshot contains no stable files: {root}")
    digest = checkpoint_content_digest(model_id, model_revision, records)
    return CheckpointManifest(
        model_id=model_id,
        model_revision=model_revision,
        snapshot_path=str(root),
        files=records,
        total_size_bytes=sum(item.size_bytes for item in records),
        checkpoint_sha256=digest,
    )


class ProbeInferenceConfig(_StrictModel):
    """Frozen deterministic inference settings for the Base Model."""

    schema_version: Literal["base-model-probe-inference-v1"] = "base-model-probe-inference-v1"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    snapshot_path: Path
    source_root: Path = Path(".")
    production_contract_path: Path | None = None
    production_contract_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prompt_variant_ids: list[Literal["clinical_mcq_letter_v1", "clinical_mcq_letter_v2"]] = Field(
        default_factory=lambda: ["clinical_mcq_letter_v1", "clinical_mcq_letter_v2"], min_length=1
    )
    batch_size: int = Field(default=4, ge=1)
    max_new_tokens: int = Field(default=8, ge=1, le=32)
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device_map: Literal["auto", "cuda", "cpu"] = "auto"
    quantization: Literal["none", "bitsandbytes_int8", "bitsandbytes_nf4"] = "none"
    llm_int8_threshold: float = Field(default=6.0, gt=0)
    seed: int = 42
    trust_remote_code: Literal[False] = False
    local_files_only: Literal[True] = True

    @field_validator("prompt_variant_ids")
    @classmethod
    def unique_prompt_variants(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("prompt_variant_ids must be unique")
        return value

    @model_validator(mode="after")
    def require_fixed_identity(self):
        if self.model_revision.lower() in {"main", "master", "latest"}:
            raise ValueError("model_revision must be an immutable revision, not a moving branch")
        if (self.production_contract_path is None) != (self.production_contract_sha256 is None):
            raise ValueError("production_contract_path and production_contract_sha256 must be supplied together")
        return self


class ChatMessage(_StrictModel):
    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class ProbePrompt(_StrictModel):
    item_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    # displayed position -> original catalogue option index.  The second
    # prompt variant rotates options deterministically to expose letter bias.
    option_order: list[int] = Field(min_length=4, max_length=4)
    messages: list[ChatMessage] = Field(min_length=2, max_length=2)

    @field_validator("option_order")
    @classmethod
    def validate_option_order(cls, value: list[int]) -> list[int]:
        if sorted(value) != [0, 1, 2, 3]:
            raise ValueError("option_order must be a permutation of [0, 1, 2, 3]")
        return value


class BackendGeneration(_StrictModel):
    rendered_prompt: str = Field(min_length=1)
    raw_output: str


class BackendRuntime(_StrictModel):
    backend: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    transformers_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_available: bool
    model_class: str = Field(min_length=1)
    tokenizer_class: str = Field(min_length=1)
    chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    requested_dtype: str = Field(min_length=1)
    quantization: str = "none"
    quantization_backend_version: str | None = None
    parameter_dtype: str = Field(min_length=1)
    model_devices: list[str] = Field(min_length=1)


class ProbeBackend(Protocol):
    @property
    def runtime(self) -> BackendRuntime: ...

    def generate(
        self, prompts: Sequence[ProbePrompt], *, batch_size: int, max_new_tokens: int, seed: int
    ) -> list[BackendGeneration]: ...


class AnswerParseResult(_StrictModel):
    status: Literal["valid", "invalid", "ambiguous"]
    answer_letter: Literal["A", "B", "C", "D"] | None = None
    answer_index: int | None = Field(default=None, ge=0, le=3)

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.status == "valid":
            if self.answer_letter is None or self.answer_index is None:
                raise ValueError("A valid parse requires both answer_letter and answer_index")
            if CHOICE_LABELS[self.answer_index] != self.answer_letter:
                raise ValueError("answer_letter and answer_index disagree")
        elif self.answer_letter is not None or self.answer_index is not None:
            raise ValueError("Invalid or ambiguous parses cannot report an answer")
        return self


_FULL_ANSWER_PATTERN = re.compile(
    r"^\s*(?:(?:最终)?答案|answer|option|选项)?\s*[:：]?\s*[\(（\[【`]*\s*([A-DＡ-Ｄ])\s*[\)）\]】`]*\s*[\.。,:：;；!！]?\s*$",
    flags=re.IGNORECASE,
)
_STANDALONE_CHOICE_PATTERN = re.compile(r"(?<![A-Za-zＡ-Ｚａ-ｚ])([A-DＡ-Ｄ])(?![A-Za-zＡ-Ｚａ-ｚ])", re.IGNORECASE)


def parse_answer_choice(raw_output: str) -> AnswerParseResult:
    """Strictly parse a deterministic A-D response.

    A complete answer such as ``A``, ``(B)``, or ``答案：C`` is accepted.  Any
    explanatory text is rejected; if it mentions multiple distinct choices it
    is labelled ambiguous.  Invalid/ambiguous responses receive zero in the
    fixed scorer instead of being repaired by a model or heuristic.
    """

    normalized = raw_output.translate(str.maketrans("ＡＢＣＤ", "ABCD")).strip()
    match = _FULL_ANSWER_PATTERN.fullmatch(normalized)
    if match:
        letter = match.group(1).upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))
        return AnswerParseResult(status="valid", answer_letter=letter, answer_index=CHOICE_LABELS.index(letter))
    candidates = {
        value.upper().translate(str.maketrans("ＡＢＣＤ", "ABCD"))
        for value in _STANDALONE_CHOICE_PATTERN.findall(normalized)
    }
    return AnswerParseResult(status="ambiguous" if len(candidates) > 1 else "invalid")


class ProbePrediction(_StrictModel):
    schema_version: Literal["base-model-probe-prediction-v1"] = "base-model-probe-prediction-v1"
    prediction_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_set_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    option_order: list[int] = Field(min_length=4, max_length=4)
    raw_output: str
    parse_status: Literal["valid", "invalid", "ambiguous"]
    parsed_answer_letter: Literal["A", "B", "C", "D"] | None = None
    parsed_display_index: int | None = Field(default=None, ge=0, le=3)
    parsed_answer_index: int | None = Field(default=None, ge=0, le=3)
    is_locked_test_item: Literal[False] = False
    derived_from_locked_test_item: Literal[False] = False

    @model_validator(mode="after")
    def validate_parse(self):
        parsed = AnswerParseResult(
            status=self.parse_status,
            answer_letter=self.parsed_answer_letter,
            answer_index=self.parsed_display_index,
        )
        if sorted(self.option_order) != [0, 1, 2, 3]:
            raise ValueError("option_order must be a permutation of [0, 1, 2, 3]")
        if parsed.status == "valid":
            if self.parsed_answer_index != self.option_order[parsed.answer_index]:
                raise ValueError("parsed_answer_index does not match the displayed option order")
        elif self.parsed_answer_index is not None:
            raise ValueError("Invalid or ambiguous parses cannot report parsed_answer_index")
        return self


class ScoringProtocol(_StrictModel):
    schema_version: Literal["base-model-probe-scoring-v1"] = "base-model-probe-scoring-v1"
    method: Literal["exact_choice_index"] = "exact_choice_index"
    accepted_labels: tuple[Literal["A", "B", "C", "D"], ...] = CHOICE_LABELS
    invalid_response_score: Literal[0] = 0
    ambiguous_response_score: Literal[0] = 0
    prompt_variant_aggregation: Literal["mean_within_item_before_cell_aggregation"] = (
        "mean_within_item_before_cell_aggregation"
    )


class ProbeRunManifest(_StrictModel):
    schema_version: Literal["base-model-probe-run-manifest-v1"] = "base-model-probe-run-manifest-v1"
    run_id: str = Field(min_length=1)
    status: Literal["complete"] = "complete"
    started_at: datetime
    finished_at: datetime
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_manifest_uri: str = Field(min_length=1)
    checkpoint_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_catalogue_uri: str = Field(min_length=1)
    probe_catalogue_file_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_set_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_count: int = Field(ge=1)
    source_artifact_count: int = Field(ge=1)
    source_artifacts_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_count: int = Field(ge=1)
    inference_config: ProbeInferenceConfig
    inference_config_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_protocol_sha256: str = Field(pattern=SHA256_PATTERN)
    scoring_protocol: ScoringProtocol
    scoring_protocol_sha256: str = Field(pattern=SHA256_PATTERN)
    implementation_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime: BackendRuntime
    predictions_uri: str = Field(min_length=1)
    predictions_sha256: str = Field(pattern=SHA256_PATTERN)
    scored_probes_uri: str = Field(min_length=1)
    scored_probes_sha256: str = Field(pattern=SHA256_PATTERN)
    valid_prediction_count: int = Field(ge=0)
    invalid_prediction_count: int = Field(ge=0)
    ambiguous_prediction_count: int = Field(ge=0)
    benchmark_test_items_accessed: Literal[False] = False
    locked_test_items_used: Literal[False] = False
    probes_derived_from_locked_test_items: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts_and_hashes(self):
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        parse_total = self.valid_prediction_count + self.invalid_prediction_count + self.ambiguous_prediction_count
        if parse_total != self.prompt_count:
            raise ValueError("Prediction parse counts must sum to prompt_count")
        if self.inference_config_sha256 != canonical_sha256(self.inference_config):
            raise ValueError("inference_config_sha256 does not match inference_config")
        if self.scoring_protocol_sha256 != canonical_sha256(self.scoring_protocol):
            raise ValueError("scoring_protocol_sha256 does not match scoring_protocol")
        return self


class ProbeRunArtifacts(_StrictModel):
    output_dir: Path
    checkpoint_manifest_path: Path
    predictions_path: Path
    scored_probes_path: Path
    run_manifest_path: Path
    run_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    probe_set_sha256: str = Field(pattern=SHA256_PATTERN)


PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "clinical_mcq_letter_v1": {
        "system": "你是医学知识能力测量中的答题模型。只依据题目选择最合适的一个选项。",
        "instruction": "请选择唯一正确答案。只输出一个大写字母 A、B、C 或 D，不要解释。",
    },
    "clinical_mcq_letter_v2": {
        "system": "你正在完成一组独立的医学四选一能力探针。",
        "instruction": "判断后仅返回对应选项字母（A/B/C/D），不要复述题目或给出理由。",
    },
}


def prompt_protocol_digest(variant_ids: Sequence[str]) -> str:
    return canonical_sha256({variant_id: PROMPT_VARIANTS[variant_id] for variant_id in variant_ids})


def build_probe_prompts(probes: Sequence[FourChoiceProbe], variant_ids: Sequence[str]) -> list[ProbePrompt]:
    prompts = []
    for probe in sorted(probes, key=lambda item: item.probe_id):
        for variant_id in variant_ids:
            template = PROMPT_VARIANTS[variant_id]
            if variant_id == "clinical_mcq_letter_v1":
                option_order = [0, 1, 2, 3]
            else:
                # Never leave variant 2 in the original order.  The stable
                # item-specific rotation controls for answer-letter priors
                # without introducing a random, unrecorded transformation.
                offset = 1 + hashlib.sha256(probe.probe_id.encode("utf-8")).digest()[0] % 3
                option_order = [(index + offset) % 4 for index in range(4)]
            displayed_options = [probe.options[index] for index in option_order]
            rendered_options = "\n".join(
                f"{label}. {text}" for label, text in zip(CHOICE_LABELS, displayed_options, strict=True)
            )
            prompts.append(
                ProbePrompt(
                    item_id=probe.probe_id,
                    cell_id=probe.cell_id,
                    variant_id=variant_id,
                    option_order=option_order,
                    messages=[
                        ChatMessage(role="system", content=template["system"]),
                        ChatMessage(
                            role="user",
                            content=f"{probe.question}\n\n{rendered_options}\n\n{template['instruction']}",
                        ),
                    ],
                )
            )
    return prompts


class TransformersProbeBackend:
    """Local-only Hugging Face backend with deterministic batched generation."""

    def __init__(self, config: ProbeInferenceConfig):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self._config = config
        snapshot_path = str(config.snapshot_path.resolve())
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]
        common = {
            "revision": config.model_revision,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(snapshot_path, **common)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        device_map: str | dict[str, str]
        if config.device_map == "auto":
            device_map = "auto"
        elif config.device_map == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device_map=cuda was requested but torch.cuda.is_available() is false")
            device_map = {"": "cuda:0"}
        else:
            device_map = {"": "cpu"}
        quantization_config = None
        quantization_backend_version = None
        if config.quantization == "bitsandbytes_int8":
            if config.device_map == "cpu":
                raise ValueError("bitsandbytes_int8 requires CUDA or automatic device placement")
            import bitsandbytes

            quantization_backend_version = bitsandbytes.__version__
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=config.llm_int8_threshold,
            )
        elif config.quantization == "bitsandbytes_nf4":
            if config.device_map == "cpu":
                raise ValueError("bitsandbytes_nf4 requires CUDA or automatic device placement")
            import bitsandbytes

            quantization_backend_version = bitsandbytes.__version__
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self._model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            **common,
        )
        self._model.eval()
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in self._model.parameters()})
        devices = sorted(
            {str(parameter.device) for parameter in self._model.parameters() if parameter.device.type != "meta"}
        )
        chat_template = self._tokenizer.chat_template
        if not chat_template:
            raise ValueError("The frozen tokenizer has no chat_template; probe prompts cannot be reproduced")
        self._runtime = BackendRuntime(
            backend="transformers-local-causal-lm",
            python_version=platform.python_version(),
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            model_class=self._model.__class__.__name__,
            tokenizer_class=self._tokenizer.__class__.__name__,
            chat_template_sha256=_text_sha256(chat_template),
            requested_dtype=config.dtype,
            quantization=config.quantization,
            quantization_backend_version=quantization_backend_version,
            parameter_dtype=",".join(parameter_dtypes),
            model_devices=devices or ["unknown"],
        )

    @property
    def runtime(self) -> BackendRuntime:
        return self._runtime

    def _input_device(self):
        device = self._model.get_input_embeddings().weight.device
        if device.type == "meta":
            device = next(parameter.device for parameter in self._model.parameters() if parameter.device.type != "meta")
        return device

    def generate(
        self, prompts: Sequence[ProbePrompt], *, batch_size: int, max_new_tokens: int, seed: int
    ) -> list[BackendGeneration]:
        torch = self._torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        outputs: list[BackendGeneration] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            rendered = [
                self._tokenizer.apply_chat_template(
                    [message.model_dump(mode="json") for message in prompt.messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch
            ]
            encoded = self._tokenizer(rendered, padding=True, return_tensors="pt")
            encoded = {name: tensor.to(self._input_device()) for name, tensor in encoded.items()}
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            input_width = encoded["input_ids"].shape[1]
            for prompt_text, token_ids in zip(rendered, generated, strict=True):
                raw_output = self._tokenizer.decode(token_ids[input_width:], skip_special_tokens=True)
                outputs.append(BackendGeneration(rendered_prompt=prompt_text, raw_output=raw_output))
        return outputs


def load_probe_catalogue(path: str | Path) -> ProbeCatalogue:
    source = Path(path)
    _reject_locked_catalog_path(source)
    rows = read_jsonl(source)
    return ProbeCatalogue(probes=[FourChoiceProbe.model_validate(row) for row in rows])


def _resolve_public_source(root: Path, relative_path: str) -> Path:
    if _contains_locked_path_marker(relative_path):
        raise ValueError("Probe source path references a locked benchmark location")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Probe source escapes source_root: {relative_path}") from exc
    return candidate


def verify_probe_source_artifacts(catalogue: ProbeCatalogue, source_root: str | Path) -> list[dict[str, Any]]:
    """Verify public documents, block files, block hashes, and exact evidence."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Probe source_root does not exist: {root}")
    verified: dict[tuple[str, str], dict[str, Any]] = {}
    block_cache: dict[str, list[dict[str, Any]]] = {}
    file_hash_cache: dict[str, str] = {}
    for probe in catalogue.probes:
        location = probe.source_location
        processed = _resolve_public_source(root, location.processed_path)
        blocks = _resolve_public_source(root, location.blocks_path)
        if str(processed) not in file_hash_cache:
            file_hash_cache[str(processed)] = file_sha256(processed) if processed.is_file() else ""
        processed_hash = file_hash_cache[str(processed)]
        if processed_hash != location.processed_sha256:
            raise ValueError(f"Processed public source hash mismatch for probe {probe.probe_id}")
        if str(blocks) not in file_hash_cache:
            file_hash_cache[str(blocks)] = file_sha256(blocks) if blocks.is_file() else ""
        blocks_hash = file_hash_cache[str(blocks)]
        if blocks_hash != location.blocks_sha256:
            raise ValueError(f"Public source block hash mismatch for probe {probe.probe_id}")
        if str(blocks) not in block_cache:
            block_cache[str(blocks)] = read_jsonl(blocks)
        rows = block_cache[str(blocks)]
        if location.block_index >= len(rows):
            raise ValueError(f"Source block index is out of range for probe {probe.probe_id}")
        block_text = rows[location.block_index].get("text")
        if not isinstance(block_text, str):
            raise ValueError(f"Source block has no text for probe {probe.probe_id}")
        if _text_sha256(block_text) != location.source_block_text_sha256:
            raise ValueError(f"Source block text hash mismatch for probe {probe.probe_id}")
        if probe.source_evidence not in block_text:
            raise ValueError(f"Source evidence is absent from the registered block for probe {probe.probe_id}")
        key = (location.processed_path, location.blocks_path)
        verified[key] = {
            "source_id": location.source_id,
            "processed_path": location.processed_path,
            "processed_sha256": location.processed_sha256,
            "blocks_path": location.blocks_path,
            "blocks_sha256": location.blocks_sha256,
        }
    return [verified[key] for key in sorted(verified)]


def verify_production_contract_binding(config: ProbeInferenceConfig) -> dict[str, str] | None:
    """Verify that the measured model is exactly the one frozen for training."""

    if config.production_contract_path is None:
        return None
    path = config.production_contract_path.resolve()
    observed_sha256 = file_sha256(path)
    if observed_sha256 != config.production_contract_sha256:
        raise ValueError("Production contract SHA-256 does not match the probe inference config")
    contract = load_production_contract(path)
    if contract.training.base_model != config.model_id:
        raise ValueError("Probe model_id does not match production contract training.base_model")
    if contract.training.base_model_revision != config.model_revision:
        raise ValueError("Probe model_revision does not match the production contract")
    if contract.training.tokenizer_revision != config.model_revision:
        raise ValueError("Probe tokenizer revision does not match the production contract")
    return {"path": str(path), "sha256": observed_sha256}


def probe_catalogue_digest(catalogue: ProbeCatalogue | Sequence[FourChoiceProbe]) -> str:
    probes = catalogue.probes if isinstance(catalogue, ProbeCatalogue) else list(catalogue)
    return canonical_sha256([probe.model_dump(mode="json") for probe in sorted(probes, key=lambda row: row.probe_id)])


def build_predictions(
    *,
    prompts: Sequence[ProbePrompt],
    generations: Sequence[BackendGeneration],
    config: ProbeInferenceConfig,
    checkpoint_sha256: str,
    probe_set_sha256: str,
) -> list[ProbePrediction]:
    if len(prompts) != len(generations):
        raise ValueError(f"Backend returned {len(generations)} generations for {len(prompts)} prompts")
    predictions = []
    for prompt, generation in zip(prompts, generations, strict=True):
        parsed = parse_answer_choice(generation.raw_output)
        original_answer_index = prompt.option_order[parsed.answer_index] if parsed.answer_index is not None else None
        predictions.append(
            ProbePrediction(
                prediction_id=f"{prompt.item_id}:{prompt.variant_id}",
                item_id=prompt.item_id,
                variant_id=prompt.variant_id,
                cell_id=prompt.cell_id,
                model_id=config.model_id,
                model_revision=config.model_revision,
                checkpoint_sha256=checkpoint_sha256,
                probe_set_sha256=probe_set_sha256,
                prompt_sha256=_text_sha256(generation.rendered_prompt),
                option_order=prompt.option_order,
                raw_output=generation.raw_output,
                parse_status=parsed.status,
                parsed_answer_letter=parsed.answer_letter,
                parsed_display_index=parsed.answer_index,
                parsed_answer_index=original_answer_index,
            )
        )
    identifiers = [prediction.prediction_id for prediction in predictions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Generated prediction_id values are not unique")
    return predictions


def score_probe_predictions(
    probes: Sequence[FourChoiceProbe], predictions: Sequence[ProbePrediction]
) -> list[ScoredProbe]:
    """Apply the frozen exact-index rubric and emit profile-compatible rows."""

    by_id = {probe.probe_id: probe for probe in probes}
    if len(by_id) != len(probes):
        raise ValueError("Probe IDs must be unique before scoring")
    scored = []
    seen = set()
    for prediction in predictions:
        if prediction.prediction_id in seen:
            raise ValueError(f"Duplicate prediction_id: {prediction.prediction_id}")
        seen.add(prediction.prediction_id)
        probe = by_id.get(prediction.item_id)
        if probe is None:
            raise ValueError(f"Prediction references an unknown probe: {prediction.item_id}")
        if prediction.cell_id != probe.cell_id:
            raise ValueError(f"Prediction cell does not match probe {probe.probe_id}")
        score = float(prediction.parse_status == "valid" and prediction.parsed_answer_index == probe.answer_index)
        scored.append(
            ScoredProbe(
                probe_id=prediction.prediction_id,
                item_id=probe.probe_id,
                variant_id=prediction.variant_id,
                cell_id=probe.cell_id,
                score=score,
                is_locked_test_item=False,
            )
        )
    expected_items = {probe.probe_id for probe in probes}
    observed_items = {prediction.item_id for prediction in predictions}
    if expected_items != observed_items:
        missing = sorted(expected_items - observed_items)
        extra = sorted(observed_items - expected_items)
        raise ValueError(f"Prediction coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return sorted(scored, key=lambda row: row.probe_id)


def run_model_probe_pipeline(
    *,
    catalogue_path: str | Path,
    output_dir: str | Path,
    config: ProbeInferenceConfig,
    backend: ProbeBackend | None = None,
) -> ProbeRunArtifacts:
    """Run, score, and freeze a complete capability-probe evidence chain."""

    started_at = datetime.now(timezone.utc)
    catalogue_source = Path(catalogue_path).resolve()
    catalogue = load_probe_catalogue(catalogue_source)
    probe_set_sha256 = probe_catalogue_digest(catalogue)
    verify_production_contract_binding(config)
    verified_sources = verify_probe_source_artifacts(catalogue, config.source_root)
    checkpoint = build_checkpoint_manifest(
        config.snapshot_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "checkpoint_manifest.json"
    predictions_path = destination / "raw_predictions.jsonl"
    scored_path = destination / "scored_probes.jsonl"
    run_manifest_path = destination / "run_manifest.json"
    write_json(checkpoint_path, checkpoint.model_dump(mode="json"))

    prompts = build_probe_prompts(catalogue.probes, config.prompt_variant_ids)
    active_backend = backend or TransformersProbeBackend(config)
    generations = active_backend.generate(
        prompts,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
    )
    predictions = build_predictions(
        prompts=prompts,
        generations=generations,
        config=config,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        probe_set_sha256=probe_set_sha256,
    )
    write_jsonl(predictions_path, (row.model_dump(mode="json") for row in predictions))
    scored = score_probe_predictions(catalogue.probes, predictions)
    write_jsonl(scored_path, (row.model_dump(mode="json") for row in scored))

    scoring_protocol = ScoringProtocol()
    counts = {
        status: sum(row.parse_status == status for row in predictions) for status in ("valid", "invalid", "ambiguous")
    }
    manifest = ProbeRunManifest(
        run_id=config.run_id,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        model_id=config.model_id,
        model_revision=config.model_revision,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        checkpoint_manifest_uri=checkpoint_path.name,
        checkpoint_manifest_sha256=file_sha256(checkpoint_path),
        probe_catalogue_uri=str(catalogue_source),
        probe_catalogue_file_sha256=file_sha256(catalogue_source),
        probe_set_sha256=probe_set_sha256,
        probe_count=len(catalogue.probes),
        source_artifact_count=len(verified_sources),
        source_artifacts_sha256=canonical_sha256(verified_sources),
        prompt_count=len(prompts),
        inference_config=config,
        inference_config_sha256=canonical_sha256(config),
        prompt_protocol_sha256=prompt_protocol_digest(config.prompt_variant_ids),
        scoring_protocol=scoring_protocol,
        scoring_protocol_sha256=canonical_sha256(scoring_protocol),
        implementation_sha256=file_sha256(Path(__file__).resolve()),
        runtime=active_backend.runtime,
        predictions_uri=predictions_path.name,
        predictions_sha256=file_sha256(predictions_path),
        scored_probes_uri=scored_path.name,
        scored_probes_sha256=file_sha256(scored_path),
        valid_prediction_count=counts["valid"],
        invalid_prediction_count=counts["invalid"],
        ambiguous_prediction_count=counts["ambiguous"],
    )
    write_json(run_manifest_path, manifest.model_dump(mode="json"))
    return ProbeRunArtifacts(
        output_dir=destination,
        checkpoint_manifest_path=checkpoint_path,
        predictions_path=predictions_path,
        scored_probes_path=scored_path,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=file_sha256(run_manifest_path),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        probe_set_sha256=probe_set_sha256,
    )


__all__ = [
    "AnswerParseResult",
    "BackendGeneration",
    "BackendRuntime",
    "CheckpointFile",
    "CheckpointManifest",
    "FourChoiceProbe",
    "ProbeCatalogue",
    "ProbeInferenceConfig",
    "ProbePrediction",
    "ProbeRunArtifacts",
    "ProbeRunManifest",
    "ProbeSourceLocation",
    "ScoringProtocol",
    "TransformersProbeBackend",
    "build_checkpoint_manifest",
    "build_predictions",
    "build_probe_prompts",
    "canonical_sha256",
    "load_probe_catalogue",
    "parse_answer_choice",
    "probe_catalogue_digest",
    "prompt_protocol_digest",
    "run_model_probe_pipeline",
    "score_probe_predictions",
    "verify_probe_source_artifacts",
    "verify_production_contract_binding",
]
