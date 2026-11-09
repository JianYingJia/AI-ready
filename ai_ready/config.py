"""Validated configuration for the L1-to-L2 pipeline."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParsingConfig(StrictConfig):
    backend: Literal["auto", "native", "docling", "marker"] = "auto"
    complex_pdf_backend: Literal["docling", "marker"] = "marker"
    marker_mode: Literal["fast", "balanced"] = "fast"
    ocr: bool = True
    force_full_page_ocr: bool = False
    table_structure: bool = True
    continue_on_error: bool = False
    chunker: Literal["hybrid", "hierarchical"] = "hybrid"
    max_tokens: int = Field(default=384, ge=32)
    sliding_window_units: int = Field(default=0, ge=0, le=10)
    sliding_window_stride: int = Field(default=1, ge=1)


class RegistryConfig(StrictConfig):
    metadata_file: Path | None = None
    default_permissions: list[str] = Field(default_factory=lambda: ["research"])
    default_version: str = "unversioned"
    simhash_max_distance: int = Field(default=3, ge=0, le=32)

    @model_validator(mode="after")
    def validate_permissions(self):
        if not self.default_permissions or any(not value.strip() for value in self.default_permissions):
            raise ValueError("default_permissions requires nonempty access groups")
        return self


class EmbeddingConfig(StrictConfig):
    model_name: str = "BAAI/bge-small-zh-v1.5"
    family: Literal["bge", "qwen"] = "bge"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=16, ge=1)
    max_seq_length: int = Field(default=512, ge=32)
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    revision: str | None = None
    local_files_only: bool = False


class CleaningConfig(StrictConfig):
    recipe: Path
    near_duplicates: bool = True
    minhash_threshold: float = Field(default=0.9, gt=0, lt=1)
    minhash_window: int = Field(default=5, ge=1)
    minhash_permutations: int = Field(default=128, ge=16)


class ClusteringConfig(StrictConfig):
    backend: Literal["hdbscan", "bertopic"] = "hdbscan"
    min_cluster_size: int = Field(default=3, ge=2)
    min_samples: int = Field(default=2, ge=1)
    n_neighbors: int = Field(default=10, ge=2)
    n_components: int = Field(default=5, ge=2)
    random_state: int = 42


class DifficultyWeights(StrictConfig):
    structure: float = Field(default=1.0, ge=0)
    language: float = Field(default=1.0, ge=0)
    knowledge_density: float = Field(default=1.0, ge=0)
    dependency: float = Field(default=1.0, ge=0)
    temporal: float = Field(default=1.0, ge=0)
    ambiguity: float = Field(default=1.0, ge=0)


class ManualTopicRule(StrictConfig):
    topic_id: str
    name: str
    keywords: list[str]
    business_weight: float = Field(default=1.0, gt=0)
    risk_weight: float = Field(default=1.0, gt=0)


class L2Config(StrictConfig):
    enabled: bool = True
    difficulty_weights: DifficultyWeights = Field(default_factory=DifficultyWeights)
    selection_budget: int = Field(default=20, ge=1)
    selector: Literal["facility_location", "random", "size_proportional", "cluster_centroid"] = "facility_location"
    random_state: int = 42
    existing_benchmark_file: Path | None = None
    target_questions_per_cell: int = Field(default=1, ge=1)
    manual_topics: list[ManualTopicRule] = Field(default_factory=list)
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "uncovered": 1.0,
            "document_coverage": 0.8,
            "long_tail": 0.8,
            "business": 1.0,
            "risk": 1.2,
            "routing_fit": 0.5,
            "task_gap": 0.6,
            "difficulty_gap": 0.6,
            "redundancy": 0.7,
        }
    )
    target_task_mix: dict[str, float] = Field(default_factory=dict)
    target_difficulty_mix: dict[str, float] = Field(default_factory=lambda: {"easy": 0.3, "medium": 0.5, "hard": 0.2})

    @model_validator(mode="after")
    def validate_weight_keys(self):
        task_types = {"factual_qa", "multi_turn", "summarization", "judgment", "comparison", "extraction"}
        levels = {"easy", "medium", "hard"}
        priority_keys = {
            "uncovered",
            "document_coverage",
            "long_tail",
            "business",
            "risk",
            "routing_fit",
            "task_gap",
            "difficulty_gap",
            "redundancy",
        }
        for actual, allowed, name in (
            (set(self.priority_weights), priority_keys, "priority_weights"),
            (set(self.target_task_mix), task_types, "target_task_mix"),
            (set(self.target_difficulty_mix), levels, "target_difficulty_mix"),
        ):
            if unknown := actual - allowed:
                raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        for name, values in (
            ("priority_weights", self.priority_weights),
            ("target_task_mix", self.target_task_mix),
            ("target_difficulty_mix", self.target_difficulty_mix),
        ):
            if any(value < 0 for value in values.values()):
                raise ValueError(f"{name} cannot contain negative values")
        if not any(value > 0 for value in self.difficulty_weights.model_dump().values()):
            raise ValueError("At least one document difficulty weight must be positive")
        if not any(value > 0 for key, value in self.priority_weights.items() if key != "redundancy"):
            raise ValueError("At least one positive demand-cell coverage weight is required")
        if self.target_task_mix and not any(value > 0 for value in self.target_task_mix.values()):
            raise ValueError("target_task_mix must contain a positive target when configured")
        if not any(value > 0 for value in self.target_difficulty_mix.values()):
            raise ValueError("target_difficulty_mix must contain a positive target")
        if len({rule.topic_id for rule in self.manual_topics}) != len(self.manual_topics):
            raise ValueError("manual topic IDs must be unique")
        return self


class StorageConfig(StrictConfig):
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"
    create_duckdb: bool = True
    create_snapshot: bool = True
    mlflow_enabled: bool = True
    mlflow_tracking_dir: Path | None = None
    mlflow_experiment: str = "ai-ready-benchmark"


class PipelineConfig(StrictConfig):
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    cleaning: CleaningConfig
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    l2: L2Config = Field(default_factory=L2Config)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def check_limits(self):
        if self.parsing.max_tokens >= self.embedding.max_seq_length:
            raise ValueError("parsing.max_tokens must be smaller than embedding.max_seq_length")
        generated_paths = [self.output_dir, self.cache_dir]
        if self.storage.mlflow_tracking_dir:
            generated_paths.append(self.storage.mlflow_tracking_dir)
        for generated in generated_paths:
            if generated == self.input_dir or self.input_dir.is_relative_to(generated):
                raise ValueError("Output/cache directories cannot contain the input directory")
            if generated.is_relative_to(self.input_dir):
                raise ValueError("Output/cache directories cannot be inside the input directory")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Resolve all filesystem paths relative to the YAML, never the shell cwd."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML must contain a mapping")
    for key in ("input_dir", "output_dir", "cache_dir"):
        data[key] = (path.parent / data[key]).resolve()
    metadata_file = data.get("registry", {}).get("metadata_file")
    if metadata_file:
        data["registry"]["metadata_file"] = (path.parent / metadata_file).resolve()
    benchmark_file = data.get("l2", {}).get("existing_benchmark_file")
    if benchmark_file:
        data["l2"]["existing_benchmark_file"] = (path.parent / benchmark_file).resolve()
    tracking_dir = data.get("storage", {}).get("mlflow_tracking_dir")
    if tracking_dir:
        data["storage"]["mlflow_tracking_dir"] = (path.parent / tracking_dir).resolve()
    data["cleaning"]["recipe"] = (path.parent / data["cleaning"]["recipe"]).resolve()
    model_name = data.get("embedding", {}).get("model_name", "")
    if model_name.startswith("."):
        data["embedding"]["model_name"] = str((path.parent / model_name).resolve())
    return PipelineConfig.model_validate(data)
