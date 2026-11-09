"""Evidence records retain raw text and source identity through cleaning."""

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version_id: str = ""
    source_uri: str
    source_sha256: str
    parser: str
    headings: list[str] = Field(default_factory=list)
    item_refs: list[str] = Field(default_factory=list)
    locations: list[dict[str, Any]] = Field(default_factory=list)
    raw_text: str


class EvidenceUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str = ""
    document_version_id: str = ""
    ordinal: int = 0
    unit_type: str = "text"
    text: str
    raw_text: str
    headings: list[str] = Field(default_factory=list)
    sources: list[Provenance]
    duplicate_ids: list[str] = Field(default_factory=list)
    near_duplicate_candidate: bool = False
    is_window_copy: bool = False
    gold_source_unit_ids: list[str] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    structure_labels: list[str] = Field(default_factory=list)
    document_difficulty: dict[str, Any] | None = None

    @property
    def embedding_text(self) -> str:
        return "\n".join([*self.headings, self.text])

    @property
    def dedup_text(self) -> str:
        # Heading context prevents merging identical phrases under different rules.
        return self.embedding_text


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    document_version_id: str
    logical_key: str
    title: str
    source_uri: str
    source_sha256: str
    file_size: int
    media_type: str
    version: str
    effective_from: str | None = None
    effective_to: str | None = None
    status: Literal["active", "superseded", "draft", "unknown"] = "unknown"
    permissions: list[str]
    parser: str = ""
    ingested_at: str
    unit_count: int = 0
    simhash64: str = ""
    exact_duplicate_of: str | None = None
    near_duplicate_version_ids: list[str] = Field(default_factory=list)
    previous_version_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DifficultyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure: float
    language: float
    knowledge_density: float
    dependency: float
    temporal: float
    ambiguity: float
    score: float
    level: Literal["low", "medium", "high"]
    raw_features: dict[str, float] = Field(default_factory=dict)


class TopicRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_id: str
    name: str
    keywords: list[str]
    representative_unit_ids: list[str]
    unit_count: int
    long_tail_score: float
    business_weight: float
    risk_weight: float
    origin: Literal["cluster", "manual", "outlier"]


class KnowledgeAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str
    topic_id: str
    unit_ids: list[str]
    document_ids: list[str]
    capabilities: list[str]
    feasible_task_types: list[str]
    document_difficulty: float
    evidence_bundle_difficulty: dict[str, Any]
    business_weight: float
    risk_weight: float


class DemandCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    topic_id: str
    task_type: Literal["factual_qa", "multi_turn", "summarization", "judgment", "comparison", "extraction"]
    target_level: Literal["easy", "medium", "hard"]
    feasible: bool
    feasibility_reasons: list[str]
    candidate_anchor_ids: list[str]
    current_valid_questions: int = 0
    priority: float
    priority_signals: dict[str, float]


class TaskRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str
    topic_id: str
    task_type: Literal["factual_qa", "multi_turn", "summarization", "judgment", "comparison", "extraction"]
    feasible: bool
    reasons: list[str]
    capabilities: list[str]
    routing_priority: float
    feasible_levels: list[Literal["easy", "medium", "hard"]]
    evidence_hops: int
    document_count: int
    reasoning_steps: int
    implicitness: float
    distractors: float
    temporal_reasoning: float
    output_constraints: float
    conversation_turns: int
    predicted_score: float
    predicted_level: Literal["easy", "medium", "hard"]
    empirical_score: float | None = None


class KnowledgeEdge(BaseModel):
    """Typed edge for querying the L1-to-L2 knowledge-demand network."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_type: str
    source_id: str
    relation: str
    target_type: str
    target_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
