"""Fully automated, domain-general, evidence-bound knowledge-unit extraction."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_ready.core.schema import EvidenceUnit, stable_id
from ai_ready.utils.io import read_jsonl, read_parquet, write_json, write_jsonl
from ai_ready.valuation.benchmark_blueprint import (
    AnnotationInference,
    OllamaAnnotationInference,
)
from ai_ready.valuation.mining import SemanticUnit

KnowledgeUnitType = Literal[
    "definition",
    "fact",
    "rule",
    "procedure",
    "causal",
    "comparison",
    "recommendation",
    "constraint",
    "exception",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeRelation(StrictModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class CandidateKnowledgeUnit(StrictModel):
    unit_type: KnowledgeUnitType
    claim: str = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    temporal_scope: str | None = None
    uncertainty: str | None = None
    entities: list[str] = Field(default_factory=list)
    relations: list[KnowledgeRelation] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("conditions", "exceptions", "entities", "evidence_quotes")
    @classmethod
    def clean_unique_values(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Knowledge-unit lists cannot contain empty strings")
        return list(dict.fromkeys(normalized))


class ExtractionResponse(StrictModel):
    units: list[CandidateKnowledgeUnit] = Field(default_factory=list)


class VerificationResponse(StrictModel):
    supported: bool
    fields_consistent: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class GenericKnowledgeUnit(StrictModel):
    knowledge_unit_id: str
    evidence_unit_id: str
    document_ids: list[str]
    headings: list[str]
    unit_type: KnowledgeUnitType
    claim: str
    conditions: list[str]
    exceptions: list[str]
    temporal_scope: str | None
    uncertainty: str | None
    entities: list[str]
    relations: list[KnowledgeRelation]
    evidence_quotes: list[str]
    evidence_offsets: list[tuple[int, int]]
    source_locations: list[dict[str, Any]]
    extraction_confidence: float
    verification_support: float
    verification_confidence: float
    verification_status: Literal["accepted", "rejected"]
    adjudicated: bool
    content_sha256: str


class ExtractionConfig(StrictModel):
    extraction_runs_per_model: int = Field(default=2, ge=1)
    verification_runs_per_model: int = Field(default=2, ge=1)
    adjudication_runs_per_model: int = Field(default=3, ge=1)
    minimum_verifier_support: float = Field(default=2 / 3, gt=0.0, le=1.0)
    minimum_verifier_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    base_seed: int = 23
    extraction_temperature: float = Field(default=0.1, ge=0.0)
    consolidation_temperature: float = Field(default=0.0, ge=0.0)
    verification_temperature: float = Field(default=0.0, ge=0.0)
    adjudication_temperature: float = Field(default=0.0, ge=0.0)


def _stable_seed(base_seed: int, evidence_id: str, model: str, run_index: int, stage: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{evidence_id}|{model}|{run_index}|{stage}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _unit_schema() -> dict[str, Any]:
    return {
        "unit_type": "definition|fact|rule|procedure|causal|comparison|recommendation|constraint|exception",
        "claim": "one self-contained atomic knowledge statement",
        "conditions": ["applicability prerequisites stated by the source"],
        "exceptions": ["exceptions, overrides, prohibitions, or contraindications stated by the source"],
        "temporal_scope": "time or version scope, or null",
        "uncertainty": "uncertainty or evidence-strength qualifier, or null",
        "entities": ["canonical entities explicitly supported by the source"],
        "relations": [{"subject": "...", "predicate": "...", "object": "..."}],
        "evidence_quotes": ["one or more exact verbatim substrings copied from source_text"],
        "confidence": "number from 0 to 1",
    }


def render_extraction_prompt(evidence: EvidenceUnit) -> dict[str, Any]:
    return {
        "schema_version": "generic-knowledge-extraction-prompt-v1",
        "evidence_unit_id": evidence.id,
        "system": (
            "Extract domain-independent atomic knowledge units from the source. A unit must be a definition, fact, "
            "rule, procedure, causal statement, comparison, recommendation, constraint, or exception. Split "
            "independent claims, but keep a claim together with conditions and exceptions needed to interpret it. "
            "Do not infer unstated knowledge. Every unit must include exact verbatim evidence quotes copied from "
            "source_text. Return JSON only."
        ),
        "headings": evidence.headings,
        "source_text": evidence.text,
        "output_schema": {"units": [_unit_schema()]},
    }


def render_consolidation_prompt(evidence: EvidenceUnit, candidates: list[CandidateKnowledgeUnit]) -> dict[str, Any]:
    return {
        "schema_version": "generic-knowledge-consolidation-prompt-v1",
        "evidence_unit_id": evidence.id,
        "system": (
            "Consolidate candidate extractions into the smallest complete set of non-duplicate atomic knowledge "
            "units. Re-read the source rather than voting mechanically. Preserve conditions, exceptions, temporal "
            "scope, and uncertainty. Delete unsupported candidates. Every evidence quote must be copied verbatim "
            "from source_text. Return JSON only."
        ),
        "headings": evidence.headings,
        "source_text": evidence.text,
        "candidate_units": [row.model_dump(mode="json") for row in candidates],
        "output_schema": {"units": [_unit_schema()]},
    }


def render_verification_prompt(
    evidence: EvidenceUnit, unit: CandidateKnowledgeUnit, *, adjudication: bool = False
) -> dict[str, Any]:
    return {
        "schema_version": (
            "generic-knowledge-adjudication-prompt-v1" if adjudication else "generic-knowledge-verification-prompt-v1"
        ),
        "evidence_unit_id": evidence.id,
        "system": (
            "Independently verify whether every material part of the candidate is supported by source_text and "
            "whether its conditions, exceptions, temporal scope, uncertainty, entities, and relations are faithful. "
            "A plausible statement is not supported unless the supplied source states it. Return JSON only."
        ),
        "headings": evidence.headings,
        "source_text": evidence.text,
        "candidate_unit": unit.model_dump(mode="json"),
        "output_schema": {
            "supported": "boolean",
            "fields_consistent": "boolean",
            "confidence": "number from 0 to 1",
            "rationale": "brief source-grounded reason",
        },
    }


def _locate_quotes(text: str, quotes: list[str]) -> list[tuple[int, int]]:
    offsets = []
    cursor_by_quote: Counter[str] = Counter()
    for quote in quotes:
        start_at = cursor_by_quote[quote]
        start = text.find(quote, start_at)
        if start < 0:
            start = text.find(quote)
        if start < 0:
            raise ValueError(f"Evidence quote is not an exact source substring: {quote!r}")
        offsets.append((start, start + len(quote)))
        cursor_by_quote[quote] = start + len(quote)
    return offsets


def _parse_extraction(payload: Mapping[str, Any], source_text: str) -> list[CandidateKnowledgeUnit]:
    response = ExtractionResponse.model_validate(payload)
    for unit in response.units:
        _locate_quotes(source_text, unit.evidence_quotes)
    return response.units


def _run_extraction_panel(
    evidence: EvidenceUnit,
    models: list[str],
    config: ExtractionConfig,
    inference: AnnotationInference,
) -> list[CandidateKnowledgeUnit]:
    candidates = []
    prompt = render_extraction_prompt(evidence)
    for model in models:
        for run_index in range(config.extraction_runs_per_model):
            seed = _stable_seed(config.base_seed, evidence.id, model, run_index, "extract")
            payload = inference(model, prompt, seed, config.extraction_temperature)
            candidates.extend(_parse_extraction(payload, evidence.text))
    return candidates


def _consolidate(
    evidence: EvidenceUnit,
    candidates: list[CandidateKnowledgeUnit],
    model: str,
    config: ExtractionConfig,
    inference: AnnotationInference,
) -> list[CandidateKnowledgeUnit]:
    prompt = render_consolidation_prompt(evidence, candidates)
    seed = _stable_seed(config.base_seed, evidence.id, model, 0, "consolidate")
    return _parse_extraction(inference(model, prompt, seed, config.consolidation_temperature), evidence.text)


def _verify_panel(
    evidence: EvidenceUnit,
    unit: CandidateKnowledgeUnit,
    models: list[str],
    runs_per_model: int,
    temperature: float,
    stage: str,
    config: ExtractionConfig,
    inference: AnnotationInference,
) -> list[VerificationResponse]:
    prompt = render_verification_prompt(evidence, unit, adjudication=stage == "adjudicate")
    responses = []
    for model in models:
        for run_index in range(runs_per_model):
            seed = _stable_seed(config.base_seed, evidence.id + unit.claim, model, run_index, stage)
            responses.append(VerificationResponse.model_validate(inference(model, prompt, seed, temperature)))
    return responses


def _verification_summary(rows: list[VerificationResponse]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    support = sum(row.supported and row.fields_consistent for row in rows) / len(rows)
    confidence = sum(row.confidence for row in rows) / len(rows)
    return support, confidence


def _source_locations(evidence: EvidenceUnit) -> list[dict[str, Any]]:
    return [source.model_dump(mode="json") for source in evidence.sources]


def _materialize_unit(
    evidence: EvidenceUnit,
    candidate: CandidateKnowledgeUnit,
    support: float,
    verification_confidence: float,
    *,
    accepted: bool,
    adjudicated: bool,
) -> GenericKnowledgeUnit:
    offsets = _locate_quotes(evidence.text, candidate.evidence_quotes)
    identity = "|".join(f"{start}:{end}" for start, end in offsets)
    content_sha = hashlib.sha256("\n".join(candidate.evidence_quotes).encode("utf-8")).hexdigest()
    return GenericKnowledgeUnit(
        knowledge_unit_id=stable_id("generic-knowledge-unit-v1", evidence.id, identity, candidate.claim),
        evidence_unit_id=evidence.id,
        document_ids=sorted({source.document_id for source in evidence.sources if source.document_id}),
        headings=evidence.headings,
        unit_type=candidate.unit_type,
        claim=candidate.claim,
        conditions=candidate.conditions,
        exceptions=candidate.exceptions,
        temporal_scope=candidate.temporal_scope,
        uncertainty=candidate.uncertainty,
        entities=candidate.entities,
        relations=candidate.relations,
        evidence_quotes=candidate.evidence_quotes,
        evidence_offsets=offsets,
        source_locations=_source_locations(evidence),
        extraction_confidence=candidate.confidence,
        verification_support=support,
        verification_confidence=verification_confidence,
        verification_status="accepted" if accepted else "rejected",
        adjudicated=adjudicated,
        content_sha256=content_sha,
    )


def extract_generic_knowledge_units(
    evidence_units: list[EvidenceUnit],
    *,
    extraction_models: list[str],
    consolidation_model: str,
    verification_models: list[str],
    adjudicator_models: list[str] | None = None,
    config: ExtractionConfig | None = None,
    inference: AnnotationInference | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> tuple[list[GenericKnowledgeUnit], list[dict[str, Any]]]:
    config = config or ExtractionConfig()
    inference = inference or OllamaAnnotationInference(base_url=ollama_url)
    adjudicator_models = adjudicator_models or verification_models
    if not extraction_models or not verification_models or not adjudicator_models:
        raise ValueError("Extraction, verification, and adjudication model panels cannot be empty")
    results = []
    dispositions = []
    for evidence in evidence_units:
        if evidence.is_window_copy:
            continue
        candidates = _run_extraction_panel(evidence, extraction_models, config, inference)
        consolidated = _consolidate(evidence, candidates, consolidation_model, config, inference)
        accepted_count = 0
        rejected_count = 0
        for candidate in consolidated:
            verification = _verify_panel(
                evidence,
                candidate,
                verification_models,
                config.verification_runs_per_model,
                config.verification_temperature,
                "verify",
                config,
                inference,
            )
            support, confidence = _verification_summary(verification)
            accepted = support >= config.minimum_verifier_support and confidence >= config.minimum_verifier_confidence
            adjudicated = False
            if not accepted:
                adjudicated = True
                adjudication = _verify_panel(
                    evidence,
                    candidate,
                    adjudicator_models,
                    config.adjudication_runs_per_model,
                    config.adjudication_temperature,
                    "adjudicate",
                    config,
                    inference,
                )
                support, confidence = _verification_summary(adjudication)
                accepted = (
                    support >= config.minimum_verifier_support and confidence >= config.minimum_verifier_confidence
                )
            results.append(
                _materialize_unit(
                    evidence,
                    candidate,
                    support,
                    confidence,
                    accepted=accepted,
                    adjudicated=adjudicated,
                )
            )
            accepted_count += int(accepted)
            rejected_count += int(not accepted)
        dispositions.append(
            {
                "evidence_unit_id": evidence.id,
                "candidate_count": len(candidates),
                "consolidated_count": len(consolidated),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "status": "accepted" if accepted_count else "no_supported_knowledge",
            }
        )
    return results, dispositions


def to_semantic_units(units: list[GenericKnowledgeUnit], *, batch_id: str) -> list[SemanticUnit]:
    """Adapt accepted generic units to the existing downstream semantic-unit contract."""

    rows = []
    for unit in units:
        if unit.verification_status != "accepted":
            continue
        rows.append(
            SemanticUnit(
                semantic_unit_id=unit.knowledge_unit_id,
                batch_id=batch_id,
                group_id=stable_id("generic-dependency-group-v1", batch_id, unit.evidence_unit_id),
                evidence_unit_ids=[unit.evidence_unit_id],
                document_ids=unit.document_ids,
                content="\n".join(unit.evidence_quotes),
                headings=unit.headings,
                activity_ids=["unmapped"],
                condition_spans=unit.conditions,
                exception_spans=unit.exceptions,
                temporal_spans=[unit.temporal_scope] if unit.temporal_scope else [],
                uncertainty_spans=[unit.uncertainty] if unit.uncertainty else [],
                source_locations=unit.source_locations,
                content_sha256=unit.content_sha256,
                duplicate_ids=[],
                near_duplicate_candidate=False,
            )
        )
    return rows


def _read_evidence_units(path: Path) -> list[EvidenceUnit]:
    rows = read_parquet(path) if path.suffix.casefold() == ".parquet" else read_jsonl(path)
    return [EvidenceUnit.model_validate(row) for row in rows]


def run_generic_knowledge_pipeline(
    *,
    records_path: str | Path,
    output_dir: str | Path,
    batch_id: str,
    extraction_models: list[str],
    consolidation_model: str,
    verification_models: list[str],
    adjudicator_models: list[str] | None = None,
    config: ExtractionConfig | None = None,
    inference: AnnotationInference | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    config = config or ExtractionConfig()
    records_path, output_dir = Path(records_path), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Knowledge extraction output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_units = _read_evidence_units(records_path)
    units, dispositions = extract_generic_knowledge_units(
        evidence_units,
        extraction_models=extraction_models,
        consolidation_model=consolidation_model,
        verification_models=verification_models,
        adjudicator_models=adjudicator_models,
        config=config,
        inference=inference,
        ollama_url=ollama_url,
    )
    semantic_units = to_semantic_units(units, batch_id=batch_id)
    knowledge_path = output_dir / "knowledge_units.jsonl"
    semantic_path = output_dir / "semantic_units.jsonl"
    disposition_path = output_dir / "dispositions.jsonl"
    write_jsonl(knowledge_path, (row.model_dump(mode="json") for row in units))
    write_jsonl(semantic_path, (row.model_dump(mode="json") for row in semantic_units))
    write_jsonl(disposition_path, dispositions)
    manifest = {
        "schema_version": "generic-knowledge-extraction-manifest-v1",
        "batch_id": batch_id,
        "human_review_used": False,
        "benchmark_blueprint_used": False,
        "records_path": str(records_path.resolve()),
        "models": {
            "extraction": extraction_models,
            "consolidation": consolidation_model,
            "verification": verification_models,
            "adjudication": adjudicator_models or verification_models,
        },
        "config": config.model_dump(mode="json"),
        "counts": {
            "evidence_units": sum(not row.is_window_copy for row in evidence_units),
            "knowledge_units": len(units),
            "accepted_units": sum(row.verification_status == "accepted" for row in units),
            "rejected_units": sum(row.verification_status == "rejected" for row in units),
            "semantic_units": len(semantic_units),
        },
        "outputs": {
            "knowledge_units": knowledge_path.name,
            "semantic_units": semantic_path.name,
            "dispositions": disposition_path.name,
        },
    }
    write_json(output_dir / "extraction_manifest.json", manifest)
    return {**manifest, "output_dir": str(output_dir.resolve())}


__all__ = [
    "CandidateKnowledgeUnit",
    "ExtractionConfig",
    "ExtractionResponse",
    "GenericKnowledgeUnit",
    "KnowledgeRelation",
    "VerificationResponse",
    "extract_generic_knowledge_units",
    "render_consolidation_prompt",
    "render_extraction_prompt",
    "render_verification_prompt",
    "run_generic_knowledge_pipeline",
    "to_semantic_units",
]
