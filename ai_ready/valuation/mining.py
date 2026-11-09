"""Deterministic, evidence-bound semantic-unit mining for valuation.

The miner tags spans that already exist in source text.  It does not invent a
question, answer, task type or difficulty label; those belong to the later,
benchmark-conditioned organization stage.
"""

import hashlib
import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from ai_ready.core.schema import EvidenceUnit, stable_id
from ai_ready.valuation.selection import CoverageObligation


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActivityRule(StrictModel):
    activity_id: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)


class SemanticUnit(StrictModel):
    semantic_unit_id: str
    batch_id: str
    group_id: str
    evidence_unit_ids: list[str]
    document_ids: list[str]
    content: str
    headings: list[str]
    activity_ids: list[str]
    condition_spans: list[str]
    exception_spans: list[str]
    temporal_spans: list[str]
    uncertainty_spans: list[str]
    source_locations: list[dict]
    content_sha256: str
    duplicate_ids: list[str]
    near_duplicate_candidate: bool


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_CONDITION = re.compile(r"如果|若|当|仅当|只在|只有|在.{0,24}情况下|适用于|对于|\bif\b|\bwhen\b", re.I)
_EXCEPTION = re.compile(r"除非|除外|例外|但是|但|不得|禁止|不适用|\bunless\b|\bexcept\b|\bmust not\b", re.I)
_TEMPORAL = re.compile(r"\b(?:19|20)\d{2}(?:[-/.年]\d{1,2})?|之前|之后|目前|当前|最新|既往|复发|一线|二线")
_UNCERTAINTY = re.compile(
    r"可能|尚不明确|证据不足|有争议|不确定|建议考虑|\bmay\b|\buncertain\b|\binsufficient evidence\b", re.I
)


def _matching_spans(text: str, pattern: re.Pattern) -> list[str]:
    spans = []
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if sentence and pattern.search(sentence):
            spans.append(sentence)
    return spans


def mine_semantic_units(
    evidence_units: list[EvidenceUnit],
    *,
    batch_id: str,
    activity_rules: list[ActivityRule],
) -> list[SemanticUnit]:
    """Map each original evidence unit to one auditable semantic unit."""

    activity_ids = [rule.activity_id for rule in activity_rules]
    if len(activity_ids) != len(set(activity_ids)):
        raise ValueError("Activity IDs must be unique")
    semantic_units = []
    for evidence in evidence_units:
        # Sliding windows are retrieval aids.  Mining them again would duplicate
        # facts and make document coverage look better than it really is.
        if evidence.is_window_copy:
            continue
        searchable = "\n".join([*evidence.headings, evidence.text]).casefold()
        matched_activities = [
            rule.activity_id if any(keyword.casefold() in searchable for keyword in rule.keywords) else None
            for rule in activity_rules
        ]
        matched_activities = [activity for activity in matched_activities if activity]
        content_sha = hashlib.sha256(evidence.text.encode("utf-8")).hexdigest()
        documents = sorted({source.document_id for source in evidence.sources if source.document_id})
        semantic_units.append(
            SemanticUnit(
                semantic_unit_id=stable_id("semantic-unit-v1", batch_id, evidence.id, content_sha),
                batch_id=batch_id,
                # The evidence unit is initially indivisible.  A future audited
                # dependency resolver may deliberately merge several IDs.
                group_id=stable_id("semantic-group-v1", batch_id, evidence.id),
                evidence_unit_ids=[evidence.id],
                document_ids=documents,
                content=evidence.text,
                headings=evidence.headings,
                activity_ids=matched_activities or ["unclassified"],
                condition_spans=_matching_spans(evidence.text, _CONDITION),
                exception_spans=_matching_spans(evidence.text, _EXCEPTION),
                temporal_spans=_matching_spans(evidence.text, _TEMPORAL),
                uncertainty_spans=_matching_spans(evidence.text, _UNCERTAINTY),
                source_locations=[source.model_dump(mode="json") for source in evidence.sources],
                content_sha256=content_sha,
                duplicate_ids=evidence.duplicate_ids,
                near_duplicate_candidate=evidence.near_duplicate_candidate,
            )
        )
    return semantic_units


def fixed_chunk_baseline(
    evidence_units: list[EvidenceUnit],
    *,
    batch_id: str,
    activity_rules: list[ActivityRule],
    chunk_characters: int = 512,
) -> list[SemanticUnit]:
    """Create a deterministic no-structure baseline from the same source units.

    This baseline deliberately does not recover condition, exception, temporal,
    or uncertainty links.  It exists only for the A1 mining ablation and must
    never replace the evidence-bound structured miner in production.
    """

    if chunk_characters <= 0:
        raise ValueError("chunk_characters must be positive")
    activity_ids = [rule.activity_id for rule in activity_rules]
    if len(activity_ids) != len(set(activity_ids)):
        raise ValueError("Activity IDs must be unique")
    chunks = []
    for evidence in evidence_units:
        if evidence.is_window_copy:
            continue
        text = evidence.text
        for chunk_index, start in enumerate(range(0, len(text), chunk_characters)):
            content = text[start : start + chunk_characters]
            if not content.strip():
                continue
            searchable = "\n".join([*evidence.headings, content]).casefold()
            matched_activities = [
                rule.activity_id
                for rule in activity_rules
                if any(keyword.casefold() in searchable for keyword in rule.keywords)
            ]
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            documents = sorted({source.document_id for source in evidence.sources if source.document_id})
            chunks.append(
                SemanticUnit(
                    semantic_unit_id=stable_id(
                        "fixed-chunk-baseline-v1",
                        batch_id,
                        evidence.id,
                        str(chunk_index),
                        content_sha,
                    ),
                    batch_id=batch_id,
                    group_id=stable_id("fixed-chunk-group-v1", batch_id, evidence.id, str(chunk_index)),
                    evidence_unit_ids=[evidence.id],
                    document_ids=documents,
                    content=content,
                    headings=evidence.headings,
                    activity_ids=matched_activities or ["unclassified"],
                    condition_spans=[],
                    exception_spans=[],
                    temporal_spans=[],
                    uncertainty_spans=[],
                    source_locations=[source.model_dump(mode="json") for source in evidence.sources],
                    content_sha256=content_sha,
                    duplicate_ids=evidence.duplicate_ids,
                    near_duplicate_candidate=evidence.near_duplicate_candidate,
                )
            )
    return chunks


def build_coverage_obligations(units: list[SemanticUnit]) -> list[CoverageObligation]:
    """Create obligations only for source evidence that actually exists.

    There is no Topic × Task × Difficulty Cartesian product here.  Conditions
    and exceptions become obligations only when their source spans are present.
    """

    by_activity: dict[str, list[str]] = defaultdict(list)
    obligations = []
    for unit in units:
        for activity in unit.activity_ids:
            by_activity[activity].append(unit.semantic_unit_id)
        obligations.append(
            CoverageObligation(
                obligation_id=stable_id("rule", unit.semantic_unit_id),
                kind="rule",
                unit_ids=[unit.semantic_unit_id],
                weight=1.0,
                capability_ids=unit.activity_ids,
            )
        )
        if unit.condition_spans:
            obligations.append(
                CoverageObligation(
                    obligation_id=stable_id("condition", unit.semantic_unit_id),
                    kind="condition",
                    unit_ids=[unit.semantic_unit_id],
                    weight=1.25,
                    capability_ids=unit.activity_ids,
                )
            )
        if unit.exception_spans:
            obligations.append(
                CoverageObligation(
                    obligation_id=stable_id("exception", unit.semantic_unit_id),
                    kind="exception",
                    unit_ids=[unit.semantic_unit_id],
                    weight=1.5,
                    capability_ids=unit.activity_ids,
                )
            )
    for activity, unit_ids in sorted(by_activity.items()):
        obligations.append(
            CoverageObligation(
                obligation_id=stable_id("activity", activity),
                kind="topic",
                unit_ids=sorted(set(unit_ids)),
                weight=1.0,
                capability_ids=[activity],
            )
        )
    return obligations
