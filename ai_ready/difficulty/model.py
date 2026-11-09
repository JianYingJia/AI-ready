"""Transparent proxy features for routing; these are not research conclusions."""

import math
import re

from ai_ready.config import DifficultyWeights
from ai_ready.core.schema import (
    DifficultyProfile,
    DocumentRecord,
    EvidenceUnit,
    TaskRoute,
)

SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d+)?%?|[一二三四五六七八九十百千万]+)(?:元|日|月|年|次|项|条|个)?")
DATE_PATTERN = re.compile(r"(?:19|20)\d{2}[年./-](?:0?[1-9]|1[0-2])(?:[月./-](?:0?[1-9]|[12]\d|3[01])日?)?")
RULE_MARKERS = ("应当", "必须", "不得", "可以", "需要", "禁止", "仅限", "至少", "不超过", "以上", "以下")
CONDITION_MARKERS = ("如果", "若", "当", "除非", "前提", "满足", "在…时", "情况下", "方可")
TEMPORAL_MARKERS = ("生效", "废止", "失效", "修订", "版本", "此前", "之后", "截至", "期间")
AMBIGUITY_MARKERS = ("除外", "例外", "但", "然而", "不得", "不适用", "另有规定", "冲突", "视情况")
ENTITY_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,16}(?:公司|机构|部门|委员会|中心|银行|贷款|账户|客户|系统)")


def _saturate(value: float, scale: float) -> float:
    return float(1.0 - math.exp(-max(value, 0.0) / scale))


def _level(score: float, labels: tuple[str, str, str]) -> str:
    return labels[0] if score < 0.34 else labels[1] if score < 0.67 else labels[2]


def document_difficulty(
    unit: EvidenceUnit, record: DocumentRecord | None, weights: DifficultyWeights
) -> DifficultyProfile:
    text = unit.text
    length = max(len(text), 1)
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]
    avg_sentence = sum(map(len, sentences)) / max(len(sentences), 1)
    pages = {loc.get("page_no") for source in unit.sources for loc in source.locations if loc.get("page_no")}
    tables = int("table" in unit.structure_labels or unit.unit_type == "table")
    lists = int("list_item" in unit.structure_labels)
    footnotes = int("footnote" in unit.structure_labels)
    condition_count = sum(text.count(marker) for marker in CONDITION_MARKERS)
    rule_count = sum(text.count(marker) for marker in RULE_MARKERS)
    temporal_count = len(DATE_PATTERN.findall(text)) + sum(text.count(marker) for marker in TEMPORAL_MARKERS)
    ambiguity_count = sum(text.count(marker) for marker in AMBIGUITY_MARKERS)
    numbers = len(NUMBER_PATTERN.findall(text))
    entities = len(set(ENTITY_PATTERN.findall(text)))
    cross_refs = len(unit.cross_references)
    versions = int(bool(record and (record.previous_version_id or record.near_duplicate_version_ids)))
    structure_raw = len(unit.headings) + 2 * tables + lists + footnotes + max(len(pages) - 1, 0)
    language_raw = avg_sentence / 30 + 3 * condition_count / max(len(sentences), 1)
    knowledge_raw = (numbers + entities + rule_count) * 100 / length
    dependency_raw = 1.5 * cross_refs + condition_count + max(len(pages) - 1, 0)
    temporal_raw = temporal_count + 2 * versions
    ambiguity_raw = ambiguity_count + 0.5 * condition_count
    signals = {
        "structure": _saturate(structure_raw, 4),
        "language": _saturate(language_raw, 3),
        "knowledge_density": _saturate(knowledge_raw, 8),
        "dependency": _saturate(dependency_raw, 3),
        "temporal": _saturate(temporal_raw, 3),
        "ambiguity": _saturate(ambiguity_raw, 3),
    }
    weight_map = weights.model_dump()
    denominator = sum(weight_map.values())
    if denominator <= 0:
        raise ValueError("At least one document difficulty weight must be positive")
    score = sum(signals[key] * weight_map[key] for key in signals) / denominator
    return DifficultyProfile(
        **signals,
        score=score,
        level=_level(score, ("low", "medium", "high")),
        raw_features={
            "character_count": float(len(text)),
            "sentence_count": float(len(sentences)),
            "average_sentence_length": avg_sentence,
            "heading_depth": float(len(unit.headings)),
            "table_count": float(tables),
            "list_count": float(lists),
            "footnote_count": float(footnotes),
            "page_count": float(len(pages)),
            "number_count": float(numbers),
            "entity_proxy_count": float(entities),
            "rule_marker_count": float(rule_count),
            "condition_count": float(condition_count),
            "cross_reference_count": float(cross_refs),
            "temporal_marker_count": float(temporal_count),
            "ambiguity_marker_count": float(ambiguity_count),
        },
    )


def capabilities_for(units: list[EvidenceUnit], records: dict[str, DocumentRecord]) -> list[str]:
    text = "\n".join(unit.text for unit in units)
    documents = {unit.document_id for unit in units}
    capabilities = {"clause_understanding"}
    if any(marker in text for marker in CONDITION_MARKERS):
        capabilities.add("conditional_reasoning")
    if any(marker in text for marker in TEMPORAL_MARKERS) or DATE_PATTERN.search(text):
        capabilities.add("temporal_judgment")
    if any(marker in text for marker in AMBIGUITY_MARKERS):
        capabilities.add("exception_handling")
    if len(documents) > 1:
        capabilities.add("cross_document_synthesis")
    if any(unit.unit_type == "table" for unit in units) or NUMBER_PATTERN.search(text):
        capabilities.add("structured_extraction")
    if any(
        records.get(unit.document_version_id) and records[unit.document_version_id].previous_version_id
        for unit in units
    ):
        capabilities.add("version_reasoning")
    return sorted(capabilities)


def evidence_bundle_difficulty(units: list[EvidenceUnit], weights: DifficultyWeights) -> DifficultyProfile:
    profiles = [unit.document_difficulty for unit in units if unit.document_difficulty]
    if not profiles:
        raise ValueError("Evidence bundle requires per-unit difficulty profiles")
    signal_names = ("structure", "language", "knowledge_density", "dependency", "temporal", "ambiguity")
    signals = {key: sum(float(profile.get(key, 0.0)) for profile in profiles) / len(profiles) for key in signal_names}
    document_count = len({unit.document_id for unit in units})
    signals["dependency"] = min(1.0, signals["dependency"] + 0.2 * max(document_count - 1, 0))
    signals["structure"] = min(1.0, signals["structure"] + 0.08 * max(len(units) - 1, 0))
    weight_map = weights.model_dump()
    denominator = sum(weight_map.values())
    if denominator <= 0:
        raise ValueError("At least one evidence-bundle difficulty weight must be positive")
    score = sum(signals[key] * weight_map[key] for key in signals) / denominator
    return DifficultyProfile(
        **signals,
        score=score,
        level=_level(score, ("low", "medium", "high")),
        raw_features={"evidence_unit_count": float(len(units)), "document_count": float(document_count)},
    )


def build_task_route(
    anchor_id: str,
    topic_id: str,
    task_type: str,
    units: list[EvidenceUnit],
    records: dict[str, DocumentRecord],
    doc_score: float,
) -> TaskRoute:
    text = "\n".join(unit.text for unit in units)
    evidence_hops = len({source_id for unit in units for source_id in unit.gold_source_unit_ids})
    document_count = len({unit.document_id for unit in units})
    version_count = len({unit.document_version_id for unit in units})
    condition_count = sum(text.count(marker) for marker in CONDITION_MARKERS)
    dates = DATE_PATTERN.findall(text)
    has_temporal = bool(dates or any(marker in text for marker in TEMPORAL_MARKERS))
    has_version_reasoning = version_count >= 2 or any(
        records.get(unit.document_version_id)
        and (
            records[unit.document_version_id].previous_version_id
            or records[unit.document_version_id].near_duplicate_version_ids
        )
        for unit in units
    )
    temporal = 1.0 if has_version_reasoning or len(set(dates)) >= 2 else (0.5 if has_temporal else 0.0)
    ambiguity = float(bool(any(marker in text for marker in AMBIGUITY_MARKERS)))
    entities = len(set(ENTITY_PATTERN.findall(text)))
    has_extractable = bool(NUMBER_PATTERN.search(text) or entities or any(unit.unit_type == "table" for unit in units))
    reasons = []
    feasible = True
    if task_type == "factual_qa":
        feasible = bool(text.strip())
        reasons.append("nonempty explicit evidence" if feasible else "empty evidence")
    elif task_type == "extraction":
        feasible = has_extractable
        reasons.append("table/entity/numeric fields available" if feasible else "no extractable structure")
    elif task_type == "summarization":
        feasible = len(text) >= 80 or evidence_hops >= 2
        reasons.append("sufficient content span" if feasible else "content too short for natural summary")
    elif task_type == "judgment":
        feasible = condition_count > 0 or ambiguity > 0
        reasons.append("condition/exception rule available" if feasible else "no judgment predicate or boundary")
    elif task_type == "comparison":
        feasible = entities >= 2 or document_count >= 2 or version_count >= 2
        reasons.append(
            "two comparable entities/documents available" if feasible else "fewer than two comparable objects"
        )
    elif task_type == "multi_turn":
        feasible = evidence_hops >= 2 or condition_count > 0
        reasons.append(
            "multiple facts or conditional follow-up available" if feasible else "no natural follow-up chain"
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    reasoning_steps = {
        "factual_qa": 1,
        "extraction": 1,
        "summarization": 2 if evidence_hops > 1 else 1,
        "judgment": 2 + int(ambiguity),
        "comparison": 2 + int(document_count > 1),
        "multi_turn": min(4, max(2, evidence_hops)),
    }[task_type]
    implicitness = min(1.0, 0.35 * condition_count + 0.3 * ambiguity)
    distractors = min(1.0, 0.35 * max(entities - 1, 0) + 0.4 * ambiguity)
    output_constraints = {
        "factual_qa": 0.15,
        "extraction": 0.45,
        "summarization": 0.5,
        "judgment": 0.7,
        "comparison": 0.65,
        "multi_turn": 0.75,
    }[task_type]
    conversation_turns = max(2, min(5, evidence_hops + 1)) if task_type == "multi_turn" else 1
    factors = [
        min(evidence_hops / 3, 1),
        min(document_count / 2, 1),
        min(reasoning_steps / 3, 1),
        implicitness,
        distractors,
        temporal,
        output_constraints,
        min(conversation_turns / 4, 1) if task_type == "multi_turn" else 0,
    ]
    task_score = sum(factors) / len(factors)
    predicted = 0.35 * doc_score + 0.65 * task_score
    predicted_level = _level(predicted, ("easy", "medium", "hard"))
    document_level = _level(doc_score, ("low", "medium", "high"))
    routing_priority = {
        "low": {
            "factual_qa": 1.0,
            "extraction": 1.0,
            "summarization": 1.0,
            "judgment": 0.6,
            "comparison": 0.5,
            "multi_turn": 0.6,
        },
        "medium": {
            "factual_qa": 0.8,
            "extraction": 0.8,
            "summarization": 1.0,
            "judgment": 1.0,
            "comparison": 1.0,
            "multi_turn": 1.0,
        },
        "high": {
            "factual_qa": 0.35,
            "extraction": 0.35,
            "summarization": 0.65,
            "judgment": 1.0,
            "comparison": 1.0,
            "multi_turn": 1.0,
        },
    }[document_level][task_type]
    levels = {predicted_level}
    if predicted < 0.55:
        levels.add("easy" if predicted < 0.34 else "medium")
    if (evidence_hops >= 2 or ambiguity or document_count >= 2) and task_type not in {"extraction"}:
        levels.add("medium")
    if (evidence_hops >= 3 or document_count >= 2 or temporal >= 1.0) and task_type in {
        "multi_turn",
        "judgment",
        "comparison",
        "summarization",
    }:
        levels.add("hard")
    return TaskRoute(
        anchor_id=anchor_id,
        topic_id=topic_id,
        task_type=task_type,
        feasible=feasible,
        reasons=reasons,
        capabilities=capabilities_for(units, records),
        routing_priority=routing_priority,
        feasible_levels=sorted(levels, key=("easy", "medium", "hard").index) if feasible else [],
        evidence_hops=evidence_hops,
        document_count=document_count,
        reasoning_steps=reasoning_steps,
        implicitness=implicitness,
        distractors=distractors,
        temporal_reasoning=temporal,
        output_constraints=output_constraints,
        conversation_turns=conversation_turns,
        predicted_score=predicted,
        predicted_level=predicted_level,
    )
