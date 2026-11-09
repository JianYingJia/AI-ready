"""Build topics, knowledge anchors, feasible demand cells and task routes."""

import math
from collections import Counter, defaultdict

import numpy as np

from ai_ready.config import DifficultyWeights, L2Config
from ai_ready.core.schema import (
    DemandCell,
    DocumentRecord,
    EvidenceUnit,
    KnowledgeAnchor,
    KnowledgeEdge,
    TopicRecord,
    stable_id,
)
from ai_ready.difficulty.model import (
    build_task_route,
    capabilities_for,
    evidence_bundle_difficulty,
)
from ai_ready.l2.task_catalog import TASK_TYPES


def _unit_representatives(unit_ids: list[str], vector_by_id: dict[str, np.ndarray], limit: int = 3) -> list[str]:
    vectors = np.vstack([vector_by_id[unit_id] for unit_id in unit_ids])
    centroid = vectors.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    scores = vectors @ centroid
    return [unit_ids[index] for index in np.argsort(-scores)[:limit]]


def build_topics(
    units: list[EvidenceUnit],
    assignments: list[dict],
    cluster_summary: dict,
    vectors: np.ndarray,
    records: list[DocumentRecord],
    config: L2Config,
) -> tuple[list[TopicRecord], dict[str, list[str]], dict[str, np.ndarray]]:
    original_units = [unit for unit in units if not unit.is_window_copy]
    indexed_by_id = {unit.id: index for index, unit in enumerate(units)}
    vector_by_id = {unit.id: vectors[indexed_by_id[unit.id]] for unit in original_units}
    label_by_id = {row["id"]: int(row["cluster_id"]) for row in assignments}
    keywords_by_label = {
        int(topic["cluster_id"]): [word["text"] for word in topic.get("keywords", []) if word.get("text")]
        for topic in cluster_summary.get("topics", [])
    }
    groups = defaultdict(list)
    for unit in original_units:
        groups[label_by_id.get(unit.id, -1)].append(unit.id)
    total = max(len(original_units), 1)
    record_by_version = {record.document_version_id: record for record in records}
    topics = []
    topic_units: dict[str, list[str]] = {}
    for label, unit_ids in sorted(groups.items()):
        if label == -1:
            for unit_id in unit_ids:
                unit = next(item for item in original_units if item.id == unit_id)
                topic_id = f"outlier-{unit_id}"
                name = unit.headings[-1] if unit.headings else f"长尾知识-{unit_id[:8]}"
                topic_units[topic_id] = [unit_id]
                record = record_by_version.get(unit.document_version_id)
                topics.append(
                    TopicRecord(
                        topic_id=topic_id,
                        name=name,
                        keywords=[],
                        representative_unit_ids=[unit_id],
                        unit_count=1,
                        long_tail_score=1.0,
                        business_weight=float((record.metadata if record else {}).get("business_weight", 1.0)),
                        risk_weight=float((record.metadata if record else {}).get("risk_weight", 1.0)),
                        origin="outlier",
                    )
                )
            continue
        topic_id = f"topic-{label}"
        keywords = keywords_by_label.get(label, [])[:10]
        heading_names = [unit.headings[-1] for unit in original_units if unit.id in unit_ids and unit.headings]
        name = (
            " / ".join(keywords[:3])
            if keywords
            else (Counter(heading_names).most_common(1)[0][0] if heading_names else topic_id)
        )
        relevant_records = [
            record_by_version[unit.document_version_id]
            for unit in original_units
            if unit.id in unit_ids and unit.document_version_id in record_by_version
        ]
        business = max([float(record.metadata.get("business_weight", 1.0)) for record in relevant_records] or [1.0])
        risk = max([float(record.metadata.get("risk_weight", 1.0)) for record in relevant_records] or [1.0])
        topic_units[topic_id] = unit_ids
        topics.append(
            TopicRecord(
                topic_id=topic_id,
                name=name,
                keywords=keywords,
                representative_unit_ids=_unit_representatives(unit_ids, vector_by_id),
                unit_count=len(unit_ids),
                long_tail_score=1.0 - math.log1p(len(unit_ids)) / math.log1p(total),
                business_weight=business,
                risk_weight=risk,
                origin="cluster",
            )
        )
    for rule in config.manual_topics:
        matched = [
            unit.id for unit in original_units if any(keyword in unit.embedding_text for keyword in rule.keywords)
        ]
        if not matched:
            continue
        topic_units[rule.topic_id] = matched
        topics.append(
            TopicRecord(
                topic_id=rule.topic_id,
                name=rule.name,
                keywords=rule.keywords,
                representative_unit_ids=_unit_representatives(matched, vector_by_id),
                unit_count=len(matched),
                long_tail_score=1.0 - math.log1p(len(matched)) / math.log1p(total),
                business_weight=rule.business_weight,
                risk_weight=rule.risk_weight,
                origin="manual",
            )
        )
    return topics, topic_units, vector_by_id


def build_anchors_and_routes(
    topics: list[TopicRecord],
    topic_units: dict[str, list[str]],
    units: list[EvidenceUnit],
    records: list[DocumentRecord],
    difficulty_weights: DifficultyWeights,
) -> tuple[list[KnowledgeAnchor], list, dict[str, list[EvidenceUnit]]]:
    unit_by_id = {unit.id: unit for unit in units}
    record_by_version = {record.document_version_id: record for record in records}
    topics_by_id = {topic.topic_id: topic for topic in topics}
    anchors = []
    anchor_units = {}
    for topic_id, member_ids in topic_units.items():
        topic = topics_by_id[topic_id]
        bundles = [[unit_id] for unit_id in member_ids]
        if len(member_ids) >= 2:
            bundles.append(topic.representative_unit_ids[: min(3, len(topic.representative_unit_ids))])
        for bundle in bundles:
            selected_units = [unit_by_id[unit_id] for unit_id in bundle]
            anchor_id = stable_id("anchor", topic_id, *bundle)
            anchor_units[anchor_id] = selected_units
            doc_score = max(
                [float(unit.document_difficulty["score"]) for unit in selected_units if unit.document_difficulty]
                or [0.0]
            )
            capabilities = capabilities_for(selected_units, record_by_version)
            bundle_profile = evidence_bundle_difficulty(selected_units, difficulty_weights)
            anchors.append(
                KnowledgeAnchor(
                    anchor_id=anchor_id,
                    topic_id=topic_id,
                    unit_ids=bundle,
                    document_ids=sorted({unit.document_id for unit in selected_units}),
                    capabilities=capabilities,
                    feasible_task_types=[],
                    document_difficulty=doc_score,
                    evidence_bundle_difficulty=bundle_profile.model_dump(mode="json"),
                    business_weight=topic.business_weight,
                    risk_weight=topic.risk_weight,
                )
            )
    routes = []
    anchor_by_id = {anchor.anchor_id: anchor for anchor in anchors}
    for anchor_id, selected_units in anchor_units.items():
        anchor = anchor_by_id[anchor_id]
        for task_type in TASK_TYPES:
            route = build_task_route(
                anchor_id,
                anchor.topic_id,
                task_type,
                selected_units,
                record_by_version,
                float(anchor.evidence_bundle_difficulty["score"]),
            )
            routes.append(route)
            if route.feasible:
                anchor.feasible_task_types.append(task_type)
    return anchors, routes, anchor_units


def build_demand_cells(
    topics: list[TopicRecord],
    anchors: list[KnowledgeAnchor],
    routes: list,
    config: L2Config,
    existing_counts: dict[tuple[str, str, str], int] | None = None,
    existing_document_ids: set[str] | None = None,
) -> list[DemandCell]:
    topic_by_id = {topic.topic_id: topic for topic in topics}
    anchor_by_id = {anchor.anchor_id: anchor for anchor in anchors}
    route_groups = defaultdict(list)
    for route in routes:
        if route.feasible:
            for level in route.feasible_levels:
                route_groups[(route.topic_id, route.task_type, level)].append(route)
    task_mix = config.target_task_mix or {task_type: 1 / len(TASK_TYPES) for task_type in TASK_TYPES}
    task_total = sum(task_mix.values()) or 1.0
    task_targets = {key: value / task_total for key, value in task_mix.items()}
    difficulty_total = sum(config.target_difficulty_mix.values()) or 1.0
    difficulty_targets = {key: value / difficulty_total for key, value in config.target_difficulty_mix.items()}
    weights = config.priority_weights
    denominator = sum(value for key, value in weights.items() if key != "redundancy") or 1.0
    cells = []
    existing_counts = existing_counts or {}
    existing_document_ids = existing_document_ids or set()
    existing_total = sum(existing_counts.values())
    existing_by_task = defaultdict(int)
    existing_by_level = defaultdict(int)
    for (_, task_type, level), count in existing_counts.items():
        existing_by_task[task_type] += count
        existing_by_level[level] += count

    def mix_gap(target: float, count: int) -> float:
        if target <= 0:
            return 0.0
        current = count / existing_total if existing_total else 0.0
        return max(target - current, 0.0) / target

    for (topic_id, task_type, level), candidates in sorted(route_groups.items()):
        topic = topic_by_id[topic_id]
        candidate_ids = sorted({route.anchor_id for route in candidates})
        candidate_document_ids = {
            document_id for anchor_id in candidate_ids for document_id in anchor_by_id[anchor_id].document_ids
        }
        current_count = existing_counts.get((topic_id, task_type, level), 0)
        signals = {
            "uncovered": float(current_count == 0),
            "document_coverage": (
                sum(document_id not in existing_document_ids for document_id in candidate_document_ids)
                / len(candidate_document_ids)
                if candidate_document_ids
                else 0.0
            ),
            "long_tail": topic.long_tail_score,
            "business": min(topic.business_weight / 2, 1.0),
            "risk": min(topic.risk_weight / 2, 1.0),
            "routing_fit": max(route.routing_priority for route in candidates),
            "task_gap": mix_gap(task_targets.get(task_type, 0.0), existing_by_task[task_type]),
            "difficulty_gap": mix_gap(difficulty_targets.get(level, 0.0), existing_by_level[level]),
            "redundancy": min(current_count / config.target_questions_per_cell, 1.0),
        }
        positive = sum(weights.get(key, 0.0) * value for key, value in signals.items() if key != "redundancy")
        priority = positive / denominator - weights.get("redundancy", 0.0) * signals["redundancy"] / denominator
        cells.append(
            DemandCell(
                cell_id=stable_id("cell", topic_id, task_type, level),
                topic_id=topic_id,
                task_type=task_type,
                target_level=level,
                feasible=True,
                feasibility_reasons=sorted({reason for route in candidates for reason in route.reasons}),
                candidate_anchor_ids=candidate_ids,
                current_valid_questions=current_count,
                priority=max(priority, 0.0),
                priority_signals=signals,
            )
        )
    return cells


def build_knowledge_edges(
    documents: list[DocumentRecord],
    units: list[EvidenceUnit],
    topic_units: dict[str, list[str]],
    anchors: list[KnowledgeAnchor],
    routes: list,
    cells: list[DemandCell],
) -> list[KnowledgeEdge]:
    """Materialize typed relations without pretending they are extracted entity facts."""
    rows: dict[str, KnowledgeEdge] = {}

    def add(source_type: str, source_id: str, relation: str, target_type: str, target_id: str, **attributes):
        edge_id = stable_id("edge", source_type, source_id, relation, target_type, target_id)
        rows[edge_id] = KnowledgeEdge(
            edge_id=edge_id,
            source_type=source_type,
            source_id=source_id,
            relation=relation,
            target_type=target_type,
            target_id=target_id,
            attributes=attributes,
        )

    for document in documents:
        add("document", document.document_id, "HAS_VERSION", "document_version", document.document_version_id)
        if document.previous_version_id:
            add(
                "document_version",
                document.document_version_id,
                "PREVIOUS_VERSION",
                "document_version",
                document.previous_version_id,
            )
    for unit in units:
        add(
            "document_version",
            unit.document_version_id,
            "CONTAINS_UNIT",
            "document_unit",
            unit.id,
            ordinal=unit.ordinal,
            unit_type=unit.unit_type,
            is_window_copy=unit.is_window_copy,
        )
        for source_id in unit.gold_source_unit_ids:
            if source_id != unit.id:
                add("document_unit", unit.id, "DERIVED_FROM_UNIT", "document_unit", source_id)
    for topic_id, unit_ids in topic_units.items():
        for unit_id in unit_ids:
            add("topic", topic_id, "HAS_MEMBER", "document_unit", unit_id)
    for anchor in anchors:
        for unit_id in anchor.unit_ids:
            add("knowledge_anchor", anchor.anchor_id, "USES_EVIDENCE", "document_unit", unit_id)
    for route in routes:
        add(
            "knowledge_anchor",
            route.anchor_id,
            "SUPPORTS_TASK" if route.feasible else "REJECTS_TASK",
            "task_type",
            route.task_type,
            predicted_level=route.predicted_level,
            reasons=route.reasons,
        )
    for cell in cells:
        for anchor_id in cell.candidate_anchor_ids:
            add("demand_cell", cell.cell_id, "HAS_CANDIDATE", "knowledge_anchor", anchor_id)
    return sorted(rows.values(), key=lambda edge: edge.edge_id)
