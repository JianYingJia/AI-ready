import numpy as np

from ai_ready.config import L2Config, ManualTopicRule
from ai_ready.l2.modeling import (
    build_anchors_and_routes,
    build_demand_cells,
    build_knowledge_edges,
    build_topics,
)
from ai_ready.l2.selection import select_anchors


def test_l2_builds_only_feasible_cells_and_keeps_manual_topic(unit_factory):
    units = [
        unit_factory("a", "第一条：申请人必须提交身份证明。", ["申请条件"]),
        unit_factory("b", "第二条：材料缺失时不得受理。", ["申请条件"]),
        unit_factory("c", "第三条：结果应在30日内通知。", ["办理期限"]),
    ]
    for index, unit in enumerate(units):
        unit.ordinal = index
        unit.gold_source_unit_ids = [unit.id]
        unit.document_difficulty = {
            "structure": 0.4,
            "language": 0.4,
            "knowledge_density": 0.4,
            "dependency": 0.4,
            "temporal": 0.4,
            "ambiguity": 0.4,
            "score": 0.4,
            "level": "medium",
        }
    vectors = np.asarray([[1.0, 0.0], [0.98, 0.02], [0.95, 0.05]], dtype=np.float32)
    assignments = [{"id": unit.id, "cluster_id": 0, "membership_strength": 1.0} for unit in units]
    config = L2Config(
        selection_budget=2,
        manual_topics=[ManualTopicRule(topic_id="risk", name="禁止规则", keywords=["不得"], risk_weight=2.0)],
    )
    topics, topic_units, vector_by_id = build_topics(
        units,
        assignments,
        {"topics": [{"cluster_id": 0, "keywords": [{"text": "申请", "weight": 1.0}]}]},
        vectors,
        [],
        config,
    )
    anchors, routes, anchor_units = build_anchors_and_routes(topics, topic_units, units, [], config.difficulty_weights)
    cells = build_demand_cells(topics, anchors, routes, config)
    first_cell = cells[0]
    existing = {(first_cell.topic_id, first_cell.task_type, first_cell.target_level): 1}
    updated_cells = build_demand_cells(topics, anchors, routes, config, existing, {""})
    updated = next(cell for cell in updated_cells if cell.cell_id == first_cell.cell_id)
    assert updated.current_valid_questions == 1
    assert updated.priority_signals["uncovered"] == 0
    assert updated.priority_signals["document_coverage"] == 0
    assert updated.priority < first_cell.priority
    assert any(topic.topic_id == "risk" and topic.origin == "manual" for topic in topics)
    assert cells
    assert all(cell.feasible and cell.candidate_anchor_ids for cell in cells)
    rejected = [route for route in routes if not route.feasible]
    assert rejected and all(route.reasons for route in rejected)
    selected = select_anchors(anchors, cells, anchor_units, vector_by_id, 2, "facility_location", 42)
    assert len(selected) == 2
    assert [row["selection_rank"] for row in selected] == [1, 2]
    edges = build_knowledge_edges([], units, topic_units, anchors, routes, cells)
    relations = {edge.relation for edge in edges}
    assert {"HAS_MEMBER", "USES_EVIDENCE", "SUPPORTS_TASK", "REJECTS_TASK", "HAS_CANDIDATE"} <= relations


def test_all_selection_baselines_are_available(unit_factory):
    units = [unit_factory(str(index), f"第{index}条规则") for index in range(3)]
    for unit in units:
        unit.gold_source_unit_ids = [unit.id]
        unit.document_difficulty = {
            "structure": 0.2,
            "language": 0.2,
            "knowledge_density": 0.2,
            "dependency": 0.2,
            "temporal": 0.2,
            "ambiguity": 0.2,
            "score": 0.2,
            "level": "low",
        }
    vectors = np.eye(3, dtype=np.float32)
    assignments = [{"id": unit.id, "cluster_id": 0, "membership_strength": 1.0} for unit in units]
    config = L2Config(selection_budget=3)
    topics, topic_units, vector_by_id = build_topics(units, assignments, {"topics": []}, vectors, [], config)
    anchors, routes, anchor_units = build_anchors_and_routes(topics, topic_units, units, [], config.difficulty_weights)
    cells = build_demand_cells(topics, anchors, routes, config)
    for method in ("random", "size_proportional", "cluster_centroid", "facility_location"):
        selected = select_anchors(anchors, cells, anchor_units, vector_by_id, 3, method, 42)
        assert len(selected) == 3
        assert all(row["selection_method"] == method for row in selected)
