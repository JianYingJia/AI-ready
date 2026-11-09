"""Classical anchor selectors; facility location is a baseline, not the claimed contribution."""

import random
from collections import defaultdict

import numpy as np

from ai_ready.core.schema import DemandCell, KnowledgeAnchor


def _anchor_vectors(
    anchors: list[KnowledgeAnchor], anchor_units: dict[str, list], vector_by_id: dict[str, np.ndarray]
) -> np.ndarray:
    rows = []
    for anchor in anchors:
        vector = np.vstack([vector_by_id[unit.id] for unit in anchor_units[anchor.anchor_id]]).mean(axis=0)
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        rows.append(vector)
    return np.ascontiguousarray(rows, dtype=np.float32)


def select_anchors(
    anchors: list[KnowledgeAnchor],
    cells: list[DemandCell],
    anchor_units: dict[str, list],
    vector_by_id: dict[str, np.ndarray],
    budget: int,
    method: str,
    random_state: int,
) -> list[dict]:
    if not anchors:
        return []
    budget = min(budget, len(anchors))
    vectors = _anchor_vectors(anchors, anchor_units, vector_by_id)
    weights = np.ones(len(anchors), dtype=np.float64)
    cell_by_anchor = defaultdict(list)
    for cell in cells:
        for anchor_id in cell.candidate_anchor_ids:
            cell_by_anchor[anchor_id].append(cell)
    for index, anchor in enumerate(anchors):
        weights[index] += sum(cell.priority for cell in cell_by_anchor[anchor.anchor_id])
    if method == "random":
        order = random.Random(random_state).sample(range(len(anchors)), budget)
        gains = [0.0] * budget
    elif method == "size_proportional":
        rng = random.Random(random_state)
        pool = list(range(len(anchors)))
        order = []
        while pool and len(order) < budget:
            selected = rng.choices(pool, weights=[weights[index] for index in pool], k=1)[0]
            pool.remove(selected)
            order.append(selected)
        gains = [float(weights[index]) for index in order]
    elif method == "cluster_centroid":
        by_topic = defaultdict(list)
        for index, anchor in enumerate(anchors):
            by_topic[anchor.topic_id].append(index)
        order = []
        for indices in by_topic.values():
            centroid = vectors[indices].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            order.append(max(indices, key=lambda index: float(vectors[index] @ centroid)))
        order = order[:budget]
        if len(order) < budget:
            remaining = [int(index) for index in np.argsort(-weights) if int(index) not in order]
            order.extend(remaining[: budget - len(order)])
        gains = [float(weights[index]) for index in order]
    elif method == "facility_location":
        similarity = np.clip(vectors @ vectors.T, 0.0, 1.0)
        covered = np.zeros(len(anchors), dtype=np.float64)
        selected_cells = set()
        remaining = set(range(len(anchors)))
        order, gains = [], []
        for _ in range(budget):
            best_index, best_gain = None, float("-inf")
            for index in sorted(remaining):
                facility_gain = float(np.sum(weights * np.maximum(similarity[:, index] - covered, 0.0)))
                new_cells = {
                    cell.cell_id
                    for cell in cell_by_anchor[anchors[index].anchor_id]
                    if cell.cell_id not in selected_cells
                }
                coverage_gain = sum(
                    cell.priority for cell in cell_by_anchor[anchors[index].anchor_id] if cell.cell_id in new_cells
                )
                gain = facility_gain + coverage_gain
                if gain > best_gain:
                    best_index, best_gain = index, gain
            if best_index is None:
                break
            order.append(best_index)
            gains.append(best_gain)
            remaining.remove(best_index)
            covered = np.maximum(covered, similarity[:, best_index])
            selected_cells.update(cell.cell_id for cell in cell_by_anchor[anchors[best_index].anchor_id])
    else:
        raise ValueError(f"Unknown selector: {method}")
    return [
        {
            **anchors[index].model_dump(mode="json"),
            "selection_rank": rank,
            "selection_method": method,
            "marginal_gain": float(gain),
            "demand_cell_ids": [cell.cell_id for cell in cell_by_anchor[anchors[index].anchor_id]],
        }
        for rank, (index, gain) in enumerate(zip(order, gains), start=1)
    ]
