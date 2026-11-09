"""A small adapter around FAISS cosine search, with persisted row identity."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ai_ready.core.schema import EvidenceUnit
from ai_ready.utils.io import read_jsonl, write_json, write_jsonl


def corpus_digest(units: list[EvidenceUnit]) -> str:
    encoded = json.dumps([(unit.id, unit.embedding_text) for unit in units], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FaissStore:
    def __init__(self, index, units: list[EvidenceUnit], metadata: dict):
        self.index = index
        self.units = units
        self.metadata = metadata

    @classmethod
    def build(cls, units: list[EvidenceUnit], vectors: np.ndarray, embedding_config: dict):
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32).copy()
        if vectors.ndim != 2 or vectors.shape[0] != len(units) or not len(units):
            raise ValueError("Expected a nonempty vector matrix aligned with evidence units")
        if len({unit.id for unit in units}) != len(units):
            raise ValueError("Duplicate evidence IDs in index input")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("FAISS input contains invalid or zero vectors")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        metadata = {
            "schema_version": 1,
            "metric": "cosine",
            "index_type": "IndexFlatIP",
            "dimension": vectors.shape[1],
            "count": len(units),
            "corpus_sha256": corpus_digest(units),
            "embedding": embedding_config,
        }
        return cls(index, units, metadata)

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=False)
        # FAISS's C++ filename APIs can fail on non-ASCII Windows paths.
        faiss.serialize_index(self.index).tofile(directory / "vectors.faiss")
        write_jsonl(directory / "records.jsonl", (unit.model_dump(mode="json") for unit in self.units))
        write_json(directory / "manifest.json", self.metadata)

    @classmethod
    def load(cls, directory: Path):
        import faiss

        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        units = [EvidenceUnit.model_validate(row) for row in read_jsonl(directory / "records.jsonl")]
        index = faiss.deserialize_index(np.fromfile(directory / "vectors.faiss", dtype=np.uint8))
        if index.ntotal != len(units) or metadata["count"] != len(units) or index.d != metadata["dimension"]:
            raise ValueError("FAISS manifest, vectors and evidence records are inconsistent")
        if corpus_digest(units) != metadata["corpus_sha256"]:
            raise ValueError("Index evidence content has changed; rebuild the index")
        return cls(index, units, metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        import faiss

        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1).copy()
        if query.shape[1] != self.index.d or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Query vector dimension or values are invalid")
        faiss.normalize_L2(query)
        scores, positions = self.index.search(query, min(top_k, self.index.ntotal))
        return [
            {"score": float(score), **self.units[int(position)].model_dump(mode="json")}
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
