import json

import numpy as np
import pytest

from ai_ready.retrieval.faiss_index import FaissStore


def test_actual_faiss_cosine_roundtrip_and_provenance(tmp_path, unit_factory):
    units = [unit_factory("a", "提前还款"), unit_factory("b", "数据库备份")]
    store = FaissStore.build(units, np.array([[3, 0], [0, 4]], dtype=np.float32), {})
    directory = tmp_path / "中文路径"
    store.save(directory)
    loaded = FaissStore.load(directory)
    hits = loaded.search(np.array([7, 0]), top_k=10)
    assert len(hits) == 2
    assert hits[0]["id"] == "a"
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["sources"][0]["source_uri"] == "file:///sample-a.txt"
    with pytest.raises(ValueError, match="dimension"):
        loaded.search(np.array([1, 2, 3]))


def test_stale_index_records_rejected(tmp_path, unit_factory):
    store = FaissStore.build([unit_factory("a", "规则")], np.array([[1, 0]], dtype=np.float32), {})
    store.save(tmp_path / "index")
    path = tmp_path / "index/records.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["text"] = "已变更规则"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        FaissStore.load(tmp_path / "index")


def test_zero_vectors_rejected(unit_factory):
    with pytest.raises(ValueError, match="zero vectors"):
        FaissStore.build([unit_factory("a", "规则")], np.array([[0, 0]], dtype=np.float32), {})
