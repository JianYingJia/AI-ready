import numpy as np

from ai_ready.clustering.topics import cluster_texts
from ai_ready.config import ClusteringConfig


def test_small_sample_keeps_noise():
    rows, summary = cluster_texts(["贷款"], np.array([[1.0, 0.0]]), ClusteringConfig())
    assert rows == [{"cluster_id": -1, "membership_strength": 0.0}]
    assert summary["status"] == "insufficient_samples"


def test_actual_hdbscan_separates_clusters_and_keeps_every_row():
    rng = np.random.default_rng(42)
    vectors = np.vstack([rng.normal(0, 0.01, (10, 3)), rng.normal(5, 0.01, (10, 3))])
    rows, summary = cluster_texts(["记录"] * 20, vectors, ClusteringConfig())
    assert len(rows) == 20
    assert summary["cluster_count"] == 2


def test_actual_bertopic_uses_given_embeddings_and_chinese_tokenizer():
    rng = np.random.default_rng(42)
    vectors = np.vstack([rng.normal(0, 0.01, (12, 5)), rng.normal(5, 0.01, (12, 5))])
    texts = ["商业贷款提前还款申请条件"] * 12 + ["数据库备份系统故障恢复"] * 12
    rows, summary = cluster_texts(texts, vectors, ClusteringConfig(backend="bertopic", n_components=2))
    assert len(rows) == len(texts)
    assert summary["topics"]
    assert any("贷款" in word["text"] for topic in summary["topics"] for word in topic["keywords"])
