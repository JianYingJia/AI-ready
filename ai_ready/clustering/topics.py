"""Cluster existing embeddings without deleting HDBSCAN noise points."""

import numpy as np

from ai_ready.config import ClusteringConfig


def chinese_tokens(text: str) -> list[str]:
    import jieba

    return [token for token in jieba.lcut(text) if token.strip() and any(char.isalnum() for char in token)]


def cluster_texts(texts: list[str], vectors: np.ndarray, config: ClusteringConfig) -> tuple[list[dict], dict]:
    import hdbscan

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Cluster vectors must be finite and aligned with text rows")
    if len(texts) < max(config.min_cluster_size, config.min_samples + 1):
        return [{"cluster_id": -1, "membership_strength": 0.0} for _ in texts], {
            "status": "insufficient_samples",
            "backend": config.backend,
            "topics": [],
        }
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",
        core_dist_n_jobs=1,
        prediction_data=config.backend == "bertopic",
    )
    if config.backend == "hdbscan":
        labels = clusterer.fit_predict(vectors)
        strengths = clusterer.probabilities_
        topics = []
    else:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=min(config.n_neighbors, len(texts) - 1),
            n_components=min(config.n_components, len(texts) - 2),
            min_dist=0.0,
            metric="cosine",
            random_state=config.random_state,
            init="random",
            n_jobs=1,
        )
        topic_model = BERTopic(
            language="multilingual",
            embedding_model=None,
            umap_model=reducer,
            hdbscan_model=clusterer,
            vectorizer_model=CountVectorizer(tokenizer=chinese_tokens, token_pattern=None, lowercase=False),
            calculate_probabilities=False,
            verbose=False,
        )
        labels, probabilities = topic_model.fit_transform(texts, embeddings=vectors)
        strengths = probabilities if probabilities is not None else np.zeros(len(texts))
        topics = [
            {
                "cluster_id": int(label),
                "keywords": [{"text": token, "weight": float(weight)} for token, weight in words],
            }
            for label, words in topic_model.get_topics().items()
        ]
    rows = [
        {"cluster_id": int(label), "membership_strength": float(strength)} for label, strength in zip(labels, strengths)
    ]
    summary = {
        "status": "completed",
        "backend": config.backend,
        "cluster_count": len(set(labels) - {-1}),
        "noise_count": sum(int(label) == -1 for label in labels),
        "topics": topics,
    }
    return rows, summary
