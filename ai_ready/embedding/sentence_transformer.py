"""Use model-specific query instructions and prevent silent truncation."""

import numpy as np

from ai_ready.config import EmbeddingConfig


class TextEmbedder:
    def __init__(self, config: EmbeddingConfig):
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable in aiReady; use device=cpu or install a CUDA PyTorch wheel"
            )
        self.model = SentenceTransformer(
            config.model_name,
            device=device,
            revision=config.revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        model_limit = self.model.max_seq_length
        if config.max_seq_length > model_limit:
            raise ValueError(f"Configured max_seq_length {config.max_seq_length} exceeds model limit {model_limit}")
        self.model.max_seq_length = config.max_seq_length
        self.dimension = int(self.model.get_embedding_dimension())

    def _query_prefix(self) -> str:
        if self.config.family == "qwen":
            return f"Instruct: {self.config.query_instruction}\nQuery: "
        return self.config.query_instruction

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prefix = self._query_prefix() if query else ""
        for start in range(0, len(texts), self.config.batch_size):
            batch = [prefix + text for text in texts[start : start + self.config.batch_size]]
            lengths = self.model.tokenizer(batch, truncation=False, padding=False, return_length=True)["length"]
            oversized = [start + i for i, length in enumerate(lengths) if length > self.config.max_seq_length]
            if oversized:
                raise ValueError(
                    f"Embedding would truncate items {oversized}; decrease chunk size or shorten the query"
                )
        vectors = self.model.encode(
            texts,
            prompt=prefix,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embedding model produced invalid vectors")
        return vectors

    def resolved_config(self) -> dict:
        config = self.config.model_dump()
        transformer = self.model[0].auto_model
        resolved_revision = getattr(transformer.config, "_commit_hash", None)
        if resolved_revision:
            config["revision"] = resolved_revision
        return config
