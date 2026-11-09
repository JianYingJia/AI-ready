from types import SimpleNamespace

import pytest

from ai_ready.config import EmbeddingConfig
from ai_ready.embedding.sentence_transformer import TextEmbedder


def test_model_specific_query_instructions():
    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.config = EmbeddingConfig(family="qwen", query_instruction="Find evidence")
    assert embedder._query_prefix() == "Instruct: Find evidence\nQuery: "
    embedder.config = EmbeddingConfig()
    assert embedder._query_prefix().startswith("为这个句子")


def test_no_silent_embedding_truncation():
    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.config = EmbeddingConfig(max_seq_length=32)
    embedder.model = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"length": [100]})
    with pytest.raises(ValueError, match="truncate"):
        embedder.encode(["too long"])
