from pathlib import Path

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from foodops_ai.knowledge.models import DocumentChunk
from foodops_ai.knowledge.vector_store import VectorStore


class _TestEmbeddingFunction(EmbeddingFunction[Documents]):
    """Small deterministic embedder for vector store tests."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for text in input:
            normalized_text = text.casefold()
            is_delivery = any(
                term in normalized_text
                for term in ("delivery", "delayed", "atraso", "atrasado")
            )
            is_refund = any(
                term in normalized_text for term in ("refund", "reembolso", "estorno")
            )
            embeddings.append([float(is_delivery), float(is_refund)])
        return embeddings

    @staticmethod
    def name() -> str:
        return "test-embedding-function"

    @classmethod
    def build_from_config(cls, config: dict[str, object]) -> "_TestEmbeddingFunction":
        del config
        return cls()

    def get_config(self) -> dict[str, object]:
        return {}


def make_chunk(
    content: str,
    source: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        source=source,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
    )


def make_store(path: Path) -> VectorStore:
    return VectorStore(
        persist_directory=path,
        embedding_function=_TestEmbeddingFunction(),
    )


def test_indexes_and_returns_relevant_chunks_with_source_metadata(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path / "vectors")
    chunks = [
        make_chunk("A delayed delivery needs follow-up.", "late_delivery.md", 0, 0, 34),
        make_chunk("Reembolso e estorno do pedido.", "refund_policy.md", 0, 0, 30),
    ]

    assert store.index_chunks(chunks) == 2
    results = store.search("pedido atrasado", limit=1)

    assert len(results) == 1
    assert results[0].content == chunks[0].content
    assert results[0].source == "late_delivery.md"
    assert results[0].chunk_index == 0
    assert results[0].start_char == 0
    assert results[0].end_char == 34
    assert results[0].distance == pytest.approx(0.0)


def test_indexing_updated_source_removes_old_trailing_chunks(tmp_path: Path) -> None:
    store = make_store(tmp_path / "vectors")
    original_chunks = [
        make_chunk("delayed delivery first part", "late.md", 0, 0, 28),
        make_chunk("delayed delivery second part", "late.md", 1, 20, 48),
    ]
    store.index_chunks(original_chunks)

    updated_chunks = [make_chunk("delayed delivery revised", "late.md", 0, 0, 24)]
    assert store.index_chunks(updated_chunks) == 1

    results = store.search("delayed delivery", limit=5)

    assert len(results) == 1
    assert results[0].content == "delayed delivery revised"


def test_persisted_chunks_are_available_after_reopening_store(tmp_path: Path) -> None:
    database_path = tmp_path / "vectors"
    store = make_store(database_path)
    store.index_chunks([make_chunk("refund approved", "refund.md", 0, 0, 15)])

    reopened_store = make_store(database_path)
    results = reopened_store.search("reembolso", limit=1)

    assert len(results) == 1
    assert results[0].source == "refund.md"


def test_empty_store_returns_no_results(tmp_path: Path) -> None:
    store = make_store(tmp_path / "vectors")

    assert store.search("late delivery") == []


def test_search_validates_query_and_limit(tmp_path: Path) -> None:
    store = make_store(tmp_path / "vectors")

    with pytest.raises(ValueError, match="query"):
        store.search("   ")
    with pytest.raises(ValueError, match="limit"):
        store.search("delivery", limit=0)
