from pathlib import Path

from foodops_ai.index_knowledge import index_knowledge_base
from foodops_ai.knowledge.models import DocumentChunk


class RecordingVectorStore:
    def __init__(self) -> None:
        self.indexed_chunks: list[DocumentChunk] = []

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        self.indexed_chunks = chunks
        return len(chunks)


def test_index_knowledge_base_loads_and_chunks_documents(tmp_path: Path) -> None:
    (tmp_path / "operations.md").write_text(
        "Fictitious delivery procedure.",
        encoding="utf-8",
    )
    vector_store = RecordingVectorStore()

    summary = index_knowledge_base(tmp_path, vector_store=vector_store)

    assert summary.document_count == 1
    assert summary.chunk_count == 1
    assert vector_store.indexed_chunks == [
        DocumentChunk(
            content="Fictitious delivery procedure.",
            source="operations.md",
            chunk_index=0,
            start_char=0,
            end_char=30,
        )
    ]


def test_index_knowledge_base_with_no_documents_does_not_create_chunks(
    tmp_path: Path,
) -> None:
    vector_store = RecordingVectorStore()

    summary = index_knowledge_base(tmp_path, vector_store=vector_store)

    assert summary.document_count == 0
    assert summary.chunk_count == 0
    assert vector_store.indexed_chunks == []
