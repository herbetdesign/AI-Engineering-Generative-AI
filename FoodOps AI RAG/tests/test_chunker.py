import pytest

from foodops_ai.knowledge.chunker import chunk_document
from foodops_ai.knowledge.models import LoadedDocument


def test_splits_document_and_preserves_source_positions_and_indices() -> None:
    document = LoadedDocument(source="policy.md", content="abcdefghij")

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=1)

    assert [chunk.content for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.source for chunk in chunks] == ["policy.md"] * 3
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
    assert all(
        document.content[chunk.start_char : chunk.end_char] == chunk.content
        for chunk in chunks
    )


def test_document_shorter_than_chunk_size_returns_one_chunk() -> None:
    document = LoadedDocument(source="short.txt", content="short text")

    chunks = chunk_document(document, chunk_size=20, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].content == "short text"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_chunks_overlap_by_requested_character_count() -> None:
    document = LoadedDocument(source="overlap.txt", content="abcdefghij")

    chunks = chunk_document(document, chunk_size=6, chunk_overlap=2)

    assert [chunk.content for chunk in chunks] == ["abcdef", "efghij"]
    assert chunks[1].start_char == chunks[0].end_char - 2


def test_empty_document_returns_no_chunks() -> None:
    document = LoadedDocument(source="empty.md", content="")

    assert chunk_document(document) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_must_be_positive(chunk_size: int) -> None:
    document = LoadedDocument(source="policy.md", content="content")

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(document, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_overlap", [-1, 4])
def test_chunk_overlap_must_be_valid(chunk_overlap: int) -> None:
    document = LoadedDocument(source="policy.md", content="content")

    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(document, chunk_size=4, chunk_overlap=chunk_overlap)
