"""Split loaded documents into overlapping character-based chunks."""

from foodops_ai.knowledge.models import DocumentChunk, LoadedDocument


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Split a document into bounded character ranges with optional overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not document.content:
        return []

    chunks: list[DocumentChunk] = []
    start_char = 0
    while start_char < len(document.content):
        end_char = min(start_char + chunk_size, len(document.content))
        content = document.content[start_char:end_char]
        if content:
            chunks.append(
                DocumentChunk(
                    content=content,
                    source=document.source,
                    chunk_index=len(chunks),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        if end_char == len(document.content):
            break
        start_char = end_char - chunk_overlap

    return chunks
