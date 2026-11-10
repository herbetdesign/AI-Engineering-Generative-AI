"""Command-line entry point for indexing local knowledge-base documents."""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from foodops_ai.config import settings
from foodops_ai.knowledge import (
    DocumentChunk,
    DocumentLoadError,
    VectorStore,
    chunk_document,
    load_documents,
)


class ChunkStore(Protocol):
    """Minimal interface needed by the knowledge indexing workflow."""

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Persist document chunks and return their count."""


@dataclass(frozen=True)
class IndexingSummary:
    """Counts produced by one knowledge-base indexing run."""

    document_count: int
    chunk_count: int


def index_knowledge_base(
    directory: Path,
    vector_store: ChunkStore | None = None,
) -> IndexingSummary:
    """Load local documents, split them into chunks, and store their embeddings."""

    documents = load_documents(directory)
    chunks: list[DocumentChunk] = [
        chunk for document in documents for chunk in chunk_document(document)
    ]
    if not chunks:
        return IndexingSummary(document_count=len(documents), chunk_count=0)

    store = (
        vector_store
        if vector_store is not None
        else VectorStore(persist_directory=settings.vector_store_path)
    )
    indexed_count = store.index_chunks(chunks)
    return IndexingSummary(
        document_count=len(documents),
        chunk_count=indexed_count,
    )


def main() -> int:
    """Index the configured local knowledge base and print a short summary."""

    try:
        summary = index_knowledge_base(settings.knowledge_base_path)
    except (DocumentLoadError, FileNotFoundError) as error:
        print(f"Não foi possível ler a base de conhecimento: {error}", file=sys.stderr)
        return 1

    if summary.document_count == 0:
        print(
            "Nenhum documento .md ou .txt encontrado em "
            f"'{settings.knowledge_base_path}'."
        )
        return 0

    print(f"Documentos lidos: {summary.document_count}")
    print(f"Chunks indexados: {summary.chunk_count}")
    print(f"Banco vetorial local: {settings.vector_store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
