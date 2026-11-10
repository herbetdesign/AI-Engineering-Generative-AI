"""Persistent local vector storage and semantic retrieval with Chroma."""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction

from foodops_ai.knowledge.models import DocumentChunk
from foodops_ai.knowledge.search_models import SearchResult


class VectorStore:
    """Index document chunks and search them in a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "foodops-knowledge",
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> None:
        """Open or create a persistent collection using cosine distance.

        When no embedding function is supplied, Chroma uses its default local
        embedding function. Its model is downloaded on first use if not cached.
        """

        self._client = chromadb.PersistentClient(path=str(persist_directory))
        collection_options: dict[str, object] = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if embedding_function is not None:
            collection_options["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**collection_options)

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Upsert chunks and remove outdated chunks for updated source files.

        Chunks are grouped by their source so a changed document replaces its
        earlier chunk set instead of leaving stale trailing chunks behind.
        Returns the number of non-empty chunks indexed.
        """

        chunks_by_source: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.content:
                chunks_by_source[chunk.source].append(chunk)

        indexed_count = 0
        for source, source_chunks in chunks_by_source.items():
            ids = [f"{source}::{chunk.chunk_index}" for chunk in source_chunks]
            existing = self._collection.get(where={"source": source}, include=[])
            existing_ids = set(existing["ids"])
            current_ids = set(ids)

            self._collection.upsert(
                ids=ids,
                documents=[chunk.content for chunk in source_chunks],
                metadatas=[
                    {
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                    for chunk in source_chunks
                ],
            )

            stale_ids = existing_ids - current_ids
            if stale_ids:
                self._collection.delete(ids=sorted(stale_ids))
            indexed_count += len(source_chunks)

        return indexed_count

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Return the closest matching chunks for a natural-language query."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        collection_size = self._collection.count()
        if collection_size == 0:
            return []

        matches = self._collection.query(
            query_texts=[query],
            n_results=min(limit, collection_size),
            include=["documents", "metadatas", "distances"],
        )

        documents = matches["documents"][0] or []
        metadatas = matches["metadatas"][0] or []
        distances = matches["distances"][0] or []

        results: list[SearchResult] = []
        for content, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            metadata = metadata or {}
            results.append(
                SearchResult(
                    content=content,
                    source=str(metadata["source"]),
                    chunk_index=int(metadata["chunk_index"]),
                    start_char=int(metadata["start_char"]),
                    end_char=int(metadata["end_char"]),
                    distance=float(distance),
                )
            )

        return results
