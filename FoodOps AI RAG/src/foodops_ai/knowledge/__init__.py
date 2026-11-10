"""Document loading, chunking, and retrieval for the knowledge base."""

from foodops_ai.knowledge.chunker import chunk_document
from foodops_ai.knowledge.loader import DocumentLoadError, load_documents
from foodops_ai.knowledge.models import DocumentChunk, LoadedDocument
from foodops_ai.knowledge.search_models import SearchResult
from foodops_ai.knowledge.vector_store import VectorStore

__all__ = [
    "DocumentChunk",
    "DocumentLoadError",
    "LoadedDocument",
    "SearchResult",
    "VectorStore",
    "chunk_document",
    "load_documents",
]
