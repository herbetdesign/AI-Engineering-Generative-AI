"""Data structures used by the knowledge document pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedDocument:
    """A text document loaded from the local knowledge base."""

    source: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    """A portion of a document with its source and character positions."""

    content: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
