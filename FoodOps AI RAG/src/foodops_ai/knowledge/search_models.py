"""Models for semantic retrieval results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its source metadata and vector distance."""

    content: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    distance: float
