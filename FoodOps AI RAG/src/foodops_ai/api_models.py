"""HTTP request and response models for the FoodOps API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Question, retrieval limit, and optional conversation identifier."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("question must not be blank")
        return normalized_value

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("conversation_id must not be blank")
        return normalized_value


class SourceReference(BaseModel):
    """Location of a retrieved chunk used as context for an answer."""

    reference_id: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int


class AskResponse(BaseModel):
    """Grounded answer, retrieved sources, and conversation identifier."""

    answer: str
    sources: list[SourceReference]
    conversation_id: str


class ConversationHistoryItem(BaseModel):
    """One stored question-answer pair."""

    question: str
    answer: str
    sources: list[SourceReference]
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    """All stored turns for a conversation."""

    conversation_id: str
    items: list[ConversationHistoryItem]


class DocumentUploadResponse(BaseModel):
    """Indexing summary returned after a document upload."""

    filename: str
    source: str
    document_count: int
    chunk_count: int
