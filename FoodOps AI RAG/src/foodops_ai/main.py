"""FastAPI application entry point."""

import logging
import re
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from foodops_ai.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    InsufficientAPICreditsError,
    MissingAPIKeyError,
    OpenAIAnswerGenerator,
)
from foodops_ai.api_models import (
    AskRequest,
    AskResponse,
    ConversationHistoryItem,
    ConversationHistoryResponse,
    DocumentUploadResponse,
    SourceReference,
)
from foodops_ai.config import settings
from foodops_ai.conversation_store import ConversationStore
from foodops_ai.knowledge import SearchResult, VectorStore, chunk_document
from foodops_ai.knowledge.loader import DocumentLoadError, load_document

logger = logging.getLogger(__name__)
app = FastAPI(title=settings.app_name)
ALLOWED_UPLOAD_EXTENSIONS = {".md", ".txt", ".pdf"}
FALLBACK_ANSWER = "Não encontrei informações suficientes na base de conhecimento."


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the process-wide persistent knowledge base."""

    return VectorStore(persist_directory=settings.vector_store_path)


@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    """Return the configured answer generator without making a network call."""

    return OpenAIAnswerGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    """Return the process-wide local conversation history store."""

    return ConversationStore(database_path=settings.conversation_db_path)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the application health status."""

    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """Save and index one Markdown, text, or text-based PDF document."""

    original_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Envie um arquivo .md, .txt ou .pdf.",
        )

    content = await file.read(settings.upload_max_size_bytes + 1)
    if len(content) > settings.upload_max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail="O arquivo excede o limite de 10 MB.",
        )
    if not content:
        raise HTTPException(status_code=422, detail="O arquivo está vazio.")

    safe_stem = (
        re.sub(r"[^A-Za-z0-9_-]+", "_", Path(original_name).stem).strip("_")
        or "document"
    )
    stored_name = f"{uuid4().hex}_{safe_stem}{suffix}"
    upload_directory = settings.knowledge_base_path / "uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)
    target_path = upload_directory / stored_name
    target_path.write_bytes(content)

    try:
        document = load_document(target_path, settings.knowledge_base_path)
        if not document.content.strip():
            raise DocumentLoadError(
                "The file contains no extractable text. Scanned PDFs need OCR."
            )
        chunks = chunk_document(document)
        if not chunks:
            raise DocumentLoadError("The file contains no indexable text.")
        indexed_count = vector_store.index_chunks(chunks)
    except DocumentLoadError as error:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        target_path.unlink(missing_ok=True)
        logger.exception("Could not index an uploaded document.")
        raise HTTPException(
            status_code=500,
            detail="Não foi possível indexar o arquivo enviado.",
        ) from error

    return DocumentUploadResponse(
        filename=original_name,
        source=document.source,
        document_count=1,
        chunk_count=indexed_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    answer_generator: AnswerGenerator = Depends(get_answer_generator),
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> AskResponse:
    """Answer a question with retrieved knowledge and recent conversation context."""

    conversation_id = request.conversation_id or str(uuid4())
    history = conversation_store.get_recent_turns(conversation_id, limit=5)
    results: list[SearchResult] = vector_store.search(request.question, request.top_k)

    if not results:
        answer = FALLBACK_ANSWER
        sources: list[SourceReference] = []
    else:
        try:
            answer = answer_generator.generate(request.question, results, history)
        except InsufficientAPICreditsError as error:
            raise HTTPException(
                status_code=402,
                detail=(
                    "A conta OpenAI está sem créditos de API. "
                    "Confira a cobrança ou os créditos disponíveis."
                ),
            ) from error
        except MissingAPIKeyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except AnswerGenerationError as error:
            raise HTTPException(
                status_code=502,
                detail="Não foi possível gerar uma resposta neste momento.",
            ) from error

        sources = [
            SourceReference(
                reference_id=f"S{index}",
                source=result.source,
                chunk_index=result.chunk_index,
                start_char=result.start_char,
                end_char=result.end_char,
            )
            for index, result in enumerate(results, start=1)
        ]

    conversation_store.add_turn(
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
        sources=[source.model_dump() for source in sources],
    )
    return AskResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
    )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
def get_conversation(
    conversation_id: str,
    conversation_store: ConversationStore = Depends(get_conversation_store),
) -> ConversationHistoryResponse:
    """Return the stored turns for a conversation."""

    turns = conversation_store.get_history(conversation_id)
    if not turns:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        items=[
            ConversationHistoryItem(
                question=turn.question,
                answer=turn.answer,
                sources=[SourceReference(**source) for source in turn.sources],
                created_at=turn.created_at,
            )
            for turn in turns
        ],
    )
