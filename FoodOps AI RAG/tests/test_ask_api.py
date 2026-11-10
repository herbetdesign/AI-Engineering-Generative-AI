from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from foodops_ai.answer_generator import (
    OpenAIAnswerGenerator,
)
from foodops_ai.conversation_store import ConversationStore, ConversationTurn
from foodops_ai.knowledge.search_models import SearchResult
from foodops_ai.main import (
    app,
    get_answer_generator,
    get_conversation_store,
    get_vector_store,
)


class FakeVectorStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_question: str | None = None
        self.last_limit: int | None = None

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.last_question = query
        self.last_limit = limit
        return self.results

    def index_chunks(self, chunks: Sequence[object]) -> int:
        return len(chunks)


class FakeAnswerGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.called = False
        self.received_sources: Sequence[SearchResult] = []
        self.received_history: Sequence[ConversationTurn] = []

    def generate(
        self,
        question: str,
        sources: Sequence[SearchResult],
        history: Sequence[ConversationTurn] = (),
    ) -> str:
        self.called = True
        self.received_sources = sources
        self.received_history = history
        return self.answer


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(output_text="A resposta está fundamentada. [S1]")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app.dependency_overrides.clear()
    app.dependency_overrides[get_conversation_store] = lambda: ConversationStore(
        tmp_path / "conversations.sqlite3"
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_ask_returns_answer_sources_and_conversation_id(
    client: TestClient,
) -> None:
    result = SearchResult(
        content="Confirme o horário estimado do pedido.",
        source="late_delivery.md",
        chunk_index=1,
        start_char=500,
        end_char=800,
        distance=0.12,
    )
    vector_store = FakeVectorStore([result])
    answer_generator = FakeAnswerGenerator(
        "Consulte o horário estimado do pedido. [S1]"
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_answer_generator] = lambda: answer_generator

    response = client.post(
        "/ask",
        json={
            "question": "Meu pedido atrasou?",
            "top_k": 2,
            "conversation_id": "delivery-session",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Consulte o horário estimado do pedido. [S1]",
        "sources": [
            {
                "reference_id": "S1",
                "source": "late_delivery.md",
                "chunk_index": 1,
                "start_char": 500,
                "end_char": 800,
            }
        ],
        "conversation_id": "delivery-session",
    }
    assert vector_store.last_question == "Meu pedido atrasou?"
    assert vector_store.last_limit == 2
    assert answer_generator.received_sources == [result]
    assert answer_generator.received_history == []


def test_ask_reuses_history_for_follow_up(client: TestClient) -> None:
    result = SearchResult(
        content="Registre os fatos observados e a orientação dada.",
        source="late_delivery.md",
        chunk_index=0,
        start_char=0,
        end_char=50,
        distance=0.1,
    )
    vector_store = FakeVectorStore([result])
    answer_generator = FakeAnswerGenerator("Registre os fatos verificados. [S1]")
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_answer_generator] = lambda: answer_generator
    conversation_id = "delivery-session"

    first = client.post(
        "/ask",
        json={"question": "Meu pedido atrasou?", "conversation_id": conversation_id},
    )
    second = client.post(
        "/ask",
        json={
            "question": "O que preciso registrar?",
            "conversation_id": conversation_id,
        },
    )

    assert first.status_code == second.status_code == 200
    assert answer_generator.received_history[-1].question == "Meu pedido atrasou?"
    history = client.get(f"/conversations/{conversation_id}")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 2


def test_ask_without_matches_saves_fallback_without_calling_llm(
    client: TestClient,
) -> None:
    vector_store = FakeVectorStore([])
    answer_generator = FakeAnswerGenerator("Should not be used")
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_answer_generator] = lambda: answer_generator

    response = client.post(
        "/ask",
        json={"question": "Onde está meu pedido?", "conversation_id": "no-match"},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "informações suficientes" in response.json()["answer"]
    assert answer_generator.called is False
    history = client.get("/conversations/no-match")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 1


def test_ask_rejects_blank_questions(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_missing_conversation_returns_not_found(client: TestClient) -> None:
    response = client.get("/conversations/unknown")

    assert response.status_code == 404


def test_ask_reports_missing_openai_api_key(client: TestClient) -> None:
    vector_store = FakeVectorStore(
        [
            SearchResult(
                content="Example context",
                source="guide.md",
                chunk_index=0,
                start_char=0,
                end_char=14,
                distance=0.1,
            )
        ]
    )
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_answer_generator] = lambda: OpenAIAnswerGenerator(
        api_key=None,
        model="gpt-6-luna",
    )

    response = client.post("/ask", json={"question": "A pergunta tem resposta?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "OPENAI_API_KEY is not configured."


def test_openai_generator_uses_responses_api_with_grounding_context() -> None:
    fake_client = FakeOpenAIClient()
    generator = OpenAIAnswerGenerator(
        api_key=SecretStr("not-a-real-key"),
        model="gpt-6-luna",
        client=fake_client,
    )
    source = SearchResult(
        content="O cliente deve consultar o horário estimado.",
        source="late_delivery.md",
        chunk_index=0,
        start_char=0,
        end_char=46,
        distance=0.1,
    )

    answer = generator.generate("Meu pedido atrasou?", [source])

    assert answer == "A resposta está fundamentada. [S1]"
    assert fake_client.responses.request is not None
    assert fake_client.responses.request["model"] == "gpt-6-luna"
    assert (
        "apenas o contexto fornecido" in fake_client.responses.request["instructions"]
    )
    assert "[S1] source=late_delivery.md" in fake_client.responses.request["input"]
