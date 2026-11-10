from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from foodops_ai.config import settings
from foodops_ai.knowledge.models import DocumentChunk
from foodops_ai.main import app, get_vector_store


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: Sequence[DocumentChunk] = ()

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        self.chunks = chunks
        return len(chunks)


@pytest.fixture
def upload_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, FakeVectorStore]:
    monkeypatch.setattr(settings, "knowledge_base_path", tmp_path / "knowledge_base")
    monkeypatch.setattr(settings, "upload_max_size_bytes", 1024)
    vector_store = FakeVectorStore()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    client = TestClient(app)
    yield client, vector_store
    app.dependency_overrides.clear()


def test_uploads_and_indexes_a_text_document(
    upload_client: tuple[TestClient, FakeVectorStore],
) -> None:
    client, vector_store = upload_client

    response = client.post(
        "/documents/upload",
        files={"file": ("procedure.txt", b"Confirm the order status.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "procedure.txt"
    assert body["source"].startswith("uploads/")
    assert body["document_count"] == 1
    assert body["chunk_count"] == 1
    assert vector_store.chunks[0].content == "Confirm the order status."


def test_upload_rejects_unsupported_extension(
    upload_client: tuple[TestClient, FakeVectorStore],
) -> None:
    client, _ = upload_client

    response = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", b"content", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_upload_rejects_files_above_configured_size(
    upload_client: tuple[TestClient, FakeVectorStore],
) -> None:
    client, _ = upload_client

    response = client.post(
        "/documents/upload",
        files={"file": ("large.txt", b"x" * 1025, "text/plain")},
    )

    assert response.status_code == 413
