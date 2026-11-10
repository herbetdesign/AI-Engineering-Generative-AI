from pathlib import Path
from types import SimpleNamespace

import pytest

from foodops_ai.knowledge.loader import (
    DocumentLoadError,
    load_document,
    load_documents,
)


def test_loads_markdown_file(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("Markdown content", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == "guide.md"
    assert documents[0].content == "Markdown content"


def test_loads_text_file_with_uppercase_extension(tmp_path: Path) -> None:
    (tmp_path / "notes.TXT").write_text("Plain text", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [(document.source, document.content) for document in documents] == [
        ("notes.TXT", "Plain text")
    ]


def test_loads_files_from_subdirectories(tmp_path: Path) -> None:
    nested_directory = tmp_path / "operations" / "delivery"
    nested_directory.mkdir(parents=True)
    (nested_directory / "late.md").write_text("Delayed order", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents[0].source == "operations/delivery/late.md"


def test_documents_are_sorted_deterministically(tmp_path: Path) -> None:
    (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "Alpha.md").write_text("a", encoding="utf-8")
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "beta.md").write_text("b", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.source for document in documents] == [
        "Alpha.md",
        "nested/beta.md",
        "zeta.txt",
    ]


def test_ignores_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "guide.csv").write_text("Ignored", encoding="utf-8")

    assert load_documents(tmp_path) == []


def test_ignores_empty_and_whitespace_only_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "spaces.txt").write_text("  \n\t", encoding="utf-8")

    assert load_documents(tmp_path) == []


def test_missing_directory_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")


def test_invalid_utf8_raises_document_load_error(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_bytes(b"\xff\xfe")

    with pytest.raises(DocumentLoadError, match="invalid.md"):
        load_documents(tmp_path)


def test_loads_pdf_text_by_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-test")

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    monkeypatch.setattr(
        "foodops_ai.knowledge.loader.PdfReader",
        lambda stream: SimpleNamespace(
            pages=[FakePage("Page one"), FakePage("Page two")]
        ),
    )

    document = load_document(pdf_path, tmp_path)

    assert document.source == "manual.pdf"
    assert document.content == "Page one\n\nPage two"


def test_pdf_parse_errors_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a PDF")

    def fail_reader(stream: object) -> None:
        raise ValueError("invalid PDF")

    monkeypatch.setattr("foodops_ai.knowledge.loader.PdfReader", fail_reader)

    with pytest.raises(DocumentLoadError, match="broken.pdf"):
        load_documents(tmp_path)
