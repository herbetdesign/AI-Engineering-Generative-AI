"""Load supported text and PDF documents from a knowledge base."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from foodops_ai.knowledge.models import LoadedDocument

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
TEXT_EXTENSIONS = {".md", ".txt"}


class DocumentLoadError(Exception):
    """Raised when a supported document cannot be read or decoded."""


def _read_pdf(path: Path, relative_source: str) -> str:
    """Extract selectable text from each page of a PDF document."""

    try:
        reader = PdfReader(BytesIO(path.read_bytes()))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise DocumentLoadError(
            f"Could not read PDF document '{relative_source}': {error}"
        ) from error


def load_document(path: Path, directory: Path) -> LoadedDocument:
    """Load one supported document and preserve its relative source path."""

    relative_source = path.relative_to(directory).as_posix()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document extension: {path.suffix}")
    try:
        if path.suffix.lower() in TEXT_EXTENSIONS:
            content = path.read_text(encoding="utf-8")
        else:
            content = _read_pdf(path, relative_source)
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentLoadError(
            f"Could not read document '{relative_source}': {error}"
        ) from error
    return LoadedDocument(source=relative_source, content=content)


def load_documents(directory: Path) -> list[LoadedDocument]:
    """Load non-empty Markdown, text, and PDF files in deterministic order.

    Paths stored in LoadedDocument.source are relative to directory and use
    forward slashes, so source names are consistent across platforms.
    """

    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: (
            path.relative_to(directory).as_posix().casefold(),
            path.relative_to(directory).as_posix(),
        ),
    )

    documents: list[LoadedDocument] = []
    for path in paths:
        document = load_document(path, directory)
        if document.content.strip():
            documents.append(document)

    return documents
