from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single loaded document before chunking."""

    content: str
    source: str   # file path or URL
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Loader(Protocol):
    def load(self, path: Path) -> list[Document]: ...


# ── PDF ───────────────────────────────────────────────────────────────────────

class PDFLoader:
    """Load a PDF file using pypdf."""

    def load(self, path: Path) -> list[Document]:
        try:
            import pypdf
        except ImportError as e:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf") from e

        reader = pypdf.PdfReader(str(path))
        pages: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        content=text,
                        source=str(path),
                        metadata={"page": i + 1, "total_pages": len(reader.pages)},
                    )
                )
        logger.debug("PDFLoader: loaded %d pages from %s", len(pages), path)
        return pages


# ── Markdown / Plain Text ─────────────────────────────────────────────────────

class MarkdownLoader:
    """Load Markdown or plain-text files."""

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [Document(content=content, source=str(path), metadata={"format": "markdown"})]


class TextLoader:
    """Load any plain-text file."""

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [Document(content=content, source=str(path), metadata={"format": "text"})]


# ── HTML ──────────────────────────────────────────────────────────────────────

class HTMLLoader:
    """Strip HTML tags and load the visible text."""

    def load(self, path: Path) -> list[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError("beautifulsoup4 is required for HTML loading: pip install beautifulsoup4") from e

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        # Remove script/style nodes
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return [Document(content=text, source=str(path), metadata={"format": "html"})]


# ── Code ──────────────────────────────────────────────────────────────────────

class CodeLoader:
    """Load source-code files verbatim, adding language metadata."""

    _EXT_MAP: dict[str, str] = {
        ".py": "python", ".rs": "rust", ".go": "go", ".ts": "typescript",
        ".js": "javascript", ".java": "java", ".cpp": "cpp", ".c": "c",
        ".rb": "ruby", ".sh": "shell",
    }

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        lang = self._EXT_MAP.get(path.suffix.lower(), "unknown")
        return [
            Document(
                content=content,
                source=str(path),
                metadata={"format": "code", "language": lang},
            )
        ]


# ── Router ────────────────────────────────────────────────────────────────────

_SUFFIX_LOADERS: dict[str, type] = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".py": CodeLoader,
    ".rs": CodeLoader,
    ".go": CodeLoader,
    ".ts": CodeLoader,
    ".js": CodeLoader,
    ".java": CodeLoader,
    ".cpp": CodeLoader,
    ".c": CodeLoader,
    ".rb": CodeLoader,
    ".sh": CodeLoader,
}


def get_loader(path: Path) -> Loader:
    """Return the appropriate loader for *path* based on its suffix."""
    loader_cls = _SUFFIX_LOADERS.get(path.suffix.lower(), TextLoader)
    return loader_cls()  # type: ignore[return-value]


def load_path(path: Path) -> Iterator[Document]:
    """Recursively load a file or directory, yielding Document objects."""
    if path.is_file():
        loader = get_loader(path)
        yield from loader.load(path)
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in _SUFFIX_LOADERS:
                try:
                    loader = get_loader(child)
                    yield from loader.load(child)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to load %s — skipping", child, exc_info=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
