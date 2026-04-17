"""
Document parsers for the Data Service.

Format-specific parsers that extract clean text with structural information.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.4: DocumentParser ABC, ExtractedDocument, DocumentSection
"""

import re
from abc import ABC, abstractmethod
from typing import List

from services.data.models import DocumentSection, ExtractedDocument


class DocumentParser(ABC):
    """Abstract parser interface (Listing 5.4)."""

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        pass


class PlainTextParser(DocumentParser):
    """Parses plain text files into paragraph-based sections."""

    def parse(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = file_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            return ExtractedDocument(sections=[])
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections = [DocumentSection(content=p) for p in paragraphs]
        return ExtractedDocument(sections=sections)


class MarkdownParser(DocumentParser):
    """Parses Markdown files, extracting headings as section boundaries."""

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        text = file_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            return ExtractedDocument(sections=[])

        sections: List[DocumentSection] = []
        heading_matches = list(self._HEADING_RE.finditer(text))

        if not heading_matches:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return ExtractedDocument(sections=[DocumentSection(content=p) for p in paragraphs])

        first_match = heading_matches[0]
        if first_match.start() > 0:
            preamble = text[: first_match.start()].strip()
            if preamble:
                sections.append(DocumentSection(content=preamble))

        for i, match in enumerate(heading_matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            start = match.end()
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
            body = text[start:end].strip()
            if body or heading:
                sections.append(DocumentSection(content=body, heading=heading, level=level))

        return ExtractedDocument(sections=sections)


_EXTENSION_MAP = {
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
}


def detect_format(filename: str, file_bytes: bytes) -> str:
    """Detect document format from magic bytes, then extension fallback."""
    if file_bytes[:5] == b"%PDF-":
        return "pdf"
    if file_bytes[:4] == b"PK\x03\x04":
        return "docx"

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    return _EXTENSION_MAP.get(ext, "text")
