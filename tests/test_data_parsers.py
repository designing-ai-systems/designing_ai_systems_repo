"""Tests for Data Service document parsers (book Listing 5.4)."""

from services.data.models import ExtractedDocument
from services.data.parsers import (
    DocumentParser,
    MarkdownParser,
    PlainTextParser,
    detect_format,
)


class TestDetectFormat:
    def test_pdf_magic_bytes(self):
        assert detect_format("file.bin", b"%PDF-1.4 rest of content") == "pdf"

    def test_docx_magic_bytes(self):
        content = b"PK\x03\x04" + b"\x00" * 100
        assert detect_format("file.bin", content) == "docx"

    def test_markdown_extension(self):
        assert detect_format("readme.md", b"# Hello") == "markdown"

    def test_markdown_extension_upper(self):
        assert detect_format("README.MD", b"# Hello") == "markdown"

    def test_txt_extension(self):
        assert detect_format("notes.txt", b"plain text") == "text"

    def test_html_extension(self):
        assert detect_format("page.html", b"<html>") == "html"

    def test_htm_extension(self):
        assert detect_format("page.htm", b"<html>") == "html"

    def test_unknown_extension(self):
        assert detect_format("data.xyz", b"unknown") == "text"

    def test_no_extension(self):
        assert detect_format("README", b"some content") == "text"


class TestPlainTextParser:
    def test_simple_text(self):
        parser = PlainTextParser()
        result = parser.parse(b"Hello, world!", "test.txt")
        assert isinstance(result, ExtractedDocument)
        assert len(result.sections) == 1
        assert result.sections[0].content == "Hello, world!"

    def test_multiline_text(self):
        parser = PlainTextParser()
        content = b"Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = parser.parse(content, "test.txt")
        assert len(result.sections) == 3
        assert result.sections[0].content == "Paragraph one."
        assert result.sections[1].content == "Paragraph two."
        assert result.sections[2].content == "Paragraph three."

    def test_utf8_content(self):
        parser = PlainTextParser()
        result = parser.parse("Héllo wörld".encode("utf-8"), "test.txt")
        assert result.sections[0].content == "Héllo wörld"

    def test_empty_content(self):
        parser = PlainTextParser()
        result = parser.parse(b"", "test.txt")
        assert len(result.sections) == 0


class TestMarkdownParser:
    def test_simple_markdown(self):
        parser = MarkdownParser()
        content = b"# Title\n\nSome paragraph text.\n\n## Section 2\n\nMore text."
        result = parser.parse(content, "doc.md")
        assert isinstance(result, ExtractedDocument)
        assert len(result.sections) >= 2

    def test_heading_extraction(self):
        parser = MarkdownParser()
        content = b"# Main Title\n\nIntro text.\n\n## Sub Section\n\nSub text."
        result = parser.parse(content, "doc.md")
        headings = [s.heading for s in result.sections if s.heading]
        assert "Main Title" in headings
        assert "Sub Section" in headings

    def test_heading_levels(self):
        parser = MarkdownParser()
        content = b"# H1\n\nText1\n\n## H2\n\nText2\n\n### H3\n\nText3"
        result = parser.parse(content, "doc.md")
        levels = {s.heading: s.level for s in result.sections if s.heading}
        assert levels["H1"] == 1
        assert levels["H2"] == 2
        assert levels["H3"] == 3

    def test_empty_markdown(self):
        parser = MarkdownParser()
        result = parser.parse(b"", "doc.md")
        assert len(result.sections) == 0

    def test_no_headings(self):
        parser = MarkdownParser()
        content = b"Just a plain paragraph.\n\nAnother paragraph."
        result = parser.parse(content, "doc.md")
        assert len(result.sections) >= 1
        assert result.text.strip() != ""


class TestDocumentParserABC:
    def test_is_abstract(self):
        import abc

        assert abc.ABC in DocumentParser.__bases__ or hasattr(DocumentParser, "__abstractmethods__")
        assert "parse" in DocumentParser.__abstractmethods__
