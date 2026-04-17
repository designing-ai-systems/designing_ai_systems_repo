"""
Chunking strategies for the Data Service.

Break extracted documents into smaller, searchable pieces.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.9: ChunkingStrategy ABC
  - Listing 5.11: Fixed-size chunking implementation
"""

from abc import ABC, abstractmethod
from typing import List

from services.data.models import Chunk, ExtractedDocument


class ChunkingStrategy(ABC):
    """Abstract chunking interface (Listing 5.9)."""

    @abstractmethod
    def chunk(self, document: ExtractedDocument) -> List[Chunk]:
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Word-count-based fixed-size chunking (Listing 5.11)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ExtractedDocument) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        words = text.split()
        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        i = 0

        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            search_start = sum(len(w) + 1 for w in words[:i])
            start_offset = text.find(chunk_words[0], search_start) if chunk_words else 0
            end_offset = start_offset + len(chunk_text)
            chunks.append(
                Chunk(text=chunk_text, start_offset=start_offset, end_offset=end_offset)
            )
            i += step

        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Splits text using a hierarchy of separators: paragraph, line, space."""

    _SEPARATORS = ["\n\n", "\n", " "]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ExtractedDocument) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []
        pieces = self._split_recursive(text, 0)
        return pieces

    def _split_recursive(self, text: str, sep_idx: int) -> List[Chunk]:
        words = text.split()
        if len(words) <= self.chunk_size:
            if text.strip():
                return [Chunk(text=text.strip(), start_offset=0, end_offset=len(text.strip()))]
            return []

        if sep_idx >= len(self._SEPARATORS):
            from services.data.models import DocumentSection

            sub_doc = ExtractedDocument(sections=[DocumentSection(content=text)])
            return FixedSizeChunking(self.chunk_size, self.chunk_overlap).chunk(sub_doc)

        sep = self._SEPARATORS[sep_idx]
        parts = text.split(sep)
        chunks: List[Chunk] = []
        current_parts: List[str] = []
        current_word_count = 0

        for part in parts:
            part_words = len(part.split())
            if current_word_count + part_words > self.chunk_size and current_parts:
                merged = sep.join(current_parts).strip()
                if merged:
                    sub_chunks = self._split_recursive(merged, sep_idx + 1)
                    chunks.extend(sub_chunks)
                current_parts = [part]
                current_word_count = part_words
            else:
                current_parts.append(part)
                current_word_count += part_words

        if current_parts:
            merged = sep.join(current_parts).strip()
            if merged:
                sub_chunks = self._split_recursive(merged, sep_idx + 1)
                chunks.extend(sub_chunks)

        return chunks


class StructureAwareChunking(ChunkingStrategy):
    """Uses document sections; prepends headings; falls back to fixed for oversized sections."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ExtractedDocument) -> List[Chunk]:
        if not document.sections:
            return []

        chunks: List[Chunk] = []
        for section in document.sections:
            content = section.content
            if section.heading:
                content = f"{section.heading}\n\n{content}"

            words = content.split()
            if len(words) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        text=content.strip(),
                        heading=section.heading,
                        start_offset=0,
                        end_offset=len(content.strip()),
                    )
                )
            else:
                from services.data.models import DocumentSection as DS

                sub_doc = ExtractedDocument(sections=[DS(content=content)])
                sub_chunks = FixedSizeChunking(self.chunk_size, self.chunk_overlap).chunk(sub_doc)
                for sc in sub_chunks:
                    sc.heading = section.heading
                chunks.extend(sub_chunks)

        return chunks
