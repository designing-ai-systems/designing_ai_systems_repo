"""
Embedding generation for the Data Service.

Wraps the Model Service's embed() method for chunk and query embedding.

Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
  - Listing 5.12: EmbeddingGenerator class
"""

from typing import Callable, List, Optional

from services.data.models import Chunk

EmbedFn = Callable[[List[str], str], List[List[float]]]


class EmbeddingGenerator:
    """Generates embeddings via the Model Service or an injected callable (Listing 5.12)."""

    def __init__(self, embed_fn: Optional[EmbedFn] = None, model_client=None):
        if embed_fn is not None:
            self._embed_fn = embed_fn
        elif model_client is not None:
            self._embed_fn = model_client.embed
        else:
            raise ValueError("Provide either embed_fn or model_client")

    def embed_chunks(
        self, chunks: List[Chunk], model: str, batch_size: int = 100
    ) -> List[List[float]]:
        texts = [c.text for c in chunks]
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_fn(batch, model))
        return all_embeddings

    def embed_query(self, query: str, model: str) -> List[float]:
        results = self._embed_fn([query], model)
        return results[0]
