"""
Embedding manager for clinical content with normalization and consistency checks.

FIX APPLIED: Embedding mismatch bug
- Root cause: The original system used different embedding models/dimensions for
  indexing vs. querying, causing cosine similarity to produce meaningless scores.
- Fix: Centralized embedding model configuration, added dimension validation,
  and normalized all embeddings to unit vectors before storage and comparison.
"""

import hashlib
import structlog
import numpy as np
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from config import get_settings

logger = structlog.get_logger(__name__)


class EmbeddingManager:
    """Manages embeddings with consistency guarantees between indexing and query time."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self._model_name = model_name or settings.openai_embedding_model
        self._embeddings = OpenAIEmbeddings(
            model=self._model_name,
            openai_api_key=settings.openai_api_key,
        )
        self._expected_dimensions: Optional[int] = None
        self._embedding_cache: dict[str, list[float]] = {}

        logger.info(
            "embedding_manager_initialized",
            model=self._model_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_embeddings(self) -> OpenAIEmbeddings:
        return self._embeddings

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _normalize(self, vector: list[float]) -> list[float]:
        """L2-normalize embedding vectors to ensure cosine similarity correctness."""
        arr = np.array(vector, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm == 0:
            logger.warning("zero_norm_embedding_detected")
            return vector
        return (arr / norm).tolist()

    def _validate_dimensions(self, vector: list[float]) -> None:
        if self._expected_dimensions is None:
            self._expected_dimensions = len(vector)
            logger.info("embedding_dimensions_set", dimensions=self._expected_dimensions)
        elif len(vector) != self._expected_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_dimensions}, "
                f"got {len(vector)}. This indicates the embedding model changed between "
                f"indexing and querying. Re-index all documents with the current model."
            )

    async def embed_text(self, text: str) -> list[float]:
        cache_key = self._cache_key(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        raw = await self._embeddings.aembed_query(text)
        self._validate_dimensions(raw)
        normalized = self._normalize(raw)
        self._embedding_cache[cache_key] = normalized

        logger.debug("text_embedded", text_length=len(text), dimensions=len(normalized))
        return normalized

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._embedding_cache:
                results.append(self._embedding_cache[cache_key])
            else:
                results.append([])
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            raw_embeddings = await self._embeddings.aembed_documents(uncached_texts)
            for idx, raw in zip(uncached_indices, raw_embeddings):
                self._validate_dimensions(raw)
                normalized = self._normalize(raw)
                cache_key = self._cache_key(texts[idx])
                self._embedding_cache[cache_key] = normalized
                results[idx] = normalized

        logger.info(
            "documents_embedded",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            computed=len(uncached_texts),
        )
        return results

    def clear_cache(self) -> None:
        self._embedding_cache.clear()
        logger.info("embedding_cache_cleared")
