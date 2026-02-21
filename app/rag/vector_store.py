"""
ChromaDB vector store with similarity threshold filtering and metadata re-ranking.

FIX APPLIED: Irrelevant content retrieval
- Root cause: The vector store returned top-K results regardless of similarity
  score, meaning low-relevance chunks were included in the context window. No
  metadata-based re-ranking existed, so a general "medication safety" chunk could
  outrank a specific "diabetes medication" chunk for a diabetes query.
- Fix: Added hard similarity threshold (configurable, default 0.78) to discard
  low-relevance matches. Implemented metadata-aware re-ranking that boosts chunks
  whose category/tags match the query intent. Added context window budget tracking
  to prevent exceeding the LLM's effective context window.
"""

import structlog
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.rag.embeddings import EmbeddingManager
from app.rag.chunking import ClinicalChunk, estimate_tokens
from config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: dict
    token_estimate: int


class ClinicalVectorStore:
    def __init__(self, embedding_manager: EmbeddingManager):
        settings = get_settings()
        self._embedding_manager = embedding_manager
        self._similarity_threshold = settings.rag_similarity_threshold
        self._top_k = settings.rag_top_k
        self._max_context_tokens = settings.rag_max_context_tokens

        self._store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_manager.langchain_embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(
            "vector_store_initialized",
            collection=settings.chroma_collection_name,
            threshold=self._similarity_threshold,
        )

    async def index_chunks(self, chunks: list[ClinicalChunk]) -> int:
        if not chunks:
            return 0

        documents = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

        ids = [
            f"{chunk.metadata.get('source_id', 'unknown')}_{i}"
            for i, chunk in enumerate(chunks)
        ]

        self._store.add_documents(documents, ids=ids)

        logger.info("chunks_indexed", count=len(chunks))
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        category_hint: str | None = None,
        tag_hints: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant clinical content with threshold filtering
        and metadata-aware re-ranking.
        """
        k = top_k or self._top_k
        fetch_k = k * 3  # over-fetch to allow filtering

        raw_results = self._store.similarity_search_with_relevance_scores(
            query, k=fetch_k
        )

        filtered = []
        for doc, score in raw_results:
            if score < self._similarity_threshold:
                logger.debug(
                    "chunk_below_threshold",
                    score=round(score, 4),
                    source=doc.metadata.get("source_id"),
                    threshold=self._similarity_threshold,
                )
                continue

            boosted_score = self._apply_metadata_boost(
                score, doc.metadata, category_hint, tag_hints
            )

            filtered.append(RetrievalResult(
                text=doc.page_content,
                score=boosted_score,
                metadata=doc.metadata,
                token_estimate=estimate_tokens(doc.page_content),
            ))

        filtered.sort(key=lambda r: r.score, reverse=True)

        budget_results = self._apply_token_budget(filtered[:k])

        logger.info(
            "retrieval_complete",
            query_length=len(query),
            raw_results=len(raw_results),
            after_threshold=len(filtered),
            after_budget=len(budget_results),
            category_hint=category_hint,
        )

        return budget_results

    def _apply_metadata_boost(
        self,
        score: float,
        metadata: dict,
        category_hint: str | None,
        tag_hints: list[str] | None,
    ) -> float:
        """Boost scores for chunks whose metadata aligns with query intent."""
        boost = 0.0

        if category_hint and metadata.get("category") == category_hint:
            boost += 0.05

        if tag_hints:
            chunk_tags = set(metadata.get("tags", []))
            overlap = chunk_tags.intersection(set(tag_hints))
            if overlap:
                boost += 0.02 * len(overlap)

        return min(score + boost, 1.0)

    def _apply_token_budget(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Trim results to fit within the context window token budget."""
        budgeted: list[RetrievalResult] = []
        total_tokens = 0

        for result in results:
            if total_tokens + result.token_estimate > self._max_context_tokens:
                logger.info(
                    "context_budget_reached",
                    included=len(budgeted),
                    total_tokens=total_tokens,
                    budget=self._max_context_tokens,
                )
                break
            budgeted.append(result)
            total_tokens += result.token_estimate

        return budgeted

    async def get_collection_stats(self) -> dict:
        collection = self._store._collection
        return {
            "name": collection.name,
            "count": collection.count(),
        }

    async def clear(self) -> None:
        """Remove all documents. Use during re-indexing."""
        collection = self._store._collection
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
        logger.info("vector_store_cleared")
