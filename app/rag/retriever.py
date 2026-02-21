"""
High-level RAG retriever that orchestrates the full pipeline:
content loading -> chunking -> indexing -> query -> context assembly.

FIX APPLIED: Context window optimization
- Root cause: The context sent to the LLM included all retrieved chunks
  concatenated without structure, often exceeding the model's effective
  attention window and diluting relevant information.
- Fix: Structured context assembly with clear section delineation, source
  attribution per chunk, and token-budget-aware truncation. Added query
  intent classification to route queries to the correct content category.
"""

import yaml
import structlog
from pathlib import Path
from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import ClinicalVectorStore, RetrievalResult
from app.rag.chunking import split_clinical_content
from config import get_settings

logger = structlog.get_logger(__name__)

INTENT_KEYWORDS: dict[str, list[str]] = {
    "medication_management": [
        "medication", "medicine", "drug", "prescription", "dose", "dosage",
        "side effect", "refill", "adherence", "taking",
    ],
    "scheduling": [
        "appointment", "schedule", "reschedule", "reminder", "visit",
        "calendar", "date", "time",
    ],
    "care_transitions": [
        "discharge", "surgery", "hospital", "post-op", "recovery",
        "follow-up", "wound", "incision",
    ],
    "chronic_care": [
        "diabetes", "hypertension", "blood pressure", "heart failure",
        "chronic", "A1C", "glucose", "monitoring", "vitals",
    ],
    "preventive_care": [
        "screening", "preventive", "mammogram", "colonoscopy",
        "vaccination", "annual", "wellness",
    ],
}


class ClinicalRAGRetriever:
    def __init__(self):
        self._embedding_manager = EmbeddingManager()
        self._vector_store = ClinicalVectorStore(self._embedding_manager)
        self._templates_loaded = False

    @property
    def vector_store(self) -> ClinicalVectorStore:
        return self._vector_store

    async def load_clinical_content(
        self, data_path: str = "clinical_data/clinical_scripts.yaml"
    ) -> int:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Clinical data file not found: {data_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        all_chunks = []

        for template in data.get("templates", []):
            chunks = split_clinical_content(
                text=template["script"],
                source_id=template["id"],
                source_category=template["category"],
                source_title=template["title"],
                tags=template.get("tags", []),
            )
            all_chunks.extend(chunks)

        for knowledge in data.get("clinical_knowledge", []):
            chunks = split_clinical_content(
                text=knowledge["content"],
                source_id=knowledge["id"],
                source_category=knowledge["category"],
                source_title=knowledge["id"].replace("_", " ").title(),
                tags=knowledge.get("tags", []),
            )
            all_chunks.extend(chunks)

        indexed = await self._vector_store.index_chunks(all_chunks)
        self._templates_loaded = True

        logger.info("clinical_content_loaded", total_chunks=indexed)
        return indexed

    def classify_intent(self, query: str) -> tuple[str | None, list[str]]:
        """Classify query intent to enable targeted retrieval."""
        query_lower = query.lower()
        scores: dict[str, int] = {}

        for category, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return None, []

        best_category = max(scores, key=scores.get)  # type: ignore[arg-type]
        relevant_tags = INTENT_KEYWORDS.get(best_category, [])

        logger.debug(
            "intent_classified",
            query=query[:80],
            category=best_category,
            score=scores[best_category],
        )
        return best_category, relevant_tags

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        category, tags = self.classify_intent(query)
        return await self._vector_store.retrieve(
            query=query,
            category_hint=category,
            tag_hints=tags,
        )

    def assemble_context(self, results: list[RetrievalResult]) -> str:
        """
        Assemble retrieved chunks into a structured context block
        with clear delineation and source attribution.
        """
        if not results:
            return "No relevant clinical content found for this query."

        settings = get_settings()
        sections: list[str] = []
        total_tokens = 0

        sections.append(
            "=== RETRIEVED CLINICAL CONTEXT ===\n"
            "Use ONLY the following verified clinical content to respond. "
            "Do not fabricate or infer information beyond what is provided.\n"
        )

        for i, result in enumerate(results, 1):
            if total_tokens + result.token_estimate > settings.rag_max_context_tokens:
                sections.append(
                    f"\n[Context truncated: {len(results) - i + 1} additional "
                    f"chunks omitted to fit context window]"
                )
                break

            source = result.metadata.get("title", "Unknown")
            section = result.metadata.get("section", "general")
            score = round(result.score, 3)

            sections.append(
                f"\n--- Source {i}: {source} (section: {section}, "
                f"relevance: {score}) ---\n{result.text}"
            )
            total_tokens += result.token_estimate

        sections.append("\n=== END CLINICAL CONTEXT ===")

        context = "\n".join(sections)
        logger.info(
            "context_assembled",
            chunks_included=min(len(results), i if 'i' in dir() else len(results)),
            total_tokens=total_tokens,
        )
        return context

    async def query(self, user_query: str) -> str:
        results = await self.retrieve(user_query)
        return self.assemble_context(results)
