"""Tests for RAG retriever: intent classification and context assembly."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.rag.retriever import ClinicalRAGRetriever
from app.rag.vector_store import RetrievalResult


@pytest.fixture
def retriever():
    with patch("app.rag.retriever.EmbeddingManager"), \
         patch("app.rag.retriever.ClinicalVectorStore"):
        r = ClinicalRAGRetriever()
        return r


class TestIntentClassification:
    def test_medication_intent(self, retriever):
        category, tags = retriever.classify_intent(
            "Is the patient taking their medication as prescribed?"
        )
        assert category == "medication_management"

    def test_scheduling_intent(self, retriever):
        category, tags = retriever.classify_intent(
            "I need to reschedule my appointment with the doctor"
        )
        assert category == "scheduling"

    def test_chronic_care_intent(self, retriever):
        category, tags = retriever.classify_intent(
            "What are the blood pressure monitoring guidelines?"
        )
        assert category == "chronic_care"

    def test_care_transition_intent(self, retriever):
        category, tags = retriever.classify_intent(
            "Post-discharge wound care after surgery"
        )
        assert category == "care_transitions"

    def test_preventive_care_intent(self, retriever):
        category, tags = retriever.classify_intent(
            "Is the patient due for a mammogram screening?"
        )
        assert category == "preventive_care"

    def test_ambiguous_query_returns_best_match(self, retriever):
        category, tags = retriever.classify_intent("general question about health")
        # Should return None since no strong keyword match
        assert category is None

    def test_multi_keyword_query(self, retriever):
        category, tags = retriever.classify_intent(
            "Check medication adherence and dosage for the prescription"
        )
        assert category == "medication_management"


class TestContextAssembly:
    def test_empty_results(self, retriever):
        context = retriever.assemble_context([])
        assert "No relevant clinical content found" in context

    def test_structured_context_output(self, retriever):
        results = [
            RetrievalResult(
                text="Take medication as prescribed.",
                score=0.92,
                metadata={
                    "title": "Medication Guide",
                    "section": "main",
                },
                token_estimate=10,
            ),
            RetrievalResult(
                text="Monitor blood pressure daily.",
                score=0.85,
                metadata={
                    "title": "BP Monitoring",
                    "section": "vitals",
                },
                token_estimate=8,
            ),
        ]

        context = retriever.assemble_context(results)
        assert "RETRIEVED CLINICAL CONTEXT" in context
        assert "Source 1: Medication Guide" in context
        assert "Source 2: BP Monitoring" in context
        assert "END CLINICAL CONTEXT" in context
        assert "0.92" in context

    def test_source_attribution_included(self, retriever):
        results = [
            RetrievalResult(
                text="Some content here.",
                score=0.90,
                metadata={"title": "Test Title", "section": "intro"},
                token_estimate=5,
            ),
        ]
        context = retriever.assemble_context(results)
        assert "Test Title" in context
        assert "intro" in context
