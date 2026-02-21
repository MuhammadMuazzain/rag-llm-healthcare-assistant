"""Tests for clinical content chunking with section-aware splitting."""

import pytest
from app.rag.chunking import (
    split_clinical_content,
    estimate_tokens,
    ClinicalChunk,
)


SAMPLE_SCRIPT = """\
Hello {patient_name}, this is your healthcare assistant from {clinic_name}.
I'm calling to check on your medication.

Are you currently taking {medication_name} as prescribed?

[IF YES]: That's great to hear. Have you experienced any side effects?

[IF NO]: I understand. Can you tell me what's been making it difficult?

[CLOSING]: Thank you for sharing. I'll note this in your record so 
Dr. {provider_name} can follow up at your next visit. Is there anything 
else I can help with today?
"""


class TestEstimateTokens:
    def test_basic_estimate(self):
        text = "Hello world, this is a test sentence."
        tokens = estimate_tokens(text)
        assert 5 <= tokens <= 15

    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_long_text(self):
        text = "word " * 500
        tokens = estimate_tokens(text)
        assert tokens > 400


class TestSplitClinicalContent:
    def test_splits_at_section_markers(self):
        chunks = split_clinical_content(
            text=SAMPLE_SCRIPT,
            source_id="test_001",
            source_category="medication_management",
            source_title="Test Script",
            tags=["medication", "test"],
        )
        assert len(chunks) >= 3
        sections = [c.metadata["section"] for c in chunks]
        assert "introduction" in sections
        assert "closing" in sections

    def test_metadata_preservation(self):
        chunks = split_clinical_content(
            text=SAMPLE_SCRIPT,
            source_id="test_001",
            source_category="medication_management",
            source_title="Test Script",
            tags=["medication", "test"],
        )
        for chunk in chunks:
            assert chunk.metadata["source_id"] == "test_001"
            assert chunk.metadata["category"] == "medication_management"
            assert "medication" in chunk.metadata["tags"]

    def test_no_section_markers(self):
        simple_text = "This is a simple text without any section markers."
        chunks = split_clinical_content(
            text=simple_text,
            source_id="simple",
            source_category="general",
            source_title="Simple",
            tags=[],
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["section"] == "main"

    def test_chunks_have_content(self):
        chunks = split_clinical_content(
            text=SAMPLE_SCRIPT,
            source_id="test_001",
            source_category="test",
            source_title="Test",
            tags=[],
        )
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0
            assert chunk.token_estimate > 0

    def test_large_section_splitting(self):
        large_text = "This is a sentence. " * 200
        chunks = split_clinical_content(
            text=large_text,
            source_id="large",
            source_category="test",
            source_title="Large Test",
            tags=[],
            max_chunk_tokens=100,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_estimate <= 120  # allow some slack
