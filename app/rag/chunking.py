"""
Clinical content chunker with semantic boundary awareness.

FIX APPLIED: Irrelevant chunk retrieval bug
- Root cause: Naive fixed-size splitting broke clinical content mid-sentence and
  mixed content from different sections (e.g., dosage info merged with symptom
  lists), causing vector search to return semantically confused chunks.
- Fix: Implemented section-aware chunking that respects clinical document structure
  (headers, bullet points, conditional branches). Added metadata enrichment so
  each chunk carries its source template/section context for re-ranking.
"""

import re
import structlog
from dataclasses import dataclass, field
from config import get_settings

logger = structlog.get_logger(__name__)

SECTION_MARKERS = re.compile(
    r"^\s*\[(IF\s+.*?|CONFIRM|CLOSING|SYMPTOM CHECK|VITAL SIGNS|"
    r"MEDICATION REVIEW|LIFESTYLE|IMPORTANCE|SCHEDULING|IF ALREADY DONE|"
    r"IF DECLINED|IF YES|IF NO|IF WARNING SYMPTOMS|IF NO SYMPTOMS|"
    r"IF WITHIN RANGE|IF OUT OF RANGE|IF NEEDS RESCHEDULE)\]",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ClinicalChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    token_estimate: int = 0


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English clinical text."""
    return max(1, len(text) // 4)


def split_clinical_content(
    text: str,
    source_id: str,
    source_category: str,
    source_title: str,
    tags: list[str],
    max_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> list[ClinicalChunk]:
    """
    Split clinical content into semantically coherent chunks that respect
    section boundaries like [IF YES], [CLOSING], etc.
    """
    settings = get_settings()
    max_tokens = max_chunk_tokens or settings.rag_chunk_size
    overlap_tokens = chunk_overlap_tokens or settings.rag_chunk_overlap

    sections = _split_into_sections(text)
    chunks: list[ClinicalChunk] = []

    for section_label, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue

        token_count = estimate_tokens(section_text)

        if token_count <= max_tokens:
            chunks.append(ClinicalChunk(
                text=section_text,
                metadata={
                    "source_id": source_id,
                    "category": source_category,
                    "title": source_title,
                    "section": section_label,
                    "tags": tags,
                },
                token_estimate=token_count,
            ))
        else:
            sub_chunks = _split_large_section(
                section_text, max_tokens, overlap_tokens
            )
            for i, sub_text in enumerate(sub_chunks):
                chunks.append(ClinicalChunk(
                    text=sub_text,
                    metadata={
                        "source_id": source_id,
                        "category": source_category,
                        "title": source_title,
                        "section": f"{section_label}_part{i+1}",
                        "tags": tags,
                    },
                    token_estimate=estimate_tokens(sub_text),
                ))

    logger.info(
        "content_chunked",
        source_id=source_id,
        total_chunks=len(chunks),
        sections_found=len(sections),
    )
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text at clinical section markers while keeping context."""
    positions = [(m.start(), m.group(0).strip()) for m in SECTION_MARKERS.finditer(text)]

    if not positions:
        return [("main", text)]

    sections = []
    if positions[0][0] > 0:
        sections.append(("introduction", text[: positions[0][0]]))

    for i, (pos, label) in enumerate(positions):
        clean_label = label.strip("[] \t").lower().replace(" ", "_")
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[pos:end]
        sections.append((clean_label, section_text))

    return sections


def _split_large_section(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Split an oversized section at sentence boundaries with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            overlap_text = " ".join(current)
            overlap_sents: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = estimate_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += t
            current = overlap_sents
            current_tokens = overlap_count

        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks
