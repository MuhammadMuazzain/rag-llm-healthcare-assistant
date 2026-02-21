# Root Cause Analysis — Healthcare RAG + Voice Agent

## Bug 1: Vector Search Returning Irrelevant Clinical Content Chunks

### Symptoms
- Queries about diabetes medication returned general medication safety chunks
- Chunks from unrelated templates appeared in retrieval results
- Context quality degraded as more content was indexed

### Root Cause
**Naive fixed-size chunking** — The original chunker split content at fixed character boundaries (e.g., every 512 characters) without regard for document structure. This caused:
1. **Cross-contamination**: A chunk boundary falling mid-sentence merged content from one clinical section (e.g., diabetes dosage) with an adjacent section (e.g., hypertension lifestyle tips), producing an embedding that matched neither query accurately.
2. **Loss of section context**: Chunks lost their association with the template they came from, making it impossible to boost relevance based on category.
3. **No similarity threshold**: The retriever returned the top-K results regardless of score, so even a 0.3 similarity chunk would be included.

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Section-aware chunking | `app/rag/chunking.py` | Splits at clinical section markers (`[IF YES]`, `[CLOSING]`, etc.) instead of fixed offsets. Each chunk stays semantically coherent. |
| Metadata enrichment | `app/rag/chunking.py` | Every chunk carries `source_id`, `category`, `title`, `section`, and `tags` for downstream filtering. |
| Similarity threshold | `app/rag/vector_store.py` | Hard cutoff at 0.78 similarity (configurable via `RAG_SIMILARITY_THRESHOLD`). Low-relevance chunks are discarded before context assembly. |
| Metadata re-ranking | `app/rag/vector_store.py` | Chunks whose `category` or `tags` match the classified query intent receive a score boost (+0.05 for category, +0.02 per matching tag). |
| Token budget | `app/rag/vector_store.py` | Context assembly stops when the cumulative token count reaches `RAG_MAX_CONTEXT_TOKENS` (default 3000), preventing context window overflow. |

### Verification
- `tests/test_chunking.py::TestSplitClinicalContent::test_splits_at_section_markers` — Confirms chunks align with section boundaries.
- `tests/test_retriever.py::TestContextAssembly::test_structured_context_output` — Confirms source attribution and structured output.

---

## Bug 2: Embedding Mismatches Causing Incorrect Template Retrieval

### Symptoms
- After changing the embedding model from `text-embedding-ada-002` to `text-embedding-3-small`, retrieval quality dropped sharply
- Some queries returned completely unrelated content
- Cosine similarity scores were inconsistent (sometimes > 1.0)

### Root Cause
**Embedding model/dimension mismatch** — The system had two separate embedding instantiation points:
1. Index-time: Used the model specified in config when bulk-loading clinical data
2. Query-time: Used a hardcoded model string in the query path

When the config was updated, only the indexing path picked up the new model. Existing vectors in ChromaDB were computed with the old model, but queries used the new model's embeddings. Since the two models produce vectors in different semantic spaces, cosine similarity became meaningless.

Additionally, raw embeddings were not L2-normalized before storage, which caused `similarity_search_with_relevance_scores` to return scores outside [0, 1] in some edge cases.

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Centralized embedding | `app/rag/embeddings.py` | Single `EmbeddingManager` class used for both indexing and querying. Model name comes from one config source (`OPENAI_EMBEDDING_MODEL`). |
| Dimension validation | `app/rag/embeddings.py` | On first embedding, records the dimension count. All subsequent embeddings are validated against it. If a mismatch is detected, a clear error message tells the user to re-index. |
| L2 normalization | `app/rag/embeddings.py` | All vectors are normalized to unit length before storage and comparison, ensuring cosine similarity stays in [0, 1]. |
| Embedding cache | `app/rag/embeddings.py` | SHA-256 content hashing prevents redundant API calls and guarantees the same text always produces the same cached vector. |

### Verification
- Dimension validation raises `ValueError` with actionable message when model changes.
- Normalized embeddings ensure cosine similarity in [0, 1].

---

## Bug 3: Context Window Optimization

### Symptoms
- Long responses from the LLM that included irrelevant padding
- LLM sometimes "hallucinated" clinical details not in the source templates
- Token usage was significantly higher than expected

### Root Cause
**Unstructured context stuffing** — All retrieved chunks were concatenated into a single text block with no delineation. The LLM could not distinguish between sources, prioritize higher-relevance chunks, or identify where one source ended and another began. Combined with a permissive system prompt, the model filled knowledge gaps with hallucinated content.

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Structured context | `app/rag/retriever.py` | Context block uses clear delimiters (`=== RETRIEVED CLINICAL CONTEXT ===`), numbered sources with metadata (title, section, relevance score), and explicit instructions to use ONLY provided content. |
| Token budget | `app/rag/vector_store.py` | Cumulative token tracking stops adding chunks when the budget is reached, with a truncation notice. |
| Intent classification | `app/rag/retriever.py` | Keyword-based intent classifier routes queries to the correct content category, improving relevance before the LLM sees the context. |
| Constrained prompting | `app/templates/prompts.py` | System prompt explicitly forbids fabrication and mandates template structure adherence. |

### Verification
- `tests/test_retriever.py::TestContextAssembly` — Confirms structured output format with source attribution.
- `tests/test_retriever.py::TestIntentClassification` — Confirms correct category routing for each query type.

---

## Bug 4: Speech Interruption Detection Not Triggering Correctly

### Symptoms
- Users could speak during TTS playback without the system detecting an interruption
- Brief pauses mid-sentence (< 300ms) were sometimes classified as end-of-speech, causing premature responses
- Rapid toggling of speech state caused the system to enter an inconsistent state

### Root Cause
**No speech state machine or debounce logic** — The original implementation used a simple boolean `is_speaking` flag toggled by every VAD event. This caused:
1. Background noise could toggle the flag, causing rapid on/off cycles
2. The system only checked `is_speaking` at the moment of TTS completion, not during playback
3. No distinction between "user started talking during TTS" (interruption) and "user started talking after TTS" (normal turn)

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| State machine | `app/voice/vapi_client.py` | Explicit `SpeechState` enum: IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED → WARNING → DISCONNECTED. Transitions are governed by role + status combinations. |
| Debounce | `app/voice/vapi_client.py` | 150ms debounce window. Speech events within this window after TTS start are ignored (likely echo/feedback, not real user speech). |
| Interruption detection | `app/voice/vapi_client.py` | If user speech is detected while `tts_active` is True AND more than `speech_debounce_ms` has elapsed since TTS started, it's classified as an interruption. |
| Interruption history | `app/voice/vapi_client.py` | Every interruption is recorded with timestamp, user speech fragment, and retry count for analytics. |

### Verification
- `tests/test_voice_state.py::TestInterruptionDetection::test_interruption_during_tts_detected`
- `tests/test_voice_state.py::TestInterruptionDetection::test_no_interruption_when_tts_inactive`
- `tests/test_voice_state.py::TestSpeechStateTransitions`

---

## Bug 5: Silence Timeout Causing Premature Disconnections

### Symptoms
- Users disconnected while waiting for the system to generate a response
- First-time callers disconnected within 2-3 seconds of the call starting
- Users who paused to think were disconnected mid-conversation

### Root Cause
**Single global silence timer** — One timer started from the last voice activity, with no awareness of the conversation phase. Specifically:
1. Timer ran during TTS playback (user can't speak while listening)
2. Timer ran during LLM processing (system is generating, user is waiting)
3. Initial greeting had the same timeout as mid-conversation, but users need more time to orient on a new call
4. No warning before disconnection

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Per-phase timeouts | `app/voice/vapi_client.py` | `SilencePhase` enum: INITIAL (10s), MID_CONVERSATION (2.5s configurable), TTS_PLAYBACK (paused), SYSTEM_PROCESSING (paused). |
| Timer pause/resume | `app/voice/vapi_client.py` | Timer automatically pauses when entering SPEAKING or PROCESSING state, resumes when entering LISTENING. |
| Warning before disconnect | `app/voice/vapi_client.py` | When silence exceeds the threshold, sends a check-in prompt ("Are you still there?") before disconnecting. Only disconnects after an additional 5 seconds of silence post-warning. |
| Phase transitions | `app/voice/vapi_client.py` | First final transcript automatically transitions from INITIAL to MID_CONVERSATION phase. |

### Verification
- `tests/test_voice_state.py::TestSilencePhases`
- `tests/test_voice_state.py::TestSilencePhases::test_transcript_switches_to_mid_conversation`

---

## Bug 6: Response Retry Logic Failing After User Interruptions

### Symptoms
- After interrupting, the conversation entered a "dead air" state
- System cancelled TTS but never generated a new response
- Multiple interruptions caused complete conversation breakdown

### Root Cause
**Missing re-invocation pipeline** — When the user interrupted:
1. The system correctly cancelled TTS playback
2. But the STT transcript from the interrupting speech was not captured
3. The LLM was not re-invoked with the new user input
4. No retry counter existed, so a bug in re-invocation (if implemented) could loop infinitely

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Interruption handler | `app/voice/vapi_client.py` | `_handle_interruption_retry()` captures the user's interrupting speech, prepends context (`[The patient interrupted...]`), and re-invokes the response callback. |
| Retry cap | `app/voice/vapi_client.py` | `voice_max_retries` (default 3) limits retry attempts. On exhaustion, returns a graceful reset message asking the user what they need. |
| Retry delay | `app/voice/vapi_client.py` | Configurable delay (`voice_retry_delay_ms`, default 1500ms) between retries prevents rapid-fire API calls. |
| Retry counter reset | `app/voice/vapi_client.py` | Successful non-interruption transcripts reset the retry counter to 0. |

### Verification
- `tests/test_voice_state.py::TestRetryLogic::test_retry_after_interruption`
- `tests/test_voice_state.py::TestRetryLogic::test_max_retries_exceeded`
- `tests/test_voice_state.py::TestRetryLogic::test_normal_transcript_resets_retry`

---

## Bug 7: Output Filtering Not Enforcing Template Structure

### Symptoms
- LLM outputs omitted required sections (e.g., `[CLOSING]`)
- Responses contained unrequested medical advice (specific dosages, diagnoses)
- Template placeholders (`{patient_name}`) appeared in final output
- Safety disclaimers were inconsistently included

### Root Cause
**No validation layer between LLM and delivery** — Raw LLM output was passed directly to the voice agent or API consumer. The system prompt alone was insufficient to guarantee structural compliance because LLMs are probabilistic and occasionally deviate from instructions.

### Fix Applied
| Component | File | Change |
|-----------|------|--------|
| Multi-rule validator | `app/validation/template_validator.py` | Checks: disallowed content patterns (dosages, diagnoses, stop-medication advice, absolute promises), section structure presence, placeholder fill status, safety language inclusion, minimum length. |
| Conformance scoring | `app/validation/template_validator.py` | Weighted score (40% sections + 30% fields + 30% safety) with configurable threshold (default 0.6). |
| Auto-correction | `app/validation/response_filter.py` | On validation failure, a constrained correction prompt reformats the response. Up to 2 correction attempts before falling back to a safe default. |
| Safe fallback | `app/validation/response_filter.py` | If all corrections fail, returns a pre-approved message directing the patient to their care team. |
| Output sanitization | `app/validation/template_validator.py` | Strips system-level artifacts (`[INTERNAL]`, `System:`, etc.) from final output. |

### Verification
- `tests/test_template_validator.py` — Full test suite covering all validation rules.
- 22 test cases covering disallowed content, section structure, required fields, safety language, and overall validation.
