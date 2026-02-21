"""
Response filtering pipeline that enforces clinical script conformance
before delivering output to the voice agent or API consumer.

FIX APPLIED: Responses deviating from approved clinical script formats
- Root cause: There was no enforcement layer between the LLM output and the
  delivery mechanism. The LLM would occasionally improvise responses, add
  conversational filler, or restructure the clinical script in ways that
  violated the approved format.
- Fix: Added a response filter that:
  1. Validates structure via ClinicalOutputValidator
  2. On validation failure, rewrites the response using a constrained prompt
     that strictly follows the template
  3. Limits retry attempts to prevent infinite loops
  4. Falls back to a safe default response if all retries fail
"""

import structlog
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.validation.template_validator import ClinicalOutputValidator, ValidationResult
from config import get_settings

logger = structlog.get_logger(__name__)

CORRECTION_SYSTEM_PROMPT = """\
You are a clinical response formatter. Your ONLY job is to reformat the provided
response to strictly follow the clinical script template structure.

Rules:
1. Preserve ALL clinical information from the original response
2. Use the exact section markers: [IF YES], [IF NO], [CLOSING], etc.
3. Do NOT add medical information not present in the original
4. Do NOT remove safety/follow-up instructions
5. Keep the tone professional and empathetic
6. Ensure all patient-specific placeholders are filled with provided values
7. Do NOT include specific dosages unless they appear in the source content

Return ONLY the reformatted response with no additional commentary.
"""

SAFE_FALLBACK_RESPONSE = (
    "Thank you for your patience. I want to make sure I provide you with "
    "accurate information. Let me connect you with a member of our care team "
    "who can assist you directly. Please hold for a moment, or you can call "
    "our office at your convenience. Your health and safety are our top priority."
)


class ClinicalResponseFilter:
    def __init__(self):
        settings = get_settings()
        self._validator = ClinicalOutputValidator()
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=1500,
        )
        self._max_correction_attempts = 2

    async def filter_response(
        self,
        response: str,
        template_id: str | None = None,
        expected_sections: list[str] | None = None,
        required_fields: list[str] | None = None,
    ) -> tuple[str, ValidationResult]:
        """
        Validate and optionally correct an LLM response.
        Returns (final_output, validation_result).
        """
        result = self._validator.validate(
            response, template_id, expected_sections, required_fields
        )

        if result.is_valid and result.sanitized_output:
            logger.info("response_passed_validation", template_id=template_id)
            return result.sanitized_output, result

        for attempt in range(1, self._max_correction_attempts + 1):
            logger.warning(
                "response_correction_attempt",
                attempt=attempt,
                original_score=result.conformance_score,
                issues=[i.code for i in result.issues],
            )

            corrected = await self._attempt_correction(
                response, result, expected_sections
            )

            result = self._validator.validate(
                corrected, template_id, expected_sections, required_fields
            )

            if result.is_valid and result.sanitized_output:
                logger.info(
                    "response_corrected_successfully",
                    attempt=attempt,
                    final_score=result.conformance_score,
                )
                return result.sanitized_output, result

            response = corrected

        logger.error(
            "response_correction_failed",
            template_id=template_id,
            final_score=result.conformance_score,
        )
        fallback_result = self._validator.validate(SAFE_FALLBACK_RESPONSE)
        return SAFE_FALLBACK_RESPONSE, fallback_result

    async def _attempt_correction(
        self,
        original_response: str,
        validation_result: ValidationResult,
        expected_sections: list[str] | None,
    ) -> str:
        issues_summary = "\n".join(
            f"- [{i.severity.value}] {i.code}: {i.message}"
            for i in validation_result.issues
        )

        section_hint = ""
        if expected_sections:
            section_hint = (
                f"\n\nRequired sections: {', '.join(f'[{s}]' for s in expected_sections)}"
            )

        correction_prompt = (
            f"The following clinical response failed validation:\n\n"
            f"--- ORIGINAL RESPONSE ---\n{original_response}\n"
            f"--- END ORIGINAL ---\n\n"
            f"Validation issues found:\n{issues_summary}\n"
            f"{section_hint}\n\n"
            f"Reformat this response to fix the issues while preserving "
            f"all clinical content. Return ONLY the corrected response."
        )

        messages = [
            SystemMessage(content=CORRECTION_SYSTEM_PROMPT),
            HumanMessage(content=correction_prompt),
        ]

        result = await self._llm.ainvoke(messages)
        return result.content
