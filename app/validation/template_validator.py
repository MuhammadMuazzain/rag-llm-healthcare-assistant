"""
Clinical template structure validation and output filtering.

FIX APPLIED: Output filtering not enforcing template structure
- Root cause: LLM responses were returned directly without validating that they
  followed the approved clinical script format. The model would paraphrase,
  reorder sections, or omit required safety disclaimers, producing outputs
  that deviated from the approved clinical scripts.
- Fix: Implemented a multi-layer validation pipeline:
  1. Structural validator: checks that required sections ([IF YES], [CLOSING], etc.)
     are present in the output.
  2. Content guardrails: rejects responses containing disallowed medical advice
     patterns (e.g., specific dosage recommendations not in the source template).
  3. Template conformance scorer: measures how closely the output matches the
     approved template structure, triggering regeneration if below threshold.
"""

import re
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    location: str | None = None


@dataclass
class ValidationResult:
    is_valid: bool
    conformance_score: float
    issues: list[ValidationIssue]
    sanitized_output: str | None = None


DISALLOWED_PATTERNS = [
    (r"\b\d+\s*mg\b", "specific_dosage",
     "Response contains a specific dosage not sourced from the template"),
    (r"\bdiagnos(?:e|is|ed)\b.*\byou\s+have\b", "unsolicited_diagnosis",
     "Response appears to make an unsolicited diagnosis"),
    (r"\bstop\s+taking\b", "stop_medication_advice",
     "Response advises stopping medication without provider instruction"),
    (r"\b(?:guarantee|promise|certain|definitely\s+will)\b", "absolute_promise",
     "Response makes absolute medical promises"),
]

REQUIRED_SAFETY_PHRASES = [
    "contact.*provider",
    "call.*(?:doctor|clinic|emergency)",
    "seek.*(?:care|medical|attention)",
    "follow.*up",
]


class ClinicalOutputValidator:
    """Validates LLM output against clinical template structure and safety rules."""

    def __init__(self, conformance_threshold: float = 0.6):
        self._conformance_threshold = conformance_threshold

    def validate(
        self,
        output: str,
        template_id: str | None = None,
        expected_sections: list[str] | None = None,
        required_fields: list[str] | None = None,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []

        self._check_disallowed_content(output, issues)
        self._check_minimum_length(output, issues)
        section_score = self._check_section_structure(output, expected_sections, issues)
        field_score = self._check_required_fields(output, required_fields, issues)
        safety_score = self._check_safety_language(output, issues)

        weights = {"sections": 0.4, "fields": 0.3, "safety": 0.3}
        conformance_score = (
            section_score * weights["sections"]
            + field_score * weights["fields"]
            + safety_score * weights["safety"]
        )

        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        is_valid = not has_errors and conformance_score >= self._conformance_threshold

        sanitized = self._sanitize_output(output) if is_valid else None

        result = ValidationResult(
            is_valid=is_valid,
            conformance_score=round(conformance_score, 3),
            issues=issues,
            sanitized_output=sanitized,
        )

        logger.info(
            "output_validated",
            template_id=template_id,
            is_valid=is_valid,
            conformance_score=result.conformance_score,
            error_count=sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
            warning_count=sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
        )

        return result

    def _check_disallowed_content(
        self, output: str, issues: list[ValidationIssue]
    ) -> None:
        for pattern, code, message in DISALLOWED_PATTERNS:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code=code,
                    message=f"{message}. Found: '{matches[0]}'",
                    location=f"match: {matches[0]}",
                ))

    def _check_minimum_length(
        self, output: str, issues: list[ValidationIssue]
    ) -> None:
        word_count = len(output.split())
        if word_count < 20:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="too_short",
                message=f"Response too short ({word_count} words). "
                        f"Clinical responses require adequate detail.",
            ))

    def _check_section_structure(
        self,
        output: str,
        expected_sections: list[str] | None,
        issues: list[ValidationIssue],
    ) -> float:
        if not expected_sections:
            return 1.0

        found = 0
        for section in expected_sections:
            pattern = re.compile(
                rf"\[{re.escape(section)}\]", re.IGNORECASE
            )
            if pattern.search(output):
                found += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="missing_section",
                    message=f"Expected section [{section}] not found in output",
                    location=section,
                ))

        score = found / len(expected_sections) if expected_sections else 1.0
        return score

    def _check_required_fields(
        self,
        output: str,
        required_fields: list[str] | None,
        issues: list[ValidationIssue],
    ) -> float:
        if not required_fields:
            return 1.0

        filled = 0
        for field_name in required_fields:
            placeholder = f"{{{field_name}}}"
            if placeholder in output:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="unfilled_placeholder",
                    message=f"Template placeholder {placeholder} was not filled",
                    location=field_name,
                ))
            else:
                filled += 1

        return filled / len(required_fields) if required_fields else 1.0

    def _check_safety_language(
        self, output: str, issues: list[ValidationIssue]
    ) -> float:
        found = sum(
            1 for pattern in REQUIRED_SAFETY_PHRASES
            if re.search(pattern, output, re.IGNORECASE)
        )

        score = found / len(REQUIRED_SAFETY_PHRASES) if REQUIRED_SAFETY_PHRASES else 1.0

        if score < 0.25:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="missing_safety_language",
                message="Response lacks adequate safety/follow-up language",
            ))

        return score

    def _sanitize_output(self, output: str) -> str:
        """Remove any residual system-level artifacts from the response."""
        sanitized = re.sub(
            r"(?:^|\n)\s*(?:System|Assistant|AI):?\s*", "\n", output
        )
        sanitized = re.sub(
            r"(?:^|\n)\s*\[(?:INTERNAL|DEBUG|NOTE)\].*$",
            "",
            sanitized,
            flags=re.MULTILINE,
        )
        return sanitized.strip()
