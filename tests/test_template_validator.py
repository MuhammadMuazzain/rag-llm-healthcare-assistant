"""Tests for clinical output validation and template structure enforcement."""

import pytest
from app.validation.template_validator import (
    ClinicalOutputValidator,
    ValidationSeverity,
)


@pytest.fixture
def validator():
    return ClinicalOutputValidator(conformance_threshold=0.6)


VALID_CLINICAL_RESPONSE = """\
Hello John, this is your healthcare assistant from Springfield Clinic.
I'm calling to check on your medication routine.

Are you currently taking Metformin as prescribed by Dr. Smith?

[IF YES]: That's great to hear. Have you experienced any side effects 
such as nausea or stomach upset?

[IF NO]: I understand. Can you tell me what's been making it difficult 
to take your medication?

[CLOSING]: Thank you for sharing that with me. I'll note this in your 
record so Dr. Smith can follow up with you at your next visit. 
Please don't hesitate to contact your provider if you have any concerns.
"""


class TestDisallowedContent:
    def test_rejects_specific_dosage(self, validator):
        bad_output = "You should take 500 mg of aspirin twice daily. Call your doctor."
        result = validator.validate(bad_output)
        codes = [i.code for i in result.issues]
        assert "specific_dosage" in codes

    def test_rejects_diagnosis(self, validator):
        bad_output = (
            "Based on your symptoms, I can diagnose that you have diabetes. "
            "Please call your doctor for a follow up visit soon."
        )
        result = validator.validate(bad_output)
        codes = [i.code for i in result.issues]
        assert "unsolicited_diagnosis" in codes

    def test_rejects_stop_medication(self, validator):
        bad_output = (
            "You should stop taking your blood pressure medication immediately. "
            "Contact your provider for more information."
        )
        result = validator.validate(bad_output)
        codes = [i.code for i in result.issues]
        assert "stop_medication_advice" in codes

    def test_rejects_absolute_promises(self, validator):
        bad_output = (
            "This treatment will definitely will cure your condition. "
            "Please follow up with your doctor for next steps."
        )
        result = validator.validate(bad_output)
        codes = [i.code for i in result.issues]
        assert "absolute_promise" in codes


class TestSectionStructure:
    def test_valid_sections_score_high(self, validator):
        result = validator.validate(
            VALID_CLINICAL_RESPONSE,
            expected_sections=["IF YES", "IF NO", "CLOSING"],
        )
        assert result.conformance_score >= 0.6

    def test_missing_sections_flagged(self, validator):
        partial = (
            "Hello John. How are you doing today? "
            "Please call your doctor if you have concerns."
        )
        result = validator.validate(
            partial,
            expected_sections=["IF YES", "IF NO", "CLOSING"],
        )
        warnings = [i for i in result.issues if i.code == "missing_section"]
        assert len(warnings) >= 2


class TestRequiredFields:
    def test_unfilled_placeholders_rejected(self, validator):
        output_with_placeholder = (
            "Hello {patient_name}, this is your healthcare assistant. "
            "Please contact your provider for any concerns about your care."
        )
        result = validator.validate(
            output_with_placeholder,
            required_fields=["patient_name"],
        )
        codes = [i.code for i in result.issues]
        assert "unfilled_placeholder" in codes

    def test_filled_fields_pass(self, validator):
        result = validator.validate(
            VALID_CLINICAL_RESPONSE,
            required_fields=["patient_name"],
        )
        unfilled = [i for i in result.issues if i.code == "unfilled_placeholder"]
        assert len(unfilled) == 0


class TestSafetyLanguage:
    def test_safety_language_present(self, validator):
        result = validator.validate(VALID_CLINICAL_RESPONSE)
        assert result.conformance_score > 0.0

    def test_missing_safety_language_warned(self, validator):
        unsafe = "Hello John, just checking in. How are you today? Have a good day."
        result = validator.validate(unsafe)
        codes = [i.code for i in result.issues]
        assert "missing_safety_language" in codes


class TestMinimumLength:
    def test_too_short_rejected(self, validator):
        result = validator.validate("Hello there.")
        codes = [i.code for i in result.issues]
        assert "too_short" in codes


class TestOverallValidation:
    def test_valid_response_passes(self, validator):
        result = validator.validate(
            VALID_CLINICAL_RESPONSE,
            template_id="medication_adherence_check",
            expected_sections=["IF YES", "IF NO", "CLOSING"],
        )
        assert result.is_valid is True
        assert result.sanitized_output is not None

    def test_invalid_response_fails(self, validator):
        result = validator.validate(
            "Take 100 mg of this medicine and you'll definitely will be fine.",
            template_id="test",
        )
        assert result.is_valid is False
