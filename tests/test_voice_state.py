"""Tests for voice agent state machine: interruptions, silence, and retry logic."""

import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.voice.vapi_client import (
    VapiVoiceClient,
    SpeechState,
    SilencePhase,
    ConversationState,
)


@pytest.fixture
def mock_settings():
    with patch("app.voice.vapi_client.get_settings") as mock:
        settings = MagicMock()
        settings.vapi_api_key = "test-key"
        settings.vapi_base_url = "https://api.vapi.ai"
        settings.vapi_assistant_id = "test-assistant"
        settings.vapi_phone_number_id = "test-phone"
        settings.voice_silence_timeout_ms = 2500
        settings.voice_interruption_threshold_ms = 300
        settings.voice_max_retries = 3
        settings.voice_retry_delay_ms = 100
        mock.return_value = settings
        yield settings


@pytest.fixture
def voice_client(mock_settings):
    client = VapiVoiceClient()
    return client


class TestSpeechStateTransitions:
    @pytest.mark.asyncio
    async def test_user_speech_start_sets_listening(self, voice_client):
        event = {
            "message": {
                "type": "speech-update",
                "status": "started",
                "role": "user",
            }
        }
        await voice_client.handle_webhook_event(event)
        assert voice_client._state.speech_state == SpeechState.LISTENING

    @pytest.mark.asyncio
    async def test_assistant_speech_start_sets_speaking(self, voice_client):
        event = {
            "message": {
                "type": "speech-update",
                "status": "started",
                "role": "assistant",
            }
        }
        await voice_client.handle_webhook_event(event)
        assert voice_client._state.speech_state == SpeechState.SPEAKING
        assert voice_client._state.tts_active is True

    @pytest.mark.asyncio
    async def test_assistant_speech_stop_resumes_listening(self, voice_client):
        voice_client._state.tts_active = True
        voice_client._state.speech_state = SpeechState.SPEAKING

        event = {
            "message": {
                "type": "speech-update",
                "status": "stopped",
                "role": "assistant",
            }
        }
        await voice_client.handle_webhook_event(event)
        assert voice_client._state.speech_state == SpeechState.LISTENING
        assert voice_client._state.tts_active is False


class TestInterruptionDetection:
    @pytest.mark.asyncio
    async def test_interruption_during_tts_detected(self, voice_client):
        voice_client._state.tts_active = True
        voice_client._state.last_tts_start = time.time() - 1.0
        voice_client._state.speech_state = SpeechState.SPEAKING

        event = {
            "message": {
                "type": "speech-update",
                "status": "started",
                "role": "user",
            }
        }
        result = await voice_client.handle_webhook_event(event)

        assert voice_client._state.speech_state == SpeechState.INTERRUPTED
        assert result == {"action": "stop_speaking"}
        assert len(voice_client._state.interruption_history) == 1

    @pytest.mark.asyncio
    async def test_no_interruption_when_tts_inactive(self, voice_client):
        voice_client._state.tts_active = False

        event = {
            "message": {
                "type": "speech-update",
                "status": "started",
                "role": "user",
            }
        }
        result = await voice_client.handle_webhook_event(event)
        assert voice_client._state.speech_state == SpeechState.LISTENING
        assert result is None


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_after_interruption(self, voice_client):
        callback = AsyncMock(return_value="Here's your answer.")
        voice_client.set_response_callback(callback)

        voice_client._state.speech_state = SpeechState.INTERRUPTED
        voice_client._state.retry_count = 0

        event = {
            "message": {
                "type": "transcript",
                "transcript": "What about my test results?",
                "role": "user",
                "transcriptType": "final",
            }
        }
        result = await voice_client.handle_webhook_event(event)

        assert result is not None
        assert "response" in result
        assert voice_client._state.retry_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, voice_client):
        voice_client._state.speech_state = SpeechState.INTERRUPTED
        voice_client._state.retry_count = 3

        event = {
            "message": {
                "type": "transcript",
                "transcript": "Hello?",
                "role": "user",
                "transcriptType": "final",
            }
        }
        result = await voice_client.handle_webhook_event(event)

        assert result is not None
        assert "specific information you need" in result["response"]
        assert voice_client._state.retry_count == 0

    @pytest.mark.asyncio
    async def test_normal_transcript_resets_retry(self, voice_client):
        callback = AsyncMock(return_value="Response here.")
        voice_client.set_response_callback(callback)
        voice_client._state.speech_state = SpeechState.LISTENING
        voice_client._state.retry_count = 2

        event = {
            "message": {
                "type": "transcript",
                "transcript": "How is my blood pressure?",
                "role": "user",
                "transcriptType": "final",
            }
        }
        result = await voice_client.handle_webhook_event(event)

        assert voice_client._state.retry_count == 0


class TestSilencePhases:
    def test_initial_phase_has_longer_timeout(self, voice_client):
        voice_client._state.silence_phase = SilencePhase.INITIAL
        timeout = voice_client._get_current_timeout()
        assert timeout == 10000

    def test_mid_conversation_uses_configured_timeout(self, voice_client):
        voice_client._state.silence_phase = SilencePhase.MID_CONVERSATION
        timeout = voice_client._get_current_timeout()
        assert timeout == 2500

    @pytest.mark.asyncio
    async def test_transcript_switches_to_mid_conversation(self, voice_client):
        callback = AsyncMock(return_value="Got it.")
        voice_client.set_response_callback(callback)
        voice_client._state.silence_phase = SilencePhase.INITIAL

        event = {
            "message": {
                "type": "transcript",
                "transcript": "Yes, I'm here.",
                "role": "user",
                "transcriptType": "final",
            }
        }
        await voice_client.handle_webhook_event(event)
        assert voice_client._state.silence_phase in (
            SilencePhase.MID_CONVERSATION,
            SilencePhase.SYSTEM_PROCESSING,
        )


class TestConversationStateReporting:
    def test_state_report_structure(self, voice_client):
        state = voice_client.get_conversation_state()
        assert "speech_state" in state
        assert "silence_phase" in state
        assert "turn_count" in state
        assert "retry_count" in state
        assert "interruption_count" in state
        assert "tts_active" in state

    @pytest.mark.asyncio
    async def test_call_end_updates_state(self, voice_client):
        event = {
            "message": {
                "type": "hang",
            }
        }
        await voice_client.handle_webhook_event(event)
        assert voice_client._state.speech_state == SpeechState.DISCONNECTED
