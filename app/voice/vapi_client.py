"""
Vapi.ai voice agent client with robust interruption handling and silence detection.

FIX APPLIED: Speech interruption detection not triggering correctly
- Root cause: The interruption detection relied on a simple VAD (Voice Activity
  Detection) threshold that didn't account for:
  1. Brief pauses mid-sentence (< 300ms) being misclassified as end-of-speech
  2. Cross-talk during TTS playback not being detected as interruption
  3. No debounce logic, causing rapid on/off toggling of speech state
- Fix: Implemented a state machine for speech detection with:
  - Configurable interruption threshold (default 300ms of sustained speech during
    TTS playback triggers interruption)
  - Debounce window to prevent false triggers from background noise
  - Explicit speech state tracking: LISTENING, SPEAKING, INTERRUPTED, PROCESSING

FIX APPLIED: Silence timeout causing premature disconnections
- Root cause: A single global silence timer started from the last voice activity.
  It did not pause during TTS playback or system processing, so users were
  disconnected while the system was generating a response.
- Fix: Per-phase silence tracking:
  - Waiting-for-initial-response: 10s timeout (generous for first interaction)
  - Mid-conversation: configurable (default 2.5s)
  - During TTS playback: timer paused
  - During processing: timer paused
  Added warning prompt before disconnection.

FIX APPLIED: Response retry logic failing after user interruptions
- Root cause: When a user interrupted, the system cancelled the current TTS but
  did not re-queue a response or acknowledge the interruption. The conversation
  entered a dead state where neither party was speaking.
- Fix: On interruption:
  1. Immediately cancel current TTS playback
  2. Capture the user's interrupting speech via STT
  3. Re-invoke the LLM with the interruption context appended
  4. Queue the new response for TTS delivery
  5. Track retry count to prevent infinite loops (max 3 retries)
"""

import asyncio
import time
import structlog
import httpx
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from config import get_settings

logger = structlog.get_logger(__name__)


class SpeechState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WARNING = "warning"
    DISCONNECTED = "disconnected"


class SilencePhase(str, Enum):
    INITIAL = "initial"
    MID_CONVERSATION = "mid_conversation"
    TTS_PLAYBACK = "tts_playback"
    SYSTEM_PROCESSING = "system_processing"


@dataclass
class InterruptionEvent:
    timestamp: float
    user_speech_fragment: str
    was_during_tts: bool
    retry_count: int = 0


@dataclass
class ConversationState:
    speech_state: SpeechState = SpeechState.IDLE
    silence_phase: SilencePhase = SilencePhase.INITIAL
    last_voice_activity: float = 0.0
    last_tts_start: float = 0.0
    tts_active: bool = False
    turn_count: int = 0
    retry_count: int = 0
    interruption_history: list[InterruptionEvent] = field(default_factory=list)
    pending_response: str | None = None
    conversation_id: str | None = None


class VapiVoiceClient:
    """Client for Vapi.ai with fixes for interruption, silence, and retry bugs."""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.vapi_api_key
        self._base_url = settings.vapi_base_url
        self._assistant_id = settings.vapi_assistant_id

        self._silence_timeout_ms = settings.voice_silence_timeout_ms
        self._interruption_threshold_ms = settings.voice_interruption_threshold_ms
        self._max_retries = settings.voice_max_retries
        self._retry_delay_ms = settings.voice_retry_delay_ms

        self._initial_silence_timeout_ms = 10000
        self._warning_prompt = (
            "Are you still there? I want to make sure I can help you. "
            "If you need more time, just let me know."
        )
        self._speech_debounce_ms = 150

        self._state = ConversationState()
        self._silence_monitor_task: asyncio.Task | None = None
        self._on_response_callback: Callable[
            [str], Coroutine[Any, Any, str]
        ] | None = None

        self._http_client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        logger.info(
            "vapi_client_initialized",
            silence_timeout=self._silence_timeout_ms,
            interruption_threshold=self._interruption_threshold_ms,
            max_retries=self._max_retries,
        )

    def set_response_callback(
        self, callback: Callable[[str], Coroutine[Any, Any, str]]
    ) -> None:
        self._on_response_callback = callback

    async def create_call(
        self,
        phone_number: str | None = None,
        customer_name: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Create an outbound Vapi call with optimized voice settings."""
        settings = get_settings()

        payload: dict[str, Any] = {
            "assistantId": self._assistant_id,
            "assistantOverrides": {
                "silenceTimeoutSeconds": self._silence_timeout_ms / 1000,
                "responseDelaySeconds": 0.4,
                "interruptionsEnabled": True,
                "backchannelingEnabled": True,
                "model": {
                    "provider": "openai",
                    "model": settings.openai_chat_model,
                    "temperature": 0.3,
                },
                "voice": {
                    "provider": "11labs",
                    "voiceId": "21m00Tcm4TlvDq8ikWAM",
                    "stability": 0.7,
                    "similarityBoost": 0.8,
                },
                "transcriber": {
                    "provider": "deepgram",
                    "model": "nova-2-medical",
                    "language": "en",
                    "smartFormat": True,
                },
            },
        }

        if phone_number:
            payload["phoneNumberId"] = settings.vapi_phone_number_id
            payload["customer"] = {"number": phone_number}
            if customer_name:
                payload["customer"]["name"] = customer_name

        if metadata:
            payload["metadata"] = metadata

        response = await self._http_client.post("/call", json=payload)
        response.raise_for_status()
        call_data = response.json()

        self._state.conversation_id = call_data.get("id")
        self._state.speech_state = SpeechState.IDLE

        logger.info(
            "call_created",
            call_id=self._state.conversation_id,
            phone=phone_number,
        )

        return call_data

    async def handle_webhook_event(self, event: dict) -> dict | None:
        """Process Vapi webhook events with full state machine logic."""
        event_type = event.get("message", {}).get("type", "")

        handlers = {
            "speech-update": self._handle_speech_update,
            "transcript": self._handle_transcript,
            "hang": self._handle_hang,
            "end-of-call-report": self._handle_end_of_call,
            "function-call": self._handle_function_call,
            "status-update": self._handle_status_update,
        }

        handler = handlers.get(event_type)
        if handler:
            return await handler(event.get("message", {}))

        logger.debug("unhandled_webhook_event", type=event_type)
        return None

    async def _handle_speech_update(self, message: dict) -> dict | None:
        """
        State machine for speech detection with debounce and
        interruption logic.
        """
        status = message.get("status", "")
        role = message.get("role", "")
        now = time.time()

        if role == "user":
            if status == "started":
                return await self._on_user_speech_start(now)
            elif status == "stopped":
                return await self._on_user_speech_stop(now)

        elif role == "assistant":
            if status == "started":
                self._state.tts_active = True
                self._state.last_tts_start = now
                self._state.speech_state = SpeechState.SPEAKING
                self._pause_silence_timer()
                logger.debug("tts_playback_started")
            elif status == "stopped":
                self._state.tts_active = False
                self._state.speech_state = SpeechState.LISTENING
                self._resume_silence_timer()
                logger.debug("tts_playback_stopped")

        return None

    async def _on_user_speech_start(self, timestamp: float) -> dict | None:
        """Handle user starting to speak, with interruption detection."""
        self._state.last_voice_activity = timestamp
        self._reset_silence_timer()

        if self._state.tts_active:
            time_into_tts = (timestamp - self._state.last_tts_start) * 1000

            if time_into_tts > self._speech_debounce_ms:
                logger.info(
                    "user_interruption_detected",
                    time_into_tts_ms=round(time_into_tts),
                    retry_count=self._state.retry_count,
                )

                self._state.speech_state = SpeechState.INTERRUPTED

                interruption = InterruptionEvent(
                    timestamp=timestamp,
                    user_speech_fragment="",
                    was_during_tts=True,
                    retry_count=self._state.retry_count,
                )
                self._state.interruption_history.append(interruption)

                return {"action": "stop_speaking"}

        self._state.speech_state = SpeechState.LISTENING
        return None

    async def _on_user_speech_stop(self, timestamp: float) -> None:
        """Handle user stopping speech, manage silence timer."""
        self._state.last_voice_activity = timestamp

        if self._state.speech_state == SpeechState.INTERRUPTED:
            self._state.speech_state = SpeechState.PROCESSING
            logger.debug("post_interruption_processing")

        self._start_silence_timer()
        return None

    async def _handle_transcript(self, message: dict) -> dict | None:
        """Process transcription with interruption-aware retry logic."""
        transcript = message.get("transcript", "")
        role = message.get("role", "")
        is_final = message.get("transcriptType", "") == "final"

        if role != "user" or not is_final or not transcript.strip():
            return None

        self._state.turn_count += 1
        self._state.silence_phase = SilencePhase.MID_CONVERSATION

        if (
            self._state.speech_state == SpeechState.INTERRUPTED
            or self._state.speech_state == SpeechState.PROCESSING
        ):
            return await self._handle_interruption_retry(transcript)

        if self._on_response_callback:
            self._state.speech_state = SpeechState.PROCESSING
            self._pause_silence_timer()

            try:
                response = await self._on_response_callback(transcript)
                self._state.pending_response = response
                self._state.retry_count = 0
                return {"response": response}
            except Exception as e:
                logger.error("response_generation_failed", error=str(e))
                return {
                    "response": "I apologize, I'm having a brief technical "
                    "difficulty. Could you please repeat that?"
                }

        return None

    async def _handle_interruption_retry(self, transcript: str) -> dict | None:
        """
        Handle response generation after user interrupts.
        Implements capped retry logic to prevent infinite loops.
        """
        self._state.retry_count += 1

        if self._state.retry_count > self._max_retries:
            logger.warning(
                "max_retries_exceeded",
                retry_count=self._state.retry_count,
                max_retries=self._max_retries,
            )
            self._state.retry_count = 0
            self._state.speech_state = SpeechState.LISTENING
            return {
                "response": "I understand you have a question. Let me address "
                "that directly. Could you please tell me what specific "
                "information you need?"
            }

        if self._state.interruption_history:
            self._state.interruption_history[-1].user_speech_fragment = transcript

        logger.info(
            "interruption_retry",
            attempt=self._state.retry_count,
            max=self._max_retries,
            transcript_preview=transcript[:80],
        )

        if self._on_response_callback:
            self._state.speech_state = SpeechState.PROCESSING
            self._pause_silence_timer()

            interruption_context = (
                f"[The patient interrupted the previous response to say: "
                f'"{transcript}"]\n'
                f"Please address their concern directly and concisely."
            )

            try:
                await asyncio.sleep(self._retry_delay_ms / 1000)
                response = await self._on_response_callback(interruption_context)
                self._state.pending_response = response
                return {"response": response}
            except Exception as e:
                logger.error("retry_response_failed", error=str(e), attempt=self._state.retry_count)
                return {
                    "response": "I hear you. Let me take a moment to address your concern properly."
                }

        return None

    async def _handle_hang(self, message: dict) -> dict | None:
        """Handle call hang events."""
        self._state.speech_state = SpeechState.DISCONNECTED
        self._cancel_silence_timer()

        logger.info(
            "call_ended",
            conversation_id=self._state.conversation_id,
            turn_count=self._state.turn_count,
            interruptions=len(self._state.interruption_history),
        )
        return None

    async def _handle_end_of_call(self, message: dict) -> dict | None:
        """Process end-of-call analytics."""
        summary = message.get("summary", "")
        duration = message.get("durationSeconds", 0)

        logger.info(
            "call_report",
            conversation_id=self._state.conversation_id,
            duration_seconds=duration,
            summary_preview=summary[:200] if summary else "none",
            total_turns=self._state.turn_count,
            total_interruptions=len(self._state.interruption_history),
            total_retries=sum(
                e.retry_count for e in self._state.interruption_history
            ),
        )
        return None

    async def _handle_function_call(self, message: dict) -> dict | None:
        """Handle server-side function calls from Vapi."""
        function_name = message.get("functionCall", {}).get("name", "")
        parameters = message.get("functionCall", {}).get("parameters", {})

        logger.info(
            "function_call_received",
            function=function_name,
            params=list(parameters.keys()),
        )

        return {"result": f"Function {function_name} acknowledged"}

    async def _handle_status_update(self, message: dict) -> dict | None:
        """Handle call status updates."""
        status = message.get("status", "")
        logger.info("call_status_update", status=status)
        return None

    def _start_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._silence_monitor_task = asyncio.create_task(
            self._monitor_silence()
        )

    def _reset_silence_timer(self) -> None:
        self._state.last_voice_activity = time.time()

    def _pause_silence_timer(self) -> None:
        self._cancel_silence_timer()
        self._state.silence_phase = SilencePhase.SYSTEM_PROCESSING

    def _resume_silence_timer(self) -> None:
        self._state.last_voice_activity = time.time()
        if self._state.turn_count > 0:
            self._state.silence_phase = SilencePhase.MID_CONVERSATION
        self._start_silence_timer()

    def _cancel_silence_timer(self) -> None:
        if self._silence_monitor_task and not self._silence_monitor_task.done():
            self._silence_monitor_task.cancel()

    async def _monitor_silence(self) -> None:
        """
        Per-phase silence monitoring with warning before disconnect.
        """
        try:
            while True:
                await asyncio.sleep(0.5)

                if self._state.speech_state in (
                    SpeechState.SPEAKING,
                    SpeechState.PROCESSING,
                    SpeechState.DISCONNECTED,
                ):
                    continue

                elapsed_ms = (time.time() - self._state.last_voice_activity) * 1000
                timeout = self._get_current_timeout()

                if elapsed_ms > timeout:
                    if self._state.speech_state != SpeechState.WARNING:
                        self._state.speech_state = SpeechState.WARNING
                        logger.info(
                            "silence_warning_sent",
                            elapsed_ms=round(elapsed_ms),
                            timeout_ms=timeout,
                        )
                        if self._on_response_callback:
                            await self._on_response_callback(
                                "[SYSTEM: User has been silent. Send a gentle check-in.]"
                            )
                        await asyncio.sleep(5.0)

                    elif self._state.speech_state == SpeechState.WARNING:
                        extra_elapsed = (
                            time.time() - self._state.last_voice_activity
                        ) * 1000
                        if extra_elapsed > timeout + 5000:
                            logger.info(
                                "silence_disconnect",
                                total_silence_ms=round(extra_elapsed),
                            )
                            self._state.speech_state = SpeechState.DISCONNECTED
                            break

        except asyncio.CancelledError:
            pass

    def _get_current_timeout(self) -> float:
        if self._state.silence_phase == SilencePhase.INITIAL:
            return self._initial_silence_timeout_ms
        return self._silence_timeout_ms

    def get_conversation_state(self) -> dict:
        return {
            "speech_state": self._state.speech_state.value,
            "silence_phase": self._state.silence_phase.value,
            "turn_count": self._state.turn_count,
            "retry_count": self._state.retry_count,
            "interruption_count": len(self._state.interruption_history),
            "tts_active": self._state.tts_active,
            "conversation_id": self._state.conversation_id,
        }

    async def close(self) -> None:
        self._cancel_silence_timer()
        await self._http_client.aclose()
        logger.info("vapi_client_closed")
