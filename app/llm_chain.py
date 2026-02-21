"""
LLM chain that ties RAG retrieval, prompt construction, and output validation together.
"""

import structlog
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.rag.retriever import ClinicalRAGRetriever
from app.validation.response_filter import ClinicalResponseFilter
from app.templates.prompts import (
    CLINICAL_SYSTEM_PROMPT,
    QUERY_RESPONSE_PROMPT,
    INTERRUPTION_RESPONSE_PROMPT,
    SILENCE_CHECKIN_PROMPT,
)
from config import get_settings

logger = structlog.get_logger(__name__)


class ClinicalLLMChain:
    def __init__(self, retriever: ClinicalRAGRetriever):
        settings = get_settings()
        self._retriever = retriever
        self._response_filter = ClinicalResponseFilter()
        self._llm = ChatOpenAI(
            model=settings.openai_chat_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.3,
            max_tokens=1000,
        )
        self._conversation_history: list[dict] = []

    async def generate_response(self, user_input: str) -> str:
        is_interruption = user_input.startswith("[The patient interrupted")
        is_silence = user_input.startswith("[SYSTEM: User has been silent")

        context = await self._retriever.query(user_input)

        system_message = CLINICAL_SYSTEM_PROMPT.format(clinical_context=context)

        if is_silence:
            user_prompt = SILENCE_CHECKIN_PROMPT
        elif is_interruption:
            user_prompt = INTERRUPTION_RESPONSE_PROMPT.format(
                interruption_context=user_input
            )
        else:
            user_prompt = QUERY_RESPONSE_PROMPT.format(user_query=user_input)

        messages = [
            SystemMessage(content=system_message),
        ]

        for entry in self._conversation_history[-6:]:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                from langchain.schema import AIMessage
                messages.append(AIMessage(content=entry["content"]))

        messages.append(HumanMessage(content=user_prompt))

        result = await self._llm.ainvoke(messages)
        raw_response = result.content

        if not is_silence:
            filtered_response, validation = await self._response_filter.filter_response(
                raw_response
            )
        else:
            filtered_response = raw_response

        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append(
            {"role": "assistant", "content": filtered_response}
        )

        logger.info(
            "response_generated",
            input_type="silence" if is_silence else "interruption" if is_interruption else "normal",
            response_length=len(filtered_response),
        )

        return filtered_response

    def clear_history(self) -> None:
        self._conversation_history.clear()
