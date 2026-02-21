from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = Field("text-embedding-3-small")
    openai_chat_model: str = Field("gpt-4o")

    vapi_api_key: str = Field(..., description="Vapi.ai API key")
    vapi_base_url: str = Field("https://api.vapi.ai")
    vapi_assistant_id: str = Field("")
    vapi_phone_number_id: str = Field("")

    chroma_persist_dir: str = Field("./chroma_db")
    chroma_collection_name: str = Field("clinical_content")

    rag_chunk_size: int = Field(512)
    rag_chunk_overlap: int = Field(64)
    rag_top_k: int = Field(5)
    rag_similarity_threshold: float = Field(0.78)
    rag_max_context_tokens: int = Field(3000)

    voice_silence_timeout_ms: int = Field(2500)
    voice_interruption_threshold_ms: int = Field(300)
    voice_max_retries: int = Field(3)
    voice_retry_delay_ms: int = Field(1500)

    app_host: str = Field("0.0.0.0")
    app_port: int = Field(8000)
    log_level: str = Field("INFO")
    environment: str = Field("development")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
