"""Shared test configuration."""

import os
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing")
os.environ.setdefault("VAPI_API_KEY", "test-vapi-key")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("OPENAI_CHAT_MODEL", "gpt-4o")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./test_chroma_db")
os.environ.setdefault("CHROMA_COLLECTION_NAME", "test_clinical")
