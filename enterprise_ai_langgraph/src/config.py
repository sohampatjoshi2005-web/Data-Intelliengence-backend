from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    brevo_api_key: str = os.getenv("BREVO_API_KEY", "")
    sender_email: str = os.getenv("SENDER_EMAIL", "support@example.com")
    api_base_url: str = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    model_name: str = os.getenv("OLLAMA_MODEL_NAME", "mistral:latest")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
    db_path: str = os.getenv("APP_DB_PATH", str(APP_ROOT / "data" / "enterprise_ai_system.db"))
    chroma_path: str = os.getenv("CHROMA_PATH", str(APP_ROOT / "startup_kb_vector"))
    kb_collection_name: str = os.getenv("KB_COLLECTION_NAME", "kb_storage")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5")
    whisper_model_name: str = os.getenv("WHISPER_MODEL_NAME", "base")


settings = Settings()
