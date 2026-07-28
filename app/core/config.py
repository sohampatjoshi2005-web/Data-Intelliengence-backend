from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Superhumanly Thoughts"
    ollama_base_url: str = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
    ollama_vision_base_url: str = os.getenv("OLLAMA_VISION_BASE_URL", "http://localhost:11434")
    ollama_api_key: str = os.getenv("OLLAMA_OPENAI_API_KEY", "ollama")
    ollama_model: str = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    structured_llm_provider: str = os.getenv("STRUCTURED_LLM_PROVIDER", "bedrock")
    bedrock_region: str = os.getenv("BEDROCK_REGION", "us-east-2")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "")
    bedrock_embed_model_id: str = os.getenv("BEDROCK_EMBED_MODEL_ID", "")
    bedrock_max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))
    bedrock_temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))
    kb_pg_dsn: str = os.getenv("KB_PG_DSN", "")
    kb_age_dsn: str = os.getenv("KB_AGE_DSN", "")
    kb_age_graph_name: str = os.getenv("KB_AGE_GRAPH_NAME", "kb_graph")
    llm_studio_url: str = os.getenv("LLM_STUDIO_URL", "")
    kb_fast_mode: bool = os.getenv("KB_FAST_MODE", "false").lower() == "true"
    kb_chunk_cap: int = int(os.getenv("KB_CHUNK_CAP", "200"))
    kb_chunk_size_tokens: int = int(os.getenv("KB_CHUNK_SIZE_TOKENS", "800"))
    kb_chunk_overlap_tokens: int = int(os.getenv("KB_CHUNK_OVERLAP_TOKENS", "150"))
    kb_skip_ner: bool = os.getenv("KB_SKIP_NER", "false").lower() == "true"
    kb_skip_pii: bool = os.getenv("KB_SKIP_PII", "false").lower() == "true"
    kb_skip_enrichment: bool = os.getenv("KB_SKIP_ENRICHMENT", "false").lower() == "true"
    kb_pii_max_chars: int = int(os.getenv("KB_PII_MAX_CHARS", "20000"))
    kb_enrich_batch_size: int = int(os.getenv("KB_ENRICH_BATCH_SIZE", "8"))
    kb_enrich_workers: int = int(os.getenv("KB_ENRICH_WORKERS", "2"))
    kb_embed_batch_size: int = int(os.getenv("KB_EMBED_BATCH_SIZE", "32"))
    kb_embed_workers: int = int(os.getenv("KB_EMBED_WORKERS", "2"))
    api_auth_enabled: bool = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
    distributed_enabled: bool = os.getenv("DISTRIBUTED_ENABLED", "false").lower() == "true"
    distributed_backend: str = os.getenv("DISTRIBUTED_BACKEND", "auto")
    dspy_enabled: bool = os.getenv("DSPY_ENABLED", "false").lower() == "true"
    dspy_model: str = os.getenv("DSPY_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest"))
    rag_v2_enabled: bool = os.getenv("RAG_V2_ENABLED", "true").lower() == "true"
    rag_query_expansions: int = int(os.getenv("RAG_QUERY_EXPANSIONS", "3"))
    rag_reranker: str = os.getenv("RAG_RERANKER", "flashrank")  # flashrank | bge
    rag_rerank_top_k: int = int(os.getenv("RAG_RERANK_TOP_K", "10"))
    rag_retrieval_pool: int = int(os.getenv("RAG_RETRIEVAL_POOL", "40"))
    rag_graph_hops: int = int(os.getenv("RAG_GRAPH_HOPS", "1"))
    rag_context_compress: bool = os.getenv("RAG_CONTEXT_COMPRESS", "true").lower() == "true"
    rag_semantic_chunking: bool = os.getenv("RAG_SEMANTIC_CHUNKING", "true").lower() == "true"
    rag_hierarchical_chunking: bool = os.getenv("RAG_HIERARCHICAL_CHUNKING", "true").lower() == "true"
    rag_semantic_model: str = os.getenv("RAG_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "")
    rag_reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
    rag_raptor_enabled: bool = os.getenv("RAG_RAPTOR_ENABLED", "true").lower() == "true"
    rag_raptor_max_sections: int = int(os.getenv("RAG_RAPTOR_MAX_SECTIONS", "12"))
    rag_raptor_workers: int = int(os.getenv("RAG_RAPTOR_WORKERS", "2"))
    rag_cache_size: int = int(os.getenv("RAG_CACHE_SIZE", "256"))
    rag_cache_ttl: int = int(os.getenv("RAG_CACHE_TTL", "900"))
    rag_redis_url: str = os.getenv("RAG_REDIS_URL", "")
    
    # Celery Configuration (Phase 2)
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_backend_url: str = os.getenv("CELERY_BACKEND_URL", "redis://localhost:6379/0")
    celery_enabled: bool = os.getenv("CELERY_ENABLED", "true").lower() == "true"
    
    # WebSocket Configuration (Phase 2)
    websocket_enabled: bool = os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true"
    
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "prod-secret-change-me")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # CORS Settings
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    
    # Email Settings
    email_enabled: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.resend.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "resend")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_from: str = os.getenv("EMAIL_FROM", "onboarding@superhumanlythoughts.com")
    email_display_name: str = os.getenv("EMAIL_DISPLAY_NAME", "Superhumanly Governance")


settings = Settings()
