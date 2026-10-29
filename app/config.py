from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""
    llm_model_answer: str = "gpt-4o"
    llm_model_grader: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_disk_cache: str = ""  # JSONL path for a persistent embedding cache ("" = off)

    # RAGAS judge (eval only). Any OpenAI-compatible endpoint: "" base URL / key = OpenAI with openai_api_key.
    # Keep one judge for every reported RAGAS number; a different judge gives different scores.
    ragas_judge_model: str = "gpt-4o-mini"
    ragas_judge_base_url: str = ""
    ragas_judge_api_key: str = ""
    ragas_judge_max_workers: int = 8  # free endpoints often allow only 1-2 concurrent requests
    # Extra request body as JSON, e.g. OpenRouter provider pinning so every run is graded by the same host:
    # {"provider": {"order": ["deepinfra"], "allow_fallbacks": false}}
    ragas_judge_extra_body: str = ""
    # Judge through a subscription CLI instead of an API ("codex" or "claude", eval/cli_llm.py); the model is then
    # ragas_judge_model ("" = the CLI's default).
    ragas_judge_cli: str = ""
    ragas_metrics: str = "faithfulness,context_precision,context_recall,answer_relevancy"

    # LLM provider selection — "groq" uses Groq's OpenAI-compatible endpoint
    # (same `openai` SDK, different base_url) for generate()/generate_with_json().
    # Embeddings always use OpenAI regardless of this setting (Groq doesn't serve them).
    llm_provider: Literal["openai", "groq", "gemini"] = "openai"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_answer: str = "openai/gpt-oss-120b"
    groq_model_grader: str = "openai/gpt-oss-120b"
    # Gemini via Google's OpenAI-compatible endpoint. GEMINI_API_KEY may hold several comma-separated keys:
    # llm_service rotates to the next key when one is rate-limited (free-tier keys have small per-key quotas).
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.1-flash-lite"  # used for both answer and grader calls

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "gst_provisions_enriched_v3"

    database_url: str = "postgresql://postgres:postgres@localhost:5432/adv_rag"

    upstash_redis_url: str = ""
    upstash_redis_token: str = ""
    cache_ttl_embeddings: int = 604_800
    cache_ttl_rag: int = 3_600
    rag_answer_cache_enabled: bool = True  # off while experimenting: every answer is computed fresh


    tavily_api_key: str = ""


    jwt_secret: str = ""
    jwt_expiration_minutes: int = 60

    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    max_tokens_per_user_daily: int = 100_000
    auth_login_rate_limit_per_min: int = 5
    auth_register_rate_limit_per_hour: int = 3

    max_input_tokens: int = 3_000
    reserved_context_tokens: int = 1_000
    reserved_output_tokens: int = 1_000

    prompt_injection_threshold: float = 0.75
    toxicity_threshold: float = 0.75
    output_toxicity_threshold: float = 0.5
    max_validation_retries: int = 2

    hyde_num_hypotheses: int = 3
    hyde_enabled_by_default: bool = False
    hybrid_search_enabled: bool = True
    sparse_scorer: str = "tfidf"  # "tfidf" (original) or "bm25"
    rrf_k: int = 60
    reranker_backend: str = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    voyage_api_key: str = ""
    voyage_model: str = "rerank-2.5"
    reranker_initial_top_k: int = 20
    reranking_enabled_by_default: bool = True
    crag_relevance_threshold: float = 0.7
    crag_ambiguous_threshold: float = 0.5
    crag_enabled_by_default: bool = True
    reflection_min_score: float = 0.85
    max_reflection_retries: int = 2
    self_reflective_enabled_by_default: bool = False

    log_json: bool = False
    log_level: str = "INFO"

    gst_corpus_path: str = "seed/docs/gst/gst_provisions_enriched_groq.jsonl"
    gst_golden_path: str = "seed/docs/gst/golden_set_with_gold_sources.jsonl"
    gst_qdrant_collection: str = "gst_provisions_enriched_v3"
    gst_chunk_max_tokens: int = 6_000
    gst_chunk_overlap_tokens: int = 200



settings = Settings()











