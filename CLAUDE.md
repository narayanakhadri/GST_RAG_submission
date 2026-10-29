# CLAUDE.md

Guidance for Claude Code in this repository. Start with `README.md` (layout, setup, results).

## What this is

A FastAPI RAG service answering Indian GST law questions over a structured corpus (one record per provision,
`seed/docs/gst/gst_provisions_enriched_groq.jsonl`, Qdrant collection `gst_provisions_enriched_v3`), plus the
thesis experiment code (`scripts/techniques/`, `scripts/thesis/`, `eval/`).

## Commands

```bash
make install / make sync     # uv
docker compose up -d         # Postgres + Qdrant
make seed && make seed-gst   # users + corpus ingestion
make api / make streamlit
uv run pytest tests/unit -v
uv run ruff check .
```

## Key code paths

- `app/api/query.py` — security layers (rate limit → token budget → input restructuring → llm-guard → PII
  redaction) run before the LangGraph; output is moderated again after. Layer order is load-bearing.
- `app/core/graph.py` — `START → generate_answer → finalize`, Postgres checkpointer.
- `app/services/rag_service.py` — `run_rag()` / `run_rag_with_trace()`: `_retrieve()` (HyDE → dense/sparse/hybrid
  search with RRF → rerank → CRAG) then `_generate()` (optional Self-RAG). Flags read via `_flag()`.
- `app/services/llm_service.py` — OpenAI / Groq / Gemini behind one interface (`settings.llm_provider`);
  Gemini rotates across comma-separated keys. Embeddings always use OpenAI.
- `app/datasets/gst/` — corpus/golden loaders and `document_mapper.py` (deterministic uuid5 point IDs, payload
  compatible with `vector_store.py`).
- `eval/gst_retrieval.py` — retrieval scores against gold provisions (recall, MRR, hit@k), split by corpus coverage.
- `scripts/thesis/final_system.py` — the final system (E17 retrieval → T31 LLM re-rank → grounded answer).

## Rules

- Set `LLM_PROVIDER=gemini` for experiments; the code default is OpenAI.
- One judge (Qwen via OpenRouter, `RAGAS_JUDGE_*`) for every reported RAGAS number.
- Tune on `dev_hard`, confirm on `test_hard`, report on all 141 fully covered questions.
- Corpus changes go to the whole corpus, never only to gold provisions. Never print API keys.
