# GST Law RAG — question answering over Indian GST law

A retrieval-augmented generation (RAG) system that answers exam-style questions on Indian Goods and Services Tax
law, with citations to the provisions it used. It is a FastAPI service (auth, LLM-security pipeline, caching,
LangGraph orchestration) plus the evaluation and experiment code behind the thesis results.

## What is in the repo

| Path | Contents |
|---|---|
| `app/` | FastAPI service: `api/` (auth, query, admin), `core/graph.py` (LangGraph), `services/` (retrieval, reranking, HyDE, CRAG, Self-RAG, LLM, cache), `security/` (input/output guard layers), `datasets/gst/` (corpus + golden loaders, provision → Qdrant point mapper) |
| `seed/docs/gst/` | Corpus `gst_provisions_enriched_groq.jsonl` (881 records → 866 points), golden sets, notification/circular sources |
| `seed/scripts/GST_Act.md` | Bare-law source text the corpus is parsed from |
| `scripts/gst_pipeline/` | Builds the corpus: parse bare law, fetch + parse notifications, LLM enrichment, golden-source updates |
| `scripts/techniques/` | 33 standalone experiments, one paper idea each (shared harness `common.py`) |
| `scripts/thesis/` | Ablation runner, question sets, final system, report tables/figures, result archiving |
| `eval/` | GST RAGAS harness (`run_gst_ragas.py`, `rescore_ragas.py`) and retrieval scoring against gold provisions (`gst_retrieval.py`) |
| `eval/results/thesis/` | All results: `archive/` (every run, gzipped), `retrieval/`, `techniques/`, `sme_v1/`, `figures/`, `results_tables.md`, `RESULTS_INDEX.md`, cached LLM outputs in `cache/` |
| `tests/unit/` | Unit tests (no external services) |

## Corpus

881 structured records, one per legal provision: CGST Act (189 sections + Schedules I–III), CGST Rules (210),
IGST Act (28), IGST Rules (9), UTGST Act (29), and 24 notifications/circulars split into 413 entries/paragraphs.
Each record carries raw text plus LLM-written metadata (summary, topics, keywords). Chunking is per provision,
not a sliding window. Golden set: 146 questions with gold provision IDs (`golden_set_with_gold_sources.textfix.jsonl`;
`golden_set_sme_v1.jsonl` adds the subject-matter-expert corrections).

## Architecture

```
request → rate limit → token budget → input restructuring → llm-guard scan → PII redaction
        → LangGraph: generate_answer (run_rag) → finalize
        → output moderation/redaction → response
run_rag: [HyDE | query decomposition] → hybrid search (dense + TF-IDF, RRF) → [rerank] → [CRAG grading + web fallback]
         → grounded generation → [Self-RAG reflection]
```

Per-request flags (`top_k`, `search_mode`, `enable_hyde`, `enable_rerank`, `enable_crag`,
`enable_self_reflective`) come from `QueryRequest` in `app/models.py`; defaults are in `app/config.py`.

**Final system** (`scripts/thesis/final_system.py`): text-embedding-3-large → hybrid dense + TF-IDF (RRF) → LLM
decomposition into sub-queries (RRF) → pool of 20 → LLM writes a short analysis of each candidate and grades
it 0–3 → top 8 (at most 3 notifications/circulars) → grounded answer by Gemini 3.1 flash-lite.

## Headline results (141 fully covered questions, required-law recall / MRR)

| System | Recall | MRR |
|---|---|---|
| Baseline: hybrid dense + TF-IDF, top 5 | 0.425 | 0.596 |
| E17: 3-large, top 8, cap 3, LLM decomposition | 0.668 | 0.785 |
| **Final: E17 + LLM pointwise re-rank (T31)** | **0.696** | **0.914** |

Final-system answers on all 146 questions (Qwen3-30B judge via OpenRouter): faithfulness 0.568, answer relevancy
0.662, answer correctness 0.733. Every ablation table: `eval/results/thesis/results_tables.md`; every run:
`eval/results/thesis/RESULTS_INDEX.md`.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker.

```bash
make install                 # uv venv + uv sync --extra dev
docker compose up -d         # Postgres 16 (:5432) + Qdrant 1.17 (:6333)
cp .env.example .env         # fill in OPENAI_API_KEY (embeddings), GEMINI_API_KEY, JWT_SECRET, ...
make seed                    # Postgres: users table + demo users (agent@demo.local/agent123, admin@demo.local/admin123)
make seed-gst                # embed the corpus into Qdrant collection gst_provisions_enriched_v3
make gst-smoke-test          # dense hit@k sanity check against the golden set
```

Run the app:

```bash
make api                     # FastAPI on :8000, interactive docs at /docs
make streamlit               # UI on :8501 (login, query with per-feature presets, eval dashboard)
```

## Keeping the law current

The corpus holds the law the thesis was evaluated on (bare law as of April 2026; notifications and circulars issued
2017–2022). A new notification or circular is added with one command, without rebuilding the corpus:

```bash
make gst-add-source ARGS='--instrument-id notification_06_2025_ct_rate --kind notification \
  --number "06/2025-Central Tax (Rate)" --date 2025-01-16 --subject "Amends 12/2017-CT(Rate)" \
  --url https://gstcouncil.gov.in/sites/default/files/2025-01/ctr06-2025.pdf'
```

`scripts/gst_pipeline/add_source.py` registers the source (so a full rebuild keeps it), converts the PDF with
Docling, splits it into paragraph/entry records, enriches only the new records (Groq; `--no-enrich` embeds the plain
text for free), and upserts only their points into Qdrant. Point IDs are deterministic, so reruns add nothing new, and
the sparse index is rebuilt from Qdrant on each query, so the new law is searchable at once. Use `--pdf FILE` for a
PDF downloaded by hand (CBIC's site needs a browser; GST 2.0 rate notification 9/2025-CT(Rate) is not on the GST
Council listing), and `--corpus/--collection/--init-from/--no-register` to try a source on a copy first. Finding new
instruments is still a manual step.

## Reproducing the results

```bash
uv run pytest tests/unit -v                                        # unit tests
uv run python -m scripts.thesis.run_ablation --summary             # rebuild the ablation table from the stored results
uv run python -m scripts.thesis.run_ablation --only hybrid_tfidf --force   # rerun one config (stored results are skipped without --force)
uv run python -m scripts.thesis.final_system --set all             # final-system answers → eval/results/<stamp>_gst_final_t31_all.json
uv run python -m eval.rescore_ragas eval/results/<run>.json --metrics faithfulness,answer_relevancy,answer_correctness
uv run python -m scripts.thesis.make_report                        # tables + figures from the stored results
uv run python -m eval.run_gst_ragas --question-set dev_hard        # RAGAS on a named question set
```

Notes:
- Experiments use `LLM_PROVIDER=gemini` (set in `.env.example`); embeddings always use OpenAI.
- Redis (Upstash) is optional: with `UPSTASH_REDIS_URL`/`TOKEN` unset, the cache, rate limiter and token budget
  run in-process.
- On Windows, set `PYTHONUTF8=1` so the scripts can print the result tables (they contain characters like "→").
- `eval/results/thesis/cache/` holds the cached LLM outputs (T31 grades, query rewrites, HyDE answers, …), so
  reruns give the same numbers without new LLM calls. The embedding cache is not included; the first run
  re-embeds through OpenAI.
- Raw run files are stored gzipped in `eval/results/thesis/archive/`; `scripts.thesis.archive_results.read_json()`
  reads either form, so `make_report` works without the uncompressed files.
- The corpus build (`make gst-build-corpus`) needs `GROQ_API_KEY`. Its intermediate files (`seed/docs/gst/build/`)
  are not included; they are regenerated.
- Scripts for the subject-matter-expert review and thesis-document assembly (`scripts/thesis/sme_*`,
  `apply_sme_review.py`, `golden_audit.py`, `codex_gold_rescore.py`, `evidence_quotes.py`, `rag_value_check.py`,
  `amount_match.py`, `build_thesis.py`, `build_report_v8.py`, `references_key.py`, `update_word_fields.py`,
  `scripts/gst_pipeline/make_review_sheet.py`) need
  review workspaces and drafts that are not part of this submission. Their outputs that the results depend on
  are included (`golden_set_sme_v1.jsonl`, `eval/results/thesis/sme_v1/`).
