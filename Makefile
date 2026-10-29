.PHONY: help install sync seed seed-gst gst-fetch-sources gst-build-corpus gst-update-golden gst-smoke-test eval-gst final-system report api streamlit test lint format


help:
	@echo "GST Law RAG — Available commands"
	@echo ""
	@echo "  make install       — create venv & install all deps (one-time)"
	@echo "  make sync          — sync deps with pyproject.toml"
	@echo "  make seed          — run DB migrations + create the demo users"
	@echo "  make seed-gst      — ingest the GST provision corpus into Qdrant"
	@echo "  make gst-smoke-test — sanity-check GST retrieval against the golden question set"
	@echo "  make gst-fetch-sources — download + convert the GST notifications/circulars (seed/docs/gst/sources/)"
	@echo "  make gst-build-corpus  — parse bare law + notifications, enrich new records via Groq, rewrite the corpus"
	@echo "  make gst-update-golden — move newly ingested law from 'missing' to 'required' in the golden set"
	@echo "  make eval-gst      — RAGAS run on a 20-question sample (hybrid+rerank+crag)"
	@echo "  make final-system  — run the final system (E17 retrieval → LLM re-rank → grounded answer)"
	@echo "  make report        — rebuild the thesis tables and figures from eval/results/thesis/"
	@echo "  make api           — start FastAPI backend (:8000)"
	@echo "  make streamlit     — start Streamlit UI (:8501)"
	@echo "  make test          — run pytest"
	@echo "  make lint          — run ruff check"
	@echo "  make format        — run ruff format"


install:
	uv python pin 3.12
	uv venv --python 3.12
	uv sync --extra dev

sync:
	uv sync --extra dev

seed:
	uv run python scripts/seed_db.py

seed-gst:
	uv run python scripts/seed_gst.py

gst-fetch-sources:
	uv run python -m scripts.gst_pipeline.fetch_sources

gst-build-corpus:
	uv run python -m scripts.gst_pipeline.build_corpus --write

gst-update-golden:
	uv run python -m scripts.gst_pipeline.update_golden_sources --write

gst-smoke-test:
	uv run python scripts/gst_smoke_test.py

eval-gst:
	uv run python -m eval.run_gst_ragas

final-system:
	uv run python -m scripts.thesis.final_system

report:
	uv run python -m scripts.thesis.make_report

api:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	uv run streamlit run scripts/streamlit_app.py


test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
