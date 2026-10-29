"""LLM query understanding for long, scenario-style questions (exam case studies).

A case study ("Satyamev Printers receives an order to print 5000 copies of a book...") is a poor search
query: most of its words describe the facts, not the law. Two rewrites are offered:

- "rewrite"   — one search query naming the legal issue and the likely provision.
- "decompose" — 2–4 sub-queries, one per legal issue (multi-query retrieval / RAG-Fusion); the caller
                retrieves for each plus the original question and fuses the rankings with RRF.

Rewrites are cached on disk (keyed by mode + prompt text + model + question) so repeated experiments don't
call the LLM again, while editing a prompt or switching model computes fresh rewrites.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from loguru import logger

from app.services.llm_service import generate_with_json, grader_model_default

CACHE_PATH = Path("eval/results/thesis/cache/query_rewrite.json")
MAX_SUBQUERIES = 4

_PROMPTS = {
    "rewrite": (
        "You turn Indian GST exam questions into one search query for a database of GST law "
        "(CGST/IGST/UTGST Acts and Rules, notifications, circulars). Ignore names, amounts and story details; "
        "name the legal issue and, if you know it, the provision (e.g. 'composite supply principal supply "
        "section 8 CGST Act'). Return JSON: {\"query\": \"...\"}"
    ),
    "decompose": (
        "You break Indian GST exam questions into the separate legal issues they raise, as search queries for "
        "a database of GST law (CGST/IGST/UTGST Acts and Rules, notifications, circulars). Ignore names, "
        "amounts and story details. Give 2-4 short queries, one per issue, each naming the legal concept and, "
        "if you know it, the provision (e.g. 'place of supply of services section 12 IGST Act', "
        "'exemption printing of books notification 12/2017'). Return JSON: {\"queries\": [\"...\"]}"
    ),
    # Step-back prompting (Zheng et al. 2023): one more general question about the underlying principle.
    "stepback": (
        "You read an Indian GST exam question and step back from its facts: write one general question about "
        "the legal principle or concept it depends on, as a search query for a database of GST law "
        "(e.g. for a question about a free gift to an employee: 'when is supply without consideration to a "
        "related person or employee taxable under GST'). Return JSON: {\"query\": \"...\"}"
    ),
}

_lock = threading.Lock()
_cache: dict[str, list[str]] | None = None


def _load_cache() -> dict[str, list[str]]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")


def cache_key(mode: str, model: str, question: str) -> str:
    prompt = hashlib.sha256(_PROMPTS[mode].encode()).hexdigest()[:12]
    return hashlib.sha256(f"{mode}|{prompt}|{model}|{question}".encode()).hexdigest()


def rewrite_queries(question: str, mode: str) -> list[str]:
    """Search queries for `question` under `mode` ("rewrite" | "decompose" | "stepback"); [] if the LLM call fails."""
    if mode not in _PROMPTS:
        raise ValueError(f"unknown query_rewrite mode: {mode}")
    model = grader_model_default()
    key = cache_key(mode, model, question)
    with _lock:
        cached = _load_cache().get(key)
    if cached is not None:
        return cached

    try:
        raw = generate_with_json(_PROMPTS[mode], question, model=model)["text"]
        data = json.loads(raw)
        queries = [data["query"]] if mode != "decompose" else list(data.get("queries") or [])[:MAX_SUBQUERIES]
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    except Exception:
        logger.exception("query rewrite failed ({}); falling back to the original question", mode)
        return []

    with _lock:
        _load_cache()[key] = queries
        _save_cache()
    return queries
