"""T31 — LLM relevance re-ranking of the pool, with vs without reasoning ("does the reranker need to think?").

Idea. Reasoning rerankers (Rank1, Weller et al. 2025; Rank-R1, Zhuang et al. 2025; ReasonRank 2025) spend tokens
thinking before judging relevance. Jedidi et al. 2025 ("Don't 'Overthink' Passage Reranking: Is Reasoning Truly
Necessary?") found the reasoning variant *underperforms* an equivalent non-reasoning pointwise reranker: once a
passage is judged relevant, scores cluster and ordering suffers. This script runs the same LLM relevance judge on
E17's 20 candidates twice — `--reasoning off`: grade only; `on`: a one-paragraph analysis per candidate first —
so the claim can be tested on legal retrieval. (The earlier listwise LLM reranker, `rerank_llm` / `c2_llm_rerank`,
orders the whole list at once; this one grades each candidate on a 0–3 scale, batched in one call.)

How it works here (free: one Gemini call per question per mode, cached).
- Prompt lists the question and the 20 candidates (label + first 700 characters of the law + summary);
  returns JSON {"grades": [0–3 per candidate]} (3 = the answer depends on it, 0 = irrelevant).
- Order: grade (descending), ties broken by the E17 rank; then cap 3 → top 8.

    uv run python -m scripts.techniques.t31_llm_pointwise_rerank eval --set dev_hard --reasoning off
    uv run python -m scripts.techniques.t31_llm_pointwise_rerank eval --set dev_hard --reasoning on
    uv run python -m scripts.techniques.t31_llm_pointwise_rerank eval --set all --reasoning on --pool bm25_citation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

CACHE = Path("eval/results/thesis/cache/pointwise_grades.json")
BASE = ("You grade how relevant each candidate provision of Indian GST law is to answering the question: 3 = the "
        "answer depends on it, 2 = useful, 1 = marginal, 0 = irrelevant.")
OFF = BASE + " Return JSON {\"grades\": [one integer per candidate, in order]}."
ON = (BASE + " First write one short analysis per candidate (what it says and whether the question needs it), then "
      "grade. Return JSON {\"analysis\": [..], \"grades\": [one integer per candidate, in order]}.")


def grades(question: str, pool: list, reasoning: bool) -> list[int]:
    from app.services.llm_service import generate_with_json, grader_model_default

    prompt = ON if reasoning else OFF
    listing = "\n\n".join(f"[{i}] {c.source}\n{c.text[:700]}" for i, c in enumerate(pool))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{prompt}|{question}|{[c.source for c in pool]}".encode()).hexdigest()
    if key not in cache:
        try:
            g = json.loads(generate_with_json(prompt, f"Question: {question}\n\nCandidates:\n{listing}")["text"])["grades"]
            cache[key] = [int(x) for x in g][: len(pool)]
        except Exception:  # noqa: BLE001
            cache[key] = []
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    g = cache[key]
    return g + [0] * (len(pool) - len(g))


def retrieve(question: str, reasoning: bool, pool_flags: dict | None = None) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve

    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None, "context_text": "law+summary",
                                **(pool_flags or {})})
    g = grades(question, pool, reasoning)
    order = sorted(range(len(pool)), key=lambda i: (-g[i], i))
    return cap_doc_types([pool[i] for i in order], get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--reasoning", choices=["off", "on"], default="off")
    ap.add_argument("--pool", choices=["e17", "bm25_citation"], default="e17",
                    help="candidate pool: E17, or E17 with BM25 + cited-provision lookup (config c2_bm25_citation)")
    args = ap.parse_args()
    use_best_stack()
    flags = {"sparse_scorer": "bm25", "citation_lookup": True} if args.pool == "bm25_citation" else {}
    suffix = "" if args.pool == "e17" else f"_{args.pool}"
    evaluate(f"t31_pointwise_{args.reasoning}{suffix}_{args.set}",
             lambda q: retrieve(q, args.reasoning == "on", flags), args.set)


if __name__ == "__main__":
    main()
