"""T28 — Multi-model answer ensemble with a selector (two generators, one judge picks).

Idea. Different LLMs make different mistakes; generating with several and letting a selector pick the best answer
reduces single-model errors. The hybrid legal RAG agent of paper 11 ("Hybrid Retrieval-Augmented Generation
Agent", 2025) generates candidates with multiple LLMs and scores them with a specialised selector before
returning the top answer (ablations show the ensemble helps). Related: self-consistency (Wang et al. 2023),
Speculative RAG's verifier (T24), LLM-as-a-judge.

How it works here (free tiers: Gemini flash-lite + Groq gpt-oss-120b; Groq allows ~200k tokens/day, so ~35
questions per day with long contexts).
1. E17 retrieval + law + summary → the same 8 provisions for both generators (grounded prompt).
2. Candidates: Gemini (settings.gemini_model) and Groq (settings.groq_model_answer).
3. Selector (Gemini, JSON): scores each candidate for legal correctness against the retrieved law, completeness and
   citations → best one returned; the candidates and scores are kept per row.

    uv run python -m scripts.techniques.t28_ensemble_selector --set dev_hard --limit 10
"""

from __future__ import annotations

import argparse
import contextlib
import json

from scripts.techniques.common import e17_chunks, generate_run, grounded_answer, use_best_stack

SELECT = ("You compare candidate answers to an Indian GST law question against the retrieved law. Score each 0-10 for "
          "legal correctness, completeness and correct citations. Return JSON {\"scores\": [..], \"best\": <0-based index>}.")
PROVIDERS = ("gemini", "groq")


@contextlib.contextmanager
def provider(name: str):
    from app.config import settings

    old = settings.llm_provider
    settings.llm_provider = name
    try:
        yield
    finally:
        settings.llm_provider = old


def answer(question: str) -> dict:
    from app.services.llm_service import generate_with_json

    chunks = e17_chunks(question)
    cands, used = [], []
    for p in PROVIDERS:
        try:
            with provider(p):
                cands.append(grounded_answer(question, chunks))
                used.append(p)
        except Exception as exc:  # noqa: BLE001  (quota / outage: keep the other candidate)
            print(f"  {p} failed: {type(exc).__name__}")
    if len(cands) == 1:
        return {"answer": cands[0], "chunks": chunks, "candidates": dict(zip(used, cands, strict=True)), "selected": used[0]}
    law = "\n\n".join(f"[{c.source}]\n{c.text[:2500]}" for c in chunks)
    listing = "\n\n".join(f"CANDIDATE {i} ({p}):\n{a}" for i, (p, a) in enumerate(zip(used, cands, strict=True)))
    with provider("gemini"):
        try:
            verdict = json.loads(generate_with_json(SELECT, f"Question: {question}\n\nRetrieved law:\n{law}\n\n{listing}")["text"])
            best = int(verdict.get("best", 0))
        except Exception:  # noqa: BLE001
            verdict, best = {}, 0
    best = best if 0 <= best < len(cands) else 0
    return {"answer": cands[best], "chunks": chunks, "candidates": dict(zip(used, cands, strict=True)),
            "selected": used[best], "selector": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run("t28_ensemble", answer, args.set, args.limit)


if __name__ == "__main__":
    main()
