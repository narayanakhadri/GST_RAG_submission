"""T23 — Self-Route: answer from RAG, fall back to long context only when the model says RAG is not enough.

Idea. Li et al. 2024 ("Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid
Approach") found long-context LLMs beat RAG on average but cost far more; Self-Route asks the model first, with
the RAG context, "can this be answered from the text? if not, say 'unanswerable'". Only those questions go to
the long-context reader — most of the long-context quality at a fraction of the tokens. Here the long-context
reader is T18 (whole chapters, ≤ 100k tokens).

How it works here (generation free with Gemini; judging as usual).
1. E17 retrieval + law + summary context (the final system).
2. Self-check prompt: answer, or reply exactly "UNANSWERABLE" if the retrieved law does not cover the question.
3. If UNANSWERABLE → T18 long context (top 3 chapters) → answer from that.
4. Saved in the RAGAS format with `route` = "rag" | "long_context" per row; report the share routed.

    uv run python -m scripts.techniques.t23_self_route --set dev_hard
"""

from __future__ import annotations

import argparse

from scripts.techniques.common import e17_chunks, generate_run, grounded_answer, use_best_stack

SELF_CHECK = ("\nIf the retrieved law does not contain what is needed to answer the question, reply with the single "
              "word UNANSWERABLE and nothing else.")


def answer(question: str) -> dict:
    from app.models import RetrievedChunk
    from scripts.techniques.t18_long_context import long_context

    chunks = e17_chunks(question)
    text = grounded_answer(question, chunks, SELF_CHECK)
    if "UNANSWERABLE" not in text.upper()[:40]:
        return {"answer": text, "chunks": chunks, "route": "rag"}
    contexts, _ = long_context(question, 3, 100_000)
    long_chunks = [RetrievedChunk(text=c, source=c.split("\n", 1)[0].split(":")[0], score=0.0) for c in contexts]
    return {"answer": grounded_answer(question, long_chunks), "chunks": long_chunks, "route": "long_context"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run("t23_self_route", answer, args.set, args.limit)


if __name__ == "__main__":
    main()
