"""T24 — Speculative RAG: several drafts from different subsets of the retrieved law, a verifier picks one.

Idea. Wang et al. 2024 ("Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting") let a small
drafter write answers from *different subsets* of the retrieved documents in parallel (each draft sees fewer
tokens and a different perspective), and a larger verifier scores the drafts and keeps the best — reported more
accurate and faster than one model reading everything. Here both roles use Gemini flash-lite (the verifier with a
scoring prompt); the subsets are built so each draft sees different provisions.

How it works here (free with Gemini: `--drafts` + 1 calls per question).
1. E17 retrieval + law + summary → 8 provisions.
2. Subsets: `--drafts` (default 3) round-robin groups of the ranked provisions (1,4,7 / 2,5,8 / 3,6), each plus
   the top-ranked provision so every draft sees the strongest hit.
3. Draft = grounded answer from that subset (with a one-line rationale).
4. Verifier: sees the question, all drafts and the full 8 provisions; returns JSON {"best": i, "scores": [...]}.
5. Saved in the RAGAS format (contexts = all 8 provisions; `drafts` and the verifier's choice kept per row).

    uv run python -m scripts.techniques.t24_speculative_rag --set dev_hard
"""

from __future__ import annotations

import argparse
import json

from scripts.techniques.common import e17_chunks, generate_run, grounded_answer, use_best_stack

VERIFY = ("You verify draft answers to an Indian GST law question against the retrieved law. Score each draft 0-10 "
          "for legal correctness, completeness (every issue answered) and correct citations. Return JSON "
          "{\"scores\": [..], \"best\": <index of the best draft, 0-based>}.")


def answer(question: str, n_drafts: int = 3) -> dict:
    from app.services.llm_service import generate_with_json

    chunks = e17_chunks(question)
    groups = [[chunks[0]] + [c for i, c in enumerate(chunks[1:], start=1) if i % n_drafts == k % n_drafts]
              for k in range(n_drafts)]
    drafts = [grounded_answer(question, g) for g in groups]
    law = "\n\n".join(f"[{c.source}]\n{c.text[:2500]}" for c in chunks)
    listing = "\n\n".join(f"DRAFT {i}:\n{d}" for i, d in enumerate(drafts))
    try:
        verdict = json.loads(generate_with_json(VERIFY, f"Question: {question}\n\nRetrieved law:\n{law}\n\n{listing}")["text"])
        best = int(verdict.get("best", 0))
    except Exception:  # noqa: BLE001
        verdict, best = {}, 0
    best = best if 0 <= best < len(drafts) else 0
    return {"answer": drafts[best], "chunks": chunks, "drafts": drafts, "verifier": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--drafts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run(f"t24_speculative_d{args.drafts}", lambda q: answer(q, args.drafts), args.set, args.limit)


if __name__ == "__main__":
    main()
