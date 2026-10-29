"""T26 — Sub-question answering: answer each decomposed issue with its own retrieval, then compose.

Idea. E17 uses decomposition only for *retrieval* (all sub-queries' hits are fused into one context). Exam case
studies raise 2–4 separate legal issues; answering each issue from the law retrieved for *that* issue, then
composing, keeps each part focused (least-to-most prompting, Zhou et al. 2023; LlamaIndex SubQuestionQueryEngine;
IRCoT / multi-hop QA decomposition in the RAG survey; the Query agent of L-MARS plans sub-questions).

How it works here (free with Gemini: n_sub + 1 calls; sub-queries come from the cached decomposition).
1. Sub-questions = the E17 decomposition queries (query_rewrite.py "decompose", cached).
2. For each: hybrid retrieval top 4 (cap 2, law + summary) → a 2–4 sentence answer citing provisions.
3. Compose: the original question + the sub-answers (+ the E17 top 8 for anything missing) → final answer in the
   usual format (grounded prompt).
4. Contexts saved = union of the sub-question provisions and the E17 provisions.

    uv run python -m scripts.techniques.t26_subquestion_answering --set dev_hard
"""

from __future__ import annotations

import argparse

from scripts.techniques.common import e17_chunks, generate_run, grounded_answer, use_best_stack

SUB = ("Answer this sub-question of an Indian GST law problem in 2-4 sentences, using only the provisions given and "
       "citing them (e.g. 'section 16(2) of the CGST Act').")
COMPOSE = ("\nYou are given answers to the sub-issues of the question (below the retrieved law). Combine them into one "
           "complete answer to the original question, resolving conflicts in favour of the retrieved law.")


def answer(question: str) -> dict:
    from app.services.llm_service import generate
    from app.services.query_rewrite import rewrite_queries
    from app.services.rag_service import _retrieve

    subs = rewrite_queries(question, "decompose")
    base = e17_chunks(question)
    used = list(base)
    notes = []
    for sq in subs:
        found = _retrieve(sq, {"search_mode": "hybrid", "enable_crag": False, "top_k": 4, "doc_type_cap": 2,
                               "context_text": "law+summary"})
        used += [c for c in found if c.source not in {u.source for u in used}]
        law = "\n\n".join(f"[{c.source}]\n{c.text[:3000]}" for c in found)
        notes.append(f"Sub-issue: {sq}\nAnswer: {generate(SUB, f'{law}\n\nSub-question: {sq}')['text'].strip()}")
    final = grounded_answer(question + "\n\nSub-issue answers:\n" + "\n\n".join(notes), base, COMPOSE)
    return {"answer": final, "chunks": used, "sub_answers": notes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run("t26_subquestions", answer, args.set, args.limit)


if __name__ == "__main__":
    main()
