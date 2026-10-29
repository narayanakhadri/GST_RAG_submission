"""T25 — Chain-of-Verification (CoVe): draft, plan verification questions, answer them from the law, revise.

Idea. Dhuliawala et al. 2023 (Meta, "Chain-of-Verification Reduces Hallucination in Large Language Models"): the
model (1) drafts an answer, (2) writes short verification questions about the claims in it ("Does section 17(5)(a)
block ITC on cars used for business?"), (3) answers each verification question *independently* (so the draft's
errors are not copied), and (4) writes the final answer using the verified facts. In RAG the verification answers
are grounded in retrieved law — here each verification question gets its own small retrieval, so a claim the first
retrieval could not support can still be checked. Related: EscavAI's post-generation fidelity check (T11), the
L-MARS judge (T10).

How it works here (free with Gemini: ≈ 3 + n_questions calls; small extra retrievals).
1. E17 + law + summary → draft (grounded prompt).
2. Plan: JSON list of ≤ `--checks` (default 4) verification questions about legal claims and citations.
3. Each: retrieve top 3 (hybrid, cap 2) for that question; answer it in ≤ 2 sentences from those provisions only.
4. Revise: the draft + the verification Q&A → final answer (fix anything contradicted, keep citations).
5. Contexts saved = the E17 provisions + the verification provisions (what the final answer could rely on).

    uv run python -m scripts.techniques.t25_chain_of_verification --set dev_hard
"""

from __future__ import annotations

import argparse
import json

from scripts.techniques.common import e17_chunks, generate_run, grounded_answer, use_best_stack

PLAN = ("List short verification questions that check the legal claims and citations of the draft answer to an "
        "Indian GST law question (which provision applies, what it says, any condition, rate or time limit). "
        "Return JSON {\"questions\": [..]}.")
VERIFY = "Answer the verification question in at most 2 sentences using only the provisions given; say 'not stated' if they do not answer it."
REVISE = ("Rewrite the draft answer to the GST question using the verification results: correct any claim they "
          "contradict, keep correct citations, do not add uncited claims. Keep the same output format as the draft.")


def answer(question: str, n_checks: int = 4) -> dict:
    from app.services.llm_service import generate, generate_with_json
    from app.services.rag_service import _retrieve

    chunks = e17_chunks(question)
    draft = grounded_answer(question, chunks)
    try:
        checks = json.loads(generate_with_json(PLAN, f"Question: {question}\n\nDraft answer:\n{draft}")["text"])
        checks = [c for c in checks.get("questions", []) if isinstance(c, str)][:n_checks]
    except Exception:  # noqa: BLE001
        checks = []
    qa, extra = [], []
    for vq in checks:
        found = _retrieve(vq, {"search_mode": "hybrid", "enable_crag": False, "top_k": 3, "doc_type_cap": 2,
                               "context_text": "law+summary"})
        extra += [c for c in found if c.source not in {x.source for x in chunks + extra}]
        law = "\n\n".join(f"[{c.source}]\n{c.text[:3000]}" for c in found)
        qa.append(f"Q: {vq}\nA: {generate(VERIFY, f'{law}\n\nVerification question: {vq}')['text'].strip()}")
    final = draft if not qa else generate(
        REVISE, f"Question: {question}\n\nDraft answer:\n{draft}\n\nVerification results:\n" + "\n\n".join(qa))["text"]
    return {"answer": final, "chunks": chunks + extra, "draft": draft, "verification": qa}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--checks", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run("t25_cove", lambda q: answer(q, args.checks), args.set, args.limit)


if __name__ == "__main__":
    main()
