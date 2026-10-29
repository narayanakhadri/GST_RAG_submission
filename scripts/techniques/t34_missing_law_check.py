"""T34 — Missing-law check: the model lists the law each step of the answer needs, and the missing provisions are fetched.

Idea (from the SME review, 2026-10-21). Wrong final answers were mostly computations that need several provisions
(3.9 required vs 2.2 for the rest); retrieval found 61% of that law (84% for the rest), and one missing rule shifts the
computed amount ("use Rule 40(2)" — Rule 40 was not retrieved). Before answering, the model plans the answer step by
step (value of supply, rate/exemption, ITC eligibility and blocking, reversal, place of supply, time limits ...) and
names the provisions each step needs; named provisions that are not in the context are looked up by ID and appended.
Unlike T12 (look up whatever a draft answer cites, on E17), the check is a plan of the law needed, is told what is
already retrieved, and runs on the final system (T31). Related: Plan-and-Solve, Self-Ask, IRCoT, L-MARS.

How it works here (free with the Gemini keys: one extra call per question, cached).
1. T31 (final system) → top 8, law + summary context.
2. Prompt: question + the retrieved provisions' labels and titles → JSON {"steps": [{"step", "provisions"}]}.
3. Citations in the plan → corpus IDs (citation_lookup.cited_ids); a notification/circular → its entry closest to the
   question. New IDs (not retrieved) are appended, at most MAX_NEW, as law + summary.
4. `answer`: grounded answer over the 8 + new (the final system's generation step).
Control (`--control`): T31's own ranking with the same number of extra slots (top 8 + n), so a gain is not just k.

    uv run python -m scripts.techniques.t34_missing_law_check eval --set dev_hard
    uv run python -m scripts.techniques.t34_missing_law_check eval --set dev_hard --control
    uv run python -m scripts.techniques.t34_missing_law_check answer --set dev_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.techniques.common import (
    E17_FLAGS,
    evaluate,
    generate_run,
    grounded_answer,
    use_best_stack,
)
from scripts.techniques.t31_llm_pointwise_rerank import grades

MAX_NEW = 4
CACHE = Path("eval/results/thesis/cache/t34_plans.json")
PLAN_PROMPT = (
    "You are an Indian GST law expert planning the answer to an exam question. Break the answer into its steps "
    "(e.g. whether it is a supply, value of supply and what is included, rate or exemption, place of supply, input "
    "tax credit eligibility and blocked credits, reversal, time limits, registration, returns, interest or penalty) "
    "and for each step name EVERY provision of the law it needs, including provisions that are NOT in the retrieved "
    "list below. Cite in the form 'section 16(2) of the CGST Act', 'rule 42 of the CGST Rules', 'section 12 of the "
    "IGST Act', 'Schedule III of the CGST Act', 'Notification No. 12/2017-Central Tax (Rate)'. Do not answer the "
    "question. Return JSON {\"steps\": [{\"step\": \"...\", \"provisions\": [\"...\"]}]}."
)


def plan(question: str, top: list) -> str:
    from app.services.llm_service import generate_with_json, grader_model_default

    listing = "\n".join(f"- {c.text.splitlines()[0][:160]}" for c in top)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{PLAN_PROMPT}|{question}|{listing}".encode()).hexdigest()
    if key not in cache:
        try:
            cache[key] = generate_with_json(PLAN_PROMPT, f"Question: {question}\n\nRetrieved provisions:\n{listing}")["text"]
        except Exception:  # noqa: BLE001
            cache[key] = ""
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache[key]


def t31_ordered(question: str) -> list:
    """T31's pool in re-ranked order (uncapped); the final system is cap_doc_types(this, 8, 3)."""
    from app.services.rag_service import _retrieve

    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None, "context_text": "law+summary"})
    g = grades(question, pool, reasoning=True)
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-g[i], i))]


def missing_law(question: str, top: list) -> list[str]:
    from app.services.citation_lookup import cited_ids
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph
    from app.services.vector_store import search_within

    graph = get_graph()
    by_inst: dict[str, list[str]] = {}
    for pid, r in graph.records.items():
        by_inst.setdefault(r["instrument_id"], []).append(pid)
    have, new = {graph.id_of(c) for c in top}, []
    for cid in cited_ids(plan(question, top), set(graph.records), set(by_inst)):
        if cid in graph.records:
            pid = cid
        elif any(h and h.startswith(cid + ".") for h in have):
            continue  # an entry of this notification/circular is already in the context
        else:
            pid = (search_within(embed_texts([question])[0], by_inst[cid], 1) or [None])[0]
        if pid and pid not in have and pid not in new:
            new.append(pid)
    return new[:MAX_NEW]


def retrieve(question: str, control: bool = False) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph, to_law_context

    graph = get_graph()
    ordered = t31_ordered(question)
    top = cap_doc_types(ordered, graph, 8, 3)
    new = missing_law(question, top)
    if control:  # same number of extra slots, filled by T31's own ranking
        return cap_doc_types(ordered, graph, 8 + len(new), 3 + len(new))
    return top + to_law_context([graph.chunk_for(p) for p in new], graph, with_summary=True)


def answer(question: str) -> dict:
    from app.services.gst_graph import get_graph

    chunks = retrieve(question)
    graph = get_graph()
    return {"answer": grounded_answer(question, chunks), "chunks": chunks,
            "added_ids": [graph.id_of(c) for c in chunks[8:]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval", "answer"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--control", action="store_true", help="T31 with the same number of extra slots")
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "eval":
        evaluate(f"t34_missing_law{'_control' if args.control else ''}_{args.set}",
                 lambda q: retrieve(q, args.control), args.set)
    else:
        generate_run("t34_missing_law", answer, args.set)


if __name__ == "__main__":
    main()
