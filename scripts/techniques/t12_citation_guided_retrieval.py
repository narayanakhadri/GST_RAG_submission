"""T12 — Citation-guided second retrieval (iterative RAG: retrieve → draft → retrieve the cited law → answer).

Idea. After a first retrieval, a draft answer often names provisions that search missed ("… blocked under
section 17(5)(a)"): the LLM's legal knowledge points to the right law even when the question's wording did not.
Looking those citations up and adding them to the context is a cheap form of iterative retrieval. Related:
ITER-RETGEN and IRCoT (iterate retrieval with generation, RAG survey), FLARE (retrieve when the draft needs it),
L-MARS refine step (T10), LQ-RAG recursive feedback.

How it works here (free with the Gemini keys: one extra LLM call per question; decompositions cached).
1. E17 retrieval → top 8.
2. Draft: a short answer from those 8 that must cite provisions ("Section 16(2) of the CGST Act" style) — the
   draft may also cite law it believes applies but did not see (marked in the prompt as allowed for the draft).
3. Citations in the draft → corpus IDs (citation_lookup.py); a cited notification/circular → its entry closest to
   the question.
4. New IDs not already retrieved are inserted after the first `KEEP` E17 results (so strong hits stay on top),
   then the doc-type cap and top 8 (or top 8 + new, flag `--grow`) are applied.

    uv run python -m scripts.techniques.t12_citation_guided_retrieval eval --set dev_hard
    uv run python -m scripts.techniques.t12_citation_guided_retrieval eval --set dev_hard --grow   # k = 8 + new
"""

from __future__ import annotations

import argparse

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

KEEP = 5
DRAFT_PROMPT = (
    "You are an Indian GST law expert. Using the retrieved provisions below and your own knowledge, write a short "
    "draft answer (max 120 words) to the question. Cite every provision you rely on in the form 'section 16(2) of "
    "the CGST Act', 'rule 42 of the CGST Rules', 'section 12 of the IGST Act', 'Notification No. 12/2017-Central "
    "Tax (Rate)' — including provisions that apply but are NOT in the retrieved text."
)


def retrieve(question: str, grow: bool = False) -> list:
    from app.services.citation_lookup import cited_ids
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.llm_service import generate
    from app.services.rag_service import _retrieve
    from app.services.vector_store import search_within

    graph = get_graph()
    first = _retrieve(question, {**E17_FLAGS, "top_k": 20})  # pool, capped below
    top = cap_doc_types(first, graph, 8, 3)
    context = "\n\n".join(f"[{c.source}]\n{c.text[:1500]}" for c in top)
    try:
        draft = generate(DRAFT_PROMPT, f"Retrieved provisions:\n{context}\n\nQuestion: {question}")["text"]
    except Exception:  # noqa: BLE001
        return top
    by_inst: dict[str, list[str]] = {}
    for pid, r in graph.records.items():
        by_inst.setdefault(r["instrument_id"], []).append(pid)
    have = {graph.id_of(c) for c in top}
    new = []
    for cid in cited_ids(draft, set(graph.records), set(by_inst)):
        pid = cid if cid in graph.records else (search_within(embed_texts([question])[0], by_inst[cid], 1) or [None])[0]
        if pid and pid not in have and pid not in new:
            new.append(pid)
    added = [graph.chunk_for(p) for p in new]
    if grow:  # E17's 8 plus every newly cited provision
        return top + added
    head = top[:KEEP] + added  # keep the strongest hits on top, then the cited law, then the rest
    seen = {graph.id_of(x) for x in head + top}
    rest = top[KEEP:] + [c for c in first if graph.id_of(c) not in seen]
    return cap_doc_types(head + rest, graph, 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--grow", action="store_true", help="keep E17's 8 and append the cited ones (k grows)")
    args = ap.parse_args()
    use_best_stack()
    name = f"t12_citation_guided{'_grow' if args.grow else ''}_{args.set}"
    evaluate(name, lambda q: retrieve(q, args.grow), args.set)


if __name__ == "__main__":
    main()
