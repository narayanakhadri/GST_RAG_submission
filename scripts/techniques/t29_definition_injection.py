"""T29 — Definition-aware RAG: add the statutory definitions a question or a retrieved provision depends on.

Idea. Legal provisions lean on defined terms ("composite supply", "intermediary", "aggregate turnover",
"export of services"); the definition lives in section 2, far from the provision that uses it. CRAwLeR
(Jalocha & Michelsen 2026, cross-reference-aware legal retrieval) shows that target chunks often depend on context
elsewhere in the act and that retrievers miss it; NitiBench (EMNLP 2025) found answers that needed a neighbouring
section defining the term. The fix here is symbolic: parse section 2 of each Act into term → clause, and
(a) retrieval: put s.2 of the right Act into the top-k when the question uses a specific defined term;
(b) generation: add only the matching definition clauses as a short "Definitions" context block (cheaper than
    the whole of s.2, which is the longest provision in the corpus).

How it works here (free, no LLM).
- `definitions()`: "(30) "composite supply" means …" → {term: (act, clause no., text)} for the CGST, IGST and
  UTGST Acts (≈ 140 terms). Generic one-word terms (goods, person, business, …) are ignored.
- `eval`: E17 + s.2 of the Act whose defined term appears in the question (inserted at rank 3) → recall.
- `definitions_block(question, chunks)`: the generation hook — matching clauses from the question and the
  retrieved texts, ≤ 6 clauses.

    uv run python -m scripts.techniques.t29_definition_injection terms "Is this a composite supply or mixed supply?"
    uv run python -m scripts.techniques.t29_definition_injection eval --set all
"""

from __future__ import annotations

import argparse
import re
import threading

from app.models import RetrievedChunk
from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

ACTS = {"cgst_act_2017.section.2": "CGST Act", "igst_act_2017.section.2": "IGST Act",
        "utgst_act_2017.section.2": "UTGST Act"}
DEF = re.compile(r"\((\d{1,3}[a-z]{0,2})\)\s*[\"“”']([^\"“”']{2,60})[\"“”']\s*(?:means|includes|shall)")
GENERIC = {"agent", "business", "goods", "services", "person", "supply", "tax", "government", "council", "court",
           "document", "market value", "money", "notification", "prescribed", "proper officer", "quarter", "recipient",
           "rules", "section", "securities", "supplier", "the act", "turnover", "assessment", "audit", "commissioner",
           "central tax", "state tax", "integrated tax", "union territory", "family", "fund", "appointed day"}
INSERT_AT = 2
_lock, _defs = threading.Lock(), {}


def definitions() -> dict[str, tuple[str, str, str]]:
    from app.services.gst_graph import get_graph

    with _lock:
        if not _defs:
            graph = get_graph()
            for sid in ACTS:
                rec = graph.records.get(sid)
                if not rec:
                    continue
                text = rec.get("raw_text") or ""
                matches = list(DEF.finditer(text))
                for i, m in enumerate(matches):
                    term = " ".join(m.group(2).lower().split())
                    end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), m.end() + 1500)
                    if term not in GENERIC and not term.endswith(" act") and (len(term.split()) > 1 or len(term) > 9):
                        _defs.setdefault(term, (sid, m.group(1), " ".join(text[m.start():end].split())[:1200]))
        return _defs


def terms_in(text: str) -> list[str]:
    low = " ".join(text.lower().split())
    return [t for t in definitions() if re.search(rf"\b{re.escape(t)}\b", low)]


def definitions_block(question: str, chunks: list[RetrievedChunk], limit: int = 6) -> RetrievedChunk | None:
    """Generation hook: one context block with the definitions used by the question and retrieved provisions."""
    found = list(dict.fromkeys(terms_in(question) + [t for c in chunks for t in terms_in(c.text)]))[:limit]
    if not found:
        return None
    lines = [f"{ACTS[definitions()[t][0]]} s.2({definitions()[t][1]}): {definitions()[t][2]}" for t in found]
    return RetrievedChunk(text="Definitions:\n" + "\n".join(lines), source="Definitions (section 2)", score=0.0)


def retrieve(question: str) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve

    graph = get_graph()
    top = _retrieve(question, E17_FLAGS)
    acts = list(dict.fromkeys(definitions()[t][0] for t in terms_in(question)))
    have = {graph.id_of(c) for c in top}
    add = [graph.chunk_for(a) for a in acts if a not in have]
    if not add:
        return top
    return cap_doc_types(top[:INSERT_AT] + add + top[INSERT_AT:], graph, 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["terms", "eval", "stats"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--set", default="all")
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "stats":
        print(len(definitions()), "specific defined terms;", sorted(definitions())[:20])
    elif args.cmd == "terms":
        for t in terms_in(args.text):
            sid, num, text = definitions()[t]
            print(f"{t} -> {ACTS[sid]} s.2({num}): {text[:150]}")
    else:
        evaluate(f"t29_definitions_{args.set}", retrieve, args.set)


if __name__ == "__main__":
    main()
