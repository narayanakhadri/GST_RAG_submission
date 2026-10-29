"""T30 — Implicit legal-concept expansion of the keyword query (LLM infers the statute's vocabulary).

Idea. A fact pattern ("the company gives its director a phone free of cost") never uses the words of the law that
governs it ("supply without consideration", "related person", "Schedule I"). Nguyen et al. (arXiv 2410.12154,
"Exploiting LLMs' Reasoning Capability to Infer Implicit Concepts in Legal Information Retrieval") let an LLM infer
the legal terms and facts implicit in the situation, add them to a term-based (BM25) query and to reformulated
queries, and ensemble — beating the best COLIEE 2022 and 2023 statute-retrieval systems. E17's decomposition
writes short search queries; this adds an explicit *keyword list for the sparse side*.

How it works here (free: one Gemini call per question, cached).
1. Expansion prompt → JSON {"terms": [8–15 statutory terms / phrases], "provisions": [likely provisions]}.
2. Sparse search (BM25 by default, `--scorer`) with question + terms; dense search with the question only.
3. Fusion: E17 pool (weight 1) + expanded sparse ranking (weight `--weight`, default 0.7) by RRF → cap 3 → top 8.

    uv run python -m scripts.techniques.t30_legal_concept_expansion expand "<question>"
    uv run python -m scripts.techniques.t30_legal_concept_expansion eval --set dev_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

CACHE = Path("eval/results/thesis/cache/legal_concepts.json")
PROMPT = ("You are an Indian GST law expert. For the question, infer the legal concepts that the governing statute "
          "would use, even if the question does not say them: statutory terms and phrases (e.g. 'supply without "
          "consideration', 'related person', 'blocked credit', 'place of supply', 'time of supply', 'composite "
          "supply'), and the provisions likely to apply. Return JSON {\"terms\": [8-15 short terms], "
          "\"provisions\": [up to 5, e.g. 'section 17(5) CGST Act']}.")


def expand(question: str) -> dict:
    from app.services.llm_service import generate_with_json, grader_model_default

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{PROMPT}|{question}".encode()).hexdigest()
    if key not in cache:
        try:
            data = json.loads(generate_with_json(PROMPT, question)["text"])
        except Exception:  # noqa: BLE001
            data = {}
        cache[key] = {"terms": [t for t in data.get("terms") or [] if isinstance(t, str)][:15],
                      "provisions": [p for p in data.get("provisions") or [] if isinstance(p, str)][:5]}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache[key]


def retrieve(question: str, scorer: str, weight: float) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve
    from app.services.sparse_vector_service import fuse_rrf
    from app.services.vector_store import sparse_search

    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    ex = expand(question)
    expanded = sparse_search(question + " " + " ".join(ex["terms"] + ex["provisions"]), top_k=20, scorer=scorer)
    return cap_doc_types(fuse_rrf([pool, expanded], weights=[1.0, weight]), get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["expand", "eval"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--scorer", default="bm25")
    ap.add_argument("--weight", type=float, default=0.7)
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "expand":
        print(json.dumps(expand(args.question), indent=1))
    else:
        evaluate(f"t30_concepts_{args.scorer}_w{args.weight}_{args.set}",
                 lambda q: retrieve(q, args.scorer, args.weight), args.set)


if __name__ == "__main__":
    main()
