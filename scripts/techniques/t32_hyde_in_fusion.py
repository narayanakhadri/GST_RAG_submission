"""T32 — HyDE as one more query inside the decomposition fusion (HyDE + RAG-Fusion).

Idea. HyDE (Gao et al. 2022) embeds a *hypothetical answer* instead of the question: answers look like the law
("Under section 17(5)(a) of the CGST Act, input tax credit is not available on motor vehicles …"), questions do
not. Alone it scored 0.485 on the baseline (vs decomposition 0.522), and E17 never used it. Here the hypothetical
answer is added as an extra query next to the original question and the decomposition sub-queries, all fused by
RRF (Query2doc, Wang et al. 2023, is the same idea for sparse retrieval: the pseudo-document expands the query).

How it works here (free: one Gemini call per question for the pseudo-answer, cached by app/services/hyde.py if
available, else here).
- Queries: [question, *decomposition, hypothetical answer (≈ 120 words)] → hybrid search each (top 20) → RRF →
  cap 3 → top 8. `--hyde-weight` scales the HyDE ranking in the fusion (default 1).

    uv run python -m scripts.techniques.t32_hyde_in_fusion eval --set dev_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.techniques.common import evaluate, use_best_stack

CACHE = Path("eval/results/thesis/cache/hyde_answers.json")
PROMPT = ("Write a short (about 120 words) answer to this Indian GST law question in the style of the law and an "
          "exam model answer, naming the provisions that apply (section/rule/notification). It is used only as a "
          "search query, so do not hedge.")


def pseudo_answer(question: str) -> str:
    from app.services.llm_service import generate, grader_model_default

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{PROMPT}|{question}".encode()).hexdigest()
    if key not in cache:
        try:
            cache[key] = generate(PROMPT, question)["text"].strip()
        except Exception:  # noqa: BLE001
            cache[key] = ""
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache[key]


def retrieve(question: str, hyde_weight: float) -> list:
    from app.config import settings
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.query_rewrite import rewrite_queries
    from app.services.rag_service import _search_once
    from app.services.sparse_vector_service import fuse_rrf

    queries = [question, *rewrite_queries(question, "decompose")]
    rankings = [_search_once(q, "hybrid", settings.reranker_initial_top_k, None) for q in queries]
    weights = [1.0] * len(rankings)
    hyde = pseudo_answer(question)
    if hyde:
        rankings.append(_search_once(hyde, "hybrid", settings.reranker_initial_top_k, None))
        weights.append(hyde_weight)
    return cap_doc_types(fuse_rrf(rankings, weights=weights), get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--hyde-weight", type=float, default=1.0)
    args = ap.parse_args()
    use_best_stack()
    evaluate(f"t32_hyde_fusion_w{args.hyde_weight}_{args.set}", lambda q: retrieve(q, args.hyde_weight), args.set)


if __name__ == "__main__":
    main()
