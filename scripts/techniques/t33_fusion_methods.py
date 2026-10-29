"""T33 — Rank-fusion methods compared: RRF vs CombSUM vs CombMNZ vs Borda (fusing E17's sub-query rankings).

Idea. E17 fuses the rankings of the question and its sub-queries with Reciprocal Rank Fusion (Cormack et al. 2009:
score = Σ 1/(k + rank)), which ignores the raw scores. Classic alternatives (Fox & Shaw 1994; Montague & Aslam
2001): CombSUM (sum of min-max-normalised scores), CombMNZ (CombSUM × number of lists that contain the item —
rewards agreement), Borda count (Σ (N − rank)). Which one to use is a standard design question; RRF is popular
because it needs no score calibration (dense cosine and TF-IDF scores live on different scales).

How it works here (free: decompositions cached; no new LLM calls).
- Rankings: hybrid search for the question and each sub-query (top 20 each), as in E17.
- Fusion by `--method` (rrf | combsum | combmnz | borda) → cap 3 → top 8. `rrf` reproduces E17 (sanity check).

    uv run python -m scripts.techniques.t33_fusion_methods eval --method combmnz --set all
"""

from __future__ import annotations

import argparse

from app.models import RetrievedChunk
from scripts.techniques.common import evaluate, use_best_stack


def fuse(rankings: list[list[RetrievedChunk]], method: str, rrf_k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    keep: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        if not ranking:
            continue
        raw = [c.score for c in ranking]
        lo, hi = min(raw), max(raw)
        n = len(ranking)
        for rank, c in enumerate(ranking):
            keep.setdefault(c.text, c)
            hits[c.text] = hits.get(c.text, 0) + 1
            if method == "rrf":
                s = 1.0 / (rrf_k + rank + 1)
            elif method in ("combsum", "combmnz"):
                s = (c.score - lo) / (hi - lo) if hi > lo else 1.0
            elif method == "borda":
                s = float(n - rank)
            else:
                raise ValueError(method)
            scores[c.text] = scores.get(c.text, 0.0) + s
    if method == "combmnz":
        scores = {t: s * hits[t] for t, s in scores.items()}
    order = sorted(scores, key=scores.get, reverse=True)
    return [RetrievedChunk(text=t, source=keep[t].source, score=scores[t]) for t in order]


def retrieve(question: str, method: str) -> list:
    from app.config import settings
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.query_rewrite import rewrite_queries
    from app.services.rag_service import _search_once

    queries = [question, *rewrite_queries(question, "decompose")]
    rankings = [_search_once(q, "hybrid", settings.reranker_initial_top_k, None) for q in queries]
    return cap_doc_types(fuse(rankings, method)[: settings.reranker_initial_top_k], get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--method", choices=["rrf", "combsum", "combmnz", "borda"], default="combmnz")
    ap.add_argument("--set", default="all")
    args = ap.parse_args()
    use_best_stack()
    evaluate(f"t33_fusion_{args.method}_{args.set}", lambda q: retrieve(q, args.method), args.set)


if __name__ == "__main__":
    main()
