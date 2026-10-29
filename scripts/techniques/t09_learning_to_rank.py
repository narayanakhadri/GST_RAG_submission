"""T09 — Learning-to-rank over the candidate pool (Gokhan & Briscoe, "Grounded Answers from Multi-Passage
Regulations: Learning-to-Rank for Regulatory RAG", NLLP 2025).

Idea. Fusion (RRF) adds up rank positions blindly. A small supervised model can learn which signals predict a
required provision: appearing in several sub-query rankings, a high dense similarity, being an Act section rather
than a rate entry, being linked to other strong candidates, being named in the question. Gokhan & Briscoe combine
lexical, semantic and graph-derived features in a feature-based LTR model and beat dense, hybrid and fusion
baselines on multi-passage regulatory QA (ObliQA-MP).

How it works here (free after the first pass; Gemini decompositions are cached).
1. Candidates = E17's pool: the question + its decomposition sub-queries, each searched hybrid top-20, fused (RRF).
2. Features per (question, candidate):
   rrf score, rrf rank, n sub-queries that found it, best rank in any ranking, rank for the original question,
   dense cosine (3-large) to the question, kind one-hot (act/rules/notification/circular), named in the question
   (citation_lookup), typed-link degree to the other candidates (Codex links), log text length.
3. Label = candidate is a required gold source (corrected annotation, instrument-level match allowed).
4. Train on the 76 questions that are in neither dev_hard nor test_hard (gold of dev/test never seen); model =
   gradient-boosted trees (sklearn HistGradientBoosting, pointwise) — LambdaMART-style listwise training is a
   drop-in alternative (LightGBM `lambdarank`).
5. Re-rank the pool by the model's probability, then the usual doc-type cap 3 → top 8; evaluate on dev/test_hard.

    uv run python -m scripts.techniques.t09_learning_to_rank features      # build + cache the feature table
    uv run python -m scripts.techniques.t09_learning_to_rank eval --set dev_hard
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from eval.gst_retrieval import gold_sets, source_matches
from scripts.techniques.common import E17_FLAGS, QUESTION_SETS, evaluate, questions, use_best_stack

CACHE = Path("eval/results/thesis/cache/ltr_features.json")
KINDS = ("act", "rules", "notification", "circular", "faq")
FEATURES = ["rrf", "rrf_rank", "n_queries", "best_rank", "orig_rank", "dense", *[f"kind_{k}" for k in KINDS],
            "cited", "link_degree", "log_len"]


def candidates(question: str) -> list[dict]:
    """E17 pool with per-candidate features (ids, not chunks)."""
    from app.services.citation_lookup import cited_ids
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph, get_typed_graph, kind
    from app.services.query_rewrite import rewrite_queries
    from app.services.rag_service import _search_once
    from app.services.sparse_vector_service import fuse_rrf
    from app.services.vector_store import provision_vectors

    graph = get_graph()
    links = get_typed_graph(("procedure_in", "rate_or_exemption_for", "clarifies", "empowered_by", "refers_to"))
    queries = [question, *rewrite_queries(question, "decompose")]
    rankings = [_search_once(q, "hybrid", settings.reranker_initial_top_k, None) for q in queries]
    fused = fuse_rrf(rankings)[: settings.reranker_initial_top_k]
    ids = [graph.id_of(c) for c in fused]
    rank_in = [{graph.id_of(c): i for i, c in enumerate(r)} for r in rankings]
    qv = embed_texts([question])[0]
    vecs = provision_vectors([i for i in ids if i])
    qn = math.sqrt(sum(x * x for x in qv))
    named = set(cited_ids(question, set(graph.records), {r["instrument_id"] for r in graph.records.values()}))
    pool = set(ids)
    out = []
    for rank, (c, pid) in enumerate(zip(fused, ids, strict=True)):
        if not pid:
            continue
        rec = graph.records[pid]
        v = vecs.get(pid)
        dense = sum(a * b for a, b in zip(qv, v, strict=True)) / (qn * math.sqrt(sum(b * b for b in v))) if v else 0.0
        ranks = [r[pid] for r in rank_in if pid in r]
        k = kind(rec)
        out.append({"id": pid, "rrf": c.score, "rrf_rank": rank, "n_queries": len(ranks),
                    "best_rank": min(ranks) if ranks else 99, "orig_rank": rank_in[0].get(pid, 99), "dense": dense,
                    **{f"kind_{x}": float(k == x) for x in KINDS},
                    "cited": float(pid in named or rec["instrument_id"] in named),
                    "link_degree": len(links.neighbours.get(pid, set()) & pool),
                    "log_len": math.log1p(len(rec.get("raw_text") or ""))})
    return out


def build_features() -> dict:
    use_best_stack()
    table = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for i, g in enumerate(questions("all"), start=1):
        if g["id"] in table:
            continue
        required = gold_sets(g, "corrected")[0]
        cands = candidates(g["question"])
        for c in cands:
            c["label"] = int(any(source_matches(s, c["id"]) for s in required))
        table[g["id"]] = cands
        if i % 10 == 0:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(table), encoding="utf-8")
            print(f"  {i} questions")
    CACHE.write_text(json.dumps(table), encoding="utf-8")
    return table


def train(table: dict):
    from sklearn.ensemble import HistGradientBoostingClassifier

    sets = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))["sets"]
    held_out = set(sets["dev_hard"]) | set(sets["test_hard"])
    X, y = [], []
    for qid, cands in table.items():
        if qid in held_out:
            continue
        for c in cands:
            X.append([c[f] for f in FEATURES])
            y.append(c["label"])
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=7)
    model.fit(X, y)
    print(f"trained on {len({q for q in table if q not in held_out})} questions, {len(y)} candidates, "
          f"{sum(y)} positives")
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["features", "eval"])
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    table = build_features()
    if args.cmd == "features":
        return
    model = train(table)
    from app.services.gst_graph import cap_doc_types, get_graph
    from scripts.techniques.common import chunks_for_ids

    by_question = {g["question"]: g["id"] for g in questions("all")}
    corpus_ids = {r["id"] for r in load_corpus(settings.gst_corpus_path)}

    def retrieve(question: str) -> list:
        cands = table[by_question[question]]
        probs = model.predict_proba([[c[f] for f in FEATURES] for c in cands])[:, 1]
        order = [c["id"] for _, c in sorted(zip(probs, cands, strict=True), key=lambda t: -t[0]) if c["id"] in corpus_ids]
        return cap_doc_types(chunks_for_ids(order), get_graph(), E17_FLAGS["top_k"], E17_FLAGS["doc_type_cap"])

    evaluate(f"t09_ltr_{args.set}", retrieve, args.set, extra={"features": FEATURES})


if __name__ == "__main__":
    main()
