"""Thesis ablation grid: run retrieval-only experiments (eval/gst_retrieval_check.run_check) and write one
JSON per configuration to eval/results/thesis/retrieval/<name>.json plus a summary table.

    uv run python -m scripts.thesis.run_ablation --group baseline
    uv run python -m scripts.thesis.run_ablation --only hybrid_bm25 hybrid_graph
    uv run python -m scripts.thesis.run_ablation --summary          # rebuild the table from saved JSONs

Groups that call an LLM (query rewriting, HyDE, LLM reranking) use settings.llm_provider — set
LLM_PROVIDER=gemini to use the free rotating keys. Existing results are skipped unless --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.gst_retrieval_check import run_check
from eval.profiles import PROFILES

OUT_DIR = Path("eval/results/thesis/retrieval")
SUMMARY = Path("eval/results/thesis/retrieval_summary.md")

HYBRID = PROFILES["hybrid"]
RERANK = {**HYBRID, "enable_rerank": True}
LARGE = {"collection": "gst_thesis_enriched_large", "settings_overrides": {"embedding_model": "text-embedding-3-large"}}
E17 = {**HYBRID, "top_k": 8, "doc_type_cap": 3, "query_rewrite": "decompose"}
GOLDEN_TEXTFIX = "seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl"
E17_TEXTFIX = {**LARGE, "golden_path": GOLDEN_TEXTFIX}
# v4 corpus; golden = textfix + citations of the new documents mapped to their IDs (update_golden_sources.py)
V4_CORPUS = "seed/docs/gst/gst_provisions_enriched_v4.jsonl"
LARGE_V4 = {"collection": "gst_thesis_enriched_large_v4", "corpus_path": V4_CORPUS,
            "golden_path": "seed/docs/gst/golden_set_v4.jsonl",
            "settings_overrides": {"embedding_model": "text-embedding-3-large", "gst_corpus_path": V4_CORPUS}}

# name -> (group, research question, description, flags, extra run_check kwargs)
CONFIGS: dict[str, tuple[str, str, str, dict, dict]] = {
    # RQ3: retrieval method
    "dense": ("baseline", "RQ3", "Dense only (text-embedding-3-small)", PROFILES["naive"], {}),
    "sparse_tfidf": ("baseline", "RQ3", "Sparse only, TF-IDF", PROFILES["sparse_only"], {}),
    "sparse_bm25": ("baseline", "RQ3", "Sparse only, BM25", {**PROFILES["sparse_only"], "sparse_scorer": "bm25"}, {}),
    "hybrid_tfidf": ("baseline", "RQ3", "Hybrid dense + TF-IDF (RRF) — system default", HYBRID, {}),
    "hybrid_bm25": ("baseline", "RQ3", "Hybrid dense + BM25 (RRF)", {**HYBRID, "sparse_scorer": "bm25"}, {}),
    "hybrid_top8": ("baseline", "RQ3", "Hybrid TF-IDF, top_k = 8", {**HYBRID, "top_k": 8}, {}),
    # RQ1: document structure — doc-type balance and graph expansion
    "hybrid_cap1": ("structure", "RQ1", "Hybrid, max 1 notification/circular in top 5", {**HYBRID, "doc_type_cap": 1}, {}),
    "hybrid_cap2": ("structure", "RQ1", "Hybrid, max 2 notifications/circulars in top 5", {**HYBRID, "doc_type_cap": 2}, {}),
    "hybrid_cap3": ("structure", "RQ1", "Hybrid, max 3 notifications/circulars in top 5", {**HYBRID, "doc_type_cap": 3}, {}),
    "hybrid_inst2": ("structure", "RQ1", "Hybrid, max 2 items per instrument (diversity)", {**HYBRID, "instrument_cap": 2}, {}),
    "hybrid_cap2_inst2": ("structure", "RQ1", "Hybrid, cap 2 notifications/circulars + max 2 per instrument",
                          {**HYBRID, "doc_type_cap": 2, "instrument_cap": 2}, {}),
    "hybrid_graph": ("structure", "RQ1/RQ2", "Hybrid + graph expansion (cross-reference neighbours)", {**HYBRID, "graph_expand": True}, {}),
    "hybrid_graph_cap2": ("structure", "RQ1/RQ2", "Hybrid + graph expansion + cap 2", {**HYBRID, "graph_expand": True, "doc_type_cap": 2}, {}),
    "hybrid_graphdense": ("structure", "RQ1/RQ2", "Hybrid + graph neighbours ranked by query similarity (w 0.5)",
                          {**HYBRID, "graph_expand": "dense"}, {}),
    "hybrid_graphdense_w1": ("structure", "RQ1/RQ2", "Hybrid + graph neighbours ranked by query similarity (w 1.0)",
                             {**HYBRID, "graph_expand": "dense", "graph_weight": 1.0}, {}),
    "hybrid_graphdense_cap2": ("structure", "RQ1/RQ2", "Hybrid + similarity-ranked graph neighbours (w 1.0) + cap 2",
                               {**HYBRID, "graph_expand": "dense", "graph_weight": 1.0, "doc_type_cap": 2}, {}),
    "hybrid_top8_cap2": ("combo", "RQ1/RQ3", "Hybrid, top_k 8, max 2 notifications/circulars",
                         {**HYBRID, "top_k": 8, "doc_type_cap": 2}, {}),
    "hybrid_top8_cap3": ("combo", "RQ1/RQ3", "Hybrid, top_k 8, max 3 notifications/circulars",
                         {**HYBRID, "top_k": 8, "doc_type_cap": 3}, {}),
    "hybrid_top8_cap4": ("combo", "RQ1/RQ3", "Hybrid, top_k 8, max 4 notifications/circulars",
                         {**HYBRID, "top_k": 8, "doc_type_cap": 4}, {}),
    "large_cap2": ("combo_large", "RQ1/RQ3", "3-large, hybrid, max 2 notifications/circulars", {**HYBRID, "doc_type_cap": 2}, LARGE),
    "large_top8": ("combo_large", "RQ3", "3-large, hybrid, top_k 8", {**HYBRID, "top_k": 8}, LARGE),
    "large_top8_cap3": ("combo_large", "RQ1/RQ3", "3-large, hybrid, top_k 8, max 3 notifications/circulars",
                        {**HYBRID, "top_k": 8, "doc_type_cap": 3}, LARGE),
    "large_dense": ("combo_large", "RQ3", "3-large, dense only", PROFILES["naive"], LARGE),
    # E17: best stack + LLM decomposition (best single technique on the baseline)
    "large_top8_cap3_decompose": ("combo_large", "RQ3", "Best stack (3-large, top_k 8, cap 3) + LLM decomposition, RRF",
                                  {**HYBRID, "top_k": 8, "doc_type_cap": 3, "query_rewrite": "decompose"}, LARGE),
    # E17 again on the fixed golden text (22 questions reworded), then on the v4 corpus (+ 3 notifications,
    # 3 CBIC FAQs) with its gold sources mapped to the new IDs. Same flags as E17; only data changes.
    "e17_v3_textfix": ("v4", "RQ3", "E17 on v3, golden text fixed", E17, {**LARGE, "golden_path": GOLDEN_TEXTFIX}),
    "e17_v4": ("v4", "RQ1", "E17 on v4 corpus (+ notifications 11/2017, 41/2017, 52/2018, CBIC FAQs)", E17, LARGE_V4),
    # E13 on top of E17 (v4): (a) Codex-corrected summaries in the embedded text (collection from
    # build_collections.py); (c) expand over Codex typed links, neighbours ordered by similarity to the question.
    "e13a_codexsum": ("e13", "RQ2", "E17 v4 + Codex-corrected summaries embedded", E17,
                      {**LARGE_V4, "collection": "gst_thesis_codexsum_large_v4"}),
    "e13c_links3": ("e13", "RQ1", "E17 v4 + typed links (procedure_in, rate_or_exemption_for, clarifies)",
                    {**E17, "graph_expand": "dense", "graph_links": ["procedure_in", "rate_or_exemption_for", "clarifies"]},
                    LARGE_V4),
    "e13c_links2": ("e13", "RQ1", "E17 v4 + typed links (procedure_in, clarifies)",
                    {**E17, "graph_expand": "dense", "graph_links": ["procedure_in", "clarifies"]}, LARGE_V4),
    # Wave 1 (v3 plan): retrieval quick wins on top of e17_v3_textfix (E17 flags, v3, golden text fixed).
    "w1a_citation": ("wave1", "RQ3", "E17 + named provisions looked up by ID and put first",
                     {**E17, "citation_lookup": True}, E17_TEXTFIX),
    "w1b_adaptive_k": ("wave1", "RQ3", "E17 + top_k by number of sub-queries (2→6, 3→7, 4→12; mean 8.4)",
                       {**E17, "adaptive_top_k": {"2": 6, "3": 7, "4": 12}}, E17_TEXTFIX),
    "w1b_fixed_k9": ("wave1", "RQ3", "Control for w1b: E17 with top_k 9", {**E17, "top_k": 9}, E17_TEXTFIX),
    "w1c_stepback": ("wave1", "RQ3", "E17 + step-back query", {**E17, "stepback": True}, E17_TEXTFIX),
    "w1d_dense03": ("wave1", "RQ3", "E17, RRF weight dense 0.3 : sparse 0.7", {**E17, "rrf_dense_weight": 0.3},
                    E17_TEXTFIX),
    "w1d_dense07": ("wave1", "RQ3", "E17, RRF weight dense 0.7 : sparse 0.3", {**E17, "rrf_dense_weight": 0.7},
                    E17_TEXTFIX),
    "w1e_mmr07": ("wave1", "RQ1", "E17 + MMR diversity λ 0.7", {**E17, "mmr_lambda": 0.7}, E17_TEXTFIX),
    "w1e_mmr05": ("wave1", "RQ1", "E17 + MMR diversity λ 0.5", {**E17, "mmr_lambda": 0.5}, E17_TEXTFIX),
    # W1f: learned sparse vectors (fastembed; collections from build_sparse_collections.py) in place of TF-IDF
    "w1f_splade": ("wave1f", "RQ3", "E17 with SPLADE++ instead of TF-IDF", {**E17, "sparse_scorer": "splade"},
                   E17_TEXTFIX),
    "w1f_bm42": ("wave1f", "RQ3", "E17 with BM42 instead of TF-IDF", {**E17, "sparse_scorer": "bm42"}, E17_TEXTFIX),
    "w1f_minicoil": ("wave1f", "RQ3", "E17 with miniCOIL instead of TF-IDF", {**E17, "sparse_scorer": "minicoil"},
                     E17_TEXTFIX),
    "w1f_bm25": ("wave1f", "RQ3", "E17 with BM25 instead of TF-IDF (classic reference)", {**E17, "sparse_scorer": "bm25"},
                 E17_TEXTFIX),
    # Combinations of the wave-1 winners and LLM re-ranking on top of E17 (not yet run).
    "c2_bm25_citation": ("combo2", "RQ3", "E17 + BM25 + cited-provision lookup",
                         {**E17, "sparse_scorer": "bm25", "citation_lookup": True}, E17_TEXTFIX),
    "c2_minicoil_citation": ("combo2", "RQ3", "E17 + miniCOIL + cited-provision lookup",
                             {**E17, "sparse_scorer": "minicoil", "citation_lookup": True}, E17_TEXTFIX),
    "c2_llm_rerank": ("combo2", "RQ3", "E17 + LLM listwise re-rank of the pool (RankGPT-style, Gemini)",
                      {**E17, "enable_rerank": True, "reranker_backend": "llm"}, E17_TEXTFIX),
    # E13b doc2query: Codex search terms + 'answers questions' appended (collection from build_collections.py)
    "e13b_doc2query": ("combo2", "RQ2", "E17 + Codex search terms and questions in the embedded text",
                       E17, {**E17_TEXTFIX, "collection": "gst_thesis_doc2query_large"}),
    "hybrid_pool20": ("diagnostic", "—", "Hybrid top 20 (upper bound for any reranker over the pool)",
                      {**HYBRID, "top_k": 20}, {}),
    # RQ3: rerankers
    "rerank_minilm": ("rerank", "RQ3", "Hybrid + MiniLM-L6 cross-encoder (512 tokens)", RERANK, {}),
    "rerank_bge_m3": ("rerank_slow", "RQ3", "Hybrid + bge-reranker-v2-m3 (8k tokens)",
                      {**RERANK, "reranker_model": "BAAI/bge-reranker-v2-m3"}, {}),
    "rerank_llm": ("llm", "RQ3", "Hybrid + LLM listwise rerank (Gemini flash-lite)", {**RERANK, "reranker_backend": "llm"}, {}),
    # RQ3: query understanding (LLM)
    "hyde": ("llm", "RQ3", "HyDE (3 hypothetical answers, dense)", {**PROFILES["naive"], "enable_hyde": True}, {}),
    "rewrite": ("llm", "RQ3", "LLM rewrites question into a legal-issue query (hybrid)", {**HYBRID, "query_rewrite": "rewrite"}, {}),
    "decompose": ("llm", "RQ3", "LLM decomposes into sub-queries + original, RRF (hybrid)", {**HYBRID, "query_rewrite": "decompose"}, {}),
    # RQ2: what text is embedded (needs the extra collections from scripts/thesis/build_collections.py)
    "embed_raw": ("embedding", "RQ2", "Embed raw provision text only (hybrid)", HYBRID,
                  {"collection": "gst_thesis_raw_small"}),
    "embed_exact": ("embedding", "RQ2", "Embed hierarchy prefix + text, no LLM metadata (hybrid)", HYBRID,
                    {"collection": "gst_thesis_exact_small"}),
    "embed_large": ("embedding", "RQ3", "Enriched text, text-embedding-3-large (hybrid)", HYBRID,
                    {"collection": "gst_thesis_enriched_large",
                     "settings_overrides": {"embedding_model": "text-embedding-3-large"}}),
}

METRICS = ("required_recall", "all_required_hit", "any_required_hit", "required_mrr")


def config_hash(flags: dict, extra: dict) -> str:
    return hashlib.sha256(json.dumps([flags, extra], sort_keys=True).encode()).hexdigest()[:12]


def run(name: str, force: bool) -> None:
    group, rq, description, flags, extra = CONFIGS[name]
    out = OUT_DIR / f"{name}.json"
    if out.exists() and not force:
        # Skip only if the saved result was produced with these exact flags; code changes need --force.
        if json.loads(out.read_text(encoding="utf-8")).get("config_hash") == config_hash(flags, extra):
            print(f"skip {name} (exists, same flags; --force to rerun after a code change)")
            return
        print(f"rerun {name}: flags changed since the saved result")
    print(f"== {name}: {description}")
    payload = run_check(flags, verbose=False, **extra)
    payload.update({"name": name, "group": group, "rq": rq, "description": description,
                    "config_hash": config_hash(flags, extra)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    full = payload["coverage_split"]["corrected"]["full"]
    print(f"   {name}: " + " ".join(f"{m}={full.get(m)}" for m in METRICS) + f" ({payload['seconds']} s)")


def summary() -> str:
    lines = ["| Config | RQ | Description | n | Required law found | All required | hit@k | MRR | Act share retrieved | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        path = OUT_DIR / f"{name}.json"
        if not path.exists():
            continue
        p = json.loads(path.read_text(encoding="utf-8"))
        full = p["coverage_split"]["corrected"]["full"]
        vals = " | ".join("—" if full.get(m) is None else f"{full[m]:.3f}" for m in METRICS)
        act = p.get("doc_kind_share", {}).get("retrieved", {}).get("act")
        lines.append(f"| {name} | {p['rq']} | {p['description']} | {full.get('n')} | {vals} | "
                     f"{'—' if act is None else f'{act:.2f}'} | {p['seconds']} |")
    table = "\n".join(lines)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("# Retrieval ablation (corrected gold sources, fully covered questions)\n\n" + table + "\n",
                       encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", nargs="*", default=[], help=f"groups: {sorted({c[0] for c in CONFIGS.values()})}")
    ap.add_argument("--only", nargs="*", default=[], help="config names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary table")
    args = ap.parse_args()

    if not args.summary:
        names = [n for n, c in CONFIGS.items() if n in args.only or c[0] in args.group]
        for name in names:
            run(name, args.force)
    print(summary())


if __name__ == "__main__":
    main()
