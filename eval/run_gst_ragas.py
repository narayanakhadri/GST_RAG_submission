"""RAGAS eval harness for the GST golden question set.

Loads the golden set (seed/docs/gst/golden_set_with_gold_sources.jsonl) via
app/datasets/gst/golden_loader.py and reuses eval/ragas_adapter.py (scoring) and
eval/reporting.py (aggregate + table printing), both generic over row dicts.

forbidden_check is always reported as passed/not-applicable — the GST golden set has no
forbidden_keywords concept; the field only exists so eval/reporting.aggregate()/print_table()
can be reused as-is.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import tiktoken

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from app.services.rag_service import run_rag_with_trace_no_cache
from eval.gst_retrieval import (
    SOURCE_VERSIONS,
    corpus_keys,
    coverage,
    gold_sets,
    label_to_id_map,
    print_split,
    score_row,
    split_report,
)
from eval.profiles import PROFILES
from eval.ragas_adapter import judge_info
from eval.ragas_adapter import run as run_ragas
from eval.reporting import aggregate, print_table

SAMPLE_SEED = 42


_ENC = tiktoken.get_encoding("o200k_base")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAGAS eval against the GST golden question set")
    ap.add_argument("--golden", default=settings.gst_golden_path, help="Path to GST golden set JSONL")
    ap.add_argument(
        "--profile", default="hybrid+rerank+crag", choices=list(PROFILES.keys()),
        help="Retrieval flag profile (from eval/profiles.py)",
    )
    ap.add_argument("--sample", type=int, default=20, help="Number of answerable goldens to sample (0 = all)")
    ap.add_argument("--output", default=None, help="Output JSON path (default: eval/results/<timestamp>_gst_<profile>.json)")
    ap.add_argument(
        "--no-web-fallback", action="store_true",
        help="Disable Tavily web fallback for this run only (in-process; .env untouched). "
        "CRAG still grades relevance, but on low relevance keeps the originally retrieved "
        "chunks instead of calling Tavily — isolates pure RAG retrieval/generation quality.",
    )
    ap.add_argument(
        "--provider", choices=["openai", "groq", "gemini"], default=None,
        help="Override LLM provider (generation/CRAG/self-reflective/HyDE) for this run only "
        "(in-process; .env untouched). RAGAS's own judge LLM always stays on OpenAI regardless, "
        "so scores stay comparable across providers. Omit to use the .env default.",
    )
    ap.add_argument(
        "--gold-sources", choices=list(SOURCE_VERSIONS), default="original",
        help="Which gold-source annotation selects answerable questions, sets --coverage buckets and fills "
        "`golden_sources` in the output. 'original' = gold_primary_sources (what earlier runs used); "
        "'corrected' = primary_required_sources. Retrieval is scored against both either way.",
    )
    ap.add_argument(
        "--coverage", choices=["all", "full", "partial"], default="all",
        help="Only evaluate questions whose required law is fully / partly in the corpus (applied before --sample).",
    )
    ap.add_argument("--flags", default="{}", help="JSON of extra/overriding flags on top of --profile, "
                    "e.g. '{\"top_k\": 8, \"doc_type_cap\": 2, \"grounded_prompt\": true}'")
    ap.add_argument("--name", default=None, help="Label used in the output filename instead of the profile name")
    ap.add_argument("--collection", default=None, help="Qdrant collection for this run only")
    ap.add_argument("--embedding-model", default=None, help="Query embedding model (must match the collection)")
    ap.add_argument("--question-set", default=None, help="Named set from eval/results/thesis/question_sets.json "
                    "(scripts/thesis/make_question_sets.py), e.g. quick40, tier_hard")
    args = ap.parse_args()

    if args.collection:
        settings.qdrant_collection = args.collection
    if args.embedding_model:
        settings.embedding_model = args.embedding_model

    if args.no_web_fallback:
        settings.tavily_api_key = ""
    if args.provider:
        settings.llm_provider = args.provider

    corpus = load_corpus(settings.gst_corpus_path)
    corpus_ids = corpus_keys(corpus)
    label_to_id = label_to_id_map(corpus)

    flags = {**PROFILES[args.profile], **json.loads(args.flags)}
    # Questions with no primary source are abstention cases, not retrieval targets.
    goldens = [g for g in load_golden_set(args.golden) if gold_sets(g, args.gold_sources)[0]]
    if args.coverage != "all":
        goldens = [g for g in goldens if coverage(g, corpus_ids, args.gold_sources) == args.coverage]
    if args.question_set:
        from scripts.thesis.make_question_sets import load_question_set

        wanted = load_question_set(args.question_set)
        goldens = [g for g in goldens if g["id"] in wanted]

    if args.sample and args.sample < len(goldens):
        rng = random.Random(SAMPLE_SEED)
        goldens = rng.sample(goldens, args.sample)

    print(
        f"GST RAGAS eval | profile={args.profile} | provider={settings.llm_provider} | "
        f"gold-sources={args.gold_sources} | coverage={args.coverage} | {len(goldens)} goldens"
    )

    rows: list[dict] = []
    skipped: list[dict] = []

    for idx, g in enumerate(goldens, start=1):
        qid = g.get("id", "")
        question = g.get("question", "")
        print(f"  [{idx}/{len(goldens)}] {qid}")
        try:
            started = time.perf_counter()
            resp, chunks = run_rag_with_trace_no_cache(question, flags)
            seconds = round(time.perf_counter() - started, 2)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": qid, "reason": f"error: {exc}"})
            continue

        rows.append(
            {
                "id": qid,
                "demonstrates_feature": "gst",
                "intent": "rag",
                "question": question,
                "answer": resp.answer,
                "contexts": [c.text for c in chunks],
                "context_tokens": sum(len(_ENC.encode(c.text)) for c in chunks),  # what the LLM read
                "seconds": seconds,  # retrieval + generation
                "ground_truth": g.get("reference_answer") or g.get("reference_answer_raw") or "",
                "actual_sources": resp.sources,
                "retrieved_sources": [c.source for c in chunks],  # rank order; resp.sources is unordered
                "golden_sources": gold_sets(g, args.gold_sources)[0],
                "forbidden_keywords": [],
            }
        )
        score_row(rows[-1], g, corpus_ids, label_to_id)

    print("Scoring with RAGAS...")
    metrics = run_ragas(rows) if rows else []
    for row, m in zip(rows, metrics, strict=True):
        row["ragas_metrics"] = m
        row["forbidden_check"] = {"passed": True, "hits": []}

    timestamp = datetime.datetime.now(datetime.UTC)
    out_path = (
        Path(args.output)
        if args.output
        else Path(f"eval/results/{timestamp:%Y%m%dT%H%M%SZ}_gst_{args.name or args.profile}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profile": f"gst_{args.profile}",
        "flags": flags,
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "llm_provider": settings.llm_provider,
        "ragas_judge": judge_info(),
        "ragas_metrics": settings.ragas_metrics,
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "mode": "gst-service",
        "gold_sources": args.gold_sources,
        "coverage_filter": args.coverage,
        "question_set": args.question_set,
        "rows": rows,
        "skipped": skipped,
        "aggregate": aggregate(rows),
        "coverage_split": {v: split_report(rows, v) for v in SOURCE_VERSIONS},
    }

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print_table(payload)
    for v in SOURCE_VERSIONS:
        print_split(payload["coverage_split"][v], v)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
