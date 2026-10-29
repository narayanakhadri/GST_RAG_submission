"""Retrieval-only GST eval: run the real retrieval pipeline over the golden questions and score which
provisions come back, split by corpus coverage — no answer generation, no RAGAS, no CRAG LLM grading.

Cheap enough to run on all 146 questions (only question embeddings, mostly cached), so it is the quick
way to compare corpora or collections, e.g. before/after ingesting new law:

    uv run python -m eval.gst_retrieval_check --collection gst_provisions_enriched_v2 \
        --corpus seed/docs/gst/build/gst_provisions_enriched_groq.before_first_rebuild.jsonl \
        --golden seed/docs/gst/golden_set_with_gold_sources.jsonl.bak
    uv run python -m eval.gst_retrieval_check                      # current settings
    uv run python -m eval.gst_retrieval_check --profile hybrid --flags '{"sparse_scorer": "bm25"}'

CRAG is always switched off here (it needs an LLM per question and can swap in web results).
`run_check()` is the reusable entry point (scripts/thesis/run_ablation.py runs a grid of configs with it).
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from collections import Counter
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from eval.gst_retrieval import (
    SOURCE_VERSIONS,
    corpus_keys,
    gold_sets,
    label_to_id_map,
    print_split,
    score_row,
    split_report,
)
from eval.profiles import PROFILES


def doc_kind(provision_id: str | None) -> str:
    """act / rules / notification / circular, from a corpus ID."""
    if not provision_id:
        return "unknown"
    head = provision_id.split(".", 1)[0]
    if head.startswith(("notification", "circular")):
        return head.split("_", 1)[0]
    return "rules" if "rules" in head else "act"


def doc_kind_shares(rows: list[dict], key: str) -> dict[str, float]:
    counts = Counter(doc_kind(i) for r in rows for i in r.get(key) or [])
    total = sum(counts.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def run_check(
    flags: dict,
    *,
    collection: str | None = None,
    corpus_path: str | None = None,
    golden_path: str | None = None,
    gold_sources: str = "corrected",
    settings_overrides: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run retrieval for every answerable golden question under `flags` and return the scored payload.

    `collection` and `settings_overrides` (e.g. {"embedding_model": "text-embedding-3-large"}) apply to this
    run only; the previous settings are restored afterwards, so a grid of runs in one process stays independent.
    """
    overrides = {**(settings_overrides or {}), **({"qdrant_collection": collection} if collection else {})}
    saved = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    try:
        return _run(flags, corpus_path, golden_path, gold_sources, verbose)
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


def _run(flags: dict, corpus_path: str | None, golden_path: str | None, gold_sources: str, verbose: bool) -> dict:
    from app.services.rag_service import _retrieve

    flags = {**flags, "enable_crag": False}
    corpus_path = corpus_path or settings.gst_corpus_path
    golden_path = golden_path or settings.gst_golden_path
    corpus = load_corpus(corpus_path)
    keys, label_to_id = corpus_keys(corpus), label_to_id_map(corpus)
    goldens = [g for g in load_golden_set(golden_path) if gold_sets(g, gold_sources)[0]]

    rows, started = [], time.perf_counter()
    for i, g in enumerate(goldens, start=1):
        chunks = _retrieve(g["question"], flags)
        row = {"id": g["id"], "retrieved_sources": [c.source for c in chunks]}
        score_row(row, g, keys, label_to_id)
        row["gold_required_ids"] = gold_sets(g, gold_sources)[0]
        rows.append(row)
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(goldens)}")

    return {
        "collection": settings.qdrant_collection,
        "corpus": corpus_path,
        "golden": golden_path,
        "flags": flags,
        "gold_sources": gold_sources,
        "seconds": round(time.perf_counter() - started, 1),
        "doc_kind_share": {
            "retrieved": doc_kind_shares(rows, "retrieved_ids"),
            "gold_required": doc_kind_shares(rows, "gold_required_ids"),
        },
        "rows": rows,
        "coverage_split": {v: split_report(rows, v) for v in SOURCE_VERSIONS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="hybrid+rerank", choices=list(PROFILES))
    ap.add_argument("--flags", default="{}", help="JSON of extra/overriding retrieval flags")
    ap.add_argument("--collection", default=None, help="Qdrant collection (default: settings.qdrant_collection)")
    ap.add_argument("--corpus", default=settings.gst_corpus_path, help="Corpus the collection was built from")
    ap.add_argument("--golden", default=settings.gst_golden_path)
    ap.add_argument("--gold-sources", choices=list(SOURCE_VERSIONS), default="corrected",
                    help="Which annotation selects the answerable questions")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    flags = {**PROFILES[args.profile], **json.loads(args.flags)}
    print(f"Retrieval check | collection={args.collection or settings.qdrant_collection} | profile={args.profile} "
          f"(CRAG off) | flags={json.loads(args.flags)} | gold={args.gold_sources}")
    payload = run_check(flags, collection=args.collection, corpus_path=args.corpus, golden_path=args.golden,
                        gold_sources=args.gold_sources)
    payload["profile"] = args.profile

    ts = datetime.datetime.now(datetime.UTC)
    out = Path(args.output or f"eval/results/{ts:%Y%m%dT%H%M%SZ}_gst_retrieval_{args.profile}_{settings.qdrant_collection}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for v in SOURCE_VERSIONS:
        print_split(payload["coverage_split"][v], v)
    print(f"\nDoc-type share: {payload['doc_kind_share']}\nWrote: {out}  ({payload['seconds']} s)")


if __name__ == "__main__":
    main()
