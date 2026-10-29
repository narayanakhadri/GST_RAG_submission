"""Retrieval scoring and corpus-coverage split for the GST eval (eval/run_gst_ragas.py).

RAGAS only sees question/answer/contexts/reference answer, so it never checks whether the *right
provisions* were retrieved. This module adds that check against the golden set's gold sources, and
buckets each question by how much of the law it needs is actually in the corpus, so pipeline quality
can be separated from corpus gaps.

Each golden record carries two generations of sources (see CLAUDE.md, "SME review workspace"):
  original  — gold_primary_sources / gold_supporting_sources / missing_gold_sources
  corrected — primary_required_sources / supporting_sources / external_sources_missing_from_corpus
Retrieval is scored against both on every run so the two can be compared.

A gold source is either a provision ID ("cgst_act_2017.section.16") or a whole instrument
("notification_12_2017_ct_rate") when the golden answer cites a notification/circular without naming the
entry; an instrument-level gold source is hit by any retrieved provision of that instrument.

Can also re-score an existing results file without re-running RAG:
    uv run python -m eval.gst_retrieval eval/results/<file>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import source_label
from app.datasets.gst.golden_loader import load_golden_set
from eval.reporting import METRIC_KEYS, aggregate

SOURCE_VERSIONS = ("original", "corrected")
COVERAGE_BUCKETS = ("full", "partial", "none")

_FIELDS = {
    "original": ("gold_primary_sources", "gold_supporting_sources", "missing_gold_sources"),
    "corrected": ("primary_required_sources", "supporting_sources", "external_sources_missing_from_corpus"),
}

RETRIEVAL_KEYS = ("required_recall", "all_required_hit", "any_required_hit", "required_mrr", "supporting_recall")


def gold_sets(golden: dict[str, Any], version: str) -> tuple[list[str], list[str], list[str]]:
    """(primary, supporting, missing-from-corpus) source lists for one annotation version."""
    primary_f, supporting_f, missing_f = _FIELDS[version]
    return golden.get(primary_f) or [], golden.get(supporting_f) or [], golden.get(missing_f) or []


def corpus_keys(corpus: list[dict[str, Any]]) -> set[str]:
    """Every ID a gold source may use to name something in the corpus: provision IDs plus instrument IDs."""
    ids = {r["id"] for r in corpus}
    return ids | {r.get("instrument_id") or r["id"].split(".", 1)[0] for r in corpus}


def source_matches(gold: str, retrieved_id: str) -> bool:
    """A provision matches itself; an instrument-level gold ID matches any provision of that instrument."""
    return retrieved_id == gold or retrieved_id.startswith(gold + ".")


def coverage(golden: dict[str, Any], corpus_ids: set[str], version: str) -> str:
    """'full' — every law the answer needs is in the corpus; 'partial' — some required law is missing;
    'none' — no required source exists in the corpus at all. `corpus_ids` should come from corpus_keys()."""
    primary, _, missing = gold_sets(golden, version)
    if not any(s in corpus_ids for s in primary):
        return "none"
    return "partial" if any(s not in corpus_ids for s in missing) else "full"


def label_to_id_map(corpus: list[dict[str, Any]]) -> dict[str, str]:
    return {source_label(r): r["id"] for r in corpus}


def retrieved_ids(sources: list[str], label_to_id: dict[str, str]) -> list[str]:
    """Map retrieved source labels to corpus IDs in rank order, de-duplicated. Labels that aren't corpus
    provisions (e.g. CRAG web-fallback URLs) are dropped."""
    out: list[str] = []
    for s in sources:
        pid = label_to_id.get(s)
        if pid and pid not in out:
            out.append(pid)
    return out


def retrieval_metrics(retrieved: list[str], primary: list[str], supporting: list[str]) -> dict[str, float | None]:
    """Retrieval quality against gold sources. Values are None when there is nothing to score against.

    required_recall   — fraction of primary sources retrieved
    all_required_hit  — 1.0 if every primary source was retrieved
    any_required_hit  — 1.0 if at least one primary source was retrieved (hit@k)
    required_mrr      — 1 / rank of the first primary source retrieved (0.0 if none)
    supporting_recall — fraction of supporting sources retrieved
    """
    def found(gold: str) -> bool:
        return any(source_matches(gold, r) for r in retrieved)

    out: dict[str, float | None] = dict.fromkeys(RETRIEVAL_KEYS)
    if primary:
        hits = [s for s in primary if found(s)]
        out["required_recall"] = len(hits) / len(primary)
        out["all_required_hit"] = float(len(hits) == len(primary))
        out["any_required_hit"] = float(bool(hits))
        ranks = [i for i, r in enumerate(retrieved, start=1) if any(source_matches(s, r) for s in primary)]
        out["required_mrr"] = 1 / ranks[0] if ranks else 0.0
    if supporting:
        out["supporting_recall"] = sum(found(s) for s in supporting) / len(supporting)
    return out


def score_row(row: dict[str, Any], golden: dict[str, Any], corpus_ids: set[str], label_to_id: dict[str, str]) -> None:
    """Add `retrieved_ids`, `coverage` and `retrieval` (per annotation version) to an eval row in place.

    Uses `retrieved_sources` (chunk sources in rank order) when present. Older results files only have
    `actual_sources`, which rag_service builds from a set — membership is right but rank order is lost,
    so required_mrr is unreliable for those (flagged via `rank_order_known`).
    """
    ranked = "retrieved_sources" in row
    ids = retrieved_ids(row["retrieved_sources"] if ranked else row.get("actual_sources") or [], label_to_id)
    row["retrieved_ids"] = ids
    row["rank_order_known"] = ranked
    row["coverage"] = {v: coverage(golden, corpus_ids, v) for v in SOURCE_VERSIONS}
    row["retrieval"] = {}
    for v in SOURCE_VERSIONS:
        primary, supporting, _ = gold_sets(golden, v)
        row["retrieval"][v] = retrieval_metrics(ids, primary, supporting)


def _mean(vals: list[Any]) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return round(mean(vals), 3) if vals else None


def split_report(rows: list[dict[str, Any]], version: str) -> dict[str, dict[str, Any]]:
    """RAGAS + retrieval aggregates per coverage bucket (plus 'all') for one annotation version."""
    scored = [r for r in rows if "retrieval" in r]
    buckets = {"all": scored} | {b: [r for r in scored if r["coverage"][version] == b] for b in COVERAGE_BUCKETS}
    report: dict[str, dict[str, Any]] = {}
    for name, bucket in buckets.items():
        ragas = aggregate(bucket) if bucket else dict.fromkeys(METRIC_KEYS)
        report[name] = {
            "n": len(bucket),
            **{k: ragas.get(k) for k in METRIC_KEYS},
            **{k: _mean([r["retrieval"][version][k] for r in bucket]) for k in RETRIEVAL_KEYS},
        }
    return report


def print_split(report: dict[str, dict[str, Any]], version: str) -> None:
    def fmt(v: Any) -> str:
        return "—" if v is None else f"{v:.2f}"

    print(f"\n## Retrieval + coverage split — gold sources: {version}")
    print("| coverage | n | req_recall | all_req | hit@k | MRR | supp_recall | faith | ctx_prec | ctx_recall | ans_rel |")
    print("|----------|---|------------|---------|-------|-----|-------------|-------|----------|------------|---------|")
    for name, m in report.items():
        print(
            f"| {name} | {m['n']} | {fmt(m['required_recall'])} | {fmt(m['all_required_hit'])} | "
            f"{fmt(m['any_required_hit'])} | {fmt(m['required_mrr'])} | {fmt(m['supporting_recall'])} | "
            f"{fmt(m['faithfulness'])} | {fmt(m['context_precision'])} | {fmt(m['context_recall'])} | "
            f"{fmt(m['answer_relevancy'])} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Add retrieval scores + coverage split to an existing GST results file")
    ap.add_argument("results", help="Path to an eval/results/*_gst_*.json file")
    ap.add_argument("--golden", default=settings.gst_golden_path)
    ap.add_argument("--corpus", default=settings.gst_corpus_path)
    ap.add_argument("--write", action="store_true", help="Write the scores back into the results file")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    corpus_ids = corpus_keys(corpus)
    label_to_id = label_to_id_map(corpus)
    goldens = {g["id"]: g for g in load_golden_set(args.golden)}

    path = Path(args.results)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [r["id"] for r in payload["rows"] if r["id"] not in goldens]
    for row in payload["rows"]:
        if row["id"] in goldens:
            score_row(row, goldens[row["id"]], corpus_ids, label_to_id)
    if missing:
        print(f"Skipped {len(missing)} row(s) whose id is not in the golden set: {missing[:5]}")

    payload["coverage_split"] = {v: split_report(payload["rows"], v) for v in SOURCE_VERSIONS}
    for v in SOURCE_VERSIONS:
        print_split(payload["coverage_split"][v], v)
    if not all(r.get("rank_order_known") for r in payload["rows"] if "retrieval" in r):
        print("\nNote: this file predates `retrieved_sources`, so rank order is unknown — ignore MRR.")

    if args.write:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nUpdated: {path}")


if __name__ == "__main__":
    main()
