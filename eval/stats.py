"""Paired significance tests between two eval runs over the same questions.

For each metric: mean of A, mean of B, mean difference B − A with a 95% paired-bootstrap confidence
interval, and a two-sided Wilcoxon signed-rank p-value (ties/zero differences handled by scipy's "zsplit").
Questions are aligned by id; questions missing a value in either run are dropped for that metric.

    uv run python -m eval.stats eval/results/thesis/retrieval/hybrid_tfidf.json eval/results/thesis/retrieval/hybrid_cap2.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from scipy.stats import wilcoxon

RETRIEVAL_METRICS = ("required_recall", "all_required_hit", "any_required_hit", "required_mrr")
RAGAS_METRICS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
BOOTSTRAP_SAMPLES = 10_000
SEED = 7


def metric_value(row: dict, metric: str, gold_version: str = "corrected") -> float | None:
    if metric in RETRIEVAL_METRICS:
        v = (row.get("retrieval") or {}).get(gold_version, {}).get(metric)
    else:
        v = (row.get("ragas_metrics") or {}).get(metric)
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    return None


def paired(rows_a: list[dict], rows_b: list[dict], metric: str, coverage: str | None = "full") -> dict:
    """Paired comparison of one metric. `coverage` restricts to a corrected-coverage bucket (None = all)."""
    b_by_id = {r["id"]: r for r in rows_b}
    pairs = []
    for a in rows_a:
        b = b_by_id.get(a["id"])
        if b is None or (coverage and (a.get("coverage") or {}).get("corrected") != coverage):
            continue
        va, vb = metric_value(a, metric), metric_value(b, metric)
        if va is not None and vb is not None:
            pairs.append((va, vb))
    if not pairs:
        return {"metric": metric, "n": 0}

    diffs = [vb - va for va, vb in pairs]
    rng = random.Random(SEED)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(BOOTSTRAP_SAMPLES))
    p = None
    if any(d != 0 for d in diffs):
        p = float(wilcoxon(diffs, zero_method="zsplit").pvalue)
    return {
        "metric": metric,
        "n": len(pairs),
        "mean_a": sum(va for va, _ in pairs) / len(pairs),
        "mean_b": sum(vb for _, vb in pairs) / len(pairs),
        "diff": sum(diffs) / len(diffs),
        "ci_low": boots[int(0.025 * BOOTSTRAP_SAMPLES)],
        "ci_high": boots[int(0.975 * BOOTSTRAP_SAMPLES) - 1],
        "p_wilcoxon": p,
        "b_better": sum(d > 0 for d in diffs),
        "a_better": sum(d < 0 for d in diffs),
    }


def format_row(name_a: str, name_b: str, r: dict) -> str:
    if not r.get("n"):
        return f"| {name_b} vs {name_a} | {r['metric']} | 0 | — | — | — | — | — |"
    p = "—" if r["p_wilcoxon"] is None else (f"{r['p_wilcoxon']:.3f}" if r["p_wilcoxon"] >= 0.001 else "<0.001")
    star = " *" if r["p_wilcoxon"] is not None and r["p_wilcoxon"] < 0.05 else ""
    return (f"| {name_b} vs {name_a} | {r['metric']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {p}{star} | {r['b_better']}/{r['a_better']} |")


HEADER = ("| Comparison | Metric | n | A | B | B − A [95% CI] | Wilcoxon p | B better / A better |\n"
          "|---|---|---|---|---|---|---|---|")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metrics", nargs="*", default=list(RETRIEVAL_METRICS))
    ap.add_argument("--coverage", default="full", help="corrected coverage bucket, or 'all'")
    args = ap.parse_args()
    ra = json.loads(Path(args.a).read_text(encoding="utf-8"))["rows"]
    rb = json.loads(Path(args.b).read_text(encoding="utf-8"))["rows"]
    print(HEADER)
    for m in args.metrics:
        print(format_row(Path(args.a).stem, Path(args.b).stem,
                         paired(ra, rb, m, None if args.coverage == "all" else args.coverage)))


if __name__ == "__main__":
    main()
