"""Paired comparison of retrieval runs against a reference, on all questions and on dev_hard / test_hard.

    uv run python -m scripts.thesis.compare_runs e17_v3_textfix w1a_citation w1b_adaptive_k ...
    uv run python -m scripts.thesis.compare_runs e17_v3_textfix --glob "w1*"

Runs are looked up in eval/results/thesis/retrieval/ and eval/results/thesis/techniques/ (by name or file stem).
Also prints the mean number of retrieved provisions (top-k actually used).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.stats import HEADER, format_row, paired

DIRS = [Path("eval/results/thesis/retrieval"), Path("eval/results/thesis/techniques")]
SETS = json.loads(Path("eval/results/thesis/question_sets.json").read_text(encoding="utf-8"))["sets"]


def load(name: str) -> list[dict]:
    for d in DIRS:
        p = d / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))["rows"]
    raise SystemExit(f"no run named {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--metrics", nargs="*", default=["required_recall", "required_mrr"])
    args = ap.parse_args()
    names = list(args.runs)
    if args.glob:
        names += sorted({p.stem for d in DIRS for p in d.glob(f"{args.glob}.json")} - {args.reference})
    ref = load(args.reference)
    for subset in ("all", "dev_hard", "test_hard"):
        ids = None if subset == "all" else set(SETS[subset])
        a = [r for r in ref if ids is None or r["id"] in ids]
        print(f"\n### {subset}\n\n{HEADER}")
        for name in names:
            b = load(name)
            for m in args.metrics:
                print(format_row(args.reference, name, paired(a, b, m, coverage=None)))
    print("\nMean provisions retrieved:")
    for name in [args.reference, *names]:
        rows = load(name)
        print(f"  {name}: {sum(len(r.get('retrieved_ids') or r.get('retrieved_sources') or []) for r in rows) / len(rows):.2f}")


if __name__ == "__main__":
    main()
