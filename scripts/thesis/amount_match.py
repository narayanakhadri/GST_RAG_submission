"""Final-amount check: does the answer reach the amounts the ICAI reference answer computes? No LLM, free.

Why: the SME rejected numerical answers whose final amount was off (a missed provision shifts the computation), and the
Qwen judge did not notice (correctness 0.74 on rejected answers vs 0.71 on the rest). This checks the numbers directly.

How. Amounts = numbers in the reference answer that are not in the question (so they were computed), >= 100, ₹ lakh/crore
expanded, section/rule/notification numbers and years removed. Per question:
- amount_recall: share of those amounts that appear in the answer (exact, ±1 for rounding);
- amount_recall_5pct: same, but an answer amount within 5% counts ("off by a few per cent");
- last_amount_hit: the reference's last computed amount (usually the final result) appears in the answer.
Only questions whose reference computes at least one amount are scored.

    uv run python -m scripts.thesis.amount_match                       # final system, SME agreement, T34 dev_hard
    uv run python -m scripts.thesis.amount_match --a A.json --b B.json  # any two answer runs, paired
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from eval.stats import HEADER, format_row, paired

GOLDEN = Path("seed/docs/gst/golden_set_sme_v1.jsonl")
FINAL = Path("eval/results/20261020T040741Z_gst_final_t31_all.json")
T34 = Path("eval/results/20261021T140444Z_gst_t34_missing_law_dev_hard.json")
VERDICTS = Path("dataset/sme_answer_verdicts_from_notes.json")
OUT = Path("eval/results/thesis/amount_match.json")

REFERENCE = re.compile(
    r"\b(?:sections?|rules?|notification(?:\s+no\.?)?|circular(?:\s+no\.?)?|para(?:graph)?|schedule|entry|"
    r"sl\.?\s*no\.?|s\.\s*no\.?|item|chapter|heading|tariff)\s*[\d/().A-Za-z-]*",
    re.I,
)
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|crores?)?", re.I)
SCALE = {"lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7}


def amounts(text: str) -> list[float]:
    """Amounts in order of appearance (lakh/crore expanded; references, years and small numbers dropped)."""
    out = []
    for m in NUMBER.finditer(REFERENCE.sub(" ", text or "")):
        raw = m.group(1).rstrip(",.")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if m.group(2):
            v *= SCALE[m.group(2).lower()]
        if v < 100 or (v.is_integer() and 2000 <= v <= 2035 and "," not in raw):
            continue
        out.append(v)
    return out


def found(v: float, pool: set[float], tol: float) -> bool:
    return any(abs(v - p) <= max(1.0, tol * v) for p in pool)


def score(question: str, reference: str, answer: str) -> dict | None:
    given = set(amounts(question))
    computed = [v for v in dict.fromkeys(amounts(reference)) if not found(v, given, 0)]
    if not computed:
        return None
    pool = set(amounts(answer))
    return {
        "n_amounts": len(computed),
        "amount_recall": mean(found(v, pool, 0) for v in computed),
        "amount_recall_5pct": mean(found(v, pool, 0.05) for v in computed),
        "last_amount_hit": float(found(computed[-1], pool, 0)),
    }


def rows(path: Path) -> dict[str, str]:
    return {r["id"]: r.get("answer") or "" for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}


METRICS = ["amount_recall", "amount_recall_5pct", "last_amount_hit"]


def load_golden() -> dict[str, dict]:
    return {g["id"]: g for g in (json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip())}


def scored(golden: dict, answers: dict[str, str]) -> dict[str, dict]:
    return {q: s for q, a in answers.items() if q in golden
            and (s := score(golden[q]["question"], golden[q].get("reference_answer") or "", a))}


def compare(path_a: Path, path_b: Path) -> dict:
    """Paired amount metrics, B vs A, on the questions both runs answered that have computed amounts."""
    golden = load_golden()
    sa, sb = scored(golden, rows(path_a)), scored(golden, rows(path_b))
    common = sorted(set(sa) & set(sb))
    # Guard: a run with a different LLM provider retrieves different law (cache misses) — make that visible.
    ra, rb = ({r["id"]: r.get("retrieved_ids") for r in json.loads(p.read_text(encoding="utf-8"))["rows"]}
              for p in (path_a, path_b))
    same = sum(set((ra[q] or [])[:8]) == set((rb[q] or [])[:8]) for q in common)
    print(f"{path_b.name} vs {path_a.name} ({len(common)} questions with computed amounts; "
          f"same top-8 retrieved law in {same}):")
    print(HEADER)
    out = {"a": str(path_a), "b": str(path_b), "per_question": {"a": sa, "b": sb}}
    for m in METRICS:
        p = paired(*([{"id": q, "ragas_metrics": s[q]} for q in common] for s in (sa, sb)), m, coverage=None)
        out[m] = p
        print(format_row("A", "B", p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", type=Path, help="baseline answer run")
    ap.add_argument("--b", type=Path, help="compared answer run")
    ap.add_argument("--out", type=Path, help="where to write the comparison (default: next to amount_match.json)")
    args = ap.parse_args()
    if args.a and args.b:
        out = args.out or OUT.with_name(f"amount_match_{args.b.stem.split('_gst_')[-1]}_vs_{args.a.stem.split('_gst_')[-1]}.json")
        out.write_text(json.dumps(compare(args.a, args.b), indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
        return
    golden = load_golden()
    final, t34 = rows(FINAL), rows(T34)
    scores = {"final": scored(golden, final), "t34": scored(golden, t34)}
    metrics = METRICS
    report: dict = {"scored": {k: len(v) for k, v in scores.items()}}

    print(f"Questions with computed amounts: final {len(scores['final'])} of {len(final)}, "
          f"T34 {len(scores['t34'])} of {len(t34)}\n")
    print("Final system, all scored questions:")
    for m in metrics:
        print(f"  {m:20s} {mean(s[m] for s in scores['final'].values()):.3f}")
        report.setdefault("final_all", {})[m] = mean(s[m] for s in scores["final"].values())

    # Validity: does the check agree with the SME? (final system answers, SME page 0)
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    wrong = {v["id"] for v in verdicts if v["sme_answer_verdict"] in ("incorrect", "partly incorrect")}
    sme = [v["id"] for v in verdicts if v["id"] in scores["final"]]
    print(f"\nAgreement with the SME ({len(sme)} reviewed questions with computed amounts):")
    for label, ids in (("SME: wrong", [q for q in sme if q in wrong]), ("SME: right", [q for q in sme if q not in wrong])):
        vals = {m: mean(scores["final"][q][m] for q in ids) if ids else float("nan") for m in metrics}
        report.setdefault("sme_agreement", {})[label] = {"n": len(ids), **vals}
        print(f"  {label} (n={len(ids):2d}) " + "  ".join(f"{m} {vals[m]:.3f}" for m in metrics))

    common = sorted(set(scores["final"]) & set(scores["t34"]))
    print(f"\nT34 vs final system, dev_hard ({len(common)} questions with computed amounts):")
    print(HEADER)
    for m in metrics:
        a, b = ([{"id": q, "ragas_metrics": scores[k][q]} for q in common] for k in ("final", "t34"))
        p = paired(a, b, m, coverage=None)
        report.setdefault("t34_vs_final", {})[m] = p
        print(format_row("final", "T34", p))
    report["per_question"] = {k: v for k, v in scores.items()}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
