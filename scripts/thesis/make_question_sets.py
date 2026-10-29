"""Split the 141 fully covered golden questions into difficulty tiers and fixed question sets.

    uv run python -m scripts.thesis.make_question_sets

Two independent notions of difficulty, both written to eval/results/thesis/question_sets.json:

- **Intrinsic tier** (`easy` / `medium` / `hard` / `very_hard`) — from the question and its gold law only, before
  any system is run, so it is fair for comparing systems: points for how many provisions the answer needs, how
  many instruments they span, whether a notification/circular is needed, and a long scenario. Points are cut
  into four tiers of roughly equal size.
- **Observed tier** — how much of the required law the retrieval configurations actually found, averaged
  over every non-diagnostic config in scripts/thesis/run_ablation.py. It describes where systems fail (error
  analysis), but it is defined by the systems under test, so don't use it to pick a winner.

Named sets (lists of question ids) for runs on a subset: `all`, `tier_<name>`, `observed_<name>`, and `quick40`,
a fixed 40-question set with 10 per intrinsic tier, for cheap RAGAS experiments. Use with
`eval.run_gst_ragas --question-set quick40` or `eval.rescore_ragas --question-set tier_hard`.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from app.services.gst_graph import SUBORDINATE_KINDS, kind
from eval.gst_retrieval import corpus_keys, coverage, gold_sets
from scripts.thesis.run_ablation import CONFIGS, OUT_DIR

QUESTION_SETS = Path("eval/results/thesis/question_sets.json")
TIERS = ("easy", "medium", "hard", "very_hard")
SEED = 7
QUICK_PER_TIER = 10


def load_question_set(name: str) -> set[str]:
    """Question ids of a named set from question_sets.json."""
    sets = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))["sets"]
    if name not in sets:
        raise SystemExit(f"unknown question set {name!r}; choose from {sorted(sets)}")
    return set(sets[name])


def features(golden: dict, kind_of: dict[str, str], instrument_of: dict[str, str]) -> dict:
    required = gold_sets(golden, "corrected")[0]
    kinds = sorted({kind_of.get(s, "unknown") for s in required})
    instruments = sorted({instrument_of.get(s, s) for s in required})
    n, words = len(required), len(golden["question"].split())
    points = (
        (0 if n == 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 7 else 4)
        + (len(instruments) >= 2) + (len(instruments) >= 3)
        + any(k in SUBORDINATE_KINDS for k in kinds)
        + (words > 150)
    )
    return {"n_required": n, "n_instruments": len(instruments), "kinds": kinds,
            "needs_notification_or_circular": any(k in SUBORDINATE_KINDS for k in kinds),
            "question_words": words, "points": int(points)}


def observed_recall() -> dict[str, float]:
    per_q: dict[str, list[float]] = {}
    for name, (group, *_rest) in CONFIGS.items():
        path = OUT_DIR / f"{name}.json"
        if group == "diagnostic" or not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            v = (row.get("retrieval") or {}).get("corrected", {}).get("required_recall")
            if v is not None:
                per_q.setdefault(row["id"], []).append(v)
    return {q: mean(v) for q, v in per_q.items()}


def observed_tier(recall: float) -> str:
    return "easy" if recall >= 0.75 else "medium" if recall >= 0.4 else "hard" if recall >= 0.1 else "very_hard"


def main() -> None:
    corpus = load_corpus(settings.gst_corpus_path)
    keys = corpus_keys(corpus)
    kind_of = {r["id"]: kind(r) for r in corpus}
    instrument_of = {r["id"]: r["instrument_id"] for r in corpus}
    for r in corpus:  # a gold source may name a whole instrument
        kind_of.setdefault(r["instrument_id"], kind(r))
        instrument_of.setdefault(r["instrument_id"], r["instrument_id"])

    goldens = [g for g in load_golden_set(settings.gst_golden_path)
               if gold_sets(g, "corrected")[0] and coverage(g, keys, "corrected") == "full"]
    feats = {g["id"]: features(g, kind_of, instrument_of) for g in goldens}

    # Rank by points (ties: more required law, then longer question) and cut into four equal-sized tiers.
    order = sorted(feats, key=lambda q: (feats[q]["points"], feats[q]["n_required"], feats[q]["question_words"], q))
    tier = {q: TIERS[min(i * len(TIERS) // len(order), len(TIERS) - 1)] for i, q in enumerate(order)}
    recall = observed_recall()
    obs = {q: observed_tier(recall[q]) for q in feats if q in recall}

    rng = random.Random(SEED)
    quick = sorted(q for t in TIERS for q in rng.sample(sorted(q for q in tier if tier[q] == t), QUICK_PER_TIER))
    sets = {"all": sorted(feats), "quick40": quick}
    sets |= {f"tier_{t}": sorted(q for q in tier if tier[q] == t) for t in TIERS}
    sets |= {f"observed_{t}": sorted(q for q in obs if obs[q] == t) for t in TIERS}

    questions = {q: {**feats[q], "tier": tier[q], "observed_tier": obs.get(q),
                     "observed_recall": round(recall[q], 3) if q in recall else None} for q in order}
    QUESTION_SETS.write_text(json.dumps({
        "description": __doc__.split("\n\n")[1],
        "n_configs_for_observed": len({n for n, c in CONFIGS.items() if c[0] != "diagnostic" and (OUT_DIR / f"{n}.json").exists()}),
        "sets": sets, "questions": questions}, indent=2), encoding="utf-8")

    print(f"Wrote {QUESTION_SETS} ({len(feats)} questions)\n")
    print("| Tier | n | required law (mean) | needs notif./circular | ≥2 instruments | mean observed recall |")
    print("|---|---|---|---|---|---|")
    for t in TIERS:
        qs = [questions[q] for q in sets[f"tier_{t}"]]
        rec = [q["observed_recall"] for q in qs if q["observed_recall"] is not None]
        print(f"| {t} | {len(qs)} | {mean(q['n_required'] for q in qs):.1f} | "
              f"{sum(q['needs_notification_or_circular'] for q in qs)} | {sum(q['n_instruments'] >= 2 for q in qs)} | "
              f"{mean(rec):.3f} |")
    print("\nIntrinsic tier × observed tier:")
    cross = Counter((questions[q]["tier"], questions[q]["observed_tier"]) for q in questions)
    print("| intrinsic \\ observed | " + " | ".join(TIERS) + " |\n|---|" + "---|" * len(TIERS))
    for t in TIERS:
        print(f"| {t} | " + " | ".join(str(cross[(t, o)]) for o in TIERS) + " |")


if __name__ == "__main__":
    main()
