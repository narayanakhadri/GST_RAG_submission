"""Re-score saved retrieval runs against the SME-reviewed golden set (no retrieval, no LLM: free).

    uv run python -m scripts.thesis.sme_rescore

Every v3 run in eval/results/thesis/{retrieval,techniques}/ is re-scored with `golden_set_sme_v1.jsonl`
(`scripts/thesis/apply_sme_review.py`) and written to eval/results/thesis/sme_v1/<dir>/<name>.json; the originals are
not touched. The gold *sources* of the original and textfix golden files are identical, so runs made with either can be
re-scored. v4 runs are skipped (their golden maps two citations to v4-only IDs). Before writing, each file is re-scored
with the golden it was made with and must reproduce its stored metrics (guards against scoring drift).
Prints the headline table (old vs SME gold) and the paired tests of the headline comparisons on the SME gold.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from statistics import mean

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from eval.gst_retrieval import (
    SOURCE_VERSIONS,
    corpus_keys,
    label_to_id_map,
    score_row,
    split_report,
)
from eval.stats import HEADER, format_row, paired

SME_GOLDEN = "seed/docs/gst/golden_set_sme_v1.jsonl"
DIRS = [Path("eval/results/thesis/retrieval"), Path("eval/results/thesis/techniques")]
OUT = Path("eval/results/thesis/sme_v1")
HEADLINE = [("hybrid_tfidf", "baseline (hybrid, top-5)"), ("e17_v3_textfix", "E17, golden text fixed"),
            ("t31_pointwise_on_all", "Final: E17 + T31"), ("e17_pool20_all", "pool of 20 (re-ranking ceiling)")]
METRICS = ("required_recall", "all_required_hit", "required_mrr")


def summary(rows: list[dict]) -> dict:
    full = [r for r in rows if (r.get("coverage") or {}).get("corrected") == "full"]
    out = {"n": len(full)}
    for m in METRICS:
        v = [r["retrieval"]["corrected"].get(m) for r in full]
        v = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
        out[m] = mean(v) if v else None
    return out


def main() -> None:
    corpus = load_corpus(settings.gst_corpus_path)
    corpus_ids, label_to_id = corpus_keys(corpus), label_to_id_map(corpus)
    sme = {g["id"]: g for g in load_golden_set(SME_GOLDEN)}
    old_goldens: dict[str, dict] = {}
    results: dict[str, tuple[dict, dict, list]] = {}
    skipped = []

    for d in DIRS:
        for path in sorted(d.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "rows" not in payload or "retrieval" not in (payload["rows"] or [{}])[0]:
                skipped.append(f"{path.name} (not a retrieval run)")
                continue
            golden_path = payload.get("golden") or (payload.get("settings") or {}).get("golden")
            if "v4" in str(golden_path) or "v4" in str(payload.get("corpus")):
                skipped.append(f"{path.name} (v4)")
                continue
            if golden_path not in old_goldens:
                old_goldens[golden_path] = {g["id"]: g for g in load_golden_set(golden_path)}
            check = copy.deepcopy(payload["rows"])
            for row in check:
                score_row(row, old_goldens[golden_path][row["id"]], corpus_ids, label_to_id)
            before, again = summary(payload["rows"]), summary(check)
            if any(abs((before[m] or 0) - (again[m] or 0)) > 1e-9 for m in METRICS):
                skipped.append(f"{path.name} (stored scores not reproduced: {before} vs {again})")
                continue
            for row in payload["rows"]:
                score_row(row, sme[row["id"]], corpus_ids, label_to_id)
            payload["golden"] = SME_GOLDEN
            payload["coverage_split"] = {v: split_report(payload["rows"], v) for v in SOURCE_VERSIONS}
            out = OUT / d.name / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            results[path.stem] = (before, summary(payload["rows"]), payload["rows"])

    print(f"Re-scored {len(results)} runs -> {OUT}/; skipped {len(skipped)}:")
    for s in skipped:
        print("  -", s)

    print("\n| Run | n (old → SME) | Recall old → SME | All-required hit old → SME | MRR old → SME |\n|---|---|---|---|---|")
    for name, label in HEADLINE:
        b, a, _ = results[name]
        print(f"| {label} | {b['n']} → {a['n']} | " + " | ".join(
            f"{b[m]:.3f} → {a[m]:.3f}" for m in METRICS) + " |")

    print("\nPaired tests on the SME gold (full-coverage questions)\n" + HEADER)
    for (a, _), (b, _) in [(HEADLINE[0], HEADLINE[1]), (HEADLINE[1], HEADLINE[2]), (HEADLINE[0], HEADLINE[2])]:
        for m in ("required_recall", "required_mrr"):
            print(format_row(a, b, paired(results[a][2], results[b][2], m)))

    changes = []
    for name, (b, a, _) in results.items():
        changes.append((abs(a["required_recall"] - b["required_recall"]), name, b["required_recall"], a["required_recall"]))
    print("\nLargest recall shifts across all re-scored runs:")
    for diff, name, b, a in sorted(changes, reverse=True)[:5]:
        print(f"  {name}: {b:.3f} → {a:.3f}")
    (OUT / "summary.json").write_text(json.dumps({n: {"old": b, "sme": a} for n, (b, a, _) in results.items()},
                                                 indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
