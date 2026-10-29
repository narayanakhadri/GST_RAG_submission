"""W1g — can retrieval confidence tell answerable from out-of-scope questions (abstention / scope detection)?

Positives: the golden questions with corrected required sources (146) (all answerable from the corpus). Negatives: 20 synthetic out-of-scope
questions (`eval/results/thesis/out_of_scope_questions.json`: income tax, customs, company law, labour law ...).
Signal: the best dense cosine score of the question against the corpus (3-large collection), the score a
system would threshold before answering. Reports AUC, the score ranges, and the threshold that misses the
fewest questions (chosen on all data, so it is optimistic — with 20 negatives this is an indication, not a tuned
system).

    uv run python -m scripts.thesis.abstention_check
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.datasets.gst.golden_loader import load_golden_set
from app.services.embedding_service import embed_texts
from app.services.vector_store import search

OOS = Path("eval/results/thesis/out_of_scope_questions.json")
GOLDEN = "seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl"
OUT = Path("eval/results/thesis/retrieval/w1g_abstention.json")


def top_score(question: str) -> float:
    hits = search(embed_texts([question])[0], top_k=1)
    return float(hits[0].score) if hits else 0.0


def main() -> None:
    settings.qdrant_collection = "gst_thesis_enriched_large"
    settings.embedding_model = "text-embedding-3-large"
    pos = [(g["id"], top_score(g["question"])) for g in load_golden_set(GOLDEN) if g["primary_required_sources"]]
    neg = [(q, top_score(q)) for q in json.loads(OOS.read_text(encoding="utf-8"))["questions"]]
    ps, ns = [s for _, s in pos], [s for _, s in neg]
    auc = sum((p > n) + 0.5 * (p == n) for p in ps for n in ns) / (len(ps) * len(ns))
    best = min(sorted(ps + ns), key=lambda t: sum(p < t for p in ps) + sum(n >= t for n in ns))
    report = {
        "n_answerable": len(ps), "n_out_of_scope": len(ns), "auc": round(auc, 3),
        "answerable_score_range": [round(min(ps), 3), round(max(ps), 3)],
        "out_of_scope_score_range": [round(min(ns), 3), round(max(ns), 3)],
        "best_threshold": round(best, 3),
        "answerable_wrongly_refused": sum(p < best for p in ps),
        "out_of_scope_wrongly_answered": sum(n >= best for n in ns),
        "out_of_scope_scores": [{"question": q, "score": round(s, 3)} for q, s in neg],
        "answerable_scores": [{"id": i, "score": round(s, 3)} for i, s in pos],
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_scores")}, indent=1))


if __name__ == "__main__":
    main()
