"""Answers of the final system side by side with the ICAI reference, for the SME: one Markdown page per question set.

Merges any number of judged run files (RAGAS output, `<run>.judge-<model>.json`) and writes, per question: the
question, the law it needs vs what the final system retrieved, the ICAI reference answer, the earlier answer the
SME priority list was built from (`eval/results/20261016T191402Z_gst_hybrid.json`, before E17/T31), the final
answer, and the judge's scores. Groups: SME priority 1, dev_hard, test_hard (a question is listed once, first group).

    uv run python -m scripts.thesis.sme_answers_report eval/results/<a>.judge-X.json [<b>.judge-X.json ...] \
        [--out docs/final_answers_sme.md] [--app-json dataset/final_system_review/answers.json]

Also writes the same records as JSON for the SME app's "Final system answers" page (dataset/sme_review_app.py); the
app never edits that file — reviews go to `answers.sme_reviewed.json` beside it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SME_BASE = Path("dataset/sme_priority_review/questions_llm_base.json")
# The run the SME priority list was built from (hybrid, before E17/T31): earlier answer + earlier recall.
EARLIER = Path("eval/results/20261016T191402Z_gst_hybrid.json")
SETS = Path("eval/results/thesis/question_sets.json")
E17 = Path("eval/results/thesis/retrieval/e17_v3_textfix.json")
T31 = Path("eval/results/thesis/techniques/t31_pointwise_on_all.json")
FINAL = Path("eval/results/20261020T040741Z_gst_final_t31_all.json")  # final-system answers for all 146
METRICS = ("faithfulness", "answer_relevancy", "answer_correctness")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _recall(path: Path) -> dict[str, float]:
    return {r["id"]: r["retrieval"]["corrected"]["required_recall"] for r in _load(path)["rows"]}


def _f(v) -> str:
    return "—" if v is None or v != v else f"{v:.2f}"


def _quote(text: str) -> str:
    return "> " + (text or "(none)").strip().replace("\n", "\n> ")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("judged", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/final_answers_sme.md"))
    ap.add_argument("--app-json", type=Path, default=Path("dataset/final_system_review/answers.json"))
    args = ap.parse_args()

    rows, judges = {}, set()
    for p in args.judged:
        d = _load(p)
        judges.add(d["ragas_judge"]["model"])
        rows.update({r["id"]: r for r in d["rows"]})
    base = {x["id"]: x for x in _load(SME_BASE)}
    sets = _load(SETS)["sets"]
    groups = {"SME priority 1": [i for i, x in base.items() if x["priority"] == 1],
              "dev_hard": sets["dev_hard"], "test_hard": sets["test_hard"]}
    from scripts.techniques.common import questions

    # Required law = the corrected gold sources the recall numbers are computed against.
    required = {g["id"]: g.get("primary_required_sources") or [] for g in questions("all")}
    e17, t31, early = _recall(E17), _recall(T31), _recall(EARLIER)
    early_rows = {r["id"]: r for r in _load(EARLIER)["rows"]}

    seen, order = set(), []
    for g, ids in groups.items():
        order += [(g, i) for i in ids if i in rows and i not in seen]
        seen |= set(ids)
    n = len(order)

    def mean(ids, m):
        v = [rows[i]["ragas_metrics"][m] for i in ids if rows[i]["ragas_metrics"].get(m) == rows[i]["ragas_metrics"].get(m)
             and rows[i]["ragas_metrics"].get(m) is not None]
        return statistics.mean(v) if v else None

    out = [f"# Final system answers for SME review ({n} questions)", "",
           "**Final system:** E17 retrieval (hybrid search, LLM question decomposition) → T31 re-ranking (the LLM "
           "analyses and grades 20 candidates, keeps 8) → grounded answer by Gemini 3.1 flash-lite over the law text "
           "+ summary.", "",
           f"**Judge:** {', '.join(sorted(judges))} (RAGAS). Faithfulness = share of the answer's claims supported by "
           "the retrieved law; relevancy = does it address the question; correctness = agreement with the ICAI "
           "reference answer.", "",
           "**Recall** = share of the law the question needs that was retrieved: earlier system (the run the SME list "
           "was built from) → E17 → final system.", "",
           "**Reference answers state the law as on 30.04.2025; the corpus is the law as of April 2026** "
           "(`docs/law_changes_2025_2026.md`), so a few \"wrong\" answers may be right under the current law.", "",
           "## Summary", "", "| Group | n | Recall earlier → E17 → final | Faithfulness | Relevancy | Correctness |",
           "|---|---|---|---|---|---|"]
    for g in [*groups, "All"]:
        ids = [i for gg, i in order if gg == g or g == "All"]
        if not ids:
            continue
        rec = [statistics.mean([src[i] for i in ids if src.get(i) is not None] or [float("nan")]) for src in
               (early, e17, t31)]
        out.append(f"| {g} | {len(ids)} | {' → '.join(_f(r) for r in rec)} | "
                   + " | ".join(_f(mean(ids, m)) for m in METRICS) + " |")
    out += ["", "## Questions", "", "| # | Group | Question | Recall earlier → E17 → final | Faith. | Relev. | Correct. |",
            "|---|---|---|---|---|---|---|"]
    for k, (g, i) in enumerate(order, 1):
        s = rows[i]["ragas_metrics"]
        out.append(f"| {k} | {g} | [{i}](#q{k}) | {_f(early.get(i))} → {_f(e17.get(i))} → "
                   f"{_f(t31.get(i))} | " + " | ".join(_f(s.get(m)) for m in METRICS) + " |")
    for k, (g, i) in enumerate(order, 1):
        r, x = rows[i], base[i]
        s = r["ragas_metrics"]
        out += ["", f'<a id="q{k}"></a>', f"## {k}. {i} ({g})", "",
                f"**Scores:** faithfulness {_f(s.get('faithfulness'))}, relevancy {_f(s.get('answer_relevancy'))}, "
                f"correctness {_f(s.get('answer_correctness'))}; recall {_f(t31.get(i))}", "",
                "**Question.**", "", _quote(x["question"]), "",
                f"**Law the answer needs:** {', '.join(required.get(i, [])) or '—'}", "",
                f"**Retrieved by the final system:** {'; '.join(r['retrieved_sources'])}", "",
                "**Reference answer (ICAI).**", "", _quote(x["reference_answer"]), "",
                "**Earlier answer** (hybrid run, 2026-10-16).", "",
                _quote((early_rows.get(i) or {}).get("answer")), "", "**Final-system answer.**", "", _quote(r["answer"]), "",
                "**SME comments:** ", ""]
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    app = [{"id": i, "group": g, "question": base[i]["question"], "reference_answer": base[i]["reference_answer"],
            "required_sources": required.get(i, []), "retrieved_ids": rows[i].get("retrieved_ids") or [],
            "retrieved_sources": rows[i]["retrieved_sources"], "answer": rows[i]["answer"],
            "scores": {**{m: rows[i]["ragas_metrics"].get(m) for m in METRICS},
                       "required_recall": t31.get(i), "required_recall_e17": e17.get(i)}}
           for g, i in order]
    args.app_json.parent.mkdir(parents=True, exist_ok=True)
    args.app_json.write_text(json.dumps(app, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    # Every final-system answer (all 146) for the priority-review page; judge scores where this report has them.
    every = {r["id"]: {"answer": r["answer"], "retrieved_ids": r.get("retrieved_ids") or [],
                       "scores": ({m: rows[r["id"]]["ragas_metrics"].get(m) for m in METRICS} if r["id"] in rows else {})
                       | {"required_recall": t31.get(r["id"])}}
             for r in _load(FINAL)["rows"]}
    (args.app_json.parent / "all_answers.json").write_text(json.dumps(every, indent=1, ensure_ascii=False, default=str),
                                                           encoding="utf-8")
    print("\n".join(out[out.index("## Summary"):out.index("## Questions")]))
    print(f"wrote {args.out} and {args.app_json}")


if __name__ == "__main__":
    main()
