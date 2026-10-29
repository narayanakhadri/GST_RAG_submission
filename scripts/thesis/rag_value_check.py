"""Does retrieval matter? Two controls for the final system on the 79 SME questions (P1 + dev_hard + test_hard).

1. Closed book — the same generator (Gemini 3.1 flash-lite) answers with NO retrieved law, from its own knowledge.
   Judged for answer correctness against the ICAI reference. Compared with the final system's correctness, this
   says how much of the answer quality comes from retrieval and how much the model already knows.
2. Question-aware faithfulness — RAGAS faithfulness checks each claim against the retrieved law only; it never
   sees the question, so claims that apply the law to the case facts ("inspection cost of ₹15,000 is included under
   s.15(2)(b)") or compute with the question's numbers count as unsupported. Re-score the same final answers with the
   question added as the first context passage: what is still unsupported comes from the model's memory.

    uv run python -m scripts.thesis.rag_value_check closed_book        # generate (free, Gemini)
    uv run python -m scripts.thesis.rag_value_check question_context   # write the re-score input
    uv run python -m eval.rescore_ragas <closed-book file> --metrics answer_correctness
    uv run python -m eval.rescore_ragas <question-context file> --metrics faithfulness
    uv run python -m scripts.thesis.rag_value_check report <final judged...> -- <closed judged> <qctx judged>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

FINAL = Path("eval/results/20261020T040741Z_gst_final_t31_all.json")
SETS = Path("eval/results/thesis/question_sets.json")
SME = Path("dataset/sme_priority_review/questions.json")
T31 = Path("eval/results/thesis/techniques/t31_pointwise_on_all.json")

CLOSED_BOOK = """You are an expert in Indian GST law (CGST, IGST and UTGST Acts and Rules, notifications and circulars).
Answer the exam question from your own knowledge, as in a suggested answer: state the conclusion, the reasons and the
provisions relied on, and show computations where asked. If you are unsure about a provision, say so."""


def sme_ids() -> set[str]:
    sets = json.loads(SETS.read_text(encoding="utf-8"))["sets"]
    p1 = {q["id"] for q in json.loads(SME.read_text(encoding="utf-8")) if q["priority"] == 1}
    return p1 | set(sets["dev_hard"]) | set(sets["test_hard"])


def closed_book() -> None:
    from app.services.llm_service import generate
    from scripts.techniques.common import generate_run

    generate_run("closed_book", lambda q: {"answer": generate(CLOSED_BOOK, f"Question: {q}")["text"], "chunks": []},
                 "all", ids=sme_ids())


def question_context() -> None:
    payload = json.loads(FINAL.read_text(encoding="utf-8"))
    ids = sme_ids()
    payload["rows"] = [{**r, "contexts": [f"The question (case facts given to the answerer):\n{r['question']}",
                                          *r["contexts"]]} for r in payload["rows"] if r["id"] in ids]
    payload["question_set"] = "sme79_question_in_context"
    out = FINAL.with_name(FINAL.stem.replace("_all", "_sme79_qctx") + ".json")
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(payload['rows'])} rows)")


def _metrics(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for p in paths:
        rows.update({r["id"]: r["ragas_metrics"] for r in json.loads(p.read_text(encoding="utf-8"))["rows"]})
    return rows


def report(final: list[Path], closed: Path, qctx: Path) -> None:
    from eval.stats import HEADER, format_row, paired

    sys.stdout.reconfigure(encoding="utf-8")
    fin, cb, qc = _metrics(final), _metrics([closed]), _metrics([qctx])
    recall = {r["id"]: r["retrieval"]["corrected"]["required_recall"]
              for r in json.loads(T31.read_text(encoding="utf-8"))["rows"]}

    def rows(m: dict, metric: str, ids=None) -> list[dict]:
        return [{"id": i, "ragas_metrics": {metric: v.get(metric)}} for i, v in m.items() if ids is None or i in ids]

    print(HEADER)
    print(format_row("closed book", "final (RAG)", paired(rows(cb, "answer_correctness"), rows(fin, "answer_correctness"),
                                                          "answer_correctness", coverage=None)))
    print(format_row("faith: law only", "faith: law + question", paired(rows(fin, "faithfulness"),
                                                                        rows(qc, "faithfulness"), "faithfulness",
                                                                        coverage=None)))
    print("\nCorrectness by how much of the required law the final system retrieved:")
    print("| retrieved required law | n | closed book | final (RAG) | gain |\n|---|---|---|---|---|")
    for name, lo, hi in (("all of it (recall = 1)", 1.0, 1.01), ("part of it", 0.001, 1.0), ("none", -1, 0.001)):
        ids = {i for i in fin if lo <= recall.get(i, 0) < hi}
        pairs = [(cb[i]["answer_correctness"], fin[i]["answer_correctness"]) for i in ids
                 if i in cb and all(isinstance(x, float) and x == x
                                    for x in (cb[i].get("answer_correctness"), fin[i].get("answer_correctness")))]
        if pairs:
            a, b = statistics.mean(p[0] for p in pairs), statistics.mean(p[1] for p in pairs)
            print(f"| {name} | {len(pairs)} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    wins = sum(1 for i in fin if i in cb and isinstance(fin[i].get("answer_correctness"), float)
               and isinstance(cb[i].get("answer_correctness"), float)
               and fin[i]["answer_correctness"] > cb[i]["answer_correctness"] + 0.05)
    losses = sum(1 for i in fin if i in cb and isinstance(fin[i].get("answer_correctness"), float)
                 and isinstance(cb[i].get("answer_correctness"), float)
                 and cb[i]["answer_correctness"] > fin[i]["answer_correctness"] + 0.05)
    print(f"\nRAG better by > 0.05 on {wins} questions, closed book better on {losses}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["closed_book", "question_context", "report"])
    ap.add_argument("files", nargs="*", type=Path, help="report: final judged files..., then closed-book, question-context")
    args = ap.parse_args()
    if args.cmd == "closed_book":
        closed_book()
    elif args.cmd == "question_context":
        question_context()
    else:
        report(args.files[:-2], args.files[-2], args.files[-1])


if __name__ == "__main__":
    main()
