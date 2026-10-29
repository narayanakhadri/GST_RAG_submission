"""Build a short SME review workbook covering only what the automated pipeline cannot verify itself:

1. Questions — golden questions where hybrid retrieval found none of the required law (or the answer said the
   context was insufficient). Each one is either a wrong gold-source annotation or a real retrieval miss.
2. Aliases — the citation → corpus-ID mappings in authority_aliases.json that are judgment calls.
3. Table rows — random entries of the large rate notifications, whose table rows were split heuristically
   from the PDF text layer, to compare against the PDF.

    uv run python -m scripts.gst_pipeline.make_review_sheet \
        --results eval/results/20261016T191402Z_gst_hybrid.json

Writes an Excel workbook (dataset/sme_review_sheet.xlsx) and the same items as JSON
(dataset/sme_priority_review/*.json), which the SME Streamlit app's "Priority review" page reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from eval.gst_retrieval import source_matches

ALIASES = Path("seed/docs/gst/sources/authority_aliases.json")
JSON_DIR = Path("dataset/sme_priority_review")
TABLE_NOTIFICATIONS = (
    "notification_12_2017_ct_rate",
    "notification_13_2017_ct_rate",
    "notification_17_2017_ct_rate",
    "notification_9_2017_it_rate",
)
ROWS_PER_TABLE = 4
SEED = 42
TEXT_LIMIT = 1500
# Phrases the generator uses when it says the retrieved law does not answer the question.
ABSTAIN_PHRASES = ("insufficient", "no information", "does not contain", "do not contain", "no mention of")
HEADER_FILL = PatternFill("solid", fgColor="DDE7F3")
SME_FILL = PatternFill("solid", fgColor="FFF4CC")

INSTRUCTIONS = [
    "GST SME review sheet — fill only the yellow columns.",
    "",
    "Tab 1 'Questions': for each golden question, retrieval found none of the law listed in 'Required sources',",
    "or the model said the context was insufficient. Read the question and reference answer and decide:",
    "  • Sources correct — the listed law is right; retrieval just missed it (no change needed).",
    "  • Sources changed — the listed law is wrong or incomplete; write the correct sources in 'Correct sources'",
    "    (e.g. 'CGST s.9; Rule 36; Notification 12/2017-CT(R) entry 5').",
    "  • Question unusable — ambiguous, not answerable from GST law, or the reference answer is wrong.",
    "  Retrieved law that actually answers the question should be noted in 'Notes'.",
    "",
    "Tab 2 'Aliases': citations in the golden set that were mapped to a corpus document by judgment. Confirm or reject.",
    "",
    "Tab 3 'Table rows': random rows of the four large rate notifications. Open the PDF link, find the Sl. No.,",
    "and check the row text (description, rate, condition) matches. Mark Match / Mismatch and describe any error.",
]


def short(text: object, limit: int = TEXT_LIMIT) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def titled(source_id: str, titles: dict[str, str]) -> str:
    return f"{source_id} — {titles[source_id]}" if source_id in titles else source_id


def corpus_titles(corpus: list[dict]) -> dict[str, str]:
    titles = {}
    for r in corpus:
        title = r.get("provision_title") or ""
        titles[r["id"]] = f"{r.get('instrument_title', '')}, {r.get('provision_type', '')} {r.get('provision_number', '')}: {title}".strip()
        titles.setdefault(r["instrument_id"], f"{r.get('instrument_title', '')} (whole instrument)")
    return titles


def write_sheet(ws, headers: list[str], rows: list[list], widths: list[int], sme_cols: list[str]) -> None:
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = SME_FILL if cell.value in sme_cols else HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows:
        ws.append(row)
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if headers[cell.column - 1] in sme_cols:
                cell.fill = SME_FILL
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions


def add_choices(ws, headers: list[str], column: str, choices: list[str], n_rows: int) -> None:
    letter = ws.cell(1, headers.index(column) + 1).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{n_rows + 1}")


RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")


def _number(value: object) -> float | None:
    """RAGAS writes NaN when it could not score a row; JSON has no NaN, so store None."""
    return value if isinstance(value, (int, float)) and value == value else None


def question_items(results: dict, goldens: dict[str, dict]) -> list[dict]:
    out = []
    for r in results["rows"]:
        g = goldens[r["id"]]
        recall = r["retrieval"]["corrected"]["required_recall"]
        abstained = any(p in (r.get("answer") or "").lower() for p in ABSTAIN_PHRASES)
        if recall and not abstained:
            continue
        retrieved = r.get("retrieved_ids") or []
        required = g.get("primary_required_sources") or []
        validation = g.get("golden_source_validation") or {}
        out.append({
            "id": r["id"],
            "priority": 1 if recall == 0 and abstained else 2 if recall == 0 else 3,
            "question": g["question"],
            "reference_answer": g.get("reference_answer") or "",
            "required_sources": required,
            "missed_sources": [s for s in required if not any(source_matches(s, rid) for rid in retrieved)],
            "added_by_correction": validation.get("sources_added") or [],
            "correction_reason": validation.get("reason") or "",
            "gold_source_notes": g.get("gold_source_notes") or {},
            "retrieved_sources": retrieved[:5],
            "model_abstained": abstained,
            "model_answer": r.get("answer") or "",
            "scores": {
                **{k: _number(r["ragas_metrics"].get(k)) for k in RAGAS_KEYS},
                **{k: r["retrieval"]["corrected"].get(k) for k in ("required_recall", "required_mrr")},
            },
        })
    out.sort(key=lambda q: (q["priority"], q["id"]))
    return out


def alias_items(goldens: list[dict]) -> list[dict]:
    data = json.loads(ALIASES.read_text(encoding="utf-8"))["aliases"]
    dumps = {g["id"]: json.dumps(g, ensure_ascii=False) for g in goldens}
    out = []
    for citation, target in data.items():
        if citation.startswith("igst_act_2017.section."):
            why = ("The IGST Act has no such section; the answer cites the CGST Act provision, which applies to IGST "
                   "via IGST Act s.20.")
        elif "(Rate)" not in citation and "_rate" in target:
            why = ("Cited without '(Rate)'; the answer refers to the rate notification of the same number "
                   "(exemption/rate schedule), not a non-rate notification.")
        else:
            continue  # exact citation → document, no judgment involved
        used_by = [gid for gid, text in dumps.items() if f'"{citation}"' in text]
        out.append({"id": citation, "citation": citation, "mapped_to": target, "why": why, "question_ids": used_by})
    return out


def table_items(corpus: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for instrument in TABLE_NOTIFICATIONS:
        entries = [r for r in corpus if r["instrument_id"] == instrument and r.get("provision_type") == "entry"]
        for r in sorted(rng.sample(entries, min(ROWS_PER_TABLE, len(entries))), key=lambda e: e["source_order"]):
            out.append({
                "id": r["id"],
                "notification": r["instrument_title"],
                "serial_number": r["provision_number"],
                "row_text": r.get("raw_text") or "",
                "law_position": r.get("law_position", ""),
                "pdf_url": r.get("source_url", ""),
                "pdf_local": f"seed/docs/gst/sources/pdf/{instrument}.pdf",
            })
    return out


def lines(ids: list[str], titles: dict[str, str]) -> str:
    return "\n".join(titled(s, titles) for s in ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="GST RAGAS results JSON (hybrid profile recommended)")
    ap.add_argument("--output", default="dataset/sme_review_sheet.xlsx")
    args = ap.parse_args()

    corpus = load_corpus(settings.gst_corpus_path)
    golden_list = load_golden_set(settings.gst_golden_path)
    titles = corpus_titles(corpus)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    questions = question_items(results, {g["id"]: g for g in golden_list})
    aliases = alias_items(golden_list)
    tables = table_items(corpus)

    wb = Workbook()
    ws = wb.active
    ws.title = "How to review"
    for line in INSTRUCTIONS:
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 120

    q_headers = ["Priority", "Question ID", "Question", "Reference answer", "Required sources",
                 "Required but not retrieved", "Added by auto-correction", "Retrieved (top 5)", "Model abstained",
                 "Model answer", "SME verdict", "Correct sources", "Notes"]
    q_rows = [[q["priority"], q["id"], short(q["question"], 3000), short(q["reference_answer"], 3000),
               lines(q["required_sources"], titles), lines(q["missed_sources"], titles),
               lines(q["added_by_correction"], titles), lines(q["retrieved_sources"], titles),
               "yes" if q["model_abstained"] else "no", short(q["model_answer"]), "", "", ""] for q in questions]
    ws = wb.create_sheet("1 Questions")
    write_sheet(ws, q_headers, q_rows, [8, 26, 60, 60, 45, 45, 35, 45, 10, 50, 22, 40, 40],
                ["SME verdict", "Correct sources", "Notes"])
    add_choices(ws, q_headers, "SME verdict", ["Sources correct", "Sources changed", "Question unusable"], len(q_rows))

    a_headers = ["Citation in golden set", "Mapped to", "Why", "Questions affected", "Question IDs", "SME verdict", "Notes"]
    a_rows = [[a["citation"], a["mapped_to"], a["why"], len(a["question_ids"]), "\n".join(a["question_ids"]), "", ""]
              for a in aliases]
    ws = wb.create_sheet("2 Aliases")
    write_sheet(ws, a_headers, a_rows, [34, 32, 60, 10, 34, 18, 40], ["SME verdict", "Notes"])
    add_choices(ws, a_headers, "SME verdict", ["Confirm", "Reject"], len(a_rows))

    t_headers = ["Notification", "Sl. No.", "Row text in corpus", "Version", "PDF", "SME verdict", "Error / notes"]
    t_rows = [[t["notification"], t["serial_number"], short(t["row_text"], 3000), t["law_position"], t["pdf_url"], "", ""]
              for t in tables]
    ws = wb.create_sheet("3 Table rows")
    write_sheet(ws, t_headers, t_rows, [34, 8, 80, 30, 40, 14, 40], ["SME verdict", "Error / notes"])
    add_choices(ws, t_headers, "SME verdict", ["Match", "Mismatch"], len(t_rows))

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in (("questions", questions), ("aliases", aliases), ("table_rows", tables)):
        (JSON_DIR / f"{name}.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(args.output)
    except PermissionError:
        print(f"WARNING: {args.output} is open in another program (e.g. Excel); workbook not rewritten")
    by_priority = {p: sum(1 for q in questions if q["priority"] == p) for p in (1, 2, 3)}
    print(f"Wrote {JSON_DIR}/ (+ workbook {args.output}): {len(questions)} questions (priority {by_priority}), "
          f"{len(aliases)} aliases, {len(tables)} table rows")


if __name__ == "__main__":
    main()
