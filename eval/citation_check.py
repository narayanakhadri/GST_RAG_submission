"""Citation correctness for GST answers (proposal RQ4 / objective 5) — deterministic, no LLM.

Parses the legal citations an answer makes ("Section 16(2) of the CGST Act", "rule 36(4)", "Schedule II",
"Notification No. 12/2017-Central Tax (Rate)", "Circular No. 92/11/2019-GST"), resolves them to corpus IDs
and scores each answer:

    n_citations           distinct provisions/instruments cited
    grounded_precision    share of citations found in the retrieved context (cited law the model was shown)
    outside_context       citations NOT in the retrieved context — from the model's own memory: possibly
                          correct, but unverifiable from the evidence (the hallucination risk RQ4 is about)
    unknown_law           citations that resolve to nothing in the corpus (wrong numbers, law not ingested)
    gold_citation_recall  share of the gold required sources the answer cites

Re-scores any GST RAGAS results file without re-running anything:

    uv run python -m eval.citation_check eval/results/<file>.json [--write]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from eval.gst_retrieval import corpus_keys, gold_sets, source_matches

_NUM = r"(\d+[A-Z]{0,2})"
_SECTION = re.compile(rf"\b(?:sections?|sec\.?|s\.)\s*{_NUM}((?:\s*\([0-9a-zA-Z]+\))*)", re.I)
_RULE = re.compile(rf"\brules?\s+{_NUM}((?:\s*\([0-9a-zA-Z]+\))*)", re.I)
_SCHEDULE = re.compile(r"\bschedule\s+(I{1,3}|IV|1|2|3)\b|\b(first|second|third)\s+schedule\b", re.I)
_NOTIFICATION = re.compile(
    r"notification\s*(?:no\.?)?\s*(\d+)\s*/\s*(\d{4})\s*[-–]?\s*"
    r"(central\s*tax|integrated\s*tax|union\s*territory\s*tax|ct|it|ut)?\s*(\(?\s*rate\s*\)?|\(r\))?",
    re.I,
)
_CIRCULAR = re.compile(r"circular\s*(?:no\.?)?\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d{4})\s*[-–]?\s*(c?gst)?", re.I)
_SCHEDULE_NUM = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "first": 1, "second": 2, "third": 3}
_ABSTAIN = ("insufficient", "no information", "does not contain", "do not contain", "no mention of")


_ACT_AFTER = re.compile(r"[\s,]*(?:of|under)\s+(?:the\s+)?(igst|integrated|utgst|union\s+territory)", re.I)


def _act_after(text: str, end: int, default: str) -> str:
    """Which Act a section/rule citation refers to: only an "of/under the IGST/UTGST ..." phrase directly
    after the citation counts, so a later "section 12 of the IGST Act" doesn't relabel an earlier rule."""
    m = _ACT_AFTER.match(text, end)
    if not m:
        return default
    return "igst" if m.group(1).lower() in ("igst", "integrated") else "utgst"


def extract_citations(answer: str, keys: set[str]) -> list[str]:
    """Distinct corpus-style IDs cited in `answer`, in first-seen order. Unresolvable notifications and
    circulars are returned as their best-guess ID so they can be counted as unknown law."""
    found: list[str] = []

    def add(cid: str) -> None:
        if cid not in found:
            found.append(cid)

    for m in _SECTION.finditer(answer):
        act = _act_after(answer, m.end(), "cgst")
        add(f"{act}_act_2017.section.{m.group(1).lower()}")
    for m in _RULE.finditer(answer):
        act = "igst" if _act_after(answer, m.end(), "cgst") == "igst" else "cgst"
        add(f"{act}_rules_2017.rule.{m.group(1).lower()}")
    for m in _SCHEDULE.finditer(answer):
        num = _SCHEDULE_NUM[(m.group(1) or m.group(2)).lower()]
        add(f"cgst_act_2017.schedule.{num}")
    for m in _NOTIFICATION.finditer(answer):
        num, year, tax, rate = m.groups()
        tax = (tax or "").lower().replace(" ", "")
        series = "it" if tax.startswith(("integrated", "it")) else "ut" if tax.startswith(("union", "ut")) else "ct"
        candidates = [f"notification_{num}_{year}_{series}{'_rate' if rate else ''}",
                      f"notification_{num}_{year}_{series}_rate", f"notification_{num}_{year}_{series}"]
        add(next((c for c in candidates if c in keys), candidates[0]))
    for m in _CIRCULAR.finditer(answer):
        a, b, year, suffix = m.groups()
        candidates = [f"circular_{a}_{b}_{year}_{(suffix or 'gst').lower()}", f"circular_{a}_{b}_{year}_gst",
                      f"circular_{a}_{b}_{year}_cgst"]
        add(next((c for c in candidates if c in keys), candidates[0]))
    return found


def _matches(a: str, b: str) -> bool:
    """Lenient match: either ID names the other or contains it (instrument ↔ its entries)."""
    return source_matches(a, b) or source_matches(b, a)


def score_answer(answer: str, retrieved: list[str], gold_required: list[str], keys: set[str]) -> dict:
    cited = extract_citations(answer or "", keys)
    known = [c for c in cited if c in keys]
    grounded = [c for c in cited if any(_matches(c, r) for r in retrieved)]
    return {
        "citations": cited,
        "n_citations": len(cited),
        "grounded_precision": len(grounded) / len(cited) if cited else None,
        "outside_context": len(cited) - len(grounded),
        "unknown_law": len(cited) - len(known),
        "gold_citation_recall": (sum(any(_matches(g, c) for c in cited) for g in gold_required) / len(gold_required)
                                 if gold_required else None),
        "abstained": any(p in (answer or "").lower() for p in _ABSTAIN),
    }


def score_results(payload: dict, gold_version: str = "corrected") -> dict:
    keys = corpus_keys(load_corpus(settings.gst_corpus_path))
    goldens = {g["id"]: g for g in load_golden_set(settings.gst_golden_path)}
    for row in payload["rows"]:
        gold = gold_sets(goldens[row["id"]], gold_version)[0] if row["id"] in goldens else []
        row["citation"] = score_answer(row.get("answer", ""), row.get("retrieved_ids") or [], gold, keys)
    payload["citation_summary"] = summarise([r["citation"] for r in payload["rows"]])
    return payload


def summarise(scores: list[dict]) -> dict:
    def avg(key: str) -> float | None:
        vals = [s[key] for s in scores if s[key] is not None]
        return round(mean(vals), 4) if vals else None

    answered = [s for s in scores if not s["abstained"]]
    return {
        "n_answers": len(scores),
        "abstained": sum(s["abstained"] for s in scores),
        "answers_with_citations": sum(1 for s in scores if s["n_citations"]),
        "mean_citations": avg("n_citations"),
        "grounded_precision": avg("grounded_precision"),
        "answers_citing_outside_context": sum(1 for s in answered if s["outside_context"]),
        "answers_citing_unknown_law": sum(1 for s in answered if s["unknown_law"]),
        "gold_citation_recall": avg("gold_citation_recall"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--write", action="store_true", help="store per-row scores back into the results file")
    args = ap.parse_args()
    for path in args.results:
        payload = score_results(json.loads(Path(path).read_text(encoding="utf-8")))
        print(Path(path).name, json.dumps(payload["citation_summary"]))
        if args.write:
            Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
