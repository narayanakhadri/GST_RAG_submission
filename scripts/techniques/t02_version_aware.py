"""T02 — Version-aware RAG: answer "as on date X" from the amendment history of each provision (proposal §7.5).

Idea. Law changes; a question is answered under the law in force on some date. The corpus is the bare law as of
April 2026, but every amended clause carries a footnote ("2 Substituted by the Finance Act, 2025, w.e.f.
01.10.2025. Prior to substitution it read …"). Parsing those footnotes gives each provision a version history:
which parts changed, when, and what the earlier text was. At query time a date ("as on 30.04.2025") selects the
version: amendments that take effect after that date are flagged and the earlier wording (when the footnote
gives it) is shown to the generator instead.

Papers: proposal V7 §7.5 (version-aware filters, "latest legal position"), VersionRAG (30: version graph + intent
routing — 90% vs 58% for naive RAG on version-sensitive questions).

How it works here (no LLM, no cost).
- `history(record)` → list of {footnote no., action (Inserted/Substituted/Omitted), effective date or
  "to be notified", amending instrument, prior text if quoted}.
- `as_on(record, date)` → the provision text with a header listing amendments not yet in force on that date
  and, for each, the earlier wording when the footnote quotes it.
- `annotate_chunks(chunks, date)` → the hook a pipeline would call before generation.
- `case_study` → runs this over the 8 golden questions whose law changed after the answers' law position
  (dataset/sme_priority_review/law_change_verdicts.json) and prints what the generator would be told.
- `stats` → how many provisions have dated amendments, by year.

    uv run python -m scripts.techniques.t02_version_aware stats
    uv run python -m scripts.techniques.t02_version_aware case_study --date 2025-04-30
    uv run python -m scripts.techniques.t02_version_aware show cgst_act_2017.section.15 --date 2025-04-30

Limitations: only what the footnotes say (the ICAI bare-law edition); notification copies in the corpus are
2019 consolidations without later amendments; no gold labels for version-sensitive retrieval, so the evaluation
is the case study, not a metric.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.models import RetrievedChunk

FOOTNOTE = re.compile(r"(?m)^\s*(\d{1,3})\s+(Inserted|Substituted|Omitted|Amended)\b([^\n]*(?:\n(?!\s*\d{1,3}\s+[A-Z])[^\n]+)*)")
WEF = re.compile(r"w\.?\s*e\.?\s*f\.?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})")
BY = re.compile(r"\b(?:by|vide)\s+(?:the\s+)?(.{5,90}?)(?:,|\s+w\.?e\.?f|\s+dated|\.\s|$)", re.I)
PRIOR = re.compile(r"prior to (?:its )?(?:substitution|omission|amendment),?\s*it\s+(?:was\s+)?(?:read|stood)\s+as(?:\s+under)?\s*:?-?\s*(.*)",
                   re.I | re.S)
VERDICTS = Path("dataset/sme_priority_review/law_change_verdicts.json")


def history(record: dict) -> list[dict]:
    out = []
    raw = record.get("raw_text") or ""
    matches = list(FOOTNOTE.finditer(raw))
    for i, m in enumerate(matches):
        body = " ".join(m.group(3).split())
        # The quoted earlier wording often follows after a blank line: read up to the next footnote.
        full = " ".join(raw[m.start() : matches[i + 1].start() if i + 1 < len(matches) else m.start() + 2000].split())
        prior = PRIOR.search(full)
        head = full[: prior.start()] if prior else full[:500]  # dates/Act before the quoted old wording
        dates = [date(int(y), int(mo), int(d)) for d, mo, y in WEF.findall(head)]
        by = BY.search(head)
        out.append({
            "footnote": int(m.group(1)), "action": m.group(2),
            "effective": dates[0].isoformat() if dates else ("to be notified" if "date to be notified" in body else None),
            "by": by.group(1).strip() if by else None,
            "prior_text": prior.group(1).strip()[:600] if prior else None,
            "note": body[:300],
        })
    return out


def not_in_force(record: dict, on: date) -> list[dict]:
    """Amendments to this provision that were not yet in force on `on`."""
    late = []
    for h in history(record):
        eff = h["effective"]
        if eff == "to be notified" or (eff and date.fromisoformat(eff) > on):
            late.append(h)
    return late


def as_on(record: dict, on: date) -> str:
    text = record.get("raw_text") or ""
    late = not_in_force(record, on)
    if not late:
        return text
    lines = [f"[Version note: law as on {on.isoformat()}. The following amendments to this provision were NOT yet in force:"]
    for h in late:
        when = h["effective"] or "date not stated"
        lines.append(f"- footnote {h['footnote']}: {h['action']} {('by ' + h['by']) if h['by'] else ''}, effective {when}."
                     + (f" Wording before the amendment: \"{h['prior_text']}\"" if h["prior_text"] else ""))
    lines.append("Apply the earlier wording where given.]")
    return "\n".join(lines) + "\n\n" + text


def annotate_chunks(chunks: list[RetrievedChunk], on: date) -> list[RetrievedChunk]:
    """Pipeline hook: prepend the version note to every retrieved provision amended after `on`."""
    from app.services.gst_graph import get_graph

    graph = get_graph()
    out = []
    for c in chunks:
        pid = graph.id_of(c)
        record = graph.records.get(pid) if pid else None
        late = not_in_force(record, on) if record else []
        if late:
            note = as_on(record, on).split("\n\n", 1)[0]
            out.append(RetrievedChunk(text=f"{note}\n\n{c.text}", source=c.source, score=c.score))
        else:
            out.append(c)
    return out


# Query routing as in VersionRAG (github.com/danielhuwiler/versionrag, src/retrieval/versionrag_retriever_parser.py):
# their LLM classifies VersionRetrieval / ChangeRetrieval / ContentRetrieval and extracts parameters. Legal
# questions signal the type in words, so a rule-based router is used here (no LLM, deterministic).
_CHANGE_Q = re.compile(r"\b(amend\w*|substitut\w*|omit\w*|insert\w*|chang\w*|w\.?e\.?f|with effect from|Finance Act,? 20\d\d)\b", re.I)
_DATE_Q = re.compile(r"\b(?:as on|as of|on|before|after|from)\s+(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", re.I)


def route(question: str) -> dict:
    """{"intent": "change" | "as_on" | "content", "date": ISO or None}."""
    m = _DATE_Q.search(question)
    when = date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat() if m else None
    if _CHANGE_Q.search(question):
        return {"intent": "change", "date": when}
    return {"intent": "as_on" if when else "content", "date": when}


def changes(record: dict, since: date | None = None, until: date | None = None) -> list[dict]:
    """ChangeRetrieval: the amendments of a provision with an effective date in [since, until]."""
    out = []
    for h in history(record):
        eff = h["effective"]
        if not eff or not eff[0].isdigit():
            out.append(h) if eff == "to be notified" else None
            continue
        d = date.fromisoformat(eff)
        if (since is None or d >= since) and (until is None or d <= until):
            out.append(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["stats", "case_study", "show"])
    ap.add_argument("provision", nargs="?")
    ap.add_argument("--date", default="2025-04-30")
    args = ap.parse_args()
    on = date.fromisoformat(args.date)
    corpus = {r["id"]: r for r in load_corpus(settings.gst_corpus_path)}
    if args.cmd == "stats":
        years, n_dated, n_any = Counter(), 0, 0
        for r in corpus.values():
            h = history(r)
            n_any += bool(h)
            n_dated += any(x["effective"] for x in h)
            years.update(x["effective"][:4] for x in h if x["effective"] and x["effective"][0].isdigit())
        print(f"{n_any} provisions with amendment footnotes, {n_dated} with an effective date; by year: "
              f"{dict(sorted(years.items()))}")
        print(f"not in force on {on}: {sum(bool(not_in_force(r, on)) for r in corpus.values())} provisions")
    elif args.cmd == "show":
        print(as_on(corpus[args.provision], on)[:3000])
    else:
        verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
        cases = {q: v["status"] for q, v in verdicts.items() if not q.startswith("_")}
        cand = json.loads(Path("dataset/sme_priority_review/law_change_candidates.json").read_text(encoding="utf-8"))
        by_q = {q["id"]: q for q in cand["questions"]}
        for qid, verdict in cases.items():
            q = by_q.get(qid)
            print(f"\n## {qid} ({verdict}): {verdicts[qid]['note'][:160]}")
            for pid in (q or {}).get("amendments", {}):
                late = not_in_force(corpus[pid], on)
                print(f"  {pid}: {len(late)} amendment(s) not in force on {on}")
                for h in late:
                    print(f"    - {h['action']} {h['effective']} {h['by'] or ''} | prior text: {'yes' if h['prior_text'] else 'no'}")


if __name__ == "__main__":
    main()
