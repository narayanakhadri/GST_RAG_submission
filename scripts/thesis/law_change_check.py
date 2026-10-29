"""Which golden questions rely on law that changed after the answers' law position (30.04.2025)?

The reference answers state the law as on 30.04.2025; the corpus is the bare law as of April 2026. For every corpus
provision, the amendment footnotes are scanned for amendments that take effect after 30.04.2025 (or 'from a date to
be notified'). Each question whose proposed gold sources (questions_llm_base.json) include such a provision is
listed with the amendment text, for a manual verdict (see docs/law_changes_2025_2026.md).

    uv run python -m scripts.thesis.law_change_check
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
BASE = Path("dataset/sme_priority_review/questions_llm_base.json")
OUT = Path("dataset/sme_priority_review/law_change_candidates.json")
LAW_POSITION = date(2025, 4, 30)

FOOTNOTE = re.compile(r"(?m)^\s*\d{1,3}\s+(?:Inserted|Substituted|Omitted|Amended)[^\n]*(?:\n(?!\s*\d{1,3}\s+[A-Z])[^\n]+)*")
WEF = re.compile(r"w\.?\s*e\.?\s*f\.?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})")


def late_amendments(text: str) -> list[str]:
    """Footnotes of amendments effective after the law position, or effective from a date still to be notified."""
    out = []
    for m in FOOTNOTE.finditer(text):
        note = " ".join(m.group(0).split())
        if not re.search(r"202[5-6]", note):
            continue
        dates = [date(int(y), int(mo), int(d)) for d, mo, y in WEF.findall(note)]
        if "date to be notified" in note or any(d > LAW_POSITION for d in dates):
            out.append(note[:400])
    return out


def main() -> None:
    corpus = {}
    for line in CORPUS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        corpus[r["id"]] = r
    changed = {pid: notes for pid, r in corpus.items() if (notes := late_amendments(r.get("raw_text") or ""))}
    rows = []
    for q in json.loads(BASE.read_text(encoding="utf-8")):
        hits = {s: changed[s] for s in q["required_sources"] if s in changed}
        if hits:
            rows.append({"id": q["id"], "roles": {s: q["llm_roles"].get(s) for s in hits}, "amendments": hits})
    OUT.write_text(json.dumps({"provisions_changed": changed, "questions": rows}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"{len(changed)} provisions amended after {LAW_POSITION} (or to be notified); "
          f"{len(rows)} questions use one of them -> {OUT}")
    for pid, notes in changed.items():
        print(f"  {pid}: {len(notes)}")


if __name__ == "__main__":
    main()
