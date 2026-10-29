"""Cited-provision lookup (thesis wave 1, W1a): if a question names a provision — "section 22", "rule 36(4)",
"Schedule I", "Notification No. 12/2017-CT (Rate)", "Circular No. 92/11/2019-GST" — fetch that provision by ID
instead of hoping search finds it. Symbolic matching next to neural search: deterministic, explainable, free.

The Act is taken from the words right after the number ("section 12 of the IGST Act"); without one, CGST is
assumed (exam questions name the IGST/UTGST Act explicitly). A notification or circular is a whole instrument;
the caller picks its most relevant entry/paragraph.
"""

from __future__ import annotations

import re

_ACT_HINTS = (
    (re.compile(r"\b(IGST|Integrated)\b", re.I), "igst"),
    (re.compile(r"\b(UTGST|Union Territory)\b", re.I), "utgst"),
)
_SECTION = re.compile(r"\b(?:section|sec\.|s\.)\s*(\d{1,3}[A-Z]{0,2})\b", re.I)
_RULE = re.compile(r"\brule\s*(\d{1,3}[A-Z]{0,2})\b", re.I)
_SCHEDULE = re.compile(r"\bSchedule\s*(III|II|I|[123])\b")
_NOTIFICATION = re.compile(
    r"\bNotification\s*No\.?\s*(\d{1,3})\s*/\s*(\d{4})\s*[-–]?\s*(Central|Integrated|CT|IT)?[^()\n]{0,12}(\(\s*Rate\s*\))?",
    re.I,
)
_CIRCULAR = re.compile(r"\bCircular\s*No\.?\s*(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{4})", re.I)
_ROMAN = {"I": "1", "II": "2", "III": "3"}
WINDOW = 60  # characters after the number in which "of the IGST Act" counts


def _act(text: str, end: int, kind: str) -> str:
    window = text[end : end + WINDOW]
    for pattern, prefix in _ACT_HINTS:
        if pattern.search(window):
            return f"{prefix}_{'rules' if kind == 'rule' else 'act'}_2017"
    return f"cgst_{'rules' if kind == 'rule' else 'act'}_2017"


def cited_ids(question: str, provision_ids: set[str], instrument_ids: set[str]) -> list[str]:
    """Corpus provision or instrument IDs the question names, in order of first mention (unknown ones dropped)."""
    found: list[tuple[int, str]] = []
    for m in _SECTION.finditer(question):
        found.append((m.start(), f"{_act(question, m.end(), 'section')}.section.{m.group(1).lower()}"))
    for m in _RULE.finditer(question):
        found.append((m.start(), f"{_act(question, m.end(), 'rule')}.rule.{m.group(1).lower()}"))
    for m in _SCHEDULE.finditer(question):
        found.append((m.start(), f"cgst_act_2017.schedule.{_ROMAN.get(m.group(1), m.group(1))}"))
    for m in _NOTIFICATION.finditer(question):
        num, year, tax, rate = m.groups()
        kind = "it" if tax and tax.lower().startswith(("i", "integrated")) else "ct"
        found.append((m.start(), f"notification_{num}_{year}_{kind}{'_rate' if rate else ''}"))
    for m in _CIRCULAR.finditer(question):
        a, b, year = m.groups()
        prefixes = {f"circular_{a}_{b}_{year}_", f"circular_{a}_{int(b):02d}_{year}_"}
        found += [(m.start(), i) for i in instrument_ids if any(i.startswith(p) for p in prefixes)]
    out: list[str] = []
    for _, cid in sorted(found):
        if (cid in provision_ids or cid in instrument_ids) and cid not in out:
            out.append(cid)
    return out
