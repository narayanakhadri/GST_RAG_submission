"""Stage 1b (parse): turn the Docling Markdown of each notification/circular into provision-shaped records.

Reads seed/docs/gst/sources/manifest.json (written by fetch_sources.py). Each table row becomes an
`entry` record (e.g. notification_12_2017_ct_rate.entry.12 — one exemption/RCM line, with the column
headers repeated so the row makes sense on its own); the text around the tables is split into
`paragraph` records at top-level paragraph numbers ("2.", "3."). Letterhead, addressees, running page
headers and signatures are dropped. Records carry the same fields as the bare-law records, plus
`instrument_subject`, `issue_date`, `law_position` and `source_url`, so enrich.py and seed_gst.py
handle them unchanged.

    uv run python -m scripts.gst_pipeline.parse_notifications --output seed/docs/gst/build/notifications_raw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCES_DIR = Path("seed/docs/gst/sources")

_BOILERPLATE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\[?\s*TO BE PUBLISHED IN THE GAZ",
        r"^Disclaimer:.*(no legal binding|for convenience)",
        r"^\[?\s*Updated version of the Notification",
        r"^(Government of India|Ministry of Finance|\(?Department of Revenue\)?|Central Board of (Indirect Taxes|Excise)|"
        r"GST Policy Wing|Tax Research Unit)\b",
        r"^F\.\s*No\.",
        r"^\[\s*F\.\s*No\.",
        r"^New Delhi\b",
        r"^To,?$",
        r"^The (Principal )?(Chief )?Commissioners?\b",
        r"^The Principal Director Generals?\b",
        r"^(Madam|Sir)\s*/\s*(Madam|Sir),?$",
        r"^Circular No\.?\s*[\d/ ]+-?\s*[A-Z]*\s*$",
        r"^Page \d+ of \d+$",
        r"^<!--.*-->$",
        r"^Hindi version (will|would) follow",
        r"^\(?[A-Z][A-Za-z. ]{2,40}\)?\s*(Principal Commissioner|Commissioner|Under Secretary|Joint Secretary|Deputy Secretary|Director|OSD)\b.*$",
        r"^(Principal Commissioner|Commissioner|Under Secretary|Joint Secretary|Deputy Secretary|Director)\b.*$",
        r"^\([A-Z][A-Za-z. ]{2,40}\)$",
        r"^Notification\s+No\.?\s*[\d/]+\s*-?\s*(Central|Integrated|Union)[^.]*$",
    )
]
_TOP_PARAGRAPH = re.compile(r"^(?:#+\s*)?(\d{1,2})\.(?!\d)\s*\S")
# Sub-paragraph starts used to split a long paragraph: "7.1 ...", "A. Free samples ...".
_SUB_PARAGRAPH = re.compile(
    r"^(?:-\s*)?(?:(?P<num>\d{1,2}\.\d{1,2})(?![.\d])|(?P<letter>[A-H])\.)\s+\S"
)
LONG_PARAGRAPH_WORDS = 600
# Clause starts inside a long paragraph: "(a) 'advertisement' means ...", "- (zza) ...", "[(zf) ..." (amended).
_CLAUSE = re.compile(r"^(?:-\s*)?\[?\(([a-z]{1,4})\)\s?\S")
CLAUSE_GROUP_WORDS = 400
# "12", "9A", "(3)", and "[3A" — consolidated copies open inserted entries with an amendment bracket.
# an opening quote mark: amending notifications quote the rows they insert ("'36B")
_SERIAL = re.compile(r"^['‘’\"“]?\[?\s*\(?(\d{1,3}\s*[A-Z]{0,2})\)?\.?\]?$", re.IGNORECASE)
_COLUMN_NUMBERS = re.compile(r"^\(?\d{1,2}\)?$")
_DESCRIPTION_HEADERS = ("description", "category", "issue", "nature", "supply", "particulars")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_boilerplate(line: str) -> bool:
    text = re.sub(r"^#+\s*", "", line).strip()
    return any(p.search(text) for p in _BOILERPLATE)


def split_cells(line: str) -> list[str]:
    return [clean_line(c) for c in line.strip().strip("|").split("|")]


def short_title(text: str, words: int = 14) -> str:
    sentence = re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0]
    parts = sentence.split()
    return " ".join(parts[:words]).rstrip(",;:-") + ("…" if len(parts) > words else "")


def parse_table_block(
    block: list[str], header: list[str] | None, entries: list[dict[str, Any]]
) -> list[str] | None:
    """Append the rows of one Markdown table block to `entries` (document-wide, so a row that Docling split
    at a page break — next block, empty serial cell — is merged into the entry it continues).
    Returns the header in effect for following blocks."""
    block_rows = 0
    for line in block:
        cells = split_cells(line)
        if all(re.fullmatch(r":?-+:?", c) or not c for c in cells):
            continue  # separator
        if all(_COLUMN_NUMBERS.match(c) or not c for c in cells):
            continue  # "(1) (2) (3)" column-number row
        if cells == header:
            continue  # header repeated on a new page
        serial = _SERIAL.match(cells[0])
        if not serial and cells[0] and block_rows == 0:
            header = cells  # first non-data row of a block is its header
            continue
        block_rows += 1
        if serial:
            entries.append(
                {
                    "number": re.sub(r"\s+", "", serial.group(1)).upper(),
                    "cells": cells,
                    "header": header,
                }
            )
        elif entries and not cells[0]:
            prev = entries[-1]["cells"]
            prev.extend([""] * (len(cells) - len(prev)))
            for i, c in enumerate(cells):
                if c:
                    prev[i] = f"{prev[i]} {c}".strip()
        elif any(cells):
            entries.append({"number": None, "cells": cells, "header": header})
    return header


def entry_text(cells: list[str], header: list[str] | None) -> tuple[str, str]:
    """('Column: value' lines, title) for one table row."""
    header = header or [f"Column {i + 1}" for i in range(len(cells))]
    lines, title = [], ""
    for i, value in enumerate(cells):
        name = header[i] if i < len(header) else f"Column {i + 1}"
        if not value or i == 0 and _SERIAL.match(value):
            continue
        lines.append(f"{name}: {value}")
        if not title and any(k in name.lower() for k in _DESCRIPTION_HEADERS):
            title = short_title(value)
    if not title and lines:
        title = short_title(lines[0].split(": ", 1)[-1])
    return "\n".join(lines), title


def parse_document(markdown: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """-> (paragraph segments, table entries), each in document order."""
    segments: list[dict[str, Any]] = [{"number": None, "lines": []}]
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            header = parse_table_block(block, header, rows)
            continue
        i += 1
        text = clean_line(line)
        if not text or is_boilerplate(text):
            continue
        m = _TOP_PARAGRAPH.match(text)
        if (
            m
            and int(m.group(1)) > 0
            and (segments[-1]["number"] is None or int(m.group(1)) > int(segments[-1]["number"]))
        ):
            segments.append({"number": m.group(1), "lines": []})
        segments[-1]["lines"].append(re.sub(r"^#+\s*", "", text))
    entries = []
    for row in rows:
        text, title = entry_text(row["cells"], row["header"])
        entries.append({"number": row["number"], "text": text, "title": title})
    return [s for s in segments if s["lines"]], entries


def first_header(markdown: str) -> list[str] | None:
    """Column names of the document's first table, as Docling read them."""
    for line in markdown.splitlines():
        if line.lstrip().startswith("|"):
            cells = split_cells(line)
            if cells and cells[0] and not _SERIAL.match(cells[0]):
                return cells
    return None


def split_long_segment(segment: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a paragraph longer than LONG_PARAGRAPH_WORDS at its sub-paragraphs (7.1, 7.2 / A., B.).
    Text before the first sub-paragraph keeps the paragraph number; parts are numbered "7.1" or "2A"."""
    if sum(len(line.split()) for line in segment["lines"]) <= LONG_PARAGRAPH_WORDS:
        return [segment]
    base = segment["number"] or "1"
    parts = [{"number": segment["number"], "lines": []}]
    for line in segment["lines"]:
        m = _SUB_PARAGRAPH.match(line)
        if m:
            sub = (
                m.group("num") if m.group("num") and m.group("num").split(".")[0] == base else None
            )
            number = sub or (f"{base}{m.group('letter')}" if m.group("letter") else None)
            if number and number not in {p["number"] for p in parts}:
                parts.append({"number": number, "lines": []})
        parts[-1]["lines"].append(line)
    parts = [p for p in parts if p["lines"]]
    if len(parts) == 1:
        return split_at_clauses(segment, base)
    return parts


def _next_clause_label(label: str | None) -> str:
    if label is None:
        return "a"
    return label[:-1] + chr(ord(label[-1]) + 1) if label[-1] < "z" else label + "a"


def split_at_clauses(segment: dict[str, Any], base: str) -> list[dict[str, Any]]:
    """Long paragraph without sub-paragraph numbers (e.g. a definitions list): group its "(a)", "(b)" ...
    clauses into parts of about CLAUSE_GROUP_WORDS, numbered like "2(a-k)"."""
    groups: list[dict[str, Any]] = [{"clauses": [], "lines": [], "words": 0}]
    prev: str | None = None
    for line in segment["lines"]:
        m = _CLAUSE.match(line)
        # Only the next label in sequence (a, b, ... z, za, ... zz, zza) is a clause boundary; "(ii)"
        # sub-items and inserted clauses like "(ia)" stay inside the current clause.
        if m and m.group(1) != _next_clause_label(prev):
            m = None
        if m:
            prev = m.group(1)
        if m and groups[-1]["clauses"] and groups[-1]["words"] >= CLAUSE_GROUP_WORDS:
            groups.append({"clauses": [], "lines": [], "words": 0})
        if m:
            groups[-1]["clauses"].append(m.group(1))
        groups[-1]["lines"].append(line)
        groups[-1]["words"] += len(line.split())
    if len(groups) == 1:
        return [segment]
    parts = []
    for i, g in enumerate(groups):
        if not g["clauses"]:
            number = base if i == 0 else f"{base}.{i + 1}"
        else:
            first, last = g["clauses"][0], g["clauses"][-1]
            number = f"{base}({first}-{last})" if first != last else f"{base}({first})"
        parts.append({"number": number, "lines": g["lines"]})
    return parts


# --- Table rows from the PDF text layer (for long consolidated rate notifications) -------------------
# Docling's table model scrambles some cells of the 40+ page consolidated notifications (word order
# lost, headers misdetected). pypdf keeps reading order; its columns come out linearised
# ("66 [Heading 9992 ...] Services provided - (a) ... Nil Nil"), which is fine for retrieval.
_FOOTNOTE_START = re.compile(
    r"^\s*\d{1,3}\s+(Inserted|Substituted|Omitted|Amended|Renumbered|Re-numbered|Added|Deleted)\b",
    re.IGNORECASE,
)
_TABLE_END = re.compile(
    r"^\s*\[?\s*\d\s*\.\s+(Definitions|Explanation|This notification|For the purposes)"
    r"|^\s*Explanation\s*\.?\s*-?\s*\W?\s*For (the )?purposes? of this notification",
    re.IGNORECASE,
)
_ROW_START = re.compile(r"^\s*\[?\s*(\d{1,3})\s*([A-Z]{0,2})\.?(?![\w.%/])\s*(.*)$")
_COLUMN_TWO = re.compile(
    r"^\[?\s*(Chapter|Heading|Section|Group|Any|Supply|Services?|Service)\b", re.IGNORECASE
)
_MARKER = re.compile(r"\]\s?\d{1,3}(?![\d.%])")  # amendment footnote markers: "]53" -> "]"
_LEADING_CODE = re.compile(
    r"^\[?\s*(?:(?:Chapter|Heading|Section|Group)\s*\d+\s*(?:,|or|and)?\s*)+\]?\s*", re.IGNORECASE
)


def pdf_text_lines(pdf_path: Path) -> list[str]:
    """PDF text lines with each page's trailing amendment footnotes and bare page numbers removed."""
    import logging

    import pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    out: list[str] = []
    for page in pypdf.PdfReader(str(pdf_path)).pages:
        lines = (page.extract_text() or "").splitlines()
        cut = next((i for i, line in enumerate(lines) if _FOOTNOTE_START.match(line)), len(lines))
        out += [
            line
            for line in lines[:cut]
            if line.strip() and not re.fullmatch(r"\s*\d{1,3}\s*", line)
        ]
    return out


def _next_row(num: int, suffix: str, rest: str, last: tuple[int, str] | None) -> bool:
    if last is None:
        return num == 1
    last_num, last_suffix = last
    if num == last_num:
        return suffix > last_suffix
    return num == last_num + 1 or (last_num < num <= last_num + 4 and bool(_COLUMN_TWO.match(rest)))


def pdf_table_entries(pdf_path: Path, columns: list[str] | None) -> list[dict[str, Any]]:
    """Split the main Table of a notification into one entry per serial number, using row order
    (each new serial must follow the previous one) to tell row starts from numbers inside cells."""
    lines = pdf_text_lines(pdf_path)
    start = next((i for i, line in enumerate(lines) if line.strip() == "Table"), None)
    if start is None:
        return []
    rows: list[dict[str, Any]] = []
    last: tuple[int, str] | None = None
    for line in lines[start + 1 :]:
        if rows and _TABLE_END.match(line):
            break
        m = _ROW_START.match(line)
        if m and _next_row(int(m.group(1)), m.group(2).upper(), m.group(3), last):
            last = (int(m.group(1)), m.group(2).upper())
            rows.append({"number": f"{m.group(1)}{m.group(2).upper()}", "lines": [m.group(3)]})
        elif rows:
            rows[-1]["lines"].append(line)
    header = f"Columns: {' | '.join(columns)}" if columns else ""
    entries = []
    for row in rows:
        body = _MARKER.sub("]", clean_line(" ".join(row["lines"])))
        if not re.sub(r"[\[\]*\s.]", "", body):
            continue  # entry omitted by amendment: "[***]"
        text = "\n".join(t for t in (header, f"Sl. No. {row['number']}: {body}") if t)
        entries.append(
            {
                "number": row["number"],
                "text": text,
                "title": short_title(_LEADING_CODE.sub("", body)),
            }
        )
    return entries


_FAQ_ROW = re.compile(r"^\s{0,6}(\d{1,2})\.\s+(\S.*)$")
# Character column where the answer column starts in pypdf's layout text of a two-column FAQ table.
FAQ_ANSWER_COLUMN = 38
_FAQ_TRAILER = re.compile(r"\(The highlighted portions.*$|\*{3,}.*$")


def pdf_faq_entries(pdf_path: Path) -> list[dict[str, Any]]:
    """One entry per row of a two-column (question | answer) FAQ table. Docling reads such tables out of
    order (answers end up under the wrong question), so the columns are cut from pypdf's layout text."""
    import logging

    import pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    rows: list[dict[str, Any]] = []
    last = 0
    for page in pypdf.PdfReader(str(pdf_path)).pages:
        for line in (page.extract_text(extraction_mode="layout") or "").splitlines():
            if not line.strip() or re.fullmatch(r"\s*Page \d+ of \d+\s*", line):
                continue
            m = _FAQ_ROW.match(line)
            if m and int(m.group(1)) in (last, last + 1):  # the source repeats "11."
                last = int(m.group(1))
                rows.append({"number": m.group(1), "q": [], "a": []})
                line = " " * (len(line) - len(m.group(2))) + m.group(2)
            if not rows:
                continue
            cut = next(
                (i for i in range(FAQ_ANSWER_COLUMN, len(line)) if line[i] != " " and line[i - 1] == " "),
                len(line),
            )
            rows[-1]["q"].append(line[:cut])
            rows[-1]["a"].append(line[cut:])
    entries = []
    for r in rows:
        q, a = (_FAQ_TRAILER.sub("", clean_line(" ".join(" ".join(part).split()))).strip() for part in (r["q"], r["a"]))
        entries.append({"number": r["number"], "text": f"Question: {q}\nAnswer: {a}", "title": short_title(q)})
    return entries


def build_records(src: dict[str, Any], markdown: str) -> list[dict[str, Any]]:
    iid = src["instrument_id"]
    kind = src["kind"]
    # FAQs and other documents carry their own title; notifications/circulars are named by number.
    title = src.get("title") or f"{'Notification' if kind == 'notification' else 'Circular'} No. {src['number']}"
    segments, entries = parse_document(markdown)
    if src.get("table_text") == "pdf":
        entries = pdf_table_entries(Path(src["pdf"]), first_header(markdown))
    elif src.get("table_text") == "faq_pdf":
        segments, entries = [], pdf_faq_entries(Path(src["pdf"]))

    def record(ptype: str, number: str, ptitle: str, raw_text: str, order: int) -> dict[str, Any]:
        return {
            "id": f"{iid}.{ptype}.{number.lower()}",
            "instrument_id": iid,
            "instrument_title": title,
            "instrument_subject": src["subject"],
            "instrument_kind": kind,
            "provision_type": ptype,
            "provision_number": number,
            "provision_title": ptitle,
            "chapter_number": None,
            "chapter_title": None,
            "status": "current_consolidated_source"
            if src.get("law_position", "").startswith("as amended")
            else "as_issued",
            "law_position": src.get("law_position", "as issued"),
            "issue_date": src["date"],
            "source_url": src["url"],
            "source_file": src.get("markdown"),
            "source_order": order,
            "raw_text": raw_text,
            "source_hash": sha256_text(raw_text),
            "character_count": len(raw_text),
            "word_count": len(raw_text.split()),
        }

    records: list[dict[str, Any]] = []
    explicit_one = any(s["number"] == "1" for s in segments)
    segments = [part for s in segments for part in split_long_segment(s)]
    # Short trailing segments ("This notification shall come into force on ...") are merged into the
    # previous one rather than becoming near-empty retrieval targets.
    merged: list[dict[str, Any]] = []
    for s in segments:
        text = "\n\n".join(s["lines"])
        if merged and len(text.split()) < 25:
            merged[-1]["text"] += "\n\n" + text
        else:
            merged.append({"number": s["number"], "text": text})
    for order, s in enumerate(merged, start=1):
        number = s["number"] or ("intro" if explicit_one else "1")
        records.append(record("paragraph", number, short_title(s["text"]), s["text"], order))

    seen: Counter[str] = Counter()
    running = 0
    for e in entries:
        running += 1
        number = e["number"] or str(running)
        seen[number] += 1
        if seen[number] > 1:
            number = f"{number}_{seen[number]}"
        records.append(record("entry", number, e["title"], e["text"], len(records) + 1))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=SOURCES_DIR / "manifest.json")
    ap.add_argument(
        "--output", type=Path, default=Path("seed/docs/gst/build/notifications_raw.jsonl")
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for src in manifest:
        records = build_records(src, Path(src["markdown"]).read_text(encoding="utf-8"))
        kinds = Counter(r["provision_type"] for r in records)
        words = sum(r["word_count"] for r in records)
        print(
            f"{src['instrument_id']:32} {kinds.get('paragraph', 0):3} paragraphs {kinds.get('entry', 0):4} entries {words:6} words"
        )
        out.extend(records)

    ids = [r["id"] for r in out]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise SystemExit(f"Duplicate IDs: {dupes[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} records -> {args.output}")


if __name__ == "__main__":
    main()
