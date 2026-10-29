"""Stage 1a: split the ICAI bare-law Markdown (GST_Act.md) into one record per provision.

Generalises the original seed/scripts/stage1.py (CGST Act + CGST Rules only) to every instrument the
golden set needs from the same book: CGST Act sections and Schedules I-III, CGST Rules, IGST Act,
IGST Rules and UTGST Act. For CGST Act/Rules the output is identical to stage1.py's, so existing
corpus records (and their enrichment) stay valid — check with `--compare-corpus`.

Instruments must appear in the order listed in INSTRUMENTS; a heading is only accepted once the
previous instrument has started, which keeps the book's tables of contents from being misread as
instrument starts.

    uv run python -m scripts.gst_pipeline.parse_bare_law --input seed/scripts/GST_Act.md \
        --output seed/docs/gst/build/bare_law_raw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# Ordered as they appear in the book. `end` = normalised heading that closes the last instrument.
INSTRUMENTS: list[dict[str, Any]] = [
    {
        "heading": "the central goods and services tax act, 2017",
        "instrument_id": "cgst_act_2017",
        "instrument_title": "The Central Goods and Services Tax Act, 2017",
        "allowed_type": "section",
        "has_schedules": True,
    },
    {
        "heading": "the central goods and services tax rules, 2017",
        "instrument_id": "cgst_rules_2017",
        "instrument_title": "The Central Goods and Services Tax Rules, 2017",
        "allowed_type": "rule",
    },
    {
        "heading": "the integrated goods and services tax act, 2017",
        "instrument_id": "igst_act_2017",
        "instrument_title": "The Integrated Goods and Services Tax Act, 2017",
        "allowed_type": "section",
    },
    {
        "heading": "the integrated goods and services tax rules, 2017",
        "instrument_id": "igst_rules_2017",
        "instrument_title": "The Integrated Goods and Services Tax Rules, 2017",
        "allowed_type": "rule",
    },
    {
        "heading": "the union territory goods and services tax act, 2017",
        "instrument_id": "utgst_act_2017",
        "instrument_title": "The Union Territory Goods and Services Tax Act, 2017",
        "allowed_type": "section",
    },
]
STOP_HEADING = "the goods and services tax (compensation to states) act, 2017"
# Editorial notes the book prints after Schedule III, before the CGST Rules.
SCHEDULES_END = "points to note"

CHAPTER_PATTERN = re.compile(
    r"^##\s*CHAPTER\s+([IVXLCDM]+)\s*(?::|-|–|—)\s*(.+?)\s*$", re.IGNORECASE
)
CHAPTER_NO_SEPARATOR_PATTERN = re.compile(
    r"^##\s*CHAPTER\s+([IVXLCDM]+)\s+(.+?)\s*$", re.IGNORECASE
)
SCHEDULE_PATTERN = re.compile(r"^##\s*SCHEDULE\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
SCHEDULE_SEE_PATTERN = re.compile(r"^(?:##\s*)?\[\s*See\s+section", re.IGNORECASE)

# A line is a provision heading only when the line itself starts like one, so references such as
PROVISION_LINE_START = re.compile(
# "under section 10" never open a new provision.
    r"^(?:##\s*)?(?:\d+\s*\[?\s*)?(?:Section|Rule)\s+\d+\s*[A-Z]?\s*\.",
    re.IGNORECASE,
)
# Also splits malformed combined lines such as "## Rule 70. [****] Rule 71. [****]".
PROVISION_TOKEN = re.compile(
    r"\b(?P<type>Section|Rule)\s+(?P<number>\d+\s*[A-Z]?)\s*\.", re.IGNORECASE
)

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


def normalize_heading(line: str) -> str:
    value = re.sub(r"^#+\s*", "", line.strip())
    return re.sub(r"\s+", " ", value).strip().lower()


def clean_title(title: str) -> str:
    """Light cleanup only; the legal wording is otherwise retained."""
    value = re.sub(r"\s+", " ", title).strip()
    value = re.sub(r"\s*\.\s*-\s*$", "", value)
    value = re.sub(r"\s*-\s*$", "", value)
    value = re.sub(r"\s*\.\s*$", "", value)
    return value.strip()


def clean_chapter_title(title: str) -> str:
    value = re.sub(r"\s+", " ", title).strip()
    value = re.sub(r"\s*\[\s*RULES?\b.*?\]\s*$", "", value, flags=re.IGNORECASE)
    return value.strip()


def normalize_body(lines: list[str]) -> str:
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_wholly_omitted(title: str) -> bool:
    """True only when the *whole provision* is shown as omitted, not just words in its title."""
    if re.search(r"\bomitted\b", title, re.IGNORECASE):
        return True
    if "***" not in title:
        return False
    remainder = re.sub(r"\d+", "", title)
    remainder = re.sub(r"[\[\]()*.,:;\-–—_/\\\s]", "", remainder)
    return remainder == ""


def extract_heading_segments(line: str) -> list[dict[str, str | None]]:
    """Parse one heading-like line into one or more provision headings."""
    stripped = line.strip()
    if not PROVISION_LINE_START.match(stripped):
        return []
    matches = list(PROVISION_TOKEN.finditer(stripped))
    if not matches:
        return []

    footnote_match = re.search(r"(\d+)\s*\[?\s*$", stripped[: matches[0].start()])
    first_footnote = footnote_match.group(1) if footnote_match else None

    segments: list[dict[str, str | None]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        segments.append(
            {
                "provision_type": match.group("type").lower(),
                "provision_number": re.sub(r"\s+", "", match.group("number")).upper(),
                "provision_title": clean_title(stripped[match.end() : end]),
                # Keep the Markdown/prefix on the first segment for debugging.
                "raw_heading": stripped[:end].strip()
                if index == 0
                else stripped[match.start() : end].strip(),
                "heading_footnote_marker": first_footnote if index == 0 else None,
            }
        )
    return segments


def provision_id(instrument: dict[str, Any], provision_type: str, provision_number: str) -> str:
    number = (
        str(ROMAN.get(provision_number, provision_number))
        if provision_type == "schedule"
        else provision_number
    )
    return f"{instrument['instrument_id']}.{provision_type}.{number.lower()}"


def build_record(
    *,
    instrument: dict[str, Any],
    chapter: dict[str, str] | None,
    provision_type: str,
    provision_number: str,
    provision_title: str,
    raw_heading: str,
    heading_footnote_marker: str | None,
    source_line_start: int,
    source_order: int,
    status: str,
) -> dict[str, Any]:
    return {
        "id": provision_id(instrument, provision_type, provision_number),
        "instrument_id": instrument["instrument_id"],
        "instrument_title": instrument["instrument_title"],
        "provision_type": provision_type,
        "provision_number": provision_number,
        "provision_title": provision_title,
        "chapter_number": chapter["chapter_number"] if chapter else None,
        "chapter_title": chapter["chapter_title"] if chapter else None,
        "status": status,
        "raw_heading": raw_heading,
        "heading_footnote_marker": heading_footnote_marker,
        "source_line_start": source_line_start,
        "source_order": source_order,
    }


def parse_bare_law(markdown: str) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    next_instrument = 0  # index into INSTRUMENTS of the next heading we accept
    current_instrument: dict[str, Any] | None = None
    current_chapter: dict[str, str] | None = None
    current_record: dict[str, Any] | None = None
    in_schedule = False
    body_lines: list[str] = []
    seen_ids: set[str] = set()
    source_order = 0

    def save_current_record(source_line_end: int) -> None:
        nonlocal current_record, body_lines
        if current_record is None:
            return
        raw_text = normalize_body(body_lines)
        current_record["raw_text"] = raw_text
        current_record["source_hash"] = sha256_text(raw_text)
        current_record["source_line_end"] = source_line_end
        current_record["character_count"] = len(raw_text)
        current_record["word_count"] = len(raw_text.split())
        records.append(current_record)
        seen_ids.add(current_record["id"])
        current_record = None
        body_lines = []

    lines = markdown.splitlines()
    for line_number, original_line in enumerate(lines, start=1):
        stripped = original_line.strip()
        normalized = normalize_heading(original_line)

        if (
            next_instrument < len(INSTRUMENTS)
            and normalized == INSTRUMENTS[next_instrument]["heading"]
        ):
            save_current_record(line_number - 1)
            current_instrument = INSTRUMENTS[next_instrument]
            next_instrument += 1
            current_chapter = None
            in_schedule = False
            print(
                f"Detected instrument at line {line_number}: {current_instrument['instrument_title']}"
            )
            continue

        if normalized == STOP_HEADING and next_instrument == len(INSTRUMENTS):
            save_current_record(line_number - 1)
            break

        if current_instrument is None:
            continue

        schedule_match = SCHEDULE_PATTERN.match(stripped)
        if current_instrument["allowed_type"] == "section" and schedule_match:
            save_current_record(line_number - 1)
            if not current_instrument.get("has_schedules"):
                current_instrument = None
                continue
            in_schedule = True
            source_order += 1
            current_record = build_record(
                instrument=current_instrument,
                chapter=None,
                provision_type="schedule",
                provision_number=schedule_match.group(1).upper(),
                provision_title="",  # filled from the first all-caps heading below
                raw_heading=stripped,
                heading_footnote_marker=None,
                source_line_start=line_number,
                source_order=source_order,
                status="current_consolidated_source",
            )
            continue

        if in_schedule:
            if normalized == SCHEDULES_END:
                save_current_record(line_number - 1)
                current_instrument = None
                in_schedule = False
                continue
            if current_record is not None and SCHEDULE_SEE_PATTERN.match(stripped):
                current_record["schedule_reference"] = re.sub(r"^#+\s*", "", stripped)
                continue
            if (
                current_record is not None
                and not current_record["provision_title"]
                and stripped.startswith("##")
            ):
                current_record["provision_title"] = re.sub(
                    r"\s+\d+\s*\[", " [", clean_title(normalized)
                ).capitalize()
                continue
            if current_record is not None:
                body_lines.append(original_line)
            continue

        chapter_match = CHAPTER_PATTERN.match(stripped) or (
            CHAPTER_NO_SEPARATOR_PATTERN.match(stripped)
            if current_instrument["instrument_id"] != "cgst_act_2017"
            and current_instrument["instrument_id"] != "cgst_rules_2017"
            else None
        )
        if chapter_match:
            save_current_record(line_number - 1)
            current_chapter = {
                "chapter_number": chapter_match.group(1).upper(),
                "chapter_title": clean_chapter_title(chapter_match.group(2)),
            }
            continue

        heading_segments = extract_heading_segments(original_line)
        if heading_segments:
            for segment in heading_segments:
                provision_type = str(segment["provision_type"])
                provision_number = str(segment["provision_number"])
                provision_title = str(segment["provision_title"])
                if provision_type != current_instrument["allowed_type"]:
                    continue

                pid = provision_id(current_instrument, provision_type, provision_number)
                if pid in seen_ids or (current_record is not None and pid == current_record["id"]):
                    warnings.append(
                        f"Ignored duplicate/historical heading at line {line_number}: {segment['raw_heading']}"
                    )
                    continue

                save_current_record(line_number - 1)
                source_order += 1
                omitted = is_wholly_omitted(provision_title)
                record = build_record(
                    instrument=current_instrument,
                    chapter=current_chapter,
                    provision_type=provision_type,
                    provision_number=provision_number,
                    provision_title="Omitted" if omitted else provision_title,
                    raw_heading=str(segment["raw_heading"]),
                    heading_footnote_marker=(
                        str(segment["heading_footnote_marker"])
                        if segment["heading_footnote_marker"] is not None
                        else None
                    ),
                    source_line_start=line_number,
                    source_order=source_order,
                    status="omitted" if omitted else "current_consolidated_source",
                )
                if omitted:
                    record.update(
                        raw_text="",
                        source_hash=sha256_text(""),
                        source_line_end=line_number,
                        character_count=0,
                        word_count=0,
                    )
                    records.append(record)
                    seen_ids.add(record["id"])
                    current_record = None
                    body_lines = []
                else:
                    current_record = record
                    body_lines = []
            continue

        if current_record is not None:
            body_lines.append(original_line)
    else:
        save_current_record(len(lines))

    return records, warnings


EXPECTED_IDS = {
    "cgst_act_2017.section.1",
    "cgst_act_2017.section.10",
    "cgst_act_2017.section.74",
    "cgst_act_2017.section.174",
    "cgst_act_2017.schedule.1",
    "cgst_act_2017.schedule.2",
    "cgst_act_2017.schedule.3",
    "cgst_rules_2017.rule.1",
    "cgst_rules_2017.rule.164",
    "igst_act_2017.section.2",
    "igst_act_2017.section.4",
    "igst_act_2017.section.7",
    "igst_act_2017.section.13",
    "igst_act_2017.section.16",
    "igst_act_2017.section.25",
    "igst_rules_2017.rule.1",
    "utgst_act_2017.section.1",
    "utgst_act_2017.section.26",
}


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r["id"] for r in records]
    duplicate_ids = sorted(i for i, c in Counter(ids).items() if c > 1)
    contamination = []
    for rid, markers in {
        "cgst_act_2017.section.174": ["## SCHEDULE I", "## Rule 1."],
        "cgst_act_2017.schedule.3": ["POINTS TO NOTE", "## Rule 1."],
        "cgst_rules_2017.rule.164": ["INTEGRATED GOODS AND SERVICES TAX ACT"],
        "igst_act_2017.section.25": ["## Rule 1."],
        "utgst_act_2017.section.26": ["COMPENSATION TO STATES"],
    }.items():
        rec = next((r for r in records if r["id"] == rid), None)
        for m in markers:
            if rec and m.lower() in rec.get("raw_text", "").lower():
                contamination.append(f"{rid} contains {m!r}")

    report: dict[str, Any] = {
        "total_records": len(records),
        "by_instrument_and_type": dict(
            Counter(f"{r['instrument_id']}/{r['provision_type']}" for r in records)
        ),
        "duplicate_ids": duplicate_ids,
        "missing_expected_ids": sorted(EXPECTED_IDS - set(ids)),
        "contamination": contamination,
        "records_without_chapter": [r["id"] for r in records if r.get("chapter_number") is None],
        "empty_titles": [r["id"] for r in records if not r.get("provision_title")],
        "omitted_records": [r["id"] for r in records if r.get("status") == "omitted"],
    }
    errors = []
    if duplicate_ids:
        errors.append(f"Duplicate IDs: {duplicate_ids}")
    if report["missing_expected_ids"]:
        errors.append(f"Expected provisions missing: {report['missing_expected_ids']}")
    if contamination:
        errors.append(f"Content from a later instrument leaked in: {contamination}")
    report["validation_errors"] = errors
    return report


RAW_FIELDS = (
    "instrument_id",
    "instrument_title",
    "provision_type",
    "provision_number",
    "provision_title",
    "chapter_number",
    "chapter_title",
    "status",
    "raw_text",
    "source_hash",
)


def compare_with_corpus(records: list[dict[str, Any]], corpus_path: Path) -> list[str]:
    """Raw-field differences between this parse and an existing (enriched) corpus, for shared IDs."""
    parsed = {r["id"]: r for r in records}
    diffs = []
    with corpus_path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            old = json.loads(line)
            new = parsed.get(old["id"])
            if new is None:
                diffs.append(f"{old['id']}: missing from new parse")
                continue
            for field in RAW_FIELDS:
                if old.get(field) != new.get(field):
                    diffs.append(f"{old['id']}: {field} differs")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=Path("seed/scripts/GST_Act.md"))
    ap.add_argument("--output", type=Path, default=Path("seed/docs/gst/build/bare_law_raw.jsonl"))
    ap.add_argument(
        "--report", type=Path, default=Path("seed/docs/gst/build/bare_law_parse_report.json")
    )
    ap.add_argument(
        "--compare-corpus",
        type=Path,
        default=None,
        help="Existing corpus JSONL; fail if any shared provision's raw fields changed",
    )
    args = ap.parse_args()

    records, warnings = parse_bare_law(args.input.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("No provisions extracted")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = validate_records(records)
    report["warnings"] = warnings
    if args.compare_corpus:
        report["corpus_diffs"] = compare_with_corpus(records, args.compare_corpus)
        if report["corpus_diffs"]:
            report["validation_errors"].append(
                f"{len(report['corpus_diffs'])} differences from {args.compare_corpus}"
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{report['total_records']} records -> {args.output}")
    for k, v in sorted(report["by_instrument_and_type"].items()):
        print(f"  {k:28} {v}")
    print(f"Warnings: {len(warnings)}  (see {args.report})")
    if report["validation_errors"]:
        for e in report["validation_errors"]:
            print(f"ERROR: {e}")
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
