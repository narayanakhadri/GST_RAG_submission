"""Stage 2: enrich raw provision records with deterministic metadata + LLM retrieval metadata.

Cleaned-up successor of seed/scripts/stage2_enrich_gst_groq.py (which built the current corpus):
same cleaning, reference extraction, prompt (PROMPT_VERSION unchanged) and retrieval-text format, so
records built here are interchangeable with the existing ones. Differences:

- The LLM call goes through the OpenAI SDK pointed at Groq's OpenAI-compatible endpoint, using
  GROQ_API_KEY / groq_base_url from .env via app.config (no keys in code). `--provider openai` uses
  OPENAI_API_KEY and settings.llm_model_grader instead — Groq's free tier caps gpt-oss-120b at ~200k
  tokens/day (~150 provisions). `enrichment.provider`/`model` record what was actually called.
- Cross-references resolve to the right instrument ("section 10" inside the IGST Act is
  igst_act_2017.section.10, not CGST), and follow an explicit Act name next to the reference.
- `--reuse <corpus.jsonl>` copies any record whose id and source_hash match an already-enriched record,
  so only new or changed provisions cost an LLM call.
- Checkpoint/resume: records already in --output are skipped.

    uv run python -m scripts.gst_pipeline.enrich --input seed/docs/gst/build/raw_all.jsonl \
        --output seed/docs/gst/build/enriched.jsonl --reuse seed/docs/gst/gst_provisions_enriched_groq.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROMPT_VERSION = "gst-enrichment-v2-groq"
DEFAULT_MAX_COMPLETION_TOKENS = 2048  # what the existing corpus was generated with
DEFAULT_REASONING_EFFORT = "low"
# The model the existing corpus was enriched with — kept fixed (not settings.groq_model_*) so new records
# match the old ones.
DEFAULT_MODEL = "openai/gpt-oss-120b"

LIST_FIELDS = (
    "topics",
    "keywords",
    "applies_to",
    "conditions",
    "exceptions",
    "obligations",
    "procedural_steps",
    "deadlines",
    "thresholds",
    "possible_questions",
)
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        **{f: {"type": "array", "items": {"type": "string"}} for f in LIST_FIELDS},
    },
    "required": ["summary", *LIST_FIELDS],
    "additionalProperties": False,
}
EMPTY_METADATA: dict[str, Any] = {"summary": "", **{f: [] for f in LIST_FIELDS}}

# ---------------------------------------------------------------------------
# Deterministic cleaning and reference extraction
# ---------------------------------------------------------------------------

HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
PAGE_CROSS_REFERENCE_PATTERN = re.compile(
    r"^\s*\[?\s*(?:Rules?|Sections?)\b.*?\b(?:page|pages|page\s+no\.)\b.*?\]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
RUNNING_HEADER_PATTERN = re.compile(
    r"^\s*(?:The\s+)?Central\s+Goods\s+and\s+Services\s+Tax\s+(?:Act|Rules),?\s*2017\s*$",
    re.IGNORECASE,
)
# Publisher/amendment-history notes, not consolidated clause text. Bracketed inserted/substituted
# legal text is kept: it normally sits in a different paragraph and is part of the current law.
AMENDMENT_NOTE_PATTERN = re.compile(
    r"^\s*\d+[A-Z]?\s+(?:Inserted|Substituted|Omitted|Amended|Renumbered|Re-numbered|Notified|Effective|"
    r"Replaced|Added|Deleted|Corrected|Modified|Prior\s+to|Before\s+its)\b",
    re.IGNORECASE | re.DOTALL,
)
AMENDMENT_NOTE_WITH_INSTRUMENT_PATTERN = re.compile(
    r"^\s*\d+[A-Z]?\s+.*?\b(?:vide|w\.e\.f\.|prior\s+to\s+its|before\s+it\s+was\s+read)\b",
    re.IGNORECASE | re.DOTALL,
)
REF_BLOCK_PATTERN = re.compile(
    r"\b(?P<kind>sections?|rules?)\s+(?P<body>\d+[A-Z]?(?:\s*\([^)]*\))*"
    r"(?:\s*(?:,|and|or|to|[-–—])\s*\d+[A-Z]?(?:\s*\([^)]*\))*)*)",
    re.IGNORECASE,
)
REF_NUMBER_PATTERN = re.compile(r"\d+[A-Z]?(?:\s*\([^)]*\))*", re.IGNORECASE)
FORM_PATTERN = re.compile(r"\bFORM\s+GST\s+[A-Z]{2,10}\s*[-–—]\s*\d+[A-Z]?\b", re.IGNORECASE)
SCHEDULE_PATTERN = re.compile(r"\bSchedule\s+(?P<number>[IVXLCDM]+|\d+)\b", re.IGNORECASE)
NOTIFICATION_PATTERN = re.compile(
    r"\bNotification\s+(?:No\.?\s*)?(?P<number>\d+/\d{4}(?:\s*[-–—]\s*[A-Z][A-Z .()]*)?)",
    re.IGNORECASE,
)
# Act named right after a reference: "section 20 of the Integrated Goods and Services Tax Act".
_ACT_AFTER_REF = re.compile(
    r"^\W{0,3}(?:of\s+(?:the\s+)?(?:said\s+)?)?(?P<act>Integrated\s+Goods\s+and\s+Services\s+Tax|IGST|"
    r"Central\s+Goods\s+and\s+Services\s+Tax|CGST|Union\s+Territory\s+Goods\s+and\s+Services\s+Tax|UTGST|"
    r"Customs|Income[- ]tax)",
    re.IGNORECASE,
)
# Which Act / Rules a bare "section N" / "rule N" means, by the record's own instrument.
_DEFAULT_ACT = {
    "igst_act_2017": "igst_act_2017",
    "igst_rules_2017": "igst_act_2017",
    "utgst_act_2017": "utgst_act_2017",
}
_DEFAULT_RULES = {"igst_rules_2017": "igst_rules_2017"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def clean_text_for_analysis(raw_text: str) -> tuple[str, list[dict[str, str]]]:
    """Text sent to the LLM; raw_text is untouched. Only high-confidence editorial material is removed,
    and every excluded paragraph is returned for auditability."""
    kept: list[str] = []
    excluded: list[dict[str, str]] = []
    for paragraph in split_paragraphs(HTML_COMMENT_PATTERN.sub("", raw_text)):
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        reason = None
        if PAGE_CROSS_REFERENCE_PATTERN.match(normalized):
            reason = "publisher_page_cross_reference"
        elif PAGE_NUMBER_PATTERN.match(normalized):
            reason = "page_number"
        elif RUNNING_HEADER_PATTERN.match(normalized):
            reason = "running_header"
        elif AMENDMENT_NOTE_PATTERN.match(normalized):
            reason = "amendment_footnote"
        elif AMENDMENT_NOTE_WITH_INSTRUMENT_PATTERN.match(normalized):
            reason = "amendment_history_note"
        if reason:
            excluded.append({"reason": reason, "text": paragraph})
        else:
            kept.append(paragraph)
    analysis_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(kept).strip())
    return analysis_text, excluded


def provision_number_key(value: str) -> tuple[int, str, str]:
    compact = re.sub(r"\s+", "", str(value).upper())
    match = re.match(r"^(\d+)([A-Z]*)(.*)$", compact)
    if not match:
        return (10**9, compact, "")
    return (int(match.group(1)), match.group(2), match.group(3))


def _target_act(explicit: str | None, record: dict[str, Any]) -> str:
    if explicit:
        e = explicit.lower()
        if e.startswith(("integrated", "igst")):
            return "igst_act_2017"
        if e.startswith(("union", "utgst")):
            return "utgst_act_2017"
        if e.startswith(("central", "cgst")):
            return "cgst_act_2017"
        return "external"
    return _DEFAULT_ACT.get(str(record.get("instrument_id")), "cgst_act_2017")


def extract_explicit_references(analysis_text: str, record: dict[str, Any]) -> dict[str, Any]:
    """Explicit Section/Rule/Form/Schedule/Notification references, without an LLM."""
    sections: set[tuple[str, str]] = set()  # (target_id, number as written, e.g. "16(2)")
    rules: set[tuple[str, str]] = set()
    own_id = record.get("id")
    for match in REF_BLOCK_PATTERN.finditer(analysis_text):
        kind = match.group("kind").lower()
        act = _ACT_AFTER_REF.match(analysis_text[match.end() : match.end() + 80])
        numbers = {
            re.sub(r"\s+", "", n).upper() for n in REF_NUMBER_PATTERN.findall(match.group("body"))
        }
        for number in numbers:
            base = re.match(r"\d+[A-Z]?", number).group(0).lower()
            if kind.startswith("section"):
                target = _target_act(act.group("act") if act else None, record)
                if target != "external":
                    sections.add((f"{target}.section.{base}", number))
            else:
                rules_instr = _DEFAULT_RULES.get(
                    str(record.get("instrument_id")), "cgst_rules_2017"
                )
                rules.add((f"{rules_instr}.rule.{base}", number))
    # No self-edge when the provision repeats its own label.
    sections = {ref for ref in sections if ref[0] != own_id}
    rules = {ref for ref in rules if ref[0] != own_id}

    def as_list(refs: set[tuple[str, str]]) -> list[dict[str, str]]:
        ordered = sorted(
            refs, key=lambda ref: (ref[0].rsplit(".", 2)[0], provision_number_key(ref[1]))
        )
        return [{"number": number, "target_id": target} for target, number in ordered]

    return {
        "sections": as_list(sections),
        "rules": as_list(rules),
        "forms": sorted(
            {
                re.sub(r"\s*[-–—]\s*", "-", re.sub(r"\s+", " ", v.upper())).strip()
                for v in FORM_PATTERN.findall(analysis_text)
            }
        ),
        "schedules": sorted(
            {m.group("number").upper() for m in SCHEDULE_PATTERN.finditer(analysis_text)}
        ),
        "notifications": sorted(
            {
                re.sub(r"\s+", " ", m.group("number")).strip()
                for m in NOTIFICATION_PATTERN.finditer(analysis_text)
            }
        ),
    }


# ---------------------------------------------------------------------------
# Long-text handling
# ---------------------------------------------------------------------------


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split an unusually long paragraph, preferring line and sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    units = [u.strip() for u in paragraph.splitlines() if u.strip()]
    if len(units) <= 1:
        units = [
            u.strip() for u in re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(\[])", paragraph) if u.strip()
        ]
    if len(units) <= 1:
        return [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            chunks.extend(unit[i : i + max_chars] for i in range(0, len(unit), max_chars))
            continue
        addition = len(unit) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append("\n".join(current))
            current, current_len = [unit], len(unit)
        else:
            current.append(unit)
            current_len += addition
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_for_llm(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [
        p for para in split_paragraphs(text) for p in split_long_paragraph(para, max_chars)
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [paragraph], len(paragraph)
        else:
            current.append(paragraph)
            current_len += addition
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Prompting and validation
# ---------------------------------------------------------------------------


def _provision_line(record: dict[str, Any]) -> str:
    return f"{str(record.get('provision_type', '')).title()} {record.get('provision_number')} - {record.get('provision_title')}"


def build_extraction_prompt(
    record: dict[str, Any], text: str, chunk_index: int, chunk_count: int
) -> str:
    partial_instruction = (
        "This is the complete provision."
        if chunk_count == 1
        else (
            f"This is part {chunk_index} of {chunk_count} of a long provision. Extract only information "
            "supported by this part. Do not imply that the part represents the whole provision."
        )
    )
    return f"""
You are extracting retrieval metadata from Indian GST legislation.

STRICT RULES:
1. Use only the legal text enclosed below.
2. Do not use outside legal knowledge.
3. Do not invent thresholds, deadlines, exceptions, procedures, forms, or legal effects.
4. Preserve legally important qualifications such as "subject to", "provided that",
   "unless", "notwithstanding", and "except".
5. Summarise the provision; do not give legal advice.
6. Each list entry must be a short, self-contained string.
7. If a field is unsupported by the supplied text, return an empty list.
8. Generate 3 to 8 natural-language questions only when they are answerable from
   the supplied text.
9. Return exactly one JSON object and no Markdown or commentary.

{partial_instruction}

Required JSON shape:
{json.dumps(EMPTY_METADATA, ensure_ascii=False, indent=2)}

Instrument: {record.get("instrument_title")}
Chapter: {record.get("chapter_number")} - {record.get("chapter_title")}
Provision: {_provision_line(record)}

--- START LEGAL TEXT ---
{text}
--- END LEGAL TEXT ---
""".strip()


def build_consolidation_prompt(
    record: dict[str, Any], partial_results: Sequence[dict[str, Any]]
) -> str:
    return f"""
You are consolidating metadata extracted from separate parts of one Indian GST
legal provision.

STRICT RULES:
1. Use only the partial extraction objects below.
2. Remove duplicates and near-duplicates.
3. Do not add legal information not present in the partial objects.
4. Produce one coherent summary of no more than 180 words.
5. Preserve conditions, exceptions, and legal qualifications.
6. Return exactly one JSON object and no Markdown or commentary.

Required JSON shape:
{json.dumps(EMPTY_METADATA, ensure_ascii=False, indent=2)}

Instrument: {record.get("instrument_title")}
Chapter: {record.get("chapter_number")} - {record.get("chapter_title")}
Provision: {_provision_line(record)}

--- START PARTIAL EXTRACTIONS ---
{json.dumps(partial_results, ensure_ascii=False, indent=2)}
--- END PARTIAL EXTRACTIONS ---
""".strip()


def extract_json_object(content: str) -> dict[str, Any]:
    value = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")
    return parsed


def normalize_string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"Every item in {field} must be a string")
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            result.append(cleaned)
    return result


def validate_and_normalize_metadata(value: dict[str, Any]) -> dict[str, Any]:
    missing = {"summary", *LIST_FIELDS} - set(value)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    if not isinstance(value.get("summary"), str):
        raise ValueError("summary must be a string")
    return {
        "summary": re.sub(r"\s+", " ", value["summary"]).strip(),
        **{f: normalize_string_list(value.get(f), f) for f in LIST_FIELDS},
    }


_CLIENT: Any | None = None


def provider_endpoint(provider: str) -> str:
    from app.config import settings

    return settings.groq_base_url if provider == "groq" else "https://api.openai.com/v1"


def get_client(provider: str) -> Any:
    """OpenAI SDK client — pointed at Groq's OpenAI-compatible endpoint for provider "groq" (same as
    app/services/llm_service.py), or at OpenAI itself. Keys come from .env via app.config."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        from app.config import settings

        key = settings.groq_api_key if provider == "groq" else settings.openai_api_key
        if not key:
            raise RuntimeError(f"No API key for provider {provider!r} (set it in .env)")
        _CLIENT = OpenAI(api_key=key, base_url=provider_endpoint(provider), timeout=120)
    return _CLIENT


def _supports_reasoning_effort(model: str) -> bool:
    return model.startswith(("openai/gpt-oss", "gpt-oss", "o1", "o3", "o4", "gpt-5"))


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return max(0.0, float(headers.get("retry-after"))) if headers is not None else None
    except (TypeError, ValueError):
        return None


def call_llm_json(*, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    system_message = (
        "You are a precise legal-information extraction component. "
        "Use only supplied text and return data matching the required JSON schema."
    )
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        if last_error is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous response failed local validation. Correct it. Validation error: {last_error}. "
                        "Return only the required JSON object."
                    ),
                }
            )
        try:
            extra = (
                {"reasoning_effort": args.reasoning_effort} if _supports_reasoning_effort(args.model) else {}
            )
            response = get_client(args.provider).chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                top_p=1,
                **extra,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "gst_retrieval_metadata",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    },
                },
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("empty response content")
            return validate_and_normalize_metadata(extract_json_object(content))
        except Exception as exc:  # noqa: BLE001 — API, network, JSON or validation failure
            last_error = exc
            if attempt < args.max_attempts:
                header_wait = _retry_after_seconds(exc)
                wait = header_wait + 0.25 if header_wait is not None else args.retry_wait * attempt
                print(
                    f"    attempt {attempt}/{args.max_attempts} failed: {exc}; retrying in {wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
    raise RuntimeError(f"No valid JSON after {args.max_attempts} attempts: {last_error}")


def enrich_text(
    record: dict[str, Any], analysis_text: str, args: argparse.Namespace
) -> tuple[dict[str, Any], int]:
    chunks = split_for_llm(analysis_text, args.max_input_chars)
    partials = []
    for index, chunk in enumerate(chunks, start=1):
        print(f"    LLM part {index}/{len(chunks)} ({len(chunk):,} chars)")
        partials.append(
            call_llm_json(
                prompt=build_extraction_prompt(record, chunk, index, len(chunks)), args=args
            )
        )
    if len(partials) == 1:
        return partials[0], 1
    print("    consolidating partial metadata")
    return call_llm_json(prompt=build_consolidation_prompt(record, partials), args=args), len(
        chunks
    )


# ---------------------------------------------------------------------------
# Derived retrieval text
# ---------------------------------------------------------------------------


def hierarchy_prefix(record: dict[str, Any]) -> str:
    """e.g. "The Central Goods and Services Tax Act, 2017. Chapter I: PRELIMINARY. Section 1: Short title...".
    Records without a chapter (schedules, notifications, circulars) skip that part; notifications and
    circulars add their subject, so every chunk names the instrument and what it is about."""
    parts = [f"{record.get('instrument_title')}."]
    if record.get("instrument_subject"):
        parts.append(f"{record['instrument_subject']}.")
    if record.get("chapter_number") or record.get("chapter_title"):
        parts.append(f"Chapter {record.get('chapter_number')}: {record.get('chapter_title')}.")
    title = record.get("provision_title")
    label = f"{str(record.get('provision_type', '')).title()} {record.get('provision_number')}"
    parts.append(f"{label}: {str(title).rstrip('.')}." if title else f"{label}.")
    return " ".join(parts)


def build_retrieval_text_exact(record: dict[str, Any], analysis_text: str) -> str:
    return f"{hierarchy_prefix(record)}\n\n{analysis_text}".strip()


def build_retrieval_text_enriched(
    record: dict[str, Any], analysis_text: str, metadata: dict[str, Any]
) -> str:
    sections = [hierarchy_prefix(record), f"Summary: {metadata.get('summary', '')}"]
    labels = {
        "topics": "Topics",
        "keywords": "Keywords",
        "applies_to": "Applies to",
        "conditions": "Conditions",
        "exceptions": "Exceptions",
        "obligations": "Obligations",
        "procedural_steps": "Procedural steps",
        "deadlines": "Deadlines",
        "thresholds": "Thresholds",
        "possible_questions": "Questions",
    }
    for field, label in labels.items():
        if metadata.get(field):
            sections.append(f"{label}: " + " | ".join(metadata[field]))
    sections.append("Authoritative provision text:\n" + analysis_text)
    return "\n\n".join(s for s in sections if s.strip()).strip()


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------


def omitted_metadata(record: dict[str, Any]) -> dict[str, Any]:
    topic = re.sub(r"\s+", " ", str(record.get("chapter_title") or "")).strip().title()
    return {
        **EMPTY_METADATA,
        "summary": f"{str(record.get('provision_type', '')).title()} {record.get('provision_number')} is shown as omitted in the consolidated source.",
        "topics": [topic] if topic else [],
        "keywords": ["omitted provision"],
    }


def process_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    analysis_text, exclusions = clean_text_for_analysis(str(record.get("raw_text") or ""))
    references = extract_explicit_references(analysis_text, record)
    skipped_reason: str | None = None
    chunk_count = 0

    if str(record.get("status", "")).lower() == "omitted":
        metadata, skipped_reason = omitted_metadata(record), "provision_marked_omitted"
    elif not analysis_text.strip():
        metadata = {
            **EMPTY_METADATA,
            "summary": "No analyzable legal text was found for this provision.",
        }
        skipped_reason = "empty_analysis_text"
    elif args.dry_run:
        metadata = {**EMPTY_METADATA, "summary": "DRY RUN: LLM enrichment was not executed."}
        skipped_reason = "dry_run"
        chunk_count = len(split_for_llm(analysis_text, args.max_input_chars))
    else:
        metadata, chunk_count = enrich_text(record, analysis_text, args)


    return {
        **record,
        "analysis_text": analysis_text,
        "analysis_text_hash": sha256_text(analysis_text),
        "analysis_exclusions": exclusions,
        "explicit_references": references,
        "llm_metadata": metadata,
        "retrieval_text_exact": build_retrieval_text_exact(record, analysis_text),
        "retrieval_text_enriched": build_retrieval_text_enriched(record, analysis_text, metadata),
        "enrichment": {
            "prompt_version": PROMPT_VERSION,
            "provider": args.provider,
            "endpoint": provider_endpoint(args.provider),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "generated_at_utc": utc_now_iso(),
            "llm_chunk_count": chunk_count,
            "max_input_chars": args.max_input_chars,
            "skipped": skipped_reason is not None,
            "skip_reason": skipped_reason,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    from app.config import settings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw records JSONL (parse_bare_law / parse_notifications)",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=None, help="Default: <output>.errors.jsonl")
    ap.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="Already-enriched corpus; records with the same id and source_hash are copied, not re-enriched",
    )
    ap.add_argument(
        "--provider", choices=("groq", "openai"), default="groq",
        help="groq (default; free tier allows ~200k tokens/day for gpt-oss-120b) or openai",
    )
    ap.add_argument("--model", default=None,
                    help=f"Default: {DEFAULT_MODEL} for groq, settings.llm_model_grader for openai")
    ap.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default=DEFAULT_REASONING_EFFORT
    )
    ap.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ids", help="Comma-separated record IDs to process")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-input-chars", type=int, default=18_000)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--retry-wait", type=float, default=2.0)
    ap.add_argument("--sleep-between-records", type=float, default=0.0)
    ap.add_argument(
        "--overwrite", action="store_true", help="Delete existing output/errors before starting"
    )
    ap.add_argument("--dry-run", action="store_true", help="Everything except the LLM call")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODEL if args.provider == "groq" else settings.llm_model_grader
    errors_path = args.errors or args.output.with_suffix(".errors.jsonl")

    if args.overwrite:
        for p in (args.output, errors_path):
            p.unlink(missing_ok=True)

    records = load_jsonl(args.input)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        records = [r for r in records if r["id"] in wanted]
    if args.limit:
        records = records[: args.limit]
    done = {r["id"] for r in load_jsonl(args.output)} if args.output.exists() else set()
    reusable = (
        {(r["id"], r.get("source_hash")): r for r in load_jsonl(args.reuse)} if args.reuse else {}
    )

    pending = [r for r in records if r["id"] not in done]
    reused = enriched = failed = 0
    print(
        f"Records: {len(records)}  already in output: {len(records) - len(pending)}  pending: {len(pending)}"
    )
    print(f"Model: {args.model} via {provider_endpoint(args.provider)}  dry-run: {args.dry_run}")

    for index, record in enumerate(pending, start=1):
        rid = record["id"]
        prior = reusable.get((rid, record.get("source_hash")))
        if prior is not None:
            append_jsonl(args.output, prior)
            reused += 1
            continue
        print(f"[{index}/{len(pending)}] {rid} — {record.get('provision_title')}")
        started = time.perf_counter()
        try:
            out = process_record(record, args)
            out["enrichment"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            append_jsonl(args.output, out)
            enriched += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            append_jsonl(
                errors_path,
                {
                    "id": rid,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "failed_at_utc": utc_now_iso(),
                },
            )
            print(f"    ERROR: {exc}", file=sys.stderr)
            if args.stop_on_error:
                raise
        if args.sleep_between_records > 0:
            time.sleep(args.sleep_between_records)

    print(f"\nReused: {reused}  enriched: {enriched}  failed: {failed}  -> {args.output}")
    if failed:
        print(f"Errors: {errors_path} (re-run the same command to retry only the failures)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
