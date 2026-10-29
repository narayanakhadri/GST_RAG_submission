from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_golden_set(path: str | Path) -> list[dict[str, Any]]:
    """Load the GST golden-question JSONL (one exam question per line).

    Tolerant of extra/unknown fields — only id/question/gold_*_sources are
    relied on by the current smoke test.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc

    return records


def answerable_questions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Questions with at least one gold_primary_sources entry.

    Rows with empty gold_primary_sources (and rows that are only answerable
    via missing_gold_sources not yet in the corpus) are excluded — they'd be
    abstention/unanswerable cases, not retrieval targets, per the brief.
    """
    return [r for r in records if r.get("gold_primary_sources")]
