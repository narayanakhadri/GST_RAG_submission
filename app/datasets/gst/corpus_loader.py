from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load the GST provision corpus JSONL (one provision per line).

    Tolerant of extra/unknown fields (the real file has more columns than any
    external spec describes) — callers should read via .get() downstream.
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

    ids = [r.get("id") for r in records]
    if len(ids) != len(set(ids)):
        duplicates = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"Duplicate provision IDs in corpus: {duplicates}")

    return records
