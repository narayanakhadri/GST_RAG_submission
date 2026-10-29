"""Stage 4: once missing law is ingested, move it from "missing" to "required" in the golden set.

The golden set lists law the corpus lacked under external_sources_missing_from_corpus (corrected
annotation) as plain citations ("Notification No. 12/2017-CT (Rate)") or not-yet-existing IDs
("cgst_act_2017.schedule.1"). For every such entry that now resolves to something in the corpus,
this script adds the corpus ID (a provision ID, or an instrument ID for a whole notification/circular)
to primary_required_sources and drops it from the missing list.

Only the corrected annotation is changed. The original annotation (gold_primary_sources /
missing_gold_sources, = legacy_golden_sources) is left as the historical baseline; score against the
updated sources with `run_gst_ragas.py --gold-sources corrected`. Each change is logged per record in
golden_source_validation.ingestion_updates. Both copies (.jsonl and .json) are rewritten; a backup of
the .jsonl is kept beside it.

    uv run python -m scripts.gst_pipeline.update_golden_sources            # dry run: print what would move
    uv run python -m scripts.gst_pipeline.update_golden_sources --write
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from eval.gst_retrieval import corpus_keys

ALIASES_PATH = Path("seed/docs/gst/sources/authority_aliases.json")


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["aliases"]


def resolve(citation: str, aliases: dict[str, str], keys: set[str]) -> str | None:
    target = aliases.get(citation, citation)
    return target if target in keys else None


def update_record(
    g: dict[str, Any], aliases: dict[str, str], keys: set[str]
) -> list[dict[str, str]]:
    primary = list(g.get("primary_required_sources") or [])
    still_missing, moves = [], []
    for citation in g.get("external_sources_missing_from_corpus") or []:
        target = resolve(citation, aliases, keys)
        if target is None:
            if citation not in still_missing:
                still_missing.append(citation)
            continue
        if target not in primary:
            primary.append(target)
        moves.append({"from_missing": citation, "to_primary": target})
    if moves:
        g["primary_required_sources"] = primary
        g["external_sources_missing_from_corpus"] = still_missing
        gsv = g.setdefault("golden_source_validation", {})
        gsv.setdefault("ingestion_updates", []).extend(moves)
    return moves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=Path(settings.gst_golden_path))
    ap.add_argument("--corpus", type=Path, default=Path(settings.gst_corpus_path))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    keys = corpus_keys(load_corpus(args.corpus))
    aliases = load_aliases()
    goldens = [json.loads(line) for line in args.golden.open(encoding="utf-8-sig") if line.strip()]

    moved: Counter[str] = Counter()
    unresolved: Counter[str] = Counter()
    for g in goldens:
        for m in update_record(g, aliases, keys):
            moved[m["to_primary"]] += 1
        for citation in g.get("external_sources_missing_from_corpus") or []:
            unresolved[citation] += 1

    print(f"{sum(moved.values())} citations resolved across {len(moved)} sources:")
    for target, n in moved.most_common():
        print(f"  {n:3}  {target}")
    print(f"\nStill missing ({len(unresolved)}):")
    for citation, n in unresolved.most_common():
        print(f"  {n:3}  {citation}")

    if not args.write:
        print("\nDry run — pass --write to update the golden set.")
        return
    backup = args.golden.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(args.golden, backup)
    with args.golden.open("w", encoding="utf-8") as f:
        for g in goldens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    json_copy = args.golden.with_suffix(".json")
    if json_copy.exists():
        json_copy.write_text(json.dumps(goldens, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nUpdated {args.golden} (+ {json_copy.name}); backup at {backup}")


if __name__ == "__main__":
    main()
