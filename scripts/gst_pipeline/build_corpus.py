"""Build the GST corpus end to end: parse -> enrich -> validate -> write settings.gst_corpus_path.

    uv run python -m scripts.gst_pipeline.build_corpus             # dry run: parse + validate, no LLM, no write
    uv run python -m scripts.gst_pipeline.build_corpus --write     # enrich new/changed records via Groq, write corpus
    uv run python -m scripts.gst_pipeline.build_corpus --write --provider openai \
        --output seed/docs/gst/gst_provisions_enriched_v4.jsonl   # new corpus file, current one left as is

Inputs: seed/scripts/GST_Act.md (bare law) and seed/docs/gst/sources/ (notifications/circulars, from
fetch_sources.py). Records whose id and source_hash match the current corpus are reused as-is, so
only new or changed provisions are sent to the LLM. Intermediate files go to seed/docs/gst/build/;
the previous corpus is backed up there before it is replaced. Re-run `make seed-gst` afterwards
(with --overwrite, so points for dropped records disappear too).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import settings
from app.datasets.gst.document_mapper import select_embedding_text, source_label
from scripts.gst_pipeline.parse_bare_law import parse_bare_law
from scripts.gst_pipeline.parse_notifications import build_records

BUILD = Path("seed/docs/gst/build")
REQUIRED = (
    "id",
    "instrument_id",
    "instrument_title",
    "provision_type",
    "provision_number",
    "raw_text",
    "source_hash",
)


def validate(records: list[dict[str, Any]], enriched: bool) -> list[str]:
    errors = []
    ids = Counter(r["id"] for r in records)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    labels = Counter(source_label(r) for r in records)
    errors += [f"duplicate source label {lbl!r}" for lbl, n in labels.items() if n > 1]
    for r in records:
        missing = [
            f
            for f in REQUIRED
            if r.get(f) in (None, "") and not (f == "raw_text" and r.get("status") == "omitted")
        ]
        if missing:
            errors.append(f"{r['id']}: missing {missing}")
        if enriched and r.get("status") != "omitted":
            text = select_embedding_text(r)
            if not text.startswith(str(r["instrument_title"])):
                errors.append(f"{r['id']}: embedded text does not start with the instrument name")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bare-law", type=Path, default=Path("seed/scripts/GST_Act.md"))
    ap.add_argument("--manifest", type=Path, default=Path("seed/docs/gst/sources/manifest.json"))
    ap.add_argument("--corpus", type=Path, default=Path(settings.gst_corpus_path))
    ap.add_argument(
        "--output", type=Path, default=None, help="Write here instead of replacing --corpus (default: --corpus)"
    )
    ap.add_argument(
        "--write", action="store_true", help="Run enrichment and replace the corpus"
    )
    ap.add_argument(
        "--provider", choices=("groq", "openai"), default="groq", help="LLM used for enrichment"
    )
    args = ap.parse_args()
    output = args.output or args.corpus

    bare, _ = parse_bare_law(args.bare_law.read_text(encoding="utf-8"))
    notes = []
    for src in json.loads(args.manifest.read_text(encoding="utf-8")):
        notes += build_records(src, Path(src["markdown"]).read_text(encoding="utf-8"))
    raw = bare + notes
    errors = validate(raw, enriched=False)
    if errors:
        sys.exit("Raw records invalid:\n  " + "\n  ".join(errors[:30]))

    current = {}
    if args.corpus.exists():
        with args.corpus.open(encoding="utf-8-sig") as f:
            current = {
                (r["id"], r.get("source_hash")): r for r in map(json.loads, filter(str.strip, f))
            }
    to_enrich = [r for r in raw if (r["id"], r["source_hash"]) not in current]
    print(f"Raw records: {len(raw)} ({len(bare)} bare law, {len(notes)} notifications/circulars)")
    print(
        f"  reused from current corpus: {len(raw) - len(to_enrich)}   need enrichment: {len(to_enrich)}"
    )
    for k, n in sorted(Counter(r["instrument_id"] for r in to_enrich).items()):
        print(f"    {k:32} {n}")

    BUILD.mkdir(parents=True, exist_ok=True)
    raw_path, enriched_path = BUILD / "raw_all.jsonl", BUILD / "enriched_all.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.write:
        print(f"\nDry run — wrote {raw_path}. Pass --write to enrich and write {output}.")
        return

    cmd = [
        sys.executable,
        "-m",
        "scripts.gst_pipeline.enrich",
        "--input",
        str(raw_path),
        "--output",
        str(enriched_path),
        "--reuse",
        str(args.corpus),
        "--provider",
        args.provider,
    ]
    if subprocess.run(cmd).returncode != 0:
        sys.exit(
            "Enrichment had failures — re-run this command to retry them; the corpus was not replaced."
        )

    with enriched_path.open(encoding="utf-8") as f:
        by_id = {r["id"]: r for r in map(json.loads, filter(str.strip, f))}
    final = [by_id[r["id"]] for r in raw if r["id"] in by_id]
    if len(final) != len(raw):
        sys.exit(
            f"{len(raw) - len(final)} records missing from {enriched_path}; corpus not replaced."
        )
    errors = validate(final, enriched=True)
    if errors:
        sys.exit("Enriched records invalid:\n  " + "\n  ".join(errors[:30]))

    backup = BUILD / f"{args.corpus.stem}.before_first_rebuild.jsonl"
    if (
        output == args.corpus and args.corpus.exists() and not backup.exists()
    ):  # keep the original hand-built corpus, not the last rebuild
        shutil.copy2(args.corpus, backup)
        print(f"Original corpus backed up to {backup}")
    with output.open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(final)} records to {output}. Next: make seed-gst (uv run python scripts/seed_gst.py --overwrite)"
    )


if __name__ == "__main__":
    main()
