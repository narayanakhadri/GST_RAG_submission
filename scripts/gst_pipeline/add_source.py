"""Add one new notification or circular to the corpus and the vector database, without a rebuild.

    uv run python -m scripts.gst_pipeline.add_source \
        --instrument-id notification_06_2025_ct_rate --kind notification \
        --number "06/2025-Central Tax (Rate)" --date 2025-01-16 \
        --subject "Amends Notification 12/2017-CT(Rate) (55th GST Council)" \
        --url https://gstcouncil.gov.in/sites/default/files/2025-01/ctr06-2025.pdf

Use --pdf <file> instead of --url for a PDF downloaded by hand (e.g. from CBIC, whose site needs a browser).
Steps, reusing the corpus pipeline:
  1. register the source in seed/docs/gst/sources/extra_sources.json (fetch_sources.SOURCES merges it, so the next
     full rebuild includes it) — skip with --no-register;
  2. download or copy the PDF, convert it with Docling, update sources/manifest.json;
  3. split it into paragraph/entry records (parse_notifications.build_records);
  4. enrich only these records with the LLM metadata (enrich.process_record; --provider groq|openai), or with
     --no-enrich embed the exact text (hierarchy prefix + text: free and immediate, enrich later);
  5. write them into the corpus file (records with the same id are replaced) and upsert only their points into
     Qdrant. Point ids are deterministic (uuid5 of the record id), so re-running adds nothing new; the sparse
     TF-IDF index is rebuilt from Qdrant on every query, so the new law is searchable at once.

--corpus/--collection default to settings.gst_corpus_path / settings.gst_qdrant_collection. To try a source without
touching them, write to a copy: --corpus seed/docs/gst/gst_provisions_live.jsonl --collection X --init-from Y
(copies collection Y's points, vectors included, into a new collection X; no re-embedding).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from app.config import settings
from scripts.gst_pipeline import fetch_sources
from scripts.gst_pipeline.build_corpus import validate
from scripts.gst_pipeline.parse_notifications import build_records

SOURCES_DIR = fetch_sources.SOURCES_DIR
EXTRA = SOURCES_DIR / "extra_sources.json"
SOURCE_FIELDS = ("instrument_id", "kind", "number", "date", "subject", "url", "law_position", "table_text", "title")


def register(src: dict[str, Any], path: Path = EXTRA) -> None:
    extra = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    extra = [s for s in extra if s["instrument_id"] != src["instrument_id"]] + [src]
    path.write_text(json.dumps(extra, ensure_ascii=False, indent=1), encoding="utf-8")


def fetch(src: dict[str, Any], pdf: Path | None, reconvert: bool, update_manifest: bool = True) -> dict[str, Any]:
    """PDF + Docling markdown in sources/, and the manifest entry for them (written to manifest.json, which the
    full rebuild reads, unless update_manifest is False)."""
    iid = src["instrument_id"]
    pdf_path, md_path = SOURCES_DIR / "pdf" / f"{iid}.pdf", SOURCES_DIR / "md" / f"{iid}.md"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf is not None:
        shutil.copyfile(pdf, pdf_path)
    elif not pdf_path.exists():
        pdf_path.write_bytes(fetch_sources.download(src["url"]))
    if reconvert or not md_path.exists():
        from docling.document_converter import DocumentConverter

        md_path.write_text(fetch_sources.convert_to_markdown(DocumentConverter(), pdf_path), encoding="utf-8")
    data = pdf_path.read_bytes()
    entry = {"law_position": "as issued", **src, "pdf": pdf_path.as_posix(), "markdown": md_path.as_posix(),
             "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if not update_manifest:
        return entry
    manifest_path = SOURCES_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    manifest = [m for m in manifest if m["instrument_id"] != iid] + [entry]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return entry


def enrich_records(records: list[dict[str, Any]], args: argparse.Namespace,
                   done: dict[tuple[str, str], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Enrich each record, reusing any already in the corpus with the same id and source_hash (no LLM call)."""
    from scripts.gst_pipeline import enrich

    ns = argparse.Namespace(
        provider=args.provider, model=args.model or (enrich.DEFAULT_MODEL if args.provider == "groq"
                                                     else settings.llm_model_grader),
        reasoning_effort=enrich.DEFAULT_REASONING_EFFORT, max_completion_tokens=enrich.DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=0.0, max_input_chars=18_000, max_attempts=3, retry_wait=2.0, dry_run=args.no_enrich,
    )
    out = []
    for i, r in enumerate(records, 1):
        prior = (done or {}).get((r["id"], r["source_hash"]))
        if prior is not None and bool(prior.get("llm_metadata")) != args.no_enrich:
            out.append(prior)
            print(f"  [{i}/{len(records)}] {r['id']} unchanged, reused")
            continue
        e = enrich.process_record(r, ns)
        if args.no_enrich:  # embed the exact text, not the dry-run placeholder summary
            e["retrieval_text_enriched"] = None
            e["llm_metadata"] = None
            e["enrichment"]["skip_reason"] = "added_without_enrichment"
        out.append(e)
        print(f"  [{i}/{len(records)}] {r['id']}{'' if args.no_enrich else ' enriched'}")
    return out


def write_corpus(records: list[dict[str, Any]], corpus: Path, base: Path | None) -> list[dict[str, Any]]:
    if not corpus.exists() and base is not None:
        shutil.copyfile(base, corpus)
    existing = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if corpus.exists() else []
    new_ids = {r["id"] for r in records}
    merged = [r for r in existing if r["id"] not in new_ids] + records
    errors = [e for e in validate(merged, enriched=False) if any(i in e for i in new_ids)]
    if errors:
        raise SystemExit("validation failed:\n  " + "\n  ".join(errors[:10]))
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
    return merged


def clone_collection(source: str, target: str) -> None:
    from app.services.vector_store import get_client

    client = get_client()
    if client.collection_exists(target):
        return
    info = client.get_collection(source)
    client.create_collection(target, vectors_config=info.config.params.vectors)
    offset, n = None, 0
    while True:
        points, offset = client.scroll(source, limit=256, offset=offset, with_payload=True, with_vectors=True)
        if points:
            from qdrant_client.models import PointStruct

            client.upsert(target, [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
            n += len(points)
        if offset is None:
            break
    print(f"cloned {n} points: {source} -> {target}")


def upsert(records: list[dict[str, Any]], collection: str) -> int:
    from app.datasets.gst.document_mapper import map_provision_to_points
    from app.services.embedding_service import embed_texts
    from app.services.gst_vector_store import upsert_gst_points

    settings.gst_qdrant_collection = collection
    points = [pt for r in records if r.get("status") != "omitted" for pt in map_provision_to_points(r)]
    for start in range(0, len(points), 32):
        batch = points[start:start + 32]
        upsert_gst_points(batch, embed_texts([p["text"] for p in batch]))
    return len(points)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instrument-id", required=True, help="e.g. notification_06_2025_ct_rate, circular_250_2025")
    ap.add_argument("--kind", choices=("notification", "circular"), required=True)
    ap.add_argument("--number", required=True, help='e.g. "06/2025-Central Tax (Rate)"')
    ap.add_argument("--date", required=True, help="issue date, YYYY-MM-DD")
    ap.add_argument("--subject", required=True)
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--url", help="PDF URL (GST Council / CBIC)")
    src_group.add_argument("--pdf", type=Path, help="a PDF downloaded by hand")
    ap.add_argument("--law-position", default="as issued")
    ap.add_argument("--table-text", choices=("pdf", "faq_pdf"), help="see fetch_sources.SOURCES")
    ap.add_argument("--no-enrich", action="store_true", help="embed the exact text; no LLM call")
    ap.add_argument("--provider", choices=("groq", "openai"), default="groq")
    ap.add_argument("--model")
    ap.add_argument("--corpus", type=Path, default=Path(settings.gst_corpus_path))
    ap.add_argument("--base-corpus", type=Path, default=Path(settings.gst_corpus_path),
                    help="copied to --corpus first if --corpus does not exist")
    ap.add_argument("--collection", default=settings.gst_qdrant_collection)
    ap.add_argument("--init-from", help="clone this collection into --collection first if it does not exist")
    ap.add_argument("--no-register", action="store_true",
                    help="do not add to extra_sources.json or manifest.json (a trial: the full rebuild stays as it is)")
    ap.add_argument("--reconvert", action="store_true")
    args = ap.parse_args()

    src = {k: v for k, v in {
        "instrument_id": args.instrument_id, "kind": args.kind, "number": args.number, "date": args.date,
        "subject": args.subject, "url": args.url or f"file:{args.pdf.name}", "law_position": args.law_position,
        "table_text": args.table_text,
    }.items() if v is not None}
    if not args.no_register:
        register(src)
    entry = fetch(src, args.pdf, args.reconvert, update_manifest=not args.no_register)
    records = build_records(entry, Path(entry["markdown"]).read_text(encoding="utf-8"))
    print(f"{args.instrument_id}: {len(records)} records")
    existing = args.corpus if args.corpus.exists() else args.base_corpus
    done = {(r["id"], r.get("source_hash")): r for r in map(json.loads, existing.read_text(encoding="utf-8").splitlines())
            if r["id"].startswith(args.instrument_id + ".")} if existing and existing.exists() else {}
    records = enrich_records(records, args, done)
    merged = write_corpus(records, args.corpus, args.base_corpus)
    if args.init_from:
        clone_collection(args.init_from, args.collection)
    n = upsert(records, args.collection)
    print(f"corpus {args.corpus}: {len(merged)} records; upserted {n} points into {args.collection}")


if __name__ == "__main__":
    main()
