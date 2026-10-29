"""Build the extra Qdrant collections for the embedding ablation (RQ2: what text to embed; RQ3: model).

Same records, chunking, point IDs and payload as scripts/seed_gst.py; only the embedded text and/or the
embedding model differ:

    gst_thesis_raw_small       raw_text only (no instrument/section heading, no LLM metadata)
    gst_thesis_exact_small     retrieval_text_exact (hierarchy prefix + text, no LLM metadata)
    gst_thesis_enriched_large  retrieval_text_enriched with text-embedding-3-large (3072 dims)
    gst_thesis_enriched_large_v4  the same on the v4 corpus (v3 + the newly fetched notifications and FAQs)
    gst_thesis_codexsum_large_v4  v4 with Codex-corrected summaries in the embedded text (thesis E13a)

The default collection (gst_provisions_enriched_v3) is the enriched / text-embedding-3-small variant;
gst_provisions_enriched_v4 is its v4 counterpart (scripts/seed_gst.py with GST_QDRANT_COLLECTION set).

    uv run python -m scripts.thesis.build_collections            # all of them
    uv run python -m scripts.thesis.build_collections --only gst_thesis_raw_small
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import map_provision_to_points
from app.services.embedding_service import embed_texts
from app.services.vector_store import get_client

V4_CORPUS = "seed/docs/gst/gst_provisions_enriched_v4.jsonl"  # v3 + 11/2017-CT(R), 41/2017-IT(R), 52/2018-CT, 3 FAQs
VARIANTS = {
    "gst_thesis_raw_small": ("raw_text", "text-embedding-3-small", 1536),
    "gst_thesis_exact_small": ("retrieval_text_exact", "text-embedding-3-small", 1536),
    "gst_thesis_enriched_large": ("retrieval_text_enriched", "text-embedding-3-large", 3072),
    "gst_thesis_enriched_large_v4": ("retrieval_text_enriched", "text-embedding-3-large", 3072, V4_CORPUS),
    # E13a: as above, with the Codex-corrected summary where Codex judged the old one wrong or partly wrong
    "gst_thesis_codexsum_large_v4": ("retrieval_text_enriched", "text-embedding-3-large", 3072, V4_CORPUS, "codex_summary"),
    # E13b doc2query (Nogueira et al. 2019): append Codex search terms + questions the provision answers (v3 corpus)
    "gst_thesis_doc2query_large": ("retrieval_text_enriched", "text-embedding-3-large", 3072, None, "doc2query"),
}
BATCH = 64
CODEX_METADATA = Path("seed/docs/gst/codex_metadata.jsonl")


def codex_summary_transform() -> Callable[[dict], dict]:
    """Swap the 'Summary: ...' paragraph of the embedded text for Codex's corrected summary (verdict wrong or
    minor_issues); every other field and every other record is unchanged."""
    fixes = {}
    for line in CODEX_METADATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            check = row["summary_check"]
            if check["verdict"] != "correct" and (check.get("corrected_summary") or "").strip():
                fixes[row["id"]] = check["corrected_summary"].strip()

    def apply(record: dict) -> dict:
        new = fixes.get(record["id"])
        text = record.get("retrieval_text_enriched") or ""
        if not new:
            return record
        swapped, n = re.subn(r"(^|\n\n)Summary: .*?(?=\n\n|$)", lambda m: f"{m.group(1)}Summary: {new}", text,
                             count=1, flags=re.S)
        if n != 1:
            raise ValueError(f"{record['id']}: no Summary paragraph in the embedded text")
        return {**record, "retrieval_text_enriched": swapped}

    apply.count = len(fixes)  # type: ignore[attr-defined]
    return apply


def doc2query_transform() -> Callable[[dict], dict]:
    """Append the Codex search terms and 'answers questions' to the embedded text (doc2query / E13b)."""
    extra = {}
    for line in CODEX_METADATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            parts = []
            if row.get("search_terms"):
                parts.append("Search terms: " + " | ".join(row["search_terms"]))
            if row.get("answers_questions"):
                parts.append("Answers questions: " + " | ".join(row["answers_questions"]))
            if parts:
                extra[row["id"]] = "\n\n".join(parts)

    def apply(record: dict) -> dict:
        add = extra.get(record["id"])
        if not add:
            return record
        return {**record, "retrieval_text_enriched": (record.get("retrieval_text_enriched") or "") + "\n\n" + add}

    return apply


TRANSFORMS = {"codex_summary": codex_summary_transform, "doc2query": doc2query_transform}


def build(name: str, text_field: str, model: str, dims: int, corpus: str | None = None,
          transform: str | None = None) -> int:
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=dims, distance=Distance.COSINE))

    points = []
    change = TRANSFORMS[transform]() if transform else (lambda r: r)
    for record in map(change, load_corpus(corpus or settings.gst_corpus_path)):
        if record.get("status") == "omitted" or not (record.get(text_field) or "").strip():
            continue
        # map_provision_to_points picks the enriched text; swap in the variant's text, keep everything else.
        variant = {**record, "retrieval_text_enriched": record[text_field].strip()}
        points.extend(map_provision_to_points(variant))

    for start in range(0, len(points), BATCH):
        batch = points[start : start + BATCH]
        vectors = embed_texts([p["text"] for p in batch], model=model)
        client.upsert(name, points=[
            PointStruct(id=p["point_id"], vector=v, payload={k: val for k, val in p.items() if k != "point_id"})
            for p, v in zip(batch, vectors, strict=True)
        ])
    return len(points)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    for name in args.only:
        n = build(name, *VARIANTS[name])
        print(f"{name}: {n} points")


if __name__ == "__main__":
    main()
