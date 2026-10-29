import argparse
import time

from loguru import logger

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import map_provision_to_points
from app.services.embedding_service import embed_texts
from app.services.gst_vector_store import ensure_gst_collection, upsert_gst_points

BATCH_SIZE = 32


def seed_gst(corpus_path: str, limit: int | None, overwrite: bool) -> dict:
    records = load_corpus(corpus_path)
    if limit:
        records = records[:limit]

    logger.info("GST corpus: {} provisions to ingest from {}", len(records), corpus_path)
    ensure_gst_collection(recreate=overwrite)

    counters = {"provisions": 0, "points": 0, "chunked_provisions": 0, "omitted_skipped": 0, "failed": 0}
    t0 = time.time()

    for idx, record in enumerate(records, start=1):
        provision_id = record.get("id", "<unknown>")
        if record.get("status") == "omitted":
            # Repealed/omitted provisions have no legal text, only "Section X is shown as omitted";
            # embedding them just adds near-identical noise to every search.
            counters["omitted_skipped"] += 1
            continue
        try:
            points = map_provision_to_points(record)
            if not points:
                logger.warning("[{}/{}] {} → no embeddable text (skipped)", idx, len(records), provision_id)
                counters["failed"] += 1
                continue

            if len(points) > 1:
                counters["chunked_provisions"] += 1
                logger.info(
                    "[{}/{}] {} → split into {} chunks (long provision)",
                    idx, len(records), provision_id, len(points),
                )

            for batch_start in range(0, len(points), BATCH_SIZE):
                batch = points[batch_start : batch_start + BATCH_SIZE]
                embeddings = embed_texts([p["text"] for p in batch])
                upsert_gst_points(batch, embeddings)

            counters["provisions"] += 1
            counters["points"] += len(points)

            if idx % 25 == 0 or idx == len(records):
                logger.info("  [{}/{}] progress — {} points upserted so far", idx, len(records), counters["points"])

        except Exception as exc:  # noqa: BLE001
            logger.warning("[{}/{}] FAILED {}: {}", idx, len(records), provision_id, type(exc).__name__)
            counters["failed"] += 1

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("GST INGESTION COMPLETE in {:.1f}s", elapsed)
    logger.info("  provisions ingested : {}", counters["provisions"])
    logger.info("  provisions chunked  : {}", counters["chunked_provisions"])
    logger.info("  omitted (skipped)   : {}", counters["omitted_skipped"])
    logger.info("  total points upserted: {}", counters["points"])
    logger.info("  failed (skipped)    : {}", counters["failed"])
    logger.info("  collection          : {}", settings.gst_qdrant_collection)
    logger.info("=" * 60)

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the GST provision corpus into Qdrant")
    parser.add_argument("--corpus", default=settings.gst_corpus_path, help="Path to corpus JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N provisions (smoke test)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recreate the GST collection before ingesting (needed to drop points for records no longer in the corpus)",
    )
    args = parser.parse_args()

    seed_gst(args.corpus, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
