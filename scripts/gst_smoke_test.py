import argparse

from loguru import logger

from app.config import settings
from app.datasets.gst.golden_loader import answerable_questions, load_golden_set
from app.services.embedding_service import embed_texts
from app.services.gst_vector_store import search_gst

TOP_K = 5


def run_smoke_test(golden_path: str, sample_size: int) -> None:
    golden = answerable_questions(load_golden_set(golden_path))
    sample = golden[:sample_size]

    logger.info(
        "GST retrieval smoke test — {} answerable questions available, sampling {}",
        len(golden), len(sample),
    )

    hits = 0
    for row in sample:
        question = row["question"]
        gold_ids = set(row.get("gold_primary_sources", []))

        query_embedding = embed_texts([question])[0]
        results = search_gst(query_embedding, top_k=TOP_K)
        returned_ids = [r.get("provision_id") for r in results]

        hit = bool(gold_ids & set(returned_ids))
        hits += int(hit)

        print(f"\nQ [{row.get('id')}]: {question}")
        print(f"  gold_primary_sources: {sorted(gold_ids)}")
        print(f"  top-{TOP_K} returned:")
        for r in results:
            marker = "  <-- gold" if r.get("provision_id") in gold_ids else ""
            print(f"    {r.get('provision_id')}  score={r['score']:.3f}{marker}")
        print(f"  hit@{TOP_K}: {'YES' if hit else 'no'}")

    print(f"\n{'=' * 40}")
    print(f"hit@{TOP_K} rate: {hits}/{len(sample)} ({hits / len(sample):.0%})" if sample else "no sample")
    print(f"collection: {settings.gst_qdrant_collection}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal GST retrieval smoke test against the golden set")
    parser.add_argument("--golden", default=settings.gst_golden_path, help="Path to golden set JSONL")
    parser.add_argument("--sample", type=int, default=10, help="Number of answerable questions to test")
    args = parser.parse_args()

    run_smoke_test(args.golden, args.sample)


if __name__ == "__main__":
    main()
