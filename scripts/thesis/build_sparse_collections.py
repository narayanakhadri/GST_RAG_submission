"""Build the learned-sparse Qdrant collections (thesis W1f): one sparse-only collection per model, same records,
chunking, point IDs, text and payload as scripts/seed_gst.py, so hits fuse with the dense collection by RRF.

    uv run python -m scripts.thesis.build_sparse_collections                 # splade, bm42, minicoil
    uv run python -m scripts.thesis.build_sparse_collections --only splade

Models run locally on CPU (fastembed / ONNX), so this is free.
"""

from __future__ import annotations

import argparse
import time

from qdrant_client.models import Modifier, PointStruct, SparseVector, SparseVectorParams

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import map_provision_to_points
from app.services.learned_sparse import MODELS, VECTOR_NAME, collection_name, model, uses_idf
from app.services.vector_store import get_client

BATCH = 32


def build(scorer: str, corpus: str) -> int:
    client = get_client()
    name = collection_name(scorer)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config={}, sparse_vectors_config={
        VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF if uses_idf(scorer) else None)})
    points = [p for r in load_corpus(corpus) if r.get("status") != "omitted" for p in map_provision_to_points(r)]
    m = model(scorer)
    for start in range(0, len(points), BATCH):
        batch = points[start:start + BATCH]
        # SPLADE scores the whole vocabulary per token (batch x 512 x 30522 floats): keep its ONNX batch small.
        embs = list(m.embed([p["text"] for p in batch], batch_size=4))
        client.upsert(name, points=[
            PointStruct(id=p["point_id"],
                        vector={VECTOR_NAME: SparseVector(indices=e.indices.tolist(), values=e.values.tolist())},
                        payload={k: v for k, v in p.items() if k != "point_id"})
            for p, e in zip(batch, embs, strict=True)])
    return len(points)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=list(MODELS))
    ap.add_argument("--corpus", default=settings.gst_corpus_path)
    args = ap.parse_args()
    for scorer in args.only:
        t0 = time.time()
        print(f"{collection_name(scorer)}: {build(scorer, args.corpus)} points ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
