"""Learned sparse retrieval (thesis wave 1, W1f): SPLADE++, BM42 and miniCOIL via fastembed, stored as Qdrant
sparse vectors — the learned counterparts of the TF-IDF / BM25 keyword scorers.

- SPLADE++ (`prithivida/Splade_PP_en_v1`) expands each text into weighted vocabulary terms, including terms that
  do not occur in it (learned term expansion), so it can bridge some vocabulary gaps.
- BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`) keeps BM25-style term matching but weights terms by a small
  transformer's attention instead of term frequency; Qdrant applies the IDF.
- miniCOIL (`Qdrant/minicoil-v1`) adds a small per-term meaning vector to BM25, so "bank" (river) and "bank"
  (finance) score differently; Qdrant applies the IDF.

Each model has its own sparse-only collection (`gst_sparse_<name>`), built by
scripts/thesis/build_sparse_collections.py from the same records and payload as the dense collection, so its hits
fuse with dense hits by RRF exactly like the TF-IDF ones. Select with the flag `sparse_scorer: "splade" | "bm42"
| "minicoil"`. Note: the models read at most ~512 tokens, so long provisions are judged on their beginning.
"""

from __future__ import annotations

import threading

from app.models import RetrievedChunk

MODELS = {
    "splade": ("prithivida/Splade_PP_en_v1", False),
    "bm42": ("Qdrant/bm42-all-minilm-l6-v2-attentions", True),
    "minicoil": ("Qdrant/minicoil-v1", True),
}
VECTOR_NAME = "sparse"
_models: dict[str, object] = {}
_lock = threading.Lock()


def collection_name(scorer: str) -> str:
    return f"gst_sparse_{scorer}"


def uses_idf(scorer: str) -> bool:
    return MODELS[scorer][1]


def model(scorer: str):
    from fastembed import SparseTextEmbedding

    with _lock:
        if scorer not in _models:
            _models[scorer] = SparseTextEmbedding(model_name=MODELS[scorer][0])
        return _models[scorer]


class LearnedSparseIndex:
    """Same interface as SparseVectorIndex / BM25Index: .search(query, top_k) -> list[RetrievedChunk]."""

    def __init__(self, scorer: str) -> None:
        self.scorer = scorer

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        from qdrant_client.models import SparseVector

        from app.services.vector_store import get_client

        emb = next(iter(model(self.scorer).query_embed(query)))
        points = get_client().query_points(
            collection_name=collection_name(self.scorer),
            query=SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()),
            using=VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        ).points
        return [RetrievedChunk(text=p.payload.get("text", ""), source=p.payload.get("source", ""), score=p.score)
                for p in points]
