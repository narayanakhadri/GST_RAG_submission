from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.models import RetrievedChunk

VECTOR_SIZE = 1536


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)

def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def upsert_chunks(chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
    ensure_collection()
    client  = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk.text, "source": chunk.source},
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)

def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    client = get_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            score=float(p.score),
        )
        for p in results
    ]


# Sparse indexes are fitted from a full scroll of the collection, which takes seconds; cache one per
# (collection, scorer) and refit only when the collection's point count changes (i.e. after re-ingestion).
_SPARSE_CACHE: dict[tuple[str, str], tuple[int, object]] = {}


def _build_sparse_index(scorer: str | None = None):
    from app.services.sparse_vector_service import BM25Index, SparseVectorIndex

    scorer = scorer or settings.sparse_scorer
    from app.services.learned_sparse import MODELS, LearnedSparseIndex

    if scorer in MODELS:  # learned sparse vectors stored in Qdrant (W1f), nothing to fit
        return LearnedSparseIndex(scorer)
    client = get_client()
    collection = settings.qdrant_collection
    count = client.count(collection_name=collection, exact=True).count
    cached = _SPARSE_CACHE.get((collection, scorer))
    if cached and cached[0] == count:
        return cached[1]

    all_points, _next_page = client.scroll(
        collection_name=collection,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    documents = [
        {
            "text": point.payload.get("text", "") if point.payload else "",
            "source": point.payload.get("source", "") if point.payload else "",
            "id": str(point.id),
        }
        for point in all_points
    ]
    sparse_index = BM25Index() if scorer == "bm25" else SparseVectorIndex()
    sparse_index.fit(documents)
    _SPARSE_CACHE[(collection, scorer)] = (count, sparse_index)
    return sparse_index


def sparse_search(query_text: str, top_k: int = 5, scorer: str | None = None) -> list[RetrievedChunk]:
    """Pure sparse search (TF-IDF or BM25, see settings.sparse_scorer) — no dense embeddings, no fusion."""
    return _build_sparse_index(scorer).search(query_text, top_k=top_k)


def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    rrf_k: int = 60,
    sparse_top_k: int = 20,
    scorer: str | None = None,
    dense_weight: float | None = None,
) -> list[RetrievedChunk]:
    """Dense + sparse fused by RRF; `dense_weight` w in (0, 1) weights the lists w : 1 − w (None = equal)."""
    from app.services.sparse_vector_service import fuse_rrf
    dense_results = search(query_embedding, top_k=sparse_top_k)
    sparse_results = _build_sparse_index(scorer).search(query_text, top_k=sparse_top_k)
    weights = None if dense_weight is None else [dense_weight, 1.0 - dense_weight]
    fused = fuse_rrf([dense_results, sparse_results], rrf_k=rrf_k, weights=weights)
    return fused[:top_k]


def provision_vectors(provision_ids: list[str]) -> dict[str, list[float]]:
    """Stored dense vectors of the given provisions (first chunk of each), for MMR diversity re-ranking."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    if not provision_ids:
        return {}
    points, _ = get_client().scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(must=[FieldCondition(key="provision_id", match=MatchAny(any=provision_ids))]),
        limit=len(provision_ids) * 4,
        with_payload=["provision_id"],
        with_vectors=True,
    )
    out: dict[str, list[float]] = {}
    for p in points:
        pid = (p.payload or {}).get("provision_id")
        if pid and pid not in out:
            out[pid] = p.vector
    return out


def search_within(query_embedding: list[float], provision_ids: list[str], top_k: int = 10) -> list[str]:
    """Dense search restricted to the given GST provision IDs; returns their IDs by similarity to the query
    (used to rank graph neighbours, see app/services/gst_graph.py)."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    if not provision_ids:
        return []
    points = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        query_filter=Filter(must=[FieldCondition(key="provision_id", match=MatchAny(any=provision_ids))]),
        limit=top_k,
        with_payload=["provision_id"],
    ).points
    return [p.payload["provision_id"] for p in points if p.payload]
