from __future__ import annotations

from typing import Any

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.services.vector_store import VECTOR_SIZE, get_client


def ensure_gst_collection(recreate: bool = False) -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    name = settings.gst_qdrant_collection

    if recreate and name in existing:
        client.delete_collection(collection_name=name)
        existing.discard(name)

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_gst_points(points: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    ensure_gst_collection()
    client = get_client()

    qdrant_points = [
        PointStruct(
            id=point["point_id"],
            vector=embedding,
            payload={k: v for k, v in point.items() if k != "point_id"},
        )
        for point, embedding in zip(points, embeddings, strict=True)
    ]
    client.upsert(collection_name=settings.gst_qdrant_collection, points=qdrant_points)


def search_gst(query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    client = get_client()
    results = client.query_points(
        collection_name=settings.gst_qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score": float(p.score),
            **(p.payload or {}),
        }
        for p in results
    ]


def collection_point_count(collection_name: str) -> int:
    client = get_client()
    return client.count(collection_name=collection_name, exact=True).count
