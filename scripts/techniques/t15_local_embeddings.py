"""T15 — Open-source local embedding models vs OpenAI text-embedding-3-large (dense retrieval, same records).

Idea. Embedding choice moved recall more than most techniques here (3-small → 3-large: +0.065). Open models run
for free on CPU and keep documents local (privacy, no API cost) — a frequent interview topic. Papers: LawRAG
(10: GTE-large best of several encoders on legal QA), the Algerian legal-RAG thesis (E5 recall 91% vs BM25),
Multimodal RAG eval (27), InLegalBERT (19; general models often beat legal BERTs, paper 6).

Models (fastembed ONNX, CPU): thenlper/gte-large, BAAI/bge-large-en-v1.5, mixedbread-ai/mxbai-embed-large-v1,
snowflake/snowflake-arctic-embed-l, nomic-ai/nomic-embed-text-v1.5, intfloat/multilingual-e5-large,
jinaai/jina-embeddings-v2-base-en (8k context). Most read at most 512 tokens: long provisions are judged on their
beginning (heading + summary first in the enriched text, which helps). Query prefixes follow each model's card
(fastembed's query_embed applies them).

How it works here.
- `build --model gte-large`: embeds the enriched text of every record (same text as the 3-large collection) into
  `gst_local_<model>`; ~15–40 min per large model on CPU.
- `eval --model gte-large`: dense-only top 8 (no hybrid, no decomposition) — compare with `--model openai`
  (dense-only on gst_thesis_enriched_large), so only the encoder differs.

    uv run python -m scripts.techniques.t15_local_embeddings build --model gte-large
    uv run python -m scripts.techniques.t15_local_embeddings eval --model gte-large --set all
    uv run python -m scripts.techniques.t15_local_embeddings eval --model openai --set all
"""

from __future__ import annotations

import argparse
import threading
import time

from app.config import settings
from scripts.techniques.common import evaluate

MODELS = {
    "gte-large": "thenlper/gte-large",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "mxbai-large": "mixedbread-ai/mxbai-embed-large-v1",
    "arctic-l": "snowflake/snowflake-arctic-embed-l",
    "nomic-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "e5-large": "intfloat/multilingual-e5-large",
    "jina-v2-base": "jinaai/jina-embeddings-v2-base-en",
}
_lock, _models = threading.Lock(), {}


def collection(name: str) -> str:
    return f"gst_local_{name.replace('.', '_').replace('-', '_')}"


def model(name: str):
    from fastembed import TextEmbedding

    with _lock:
        if name not in _models:
            _models[name] = TextEmbedding(model_name=MODELS[name])
        return _models[name]


def build(name: str) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import map_provision_to_points
    from app.services.vector_store import get_client

    points = [p for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"
              for p in map_provision_to_points(r)]
    m, t0 = model(name), time.time()
    first = next(iter(m.embed([points[0]["text"]])))
    client, coll = get_client(), collection(name)
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=VectorParams(size=len(first), distance=Distance.COSINE))
    for s in range(0, len(points), 32):
        batch = points[s : s + 32]
        vecs = list(m.embed([p["text"] for p in batch], batch_size=8))
        client.upsert(coll, points=[PointStruct(id=p["point_id"], vector=v.tolist(),
                                                payload={k: x for k, x in p.items() if k != "point_id"})
                                    for p, v in zip(batch, vecs, strict=True)])
        if s % 320 == 0:
            print(f"  {s}/{len(points)} ({time.time() - t0:.0f} s)")
    print(f"{coll}: {len(points)} points, {len(first)} dims, {time.time() - t0:.0f} s")


def retrieve(question: str, name: str, top_k: int = 8) -> list:
    from app.models import RetrievedChunk
    from app.services.embedding_service import embed_texts
    from app.services.vector_store import get_client

    if name == "openai":
        coll, vec = "gst_thesis_enriched_large", embed_texts([question], model="text-embedding-3-large")[0]
    else:
        coll, vec = collection(name), next(iter(model(name).query_embed(question))).tolist()
    pts = get_client().query_points(coll, query=vec, limit=top_k, with_payload=["text", "source"]).points
    return [RetrievedChunk(text=p.payload["text"], source=p.payload["source"], score=p.score) for p in pts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--model", choices=[*MODELS, "openai"], default="gte-large")
    ap.add_argument("--set", default="all")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.model)
    else:
        evaluate(f"t15_dense_{args.model}_{args.set}", lambda q: retrieve(q, args.model), args.set)


if __name__ == "__main__":
    main()
