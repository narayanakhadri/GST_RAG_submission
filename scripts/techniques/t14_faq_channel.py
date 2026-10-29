"""T14 — Secondary sources in their own channel: FAQ question-matching instead of FAQs in the main pool (v4).

Idea. In v4 the CBIC FAQs crowded the Act out of the top-k (they restate s.52 question by question; recall fell
0.668 → 0.643). Treat FAQs like an FAQ system does: match the *user question against FAQ questions*
(Dense-to-Question and Sparse-to-Answer, paper 3: dense retrieval on the question side; GST FAQ answering by
question clustering, paper 17), and add at most one FAQ, only when it is a close match. The statute pool is then
untouched, and an FAQ is shown only when it answers the very question asked.

How it works here.
- Index: the "Question: …" part of every FAQ entry/paragraph in the v4 corpus, embedded with 3-large into
  `gst_faq_questions_large` (≈ 150 short texts, < $0.01).
- Retrieval: E17 on the v4 collection with FAQ records removed from the pool, cap 3 → top 8; then if the best FAQ
  question similarity ≥ `--threshold` (default 0.75), that FAQ is inserted at position `--position` (default 4)
  and the list is cut back to 8.
- Gold: golden_set_v4.jsonl (FAQ part 2 is required for p10_q007); compare with e17_v4 and e17_v3_textfix.

    uv run python -m scripts.techniques.t14_faq_channel build
    uv run python -m scripts.techniques.t14_faq_channel eval --set all --threshold 0.75
"""

from __future__ import annotations

import argparse
import re
import uuid

from scripts.techniques.common import E17_FLAGS, evaluate

V4_CORPUS = "seed/docs/gst/gst_provisions_enriched_v4.jsonl"
V4_GOLDEN = "seed/docs/gst/golden_set_v4.jsonl"
V4_COLLECTION = "gst_thesis_enriched_large_v4"
FAQ_COLLECTION = "gst_faq_questions_large"
MODEL, DIMS = "text-embedding-3-large", 3072


def use_v4() -> None:
    from app.config import settings

    settings.qdrant_collection = V4_COLLECTION
    settings.embedding_model = MODEL
    settings.gst_corpus_path = V4_CORPUS


def faq_question(text: str) -> str:
    m = re.search(r"Question:\s*(.+?)(?:\nAnswer:|$)", text, re.S)
    return (m.group(1) if m else text.split("?")[0] + "?").strip()[:600]


def build() -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import source_label
    from app.services.embedding_service import embed_texts
    from app.services.vector_store import get_client

    faqs = [r for r in load_corpus(V4_CORPUS) if r["instrument_id"].startswith("faq_")]
    qs = [faq_question(r["raw_text"]) for r in faqs]
    vecs = embed_texts(qs, model=MODEL)
    client = get_client()
    if client.collection_exists(FAQ_COLLECTION):
        client.delete_collection(FAQ_COLLECTION)
    client.create_collection(FAQ_COLLECTION, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    client.upsert(FAQ_COLLECTION, points=[
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["id"])), vector=v,
                    payload={"provision_id": r["id"], "question": q, "source": source_label(r)})
        for r, q, v in zip(faqs, qs, vecs, strict=True)])
    print(f"{FAQ_COLLECTION}: {len(faqs)} FAQ questions")


def retrieve(question: str, threshold: float, position: int) -> list:
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import cap_doc_types, get_graph, kind
    from app.services.rag_service import _retrieve
    from app.services.vector_store import get_client

    graph = get_graph(V4_CORPUS)
    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    statute = [c for c in pool if (pid := graph.id_of(c)) and kind(graph.records[pid]) != "faq"]
    top = cap_doc_types(statute, graph, 8, 3)
    hit = get_client().query_points(FAQ_COLLECTION, query=embed_texts([question], model=MODEL)[0], limit=1,
                                    with_payload=True).points
    if hit and hit[0].score >= threshold:
        faq = graph.chunk_for(hit[0].payload["provision_id"])
        top = (top[:position] + [faq] + top[position:])[:8]
    return top


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--set", default="all")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--position", type=int, default=4)
    args = ap.parse_args()
    use_v4()
    if args.cmd == "build":
        build()
    else:
        evaluate(f"t14_faq_channel_t{args.threshold}_{args.set}", lambda q: retrieve(q, args.threshold, args.position),
                 args.set, golden=V4_GOLDEN, corpus_path=V4_CORPUS)


if __name__ == "__main__":
    main()
