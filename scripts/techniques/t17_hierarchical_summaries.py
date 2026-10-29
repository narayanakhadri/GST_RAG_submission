"""T17 — Hierarchical, summary-first retrieval (RAPTOR-style tree, two levels): chapters/instruments → provisions.

Idea. A question about composition levy should first land in "CGST Act, Chapter II — Administration… / section 10"
territory, then pick provisions inside it. Summary nodes for each chapter (Acts, Rules) and each notification /
circular give a coarse map. Sources: RAPTOR (Sarthi et al. 2024; RAG survey) builds a tree of recursive cluster
summaries and retrieves over the "collapsed tree"; Hybrid Hierarchical Retrieval (paper 2; code
github.com/ghuhan17/HybridHierarchicalRetrieval: document-level retrieval first, then passages, dense + sparse at
both stages); abstract-first retrieval (paper 24).

How it works here (no LLM: nodes are built from existing text; ≈ 90 nodes, < $0.01 to embed with 3-large).
- Node text = "<instrument>, Chapter N: <title>" + the one-line summaries of its provisions (Codex-corrected where
  available), cut to ~6,000 characters. RAPTOR clusters by embedding; the legal hierarchy already gives the
  clusters, so no clustering is needed.
- `--mode filter`  (HHR style): top `--chapters` nodes → dense search restricted to their provisions → top 8.
- `--mode fuse`    : E17 pool fused (RRF) with the chapter-restricted ranking (weight 0.5) → cap 3 → top 8.

    uv run python -m scripts.techniques.t17_hierarchical_summaries build
    uv run python -m scripts.techniques.t17_hierarchical_summaries eval --mode fuse --set dev_hard
"""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

COLLECTION = "gst_chapter_nodes_large"
MODEL, DIMS = "text-embedding-3-large", 3072
NODE_CHARS = 6000


def nodes() -> dict[str, dict]:
    from app.config import settings
    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import source_label
    from app.services.gst_graph import summary_of

    groups: dict[str, dict] = defaultdict(lambda: {"members": [], "lines": [], "title": ""})
    for r in load_corpus(settings.gst_corpus_path):
        if r.get("status") == "omitted":
            continue
        if r.get("chapter_number"):
            key = f"{r['instrument_id']}#ch{r['chapter_number']}"
            title = f"{r['instrument_title']}, Chapter {r['chapter_number']}: {r.get('chapter_title') or ''}"
        else:  # notifications, circulars, schedules: the instrument is the node
            key = r["instrument_id"]
            title = f"{r['instrument_title']}" + (f" — {r['instrument_subject']}" if r.get("instrument_subject") else "")
        g = groups[key]
        g["title"] = title
        g["members"].append(r["id"])
        g["lines"].append(f"- {source_label(r).split(', ')[-1]}: {summary_of(r)[:200]}")
    return {k: {**v, "text": (v["title"] + "\n" + "\n".join(v["lines"]))[:NODE_CHARS]} for k, v in groups.items()}


def build() -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.services.embedding_service import embed_texts
    from app.services.vector_store import get_client

    ns = nodes()
    keys = list(ns)
    vecs = embed_texts([ns[k]["text"] for k in keys], model=MODEL)
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    client.upsert(COLLECTION, points=[PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, k)), vector=v,
                                                  payload={"node": k, "title": ns[k]["title"], "members": ns[k]["members"]})
                                      for k, v in zip(keys, vecs, strict=True)])
    print(f"{COLLECTION}: {len(keys)} nodes")


def chapter_ranking(question: str, n_chapters: int) -> list:
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph
    from app.services.vector_store import get_client, search_within

    q = embed_texts([question])[0]
    hits = get_client().query_points(COLLECTION, query=q, limit=n_chapters, with_payload=["members"]).points
    members = [m for h in hits for m in h.payload["members"]]
    graph = get_graph()
    return [graph.chunk_for(i) for i in search_within(q, members, top_k=20)]


def retrieve(question: str, mode: str, n_chapters: int) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve
    from app.services.sparse_vector_service import fuse_rrf

    ranked = chapter_ranking(question, n_chapters)
    if mode == "filter":
        return cap_doc_types(ranked, get_graph(), 8, 3)
    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    return cap_doc_types(fuse_rrf([pool, ranked], weights=[1.0, 0.5]), get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval", "show"])
    ap.add_argument("--mode", choices=["filter", "fuse"], default="fuse")
    ap.add_argument("--chapters", type=int, default=3)
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "show":
        ns = nodes()
        print(len(ns), "nodes;", next(iter(ns.values()))["text"][:600])
    elif args.cmd == "build":
        build()
    else:
        evaluate(f"t17_hier_{args.mode}_c{args.chapters}_{args.set}",
                 lambda q: retrieve(q, args.mode, args.chapters), args.set)


if __name__ == "__main__":
    main()
