"""T27 — GraphRAG-style community summaries: cluster the provision graph, summarise each community, retrieve by it.

Idea. Microsoft GraphRAG (Edge et al. 2024, "From Local to Global") builds an entity graph, finds communities with
the Leiden algorithm, writes an LLM report per community, and answers broad questions from those reports ("global
search"); RAPTOR clusters chunks and summarises recursively. T17 used the legal chapters as clusters; here the
clusters are *learned from the links* (cross-references + Codex typed links), so a community can cut across Acts,
Rules and notifications — e.g. section 16 + rules 36–37 + the ITC circulars.

How it works here.
- Graph: as T22 (provisions, cross-reference + typed-link edges). Communities: Louvain (networkx; Leiden needs
  igraph) with resolution `--resolution` (default 8: ≈ 55 communities, median 7 provisions; 1.0 gives 13 huge ones
  around hub provisions such as s.2 and s.9); communities smaller than 3 are skipped.
- Report per community: Gemini writes 3–5 sentences on what the provisions jointly regulate (free, cached in
  eval/results/thesis/cache/graphrag_reports.json); node text = report + member labels.
- Index: reports embedded with 3-large into `gst_graphrag_communities_large` (≈ 50–120 nodes, < $0.01).
- Retrieval ("local-global" hybrid): top `--communities` reports → their members ranked by dense similarity →
  fused (RRF weight 0.5) with the E17 pool → cap 3 → top 8.

    uv run python -m scripts.techniques.t27_graphrag_communities build
    uv run python -m scripts.techniques.t27_graphrag_communities eval --set dev_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

COLLECTION = "gst_graphrag_communities_large"
CACHE = Path("eval/results/thesis/cache/graphrag_reports.json")
MODEL, DIMS = "text-embedding-3-large", 3072
REPORT = ("You write a short report (3-5 sentences) on a group of linked Indian GST provisions: what they jointly "
          "regulate, which provision is central, and how the others relate to it. Plain text.")


def communities(resolution: float) -> list[list[str]]:
    import networkx as nx

    from scripts.techniques.t22_personalized_pagerank import RELATIONS, nx_graph

    G = nx_graph(RELATIONS)
    comms = nx.community.louvain_communities(G, resolution=resolution, seed=7)
    return [sorted(c) for c in comms if len(c) >= 3]


def report(members: list[str]) -> str:
    from app.services.gst_graph import get_graph, summary_of
    from app.services.llm_service import generate

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256("|".join(members).encode()).hexdigest()
    if key not in cache:
        graph = get_graph()
        lines = "\n".join(f"- {graph.chunk_for(m).source}: {summary_of(graph.records[m])[:250]}" for m in members[:40])
        try:
            cache[key] = generate(REPORT, lines)["text"].strip()
        except Exception:  # noqa: BLE001
            cache[key] = lines[:1500]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache[key]


def build(resolution: float) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph
    from app.services.vector_store import get_client

    graph = get_graph()
    comms = communities(resolution)
    print(f"{len(comms)} communities (≥ 3 provisions); sizes: {sorted(len(c) for c in comms)[-10:]} …")
    texts = []
    for i, c in enumerate(comms, start=1):
        texts.append(report(c) + "\nProvisions: " + "; ".join(graph.chunk_for(m).source for m in c[:30]))
        if i % 10 == 0:
            print(f"  {i} reports")
    vecs = embed_texts(texts, model=MODEL)
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    client.upsert(COLLECTION, points=[PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(c))), vector=v,
                                                  payload={"members": c, "report": t[:2000]})
                                      for c, t, v in zip(comms, texts, vecs, strict=True)])
    print(f"{COLLECTION}: {len(comms)} community nodes")


def retrieve(question: str, n: int) -> list:
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve
    from app.services.sparse_vector_service import fuse_rrf
    from app.services.vector_store import get_client, search_within

    graph = get_graph()
    q = embed_texts([question])[0]
    hits = get_client().query_points(COLLECTION, query=q, limit=n, with_payload=["members"]).points
    members = [m for h in hits for m in h.payload["members"]]
    ranked = [graph.chunk_for(i) for i in search_within(q, members, top_k=20)]
    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    return cap_doc_types(fuse_rrf([pool, ranked], weights=[1.0, 0.5]), graph, 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval", "show"])
    ap.add_argument("--resolution", type=float, default=8.0)  # 1.0 gives 13 huge communities (hubs s.2, s.9)
    ap.add_argument("--communities", type=int, default=3)
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "show":
        cs = communities(args.resolution)
        print(len(cs), "communities; largest:", cs[max(range(len(cs)), key=lambda i: len(cs[i]))][:12])
    elif args.cmd == "build":
        build(args.resolution)
    else:
        evaluate(f"t27_graphrag_c{args.communities}_{args.set}", lambda q: retrieve(q, args.communities), args.set)


if __name__ == "__main__":
    main()
