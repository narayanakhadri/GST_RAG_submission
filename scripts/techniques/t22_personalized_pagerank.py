"""T22 — Personalized PageRank over the provision graph (HippoRAG / HippoRAG 2 style graph retrieval).

Idea. HippoRAG (Gutiérrez et al., NeurIPS 2024) and HippoRAG 2 ("From RAG to Memory", ICML 2025; code
github.com/OSU-NLP-Group/HippoRAG) store knowledge as a graph and retrieve with Personalized PageRank (PPR): the
nodes that match the query are the "seeds", PPR spreads probability along the edges, and nodes reachable from
*several* seeds collect the most — multi-hop association in one step. HippoRAG 2 adds passage nodes to the graph
and an LLM filter on the seed triples. E13c expanded only the direct neighbours of each hit and gave no gain; PPR
differs by rewarding provisions connected to many hits at once (e.g. the rule that implements two retrieved
sections).

How it works here (free; networkx, no LLM).
- Nodes: provisions (passage nodes in HippoRAG 2 terms). Edges (undirected): parsed cross-references of the
  corpus + Codex typed links of the chosen relations (`--relations`, default procedure_in, rate_or_exemption_for,
  clarifies, empowered_by, refers_to, defines_term_used_in).
- Seeds: E17's top 20 with weight = fused score (the query-to-node match).
- PPR with damping `--alpha` (HippoRAG uses 0.5) → scores for all nodes.
- Final ranking: RRF of the E17 rank (weight 1) and the PPR rank (weight `--weight`, default 0.5) over the union,
  then doc-type cap 3 → top 8.

    uv run python -m scripts.techniques.t22_personalized_pagerank eval --set dev_hard
"""

from __future__ import annotations

import argparse
import threading

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

RELATIONS = ("procedure_in", "rate_or_exemption_for", "clarifies", "empowered_by", "refers_to", "defines_term_used_in")
_lock, _graph = threading.Lock(), {}


def nx_graph(relations: tuple[str, ...]):
    import networkx as nx

    from app.services.gst_graph import get_graph, get_typed_graph

    with _lock:
        if relations not in _graph:
            g = nx.Graph()
            base = get_graph()
            g.add_nodes_from(base.records)
            for a, ns in base.neighbours.items():
                g.add_edges_from((a, b) for b in ns)
            for a, ns in get_typed_graph(relations).neighbours.items():
                g.add_edges_from((a, b) for b in ns)
            _graph[relations] = g
        return _graph[relations]


def retrieve(question: str, alpha: float, weight: float, relations: tuple[str, ...]) -> list:
    import networkx as nx

    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve

    graph = get_graph()
    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    seeds = {}
    for c in pool:
        pid = graph.id_of(c)
        if pid:
            seeds[pid] = seeds.get(pid, 0.0) + max(c.score, 1e-6)
    if not seeds:
        return cap_doc_types(pool, graph, 8, 3)
    G = nx_graph(relations)
    ppr = nx.pagerank(G, alpha=alpha, personalization=seeds, max_iter=200, tol=1e-8)
    ppr_rank = {pid: r for r, pid in enumerate(sorted(ppr, key=ppr.get, reverse=True)[:60])}
    e17_rank = {graph.id_of(c): r for r, c in enumerate(pool)}
    fused = {}
    for pid in set(ppr_rank) | set(e17_rank):
        if pid is None:
            continue
        fused[pid] = (1 / (60 + e17_rank[pid] + 1) if pid in e17_rank else 0.0) + \
                     (weight / (60 + ppr_rank[pid] + 1) if pid in ppr_rank else 0.0)
    order = sorted(fused, key=fused.get, reverse=True)
    return cap_doc_types([graph.chunk_for(p) for p in order], graph, 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--weight", type=float, default=0.5)
    args = ap.parse_args()
    use_best_stack()
    evaluate(f"t22_ppr_a{args.alpha}_w{args.weight}_{args.set}",
             lambda q: retrieve(q, args.alpha, args.weight, RELATIONS), args.set)


if __name__ == "__main__":
    main()
