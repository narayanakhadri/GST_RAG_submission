"""T03 — GST knowledge graph in Neo4j + Cypher "context grouping" (proposal §7.4–7.5).

Idea. Sections, rules, notifications and circulars form a legal map: a rule prescribes the procedure of a section,
a notification entry sets the rate or exemption for supplies under a section, a circular clarifies a section.
Stored as a property graph, one Cypher query answers "give me everything attached to section 16": its rules,
clarifying circulars, rate entries, definitions — the proposal's context grouping.

The thesis experiments used an in-memory graph (app/services/gst_graph.py; 881 nodes do not need a database),
and E13c showed that expanding retrieval over typed links gave no recall gain on top of decomposition. This
script builds the same graph in Neo4j so it can be explored, queried and demonstrated (interviews), and offers
a Cypher-based expansion that plugs into the technique harness for comparison.

Graph.
- Nodes `(:Provision {id, label, kind, instrument, title, summary})` — kind = act / rules / notification /
  circular / faq. Plus `(:Instrument {id, title})` with `(:Provision)-[:PART_OF]->(:Instrument)`.
- Edges from the Codex sidecar (seed/docs/gst/codex_metadata.jsonl), one relationship type per relation:
  REFERS_TO, EMPOWERED_BY, RATE_OR_EXEMPTION_FOR, PROCEDURE_IN, CLARIFIES, COUNTERPART, AMENDS,
  DEFINES_TERM_USED_IN (with the `evidence` text as a property), and CROSS_REF for the parsed explicit
  references of the corpus.

Run Neo4j locally (no install besides Docker):
    docker run -d --name gst-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/gstgraph123 neo4j:5
    uv run python -m scripts.techniques.t03_neo4j_graph load           # nodes + edges via the HTTP API
    uv run python -m scripts.techniques.t03_neo4j_graph group cgst_act_2017.section.16
    uv run python -m scripts.techniques.t03_neo4j_graph eval --set dev_hard   # E17 + Cypher expansion
Browse at http://localhost:7474, e.g.
    MATCH (s:Provision {id:'cgst_act_2017.section.16'})-[r]-(n) RETURN s, r, n
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import source_label
from app.services.gst_graph import CODEX_LINKS, kind, summary_of

URL = os.environ.get("NEO4J_HTTP", "http://localhost:7474/db/neo4j/tx/commit")
AUTH = os.environ.get("NEO4J_AUTH", "neo4j/gstgraph123")
GROUP_RELATIONS = ("PROCEDURE_IN", "CLARIFIES", "RATE_OR_EXEMPTION_FOR", "DEFINES_TERM_USED_IN", "EMPOWERED_BY")


def cypher(statements: list[tuple[str, dict]]) -> list[dict]:
    user, pwd = AUTH.split("/", 1)
    body = json.dumps({"statements": [{"statement": s, "parameters": p} for s, p in statements]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read())
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["results"]


def load() -> None:
    records = [r for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"]
    nodes = [{"id": r["id"], "label": source_label(r), "kind": kind(r), "instrument": r["instrument_id"],
              "title": r.get("provision_title") or "", "summary": summary_of(r)[:500]} for r in records]
    cypher([("CREATE CONSTRAINT prov_id IF NOT EXISTS FOR (p:Provision) REQUIRE p.id IS UNIQUE", {}),
            ("MATCH (n) DETACH DELETE n", {})])
    for s in range(0, len(nodes), 200):
        cypher([("UNWIND $rows AS row MERGE (p:Provision {id: row.id}) SET p += row "
                 "MERGE (i:Instrument {id: row.instrument}) MERGE (p)-[:PART_OF]->(i)", {"rows": nodes[s:s + 200]})])
    edges: dict[str, list[dict]] = {}
    for line in CODEX_LINKS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            for link in row.get("links") or []:
                edges.setdefault(link["relation"].upper(), []).append(
                    {"a": row["id"], "b": link["target_id"], "evidence": link.get("evidence", "")[:300]})
    for r in records:
        for t in ("sections", "rules", "schedules"):
            for ref in (r.get("explicit_references") or {}).get(t) or []:
                if isinstance(ref, dict) and ref.get("target_id"):
                    edges.setdefault("CROSS_REF", []).append({"a": r["id"], "b": ref["target_id"], "evidence": ""})
    for rel, rows in edges.items():  # relationship types cannot be parameters in Cypher
        for s in range(0, len(rows), 500):
            cypher([(f"UNWIND $rows AS row MATCH (a:Provision {{id: row.a}}), (b:Provision {{id: row.b}}) "
                     f"MERGE (a)-[e:{rel}]->(b) SET e.evidence = row.evidence", {"rows": rows[s:s + 500]})])
    counts = cypher([("MATCH (p:Provision) RETURN count(p)", {}), ("MATCH ()-[e]->() RETURN type(e), count(e)", {})])
    print("nodes:", counts[0]["data"][0]["row"][0])
    for d in counts[1]["data"]:
        print(" ", *d["row"])


def group(provision_id: str, limit: int = 15) -> list[dict]:
    """Context grouping: the provisions attached to `provision_id` by the legally meaningful relations."""
    rels = "|".join(GROUP_RELATIONS)
    res = cypher([(f"MATCH (s:Provision {{id: $id}})-[e:{rels}]-(n:Provision) "
                   "RETURN n.id, n.label, type(e), startNode(e).id = s.id AS outgoing, e.evidence LIMIT $limit",
                   {"id": provision_id, "limit": limit})])
    return [dict(zip(["id", "label", "relation", "outgoing", "evidence"], d["row"], strict=True)) for d in res[0]["data"]]


def retrieve(question: str) -> list:
    """E17 retrieval, then the Cypher neighbours of the top 3 hits ranked by similarity to the question
    (same fusion as E13c, but the graph lives in Neo4j)."""
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import expand_with_graph, get_graph
    from app.services.rag_service import _retrieve
    from app.services.vector_store import search_within
    from scripts.techniques.common import E17_FLAGS

    chunks = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    graph = get_graph()
    seeds = [graph.id_of(c) for c in chunks[:3]]

    class CypherGraph:  # duck-typed for expand_with_graph: neighbours from Neo4j instead of memory
        records, label_to_id, id_of, chunk_for = graph.records, graph.label_to_id, graph.id_of, graph.chunk_for
        neighbours = {s: {n["id"] for n in group(s)} for s in seeds if s}

    q = embed_texts([question])[0]
    expanded = expand_with_graph(chunks, CypherGraph, seeds=3,
                                 rank_neighbours=lambda ids: search_within(q, ids, top_k=len(ids)))
    from app.services.gst_graph import cap_doc_types

    return cap_doc_types(expanded, graph, 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["load", "group", "eval"])
    ap.add_argument("provision", nargs="?")
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    if args.cmd == "load":
        load()
    elif args.cmd == "group":
        for n in group(args.provision):
            arrow = "->" if n["outgoing"] else "<-"
            print(f"{arrow} {n['relation']:24} {n['label']}  | {n['evidence'][:90]}")
    else:
        from scripts.techniques.common import evaluate, use_best_stack

        use_best_stack()
        evaluate(f"t03_neo4j_expand_{args.set}", retrieve, args.set)


if __name__ == "__main__":
    main()
