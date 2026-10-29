"""T13 — Query routing to document families (adaptive / routed retrieval).

Idea. Different GST questions live in different instruments: rates and exemptions in notifications, place of
supply in the IGST Act, procedures in the Rules, clarifications in circulars. One shared index makes all of them
compete; a router can predict which families a question needs and give each its own retrieval channel.
Sources: RAG survey ("routing … selecting the optimal pathway for a query"; semantic-router), HyPA-RAG (33: a query
classifier sets retrieval parameters), VersionRAG (30: intent classification routes to specialised retrieval),
NitiBench (2025: hierarchy-aware / section-based retrieval for tax law).

How it works here (free: one Gemini call per question, cached).
1. Router: Gemini picks 1–3 families from {cgst_act, cgst_rules, igst_act, utgst_act, rate_notifications,
   exemption_notifications, other_notifications, circulars} and returns JSON.
2. For each family: dense search restricted to its instruments (Qdrant payload filter on instrument_id), top 10.
3. Fusion: E17's ranked pool (weight 1) + each family ranking (weight `--weight`, default 0.5) by RRF, then the
   usual doc-type cap 3 → top 8.

    uv run python -m scripts.techniques.t13_query_routing route "What is the GST rate on restaurant service?"
    uv run python -m scripts.techniques.t13_query_routing eval --set dev_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

CACHE = Path("eval/results/thesis/cache/query_routes.json")
FAMILIES = {
    "cgst_act": ["cgst_act_2017"],
    "cgst_rules": ["cgst_rules_2017"],
    "igst_act": ["igst_act_2017", "igst_rules_2017"],
    "utgst_act": ["utgst_act_2017"],
    "rate_notifications": ["notification_1_2017_ct_rate", "notification_11_2017_ct_rate", "notification_8_2017_it_rate"],
    "exemption_notifications": ["notification_12_2017_ct_rate", "notification_9_2017_it_rate",
                                "notification_2_2019_ct_rate"],
    "reverse_charge_notifications": ["notification_13_2017_ct_rate", "notification_10_2017_it_rate"],
    "other_notifications": ["notification_10_2019_ct", "notification_14_2019_ct", "notification_16_2020_ct",
                            "notification_50_2018_ct", "notification_66_2017_ct", "notification_17_2017_ct_rate"],
    "circulars": ["circular_"],  # prefix: every circular
}
ROUTER_PROMPT = ("You route Indian GST law questions to the document families that answer them. Families: "
                 + ", ".join(FAMILIES) + ". Return JSON {\"families\": [1-3 names, most important first]}.")


def route(question: str) -> list[str]:
    from app.services.llm_service import generate_with_json, grader_model_default

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{ROUTER_PROMPT}|{question}".encode()).hexdigest()
    if key not in cache:
        try:
            fams = json.loads(generate_with_json(ROUTER_PROMPT, question)["text"]).get("families") or []
        except Exception:  # noqa: BLE001
            fams = []
        cache[key] = [f for f in fams if f in FAMILIES][:3]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def family_search(question: str, family: str, top_k: int = 10) -> list:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from app.config import settings
    from app.models import RetrievedChunk
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph
    from app.services.vector_store import get_client

    insts = FAMILIES[family]
    if insts == ["circular_"]:
        insts = sorted({r["instrument_id"] for r in get_graph().records.values() if r["instrument_id"].startswith("circular_")})
    flt = Filter(must=[FieldCondition(key="instrument_id", match=MatchAny(any=insts))])
    pts = get_client().query_points(settings.qdrant_collection, query=embed_texts([question])[0], query_filter=flt,
                                    limit=top_k, with_payload=["text", "source"]).points
    return [RetrievedChunk(text=p.payload["text"], source=p.payload["source"], score=p.score) for p in pts]


def retrieve(question: str, weight: float = 0.5) -> list:
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve
    from app.services.sparse_vector_service import fuse_rrf

    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    lists = [pool] + [family_search(question, f) for f in route(question)]
    fused = fuse_rrf(lists, weights=[1.0] + [weight] * (len(lists) - 1))
    return cap_doc_types(fused, get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["route", "eval"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--weight", type=float, default=0.5)
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "route":
        print(route(args.question))
    else:
        evaluate(f"t13_routing_w{args.weight}_{args.set}", lambda q: retrieve(q, args.weight), args.set)


if __name__ == "__main__":
    main()
