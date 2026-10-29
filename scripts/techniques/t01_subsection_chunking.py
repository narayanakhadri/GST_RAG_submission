"""T01 — Sub-section chunking + chunk context (SAC / Contextual Retrieval) + small-to-big / HiChunk auto-merge.

Idea. One record per provision makes long sections (s.16, s.17, rule 36 …) a single embedding: a question about
s.17(5) has to match a vector that also encodes s.17(1)–(4). Splitting at the sub-section markers "(1)", "(2)" …
gives smaller, sharper units — but a sub-section alone loses its context ("such person" — which person?). Three
ways to give each child its context back are offered (`--context`):

- none        heading only: "<Act>, Section 16: Eligibility and conditions for taking input tax credit — (2)".
- sac         Summary-Augmented Chunking (Reuter et al., paper 4; code github.com/DevelopedByMarkus/
              summary-augmented-chunking): their chunks are "[document summary] <~150-char summary>\n[content]
              <chunk>". Here the "document" is the parent provision; its (Codex-corrected) summary is cut to 170
              characters as in their code (`summary_truncation_length = 170`).
- contextual  Anthropic Contextual Retrieval (2024): an LLM writes 50–100 tokens situating each child inside its
              parent ("This sub-section of section 16 lists the four conditions …"), prepended before embedding
              (and before the sparse index). Uses the free Gemini keys; cached on disk.

Retrieval (`--mode`):
- parent   small-to-big / parent-document retrieval (LawRAG, paper 10): search children, return their parent
           provisions; a parent hit by several children gets +0.02 per extra child.
- hichunk  HiChunk Auto-Merge (paper 31; code github.com/TencentYoutuResearch/HiChunk, `retrieval_algo.py`,
           `check_merge_condition`): return the *children* as context, and replace children by their parent only
           when (1) ≥ 2 children of that parent are retrieved, (2) their characters reach
           threshold = (used_tokens / budget) · parent_chars / 3 + parent_chars / 3, and (3) the rest of the parent
           fits in the remaining token budget. Context is smaller than "parent"; recall is scored on parent IDs.

    uv run python -m scripts.techniques.t01_subsection_chunking stats
    uv run python -m scripts.techniques.t01_subsection_chunking build --context sac         # ≈ $0.12 (asks)
    uv run python -m scripts.techniques.t01_subsection_chunking eval --context sac --mode hichunk --set dev_hard

Compare with: large_dense (one vector per provision, same model) and e17_v3_textfix (best stack).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import defaultdict
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import source_label
from app.models import RetrievedChunk
from app.services.embedding_service import embed_texts
from scripts.techniques.common import chunks_for_ids, evaluate

MODEL, DIMS = "text-embedding-3-large", 3072
SUBSECTION = re.compile(r"^\s*[-–]?\s*\d*\s*\[?\((\d{1,2}[A-Z]{0,2})\)", re.M)
SPLIT_KINDS = ("cgst_act_2017", "cgst_rules_2017", "igst_act_2017", "igst_rules_2017", "utgst_act_2017")
MERGE_BONUS = 0.02
SAC_CHARS = 170
TOKEN_BUDGET = 6000  # context budget for hichunk mode (tokens ≈ chars / 4)
CONTEXT_CACHE = Path("eval/results/thesis/cache/contextual_chunks.json")
CONTEXT_PROMPT = (
    "Here is a provision of Indian GST law:\n<provision>\n{parent}\n</provision>\n"
    "Here is a part of it:\n<part>\n{child}\n</part>\n"
    "Write 1-2 sentences (at most 80 words) that situate this part within the provision, to improve search "
    "retrieval of the part: say which provision it belongs to and what it covers. Answer only with the sentences."
)


def collection(context: str) -> str:
    return f"gst_thesis_subsec_{context}_large"


def heading(r: dict) -> str:
    title = r.get("provision_title") or ""
    return source_label(r) + (f": {title}" if title else "")  # the label already starts with the instrument


def children(r: dict) -> list[tuple[str, str]]:
    """[(child key, body text)] — the sub-sections of a provision, or the whole provision if it has < 2 markers."""
    raw = r.get("raw_text") or ""
    marks = list(SUBSECTION.finditer(raw)) if r["instrument_id"] in SPLIT_KINDS else []
    seen, starts = set(), []
    for m in marks:  # first occurrence of each number, in order ("(1)" of a later clause list is not a split)
        if m.group(1) not in seen:
            seen.add(m.group(1))
            starts.append((m.group(1), m.start()))
    if len(starts) < 2:
        return [("all", raw.strip())]
    out = []
    if starts[0][1] > 40:  # text before (1), e.g. an opening sentence
        out.append(("pre", raw[: starts[0][1]].strip()))
    for i, (num, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(raw)
        out.append((num, raw[start:end].strip()))
    return out


def _contextual(r: dict, key: str, body: str, cache: dict) -> str:
    from app.services.llm_service import generate

    ck = hashlib.sha256(f"{r['id']}#{key}#{body[:200]}".encode()).hexdigest()
    if ck not in cache:
        parent = (r.get("raw_text") or "")[:12000]
        cache[ck] = generate("You write short retrieval context.", CONTEXT_PROMPT.format(parent=parent, child=body[:4000]))[
            "text"].strip()
    return cache[ck]


def unit_text(r: dict, key: str, body: str, context: str, cache: dict | None = None) -> str:
    head = heading(r) + ("" if key in ("all", "pre") else f" — sub-section ({key})")
    if context == "sac" and key != "all":
        from app.services.gst_graph import summary_of

        return f"{head}\n[document summary] {summary_of(r)[:SAC_CHARS]}\n[content] {body}"
    if context == "contextual" and key != "all":
        return f"{head}\n{_contextual(r, key, body, cache if cache is not None else {})}\n\n{body}"
    return f"{head}\n\n{body}"


def units(context: str) -> list[tuple[dict, str, str]]:
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8")) if CONTEXT_CACHE.exists() else {}
    out = []
    for r in load_corpus(settings.gst_corpus_path):
        if r.get("status") == "omitted":
            continue
        for key, body in children(r):
            out.append((r, key, unit_text(r, key, body, context, cache)[:24000]))
    if context == "contextual":
        CONTEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CONTEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


def build(context: str) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.services.vector_store import get_client

    if context == "contextual":
        print("Contextual mode: one Gemini call per sub-section (~1,500 calls, free keys, cached).")
    us = units(context)
    tokens = sum(len(t) for *_, t in us) / 4 / 1e6
    print(f"{len(us)} units ≈ {tokens:.2f}M tokens ≈ ${tokens * 0.13:.2f} with {MODEL}")
    if input(f"Embed into {collection(context)}? [y/N] ").strip().lower() != "y":
        return
    client = get_client()
    name = collection(context)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    for s in range(0, len(us), 64):
        batch = us[s : s + 64]
        vecs = embed_texts([t for *_, t in batch], model=MODEL)
        client.upsert(name, points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r['id']}#{key}")), vector=v,
                        payload={"parent_id": r["id"], "child": key, "text": t, "source": source_label(r),
                                 "chars": len(t)})
            for (r, key, t), v in zip(batch, vecs, strict=True)])
    print(f"{name}: {len(us)} points")


def _hits(question: str, context: str, pool: int):
    from app.services.vector_store import get_client

    vec = embed_texts([question], model=MODEL)[0]
    return get_client().query_points(collection(context), query=vec, limit=pool,
                                     with_payload=["parent_id", "child", "text", "source", "chars"]).points


def retrieve_parent(question: str, context: str, top_k: int = 8, pool: int = 40) -> list[RetrievedChunk]:
    best, extra = {}, defaultdict(int)
    for h in _hits(question, context, pool):
        pid = h.payload["parent_id"]
        if pid in best:
            extra[pid] += 1
        else:
            best[pid] = h.score
    score = {pid: s + MERGE_BONUS * extra[pid] for pid, s in best.items()}
    return chunks_for_ids(sorted(score, key=score.get, reverse=True)[:top_k])


def retrieve_hichunk(question: str, context: str, top_k: int = 8, pool: int = 40) -> list[RetrievedChunk]:
    """HiChunk auto-merge over a two-level tree (parent provision → sub-sections), token budget TOKEN_BUDGET."""
    from app.services.gst_graph import get_graph

    graph = get_graph()
    picked: list[dict] = []  # {"parent", "children": [payload], "merged": bool}
    used = 0
    for h in _hits(question, context, pool):
        p = h.payload
        group = next((g for g in picked if g["parent"] == p["parent_id"]), None)
        if group and group["merged"]:
            continue
        if group is None:
            if len(picked) == top_k or used >= TOKEN_BUDGET:
                continue
            group = {"parent": p["parent_id"], "children": [], "merged": False}
            picked.append(group)
        group["children"].append(p)
        used += p["chars"] // 4
        record = graph.records.get(p["parent_id"])
        parent_chars = len(record.get("raw_text") or "") if record else 0
        child_chars = sum(c["chars"] for c in group["children"])
        threshold = (used / TOKEN_BUDGET) * parent_chars / 3 + parent_chars / 3
        fits = (parent_chars - child_chars) // 4 <= TOKEN_BUDGET - used
        if len(group["children"]) >= 2 and child_chars >= threshold and fits:
            group["merged"] = True
            used += (parent_chars - child_chars) // 4
    out = []
    for g in picked:
        if g["merged"] or g["children"][0]["child"] == "all":
            out += chunks_for_ids([g["parent"]])
        else:  # children as context, labelled with the parent so recall is scored on the provision
            out.append(RetrievedChunk(text="\n\n".join(c["text"] for c in g["children"]),
                                      source=g["children"][0]["source"], score=0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval", "stats"])
    ap.add_argument("--context", choices=["none", "sac", "contextual"], default="sac")
    ap.add_argument("--mode", choices=["parent", "hichunk"], default="parent")
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    if args.cmd == "stats":
        records = [r for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"]
        split = [r for r in records if len(children(r)) > 1]
        print(f"{len(split)} of {len(records)} provisions split; {sum(len(children(r)) for r in records)} units")
    elif args.cmd == "build":
        build(args.context)
    else:
        settings.embedding_model = MODEL
        fn = retrieve_parent if args.mode == "parent" else retrieve_hichunk
        evaluate(f"t01_subsec_{args.context}_{args.mode}_{args.set}", lambda q: fn(q, args.context), args.set)


if __name__ == "__main__":
    main()
