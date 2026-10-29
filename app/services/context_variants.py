"""Generation-context variants (techniques T04–T06; flag `context_variants`, a list applied in order).

- "relations"  (T04, proposal §7.6, HyPA-RAG KG triplets): append one extra context block that lists how the
  retrieved provisions relate — "Rule 36 prescribes the procedure of Section 16", "Circular 92/11/2019 clarifies
  Section 7" — from the Codex typed links, so the model sees the legal map, not just loose passages.
- "edge_order" (T05, "lost in the middle", Liu et al. 2023; RAG survey): LLMs use the start and end of a long
  context best, so the ranked chunks are re-ordered 1, 3, 5, …, 6, 4, 2 — the strongest at both edges.
- "compress"   (T06, contextual compression: LLMLingua / RECOMP family, RAG survey): keep each provision's
  heading and only its passages most similar to the question (TF-IDF, free), in their original order —
  cuts tokens; the risk is dropping a proviso the answer needs.

Each variant takes and returns list[RetrievedChunk]; sources and ranking stay traceable.
"""

from __future__ import annotations

import re

from app.models import RetrievedChunk

RELATIONS_SOURCE = "Relations between the retrieved provisions (typed links)"
_RELATION_TEXT = {
    "procedure_in": "{a} has its procedure/conditions prescribed in {b}",
    "rate_or_exemption_for": "{a} sets the rate or exemption for supplies under {b}",
    "clarifies": "{a} clarifies {b}",
    "empowered_by": "{a} is issued under the power in {b}",
    "defines_term_used_in": "{a} defines a term used in {b}",
    "amends": "{a} amends {b}",
    "counterpart": "{a} is the counterpart of {b} in another Act",
}
COMPRESS_PASSAGES = 3


def relations(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    import json

    from app.services.gst_graph import CODEX_LINKS, get_graph

    graph = get_graph()
    ids = [graph.id_of(c) for c in chunks]
    present = {i for i in ids if i}
    lines = []
    for line in CODEX_LINKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in present:
            continue
        for link in row.get("links") or []:
            b, rel = link.get("target_id"), link.get("relation")
            if b in present and rel in _RELATION_TEXT:
                la, lb = graph.chunk_for(row["id"]).source, graph.chunk_for(b).source
                lines.append("- " + _RELATION_TEXT[rel].format(a=la, b=lb))
    if not lines:
        return chunks
    block = "How the retrieved provisions relate:\n" + "\n".join(dict.fromkeys(lines))
    return [*chunks, RetrievedChunk(text=block, source=RELATIONS_SOURCE, score=0.0)]


def edge_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    front, back = chunks[0::2], chunks[1::2]
    return front + back[::-1]


def _passages(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def compress(chunks: list[RetrievedChunk], question: str, keep: int = COMPRESS_PASSAGES) -> list[RetrievedChunk]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    out = []
    for c in chunks:
        parts = _passages(c.text)
        if len(parts) <= keep + 1:
            out.append(c)
            continue
        head, body = parts[0], parts[1:]
        vec = TfidfVectorizer(stop_words="english").fit(body + [question])
        sims = linear_kernel(vec.transform([question]), vec.transform(body))[0]
        top = sorted(sorted(range(len(body)), key=lambda i: -sims[i])[:keep])
        text = "\n\n".join([head, *(body[i] for i in top), "[... other sub-sections omitted ...]"])
        out.append(RetrievedChunk(text=text, source=c.source, score=c.score))
    return out


def apply(chunks: list[RetrievedChunk], question: str, variants: list[str]) -> list[RetrievedChunk]:
    for v in variants:
        if v == "relations":
            chunks = relations(chunks)
        elif v == "edge_order":
            chunks = edge_order(chunks)
        elif v == "compress":
            chunks = compress(chunks, question)
        else:
            raise ValueError(f"unknown context variant {v!r}")
    return chunks
