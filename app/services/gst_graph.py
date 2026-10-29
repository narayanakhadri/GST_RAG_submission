"""Lightweight GST provision graph and two retrieval post-processors built on it.

The research proposal (§7.4–7.5) links Sections ↔ Rules ↔ Notifications ↔ Circulars in a knowledge graph
and expands retrieval with linked provisions. This module builds that graph in memory from the corpus
JSONL (no Neo4j needed at this size: 881 nodes):

- An edge for every resolved cross-reference in a record's `explicit_references`
  (sections / rules / schedules with a `target_id`), stored in both directions.
- Notification/circular entries usually carry no references of their own, but the instrument's preamble
  paragraphs do ("in exercise of the powers conferred by section 11(1)..."). Each entry is linked to the
  provisions its instrument's paragraphs reference.

`expand_with_graph()` adds the neighbours of the top hits (proposal: "context grouping"), and
`cap_doc_types()` limits how many notification/circular items fill the final top-k, because short
rate-table entries otherwise crowd out the Act sections the answers need.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

from app.config import settings
from app.datasets.gst.document_mapper import select_embedding_text, source_label
from app.models import RetrievedChunk

# CBIC FAQs (v4 corpus) are clarifications like circulars: short, numerous records that would otherwise fill the
# top-k on their topic (TCS questions lost s.52 to eight FAQ entries), so the doc-type cap counts them too.
SUBORDINATE_KINDS = ("notification", "circular", "faq")
_REFERENCE_TYPES = ("sections", "rules", "schedules")


class ProvisionGraph:
    def __init__(self, records: list[dict]) -> None:
        self.records = {r["id"]: r for r in records}
        self.label_to_id = {source_label(r): r["id"] for r in records}
        self.neighbours: dict[str, set[str]] = defaultdict(set)

        def link(a: str, b: str) -> None:
            if a != b and a in self.records and b in self.records:
                self.neighbours[a].add(b)
                self.neighbours[b].add(a)

        instrument_refs: dict[str, set[str]] = defaultdict(set)
        for r in records:
            for target in _reference_targets(r):
                link(r["id"], target)
                if kind(r) in SUBORDINATE_KINDS:
                    instrument_refs[r["instrument_id"]].add(target)
        for r in records:
            if kind(r) in SUBORDINATE_KINDS and r.get("provision_type") == "entry":
                for target in instrument_refs[r["instrument_id"]]:
                    link(r["id"], target)

    def id_of(self, chunk: RetrievedChunk) -> str | None:
        return self.label_to_id.get(chunk.source)

    def chunk_for(self, provision_id: str, score: float = 0.0) -> RetrievedChunk:
        """The chunk retrieval would return for this provision (same text and label as the Qdrant payload)."""
        record = self.records[provision_id]
        text = select_embedding_text(record)
        return RetrievedChunk(text=text, source=source_label(record), score=score)


def law_text(record: dict) -> str:
    """The provision as law only: its label, title and raw text, without the LLM-written metadata that the embedded
    text carries (summary, topics, keywords, possible questions)."""
    title = record.get("provision_title") or ""
    heading = source_label(record) + (f": {title}" if title else "")
    return f"{heading}\n\n{(record.get('raw_text') or '').strip()}"


_corrected_summaries: dict[str, str] | None = None


def corrected_summaries() -> dict[str, str]:
    """Codex-corrected summaries (codex_metadata.jsonl) for the provisions whose summary Codex judged wrong or
    partly wrong."""
    global _corrected_summaries
    if _corrected_summaries is None:
        out = {}
        if CODEX_LINKS.exists():
            for line in CODEX_LINKS.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    check = row.get("summary_check") or {}
                    if check.get("verdict") != "correct" and (check.get("corrected_summary") or "").strip():
                        out[row["id"]] = check["corrected_summary"].strip()
        _corrected_summaries = out
    return _corrected_summaries


def summary_of(record: dict) -> str:
    return corrected_summaries().get(record["id"]) or ((record.get("llm_metadata") or {}).get("summary") or "").strip()


def to_law_context(chunks: list[RetrievedChunk], graph: ProvisionGraph, with_summary: bool = False) -> list[RetrievedChunk]:
    """Swap each retrieved chunk's text for the provision's law text (one chunk per provision; ranking, sources and
    scores unchanged), optionally preceded by its summary. Chunks that don't map to a corpus record are kept."""
    out, seen = [], set()
    for chunk in chunks:
        pid = graph.id_of(chunk)
        if pid is None:
            out.append(chunk)
        elif pid not in seen:
            seen.add(pid)
            record = graph.records[pid]
            text = law_text(record)
            summary = summary_of(record) if with_summary else ""
            if summary:
                heading, _, body = text.partition("\n\n")
                text = f"{heading}\n\nSummary: {summary}\n\n{body}"
            out.append(RetrievedChunk(text=text, source=chunk.source, score=chunk.score))
    return out


def kind(record: dict) -> str:
    """"notification" / "circular" for subordinate instruments, else "act" or "rules"."""
    if record.get("instrument_kind"):
        return record["instrument_kind"]
    return "rules" if "rules" in record.get("instrument_id", "") else "act"


def _reference_targets(record: dict) -> list[str]:
    refs = record.get("explicit_references") or {}
    return [ref["target_id"] for t in _REFERENCE_TYPES for ref in refs.get(t) or []
            if isinstance(ref, dict) and ref.get("target_id")]


class TypedLinkGraph(ProvisionGraph):
    """Graph from the Codex sidecar's typed links (seed/docs/gst/codex_metadata.jsonl), keeping only the given
    relation types (thesis E13c). A link is stored in both directions: a retrieved section brings in the rule
    that prescribes its procedure (`procedure_in`) and the circulars that clarify it (`clarifies`, stored on the
    circular), and a retrieved rule or circular brings in its section."""

    def __init__(self, records: list[dict], links_path: Path, relations: tuple[str, ...]) -> None:
        self.records = {r["id"]: r for r in records}
        self.label_to_id = {source_label(r): r["id"] for r in records}
        self.neighbours = defaultdict(set)
        for line in links_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for link in row.get("links") or []:
                if link.get("relation") not in relations:
                    continue
                # A whole-instrument target ("notification_12_2017_ct_rate") is not a node; skip it.
                a, b = row["id"], link.get("target_id")
                if a != b and a in self.records and b in self.records:
                    self.neighbours[a].add(b)
                    self.neighbours[b].add(a)


CODEX_LINKS = Path("seed/docs/gst/codex_metadata.jsonl")
_graphs: dict[tuple, ProvisionGraph] = {}
_graph_lock = threading.Lock()


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_graph(corpus_path: str | None = None) -> ProvisionGraph:
    """Cross-reference graph of the corpus, cached per corpus path (runs on different corpora can share a
    process: a graph of the wrong corpus would not know the other corpus's records)."""
    path = str(corpus_path or settings.gst_corpus_path)
    with _graph_lock:
        if path not in _graphs:
            _graphs[path] = ProvisionGraph(_load_records(Path(path)))
        return _graphs[path]


def get_typed_graph(relations: tuple[str, ...], corpus_path: str | None = None) -> ProvisionGraph:
    path = str(corpus_path or settings.gst_corpus_path)
    key = (path, "codex", tuple(sorted(relations)))
    with _graph_lock:
        if key not in _graphs:
            _graphs[key] = TypedLinkGraph(_load_records(Path(path)), CODEX_LINKS, tuple(relations))
        return _graphs[key]


def expand_with_graph(
    chunks: list[RetrievedChunk],
    graph: ProvisionGraph,
    seeds: int = 5,
    max_new: int = 10,
    weight: float = 0.5,
    rrf_k: int = 60,
    rank_neighbours=None,
) -> list[RetrievedChunk]:
    """Re-rank `chunks` together with the graph neighbours of its top `seeds` items.

    Neighbours are ordered either by link strength — the sum of 1/(rrf_k + rank) over the seeds that link
    to them, so a provision linked from several top hits ranks higher — or, if `rank_neighbours(ids)` is
    given, by that function (e.g. dense similarity to the query, restricted to the neighbour set). The
    neighbour list is then fused with the original ranking by weighted RRF (`weight` < 1 keeps direct hits
    ahead of equally ranked neighbours).
    """
    present = {graph.id_of(c) for c in chunks}
    neighbour_score: dict[str, float] = defaultdict(float)
    for rank, chunk in enumerate(chunks[:seeds]):
        pid = graph.id_of(chunk)
        for n in graph.neighbours.get(pid, ()) if pid else ():
            if n not in present:
                neighbour_score[n] += 1.0 / (rrf_k + rank + 1)
    if rank_neighbours is not None and neighbour_score:
        new_ids = [n for n in rank_neighbours(list(neighbour_score)) if n in neighbour_score][:max_new]
    else:
        new_ids = sorted(neighbour_score, key=neighbour_score.get, reverse=True)[:max_new]

    fused: dict[str, float] = defaultdict(float)
    by_key: dict[str, RetrievedChunk] = {}
    for rank, chunk in enumerate(chunks):
        fused[chunk.source] += 1.0 / (rrf_k + rank + 1)
        by_key.setdefault(chunk.source, chunk)
    for rank, pid in enumerate(new_ids):
        chunk = graph.chunk_for(pid)
        fused[chunk.source] += weight / (rrf_k + rank + 1)
        by_key.setdefault(chunk.source, chunk)
    ordered = sorted(fused, key=fused.get, reverse=True)
    return [RetrievedChunk(text=by_key[k].text, source=k, score=fused[k]) for k in ordered]


def cap_doc_types(
    chunks: list[RetrievedChunk],
    graph: ProvisionGraph,
    top_k: int,
    cap: int | None = None,
    per_instrument: int | None = None,
) -> list[RetrievedChunk]:
    """Take the top `top_k` chunks in order, allowing at most `cap` notification/circular items in total and
    at most `per_instrument` items from any one instrument (None = no limit); if that leaves fewer than
    `top_k`, fill up with the skipped items in their original order."""
    chosen, skipped, subordinate = [], [], 0
    per: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        pid = graph.id_of(chunk)
        record = graph.records.get(pid) if pid else None
        is_sub = record is not None and kind(record) in SUBORDINATE_KINDS
        instrument = record["instrument_id"] if record else None
        if (cap is not None and is_sub and subordinate >= cap) or (
            per_instrument is not None and instrument and per[instrument] >= per_instrument
        ):
            skipped.append(chunk)
            continue
        chosen.append(chunk)
        subordinate += is_sub
        if instrument:
            per[instrument] += 1
        if len(chosen) == top_k:
            return chosen
    return chosen + skipped[: top_k - len(chosen)]
