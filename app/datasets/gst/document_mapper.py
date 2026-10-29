from __future__ import annotations

import uuid
from typing import Any

from app.config import settings
from app.security.input_restructuring import count_tokens

_METADATA_FIELDS = (
    "instrument_id",
    "instrument_title",
    "provision_type",
    "provision_number",
    "provision_title",
    "chapter_number",
    "chapter_title",
    "source_hash",
)


def select_embedding_text(record: dict[str, Any]) -> str:
    """Pick the text to embed for a provision.

    Prefers retrieval_text_enriched (summary + full text + topics/keywords/
    conditions/obligations) per the chosen embedding profile, falling back
    gracefully if a record is missing it.
    """
    for field in ("retrieval_text_enriched", "retrieval_text_exact", "analysis_text", "raw_text"):
        text = record.get(field)
        if text and text.strip():
            return text.strip()
    return ""


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    """Split text into overlapping token windows only if it exceeds max_tokens.

    Most provisions are well under the limit and pass through unchanged as a
    single chunk; only the oversized minority get split, using tiktoken
    (cl100k_base, same encoding as app.security.input_restructuring) so chunk
    boundaries are measured in the same units as the embedding model's limit.
    """
    max_tokens = max_tokens if max_tokens is not None else settings.gst_chunk_max_tokens
    overlap_tokens = (
        overlap_tokens if overlap_tokens is not None else settings.gst_chunk_overlap_tokens
    )

    if count_tokens(text) <= max_tokens:
        return [text]

    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    stride = max(max_tokens - overlap_tokens, 1)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride

    return chunks


def _point_id(provision_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point ID.

    Qdrant point IDs must be an unsigned int or UUID — the corpus's natural
    string IDs (e.g. "cgst_act_2017.section.1") aren't valid directly. Using
    uuid5 (deterministic) instead of uuid4 (random) means re-running ingestion
    upserts the same points instead of duplicating them.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{provision_id}::chunk{chunk_index}"))


def source_label(record: dict[str, Any]) -> str:
    """Human-readable `source` payload value, e.g. "The Central Goods and Services Tax Act, 2017, Section 16".

    This is what retrieval returns as RetrievedChunk.source; eval/gst_retrieval.py maps it back to the
    corpus `id`, so keep the two in sync via this one function.
    """
    return (
        f"{record.get('instrument_title', '')}, "
        f"{str(record.get('provision_type', '')).capitalize()} {record.get('provision_number', '')}"
    ).strip()


def map_provision_to_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one corpus record into 1+ Qdrant point dicts (id, text, payload)."""
    provision_id = record.get("id", "")
    text = select_embedding_text(record)
    if not text:
        return []

    chunks = chunk_text(text)
    metadata = {field: record.get(field) for field in _METADATA_FIELDS}
    source = source_label(record)

    points = []
    for idx, chunk in enumerate(chunks):
        points.append(
            {
                "point_id": _point_id(provision_id, idx),
                "provision_id": provision_id,
                "source": source,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "text": chunk,
                **metadata,
            }
        )
    return points
