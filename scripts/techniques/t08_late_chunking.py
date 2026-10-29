"""T08 — Late chunking (Günther et al., Jina AI, 2024/2025) vs naive chunking, with a local long-context embedder.

Idea. Naive chunking embeds each sub-section on its own, so "such person" or "the said goods" lose their referent.
Late chunking runs the *whole* provision (up to 8,192 tokens) through a long-context embedding model once, then
mean-pools the token embeddings inside each sub-section's span. Every sub-section vector has "read" the full
provision through attention — context without any LLM-written summary (compare T01 sac / contextual, which add
context as text). Same idea as Anthropic's Contextual Retrieval, but done inside the embedding model.

How it works here (free: the model runs locally on CPU).
- Model: jinaai/jina-embeddings-v2-small-en (8k context, 512 dims; the model used in the late-chunking paper
  family; needs trust_remote_code).
- Units: the same sub-sections as T01 (`children()`), prefixed by the provision heading.
- `build --method late`: one forward pass per provision → token embeddings → mean over each child's token span
  (found through the tokenizer's offset mapping) → Qdrant `gst_thesis_late_jina` (payload parent_id).
- `build --method naive`: each child encoded separately → `gst_thesis_naive_jina` (the control).
- `eval`: dense search over children → parent provisions (as T01 "parent" mode) → recall on dev_hard.

    uv run python -m scripts.techniques.t08_late_chunking build --method late
    uv run python -m scripts.techniques.t08_late_chunking build --method naive
    uv run python -m scripts.techniques.t08_late_chunking eval --method late --set dev_hard
"""

from __future__ import annotations

import argparse
import threading
import uuid
from collections import defaultdict

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import source_label
from scripts.techniques.common import chunks_for_ids, evaluate
from scripts.techniques.t01_subsection_chunking import children, heading

MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"
DIMS, MAX_TOKENS = 512, 8192
_lock, _model = threading.Lock(), {}


def collection(method: str) -> str:
    return f"gst_thesis_{method}_jina"


def load_model():
    from transformers import AutoModel, AutoTokenizer

    with _lock:
        if not _model:
            _model["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model["model"] = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()
        return _model["tok"], _model["model"]


def _token_embeddings(text: str):
    import torch

    tok, model = load_model()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0]
    with torch.no_grad():
        out = model(**enc).last_hidden_state[0]
    return out, offsets


def _normalise(v):
    return (v / (v.norm() + 1e-9)).tolist()


def encode_query(text: str) -> list[float]:
    emb, _ = _token_embeddings(text)
    return _normalise(emb.mean(0))


def provision_units(r: dict) -> tuple[str, list[tuple[str, int, int]]]:
    """Full text + (child key, start char, end char) spans of each child inside it."""
    head = heading(r) + "\n\n"
    text, spans = head, []
    for key, body in children(r):
        start = len(text)
        text += body + "\n\n"
        spans.append((key, start, start + len(body)))
    return text, spans


def late_vectors(r: dict) -> list[tuple[str, list[float]]]:
    text, spans = provision_units(r)
    emb, offsets = _token_embeddings(text)
    out = []
    for key, s, e in spans:
        idx = [i for i, (a, b) in enumerate(offsets.tolist()) if b > a and a >= s and b <= e]
        if not idx:  # child beyond the 8k window: fall back to encoding it alone
            out.append((key, encode_query(heading(r) + "\n\n" + text[s:e])))
            continue
        out.append((key, _normalise(emb[idx].mean(0))))
    return out


def naive_vectors(r: dict) -> list[tuple[str, list[float]]]:
    return [(key, encode_query(f"{heading(r)}\n\n{body}")) for key, body in children(r)]


def build(method: str) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from app.services.vector_store import get_client

    client, name = get_client(), collection(method)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE))
    records = [r for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"]
    fn = late_vectors if method == "late" else naive_vectors
    n = 0
    for i, r in enumerate(records, start=1):
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r['id']}#{key}")), vector=v,
                              payload={"parent_id": r["id"], "child": key, "source": source_label(r)})
                  for key, v in fn(r)]
        client.upsert(name, points=points)
        n += len(points)
        if i % 100 == 0:
            print(f"  {i}/{len(records)} provisions, {n} units")
    print(f"{name}: {n} points")


def retrieve(question: str, method: str, top_k: int = 8, pool: int = 40) -> list:
    from app.services.vector_store import get_client

    hits = get_client().query_points(collection(method), query=encode_query(question), limit=pool,
                                     with_payload=["parent_id"]).points
    best, extra = {}, defaultdict(int)
    for h in hits:
        pid = h.payload["parent_id"]
        if pid in best:
            extra[pid] += 1
        else:
            best[pid] = h.score
    score = {pid: s + 0.02 * extra[pid] for pid, s in best.items()}
    return chunks_for_ids(sorted(score, key=score.get, reverse=True)[:top_k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--method", choices=["late", "naive"], default="late")
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.method)
    else:
        evaluate(f"t08_{args.method}_jina_{args.set}", lambda q: retrieve(q, args.method), args.set)


if __name__ == "__main__":
    main()
