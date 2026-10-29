"""T19 — ColBERT late-interaction re-ranking (Khattab & Zaharia 2020; ColBERTv2 2022) of the E17 pool.

Idea. Single-vector embeddings squeeze a whole provision into one point; cross-encoders read query + document
jointly but are slow and (here, MiniLM) truncate at 512 tokens and *hurt* recall (0.328). ColBERT keeps one vector
per token and scores with MaxSim: every query token finds its best-matching document token, and the maxima are
summed. Fine-grained like a cross-encoder, but document vectors can be pre-computed — a standard "third way" in
interviews (dense bi-encoder vs cross-encoder vs late interaction). Qdrant also supports multivectors natively
(MaxSim comparator) if one wants to index instead of re-rank.

How it works here (free; local CPU via fastembed).
- Model: answerdotai/answerai-colbert-small-v1 (fast, 96-dim token vectors) or colbert-ir/colbertv2.0 (`--model`).
- E17 pool (top 20, no cap) → ColBERT MaxSim score of each candidate's embedded text vs the question →
  final score = MaxSim (`--mode replace`) or RRF of E17 rank and ColBERT rank (`--mode fuse`) → cap 3 → top 8.
- Document token vectors are cached on disk per (model, provision) so repeated runs are fast.

    uv run python -m scripts.techniques.t19_colbert_rerank eval --set dev_hard --mode fuse
"""

from __future__ import annotations

import argparse
import hashlib
import threading
from pathlib import Path

import numpy as np

from scripts.techniques.common import E17_FLAGS, evaluate, use_best_stack

MODELS = {"small": "answerdotai/answerai-colbert-small-v1", "v2": "colbert-ir/colbertv2.0"}
CACHE_DIR = Path("eval/results/thesis/cache/colbert")
_lock, _models = threading.Lock(), {}


def model(name: str):
    from fastembed import LateInteractionTextEmbedding

    with _lock:
        if name not in _models:
            _models[name] = LateInteractionTextEmbedding(model_name=MODELS[name])
        return _models[name]


def doc_vectors(name: str, text: str) -> np.ndarray:
    key = hashlib.sha256(f"{name}|{text}".encode()).hexdigest()[:24]
    path = CACHE_DIR / name / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vec = np.asarray(next(iter(model(name).embed([text]))), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    return float((q @ d.T).max(axis=1).sum())


def retrieve(question: str, name: str, mode: str) -> list:
    from app.models import RetrievedChunk
    from app.services.gst_graph import cap_doc_types, get_graph
    from app.services.rag_service import _retrieve

    pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None})
    q = np.asarray(next(iter(model(name).query_embed(question))), dtype=np.float32)
    scores = [maxsim(q, doc_vectors(name, c.text)) for c in pool]
    by_colbert = sorted(range(len(pool)), key=lambda i: -scores[i])
    if mode == "replace":
        order = by_colbert
    else:  # RRF of the E17 rank and the ColBERT rank
        rank = {i: r for r, i in enumerate(by_colbert)}
        order = sorted(range(len(pool)), key=lambda i: -(1 / (60 + i + 1) + 1 / (60 + rank[i] + 1)))
    ranked = [RetrievedChunk(text=pool[i].text, source=pool[i].source, score=scores[i]) for i in order]
    return cap_doc_types(ranked, get_graph(), 8, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--model", choices=list(MODELS), default="small")
    ap.add_argument("--mode", choices=["replace", "fuse"], default="fuse")
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    use_best_stack()
    evaluate(f"t19_colbert_{args.model}_{args.mode}_{args.set}", lambda q: retrieve(q, args.model, args.mode), args.set)


if __name__ == "__main__":
    main()
