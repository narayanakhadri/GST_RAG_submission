"""T16 — Fine-tuning an open embedding model on synthetic GST question → provision pairs.

Idea. General embedders do not know that "free gifts to employees" is about Schedule I para 2 or that "blocked
credit" means s.17(5). Contrastive fine-tuning on (question, provision) pairs teaches the vocabulary of this corpus.
LQ-RAG (Legal Query RAG, 2025) reports +13% hit rate and +15% MRR from a fine-tuned legal embedder; the academic
RAG framework (paper 24) also fine-tunes its embedder. Training data must not come from the golden set (leakage):
here it is synthetic — the questions Codex wrote for each provision (`answers_questions`, 1,094) plus the
enrichment LLM's `possible_questions` — all generated from the law text alone.

How it works here (free; CPU).
- Base model: BAAI/bge-small-en-v1.5 (33M parameters, trains on CPU in ~10–30 min) or bge-base (`--base`).
- Pairs: (question, provision embedded text); loss = MultipleNegativesRankingLoss (in-batch negatives, the
  standard recipe for retrieval fine-tuning); 1 epoch, batch 32, lr 2e-5.
- Hold-out check: 10% of provisions are kept out of training; their questions give a quick before/after hit@8.
- `build`: embeds the corpus with the tuned model into `gst_ft_<base>`; `eval` = dense-only top 8 on the golden set
  (compare with the same base model untuned: `--untuned`, and with t15's openai dense control).

    uv run python -m scripts.techniques.t16_finetune_embeddings train
    uv run python -m scripts.techniques.t16_finetune_embeddings build
    uv run python -m scripts.techniques.t16_finetune_embeddings eval --set all
    uv run python -m scripts.techniques.t16_finetune_embeddings eval --set all --untuned
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.config import settings
from scripts.techniques.common import evaluate

OUT = Path("eval/results/thesis/models")
CODEX = Path("seed/docs/gst/codex_metadata.jsonl")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction


def pairs() -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, str]]:
    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import select_embedding_text

    recs = {r["id"]: r for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"}
    docs = {pid: select_embedding_text(r)[:4000] for pid, r in recs.items()}
    qs: list[tuple[str, str]] = []
    for line in CODEX.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qs += [(q, row["id"]) for q in row.get("answers_questions") or [] if row["id"] in docs]
    for pid, r in recs.items():
        qs += [(q, pid) for q in (r.get("llm_metadata") or {}).get("possible_questions") or []]
    ids = sorted(docs)
    random.Random(7).shuffle(ids)
    held = set(ids[: len(ids) // 10])
    train = [(q, docs[p]) for q, p in qs if p not in held]
    test = [(q, p) for q, p in qs if p in held]
    return train, test, docs


def train(base: str) -> None:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    tr, _, _ = pairs()
    print(f"{len(tr)} training pairs")
    model = SentenceTransformer(base)
    data = DataLoader([InputExample(texts=[QUERY_PREFIX + q, d]) for q, d in tr], shuffle=True, batch_size=32)
    model.fit(train_objectives=[(data, losses.MultipleNegativesRankingLoss(model))], epochs=1, warmup_steps=20,
              optimizer_params={"lr": 2e-5}, show_progress_bar=True)
    path = OUT / f"ft_{base.split('/')[-1]}"
    model.save(str(path))
    print(f"saved {path}")
    heldout_check(base, str(path))


def heldout_check(base: str, tuned: str) -> None:
    """hit@8 of held-out provisions' questions over all provisions, before vs after tuning."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _, test, docs = pairs()
    ids = list(docs)
    for name in (base, tuned):
        m = SentenceTransformer(name)
        D = m.encode([docs[i] for i in ids], batch_size=16, normalize_embeddings=True)
        Q = m.encode([QUERY_PREFIX + q for q, _ in test], batch_size=32, normalize_embeddings=True)
        top = np.argsort(-(Q @ D.T), axis=1)[:, :8]
        hit = np.mean([ids.index(p) in row for (_, p), row in zip(test, top, strict=True)])
        print(f"{name}: held-out hit@8 = {hit:.3f} on {len(test)} questions")


def build(model_path: str, coll: str) -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import map_provision_to_points
    from app.services.vector_store import get_client

    m = SentenceTransformer(model_path)
    points = [p for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"
              for p in map_provision_to_points(r)]
    vecs = m.encode([p["text"] for p in points], batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    client = get_client()
    if client.collection_exists(coll):
        client.delete_collection(coll)
    client.create_collection(coll, vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE))
    for s in range(0, len(points), 128):
        client.upsert(coll, points=[PointStruct(id=p["point_id"], vector=v.tolist(),
                                                payload={k: x for k, x in p.items() if k != "point_id"})
                                    for p, v in zip(points[s:s + 128], vecs[s:s + 128], strict=True)])
    print(f"{coll}: {len(points)} points")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["train", "build", "eval"])
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--untuned", action="store_true", help="use the base model instead of the fine-tuned one")
    ap.add_argument("--set", default="all")
    args = ap.parse_args()
    short = args.base.split("/")[-1]
    model_path = args.base if args.untuned else str(OUT / f"ft_{short}")
    coll = f"gst_{'base' if args.untuned else 'ft'}_{short.replace('-', '_').replace('.', '_')}"
    if args.cmd == "train":
        train(args.base)
    elif args.cmd == "build":
        build(model_path, coll)
    else:
        from sentence_transformers import SentenceTransformer

        from app.models import RetrievedChunk
        from app.services.vector_store import get_client

        m = SentenceTransformer(model_path)

        def retrieve(q: str) -> list:
            v = m.encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()
            pts = get_client().query_points(coll, query=v, limit=8, with_payload=["text", "source"]).points
            return [RetrievedChunk(text=p.payload["text"], source=p.payload["source"], score=p.score) for p in pts]

        evaluate(f"t16_{'untuned' if args.untuned else 'finetuned'}_{short}_{args.set}", retrieve, args.set)


if __name__ == "__main__":
    main()
