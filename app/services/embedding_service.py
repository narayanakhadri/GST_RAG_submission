import hashlib
import json
import threading
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.services.query_cache_service import query_cache

openai_client = OpenAI(api_key=settings.openai_api_key)

DEFAULT_CACHE_MODEL = "text-embedding-3-small"

# Optional persistent cache (settings.embedding_disk_cache, e.g. for eval runs): one JSON line per
# (model, text) so repeated experiments don't re-embed the same questions, and keep working offline.
_disk: dict[str, list[float]] | None = None
_disk_lock = threading.Lock()


def _disk_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{text}".encode()).hexdigest()


def _disk_cache() -> dict[str, list[float]]:
    global _disk
    if _disk is None:
        _disk = {}
        path = Path(settings.embedding_disk_cache)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    _disk[row["k"]] = row["v"]
    return _disk


def _disk_store(entries: dict[str, list[float]]) -> None:
    path = Path(settings.embedding_disk_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for k, v in entries.items():
            f.write(json.dumps({"k": k, "v": v}) + "\n")


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    if model is None:
        model = settings.embedding_model

    # The cache is keyed by text; prefix the model for non-default models so vectors from different
    # models (and dimensions) never mix. Default-model keys are unchanged, so existing cache entries stay valid.
    prefix = "" if model == DEFAULT_CACHE_MODEL else f"{model}::"
    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = query_cache.get_embedding(prefix + text)
        if cached is None and settings.embedding_disk_cache:
            with _disk_lock:
                cached = _disk_cache().get(_disk_key(model, text))
        if cached is not None:
            results[i] = cached

        else:
            miss_indices.append(i)
            miss_texts.append(text)

    if miss_texts:
        response = openai_client.embeddings.create(input=miss_texts, model=model)
        for idx_in_misses, item in enumerate(response.data):
            original_idx = miss_indices[idx_in_misses]
            vector = item.embedding
            results[original_idx] = vector
            query_cache.set_embedding(prefix + miss_texts[idx_in_misses], vector)
        if settings.embedding_disk_cache:
            new = {_disk_key(model, t): results[i] for i, t in zip(miss_indices, miss_texts, strict=True)}
            with _disk_lock:
                _disk_cache().update(new)
                _disk_store(new)

    return [r for r in results if r is not None]

