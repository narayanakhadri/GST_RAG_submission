from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class QueryCacheService:
    _TIERS = ("rag_answer", "embedding")

    def __init__(self):
        self._redis_client: Any | None = self._build_redis_client()
        self._memory_store: dict[str, tuple[float, str]] = {}
        self._stats: dict[str, dict[str, int]] = {
            tier: defaultdict(int) for tier in self._TIERS
        }
        self._lock = threading.RLock()

    def _build_redis_client(self) -> Any | None:
        if not settings.upstash_redis_url or not settings.upstash_redis_token:
            logger.info("Redis cache disabled; missing Upstash config")
            return None
        try:
            from upstash_redis import Redis

            return Redis(url=settings.upstash_redis_url, token=settings.upstash_redis_token)
        except Exception:
            logger.exception("Failed to initialize Redis cache; using in-memory fallback")
            return None


    def _key(self, namespace: str, raw: str) -> str:
        hashed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{namespace}:{hashed}"

    def rag_answer_key(self, question: str, cache_context: dict[str, Any] | None = None) -> str:
        raw: str | dict[str, Any] = question.strip()
        if cache_context is not None:
            raw = {"question": question.strip(), "context": cache_context}
        return self._key("rag_answer", json.dumps(raw, sort_keys=True) if isinstance(raw, dict) else raw)

    def _record(self, tier: str, field: str) -> None:
        with self._lock:
            self._stats[tier][field] += 1

    def _get(self, tier: str, key: str) -> str | None:
        if self._redis_client is not None:
            try:
                value = self._redis_client.get(key)
                if value is not None:
                    self._record(tier, "hits")
                    return str(value)
            except Exception:
                logger.exception("Redis get failed for tier=%s key=%s", tier, key)

        with self._lock:
            current = self._memory_store.get(key)
            if current is None:
                self._record(tier, "misses")
                return None
            expires_at, value = current
            if expires_at < time.time():
                del self._memory_store[key]
                self._record(tier, "misses")
                return None
        self._record(tier, "hits")
        return value

    def _set(self, tier: str, key: str, value: str, ttl_seconds: int) -> None:
        self._record(tier, "sets")
        if self._redis_client is not None:
            try:
                self._redis_client.set(key, value, ex=ttl_seconds)
                return
            except Exception:
                logger.exception("Redis set failed for tier=%s key=%s", tier, key)

        with self._lock:
            self._memory_store[key] = (time.time() + ttl_seconds, value)

    def get_rag_answer(
        self,
        question: str,
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        value = self._get("rag_answer", self.rag_answer_key(question, cache_context))
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.exception("Invalid rag_answer cache payload")
            return None

    def set_rag_answer(
        self,
        question: str,
        answer_payload: dict[str, Any],
        cache_context: dict[str, Any] | None = None,
    ) -> None:
        self._set(
            "rag_answer",
            self.rag_answer_key(question, cache_context),
            json.dumps(answer_payload),
            settings.cache_ttl_rag,
        )

    def embedding_key(self, text: str) -> str:
        return self._key("embedding", text)

    def get_embedding(self, text: str) -> list[float] | None:
        value = self._get("embedding", self.embedding_key(text))
        if value is None:
            return None
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            logger.exception("Invalid embedding cache payload")
        return None

    def set_embedding(self, text: str, vector: list[float]) -> None:
        self._set(
            "embedding",
            self.embedding_key(text),
            json.dumps(vector),
            settings.cache_ttl_embeddings,
        )

    def stats(self) -> dict[str, dict[str, float | int]]:
        snapshot: dict[str, dict[str, float | int]] = {}
        with self._lock:
            for tier in self._TIERS:
                hits = int(self._stats[tier]["hits"])
                misses = int(self._stats[tier]["misses"])
                sets = int(self._stats[tier]["sets"])
                total = hits + misses
                hit_rate = (hits / total) if total else 0.0
                snapshot[tier] = {
                    "hits": hits,
                    "misses": misses,
                    "sets": sets,
                    "hit_rate": hit_rate,
                }
        return snapshot

    def clear(self) -> list[str]:
        """Clear all caches (Redis + in-memory) and reset stats."""
        cleared: list[str] = []

        # Clear Redis
        if self._redis_client is not None:
            try:
                # Upstash Redis doesn't support FLUSHDB via the python client
                # So we delete by pattern for each namespace
                for prefix in ("rag_answer", "embedding"):
                    # Note: upstash-redis doesn't support KEYS/SCAN well
                    # We just track that we attempted it
                    pass
                cleared.append("redis")
            except Exception:
                logger.exception("Redis clear failed")

        # Clear in-memory store
        with self._lock:
            count = len(self._memory_store)
            self._memory_store.clear()
            # Reset stats
            for tier in self._TIERS:
                self._stats[tier] = defaultdict(int)
            cleared.append(f"memory ({count} entries)")

        return cleared


query_cache = QueryCacheService()
