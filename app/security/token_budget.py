"""Per-user daily token budget tracked in Redis."""

import datetime
import threading

from upstash_redis import Redis

from app.config import settings

_redis_client: Redis | None = None

# In-process fallback when Upstash isn't configured (same rule as query_cache_service). Keys include the date,
# so yesterday's counters are simply never read again.
_memory_used: dict[str, int] = {}
_memory_lock = threading.Lock()


def _redis_configured() -> bool:
    return bool(settings.upstash_redis_url and settings.upstash_redis_token)


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            url=settings.upstash_redis_url,
            token=settings.upstash_redis_token,
        )
    return _redis_client


class TokenBudget:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def _key(self, user_id: str) -> str:
        today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        return f"token_budget:{user_id}:{today}"

    def check_budget(self, user_id: str, estimated_tokens: int) -> tuple[bool, int]:
        key = self._key(user_id)
        if _redis_configured():
            used_str = get_redis_client().get(key)
            used = int(used_str) if used_str is not None else 0
        else:
            with _memory_lock:
                used = _memory_used.get(key, 0)
        remaining = self.max_tokens - used
        ok = estimated_tokens <= remaining
        return ok, remaining

    def consume(self, user_id: str, actual_tokens: int) -> dict:
        key = self._key(user_id)
        if not _redis_configured():
            with _memory_lock:
                used = _memory_used[key] = _memory_used.get(key, 0) + actual_tokens
            return {
                "used": used,
                "limit": self.max_tokens,
                "remaining": max(0, self.max_tokens - used),
                "tokens_charged": actual_tokens,
            }

        client = get_redis_client()
        used = client.incrby(key, actual_tokens)

        # Set TTL to seconds-until-midnight on first write
        ttl = client.ttl(key)
        if ttl == -1:
            now = datetime.datetime.now(datetime.UTC)
            midnight = (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            seconds_until_midnight = int((midnight - now).total_seconds())
            client.expire(key, seconds_until_midnight)

        remaining = max(0, self.max_tokens - used)
        return {
            "used": used,
            "limit": self.max_tokens,
            "remaining": remaining,
            "tokens_charged": actual_tokens,
        }


_budget = TokenBudget(max_tokens=settings.max_tokens_per_user_daily)


def check_budget(user_id: str, estimated_tokens: int) -> tuple[bool, int]:
    return _budget.check_budget(user_id, estimated_tokens)


def consume_budget(user_id: str, actual_tokens: int) -> dict:
    return _budget.consume(user_id, actual_tokens)
