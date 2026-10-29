import threading
import time
from collections import defaultdict

from upstash_redis import Redis

from app.config import settings

_redis_client: Redis | None = None

# In-process fallback when Upstash isn't configured (same rule as query_cache_service): per-key request timestamps.
_memory_windows: dict[str, list[float]] = defaultdict(list)
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



class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, key: str) -> tuple[bool, int, int]:
        now = time.time()
        window_start = now - self.window_seconds

        if _redis_configured():
            client = get_redis_client()
            pipe = client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds)
            results = pipe.exec()
            request_count: int = results[2]  # type: ignore[assignment]
        else:
            with _memory_lock:
                window = [t for t in _memory_windows[key] if t > window_start]
                window.append(now)
                _memory_windows[key] = window
                request_count = len(window)

        remaining = max(0, self.max_requests - request_count)
        allowed = request_count <= self.max_requests

        return allowed, remaining, request_count


def is_allowed_ip(ip: str, route: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:ip:{ip}:{route}"
    return limiter.is_allowed(key)


def is_allowed_user(
    user_id: str, limit: int = 20, window_seconds: int = 60
) -> tuple[bool, int, int]:
    limiter = RateLimiter(max_requests=limit, window_seconds=window_seconds)
    key = f"rate_limit:user:{user_id}"
    return limiter.is_allowed(key)
