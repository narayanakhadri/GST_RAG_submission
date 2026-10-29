import threading
import time

from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config import settings


def _get_client() -> OpenAI:
    # Groq's and Gemini's APIs are OpenAI-compatible, so the same SDK works against them —
    # just a different base_url/api_key. Built per-call (cheap, no network
    # cost) rather than as a module-level singleton, so settings.llm_provider
    # can be flipped at runtime (e.g. for A/B eval comparisons).
    if settings.llm_provider == "groq":
        return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    if settings.llm_provider == "gemini":
        return OpenAI(api_key=_gemini_keys()[_gemini_state["index"]], base_url=settings.gemini_base_url, max_retries=0)
    return OpenAI(api_key=settings.openai_api_key)


# --- Gemini key rotation -----------------------------------------------------------------------
# Free-tier Gemini keys have small per-minute/per-day quotas. GEMINI_API_KEY may hold several
# comma-separated keys; on a rate limit (429), overload (5xx) or timeout the call moves on to the next key, and
# only after every key has failed in a row does it back off and go round again.
GEMINI_MAX_ROUNDS = 6
GEMINI_BACKOFF_SECONDS = 20.0
GEMINI_TIMEOUT_SECONDS = 60.0
_gemini_state = {"index": 0}
_gemini_lock = threading.Lock()


def _gemini_keys() -> list[str]:
    keys = [k.strip() for k in settings.gemini_api_key.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return keys


def _create(**kwargs):
    if settings.llm_provider != "gemini":
        return _get_client().chat.completions.create(**kwargs)

    keys = _gemini_keys()
    last_error: Exception | None = None
    for round_number in range(GEMINI_MAX_ROUNDS):
        for _ in range(len(keys)):
            with _gemini_lock:
                index = _gemini_state["index"] % len(keys)
            try:
                client = OpenAI(
                    api_key=keys[index], base_url=settings.gemini_base_url, max_retries=0, timeout=GEMINI_TIMEOUT_SECONDS
                )
                return client.chat.completions.create(**kwargs)
            except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as exc:
                status = getattr(exc, "status_code", None)
                if isinstance(exc, APIStatusError) and not isinstance(exc, RateLimitError) and status < 500:
                    raise  # a real request error (bad model, bad input) — another key won't help
                last_error = exc
                with _gemini_lock:
                    if _gemini_state["index"] % len(keys) == index:
                        _gemini_state["index"] = (index + 1) % len(keys)
                logger.debug("Gemini key #{} failed ({}); rotating", index + 1, status)
        wait = GEMINI_BACKOFF_SECONDS * (round_number + 1)
        logger.warning("All {} Gemini keys rate-limited; waiting {:.0f}s", len(keys), wait)
        time.sleep(wait)
    raise RuntimeError(f"Gemini: all keys still failing after {GEMINI_MAX_ROUNDS} rounds: {last_error}")


def answer_model_default() -> str:
    """The provider-aware default model for generate() — exposed so callers that
    need to name a model explicitly (rather than relying on generate()'s default)
    still respect settings.llm_provider instead of hardcoding an OpenAI model id."""
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    return settings.groq_model_answer if settings.llm_provider == "groq" else settings.llm_model_answer


def grader_model_default() -> str:
    """Same as answer_model_default(), for the cheaper grading/classification model."""
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    return settings.groq_model_grader if settings.llm_provider == "groq" else settings.llm_model_grader


def _usage(response) -> dict:
    return {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }


def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0) -> dict:
    if model is None:
        model = answer_model_default()

    response = _create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return {"text": response.choices[0].message.content or "", "usage": _usage(response)}


def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:

    if model is None:
        model = grader_model_default()

    response = _create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return {"text": response.choices[0].message.content or "", "usage": _usage(response)}
