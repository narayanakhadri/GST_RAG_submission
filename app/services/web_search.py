import logging

from app.config import settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_TAVILY_QUERY_CHARS = 400

_SHORTEN_SYSTEM_PROMPT = (
    "Rewrite the user's question as a short, concise web search query "
    f"(well under {_MAX_TAVILY_QUERY_CHARS} characters) that captures its core intent. "
    "Drop any fact-pattern narrative, scenario details, or numbers not needed to search. "
    "Return ONLY the search query text — no quotes, no explanation."
)


def _shorten_query(query: str) -> str:
    try:
        from app.services.llm_service import generate, grader_model_default

        result = generate(_SHORTEN_SYSTEM_PROMPT, query, model=grader_model_default(), temperature=0.0)
        shortened = result["text"].strip().strip('"')
        return shortened[:_MAX_TAVILY_QUERY_CHARS] or query[:_MAX_TAVILY_QUERY_CHARS]
    except Exception:
        logger.exception("Query shortening failed; falling back to truncation")
        return query[:_MAX_TAVILY_QUERY_CHARS]


def search_web(query: str, max_results: int = 5) -> list[RetrievedChunk]:

    if not settings.tavily_api_key:
        raise ValueError("Tavily API key not configured")

    if len(query) > _MAX_TAVILY_QUERY_CHARS:
        query = _shorten_query(query)

    try:
        import tavily
        client = tavily.TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        return [
            RetrievedChunk(
                text=result["content"],
                source=result["url"],
                score=result.get("score", 0.0),
            )
            for result in results
        ]
    except Exception:
        logger.exception("Tavily web search failed")
        return []
