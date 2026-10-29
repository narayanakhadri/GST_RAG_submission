from operator import add
from typing import Annotated, TypedDict

from app.models import CRAGEvaluation, ReflectionResult, RetrievedChunk


class GraphState(TypedDict):
    question: str
    user_id: str
    flags: dict


    hypotheses: list[str]
    retrieved_chunks: Annotated[list[RetrievedChunk], add]
    reranked_chunks: list[RetrievedChunk] | None
    spotlighted_context: str | None
    crag_evaluation: CRAGEvaluation | None
    web_results: list[RetrievedChunk]
    rag_cache_hit: bool

    raw_answer: str | None
    reflection: ReflectionResult | None
    reflection_iterations: int
    refined_question: str | None

    final_answer: str | None
    sources: list[str]
    confidence: float | None
    chunk_previews: list[dict]

    cache_hits: dict[str, bool]
    cost_saved_usd: float
