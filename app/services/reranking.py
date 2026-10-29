import logging
from typing import cast

from app.config import settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Loading a cross-encoder takes seconds; keep one per model name for the life of the process.
_LOCAL_MODELS: dict[str, object] = {}
LLM_PASSAGE_CHARS = 1500

_LLM_RERANK_PROMPT = (
    "You rank passages of Indian GST law (Acts, Rules, notifications, circulars) by how much each is needed "
    "to answer the question. Consider the legal issue, not word overlap. Return JSON: "
    '{"ranking": [passage numbers, most useful first]} listing every passage number once.'
)


class Reranker:
    def __init__(self, backend: str | None = None, model: str | None = None) -> None:
        self.backend = backend or settings.reranker_backend
        self.model_name = model or settings.reranker_model
        self._voyage_client: object | None = None

    def _load_local_model(self) -> object:
        if self.model_name not in _LOCAL_MODELS:
            from sentence_transformers import CrossEncoder

            _LOCAL_MODELS[self.model_name] = CrossEncoder(self.model_name)
        return _LOCAL_MODELS[self.model_name]

    def _load_voyage_client(self) -> object:
        if self._voyage_client is None:
            import voyageai

            if not settings.voyage_api_key:
                raise ValueError("Voyage API key is required for voyage reranker backend")
            self._voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        return self._voyage_client


    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        top_k = top_k or settings.reranker_initial_top_k
        top_k = min(top_k, len(chunks))

        try:
            if self.backend == "voyage":
                return self._rerank_voyage(query, chunks, top_k)
            if self.backend == "llm":
                return self._rerank_llm(query, chunks, top_k)
            return self._rerank_local(query, chunks, top_k)
        except Exception:
            logger.exception("Reranking failed, returning original order")
            return chunks[:top_k]


    def _rerank_local(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        model = self._load_local_model()
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = cast("list[float]", model.predict(pairs))

        scored = [
            RetrievedChunk(text=chunk.text, source=chunk.source, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


    def _rerank_voyage(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        client = self._load_voyage_client()
        documents = [chunk.text for chunk in chunks]
        result = client.rerank(
            query=query,
            documents=documents,
            model=settings.voyage_model,
            top_k=top_k,
        )

        # Map results back to RetrievedChunk
        reranked: list[RetrievedChunk] = []
        for item in result.results:
            idx = item.index
            chunk = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    score=float(item.relevance_score),
                )
            )
        return reranked


    def _rerank_llm(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Listwise reranking: one LLM call sees all candidates and returns their order (RankGPT-style)."""
        import json

        from app.services.llm_service import generate_with_json

        passages = "\n\n".join(
            f"[{i}] {chunk.source}\n{chunk.text[:LLM_PASSAGE_CHARS]}" for i, chunk in enumerate(chunks, start=1)
        )
        raw = generate_with_json(_LLM_RERANK_PROMPT, f"Question:\n{query}\n\nPassages:\n{passages}")["text"]
        order: list[int] = []
        for n in json.loads(raw).get("ranking") or []:
            if isinstance(n, int) and 1 <= n <= len(chunks) and n - 1 not in order:
                order.append(n - 1)
        order += [i for i in range(len(chunks)) if i not in order]  # anything the LLM left out keeps its place
        return [
            RetrievedChunk(text=chunks[i].text, source=chunks[i].source, score=float(len(order) - rank))
            for rank, i in enumerate(order[:top_k])
        ]

