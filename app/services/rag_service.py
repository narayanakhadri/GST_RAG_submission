from __future__ import annotations

from loguru import logger

from app.config import settings
from app.models import (
    ChatResponse,
    ResponseMetadata,
    RetrievedChunk,
    RetrievedChunkPreview,
)
from app.security.output_validator import validate_with_retry
from app.security.spotlighting import build_spotlighted_context
from app.security.system_prompt import build_system_prompt
from app.services.citation_lookup import cited_ids
from app.services.crag import crag_pipeline
from app.services.embedding_service import embed_texts
from app.services.gst_graph import (
    cap_doc_types,
    expand_with_graph,
    get_graph,
    get_typed_graph,
    to_law_context,
)
from app.services.hyde import HyDERetriever
from app.services.llm_service import answer_model_default, generate
from app.services.query_cache_service import query_cache
from app.services.query_rewrite import rewrite_queries
from app.services.reranking import Reranker
from app.services.self_reflective import reflect_on_answer, should_regenerate
from app.services.sparse_vector_service import fuse_rrf
from app.services.vector_store import (
    hybrid_search,
    provision_vectors,
    search,
    search_within,
    sparse_search,
)


def _flag(flags: dict | None, key: str, default):
    if not isinstance(flags, dict):
        return default
    return flags.get(key, default)








def _search_once(query: str, mode: str, top_k: int, scorer: str | None,
                 dense_weight: float | None = None) -> list[RetrievedChunk]:
    if mode == "sparse":
        return sparse_search(query, top_k=top_k, scorer=scorer)
    query_embedding = embed_texts([query])[0]
    if mode == "hybrid":
        return hybrid_search(query_embedding, query, top_k=top_k, scorer=scorer, dense_weight=dense_weight)
    return search(query_embedding, top_k=top_k)


def _with_cited_provisions(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """W1a: put the provisions the question names first (a named notification/circular contributes its entry or
    paragraph closest to the question). Already-retrieved copies are moved up, not duplicated."""
    graph = get_graph()
    provision_ids = set(graph.records)
    by_instrument: dict[str, list[str]] = {}
    for pid, r in graph.records.items():
        by_instrument.setdefault(r["instrument_id"], []).append(pid)
    forced: list[str] = []
    for cid in cited_ids(question, provision_ids, set(by_instrument)):
        if cid in provision_ids:
            forced.append(cid)
        else:  # whole instrument: its record most similar to the question
            best = search_within(embed_texts([question])[0], by_instrument[cid], top_k=1)
            forced += best[:1]
    if not forced:
        return chunks
    head = [graph.chunk_for(pid, score=1.0) for pid in dict.fromkeys(forced)]
    labels = {c.source for c in head}
    return head + [c for c in chunks if c.source not in labels]


def _mmr(question: str, chunks: list[RetrievedChunk], lam: float) -> list[RetrievedChunk]:
    """W1e: maximal marginal relevance over the candidate pool. Relevance = the fused score scaled to [0, 1];
    redundancy = cosine similarity to the provisions already picked. Returns the whole pool re-ordered."""
    import math

    graph = get_graph()
    ids = [graph.id_of(c) for c in chunks]
    vectors = provision_vectors([i for i in ids if i])
    top = max((c.score for c in chunks), default=0.0) or 1.0

    def cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)

    remaining, picked, picked_vecs = list(range(len(chunks))), [], []
    while remaining:
        def gain(i: int) -> float:
            v = vectors.get(ids[i]) if ids[i] else None
            redundancy = max((cos(v, p) for p in picked_vecs), default=0.0) if v is not None else 0.0
            return lam * chunks[i].score / top - (1 - lam) * redundancy

        best = max(remaining, key=gain)
        remaining.remove(best)
        picked.append(chunks[best])
        if ids[best] and ids[best] in vectors:
            picked_vecs.append(vectors[ids[best]])
    return picked


def _retrieve(question: str, flags: dict | None = None) -> list[RetrievedChunk]:
    """Retrieval pipeline. Besides the original flags (search_mode, enable_hyde, enable_rerank, enable_crag,
    top_k) it takes experiment flags, all off by default so the default behaviour is unchanged:

    sparse_scorer   "tfidf" | "bm25"            — keyword scorer for sparse/hybrid search
    query_rewrite   None | "rewrite" | "decompose" — LLM search queries (query_rewrite.py), fused with RRF
    graph_expand    False | True | "dense"       — add graph neighbours of the top hits (gst_graph.py);
                                                   "dense" orders them by similarity to the question
    graph_weight    float (0.5)                  — RRF weight of the neighbour list
    graph_links     None | list of relations     — expand over the Codex typed links of these relation types
                                                   (e.g. ["procedure_in", "clarifies"]) instead of the
                                                   cross-reference graph (thesis E13c)
    doc_type_cap    int | None                   — max notification/circular items in the final top_k
    instrument_cap  int | None                   — max items from any one instrument in the final top_k
    reranker_backend / reranker_model            — override the reranker for this request
    Wave 1 (thesis v3):
    citation_lookup bool                         — provisions the question names go first (citation_lookup.py)
    adaptive_top_k  {"2": 6, "3": 7, "4": 12}    — top_k by the number of decomposition sub-queries
    stepback        bool                         — add a step-back query (general principle) to the queries
    rrf_dense_weight float | None                — dense : sparse RRF weight w : 1 − w inside hybrid search
    mmr_lambda      float | None                 — MMR diversity re-ranking of the pool (1 = relevance only)
    """
    final_top_k = int(_flag(flags, "top_k", 5))
    dense_weight = _flag(flags, "rrf_dense_weight", None)
    mmr_lambda = _flag(flags, "mmr_lambda", None)
    mode = _flag(flags, "search_mode", "dense")
    rerank = bool(_flag(flags, "enable_rerank", False))
    hyde = bool(_flag(flags, "enable_hyde", False))
    enable_crag = bool(_flag(flags, "enable_crag", settings.crag_enabled_by_default))
    scorer = _flag(flags, "sparse_scorer", None)
    rewrite_mode = _flag(flags, "query_rewrite", None)
    graph_expand = _flag(flags, "graph_expand", False)
    doc_type_cap = _flag(flags, "doc_type_cap", None)
    instrument_cap = _flag(flags, "instrument_cap", None)

    # Post-processing steps need a candidate pool larger than the final top_k.
    needs_pool = (rerank or graph_expand or doc_type_cap is not None or instrument_cap is not None
                  or mmr_lambda is not None)
    retrieve_k = settings.reranker_initial_top_k if needs_pool else final_top_k

    if hyde:
        chunks = HyDERetriever().retrieve(question, top_k=retrieve_k)
    elif rewrite_mode:
        queries = rewrite_queries(question, rewrite_mode)
        adaptive = _flag(flags, "adaptive_top_k", None)
        if adaptive and rewrite_mode == "decompose":
            final_top_k = int(adaptive.get(str(len(queries)), final_top_k))
        if rewrite_mode == "decompose" or not queries:
            queries = [question, *queries]  # decomposition keeps the original question as one of the queries
        if _flag(flags, "stepback", False):
            queries += rewrite_queries(question, "stepback")
        rankings = [_search_once(q, mode, settings.reranker_initial_top_k, scorer, dense_weight) for q in queries]
        chunks = fuse_rrf(rankings)[:retrieve_k] if len(rankings) > 1 else rankings[0][:retrieve_k]
    else:
        chunks = _search_once(question, mode, retrieve_k, scorer, dense_weight)

    if _flag(flags, "citation_lookup", False):
        chunks = _with_cited_provisions(question, chunks)

    if graph_expand and chunks:
        ranker = None
        if graph_expand == "dense":  # order neighbours by dense similarity to the question
            query_embedding = embed_texts([question])[0]
            ranker = lambda ids: search_within(query_embedding, ids, top_k=len(ids))  # noqa: E731
        links = _flag(flags, "graph_links", None)
        graph = get_typed_graph(tuple(links)) if links else get_graph()
        chunks = expand_with_graph(chunks, graph, weight=float(_flag(flags, "graph_weight", 0.5)),
                                   rank_neighbours=ranker)

    if rerank and chunks:
        reranker = Reranker(_flag(flags, "reranker_backend", None), _flag(flags, "reranker_model", None))
        chunks = reranker.rerank(question, chunks, top_k=len(chunks))

    if mmr_lambda is not None and chunks:
        chunks = _mmr(question, chunks, float(mmr_lambda))

    if doc_type_cap is not None or instrument_cap is not None:
        chunks = cap_doc_types(chunks, get_graph(), final_top_k,
                               None if doc_type_cap is None else int(doc_type_cap),
                               None if instrument_cap is None else int(instrument_cap))
    else:
        chunks = chunks[:final_top_k]

    # Search on the enriched text, but give the LLM (and CRAG, and the RAGAS judge) the law only (thesis E12),
    # or the law plus its one-paragraph summary, Codex-corrected where Codex fixed it (E12b).
    context_text = _flag(flags, "context_text", "embedded")
    if context_text in ("law", "law+summary"):
        chunks = to_law_context(chunks, get_graph(), with_summary=context_text == "law+summary")
    if _flag(flags, "context_variants", None):  # T04–T06: relations block, edge ordering, compression
        from app.services.context_variants import apply as apply_context_variants

        chunks = apply_context_variants(chunks, question, list(_flag(flags, "context_variants", [])))
    if _flag(flags, "as_on_date", None):  # T02: version notes for amendments not yet in force on that date
        from datetime import date

        from scripts.techniques.t02_version_aware import annotate_chunks

        chunks = annotate_chunks(chunks, date.fromisoformat(_flag(flags, "as_on_date", None)))

    # CRAG: grade chunks + fall back to web search if irrelevant
    chunks, evaluation, used_web = crag_pipeline(
        question=question,
        chunks=chunks,
        enable_crag=enable_crag,
    )
    logger.info(
        "CRAG | enabled={} score={} label={} used_web={}",
        enable_crag,
        evaluation.relevance_score,
        evaluation.relevance_label,
        used_web,
    )

    return chunks


def _generate(
    question: str,
    chunks: list[RetrievedChunk],
    flags: dict | None = None,
) -> ChatResponse:
    enable_self_reflective = bool(_flag(flags, "enable_self_reflective", False))

    spotlighted = build_spotlighted_context(chunks)
    system = build_system_prompt(grounded=bool(_flag(flags, "grounded_prompt", False)),
                                 variant=_flag(flags, "prompt_variant", None))

    def _raw(q: str) -> str:
        return generate(system, f"{spotlighted}\n\nQuestion: {q}")["text"]

    def _extract_answer(raw_text: str) -> str:
        # The system prompt instructs the LLM to return a JSON envelope
        # ({"answer": ..., "sources": [...], "confidence": ...}); unwrap it so
        # downstream consumers (reflection, the final response) see clean
        # answer text instead of the raw JSON/markdown-fenced blob.
        try:
            return validate_with_retry(raw_text, lambda prompt, _err: generate(system, prompt)["text"]).answer
        except Exception:
            logger.warning("Output validation failed; falling back to raw LLM text")
            return raw_text

    working_q = question
    raw = _raw(working_q)
    answer_text = _extract_answer(raw)

    # Self-RAG: reflect on the answer; refine the question and retry if weak.
    iterations = 0
    last_score: float | None = None
    final_refined: str | None = None
    if enable_self_reflective:
        while True:
            reflection = reflect_on_answer(
                question=working_q,
                answer=answer_text,
                context=spotlighted,
            )
            last_score = float(reflection.reflection_score)
            if not should_regenerate(reflection, iterations):
                break
            final_refined = reflection.refined_question or working_q
            working_q = final_refined
            raw = _raw(working_q)
            answer_text = _extract_answer(raw)
            iterations += 1

    chunk_previews = [
        RetrievedChunkPreview(text=c.text, source=c.source, score=c.score) for c in chunks
    ]
    return ChatResponse(
        answer=answer_text,
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews,
            reflection_iterations=iterations,
            reflection_score=last_score,
            refined_question=final_refined,
        ),
    )



def run_rag(question: str, flags: dict | int | None = None) -> ChatResponse:
    cache_ctx = (
        _cache_context(flags) if isinstance(flags, dict) else _cache_context(None)
    )
    cached = query_cache.get_rag_answer(question, cache_ctx) if settings.rag_answer_cache_enabled else None
    if cached is not None:
        resp = ChatResponse(**cached)
        resp.cache_hit = True  # set on the schema (added in L8)
        resp.metadata.cache_hit = True
        return resp

    logger.info(
        "L8 query | mode={} rerank={} hyde={} crag={} self_rag={} top_k={}",
        _flag(flags, "search_mode", "dense"),
        _flag(flags, "enable_rerank", False),
        _flag(flags, "enable_hyde", False),
        _flag(flags, "enable_crag", settings.crag_enabled_by_default),
        _flag(flags, "enable_self_reflective", False),
        int(_flag(flags, "top_k", 5)),
    )
    chunks = _retrieve(question, flags=flags if isinstance(flags, dict) else None)
    response = _generate(question, chunks, flags=flags if isinstance(flags, dict) else None)

    if settings.rag_answer_cache_enabled:
        query_cache.set_rag_answer(question, response.model_dump(), cache_ctx)
    return response


# Bump when retrieval/generation logic or the system prompt changes, so cached answers from the old code
# are never served (they would otherwise live for cache_ttl_rag after a restart with a persistent Redis).
ANSWER_CACHE_VERSION = 2


def _cache_context(flags: dict | None) -> dict:
    """Everything that changes the answer: every request flag, plus the collection, embedding model and LLM."""
    extra = {k: v for k, v in (flags or {}).items() if v is not None}
    return {
        **dict(sorted(extra.items())),
        "search_mode": _flag(flags, "search_mode", "dense"),
        "enable_hyde": bool(_flag(flags, "enable_hyde", False)),
        "enable_rerank": bool(_flag(flags, "enable_rerank", False)),
        "enable_crag": bool(_flag(flags, "enable_crag", settings.crag_enabled_by_default)),
        "enable_self_reflective": bool(_flag(flags, "enable_self_reflective", False)),
        "top_k": int(_flag(flags, "top_k", 5)),
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "llm": f"{settings.llm_provider}:{answer_model_default()}",
        "version": ANSWER_CACHE_VERSION,
    }





def run_rag_with_trace(
    question: str, flags: dict | int | None = None
) -> tuple[ChatResponse, list[RetrievedChunk]]:
    chunks = _retrieve(question, flags=flags if isinstance(flags, dict) else None)
    response = _generate(question, chunks, flags=flags if isinstance(flags, dict) else None)
    return response, chunks



run_rag_with_trace_no_cache = run_rag_with_trace
