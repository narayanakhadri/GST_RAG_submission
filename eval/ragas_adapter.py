from __future__ import annotations

import json
import os

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings

ALL_METRICS = {
    "faithfulness": faithfulness,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "answer_relevancy": answer_relevancy,
    # T20: agreement with the reference answer (claim-level F1 judged by the LLM + embedding similarity).
    "answer_correctness": answer_correctness,
}


def selected_metrics(names: str | list[str] | None = None) -> list:
    """Metrics by name (comma-separated string or list); default `settings.ragas_metrics`."""
    if names is None:
        names = settings.ragas_metrics
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in names if n not in ALL_METRICS]
    if unknown:
        raise ValueError(f"unknown RAGAS metric(s): {unknown}; choose from {list(ALL_METRICS)}")
    return [ALL_METRICS[n] for n in names]


def _extra_body() -> dict | None:
    body = json.loads(settings.ragas_judge_extra_body) if settings.ragas_judge_extra_body else {}
    # Qwen (DashScope) hybrid models think by default: slower, more output tokens, no better at grading.
    if "dashscope" in settings.ragas_judge_base_url:
        body.setdefault("enable_thinking", False)
    return body or None


def judge_info() -> dict:
    """What to record in a results file so runs scored by different judges are never mixed up."""
    if settings.ragas_judge_cli:
        return {"model": settings.ragas_judge_model or "default", "base_url": f"cli:{settings.ragas_judge_cli}",
                "extra_body": None}
    return {
        "model": settings.ragas_judge_model,
        "base_url": settings.ragas_judge_base_url or "openai",
        "extra_body": _extra_body(),
    }


def _run_config() -> RunConfig:
    # A subscription CLI starts a process per call and queues under its rate limit: allow longer per call.
    timeout = 1200 if settings.ragas_judge_cli else 300
    return RunConfig(max_workers=settings.ragas_judge_max_workers, max_retries=10, max_wait=90, timeout=timeout)


def _get_ragas_llm():
    """The judge LLM: OpenAI by default, or any OpenAI-compatible endpoint (settings.ragas_judge_*)."""
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.ragas_judge_cli:
        from eval.cli_llm import CliChatModel

        return LangchainLLMWrapper(CliChatModel(cli=settings.ragas_judge_cli, model=settings.ragas_judge_model),
                                   _run_config())
    base_url = settings.ragas_judge_base_url or None
    extra_body = _extra_body()
    chat = ChatOpenAI(
        model=settings.ragas_judge_model,
        api_key=settings.ragas_judge_api_key or settings.openai_api_key,
        base_url=base_url,
        temperature=0,
        extra_body=extra_body,
    )
    return LangchainLLMWrapper(chat, _run_config())

def _get_ragas_embeddings():
    """Create Ragas-compatible embeddings using app settings."""
    lc_emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return LangchainEmbeddingsWrapper(lc_emb)

def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "user_input": [r["question"] for r in rows],
            "response": [r["answer"] for r in rows],
            "retrieved_contexts": [r["contexts"] for r in rows],
            "reference": [r["ground_truth"] for r in rows],
        }
    )

def run(rows: list[dict], metrics: str | list[str] | None = None) -> list[dict]:
    if not rows:
        return []

    chosen = selected_metrics(metrics)
    ds = build_dataset(rows)
    result = evaluate(
        ds,
        metrics=chosen,
        llm=_get_ragas_llm(),
        # Only answer_relevancy and answer_correctness embed text; skip the OpenAI client otherwise.
        embeddings=_get_ragas_embeddings() if (answer_relevancy in chosen or answer_correctness in chosen) else None,
        run_config=_run_config(),
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")
