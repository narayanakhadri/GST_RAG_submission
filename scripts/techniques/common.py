"""Shared harness for the technique scripts: score any retrieval function exactly like the thesis ablation does.

A technique only has to supply `retrieve(question) -> list[RetrievedChunk]` (or a list of corpus provision IDs).
`evaluate()` runs it over a question set, scores every question against the corrected gold sources with
eval/gst_retrieval.py (required-law recall, all-required hit, MRR), saves
`eval/results/thesis/techniques/<name>.json`, and prints a paired comparison with the reference run
(`e17_v3_textfix`, the best system) on the same questions.

    from scripts.techniques.common import evaluate, use_best_stack
    use_best_stack()                       # 3-large collection + E17 settings
    evaluate("my_idea", my_retrieve, question_set="dev_hard")

Question sets: dev_hard / test_hard (35 each, eval/results/thesis/question_sets.json) or "all" (146).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from app.models import RetrievedChunk
from eval.gst_retrieval import corpus_keys, gold_sets, label_to_id_map, score_row
from eval.stats import HEADER, format_row, paired

GOLDEN = "seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl"
QUESTION_SETS = Path("eval/results/thesis/question_sets.json")
OUT_DIR = Path("eval/results/thesis/techniques")
REFERENCE = Path("eval/results/thesis/retrieval/e17_v3_textfix.json")

# The best retrieval system (E17): flags for app.services.rag_service._retrieve.
E17_FLAGS = {"search_mode": "hybrid", "enable_rerank": False, "enable_crag": False, "top_k": 8,
             "doc_type_cap": 3, "query_rewrite": "decompose"}


def use_best_stack() -> None:
    """Point retrieval at E17's collection and embedding model (in-process only, .env untouched).
    Refuses to run on a paid provider: .env defaults to OpenAI, and a run without LLM_PROVIDER=gemini both costs money
    and misses every Gemini cache (different retrieval). Set ALLOW_PAID_LLM=1 to run on another provider on purpose."""
    if settings.llm_provider != "gemini" and os.environ.get("ALLOW_PAID_LLM") != "1":
        raise SystemExit(f"LLM provider is {settings.llm_provider!r}, not 'gemini' (paid, and not the cached model). "
                         "Prefix the command with LLM_PROVIDER=gemini, or set ALLOW_PAID_LLM=1 on purpose.")
    settings.qdrant_collection = "gst_thesis_enriched_large"
    settings.embedding_model = "text-embedding-3-large"


def questions(question_set: str = "dev_hard", golden: str = GOLDEN) -> list[dict]:
    goldens = [g for g in load_golden_set(golden) if gold_sets(g, "corrected")[0]]
    if question_set == "all":
        return goldens
    ids = set(json.loads(QUESTION_SETS.read_text(encoding="utf-8"))["sets"][question_set])
    return [g for g in goldens if g["id"] in ids]


def labels_for(ids: list[str], corpus: list[dict]) -> list[str]:
    """Corpus IDs -> the source labels retrieval returns (so ID-returning techniques score the same way)."""
    from app.datasets.gst.document_mapper import source_label

    by_id = {r["id"]: r for r in corpus}
    return [source_label(by_id[i]) for i in ids if i in by_id]


def evaluate(name: str, retrieve: Callable[[str], list], question_set: str = "dev_hard", golden: str = GOLDEN,
             corpus_path: str | None = None, extra: dict | None = None) -> dict:
    corpus = load_corpus(corpus_path or settings.gst_corpus_path)
    keys, label_to_id = corpus_keys(corpus), label_to_id_map(corpus)
    rows, t0 = [], time.perf_counter()
    for i, g in enumerate(questions(question_set, golden), start=1):
        out = retrieve(g["question"])
        sources = [c.source for c in out] if out and isinstance(out[0], RetrievedChunk) else labels_for(out, corpus)
        row = {"id": g["id"], "retrieved_sources": sources}
        score_row(row, g, keys, label_to_id)
        rows.append(row)
        if i % 10 == 0:
            print(f"  {i} questions")
    payload = {"name": name, "question_set": question_set, "golden": golden, "seconds": round(time.perf_counter() - t0, 1),
               "settings": {"collection": settings.qdrant_collection, "embedding_model": settings.embedding_model},
               **(extra or {}), "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    compare(name, rows)
    return payload


def compare(name: str, rows: list[dict]) -> None:
    """Paired comparison with the reference (E17 on the fixed golden text) on the same questions."""
    if not REFERENCE.exists():
        return
    ids = {r["id"] for r in rows}
    ref = [r for r in json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"] if r["id"] in ids]
    print(HEADER)
    for metric in ("required_recall", "all_required_hit", "required_mrr"):
        print(format_row("e17_v3_textfix", name, paired(ref, rows, metric, coverage=None)))


def generate_run(name: str, answer_fn: Callable[[str], dict], question_set: str = "dev_hard",
                 limit: int = 0, ids: set[str] | None = None) -> Path:
    """Run a generation technique over a question set and save it in the eval/run_gst_ragas.py format, so the same
    judge scores it:  uv run python -m eval.rescore_ragas <file>  (and T20 for answer correctness).
    `answer_fn(question)` returns {"answer": str, "chunks": list[RetrievedChunk], ...extra fields}."""
    import datetime

    from app.services.gst_graph import get_graph

    graph = get_graph()
    rows = []
    for g in [g for g in questions(question_set) if ids is None or g["id"] in ids][: limit or None]:
        t0 = time.perf_counter()
        try:
            out = answer_fn(g["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"{g['id']}: failed ({type(exc).__name__}: {exc})")
            continue
        chunks = out.pop("chunks", [])
        rows.append({"id": g["id"], "question": g["question"], "answer": out.pop("answer", ""),
                     "contexts": [c.text for c in chunks], "ground_truth": g.get("reference_answer", ""),
                     "retrieved_sources": [c.source for c in chunks],
                     "retrieved_ids": [i for i in (graph.id_of(c) for c in chunks) if i],
                     "context_tokens": sum(len(c.text) for c in chunks) // 4,
                     "seconds": round(time.perf_counter() - t0, 2), **out})
        print(f"  {g['id']}: {rows[-1]['seconds']} s")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(f"eval/results/{stamp}_gst_{name}_{question_set}.json")
    path.write_text(json.dumps({"profile": name, "question_set": question_set, "rows": rows}, indent=1,
                               ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}\nscore: uv run python -m eval.rescore_ragas {path}")
    return path


def e17_chunks(question: str, context_text: str = "law+summary") -> list[RetrievedChunk]:
    """The final-system retrieval: E17 + the E12b generation context."""
    from app.services.rag_service import _retrieve

    return _retrieve(question, {**E17_FLAGS, "context_text": context_text})


def grounded_answer(question: str, chunks: list[RetrievedChunk], extra_instructions: str = "") -> str:
    """Answer with the grounded system prompt over the given chunks (the final system's generation step).
    Same as rag_service._generate: spotlighted context, and the JSON envelope ({"answer", "sources", ...}) the
    system prompt asks for is unwrapped to the answer text (raw text if validation fails)."""
    from app.security.output_validator import validate_with_retry
    from app.security.spotlighting import build_spotlighted_context
    from app.security.system_prompt import build_system_prompt
    from app.services.llm_service import generate

    system = build_system_prompt(grounded=True) + extra_instructions
    raw = generate(system, f"{build_spotlighted_context(chunks)}\n\nQuestion: {question}")["text"]
    try:
        return validate_with_retry(raw, lambda prompt, _err: generate(system, prompt)["text"]).answer
    except Exception:  # noqa: BLE001
        return raw


def chunks_for_ids(ids: list[str]) -> list[RetrievedChunk]:
    """Corpus IDs -> the chunks retrieval would return (text + label), via the provision graph."""
    from app.services.gst_graph import get_graph

    graph = get_graph()
    return [graph.chunk_for(i) for i in dict.fromkeys(ids) if i in graph.records]
