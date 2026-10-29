"""T18 — Long-context generation vs RAG: give the model whole chapters instead of 8 retrieved provisions.

Idea. Million-token context windows raise the question "do we still need precise retrieval?". LongRAG (Jiang et
al. 2024) retrieves large units (whole documents) and lets a long-context reader find the answer; NitiBench (EMNLP
2025, Thai tax law) found long-context LLMs *underperform* RAG on legal QA. This script reproduces that comparison
on GST: the reader gets the full law text of the top `--chapters` chapters/instruments (T17's chapter nodes), up
to `--max-tokens`, instead of E17's 8 provisions.

Output: a results file in the eval/run_gst_ragas.py format (answer, contexts, ground truth, retrieved ids), so the
same judge scores it:  uv run python -m eval.rescore_ragas eval/results/<file>.json
Cost: generation free (Gemini flash-lite, 1M context); judging ≈ $0.06 per 35 questions but with very long
contexts the faithfulness judge reads much more — expect ≈ 3–5× the usual judge cost; use --limit first.

    uv run python -m scripts.techniques.t18_long_context run --set dev_hard --chapters 3 --limit 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

from scripts.techniques.common import questions, use_best_stack

CHARS_PER_TOKEN = 4


def long_context(question: str, n_chapters: int, max_tokens: int) -> tuple[list[str], list[str]]:
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph, law_text
    from app.services.vector_store import get_client
    from scripts.techniques.t17_hierarchical_summaries import COLLECTION

    graph = get_graph()
    hits = get_client().query_points(COLLECTION, query=embed_texts([question])[0], limit=n_chapters,
                                     with_payload=["members", "title"]).points
    contexts, ids, used = [], [], 0
    for h in hits:
        for pid in h.payload["members"]:
            text = law_text(graph.records[pid])
            if used + len(text) > max_tokens * CHARS_PER_TOKEN:
                break
            contexts.append(text)
            ids.append(pid)
            used += len(text)
    return contexts, ids


def main() -> None:
    from app.security.system_prompt import build_system_prompt
    from app.services.llm_service import generate

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--set", default="dev_hard")
    ap.add_argument("--chapters", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    rows = []
    qs = questions(args.set)[: args.limit or None]
    for g in qs:
        t0 = time.time()
        contexts, ids = long_context(g["question"], args.chapters, args.max_tokens)
        prompt = "\n\n".join(contexts) + f"\n\nQuestion: {g['question']}"
        try:
            answer = generate(build_system_prompt(grounded=True), prompt)["text"]
        except Exception as exc:  # noqa: BLE001
            answer = ""
            print(f"{g['id']}: generation failed ({type(exc).__name__})")
        rows.append({"id": g["id"], "question": g["question"], "answer": answer, "contexts": contexts,
                     "context_tokens": sum(len(c) for c in contexts) // CHARS_PER_TOKEN,
                     "seconds": round(time.time() - t0, 2), "ground_truth": g.get("reference_answer", ""),
                     "retrieved_ids": ids, "retrieved_sources": [c.split("\n", 1)[0].split(":")[0] for c in contexts]})
        print(f"{g['id']}: {rows[-1]['context_tokens']} tokens, {rows[-1]['seconds']} s")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(f"eval/results/{stamp}_gst_t18_longcontext_c{args.chapters}_{args.set}.json")
    out.write_text(json.dumps({"profile": "t18_long_context", "flags": vars(args), "question_set": args.set,
                               "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}; score with: uv run python -m eval.rescore_ragas {out}")


if __name__ == "__main__":
    main()
