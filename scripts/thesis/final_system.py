"""The final system end to end: E17 retrieval → T31 reasoning re-rank → grounded answer over law + summary (E12b).

Writes a RAGAS-format run (eval/results/<stamp>_gst_final_t31_<set>.json) with the answers; the judge scores it
afterwards, with the same pinned Qwen judge as every other number:

    uv run python -m scripts.thesis.final_system --set all
    uv run python -m eval.rescore_ragas eval/results/<run>.json --metrics faithfulness,answer_relevancy,answer_correctness

Generation is free (Gemini keys); T31 grades are cached from the retrieval run (t31_pointwise_on_all).
"""

from __future__ import annotations

import argparse

from scripts.techniques.common import generate_run, grounded_answer, use_best_stack
from scripts.techniques.t31_llm_pointwise_rerank import retrieve


def answer(question: str) -> dict:
    chunks = retrieve(question, reasoning=True)
    return {"answer": grounded_answer(question, chunks), "chunks": chunks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    use_best_stack()
    generate_run("final_t31", answer, args.set, args.limit)


if __name__ == "__main__":
    main()
