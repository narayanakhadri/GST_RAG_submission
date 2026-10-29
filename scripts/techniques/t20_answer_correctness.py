"""T20 — Answer correctness against the ICAI reference answers (the metric the thesis lacked).

Idea. Faithfulness says "the answer sticks to the retrieved text"; answer relevancy says "it addresses the
question". Neither says "it reaches the same conclusion as the expert answer". RAGAS answer_correctness
compares the answer with the reference: the judge splits both into claims and counts true positives / false
positives / false negatives (factual F1, weight 0.75) and adds embedding similarity (weight 0.25). Sources: RAGAS
(Es et al. 2023); NyayaRAG (paper 9: G-Eval vs reference), LawRAG (paper 10: BERTScore vs reference),
AI tax attorneys (paper 14: expert comparison). Caveat: the references state the law as on 30.04.2025
(docs/law_changes_2025_2026.md) — 8 questions may be "wrong" against an up-to-date corpus.

How it works here: a thin wrapper around eval/rescore_ragas.py with the metric added to eval/ragas_adapter.py
(`answer_correctness`), same pinned Qwen judge as every other number. Re-scores saved runs — no new generation.
Cost ≈ $0.04–0.06 per 35 questions (the judge reads answer + reference, not the long context).

    uv run python -m scripts.techniques.t20_answer_correctness eval/results/<run>.json [--limit 5]
    # then: uv run python -m scripts.techniques.compare_ragas <base>.judge-*.json <variant>.judge-*.json
"""

from __future__ import annotations

import sys


def main() -> None:
    from eval import rescore_ragas

    args = [a for a in sys.argv[1:]]
    sys.argv = ["rescore_ragas", *args, "--metrics", "answer_correctness"]
    rescore_ragas.main()


if __name__ == "__main__":
    main()
