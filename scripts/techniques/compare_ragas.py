"""Compare two RAGAS result files (same questions): faithfulness, answer relevancy, citation checks, context size.

    uv run python -m scripts.techniques.compare_ragas eval/results/<base>.json eval/results/<variant>.json
"""

from __future__ import annotations

import json
import statistics
import sys

from eval.stats import HEADER, format_row, paired


def main() -> None:
    a_path, b_path = sys.argv[1], sys.argv[2]
    a = json.load(open(a_path, encoding="utf-8"))["rows"]
    b = json.load(open(b_path, encoding="utf-8"))["rows"]
    print(HEADER)
    for m in ("faithfulness", "answer_relevancy", "answer_correctness"):
        r = paired(a, b, m, coverage=None)
        if r.get("n"):
            print(format_row("A", "B", r))
    for name, rows in (("A", a), ("B", b)):
        toks = [r["context_tokens"] for r in rows if r.get("context_tokens")]
        secs = [r["seconds"] for r in rows if r.get("seconds")]
        print(f"{name}: median context tokens {statistics.median(toks) if toks else '—'}, "
              f"median seconds {statistics.median(secs) if secs else '—'}")
    from eval import citation_check

    sys.argv = ["citation_check", a_path, b_path]
    citation_check.main()


if __name__ == "__main__":
    main()
