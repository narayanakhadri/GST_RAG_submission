"""Two proposal promises tested on the final system (E17 → T31 → grounded answer over law + summary).

1. `version` — RQ1/RQ4 "do not mix old law with new updates" (proposal §7.5, T02). The 8 golden questions whose
   law changed after the reference answers' law date (30.04.2025; dataset/sme_priority_review/law_change_verdicts.json)
   are answered "as on 30.04.2025": every retrieved provision amended later gets a version note (the amendment, its
   effective date, and the earlier wording when the footnote quotes it), and the prompt asks for the law on that date.
   Baseline = the final system's normal answers (law as in the corpus, April 2026). Both judged for correctness
   against the references; plus how many answers received a version note at all.
2. `relations` — proposal §7.6 "key graph relationships in the prompt" (T04): the same final system with the graph
   relations between the retrieved provisions ("Rule 36 is made under Section 16", "Notification 12/2017 exempts
   under Section 11") added to the context. On the 79 SME questions, paired with the final system's scores.

    uv run python -m scripts.thesis.final_variants version
    uv run python -m scripts.thesis.final_variants relations
    (then eval.rescore_ragas on the files they print; compare with scripts.techniques.compare_ragas)
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from scripts.techniques.common import generate_run, grounded_answer, use_best_stack
from scripts.techniques.t31_llm_pointwise_rerank import retrieve

VERDICTS = Path("dataset/sme_priority_review/law_change_verdicts.json")
AS_ON = date(2025, 4, 30)
AS_ON_RULE = ("\n\nAnswer under the law as on 30.04.2025. Where a provision carries a version note saying an amendment "
              "was not yet in force on that date, apply the earlier wording and say that the law has since changed.")


def law_change_ids() -> set[str]:
    return {k for k in json.loads(VERDICTS.read_text(encoding="utf-8")) if not k.startswith("_")}


def version_answer(question: str) -> dict:
    from scripts.techniques.t02_version_aware import annotate_chunks

    chunks = retrieve(question, reasoning=True)
    noted = annotate_chunks(chunks, AS_ON)
    n_notes = sum(1 for a, b in zip(chunks, noted, strict=True) if a.text != b.text)
    return {"answer": grounded_answer(question, noted, AS_ON_RULE), "chunks": noted, "version_notes": n_notes}


def relations_answer(question: str) -> dict:
    from app.services.context_variants import apply

    chunks = apply(retrieve(question, reasoning=True), question, ["relations"])
    return {"answer": grounded_answer(question, chunks), "chunks": chunks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["version", "relations"])
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "version":
        generate_run("final_t31_as_on_2025", version_answer, "all", ids=law_change_ids())
    else:
        from scripts.thesis.rag_value_check import sme_ids

        generate_run("final_t31_relations", relations_answer, "all", ids=sme_ids())


if __name__ == "__main__":
    main()
