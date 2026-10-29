"""Apply the SME's page-0 review to a new golden file, and pull the SME's answer verdicts out of the review notes.

    uv run python -m scripts.thesis.apply_sme_review

Reads (never edits) `dataset/sme_priority_review/questions_llm_base.sme_reviewed.json` and the fixed golden text
`golden_set_with_gold_sources.textfix.jsonl`. Writes:

- `seed/docs/gst/golden_set_sme_v1.jsonl` — the textfix golden set; for every question the SME reviewed, the corrected
  annotation (`primary_required_sources` / `supporting_sources` / `external_sources_missing_from_corpus`) is replaced
  by the SME's final list (role R -> required, S -> supporting). Unreviewed questions keep the corrected annotation.
  The original annotation (`gold_primary_sources` ...) is left as it is. Each changed record gets an `sme_review` block.
- `dataset/sme_answer_verdicts_from_notes.json` — page 0 showed the final system's answer (collapsed, from
  `final_system_review/all_answers.json`); page 0 has one free-text note, so an answer verdict is taken only from
  notes that say the answer is wrong. Other notes ("No changes required") are about the sources: no answer verdict.

The older copy (`questions.sme_reviewed.json`, 4 questions, 2026-10-19) is superseded by the LLM-base review where the
two overlap (decision 2026-10-21: p5u1_q003 keeps s.8, as in the newer review).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

REVIEW = Path("dataset/sme_priority_review/questions_llm_base.sme_reviewed.json")
ANSWERS = Path("dataset/final_system_review/all_answers.json")
GOLDEN_IN = Path("seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl")
GOLDEN_OUT = Path("seed/docs/gst/golden_set_sme_v1.jsonl")
VERDICTS_OUT = Path("dataset/sme_answer_verdicts_from_notes.json")

REVIEWED = ("Sources changed", "Sources correct")
# A question is "numerical" when the answer is a computation: tax liability, ITC, value of supply, interest ...
NUMERICAL = re.compile(r"comput|calculat|working note|₹|\brs\.?\s*\d|\bper cent\b|%", re.I)


def answer_verdict(note: str) -> str | None:
    n = (note or "").lower()
    if "except for" in n and "correct" in n:
        return "partly incorrect"
    if "incorrect" in n or "not applied correctly" in n or "not concluded" in n:
        return "incorrect"
    return None


def is_numerical(g: dict) -> bool:
    return bool(g.get("contains_table")) or bool(NUMERICAL.search(g.get("reference_answer") or ""))


def main() -> None:
    review = {r["id"]: r for r in json.loads(REVIEW.read_text(encoding="utf-8"))
              if r.get("sme_status") in REVIEWED}
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    goldens = [json.loads(line) for line in GOLDEN_IN.read_text(encoding="utf-8").splitlines() if line.strip()]

    changed = 0
    for g in goldens:
        r = review.get(g["id"])
        if not r:
            continue
        roles = r.get("sme_final_roles") or {s: (r.get("llm_roles") or {}).get(s, "S") for s in r["required_sources"]}
        before = (g.get("primary_required_sources"), g.get("supporting_sources"))
        g["primary_required_sources"] = [s for s, v in roles.items() if v == "R"]
        g["supporting_sources"] = [s for s, v in roles.items() if v == "S"]
        external = list(g.get("external_sources_missing_from_corpus") or [])
        g["external_sources_missing_from_corpus"] = external + [e for e in r.get("sme_added_external") or []
                                                                if e not in external]
        g["sme_review"] = {"reviewer": r.get("sme_reviewer"), "reviewed_at": r.get("sme_reviewed_at"),
                           "status": r["sme_status"], "notes": r.get("sme_notes"),
                           "corrected_before_sme": {"primary_required_sources": before[0],
                                                    "supporting_sources": before[1]}}
        changed += before != (g["primary_required_sources"], g["supporting_sources"])
    GOLDEN_OUT.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in goldens), encoding="utf-8")
    print(f"{GOLDEN_OUT}: {len(goldens)} records, {len(review)} SME-reviewed, "
          f"required/supporting lists differ from the corrected annotation on {changed}")

    by_id = {g["id"]: g for g in goldens}
    rows = []
    for rid, r in review.items():
        a = answers.get(rid) or {}
        scores = a.get("scores") or {}
        rows.append({"id": rid, "priority": r.get("priority"), "numerical": is_numerical(by_id[rid]),
                     "sme_answer_verdict": answer_verdict(r.get("sme_notes")), "sme_notes": r.get("sme_notes"),
                     "sme_reviewed_at": r.get("sme_reviewed_at"), "question": r["question"],
                     "reference_answer": r["reference_answer"], "final_answer": a.get("answer"),
                     "qwen_scores": scores})
    VERDICTS_OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{VERDICTS_OUT}: {len(rows)} questions")

    def corr(rs):
        v = [x["qwen_scores"].get("answer_correctness") for x in rs]
        v = [x for x in v if isinstance(x, float) and x == x]
        return f"{mean(v):.3f} (n={len(v)})" if v else "n/a"

    print("\nSME answer verdicts on the final system (24 reviewed questions)")
    for label, sel in [("incorrect / partly", lambda x: x["sme_answer_verdict"]),
                       ("no answer verdict", lambda x: not x["sme_answer_verdict"])]:
        rs = [x for x in rows if sel(x)]
        print(f"  {label:20} {len(rs):2}  numerical {sum(x['numerical'] for x in rs):2}  Qwen correctness {corr(rs)}")
    for num in (True, False):
        rs = [x for x in rows if x["numerical"] == num]
        bad = sum(bool(x["sme_answer_verdict"]) for x in rs)
        print(f"  {'numerical' if num else 'non-numerical':20} {len(rs):2}  SME says wrong {bad}")

    print("\nAll 146 final-system answers, Qwen correctness by question type")
    for num in (True, False):
        rs = [{"qwen_scores": (answers.get(g["id"]) or {}).get("scores") or {}} for g in goldens
              if is_numerical(g) == num]
        print(f"  {'numerical' if num else 'non-numerical':20} {len(rs):3}  {corr(rs)}")


if __name__ == "__main__":
    main()
