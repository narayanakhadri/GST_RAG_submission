"""Build the LLM-proposed base the SME reviews: dataset/sme_priority_review/questions_llm_base.json.

All 146 golden questions: the 49 priority records of questions.json (priority 1-3) and the other 97 (priority 4,
built from the golden set). For each,
- required_sources = the pre-check's proposal (sources it keeps, required first, + missing law that is in the corpus);
  the old list stays in required_sources_before_llm, and what the pre-check dropped in llm_removed_sources;
- question / reference answer come from the text-fixed golden set (scripts/gst_pipeline/fix_golden_text.py); where
  that differs, the old text stays in question_before_llm / reference_answer_before_llm and llm_text_fixes says why;
- llm_bucket / llm_note / llm_confidence / llm_external_suggestions carry the pre-check result;
- records are ordered look_closely -> minor -> confirm, priority questions first within each bucket.

questions.json, the golden set and every *.sme_reviewed.* file are only read.

    uv run python -m scripts.thesis.sme_llm_base
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.thesis.sme_precheck_compare import (
    DIR,
    GOLDEN,
    checker_sets,
    corpus_keys,
    load_jsonl,
    load_verdicts,
    triage,
)

QUESTIONS = DIR / "questions.json"
GOLDEN_TEXTFIX = Path("seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl")
OUT = DIR / "questions_llm_base.json"
OTHER = 4  # priority value for the 97 questions that are not on the priority list


def from_golden(g: dict) -> dict:
    """A record shaped like questions.json for a question that is not on the priority list (no RAG run data)."""
    validation = g.get("golden_source_validation") or {}
    return {
        "id": g["id"], "priority": OTHER, "question": g["question"], "reference_answer": g["reference_answer"],
        "required_sources": list(dict.fromkeys(g["primary_required_sources"] + g["supporting_sources"])),
        "added_by_correction": validation.get("sources_added") or [], "correction_reason": validation.get("reason"),
        "gold_source_notes": g.get("gold_source_notes") or {},
        "retrieved_sources": [], "missed_sources": [], "model_answer": "", "model_abstained": False, "scores": {},
    }


def main() -> None:
    keys = corpus_keys()
    gold = {r["id"]: r for r in load_jsonl(GOLDEN)}
    verdicts = {v["id"]: v for v in load_verdicts()}
    fixed = {r["id"]: r for r in load_jsonl(GOLDEN_TEXTFIX)}
    order = {"look_closely": 0, "minor": 1, "confirm": 2}

    priority = {r["id"]: r for r in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    out = []
    for qid, g in gold.items():
        if qid not in verdicts:
            continue  # not pre-checked
        r = priority.get(qid) or from_golden(g)
        v = verdicts[qid]
        t = triage(v, g, keys)
        kept, external = checker_sets(v, keys)
        proposal = [s for s in kept if kept[s] == "R"] + [s for s in kept if kept[s] == "S"]
        rec = dict(r)
        rec["required_sources_before_llm"] = r["required_sources"]
        rec["required_sources"] = proposal
        rec["missed_sources"] = [s for s in proposal if s not in r["retrieved_sources"]]
        rec["llm_roles"] = kept
        rec["llm_removed_sources"] = [s for s, role in v["verdicts"].items() if role == "X"]
        rec["llm_external_suggestions"] = external
        rec["llm_bucket"] = t["bucket"]
        rec["llm_confidence"] = v["confidence"]
        rec["llm_note"] = v["notes"]
        rec["llm_text_fixes"] = []
        f = fixed[qid]
        for field in ("question", "reference_answer"):
            if f[field].strip() != r[field].strip():
                rec[f"{field}_before_llm"] = r[field]
                rec[field] = f[field]
        if any(f"{field}_before_llm" in rec for field in ("question", "reference_answer")):
            rec["llm_text_fixes"].append(f.get("text_fix_note") or "Text fixed in the golden set.")
        out.append(rec)

    out.sort(key=lambda x: (order[x["llm_bucket"]], x["priority"]))
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} records ({sum(x['priority'] != OTHER for x in out)} priority) -> {OUT}; "
          f"text fixes: {sum(bool(x['llm_text_fixes']) for x in out)}")


if __name__ == "__main__":
    main()
