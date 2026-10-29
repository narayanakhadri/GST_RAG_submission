"""Triage the 49 SME priority questions with the LLM pre-check verdicts, and (later) measure LLM-SME agreement.

Input: dataset/sme_priority_review/llm_precheck_verdicts.jsonl, one row per question, written blind by the checker
(Claude, from the packets of scripts/thesis/sme_precheck_packets.py): a verdict per listed source
(R = required, S = supporting, X = not needed), missing law, confidence, notes.

    uv run python -m scripts.thesis.sme_precheck_compare              # triage -> llm_precheck.json + .md
    uv run python -m scripts.thesis.sme_precheck_compare --agreement  # LLM vs SME on the reviewed questions

The triage compares the checker with the *corrected* annotation (primary_required_sources + supporting_sources),
which is what the SME app reviews. The golden set and the SME review copy are only read, never written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DIR = Path("dataset/sme_priority_review")
# The 49 priority questions, then the other 97 (same blind method, checked afterwards).
VERDICT_FILES = [DIR / "llm_precheck_verdicts.jsonl", DIR / "llm_precheck_verdicts_rest.jsonl"]
SME_COPY = DIR / "questions.sme_reviewed.json"
GOLDEN = Path("seed/docs/gst/golden_set_with_gold_sources.jsonl")
CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_verdicts() -> list[dict]:
    return [v for f in VERDICT_FILES if f.exists() for v in load_jsonl(f)]


def corpus_keys() -> set[str]:
    keys = set()
    for r in load_jsonl(CORPUS):
        keys.add(r["id"])
        keys.add(r["instrument_id"])
    return keys


def checker_sets(v: dict, keys: set[str]) -> tuple[dict[str, str], list[str]]:
    """Kept sources with role (listed R/S + missing law in the corpus) and missing law outside the corpus."""
    kept = {s: role for s, role in v["verdicts"].items() if role in ("R", "S")}
    external = []
    for m in v["missing"]:
        if m["source"] in keys:
            kept.setdefault(m["source"], m["role"])
        else:
            external.append(m["source"])
    return kept, external


PRIORITY_IDS = {q["id"] for q in json.loads((DIR / "questions.json").read_text(encoding="utf-8"))}


def triage(v: dict, g: dict, keys: set[str]) -> dict:
    corrected = {s: "R" for s in g["primary_required_sources"]}
    corrected.update({s: "S" for s in g["supporting_sources"] if s not in corrected})
    kept, external = checker_sets(v, keys)

    drop = sorted(s for s in corrected if v["verdicts"].get(s) == "X")
    add = sorted(s for s in kept if s not in corrected)
    role = sorted(s for s in corrected if s in kept and kept[s] != corrected[s])
    data_error = "DATA ERROR" in v["notes"]

    added_by_correction = g.get("golden_source_validation", {}).get("sources_added") or []
    sides = {s: ("corrected" if s in kept else "original") for s in added_by_correction if s in v["verdicts"] or s in kept}

    # Major: a required source would change (corrected R dropped, checker R added) or the record itself is broken.
    major = [s for s in drop if corrected[s] == "R"] + [s for s in add if kept[s] == "R"]
    if major or data_error or v["confidence"] == "low":
        bucket = "look_closely"
    elif drop or add or role or external:
        bucket = "minor"
    else:
        bucket = "confirm"
    return {
        "id": v["id"],
        "bucket": bucket,
        "priority": v["id"] in PRIORITY_IDS,
        "checker_blind": v.get("blind", True),
        "confidence": v["confidence"],
        "data_error": data_error,
        "corrected_annotation": corrected,
        "checker_keeps": kept,
        "checker_drops_from_corrected": drop,
        "checker_adds": add,
        "role_differences": role,
        "missing_outside_corpus": external,
        "correction_additions_checker_sides_with": sides,
        "notes": v["notes"],
    }


def write_triage() -> None:
    keys = corpus_keys()
    gold = {r["id"]: r for r in load_jsonl(GOLDEN)}
    rows = [triage(v, gold[v["id"]], keys) for v in load_verdicts()]
    order = {"look_closely": 0, "minor": 1, "confirm": 2}
    rows.sort(key=lambda r: order[r["bucket"]])

    buckets = Counter(r["bucket"] for r in rows)
    sides = Counter(s for r in rows for s in r["correction_additions_checker_sides_with"].values())
    summary = {
        "questions": len(rows),
        "priority_questions": sum(r["priority"] for r in rows),
        "buckets": dict(buckets),
        "data_errors": sum(r["data_error"] for r in rows),
        "with_drops": sum(bool(r["checker_drops_from_corrected"]) for r in rows),
        "with_adds": sum(bool(r["checker_adds"]) for r in rows),
        "correction_additions_checker_sides_with": dict(sides),
        "method": "Checker = Claude (claude-opus-5-5) in the chat, no API cost. Saw only question, reference answer, the "
        "sorted union of original + corrected gold sources (provenance hidden) and their law text (first ~800 "
        "tokens each; cited clauses only for s.2; for whole notifications the 4 records closest to the "
        "question). For priority questions 31-49 and all 97 other questions the law text was consulted only where needed (question, reference answer and source titles read in full). Never saw retrieval "
        "output or model answers, except question 1 (87619bos-aps2102-p5u1_q003), marked checker_blind=false.",
    }
    (DIR / "llm_precheck.json").write_text(json.dumps({"summary": summary, "questions": rows}, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# LLM pre-check of the 49 SME priority questions", "", "LLM-assisted, SME-verified: the SME has the final say.", ""]
    lines += [f"- {k}: {v}" for k, v in summary.items() if k != "method"] + ["", summary["method"], ""]
    for r in rows:
        lines.append(f"## [{r['bucket']}] {r['id']}" + ("" if r["checker_blind"] else " (checker not blind)"))
        for label, key in [("Drop", "checker_drops_from_corrected"), ("Add", "checker_adds"), ("Role", "role_differences"),
                           ("Outside corpus", "missing_outside_corpus")]:
            if r[key]:
                lines.append(f"- {label}: {', '.join(r[key])}")
        lines += [f"- Notes: {r['notes']}", ""]
    (DIR / "llm_precheck.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def agreement() -> None:
    """Per listed source: does the checker keep/drop it the same way as the SME (sme_final_sources)?"""
    keys = corpus_keys()
    gold = {r["id"]: r for r in load_jsonl(GOLDEN)}
    verdicts = {v["id"]: v for v in load_verdicts()}
    reviewed = [r for r in json.loads(SME_COPY.read_text(encoding="utf-8")) if r.get("sme_status") not in (None, "", "Unreviewed")]
    agree = total = q_exact = 0
    for r in reviewed:
        v, g = verdicts[r["id"]], gold[r["id"]]
        kept, _ = checker_sets(v, keys)
        sme = set(r.get("sme_final_sources") or [])
        listed = set(g["primary_required_sources"]) | set(g["supporting_sources"]) | set(kept) | sme
        same = [(s in kept) == (s in sme) for s in listed]
        agree, total = agree + sum(same), total + len(same)
        q_exact += all(same)
        print(f"{r['id']}: {sum(same)}/{len(same)} sources agree; checker={sorted(kept)} sme={sorted(sme)}")
    if total:
        print(f"\n{len(reviewed)} reviewed questions: source-level agreement {agree}/{total} = {agree / total:.2f}; "
              f"questions with full agreement {q_exact}/{len(reviewed)}")


def edit_rate() -> None:
    """How much the SME changed the LLM proposal (questions_llm_base.sme_reviewed.json): the number to report as
    'gold sources proposed by an LLM, verified by an SME'."""
    path = DIR / "questions_llm_base.sme_reviewed.json"
    if not path.exists():
        print(f"{path} not found - the SME has not opened the LLM base yet.")
        return
    reviewed = [r for r in json.loads(path.read_text(encoding="utf-8")) if r.get("sme_status") not in (None, "", "Unreviewed")]
    unchanged = removed = added = proposed = role_changed = 0
    for r in reviewed:
        ext_changed = sorted(r.get("sme_added_external") or []) != sorted(r.get("llm_external_suggestions") or [])
        rm, ad = r.get("sme_removed_sources") or [], r.get("sme_added_sources") or []
        rc = r.get("sme_role_changes") or {}
        unchanged += not (rm or ad or rc or ext_changed)
        removed, added, proposed = removed + len(rm), added + len(ad), proposed + len(r["required_sources"])
        role_changed += len(rc)
        if rm or ad or rc or ext_changed:
            print(f"{r['id']} [{r['llm_bucket']}]: removed {rm} added {ad} role changes {rc}"
                  + (" external changed" if ext_changed else ""))
    if reviewed:
        print(f"\n{len(reviewed)} reviewed: accepted unchanged {unchanged}/{len(reviewed)} = {unchanged / len(reviewed):.2f}; "
              f"of {proposed} proposed sources the SME removed {removed} ({removed / max(proposed, 1):.2f}), "
              f"changed the role of {role_changed} and added {added}; "
              f"unusable: {sum(r.get('sme_status') == 'Question unusable' for r in reviewed)}; "
              f"bulk-accepted without opening ('confirm' questions): {sum(bool(r.get('sme_bulk_accepted')) for r in reviewed)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", action="store_true", help="LLM vs SME on the original list (questions.sme_reviewed.json)")
    ap.add_argument("--edit-rate", action="store_true", help="how much the SME changed the LLM base")
    args = ap.parse_args()
    (agreement if args.agreement else edit_rate if args.edit_rate else write_triage)()
