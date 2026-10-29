"""Golden-set audit by an agent (Codex / Claude via subscription): is each ICAI reference answer right, current and
well-paired, and how does it compare with the final system's answer? Files in, files out, as subscription_judge.py.

    judge/golden_audit/
      ref/<qid>.md      question, ICAI reference answer, gold sources, and the law text (gold + retrieved provisions)
      system/<qid>.md   the final system's answer (opened only after the reference verdict is written)
      out/<qid>.json    the auditor's verdict (judge/GOLDEN_AUDIT_INSTRUCTIONS.md)

    uv run python -m scripts.thesis.golden_audit export [--ids-from FILE]
    uv run python -m scripts.thesis.golden_audit status
    uv run python -m scripts.thesis.golden_audit summary     # -> judge/golden_audit/summary.md + audit.csv for the SME

The golden set is never edited: the summary is a review list for the expert, who has the final say.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

GOLDEN = Path("seed/docs/gst/golden_set_sme_v1.jsonl")
FINAL = Path("eval/results/20261020T040741Z_gst_final_t31_all.json")
CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
ROOT = Path("judge/golden_audit")
MAX_WORDS = 1200  # per provision; longer rules are cut (the auditor is told)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def law_block(pid: str, corpus: dict[str, dict], role: str) -> str:
    r = corpus.get(pid)
    if r is None:
        return f"### {pid} ({role})\n\n(not in the corpus)\n"
    if r["status"] == "omitted":
        return f"### {r['instrument_title']}, {r['provision_type']} {r['provision_number']} ({role})\n\n(omitted / repealed)\n"
    words = r["raw_text"].split()
    text = " ".join(words[:MAX_WORDS]) + (" […truncated]" if len(words) > MAX_WORDS else "")
    title = f"{r['instrument_title']}, {r['provision_type']} {r['provision_number']}"
    if r.get("provision_title"):
        title += f" — {r['provision_title']}"
    return f"### {title} ({role}; id `{pid}`)\n\n{text}\n"


def export(ids_from: Path | None) -> None:
    corpus = {r["id"]: r for r in jsonl(CORPUS)}
    final = {r["id"]: r for r in json.loads(FINAL.read_text(encoding="utf-8"))["rows"]}
    wanted = set(ids_from.read_text(encoding="utf-8").split()) if ids_from else None
    for sub in ("ref", "system", "out"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for g in jsonl(GOLDEN):
        if wanted is not None and g["id"] not in wanted:
            continue
        required = g.get("primary_required_sources") or []
        supporting = g.get("supporting_sources") or []
        missing = g.get("external_sources_missing_from_corpus") or []
        retrieved = (final.get(g["id"]) or {}).get("retrieved_ids") or []
        blocks, seen = [], set()
        for role, ids in (("required", required), ("supporting", supporting), ("retrieved by the system", retrieved)):
            for pid in ids:
                if pid in corpus and pid not in seen:
                    seen.add(pid)
                    blocks.append(law_block(pid, corpus, role))
                elif pid not in corpus and role != "retrieved by the system":
                    entries = [i for i in retrieved if i.startswith(pid + ".")]
                    blocks.append(f"### {pid} ({role}; whole instrument)\n\nThe gold source is the whole notification/"
                                  f"circular. Entries of it in this file: {', '.join(entries) or 'none'}.\n")
        sources = (f"- Required: {', '.join(required) or '(none)'}\n- Supporting: {', '.join(supporting) or '(none)'}\n"
                   f"- Cited by the reference but not in the corpus: {', '.join(map(str, missing)) or '(none)'}\n")
        (ROOT / "ref" / f"{g['id']}.md").write_text(
            f"# {g['id']}\n\nLaw position of the reference answer: {g.get('law_position_as_on') or '30.04.2025'}. "
            f"Corpus law: April 2026.\n\n## Question\n\n{g['question']}\n\n## ICAI reference answer\n\n"
            f"{g['reference_answer']}\n\n## Gold sources (annotation)\n\n{sources}\n## Law text\n\n"
            + "\n".join(blocks), encoding="utf-8")
        row = final.get(g["id"])
        (ROOT / "system" / f"{g['id']}.md").write_text(
            f"# {g['id']} — final system answer\n\nProvisions it was given: {', '.join(retrieved) or '(none)'}\n\n"
            f"## Answer\n\n{row['answer'] if row else '(no answer generated)'}\n", encoding="utf-8")
        n += 1
    print(f"exported {n} questions to {ROOT}")


def verdicts() -> list[dict]:
    out = []
    for f in sorted((ROOT / "out").glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"bad JSON in {f}: {exc}")
    return out


def status() -> None:
    todo = sorted(p.stem for p in (ROOT / "ref").glob("*.md"))
    done = {v.get("qid") for v in verdicts()}
    print(f"{len(done & set(todo))} of {len(todo)} done; next: {[q for q in todo if q not in done][:5]}")


def summary() -> None:
    vs = verdicts()
    if not vs:
        raise SystemExit("no verdicts yet")
    ref = [v.get("reference", {}) for v in vs]
    lines = ["# Golden-set audit (generated by scripts/thesis/golden_audit.py)", "",
             f"{len(vs)} questions audited by {', '.join(sorted({v.get('auditor', '?') for v in vs}))}. "
             "LLM audit — a review list for the expert, not a correction.", ""]
    for field in ("legal_correctness", "current_law_2026", "completeness", "arithmetic", "citations",
                  "question_answer_pairing", "ambiguity"):
        c = Counter(r.get(field, {}).get("rating", "?") for r in ref)
        lines.append(f"- **{field}**: " + ", ".join(f"{k} {n}" for k, n in c.most_common()))
    cmp_ = [v.get("comparison", {}) for v in vs]
    lines += ["", "- **conclusion agreement**: " + ", ".join(f"{k} {n}" for k, n in Counter(
        c.get("conclusion_agreement", "?") for c in cmp_).most_common()),
              "- **who is right (when they differ)**: " + ", ".join(f"{k} {n}" for k, n in Counter(
                  c.get("who_is_right", "?") for c in cmp_ if c.get("conclusion_agreement") != "same").most_common()),
              "- **item verdict**: " + ", ".join(f"{k} {n}" for k, n in Counter(
                  v.get("item_verdict", "?") for v in vs).most_common()), "",
              "## For the expert (item verdict other than keep, or expert check requested)", "",
              "| qid | verdict | confidence | issue |", "|---|---|---|---|"]
    flagged = [v for v in vs if v.get("item_verdict") != "keep" or v.get("needs_expert")]
    for v in sorted(flagged, key=lambda v: (v.get("confidence") != "high", v.get("qid", ""))):
        lines.append(f"| {v.get('qid')} | {v.get('item_verdict')} | {v.get('confidence')} | "
                     f"{(v.get('summary') or '').replace('|', '/')} |")
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "audit.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "item_verdict", "confidence", "needs_expert", "legal_correctness", "current_law_2026",
                    "arithmetic", "pairing", "conclusion_agreement", "who_is_right", "summary"])
        for v in vs:
            r, c = v.get("reference", {}), v.get("comparison", {})
            w.writerow([v.get("qid"), v.get("item_verdict"), v.get("confidence"), v.get("needs_expert"),
                        r.get("legal_correctness", {}).get("rating"), r.get("current_law_2026", {}).get("rating"),
                        r.get("arithmetic", {}).get("rating"), r.get("question_answer_pairing", {}).get("rating"),
                        c.get("conclusion_agreement"), c.get("who_is_right"), v.get("summary")])
    print(f"wrote {ROOT / 'summary.md'} and {ROOT / 'audit.csv'} ({len(flagged)} flagged)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ids-from", type=Path)
    sub.add_parser("status")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.ids_from)
    elif a.cmd == "status":
        status()
    else:
        summary()


if __name__ == "__main__":
    main()
