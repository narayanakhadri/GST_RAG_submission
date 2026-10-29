"""Evidence passage per (question, source): the part of the law that best matches the reference answer.

For every source the SME sees on page 0 (proposed + removed by the pre-check), the full law text (no cap) is split
into passages (paragraphs / list items, footnotes dropped) and ranked by TF-IDF cosine similarity to the question +
reference answer, with a bonus for the clause numbers the answer cites ("section 17(5)(b)" -> "(5)", "(b)").
The best passage is the SME's evidence line; scores also drive the second pass of the pre-check (a source the
pre-check dropped but whose best passage matches the answer well is re-examined).

    uv run python -m scripts.thesis.evidence_quotes      # -> dataset/sme_priority_review/evidence.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
BASE = Path("dataset/sme_priority_review/questions_llm_base.json")
OUT = Path("dataset/sme_priority_review/evidence.json")
MAX_CHARS = 600

FOOTNOTE = re.compile(r"^\s*\d{1,3}\s+(Inserted|Substituted|Omitted|Amended|Refer|Prior|\*)", re.I)


def passages(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-\d]+\.?\s*\()", text):
        block = " ".join(block.split())
        if len(block) < 40 or FOOTNOTE.match(block):
            continue
        while len(block) > MAX_CHARS:  # long paragraphs: windows at sentence ends
            cut = block.rfind("; ", 0, MAX_CHARS)
            cut = cut if cut > 200 else MAX_CHARS
            out.append(block[: cut + 1].strip())
            block = block[cut + 1:].strip()
        if len(block) >= 40:
            out.append(block)
    return out


def cited_clauses(answer: str, sid: str) -> set[str]:
    """Clause markers the answer cites for this provision number, e.g. 'section 17(5)(b)' -> {'(5)', '(b)'}."""
    num = sid.rsplit(".", 1)[-1]
    found = set()
    for m in re.finditer(rf"(?:section|rule|sec\.?|s\.)\s*{re.escape(num)}((?:\s*\(\w+\))+)", answer, re.I):
        found.update(re.findall(r"\(\w+\)", m.group(1)))
    return found


def main() -> None:
    corpus = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["id"]: r for r in corpus}
    by_inst: dict[str, list[dict]] = {}
    for r in corpus:
        by_inst.setdefault(r["instrument_id"], []).append(r)

    def source_passages(sid: str) -> list[tuple[str, str]]:
        """(record id, passage) pairs; a whole instrument contributes the passages of all its records."""
        recs = [by_id[sid]] if sid in by_id else by_inst.get(sid, [])
        return [(r["id"], p) for r in recs for p in passages(r.get("raw_text") or "")]

    base = json.loads(BASE.read_text(encoding="utf-8"))
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    vec.fit([p for r in corpus for p in passages(r.get("raw_text") or "")] + [q["reference_answer"] for q in base])

    out: dict[str, dict] = {}
    for q in base:
        query = q["question"] + "\n" + q["reference_answer"]
        qv = vec.transform([query])
        ev = {}
        for sid in dict.fromkeys(q["required_sources"] + (q.get("llm_removed_sources") or [])):
            cands = source_passages(sid)
            if not cands:
                continue
            sims = linear_kernel(qv, vec.transform([p for _, p in cands]))[0]
            clauses = cited_clauses(q["reference_answer"], sid)
            scored = [(s + 0.05 * sum(c in cands[i][1][:40] for c in clauses), i) for i, s in enumerate(sims)]
            best_score, best = max(scored)
            rid, text = cands[best]
            ev[sid] = {"passage": text, "from": rid, "score": round(float(best_score), 3)}
        out[q["id"]] = ev
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"evidence for {sum(len(v) for v in out.values())} (question, source) pairs -> {OUT}")


if __name__ == "__main__":
    main()
