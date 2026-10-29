"""Build blind review packets for the LLM pre-check of the 49 SME priority questions.

The checker (here: Claude in the chat, no API call) must not see retrieval output or model answers, and must not
know which gold sources the correction added. So only the `id` is taken from questions.json; question, reference
answer and gold sources come from the golden set, and the sources shown are the sorted union of the original and
the corrected annotation. Packets go to --out (markdown, a few questions per file).

    uv run python -m scripts.thesis.sme_precheck_packets --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tiktoken

QUESTIONS = Path("dataset/sme_priority_review/questions.json")
GOLDEN = Path("seed/docs/gst/golden_set_with_gold_sources.jsonl")
CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
MAX_TOKENS = 800  # per source; enough to see what a provision covers
INSTRUMENT_RECORDS = 4

enc = tiktoken.get_encoding("cl100k_base")


def cap(text: str, limit: int = MAX_TOKENS) -> str:
    toks = enc.encode(text)
    return text if len(toks) <= limit else enc.decode(toks[:limit]) + "\n[... truncated ...]"


def section2_clauses(raw: str, answer: str) -> str:
    """Only the clauses of a definitions section that the reference answer cites, e.g. 2(30)."""
    cited = set(re.findall(r"(?:section|sec\.?|s\.)\s*2\s*\((\w+)\)", answer, re.I))
    if not cited:
        return cap(raw)
    parts = re.split(r"(?m)^(?=- \(\w+\))", raw)
    keep = [p for p in parts if (m := re.match(r"- \((\w+)\)", p)) and m.group(1) in cited]
    return f"[only the cited clauses {sorted(cited)}]\n" + "\n".join(keep) if keep else cap(raw)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def instrument_excerpt(inst: str, records: list[dict], query: str) -> str:
    """Whole notification/circular cited: show its records that overlap most with question + answer."""
    q = words(query)
    ranked = sorted(records, key=lambda r: -len(q & words(r["raw_text"])))[:INSTRUMENT_RECORDS]
    body = "\n\n".join(f"<{r['id']}>\n{cap(r['raw_text'], 500)}" for r in ranked)
    return f"[whole instrument cited; {len(records)} records, the {len(ranked)} closest to the question shown]\n{body}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--golden", type=Path, default=GOLDEN, help="e.g. the text-fixed golden set")
    ap.add_argument("--rest", action="store_true", help="every golden question except the 49 priority ones")
    args = ap.parse_args()

    priority = [q["id"] for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))]  # nothing else from this file
    gold = {r["id"]: r for r in map(json.loads, args.golden.read_text(encoding="utf-8").splitlines())}
    ids = [i for i in gold if i not in set(priority)] if args.rest else priority
    corpus = {r["id"]: r for r in map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines())}
    by_inst: dict[str, list[dict]] = {}
    for r in corpus.values():
        by_inst.setdefault(r["instrument_id"], []).append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    blocks = []
    for n, qid in enumerate(ids, 1):
        g = gold[qid]
        sources = sorted(
            set(g["gold_primary_sources"] + g["gold_supporting_sources"] + g["primary_required_sources"] + g["supporting_sources"])
        )
        external = sorted(set(g["missing_gold_sources"] + g["external_sources_missing_from_corpus"]))
        query = g["question"] + " " + g["reference_answer"]
        lines = [f"# Q{n:02d} {qid}", "## Question", g["question"], "## Reference answer", g["reference_answer"]]
        lines.append("## Listed sources (sorted; judge each)")
        for s in sources:
            if s in corpus:
                r = corpus[s]
                text = section2_clauses(r["raw_text"], g["reference_answer"]) if s.endswith(".section.2") else cap(r["raw_text"])
                lines.append(f"### {s} — {r.get('provision_title') or ''}\n{text}")
            elif s in by_inst:
                lines.append(f"### {s}\n{instrument_excerpt(s, by_inst[s], query)}")
            else:
                lines.append(f"### {s}\n[not in corpus]")
        if external:
            lines.append("## Listed citations not in the corpus\n" + "\n".join(f"- {e}" for e in external))
        blocks.append("\n\n".join(lines))

    for i in range(0, len(blocks), args.per_file):
        (args.out / f"packet_{i // args.per_file + 1:02d}.md").write_text("\n\n---\n\n".join(blocks[i : i + args.per_file]), encoding="utf-8")
    print(f"{len(blocks)} questions -> {args.out}")


if __name__ == "__main__":
    main()
