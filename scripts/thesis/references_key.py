"""Citation key for Mendeley: number every reference, mark every in-text citation, list papers tried but not cited.

    uv run python -m scripts.thesis.references_key                  # rewrite docs/thesis/references_key.md + markers in the draft
    uv run python -m scripts.thesis.references_key --docx docs/thesis/GST_RAG_Thesis_v1.docx   # also mark a Word file

Reads the typed bibliography of docs/thesis/thesis_reference_draft.md (after REFERENCES_TYPED), gives each entry a
marker R01, R02, … in alphabetical order, and writes docs/thesis/references_key.md: marker, Harvard entry, a link to
find it (arXiv / DOI / URL in the entry, otherwise a Google Scholar search on the title) and where it is cited.
Every in-text citation in the draft — "(Es et al., 2024)", "Magesh et al. (2024)", "(Government of India, 2017a;
2017b)" — gets the marker(s) after it, e.g. "(Es et al., 2024) [R12]". In Word: search "[R", insert the citation
from Mendeley, delete the marker. A second table lists the papers behind techniques that were tried
(scripts/thesis/make_thesis_figures.py TECHNIQUE_MAP, docs/techniques_catalog.md) but are not in the bibliography.
Re-running is safe: existing markers are replaced, not duplicated.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import quote_plus

DRAFT = Path("docs/thesis/thesis_reference_draft.md")
OUT = Path("docs/thesis/references_key.md")
MARKER = re.compile(r" \[R\d{2}(?:, R\d{2})*\]")
YEAR = r"\d{4}[a-z]?"
PAREN = re.compile(rf"\(([^()]*?{YEAR})\)")
NARRATIVE = re.compile(rf"([A-Z][\w'’\-]*(?: [A-Z][\w'’\-]+)*)(?:(?:, [A-Z][\w'’\-]+)* and [A-Z][\w'’\-]+| et al\.)? \(({YEAR})\)")

# Papers behind techniques that were tried but are not (yet) in the bibliography: (technique, title, authors, year).
NOT_CITED = [
    ("T17 summary-first retrieval", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval", "Sarthi et al.", "2024"),
    ("T12 second retrieval", "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)", "Trivedi et al.", "2023"),
    ("T12 second retrieval", "Active Retrieval Augmented Generation (FLARE)", "Jiang et al.", "2023"),
    ("T06 compression", "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models", "Jiang et al.", "2023"),
    ("T06 compression", "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation", "Xu et al.", "2024"),
    ("T07 answer structure", "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models", "Yu et al.", "2023"),
    ("T24 drafting", "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting", "Wang et al.", "2024"),
    ("T25 verification", "Chain-of-Verification Reduces Hallucination in Large Language Models", "Dhuliawala et al.", "2023"),
    ("T05 context order", "Lost in the Middle: How Language Models Use Long Contexts", "Liu et al.", "2024"),
    ("E13b doc2query", "Document Expansion by Query Prediction", "Nogueira et al.", "2019"),
    ("T19 late interaction", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", "Santhanam et al.", "2022"),
    ("T34 missing-law check", "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models", "Wang et al.", "2023"),
    ("T34 missing-law check", "Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask)", "Press et al.", "2023"),
    ("T26 sub-questions", "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models", "Zhou et al.", "2023"),
    ("T20 answer correctness", "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment", "Liu et al.", "2023"),
    ("T28 ensemble", "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "Wang et al.", "2023"),
    ("T18 long context", "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs", "Jiang et al.", "2024"),
    ("W1b adaptive top-k", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity", "Jeong et al.", "2024"),
    ("W1d weighted RRF", "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers", "Sawarkar et al.", "2024"),
    ("T31 reasoning re-rank", "ReasonRank: Empowering Passage Ranking with Strong Reasoning Ability", "Liu et al.", "2025"),
    ("T22 PageRank", "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models", "Gutiérrez et al.", "2024"),
    ("T15 open embedders", "Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)", "Wang et al.", "2022"),
    ("T15 open embedders", "Towards General Text Embeddings with Multi-stage Contrastive Learning (GTE)", "Li et al.", "2023"),
    ("T15 open embedders", "C-Pack: Packed Resources For General Chinese Embeddings (BGE)", "Xiao et al.", "2023"),
    ("T16 fine-tuned embedder", "LQ-RAG (legal query RAG with a fine-tuned embedder) — CHECK exact title", "", ""),
    ("T09 / T07b learning-to-rank, obligations", "Gokhan & Briscoe, NLLP workshop 2025 — CHECK exact title", "Gokhan and Briscoe", "2025"),
    ("T29 definitions", "CRAwLeR (cross-reference dependence in legal retrieval) — CHECK exact title", "", "2026"),
    ("T11 citation audit", "EscavAI (citation grounding for legal answers) — CHECK exact title", "", "2026"),
    ("T01 sub-section chunking", "SAC (summary-augmented chunking) — CHECK exact title", "", ""),
    ("W1f learned sparse", "Qdrant BM42 and miniCOIL announcements (Qdrant blog) — CHECK URLs", "Qdrant", "2024–2025"),
]


def scholar(title: str) -> str:
    return "https://scholar.google.com/scholar?q=" + quote_plus(re.sub(r"\s+—.*$", "", title))


def bibliography(text: str) -> list[dict]:
    body = text.split("REFERENCES_TYPED", 1)[1].split("\n# Appendix A", 1)[0]
    entries = []
    for line in body.splitlines():
        m = re.match(rf"(.+?) \(({YEAR})\) (.+)", line.strip())
        if not m or line.startswith(">"):
            continue
        authors, year, rest = m.groups()
        name = authors.split(",")[0].strip() if "," in authors else authors.strip()
        if m2 := re.search(r"arXiv:(\d{4}\.\d{4,5})", rest):
            link = f"https://arxiv.org/abs/{m2.group(1)}"
        elif m2 := re.search(r"(https?://\S+?)\.?$", rest):
            link = m2.group(1)
        elif m2 := re.search(r"doi[: ]\s*(10\.\S+)", rest, re.I):
            link = f"https://doi.org/{m2.group(1).rstrip('.')}"
        else:
            title = re.search(r"'([^']+)'", rest)
            link = scholar(title.group(1) if title else rest.split(".")[0])
        entries.append({"name": name, "year": year, "entry": line.strip(), "link": link, "cited_in": []})
    for i, e in enumerate(entries, 1):
        e["marker"] = f"R{i:02d}"
    return entries


def cite_name(part: str) -> str:
    part = re.sub(r"^(?:e\.g\.|see|cf\.)\s+", "", part.strip())
    return re.split(r" et al\.| and |,", part)[0].strip()


def markers_for(group: str, index: dict, last: str | None = None) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for part in group.split(";"):
        part = part.strip()
        years = re.findall(YEAR, part)
        if not years:
            continue
        name = cite_name(part) if not re.fullmatch(YEAR, part) else last
        if re.fullmatch(YEAR, part) and last is None:
            continue
        last = name
        for y in years:
            key = (name, y)
            (found if key in index else missing).append(index.get(key, f"{name} {y}"))
    return found, missing


def mark_text(text: str, index: dict, section: str = "", seen: dict | None = None) -> tuple[str, list[str]]:
    """Add markers after every citation in one line/paragraph; returns the new text and unmatched citations."""
    text = MARKER.sub("", text)
    missing: list[str] = []

    def paren(m):
        found, miss = markers_for(m.group(1), index)
        missing.extend(miss)
        if seen is not None:
            for mk in found:
                seen.setdefault(mk, set()).add(section)
        return m.group(0) + (f" [{', '.join(dict.fromkeys(found))}]" if found else "")

    def narrative(m):
        # "Press Information Bureau (2026)" matches whole; "the study by Magesh et al. (2024)" ends in a surname
        words = m.group(1).split()
        key = next(((" ".join(words[i:]), m.group(2)) for i in range(len(words))
                    if (" ".join(words[i:]), m.group(2)) in index), None)
        if key is None or m.string[m.end():m.end() + 2] == " [":
            return m.group(0)
        if seen is not None:
            seen.setdefault(index[key], set()).add(section)
        return m.group(0) + f" [{index[key]}]"

    text = PAREN.sub(paren, text)
    text = NARRATIVE.sub(narrative, text)
    return text, [m for m in missing if len(m.split()) <= 5 and m[:1].isupper()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docx", type=Path, action="append", default=[], help="Word file(s) to mark as well")
    args = ap.parse_args()

    text = DRAFT.read_text(encoding="utf-8")
    entries = bibliography(text)
    index = {(e["name"], e["year"]): e["marker"] for e in entries}
    head, tail = text.split("REFERENCES_TYPED", 1)
    out_lines, missing, seen, section = [], [], {}, ""
    for line in head.splitlines():
        if m := re.match(r"#{1,3} (.+)", line):
            section = m.group(1)
        if line.startswith(("FIG ", "RESULTS ", "CODE ", "CAPTION ", "<!--", "#")) or line.startswith("> CHECK"):
            out_lines.append(line)
            continue
        new, miss = mark_text(line, index, section, seen)
        out_lines.append(new)
        missing += miss
    DRAFT.write_text("\n".join(out_lines) + "\nREFERENCES_TYPED" + tail, encoding="utf-8")

    for e in entries:
        e["cited_in"] = sorted(seen.get(e["marker"], []))
    rows = ["# Reference key — markers for Mendeley", "",
            "Generated by `scripts/thesis/references_key.py` from the bibliography of `thesis_reference_draft.md`. In the",
            "Word file, search for `[R`, insert the citation from Mendeley (search by the link or the title), then delete",
            "the marker. Verify every entry before importing it (the draft's CHECK note applies).", "",
            "## Cited in the draft", "", "| Marker | Reference (Harvard) | Find it | Cited in |", "|---|---|---|---|"]
    for e in entries:
        where = "; ".join(e["cited_in"]) or "not cited in the text (remove or cite)"
        rows.append(f"| {e['marker']} | {e['entry'].replace('|', '/')} | {e['link']} | {where} |")
    rows += ["", "## Papers behind techniques tried, not in the bibliography yet", "",
             "Add the ones you discuss in Chapter 2 or Appendix B; titles marked CHECK need confirming.", "",
             "| Technique | Paper | Authors, year | Find it |", "|---|---|---|---|"]
    for tech, title, authors, year in NOT_CITED:
        rows.append(f"| {tech} | {title} | {', '.join(x for x in (authors, year) if x)} | {scholar(title)} |")
    if missing:
        rows += ["", "## In-text citations with no bibliography entry", ""] + [f"- {m}" for m in sorted(set(missing))]
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(entries)} references, {sum(1 for e in entries if e['cited_in'])} cited; "
          f"{len(set(missing))} unmatched citations -> {OUT}")

    for path in args.docx:
        from docx import Document

        doc = Document(str(path))
        changed = 0
        paragraphs = list(doc.paragraphs) + [p for t in doc.tables for r in t.rows for c in r.cells for p in c.paragraphs]
        for p in paragraphs:
            if p.style is not None and p.style.name in ("Bibliography", "Writing Brief"):
                continue
            for run in p.runs:
                new, _ = mark_text(run.text, index)
                if new != run.text:
                    run.text = new
                    changed += 1
        doc.save(str(path))
        print(f"{path}: markers added in {changed} runs")


if __name__ == "__main__":
    main()
