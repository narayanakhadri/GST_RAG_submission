"""Convert a Markdown draft (headings, paragraphs, bullet/numbered lists, tables, **bold**, *italic*, `code`,
> quotes) to .docx with python-docx, so it can be merged into the thesis document in Word.

    uv run python -m scripts.thesis.md_to_docx docs/thesis_results_draft.md [--out docs/thesis_results_draft.docx]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def add_inline(par, text: str) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("[") and "](" in part:
            par.add_run(part[1:part.index("](")])
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def add_table(doc, lines: list[str]) -> None:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines
            if not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", ln.strip())]
    width = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(width):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, row[j] if j < len(row) else "")
            for run in par.runs:
                run.font.size = Pt(9)
                if i == 0:
                    run.bold = True
    doc.add_paragraph()


def convert(md: str, out: Path) -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    lines = md.splitlines()
    i, para = 0, []

    def flush():
        if para:
            add_inline(doc.add_paragraph(), " ".join(s.strip() for s in para))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush()
        elif s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        elif m := re.match(r"(#{1,4})\s+(.*)", s):
            flush()
            doc.add_heading(re.sub(r"\*\*|\[RESULTS\]", "", m.group(2)).strip(), level=min(len(m.group(1)), 4) - 1 or 0)
        elif m := re.match(r"([-*]|\d+\.)\s+(.*)", s):
            flush()
            text = m.group(2)
            # continuation lines of the same list item are indented
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip() \
                    and not re.match(r"\s*([-*]|\d+\.)\s", lines[i + 1]):
                i += 1
                text += " " + lines[i].strip()
            add_inline(doc.add_paragraph(style="List Number" if m.group(1)[0].isdigit() else "List Bullet"), text)
        elif s.startswith(">"):
            flush()
            par = doc.add_paragraph(style="Quote")
            add_inline(par, s.lstrip("> ").strip())
        else:
            para.append(s)
        i += 1
    flush()
    doc.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.markdown.with_suffix(".docx")
    convert(args.markdown.read_text(encoding="utf-8"), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
