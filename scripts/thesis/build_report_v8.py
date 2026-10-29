"""Thesis draft V8: the proposal (V7 .docx) with its methodology rewritten to what was built and the evaluation,
results, discussion, limitations, future work and conclusion added. The text comes from docs/report_v8_content.md
(`@@ name` blocks); V7's citations (Mendeley fields), figures, captions and bibliography are kept, because only
paragraphs without fields are rewritten and everything else is inserted around them.

    uv run python -m scripts.thesis.build_report_v8 [--out "docs/A Contextual Question Answering System_V8_draft.docx"]
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

from scripts.thesis.md_to_docx import add_inline

V7 = Path("docs/A Contextual Question Answering SystemV7.docx")
CONTENT = Path("docs/report_v8_content.md")
OUT = Path("docs/A Contextual Question Answering System_V8_draft.docx")
BULLET_NUM_ID = "5"  # V7's bullet list
NEW_ABBREVIATIONS = [
    ("RRF", "Reciprocal Rank Fusion"), ("TF-IDF", "Term Frequency–Inverse Document Frequency"),
    ("HyDE", "Hypothetical Document Embeddings"), ("RAGAS", "Retrieval-Augmented Generation Assessment"),
    ("ICAI", "Institute of Chartered Accountants of India"), ("SME", "Subject-Matter Expert"),
    ("CI", "Confidence Interval"), ("n.s.", "not significant (p ≥ 0.05)"),
]


def blocks(text: str) -> dict[str, list[str]]:
    out, name = {}, None
    for line in text.splitlines():
        if m := re.match(r"@@ (\w+)\s*$", line):
            name = m.group(1)
            out[name] = []
        elif name:
            out[name].append(line)
    return out


class Builder:
    def __init__(self, doc):
        self.doc = doc
        self.body = doc.element.body

    # --- finding and editing V7 -------------------------------------------------------------------------------
    def paragraphs(self):
        return [Paragraph(el, self.doc) for el in self.body.iterchildren() if el.tag == qn("w:p")]

    def find(self, prefix: str, heading: bool = False) -> Paragraph:
        hits = [p for p in self.paragraphs() if p.text.strip().startswith(prefix)
                and (not heading or p.style.name.startswith("Heading"))]
        if len(hits) != 1:
            raise SystemExit(f"anchor {prefix!r}: {len(hits)} matches")
        return hits[0]

    @staticmethod
    def set_text(p: Paragraph, text: str) -> None:
        if p._p.findall(".//" + qn("w:fldSimple")) or p._p.findall(".//" + qn("w:sdt")):
            raise SystemExit(f"refusing to rewrite a paragraph with fields: {p.text[:60]!r}")
        runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)

    def replace_body(self, heading_prefix: str, lines: list[str]) -> None:
        """Delete everything between a heading and the next heading, and put the block there."""
        el = self.find(heading_prefix, heading=True)._p.getnext()
        while not self._is_heading(el):
            nxt = el.getnext()
            self.body.remove(el)
            el = nxt
        self.insert_before(el, lines)

    @staticmethod
    def _is_heading(el) -> bool:
        style = el.find(qn("w:pPr") + "/" + qn("w:pStyle")) if el.tag == qn("w:p") else None
        return style is not None and style.get(qn("w:val")).startswith("Heading")

    # --- inserting Markdown -------------------------------------------------------------------------------------
    def insert_before(self, anchor, lines: list[str]) -> None:
        for el in self.render(lines):
            anchor.addprevious(el)

    def render(self, lines: list[str]) -> list:
        els, para, i = [], [], 0

        def flush():
            if para:
                els.append(self.paragraph(" ".join(s.strip() for s in para)))
                para.clear()

        while i < len(lines):
            s = lines[i].strip()
            if not s:
                flush()
            elif s.startswith("|"):
                flush()
                block = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    block.append(lines[i])
                    i += 1
                els.append(self.table(block))
                els.append(self.paragraph(""))
                continue
            elif m := re.match(r"(#{2,3})\s+(.*)", s):
                flush()
                els.append(self.paragraph(m.group(2), style="Heading 1" if len(m.group(1)) == 2 else "Heading 2"))
            elif s.startswith("CAPTION "):
                flush()
                kind, text = s[len("CAPTION "):].split(":", 1)
                els.append(self.caption(kind.strip(), text.strip()))
            elif s.startswith("FIGURE "):
                flush()
                els.append(self.figure(s[len("FIGURE "):].strip()))
            elif s.startswith("- "):
                flush()
                text = s[2:]
                while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip():
                    i += 1
                    text += " " + lines[i].strip()
                els.append(self.paragraph(text, bullet=True))
            else:
                para.append(s)
            i += 1
        flush()
        return els

    def paragraph(self, text: str, style: str | None = None, bullet: bool = False):
        p = self.doc.add_paragraph(style=style)
        add_inline(p, text)
        if bullet:
            ppr = p._p.get_or_add_pPr()
            num = OxmlElement("w:numPr")
            for tag, val in (("w:ilvl", "0"), ("w:numId", BULLET_NUM_ID)):
                e = OxmlElement(tag)
                e.set(qn("w:val"), val)
                num.append(e)
            ppr.append(num)
        return self._detach(p._p)

    def caption(self, kind: str, text: str):
        """Numbered like V7's captions (SEQ field), so they appear in the lists of figures and tables."""
        p = self.doc.add_paragraph(style="Caption")
        if kind == "Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.paragraph_format.keep_with_next = True
        p.add_run(f"{kind} ")
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), f" SEQ {kind} \\* ARABIC ")
        r = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = "0"
        r.append(t)
        fld.append(r)
        p._p.append(fld)
        p.add_run(f": {text}")
        return self._detach(p._p)

    def figure(self, path: str):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(15))
        return self._detach(p._p)

    def table(self, lines: list[str]):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines
                if not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", ln.strip())]
        width = max(len(r) for r in rows)
        table = self.doc.add_table(rows=len(rows), cols=width)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        # the first column holds the labels: give it the room, share the rest (text width ≈ 16 cm)
        widths = [11.0, 5.0] if width == 2 else [6.0] + [10.0 / (width - 1)] * (width - 1)
        for col, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            col.set(qn("w:w"), str(int(Cm(w).twips)))
        for i, row in enumerate(rows):
            for j in range(width):
                table.cell(i, j).width = Cm(widths[j])
                par = table.cell(i, j).paragraphs[0]
                add_inline(par, row[j] if j < len(row) else "")
                for run in par.runs:
                    run.font.size = Pt(9)
                    if i == 0:
                        run.bold = True
        return self._detach(table._tbl)

    def _detach(self, el):
        self.body.remove(el)
        return el

    # --- V7 extras ----------------------------------------------------------------------------------------------
    def add_abbreviations(self) -> None:
        table = self.doc.tables[0]
        template = table.rows[-1]._tr
        for short, long in NEW_ABBREVIATIONS:
            tr = copy.deepcopy(template)
            template.addnext(tr)
            template = tr
            for cell_el, text in zip(tr.findall(qn("w:tc")), (short, long)):
                ps = cell_el.findall(qn("w:p"))
                self.set_text(Paragraph(ps[0], self.doc), text)

    def update_fields_on_open(self) -> None:
        settings = self.doc.settings.element
        if settings.find(qn("w:updateFields")) is not None:
            return
        e = OxmlElement("w:updateFields")
        e.set(qn("w:val"), "true")
        # the schema fixes the order: updateFields comes before these
        later = {qn(f"w:{t}") for t in ("hdrShapeDefaults", "footnotePr", "endnotePr", "compat", "docVars", "rsids")}
        nxt = next((c for c in settings.iterchildren() if c.tag in later), None)
        if nxt is not None:
            nxt.addprevious(e)
        else:
            settings.append(e)


def build(out: Path) -> None:
    content = blocks(CONTENT.read_text(encoding="utf-8"))
    b = Builder(Document(str(V7)))

    # cover
    b.set_text(b.find("Research Proposal"), "Thesis Draft: Methodology and Results")
    note = copy.deepcopy(b.find("Thesis Draft")._p)
    b.find("Thesis Draft")._p.addnext(note)
    b.set_text(Paragraph(note, b.doc), "Working draft prepared with AI assistance; to be rewritten by the author")
    b.set_text(b.find("December"), "October 2026")

    # abstract: keep V7's problem statement, replace the two paragraphs about the planned work
    first, second = b.find("This project aims"), b.find("The expected outcome")
    anchor = second._p.getnext()
    for p in (first, second):
        b.body.remove(p._p)
    b.insert_before(anchor, content["abstract"])

    b.add_abbreviations()
    b.insert_before(b.find("3. Research Questions", heading=True)._p, content["lit_recent"])
    b.set_text(b.find("To achieve this aim, the study will focus on"), "To achieve this aim, the study focused on:")
    b.insert_before(b.find("7. Research Methodology", heading=True)._p, content["scope_delivered"])

    # methodology: V7's design and Figure 2 stay; each block's body becomes what was built
    tail = b.find("The next sub-sections explain")
    b.insert_before(tail._p, content["m7_intro_tail"])
    b.body.remove(tail._p)
    for prefix, name in (("7.1", "m7_1"), ("7.2", "m7_2"), ("7.3", "m7_3"), ("7.4", "m7_4"), ("7.5", "m7_5"),
                         ("7.6", "m7_6")):
        b.replace_body(prefix, content[name])

    b.set_text(b.find("8. Requirements and Resources", heading=True), "8. Implementation and Resources Used")
    b.replace_body("8. Implementation", content["s8_intro"])
    for prefix, title, name in (("8.1", "8.1 Software Used", "s8_1"), ("8.2", "8.2 Hardware Used", "s8_2"),
                                ("8.3", "8.3 Data Sources Used", "s8_3"), ("8.4", "8.4 Ethical Considerations", "s8_4")):
        b.set_text(b.find(prefix, heading=True), title)
        b.replace_body(prefix, content[name])

    b.set_text(b.find("The project will be completed in six months"),
               "The proposal planned the project over six months in five main stages, shown in the table below; some "
               "stages overlap because the literature review and the writing continued while the system was built.")
    plan = b.find("9. Research Plan", heading=True)
    b.insert_before(plan._p, content["chapters"])
    b.set_text(plan, "14. Research Plan and Actual Timeline")
    reference = b.find("Reference", heading=True)
    b.insert_before(reference._p, content["research_plan_actual"] + content["conclusion"])
    b.insert_before(b.body[-1], content["extra_refs"])  # after the bibliography, before the section properties

    b.update_fields_on_open()
    b.doc.save(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
