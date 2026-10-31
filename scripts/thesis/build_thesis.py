"""Thesis skeleton in the chapter format of the LJMU model thesis: front matter, Chapters 1–6, references, appendices.

Each section carries a grey "Writing Brief" (what to write, with the numbers and their sources) instead of prose, so
the author writes the text; figures, result tables, code listings and the proposal (Appendix A) are filled in. The
structure and the briefs come from docs/thesis/thesis_outline.md (directives are listed at its top); V7 is the
template, so its styles, cover, Mendeley citations, bibliography and lists of figures/tables are kept.

    uv run python -m scripts.thesis.make_thesis_figures      # first
    uv run python -m scripts.thesis.build_thesis [--out docs/thesis/GST_RAG_Thesis_skeleton.docx]
    uv run python -m scripts.thesis.build_thesis --outline docs/thesis/thesis_reference_draft.md
        --out docs/thesis/GST_RAG_Thesis_reference_draft.docx --cover-note "REFERENCE DRAFT (AI-assisted) - not for submission"
        (one command line)

After writing, the fields (TOC, lists of figures and tables, figure/table numbers) are updated in Word itself via
scripts/thesis/update_word_fields.py (Windows, Word installed, the file closed); --no-fields skips that step.

Text may refer to figures and tables as {fig:<id>} / {tab:<id>}; they are replaced by "Figure c.n" / "Table c.n" (numbered
within each chapter or appendix; figures by their id in figures.json, tables by `id=<id>` on a RESULTS line or
`CAPTION Table <id>: ...`).

Layout follows the upGrad/LJMU templates (docs/thesis/instructions/THESIS_REQUIREMENTS_SUMMARY.md): A4, margins
3/2/2.5/2.5 cm (from V7), Times New Roman 12 everywhere, body 1.5 lines justified, headings 12 pt bold, chapter
headings centred in capitals on two lines ("CHAPTER 2" / "LITERATURE REVIEW"), section numbers followed by a tab,
captions "Table 2.1 Title", the cover as in the template, front matter in the order Declaration, TOC, Dedication,
Acknowledgements, Abstract, List of Tables, List of Figures, List of Abbreviations.

To remove every brief in Word: click in one, Home → Select → Select All Text With Similar Formatting, Delete.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph

from scripts.thesis.build_report_v8 import V7, Builder

OUTLINE = Path("docs/thesis/thesis_outline.md")
FIGURES = Path("eval/results/thesis/figures/thesis/figures.json")
RESULTS = Path("eval/results/thesis/results_tables.md")
DATA_TABLES = Path("eval/results/thesis/figures/thesis/data_tables.md")
OUT = Path("docs/thesis/GST_RAG_Thesis_skeleton.docx")
TEXT_WIDTH_CM = 16.0
BRIEF = "Writing Brief"
PROPOSAL_HEADING = "Proposal Heading"
EXTRA_ABBREVIATIONS = [
    ("RRF", "Reciprocal Rank Fusion"), ("TF-IDF", "Term Frequency–Inverse Document Frequency"),
    ("MRR", "Mean Reciprocal Rank"), ("HyDE", "Hypothetical Document Embeddings"),
    ("RAGAS", "Retrieval-Augmented Generation Assessment"), ("ICAI", "Institute of Chartered Accountants of India"),
    ("SME", "Subject-Matter Expert"), ("CI", "Confidence Interval"), ("ITC", "Input Tax Credit"),
    ("PPR", "Personalized PageRank"), ("n.s.", "not significant (p ≥ 0.05)"),
]
_ids = itertools.count(900_000)


def sig_stars(text: str) -> str:
    """'+0.044*' marks significance; a bare '*' would start Markdown italics in add_inline, so use '∗' instead."""
    return re.sub(r"(?<=[\d)])\*|(?<=\s)\*(?=[\s;,.)|]|$)", "∗", text)


def shade(ppr, fill: str) -> None:
    """Grey paragraph background; w:shd has a fixed place among the pPr children."""
    shd = OxmlElement("w:shd")
    for k, v in (("w:val", "clear"), ("w:color", "auto"), ("w:fill", fill)):
        shd.set(qn(k), v)
    later = {qn(f"w:{t}") for t in ("tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct",
                                     "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing",
                                     "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
                                     "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
                                     "sectPr", "pPrChange")}
    nxt = next((c for c in ppr if c.tag in later), None)
    nxt.addprevious(shd) if nxt is not None else ppr.append(shd)


def plain_font(style) -> None:
    """Times New Roman in black: theme fonts and theme colours would override the explicit values."""
    style.font.name = "Times New Roman"
    fonts = style.element.get_or_add_rPr().find(qn("w:rFonts"))
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{attr}"), None)
    for attr in ("eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), "Times New Roman")
    style.font.color.rgb = RGBColor(0, 0, 0)
    color = style.element.rPr.find(qn("w:color"))
    for attr in ("themeColor", "themeShade", "themeTint"):
        color.attrib.pop(qn(f"w:{attr}"), None)


def heading_text(text: str, level: int) -> str:
    """'Chapter 2: Literature Review' → 'CHAPTER 2' / 'LITERATURE REVIEW' (two lines); other level-1 headings in
    capitals; '2.1 Title' → '2.1<tab>Title'."""
    if level == 1:
        if m := re.match(r"Chapter (\d+):\s*(.*)", text):
            return f"CHAPTER {m.group(1)}\n{m.group(2).upper()}"
        return text.upper()
    return re.sub(r"^(\d+(?:\.\d+)+)\s+", "\\1\t", text)


def chapter_label(text: str) -> str | None:
    """The prefix of figure and table numbers under a level-1 heading: '3' for Chapter 3, 'B' for Appendix B."""
    m = re.match(r"(?:Chapter (\d+)|Appendix ([A-Z]))\b", text)
    return (m.group(1) or m.group(2)) if m else None


class ThesisBuilder(Builder):
    def __init__(self, doc, numbers: dict[str, str] | None = None):
        super().__init__(doc)
        self.numbers = numbers or {}
        self.chapter, self._count = "", {}
        self.figures = {f["id"]: f for f in json.loads(FIGURES.read_text(encoding="utf-8"))}
        self.results = RESULTS.read_text(encoding="utf-8").splitlines() + DATA_TABLES.read_text(encoding="utf-8").splitlines()
        self.styles()

    # --- styles -------------------------------------------------------------------------------------------------
    def styles(self) -> None:
        names = {s.name for s in self.doc.styles}
        if BRIEF not in names:
            st = self.doc.styles.add_style(BRIEF, WD_STYLE_TYPE.PARAGRAPH)
            st.base_style = self.doc.styles["Normal"]
            st.font.size = Pt(9.5)
            st.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
            st.paragraph_format.space_after = Pt(2)
            shade(st.element.get_or_add_pPr(), "E7E6E6")
        if PROPOSAL_HEADING not in names:
            st = self.doc.styles.add_style(PROPOSAL_HEADING, WD_STYLE_TYPE.PARAGRAPH)
            st.base_style = self.doc.styles["Normal"]
            st.font.bold = True
            st.font.size = Pt(12)
            st.paragraph_format.keep_with_next = True
            st.paragraph_format.space_before = Pt(12)

    def body_format(self) -> None:
        normal = self.doc.styles["Normal"]
        normal.font.name = "Times New Roman"
        normal.font.size = Pt(12)
        normal.paragraph_format.line_spacing = 1.5
        normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        normal.paragraph_format.space_after = Pt(6)
        # headings and captions: Times New Roman 12, black (V7's are theme fonts, 14–20 pt, blue)
        for name, align, before, after in (("Heading 1", WD_ALIGN_PARAGRAPH.CENTER, 0, 18),
                                           ("TOC Heading", WD_ALIGN_PARAGRAPH.CENTER, 0, 18),
                                           ("Heading 2", WD_ALIGN_PARAGRAPH.LEFT, 18, 0),
                                           ("Heading 3", WD_ALIGN_PARAGRAPH.LEFT, 18, 0),
                                           ("Caption", None, 0, 6)):
            st = self.doc.styles[name]
            plain_font(st)
            st.font.size = Pt(12)
            st.font.bold = name != "Caption"
            st.font.italic = False
            pf = st.paragraph_format
            pf.line_spacing = 1.5
            pf.space_before, pf.space_after = Pt(before), Pt(after)
            if align is not None:
                pf.alignment = align
            if name in ("Heading 2", "Heading 3"):
                pf.tab_stops.add_tab_stop(Cm(1.27))
                pf.left_indent = pf.first_line_indent = Cm(0)

    @staticmethod
    def compact(el) -> None:
        """Tables and code: single spacing, left-aligned (the body style is 1.5 lines, justified)."""
        for p in el.iter(qn("w:p")):
            pf = Paragraph(p, None).paragraph_format
            pf.line_spacing = 1.0
            pf.space_after = Pt(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def refs(self, text: str) -> str:
        def number(m):
            key = f"{m.group(1)}:{m.group(2)}"
            if key not in self.numbers:
                raise SystemExit(f"unknown reference {{{key}}}")
            return f"{'Figure' if m.group(1) == 'fig' else 'Table'} {self.numbers[key]}"
        return re.sub(r"\{(fig|tab):([\w.-]+)\}", number, text)

    def paragraph(self, text: str, style: str | None = None, bullet: bool = False):
        return super().paragraph(self.refs(text), style=style, bullet=bullet)

    def caption(self, kind: str, text: str):
        """'Table 2.1 Title': the chapter number as text, then a SEQ field that restarts at every level-1 heading
        (\\s 1), so the lists of figures and tables pick the captions up; the cached number is filled in."""
        self._count[kind] = self._count.get(kind, 0) + 1
        p = self.doc.add_paragraph(style="Caption")
        if kind == "Figure":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.paragraph_format.keep_with_next = True
        p.add_run(f"{kind} {self.chapter}.").bold = True
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), f" SEQ {kind} \\* ARABIC \\s 1 ")
        r = OxmlElement("w:r")
        rpr = OxmlElement("w:rPr")
        rpr.append(OxmlElement("w:b"))
        r.append(rpr)
        t = OxmlElement("w:t")
        t.text = str(self._count[kind])
        r.append(t)
        fld.append(r)
        p._p.append(fld)
        p.add_run(f" {text}")
        return self._detach(p._p)

    # --- pieces -------------------------------------------------------------------------------------------------
    def heading(self, text: str, level: int):
        if level == 1:
            self.chapter, self._count = chapter_label(text) or "", {}
        el = self.paragraph(heading_text(text, level), style=f"Heading {level}")
        if level == 1:
            Paragraph(el, self.doc).paragraph_format.page_break_before = True
        return el

    def brief(self, text: str, bullet: bool):
        return self.paragraph(sig_stars(text), style=BRIEF, bullet=bullet)

    def figure_with_caption(self, fig_id: str) -> list:
        f = self.figures.get(fig_id)
        if f is None:
            return [self.brief(f"[missing figure {fig_id}: run make_thesis_figures]", False)]
        return [self.figure(f["file"]), self.caption("Figure", f["caption"])]

    def results_table(self, spec: str) -> list:
        drop = None
        spec = re.sub(r"\s+id=[\w.-]+", "", spec)
        if m := re.search(r"\s+drop=(.+)$", spec):
            drop, spec = m.group(1).strip(), spec[: m.start()]
        start = next((i for i, ln in enumerate(self.results) if re.fullmatch(r"#{2,3} " + re.escape(spec), ln)), None)
        if start is None:
            raise SystemExit(f"results table not found: {spec!r}")
        lines = list(itertools.takewhile(lambda ln: ln.startswith("|"),
                                         itertools.dropwhile(lambda ln: not ln.startswith("|"), self.results[start + 1:])))
        if drop:
            head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
            j = head.index(drop)
            lines = ["|" + "|".join(c for k, c in enumerate(ln.strip().strip("|").split("|")) if k != j) + "|"
                     for ln in lines]
        title = re.sub(r"^RQ\d — ", "", spec)
        return [self.caption("Table", title[0].upper() + title[1:]), self.fitted_table(lines)]

    def fitted_table(self, lines: list[str]):
        """Builder.table, with column widths in proportion to their longest cell (capped), filling the text width."""
        lines = [sig_stars(ln) for ln in lines]
        tbl = self.table(lines)
        self.compact(tbl)
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if not set(ln) <= set("|-: ")]
        n = max(len(r) for r in rows)
        # room for the longest cell (capped: long text wraps) and at least the longest word of the header
        need = [max(min(max(len(r[j]) if j < len(r) else 0 for r in rows[1:] or rows), 28),
                    max(map(len, rows[0][j].split())) if j < len(rows[0]) and rows[0][j] else 0) + 3 for j in range(n)]
        widths = [TEXT_WIDTH_CM * w / sum(need) for w in need]
        for col, w in zip(tbl.findall(qn("w:tblGrid") + "/" + qn("w:gridCol")), widths, strict=False):
            col.set(qn("w:w"), str(int(Cm(w).twips)))
        for tr in tbl.findall(qn("w:tr")):
            for tc, w in zip(tr.findall(qn("w:tc")), widths, strict=False):
                tcw = tc.find(qn("w:tcPr") + "/" + qn("w:tcW"))
                tcw.set(qn("w:w"), str(int(Cm(w).twips)))
                tcw.set(qn("w:type"), "dxa")
        return tbl

    def code(self, spec: str) -> list:
        where, title = (s.strip() for s in spec.split("|", 1))
        path, first, last = where.split()
        text = Path(path).read_text(encoding="utf-8").splitlines()[int(first) - 1:int(last)]
        head = self.doc.add_paragraph()
        head.add_run(f"{title} ({path}, lines {first}–{last})").bold = True
        head.paragraph_format.keep_with_next = True
        p = self.doc.add_paragraph()
        for i, line in enumerate(text):
            run = p.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(7.5)
            if i < len(text) - 1:
                run.add_break()
        shade(p._p.get_or_add_pPr(), "F2F2F2")
        self.compact(p._p)
        return [self._detach(head._p), self._detach(p._p)]

    # --- sections and page numbers --------------------------------------------------------------------------------
    @staticmethod
    def set_page_numbers(sect, fmt: str, title_page: bool) -> None:
        for tag in ("w:pgNumType", "w:titlePg"):
            for old in sect.findall(qn(tag)):
                sect.remove(old)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), fmt)
        pg.set(qn("w:start"), "1")
        later = [qn(f"w:{t}") for t in ("cols", "formProt", "vAlign", "noEndnote", "titlePg", "textDirection", "bidi",
                                        "rtlGutter", "docGrid", "printerSettings")]
        nxt = next((c for c in sect if c.tag in later), None)
        nxt.addprevious(pg) if nxt is not None else sect.append(pg)
        if title_page:
            tp = OxmlElement("w:titlePg")
            nxt = next((c for c in sect if c.tag in later[5:]), None)
            nxt.addprevious(tp) if nxt is not None else sect.append(tp)

    def section_break(self, final_sect):
        """End the front matter: roman page numbers up to here (none on the cover), arabic from Chapter 1."""
        sect = copy.deepcopy(final_sect)
        self.set_page_numbers(sect, "lowerRoman", title_page=True)
        self.set_page_numbers(final_sect, "decimal", title_page=False)
        p = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(sect)
        p.append(ppr)
        return p

    # --- the proposal as Appendix A -------------------------------------------------------------------------------
    def proposal(self, elements: list) -> list:
        """V7's body, kept out of the thesis TOC and lists: headings restyled, captions renumbered under their own
        sequence, bookmarks and the TOC removed, and the ids Word requires to be unique renumbered."""
        style_id = self.doc.styles[PROPOSAL_HEADING].style_id
        out = []
        for el in (copy.deepcopy(e) for e in elements):
            if el.tag == qn("w:sdt") and el.find(".//" + qn("w:docPartGallery")) is not None and \
                    "Contents" in el.find(".//" + qn("w:docPartGallery")).get(qn("w:val")):
                continue
            for ps in el.iter(qn("w:pStyle")):
                if ps.get(qn("w:val")).startswith("Heading"):
                    ps.set(qn("w:val"), style_id)
            for f in el.iter(qn("w:fldSimple")):
                f.set(qn("w:instr"), re.sub(r"SEQ (Figure|Table)", r"SEQ Proposal\1", f.get(qn("w:instr"))))
            for t in el.iter(qn("w:instrText")):
                t.text = re.sub(r'\\c "(Figure|Table)"', r'\\c "Proposal\1"', t.text or "")
            for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
                for b in list(el.iter(qn(tag))):
                    b.getparent().remove(b)
            for i in el.iter(qn("w:id")):
                if i.getparent().tag == qn("w:sdtPr"):
                    i.set(qn("w:val"), str(next(_ids)))
            for dp in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"):
                dp.set("id", str(next(_ids)))
            out.append(el)
        return out

    def add_abbreviations(self) -> None:
        """Like Builder.add_abbreviations, but only for abbreviations V7's table does not have yet."""
        table = self.doc.tables[0]
        have = {r.cells[0].text.strip() for r in table.rows}
        template = table.rows[-1]._tr
        for short, long in (a for a in EXTRA_ABBREVIATIONS if a[0] not in have):
            tr = copy.deepcopy(template)
            template.addnext(tr)
            template = tr
            for cell_el, text in zip(tr.findall(qn("w:tc")), (short, long), strict=False):
                self.set_text(Paragraph(cell_el.findall(qn("w:p"))[0], self.doc), text)


def outline_lines(outline: Path) -> list[str]:
    lines = outline.read_text(encoding="utf-8").splitlines()
    return lines[next(i for i, ln in enumerate(lines) if ln.strip() == "-->") + 1:]


def numbering(lines: list[str]) -> dict[str, str]:
    """Figure and table numbers ('3.2') in document order, so the text can cite them before they appear."""
    numbers, chapter, fig, tab = {}, "", 0, 0
    for s in (ln.strip() for ln in lines):
        if m := re.match(r"#\s+(.*)", s):
            chapter, fig, tab = chapter_label(m.group(1)) or "", 0, 0
        elif s.startswith("FIG "):
            fig += 1
            numbers[f"fig:{s[4:].strip()}"] = f"{chapter}.{fig}"
        elif s.startswith("RESULTS "):
            tab += 1
            if m := re.search(r"\sid=([\w.-]+)", s):
                numbers[f"tab:{m.group(1)}"] = f"{chapter}.{tab}"
        elif m := re.match(r"CAPTION (Figure|Table)\s*([\w.-]*)\s*:", s):
            if m.group(1) == "Figure":
                fig += 1
            else:
                tab += 1
            if m.group(2):
                numbers[f"{'fig' if m.group(1) == 'Figure' else 'tab'}:{m.group(2)}"] = \
                    f"{chapter}.{fig if m.group(1) == 'Figure' else tab}"
    return numbers


def build(out: Path, outline: Path = OUTLINE, cover_note: str | None = None) -> None:
    doc = Document(str(V7))
    lines = outline_lines(outline)
    b = ThesisBuilder(doc, numbering(lines))
    b.body_format()
    body = b.body
    elements = list(body)
    final_sect = elements[-1]
    original = elements[:-1]

    def text(el) -> str:
        return Paragraph(el, doc).text.strip() if el.tag == qn("w:p") else ""

    def index(pred) -> int:
        return next(i for i, el in enumerate(original) if pred(el))

    title = text(original[index(lambda el: text(el).startswith("A Contextual Question Answering System"))])
    toc = next(el for el in original if el.tag == qn("w:sdt") and el.find(".//" + qn("w:docPartGallery")) is not None)
    lof_at, lot_at = index(lambda el: text(el) == "LIST OF FIGURES"), index(lambda el: text(el) == "LIST OF TABLES")
    abbr_at = index(lambda el: text(el) == "LIST OF ABBREVIATIONS")
    # the template's order: tables, figures, abbreviations (V7 has figures first)
    lists = original[lot_at:abbr_at] + original[lof_at:lot_at] + original[abbr_at:abbr_at + 2]
    ref_at = index(lambda el: text(el) == "Reference")
    references = original[ref_at:ref_at + 2]
    proposal = b.proposal(original)
    for el in original:
        body.remove(el)

    def add(els) -> None:
        for el in els if isinstance(els, list) else [els]:
            final_sect.addprevious(el)

    def cover_line(s: str = "", small_caps: bool = False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        if s:
            p.add_run(s).font.small_caps = small_caps
        return b._detach(p._p)

    # cover, as in the template: blank lines (1.5-spaced, TNR 12) between title, name, report type and month;
    # 4 lines after a two-line title
    def blanks(n: int) -> list:
        return [cover_line() for _ in range(n)]

    add(blanks(7) + [cover_line(title, small_caps=True)] + blanks(4) + [cover_line("Narayana Khadri", small_caps=True)]
        + blanks(4) + [cover_line("Final Thesis Report")])
    add(blanks(4) + [cover_line(cover_note)] + blanks(5) if cover_note else blanks(11))
    add(cover_line("OCTOBER 2026"))

    i, para = 0, []

    def flush():
        if para:
            add(b.paragraph(sig_stars(" ".join(para))))
            para.clear()

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush()
        elif s == "COVER":
            pass
        elif s == "DECLARATION":
            add(b.heading("Declaration", 1))
            add(b.brief("Use the university's declaration wording (ask the supervisor for the template): the work is "
                        "your own, sources are acknowledged, and how AI tools were used. Sign and date.", False))
        elif s == "TOC":
            heading = b.paragraph("TABLE OF CONTENTS", style="TOC Heading")
            Paragraph(heading, doc).paragraph_format.page_break_before = True
            for p in toc.iter(qn("w:p")):  # V7's own "Contents" title inside the TOC block
                if Paragraph(p, doc).text.strip() == "Contents":
                    p.getparent().remove(p)
                    break
            add([heading, toc])
        elif s == "DEDICATION":
            add(b.heading("Dedication", 1))
            add(b.brief("Optional: one or two lines of your own (for example, to family). Delete the page if not "
                        "wanted.", False))
        elif s == "LISTS":
            for el in lists:
                if text(el).startswith("LIST OF"):
                    Paragraph(el, doc).paragraph_format.page_break_before = True
                add(el)
            b.add_abbreviations()
        elif s == "END_FRONT_MATTER":
            flush()
            add(b.section_break(final_sect))
        elif s == "REFERENCES_TYPED":
            flush()
            add(b.heading("References", 1))
        elif s == "REFERENCES":
            flush()
            head, bib = references
            b.set_text(Paragraph(head, doc), "REFERENCES")
            Paragraph(head, doc).paragraph_format.page_break_before = True
            add([head, bib])
        elif s == "PROPOSAL":
            add(proposal)
        elif m := re.match(r"(#{1,3})\s+(.*)", s):
            flush()
            add(b.heading(m.group(2), len(m.group(1))))
        elif s.startswith(">"):
            flush()
            body_text = s[1:].strip()
            if body_text.startswith("- "):
                text_ = body_text[2:]
                while i + 1 < len(lines) and re.match(r">\s{2,}\S", lines[i + 1]):
                    i += 1
                    text_ += " " + lines[i][1:].strip()
                add(b.brief(text_, True))
            else:
                while i + 1 < len(lines) and lines[i + 1].startswith(">") and not lines[i + 1][1:].strip().startswith("- "):
                    i += 1
                    body_text += " " + lines[i][1:].strip()
                add(b.brief(body_text, False))
        elif s.startswith("- "):
            flush()
            text_ = s[2:]
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip():
                i += 1
                text_ += " " + lines[i].strip()
            add(b.paragraph(sig_stars(text_), bullet=True))
        elif s.startswith("FIG "):
            flush()
            add(b.figure_with_caption(s[4:].strip()))
        elif s.startswith("RESULTS "):
            flush()
            add(b.results_table(s[8:].strip()))
            add(b.paragraph(""))
        elif s.startswith("CODE "):
            flush()
            add(b.code(s[5:]))
        elif s.startswith("CAPTION "):
            flush()
            kind, title = s[8:].split(":", 1)
            kind = kind.split()[0]
            block = []
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                block.append(lines[i])
            add([b.caption(kind.strip(), title.strip()), b.fitted_table(block), b.paragraph("")])
        else:
            para.append(s)
        i += 1
    flush()

    b.update_fields_on_open()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--outline", type=Path, default=OUTLINE)
    ap.add_argument("--cover-note")
    ap.add_argument("--no-fields", action="store_true",
                    help="skip updating the fields in Word (TOC, lists, numbers stay stale until Word updates them)")
    a = ap.parse_args()
    build(a.out, a.outline, a.cover_note)
    if not a.no_fields:
        from scripts.thesis.update_word_fields import update

        update([a.out])


if __name__ == "__main__":
    main()
