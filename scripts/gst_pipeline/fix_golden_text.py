"""Mechanical text fixes for the GST golden set -> seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl.

The question extraction split the study material in the wrong places: the last line of a question (or the case facts
of its sub-parts) ended up at the start of the next record, and the next record's own question at the end of a later
one. Answers were paired correctly, so each chain is repaired by moving text back to the record whose answer it
belongs to. Page artifacts ('avv', running headers, page numbers) are removed everywhere. No text is written: every
fix only moves or deletes extracted text. Gold sources are not touched.

Each fixed record keeps question_before_textfix / reference_answer_before_textfix and a text_fix_note.
The original golden file is only read.

    uv run python -m scripts.gst_pipeline.fix_golden_text
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path("seed/docs/gst/golden_set_with_gold_sources.jsonl")
OUT = Path("seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl")

ARTIFACTS = [
    re.compile(r"(?m)^avv\s*$\n?"),                       # page mark of the PDF conversion
    re.compile(r"(?m)^#+ *GOODS AND SERVICES TAX\s*$\n?"),  # running header
    re.compile(r"(?m)^#+ *[A-Z][A-Z ]+\.\d+\s*$\n?"),      # running header with page number, e.g. 'VALUE OF SUPPLY.103'
    re.compile(r"(?m)^\d{1,3}\s*$\n?"),                    # bare page number
]


def clean(text: str) -> str:
    for pat in ARTIFACTS:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def after(text: str, marker: str) -> str:
    return text[text.index(marker):]


def before(text: str, marker: str) -> str:
    return text[: text.index(marker)].rstrip()


def between(text: str, start: str, end: str | None = None) -> str:
    i = text.index(start)
    return (text[i: text.index(end, i)] if end else text[i:]).strip()


def drop_numbering(text: str) -> str:
    """'2. (i) ...' list numbers that the PDF conversion put in front of sub-parts."""
    return re.sub(r"(?m)^\d+\. (?=\()", "", text)


def chain_fixes(g: dict[str, dict]) -> dict[str, tuple[dict, str]]:
    """id -> ({field: new text}, note)."""
    Q = {i: r["question"] for i, r in g.items()}
    A = {i: r["reference_answer"] for i, r in g.items()}
    f: dict[str, tuple[dict, str]] = {}
    moved = "The extraction split the question in the wrong place; text moved back to the record whose answer it belongs to."

    # p5u3: last line of q001 sat at the start of q002.
    p = "87621bos-aps2102-p5u3_"
    tail = before(Q[p + "q002"], "\n2. Damani")
    f[p + "q001"] = ({"question": Q[p + "q001"] + "\n\n" + tail}, moved)
    f[p + "q002"] = ({"question": after(Q[p + "q002"], "Damani Industries").replace("\n4. Will your answer", "\n\nWill your answer")}, moved)

    # p5u4: sub-parts (a)/(b) of q001 sat in q002/q003; q002's question sat in q003, q003's after it.
    p = "87622bos-aps2102-p5u4_"
    q3 = Q[p + "q003"]
    f[p + "q001"] = ({"question": Q[p + "q001"] + "\n\n" + Q[p + "q002"] + "\n\n" + before(q3, "\n2. Shiva Medical")}, moved)
    f[p + "q002"] = ({"question": between(q3, "Shiva Medical Centre", "\n3. Vedanta Hospital").replace("\n- ", "\n")}, moved)
    f[p + "q003"] = ({"question": after(q3, "Vedanta Hospital")}, moved)

    # p5u5: last line of q001 sat at the start of q002.
    p = "87623bos-aps2102-p5u5_"
    f[p + "q001"] = ({"question": Q[p + "q001"] + "\n\n" + before(Q[p + "q002"], "\n2. I buy")}, moved)
    f[p + "q002"] = ({"question": after(Q[p + "q002"], "I buy a set").replace("\n4. Is this correct", "\n\nIs this correct")}, moved)

    # ch6: sub-parts (i)-(iii) of q003 sat in q004-q006; the questions of q004-q006 all sat in q006.
    p = "88087bos-aps2102-ch6_"
    q6 = Q[p + "q006"]
    f[p + "q003"] = ({"question": "\n".join([Q[p + "q003"], Q[p + "q004"], Q[p + "q005"], before(q6, "\n4. Rajesh")])}, moved)
    f[p + "q004"] = ({"question": between(q6, "Rajesh & Co., a partnership firm", "5. The supplies of commodity")}, moved)
    f[p + "q005"] = ({"question": between(q6, "The supplies of commodity 'y'", "6. Easy Coupons Ltd.")}, moved)
    f[p + "q006"] = ({"question": after(q6, "Easy Coupons Ltd.")}, moved)

    # p7: last line of q001 sat at the start of q002; the end of q004's answer sat at the start of q005's answer.
    p = "88886bos-aps2102-m2-p7_"
    f[p + "q001"] = ({"question": Q[p + "q001"] + "\n\n" + before(Q[p + "q002"], "\n2. Jumbo")}, moved)
    f[p + "q002"] = ({"question": after(Q[p + "q002"], "Jumbo Sales Pvt. Ltd.")}, moved)
    a5 = A[p + "q005"]
    f[p + "q004"] = ({"reference_answer": A[p + "q004"].replace("\n4. o Tumkur", "\n- o Tumkur") + "\n" + before(a5, "5. As per section 16")}, moved)
    f[p + "q005"] = ({"reference_answer": after(a5, "As per section 16, Mohan Ltd.")}, moved)

    # p9: sub-parts (i)-(iii) of q001 sat in q002-q004; the questions of q002-q004 all sat in q004.
    p = "88888bos-aps2102-m2-p9_"
    q4 = Q[p + "q004"]
    f[p + "q001"] = ({"question": "\n".join([Q[p + "q001"], Q[p + "q002"], Q[p + "q003"], before(q4, "\n2. Avtaar")])}, moved)
    f[p + "q002"] = ({"question": between(q4, "Avtaar Enterprises", "3. Discuss the provisions")}, moved)
    f[p + "q003"] = ({"question": drop_numbering(between(q4, "Discuss the provisions", "4. Pari & Sons"))}, moved)
    f[p + "q004"] = ({"question": after(q4, "Pari & Sons is an unregistered")}, moved)

    # p12: sub-part (b) of q004 sat at the start of q005.
    p = "88891bos-aps2102-m2-p12_"
    f[p + "q004"] = ({"question": Q[p + "q004"].replace("\n4. (a)", "\n(a)") + "\n" + before(Q[p + "q005"], "\n5. A is an")}, moved)
    f[p + "q005"] = ({"question": after(Q[p + "q005"], "A is an e-commerce operator")}, moved)

    # p5u4: the tent table of q007 (i) sat inside q008's question, between its items (1) and (2).
    p = "87622bos-aps2102-p5u4_"
    q8 = Q[p + "q008"]
    table = between(q8, "|   No. of tents rented", "- (2) Bus fees")
    f[p + "q007"] = ({"question": Q[p + "q007"].replace(
        "on 8th December is as under:", "on 8th December is as under:\n\n" + table + "\n", 1)}, moved)
    f[p + "q008"] = ({"question": q8.replace(table, "").replace("\n\n\n", "\n")}, moved)

    # p7: the last part of q015's answer (B & D Company stock / ITC table) sat at the start of q016's answer.
    p = "88886bos-aps2102-m2-p7_"
    a16 = A[p + "q016"]
    head = before(a16, "| S. No.   | Particulars")
    f[p + "q015"] = ({"reference_answer": A[p + "q015"] + "\n\n" + head}, moved)
    f[p + "q016"] = ({"reference_answer": after(a16, "| S. No.   | Particulars")}, moved)

    # Tables that Docling turned into an image or stripped of labels: text copied from the text layer of the ICAI
    # PDF (resource.cdn.icai.org/<source_file>.pdf; p5u4 page 157, p5u2 page 81).
    from_pdf = "Table restored from the text layer of the original ICAI PDF (the Markdown conversion had lost it)."
    i = "87622bos-aps2102-p5u4_q014"
    f[i] = ({"question": Q[i].replace("during the month of April:\n\n", "during the month of April:\n\n"
        "| S. No. | Description of services provided |\n|---|---|\n"
        "| (i) | Transportation of students & staff of 'Shiksha University', a Deemed University |\n"
        "| (ii) | Catering services provided to 'Rank CBSE School' |\n"
        "| (iii) | Security personnel services provided to 'Win CBSE School', for its annual sports day held at SAI "
        "Sports Complex owned by Government of India |\n"
        "| (iv) | Supply of online periodical science journal to 'Merit CBSE School' for its higher secondary students |\n"
        "| (v) | Services, in relation to placement of students, to 'SKILL', a Government recognized vocational "
        "training college |\n\n", 1)}, from_pdf)
    i = "87620bos-aps2102-p5u2_q008"
    f[i] = ({"question": Q[i].replace("The firm specializes\n\nin providing", "The firm specializes in providing").replace(
        "| Amount (in ₹ )   |\n|------------------|\n| 10 lakh          |\n| 20 lakh          |",
        "| Quarter | Amount (in ₹) |\n|---|---|\n| Apr-Jun | 10 lakh |\n| Jul-Sep | 20 lakh |")}, from_pdf)

    # p13: stray line from another question inside q006.
    i = "88892bos-aps2102-m2-p13_q006"
    f[i] = ({"question": Q[i].replace(
        "taken a view\n\nwithin a period not exceeding 90 days from the date of service of the notice\n\n- that", "taken a view that")},
        "Removed a stray line ('within a period not exceeding 90 days ...') that belongs to another question.")
    return f


def main() -> None:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    g = {r["id"]: r for r in rows}
    fixes = chain_fixes(g)
    n_chain = n_artifact = 0
    for r in rows:
        new, notes = {}, []
        if r["id"] in fixes:
            new.update(fixes[r["id"]][0])
            notes.append(fixes[r["id"]][1])
        for field in ("question", "reference_answer"):
            text = clean(new.get(field, r[field]))
            if text != r[field]:
                if field not in new:
                    notes.append("Removed page artifacts (page marks, running headers, page numbers).")
                new[field] = text
        if new:
            n_chain += r["id"] in fixes
            n_artifact += r["id"] not in fixes
            for field, text in new.items():
                r[f"{field}_before_textfix"] = r[field]
                r[field] = text
            r["text_fix_note"] = " ".join(dict.fromkeys(notes))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} records -> {OUT}: {n_chain} chain fixes, {n_artifact} artifact-only fixes")


if __name__ == "__main__":
    main()
