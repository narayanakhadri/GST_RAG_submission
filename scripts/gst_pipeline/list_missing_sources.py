"""List the notifications and circulars the golden set needs but the corpus lacks -> seed/docs/gst/sources/to_ingest.json.

Three evidence sources, each kept per entry:
- "answer_text": the instrument is named in a reference answer or question (regex over all 146 records);
- "golden_annotation": it is in missing_gold_sources / external_sources_missing_from_corpus;
- "llm_precheck": the pre-check suggested it (dataset/sme_priority_review/llm_precheck_verdicts*.jsonl).

An instrument already in the corpus (by instrument_id, or via sources/authority_aliases.json) is left out. The output
is a work list for fetch_sources.py: url is left empty to be filled from the gstcouncil.gov.in / CBIC listings.
Nothing is downloaded or ingested here.

    uv run python -m scripts.gst_pipeline.list_missing_sources
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

GOLDEN = Path("seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl")
CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
ALIASES = Path("seed/docs/gst/sources/authority_aliases.json")
PRECHECK = [Path("dataset/sme_priority_review/llm_precheck_verdicts.jsonl"),
            Path("dataset/sme_priority_review/llm_precheck_verdicts_rest.jsonl")]
OUT = Path("seed/docs/gst/sources/to_ingest.json")

CITATION = re.compile(
    r"(?P<kind>Notification|Circular)\s+No\.?\s*(?P<num>\d+(?:/\d+)*/(?:20)?\d{2})\s*[-–]?\s*"
    r"(?P<series>Central\s+Tax|Integrated\s+Tax|Union\s+Territory\s+Tax|CT|IT|UTT|GST)?\s*(?P<rate>\(\s*(?:Rate|R)\s*\))?",
    re.I,
)
# Answers that name an instrument without its series: the alias file explains each.
OVERRIDES = {"notification_41_2017_ct": "notification_41_2017_it_rate"}  # merchant exports, IGST (Rate)
# Notifications the answers cite as 'Notification No. X/2017-CT' although they are rate notifications.
RATE_BY_CONTEXT = {"notification_12_2017_ct", "notification_13_2017_ct", "notification_17_2017_ct",
                   "notification_9_2017_it", "notification_10_2017_it", "notification_11_2017_ct"}


def instrument_id(kind: str, num: str, series: str | None, rate: bool) -> str:
    parts = num.split("/")
    if len(parts[-1]) == 2:
        parts[-1] = "20" + parts[-1]
    if kind.lower() == "circular":
        return "circular_" + "_".join(parts) + "_gst"
    s = (series or "CT").lower().replace(" ", "")
    s = {"centraltax": "ct", "integratedtax": "it", "unionterritorytax": "utt"}.get(s, s)
    iid = f"notification_{int(parts[0])}_{parts[-1]}_{s}"
    iid = iid + "_rate" if rate or iid in RATE_BY_CONTEXT else iid
    return OVERRIDES.get(iid, iid)


def in_corpus(iid: str, corpus: set[str]) -> bool:
    """Circulars are often cited short ('Circular No. 44/2018'): match on serial number and year."""
    if iid in corpus:
        return True
    p = iid.split("_")
    return p[0] == "circular" and any(c.startswith(f"circular_{int(p[1])}_") and f"_{p[-2]}_" in c + "_" for c in corpus)


def label(iid: str) -> str:
    p = iid.split("_")
    if p[0] == "circular":
        return f"Circular No. {'/'.join(p[1:-1])}-GST"
    series = {"ct": "Central Tax", "it": "Integrated Tax", "utt": "Union Territory Tax"}[p[3]]
    return f"Notification No. {p[1]}/{p[2]}-{series}" + (" (Rate)" if iid.endswith("_rate") else "")


def main() -> None:
    corpus = {json.loads(line)["instrument_id"] for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))["aliases"]
    found: dict[str, dict] = defaultdict(lambda: {"cited_in": set(), "evidence": set(), "citations": set()})

    def add(iid: str, qid: str, evidence: str, text: str) -> None:
        if in_corpus(iid, corpus) or aliases.get(text) in corpus:
            return
        found[iid]["cited_in"].add(qid)
        found[iid]["evidence"].add(evidence)
        found[iid]["citations"].add(text)

    def from_text(text: str, qid: str, evidence: str) -> int:
        n = 0
        for m in CITATION.finditer(text):
            add(instrument_id(m["kind"], m["num"], m["series"], bool(m["rate"])), qid, evidence, m.group(0).strip(" -–"))
            n += 1
        return n

    for r in map(json.loads, GOLDEN.read_text(encoding="utf-8").splitlines()):
        from_text(r["question"] + "\n" + r["reference_answer"], r["id"], "answer_text")
        for cite in set(r.get("missing_gold_sources") or []) | set(r.get("external_sources_missing_from_corpus") or []):
            if not from_text(cite, r["id"], "golden_annotation") and cite not in corpus and aliases.get(cite) not in corpus \
                    and cite.split(".")[0] not in {"cgst_act_2017", "igst_act_2017", "cgst_rules_2017", "igst_rules_2017", "utgst_act_2017"}:
                found["other:" + cite]["cited_in"].add(r["id"])
                found["other:" + cite]["evidence"].add("golden_annotation")
                found["other:" + cite]["citations"].add(cite)
    for v in (json.loads(line) for f in PRECHECK if f.exists() for line in f.read_text(encoding="utf-8").splitlines()):
        for m in v["missing"]:
            if m["source"] not in corpus and not from_text(m["source"], v["id"], "llm_precheck") and "." not in m["source"].split()[0]:
                found["other:" + m["source"]]["cited_in"].add(v["id"])
                found["other:" + m["source"]]["evidence"].add("llm_precheck")
                found["other:" + m["source"]]["citations"].add(m["source"])

    rows = []
    for iid, d in found.items():
        other = iid.startswith("other:")
        rows.append({
            "instrument_id": None if other else iid,
            "kind": "other" if other else iid.split("_")[0],
            "number": iid[6:] if other else label(iid),
            "cited_in": sorted(d["cited_in"]),
            "evidence": sorted(d["evidence"]),
            "citations_seen": sorted(d["citations"]),
            "url": "",
        })
    rows.sort(key=lambda r: (r["kind"] == "other", -len(r["cited_in"]), r["number"]))
    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} instruments not in the corpus -> {OUT}")
    for r in rows:
        print(f"{len(r['cited_in']):3d}  {r['number']:<60} {','.join(r['evidence'])}")


if __name__ == "__main__":
    main()
