"""Hand the corpus to Codex (ChatGPT subscription) to check and extend its metadata, then validate the result.

    uv run python -m scripts.gst_pipeline.codex_kit export   # writes seed/docs/gst/codex/{catalog.tsv,batches/}
    #   ... run Codex on each batch, following docs/codex_metadata_task.md ...
    uv run python -m scripts.gst_pipeline.codex_kit check    # validates codex/out/*.jsonl -> codex_metadata.jsonl

Codex writes one JSON line per provision (schema in docs/codex_metadata_task.md): a verdict on the existing
LLM summary/metadata, typed links to other provisions (section -> rule -> notification -> circular), search
terms a question would use, generic questions the provision answers, and — for long provisions (listed in
codex/long_provisions.txt) — `units`: the provision split into its logical parts (sub-sections, provisos, explanations),
each located by the exact text it starts with, so any later chunking technique can split there. `check` rejects lines with unknown
ids, link targets outside the corpus, unknown relation types or missing evidence, and writes the valid lines to
seed/docs/gst/codex_metadata.jsonl. The corpus itself is never modified; experiments read the sidecar file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.document_mapper import source_label
from app.services.gst_graph import kind
from eval.gst_retrieval import corpus_keys

KIT = Path("seed/docs/gst/codex")
SIDECAR = Path("seed/docs/gst/codex_metadata.jsonl")
BATCH_SIZE = 25
RELATIONS = {
    "procedure_in": "this section's procedure/conditions are prescribed in the target rule (or form)",
    "empowered_by": "this rule/notification is made under the power in the target section",
    "rate_or_exemption_for": "this notification entry sets a rate/exemption for supplies governed by the target",
    "clarifies": "this circular paragraph clarifies the target provision",
    "amends": "this provision amends or substitutes the target",
    "defines_term_used_in": "this provision defines a term the target relies on",
    "counterpart": "same rule in another Act (e.g. CGST section <-> IGST/UTGST section)",
    "refers_to": "any other explicit or necessary reference",
}
VERDICTS = {"correct", "minor_issues", "wrong"}
MAX_LINKS, MAX_TERMS, MAX_QUESTIONS, MAX_EVIDENCE = 12, 15, 4, 300
LONG_TOKENS = 800  # provisions with more raw-text tokens than this must be split into `units`
MIN_UNITS, MAX_UNITS, MIN_START = 2, 25, 20


def export() -> None:
    records = [r for r in load_corpus(settings.gst_corpus_path) if r.get("status") != "omitted"]
    KIT.mkdir(parents=True, exist_ok=True)
    (KIT / "out").mkdir(exist_ok=True)
    with (KIT / "catalog.tsv").open("w", encoding="utf-8") as f:
        f.write("id\tkind\tlabel\ttitle\n")
        instruments = {}
        for r in records:
            f.write(f"{r['id']}\t{kind(r)}\t{source_label(r)}\t{(r.get('provision_title') or '').replace(chr(9), ' ')}\n")
            instruments.setdefault(r["instrument_id"], r)
        for iid, r in instruments.items():  # a whole instrument is also a valid link target
            f.write(f"{iid}\t{kind(r)}\t{r.get('instrument_title') or iid}\t(whole instrument)\n")
    enc = tiktoken.get_encoding("o200k_base")
    long_ids = [r["id"] for r in records if len(enc.encode(r.get("raw_text") or "")) > LONG_TOKENS]
    (KIT / "long_provisions.txt").write_text("\n".join(long_ids) + "\n", encoding="utf-8")
    batches = KIT / "batches"
    batches.mkdir(exist_ok=True)
    for old in batches.glob("batch_*.txt"):
        old.unlink()
    for n, start in enumerate(range(0, len(records), BATCH_SIZE), start=1):
        (batches / f"batch_{n:03d}.txt").write_text(
            "\n".join(r["id"] for r in records[start:start + BATCH_SIZE]) + "\n", encoding="utf-8")
    print(f"{len(records)} provisions in {n} batches of {BATCH_SIZE}: {batches}/\ncatalog: {KIT / 'catalog.tsv'}\n"
          f"{len(long_ids)} long provisions (> {LONG_TOKENS} tokens) need units: {KIT / 'long_provisions.txt'}\n"
          f"Codex writes to {KIT / 'out'}/batch_NNN.jsonl — see docs/codex_metadata_task.md")


def _validate_units(units: list, raw: str) -> list[str]:
    """Each unit must start with text that occurs in the raw law, in order; the first one at the top."""
    errs, pos = [], -1
    if not MIN_UNITS <= len(units) <= MAX_UNITS:
        errs.append(f"{len(units)} units (need {MIN_UNITS}-{MAX_UNITS})")
    for i, unit in enumerate(units):
        start = unit.get("starts_with") or ""
        if len(start.strip()) < MIN_START:
            errs.append(f"unit {i + 1}: starts_with shorter than {MIN_START} chars")
            continue
        found = raw.find(start, pos + 1)
        if found < 0:
            errs.append(f"unit {i + 1} ({unit.get('unit')}): starts_with not found (after the previous unit) in raw_text")
            continue
        if i == 0 and raw[:found].strip(" -\n\t"):
            errs.append("unit 1 must start at the beginning of raw_text")
        pos = found
        if not (unit.get("unit") and unit.get("summary")):
            errs.append(f"unit {i + 1}: unit label and summary are required")
    return errs


def _validate(line: dict, wanted: set[str], keys: set[str], raw: dict[str, str], long_ids: set[str]) -> list[str]:
    errs = []
    if line.get("id") in long_ids:
        if not line.get("units"):
            errs.append("long provision: `units` missing")
        else:
            errs += _validate_units(line["units"], raw.get(line["id"], ""))
    if line.get("id") not in wanted:
        errs.append(f"id {line.get('id')!r} is not in this batch")
    for part in ("summary_check", "metadata_check"):
        v = (line.get(part) or {}).get("verdict")
        if v not in VERDICTS:
            errs.append(f"{part}.verdict {v!r} not in {sorted(VERDICTS)}")
    links = line.get("links") or []
    if len(links) > MAX_LINKS:
        errs.append(f"{len(links)} links (max {MAX_LINKS})")
    for link in links:
        if link.get("target_id") not in keys:
            errs.append(f"link target {link.get('target_id')!r} not in catalog.tsv")
        if link.get("target_id") == line.get("id"):
            errs.append("link to itself")
        if link.get("relation") not in RELATIONS:
            errs.append(f"relation {link.get('relation')!r} unknown")
        if not (link.get("evidence") or "").strip() or len(link.get("evidence", "")) > MAX_EVIDENCE:
            errs.append(f"link to {link.get('target_id')}: evidence missing or > {MAX_EVIDENCE} chars")
    if len(line.get("search_terms") or []) > MAX_TERMS:
        errs.append(f"more than {MAX_TERMS} search_terms")
    if len(line.get("answers_questions") or []) > MAX_QUESTIONS:
        errs.append(f"more than {MAX_QUESTIONS} answers_questions")
    return errs


def check() -> None:
    corpus = load_corpus(settings.gst_corpus_path)
    keys = corpus_keys(corpus)
    raw = {r["id"]: r.get("raw_text") or "" for r in corpus}
    long_path = KIT / "long_provisions.txt"
    long_ids = set(long_path.read_text(encoding="utf-8").split()) if long_path.exists() else set()
    valid: dict[str, dict] = {}
    report = []
    for batch in sorted((KIT / "batches").glob("batch_*.txt")):
        wanted = set(batch.read_text(encoding="utf-8").split())
        out = KIT / "out" / f"{batch.stem}.jsonl"
        if not out.exists():
            report.append(f"{batch.stem}: not done")
            continue
        bad, seen = [], set()
        for n, raw_line in enumerate(out.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                line = json.loads(raw_line)
            except json.JSONDecodeError as e:
                bad.append(f"line {n}: not JSON ({e.msg})")
                continue
            errs = _validate(line, wanted, keys, raw, long_ids)
            if errs:
                bad.append(f"line {n} ({line.get('id')}): " + "; ".join(errs))
            else:
                valid[line["id"]] = line
                seen.add(line["id"])
        missing = sorted(wanted - seen)
        status = "ok" if not bad and not missing else "NEEDS FIX"
        report.append(f"{batch.stem}: {status} — {len(seen)}/{len(wanted)} valid"
                      + (f"; missing: {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}" if missing else ""))
        report += [f"    {b}" for b in bad[:20]]
    SIDECAR.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in valid.values()), encoding="utf-8")
    (KIT / "check_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    links = sum(len(v.get("links") or []) for v in valid.values())
    wrong = sum((v["summary_check"]["verdict"] == "wrong") for v in valid.values())
    units = sum(len(v.get("units") or []) for v in valid.values())
    print("\n".join(report))
    print(f"\n{len(valid)} valid provisions, {links} links, {units} units, {wrong} summaries judged wrong -> {SIDECAR}\n"
          f"report: {KIT / 'check_report.txt'} (give Codex the NEEDS FIX lines to correct)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "check"])
    args = ap.parse_args()
    export() if args.command == "export" else check()


if __name__ == "__main__":
    main()
