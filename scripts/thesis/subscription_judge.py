"""Free LLM judge via a chat subscription (Claude Code, Codex CLI, ...): files in, files out.

The judge is any agent that can read and write files. It follows `judge/JUDGE_INSTRUCTIONS.md`; this script only
prepares its input and turns its output into the same results format as eval/rescore_ragas.py, so
scripts/techniques/compare_ragas.py and the thesis tables work unchanged.

    judge/
      JUDGE_INSTRUCTIONS.md          rubric + output format (fixed)
      reference/<qid>.md             question, ICAI reference answer, required law (fixed; `references`)
      runs/<run>/manifest.json       item -> source run file + question id (the judge must not open it)
      runs/<run>/items/<item>.md     one answer to grade: question, retrieved law, answer
      runs/<run>/out/<item>.json     the judge's verdict (one file per item: resumable, parallel-safe)

    uv run python -m scripts.thesis.subscription_judge references
    uv run python -m scripts.thesis.subscription_judge export final_t31_all eval/results/<run>.json [--ids-from F] [--limit N] [--blind]
    uv run python -m scripts.thesis.subscription_judge status final_t31_all
    uv run python -m scripts.thesis.subscription_judge import final_t31_all      # -> <run>.judge-<judge>.json next to the source
    uv run python -m scripts.thesis.subscription_judge agree <a>.judge-X.json <b>.judge-Y.json   # judge vs judge

Scores are computed here from the judge's claim lists, not taken from the judge's own arithmetic:
- faithfulness        = answer claims supported by the retrieved law / all answer claims        (as RAGAS)
- answer_correctness  = claim F1 vs the reference: TP / (TP + 0.5 (FP + FN)), FP = answer claims not in or
                        contradicting the reference, FN = reference points the answer misses     (RAGAS factual part)
- correctness_lenient = the same, but claims that are merely absent from the reference are not penalised
- answer_relevancy    = (1–5 rubric − 1) / 4  (RAGAS uses embeddings here, so this one is not comparable)
- conclusion          = 1 same / 0.5 partly / 0 different or abstained (exam-style: right final answer?)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("judge")
REF_DIR = ROOT / "reference"
RUNS = ROOT / "runs"
METRICS = ("faithfulness", "answer_correctness", "correctness_lenient", "answer_relevancy", "conclusion")


def references() -> None:
    from app.config import settings
    from app.datasets.gst.corpus_loader import load_corpus
    from app.datasets.gst.document_mapper import source_label
    from scripts.techniques.common import questions

    labels = {r["id"]: source_label(r) for r in load_corpus(settings.gst_corpus_path)}
    REF_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for g in questions("all"):
        req = g.get("primary_required_sources") or []
        law = "\n".join(f"- {labels.get(s, s)}" for s in req) or "- (none listed)"
        (REF_DIR / f"{g['id']}.md").write_text(
            f"# Reference {g['id']}\n\n## Question\n\n{g['question']}\n\n## Reference answer (ICAI)\n\n"
            f"{g['reference_answer']}\n\n## Law the answer needs\n\n{law}\n", encoding="utf-8")
        n += 1
    print(f"wrote {n} reference files to {REF_DIR}")


def export(name: str, files: list[Path], ids_from: Path | None, limit: int, blind: bool) -> None:
    wanted = set(ids_from.read_text(encoding="utf-8").split()) if ids_from else None
    run = f"blind_{hashlib.sha256(name.encode()).hexdigest()[:8]}" if blind else name
    items_dir = RUNS / run / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (RUNS / run / "out").mkdir(exist_ok=True)
    manifest, k = {"name": name, "items": {}}, 0
    for spec in files:
        # "run.json@ids.txt" restricts that one file to the listed question ids (mix systems in one run).
        path, _, ids_file = str(spec).partition("@")
        f = Path(path)
        keep = set(Path(ids_file).read_text(encoding="utf-8").split()) if ids_file else wanted
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if (keep and r["id"] not in keep) or not (r.get("answer") or "").strip():
                continue
            if limit and k >= limit:
                break
            k += 1
            item = f"{k:03d}"
            ctx = "\n\n".join(f"### Passage {i + 1}\n\n{c}" for i, c in enumerate(r["contexts"]))
            (items_dir / f"{item}.md").write_text(
                f"# Item {item}\n\nqid: {r['id']}\n\n## Question\n\n{r['question']}\n\n## Answer to grade\n\n"
                f"{r['answer']}\n\n## Retrieved law (the only context the answer was allowed to use)\n\n{ctx}\n",
                encoding="utf-8")
            manifest["items"][item] = {"source": str(f), "qid": r["id"]}
    (RUNS / run / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"exported {k} items to {RUNS / run}" + (f"  (blind name for '{name}')" if blind else ""))


def _scores(v: dict) -> dict:
    claims = v.get("answer_claims") or []
    points = v.get("reference_points") or []
    sup = sum(1 for c in claims if c.get("supported_by_context") is True)
    tp = sum(1 for c in claims if c.get("matches_reference") == "yes")
    fp_contra = sum(1 for c in claims if c.get("matches_reference") == "contradicts")
    fp_absent = sum(1 for c in claims if c.get("matches_reference") == "not_in_reference")
    fn = sum(1 for p in points if not p.get("covered_by_answer"))

    def f1(fp: int) -> float | None:
        den = tp + 0.5 * (fp + fn)
        return round(tp / den, 4) if den else None

    rel = v.get("relevancy_1to5")
    return {"faithfulness": round(sup / len(claims), 4) if claims else None,
            "answer_correctness": f1(fp_contra + fp_absent), "correctness_lenient": f1(fp_contra),
            "answer_relevancy": round((float(rel) - 1) / 4, 4) if rel is not None else None,
            "conclusion": {"same": 1.0, "partly": 0.5, "different": 0.0, "abstained": 0.0}.get(v.get("conclusion"))}


def _load(run: str) -> tuple[dict, dict]:
    manifest = json.loads((RUNS / run / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for p in sorted((RUNS / run / "out").glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  {p.name}: invalid JSON ({exc})")
    return manifest, out


def status(run: str) -> None:
    manifest, out = _load(run)
    todo = [i for i in manifest["items"] if i not in out]
    bad = [i for i, v in out.items() if v.get("qid") != manifest["items"].get(i, {}).get("qid")
           or v.get("conclusion") not in ("same", "partly", "different", "abstained")
           or not isinstance(v.get("answer_claims"), list) or not isinstance(v.get("reference_points"), list)]
    print(f"{run}: {len(out)}/{len(manifest['items'])} graded; to do: {' '.join(todo) or 'none'}")
    if bad:
        print(f"  malformed (re-grade): {' '.join(bad)}")


def import_run(run: str) -> None:
    manifest, out = _load(run)
    judges = {v.get("judge", "unknown") for v in out.values()}
    if len(judges) != 1:
        raise SystemExit(f"one judge per import, found {sorted(judges)}")
    judge = judges.pop()
    tag = "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in judge)
    by_source: dict[str, dict[str, dict]] = {}
    for item, v in out.items():
        m = manifest["items"][item]
        by_source.setdefault(m["source"], {})[m["qid"]] = {**_scores(v), "item": item}
    for source, scores in by_source.items():
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        rows = [{**r, "ragas_metrics": scores[r["id"]]} for r in payload["rows"] if r["id"] in scores]
        means = {m: (round(statistics.mean(x), 3) if (x := [r["ragas_metrics"][m] for r in rows
                                                          if r["ragas_metrics"][m] is not None]) else None)
                 for m in METRICS}
        payload.update(rows=rows, aggregate=means, ragas_judge={"model": judge, "via": "subscription (files)"},
                       ragas_metrics=",".join(METRICS), rescored_from=source, judge_run=str(RUNS / run))
        dest = Path(source).with_name(f"{Path(source).stem}.judge-{tag}.json")
        dest.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{dest}: {len(rows)} rows  {means}")


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(x: list[float]) -> list[float]:
        order = sorted(range(len(x)), key=x.__getitem__)
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    return statistics.correlation(ranks(a), ranks(b)) if len(a) > 2 else float("nan")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and x == x  # x == x drops NaN (a judge call that timed out)


def agree(a_path: Path, b_path: Path) -> None:
    a = {r["id"]: r["ragas_metrics"] for r in json.loads(a_path.read_text(encoding="utf-8"))["rows"]}
    b = {r["id"]: r["ragas_metrics"] for r in json.loads(b_path.read_text(encoding="utf-8"))["rows"]}
    print(f"A = {a_path.name}\nB = {b_path.name}\n| metric | n | mean A | mean B | mean abs diff | Spearman |\n|---|---|---|---|---|---|")
    for m in sorted({k for v in a.values() for k in v} & {k for v in b.values() for k in v}):
        pairs = [(a[q][m], b[q][m]) for q in a.keys() & b.keys()
                 if _num(a[q].get(m)) and _num(b[q].get(m))]
        if not pairs:
            continue
        x, y = zip(*pairs, strict=True)
        print(f"| {m} | {len(x)} | {statistics.mean(x):.3f} | {statistics.mean(y):.3f} | "
              f"{statistics.mean(abs(i - j) for i, j in pairs):.3f} | {_spearman(list(x), list(y)):.2f} |")


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("references")
    e = sub.add_parser("export")
    e.add_argument("name")
    e.add_argument("files", nargs="+", type=Path, help="run files; path@ids.txt keeps only those question ids")
    e.add_argument("--ids-from", type=Path, help="text file with question ids (whitespace-separated)")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--blind", action="store_true", help="neutral folder name, so the judge cannot tell the technique")
    for c in ("status", "import"):
        sub.add_parser(c).add_argument("run")
    g = sub.add_parser("agree")
    g.add_argument("a", type=Path)
    g.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "references":
        references()
    elif args.cmd == "export":
        export(args.name, args.files, args.ids_from, args.limit, args.blind)
    elif args.cmd == "status":
        status(args.run)
    elif args.cmd == "import":
        import_run(args.run)
    else:
        agree(args.a, args.b)


if __name__ == "__main__":
    main()
