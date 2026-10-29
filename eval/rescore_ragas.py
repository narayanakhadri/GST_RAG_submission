"""Re-score the saved answers of a GST RAGAS results file with the configured judge — no RAG, no generation.

Used to grade a run whose scoring failed, or to put every run under the same judge:

    uv run python -m eval.rescore_ragas eval/results/<run>.json [more.json ...] \
        [--metrics faithfulness,answer_relevancy] [--judge-model M --judge-base-url URL] [--limit 20]
    uv run python -m eval.rescore_ragas eval/results/<run>.json --judge-cli codex   # subscription CLI as judge (eval/cli_llm.py)

The judge's key comes from RAGAS_JUDGE_API_KEY (or OPENAI_API_KEY for OpenAI). Output goes next to each input
as `<stem>.judge-<model>.json`; the input file is never modified. Rows whose answer is empty are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from app.config import settings
from eval.ragas_adapter import judge_info
from eval.ragas_adapter import run as run_ragas
from eval.reporting import aggregate


def rescore(path: Path, metrics: str, limit: int = 0, batch: int = 20, question_set: str | None = None,
            overwrite: bool = False, retry_missing: bool = False) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["rows"] if (r.get("answer") or "").strip()]
    if question_set:
        from scripts.thesis.make_question_sets import load_question_set

        wanted = load_question_set(question_set)
        rows = [r for r in rows if r["id"] in wanted]
    if limit:
        rows = rows[:limit]
    judge_name = (f"{settings.ragas_judge_cli}-{settings.ragas_judge_model or 'default'}" if settings.ragas_judge_cli
                  else settings.ragas_judge_model)
    tag = re.sub(r"[^A-Za-z0-9.-]+", "-", judge_name)
    suffix = (f".{question_set}" if question_set else "") + (f".limit{limit}" if limit else "")
    out = path.with_name(f"{path.stem}.judge-{tag}{suffix}.json")
    if out.exists() and retry_missing:
        # Keep every complete row; re-judge only rows with a missing metric (a judge call that timed out -> NaN).
        done = json.loads(out.read_text(encoding="utf-8"))
        if done.get("ragas_metrics") != metrics or done.get("ragas_judge") != judge_info():
            raise SystemExit(f"{out} was scored with other metrics/judge settings; --retry-missing needs the same ones")
        names = [n.strip() for n in metrics.split(",")]
        complete = {r["id"]: r["ragas_metrics"] for r in done["rows"]
                    if all(isinstance(r["ragas_metrics"].get(n), (int, float)) and r["ragas_metrics"][n] == r["ragas_metrics"][n]
                           for n in names)}
        out.with_suffix(".partial.json").write_text(json.dumps(complete, default=str), encoding="utf-8")
        print(f"{out.name}: re-judging {len(rows) - len(complete)} rows with missing scores", flush=True)
    elif out.exists() and not overwrite:
        done = json.loads(out.read_text(encoding="utf-8"))
        if done.get("ragas_metrics") == metrics and done.get("ragas_judge") == judge_info():
            print(f"{path.name}: already scored -> {out.name}, skipping", flush=True)
            return out
        raise SystemExit(f"{out} exists but was scored with other metrics/judge settings; pass --overwrite to replace it")

    # Scores are checkpointed per batch, so an interrupted run resumes instead of paying for the same calls again.
    checkpoint = out.with_suffix(".partial.json")
    by_id: dict[str, dict] = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    todo = [r for r in rows if r["id"] not in by_id]
    print(f"{path.name}: scoring {len(todo)} of {len(rows)} rows with {judge_name} [{metrics}]"
          f"{f' (resuming, {len(by_id)} done)' if by_id else ''}", flush=True)
    for start in range(0, len(todo), batch):
        part = todo[start:start + batch]
        by_id.update({r["id"]: m for r, m in zip(part, run_ragas(part, metrics), strict=True)})
        checkpoint.write_text(json.dumps(by_id, default=str), encoding="utf-8")
        print(f"  {len(by_id)}/{len(rows)} scored", flush=True)
    names = [n.strip() for n in metrics.split(",")]
    out_rows = []
    for r in payload["rows"]:
        if r["id"] not in by_id:
            continue  # empty answer, or outside --limit / --question-set
        m = by_id[r["id"]]
        out_rows.append({**r, "ragas_metrics": {k: m.get(k) for k in names}})
    payload.update(
        rows=out_rows,
        aggregate=aggregate(out_rows),
        ragas_judge=judge_info(),
        ragas_metrics=metrics,
        rescored_from=str(path),
    )
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"  aggregate: {payload['aggregate']}\n  wrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--metrics", default=settings.ragas_metrics)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-cli", choices=["codex", "claude"], default=None,
                    help="judge through a subscription CLI (eval/cli_llm.py); --judge-model is then the CLI's model")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N rows (calibration / smoke test)")
    ap.add_argument("--question-set", default=None, help="Named set from question_sets.json, e.g. quick40")
    ap.add_argument("--overwrite", action="store_true", help="Re-score even if an output with other settings exists")
    ap.add_argument("--retry-missing", action="store_true",
                    help="Re-judge only rows of an existing output whose score is missing (judge timeout)")
    args = ap.parse_args()
    if args.judge_cli:
        settings.ragas_judge_cli = args.judge_cli
        settings.ragas_judge_model = args.judge_model or ""  # not the API judge from .env
        settings.ragas_judge_max_workers = args.workers or 3
    elif args.judge_model:
        settings.ragas_judge_model = args.judge_model
    if args.judge_base_url is not None:
        settings.ragas_judge_base_url = args.judge_base_url
    if args.workers:
        settings.ragas_judge_max_workers = args.workers
    sys.stdout.reconfigure(encoding="utf-8")
    for path in args.results:
        rescore(path, args.metrics, args.limit, question_set=args.question_set, overwrite=args.overwrite,
                retry_missing=args.retry_missing)


if __name__ == "__main__":
    main()
