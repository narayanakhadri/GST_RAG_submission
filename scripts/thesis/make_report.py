"""Build the thesis results tables and figures from the saved experiment JSONs — never typed by hand.

    uv run python -m scripts.thesis.make_report

Reads eval/results/thesis/retrieval/*.json (scripts/thesis/run_ablation.py) and the RAGAS runs listed in
eval/results/thesis/ragas_runs.json ({"label": "path/to/results.json", ...}), and writes
eval/results/thesis/results_tables.md plus PNG figures in eval/results/thesis/figures/.
All retrieval numbers are on the 141 fully covered questions with corrected gold sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.citation_check import score_results  # noqa: E402
from eval.stats import HEADER, RAGAS_METRICS, format_row, metric_value, paired  # noqa: E402
from scripts.thesis.archive_results import read_json  # noqa: E402
from scripts.thesis.run_ablation import CONFIGS, OUT_DIR  # noqa: E402

THESIS_DIR = Path("eval/results/thesis")
FIG_DIR = THESIS_DIR / "figures"
RAGAS_RUNS = THESIS_DIR / "ragas_runs.json"
BASELINE = "hybrid_tfidf"
METRICS = ("required_recall", "all_required_hit", "any_required_hit", "required_mrr")
LABELS = {"required_recall": "Required law found", "all_required_hit": "All required found",
          "any_required_hit": "hit@k", "required_mrr": "MRR"}
TABLES = [
    ("RQ3 — retrieval method", ["dense", "sparse_tfidf", "sparse_bm25", "hybrid_tfidf", "hybrid_bm25", "hybrid_top8"]),
    ("RQ2 — what text is embedded, and the embedding model", ["embed_raw", "embed_exact", "hybrid_tfidf", "embed_large"]),
    ("RQ1 — document structure: doc-type balance, diversity and graph expansion",
     ["hybrid_tfidf", "hybrid_cap1", "hybrid_cap2", "hybrid_cap3", "hybrid_inst2", "hybrid_cap2_inst2", "hybrid_graph",
      "hybrid_graph_cap2", "hybrid_graphdense", "hybrid_graphdense_w1", "hybrid_graphdense_cap2"]),
    ("RQ3 — reranking", ["hybrid_tfidf", "rerank_minilm", "rerank_bge_m3", "rerank_llm", "hybrid_pool20"]),
    ("RQ3 — query understanding (LLM)", ["hybrid_tfidf", "hyde", "rewrite", "decompose"]),
    ("Final combination (E17)", ["hybrid_tfidf", "decompose", "large_top8_cap3", "large_top8_cap3_decompose"]),
]


def load(name: str) -> dict | None:
    path = OUT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def full_rows(payload: dict) -> list[dict]:
    return [r for r in payload["rows"] if (r.get("coverage") or {}).get("corrected") == "full"]


def ablation_table(names: list[str], runs: dict[str, dict]) -> str:
    base = runs.get(BASELINE)
    lines = ["| Config | Description | " + " | ".join(LABELS[m] for m in METRICS) + " | Δ recall vs baseline [95% CI] | p |",
             "|---|---|" + "---|" * len(METRICS) + "---|---|"]
    for name in names:
        p = runs.get(name)
        if p is None:
            continue
        agg = p["coverage_split"]["corrected"]["full"]
        cells = " | ".join("—" if agg.get(m) is None else f"{agg[m]:.3f}" for m in METRICS)
        delta = pval = "—"
        if base is not None and name != BASELINE:
            r = paired(base["rows"], p["rows"], "required_recall")
            if r.get("n"):
                delta = f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                pval = "—" if r["p_wilcoxon"] is None else f"{r['p_wilcoxon']:.3f}" + (" *" if r["p_wilcoxon"] < 0.05 else "")
        bold = "**" if name == BASELINE else ""
        lines.append(f"| {bold}{name}{bold} | {CONFIGS[name][2]} | {cells} | {delta} | {pval} |")
    return "\n".join(lines)


def doc_kind_table(runs: dict[str, dict], names: list[str]) -> str:
    kinds = ("act", "rules", "notification", "circular")
    base = runs.get(BASELINE)
    lines = ["| | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    if base:
        gold = base["doc_kind_share"]["gold_required"]
        lines.append("| **gold required law** | " + " | ".join(f"{gold.get(k, 0):.2f}" for k in kinds) + " |")
    for name in names:
        p = runs.get(name)
        if p:
            share = p["doc_kind_share"]["retrieved"]
            lines.append(f"| retrieved: {name} | " + " | ".join(f"{share.get(k, 0):.2f}" for k in kinds) + " |")
    return "\n".join(lines)


def error_taxonomy(runs: dict[str, dict]) -> str:
    """Why the baseline misses gold law: classify every missed (question, gold source) pair."""
    from app.services.gst_graph import get_graph
    from eval.gst_retrieval import source_matches
    from eval.gst_retrieval_check import doc_kind

    base, pool = runs.get(BASELINE), runs.get("hybrid_pool20")
    if base is None or pool is None:
        return "_Needs hybrid_tfidf and hybrid_pool20 results._"
    graph = get_graph()
    pool_ids = {r["id"]: r["retrieved_ids"] for r in pool["rows"]}
    counts = {"in top-20 pool, ranked 6–20 (ranking problem)": 0,
              "not in pool, but a graph neighbour of the top-5": 0,
              "not in pool or graph; top-5 dominated by notifications/circulars (crowding)": 0,
              "not found by any signal (vocabulary / query problem)": 0}
    by_kind: dict[str, int] = {}
    total = 0
    for r in full_rows(base):
        top = r["retrieved_ids"][:5]
        neighbours = set().union(*[graph.neighbours.get(p, set()) for p in top]) if top else set()
        subordinate = sum(doc_kind(t) in ("notification", "circular") for t in top)
        for gold in r["gold_required_ids"]:
            if any(source_matches(gold, t) for t in top):
                continue
            total += 1
            by_kind[doc_kind(gold)] = by_kind.get(doc_kind(gold), 0) + 1
            if any(source_matches(gold, t) for t in pool_ids.get(r["id"], [])):
                key = "in top-20 pool, ranked 6–20 (ranking problem)"
            elif any(source_matches(gold, n) for n in neighbours):
                key = "not in pool, but a graph neighbour of the top-5"
            elif subordinate >= 3 and doc_kind(gold) in ("act", "rules"):
                key = "not in pool or graph; top-5 dominated by notifications/circulars (crowding)"
            else:
                key = "not found by any signal (vocabulary / query problem)"
            counts[key] += 1
    lines = [f"{total} (question, gold source) pairs missed by the baseline top-5. Categories are checked in order.", "",
             "| Category | Missed pairs | Share |", "|---|---|---|"]
    lines += [f"| {k} | {v} | {v / total:.0%} |" for k, v in counts.items()]
    lines += ["", "Missed gold sources by document type: " +
              ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]))]
    return "\n".join(lines)


CITATION_METRICS = ("grounded_precision", "gold_citation_recall")


def _as_metric_rows(rows: list[dict], source: str) -> list[dict]:
    """Rows reshaped so eval.stats.paired() can test a per-row citation score like a RAGAS metric."""
    return [{"id": r["id"], "ragas_metrics": dict(r.get(source) or {})} for r in rows]


def _judge_label(payload: dict) -> str:
    j = payload.get("ragas_judge") or {}
    return j.get("model", "gpt-4o-mini")  # runs before the judge was recorded used gpt-4o-mini


def ragas_section() -> str:
    """RAGAS + citation table for the runs in ragas_runs.json.

    Each entry is either "path" or {"path": ..., "gpt-4o-mini": ...}; the second path (same answers, earlier
    judge) is used only for the judge-agreement table."""
    if not RAGAS_RUNS.exists():
        return "_No RAGAS runs registered yet (eval/results/thesis/ragas_runs.json)._"
    spec = {k: (v if isinstance(v, dict) else {"path": v}) for k, v in json.loads(RAGAS_RUNS.read_text(encoding="utf-8")).items()}
    loaded = {label: score_results(read_json(v["path"])) for label, v in spec.items()}
    metrics = [m for m in RAGAS_METRICS if any((p.get("aggregate") or {}).get(m) is not None for p in loaded.values())]
    judges = sorted({_judge_label(p) for p in loaded.values()})
    names = {"faithfulness": "Faithfulness", "context_precision": "Context precision",
             "context_recall": "Context recall", "answer_relevancy": "Answer relevancy"}
    lines = [f"RAGAS judge: **{', '.join(judges)}**" + (" — WARNING: mixed judges" if len(judges) > 1 else ""), "",
             "| Run | n | " + " | ".join(names[m] for m in metrics) + " | Abstained | Grounded citation precision | "
             "Answers citing law outside context | Gold law cited |",
             "|---|---|" + "---|" * len(metrics) + "---|---|---|---|"]
    for label, p in loaded.items():
        agg = p.get("aggregate") or {}
        c = p["citation_summary"]
        vals = " | ".join("—" if agg.get(m) is None else f"{agg[m]:.3f}" for m in metrics)
        lines.append(f"| {label} | {c['n_answers']} | {vals} | {c['abstained']} | {c['grounded_precision']:.3f} | "
                     f"{c['answers_citing_outside_context']} | {c['gold_citation_recall']:.3f} |")
    labels = list(loaded)
    if len(labels) > 1:
        lines += ["", f"Paired tests vs **{labels[0]}** (questions answered in both runs):", "", HEADER]
        base = loaded[labels[0]]["rows"]
        for label in labels[1:]:
            rows = loaded[label]["rows"]
            for m in metrics:
                lines.append(format_row(labels[0], label, paired(base, rows, m, coverage=None)))
            for m in CITATION_METRICS:
                lines.append(format_row(labels[0], label, paired(
                    _as_metric_rows(base, "citation"), _as_metric_rows(rows, "citation"), m, coverage=None)))
    lines += ["", judge_agreement(spec, loaded, metrics)]
    return "\n".join(lines)


def judge_agreement(spec: dict, loaded: dict, metrics: list[str]) -> str:
    """Same answers scored by two judges: means and Spearman rank correlation per metric."""
    from scipy.stats import spearmanr

    lines = ["### Judge agreement (same answers, two judges)", "",
             "| Run | Metric | n | gpt-4o-mini mean | new judge mean | Spearman ρ |", "|---|---|---|---|---|---|"]
    any_row = False
    for label, v in spec.items():
        if "gpt-4o-mini" not in v:
            continue
        old = {r["id"]: r for r in read_json(v["gpt-4o-mini"])["rows"]}
        for m in metrics:
            pairs = []
            for r in loaded[label]["rows"]:
                a = metric_value(old[r["id"]], m) if r["id"] in old else None
                b = metric_value(r, m)
                if a is not None and b is not None:
                    pairs.append((a, b))
            if len(pairs) < 3:
                continue
            xs, ys = zip(*pairs, strict=True)
            rho = spearmanr(xs, ys)[0]
            lines.append(f"| {label} | {m} | {len(pairs)} | {sum(xs) / len(xs):.3f} | {sum(ys) / len(ys):.3f} | {rho:.2f} |")
            any_row = True
    return "\n".join(lines) if any_row else ""


TIER_CONFIGS = ["dense", "sparse_bm25", "hybrid_tfidf", "hybrid_top8", "embed_large", "hyde", "rerank_llm",
                "rewrite", "decompose", "large_top8_cap3", "large_top8_cap3_decompose"]


def tier_section(runs: dict[str, dict]) -> str:
    """Required-law recall per intrinsic difficulty tier (scripts/thesis/make_question_sets.py)."""
    from scripts.thesis.make_question_sets import QUESTION_SETS, TIERS

    if not QUESTION_SETS.exists():
        return "_No question tiers yet (scripts/thesis/make_question_sets.py)._"
    qs = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))
    tier_of = {q: v["tier"] for q, v in qs["questions"].items()}
    sizes = {t: len(qs["sets"][f"tier_{t}"]) for t in TIERS}
    lines = ["Tiers are set from the question and its gold law only (number of required provisions, instruments "
             "spanned, notification/circular needed, scenario length), before any system is run.", "",
             "| Config | " + " | ".join(f"{t} (n={sizes[t]})" for t in TIERS) + " |", "|---|" + "---|" * len(TIERS)]
    for name in [n for n in TIER_CONFIGS if n in runs]:
        cells = []
        for t in TIERS:
            vals = [v for r in runs[name]["rows"] if tier_of.get(r["id"]) == t
                    and (v := metric_value(r, "required_recall")) is not None]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figures(runs: dict[str, dict]) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    names = [n for n in CONFIGS if n in runs and CONFIGS[n][0] != "diagnostic"]
    rec = [runs[n]["coverage_split"]["corrected"]["full"]["required_recall"] for n in names]
    mrr = [runs[n]["coverage_split"]["corrected"]["full"]["required_mrr"] for n in names]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(names) + 1.5))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], rec, height=0.4, label="Required law found (recall)")
    ax.barh([i - 0.2 for i in y], mrr, height=0.4, label="MRR")
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.axvline(runs[BASELINE]["coverage_split"]["corrected"]["full"]["required_recall"], ls="--", c="grey", lw=1)
    ax.set_xlim(0, 1)
    ax.set_title("Retrieval ablation — 141 fully covered GST questions (dashed: baseline recall)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrieval_ablation.png", dpi=150)
    plt.close(fig)
    made.append("retrieval_ablation.png")

    kinds = ("act", "rules", "notification", "circular")
    series = {"gold required": runs[BASELINE]["doc_kind_share"]["gold_required"],
              "retrieved (hybrid)": runs[BASELINE]["doc_kind_share"]["retrieved"]}
    for extra in ("hybrid_cap2", "hybrid_graphdense_cap2"):
        if extra in runs:
            series[f"retrieved ({extra})"] = runs[extra]["doc_kind_share"]["retrieved"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(series)
    for i, (label, share) in enumerate(series.items()):
        ax.bar([k + i * width for k in range(len(kinds))], [share.get(k, 0) for k in kinds], width, label=label)
    ax.set_xticks([k + width * (len(series) - 1) / 2 for k in range(len(kinds))], kinds)
    ax.set_ylabel("share of items")
    ax.set_title("Document types: gold law vs retrieved top-5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "doc_type_share.png", dpi=150)
    plt.close(fig)
    made.append("doc_type_share.png")
    return made


def main() -> None:
    runs = {n: p for n in CONFIGS if (p := load(n)) is not None}
    parts = ["# Thesis results (generated by scripts/thesis/make_report.py)", "",
             f"Retrieval: 141 fully covered questions, corrected gold sources, top-5 unless stated. Baseline: "
             f"**{BASELINE}** (the system default). Δ = paired difference in required-law recall vs the baseline, "
             "95% bootstrap CI; p = Wilcoxon signed-rank (* < 0.05).", ""]
    for title, names in TABLES:
        parts += [f"## {title}", "", ablation_table(names, runs), ""]
    parts += ["## Document-type share of gold law vs retrieved items", "",
              doc_kind_table(runs, ["hybrid_tfidf", "dense", "hybrid_cap2", "hybrid_graphdense_cap2"]), "",
              "## Required-law recall by question difficulty", "", tier_section(runs), "",
              "## Error analysis — why the baseline misses gold law", "", error_taxonomy(runs), "",
              "## Answer quality (RAGAS) and citation correctness", "", ragas_section(), ""]
    if BASELINE in runs:
        parts += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in figures(runs)]
    out = THESIS_DIR / "results_tables.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
