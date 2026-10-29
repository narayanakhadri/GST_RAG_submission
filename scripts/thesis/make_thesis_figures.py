"""Figures for the thesis chapters (data, method diagrams, results), drawn from the corpus, the golden set and the
saved experiment JSONs — never typed by hand. Complements scripts/thesis/make_report.py, whose two figures are reused.

    uv run python -m scripts.thesis.make_thesis_figures

Writes PNGs to eval/results/thesis/figures/thesis/ and a list of them (with the caption each should get) to
eval/results/thesis/figures/thesis/figures.json, plus data tables in data_tables.md; scripts/thesis/build_thesis.py
reads both.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from eval.gst_retrieval import source_matches  # noqa: E402
from eval.stats import metric_value, paired  # noqa: E402
from scripts.thesis.make_report import BASELINE, load  # noqa: E402

CORPUS = Path("seed/docs/gst/gst_provisions_enriched_groq.jsonl")
GOLDEN = Path("seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl")
QUESTION_SETS = Path("eval/results/thesis/question_sets.json")
TECH = Path("eval/results/thesis/techniques")
RESULTS = Path("eval/results")
OUT = Path("eval/results/thesis/figures/thesis")
KINDS = ("act", "rules", "notification", "circular")
COLORS = {"act": "#3b6ea8", "rules": "#5aa469", "notification": "#e0a030", "circular": "#c05050"}
TIERS = ("easy", "medium", "hard", "very_hard")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

made: list[dict] = []


def kind_of(provision_or_instrument_id: str) -> str:
    inst = provision_or_instrument_id.split(".")[0]
    if inst.startswith("notification_"):
        return "notification"
    if inst.startswith("circular_"):
        return "circular"
    return "rules" if inst.endswith("rules_2017") else "act"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig, name: str, caption: str, chapter: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)
    made.append({"id": name, "file": str(OUT / f"{name}.png"), "caption": caption, "chapter": chapter})
    print(f"  {name}")


def recall(rows: list[dict], metric: str = "required_recall") -> float:
    vals = [v for r in rows if (v := metric_value(r, metric)) is not None]
    return sum(vals) / len(vals)


def full(rows: list[dict]) -> list[dict]:
    return [r for r in rows if (r.get("coverage") or {}).get("corrected") == "full"]


def tech(name: str) -> list[dict] | None:
    path = TECH / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else None


# --- diagrams ---------------------------------------------------------------------------------------------------
def flow(name: str, caption: str, chapter: str, steps: list[tuple[str, str]], cols: int, size: tuple[float, float],
         arrows: list[tuple[int, int]] | None = None) -> None:
    """Boxes laid out row by row (snake order), joined by arrows; steps = (title, detail)."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_axis_off()
    rows = (len(steps) + cols - 1) // cols
    w, h = 0.8 / cols, 0.55 / rows
    centers = []
    for i, (title, detail) in enumerate(steps):
        r, c = divmod(i, cols)
        if r % 2:
            c = cols - 1 - c
        x, y = (c + 0.5) / cols, 1 - (r + 0.5) / rows
        centers.append((x, y))
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01", fc="#eef3fa",
                                    ec="#3b6ea8", lw=1.2, transform=ax.transAxes))
        ax.text(x, y + h * 0.18, title, ha="center", va="center", weight="bold", fontsize=9.5, transform=ax.transAxes)
        ax.text(x, y - h * 0.18, detail, ha="center", va="center", fontsize=8, transform=ax.transAxes, wrap=True)
    for a, b in arrows or [(i, i + 1) for i in range(len(steps) - 1)]:
        (x1, y1), (x2, y2) = centers[a], centers[b]
        if abs(y1 - y2) < 1e-6:
            x1, x2 = x1 + (w / 2 if x2 > x1 else -w / 2), x2 - (w / 2 if x2 > x1 else -w / 2)
        else:
            y1, y2 = y1 - h / 2, y2 + h / 2
        ax.annotate("", (x2, y2), (x1, y1), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "-|>", "color": "#555", "lw": 1.2})
    save(fig, name, caption, chapter)


def diagrams() -> None:
    flow("research_approach", "Research approach: from data collection to staged evaluation", "3", [
        ("1. Data collection", "Bare law (ICAI, Apr 2026)\n24 notifications/circulars\n146 ICAI exam questions"),
        ("2. Corpus building", "one record per section/rule/\nentry/paragraph (881)"),
        ("3. Enrichment", "LLM summary, topics, keywords\n+ cross-references"),
        ("4. Gold annotation", "required law per question\ncorrected + expert-reviewed"),
        ("5. Baseline system", "hybrid dense + TF-IDF, RRF\ntop-5"),
        ("6. Ablations", "one technique at a time\nretrieval metrics, paired tests"),
        ("7. Staged gates", "dev_hard (35) → test_hard (35)\n→ all 141/146"),
        ("8. Answer evaluation", "RAGAS judge, citations,\nfinal amount, expert review"),
    ], cols=4, size=(11, 4.2))
    flow("evaluation_framework", "Evaluation framework: how each technique is measured and admitted", "3", [
        ("Gold sources", "required provisions per question\n(corrected, expert-reviewed)"),
        ("Retrieval run", "named configuration,\nflags + config hash saved"),
        ("Retrieval metrics", "required-law recall, all-required,\nhit@k, MRR"),
        ("Paired statistics", "bootstrap 95% CI (10,000)\n+ Wilcoxon signed-rank"),
        ("Gate 1: dev_hard", "35 hard questions\n(ideas tried here)"),
        ("Gate 2: test_hard", "35 held-out hard questions\n(confirmation)"),
        ("All questions", "141 fully covered\n(reported numbers)"),
        ("Answer evaluation", "RAGAS judge, citations,\nfinal amount, expert review"),
    ], cols=4, size=(11, 4.2))
    flow("baseline_pipeline", "Baseline retrieval pipeline (hybrid dense + TF-IDF with RRF)", "3", [
        ("Question", "user question\n(exam scenario)"),
        ("Two searches", "dense: 3-small, cosine\nsparse: TF-IDF"),
        ("Rank fusion", "Reciprocal Rank\nFusion, k = 60"),
        ("Top 5", "provisions passed\nto the generator"),
        ("Answer", "grounded prompt\nwith citations"),
    ], cols=5, size=(11, 2.2))
    flow("system_architecture", "System architecture: request flow through the FastAPI service", "4", [
        ("Client", "Streamlit UI / REST\n(JWT login)"),
        ("Security layers", "rate limit, token budget,\ninjection scan, PII redaction"),
        ("LangGraph", "route_intent → retrieve_rag\n→ generate_answer → finalize"),
        ("Retrieval", "Qdrant dense + TF-IDF sparse\nRRF, re-rank, caps"),
        ("Generation", "Gemini 3.1 flash-lite\ngrounded prompt, citations"),
        ("Output checks", "moderation, PII redaction\n→ answer + sources"),
    ], cols=3, size=(11, 3.8))
    flow("corpus_pipeline", "Corpus construction pipeline (scripts/gst_pipeline/)", "4", [
        ("parse_bare_law.py", "ICAI bare-law markdown →\nsections, rules, schedules"),
        ("fetch_sources.py", "GST Council PDFs →\nDocling markdown"),
        ("parse_notifications.py", "table rows → entries\nnumbered paras → paragraphs"),
        ("enrich.py", "LLM metadata +\nretrieval_text_enriched"),
        ("build_corpus.py", "validate IDs, labels\n→ 881-record JSONL"),
        ("seed_gst.py", "embed (OpenAI) →\nQdrant, 866 points"),
    ], cols=3, size=(11, 3.8))
    flow("final_pipeline", "Final retrieval and answer pipeline (E17 + T31)", "4", [
        ("Question", "long exam scenario\n(median 112 words)"),
        ("Decomposition", "LLM writes one sub-query\nper legal issue"),
        ("Hybrid search", "text-embedding-3-large +\nTF-IDF per query, RRF"),
        ("Pool of 20", "fused candidates\n(holds 0.73 of gold law)"),
        ("T31 re-rank", "LLM analyses each candidate,\ngrades 0–3"),
        ("Top 8, cap 3", "max 3 notification/\ncircular entries"),
        ("Grounded answer", "law + summary context,\ncite every provision"),
    ], cols=4, size=(11, 4.2))


# --- data (chapter 4) -------------------------------------------------------------------------------------------
def data_figures(corpus: list[dict], golden: list[dict], qsets: dict) -> None:
    by_inst = Counter(r["instrument_id"] for r in corpus)
    big = [(k, v) for k, v in by_inst.most_common() if v >= 9]
    rest = sum(v for k, v in by_inst.items() if v < 9)
    labels = [k.replace("_2017", "").replace("notification_", "notif. ").replace("circular_", "circ. ") for k, _ in big]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(labels + [f"{sum(1 for v in by_inst.values() if v < 9)} other instruments"], [v for _, v in big] + [rest],
            color=[COLORS[kind_of(k)] for k, _ in big] + ["#999"])
    ax.invert_yaxis()
    ax.set_xlabel("records")
    ax.set_title(f"Corpus records by instrument (n = {len(corpus)}, {len(by_inst)} instruments)")
    save(fig, "corpus_by_instrument", "Corpus records by instrument, coloured by document type", "4")

    kinds = Counter(kind_of(r["id"]) for r in corpus)
    ptypes = Counter(r["provision_type"] for r in corpus)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.bar(KINDS, [kinds[k] for k in KINDS], color=[COLORS[k] for k in KINDS])
    a1.set_title("Records by document type")
    for i, k in enumerate(KINDS):
        a1.text(i, kinds[k], str(kinds[k]), ha="center", va="bottom")
    a2.bar(list(ptypes), list(ptypes.values()), color="#777")
    a2.set_title("Records by unit of segmentation")
    for i, v in enumerate(ptypes.values()):
        a2.text(i, v, str(v), ha="center", va="bottom")
    save(fig, "corpus_by_type", "Corpus records by document type and by segmentation unit", "4")

    fig, ax = plt.subplots(figsize=(9, 4))
    data = [[r["word_count"] for r in corpus if kind_of(r["id"]) == k and r["status"] != "omitted"] for k in KINDS]
    ax.boxplot(data, tick_labels=[f"{k}\n(median {statistics.median(d):.0f})" for k, d in zip(KINDS, data, strict=True)],
               showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("words per record (log scale)")
    ax.set_title("Length of corpus records by document type")
    save(fig, "record_length", "Length of corpus records (words) by document type", "4")

    raw = [len(r["raw_text"].split()) for r in corpus if r["status"] != "omitted"]
    enr = [len(r["retrieval_text_enriched"].split()) for r in corpus if r["status"] != "omitted"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    ax.hist([raw, enr], bins=bins, label=["raw provision text", "enriched text (embedded)"], color=["#999", "#3b6ea8"])
    ax.set_xscale("log")
    ax.set_xlabel("words (log scale)")
    ax.set_ylabel("records")
    ax.legend()
    ax.set_title(f"Raw vs enriched text: median {statistics.median(raw):.0f} → {statistics.median(enr):.0f} words")
    save(fig, "enrichment_length", "Effect of LLM enrichment on the length of the embedded text", "4")

    refs = [sum(len(v) for v in r["explicit_references"].values()) for r in corpus]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    cap = 15
    counts = Counter(min(n, cap) for n in refs)
    ax.bar(range(cap + 1), [counts[i] for i in range(cap + 1)], color="#5aa469")
    ax.set_xticks(range(cap + 1), [str(i) for i in range(cap)] + [f"{cap}+"])
    ax.set_xlabel("explicit cross-references in the record")
    ax.set_ylabel("records")
    ax.set_title(f"Cross-references per record ({sum(1 for n in refs if n)} of {len(refs)} records cite other law)")
    save(fig, "cross_references", "Explicit cross-references per corpus record (basis of the provision graph)", "4")

    qwords = [len(g["question"].split()) for g in golden]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(qwords, bins=25, color="#3b6ea8")
    ax.axvline(statistics.median(qwords), ls="--", c="k", lw=1)
    ax.set_xlabel("words in the question")
    ax.set_ylabel("questions")
    ax.set_title(f"Question length (n = {len(golden)}, median {statistics.median(qwords):.0f} words)")
    save(fig, "question_length", "Length of the ICAI evaluation questions", "4")

    nreq = [len(g["primary_required_sources"]) for g in golden]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    c = Counter(nreq)
    a1.bar(sorted(c), [c[k] for k in sorted(c)], color="#3b6ea8")
    a1.xaxis.set_major_locator(MaxNLocator(integer=True))
    a1.set_xlabel("required provisions per question")
    a1.set_ylabel("questions")
    a1.set_title(f"Required law per question (mean {statistics.mean(nreq):.1f})")
    gk = Counter(kind_of(s) for g in golden for s in g["primary_required_sources"])
    a2.pie([gk[k] for k in KINDS], labels=[f"{k} ({gk[k]})" for k in KINDS], colors=[COLORS[k] for k in KINDS],
           autopct="%1.0f%%", startangle=90)
    a2.set_title("Document type of the required law")
    save(fig, "gold_sources", "Gold (required) sources per question and their document types", "4")

    q = qsets["questions"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    no = [sum(1 for v in q.values() if v["tier"] == t and not v["needs_notification_or_circular"]) for t in TIERS]
    yes = [sum(1 for v in q.values() if v["tier"] == t and v["needs_notification_or_circular"]) for t in TIERS]
    ax.bar(TIERS, no, label="Acts/Rules only", color="#3b6ea8")
    ax.bar(TIERS, yes, bottom=no, label="needs a notification or circular", color="#e0a030")
    ax.set_ylabel("questions")
    ax.legend()
    ax.set_title("Difficulty tiers (set from the question and its gold law, before any run)")
    save(fig, "difficulty_tiers", "Questions by difficulty tier and by the kind of law they need", "4")

    units = Counter(re.sub(r"^\d+bos-aps2102-", "", g["source_file"]).removesuffix(".md") for g in golden)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(list(units), list(units.values()), color="#777")
    ax.set_xlabel("ICAI study-material unit (source file)")
    ax.set_ylabel("questions")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Where the evaluation questions come from")
    save(fig, "question_sources", "Evaluation questions by ICAI study-material unit", "4")


# --- results (chapter 5) ----------------------------------------------------------------------------------------
def results_figures(golden: list[dict]) -> None:
    for f in ("retrieval_ablation", "doc_type_share"):
        src = Path("eval/results/thesis/figures") / f"{f}.png"
        if src.exists():
            made.append({"id": f, "file": str(src), "chapter": "5", "caption": {
                "retrieval_ablation": "Required-law recall and MRR of every retrieval configuration (141 questions)",
                "doc_type_share": "Document types of the gold law and of the retrieved top 5"}[f]})

    steps = [("Baseline\nhybrid top-5", "hybrid_tfidf"), ("+ 3-large\nembeddings", "embed_large"),
             ("+ top-8,\ncap 3", "large_top8_cap3"), ("+ decomposition\n(E17)", "large_top8_cap3_decompose"),
             ("E17, fixed\nquestion text", "e17_v3_textfix")]
    rec, mrr, labels = [], [], []
    for label, name in steps:
        if (p := load(name)) is not None:
            s = p["coverage_split"]["corrected"]["full"]
            rec.append(s["required_recall"])
            mrr.append(s["required_mrr"])
            labels.append(label)
    t31 = tech("t31_pointwise_on_all")
    if t31:
        labels.append("+ T31 re-rank\n(final)")
        rec.append(recall(full(t31)))
        mrr.append(recall(full(t31), "required_mrr"))
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(labels))
    ax.plot(x, rec, "o-", label="required-law recall", c="#3b6ea8")
    ax.plot(x, mrr, "s-", label="MRR", c="#c05050")
    for i in x:
        ax.text(i, rec[i] - 0.05, f"{rec[i]:.3f}", ha="center", fontsize=8.5, c="#3b6ea8")
        ax.text(i, mrr[i] + 0.025, f"{mrr[i]:.3f}", ha="center", fontsize=8.5, c="#c05050")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper left")
    ax.set_title("From the baseline to the final system (fully covered questions)")
    save(fig, "progression", "Step-by-step improvement from the baseline to the final system", "5")

    base = load(BASELINE)
    configs = [("dense", "Dense only"), ("sparse_bm25", "Sparse BM25"), ("hybrid_top8", "Hybrid top-8"),
               ("embed_raw", "Embed raw text"), ("embed_large", "3-large embeddings"), ("hybrid_cap2", "Cap 2"),
               ("hybrid_graph", "Graph expansion"), ("hybrid_graphdense_w1", "Graph (similarity)"),
               ("rerank_minilm", "MiniLM cross-encoder"), ("rerank_llm", "LLM listwise rerank"), ("hyde", "HyDE"),
               ("rewrite", "LLM rewrite"), ("decompose", "LLM decomposition"),
               ("large_top8_cap3_decompose", "E17 (combined)")]
    forest(base["rows"], [(label, load(n)["rows"]) for n, label in configs if load(n)], "forest_vs_baseline",
           "Change in required-law recall against the baseline, paired, with 95% bootstrap CI",
           "Δ required-law recall vs baseline (hybrid top-5)", coverage="full")

    pairs = [("T09 learning-to-rank", "t09_ltr"), ("T12 citation-guided", "t12_citation_guided"),
             ("T13 query routing", "t13_routing_w0.5"), ("T17 hierarchical summaries", "t17_hier_fuse_c3"),
             ("T19 ColBERT fusion", "t19_colbert_small_fuse"), ("T22 Personalized PageRank", "t22_ppr_a0.5_w0.5"),
             ("T27 GraphRAG communities", "t27_graphrag_c3"), ("T30 legal concepts", "t30_concepts_bm25_w0.7"),
             ("T32 HyDE in fusion", "t32_hyde_fusion_w1.0"), ("T31 re-rank, no reasoning", "t31_pointwise_off"),
             ("T31 re-rank with reasoning", "t31_pointwise_on")]
    ref = load("e17_v3_textfix")
    if ref:
        for split in ("dev_hard", "test_hard"):
            items = [(label, rows) for label, n in pairs if (rows := tech(f"{n}_{split}"))]
            forest(ref["rows"], items, f"forest_techniques_{split}",
                   f"Techniques on top of E17, {split} (35 questions): change in required-law recall, 95% CI",
                   f"Δ required-law recall vs E17 ({split})", coverage=None)

    if base:
        tiers = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))["questions"]
        names = [("hybrid_tfidf", "Baseline"), ("embed_large", "3-large"), ("decompose", "Decomposition"),
                 ("large_top8_cap3_decompose", "E17")]
        series = [(label, load(n)["rows"]) for n, label in names if load(n)]
        if t31:
            series.append(("Final (T31)", t31))
        fig, ax = plt.subplots(figsize=(9, 4))
        width = 0.8 / len(series)
        for i, (label, rows) in enumerate(series):
            vals = [recall([r for r in full(rows) if tiers.get(r["id"], {}).get("tier") == t]) for t in TIERS]
            ax.bar([k + i * width for k in range(len(TIERS))], vals, width, label=label)
        ax.set_xticks([k + width * (len(series) - 1) / 2 for k in range(len(TIERS))], TIERS)
        ax.set_ylabel("required-law recall")
        ax.set_ylim(0, 1)
        ax.legend(ncol=3, fontsize=8.5)
        ax.set_title("Required-law recall by difficulty tier")
        save(fig, "recall_by_tier", "Required-law recall by question difficulty tier", "5")

    errors = [("ranked 6–20\n(ranking)", 65), ("graph neighbour\nof the top-5", 115), ("crowded out by\nnotifications", 68),
              ("no signal\n(vocabulary)", 114)]
    errors = error_buckets(base) or errors
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([e[0] for e in errors], [e[1] for e in errors], color=["#3b6ea8", "#5aa469", "#e0a030", "#c05050"])
    total = sum(e[1] for e in errors)
    for i, (_, v) in enumerate(errors):
        ax.text(i, v, f"{v} ({v / total:.0%})", ha="center", va="bottom")
    ax.set_ylabel("missed (question, law) pairs")
    ax.set_title(f"Why the baseline misses required law ({total} missed pairs)")
    save(fig, "error_taxonomy", "Error analysis: why the baseline top 5 misses required law", "5")

    pool = tech("e17_pool20_all")
    if pool:
        gold = {g["id"]: g["primary_required_sources"] for g in golden}
        rows = [r for r in full(pool) if gold.get(r["id"])]
        ks = range(1, 21)
        curve = [statistics.mean(sum(any(source_matches(s, i) for i in r["retrieved_ids"][:k]) for s in gold[r["id"]])
                                 / len(gold[r["id"]]) for r in rows) for k in ks]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(list(ks), curve, "o-", ms=3)
        ax.set_xticks(list(ks))
        ax.axvline(8, ls="--", c="grey", lw=1)
        ax.text(8.3, curve[0], "final top-k = 8", fontsize=8.5)
        ax.set_xlabel("k (candidates kept, E17 fused ranking)")
        ax.set_ylabel("required-law recall@k")
        ax.set_title(f"How much required law the candidate pool holds (recall@20 = {curve[-1]:.3f})")
        save(fig, "recall_at_k", "Required-law recall at k over the 20-candidate pool (E17 ranking)", "5")

    t34 = [(s, tech(f"t34_missing_law_{s}"), tech(f"t34_missing_law_control_{s}"), tech(f"t31_pointwise_on_{s}"))
           for s in ("dev_hard", "test_hard")]
    if all(a and b and c for _, a, b, c in t34):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.27
        for i, (label, idx) in enumerate([("final system (T31)", 3), ("control: extra slots", 2), ("T34 missing-law", 1)]):
            vals = [recall(t[idx]) for t in t34]
            ax.bar([k + i * width for k in range(2)], vals, width, label=label)
            for k, v in enumerate(vals):
                ax.text(k + i * width, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks([k + width for k in range(2)], ["dev_hard", "test_hard"])
        ax.set_ylabel("required-law recall")
        ax.set_ylim(0, 0.8)
        ax.legend(fontsize=8.5)
        ax.set_title("T34: fetching the law the answer plan names")
        save(fig, "t34_recall", "T34 missing-law check: retrieval gain against the final system and a control", "5")

    judge_scatter()
    amount_figure()


def error_buckets(base: dict | None) -> list[tuple[str, int]] | None:
    """Re-read the counts from the generated results table so the figure and the table cannot disagree."""
    table = Path("eval/results/thesis/results_tables.md").read_text(encoding="utf-8")
    m = re.findall(r"\| ([^|]+) \| (\d+) \| \d+% \|", table)
    if len(m) != 4:
        return None
    short = ["ranked 6–20\n(ranking)", "graph neighbour\nof the top-5", "crowded out by\nnotifications",
             "no signal\n(vocabulary)"]
    return [(s, int(n)) for s, (_, n) in zip(short, m, strict=True)]


def forest(ref_rows: list[dict], items: list[tuple[str, list[dict]]], name: str, caption: str, xlabel: str,
           coverage: str | None) -> None:
    stats = [(label, paired(ref_rows, rows, "required_recall", coverage=coverage)) for label, rows in items]
    stats = [(label, s) for label, s in stats if s.get("n")]
    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(stats) + 1.4))
    for i, (_label, s) in enumerate(stats):
        sig = s["p_wilcoxon"] is not None and s["p_wilcoxon"] < 0.05
        col = ("#3b6ea8" if s["diff"] > 0 else "#c05050") if sig else "#888"
        ax.plot([s["ci_low"], s["ci_high"]], [i, i], c=col, lw=2)
        ax.plot(s["diff"], i, "o", c=col)
        ax.text(max(s["ci_high"], 0) + 0.01, i, f"{s['diff']:+.3f}" + (" *" if sig else ""), va="center", fontsize=8)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(range(len(stats)), [label for label, _ in stats])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel + "   (* p < 0.05, Wilcoxon)")
    save(fig, name, caption, "5")


def judge_scatter() -> None:
    a = RESULTS / "20261016T191402Z_gst_hybrid.json"
    b = RESULTS / "20261016T191402Z_gst_hybrid.judge-qwen-qwen3-30b-a3b-instruct-2507.json"
    if not (a.exists() and b.exists()):
        return
    ra = {r["id"]: r for r in json.loads(a.read_text(encoding="utf-8"))["rows"]}
    rb = {r["id"]: r for r in json.loads(b.read_text(encoding="utf-8"))["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("faithfulness", "answer_relevancy"), strict=True):
        pts = [(va, vb) for i in ra if i in rb and (va := metric_value(ra[i], metric)) is not None
               and (vb := metric_value(rb[i], metric)) is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, alpha=0.6)
        ax.plot([0, 1], [0, 1], c="grey", lw=0.8, ls="--")
        ax.set_xlabel("gpt-4o-mini judge")
        ax.set_ylabel("Qwen3-30B judge")
        ax.set_title(f"{metric} (n = {len(pts)})")
    save(fig, "judge_agreement", "Two LLM judges scoring the same baseline answers", "5")


def amount_figure() -> None:
    path = Path("eval/results/thesis/amount_match.json")
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    fa, sme = d["final_all"], d["sme_agreement"]
    labels = ["amounts reached\n(exact)", "amounts reached\n(within 5%)", "final amount\nreached"]
    keys = ["amount_recall", "amount_recall_5pct", "last_amount_hit"]
    series = [(f"final system, all ({d['scored']['final']} q)", fa)] + [
        (f"{k} ({v['n']} q)", v) for k, v in sme.items()]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / len(series)
    for i, (label, s) in enumerate(series):
        vals = [s[k] for k in keys]
        ax.bar([k + i * width for k in range(3)], vals, width, label=label)
        for k, v in enumerate(vals):
            ax.text(k + i * width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([k + width * (len(series) - 1) / 2 for k in range(3)], labels)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8.5)
    ax.set_title("Final-amount check on numerical questions")
    save(fig, "amount_match", "Deterministic final-amount check, overall and split by the expert's verdict", "5")


def data_tables(corpus: list[dict], golden: list[dict]) -> None:
    lines = ["# Data tables (generated by scripts/thesis/make_thesis_figures.py)", "", "## Corpus records by document type",
             "", "| Document type | Records | Indexed (not omitted) | Median words (indexed) |", "|---|---|---|---|"]
    names = {"act": "Act sections and schedules", "rules": "Rules", "notification": "Notification entries/paragraphs",
             "circular": "Circular paragraphs"}
    for k in KINDS:
        rows = [r for r in corpus if kind_of(r["id"]) == k]
        live = [r for r in rows if r["status"] != "omitted"]
        lines.append(f"| {names[k]} | {len(rows)} | {len(live)} | {statistics.median(r['word_count'] for r in live):.0f} |")
    live = [r for r in corpus if r["status"] != "omitted"]
    lines += [f"| Total ({len({r['instrument_id'] for r in corpus})} instruments) | {len(corpus)} | {len(live)} | "
              f"{statistics.median(r['word_count'] for r in live):.0f} |", "",
              "## Evaluation set summary", "", "| Measure | Value |", "|---|---|",
              f"| Questions | {len(golden)} |",
              f"| Median question length (words) | {statistics.median(len(g['question'].split()) for g in golden):.0f} |",
              f"| Required provisions per question (mean / median) | "
              f"{statistics.mean(len(g['primary_required_sources']) for g in golden):.1f} / "
              f"{statistics.median(len(g['primary_required_sources']) for g in golden):.0f} |",
              f"| Questions needing a notification or circular | "
              f"{sum(any(kind_of(s) in ('notification', 'circular') for s in g['primary_required_sources']) for g in golden)} |",
              f"| ICAI source units | {len({g['source_file'] for g in golden})} |", ""]
    (OUT / "data_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    corpus = jsonl(CORPUS)
    golden = jsonl(GOLDEN)
    qsets = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))
    diagrams()
    data_figures(corpus, golden, qsets)
    results_figures(golden)
    data_tables(corpus, golden)
    (OUT / "figures.json").write_text(json.dumps(made, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(made)} figures → {OUT}")


if __name__ == "__main__":
    main()
