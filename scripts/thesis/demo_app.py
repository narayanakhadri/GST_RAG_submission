"""Thesis demo: the final system (E17 → T31 reasoning re-rank → grounded answer) on one golden question at a time.

Pick a question, see the system's answer beside the gold answer, the required-law recall / MRR against the corrected
gold sources, and the RAGAS scores (Qwen judge) of the reported run. "Run live" re-runs the whole pipeline and shows
each step: decomposed sub-queries, the 20-candidate pool with T31 grades, the final 8 provisions, the answer.

    docker compose up -d qdrant
    uv run python -m scripts.thesis.build_collections --only gst_thesis_enriched_large   # once: 3-large embeddings
    uv run streamlit run scripts/thesis/demo_app.py

Without Qdrant the page still works on stored answers; only "Run live" needs it.

Live runs use Gemini (forced here) and the on-disk caches, so golden questions are mostly cache hits; a typed question
needs one OpenAI embedding call per (sub-)query. Live answers are not judged: RAGAS scores come from the reported run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["LLM_PROVIDER"] = "gemini"  # never the paid default (.env says openai)
os.environ["UPSTASH_REDIS_URL"] = ""  # dead host; in-memory cache instead
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402

RUNS = json.loads(Path("eval/results/thesis/ragas_runs.json").read_text(encoding="utf-8"))
FINAL = RUNS["final system (E17 + T31 + grounded, law+summary), all 146"]
QCTX = RUNS["final system, question in the judge's context (faithfulness), 79 SME q"]["path"]
CLOSED = RUNS["closed book (no retrieval, correctness), 79 SME q"]["path"]
RETRIEVAL = Path("eval/results/thesis/techniques/t31_pointwise_on_all.json")


def rows_by_id(path: str | Path) -> dict[str, dict]:
    # read_json falls back to the gzipped copy in the archive
    from scripts.thesis.archive_results import read_json

    return {r["id"]: r for r in read_json(path)["rows"]}


@st.cache_resource(show_spinner="Loading golden set, stored runs and corpus…")
def load() -> dict:
    from app.datasets.gst.golden_loader import load_golden_set
    from app.services.gst_graph import get_graph
    from scripts.techniques.common import GOLDEN, QUESTION_SETS, use_best_stack

    use_best_stack()
    sets = json.loads(QUESTION_SETS.read_text(encoding="utf-8"))["sets"]
    judged: dict[str, dict] = {}
    for p in FINAL["paths"]:
        judged |= rows_by_id(p)
    qctx = rows_by_id(QCTX)  # its rows are the 79 SME questions
    return {"golden": {g["id"]: g for g in load_golden_set(GOLDEN)}, "graph": get_graph(),
            "sets": {"dev_hard": sets["dev_hard"], "test_hard": sets["test_hard"], "SME 79": sorted(qctx)},
            "answers": rows_by_id(FINAL["answers"]), "judged": judged, "qctx": qctx,
            "closed": rows_by_id(CLOSED), "retrieval": rows_by_id(RETRIEVAL)}


def label(pid: str, graph) -> str:
    r = graph.records.get(pid)
    if not r:
        return f"{pid} (whole instrument)"
    return f"{r['instrument_title']} — {r['provision_type']} {r['provision_number']}: {r.get('provision_title') or ''}"


def gold(g: dict) -> tuple[list[str], list[str], list[str]]:
    from eval.gst_retrieval import gold_sets

    return gold_sets(g, "corrected")


def run_live(question: str, data: dict) -> dict:
    """Same steps as scripts/techniques/t31_llm_pointwise_rerank.retrieve + final_system.answer, kept apart so the
    page can show them."""
    from app.services.gst_graph import cap_doc_types
    from app.services.query_rewrite import rewrite_queries
    from app.services.rag_service import _retrieve
    from scripts.techniques.common import E17_FLAGS, grounded_answer
    from scripts.techniques.t31_llm_pointwise_rerank import grades

    graph, t0, steps = data["graph"], time.perf_counter(), {}
    with st.status("Running the final system…", expanded=True) as status:
        st.write("1. Decomposing the question into sub-queries (E17)")
        steps["subqueries"] = rewrite_queries(question, "decompose")
        st.write("2. Hybrid search per query, RRF fusion → 20-candidate pool")
        pool = _retrieve(question, {**E17_FLAGS, "top_k": 20, "doc_type_cap": None, "context_text": "law+summary"})
        st.write("3. T31: LLM grades every candidate 0–3, with reasoning")
        g = grades(question, pool, reasoning=True)
        order = sorted(range(len(pool)), key=lambda i: (-g[i], i))
        final = cap_doc_types([pool[i] for i in order], graph, 8, 3)
        st.write("4. Grounded answer over the final 8 provisions (law + summary)")
        answer = grounded_answer(question, final)
        status.update(label=f"Done in {time.perf_counter() - t0:.1f} s", state="complete")
    grade_of = {graph.id_of(c): g[i] for i, c in enumerate(pool)}
    steps["pool"] = [(graph.id_of(c), g[i]) for i, c in enumerate(pool)]
    steps["final_ids"] = [i for i in (graph.id_of(c) for c in final) if i]
    steps["grade_of"] = grade_of
    steps["answer"] = answer
    steps["seconds"] = time.perf_counter() - t0
    return steps


def metric(col, name: str, value, help_text: str = "") -> None:
    col.metric(name, "—" if value is None or value != value else f"{value:.2f}", help=help_text or None)


def main() -> None:
    st.set_page_config(page_title="GST RAG — thesis demo", layout="wide")
    data = load()
    golden, graph = data["golden"], data["graph"]

    with st.sidebar:
        st.header("Question")
        source = st.radio("Pick from", ["dev_hard", "test_hard", "SME 79", "all 146", "type my own"])
        if source == "type my own":
            qid, g = None, None
            question = st.text_area("Your GST question", height=160)
        else:
            ids = list(golden) if source == "all 146" else [i for i in data["sets"][source] if i in golden]
            qid = st.selectbox(f"{len(ids)} questions", ids,
                               format_func=lambda i: f"{i.split('_')[-1]} · {golden[i]['question'][:70]}…")
            g = golden[qid]
            question = g["question"]
        live = st.button("Run live", type="primary", disabled=not question.strip(),
                         help="Re-run the whole pipeline now (Gemini, cached). Otherwise the reported run is shown.")
        st.caption("Final system: E17 (decompose + hybrid + RRF, cap 3) → T31 LLM re-rank with reasoning → "
                   "grounded answer over law + summary. Judge: Qwen3-30B via OpenRouter.")

    st.title("GST law RAG — final system demo")
    st.markdown(f"**Question** {'`' + qid + '`' if qid else ''}")
    st.info(question or "Type a question in the sidebar.")
    if not question.strip():
        return

    key = f"live::{question}"
    if live:
        try:
            st.session_state[key] = run_live(question, data)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live run failed ({type(exc).__name__}: {exc}). Is Qdrant up? Showing the reported run instead.")
    result = st.session_state.get(key)
    stored = data["answers"].get(qid) if qid else None
    if result:
        answer, final_ids, where = result["answer"], result["final_ids"], f"live run, {result['seconds']:.1f} s"
    elif stored:
        answer, final_ids, where = stored["answer"], stored["retrieved_ids"], "reported run (2026-10-20)"
    else:
        st.warning("No stored answer for a typed question: press **Run live**.")
        return

    primary, supporting, missing = gold(g) if g else ([], [], [])
    if primary:
        from eval.gst_retrieval import retrieval_metrics

        m = retrieval_metrics(final_ids, primary, supporting)
        judged = data["judged"].get(qid, {}).get("ragas_metrics", {})
        qctx = data["qctx"].get(qid, {}).get("ragas_metrics", {})
        st.subheader("Scores for this question")
        c = st.columns(6)
        metric(c[0], "Required-law recall", m["required_recall"], "Share of the required provisions in the final 8")
        metric(c[1], "All required found", m["all_required_hit"])
        metric(c[2], "MRR", m["required_mrr"], "1 / rank of the first required provision")
        metric(c[3], "Correctness", judged.get("answer_correctness"), "RAGAS answer correctness vs gold answer")
        metric(c[4], "Faithfulness", judged.get("faithfulness"), "RAGAS: claims supported by the retrieved law")
        metric(c[5], "Faithfulness (q-aware)", qctx.get("faithfulness"),
               "Judge also sees the question's case facts (79 SME questions only)")
        st.caption("Recall and MRR are computed for the answer shown. RAGAS scores belong to the reported run"
                   + (" — the live answer is not judged." if result else "."))

    left, right = st.columns(2)
    with left:
        st.subheader("System answer")
        st.caption(where)
        st.markdown(answer)
    with right:
        st.subheader("Gold answer")
        st.caption("Reference answer from the golden set")
        st.markdown(g["reference_answer"] if g else "_No gold answer for a typed question._")

    st.subheader("Law used")
    rows = []
    for rank, pid in enumerate(final_ids, start=1):
        from eval.gst_retrieval import source_matches

        row = {"rank": rank, "provision": label(pid, graph),
               "required": "yes" if any(source_matches(s, pid) for s in primary) else ""}
        if result:
            row["T31 grade"] = result["grade_of"].get(pid)
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")

    if primary:
        from eval.gst_retrieval import source_matches

        st.markdown("**Required law (corrected gold)**")
        for s in primary:
            hit = any(source_matches(s, r) for r in final_ids)
            st.markdown(f"- {'✅' if hit else '❌'} {label(s, graph)}")
        if missing:
            st.markdown("**Required law not in the corpus** (no retriever can find it)")
            for s in missing:
                st.markdown(f"- {s}")

    if result:
        with st.expander("Pipeline steps (live run)"):
            st.markdown("**Sub-queries (decompose)**")
            for q in result["subqueries"] or ["(none — the original question only)"]:
                st.markdown(f"- {q}")
            st.markdown("**20-candidate pool, E17 order, with T31 grades (3 = the answer depends on it)**")
            st.dataframe([{"E17 rank": i + 1, "grade": gr, "provision": label(pid, graph) if pid else "?",
                           "in final 8": "yes" if pid in result["final_ids"] else ""}
                          for i, (pid, gr) in enumerate(result["pool"])], hide_index=True, width="stretch")

    closed = data["closed"].get(qid) if qid else None
    if closed:
        with st.expander("Closed book: the same model with no retrieval"):
            cm = closed.get("ragas_metrics", {})
            st.caption(f"Correctness {cm.get('answer_correctness', 0):.2f} (Qwen judge)")
            st.markdown(closed["answer"])

    with st.expander("Reported totals (whole golden set)"):
        ret = [r["retrieval"]["corrected"] for r in data["retrieval"].values()
               if r["retrieval"]["corrected"]["required_recall"] is not None]
        jm = [r.get("ragas_metrics", {}) for r in data["judged"].values()]

        def avg(vals):
            vals = [v for v in vals if v is not None and v == v]  # judge rows may hold NaN
            return sum(vals) / len(vals) if vals else None

        c = st.columns(5)
        metric(c[0], f"Recall ({len(ret)} q)", avg(r["required_recall"] for r in ret))
        metric(c[1], "MRR", avg(r["required_mrr"] for r in ret))
        scored = [m["answer_correctness"] for m in jm if m.get("answer_correctness") == m.get("answer_correctness")]
        metric(c[2], f"Correctness ({len(scored)} q)", avg(scored))
        metric(c[3], "Faithfulness", avg(m.get("faithfulness") for m in jm))
        metric(c[4], "Faithfulness q-aware (79)",
               avg(r.get("ragas_metrics", {}).get("faithfulness") for r in data["qctx"].values()))


main()
