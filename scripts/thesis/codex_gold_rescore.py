"""Sensitivity check: do the retrieval results and conclusions hold on gold sources corrected by the Codex audit?

    uv run python -m scripts.thesis.codex_gold_rescore

The audit (judge/golden_audit/out/, scripts/thesis/golden_audit.py) lists per question the required sources the
reasoning does not need and the ones it is missing. This builds an *audited* gold from the SME golden set the audit
read (`golden_set_sme_v1.jsonl`): unnecessary required sources move to supporting, missing ones become required
(names like "CGST Act, section 22" are mapped to corpus IDs; law outside the corpus is recorded as missing from it,
which makes that question partially covered). The audited gold is an LLM suggestion and is written next to the results,
never into seed/: it is used only to test whether the conclusions depend on the gold's imperfections.

Every saved run is re-scored on the audited questions with both golds (no retrieval, no LLM: free), and every paired
comparison the thesis reports is recomputed. Output: eval/results/thesis/codex_audit/{golden_codex_audit.jsonl,report.md}.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from app.config import settings
from app.datasets.gst.corpus_loader import load_corpus
from app.datasets.gst.golden_loader import load_golden_set
from eval.gst_retrieval import corpus_keys, label_to_id_map, score_row
from eval.stats import paired
from scripts.thesis.sme_rescore import SME_GOLDEN

AUDIT = Path("judge/golden_audit/out")
OUT = Path("eval/results/thesis/codex_audit")
RETRIEVAL = Path("eval/results/thesis/retrieval")
TECH = Path("eval/results/thesis/techniques")
METRICS = ("required_recall", "required_mrr")
# (name, file, reference, reference file): the comparisons the thesis reports
COMPARISONS = [(n, RETRIEVAL / f"{n}.json", "hybrid_tfidf", RETRIEVAL / "hybrid_tfidf.json") for n in (
    "dense", "sparse_tfidf", "sparse_bm25", "hybrid_bm25", "hybrid_top8", "embed_raw", "embed_exact", "embed_large",
    "hybrid_cap2", "hybrid_graph", "hybrid_graphdense_w1", "rerank_minilm", "rerank_llm", "hyde", "rewrite",
    "decompose", "large_top8_cap3", "large_top8_cap3_decompose")] + [
    ("t31_pointwise_on_all", TECH / "t31_pointwise_on_all.json", "e17_v3_textfix", RETRIEVAL / "e17_v3_textfix.json"),
    ("hybrid_tfidf → final", TECH / "t31_pointwise_on_all.json", "hybrid_tfidf", RETRIEVAL / "hybrid_tfidf.json")] + [
    (f"{t}_{s}", TECH / f"{t}_{s}.json", "e17_v3_textfix", RETRIEVAL / "e17_v3_textfix.json")
    for t in ("t31_pointwise_on", "t31_pointwise_off", "t22_ppr_a0.5_w0.5", "t32_hyde_fusion_w1.0", "t27_graphrag_c3",
              "t09_ltr") for s in ("dev_hard", "test_hard")] + [
    (f"t34_missing_law_{s}", TECH / f"t34_missing_law_{s}.json", f"t31_pointwise_on_{s}",
     TECH / f"t31_pointwise_on_{s}.json") for s in ("dev_hard", "test_hard")]
NAME = re.compile(r"(CGST|IGST|UTGST)\s+(Act|Rules),?\s+(section|rule|schedule)\s+([0-9IVX]+[A-Z]?)", re.I)


def to_id(name: str, keys: set[str]) -> str | None:
    if name in keys:
        return name
    if m := NAME.search(name):
        act, kind, unit, num = (g.lower() for g in m.groups())
        pid = f"{act}_{'rules' if kind == 'rules' else 'act'}_2017.{unit}.{num.upper() if num[-1].isalpha() else num}"
        if unit == "schedule":
            pid = f"{act}_act_2017.schedule.{ {'i': '1', 'ii': '2', 'iii': '3'}.get(num, num)}"
        for cand in (pid, pid.lower()):
            if cand in keys:
                return cand
    return None


def audited_golden(keys: set[str]) -> tuple[dict[str, dict], dict]:
    golden = {g["id"]: g for g in load_golden_set(SME_GOLDEN)}
    stats = {"questions": 0, "removed": 0, "added": 0, "outside_corpus": []}
    for f in sorted(AUDIT.glob("*.json")):
        v = json.loads(f.read_text(encoding="utf-8"))
        g = golden.get(v.get("qid"))
        if g is None:
            continue
        gs = v["reference"].get("gold_sources") or {}
        unnecessary = set(gs.get("unnecessary_required") or [])
        req = [s for s in g.get("primary_required_sources") or [] if s not in unnecessary]
        sup = list(g.get("supporting_sources") or [])
        missing_corpus = list(g.get("external_sources_missing_from_corpus") or [])
        stats["removed"] += len(g.get("primary_required_sources") or []) - len(req)
        sup += [s for s in unnecessary if s not in sup]
        for name in gs.get("missing_required") or []:
            pid = to_id(name, keys)
            if pid and pid not in req:
                req.append(pid)
                stats["added"] += 1
            elif not pid:
                missing_corpus.append(name)
                stats["outside_corpus"].append(f"{v['qid']}: {name}")
        g["primary_required_sources"], g["supporting_sources"] = req, [s for s in sup if s not in req]
        g["external_sources_missing_from_corpus"] = missing_corpus
        g["codex_audited"] = True
        stats["questions"] += 1
    return golden, stats


def rescore(path: Path, golden: dict[str, dict], ids: set[str], keys, labels) -> list[dict]:
    rows = [copy.deepcopy(r) for r in json.loads(path.read_text(encoding="utf-8"))["rows"] if r["id"] in ids]
    for r in rows:
        score_row(r, golden[r["id"]], keys, labels)
    return rows


def fmt(r: dict, m: str) -> str:
    if not r.get("n"):
        return "—"
    p = r["p_wilcoxon"]
    star = "*" if p is not None and p < 0.05 else ""
    return f"{r['diff']:+.3f}{star} (n {r['n']})"


def main() -> None:
    corpus = load_corpus(settings.gst_corpus_path)
    keys, labels = corpus_keys(corpus), label_to_id_map(corpus)
    sme = {g["id"]: g for g in load_golden_set(SME_GOLDEN)}
    audited, stats = audited_golden(keys)
    ids = {q for q, g in audited.items() if g.get("codex_audited")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "golden_codex_audit.jsonl").write_text(
        "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in audited.values()), encoding="utf-8")

    lines = ["# Sensitivity check: results on gold sources corrected by the Codex audit", "",
             f"Generated by scripts/thesis/codex_gold_rescore.py. {stats['questions']} audited questions; required sources "
             f"removed {stats['removed']}, added {stats['added']}; {len(stats['outside_corpus'])} suggested sources are "
             "outside the corpus (those questions become partially covered and leave the fully covered set).",
             "Both columns use the same audited questions, fully covered under each gold. Δ = paired difference vs the "
             "reference; * Wilcoxon p < 0.05.", "",
             "| Comparison | Reference | Δ recall: SME gold | Δ recall: audited gold | Δ MRR: SME gold | Δ MRR: audited gold | "
             "Conclusion changes? |", "|---|---|---|---|---|---|---|"]
    levels = ["| Run | Recall SME → audited | MRR SME → audited |", "|---|---|---|"]
    seen = set()
    for name, path, ref, ref_path in COMPARISONS:
        if not (path.exists() and ref_path.exists()):
            continue
        res = {}
        for label, gold in (("sme", sme), ("audit", audited)):
            a, b = rescore(ref_path, gold, ids, keys, labels), rescore(path, gold, ids, keys, labels)
            res[label] = {m: paired(a, b, m) for m in METRICS}
            for run, rows in ((ref, a), (name, b)):
                full = [r for r in rows if r["coverage"]["corrected"] == "full"]
                res.setdefault("lvl", {}).setdefault(run, {})[label] = {
                    m: sum(r["retrieval"]["corrected"][m] for r in full) / len(full) for m in METRICS} if full else None
        s, a = res["sme"]["required_recall"], res["audit"]["required_recall"]
        sig = lambda r: r.get("n") and r["p_wilcoxon"] is not None and r["p_wilcoxon"] < 0.05  # noqa: E731
        changed = "sign flips" if s.get("n") and a.get("n") and (s["diff"] > 0) != (a["diff"] > 0) and \
            (abs(s["diff"]) > 0.005 or abs(a["diff"]) > 0.005) else (
            "significance changes" if bool(sig(s)) != bool(sig(a)) else "no")
        lines.append(f"| {name} | {ref} | {fmt(s, 'r')} | {fmt(a, 'r')} | {fmt(res['sme']['required_mrr'], 'm')} | "
                     f"{fmt(res['audit']['required_mrr'], 'm')} | {changed} |")
        for run, v in res["lvl"].items():
            if run not in seen and v.get("sme") and v.get("audit"):
                seen.add(run)
                levels.append(f"| {run} | {v['sme']['required_recall']:.3f} → {v['audit']['required_recall']:.3f} | "
                              f"{v['sme']['required_mrr']:.3f} → {v['audit']['required_mrr']:.3f} |")
    lines += ["", "## Score levels on the audited questions (fully covered)", ""] + levels
    lines += ["", "## Suggested sources outside the corpus", ""] + [f"- {s}" for s in stats["outside_corpus"]]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
