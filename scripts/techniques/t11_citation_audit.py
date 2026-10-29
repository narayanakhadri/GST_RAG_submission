"""T11 — Citation audit of generated answers: existence + in-context + fidelity (entailment).

Idea. A legal answer is only trustworthy if every provision it cites (1) exists, (2) was actually retrieved, and
(3) says what the answer claims. Sources:
- EscavAI (PROPOR 2026, "Grounded in Law: A Multi-Stage Anti-Hallucination Pipeline for Legal RAG"): after
  generation, taggers extract legislation mentions, normalise them, check existence against the official
  database, verify fidelity against the official text, and trigger targeted rewrites (6.5% of answers corrected).
- COLIEE statute entailment (paper 8): a DeBERTa NLI model decides whether a statute entails a statement.
- Citation-grounding metrics (Ovcharov 2026, arXiv 2606.00898): existence, relevance, temporal validity.

How it works here (free: local NLI model, no LLM).
1. Citations in each answer → corpus IDs with app/services/citation_lookup.py (sections, rules, schedules,
   notifications, circulars; a sub-section "(5)" is kept to pick the passage).
2. existence  = the ID is in the corpus;  in_context = the provision was among the retrieved contexts.
3. fidelity   = for each sentence that cites a provision: premise = the provision passage most similar to the
   sentence (the cited sub-section if given), hypothesis = the sentence → NLI label (entailment / neutral /
   contradiction) with cross-encoder/nli-deberta-v3-base.
   Caveat: the NLI model is general-domain; legal sentences that paraphrase or combine provisions often come out
   "neutral", and a wrong premise passage shows up as "contradiction" — treat the shares as a screening signal and
   read the flagged list, not as a hallucination rate. The in-context rate is the robust number.
4. Report per run: citations per answer, existence rate, in-context rate, entailed / neutral / contradicted shares,
   and the list of contradicted citations (candidates for a targeted rewrite).

    uv run python -m scripts.techniques.t11_citation_audit eval/results/<run>.json [<run2>.json ...]
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections import Counter
from pathlib import Path

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
LABELS = ("contradiction", "entailment", "neutral")  # output order of this model
_lock, _nli = threading.Lock(), {}


def nli(pairs: list[tuple[str, str]]) -> list[str]:
    from sentence_transformers import CrossEncoder

    with _lock:
        if "m" not in _nli:
            _nli["m"] = CrossEncoder(NLI_MODEL)
    scores = _nli["m"].predict(pairs, batch_size=8)
    return [LABELS[int(s.argmax())] for s in scores]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text) if len(s.strip()) > 20]


def premise_for(record_text: str, sentence: str, cited: str) -> str:
    m = re.search(r"\((\d{1,2}[A-Z]?)\)", cited)
    if m and (pos := record_text.find(f"({m.group(1)})")) >= 0:
        return record_text[pos : pos + 1500]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    parts = [p for p in re.split(r"\n\s*\n", record_text) if p.strip()] or [record_text]
    vec = TfidfVectorizer(stop_words="english").fit(parts + [sentence])
    best = linear_kernel(vec.transform([sentence]), vec.transform(parts))[0].argmax()
    return parts[best][:1500]


def answer_text(row: dict) -> str:
    a = row.get("answer") or ""
    try:
        return json.loads(a).get("answer", a)
    except Exception:  # noqa: BLE001
        return a


def audit(path: str) -> dict:
    from app.services.citation_lookup import cited_ids
    from app.services.gst_graph import get_graph

    graph = get_graph()
    provisions = set(graph.records)
    instruments = {r["instrument_id"] for r in graph.records.values()}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    counts, flagged, n_cit = Counter(), [], 0
    for row in rows:
        retrieved = set(row.get("retrieved_ids") or [])
        for sent in sentences(answer_text(row)):
            for cid in cited_ids(sent, provisions, instruments):
                n_cit += 1
                counts["exists"] += 1  # cited_ids only returns corpus IDs; unresolved mentions are not counted
                in_ctx = cid in retrieved or any(r.startswith(cid + ".") for r in retrieved)
                counts["in_context"] += in_ctx
                pid = cid if cid in graph.records else None
                if pid is None:  # a notification/circular: the entry or paragraph the sentence names, else a retrieved one
                    m = re.search(r"\b(Entry|S\.?\s*No\.?|Sl\.?\s*No\.?|para(?:graph)?)\s*(\d{1,3}[A-Z]{0,2})", sent, re.I)
                    kind_ = "paragraph" if m and m.group(1).lower().startswith("para") else "entry"
                    named = f"{cid}.{kind_}.{m.group(2).lower()}" if m else None
                    pid = named if named in graph.records else next((r for r in retrieved if r.startswith(cid + ".")), None)
                if not pid:
                    continue
                label = nli([(premise_for(graph.records[pid].get("raw_text") or "", sent, sent), sent)])[0]
                counts[label] += 1
                if label == "contradiction":
                    flagged.append({"id": row["id"], "citation": cid, "sentence": sent[:300]})
    judged = counts["entailment"] + counts["neutral"] + counts["contradiction"]
    report = {"run": Path(path).name, "answers": len(rows), "citations": n_cit,
              "citations_per_answer": round(n_cit / max(len(rows), 1), 2),
              "in_context_rate": round(counts["in_context"] / max(n_cit, 1), 3),
              **{f"{k}_share": round(counts[k] / max(judged, 1), 3) for k in ("entailment", "neutral", "contradiction")},
              "contradicted": flagged}
    print(json.dumps({k: v for k, v in report.items() if k != "contradicted"}))
    return report


def main() -> None:
    out = [audit(p) for p in sys.argv[1:]]
    Path("eval/results/thesis/techniques").mkdir(parents=True, exist_ok=True)
    Path("eval/results/thesis/techniques/t11_citation_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
