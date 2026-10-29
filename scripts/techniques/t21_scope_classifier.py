"""T21 — LLM scope check before answering (abstention for out-of-domain questions), follow-up to W1g.

Idea. W1g showed that a retrieval-score threshold separates GST questions from other Indian tax/legal questions
only partly (AUC 0.92; at the best threshold 8 of 20 out-of-scope questions are still answered): income-tax and
customs questions use the same vocabulary. A cheap LLM classifier ("is this answerable from GST law?") is the
usual guard in production RAG (RAG survey: routing / adaptive retrieval; Hybrid RAG agent, paper 11: retrieve only
when the knowledge base covers the query; CRAG: grade before answering).

How it works here (free: one Gemini call per question, cached).
- Prompt: in scope = CGST/IGST/UTGST Acts and Rules, GST rate/exemption/RCM notifications, CBIC GST circulars;
  out of scope = income tax, customs duty, company law, labour law, stamp duty, FEMA, … → JSON {"in_scope": bool}.
- Evaluation: the 146 golden questions (all in scope) + the 20 synthetic out-of-scope questions of W1g;
  reports accuracy, wrongly refused GST questions and wrongly answered out-of-scope ones; also the combination
  "answer only if the LLM says in scope AND the retrieval score ≥ W1g threshold".

    uv run python -m scripts.techniques.t21_scope_classifier
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE = Path("eval/results/thesis/cache/scope_classifier.json")
OOS = Path("eval/results/thesis/out_of_scope_questions.json")
W1G = Path("eval/results/thesis/retrieval/w1g_abstention.json")
OUT = Path("eval/results/thesis/techniques/t21_scope_classifier.json")
PROMPT = ("You decide whether a question can be answered from Indian GST law: the CGST, IGST and UTGST Acts and "
          "Rules, GST rate / exemption / reverse-charge notifications and CBIC GST circulars. Questions about income "
          "tax, customs duty, excise on petroleum, company law, labour law, stamp duty, FEMA or other laws are out of "
          "scope, even if they mention tax. Return JSON {\"in_scope\": true|false, \"reason\": \"...\"}.")


def in_scope(question: str) -> bool:
    from app.services.llm_service import generate_with_json, grader_model_default

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = hashlib.sha256(f"{grader_model_default()}|{PROMPT}|{question}".encode()).hexdigest()
    if key not in cache:
        try:
            cache[key] = bool(json.loads(generate_with_json(PROMPT, question)["text"]).get("in_scope", True))
        except Exception:  # noqa: BLE001
            cache[key] = True  # fail open: answer rather than refuse
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache[key]


def main() -> None:
    from scripts.techniques.common import questions

    pos = [(g["id"], g["question"]) for g in questions("all")]
    neg = json.loads(OOS.read_text(encoding="utf-8"))["questions"]
    p_ok = [in_scope(q) for _, q in pos]
    n_ok = [in_scope(q) for q in neg]
    report = {"gst_questions": len(pos), "out_of_scope": len(neg),
              "gst_wrongly_refused": sum(not x for x in p_ok), "out_of_scope_wrongly_answered": sum(n_ok),
              "accuracy": round((sum(p_ok) + sum(not x for x in n_ok)) / (len(pos) + len(neg)), 3)}
    if W1G.exists():  # combine with the retrieval-score threshold of W1g
        w = json.loads(W1G.read_text(encoding="utf-8"))
        t = w["best_threshold"]
        ps = {r["id"]: r["score"] for r in w["answerable_scores"]}
        ns = {r["question"]: r["score"] for r in w["out_of_scope_scores"]}
        report["combined_gst_wrongly_refused"] = sum(not (ok and ps.get(i, 1) >= t) for (i, _), ok in zip(pos, p_ok, strict=True))
        report["combined_out_of_scope_wrongly_answered"] = sum(ok and ns.get(q, 0) >= t for q, ok in zip(neg, n_ok, strict=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
