"""T10 — Agentic RAG: a tool-using agent with a search → judge → refine loop.

Idea. Fixed pipelines retrieve once. An agent decides itself what to search next, reads results, looks up the
provisions they mention, computes amounts, and stops when a judge says the evidence is sufficient. Sources:
- L-MARS (Wang & Yuan, arXiv 2509.00761, 2025; code github.com/boqiny/L-MARS): Query agent plans sub-questions,
  Search agent retrieves, Judge agent checks evidence sufficiency (incl. jurisdiction / temporal validity) and
  loops "search-judge-refine" until a threshold is met.
- Search-R1 / Search-o1 / R1-Searcher (2025): the LLM interleaves reasoning with search calls (there trained with
  RL; here prompted, no training).
- ReAct (Yao et al. 2023): thought → action → observation loop.

Tools (all over this corpus; no web):
- search(query)            E17-style hybrid search on the 3-large collection, top 5 (labels + 300-char snippets)
- lookup(citation)         "section 16 CGST", "rule 36(4)", "Notification 12/2017-CT (Rate)" → the provision text
                           (citation_lookup.py, symbolic, exact)
- calculate(expression)    safe arithmetic (AST-evaluated: + − × ÷ %, parentheses) for tax computations
- finish()                 ends the research phase
After each research round a Judge prompt returns {"sufficient": bool, "missing": "..."}; if not sufficient, the
missing part becomes the next round's focus (max 3 rounds × 4 steps). The final answer is generated from the
collected evidence with the grounded prompt; evidence provisions are the "retrieved set" for scoring.

Cost: free with the Gemini keys (≈ 6–12 calls per question). Slower than E17 (sequential LLM calls).

    uv run python -m scripts.techniques.t10_agentic_rag ask "Is ITC available on a car used by the director?"
    uv run python -m scripts.techniques.t10_agentic_rag eval --set dev_hard        # retrieval of the evidence set
"""

from __future__ import annotations

import argparse
import ast
import json
import operator

from app.models import RetrievedChunk
from scripts.techniques.common import evaluate, use_best_stack

MAX_ROUNDS, MAX_STEPS, TOP = 3, 4, 5
AGENT_PROMPT = """You research Indian GST law questions using tools over a database of the CGST/IGST/UTGST Acts and
Rules, notifications and circulars. Each turn, choose ONE action and return JSON:
{"thought": "...", "action": "search" | "lookup" | "calculate" | "finish", "input": "..."}
- search: a short legal search query (concept + provision if known), e.g. "blocked credit motor vehicles section 17(5)"
- lookup: a citation to read in full, e.g. "section 17 CGST Act", "rule 42", "Notification No. 12/2017-CT (Rate)"
- calculate: an arithmetic expression, e.g. "250000 * 18 / 100"
- finish: when the evidence below covers every legal issue of the question
Prefer looking up provisions that search results or your knowledge point to. Do not repeat a query."""
JUDGE_PROMPT = """You check whether collected evidence is sufficient to answer an Indian GST law question.
Return JSON {"sufficient": true|false, "missing": "what legal issue or provision is still missing"}."""
ANSWER_SYSTEM = None  # set lazily: grounded system prompt

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expr: str) -> str:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")

    try:
        return str(round(ev(ast.parse(expr.replace(",", "").replace("×", "*").replace("÷", "/"), mode="eval").body), 4))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def as_dict(data) -> dict:
    """The model sometimes wraps its JSON object in a list; take the first object."""
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), {})
    return data if isinstance(data, dict) else {}


def tool_search(query: str) -> list[RetrievedChunk]:
    from app.services.rag_service import _retrieve

    return _retrieve(query, {"search_mode": "hybrid", "enable_crag": False, "top_k": TOP, "doc_type_cap": 2})


def tool_lookup(citation: str) -> list[RetrievedChunk]:
    from app.services.citation_lookup import cited_ids
    from app.services.embedding_service import embed_texts
    from app.services.gst_graph import get_graph
    from app.services.vector_store import search_within

    graph = get_graph()
    text = citation if any(w in citation.lower() for w in ("section", "rule", "schedule", "notification", "circular")) \
        else f"section {citation}"
    by_inst: dict[str, list[str]] = {}
    for pid, r in graph.records.items():
        by_inst.setdefault(r["instrument_id"], []).append(pid)
    out = []
    for cid in cited_ids(text, set(graph.records), set(by_inst)):
        pid = cid if cid in graph.records else (search_within(embed_texts([citation])[0], by_inst[cid], 1) or [None])[0]
        if pid:
            out.append(graph.chunk_for(pid))
    return out


def view(chunk: RetrievedChunk, question: str, hint: str = "", chars: int = 1200) -> str:
    """The part of a provision worth showing: from a cited sub-section ("(5)") if the hint names one, else the
    passages most similar to the question (context_variants.compress)."""
    import re

    from app.services.context_variants import compress

    m = re.search(r"\((\d{1,2}[A-Z]?)\)", hint)
    if m:
        pos = chunk.text.find(f"({m.group(1)})")
        if pos >= 0:
            return chunk.text[pos : pos + chars]
    return compress([chunk], question, keep=2)[0].text[:chars]


def run(question: str, verbose: bool = False) -> dict:
    from app.security.system_prompt import build_system_prompt
    from app.services.llm_service import generate, generate_with_json

    evidence: dict[str, RetrievedChunk] = {}
    views: dict[str, str] = {}
    log, focus, calcs, done = [], "", [], set()
    for rnd in range(MAX_ROUNDS):
        for _ in range(MAX_STEPS):
            notes = "\n".join(f"- {s}: {v.replace(chr(10), ' ')}" for s, v in list(views.items())[-8:])
            user = (f"Question: {question}\n" + (f"Focus now on: {focus}\n" if focus else "")
                    + f"Evidence so far:\n{notes or '(none)'}\nPrevious actions: {json.dumps(log[-6:])}")
            try:
                step = as_dict(json.loads(generate_with_json(AGENT_PROMPT, user)["text"]))
            except Exception:  # noqa: BLE001
                break
            action, arg = step.get("action"), str(step.get("input", ""))
            if (action, arg.lower()) in done and action != "finish":
                log.append({"round": rnd, "action": action, "input": arg, "observation": "already done - choose another"})
                continue
            done.add((action, arg.lower()))
            if verbose:
                print(f"[{rnd}] {action}: {arg}  — {step.get('thought', '')[:120]}")
            if action == "finish":
                break
            if action == "calculate":
                calcs.append(f"{arg} = {calculate(arg)}")
                log.append({"round": rnd, "action": action, "input": arg, "observation": calcs[-1]})
                continue
            found = tool_search(arg) if action == "search" else tool_lookup(arg) if action == "lookup" else []
            for c in found:
                evidence.setdefault(c.source, c)
                views[c.source] = view(c, question, arg if action == "lookup" else "")
            log.append({"round": rnd, "action": action, "input": arg, "observation": [c.source for c in found]})
        notes = "\n".join(f"- {s}: {v.replace(chr(10), ' ')}" for s, v in views.items())
        try:
            verdict = as_dict(json.loads(generate_with_json(JUDGE_PROMPT, f"Question: {question}\nEvidence:\n{notes}")["text"]))
        except Exception:  # noqa: BLE001
            verdict = {"sufficient": True}
        log.append({"round": rnd, "judge": verdict})
        if verbose:
            print(f"  judge: {verdict}")
        if verdict.get("sufficient"):
            break
        focus = verdict.get("missing", "")
    context = "\n\n".join(f"[{c.source}]\n{c.text[:6000]}" for c in evidence.values())
    calc_note = ("\nCalculations done:\n" + "\n".join(calcs)) if calcs else ""
    answer = generate(build_system_prompt(grounded=True),
                      f"Retrieved law:\n{context}{calc_note}\n\nQuestion: {question}")["text"]
    return {"answer": answer, "evidence": list(evidence.values()), "log": log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["ask", "eval"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--set", default="dev_hard")
    args = ap.parse_args()
    use_best_stack()
    if args.cmd == "ask":
        out = run(args.question, verbose=True)
        print("\nEVIDENCE:", [c.source for c in out["evidence"]])
        print("\nANSWER:", out["answer"][:3000])
    else:
        # Scored on the evidence set, capped at 8 in collection order (first found first).
        evaluate(f"t10_agentic_{args.set}", lambda q: run(q)["evidence"][:8], args.set)


if __name__ == "__main__":
    main()
