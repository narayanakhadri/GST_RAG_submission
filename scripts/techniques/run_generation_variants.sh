#!/usr/bin/env bash
# Generation-side techniques (T02, T04–T07, T10) as RAGAS runs on a question set (default dev_hard, ≈ $0.06 each with
# the Qwen judge). Base = the planned final system: E17 retrieval + grounded prompt + law + summary context.
# Compare each run with the base:  uv run python -m scripts.techniques.compare_ragas <base.json> <variant.json>
#
#   bash scripts/techniques/run_generation_variants.sh dev_hard            # all variants
#   bash scripts/techniques/run_generation_variants.sh dev_hard irac       # one variant
set -u
cd "$(dirname "$0")/../.."
export UPSTASH_REDIS_URL= PYTHONIOENCODING=utf-8 LLM_PROVIDER=gemini
SET="${1:-dev_hard}"; ONLY="${2:-}"
COMMON=(--provider gemini --no-web-fallback --gold-sources corrected --coverage full --sample 0
        --golden seed/docs/gst/golden_set_with_gold_sources.textfix.jsonl
        --collection gst_thesis_enriched_large --embedding-model text-embedding-3-large --question-set "$SET")
BASE='"search_mode":"hybrid","enable_rerank":false,"enable_crag":false,"top_k":8,"doc_type_cap":3,"query_rewrite":"decompose","grounded_prompt":true,"context_text":"law+summary"'
declare -A VARIANTS=(
  [base]=""
  [relations]=',"context_variants":["relations"]'          # T04 graph relations in the prompt
  [edge_order]=',"context_variants":["edge_order"]'        # T05 lost-in-the-middle ordering
  [compress]=',"context_variants":["compress"]'            # T06 contextual compression
  [irac]=',"prompt_variant":"irac"'                        # T07 IRAC answer structure
  [chain_of_note]=',"prompt_variant":"chain_of_note"'      # T07 Chain-of-Note
  [obligations]=',"prompt_variant":"obligations"'          # T07 obligation-centric prompting (Gokhan & Briscoe 2025)
  [as_on_2025]=',"as_on_date":"2025-04-30"'                # T02 version notes (answers state the law on 30.04.2025)
  [citation_lookup]=',"citation_lookup":true'              # W1a end to end
  [bm25_citation]=',"sparse_scorer":"bm25","citation_lookup":true'   # C2 retrieval end to end
  [crag]=',"enable_crag":true'                             # CRAG chunk grading (web fallback off via --no-web-fallback)
  [self_rag]=',"enable_self_reflective":true'              # Self-RAG reflection + retry
)
# Separate scripts (same RAGAS format, score with eval.rescore_ragas): t23_self_route, t24_speculative_rag,
# t25_chain_of_verification, t26_subquestion_answering, t28_ensemble_selector, t18_long_context.
for name in "${!VARIANTS[@]}"; do
  [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
  .venv/Scripts/python.exe -m eval.run_gst_ragas --profile hybrid "${COMMON[@]}" \
    --flags "{${BASE}${VARIANTS[$name]}}" --name "gen_${name}_${SET}" > "eval/results/thesis/logs/gen_${name}_${SET}.log" 2>&1
  echo "done $name"
done
