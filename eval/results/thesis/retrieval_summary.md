# Retrieval ablation (corrected gold sources, fully covered questions)

| Config | RQ | Description | n | Required law found | All required | hit@k | MRR | Act share retrieved | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| dense | RQ3 | Dense only (text-embedding-3-small) | 141 | 0.392 | 0.227 | 0.716 | 0.570 | 0.28 | 47.6 |
| sparse_tfidf | RQ3 | Sparse only, TF-IDF | 141 | 0.352 | 0.191 | 0.667 | 0.442 | 0.33 | 8.6 |
| sparse_bm25 | RQ3 | Sparse only, BM25 | 141 | 0.382 | 0.206 | 0.738 | 0.529 | 0.34 | 9.5 |
| hybrid_tfidf | RQ3 | Hybrid dense + TF-IDF (RRF) — system default | 141 | 0.425 | 0.248 | 0.766 | 0.596 | 0.34 | 36.3 |
| hybrid_bm25 | RQ3 | Hybrid dense + BM25 (RRF) | 141 | 0.396 | 0.220 | 0.723 | 0.563 | 0.32 | 17.3 |
| hybrid_top8 | RQ3 | Hybrid TF-IDF, top_k = 8 | 141 | 0.470 | 0.270 | 0.809 | 0.602 | 0.31 | 17.2 |
| hybrid_cap1 | RQ1 | Hybrid, max 1 notification/circular in top 5 | 141 | 0.429 | 0.255 | 0.773 | 0.593 | 0.45 | 20.7 |
| hybrid_cap2 | RQ1 | Hybrid, max 2 notifications/circulars in top 5 | 141 | 0.434 | 0.255 | 0.780 | 0.599 | 0.41 | 36.7 |
| hybrid_cap3 | RQ1 | Hybrid, max 3 notifications/circulars in top 5 | 141 | 0.428 | 0.248 | 0.773 | 0.597 | 0.38 | 29.0 |
| hybrid_inst2 | RQ1 | Hybrid, max 2 items per instrument (diversity) | 141 | 0.401 | 0.220 | 0.738 | 0.589 | 0.31 | 123.4 |
| hybrid_cap2_inst2 | RQ1 | Hybrid, cap 2 notifications/circulars + max 2 per instrument | 141 | 0.398 | 0.220 | 0.738 | 0.589 | 0.38 | 58.8 |
| hybrid_graph | RQ1/RQ2 | Hybrid + graph expansion (cross-reference neighbours) | 141 | 0.425 | 0.248 | 0.766 | 0.574 | 0.33 | 14.4 |
| hybrid_graph_cap2 | RQ1/RQ2 | Hybrid + graph expansion + cap 2 | 141 | 0.434 | 0.255 | 0.780 | 0.578 | 0.44 | 16.2 |
| hybrid_graphdense | RQ1/RQ2 | Hybrid + graph neighbours ranked by query similarity (w 0.5) | 141 | 0.412 | 0.234 | 0.738 | 0.567 | 0.38 | 153.9 |
| hybrid_graphdense_w1 | RQ1/RQ2 | Hybrid + graph neighbours ranked by query similarity (w 1.0) | 141 | 0.378 | 0.213 | 0.709 | 0.434 | 0.44 | 99.8 |
| hybrid_graphdense_cap2 | RQ1/RQ2 | Hybrid + similarity-ranked graph neighbours (w 1.0) + cap 2 | 141 | 0.374 | 0.206 | 0.688 | 0.429 | 0.50 | 90.8 |
| hybrid_top8_cap2 | RQ1/RQ3 | Hybrid, top_k 8, max 2 notifications/circulars | 141 | 0.466 | 0.270 | 0.809 | 0.603 | 0.39 | 87.2 |
| hybrid_top8_cap3 | RQ1/RQ3 | Hybrid, top_k 8, max 3 notifications/circulars | 141 | 0.471 | 0.270 | 0.816 | 0.604 | 0.37 | 17.1 |
| hybrid_top8_cap4 | RQ1/RQ3 | Hybrid, top_k 8, max 4 notifications/circulars | 141 | 0.469 | 0.270 | 0.809 | 0.602 | 0.36 | 16.9 |
| large_cap2 | RQ1/RQ3 | 3-large, hybrid, max 2 notifications/circulars | 141 | 0.494 | 0.298 | 0.837 | 0.663 | 0.42 | 68.0 |
| large_top8 | RQ3 | 3-large, hybrid, top_k 8 | 141 | 0.529 | 0.326 | 0.851 | 0.664 | 0.31 | 16.2 |
| large_top8_cap3 | RQ1/RQ3 | 3-large, hybrid, top_k 8, max 3 notifications/circulars | 141 | 0.551 | 0.340 | 0.894 | 0.670 | 0.38 | 36.5 |
| large_dense | RQ3 | 3-large, dense only | 141 | 0.487 | 0.291 | 0.837 | 0.656 | 0.30 | 13.5 |
| large_top8_cap3_decompose | RQ3 | Best stack (3-large, top_k 8, cap 3) + LLM decomposition, RRF | 141 | 0.651 | 0.404 | 0.979 | 0.779 | 0.45 | 286.4 |
| e17_v3_textfix | RQ3 | E17 on v3, golden text fixed | 141 | 0.668 | 0.404 | 1.000 | 0.783 | 0.45 | 1163.2 |
| e17_v4 | RQ1 | E17 on v4 corpus (+ notifications 11/2017, 41/2017, 52/2018, CBIC FAQs) | 143 | 0.639 | 0.378 | 0.986 | 0.752 | 0.47 | 82.3 |
| e13a_codexsum | RQ2 | E17 v4 + Codex-corrected summaries embedded | 143 | 0.626 | 0.371 | 0.972 | 0.743 | 0.47 | 75.2 |
| e13c_links3 | RQ1 | E17 v4 + typed links (procedure_in, rate_or_exemption_for, clarifies) | 143 | 0.645 | 0.385 | 0.986 | 0.731 | 0.50 | 83.8 |
| e13c_links2 | RQ1 | E17 v4 + typed links (procedure_in, clarifies) | 143 | 0.638 | 0.378 | 0.986 | 0.731 | 0.48 | 84.9 |
| w1a_citation | RQ3 | E17 + named provisions looked up by ID and put first | 141 | 0.671 | 0.411 | 1.000 | 0.814 | 0.45 | 129.8 |
| w1b_adaptive_k | RQ3 | E17 + top_k by number of sub-queries (2→6, 3→7, 4→12; mean 8.4) | 141 | 0.665 | 0.404 | 0.993 | 0.782 | 0.44 | 143.8 |
| w1b_fixed_k9 | RQ3 | Control for w1b: E17 with top_k 9 | 141 | 0.676 | 0.411 | 1.000 | 0.783 | 0.44 | 255.2 |
| w1c_stepback | RQ3 | E17 + step-back query | 141 | 0.658 | 0.426 | 0.979 | 0.792 | 0.44 | 1003.1 |
| w1d_dense03 | RQ3 | E17, RRF weight dense 0.3 : sparse 0.7 | 141 | 0.600 | 0.348 | 0.929 | 0.709 | 0.43 | 256.6 |
| w1d_dense07 | RQ3 | E17, RRF weight dense 0.7 : sparse 0.3 | 141 | 0.653 | 0.390 | 1.000 | 0.773 | 0.45 | 272.0 |
| w1e_mmr07 | RQ1 | E17 + MMR diversity λ 0.7 | 141 | 0.656 | 0.404 | 0.986 | 0.777 | 0.43 | 956.0 |
| w1e_mmr05 | RQ1 | E17 + MMR diversity λ 0.5 | 141 | 0.626 | 0.369 | 0.965 | 0.765 | 0.43 | 761.8 |
| w1f_splade | RQ3 | E17 with SPLADE++ instead of TF-IDF | 141 | 0.649 | 0.404 | 0.993 | 0.787 | 0.51 | 427.7 |
| w1f_bm42 | RQ3 | E17 with BM42 instead of TF-IDF | 141 | 0.661 | 0.418 | 0.986 | 0.798 | 0.44 | 178.7 |
| w1f_minicoil | RQ3 | E17 with miniCOIL instead of TF-IDF | 141 | 0.672 | 0.426 | 1.000 | 0.846 | 0.43 | 227.7 |
| w1f_bm25 | RQ3 | E17 with BM25 instead of TF-IDF (classic reference) | 141 | 0.675 | 0.411 | 1.000 | 0.845 | 0.43 | 171.7 |
| c2_bm25_citation | RQ3 | E17 + BM25 + cited-provision lookup | 141 | 0.676 | 0.411 | 1.000 | 0.876 | 0.43 | 84.8 |
| c2_minicoil_citation | RQ3 | E17 + miniCOIL + cited-provision lookup | 141 | 0.676 | 0.426 | 1.000 | 0.878 | 0.43 | 136.2 |
| hybrid_pool20 | — | Hybrid top 20 (upper bound for any reranker over the pool) | 141 | 0.551 | 0.326 | 0.879 | 0.607 | 0.29 | 49.3 |
| rerank_minilm | RQ3 | Hybrid + MiniLM-L6 cross-encoder (512 tokens) | 141 | 0.328 | 0.170 | 0.645 | 0.423 | 0.34 | 801.9 |
| rerank_llm | RQ3 | Hybrid + LLM listwise rerank (Gemini flash-lite) | 141 | 0.503 | 0.291 | 0.851 | 0.782 | 0.38 | 868.1 |
| hyde | RQ3 | HyDE (3 hypothetical answers, dense) | 141 | 0.485 | 0.291 | 0.837 | 0.665 | 0.35 | 1273.5 |
| rewrite | RQ3 | LLM rewrites question into a legal-issue query (hybrid) | 141 | 0.496 | 0.298 | 0.858 | 0.670 | 0.35 | 333.2 |
| decompose | RQ3 | LLM decomposes into sub-queries + original, RRF (hybrid) | 141 | 0.522 | 0.319 | 0.865 | 0.654 | 0.40 | 741.4 |
| embed_raw | RQ2 | Embed raw provision text only (hybrid) | 141 | 0.361 | 0.199 | 0.688 | 0.470 | 0.27 | 49.6 |
| embed_exact | RQ2 | Embed hierarchy prefix + text, no LLM metadata (hybrid) | 141 | 0.403 | 0.234 | 0.745 | 0.521 | 0.31 | 49.5 |
| embed_large | RQ3 | Enriched text, text-embedding-3-large (hybrid) | 141 | 0.490 | 0.291 | 0.823 | 0.660 | 0.34 | 107.6 |
