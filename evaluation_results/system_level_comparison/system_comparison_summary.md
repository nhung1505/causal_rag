# BLHS System-Level Baseline Comparison

- Comparator version: `1.0-system-level-paired-comparison`
- Reference method: `full_legal_scm`
- Shared end-to-end quality metrics include every gold sample; a missing or failed prediction receives zero.
- Retrieval/path cells are `N/A` when a method does not expose that component.
- Top-1 path is computed only on samples with a linear gold path; branch/convergence samples are excluded.

## Main comparison

| Method | Success | Decision Acc. | Macro-F1 | Answer F1 | ROUGE-L | Citation F1 | Article R@5 | Rule R@5 | Event R@5 | Top-1 Path | Avg. sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 100.00% | 0.00% | 0.00% | 12.98% | 11.27% | 0.00% | N/A | N/A | N/A | N/A | 0.848 |
| vanilla_rag | 100.00% | 59.60% | 69.42% | 24.71% | 22.76% | 49.78% | 60.07% | N/A | N/A | N/A | 2.475 |
| causal_path_rag | 100.00% | 90.80% | 84.36% | 36.53% | 35.82% | 58.70% | 75.40% | 71.20% | 74.79% | 38.56% | 2.121 |
| full_legal_scm | 100.00% | 92.00% | 97.73% | 35.70% | 34.90% | 55.39% | 75.00% | 70.80% | 74.66% | 37.29% | 5.950 |

## Paired significance against the reference

Positive delta means the reference method scores higher.

| Comparator | Metric | Delta | 95% CI | Test | p | Holm p |
|---|---|---:|---:|---|---:|---:|
| llm_only | decision_accuracy | 0.9200 | [0.8840, 0.9520] | exact_mcnemar | 0.0000 | 0.0000 |
| llm_only | answer_token_f1 | 0.2273 | [0.1934, 0.2614] | paired_randomization | 0.0002 | 0.0020 |
| llm_only | answer_rouge_l_f1 | 0.2363 | [0.2044, 0.2698] | paired_randomization | 0.0002 | 0.0020 |
| llm_only | citation_f1 | 0.5539 | [0.5092, 0.5999] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | decision_accuracy | 0.3240 | [0.2560, 0.3920] | exact_mcnemar | 0.0000 | 0.0000 |
| vanilla_rag | answer_token_f1 | 0.1100 | [0.0732, 0.1462] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | answer_rouge_l_f1 | 0.1214 | [0.0856, 0.1573] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | citation_f1 | 0.0561 | [0.0042, 0.1089] | paired_randomization | 0.0402 | 0.1608 |
| causal_path_rag | decision_accuracy | 0.0120 | [-0.0320, 0.0560] | exact_mcnemar | 0.7201 | 0.8296 |
| causal_path_rag | answer_token_f1 | -0.0083 | [-0.0252, 0.0084] | paired_randomization | 0.3327 | 0.8296 |
| causal_path_rag | answer_rouge_l_f1 | -0.0091 | [-0.0258, 0.0066] | paired_randomization | 0.2765 | 0.8296 |
| causal_path_rag | citation_f1 | -0.0331 | [-0.0576, -0.0098] | paired_randomization | 0.0042 | 0.0210 |
