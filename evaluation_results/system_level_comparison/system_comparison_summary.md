# BLHS System-Level Baseline Comparison

- Comparator version: `1.0-system-level-paired-comparison`
- Reference method: `full_legal_scm`
- Shared end-to-end quality metrics include every gold sample; a missing or failed prediction receives zero.
- Retrieval/path cells are `N/A` when a method does not expose that component.
- Top-1 path is computed only on samples with a linear gold path; branch/convergence samples are excluded.

## Main comparison

| Method | Success | Decision Acc. | Macro-F1 | Answer F1 | ROUGE-L | Citation F1 | Article R@5 | Rule R@5 | Event R@5 | Top-1 Path | Avg. sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 100.00% | 0.00% | 0.00% | 12.98% | 11.27% | 0.00% | N/A | N/A | N/A | N/A | 0.841 |
| vanilla_rag | 100.00% | 55.60% | 67.60% | 23.28% | 21.45% | 49.37% | 60.07% | N/A | N/A | N/A | 2.308 |
| causal_path_rag | 100.00% | 90.80% | 84.36% | 36.53% | 35.82% | 58.70% | 75.40% | 71.20% | 74.79% | 38.56% | 2.107 |
| full_legal_scm | 100.00% | 96.80% | 93.31% | 36.32% | 35.73% | 59.30% | 75.40% | 71.20% | 74.79% | 38.56% | 3.324 |

## Paired significance against the reference

Positive delta means the reference method scores higher.

| Comparator | Metric | Delta | 95% CI | Test | p | Holm p |
|---|---|---:|---:|---|---:|---:|
| llm_only | decision_accuracy | 0.9680 | [0.9440, 0.9880] | exact_mcnemar | 0.0000 | 0.0000 |
| llm_only | answer_token_f1 | 0.2335 | [0.1997, 0.2680] | paired_randomization | 0.0002 | 0.0020 |
| llm_only | answer_rouge_l_f1 | 0.2445 | [0.2123, 0.2779] | paired_randomization | 0.0002 | 0.0020 |
| llm_only | citation_f1 | 0.5930 | [0.5481, 0.6394] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | decision_accuracy | 0.4120 | [0.3480, 0.4760] | exact_mcnemar | 0.0000 | 0.0000 |
| vanilla_rag | answer_token_f1 | 0.1304 | [0.0959, 0.1642] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | answer_rouge_l_f1 | 0.1428 | [0.1098, 0.1769] | paired_randomization | 0.0002 | 0.0020 |
| vanilla_rag | citation_f1 | 0.0993 | [0.0476, 0.1524] | paired_randomization | 0.0002 | 0.0020 |
| causal_path_rag | decision_accuracy | 0.0600 | [0.0280, 0.0960] | exact_mcnemar | 0.0007 | 0.0029 |
| causal_path_rag | answer_token_f1 | -0.0021 | [-0.0183, 0.0143] | paired_randomization | 0.8084 | 1.0000 |
| causal_path_rag | answer_rouge_l_f1 | -0.0009 | [-0.0171, 0.0156] | paired_randomization | 0.9178 | 1.0000 |
| causal_path_rag | citation_f1 | 0.0059 | [-0.0072, 0.0203] | paired_randomization | 0.4135 | 1.0000 |
