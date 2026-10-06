# BLHS System-Level Baseline Comparison

- Comparator version: `1.1-system-level-paired-audited-comparison`
- Reference method: `full_legal_scm`
- Shared end-to-end quality metrics include every gold sample; a missing or failed prediction receives zero.
- Retrieval/path cells are `N/A` when a method does not expose that component.
- Top-1 path is computed only on samples with a linear gold path; branch/convergence samples are excluded.

## Main comparison

| Method | Success | Decision Acc. | Balanced Acc. | Macro-F1 | Answer F1 | ROUGE-L | Citation F1 | Article R@5 | Rule R@5 | Event R@5 | Top-1 Path | Avg. sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 100.00% | 4.40% | 27.50% | 22.00% | 15.08% | 13.19% | 0.00% | N/A | N/A | N/A | N/A | 1.031 |
| llm_only_forced_binary | 100.00% | 9.20% | 50.65% | 8.78% | 18.44% | 16.19% | 0.27% | N/A | N/A | N/A | N/A | 0.900 |
| vanilla_rag | 100.00% | 55.20% | 71.09% | 49.88% | 28.18% | 25.68% | 50.75% | 60.07% | N/A | N/A | N/A | 2.369 |
| causal_path_rag | 100.00% | 88.80% | 89.35% | 76.30% | 37.27% | 36.40% | 58.41% | 75.40% | 71.20% | 74.79% | 38.56% | 1.947 |
| full_legal_scm | 100.00% | 96.80% | 98.26% | 91.67% | 37.60% | 36.92% | 58.89% | 75.40% | 71.20% | 74.79% | 38.56% | 3.238 |

## Decision and verification audit

| Method | Non-abstain coverage | Abstain | Factual route | Direct-edge route | do() requested | do() executed | Committed | Step-3 fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_only | 12.00% | 220 | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_only_forced_binary | 100.00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vanilla_rag | 86.40% | 34 | 0 | 0 | 0 | 0 | 0 | 0 |
| causal_path_rag | 98.80% | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| full_legal_scm | 99.60% | 1 | 230 | 13 | 7 | 7 | 230 | 20 |

## Paired significance against the reference

Positive delta means the reference method scores higher.

| Comparator | Metric | Delta | 95% CI | Test | p | Holm p |
|---|---|---:|---:|---|---:|---:|
| llm_only | decision_accuracy | 0.9240 | [0.8920, 0.9560] | exact_mcnemar | 0.0000 | 0.0000 |
| llm_only | answer_token_f1 | 0.2252 | [0.1907, 0.2617] | paired_randomization | 0.0002 | 0.0024 |
| llm_only | answer_rouge_l_f1 | 0.2372 | [0.2026, 0.2719] | paired_randomization | 0.0002 | 0.0024 |
| llm_only | citation_f1 | 0.5889 | [0.5427, 0.6349] | paired_randomization | 0.0002 | 0.0024 |
| llm_only_forced_binary | decision_accuracy | 0.8760 | [0.8360, 0.9160] | exact_mcnemar | 0.0000 | 0.0000 |
| llm_only_forced_binary | answer_token_f1 | 0.1916 | [0.1593, 0.2230] | paired_randomization | 0.0002 | 0.0024 |
| llm_only_forced_binary | answer_rouge_l_f1 | 0.2073 | [0.1766, 0.2386] | paired_randomization | 0.0002 | 0.0024 |
| llm_only_forced_binary | citation_f1 | 0.5862 | [0.5418, 0.6313] | paired_randomization | 0.0002 | 0.0024 |
| vanilla_rag | decision_accuracy | 0.4160 | [0.3520, 0.4800] | exact_mcnemar | 0.0000 | 0.0000 |
| vanilla_rag | answer_token_f1 | 0.0942 | [0.0626, 0.1261] | paired_randomization | 0.0002 | 0.0024 |
| vanilla_rag | answer_rouge_l_f1 | 0.1123 | [0.0814, 0.1439] | paired_randomization | 0.0002 | 0.0024 |
| vanilla_rag | citation_f1 | 0.0814 | [0.0282, 0.1356] | paired_randomization | 0.0024 | 0.0096 |
| causal_path_rag | decision_accuracy | 0.0800 | [0.0440, 0.1200] | exact_mcnemar | 0.0001 | 0.0011 |
| causal_path_rag | answer_token_f1 | 0.0033 | [-0.0119, 0.0185] | paired_randomization | 0.6671 | 1.0000 |
| causal_path_rag | answer_rouge_l_f1 | 0.0052 | [-0.0095, 0.0200] | paired_randomization | 0.5161 | 1.0000 |
| causal_path_rag | citation_f1 | 0.0047 | [-0.0100, 0.0195] | paired_randomization | 0.5315 | 1.0000 |

## Contract warnings

- Generation contract mismatch llm_only vs llm_only_forced_binary: {'decision_policy': ('abstain_allowed', 'forced_binary')}
