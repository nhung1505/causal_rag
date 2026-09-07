# BLHS CausalRAG — Detailed Evaluation Metrics

- Step 7 version: `2.0-current-pipeline-schema`
- Samples: **250**
- Prediction coverage: **100.00%**
- Success rate: **100.00%**

## Main metrics

| Metric | Value |
|---|---:|
| Rule Recall@5 | 69.93% |
| Event Recall@5 | 74.66% |
| Top-1 Exact Path | 35.20% |
| Oracle Exact Path | 71.20% |
| Verification Accuracy | 92.00% |
| Verification Macro-F1 | 97.73% |
| Verification Balanced Accuracy | 95.65% |
| Answer Token F1 | 68.69% |
| Answer ROUGE-L F1 | 67.67% |
| Citation F1 | 62.86% |

## Interpretation notes

- Top-1 path evaluates the path actually selected by the pipeline.
- Oracle path is a diagnostic over returned causal paths; it must not replace top-1 results.
- The gap between oracle and top-1 exact path indicates a path-ranking problem.
- Verification Macro-F1 and Balanced Accuracy should be reported with Accuracy when classes are imbalanced.
