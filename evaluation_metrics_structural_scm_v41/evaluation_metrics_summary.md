# BLHS CausalRAG — Detailed Evaluation Metrics

- Step 7 version: `2.0-current-pipeline-schema`
- Samples: **250**
- Prediction coverage: **100.00%**
- Success rate: **100.00%**

## Main metrics

| Metric | Value |
|---|---:|
| Rule Recall@5 | 69.53% |
| Event Recall@5 | 74.66% |
| Top-1 Exact Path | 35.20% |
| Oracle Exact Path | 71.20% |
| Verification Accuracy | 87.20% |
| Verification Macro-F1 | 96.26% |
| Verification Balanced Accuracy | 93.04% |
| Answer Token F1 | 23.74% |
| Answer ROUGE-L F1 | 21.77% |
| Citation F1 | 53.92% |

## Interpretation notes

- Top-1 path evaluates the path actually selected by the pipeline.
- Oracle path is a diagnostic over returned causal paths; it must not replace top-1 results.
- The gap between oracle and top-1 exact path indicates a path-ranking problem.
- Verification Macro-F1 and Balanced Accuracy should be reported with Accuracy when classes are imbalanced.
