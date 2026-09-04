# BLHS CausalRAG — Detailed Evaluation Metrics

- Step 7 version: `2.0-current-pipeline-schema`
- Samples: **250**
- Prediction coverage: **100.00%**
- Success rate: **100.00%**

## Main metrics

| Metric | Value |
|---|---:|
| Rule Recall@5 | 0.00% |
| Event Recall@5 | 0.00% |
| Top-1 Exact Path | 5.60% |
| Oracle Exact Path | 5.60% |
| Verification Accuracy | 34.00% |
| Verification Macro-F1 | 44.55% |
| Verification Balanced Accuracy | 43.59% |
| Answer Token F1 | 23.20% |
| Answer ROUGE-L F1 | 20.38% |
| Citation F1 | 0.53% |

## Interpretation notes

- Top-1 path evaluates the path actually selected by the pipeline.
- Oracle path is a diagnostic over returned causal paths; it must not replace top-1 results.
- The gap between oracle and top-1 exact path indicates a path-ranking problem.
- Verification Macro-F1 and Balanced Accuracy should be reported with Accuracy when classes are imbalanced.
