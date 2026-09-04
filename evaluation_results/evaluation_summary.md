# Báo cáo đánh giá CausalRAG trên BLHS Multi-hop Benchmark

- Thời điểm chạy: `2026-07-25T13:18:32.485352+00:00`
- Tổng số câu benchmark: **250**
- Prediction coverage: **100.00%**

## 1. Retrieval

| Chỉ số | Kết quả |
|---|---:|
| Rule Hit@5 | 88.40% |
| Rule Recall@5 | 69.53% |
| Rule Precision@5 | 28.80% |
| Rule MRR | 77.23% |
| Rule MAP | 61.18% |
| Event Hit@5 | 89.20% |
| Event Recall@5 | 74.66% |
| Event Precision@5 | 45.76% |
| Event MRR | 79.72% |

## 2. Causal path

| Chỉ số | Kết quả |
|---|---:|
| Exact path accuracy | 37.29% |
| Hop accuracy | 56.78% |
| Path edge F1 | 57.34% |
| Path event F1 | 65.14% |

## 3. Counterfactual verification

| Chỉ số | Kết quả |
|---|---:|
| Verification accuracy | 87.20% |
| Counterfactual precision | 100.00% |
| Counterfactual recall | 100.00% |
| Counterfactual F1 | 100.00% |

## 4. Answer generation

| Chỉ số | Kết quả |
|---|---:|
| Exact Match | 0.00% |
| Token F1 | 23.74% |
| ROUGE-L | 28.98% |
| BERTScore F1 | N/A |

## 5. Citation và evidence

| Chỉ số | Kết quả |
|---|---:|
| Citation precision | 51.63% |
| Citation recall | 64.59% |
| Citation F1 | 53.92% |
| Evidence coverage | 64.59% |

## 6. Ghi chú

- Số câu có gold path tuyến tính: **236**.
- Số câu counterfactual: **20**.
- BERTScore đang tắt. Thêm `--bertscore` để bật.

## 7. Kết quả theo loại câu hỏi

| Loại câu hỏi | Số câu | Rule Recall@5 | Path Accuracy | Token F1 | Citation F1 |
|---|---:|---:|---:|---:|---:|
| branch_reasoning | 4 | 60.00% | N/A | 24.32% | 39.22% |
| bridge_convergence_fallback | 6 | 100.00% | 83.33% | 10.05% | 90.00% |
| bridge_event | 40 | 81.25% | 50.00% | 12.82% | 68.23% |
| causal_chain_explanation | 35 | 78.57% | 40.00% | 33.59% | 64.92% |
| convergence_reasoning | 10 | 59.33% | N/A | 23.37% | 53.49% |
| forward_multihop | 70 | 69.29% | 35.71% | 28.13% | 45.83% |
| reverse_reasoning | 45 | 46.67% | 8.89% | 17.37% | 34.53% |
| yes_no_counterexample | 20 | 82.50% | 55.00% | 25.78% | 63.93% |
| yes_no_positive | 20 | 67.50% | 45.00% | 29.44% | 60.33% |
