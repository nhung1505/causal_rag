# Báo cáo đánh giá CausalRAG trên BLHS Multi-hop Benchmark

- Thời điểm chạy: `2026-07-25T03:18:51.656869+00:00`
- Tổng số câu benchmark: **250**
- Prediction coverage: **100.00%**

## 1. Retrieval

| Chỉ số | Kết quả |
|---|---:|
| Rule Hit@5 | 82.00% |
| Rule Recall@5 | 53.27% |
| Rule Precision@5 | 22.08% |
| Rule MRR | 74.42% |
| Rule MAP | 47.88% |
| Event Hit@5 | 90.00% |
| Event Recall@5 | 39.73% |
| Event Precision@5 | 24.16% |
| Event MRR | 81.38% |

## 2. Causal path

| Chỉ số | Kết quả |
|---|---:|
| Exact path accuracy | 0.00% |
| Hop accuracy | 15.04% |
| Path edge F1 | 24.01% |
| Path event F1 | 35.42% |

## 3. Counterfactual verification

| Chỉ số | Kết quả |
|---|---:|
| Verification accuracy | 92.00% |
| Counterfactual precision | 0.00% |
| Counterfactual recall | 0.00% |
| Counterfactual F1 | 0.00% |

## 4. Answer generation

| Chỉ số | Kết quả |
|---|---:|
| Exact Match | 0.00% |
| Token F1 | 19.06% |
| ROUGE-L | 20.05% |
| BERTScore F1 | 65.48% |

## 5. Citation và evidence

| Chỉ số | Kết quả |
|---|---:|
| Citation precision | 17.58% |
| Citation recall | 53.52% |
| Citation F1 | 24.97% |
| Evidence coverage | 53.52% |

## 6. Ghi chú

- Số câu có gold path tuyến tính: **236**.
- Số câu counterfactual: **20**.

## 7. Kết quả theo loại câu hỏi

| Loại câu hỏi | Số câu | Rule Recall@5 | Path Accuracy | Token F1 | Citation F1 |
|---|---:|---:|---:|---:|---:|
| branch_reasoning | 4 | 70.00% | N/A | 21.99% | 33.45% |
| bridge_convergence_fallback | 6 | 33.33% | 0.00% | 4.89% | 28.84% |
| bridge_event | 40 | 55.00% | 0.00% | 5.94% | 26.24% |
| causal_chain_explanation | 35 | 64.29% | 0.00% | 23.99% | 25.80% |
| convergence_reasoning | 10 | 28.67% | N/A | 22.42% | 15.43% |
| forward_multihop | 70 | 52.86% | 0.00% | 21.12% | 23.71% |
| reverse_reasoning | 45 | 46.67% | 0.00% | 20.24% | 21.43% |
| yes_no_counterexample | 20 | 82.50% | 0.00% | 28.01% | 30.51% |
| yes_no_positive | 20 | 32.50% | 0.00% | 19.88% | 29.81% |
