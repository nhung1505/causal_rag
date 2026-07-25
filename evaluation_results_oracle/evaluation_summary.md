# Báo cáo đánh giá CausalRAG trên BLHS Multi-hop Benchmark

- Thời điểm chạy: `2026-07-24T01:51:15.848123+00:00`
- Tổng số câu benchmark: **250**
- Prediction coverage: **100.00%**

## 1. Retrieval

| Chỉ số | Kết quả |
|---|---:|
| Rule Hit@5 | 100.00% |
| Rule Recall@5 | 100.00% |
| Rule Precision@5 | 100.00% |
| Rule MRR | 100.00% |
| Rule MAP | 100.00% |
| Event Hit@5 | 100.00% |
| Event Recall@5 | 99.80% |
| Event Precision@5 | 100.00% |
| Event MRR | 100.00% |

## 2. Causal path

| Chỉ số | Kết quả |
|---|---:|
| Exact path accuracy | 100.00% |
| Hop accuracy | 100.00% |
| Path edge F1 | 100.00% |
| Path event F1 | 100.00% |

## 3. Counterfactual verification

| Chỉ số | Kết quả |
|---|---:|
| Verification accuracy | 100.00% |
| Counterfactual precision | 100.00% |
| Counterfactual recall | 100.00% |
| Counterfactual F1 | 100.00% |

## 4. Answer generation

| Chỉ số | Kết quả |
|---|---:|
| Exact Match | 100.00% |
| Token F1 | 100.00% |
| ROUGE-L | 100.00% |
| BERTScore F1 | N/A |

## 5. Citation và evidence

| Chỉ số | Kết quả |
|---|---:|
| Citation precision | 100.00% |
| Citation recall | 100.00% |
| Citation F1 | 100.00% |
| Evidence coverage | 100.00% |

## 6. Ghi chú

- Số câu có gold path tuyến tính: **236**.
- Số câu counterfactual: **20**.
- BERTScore đang tắt. Thêm `--bertscore` để bật.

## 7. Kết quả theo loại câu hỏi

| Loại câu hỏi | Số câu | Rule Recall@5 | Path Accuracy | Token F1 | Citation F1 |
|---|---:|---:|---:|---:|---:|
| branch_reasoning | 4 | 100.00% | N/A | 100.00% | 100.00% |
| bridge_convergence_fallback | 6 | 100.00% | 100.00% | 100.00% | 100.00% |
| bridge_event | 40 | 100.00% | 100.00% | 100.00% | 100.00% |
| causal_chain_explanation | 35 | 100.00% | 100.00% | 100.00% | 100.00% |
| convergence_reasoning | 10 | 100.00% | N/A | 100.00% | 100.00% |
| forward_multihop | 70 | 100.00% | 100.00% | 100.00% | 100.00% |
| reverse_reasoning | 45 | 100.00% | 100.00% | 100.00% | 100.00% |
| yes_no_counterexample | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
| yes_no_positive | 20 | 100.00% | 100.00% | 100.00% | 100.00% |
