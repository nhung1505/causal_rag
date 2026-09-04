# Báo cáo đánh giá CausalRAG trên BLHS Multi-hop Benchmark

- Thời điểm chạy: `2026-07-25T06:50:38.339240+00:00`
- Tổng số câu benchmark: **5**
- Prediction coverage: **100.00%**

## 1. Retrieval

| Chỉ số | Kết quả |
|---|---:|
| Rule Hit@5 | 100.00% |
| Rule Recall@5 | 100.00% |
| Rule Precision@5 | 40.00% |
| Rule MRR | 86.67% |
| Rule MAP | 85.00% |
| Event Hit@5 | 100.00% |
| Event Recall@5 | 100.00% |
| Event Precision@5 | 60.00% |
| Event MRR | 90.00% |

## 2. Causal path

| Chỉ số | Kết quả |
|---|---:|
| Exact path accuracy | 80.00% |
| Hop accuracy | 90.00% |
| Path edge F1 | 90.00% |
| Path event F1 | 93.33% |

## 3. Counterfactual verification

| Chỉ số | Kết quả |
|---|---:|
| Verification accuracy | 80.00% |
| Counterfactual precision | 0.00% |
| Counterfactual recall | 0.00% |
| Counterfactual F1 | 0.00% |

## 4. Answer generation

| Chỉ số | Kết quả |
|---|---:|
| Exact Match | 0.00% |
| Token F1 | 26.86% |
| ROUGE-L | 31.89% |
| BERTScore F1 | N/A |

## 5. Citation và evidence

| Chỉ số | Kết quả |
|---|---:|
| Citation precision | 86.67% |
| Citation recall | 100.00% |
| Citation F1 | 92.00% |
| Evidence coverage | 100.00% |

## 6. Ghi chú

- Số câu có gold path tuyến tính: **5**.
- Số câu counterfactual: **0**.
- BERTScore đang tắt. Thêm `--bertscore` để bật.

## 7. Kết quả theo loại câu hỏi

| Loại câu hỏi | Số câu | Rule Recall@5 | Path Accuracy | Token F1 | Citation F1 |
|---|---:|---:|---:|---:|---:|
| forward_multihop | 5 | 100.00% | 80.00% | 26.86% | 92.00% |
