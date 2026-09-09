ssh nhung
conda activate nhungnt
cd workspace/nhungnt/causal_rag/
ollama serve

python -m py_compile 3_multi_hop_causal_retriever.py
python -m py_compile 4_counterfactual_verification.py

** chạy file 1
python 1_build_legal_causal_graph.py \
  --input data/blhs_rules_final_all_normalized.json


** chạy file 2:
python 2_build_causal_memory.py

or 

python 2_build_causal_memory.py \
--input data/blhs_rules_final_all_normalized.json \
--graph data/legal_causal_knowledge_graph.graphml

or (bỏ event memory)

python 2_build_causal_memory.py --rule-only

** chạy file 3
python 3_multi_hop_causal_retriever.py \
  "Người phạm tội chưa đạt phải chịu trách nhiệm hình sự như thế nào?"

or

python 3_multi_hop_causal_retriever.py \
  "Điều kiện để được xóa án tích là gì?" \
  --max-hops 2 \
  --event-top-k 8 \
  --final-top-k 12

** chạy file 4
python 4_counterfactual_verification.py

or

python 4_counterfactual_verification.py \
  --retrieval-result data/retrieval_result.json \
  --max-cf-hops 3 \
  --cf-top-k 5 \
  --mapping-threshold 0.42 \
  --keep-threshold 0.52 \
  --reject-threshold 0.34


rm -f data/pipeline_predictions.json
rm -f data/pipeline_predictions.jsonl
rm -f data/pipeline_errors.json
rm -f data/pipeline_run_log.json
rm -rf data/pipeline_intermediate

python 5_5_generate_pipeline_predictions.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --provider extractive \
  --limit 5 \
  --disable-semantic-mapping \
  --keep-intermediate

  python 5_5_generate_pipeline_predictions.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --provider ollama \
  --answer-model qwen3:8b \
  --disable-semantic-mapping \
  --resume


  python 6_run_evaluation_pipeline.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --oracle \
  --output-dir evaluation_results_oracle

  python 6_run_evaluation_pipeline.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --predictions data/pipeline_predictions.json \
  --output-dir evaluation_results


  -----

  python -m py_compile \
  4_counterfactual_verification.py \
  5_5_generate_pipeline_predictions.py
bash

python 5_5_generate_pipeline_predictions.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --counterfactual-mode structural_scm \
  --rules data/blhs_rules_final_all_normalized.json \
  --provider extractive \
  --disable-semantic-mapping \
  --output data/pipeline_predictions_structural_scm_v41.json \
  --jsonl-output data/pipeline_predictions_structural_scm_v41.jsonl \
  --errors-output data/pipeline_errors_structural_scm_v41.json \
  --run-log data/pipeline_run_log_structural_scm_v41.json \
  --work-dir data/pipeline_intermediate_structural_scm_v41 \
  --resume
bash

python 5_5_generate_pipeline_predictions.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --counterfactual-mode path_ablation \
  --rules data/blhs_rules_final_all_normalized.json \
  --provider extractive \
  --disable-semantic-mapping \
  --output data/pipeline_predictions_path_ablation_v41.json \
  --jsonl-output data/pipeline_predictions_path_ablation_v41.jsonl \
  --errors-output data/pipeline_errors_path_ablation_v41.json \
  --run-log data/pipeline_run_log_path_ablation_v41.json \
  --work-dir data/pipeline_intermediate_path_ablation_v41 \
  --resume
bash

python 7_compute_evaluation_metrics.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --predictions data/pipeline_predictions_structural_scm_v41.json \
  --output-dir evaluation_metrics_structural_scm_v41 \
  --strict
bash

python 7_compute_evaluation_metrics.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --predictions data/pipeline_predictions_path_ablation_v41.json \
  --output-dir evaluation_metrics_path_ablation_v41 \
  --strict


---

## System-level baselines (Ollama `qwen3:8b`)

Dùng `10_generate_system_baselines.py` cho thí nghiệm chính. Runner này giữ nguyên một generation contract cho cả bốn cấu hình: cùng system prompt, JSON schema, parser, citation policy, `qwen3:8b`, `temperature=0.1`, `max_tokens=1200`, `num_ctx=32768`, seed 42, context budget 18.000 ký tự và không có extractive fallback. Chỉ chiến lược tạo context/verification thay đổi.

Bốn mode:

- `llm_only`: không retrieval, graph hoặc verifier.
- `vanilla_rag`: BGE-M3 truy hồi top-5 điều luật thô, không graph.
- `causal_path_rag`: chạy Step 3 và đưa causal path/rule evidence cho LLM; hoàn toàn không gọi Step 4.
- `full_legal_scm`: Step 3 → query-aware Step 4 `structural_scm`/LegalSCM → cùng LLM.

Chạy từ thư mục gốc repo trên server:

```bash
python 10_generate_system_baselines.py --mode llm_only --model qwen3:8b --resume
python 10_generate_system_baselines.py --mode vanilla_rag --model qwen3:8b --resume
python 10_generate_system_baselines.py --mode causal_path_rag --model qwen3:8b --resume
python 10_generate_system_baselines.py --mode full_legal_scm --model qwen3:8b --resume
```

Output mặc định:

```text
data/baselines/system_level/llm_only_qwen3_8b_predictions.json
data/baselines/system_level/vanilla_rag_qwen3_8b_predictions.json
data/baselines/system_level/causal_path_rag_qwen3_8b_predictions.json
data/baselines/system_level/full_legal_scm_qwen3_8b_predictions.json
```

Mỗi mode còn tạo checkpoint `.jsonl`, error file và run log cùng prefix. `--resume` chỉ tiếp tục khi configuration fingerprint khớp run cũ. Dùng `--overwrite` nếu chủ động muốn chạy lại từ đầu.

### LegalSCM fallback policy

Mode `full_legal_scm` mặc định không cho node-deletion fallback âm thầm được tính như kết quả LegalSCM. Nếu primary path dùng structural fallback, effective decision được chuyển thành `UNCERTAIN` và audit metadata ghi `STRUCTURAL_FALLBACK_BLOCKED`. Chỉ dùng `--allow-scm-fallback` cho phân tích chẩn đoán, không dùng cho bảng main comparison.

### So sánh bốn hệ thống

Sau khi cả bốn prediction file hoàn tất không lỗi:

```bash
python 11_compare_system_baselines.py \
  --predictions \
    llm_only=data/baselines/system_level/llm_only_qwen3_8b_predictions.json \
    vanilla_rag=data/baselines/system_level/vanilla_rag_qwen3_8b_predictions.json \
    causal_path_rag=data/baselines/system_level/causal_path_rag_qwen3_8b_predictions.json \
    full_legal_scm=data/baselines/system_level/full_legal_scm_qwen3_8b_predictions.json \
  --reference full_legal_scm \
  --require-complete-success
```

Comparator từ chối các file không dùng cùng generation contract. Shared metrics (decision, answer, citation) được tính trên toàn bộ gold samples; prediction thiếu/lỗi nhận 0 thay vì bị loại khỏi mẫu số. Retrieval/path metric không áp dụng được ghi `N/A`, không ghi 0. Linear path metric loại 14 mẫu branch/convergence không có `gold_path` tuyến tính.

Artifact so sánh mặc định:

```text
evaluation_results/system_level_comparison/system_comparison_report.json
evaluation_results/system_level_comparison/system_comparison_by_sample.csv
evaluation_results/system_level_comparison/system_comparison_significance.csv
evaluation_results/system_level_comparison/system_comparison_summary.md
```

Báo cáo gồm bootstrap 95% CI, exact McNemar cho decision accuracy, paired randomization cho các metric liên tục và Holm correction. `7_compute_evaluation_metrics.py` vẫn được giữ cho diagnostic chi tiết của từng causal pipeline/artifact cũ; dùng `11_compare_system_baselines.py` cho bảng so sánh cấp hệ thống trong paper.
