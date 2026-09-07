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