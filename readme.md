ssh nhung
conda activate nhungnt
cd workspace/nhungnt/causal_rag/
ollama serve

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