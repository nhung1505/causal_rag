ssh nhung
conda activate nhungnt
cd workspace/nhungnt/causal_rag/
ollama serve

chạy file 1
python 1_build_legal_causal_graph.py \
  --input data/blhs_rules_final_all_normalized.json


  file 2:
  python 2_build_causal_memory.py

  or 

  python 2_build_causal_memory.py \
  --input data/blhs_rules_final_all_normalized.json \
  --graph data/legal_causal_knowledge_graph.graphml

  or (bỏ event memory)

  python 2_build_causal_memory.py --rule-only