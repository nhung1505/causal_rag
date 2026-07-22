VIETNAMESE LEGAL CAUSALRAG EVALUATION DATASET
================================================

Nguồn: Văn bản đã dán (1)(13).txt
Số causal chains: 35
Số mẫu: 70

Phân bố:
- Factual multi-hop: 35
- Counterfactual path invalidation: 12
- Counterfactual có alternative candidate: 23
- Mẫu 3-hop: 8
- Mẫu 2-hop: 62

Các trường chính:
- question_id: mã câu hỏi
- task_type: loại nhiệm vụ
- question: câu hỏi tiếng Việt
- gold_rule_ids: rule chuẩn
- gold_article_ids: điều luật chuẩn
- gold_path.nodes: chuỗi node chuẩn
- gold_path.edges: các cạnh chuẩn
- intervention: can thiệp counterfactual
- gold_counterfactual_status: trạng thái chuẩn
- gold_final_effect_reachable: hậu quả cuối còn reachable hay không
- gold_answer_label: nhãn đáp án
- gold_answer: đáp án tham chiếu
- alternative_candidates: chain thay thế tiềm năng, luôn cần xác minh điều kiện

Lưu ý:
1. Dataset bám sát causal chains đã trích xuất, không tự sửa hoặc xác minh lại luật gốc.
2. INVALIDATED nghĩa là chain đang xét bị vô hiệu, không đồng nghĩa hậu quả pháp lý tuyệt đối không thể xảy ra.
3. Alternative candidate không phải evidence hợp lệ nếu điều kiện chưa được xác minh.
