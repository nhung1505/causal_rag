# Tiến độ slide luận văn — HOÀN THÀNH (2026-11-12)

- Làm trực tiếp trong session, KHÔNG dùng workflow (yêu cầu của người dùng).
- Bộ slide: `slides/luan_van/` → `main.pdf` (55 trang PDF). Biên dịch bằng pdfLaTeX: `latexmk -pdf main.tex`
  (Overleaf: chọn compiler pdfLaTeX).
- Kiểm tra lại bố cục sau khi sửa: trong `slides/luan_van/` chạy `zsh build/build.sh`
  (biên dịch, render PNG vào `build/render`, kiểm tra nội dung tràn ra ngoài vùng an toàn).
  Xem nhanh 4 trang: `/usr/bin/python3 build/sheet.py s.png 13 14 15 16`.

## Cấu trúc
- Trang intro, trang tiêu đề, mục lục, 5 trang chia phần, trang cảm ơn, trang chia "Phụ lục".
- 36 frame nội dung: Giới thiệu 7, Dữ liệu và phương pháp 15, Thực nghiệm 3, Kết quả 8, Kết luận 3.
- Phụ lục 8 frame (A1–A7, A7 tách 2 trang) đặt sau trang cảm ơn, dùng khi hỏi đáp.
- Ví dụ xuyên suốt: `BLHS_MH_0081`; ví dụ phụ: `BLHS_MH_0221` (DIRECT), `BLHS_MH_0223` (can thiệp).
- Một `\pause` duy nhất ở slide "Ví dụ xuyên suốt: một câu hỏi, năm câu trả lời".

## Kiểm tra đã làm
- Không lỗi biên dịch, không tham chiếu thiếu; 55/55 trang đạt kiểm tra bố cục; đã xem ảnh từng trang.
- Số liệu đối chiếu lại với `evaluation_results/system_level_comparison_v2/*`, `data/legal_causal_graph_stats.json`,
  `data/blhs_rules_final_all_normalized.json`, `data/1_raw_data.json`, `data/causal_memory.csv`,
  `data/baselines/*_predictions.json`, run log của Full, metadata benchmark, mã `3_multi_hop_causal_retriever.py`.
- `beamerthemeHUST.sty` giống hệt template; không còn nội dung CNN/Oxford Pet.

## Điểm cần người dùng quyết định
1. Thứ tự GVHD: bìa `luan_van_ver1.pdf` ghi "PGS.TS. Nguyễn Thị Kim Anh & TS. Nguyễn Quốc Tuấn";
   slide đang theo thứ tự người dùng nhắn (TS. Nguyễn Quốc Tuấn trước). Sửa ở `\thesisAdvisors` trong `main.tex`.
2. Gộp modality "có thể được" với "được" khi chuẩn hóa sự kiện: chưa có trong paper/luận văn nên không đưa lên slide.
3. Bài báo nói thiếu hỗ trợ → UNKNOWN, còn code đặt sự kiện nội sinh không có rule kích hoạt → FALSE;
   slide theo code (F2.9, A2).
4. Không trình bày Causal Path RAG như CausalRAG của Wang và cs. (2025).
5. Repo: `causal_core/` đang bị xóa trong git và có thư mục tên `|` chứa `legal_scm.py`, `schema.py` (chưa đụng tới).
