# Cập nhật yêu cầu từ người dùng (2026-11-11)

Các điểm dưới đây THAY THẾ brief gốc của workflow ở mọi chỗ mâu thuẫn (số frame, ngày, placeholder GVHD, văn phong).

## 1. Thời lượng
- Trình bày khoảng **30 phút** → khoảng **30–36 frame nội dung** (không tính intro, trang tiêu đề, mục lục, trang chia phần, trang cảm ơn).
- Mốc "18–22 frame" trong brief gốc không còn áp dụng.

## 2. Trang tiêu đề (không còn placeholder)
- Học viên: **Nguyễn Thị Nhung -- 20251329M**
- Giảng viên hướng dẫn: **TS. Nguyễn Quốc Tuấn** và **PGS.TS. Nguyễn Thị Kim Anh**
- Chuyên ngành: **Thạc sĩ Khoa học Máy tính**
- **Đại học Bách khoa Hà Nội**
- Thời gian báo cáo: **Tháng 12 năm 2026** (thay cho "Tháng 11 năm 2026")
- Tên đề tài: vẫn lấy từ bìa `luan_van_ver1.pdf` (ưu tiên) hoặc `paper/vi/main.tex`.
- Trang tiêu đề phải hiển thị gọn, không tràn (kiểm tra bằng ảnh render).

## 3. Vị trí
- Giữ bộ slide mới ở `slides/luan_van/`. Không sửa `slides/main.tex` và thư mục template.

## 4. Văn phong: đậm chất nghiên cứu khoa học, nhưng sáng tạo và dễ hiểu
**Tính khoa học**
- Thuật ngữ và ký hiệu chính xác, nhất quán với `paper/vi`.
- Nêu rõ câu hỏi nghiên cứu (RQ) và đóng góp ở phần Giới thiệu; phần Kết luận trả lời lại từng RQ bằng số liệu.
- Mỗi kết luận đi kèm số liệu hoặc kiểm định (CI, p-value) lấy từ file nguồn.
- Trích dẫn ngắn (tác giả, năm) cho nghiên cứu liên quan và nền tảng lý thuyết.
- Có phần hạn chế / threats to validity.

**Tính dễ hiểu và sáng tạo**
- Một **ví dụ xuyên suốt**: chọn một câu hỏi thật trong benchmark, theo nó qua từng bước pipeline (graph → memory → retrieval → verification → answer). Ghi rõ sample id.
- "Trực giác trước, công thức sau": mỗi công thức có một dòng diễn giải bằng lời.
- Hộp "Thông điệp chính" ngắn ở các slide quan trọng.
- Sơ đồ TikZ với màu nhất quán cho từng bước pipeline; có lộ trình nhỏ cho biết đang ở bước nào.
- So sánh trước/sau trên cùng một câu hỏi (ví dụ LLM-only so với Full LegalSCM).
- Sáng tạo qua bố cục, sơ đồ, màu và ví dụ; giữ nghiêm túc: không emoji, không tải clipart/hình từ internet (tự vẽ bằng TikZ/pgfplots).
- Overlay (`\pause`) chỉ dùng hạn chế.
- Công thức quá nặng có thể chuyển sang phần **Phụ lục (backup)** đặt sau trang cảm ơn để dùng khi hỏi đáp.
