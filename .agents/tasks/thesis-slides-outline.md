# YÊU CẦU CẬP NHẬT TỪ NGƯỜI DÙNG (ưu tiên cao nhất)

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

---

# Thesis defense deck — frame-by-frame outline and implementation plan

- **Project:** LegalSCM-RAG (master's thesis, HUST). **Deliverable:** `slides/luan_van/` (Beamer, HUST template copied from `slides/slide_hoc_sau_va_ung_dung/`).
- **Repo root:** `/Users/nhungnguyen/Documents/cao_hoc/do_an/causal_rag` (all paths below are relative to it unless absolute).
- **Planner scope:** this file only. The planner did not modify anything else and made no git commit. The coder must not modify `slides/main.tex`, `slides/slide_hoc_sau_va_ung_dung/**`, or anything outside `slides/luan_van/`, and must not commit unless the workflow explicitly says so.
- **Language rule:** instructions in this file are in English; every string that appears on a slide is given in Vietnamese with full diacritics and must be copied as written (minor line-breaking is fine; changing wording or numbers is not).
- **Pre-existing repo state (do not touch):** `git status` shows `causal_core/` deleted and an untracked folder literally named `|` that contains `legal_scm.py` and `schema.py` (the old `causal_core` files). It is unrelated to the slides; it is cited below as `|/legal_scm.py` and must be reported to the user in the final report.

## 0. Source keys (every number on a slide must come from one of these)

| Key | File |
|---|---|
| S-sum | `evaluation_results/system_level_comparison_v2/system_comparison_summary.md` |
| S-sig | `evaluation_results/system_level_comparison_v2/system_comparison_significance.csv` |
| S-rep | `evaluation_results/system_level_comparison_v2/system_comparison_report.json` |
| S-bys | `evaluation_results/system_level_comparison_v2/system_comparison_by_sample.csv` |
| G | `data/legal_causal_graph_stats.json` |
| B | `data/blhs_multihop_benchmark_250.json` |
| R | `data/blhs_rules_final_all_normalized.json` (2,884 rules) |
| RAW | `data/1_raw_data.json` (406 articles) |
| MEM | `data/causal_memory.csv` (4,698 rows) and `data/causal_memory_embeddings.npy` (shape 4698×1024) |
| P-full / P-cpr / P-van / P-cba / P-cbf | `data/baselines/{full_legal_scm, causal_path_rag, vanilla_rag, llm_only, llm_only_forced_binary}_qwen3_8b_predictions.json` |
| LOG-full | `data/baselines/full_legal_scm_qwen3_8b_run_log.json` |
| PV1…PV7, PVR, PVM | `paper/vi/1_INTRODUCTION.tex` … `7_conclusion.tex`, `References.tex`, `main.tex` |
| PE1 | `paper/en/1_INTRODUCTION.tex` |
| LV1 | `luan_van_ver1.pdf` (35-page proposal-style slide deck; page numbers given as "tr. N") |
| DECK | `slides/main.tex` (user's minimal 10-frame deck; "frame N") |
| A1 / A2 | `a1_extract_rules.py` / `a2_normalize_events_fixed.py` |
| C2 / C3 | `2_build_causal_memory.py` / `3_multi_hop_causal_retriever.py` |
| SCM | `|/legal_scm.py` (formerly `causal_core/legal_scm.py`) |
| AN | `.agents/tasks/source-code-analysis-2026-10-06.md` |
| README | `readme.md` |

"computed from X" means a simple count/division over that file, done by the planner; the exact value is given here so the coder does not recompute.

## 1. Decisions (each with a short rationale)

**D1 — Title page content.**
- `\title`: **Cải thiện Multi-hop Question Answering bằng cách kết hợp Causal Graph và Counterfactual Reasoning trong RAG** (official title from the LV1 cover, tr. 2; the user's update says the cover title has priority).
- `\subtitle`: **LegalSCM-RAG cho hỏi đáp pháp luật đa bước tiếng Việt** (a shortened form of the paper title, so the audience links the thesis title to the system name used on every slide).
- The paper title (PVM) is recorded but not shown in full: *LegalSCM-RAG: Truy xuất đường dẫn nhân quả và xác minh cấu trúc chọn lọc cho hỏi đáp pháp luật đa bước tiếng Việt*.
- `\author{Nguyễn Thị Nhung -- 20251329M}`; advisors line: **Giảng viên hướng dẫn: TS. Nguyễn Quốc Tuấn và PGS.TS. Nguyễn Thị Kim Anh** (order and wording from the user update; LV1 lists the opposite order, see conflict C2); **Chuyên ngành: Thạc sĩ Khoa học Máy tính** (user update; LV1 says "Khoa học máy tính"); `\institute{Đại học Bách khoa Hà Nội}`; `\date{Tháng 12 năm 2026}`. No placeholders anywhere.
- The template's hard-coded **"Nhóm 7:"** label is removed by overriding `\husttitlepage` in `main.tex` (code in §3.4). Rationale: single-author thesis; overriding in `main.tex` keeps `beamerthemeHUST.sty` byte-identical to the template.
- Title-page geometry was prototyped on this machine (pdflatex, TeX Live 2025): with `text width=0.80\paperwidth`, `\titlePageTitleX=-1.0cm`, `\titlePageTitleY=-0.45cm`, `\Large` uppercase title (3 lines), `\normalsize` subtitle and a `\small` 4-line info block, the text block occupied y ≈ 2.64–7.23 cm, x 0.6–13.05 cm, inside the free area of `hust-blue-169-title.png` (free area measured: x 0.5–13.4 cm, y 2.1–7.6 cm); the advisors line stayed on one line. The template's own `\LARGE` would need 4 title lines and overflow, hence `\Large`. All `latex` code blocks of §3 were compiled together in a scratch copy outside the repo (exit 0, no errors, no overfull vbox) and the §10 checker reported 0 problems on that build.

**D2 — Size and timing.** 35 content frames (S1: 7, S2: 14, S3: 3, S4: 8, S5: 3) for a 30-minute talk (~51 s per frame), plus intro, title, TOC, 5 section pages and the thank-you page; 7 backup frames after the thank-you page. Timing guide: S1 ≈ 5.5 min, S2 ≈ 12 min, S3 ≈ 2.5 min, S4 ≈ 7.5 min, S5 ≈ 2.5 min. Only one frame may be split if it overflows (F2.3 → F2.3a/b, giving 36); any other overflow must be solved by moving material to the appendix, never by going above 36 or shrinking text below `\scriptsize`.

**D3 — Running example: `BLHS_MH_0081`.** Reverse-reasoning, 2-hop question about excessive self-defence (Điều 22 → Điều 51). Chosen because: (i) it is one of the 23 samples where Full LegalSCM is right and Causal Path RAG is wrong (S-bys); (ii) Step 3 retrieves the exact gold path (`top1_exact_path_match = 1.0`) and both systems cite exactly Điều 22 and Điều 51 (citation F1 = 1.0), so the only difference is Step 4 — the clearest possible illustration of the thesis claim; (iii) every intermediate value (memory ids, 33 candidate paths, path score, fixed point, commit gate) is stored in P-full; (iv) the legal content (phòng vệ chính đáng) is intuitive for a general committee; (v) all events involved have modality `UNSPECIFIED`, so the example does not touch the modality-merge issue (see D11). Two auxiliary real samples from the same legal neighbourhood are used where the running example cannot show a route: `BLHS_MH_0221` (DIRECT route) and `BLHS_MH_0223` (INTERVENTION route). Full data in §6.

**D4 — Figures.** All figures are drawn with TikZ/pgfplots; no image files except the theme backgrounds. `slides/slide_hoc_sau_va_ung_dung/figures/causalrag_architecture.png` is **not reused**: it is the CausalRAG architecture of Wang và cs. (2025) (Upload Doc → text-based graph construction → initial/extended nodes → causal summary → LLM), it has no rule extraction, no LegalSCM verification and no commit gate, and the user forbids figures taken from outside sources. The other template figures (CNN/Grad-CAM) are unrelated and are not copied.

**D5 — Theme adaptation.** Copy `beamerthemeHUST.sty` verbatim (must stay byte-identical, checked with `cmp`). All thesis-specific changes live in `main.tex`: title-page override (D1), frametitle override that adds the roadmap (D6), colours and macros (§3).

**D6 — Pipeline roadmap ("lộ trình nhỏ").** Six small boxes (Dữ liệu, Đồ thị, Bộ nhớ, Truy xuất, Xác minh, Trả lời) drawn in the footer strip, below the footer line, by the frametitle template whenever `\setroadmap{k}` (k = 0…5) is active; `\setroadmap{-1}` turns it off. Prototype measurement: boxes at y 8.50–8.81 cm, x 8.06–14.72 cm, clear of the footer line (y 8.38–8.40 cm), the footer logo (x ≤ 2.01 cm) and the page number (x ≈ 15.2–15.5 cm), and it does not move any frame content. Rationale: a header-bar roadmap would collide with the assertion-style frame titles (titles up to ~60 characters span x 0.5–14.3 cm, measured).

**D7 — "Thông điệp chính" box.** Macro `\keymsg{…}`: a rounded `beamercolorbox` (gold tint) placed **in the normal flow** after the main content, preceded by `\vspace{0.15cm}` (or `\vfill` when the frame is short). Do **not** push it to the bottom with `\vspace{0pt plus 1filll}`: the prototype showed the frame body extends to y ≈ 8.9 cm, so a bottom-anchored box covers the footer line and roadmap. Safe content bottom: y ≤ 8.0 cm.

**D8 — Number format (Vietnamese convention, as in PV and the template).** Decimal comma, thousands dot: `96,80%`, `5.134`, `14.420`, `0,705`. In math mode write `0{,}705`, `8{,}8\times10^{-5}`, intervals `[4{,}4;\,12{,}0]` (semicolon separator, as in PV6). pgfplots uses `use comma, 1000 sep={.}`. Times: `3,238 giây`.

**D9 — Notation and names (consistent with paper/vi).** System names exactly: **Closed-book (abstain)**, **Closed-book (forced)**, **Vanilla RAG**, **Causal Path RAG**, **Full LegalSCM** (PV6 table names). In running text, closed-book = "LLM chỉ nhận câu hỏi". Labels are typeset in typewriter via macros (`\labSup`, `\labRej`, `\labUnc`) because `\textsc` has no small caps in T5 sans and would print lowercase. Use `do(X_m=0)` (PV3 body notation; the abstract's `do(X_M=0)` is not used). Metric names in English as in PV: Decision Accuracy, Balanced Accuracy, Macro-F1, Answer F1 (Token-F1), ROUGE-L, Citation F1, Recall@5, MRR, MAP, Top-1/Oracle Exact Path. Abbreviations defined on F1.2 and reused: PVCĐ = phòng vệ chính đáng, TNHS = trách nhiệm hình sự, BLHS = Bộ luật Hình sự. Short citations as "(Tác giả và cs., năm)", following PV's "và cộng sự".

**D10 — Colours (fixed palette).**
- Pipeline steps (diagrams, roadmap, running-example boxes on method frames): B0 Dữ liệu `stepData`=schemaGray `5A6872`; B1 Đồ thị `stepGraph`=schemaCyan `00A6D6`; B2 Bộ nhớ `stepMemory`=schemaGreen `1C8C66`; B3 Truy xuất `stepRetrieve`=schemaGold `D89B16`; B4 Xác minh `stepVerify`=`7A3E9D` (new purple); B5 Trả lời `stepAnswer`=schemaNavy `073B5C`.
- Systems (charts): Closed-book (abstain) `schemaGray!45`, Closed-book (forced) `schemaGray!80`, Vanilla RAG `hustprimary!55`, Causal Path RAG = `stepRetrieve`, Full LegalSCM = `stepVerify` (a system takes the colour of the most advanced pipeline step it has).
- Decision labels: SUPPORTED green (`schemaGreen`), REJECT_DIRECT_CLAIM red (`schemaRed`), UNCERTAIN gray (`schemaGray`).
- Template blocks keep their template colours: `block` (navy title / light body) for definitions and inputs, `alertblock` (red) for cautions and limitations, `exampleblock` (green) only for the Full LegalSCM result on F4.6.

**D11 — Limitations decision (brief item (a)).** The limitations frame (F5.2) is based on PV7 and DECK frame 10: internal validity (questions generated from the same graph), only 7 intervention cases (both systems 7/7), single-antecedent data so conjunction is not validated, path ranking bottleneck, runtime overhead, answer/citation not improved vs Causal Path RAG, deterministic normative model without abduction. The modality issue flagged in AN (H-1: "có thể được" merged with "được", possibility treated as fact) is **not discussed in paper/vi, paper/en or LV1** (searched: no occurrence of modality/"có thể được"/POSSIBLE in `paper/`; LV1 text has none). Therefore it is **left off all slides, including the appendix**, and listed in §9 for the final report. Consequence for the coder: never show bridge-event names containing "||" or "Có thể được …" (e.g. "Được xóa án tích || Có thể được xóa án tích"), and do not mention the 241 remapped endpoints or the 89 rules with `POSSIBLE_*` modality.

**D12 — Semantics wording (paper vs implementation, conflict C7).** PV3 says "không có hỗ trợ hoặc có hỗ trợ xung đột tạo trạng thái unknown", but the implementation (SCM docstring, lines 7–11) sets endogenous events with no active rule to FALSE (closed world) and keeps root events UNKNOWN until given. The intervention example (`BLHS_MH_0223`, outcome TRUE → FALSE) depends on the implemented behaviour. Slides describe the implemented convention explicitly on F2.9 (one bullet) and in appendix A2; the discrepancy goes to §9 so the user can align the paper.

**D13 — Baseline naming (conflict C5).** LV1 (tr. 12, 15, 34) promises a comparison with "CausalRAG"; the experiments compare with **Causal Path RAG**, which is this pipeline's Step 3 without Step 4 (README; PV5), not the implementation of Wang và cs. (2025). Slides must never claim a comparison against Wang và cs.'s system; RQ4 is reworded accordingly.

**D14 — Overlays.** Exactly one `\pause` in the whole deck, on F4.6 (reveals the Full LegalSCM block after the four baselines). Rationale: the theme prints `\insertpagenumber`, so every overlay adds a PDF page and makes page numbers jump.

**D15 — Appendix.** After `\hustthankyou`: `\appendix`, `\showPageNumber`, then `\input{6_phu_luc}` which starts with `\section{Phụ lục}` (produces a "Phụ lục" section page via the theme's `\AtBeginSection`). Prototype-verified: because `\appendix` opens a new part, the TOC frame at the start lists only the 5 main sections.

## 2. Files to create (all inside `slides/luan_van/`)

| File | Content |
|---|---|
| `beamerthemeHUST.sty` | Byte-identical copy of `slides/slide_hoc_sau_va_ung_dung/beamerthemeHUST.sty`. |
| `images/hust-blue-169-intro.png` | Used by `\hustintropage`. |
| `images/hust-blue-169-title.png` | Used by the (overridden) `\husttitlepage`. |
| `images/hust-blue-169-content-full.png` | Default background set at package load by `\useStyleFull` (thin bar); used by the TOC frame, which comes before `\useStyleHustFull`. |
| `images/hust-blue-169-content-hust-full.png` | Background of all content frames after `\useStyleHustFull` + `\useThinBar`. |
| `images/hust-blue-169-section.png` | Section divider pages (`\AtBeginSection` → `\hustsectionpage`). |
| `images/hust-blue-169-end.png` | `\hustthankyou`. |
| `main.tex` | Preamble (§3), title data, document skeleton (§3.6). |
| `1_gioi_thieu.tex` | `\section{Giới thiệu}` + F1.1–F1.7. |
| `2_du_lieu_va_phuong_phap.tex` | `\section{Dữ liệu và phương pháp}` + F2.1–F2.14. |
| `3_thuc_nghiem_va_danh_gia.tex` | `\section{Thực nghiệm và đánh giá}` + F3.1–F3.3. |
| `4_ket_qua_va_phan_tich.tex` | `\section{Kết quả và phân tích}` + F4.1–F4.8. |
| `5_ket_luan.tex` | `\section{Kết luận}` + F5.1–F5.3. |
| `6_phu_luc.tex` | `\section{Phụ lục}` + A1–A7 (backup). |
| `build/` (scratch) | Rendered PNGs and `check_layout.py` for QA (§10). Not part of the deliverable. |

The 6 PNGs are the only images the theme loads in this configuration (`\hustbg{x}` = `hust-blue-169-x.png`, `\graphicspath{{images/}}`). Not needed: `*-thick.png` (only with `\useThickBar`), `*-end-empty.png` (only `\hustcontactpage`), `*-content-logo.png`, `*-content-hust.png`, all `red`/`43` variants, `scratch_cnn-1.png`. Copy only the six.

## 3. `main.tex` specification

### 3.1 Preamble (template preamble kept, additions marked NEW)
```latex
\documentclass[t,aspectratio=169]{beamer}
\usepackage[utf8]{vietnam}            % keep: vntex/T5; do NOT add inputenc/fontenc/babel
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\usepackage{booktabs,array,multirow}
\usepackage{colortbl}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit,backgrounds}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{pgfplots}                                             % NEW
\pgfplotsset{compat=1.18,every axis/.append style={font=\scriptsize}}  % NEW
\pgfkeys{/pgf/number format/.cd,use comma,1000 sep={.}}           % NEW (D8)

\def\slideContentLeftMargin{0.7cm}
\def\slideContentRightMargin{0.9cm}
\def\hustHeaderHeight{0.9cm}
\def\slideContentBotMargin{3.5cm}
\usepackage[blue,169]{beamerthemeHUST}

\renewcommand{\titlePageTitleX}{-1.0cm}    % changed from -0.5cm (D1)
\renewcommand{\titlePageTitleY}{-0.45cm}   % changed from -0.5cm (D1)
\renewcommand{\hustTitleAlign}{\alignLeft}
\renewcommand{\hustSubtitleAlign}{\alignLeft}
\renewcommand{\hustAuthorAlign}{\alignLeft}
\renewcommand{\hustInstAlign}{\alignLeft}
```
Then the template's colour block unchanged (`schemaCyan`, `schemaNavy`, `schemaRed`, `schemaGreen`, `schemaGold`, `schemaLight`, `schemaGray`, the six `\setbeamercolor{block …}` lines and `alerted text`), and the template's `\code`, `\metric`, `\stepbox` macros unchanged.

### 3.2 NEW colours and small macros
```latex
\colorlet{stepData}{schemaGray}
\colorlet{stepGraph}{schemaCyan}
\colorlet{stepMemory}{schemaGreen}
\colorlet{stepRetrieve}{schemaGold}
\definecolor{stepVerify}{HTML}{7A3E9D}
\colorlet{stepAnswer}{schemaNavy}
\colorlet{sysCBa}{schemaGray!45}      % Closed-book (abstain)
\colorlet{sysCBf}{schemaGray!80}      % Closed-book (forced)
\colorlet{sysVan}{hustprimary!55}     % Vanilla RAG
\colorlet{sysCPR}{stepRetrieve}       % Causal Path RAG
\colorlet{sysFull}{stepVerify}        % Full LegalSCM

\newcommand{\labSup}{\textcolor{schemaGreen}{\texttt{SUPPORTED}}}
\newcommand{\labRej}{\textcolor{schemaRed}{\texttt{REJECT\_DIRECT\_CLAIM}}}
\newcommand{\labRejS}{\textcolor{schemaRed}{\texttt{REJECT}}}   % only in narrow tables, with a legend line
\newcommand{\labUnc}{\textcolor{schemaGray}{\texttt{UNCERTAIN}}}
\newcommand{\ok}{\textcolor{schemaGreen}{$\checkmark$}}
\newcommand{\ko}{\textcolor{schemaRed}{$\times$}}
\newcommand{\exid}{\texttt{BLHS\_MH\_0081}}

\setbeamercolor{keymsg}{fg=black,bg=schemaGold!16}
\newcommand{\keymsg}[1]{%
  \begin{beamercolorbox}[rounded=true,sep=0.12em,wd=\textwidth]{keymsg}
    \small\textbf{\color{schemaNavy}Thông điệp chính:} #1
  \end{beamercolorbox}}

% example box in the colour of the current step
\newcommand{\exboxof}[4]{% #1 colour, #2 text width, #3 sample id, #4 body
  \begin{tikzpicture}\node[draw=#1,fill=#1!7,rounded corners=3pt,line width=0.7pt,
    inner sep=4pt,text width=#2,align=left,font=\footnotesize]
    {{\bfseries\color{#1!80!black}Ví dụ #3.}\ #4};\end{tikzpicture}}
% running-example box (BLHS_MH_0081)
\newcommand{\exbox}[3]{\exboxof{#1}{#2}{\exid}{#3}}
% Default width argument: 0.95\linewidth. Where this outline writes \exbox{colour}{…}{…},
% "…" in the width slot means this default.

% research-question card (F1.6 and F5.1 use the same colours per RQ)
\newcommand{\rqcard}[4]{% #1 colour, #2 tag, #3 title, #4 body
  \begin{tikzpicture}\node[draw=#1,fill=#1!8,rounded corners=3pt,line width=0.7pt,
    inner sep=4pt,text width=0.92\linewidth,align=left,font=\small]
    {\textbf{\color{#1!80!black}#2 · #3}\\[1pt] #4};\end{tikzpicture}}

\tikzset{
  stepnode/.style={rounded corners=2pt,draw=#1,fill=#1!10,line width=0.7pt,
                   align=center,font=\scriptsize,inner sep=3pt,minimum height=0.9cm},
  evnode/.style={rounded corners=3pt,draw=#1,fill=#1!8,line width=0.6pt,align=center,
                 font=\scriptsize,text width=2.55cm,inner sep=2pt,minimum height=0.75cm},
  bridgenode/.style={evnode=#1,double,double distance=0.9pt},
  flow/.style={-{Latex[length=1.8mm]},line width=0.6pt,draw=schemaGray},
  causes/.style={-{Latex[length=2mm]},line width=0.9pt,draw=#1},
  elab/.style={font=\scriptsize,fill=white,inner sep=1pt},
  faded/.style={opacity=0.35},
  tag/.style={font=\scriptsize,text=schemaGray,inner sep=1pt}
}
```

### 3.3 NEW roadmap + frametitle override (prototype-tested; copy exactly)
```latex
\newcommand{\rmnode}[4]{% #1 index, #2 label, #3 colour, #4 current step
  \ifnum#1=#4\relax
    \node[rounded corners=1.5pt,minimum height=0.30cm,minimum width=1.05cm,inner sep=1pt,
          font=\tiny\bfseries,fill=#3,draw=#3,text=white]
      at ($(current page.south east)+(-7.40cm+#1*1.12cm,0.34cm)$) {#2};
  \else
    \node[rounded corners=1.5pt,minimum height=0.30cm,minimum width=1.05cm,inner sep=1pt,
          font=\tiny,fill=#3!12,draw=#3!50,text=#3!75!black]
      at ($(current page.south east)+(-7.40cm+#1*1.12cm,0.34cm)$) {#2};
  \fi}
\newcommand{\hustRoadmapStep}{-1}
\newcommand{\setroadmap}[1]{\renewcommand{\hustRoadmapStep}{#1}}
\newcommand{\drawroadmap}[1]{%
  \begin{tikzpicture}[remember picture,overlay]
    \rmnode{0}{Dữ liệu}{stepData}{#1}
    \rmnode{1}{Đồ thị}{stepGraph}{#1}
    \rmnode{2}{Bộ nhớ}{stepMemory}{#1}
    \rmnode{3}{Truy xuất}{stepRetrieve}{#1}
    \rmnode{4}{Xác minh}{stepVerify}{#1}
    \rmnode{5}{Trả lời}{stepAnswer}{#1}
  \end{tikzpicture}}
% same as the theme's frametitle template, plus the roadmap hook
\setbeamertemplate{frametitle}{%
  \nointerlineskip
  \begin{tikzpicture}[remember picture, overlay]
    \node[anchor=west, xshift=0.3cm, yshift=-0.4cm] at (current page.north west)
      {\usebeamerfont{frametitle}\usebeamercolor[fg]{frametitle}\insertframetitle};
  \end{tikzpicture}%
  \ifnum\hustRoadmapStep>-1\relax\drawroadmap{\hustRoadmapStep}\fi}
```
The roadmap labels are the only `\tiny` text allowed in the deck. `\setroadmap{k}` is placed **between** frames (outside `frame` environments).

### 3.4 NEW title-page override (prototype-tested geometry; removes "Nhóm 7:")
```latex
\newcommand{\thesisAdvisors}{TS. Nguyễn Quốc Tuấn và PGS.TS. Nguyễn Thị Kim Anh}
\newcommand{\thesisMajor}{Thạc sĩ Khoa học Máy tính}
\renewcommand{\husttitlepage}{%
  {\setbeamertemplate{background}{\includegraphics[width=\paperwidth,height=\paperheight]{\hustbg{title}}\drawHustPageNum}%
   \begin{frame}[plain]
    \begin{tikzpicture}[remember picture,overlay]
      \node[anchor=center,xshift=\titlePageTitleX,yshift=\titlePageTitleY,text width=0.80\paperwidth]
        at (current page.center) {%
        {\raggedright\Large\bfseries\color{hustprimary}\MakeUppercase{\inserttitle}\par}
        \vspace{0.14cm}
        {\raggedright\normalsize\bfseries\color{black!80}\insertsubtitle\par}
        \vspace{0.32cm}
        {\raggedright\small
          {\color{hustprimary}\bfseries Học viên:} \insertauthor\par
          {\color{hustprimary}\bfseries Giảng viên hướng dẫn:} \thesisAdvisors\par
          {\color{hustprimary}\bfseries Chuyên ngành:} \thesisMajor\par
          \vspace{0.12cm}
          {\color{hustprimary}\bfseries\insertinstitute} \quad\textbar\quad \insertdate\par}
      };
    \end{tikzpicture}
   \end{frame}}}
```
Acceptance: each info item on one line; block inside x ≤ 13.4 cm and 2.1 ≤ y ≤ 7.6 cm (checked by `check_layout.py`, §10). If the advisors line wraps, change only the info block from `\small` to `\footnotesize`.

### 3.5 Title data
```latex
\title{Cải thiện Multi-hop Question Answering bằng cách kết hợp Causal Graph và Counterfactual Reasoning trong RAG}
\subtitle{LegalSCM-RAG cho hỏi đáp pháp luật đa bước tiếng Việt}
\author{Nguyễn Thị Nhung -- 20251329M}
\institute{Đại học Bách khoa Hà Nội}
\date{Tháng 12 năm 2026}
```

### 3.6 Document skeleton
```latex
\begin{document}
\hustintropage{}
\husttitlepage
\begin{frame}{Nội dung trình bày}
  \tableofcontents[hideallsubsections]
\end{frame}
\resetPageCount
\showPageNumber
\useStyleHustFull
\useThinBar
\input{1_gioi_thieu}
\input{2_du_lieu_va_phuong_phap}
\input{3_thuc_nghiem_va_danh_gia}
\input{4_ket_qua_va_phan_tich}
\input{5_ket_luan}
\hidePageNumber
\hustthankyou
\appendix
\showPageNumber
\input{6_phu_luc}
\end{document}
```
Keep the template's `\resetPageCount` even though the theme prints `\insertpagenumber` (it has no visible effect; do not "fix" the theme).

### 3.7 Layout rules for every frame
- Text sizes: body `\normalsize`/`\small`; dense frames `\footnotesize`; tables and TikZ labels no smaller than `\scriptsize` (roadmap excepted). Split or move to the appendix instead of shrinking.
- Frame titles ≤ 60 characters, one line (measured: ~0.22 cm per character at the frametitle font; the bar is 16 cm wide).
- Design target: content ends above y = 8.0 cm (the footer logo starts at y 8.08 cm, the footer line at 8.38 cm, roadmap and page number below it). The checker tolerates up to 8.05 cm and flags anything right of x = 15.35 cm (full-width blocks normally reach 15.22 cm).
- Use `\begin{columns}[T,onlytextwidth]` as in the template; small negative `\vspace` at the top of a frame is allowed, as the template does.
- No file paths, sample-file names or code identifiers on slides except the sample ids (`BLHS_MH_…`), event/rule/article identifiers shown in the running example, and method names. Sources stay in this outline.
- Each formula is followed by one line starting with "Diễn giải:" (or the plain-language sentence given below).

## 4. Adapting content from the user's minimal deck (`slides/main.tex`)
The 10 DECK frames are reused (split and restyled) on F2.1, F2.4, F2.6–F2.8, F2.10–F2.14, F3.2–F3.3, F4.1, F4.3, F4.5. Required adaptations:
- **Encoding/fonts:** DECK uses `\usepackage[utf8]{inputenc}`, `[T5]{fontenc}`, `[vietnamese]{babel}`, `\usetheme{default}`, `\usefonttheme{serif}`. Drop all of them; the template's `\usepackage[utf8]{vietnam}` already provides UTF-8 + T5. Do not copy DECK's `\setbeamertemplate{frametitle|footline|itemize item|blocks}`, `AcademicBlue` colours or block colours.
- **TikZ styles:** DECK's `process` → `stepnode=<step colour>` (step-coloured, as the template's `\stepbox` idea); DECK's `flow` → `flow` (gray, same arrow tip `Latex[length=1.8mm]`); DECK's `entity` circles → `evnode` rounded rectangles (Vietnamese event names are long). DECK loads only `arrows.meta,positioning`; the template loads more, no change needed.
- **Labels:** DECK's `\textsc{supported}` etc. → `\labSup`, `\labRej`, `\labUnc` (D9).
- **Numbers:** DECK uses decimal points (`96.80`, `0.54`) → decimal commas (D8).
- **Tables:** DECK's `\resizebox{0.99\textwidth}{!}{…}` → natural-size `booktabs` table at `\footnotesize`/`\small` with `\setlength{\tabcolsep}{…}`; never scale a table below `\scriptsize`.
- **Footnote-size citations:** DECK's `\tiny` reference lines (e.g. "Nền tảng: Kleene (1952) …") → `\scriptsize` in `schemaGray`.

## 5. Frame-by-frame outline

Structural pages (not counted): intro (`\hustintropage{}`), title (`\husttitlepage`), TOC frame "Nội dung trình bày", 5 section pages (automatic), thank-you (`\hustthankyou`), "Phụ lục" section page (automatic).

Format of each entry: **file — title** · roadmap state · layout · slide text (Vietnamese) · figure/table plan · key message · sources.

### Section 1 — `1_gioi_thieu.tex` — `\section{Giới thiệu}` — `\setroadmap{-1}` (7 frames)

#### F1.1 — "LLM trả lời trôi chảy nhưng chưa đủ căn cứ pháp lý"
- Layout: two columns 0.56 / 0.40.
- Left bullets:
  - LLM đạt kết quả tốt trên nhiều bài toán hỏi đáp, nhưng hỏi đáp pháp luật đáng tin cậy cần đồng thời:
    - căn cứ luật thành văn phù hợp;
    - chuỗi suy luận truy vết được: điều kiện → sự kiện trung gian → hệ quả pháp lý (Chalkidis và cs., 2020; Guha và cs., 2023).
  - RAG bổ sung bằng chứng bên ngoài trước bước sinh (Lewis và cs., 2020; Gao và cs., 2024).
  - Nhưng truy xuất ngữ nghĩa thường xem các đoạn văn là độc lập, không biểu diễn tường minh quan hệ điều kiện–hệ quả nhiều bước.
- Right TikZ (two stacked mini-flows): top, all `schemaGray`, caption "RAG cơ bản": `Câu hỏi` → three small boxes `đoạn 1`, `đoạn 2`, `đoạn 3` (no arrows between them) → `LLM` → `Câu trả lời`. Bottom, caption "Điều bài toán cần": `Điều kiện` (stepGraph) → label "Điều a" → `Sự kiện trung gian` (bridgenode) → label "Điều b" → `Hệ quả pháp lý`.
- `\keymsg{Truy xuất đúng tài liệu chưa chắc dẫn đến suy luận đúng.}`
- Sources: PV1 ¶1; LV1 tr. 5–7 (key message wording from tr. 7). No numbers.

#### F1.2 — "Ví dụ xuyên suốt: câu hỏi cần nối hai điều luật"
- Layout: block on top; TikZ chain in the middle; two columns at the bottom (0.52 / 0.44).
- `block` titled "Câu hỏi \exid{} (suy luận ngược, 2 bước)": «Kết quả “Có tình tiết giảm nhẹ trách nhiệm hình sự” trong chain này bắt nguồn từ sự kiện nào trước khi đi qua bước “Vượt quá giới hạn phòng vệ chính đáng”?»
- TikZ chain (three `evnode`s, left to right): `Chống trả rõ ràng quá mức cần thiết` (dashed border, small tag above "cần tìm", small tag below "đáp án chuẩn") —[`causes=stepGraph`, label "rule 47 · Điều 22"]→ `Vượt quá giới hạn phòng vệ chính đáng` (`bridgenode`) —[label "rule 106 · Điều 51"]→ `Có tình tiết giảm nhẹ trách nhiệm hình sự`.
- Bottom-left `block` "Vì sao khó?":
  - hỏi ngược: từ hệ quả quay về điều kiện ban đầu;
  - phải nối hai điều luật khác nhau: Điều 22 (Phòng vệ chính đáng) và Điều 51 (Các tình tiết giảm nhẹ trách nhiệm hình sự);
  - sự kiện trung gian không phải là đáp án nhưng bắt buộc phải đi qua.
- Bottom-right `alertblock` "Closed-book (abstain) trả lời": “Không đủ thông tin để xác định sự kiện trước bước ‘Vượt quá giới hạn phòng vệ chính đáng’.” → \labUnc
- Footer line (`\scriptsize`, gray): "Viết tắt: PVCĐ – phòng vệ chính đáng; TNHS – trách nhiệm hình sự; BLHS – Bộ luật Hình sự. Ví dụ này được theo dõi qua từng bước ở Phần 2 và so sánh ở Phần 4."
- No key-message box (space).
- Sources: B (`BLHS_MH_0081`: question, question_type `reverse_reasoning`, hop_count 2, supporting_rule_ids 47/106, supporting_article_ids 22/51, evaluation.gold_answer, evaluation.gold_decision SUPPORTED); RAW (titles of Điều 22 and Điều 51); P-cba (`BLHS_MH_0081`: verification_decision UNCERTAIN, final_answer).

#### F1.3 — "Bài toán: quyết định, câu trả lời và căn cứ truy vết được"
- Layout: two `block`s side by side (0.48 / 0.48), formula centred below, label table, small alert line.
- `block` "Đầu vào": câu hỏi pháp lý $q$; đồ thị có hướng $G=(V,E)$ — nút: sự kiện pháp lý đã chuẩn hóa, cạnh: một hoặc nhiều rule điều kiện–hệ quả; tập rule $\mathcal R$ xác định ngữ nghĩa suy luận.
- `block` "Đầu ra": đường suy luận $P^{*}=v_0\xrightarrow{r_1}\cdots\xrightarrow{r_h}v_h$; quyết định $z$; bằng chứng $\mathcal E^{*}$; câu trả lời $a$; căn cứ trích dẫn $C$.
- Formula: $f:(q,G,\mathcal R)\rightarrow(P^{*},z,\mathcal E^{*},a,C)$ — "Diễn giải: hệ thống phải trả về chuỗi lập luận và quyết định đã kiểm chứng, không chỉ một câu trả lời."
- Label table (`\footnotesize`, 2 columns, booktabs):
  - \labSup — luật (qua đồ thị và rule) hỗ trợ quan hệ được hỏi.
  - \labRej — bác bỏ khẳng định “trực tiếp” hoặc “bỏ qua được bước trung gian” khi luật chỉ hỗ trợ chuỗi gián tiếp $A\to B\to C$.
  - \labUnc — không đủ căn cứ để kết luận (từ chối trả lời).
- Small alert line (`\scriptsize`, `schemaRed`): "Lưu ý: cạnh “nhân quả” là quan hệ phụ thuộc quy phạm trích từ luật thành văn, không phải hiệu ứng nhân quả ước lượng từ dữ liệu quan sát."
- Sources: PV3 (definitions of $G$, $\mathcal R$, $P_i$, $P^{*}$, $z$, $f$); PV1 ¶2 (normative dependencies); B metadata.construction_notes ("Mẫu phản ví dụ chỉ phủ nhận cạnh trực tiếp A→C khi graph chỉ hỗ trợ A→B→C").

#### F1.4 — "Nghiên cứu liên quan: RAG, đồ thị và suy luận quy tắc"
- Layout: one `booktabs` table, `\scriptsize`, columns `p{0.27\textwidth} p{0.33\textwidth} p{0.34\textwidth}`; header row: **Hướng** | **Ý tưởng chính** | **Hạn chế với hỏi đáp pháp luật đa bước**.
- Rows:
  1. RAG và truy xuất dense (Lewis và cs., 2020; Karpukhin và cs., 2020; Izacard & Grave, 2021; Asai và cs., 2024; Sarthi và cs., 2024) | ghép bằng chứng truy xuất vào LLM; truy xuất thích ứng, tổ chức tri thức phân cấp | bằng chứng chủ yếu là đoạn văn độc lập, thiếu cấu trúc suy luận tường minh
  2. Hỏi đáp đa bước (Yang và cs., 2018; Trivedi và cs., 2022) | câu trả lời cần nhiều bằng chứng liên kết | truy xuất từng đoạn liên quan riêng lẻ là chưa đủ
  3. RAG trên đồ thị và quan hệ nhân quả (Edge và cs., 2024; Sun và cs., 2024; Wang và cs., 2025; Parekh và cs., 2025) | tổ chức bằng chứng theo quan hệ; duyệt đồ thị; chuỗi nhân quả | tính đạt được trên đồ thị không tự xác định ngữ nghĩa quy tắc pháp luật
  4. Suy luận phản thực và SCM (Guo và cs., 2023; Pearl, 2009) | mô hình có thể đúng nhờ suy luận rời rạc; can thiệp khác xóa nút | chưa tích hợp vào pipeline RAG pháp luật
  5. Logic ba giá trị, lập trình logic đại số tuyến tính (Kleene, 1952; Fitting, 1985; Sato và cs., 2020; Sakama và cs., 2021) | ngữ nghĩa điểm bất động; biểu diễn vector, ma trận, tensor | ít được kết hợp với truy xuất và hỏi đáp pháp luật
- Footer (`\scriptsize`): "Benchmark pháp luật: LexGLUE, CaseHOLD, LegalBench, LEGAL-BERT (Chalkidis và cs., 2022; Zheng và cs., 2021; Guha và cs., 2023; Chalkidis và cs., 2020)."
- Sources: PV2 (all claims); PVR (authors, years); LV1 tr. 21 (row 4 limitation wording).

#### F1.5 — "Khoảng trống nghiên cứu và vị trí của luận văn"
- Layout: two columns 0.52 / 0.44.
- Left: "Chưa có framework hỏi đáp pháp luật đồng thời:" then 5 bullets:
  - truy xuất và xếp hạng đường suy luận pháp luật nhiều bước;
  - biểu diễn trạng thái pháp lý chưa đầy đủ bằng ngữ nghĩa ba giá trị;
  - xử lý phép hội trong một rule và tuyển thay thế giữa nhiều rule;
  - phân biệt xóa nút tô-pô với can thiệp cấu trúc;
  - commit có chọn lọc các quyết định đã xác minh trước bước sinh.
- Right TikZ: three overlapping circles (fill opacity 0.15, draw opacity 0.8): "Truy xuất ngữ nghĩa (RAG)" in `stepMemory`, "Kết nối cấu trúc (Graph/Causal RAG)" in `stepGraph`, "Ngữ nghĩa quy tắc hình thức (logic ba giá trị, SCM)" in `stepVerify`; centre label **LegalSCM-RAG** (bold, `schemaNavy`).
- `\keymsg{LegalSCM-RAG tích hợp truy xuất đường dẫn nhân quả, suy luận điểm bất động ba giá trị, xác minh phụ thuộc câu hỏi và cổng commit có kiểm toán.}`
- Sources: PV2 §"Khoảng trống nghiên cứu" (5 items and the final sentence).

#### F1.6 — "Bốn câu hỏi nghiên cứu"
- Layout: 2×2 grid of `\rqcard`s (two `columns`, two cards each), then one `\scriptsize` line.
- Cards (colour, tag, title, body):
  - `stepMemory`, RQ1, Truy xuất có cấu trúc — "Truy xuất trên rule, sự kiện và đường dẫn có tìm đúng căn cứ pháp lý tốt hơn truy xuất điều luật thô (Vanilla RAG) không?"
  - `stepRetrieve`, RQ2, Đường suy luận đa bước — "Truy xuất theo đường dẫn nhân quả có chọn đúng chuỗi suy luận nhiều bước không?"
  - `stepVerify`, RQ3, Xác minh chọn lọc — "Xác minh LegalSCM (factual, cạnh trực tiếp, $do(X_m=0)$) có cải thiện quyết định so với chỉ truy xuất đường dẫn không? Lợi ích đến từ route nào?"
  - `stepAnswer`, RQ4, Hiệu quả đầu–cuối — "Câu trả lời và trích dẫn có tốt hơn LLM closed-book, Vanilla RAG và Causal Path RAG không?"
- Bottom line: "Mỗi câu hỏi được trả lời bằng số liệu và kiểm định ở Phần 5."
- Sources: LV1 tr. 12 (RQ1–RQ4, adapted: "CausalRAG" → "Causal Path RAG", see D13/C5).

#### F1.7 — "Đóng góp của luận văn"
- Layout: three `block`s in columns (0.32 each), then a row of 4 `\metric` cards.
- `block` "Dữ liệu và tri thức": 2.884 quy tắc điều kiện–hệ quả từ 395 điều BLHS; đồ thị nhân quả pháp lý 5.134 nút, 14.420 cạnh; benchmark 250 câu hỏi đa bước có gold rule, sự kiện, đường, trích dẫn.
- `block` "Phương pháp": truy xuất và xếp hạng đường suy luận đa bước trên quan hệ điều kiện–hệ quả; formulation quy tắc ba giá trị đại số tuyến tính: hội, tuyển thay thế, can thiệp cấu trúc tường minh; xác minh phụ thuộc câu hỏi và cổng commit chọn lọc.
- `block` "Đánh giá": so sánh có kiểm soát năm cấu hình; kiểm định theo cặp (bootstrap, McNemar, hiệu chỉnh Holm); kiểm toán route, commit, abstention và can thiệp.
- Metric row: `\metric{96,80\%}{Decision Accuracy của Full LegalSCM}` `\metric{+8,0}{điểm so với Causal Path RAG}` `\metric{0,0011}{$p$ hiệu chỉnh Holm}` `\metric{250}{câu hỏi BLHS đa bước}`.
- Sources: R (2.884 rules; 395 distinct `article_id`), G (5.134 nodes, 14.420 edges), B (250), PV1 (contribution list), LV1 tr. 15 (data contributions), S-sum (96,80%), S-sig (delta 0.08, `p_value_holm` 0.001144 → 0,0011).

### Section 2 — `2_du_lieu_va_phuong_phap.tex` — `\section{Dữ liệu và phương pháp}` (14 frames)

#### F2.1 — "LegalSCM-RAG: hai pha, năm bước" · `\setroadmap{-1}` (the diagram itself is the map)
- TikZ pipeline, full width, `\scriptsize`, two rows, each row on a light background rectangle (`backgrounds` + `fit`, fill `schemaLight`, rounded) with a left caption:
  - Row 1 caption "Ngoại tuyến (một lần)": `BLHS · 406 điều` (stepnode=stepData) → `(a1) trích xuất rule · (a2) chuẩn hóa sự kiện` (stepnode=stepData) → `Bước 1 · Đồ thị nhân quả pháp lý · 5.134 nút` (stepnode=stepGraph) → `Bước 2 · Bộ nhớ nhân quả · 4.698 vector` (stepnode=stepMemory).
  - Row 2 caption "Theo từng câu hỏi": `Câu hỏi q` (gray) → `Bước 3 · Truy xuất đường dẫn → P*` (stepnode=stepRetrieve) → `Bước 4 · Xác minh LegalSCM theo loại câu hỏi` (stepnode=stepVerify) → diamond `Commit?` (`shape=diamond`, draw=stepVerify, `aspect=1.6`) → `Bước 5 · Sinh câu trả lời + trích dẫn → (a, C)` (stepnode=stepAnswer). From the diamond a dashed arrow labelled "không đạt: giữ ngữ cảnh Bước 3" goes to Bước 5.
  - Dashed gray arrows from Bước 1 and Bước 2 down to Bước 3 (labelled "tài nguyên"), and from Bước 1 to Bước 4 (labelled "tập rule $\mathcal R$").
- Below: $q\;\rightarrow\;P^{*}\;\rightarrow\;(z,\mathcal E^{*})\;\rightarrow\;(a,C)$ — "Diễn giải: chọn đường suy luận → ra quyết định có xác minh → sinh câu trả lời kèm căn cứ."
- `\scriptsize` line: "LegalSCM (Legal Structural Causal Model): mô hình quy tắc quy phạm ba giá trị, hỗ trợ can thiệp cấu trúc."
- `\keymsg{Điểm mới: bước xác minh LegalSCM có chọn lọc được đặt giữa truy xuất đường dẫn và sinh câu trả lời.}`
- Sources: PV4 eq. (overall_pipeline); DECK frame 1; PE1 (expansion of LegalSCM); LV1 tr. 23 (key-message idea); RAW (406), G (5.134), MEM (4.698).

#### F2.2 — "Dữ liệu nguồn: 406 điều của Bộ luật Hình sự" · `\setroadmap{0}`
- Layout: bullets (top), row of 4 `\metric` cards (bottom).
- Bullets:
  - Văn bản: Văn bản hợp nhất số 135/VBHN-VPQH – Bộ luật Hình sự (Văn phòng Quốc hội, 05/09/2025).
  - Dữ liệu thô: 406 điều, mã 1–426 (không có 20 mã: Điều 90–107, 292, 317); mỗi bản ghi gồm `article_id`, `article_title`, `content`.
  - Sau trích xuất: 2.884 rule từ 395 điều; 11 điều không sinh rule nào.
  - Trung bình 7,30 rule/điều, nhiều nhất 41 rule (Điều 251); 41 chủ thể pháp lý.
  - Nhãn chất lượng của rule cuối: 2.654 `CURATED_FROM_STATUTORY_CLAUSE`, 230 `CURATED`.
- Metrics: `\metric{406}{điều luật (dữ liệu thô)}` `\metric{2.884}{quy tắc điều kiện–hệ quả}` `\metric{1.814}{sự kiện chuẩn hóa}` `\metric{41}{chủ thể pháp lý}`.
- Sources: PVR (`blhs2025`); RAW (406 records, ids 1–426; missing ids computed from RAW: 90–107, 292, 317); R (2.884 rules; 395 distinct articles; quality_status counts 2.654/230; legal_subject distinct 41; mean 2.884/395 = 7,30 and max 41 for Điều 251 computed from R); RAW vs R (11 articles without rules: 1, 4, 30, 31, 42, 46, 122, 352, 367, 392, 426 — list kept here, not on the slide); G (1.814 EVENT, 41 SUBJECT).

#### F2.3 — "Tiền xử lý: trích xuất quy tắc và chuẩn hóa sự kiện" · `\setroadmap{0}`
- Layout: two columns 0.49 / 0.47, `\footnotesize`. (If it overflows: split into F2.3a "(a1) …" and F2.3b "(a2) …"; this is the only allowed split.)
- Left `block` "(a1) Trích xuất quy tắc bằng LLM":
  - Mỗi điều luật → Qwen3-8B (Ollama, temperature 0, đầu ra theo JSON schema) → danh sách rule {`legal_subject`, `condition`, `effect`, `rule_text`}.
  - `rule_text` theo mẫu “Nếu ⟨điều kiện⟩ thì ⟨hệ quả⟩.”; chỉ dùng nội dung trong điều luật; khử trùng lặp; thử lại tối đa 3 lần.
  - Excerpt box (gray, italic): Điều 22, khoản 2: “Vượt quá giới hạn phòng vệ chính đáng là hành vi chống trả rõ ràng quá mức cần thiết, không phù hợp với tính chất và mức độ nguy hiểm cho xã hội của hành vi xâm hại.”
  - Down arrow, then box "rule 47 (bản cuối, đã rà soát)": “Nếu chống trả rõ ràng quá mức cần thiết và không phù hợp với tính chất, mức độ nguy hiểm của hành vi xâm hại thì người phòng vệ được xác định là vượt quá giới hạn phòng vệ chính đáng.”
- Right `block` "(a2) Chuẩn hóa sự kiện pháp lý":
  - Các cách diễn đạt của cùng một sự kiện dùng chung `event_id`: “phải chịu TNHS”, “chịu TNHS”, “bị truy cứu TNHS” → *Chịu trách nhiệm hình sự*.
  - Không gộp khái niệm chỉ gần nghĩa: “Miễn TNHS” ≠ “Không phải chịu TNHS”; “Phạm tội” ≠ “Chịu TNHS”.
  - Ứng viên: $s_{\text{lex}}=0{,}45\,\text{Jaccard}+0{,}35\,\text{SeqMatch}+0{,}20\,\text{containment}$ — "Diễn giải: kết hợp độ trùng từ, độ giống chuỗi và mức bao hàm."
  - LLM chọn `USE_EXISTING` hoặc `CREATE_NEW`; chỉ dùng sự kiện có sẵn khi điểm ≥ 0,60, đúng vai trò và không tạo vòng tự thân.
  - Kết quả: 1.814 sự kiện chuẩn (1.416 ở phía điều kiện, 412 ở phía hệ quả, 14 ở cả hai).
- `\keymsg{Chuẩn hóa nối các điều luật: hệ quả của rule 47 (Điều 22) và điều kiện của rule 106 (Điều 51) cùng là sự kiện “Vượt quá giới hạn phòng vệ chính đáng”.}`
- Sources: A1 (line 15 `DEFAULT_MODEL = "qwen3:8b"`; line 215 `--temperature` default 0.0; line 219 `--max-attempts` default 3; JSON schema lines 180–195; prompt rules lines 60–135); A2 (prompt lines 26–100 incl. the two "không được gộp" examples; line 207 min score 0.60; lines 334–347 lexical score; lines 982–996 forced `CREATE_NEW`); RAW (Điều 22 text); R (rule 47 `rule_text`, `quality_status` CURATED; rule 106 condition event); G `event_role_counts` (condition_only 1.402 + both 14 = 1.416; effect_only 398 + 14 = 412).
- Do **not** quote A2's example "Có thể được miễn ≠ Được miễn" (D11).

#### F2.4 — "Bước 1: đồ thị nhân quả pháp lý dị thể" · `\setroadmap{1}`
- Layout: two columns 0.50 / 0.46.
- Left TikZ schema (restyled DECK frame 2): `ARTICLE` (gray) —HAS_RULE→ `RULE $r_j$` (stepGraph) —HAS_SUBJECT→ `SUBJECT`; `RULE` —HAS_CONDITION→ `EVENT $u_j$`; `RULE` —HAS_EFFECT→ `EVENT $v_j$`; thick `causes=stepGraph` edge $u_j$ —CAUSES→ $v_j$. Edge labels `\scriptsize`.
  - $r_j:\;u_j\xrightarrow{\;r_j,\,a_j\;}v_j$ — "Diễn giải: mỗi rule là một cạnh CAUSES mang mã rule và điều luật nguồn."
  - $A\xrightarrow{r_1}B,\;B\xrightarrow{r_2}C\;\Longrightarrow\;A\rightarrow B\rightarrow C$ — "Diễn giải: hai rule nối thành chuỗi khi hệ quả của rule trước trùng điều kiện của rule sau."
- Right `booktabs` table (`\footnotesize`):
  - Nút | 5.134 (395 ARTICLE · 2.884 RULE · 1.814 EVENT · 41 SUBJECT)
  - Cạnh (đa đồ thị) | 14.420 = 5 loại × 2.884
  - Cặp sự kiện CAUSES duy nhất | 2.122
  - Sự kiện cầu nối | 14
  - Chuỗi hai bước | 112
  - Đồ thị sự kiện | DAG, 248 thành phần liên thông yếu
- Right bullets under the table: biểu diễn bằng NetworkX MultiDiGraph; cạnh CAUSES giữ `rule_id`, `article_id` → nhiều rule cùng hỗ trợ một cặp sự kiện mà không mất nguồn gốc.
- `\keymsg{Đồ thị phục vụ truy xuất và phân tích tô-pô; ngữ nghĩa suy luận do tập rule $\mathcal R$ quyết định.}`
- Sources: G (`total_nodes` 5134, `total_edges` 14420, `node_type_counts`, `edge_type_counts` all 2884, `causal_edges` 2122, `bridge_event_nodes` 14, `two_hop_causal_chains` 112, `causal_graph_is_dag` true, `causal_weak_components` 248); DECK frame 2; PV3 ¶1.

#### F2.5 — "Lát cắt thật của đồ thị quanh ví dụ xuyên suốt" · `\setroadmap{1}`
- TikZ, full width, three columns of nodes (`evnode`, text width 2.6 cm), edge labels `elab` "rule N · Điều M":
  - Column 1 (điều kiện): C1 `Chống trả cần thiết để bảo vệ lợi ích chính đáng`; C2 `Gây thiệt hại nhỏ hơn để tránh thiệt hại lớn hơn`; C3 `Chống trả rõ ràng quá mức cần thiết`.
  - Column 2 (cầu nối, `bridgenode=stepGraph`): B1 `Phòng vệ chính đáng`; B2 `Tình thế cấp thiết`; B3 `Vượt quá giới hạn phòng vệ chính đáng`.
  - Column 3 (hệ quả): E1 `Không bị coi là tội phạm`; E2 `Chịu trách nhiệm hình sự`; E3 `Có tình tiết giảm nhẹ trách nhiệm hình sự`.
  - Edges: C1→B1 (rule 45 · Điều 22); B1→E1 (rule 46 · Điều 22); C2→B2 (rule 49 · Điều 23); B2→E1 (rule 50 · Điều 23); C3→B3 (rule 47 · Điều 22); B3→E2 (rule 48 · Điều 22); B3→E3 (rule 106 · Điều 51).
  - Arrange rows top→bottom as PVCĐ (C1,B1), Tình thế cấp thiết (C2,B2), Vượt quá (C3,B3) and place E1 between rows 1–2, E2 and E3 beside row 3, so that no edges cross.
  - Highlight the running-example path C3→B3→E3 with `causes=stepGraph` at 1.4 pt and a small tag "đường của ví dụ"; draw the other edges at 0.6 pt in `stepGraph!60`.
  - Gray `tag`s next to the effects: E1 "← 5 rule (Điều 8, 22–25)", E2 "← 18 rule", E3 "← 28 rule (Điều 51, 84)".
- Bullets below (`\footnotesize`):
  - 3 trong 14 sự kiện cầu nối nằm trong lát cắt này; có rẽ nhánh (một cầu nối → hai hệ quả) và hội tụ (hai đường → “Không bị coi là tội phạm”, tức tuyển giữa các rule).
  - Đường của ví dụ: rule 47 (Điều 22) → rule 106 (Điều 51).
- `\keymsg{Sự kiện cầu nối biến hai điều luật rời rạc thành một chuỗi suy luận hai bước.}`
- Sources: R (rules 45, 46, 47, 48, 49, 50, 106: condition/effect events and `article_id`; full rule texts in §6.3); G `bridge_events` (contains `phong_ve_chinh_dang`, `tinh_the_cap_thiet`, `vuot_qua_gioi_han_phong_ve_chinh_dang`); in-degrees computed from R (rules whose `effect_event` is the node: `khong_bi_coi_la_toi_pham` 5 rules from Điều 8, 22, 23, 24, 25; `chiu_trach_nhiem_hinh_su` 18; `co_tinh_tiet_giam_nhe_trach_nhiem_hinh_su` 28 from Điều 51, 84).

#### F2.6 — "Bước 2: bộ nhớ nhân quả cho tìm kiếm ngữ nghĩa" · `\setroadmap{2}`
- Top TikZ flow (stepMemory, restyled DECK frame 3): `RULE memory · chủ thể, điều kiện, hệ quả, sự kiện, điều luật` and `EVENT memory · tên, alias, vai trò, ngữ cảnh` (stacked) → `BGE-M3 · 1.024 chiều` → `Chuẩn hóa L2` → `FAISS IndexFlatIP`.
- Bottom two columns (0.50 / 0.46):
  - Left: "Trực giác: mỗi rule và mỗi sự kiện thành một vector; câu hỏi tìm các vector gần nhất."  
    $\hat{\mathbf z}_j=\mathbf z_j/\|\mathbf z_j\|_2,\qquad s(q,c_j)=\hat{\mathbf q}^{\top}\hat{\mathbf z}_j=\cos(\mathbf q,\mathbf z_j)$ — "Diễn giải: sau chuẩn hóa, tích vô hướng chính là cosine; IndexFlatIP tìm kiếm chính xác, không xấp xỉ."  
    Citation `\scriptsize`: BGE-M3 (Chen và cs., 2024).
  - Right bullets: 4.698 vector = 2.884 RULE + 1.814 EVENT; `memory_id` trùng vị trí vector trong index; kiểm tra nhất quán CSV – embedding – FAISS; vai trò EVENT: CONDITION 1.402, EFFECT 398, BRIDGE 14.
  - Right `\exbox{stepMemory}{0.9\linewidth}{…}`: "EVENT \#2956 Chống trả rõ ràng quá mức cần thiết (CONDITION); \#2957 Vượt quá giới hạn PVCĐ (BRIDGE); \#3047 Có tình tiết giảm nhẹ TNHS (EFFECT). Rule gần câu hỏi nhất: RULE 106 (Điều 51), cos = 0,653."
- Sources: C2 (BGE-M3, L2 normalisation, `faiss.IndexFlatIP`); MEM (4.698 rows; `memory_type` RULE 2.884 / EVENT 1.814; `event_role` CONDITION 1.402 / EFFECT 398 / BRIDGE 14; `.npy` shape (4698, 1024)); P-full `BLHS_MH_0081` `retrieval.retrieved_events` (memory_id 2956/2957/3047 and roles) and `retrieval.direct_rule_hits[0]` (RULE::106, score 0.6535); DECK frame 3; PVR (`chen2024bgem3`).

#### F2.7 — "Bước 3: truy xuất đường dẫn theo hướng câu hỏi" · `\setroadmap{3}`
- Layout: two columns 0.47 / 0.49.
- Left `enumerate` (`\small`):
  1. Mã hóa câu hỏi bằng BGE-M3.
  2. FAISS lấy 100 RULE/EVENT gần nhất.
  3. Chọn seed: EVENT top-8 ($s\ge0{,}20$) và RULE top-8 ($s\ge0{,}15$).
  4. Suy hướng từ ý định câu hỏi: xuôi (forward), ngược (backward) hoặc cả hai.
  5. BFS tối đa 2 hop, không lặp nút, tối đa 30 đường mỗi seed.
  6. Gom rule từ seed, đường và kết quả trực tiếp → xếp hạng → 12 bằng chứng.
- Right: `\exbox{stepRetrieve}{0.9\linewidth}{Cụm “bắt nguồn từ” → hướng ngược (backward).}` then a vertical funnel TikZ (stepRetrieve shades, widths decreasing): `100 kết quả ngữ nghĩa` → `8 seed sự kiện + 8 rule trực tiếp` → `33 đường ứng viên (1 đường 2 bước, 32 đường 1 bước)` → `36 rule ứng viên` → `12 bằng chứng cuối`.
- `\keymsg{Khác RAG thường: ngữ cảnh gồm sự kiện, rule và quan hệ có hướng nối chúng, không chỉ các đoạn văn độc lập.}`
- Sources: LOG-full `configuration` (semantic_pool_size 100, event_top_k 8, direct_rule_top_k 8, min_event_score 0.2, min_rule_score 0.15, max_hops 2, max_paths_per_event 30, final_top_k 12, direction "both"); P-full `BLHS_MH_0081` `retrieval.statistics` (semantic_hits 100, retrieved_event_seeds 8, direct_rule_hits 8, causal_paths 33, candidate_rules 36, final_evidence 12, effective_direction backward) and `retrieval.causal_paths` (33 paths: 1 with 2 rules, 32 with 1 rule, all backward); DECK frame 4; PV4 (last ¶ of "Truy xuất đường dẫn nhân quả").

#### F2.8 — "Xếp hạng đường dẫn: ưu tiên đường đủ bước qua cầu nối" · `\setroadmap{3}`
- Line 1 (trực giác): "Đường tốt: bắt đầu từ seed gần câu hỏi, được rule hỗ trợ, đi qua sự kiện cầu nối và đủ số bước."
- Paper form: $P^{*}=\arg\max_{P_i\in\mathcal P(q)}S(P_i,q)$ — "Diễn giải: chọn đường có điểm phù hợp ngữ nghĩa và cấu trúc cao nhất."
- Implemented score: $S_{\text{path}}=\operatorname{clip}_{[0,1]}\big(0{,}54\,s_{\text{seed}}+0{,}28\,s_{\text{sup}}+b_{\text{bridge}}+b_{\text{hop}}\big)$, $b_{\text{bridge}}=\min(0{,}18;\,0{,}06\,N_{\text{bridge}})$, $b_{\text{hop}}=0{,}18$ nếu đường đủ 2 bước — "Diễn giải: độ gần của seed chiếm phần lớn; hai khoản thưởng ưu tiên đường hai bước qua cầu nối."
- `\exbox{stepRetrieve}{…}`: $S(P_0)=0{,}54\cdot0{,}602+0{,}28\cdot0{,}5+0{,}06+0{,}18=0{,}705$ → hạng 1/33; đường 1 bước tốt nhất: $S(P_1)=0{,}481$.
- Line: "Điểm bằng chứng của rule kết hợp 5 tín hiệu (Phụ lục A1): rule 106 = 0,719 (hạng 1), rule 47 = 0,642 (hạng 2)."
- `\keymsg{Với ví dụ, đường đúng đứng đầu; trên toàn benchmark, xếp hạng đường vẫn là nút thắt (Phần 4).}`
- Sources: C3 lines 49–55 (`FULL_HOP_COMPLETION_BONUS` 0.18, `PARTIAL_HOP_COMPLETION_BONUS` 0.03, `BRIDGE_EVENT_BONUS` 0.06) and 1140–1206 (`_score_path`); PV4 ($P^{*}$, $S$); P-full `BLHS_MH_0081` `retrieval.causal_paths[0]` (seed_similarity 0.60206, graph_score 0.70511), `causal_paths[1]` (graph_score 0.48094), `retrieval.retrieved_rules` (rule 106 final_score 0.7190, rule 47 0.6418); $s_{\text{sup}}=0{,}5$ computed from C3 (one supporting rule per hop: $\ln 2/\ln 4$); DECK frame 4.

#### F2.9 — "Bước 4: mỗi sự kiện pháp lý có ba trạng thái" · `\setroadmap{4}`
- Layout: two columns 0.50 / 0.46.
- Left bullets:
  - Trực giác: thông tin pháp lý thường chưa đầy đủ — “chưa biết” khác “sai”.
  - $X_i\in\{1,0,\bot\}$ = TRUE / FALSE / UNKNOWN; $\bot$ là thiếu hoặc xung đột hỗ trợ, không phải giá trị nằm giữa 0 và 1.
  - Rule kích hoạt khi mọi điều kiện thỏa (AND trong rule); hệ quả đúng khi ít nhất một rule kích hoạt (OR giữa các rule); lặp đến điểm bất động.
  - Quy ước cài đặt: sự kiện gốc chưa có trong ngữ cảnh → UNKNOWN; sự kiện là hệ quả của rule nhưng không rule nào kích hoạt → FALSE (thế giới đóng); hỗ trợ TRUE/FALSE xung đột → UNKNOWN.
  - Citation `\scriptsize`: Kleene (1952); Fitting (1985).
- Right: `\exbox{stepVerify}{…}{Điểm bất động factual}` followed by a small table (`\footnotesize`, cells TRUE in `schemaGreen!25`, "—" in `schemaGray!15`):

  | Vòng | Chống trả rõ ràng quá mức cần thiết | Vượt quá giới hạn PVCĐ | Có tình tiết giảm nhẹ TNHS |
  |---|---|---|---|
  | Ngữ cảnh | TRUE | — | — |
  | Sau rule 47 (Điều 22) | TRUE | TRUE | — |
  | Sau rule 106 (Điều 51) | TRUE | TRUE | TRUE |

  Caption: "Hội tụ sau 3 vòng; mọi hop có rule kích hoạt (47, 106); không có xung đột → outcome TRUE. (— : chưa được suy ra)"
- `\keymsg{Kết luận chỉ được chấp nhận khi chuỗi rule thực sự suy ra được hệ quả.}`
- Sources: PV3 (§"Ngữ nghĩa ba giá trị…": $X_i$, meaning of unknown, fixed point); SCM module docstring lines 7–11 (closed world for endogenous events, root events UNKNOWN, conflict → UNKNOWN; see D12/C7); P-full `BLHS_MH_0081` `verification.path_verifications[0].counterfactual_summary` (factual_context {chong_tra_ro_rang_qua_muc_can_thiet: true}, factual_states all "true", factual_iterations 3, path_rule_coverage activated 47 and 106, factual_conflicting_event_ids []).

#### F2.10 — "Biểu diễn thưa: AND trong một rule, OR giữa các rule" · `\setroadmap{4}`
- Layout: two columns 0.55 / 0.41.
- Left, four formulas, each followed by "Diễn giải:":
  - $(t_i,f_i)=(1,0)$ nếu $X_i=1$; $(0,1)$ nếu $X_i=0$; $(0,0)$ nếu $X_i=\bot$ — "hai kênh hỗ trợ: đúng và sai."
  - $\mathbf b=\mathbb I[\mathbf B^{+}\mathbf t+\mathbf B^{-}\mathbf f=\mathbf k],\quad \mathbf a=\mathbf b\odot(\mathbf 1-\mathbf e)$ — "rule kích hoạt khi đủ mọi literal trong thân ($\mathbf k$ = số literal) và không bị ngoại lệ $\mathbf e$ chặn."
  - $\mathbf d^{+}=\mathbb I[\mathbf H^{+}\mathbf a\ge1],\quad \mathbf d^{-}=\mathbb I[\mathbf H^{-}\mathbf a\ge1]$ — "hệ quả có hỗ trợ khi ít nhất một rule kích hoạt."
  - $\mathbf x^{F}=\operatorname{Fix}(\Phi_{\mathcal R},\mathbf x^{(0)})$ — "lặp đến khi trạng thái không đổi."
- Right `block` "Hội và tuyển": $(A\wedge B)\to Y$: một rule có hai literal; $(A\vee B)\to Y$: chuẩn hóa thành $A\to Y$ và $B\to Y$.
- Right `block` "Quy mô trong thực nghiệm": $\mathbf B^{+}\in\{0,1\}^{2884\times1814}$, $\mathbf H^{+}\in\{0,1\}^{1814\times2884}$, mỗi ma trận 2.884 phần tử khác 0; $\mathbf B^{-}$, $\mathbf H^{-}$ và ngoại lệ rỗng; 187 sự kiện có từ 2 rule hỗ trợ trở lên (tuyển); runtime suy luận ký hiệu tới điểm bất động, không nhân ma trận dày.
- `alertblock` "Giới hạn": "Engine hỗ trợ nhiều literal và ngoại lệ, nhưng dữ liệu hiện tại mỗi rule chỉ có một tiền đề dương (0 rule hội) → phép hội chưa được kiểm chứng thực nghiệm."
- Citation `\scriptsize`: Sato và cs. (2020); Sakama và cs. (2021).
- Sources: PV3 eqs. two_channel_state, rule_activation, head_support, factual_fixed_point, and the one-antecedent limitation ¶; P-full `verification.configuration.rule_semantics` (matrix_shapes condition_true [2884, 1814], head_true [1814, 2884]; matrix_nnz condition_true 2884, head_true 2884, others 0; conjunctive_rule_count 0; rules_with_exceptions 0; alternative_support_event_count 187; runtime_engine SYMBOLIC_FIXED_POINT); 187 cross-checked with R (effect events with ≥ 2 rules = 187 of 412); DECK frame 5.

#### F2.11 — "Ba route xác minh theo loại câu hỏi" · `\setroadmap{4}`
- `booktabs` table (`\footnotesize`), columns: **Route** | **Dạng câu hỏi** | **Cách xác minh** | **Số câu**:
  - FACTUAL (xác thực factual) | “A có dẫn đến Y?” (hỏi xuôi hoặc ngược) | gán seed = TRUE; yêu cầu outcome TRUE, mọi hop có rule kích hoạt, không xung đột | 230
  - DIRECT (cạnh trực tiếp) | “A trực tiếp gây ra Y?” | ground hai đầu mút (≥ 0,80), rồi kiểm tra cạnh $A\to Y$ | 13
  - INTERVENTION (can thiệp) | “Bỏ qua bước $m$, Y còn xảy ra?” | ground sự kiện trung gian (≥ 0,80), chạy $do(X_m=0)$, suy luận lại điểm bất động | 7
  - Colour each route name with `stepVerify` and make the counts bold.
- Two example lines below:
  - FACTUAL — \exid{} (ví dụ xuyên suốt) → \labSup.
  - DIRECT — `BLHS_MH_0221`: «Nhận định “Chống trả rõ ràng quá mức cần thiết” trực tiếp gây ra “Chịu trách nhiệm hình sự” mà không cần sự kiện trung gian là đúng hay sai?» → hai đầu mút được ground với độ tin cậy 1,0; đồ thị chỉ có đường qua “Vượt quá giới hạn PVCĐ” (rule 47 → rule 48) → \labRej. (Refer back to the slice on F2.5.)
- `\scriptsize` footnote: "Router dựa trên dấu hiệu từ vựng của câu hỏi; chỉ đường chính (path 0) được xác minh cấu trúc."
- Sources: DECK frame 6; PV4 (three routes); S-sum audit table (Factual route 230, Direct-edge route 13, do() requested/executed 7); LOG-full (scm_grounding_threshold 0.8); B `BLHS_MH_0221` question; P-full `BLHS_MH_0221` (`runner_scm_guard.endpoint_alignment_confidence` 1.0, `decision_explanation`, path rules 47 and 48, decision REJECT_DIRECT_CLAIM); AN §4 (lexical/regex router, path-0-only structural verification).

#### F2.12 — "Can thiệp $do(X_m=0)$ không phải là xóa nút" · `\setroadmap{4}`
- Top: two TikZ panels side by side, both showing $A\to m\to Y$ with a dashed alternative $A\to D\to Y$:
  - Left panel, title "Xóa nút (chẩn đoán tô-pô)": node $m$ crossed out; $G^{-m}=G[V\setminus\{m\}]$; $\rho_{\text{top}}=\mathbb I[\,Y\text{ đạt được từ }s\text{ trong }G^{-m}\,]$ — "Diễn giải: chỉ hỏi còn đường khác hay không (giới hạn 3 hop, 30 đường)."
  - Right panel, title "Can thiệp cấu trúc": node $m$ labelled "= FALSE", incoming mechanism edge cut; $\mathcal R^{do(X_m=0)}=\{r:\operatorname{var}(\operatorname{head}(r))\neq m\}$, $\mathbf x^{CF(m)}=\operatorname{Fix}\big(\Phi_{\mathcal R^{do(X_m=0)}},\mathbf x^{(0)}[X_m\leftarrow0]\big)$ — "Diễn giải: cố định $m$ = FALSE, bỏ cơ chế sinh ra $m$, rồi suy luận lại theo luật."
  - Between the panels: $\boxed{G^{-m}\neq do(X_m=0)}$ (Pearl, 2009).
- Bottom box `\exboxof{stepVerify}{0.95\linewidth}{\texttt{BLHS\_MH\_0223}}{…}` (not `\exbox`, which is reserved for the running example): «Có thể bỏ qua bước “Tình thế cấp thiết” và kết luận rằng “Gây thiệt hại nhỏ hơn để tránh thiệt hại lớn hơn” trực tiếp dẫn đến “Không bị coi là tội phạm” hay không?»
  - Factual: Gây thiệt hại nhỏ hơn… = TRUE → Tình thế cấp thiết = TRUE → Không bị coi là tội phạm = TRUE.
  - $do(\text{Tình thế cấp thiết}=0)$: Không bị coi là tội phạm: TRUE → FALSE (rule 49 và 50 không còn kích hoạt) ⇒ sự kiện trung gian là NECESSARY ⇒ không thể bỏ qua ⇒ \labRej (điểm 0,723, đã commit).
- `\scriptsize` line: "Can thiệp tương đối với mô hình quy tắc đã mã hóa; không có bước abduction biến ngoại sinh."
- `\keymsg{Xóa nút hỏi “còn đường khác không?”; $do(X_m=0)$ hỏi “theo luật, hệ quả còn được suy ra không?”.}`
- Sources: PV3 (§"Chẩn đoán tô-pô và can thiệp cấu trúc": $G^{-m}$, $\rho_{\text{top}}$, $\mathcal R^{do}$, $\mathbf x^{CF(m)}$, necessity rule, no abduction); PV2 (Pearl 2009); DECK frame 6 (3 hops, 30 paths; boxed inequality); LOG-full (max_cf_hops 3, max_cf_paths 30); B `BLHS_MH_0223` question; P-full `BLHS_MH_0223` (`counterfactual_summary.factual_states` all true; `interventions[0]`: assignment {tinh_the_cap_thiet: false}, factual_outcome true, counterfactual_outcome false, status NECESSARY, disabled_rule_ids [49, 50]; decision_score 0.72344; guard VERIFIER_COMMITTED; mediator_alignment_confidence 1.0).

#### F2.13 — "Cổng commit: chỉ tin bộ xác minh khi đủ điều kiện" · `\setroadmap{4}`
- Top TikZ (restyled DECK frame 7): `Bước 4: quyết định + audit` (stepVerify) → diamond `Đạt cổng?` → upper branch "đạt" → `$z_{\text{SCM}}$ + bằng chứng đã xác minh` → `LLM chung → $(a,C)$` (stepAnswer); lower branch "không đạt" → `giữ ngữ cảnh Bước 3; LLM tự quyết` → same LLM node.
- Formula: $G_{\text{commit}}=\mathbb I\big[g_{\text{ground}}\wedge g_{\text{path}}\wedge(s_{\text{SCM}}\ge\tau)\wedge g_{\text{consistent}}\big],\ \tau=0{,}55$ — "Diễn giải: chỉ khi cả bốn điều kiện đạt, quyết định của bộ xác minh mới có thẩm quyền."
- $z=z_{\text{SCM}}$ nếu $G_{\text{commit}}=1$, ngược lại $z=z_{\text{LLM}}$; $\mathcal E^{*}=\mathcal E_{\text{verified}}$ hoặc $\mathcal E_{\text{path}}$.
- Bullet: Khi không đạt: không ép \labUnc, không biến chẩn đoán tô-pô thành quyết định SCM → Full LegalSCM là hệ lai có chọn lọc.
- Right `\exbox{stepVerify}{…}` checklist (use `\ok`): đường chính hợp lệ (path 0); mọi hop có rule kích hoạt; 0 xung đột; outcome TRUE; $s_{\text{SCM}}=0{,}577\ge0{,}55$ → `VERIFIER_COMMITTED`.
- `\keymsg{Kết luận hình thức khi đủ căn cứ; quay về ngữ cảnh truy xuất khi không.}`
- Sources: PV4 (eq. selective_commit, §"Cổng cam kết chọn lọc"); LOG-full (scm_commit_threshold 0.55, scm_grounding_threshold 0.8); DECK frame 7 (commit conditions); P-full `BLHS_MH_0081` `verification.runner_scm_guard` (status VERIFIER_COMMITTED, primary_path_ids [0], all_hops_covered true, structural_conflict_count 0, factual_outcome true, original_decision_score 0.57681).

#### F2.14 — "Bước 5: sinh câu trả lời và vết suy luận đầy đủ" · `\setroadmap{5}`
- Top two columns (0.42 / 0.54):
  - Left: $I_q=(q,P^{*},z,\mathcal E^{*}),\qquad (a,C)=f_\theta(I_q)$ — "Diễn giải: LLM diễn đạt quyết định và bằng chứng được cung cấp, kèm trích dẫn."
  - Right bullets: cùng Qwen3-8B, prompt, JSON schema, parser và seed cho mọi hệ thống; không dùng fallback trích xuất; trích dẫn chỉ trong tập điều luật được cung cấp; ghi audit: route, grounding, can thiệp, commit, fallback, bằng chứng cuối.
- Bottom: trace strip titled "Ví dụ \exid{} qua năm bước" — five `stepnode`s (text width 2.5 cm, min height 1.9 cm, `\scriptsize`) joined by `flow` arrows:
  1. (stepGraph) **1 · Đồ thị** — chuỗi rule 47 (Điều 22) → rule 106 (Điều 51) có sẵn.
  2. (stepMemory) **2 · Bộ nhớ** — EVENT \#2956, \#2957, \#3047; RULE 106 cos 0,653.
  3. (stepRetrieve) **3 · Truy xuất** — hướng ngược; 33 đường; $P_0$ hạng 1 (0,705).
  4. (stepVerify) **4 · Xác minh** — factual; 3 vòng; outcome TRUE; 0,577 ≥ 0,55 → commit.
  5. (stepAnswer) **5 · Trả lời** — \labSup; “Sự kiện trước là ‘Chống trả rõ ràng quá mức cần thiết’.”; Điều 22, 51; 2,83 giây.
- Sources: PV4 (§"Sinh câu trả lời và khả năng kiểm toán"); DECK frame 7 (generation contract; extractive fallback off); S-rep `generation_signatures` (identical contract fields for all systems, `extractive_fallback` false); P-full `BLHS_MH_0081` (`final_answer`, `citations` [Điều 22, Điều 51], `runtime_seconds` 2.829465, and the values cited on F2.6–F2.13).

### Section 3 — `3_thuc_nghiem_va_danh_gia.tex` — `\setroadmap{-1}` before `\section{Thực nghiệm và đánh giá}` (3 frames)

#### F3.1 — "Benchmark: 250 câu hỏi đa bước từ đồ thị BLHS"
- Layout: two columns 0.52 / 0.44.
- Left pgfplots `xbar` chart (bars `schemaNavy`; the "Suy luận ngược" bar in `stepRetrieve` with tag "loại của ví dụ"), categories top→bottom with counts as `nodes near coords`: Đa bước xuôi 70; Suy luận ngược 45; Sự kiện cầu nối 40; Giải thích chuỗi 35; Có/không – khẳng định 20; Có/không – phản ví dụ 20; Hội tụ 10; Cầu nối/hội tụ (dự phòng) 6; Rẽ nhánh 4.
- Right bullets:
  - Nhãn chuẩn: 230 \labSup, 20 \labRej; không có \labUnc.
  - 236 câu có đường chuẩn tuyến tính; 14 câu rẽ nhánh/hội tụ không tính vào chỉ số đường.
  - 20 mẫu âm: 13 khẳng định cạnh trực tiếp và 7 câu bỏ qua sự kiện trung gian.
  - Mỗi câu có gold: rule, sự kiện, đường, câu trả lời, trích dẫn; các trường gold không được truy cập khi sinh dự đoán.
- `alertblock` "Lưu ý về tính hợp lệ": "Câu hỏi và đồ thị cùng một nguồn có cấu trúc → thực nghiệm đo tính hợp lệ nội tại; benchmark cần được rà soát thủ công trước khi dùng làm chuẩn chính thức."
- Sources: B `metadata.actual_distribution` (70/45/40/35/20/20/10/6/4), `metadata.validation_summary` (questions_with_linear_gold_path 236; gold_decision_distribution 230/20), `metadata.construction_notes`; PV5 ¶1 (gold never accessed; internal validity); S-sum (13 direct-edge, 7 do()); README (linear path metrics exclude 14 branch/convergence samples).

#### F3.2 — "Năm cấu hình: chỉ thay ngữ cảnh và bước xác minh"
- Layout: TikZ "ladder" (left, 0.60) + `block` (right, 0.36).
- Ladder: five rows; each row = system tag (rounded box in its system colour, D10) followed by chips (`stepnode` in step colours) and an arrow to a gray `LLM` chip:
  1. Closed-book (abstain): [Câu hỏi] → LLM (cho phép \labUnc)
  2. Closed-book (forced): [Câu hỏi] → LLM (buộc chọn \labSup hoặc \labRej) — tag "đối chứng độ nhạy"
  3. Vanilla RAG: [Câu hỏi] + [5 điều luật thô · BGE-M3/FAISS] → LLM
  4. Causal Path RAG: [Câu hỏi] + [đường dẫn, rule, sự kiện · Bước 3] → LLM
  5. Full LegalSCM: [Câu hỏi] + [Bước 3] + [xác minh LegalSCM + cổng commit · Bước 4] → LLM
- Right `block` "Nguyên tắc so sánh công bằng": bốn cấu hình cho phép abstention dùng cùng bộ sinh, quy tắc prompt, JSON schema, parser, ngân sách ngữ cảnh và tham số giải mã; biến thể forced chỉ đổi chính sách quyết định; không hệ thống nào dùng fallback trích xuất; Causal Path RAG và Full dùng đầu ra truy xuất giống hệt nhau.
- `\keymsg{So sánh Full LegalSCM với Causal Path RAG cô lập đóng góp của Bước 4.}`
- Sources: PV5 §"Dữ liệu và các hệ thống so sánh"; README (the four modes, top-5 raw articles); S-rep `generation_signatures` and `generation_contract_warnings` (forced differs only in decision_policy); PV6 §"Truy xuất…" (identical retrieval outputs); DECK frame 8.

#### F3.3 — "Cấu hình chung, chỉ số và kiểm định thống kê"
- Layout: two columns 0.45 / 0.51.
- Left `booktabs` table (`\footnotesize`), header **Tham số** | **Giá trị**:
  - Bộ sinh / embedding | Qwen3-8B (Ollama) / BGE-M3
  - Nhiệt độ / seed | 0,1 / 42
  - Token đầu ra / cửa sổ ngữ cảnh | 1.200 / 32.768
  - Ngân sách ngữ cảnh | 18.000 ký tự
  - Top-$K$ Vanilla / sự kiện / rule | 5 / 8 / 8
  - Bằng chứng cuối / số hop tối đa | 12 / 2
  - Ngưỡng commit / grounding | 0,55 / 0,80
- Right `block` "Chỉ số":
  - Quyết định: Decision Accuracy, Balanced Accuracy, Macro-F1 (trung bình trên 2 lớp có trong gold). $\text{B.Acc}=\tfrac12\big(\text{Recall}_{\text{SUP}}+\text{Recall}_{\text{REJ}}\big)$ — "Diễn giải: trung bình recall của hai lớp, không bị lớp đa số chi phối."
  - Câu trả lời: Token-F1, ROUGE-L; trích dẫn: F1 theo tập điều luật.
  - Truy xuất: Recall@5, MRR, MAP; đường: Top-1 và Oracle Exact Path (Oracle chỉ để chẩn đoán).
  - Dự đoán lỗi nhận điểm 0; thời gian tính trên các dự đoán thành công.
- Right `block` "Kiểm định theo cặp trên cùng 250 câu": bootstrap 5.000 mẫu → CI 95%; McNemar chính xác cho Decision Accuracy; 5.000 phép đổi dấu ngẫu nhiên cho Answer F1, ROUGE-L, Citation F1; hiệu chỉnh Holm.
- Sources: PV5 (configuration table; §"Chỉ số và kiểm định thống kê"); LOG-full `configuration` (model qwen3:8b, temperature 0.1, seed 42, max_tokens 1200, num_ctx 32768, max_context_chars 18000, vanilla_top_k 5, event_top_k 8, direct_rule_top_k 8, final_top_k 12, max_hops 2, scm_commit_threshold 0.55, scm_grounding_threshold 0.8); S-rep `statistical_settings` (5000 bootstrap, 5000 permutation, seed 42, Holm, exact McNemar, paired sign-flip).

### Section 4 — `4_ket_qua_va_phan_tich.tex` — `\section{Kết quả và phân tích}` (8 frames)

#### F4.1 — "Full LegalSCM đạt độ chính xác quyết định cao nhất"
- `booktabs` table (`\small`; `\setlength{\tabcolsep}{0.16cm}`), group headers with `\cmidrule`: "Quyết định" over the first 3 metrics, "Văn bản và trích dẫn" over the next 3, "Chi phí" over the last.
  - Columns: **Phương pháp** | **Acc.** | **B.Acc.** | **M-F1** | **Ans.F1** | **R-L** | **Cit.F1** | **Giây**
  - Closed-book (abstain) | 4,40 | 27,50 | 22,00 | 15,08 | 13,19 | 0,00 | 1,031
  - Closed-book (forced) | 9,20 | 50,65 | 8,78 | 18,44 | 16,19 | 0,27 | 0,900
  - Vanilla RAG | 55,20 | 71,09 | 49,88 | 28,18 | 25,68 | 50,75 | 2,369
  - Causal Path RAG | 88,80 | 89,35 | 76,30 | 37,27 | 36,40 | 58,41 | 1,947
  - **Full LegalSCM** (row `\rowcolor{schemaLight}`) | **96,80** | **98,26** | **91,67** | **37,60** | **36,92** | **58,89** | 3,238
- Note (`\scriptsize`): "Đơn vị %, trừ thời gian (giây/câu, trung bình trên dự đoán thành công). Cả 5 lượt chạy sinh đủ 250/250 dự đoán, không lỗi pipeline."
- `\keymsg{Quyết định tăng theo từng thành phần: 55,20 (Vanilla RAG) → 88,80 (đường dẫn) → 96,80 (xác minh); chất lượng văn bản gần như không đổi giữa hai hệ thống cuối.}`
- Sources: S-sum "Main comparison" table (all 35 numbers); PV6 table `tab:main_results` (identical) and ¶1 (250/250, no pipeline errors); DECK frame 9.

#### F4.2 — "Accuracy thô dễ gây hiểu lầm khi dữ liệu lệch lớp"
- Layout: two columns 0.55 / 0.41.
- Left pgfplots grouped `ybar` (x: the 5 systems, short labels "CB (abstain)", "CB (forced)", "Vanilla", "Causal Path", "Full"; bars: Acc. `hustprimary`, B.Acc. `hustprimary!60`, Macro-F1 `hustprimary!30`; ymin 0, ymax 100; legend top-left). Dashed `schemaRed` horizontal line at y = 92 with label "luôn đoán SUPPORTED: Acc. 92%". Values: Acc 4,40 / 9,20 / 55,20 / 88,80 / 96,80; B.Acc 27,50 / 50,65 / 71,09 / 89,35 / 98,26; Macro-F1 22,00 / 8,78 / 49,88 / 76,30 / 91,67.
- Right bullets:
  - Lớp lệch 230/20: một bộ phân loại luôn đoán \labSup đã đạt 92% Accuracy → cần Balanced Accuracy và Macro-F1.
  - Closed-book (abstain): 220/250 câu trả \labUnc → chỉ 11 câu đúng (4,40%).
  - Closed-book (forced): 247/250 câu đoán \labRej → 9,20%; bỏ abstention tạo thiên lệch bác bỏ chứ không khôi phục năng lực pháp luật.
  - Điểm closed-book phản ánh hành vi dưới hợp đồng nhãn định hướng xác minh khi không có bằng chứng, không phải năng lực pháp luật tổng quát của Qwen3-8B.
- `\keymsg{Luôn báo cáo Balanced Accuracy và Macro-F1 cùng với Accuracy.}`
- Sources: S-sum (Acc., B.Acc., Macro-F1); S-rep `prediction_run_audits` (llm_only decision_counts UNCERTAIN 220 / REJECT 30; llm_only_forced_binary REJECT 247 / SUPPORTED 3); PV6 ¶2 (11 correct; 92%); PV7 ¶2.

#### F4.3 — "Mức tăng +8,0 điểm có ý nghĩa thống kê"
- Layout: two columns 0.42 / 0.54.
- Left TikZ 2×2 matrix (cells 1.6 cm): rows "Full đúng", "Full sai"; columns "Causal Path đúng", "Causal Path sai"; cells: 219 | **23** (fill `stepVerify!35`) / 3 (fill `stepRetrieve!35`) | 5. Margins: "Full: 242/250", "Causal Path RAG: 222/250". Caption: "McNemar chỉ dựa vào các cặp bất đồng: 23 so với 3."
- Right bullets:
  - $\Delta\text{Acc}=+8{,}0$ điểm; CI 95% bootstrap $[4{,}4;\,12{,}0]$.
  - McNemar chính xác: $p=8{,}8\times10^{-5}$; hiệu chỉnh Holm: $p=0{,}0011$.
  - Macro-F1: 76,30 → 91,67; F1 của lớp âm: 59,02 → 85,11.
  - Answer F1 / ROUGE-L / Citation F1: +0,33 / +0,52 / +0,47 điểm; CI đều chứa 0; $p$ Holm = 1,0.
- Right small `booktabs` table (`\scriptsize`) "Full LegalSCM so với các baseline khác": Vanilla RAG | +41,6 | [35,2; 48,0] | $<0{,}001$ ; Closed-book (forced) | +87,6 | [83,6; 91,6] | $<0{,}001$ ; Closed-book (abstain) | +92,4 | [89,2; 95,6] | $<0{,}001$ (columns: Baseline | ΔAcc (điểm) | CI 95% | $p$ Holm).
- `\keymsg{Cải thiện có ý nghĩa nằm ở quyết định có cấu trúc; văn bản tự do và trích dẫn chưa khác biệt.}`
- Sources: S-sig (all deltas, CIs, p, Holm; `reference_only_correct` 23 / `comparator_only_correct` 3 for causal_path_rag; Holm p 0.0 for the other decision rows → reported as "< 0,001"); PV6 §"Ảnh hưởng của xác minh chọn lọc" (242/222, 23/3/219/5, negative-class F1 59,02 → 85,11); S-rep `per_class` (REJECT f1 0.590164 / 0.851064); S-sum (Macro-F1).

#### F4.4 — "Lợi ích đến từ việc sửa các lần bác bỏ nhầm"
- Layout: two confusion matrices side by side (tabular, `\footnotesize`, `\cellcolor` intensity ∝ count/row total, colour of the system), rows = gold (\labSup 230, \labRejS 20), columns = dự đoán (SUP / REJ / UNC):
  - Causal Path RAG (`stepRetrieve`): SUP → 204 / 23 / 3; REJ → 2 / 18 / 0.
  - Full LegalSCM (`stepVerify`): SUP → 222 / 7 / 1; REJ → 0 / 20 / 0.
- Bullets below:
  - Causal Path RAG bác bỏ nhầm 23 câu \labSup; Full LegalSCM còn 7.
  - Full đúng 20/20 mẫu âm (Causal Path RAG: 18/20).
  - Precision lớp REJECT: 43,90% → 74,07%; recall lớp SUPPORTED: 88,70% → 96,52%.
- Legend line: "REJECT = \labRej."
- `\keymsg{Trên cùng ngữ cảnh truy xuất, bộ xác minh chủ yếu sửa các lần LLM bác bỏ nhầm câu được luật hỗ trợ.}`
- Sources: S-rep `aggregates.{causal_path_rag,full_legal_scm}.verification.confusion_matrix` and `per_class` (REJECT precision 0.439024 / 0.740741; SUPPORTED recall 0.886957 / 0.965217); PV6 (identical retrieval outputs).

#### F4.5 — "Mọi lỗi của Full xảy ra khi thiếu đường chính"
- TikZ flow, left→right, `\scriptsize`: `250 câu` → three route boxes (stepVerify shades) `Factual 230`, `Cạnh trực tiếp 13`, `Can thiệp 7` → gate boxes: Factual splits into `Commit 210` and `Fallback 20 · không có đường chính` (gray, dashed); Cạnh trực tiếp → `Commit 13`; Can thiệp → `Commit 7`. Two outcome boxes on the right:
  - from all commits: "230 câu đã commit: 0 lỗi; 23 câu chỉ Full đúng = 21 factual + 2 cạnh trực tiếp + 0 can thiệp" (green border);
  - from fallback: "20 câu fallback: 8 lỗi của Full (7 câu suy luận ngược); Causal Path RAG đúng thêm 3 câu; Full không có cải thiện riêng" (red border).
- Bullet below: "$do(X_m=0)$ được yêu cầu và thực thi 7/7; Causal Path RAG cũng đúng 7/7 câu này → chưa tách được lợi ích riêng của can thiệp."
- `\keymsg{Điểm yếu còn lại nằm ở truy xuất (thiếu đường chính), không ở các quyết định đã commit.}`
- Sources: S-sum audit table (factual 230, direct-edge 13, do() requested 7 / executed 7, committed 230, Step-3 fallback 20); S-bys (route × guard: factual+committed 210, factual+NO_PRIMARY_PATH 20, direct+committed 13, intervention+committed 7; all 8 Full errors have guard NO_PRIMARY_PATH; intervention samples 0211/0214/0217/0220/0223/0226/0229 correct for both systems); PV6 §"Kiểm toán xác minh và lỗi" (21 + 2, 0 from intervention, 8 errors, 7 reverse-reasoning, 3 Causal-only in fallback); DECK frame 10.

#### F4.6 — "Ví dụ xuyên suốt: một câu hỏi, năm câu trả lời"
- Top line (`\footnotesize`): "\exid{} — gold: \labSup; đáp án “Chống trả rõ ràng quá mức cần thiết”; căn cứ Điều 22, Điều 51."
- `booktabs` table (`\footnotesize`), columns **Hệ thống** | **Quyết định** | **Câu trả lời (nguyên văn; PVCĐ viết tắt)** | **Trích dẫn** | **Kết quả**:
  - Closed-book (abstain) | \labUnc | “Không đủ thông tin để xác định sự kiện trước bước ‘Vượt quá giới hạn PVCĐ’.” | — | \ko
  - Closed-book (forced) | \labRejS | (như trên) | — | \ko
  - Vanilla RAG | \labSup | “Sự kiện trước đó là ‘Phạm tội trong trường hợp vượt quá giới hạn PVCĐ’.” | Điều 51, 126 | nhãn \ok, nội dung \ko
  - Causal Path RAG | \labRejS | “Không đủ căn cứ xác định sự kiện trước ‘Vượt quá giới hạn PVCĐ’.” | Điều 22, 51 | \ko
- `\pause` (the only overlay in the deck)
- `exampleblock` "Full LegalSCM": \labSup — “Sự kiện trước là ‘Chống trả rõ ràng quá mức cần thiết’.” — Điều 22, 51 — \ok (đã commit: 0,577 ≥ 0,55; 2,83 giây).
- Bullet: "Causal Path RAG và Full LegalSCM nhận cùng 8 bằng chứng (Điều 22 và 51); chỉ Full có kết quả xác minh trong prompt."
- `\scriptsize` note: "Đúng nhãn chưa chắc đúng lập luận (Vanilla RAG). Token-F1 của câu “không đủ thông tin” vẫn là 36,67% → chỉ số trùng từ không đo đúng/sai nội dung."
- `\keymsg{Truy xuất đúng chưa đủ; xác minh biến bằng chứng đúng thành quyết định đúng.}`
- Sources: P-cba, P-cbf, P-van, P-cpr, P-full — `BLHS_MH_0081`: `verification_decision`, `final_answer`, `citations`, `pipeline_metadata.selected_evidence_count` (8 for both Causal and Full), `pipeline_metadata.prompt_includes_verification` (False / True), `retrieved_article_ids` (22, 51); S-bys `BLHS_MH_0081` rows (`decision_accuracy`; llm_only `answer_token_f1` 0.366667; citation_f1 1.0 for Causal/Full, 0.5 for Vanilla). Full data in §6.

#### F4.7 — "Nút thắt: đường đúng có trong ứng viên nhưng chưa xếp đầu"
- Layout: two columns 0.47 / 0.49, two pgfplots `ybar` charts (ymin 0, ymax 100, values as nodes near coords).
  - Left "Truy xuất (Recall@5)": Article — Vanilla RAG 60,07 (`sysVan`) vs Causal Path RAG = Full 75,40 (`sysCPR`); Rule 71,20; Event 74,79 (Vanilla RAG: không áp dụng). Caption: "Causal Path RAG và Full dùng cùng đầu ra truy xuất."
  - Right "Đường suy luận (236 câu có đường tuyến tính)": Exact Path — Top-1 38,56 vs Oracle 75,42; Edge F1 — Top-1 58,19 vs Oracle 79,80. Brace on Exact Path labelled "khoảng cách 36,86 điểm".
- Bullets: đường phù hợp thường đã được truy xuất nhưng chưa được xếp hạng đầu tiên; cả 8 lỗi của Full xảy ra khi không có đường chính; MRR: điều luật 82,87%, rule 78,12%, sự kiện 80,42% (Vanilla RAG: điều luật 72,84%).
- `\keymsg{Ưu tiên tiếp theo: cải thiện xếp hạng đường và grounding, trước khi tăng độ phức tạp của SCM.}`
- Sources: S-sum (Article R@5 60,07 / 75,40; Rule R@5 71,20; Event R@5 74,79; Top-1 Path 38,56); S-rep `aggregates.causal_path_rag` (`causal_path`: applicable_sample_count 236, top1_exact_path_match 0.385593, top1_edge_f1 0.581921, oracle_exact_path_match 0.754237, oracle_edge_f1 0.798023; `retrieval`: article_mrr 0.828744, rule_mrr 0.7812, event_mrr 0.804238) and `aggregates.vanilla_rag.retrieval.article_mrr` 0.7284; PV6 §"Truy xuất, xếp hạng đường và thời gian chạy"; PV7 ¶4; AN §9; gap 36,86 = 75,42 − 38,56 (computed).

#### F4.8 — "Đánh đổi: +8,0 điểm quyết định, +66% thời gian"
- Layout: two columns 0.55 / 0.41.
- Left pgfplots scatter: x "Thời gian trung bình (giây/câu)" 0–3,5; y "Decision Accuracy (%)" 0–100; five labelled points in system colours: Closed-book (abstain) (1,031; 4,40), Closed-book (forced) (0,900; 9,20), Vanilla RAG (2,369; 55,20), Causal Path RAG (1,947; 88,80), Full LegalSCM (3,238; 96,80); arrow Causal Path → Full labelled "+1,29 giây; +8,0 điểm".
- Right bullets:
  - Full LegalSCM: 3,238 giây/câu so với 1,947 giây (≈ +66%); trung vị 3,118 so với 1,795 giây.
  - 250 câu của Full LegalSCM chạy trong 956,75 giây (≈ 16 phút).
  - Một phần chi phí đến từ việc dựng lại tài nguyên đồ thị và bộ nhớ cho từng câu ở Bước 4 → có thể nạp một lần cho cả lượt chạy.
- `\keymsg{Đổi thêm 1,29 giây mỗi câu lấy +8,0 điểm Decision Accuracy.}`
- Sources: S-sum (Avg. sec); S-rep `aggregates.*.runtime.median_seconds_successful_only` (3.11831; 1.795336); PV6 last ¶ (≈ 66%); LOG-full `elapsed_seconds` 956.752942; AN §7 M-4 (per-sample reload of graph/memory in Step 4); 1,29 = 3,238 − 1,947 (computed).

### Section 5 — `5_ket_luan.tex` — `\section{Kết luận}` (3 frames)

#### F5.1 — "Trả lời bốn câu hỏi nghiên cứu"
- Layout: 2×2 `\rqcard`s with the same colours and tags as F1.6; each card body starts with a bold verdict.
  - RQ1 · Truy xuất có cấu trúc — **Có.** Article Recall@5 75,40% so với 60,07% (Vanilla RAG); Rule/Event Recall@5: 71,20% / 74,79%.
  - RQ2 · Đường suy luận đa bước — **Một phần.** Đường phù hợp có trong tập ứng viên ở 75,42% câu (Oracle) nhưng chỉ 38,56% đứng đầu (Top-1) → xếp hạng là nút thắt; 8/8 lỗi của Full xảy ra khi thiếu đường chính.
  - RQ3 · Xác minh chọn lọc — **Có, cho quyết định.** +8,0 điểm Decision Accuracy (CI 95% [4,4; 12,0]; $p$ Holm = 0,0011); 23 câu chỉ Full đúng: 21 factual + 2 cạnh trực tiếp; lợi ích riêng của $do(X_m=0)$ chưa tách được (7/7 ở cả hai hệ thống).
  - RQ4 · Hiệu quả đầu–cuối — **Có so với closed-book và Vanilla RAG; chưa so với Causal Path RAG.** Answer F1 37,60% so với 15,08% / 28,18%; Citation F1 58,89% so với 0,00% / 50,75% ($p$ Holm ≤ 0,0096); so với Causal Path RAG: không khác biệt có ý nghĩa ($p$ Holm = 1,0).
- Sources: S-sum (R@5, Top-1, Answer F1, Citation F1); S-rep (oracle 0.754237); S-sig (causal 0.08 [0.044, 0.12], Holm 0.001144; vs vanilla/llm_only answer and citation Holm 0.0024 and 0.009598; vs causal Holm 1.0); PV6 (21 + 2; 7/7; 8 errors after fallback); PV7.

#### F5.2 — "Hạn chế và các mối đe dọa tính hợp lệ"
- Layout: 2×2 `alertblock`/`block` grid (`\footnotesize`), then `\keymsg`.
  - `block` "Tính hợp lệ nội tại": câu hỏi được sinh từ chính đồ thị và nguồn có cấu trúc mà hệ thống truy xuất; benchmark chưa được rà soát thủ công.
  - `block` "Tính hợp lệ bên ngoài": chỉ Bộ luật Hình sự và 250 câu sinh từ đồ thị; cần câu hỏi do chuyên gia viết hoặc rà soát, nằm ngoài đồ thị.
  - `block` "Kết luận thống kê": chỉ 20 mẫu âm và 7 câu can thiệp → chưa đủ để tách lợi ích của $do(X_m=0)$; lớp lệch 230/20, không có gold \labUnc.
  - `block` "Tính hợp lệ cấu trúc": mỗi rule chỉ một tiền đề dương (0 rule hội, 0 ngoại lệ) → phép hội chưa được kiểm chứng; Token-F1/ROUGE-L đo trùng từ và tính nhất quán câu trả lời–quyết định chưa được ép buộc; LegalSCM là mô hình quy tắc quy phạm tất định, không khám phá nhân quả, không có abduction.
- `\keymsg{96,80\% là bằng chứng nội tại trong môi trường có kiểm soát, chưa phải năng lực tư vấn pháp lý.}`
- Sources: PV5 ¶1 (internal validity); PV7 (all limitations: 7/7, one antecedent, path ranking, small graph-derived benchmark, expert review, deterministic model, no abduction, answer–decision consistency as future work); PV3 (one positive antecedent); P-full `rule_semantics` (0 conjunctive rules, 0 exceptions); B `construction_notes` (manual review needed); S-bys `BLHS_MH_0081` (Token-F1 example used on F4.6); DECK frame 5 and 10; AN §9 (no legal-advice use). Decision D11: no modality item.

#### F5.3 — "Kết luận và hướng phát triển"
- Layout: two `block`s (0.49 / 0.47), then 3 `\metric` cards.
- `block` "Kết luận":
  - LegalSCM-RAG kết hợp truy xuất đường dẫn nhân quả, suy luận quy tắc ba giá trị và xác minh cấu trúc chọn lọc cho hỏi đáp pháp luật đa bước tiếng Việt.
  - Trên 250 câu: Decision Accuracy 96,80%, cao hơn Causal Path RAG 8,0 điểm với bằng chứng thống kê theo cặp; câu trả lời và trích dẫn chưa cải thiện có ý nghĩa.
  - Kết quả ủng hộ xác minh factual và kiểm tra khẳng định trực tiếp có chọn lọc; lợi ích độc lập của $do(X_m=0)$ chưa được xác lập.
- `block` "Hướng phát triển":
  - mở rộng benchmark về can thiệp sự kiện trung gian và quy tắc nhiều tiền đề;
  - cải thiện xếp hạng đường và grounding;
  - ép tính nhất quán giữa câu trả lời và quyết định;
  - đánh giá chuyên gia trên câu hỏi không sinh từ đồ thị nguồn;
  - giảm thời gian chạy bằng cách nạp đồ thị và bộ nhớ một lần cho cả lượt chạy.
- Metrics: `\metric{96,80\%}{Decision Accuracy}` `\metric{+8,0}{điểm so với Causal Path RAG}` `\metric{0,0011}{$p$ hiệu chỉnh Holm}`.
- Sources: PV7 §"Kết luận" (summary and future work); S-sum; S-sig; AN §7 M-4 (runtime direction).

Then: `\hidePageNumber` and `\hustthankyou` (structural).

### Appendix — `6_phu_luc.tex` — `\setroadmap{-1}`, `\section{Phụ lục}` (7 backup frames, not counted)

#### A1 — "Phụ lục A1 – Công thức xếp hạng của Bước 3"
- $s_{\text{sup}}=\min\!\Big(1,\ \tfrac1h\sum_{k=1}^{h}\ln\big(1+\max(1,n_k)\big)\big/\ln4\Big)$, $n_k$: số rule hỗ trợ bước $k$ — "Diễn giải: bước được nhiều rule hỗ trợ thì tin cậy hơn, bão hòa ở 1."
- $b_{\text{hop}}=0{,}18$ nếu $h=h^{*}=2$; ngược lại $0{,}03\cdot h/h^{*}$.
- $S_{\text{rule}}=0{,}32\,S_{\text{sem}}+0{,}36\,S_{\text{graph}}+0{,}10\,S_{\text{seed}}+0{,}07\,S_{\text{direct}}+0{,}15\,S_{\text{priority}}$ — "Diễn giải: trọng số đồ thị/đường cao để giữ rule ở bước thứ hai dù câu hỏi chủ yếu nhắc đến sự kiện đầu chuỗi."
- Ví dụ: rule 106 = 0,32·0,653 + 0,36·0,705 + 0,10·0,604 + 0,07·0,653 + 0,15·1,0 = 0,719; rule 47 = 0,32·0,555 + 0,36·0,705 + 0,10·0,604 + 0 + 0,15·1,0 = 0,642.
- Cấu hình: 100 kết quả ngữ nghĩa; tối đa 200 rule ứng viên; 30 đường mỗi seed; 12 bằng chứng; ngưỡng sự kiện 0,20, rule 0,15.
- Sources: C3 lines 49–55, 1140–1206; P-full `retrieval.configuration.score_weights` (0.32/0.36/0.10/0.07/0.15) and `retrieval.retrieved_rules` (rule 106: semantic 0.6535, graph 0.7051, seed 0.6035, direct 0.6535, priority 1.0, final 0.7190; rule 47: 0.5549, 0.7051, 0.6035, 0.0, 1.0, 0.6418); LOG-full; DECK frame 4.

#### A2 — "Phụ lục A2 – Ngữ nghĩa LegalSCM và quy ước cài đặt"
- Full formulation: $X_i$, $(t_i,f_i)$, $\mathbf B^{\pm}\in\{0,1\}^{p\times n}$, $\mathbf H^{\pm}\in\{0,1\}^{n\times p}$, $\mathbf k=(\mathbf B^{+}+\mathbf B^{-})\mathbf 1$, $\mathbf b$, $\mathbf a$, $\mathbf d^{\pm}$, $\mathbf x^{F}$ ($n=|V|$, $p=|\mathcal R|$).
- Theo bài báo: chỉ có hỗ trợ dương → TRUE; chỉ có hỗ trợ âm → FALSE; xung đột → UNKNOWN.
- Theo cài đặt: sự kiện nội sinh (là hệ quả của ít nhất một rule) mặc định FALSE khi không cơ chế nào sinh ra; sự kiện gốc UNKNOWN cho đến khi có trong ngữ cảnh; xung đột → UNKNOWN.
- Kích thước thực nghiệm: 1.814 sự kiện, 2.884 rule; 187 sự kiện có hỗ trợ thay thế; 0 rule hội; 0 ngoại lệ.
- Sources: PV3; SCM docstring lines 7–11; P-full `rule_semantics`. (C7.)

#### A3 — "Phụ lục A3 – Can thiệp cấu trúc và chẩn đoán xóa nút"
- $\mathcal M(P^{*})=\{v_1,\dots,v_{h-1}\}$; $G^{-m}=G[V\setminus\{m\}]$; $\rho_{\text{top}}(s,Y\mid m)=\mathbb I[Y\text{ đạt được từ }s\text{ trong }G^{-m}]$.
- $do(X_m=0)$, $0\equiv\text{FALSE}$; $\mathcal R^{do(X_m=0)}=\{r\in\mathcal R\mid\operatorname{var}(\operatorname{head}(r))\neq m\}$; $\mathbf x^{CF(m)}=\operatorname{Fix}(\Phi_{\mathcal R^{do(X_m=0)}},\mathbf x^{(0)}[X_m\leftarrow0])$.
- $\nu(m;Y)$: NECESSARY nếu $x_Y^{F}=1,\ x_Y^{CF(m)}=0$; NON_NECESSARY nếu cả hai bằng 1; chưa xác định trong các trường hợp còn lại.
- Phạm vi: chỉ đường chính (path 0) được xác minh cấu trúc; chẩn đoán xóa nút giới hạn 3 hop, 30 đường; can thiệp tương đối với mô hình đã mã hóa, không có abduction.
- Sources: PV3; DECK frame 6; LOG-full (max_cf_hops 3, max_cf_paths 30); AN §4.

#### A4 — "Phụ lục A4 – Kiểm định theo cặp đầy đủ (tham chiếu: Full LegalSCM)"
- `booktabs` table (`\scriptsize`), columns: So với | Chỉ số | Δ (điểm) | CI 95% | Kiểm định | $p$ Holm. Rows (Δ and CI ×100 from S-sig):
  - Closed-book (abstain): Decision Acc. +92,40 [89,20; 95,60] McNemar <0,001 · Answer F1 +22,52 [19,07; 26,17] 0,0024 · ROUGE-L +23,72 [20,26; 27,19] 0,0024 · Citation F1 +58,89 [54,27; 63,49] 0,0024
  - Closed-book (forced): +87,60 [83,60; 91,60] <0,001 · +19,16 [15,93; 22,30] 0,0024 · +20,73 [17,66; 23,86] 0,0024 · +58,62 [54,18; 63,13] 0,0024
  - Vanilla RAG: +41,60 [35,20; 48,00] <0,001 · +9,42 [6,26; 12,61] 0,0024 · +11,23 [8,14; 14,39] 0,0024 · +8,14 [2,82; 13,56] 0,0096
  - Causal Path RAG: +8,00 [4,40; 12,00] 0,0011 · +0,33 [−1,19; 1,85] 1,0 · +0,52 [−0,95; 2,00] 1,0 · +0,47 [−1,00; 1,95] 1,0
- Footer: "Cặp bất đồng (Full đúng / baseline đúng): 231/0, 219/0, 107/3, 23/3. Kiểm định liên tục: đổi dấu ngẫu nhiên theo cặp, 5.000 lần."
- Sources: S-sig (every cell, including `reference_only_correct`/`comparator_only_correct`); S-rep `statistical_settings`.

#### A5 — "Phụ lục A5 – Chỉ số theo lớp và khoảng tin cậy"
- `booktabs` table (`\scriptsize`), columns: Hệ thống | SUP P / R / F1 | REJ P / R / F1 | Decision Acc. [CI 95%]:
  - Closed-book (abstain): 0 / 0 / 0 | 36,67 / 55,00 / 44,00 | 4,40 [2,00; 7,20]
  - Closed-book (forced): 100 / 1,30 / 2,58 | 8,10 / 100 / 14,98 | 9,20 [6,00; 12,80]
  - Vanilla RAG: 98,36 / 52,17 / 68,18 | 19,15 / 90,00 / 31,58 | 55,20 [49,20; 61,20]
  - Causal Path RAG: 99,03 / 88,70 / 93,58 | 43,90 / 90,00 / 59,02 | 88,80 [84,80; 92,40]
  - Full LegalSCM: 100 / 96,52 / 98,23 | 74,07 / 100 / 85,11 | 96,80 [94,40; 98,80]
- Footer: "Macro-F1 = trung bình F1 của hai lớp; lớp UNCERTAIN không có mẫu gold."
- Sources: S-rep `aggregates.*.verification.per_class` and `bootstrap_confidence_intervals.*.decision_accuracy`.

#### A6 — "Phụ lục A6 – Tám lỗi của Full LegalSCM"
- `booktabs` table (`\scriptsize`): Mẫu | Loại câu hỏi | Full | Causal Path RAG | Trạng thái cổng
  - BLHS_MH_0072, 0074, 0076, 0078, 0080 | suy luận ngược | \labRejS | \labRejS (cũng sai) | NO_PRIMARY_PATH
  - BLHS_MH_0088 | suy luận ngược | \labUnc | \labSup (đúng) | NO_PRIMARY_PATH
  - BLHS_MH_0108 | suy luận ngược | \labRejS | \labSup (đúng) | NO_PRIMARY_PATH
  - BLHS_MH_0209 | có/không – khẳng định | \labRejS | \labSup (đúng) | NO_PRIMARY_PATH
- Footer: "20 câu fallback: 18 suy luận ngược + 2 có/không – khẳng định; Full đúng 12/20, Causal Path RAG đúng 15/20; không có lỗi nào sau khi bộ xác minh commit."
- Sources: S-bys (per-sample predicted_decision, question_type, guard_status for both methods; fallback counts computed from S-bys).

#### A7 — "Phụ lục A7 – Tài liệu tham khảo chính"
- Two columns, `\scriptsize`, short form "Tác giả (năm). Tiêu đề rút gọn. Nơi công bố." for the 20 entries of PVR in this order: Lewis 2020; Karpukhin 2020; Izacard & Grave 2021; Asai 2024; Sarthi 2024; Gao 2024; Yang 2018; Trivedi 2022; Edge 2024; Sun 2024; Wang 2025; Parekh 2025; Guo 2023; Pearl 2009; Kleene 1952; Fitting 1985; Sato 2020; Sakama 2021; Chen 2024 (BGE-M3); Chalkidis 2020/2022, Zheng 2021, Guha 2023; Văn phòng Quốc hội 2025 (VBHN 135). If it does not fit at `\scriptsize`, split into A7a/A7b (appendix frames are not counted).
- Sources: PVR.

## 6. Running example data (verbatim from the sources)

### 6.1 Sample `BLHS_MH_0081` (benchmark record, B)
- question_type `reverse_reasoning`; hop_count 2; expected_label SUPPORTED; `evaluation.gold_decision` SUPPORTED. (Benchmark field `difficulty` = "hard" but `evaluation.difficulty.overall_level` = "medium"; S-bys uses "medium". Do not show difficulty on slides.)
- Question: «Kết quả “Có tình tiết giảm nhẹ trách nhiệm hình sự” trong chain này bắt nguồn từ sự kiện nào trước khi đi qua bước “Vượt quá giới hạn phòng vệ chính đáng”?»
- Gold answer: «Sự kiện khởi đầu là “Chống trả rõ ràng quá mức cần thiết”. Chuỗi đầy đủ: Chống trả rõ ràng quá mức cần thiết → Vượt quá giới hạn phòng vệ chính đáng → Có tình tiết giảm nhẹ trách nhiệm hình sự.»
- Gold path: hop 1 `chong_tra_ro_rang_qua_muc_can_thiet` → `vuot_qua_gioi_han_phong_ve_chinh_dang` (rule 47, Điều 22); hop 2 → `co_tinh_tiet_giam_nhe_trach_nhiem_hinh_su` (rule 106, Điều 51). Gold citations: Điều 22, Điều 51.

### 6.2 Statutory text (RAW)
- Điều 22 (Phòng vệ chính đáng), khoản 2: «Vượt quá giới hạn phòng vệ chính đáng là hành vi chống trả rõ ràng quá mức cần thiết, không phù hợp với tính chất và mức độ nguy hiểm cho xã hội của hành vi xâm hại. Người có hành vi vượt quá giới hạn phòng vệ chính đáng phải chịu trách nhiệm hình sự theo quy định của Bộ luật này.»
- Điều 51 (Các tình tiết giảm nhẹ trách nhiệm hình sự), khoản 1, điểm c: «c) Phạm tội trong trường hợp vượt quá giới hạn phòng vệ chính đáng;»
- Điều 23 (Tình thế cấp thiết), khoản 1 ends: «Hành vi gây thiệt hại trong tình thế cấp thiết không phải là tội phạm.»

### 6.3 Rules of the graph slice (R; all modalities `UNSPECIFIED`)
| Rule | Điều | Chủ thể | rule_text | condition event → effect event |
|---|---|---|---|---|
| 45 | 22 | Người phòng vệ | Nếu chống trả cần thiết để bảo vệ quyền, lợi ích chính đáng trước hành vi xâm phạm thì người phòng vệ được xác định là phòng vệ chính đáng. | Chống trả cần thiết để bảo vệ lợi ích chính đáng → Phòng vệ chính đáng |
| 46 | 22 | Người phòng vệ | Nếu thực hiện hành vi phòng vệ chính đáng thì người phòng vệ không bị coi là phạm tội. | Phòng vệ chính đáng → Không bị coi là tội phạm |
| 47 | 22 | Người phòng vệ | Nếu chống trả rõ ràng quá mức cần thiết và không phù hợp với tính chất, mức độ nguy hiểm của hành vi xâm hại thì người phòng vệ được xác định là vượt quá giới hạn phòng vệ chính đáng. | Chống trả rõ ràng quá mức cần thiết → Vượt quá giới hạn phòng vệ chính đáng |
| 48 | 22 | Người phòng vệ | Nếu vượt quá giới hạn phòng vệ chính đáng thì người phòng vệ phải chịu trách nhiệm hình sự. | Vượt quá giới hạn phòng vệ chính đáng → Chịu trách nhiệm hình sự |
| 49 | 23 | Người gây thiệt hại | Nếu không còn cách nào khác để tránh thiệt hại lớn hơn và gây ra thiệt hại nhỏ hơn thiệt hại cần ngăn ngừa thì người gây thiệt hại được xác định là hành động trong tình thế cấp thiết. | Gây thiệt hại nhỏ hơn để tránh thiệt hại lớn hơn → Tình thế cấp thiết |
| 50 | 23 | Người gây thiệt hại | Nếu gây thiệt hại trong tình thế cấp thiết thì người gây thiệt hại không bị coi là phạm tội. | Tình thế cấp thiết → Không bị coi là tội phạm |
| 106 | 51 | Tòa án | Nếu người phạm tội phạm tội trong trường hợp vượt quá giới hạn phòng vệ chính đáng thì tòa án xác định đây là tình tiết giảm nhẹ trách nhiệm hình sự. | Vượt quá giới hạn phòng vệ chính đáng → Có tình tiết giảm nhẹ trách nhiệm hình sự |

In-degree (computed from R): Không bị coi là tội phạm ← 5 rules (Điều 8, 22, 23, 24, 25); Chịu trách nhiệm hình sự ← 18 rules; Có tình tiết giảm nhẹ trách nhiệm hình sự ← 28 rules (Điều 51, 84); Vượt quá giới hạn phòng vệ chính đáng: 1 in, 2 out. Note the rule-47 condition contains "… và không phù hợp …" but is encoded as one positive antecedent event (single-antecedent data, F2.10).

### 6.4 Pipeline trace in Full LegalSCM (P-full, `id == "BLHS_MH_0081"`)
- **Step 2 (memory):** `retrieval.retrieved_events`: memory_id 2956 `chong_tra_ro_rang_qua_muc_can_thiet` role CONDITION; 2957 `vuot_qua_gioi_han_phong_ve_chinh_dang` role BRIDGE; 3047 `co_tinh_tiet_giam_nhe_trach_nhiem_hinh_su` role EFFECT (plus 5 other seeds with score 0.4801: Lần đầu phạm tội ít nghiêm trọng; Vượt quá yêu cầu của tình thế cấp thiết; Vượt quá mức cần thiết khi bắt giữ người phạm tội; Ngăn chặn hoặc làm giảm tác hại của tội phạm; Tự nguyện khắc phục hậu quả). `retrieval.direct_rule_hits[0]`: RULE::106 (memory_id 105, Điều 51), score 0.6535.
- **Step 3 (retrieval):** `statistics`: semantic_hits 100, retrieved_event_seeds 8, direct_rule_hits 8, effective_direction backward (requested "both"), causal_paths 33, candidate_rules 36, final_evidence 12. `causal_paths[0]`: seed `co_tinh_tiet_giam_nhe_trach_nhiem_hinh_su` (seed_similarity 0.60206), backward, events chống trả → vượt quá → có tình tiết, rule_ids [47, 106], graph_score 0.70511 (= 0.54·0.60206 + 0.28·0.5 + 0.06 + 0.18). `causal_paths[1]`: 1 hop from seed `vuot_qua…` (0.6035), graph_score 0.48094. 32 of the 33 paths are 1-hop. Rule evidence: rule 106 final 0.7190 (rank 1), rule 47 final 0.6418 (rank 2); `retrieved_rule_ids` = 47, 106, 112, 107, 108, 104, 110, 114, 113, 109, 105, 111; retrieved articles 22, 51.
- **Step 4 (verification):** `query_analysis.claim_type` STANDARD_CAUSAL; route `legal_scm_factual_validation`; no do() requested. `path_verifications[0].counterfactual_summary`: factual_context {chong_tra_ro_rang_qua_muc_can_thiet: true}; factual_states: all three events "true"; factual_iterations 3; path_rule_coverage all hops covered (activated 47, 106; missing none); factual_conflicting_event_ids []. Path status SUPPORTED, consistency/decision score 0.57681. `statistics`: 33 paths all SUPPORTED; verified_evidence 10, uncertain 0, removed 0.
- **Commit gate:** `runner_scm_guard`: status VERIFIER_COMMITTED ("Verifier decision passed the selective commit gate."), primary_path_ids [0], all_hops_covered true, structural_conflict_count 0, factual_outcome true, scm_commit_threshold 0.55, scm_grounding_threshold 0.8, original/effective decision SUPPORTED with score 0.57681.
- **Step 5 (answer):** final_answer «Sự kiện trước là 'Chống trả rõ ràng quá mức cần thiết'.»; citations [Điều 22, Điều 51]; `generation.authoritative_decision_applied` true, `decision_overridden` false; runtime_seconds 2.829465; `pipeline_metadata.selected_evidence_count` 8, `prompt_includes_verification` true.

### 6.5 The same question in the four other systems
| System (file) | Decision | final_answer (verbatim) | citations | Answer F1 / Citation F1 (S-bys) | runtime (s) |
|---|---|---|---|---|---|
| Closed-book (abstain) (P-cba) | UNCERTAIN | Không đủ thông tin để xác định sự kiện trước bước 'Vượt quá giới hạn phòng vệ chính đáng'. | [] | 0.3667 / 0 | 0.842 |
| Closed-book (forced) (P-cbf) | REJECT_DIRECT_CLAIM | Không đủ thông tin để xác định sự kiện trước bước 'Vượt quá giới hạn phòng vệ chính đáng'. | [] | 0.3667 / 0 | 0.877 |
| Vanilla RAG (P-van) | SUPPORTED | Sự kiện trước đó là 'Phạm tội trong trường hợp vượt quá giới hạn phòng vệ chính đáng'. | [Điều 51, Điều 126] (retrieved 51, 52, 84, 29, 126) | 0.3729 / 0.5 | 1.967 |
| Causal Path RAG (P-cpr) | REJECT_DIRECT_CLAIM | Không đủ căn cứ xác định sự kiện trước 'Vượt quá giới hạn phòng vệ chính đáng'. | [Điều 22, Điều 51] | 0.3793 / 1.0 | 1.596 |
| Full LegalSCM (P-full) | SUPPORTED | Sự kiện trước là 'Chống trả rõ ràng quá mức cần thiết'. | [Điều 22, Điều 51] | 0.4151 / 1.0 | 2.829 |

Causal Path RAG: `step4_bypassed` true, `prompt_includes_verification` false, `selected_evidence_count` 8, same `retrieved_rule_ids` as Full. Both Causal and Full: `top1_exact_path_match` 1.0, rule/article Recall@5 1.0 (S-bys).

### 6.6 Auxiliary examples (same legal neighbourhood)
- **`BLHS_MH_0221` (DIRECT route, F2.11).** Question: «Nhận định “Chống trả rõ ràng quá mức cần thiết” trực tiếp gây ra “Chịu trách nhiệm hình sự” mà không cần sự kiện trung gian là đúng hay sai?» Gold NO_DIRECT_EDGE → REJECT_DIRECT_CLAIM. P-full: claim_type DIRECT_CAUSAL_CLAIM, route `grounded_direct_edge_topology`, endpoint_alignment_confidence 1.0, decision REJECT_DIRECT_CLAIM (score 0.65), committed; explanation "Grounded cause/effect không có CAUSES edge trực tiếp; graph chỉ hỗ trợ quan hệ gián tiếp hoặc không hỗ trợ."; path rules 47, 48 (Điều 22); final_answer «Sai», citations [Điều 22, Điều 23, Điều 24] (not shown on slides). Causal Path RAG also correct.
- **`BLHS_MH_0223` (INTERVENTION route, F2.12).** Question: «Có thể bỏ qua bước “Tình thế cấp thiết” và kết luận rằng “Gây thiệt hại nhỏ hơn để tránh thiệt hại lớn hơn” trực tiếp dẫn đến “Không bị coi là tội phạm” hay không?» Gold NO_DIRECT_EDGE → REJECT_DIRECT_CLAIM. P-full: claim_type MEDIATOR_OMITTABLE_CLAIM, route `legal_scm_factual_do_intervention`, mediator_alignment_confidence 1.0; factual_states gay_thiet_hai… / tinh_the_cap_thiet / khong_bi_coi_la_toi_pham all true (3 iterations); intervention {tinh_the_cap_thiet: false}: factual_outcome true → counterfactual_outcome false, outcome_changed true, status NECESSARY, disabled_rule_ids [49, 50]; decision REJECT_DIRECT_CLAIM (score 0.72344), committed; final_answer «Không thể bỏ qua bước 'Tình thế cấp thiết' để kết luận trực tiếp rằng 'Gây thiệt hại nhỏ hơn để tránh thiệt hại lớn hơn' dẫn đến 'Không bị coi là tội phạm'.», citations [Điều 23, Điều 29]. Causal Path RAG also correct (as in all 7 intervention cases).

## 7. Cross-check of the brief's known numbers

| Quantity | Brief | Value in source | Source | Status |
|---|---|---|---|---|
| Articles | 406 unique, IDs 1–426 | 406 records; ids 1–426; 20 ids absent (90–107, 292, 317) | RAW | match |
| Graph | 5,134 nodes / 14,420 multi-edges | 5134 / 14420 (5 edge types × 2884) | G | match |
| Node types | 395 / 2,884 / 1,814 / 41 | same | G | match |
| Unique causal event pairs | 2,122 | `causal_edges` 2122; also 2122 distinct (condition, effect) pairs in R | G, R | match (B metadata names the 2,884 rule-level edges `causal_edges` — naming only, C8) |
| Bridge events / two-hop chains | 14 / 112 | 14 / 112 | G | match |
| DAG / weak components | DAG / 248 | true / 248 | G | match |
| Memory | 4,698 = 2,884 RULE + 1,814 EVENT | 4698 rows (RULE 2884, EVENT 1814); embeddings (4698, 1024) | MEM | match |
| Encoder / index | BGE-M3 + FAISS IndexFlatIP | same | C2, LOG-full | match |
| Benchmark | 250; 230/20; no gold UNCERTAIN; 236 linear | same | B | match |
| Negative routes | 13 direct + 7 intervention | 13 / 7 | S-sum, S-bys | match |
| Setup | Qwen3-8B, BGE-M3, T 0.1, seed 42, 1200 tok, 32768 ctx, 18,000 chars | same (+ thinking disabled, JSON format) | LOG-full, PV5, S-rep | match |
| Main results (35 values) | as in brief | identical | S-sum, PV6 | match |
| Full vs Causal | 242 vs 222; 23/3/219/5; +8.0 [4.4, 12.0]; p 8.8e-5; Holm 0.0011 | identical (Holm 0.001144) | S-sig, S-bys, PV6 | match |
| Text metrics | Holm p = 1.0 | 1.0, 1.0, 1.0 | S-sig | match |
| Audit | 230/13/7; committed 230; fallback 20; 8 errors after fallback; 7/7 | identical | S-sum, S-bys, PV6 | match |
| Path | Top-1 38.56 vs Oracle 75.42 | 0.385593 / 0.754237 | S-rep, S-sum, PV6 | match |
| Runtime | 3.238 vs 1.947 (~66%) | identical; ratio 1.663 | S-sum, PV6 | match |
| Closed-book | 220/250 UNCERTAIN; 247/250 REJECT | identical | S-rep, PV6 | match |

No numeric mismatch was found among the brief's numbers; the conflicts are about naming, versions and semantics (§8).

## 8. Source conflicts and mismatches (and how the slides handle them)

- **C1 — Title.** LV1 cover: "Cải thiện Multi-hop Question Answering bằng cách kết hợp Causal Graph và Counterfactual Reasoning trong RAG"; PVM: "LegalSCM-RAG: Truy xuất đường dẫn nhân quả và xác minh cấu trúc chọn lọc cho hỏi đáp pháp luật đa bước tiếng Việt". Slides: LV1 title + shortened subtitle (D1).
- **C2 — Advisor order.** LV1 tr. 2: "GVHD: PGS.TS. Nguyễn Thị Kim Anh & TS. Nguyễn Quốc Tuấn"; user update: "TS. Nguyễn Quốc Tuấn và PGS.TS. Nguyễn Thị Kim Anh" (same order as the paper's author list). Slides follow the user update.
- **C3 — Major.** LV1: "Chuyên ngành: Khoa học máy tính"; user update: "Thạc sĩ Khoa học Máy tính". Slides follow the user update.
- **C4 — Nature and age of LV1.** `luan_van_ver1.pdf` is a 35-page proposal-style deck (motivation, objectives, related work, proposed method; no data chapter, no results). Its method pages (tr. 29–30) describe counterfactual verification as node deletion ("Loại bỏ mediator → A→C?"), which the paper now treats only as a topological diagnostic, distinct from $do(X_m=0)$. Slides follow the paper. It also shows the CausalRAG figure "Nguồn: Wang et al." (tr. 9), not reused (D4).
- **C5 — Baseline "CausalRAG" vs "Causal Path RAG".** LV1 tr. 12/15/34 compare against "CausalRAG"; the experiments use Causal Path RAG = Step 3 without Step 4 (README, PV5). RQ4 and all slides use "Causal Path RAG"; no claim of comparison with Wang và cs. (2025).
- **C6 — Path-ranking formula.** PV4 gives the generic $S(P_i,q)=\alpha S_{\text{sem}}+\beta S_{\text{graph}}+\gamma S_{\text{seed}}+\delta S_{\text{path}}$ without weights; the code (C3) and DECK use a path score (0,54/0,28/bridge/hop) and a separate rule score (0,32/0,36/0,10/0,07/0,15). Slides: generic $\arg\max$ from the paper + implemented $S_{\text{path}}$ (F2.8); $S_{\text{rule}}$ in A1.
- **C7 — UNKNOWN vs closed world.** PV3: "không có hỗ trợ hoặc có hỗ trợ xung đột tạo trạng thái unknown"; SCM: endogenous events with no active rule default to FALSE, root events UNKNOWN until supplied. The do() result of `BLHS_MH_0223` (TRUE → FALSE) relies on the implemented rule. Slides state the implemented convention (F2.9, A2); flag for the paper (§9).
- **C8 — "causal_edges" naming.** B metadata `source_graph_statistics.causal_edges` = 2,884 (rule-level CAUSES edges); G `causal_edges` = 2,122 (unique event pairs). Slides: "14.420 cạnh = 5 loại × 2.884" and "2.122 cặp sự kiện CAUSES duy nhất".
- **C9 — 406 vs 395 articles.** 11 raw articles produce no rule (ids 1, 4, 30, 31, 42, 46, 122, 352, 367, 392, 426); not a contradiction, explained on F2.2.
- **C10 — Rules are curated, not raw a1 output.** All 2,884 final rules carry `quality_status` CURATED (230) or CURATED_FROM_STATUTORY_CLAUSE (2,654); AN H-3 notes the curation step between a2 and the final file is not in the repo. F2.3 labels rule 47 "bản cuối, đã rà soát" and does not claim a1 produced it unchanged.
- **C11 — Legacy evaluation folders.** `evaluation_metrics_detailed/`, `evaluation_metrics_*_v41/` etc. contain older Step-7 numbers (e.g. Oracle exact path 0.712; AN M-9 reports Top-1 35,20% / Oracle 71,20%). Slides use only `system_level_comparison_v2`.
- **C12 — Notation.** PV abstract uses $do(X_M=0)$, the body $do(X_m=0)$ → slides use $do(X_m=0)$. PV tables use decimal points while PV text uses commas → slides use commas (D8). PV names closed-book baselines "Closed-book (abstain/forced)", DECK and artifacts "LLM-only/llm_only" → slides use the PV table names (D9).
- **C13 — Difficulty field.** `BLHS_MH_0081` has `difficulty` "hard" but `evaluation.difficulty.overall_level` "medium" (S-bys uses the latter). Difficulty is not shown.

## 9. Limitations decision and items for the final report to the user

Limitations on slides: F5.2 (D11). Items **not** put on slides, to be reported to the user:
1. **Modality merge (AN H-1)** — "có thể được" (POSSIBLE_GRANTED / POSSIBLE_IMPOSED) merged with "được" in event normalization; 89 rules have a `POSSIBLE_*` endpoint modality and several bridge events mix both (e.g. "Được xóa án tích || Có thể được xóa án tích"). Not discussed in paper/vi, paper/en or LV1, so left off the slides by decision (a). The running example and both auxiliary examples avoid these events.
2. **Paper vs code on UNKNOWN (C7)** — the paper should state the closed-world default for endogenous events, or the code should change; the slides follow the code.
3. **Decision vs exact path (observation from S-bys, not in the paper)** — among the 23 Full-only correct samples only 3 have `top1_exact_path_match = 1` (0081, 0213, 0219). Exact path match is strict at rule level, so this needs analysis before any claim; not on slides.
4. **Repository state** — `causal_core/` appears deleted in git and an untracked folder named `|` holds `legal_scm.py` and `schema.py`; probably an accidental rename. Not touched.
5. **Baseline naming (C5) and proposal-era method description (C4)** — the defense should not present Causal Path RAG as Wang và cs.'s CausalRAG.
6. **Data lineage (C10)** — the curation step from a2 output to the final rule file is not in the repo.

## 10. Build and verification procedure (used by every step in §11)

All commands run from `/Users/nhungnguyen/Documents/cao_hoc/do_an/causal_rag/slides/luan_van` (use this absolute path; relative paths from another cwd land elsewhere). Tools verified on this machine: `latexmk`/`pdflatex` (TeX Live 2025, includes `vietnam.sty`, `beamer.cls`, `pgfplots.sty`), Ghostscript `gs`, and `/usr/bin/python3` (3.9) with PIL 9.2.0. `pdftotext`/`pdfinfo` are not installed.

1. **Build:** `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` → exit code 0 (latexmk runs the second pass needed by `remember picture`).
2. **Log:** `grep -nE "^! |Overfull \\\\vbox|Undefined control sequence|Missing character" main.log` → no output; `grep -nE "Overfull \\\\hbox \(([3-9]|[1-9][0-9]+)\." main.log` → no output (hbox overfull ≤ 3 pt tolerated). Font-shape substitution notices (`T5/lmss/m/it` → `sl`) also appear in the template build and are acceptable.
3. **Render:** `mkdir -p build/render && rm -f build/render/*.png && gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r203.2 -o build/render/p%02d.png main.pdf` (203.2 dpi gives 1280×720 px, the native size of the background PNGs, so the checker compares pixel-to-pixel without resampling; at other resolutions the footer produces false positives — tested); page count: `gs -q -dNODISPLAY -dNOSAFER -c "(main.pdf) (r) file runpdfbegin pdfpagecount = quit"`.
4. **Layout check:** save the script below as `build/check_layout.py` and run `/usr/bin/python3 build/check_layout.py` → last line `pages with problems: 0`. (The script was tested on a scratch build of the template deck: it runs in ~2 s for 31 pages and flags exactly the template pages whose content touches the footer logo/line; template blocks reach x = 15.22 cm, which is why the right limit is 15.35 cm.)
5. **Visual check:** open the rendered PNGs of the pages changed in the step with the file viewer (they are inside the workspace) and confirm: Vietnamese diacritics render, no overlap with the header bar, footer logo/line, roadmap or page number; roadmap highlights the right step; colours follow D10.
6. **Frame count:** `grep -c '\\begin{frame}' 1_gioi_thieu.tex 2_du_lieu_va_phuong_phap.tex 3_thuc_nghiem_va_danh_gia.tex 4_ket_qua_va_phan_tich.tex 5_ket_luan.tex` → 7, 14, 3, 8, 3 at the end (36 total only if F2.3 was split).

```python
# build/check_layout.py — flags content outside the safe area of the HUST backgrounds
from PIL import Image, ImageChops
import glob, os
BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "..", "images")
KINDS = ["intro", "title", "content-full", "content-hust-full", "section", "end"]
BGS = {k: Image.open(os.path.join(IMG, "hust-blue-169-%s.png" % k)).convert("RGB") for k in KINDS}
problems = 0
for fn in sorted(glob.glob(os.path.join(BASE, "render", "p*.png"))):
    a = Image.open(fn).convert("RGB"); W, H = a.size
    best = None
    for k, bg in BGS.items():
        d = ImageChops.difference(a, bg.resize(a.size)).convert("L").point(lambda v: 255 if v > 40 else 0)
        frac = d.histogram()[255] / float(W * H)
        if best is None or frac < best[0]:
            best = (frac, k, d)
    frac, kind, d = best
    px = d.load()
    def count(x0, y0, x1, y1, allow=()):
        n = 0
        for y in range(int(y0 / 9.0 * H), min(H, int(y1 / 9.0 * H))):
            for x in range(int(x0 / 16.0 * W), min(W, int(x1 / 16.0 * W))):
                if px[x, y]:
                    xc, yc = x / float(W) * 16, y / float(H) * 9
                    if any(ax0 <= xc <= ax1 and ay0 <= yc <= ay1 for (ax0, ay0, ax1, ay1) in allow):
                        continue
                    n += 1
        return n
    msgs = []
    if kind in ("content-full", "content-hust-full"):
        nb = count(0, 8.05, 16, 9.0, allow=[(7.9, 8.42, 14.9, 8.95), (14.95, 8.40, 15.75, 8.98)])
        nr = count(15.35, 1.1, 16, 8.0)
        if nb > 20: msgs.append("content below 8.05 cm (%d px)" % nb)
        if nr > 20: msgs.append("content right of 15.35 cm (%d px)" % nr)
    if kind == "title":
        bb = d.getbbox()
        if bb:
            x1, y0, y1 = bb[2] / float(W) * 16, bb[1] / float(H) * 9, bb[3] / float(H) * 9
            if x1 > 13.45 or y0 < 2.05 or y1 > 7.65:
                msgs.append("title block outside safe box: x1=%.2f y0=%.2f y1=%.2f" % (x1, y0, y1))
    if msgs:
        problems += 1
        print(os.path.basename(fn), kind, "; ".join(msgs))
print("pages with problems:", problems)
```

## 11. Implementation plan (ordered; each step leaves a buildable deck)

- [ ] 1. Scaffold `slides/luan_van/`: copy `beamerthemeHUST.sty` byte-for-byte; copy the six PNGs of §2 into `images/`; write `main.tex` exactly per §3 (preamble, colours/macros §3.2, roadmap/frametitle §3.3, title page §3.4, title data §3.5, skeleton §3.6); create the six section files containing only their `\section{…}` line (and `\setroadmap{-1}` before `\section` in files 3 and 6), so the deck builds with structural pages only.
      Files: `slides/luan_van/{main.tex, beamerthemeHUST.sty, images/*.png, 1_gioi_thieu.tex, 2_du_lieu_va_phuong_phap.tex, 3_thuc_nghiem_va_danh_gia.tex, 4_ket_qua_va_phan_tich.tex, 5_ket_luan.tex, 6_phu_luc.tex}`
      Verify: §10 steps 1–5 pass; `cmp beamerthemeHUST.sty ../slide_hoc_sau_va_ung_dung/beamerthemeHUST.sty` prints nothing; in the render, page 2 shows the title in at most 3 uppercase lines, the subtitle, and the 4 info lines exactly as D1 with no "Nhóm 7:" (check_layout reports no title problem); page 3 (TOC) lists exactly the 5 main sections and no "Phụ lục"; the last pages are the thank-you page and the "Phụ lục" section page.

- [ ] 2. Section 1: write F1.1–F1.7 in `1_gioi_thieu.tex` (§5, Section 1), including the RQ cards and the Venn and chain TikZ figures.
      Files: `slides/luan_van/1_gioi_thieu.tex`
      Verify: §10 steps 1–5 pass; 7 frames in the file; every number on these frames appears in the F1.x "Sources" lines.

- [ ] 3. Section 2, data and graph: F2.1–F2.5 (pipeline overview, data source, preprocessing, graph schema and stats, real graph slice) with `\setroadmap{-1}`, `{0}`, `{0}`, `{1}`, `{1}` placed before the respective frames. Depends on step 1 (step colours, `stepnode`/`evnode`/`bridgenode` styles).
      Files: `slides/luan_van/2_du_lieu_va_phuong_phap.tex`
      Verify: §10 steps 1–5 pass; roadmap visible and highlighting Dữ liệu on F2.2–F2.3 and Đồ thị on F2.4–F2.5, absent on F2.1; if F2.3 overflows, split it into F2.3a/F2.3b (only allowed split).

- [ ] 4. Section 2, memory, retrieval and semantics: F2.6–F2.10 (`\setroadmap{2}`, `{3}`, `{3}`, `{4}`, `{4}`).
      Files: `slides/luan_van/2_du_lieu_va_phuong_phap.tex`
      Verify: §10 steps 1–5 pass; every formula has its "Diễn giải" line; the worked numbers 0,705 / 0,481 / 0,719 / 0,642 / 3 vòng match §6.4.

- [ ] 5. Section 2, verification and answer: F2.11–F2.14 (`\setroadmap{4}` ×3, then `{5}`), including the two-panel do()/node-deletion figure, the commit-gate diagram and the five-step trace strip.
      Files: `slides/luan_van/2_du_lieu_va_phuong_phap.tex`
      Verify: §10 steps 1–5 pass; 14 frames in the file (15 if F2.3 was split); the trace strip values equal §6.4.

- [ ] 6. Section 3: F3.1–F3.3 (`\setroadmap{-1}` before `\section`), including the pgfplots question-type chart and the five-configuration ladder.
      Files: `slides/luan_van/3_thuc_nghiem_va_danh_gia.tex`
      Verify: §10 steps 1–5 pass; no roadmap on these pages; chart tick labels use decimal commas.

- [ ] 7. Section 4, results: F4.1–F4.4 (main table, imbalance chart, significance matrix, confusion matrices).
      Files: `slides/luan_van/4_ket_qua_va_phan_tich.tex`
      Verify: §10 steps 1–5 pass; every table value equals §5 F4.1–F4.4 (which equal S-sum/S-sig/S-rep).

- [ ] 8. Section 4, analysis: F4.5–F4.8 (audit flow, running-example before/after with the deck's only `\pause`, retrieval/path charts, runtime scatter).
      Files: `slides/luan_van/4_ket_qua_va_phan_tich.tex`
      Verify: §10 steps 1–5 pass; `grep -c '\\pause' *.tex` totals 1; F4.6 renders as two PDF pages (before/after reveal); quotes in F4.6 match §6.5 verbatim (with "PVCĐ" abbreviation only).

- [ ] 9. Section 5: F5.1–F5.3 (RQ answers with the same card colours as F1.6, threats to validity, conclusion and future work).
      Files: `slides/luan_van/5_ket_luan.tex`
      Verify: §10 steps 1–6 pass; content frame counts 7/14/3/8/3; each RQ answer cites the numbers listed in F5.1 "Sources"; no modality item anywhere (D11).

- [ ] 10. Appendix: A1–A7 in `6_phu_luc.tex` after the thank-you page (`\appendix` and `\showPageNumber` already in `main.tex`).
      Files: `slides/luan_van/6_phu_luc.tex`
      Verify: §10 steps 1–5 pass; appendix pages come after the thank-you page and show page numbers; TOC still lists only 5 sections.

- [ ] 11. Final QA on a clean build: `latexmk -C` then §10 steps 1–6. Then audit: (a) every number on every slide appears in §5/§6 of this outline with its source; (b) no image files used except the six backgrounds (`grep -n includegraphics *.tex` → only inside `main.tex` title override); (c) no text below `\scriptsize` except the roadmap (`grep -n '\\tiny' *.tex` → only in `main.tex` `\rmnode`); (d) no emoji, no "Nhóm 7", no "Wang và cs." comparison claim, no "có thể được"/modality content (`grep -n -i "nhóm 7\|có thể được\|modality" *.tex` → nothing); (e) `slides/main.tex` and `slides/slide_hoc_sau_va_ung_dung/` unchanged (`git status --short slides/` shows only the new `slides/luan_van/` folder besides the pre-existing untracked template folder); (f) look at every rendered page once. Leave `main.pdf` in `slides/luan_van/`; do not commit.
      Files: none new (fixes only).
      Verify: all checks above pass; expected page count ≈ 53 (1 intro + 1 title + 1 TOC + 5 section pages + 35 frames + 1 overlay page + 1 thank-you + 1 appendix section page + 7 appendix frames), ±1 for allowed splits.

