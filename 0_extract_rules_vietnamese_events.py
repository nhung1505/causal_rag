from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests


DEFAULT_INPUT = "data/1_blhs_articles_parsed.json"
DEFAULT_OUTPUT = "data/4_blhs_merged.json"
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"


SYSTEM_INSTRUCTION = """
Bạn là chuyên gia trích xuất quy tắc pháp luật từ Bộ luật Hình sự Việt Nam.

Nhiệm vụ:
- Đọc toàn bộ nội dung của một điều luật.
- Trích xuất các quy tắc pháp lý có cấu trúc điều kiện -> hệ quả.
- Một điều luật có thể tạo ra nhiều quy tắc.
- Chỉ trích xuất nội dung được thể hiện trực tiếp trong điều luật.
- Không tự bổ sung kiến thức ngoài văn bản.
- Không gộp các quy tắc có điều kiện hoặc hệ quả khác nhau.

Mỗi quy tắc phải có:
1. legal_subject:
   Chủ thể pháp lý chịu sự điều chỉnh của quy tắc.

2. condition:
   Điều kiện, tình huống hoặc sự kiện làm phát sinh quy tắc.
   Viết bằng tiếng Việt có dấu, ngắn gọn nhưng đầy đủ ý nghĩa pháp lý.

3. effect:
   Hệ quả pháp lý khi điều kiện xảy ra.
   Viết bằng tiếng Việt có dấu, ngắn gọn nhưng đầy đủ ý nghĩa pháp lý.

4. condition_event:
   Nhãn sự kiện chuẩn hóa của condition.
   Bắt buộc:
   - tiếng Việt không dấu;
   - chỉ dùng chữ thường, chữ số và dấu gạch dưới;
   - không dùng dấu cách;
   - diễn đạt rõ nghĩa;
   - không thêm tiền tố event_;
   - không viết bằng tiếng Anh.

5. effect_event:
   Nhãn sự kiện chuẩn hóa của effect.
   Bắt buộc:
   - tiếng Việt không dấu;
   - chỉ dùng chữ thường, chữ số và dấu gạch dưới;
   - không dùng dấu cách;
   - diễn đạt rõ nghĩa;
   - không thêm tiền tố event_;
   - không viết bằng tiếng Anh.

6. rule_text:
   Viết lại quy tắc theo mẫu tự nhiên:
   "Nếu <condition> thì <effect>."

Nguyên tắc quan trọng:
- Nếu điều luật chỉ mô tả khái niệm, mục đích, danh sách hoặc nguyên tắc chung
  nhưng không hình thành quan hệ điều kiện -> hệ quả rõ ràng, trả về [].
- Không được tạo condition hoặc effect rỗng.
- Không được trả về markdown.
- Không được giải thích.
- Chỉ trả về một JSON array hợp lệ.

Ví dụ đầu ra:
[
  {
    "legal_subject": "Người phạm tội",
    "condition": "phạm tội ít nghiêm trọng và có nhiều tình tiết giảm nhẹ nhưng chưa đến mức miễn hình phạt",
    "effect": "được áp dụng hình phạt cảnh cáo",
    "condition_event": "pham_toi_it_nghiem_trong_co_nhieu_tinh_tiet_giam_nhe",
    "effect_event": "duoc_ap_dung_hinh_phat_canh_cao",
    "rule_text": "Nếu người phạm tội ít nghiêm trọng và có nhiều tình tiết giảm nhẹ nhưng chưa đến mức miễn hình phạt thì được áp dụng hình phạt cảnh cáo."
  }
]
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trích xuất rule từ 1_blhs_articles_parsed.json và tạo "
            "4_blhs_merged.json."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_GENERATE_URL,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1400,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--start-article",
        type=int,
        default=None,
        help="Chỉ xử lý từ article_id này trở đi.",
    )
    parser.add_argument(
        "--end-article",
        type=int,
        default=None,
        help="Chỉ xử lý đến article_id này.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số điều luật để chạy thử.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Thời gian nghỉ giữa hai request.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua các article_id đã có trong output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa output cũ và chạy lại từ đầu.",
    )
    return parser.parse_args()


def load_json(path_value: str) -> Any:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path_value: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def remove_accents(text: str) -> str:
    text = str(text or "").replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(
        char
        for char in decomposed
        if unicodedata.category(char) != "Mn"
    )


def normalize_event_label(value: Any) -> str:
    text = remove_accents(str(value or "")).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    text = raw_text.strip()

    # Loại bỏ reasoning tag nếu model sinh ra.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Loại bỏ code fence.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Không tìm thấy JSON array trong phản hồi của model."
            )

        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, list):
        raise ValueError("Phản hồi của model không phải JSON array.")

    return [
        item
        for item in payload
        if isinstance(item, dict)
    ]


def build_prompt(article: dict[str, Any]) -> str:
    article_id = article.get("article_id", "")
    article_title = normalize_text(article.get("article_title"))
    content = str(article.get("content") or "").strip()

    return f"""
{SYSTEM_INSTRUCTION}

ĐIỀU LUẬT CẦN TRÍCH XUẤT

article_id: {article_id}
article_title: {article_title}

content:
{content}

Chỉ trả về JSON array.
""".strip()


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    num_predict: int,
    timeout: int,
) -> str:
    response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    result = payload.get("response")

    if not isinstance(result, str):
        raise ValueError(
            "Ollama không trả về trường response hợp lệ."
        )

    return result


def validate_and_prepare_rule(
    raw_rule: dict[str, Any],
    article: dict[str, Any],
) -> dict[str, Any] | None:
    legal_subject = normalize_text(raw_rule.get("legal_subject"))
    condition = normalize_text(raw_rule.get("condition"))
    effect = normalize_text(raw_rule.get("effect"))

    if not condition or not effect:
        return None

    condition_event = normalize_event_label(
        raw_rule.get("condition_event")
        or condition
    )
    effect_event = normalize_event_label(
        raw_rule.get("effect_event")
        or effect
    )

    if not condition_event or not effect_event:
        return None

    rule_text = normalize_text(raw_rule.get("rule_text"))
    if not rule_text:
        subject_prefix = (
            f"{legal_subject} "
            if legal_subject
            else ""
        )
        rule_text = (
            f"Nếu {subject_prefix}{condition} thì {effect}."
        )

    if not rule_text.endswith((".", "?", "!")):
        rule_text += "."

    return {
        "index": 0,
        "article_id": article.get("article_id"),
        "legal_subject": legal_subject,
        "condition": condition,
        "effect": effect,
        "condition_event": condition_event,
        "effect_event": effect_event,
        "rule_text": rule_text,
        "article_title": normalize_text(
            article.get("article_title")
        ),
        "content": str(article.get("content") or "").strip(),
    }


def rule_signature(rule: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rule.get("article_id"),
        normalize_event_label(rule.get("condition_event")),
        normalize_event_label(rule.get("effect_event")),
    )


def deduplicate_rules(
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for rule in rules:
        signature = rule_signature(rule)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(rule)

    return result


def reindex_rules(
    rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for index, rule in enumerate(rules, start=1):
        rule["index"] = index
    return rules


def main() -> None:
    args = parse_args()

    articles = load_json(args.input)
    if not isinstance(articles, list):
        raise ValueError(
            "File input phải là JSON array các điều luật."
        )

    selected_articles: list[dict[str, Any]] = []

    for article in articles:
        if not isinstance(article, dict):
            continue

        article_id = article.get("article_id")
        try:
            numeric_article_id = int(article_id)
        except (TypeError, ValueError):
            continue

        if (
            args.start_article is not None
            and numeric_article_id < args.start_article
        ):
            continue

        if (
            args.end_article is not None
            and numeric_article_id > args.end_article
        ):
            continue

        selected_articles.append(article)

    if args.limit is not None:
        selected_articles = selected_articles[: args.limit]

    output_path = Path(args.output)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    existing_rules: list[dict[str, Any]] = []
    processed_article_ids: set[int] = set()

    if output_path.exists():
        existing_payload = load_json(str(output_path))
        if isinstance(existing_payload, list):
            existing_rules = [
                item
                for item in existing_payload
                if isinstance(item, dict)
            ]

            if args.resume:
                for rule in existing_rules:
                    try:
                        processed_article_ids.add(
                            int(rule.get("article_id"))
                        )
                    except (TypeError, ValueError):
                        pass

    all_rules = existing_rules.copy()

    total = len(selected_articles)
    success_count = 0
    failed_count = 0
    empty_count = 0

    print(f"Input articles: {total}")
    print(f"Existing rules: {len(existing_rules)}")
    print(f"Model: {args.model}")
    print("-" * 72)

    for position, article in enumerate(
        selected_articles,
        start=1,
    ):
        article_id = article.get("article_id")
        title = normalize_text(article.get("article_title"))

        try:
            numeric_article_id = int(article_id)
        except (TypeError, ValueError):
            print(
                f"[{position}/{total}] Skip invalid article_id: "
                f"{article_id}"
            )
            continue

        if (
            args.resume
            and numeric_article_id in processed_article_ids
        ):
            print(
                f"[{position}/{total}] Skip Điều {article_id}: "
                "đã có trong output"
            )
            continue

        print(
            f"[{position}/{total}] Processing Điều "
            f"{article_id}: {title}"
        )

        started_at = time.time()

        try:
            prompt = build_prompt(article)
            raw_response = call_ollama(
                prompt=prompt,
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                num_predict=args.num_predict,
                timeout=args.timeout,
            )

            raw_rules = extract_json_array(raw_response)

            prepared_rules: list[dict[str, Any]] = []
            for raw_rule in raw_rules:
                rule = validate_and_prepare_rule(
                    raw_rule,
                    article,
                )
                if rule is not None:
                    prepared_rules.append(rule)

            prepared_rules = deduplicate_rules(
                prepared_rules
            )

            if not prepared_rules:
                empty_count += 1
                print(
                    f"  -> Không có rule hợp lệ "
                    f"({time.time() - started_at:.2f}s)"
                )
            else:
                all_rules.extend(prepared_rules)
                success_count += 1
                print(
                    f"  -> {len(prepared_rules)} rules "
                    f"({time.time() - started_at:.2f}s)"
                )

            all_rules = deduplicate_rules(all_rules)
            all_rules = reindex_rules(all_rules)
            save_json(all_rules, args.output)

        except Exception as error:
            failed_count += 1
            print(
                f"  -> ERROR: {type(error).__name__}: {error}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    all_rules = deduplicate_rules(all_rules)
    all_rules = reindex_rules(all_rules)
    save_json(all_rules, args.output)

    print("\n" + "=" * 72)
    print("COMPLETED")
    print("=" * 72)
    print(f"Articles with extracted rules: {success_count}")
    print(f"Articles without valid rules: {empty_count}")
    print(f"Failed articles: {failed_count}")
    print(f"Total output rules: {len(all_rules)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
