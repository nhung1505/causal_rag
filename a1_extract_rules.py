from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_INPUT = "data/1_blhs_articles_parsed.json"
DEFAULT_OUTPUT = "data/2_blhs_rules_raw.json"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


SYSTEM_INSTRUCTION = """
Bạn là chuyên gia trích xuất quy tắc pháp luật từ Bộ luật Hình sự Việt Nam.

Mục tiêu:
Chuyển nội dung của từng điều luật thành các quy tắc pháp lý có cấu trúc:
điều kiện hoặc tình huống pháp lý -> kết luận, hệ quả, nghĩa vụ, quyền,
sự cho phép, sự cấm đoán, phạm vi áp dụng, phân loại hoặc ngoại lệ pháp lý.

PHẢI TRÍCH XUẤT các dạng quy tắc sau:
- Điều kiện -> trách nhiệm pháp lý.
- Điều kiện -> hình phạt hoặc biện pháp xử lý.
- Điều kiện -> được phép hoặc có thể áp dụng.
- Điều kiện -> bắt buộc phải thực hiện.
- Điều kiện -> không được áp dụng.
- Điều kiện -> được miễn hoặc không phải chịu trách nhiệm.
- Đối tượng hoặc phạm vi -> quy định pháp luật được áp dụng.
- Đặc điểm hoặc mức hình phạt -> phân loại tội phạm.
- Trường hợp chung -> ngoại lệ.
- Hành vi hoặc trạng thái -> kết luận pháp lý.
- Chủ thể -> nghĩa vụ, trách nhiệm hoặc quyền của chủ thể.

Một điều luật có thể tạo ra nhiều quy tắc.
Mỗi khoản, điểm hoặc ngoại lệ khác nhau nên được tách thành một rule riêng
khi chúng có condition hoặc effect khác nhau.

KHÔNG được trả về danh sách rỗng chỉ vì điều luật:
- có tên là nguyên tắc;
- quy định hiệu lực;
- quy định trách nhiệm hoặc nghĩa vụ;
- định nghĩa hoặc phân loại;
- chứa từ "được", "phải", "không được", "có thể", "chỉ", "trừ trường hợp";
- không sử dụng trực tiếp cấu trúc "nếu ... thì ...".

Chỉ trả về {"rules": []} khi điều luật thực sự không thể chuyển thành bất kỳ
quan hệ đầu vào -> kết luận pháp lý nào, chẳng hạn chỉ nêu mục tiêu chung mà
không xác định chủ thể, điều kiện, phạm vi, nghĩa vụ, quyền hoặc hệ quả cụ thể.

Mỗi rule phải có đúng các trường:

1. legal_subject
Chủ thể pháp lý chịu sự điều chỉnh.
Ví dụ:
- Người phạm tội
- Pháp nhân thương mại
- Người nước ngoài
- Tòa án
- Cơ quan, tổ chức
- Mọi công dân

2. condition
Điều kiện, tình huống, hành vi, đặc điểm, phạm vi hoặc trường hợp áp dụng.
Có thể là:
- một điều kiện thực tế;
- một loại chủ thể;
- một mức hình phạt;
- một địa điểm hoặc thời điểm;
- một trường hợp luật định;
- một hành vi hoặc trạng thái pháp lý.

Yêu cầu:
- Viết bằng tiếng Việt có dấu.
- Ngắn gọn nhưng đủ nghĩa pháp lý.
- Không được rỗng.
- Không tự bổ sung nội dung ngoài điều luật.

3. effect
Kết luận hoặc hệ quả pháp lý tương ứng.
Có thể là:
- phải chịu trách nhiệm;
- được áp dụng;
- không được áp dụng;
- được miễn;
- được phân loại;
- có trách nhiệm thực hiện;
- thuộc phạm vi điều chỉnh;
- không phải là tội phạm.

Yêu cầu:
- Viết bằng tiếng Việt có dấu.
- Ngắn gọn nhưng đủ nghĩa pháp lý.
- Không được rỗng.
- Không tự bổ sung nội dung ngoài điều luật.

4. rule_text
Viết lại rule thành một câu tự nhiên theo mẫu:
"Nếu <condition> thì <effect>."

Có thể đưa legal_subject vào câu để câu rõ nghĩa.

Nguyên tắc trích xuất:
- Chỉ sử dụng nội dung trực tiếp trong điều luật.
- Không dùng kiến thức ngoài văn bản.
- Không diễn giải quá xa nội dung gốc.
- Không gộp các rule có condition hoặc effect khác nhau.
- Không tạo rule trùng lặp.
- Không tạo condition hoặc effect rỗng.
- Không trả về Markdown.
- Không giải thích.
- Chỉ trả về JSON object có khóa "rules".
- Giá trị "rules" phải là JSON array.

Ví dụ 1

Văn bản:
"Chỉ người nào phạm một tội đã được Bộ luật Hình sự quy định mới phải chịu
trách nhiệm hình sự."

Đầu ra:
{
  "rules": [
    {
      "legal_subject": "Người phạm tội",
      "condition": "phạm một tội đã được Bộ luật Hình sự quy định",
      "effect": "phải chịu trách nhiệm hình sự",
      "rule_text": "Nếu một người phạm một tội đã được Bộ luật Hình sự quy định thì người đó phải chịu trách nhiệm hình sự."
    }
  ]
}

Ví dụ 2

Văn bản:
"Bộ luật Hình sự được áp dụng đối với mọi hành vi phạm tội thực hiện trên
lãnh thổ Việt Nam."

Đầu ra:
{
  "rules": [
    {
      "legal_subject": "Hành vi phạm tội",
      "condition": "được thực hiện trên lãnh thổ Việt Nam",
      "effect": "Bộ luật Hình sự được áp dụng",
      "rule_text": "Nếu hành vi phạm tội được thực hiện trên lãnh thổ Việt Nam thì Bộ luật Hình sự được áp dụng."
    }
  ]
}

Ví dụ 3

Văn bản:
"Tội phạm có mức cao nhất của khung hình phạt đến 03 năm tù là tội phạm
ít nghiêm trọng."

Đầu ra:
{
  "rules": [
    {
      "legal_subject": "Tội phạm",
      "condition": "mức cao nhất của khung hình phạt là phạt tiền, cải tạo không giam giữ hoặc phạt tù đến 03 năm",
      "effect": "được phân loại là tội phạm ít nghiêm trọng",
      "rule_text": "Nếu mức cao nhất của khung hình phạt là phạt tiền, cải tạo không giam giữ hoặc phạt tù đến 03 năm thì tội phạm được phân loại là tội phạm ít nghiêm trọng."
    }
  ]
}
""".strip()


RULES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "legal_subject": {"type": "string"},
                    "condition": {"type": "string"},
                    "effect": {"type": "string"},
                    "rule_text": {"type": "string"},
                },
                "required": [
                    "legal_subject",
                    "condition",
                    "effect",
                    "rule_text",
                ],
            },
        }
    },
    "required": ["rules"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trích xuất rule thô từ 1_blhs_articles_parsed.json "
            "và tạo 2_blhs_rules_raw.json."
        )
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.2)

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
        "--resume",
        action="store_true",
        help="Bỏ qua các điều đã xử lý thành công hoặc xác định không có rule.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa output và trạng thái cũ trước khi chạy.",
    )

    return parser.parse_args()


def load_json(path_value: str | Path) -> Any:
    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_save_json(data: Any, path_value: str | Path) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_model_response(raw_text: str) -> str:
    text = str(raw_text or "").strip()

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text).strip()

    return text


def extract_balanced_json(
    text: str,
    opening: str,
    closing: str,
) -> str | None:
    start = text.find(opening)

    if start == -1:
        return None

    depth = 0
    inside_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]

        if inside_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                inside_string = False
            continue

        if character == '"':
            inside_string = True
            continue

        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1

            if depth == 0:
                return text[start : index + 1]

    return None


def extract_rules_payload(raw_text: str) -> list[dict[str, Any]]:
    text = clean_model_response(raw_text)

    if not text:
        raise ValueError("Phản hồi của model rỗng.")

    payload: Any = None
    errors: list[str] = []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        errors.append(f"whole_response={error}")

    if payload is None:
        object_text = extract_balanced_json(text, "{", "}")

        if object_text:
            try:
                payload = json.loads(object_text)
            except json.JSONDecodeError as error:
                errors.append(f"object={error}")

    if payload is None:
        array_text = extract_balanced_json(text, "[", "]")

        if array_text:
            try:
                payload = json.loads(array_text)
            except json.JSONDecodeError as error:
                errors.append(f"array={error}")

    if payload is None:
        preview = text[:800].replace("\n", " ")
        details = " | ".join(errors)

        raise ValueError(
            "Không tìm thấy JSON hợp lệ trong phản hồi của model. "
            f"Preview: {preview}. Parse errors: {details}"
        )

    if isinstance(payload, dict):
        for key in (
            "rules",
            "legal_rules",
            "data",
            "results",
            "output",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                payload = value
                break
        else:
            if "condition" in payload and "effect" in payload:
                payload = [payload]

    if not isinstance(payload, list):
        raise ValueError(
            "JSON hợp lệ nhưng không chứa danh sách rule."
        )

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

Chỉ trả về JSON object có dạng:
{{"rules": [...]}}
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
            "format": RULES_JSON_SCHEMA,
            "think": False,
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


def extract_rules_with_retry(
    article: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str]:
    prompt = build_prompt(article)
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(1, args.max_attempts + 1):
        try:
            last_response = call_ollama(
                prompt=prompt,
                model=args.model,
                ollama_url=args.ollama_url,
                temperature=args.temperature,
                num_predict=args.num_predict,
                timeout=args.timeout,
            )

            rules = extract_rules_payload(last_response)
            return rules, last_response

        except Exception as error:
            last_error = error

            print(
                f"  -> Attempt {attempt}/{args.max_attempts} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < args.max_attempts:
                time.sleep(args.retry_delay)

    raise ValueError(
        f"Không trích xuất được sau {args.max_attempts} lần. "
        f"Lỗi cuối: {last_error}"
    )


def validate_and_prepare_rule(
    raw_rule: dict[str, Any],
    article: dict[str, Any],
) -> dict[str, Any] | None:
    legal_subject = normalize_text(raw_rule.get("legal_subject"))
    condition = normalize_text(raw_rule.get("condition"))
    effect = normalize_text(raw_rule.get("effect"))
    rule_text = normalize_text(raw_rule.get("rule_text"))

    if not condition or not effect:
        return None

    if not rule_text:
        subject_part = f"{legal_subject} " if legal_subject else ""
        rule_text = f"Nếu {subject_part}{condition} thì {effect}."

    if not rule_text.endswith((".", "?", "!")):
        rule_text += "."

    return {
        "index": 0,
        "article_id": article.get("article_id"),
        "legal_subject": legal_subject,
        "condition": condition,
        "effect": effect,
        "rule_text": rule_text,
        "article_title": normalize_text(
            article.get("article_title")
        ),
        "content": str(article.get("content") or "").strip(),
    }


def rule_signature(rule: dict[str, Any]) -> tuple[Any, ...]:
    return (
        rule.get("article_id"),
        normalize_text(rule.get("legal_subject")).lower(),
        normalize_text(rule.get("condition")).lower(),
        normalize_text(rule.get("effect")).lower(),
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


def get_support_paths(output_path: Path) -> dict[str, Path]:
    base_name = output_path.stem

    return {
        "state": output_path.parent / f"{base_name}_state.json",
        "errors": output_path.parent / f"{base_name}_errors",
        "empty": output_path.parent / f"{base_name}_empty_responses",
    }


def load_state(state_path: Path) -> dict[str, list[int]]:
    default_state = {
        "processed_article_ids": [],
        "empty_article_ids": [],
        "failed_article_ids": [],
    }

    if not state_path.exists():
        return default_state

    try:
        payload = load_json(state_path)
    except Exception:
        return default_state

    if not isinstance(payload, dict):
        return default_state

    result: dict[str, list[int]] = {}

    for key in default_state:
        values = payload.get(key, [])

        if not isinstance(values, list):
            values = []

        clean_values: list[int] = []

        for value in values:
            try:
                clean_values.append(int(value))
            except (TypeError, ValueError):
                continue

        result[key] = sorted(set(clean_values))

    return result


def save_state(
    state: dict[str, set[int]],
    state_path: Path,
) -> None:
    serializable = {
        key: sorted(values)
        for key, values in state.items()
    }

    atomic_save_json(serializable, state_path)


def save_error_artifacts(
    errors_directory: Path,
    article: dict[str, Any],
    raw_response: str,
    error: Exception,
) -> None:
    errors_directory.mkdir(parents=True, exist_ok=True)

    article_id = article.get("article_id", "unknown")

    response_path = (
        errors_directory
        / f"article_{article_id}_response.txt"
    )
    metadata_path = (
        errors_directory
        / f"article_{article_id}_error.json"
    )

    response_path.write_text(
        raw_response or "",
        encoding="utf-8",
    )

    atomic_save_json(
        {
            "article_id": article_id,
            "article_title": normalize_text(
                article.get("article_title")
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "raw_response_file": str(response_path),
        },
        metadata_path,
    )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    support_paths = get_support_paths(output_path)

    state_path = support_paths["state"]
    errors_directory = support_paths["errors"]
    empty_directory = support_paths["empty"]

    if args.overwrite:
        if output_path.exists():
            output_path.unlink()

        if state_path.exists():
            state_path.unlink()

    articles = load_json(input_path)

    if not isinstance(articles, list):
        raise ValueError(
            "File input phải là JSON array các điều luật."
        )

    selected_articles: list[dict[str, Any]] = []

    for article in articles:
        if not isinstance(article, dict):
            continue

        try:
            article_id = int(article.get("article_id"))
        except (TypeError, ValueError):
            continue

        if (
            args.start_article is not None
            and article_id < args.start_article
        ):
            continue

        if (
            args.end_article is not None
            and article_id > args.end_article
        ):
            continue

        selected_articles.append(article)

    if args.limit is not None:
        selected_articles = selected_articles[: args.limit]

    existing_rules: list[dict[str, Any]] = []

    if output_path.exists():
        existing_payload = load_json(output_path)

        if isinstance(existing_payload, list):
            existing_rules = [
                item
                for item in existing_payload
                if isinstance(item, dict)
            ]

    loaded_state = load_state(state_path)

    state: dict[str, set[int]] = {
        "processed_article_ids": set(
            loaded_state["processed_article_ids"]
        ),
        "empty_article_ids": set(
            loaded_state["empty_article_ids"]
        ),
        "failed_article_ids": set(
            loaded_state["failed_article_ids"]
        ),
    }

    for rule in existing_rules:
        try:
            state["processed_article_ids"].add(
                int(rule.get("article_id"))
            )
        except (TypeError, ValueError):
            continue

    all_rules = existing_rules.copy()

    success_count = 0
    empty_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"Input articles: {len(selected_articles)}")
    print(f"Existing rules: {len(existing_rules)}")
    print(f"Model: {args.model}")
    print(f"num_predict: {args.num_predict}")
    print(f"max_attempts: {args.max_attempts}")
    print("-" * 72)

    for position, article in enumerate(
        selected_articles,
        start=1,
    ):
        article_id = int(article["article_id"])
        article_title = normalize_text(
            article.get("article_title")
        )

        already_finished = (
            article_id in state["processed_article_ids"]
            or article_id in state["empty_article_ids"]
        )

        if args.resume and already_finished:
            skipped_count += 1
            print(
                f"[{position}/{len(selected_articles)}] "
                f"Skip Điều {article_id}: đã xử lý"
            )
            continue

        print(
            f"[{position}/{len(selected_articles)}] "
            f"Processing Điều {article_id}: {article_title}"
        )

        started_at = time.time()
        raw_response = ""

        try:
            raw_rules, raw_response = extract_rules_with_retry(
                article=article,
                args=args,
            )

            prepared_rules: list[dict[str, Any]] = []

            for raw_rule in raw_rules:
                prepared_rule = validate_and_prepare_rule(
                    raw_rule,
                    article,
                )

                if prepared_rule is not None:
                    prepared_rules.append(prepared_rule)

            prepared_rules = deduplicate_rules(
                prepared_rules
            )

            all_rules = [
                rule
                for rule in all_rules
                if int(rule.get("article_id", -1)) != article_id
            ]

            if prepared_rules:
                all_rules.extend(prepared_rules)
                success_count += 1

                state["processed_article_ids"].add(article_id)
                state["empty_article_ids"].discard(article_id)
                state["failed_article_ids"].discard(article_id)

                print(
                    f"  -> {len(prepared_rules)} rules "
                    f"({time.time() - started_at:.2f}s)"
                )
            else:
                empty_count += 1

                state["empty_article_ids"].add(article_id)
                state["processed_article_ids"].discard(article_id)
                state["failed_article_ids"].discard(article_id)

                empty_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                empty_response_path = (
                    empty_directory
                    / f"article_{article_id}_response.txt"
                )

                empty_response_path.write_text(
                    raw_response or "",
                    encoding="utf-8",
                )

                print(
                    "  -> Không có rule hợp lệ "
                    f"({time.time() - started_at:.2f}s)"
                )
                print(
                    "  -> Empty response saved in: "
                    f"{empty_response_path}"
                )

            all_rules = reindex_rules(
                deduplicate_rules(all_rules)
            )

            atomic_save_json(all_rules, output_path)
            save_state(state, state_path)

        except Exception as error:
            failed_count += 1

            state["failed_article_ids"].add(article_id)
            state["processed_article_ids"].discard(article_id)
            state["empty_article_ids"].discard(article_id)

            save_error_artifacts(
                errors_directory=errors_directory,
                article=article,
                raw_response=raw_response,
                error=error,
            )

            save_state(state, state_path)

            print(
                f"  -> ERROR: {type(error).__name__}: {error}"
            )
            print(
                "  -> Raw response/error saved in: "
                f"{errors_directory}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    all_rules = reindex_rules(
        deduplicate_rules(all_rules)
    )

    atomic_save_json(all_rules, output_path)
    save_state(state, state_path)

    print("\n" + "=" * 72)
    print("COMPLETED")
    print("=" * 72)
    print(f"Articles with extracted rules: {success_count}")
    print(f"Articles without valid rules: {empty_count}")
    print(f"Failed articles: {failed_count}")
    print(f"Skipped articles: {skipped_count}")
    print(f"Total output rules: {len(all_rules)}")
    print(f"Output: {output_path}")
    print(f"State: {state_path}")
    print(f"Error directory: {errors_directory}")
    print(f"Empty response directory: {empty_directory}")


if __name__ == "__main__":
    main()
