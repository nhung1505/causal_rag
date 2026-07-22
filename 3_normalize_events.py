from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


DEFAULT_INPUT = "data/2_blhs_rules_raw.json"
DEFAULT_OUTPUT = "data/4_blhs_merged.json"
DEFAULT_CATALOG = "data/3_event_catalog.json"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"


NORMALIZATION_INSTRUCTION = """
Bạn là chuyên gia chuẩn hóa sự kiện pháp lý trong Bộ luật Hình sự Việt Nam.

Mục tiêu:
Ánh xạ một cụm condition hoặc effect tự nhiên vào một sự kiện chuẩn trong
event catalog để những câu khác nhau nhưng biểu đạt CÙNG MỘT SỰ KIỆN PHÁP LÝ
có chung event_id.

Đây là chuẩn hóa NGỮ NGHĨA, không phải chỉ bỏ dấu hoặc nối từ bằng dấu gạch dưới.

Ví dụ có thể cùng một sự kiện:
- "phải chịu trách nhiệm hình sự"
- "chịu trách nhiệm hình sự"
- "bị truy cứu trách nhiệm hình sự"
Có thể cùng ánh xạ về sự kiện chuẩn "Chịu trách nhiệm hình sự" nếu ngữ cảnh
không tạo ra khác biệt pháp lý đáng kể.

Nhưng KHÔNG được gộp các khái niệm chỉ gần nghĩa hoặc có quan hệ với nhau:
- "Miễn trách nhiệm hình sự" khác "Không phải chịu trách nhiệm hình sự".
- "Không truy cứu trách nhiệm hình sự" khác "Miễn trách nhiệm hình sự".
- "Phạm tội" khác "Chịu trách nhiệm hình sự".
- "Có nhiều tình tiết giảm nhẹ" khác "Được giảm hình phạt".
- "Tội phạm ít nghiêm trọng" khác "Phạm tội ít nghiêm trọng".
- "Có thể được miễn" khác "Được miễn" nếu tính khả năng là nội dung pháp lý
  quan trọng của văn bản.
- Các ngưỡng tuổi, mức hình phạt, địa điểm, thời điểm hoặc đối tượng khác nhau
  không được gộp nếu chúng làm thay đổi phạm vi áp dụng.

Nhiệm vụ:
1. Đọc cụm từ cần chuẩn hóa và toàn bộ rule context.
2. So sánh với các candidate events.
3. Chọn action = "USE_EXISTING" chỉ khi candidate biểu đạt cùng một trạng thái,
   hành vi, điều kiện hoặc hệ quả pháp lý.
4. Chọn action = "CREATE_NEW" nếu không candidate nào tương đương pháp lý.
5. Không chọn chỉ vì có nhiều từ giống nhau.
6. Không tạo event mới chỉ vì khác cách diễn đạt bề mặt.

Khi CREATE_NEW:
- event_name phải là tên sự kiện chuẩn, tiếng Việt có dấu, ngắn gọn và độc lập.
- event_name nên dùng dạng danh từ hoặc động từ pháp lý rõ nghĩa.
- description phải mô tả ranh giới ngữ nghĩa để phân biệt với event gần nghĩa.
- event_type chọn một trong:
  ACTION,
  STATE,
  LEGAL_CONDITION,
  LEGAL_CONSEQUENCE,
  SANCTION,
  OBLIGATION,
  PERMISSION,
  PROHIBITION,
  EXEMPTION,
  CLASSIFICATION,
  SCOPE_APPLICATION,
  PROCEDURE,
  OTHER.
- canonical_alias là cách diễn đạt chuẩn gần với cụm đầu vào.

Khi USE_EXISTING:
- selected_event_id phải đúng bằng event_id của một candidate.
- event_name, description và event_type có thể để rỗng.
- canonical_alias là cụm đầu vào đã được làm gọn nhưng không thay đổi nghĩa.

Chỉ trả về JSON, không giải thích, không Markdown.
""".strip()


NORMALIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["USE_EXISTING", "CREATE_NEW"],
        },
        "selected_event_id": {"type": "string"},
        "event_name": {"type": "string"},
        "description": {"type": "string"},
        "event_type": {
            "type": "string",
            "enum": [
                "ACTION",
                "STATE",
                "LEGAL_CONDITION",
                "LEGAL_CONSEQUENCE",
                "SANCTION",
                "OBLIGATION",
                "PERMISSION",
                "PROHIBITION",
                "EXEMPTION",
                "CLASSIFICATION",
                "SCOPE_APPLICATION",
                "PROCEDURE",
                "OTHER",
            ],
        },
        "canonical_alias": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "action",
        "selected_event_id",
        "event_name",
        "description",
        "event_type",
        "canonical_alias",
        "reason",
    ],
}


VALID_EVENT_TYPES = {
    "ACTION",
    "STATE",
    "LEGAL_CONDITION",
    "LEGAL_CONSEQUENCE",
    "SANCTION",
    "OBLIGATION",
    "PERMISSION",
    "PROHIBITION",
    "EXEMPTION",
    "CLASSIFICATION",
    "SCOPE_APPLICATION",
    "PROCEDURE",
    "OTHER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chuẩn hóa condition/effect thành các event dùng chung để xây dựng "
            "causal graph."
        )
    )

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Model embedding của Ollama, ví dụ nomic-embed-text. "
            "Bỏ trống để chỉ dùng lexical matching."
        ),
    )
    parser.add_argument(
        "--ollama-generate-url",
        default=DEFAULT_OLLAMA_GENERATE_URL,
    )
    parser.add_argument(
        "--ollama-embed-url",
        default=DEFAULT_OLLAMA_EMBED_URL,
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.1)

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Số candidate events đưa cho LLM so sánh.",
    )
    parser.add_argument(
        "--exact-alias-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự động dùng event khi alias trùng chính xác sau chuẩn hóa.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Chỉ xử lý từ rule index này.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Chỉ xử lý đến rule index này.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số rule để chạy thử.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ output hiện có và bỏ qua rule đã chuẩn hóa.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa output, catalog và state cũ trước khi chạy.",
    )

    return parser.parse_args()


def load_json(path_value: str | Path, default: Any = None) -> Any:
    path = Path(path_value)

    if not path.exists():
        if default is not None:
            return default
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


def normalize_spaces(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_for_matching(value: Any) -> str:
    text = normalize_spaces(value).lower()
    text = text.replace("đ", "d")

    text = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    slug = normalize_for_matching(value).replace(" ", "_")
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "su_kien"


def token_set(value: str) -> set[str]:
    return set(normalize_for_matching(value).split())


def lexical_similarity(first: str, second: str) -> float:
    normalized_first = normalize_for_matching(first)
    normalized_second = normalize_for_matching(second)

    if not normalized_first or not normalized_second:
        return 0.0

    first_tokens = token_set(normalized_first)
    second_tokens = token_set(normalized_second)

    union = first_tokens | second_tokens
    intersection = first_tokens & second_tokens

    jaccard = len(intersection) / len(union) if union else 0.0
    sequence = SequenceMatcher(
        None,
        normalized_first,
        normalized_second,
    ).ratio()

    containment = 0.0
    shorter = min(len(first_tokens), len(second_tokens))

    if shorter:
        containment = len(intersection) / shorter

    return 0.45 * jaccard + 0.35 * sequence + 0.20 * containment


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if not first or not second or len(first) != len(second):
        return 0.0

    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def call_embedding(
    text: str,
    model: str,
    url: str,
    timeout: int,
) -> list[float]:
    response = requests.post(
        url,
        json={
            "model": model,
            "input": text,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    embeddings = payload.get("embeddings")

    if (
        not isinstance(embeddings, list)
        or not embeddings
        or not isinstance(embeddings[0], list)
    ):
        raise ValueError("Ollama không trả về embeddings hợp lệ.")

    return [float(value) for value in embeddings[0]]


def event_search_text(event: dict[str, Any]) -> str:
    parts = [
        normalize_spaces(event.get("event_name")),
        normalize_spaces(event.get("description")),
    ]

    aliases = event.get("aliases", [])

    if isinstance(aliases, list):
        parts.extend(normalize_spaces(alias) for alias in aliases)

    return " | ".join(part for part in parts if part)


def get_event_embedding(
    event: dict[str, Any],
    args: argparse.Namespace,
    embedding_cache: dict[str, list[float]],
) -> list[float] | None:
    if not args.embedding_model:
        return None

    event_id = str(event.get("event_id") or "")

    if event_id in embedding_cache:
        return embedding_cache[event_id]

    embedding = call_embedding(
        text=event_search_text(event),
        model=args.embedding_model,
        url=args.ollama_embed_url,
        timeout=args.timeout,
    )
    embedding_cache[event_id] = embedding
    return embedding


def find_exact_alias(
    phrase: str,
    catalog: list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized_phrase = normalize_for_matching(phrase)

    for event in catalog:
        values = [event.get("event_name", "")]
        aliases = event.get("aliases", [])

        if isinstance(aliases, list):
            values.extend(aliases)

        for value in values:
            if normalize_for_matching(value) == normalized_phrase:
                return event

    return None


def rank_candidates(
    phrase: str,
    catalog: list[dict[str, Any]],
    args: argparse.Namespace,
    embedding_cache: dict[str, list[float]],
) -> list[dict[str, Any]]:
    if not catalog:
        return []

    phrase_embedding: list[float] | None = None

    if args.embedding_model:
        try:
            phrase_embedding = call_embedding(
                text=phrase,
                model=args.embedding_model,
                url=args.ollama_embed_url,
                timeout=args.timeout,
            )
        except Exception as error:
            print(
                "    Embedding unavailable, fallback lexical: "
                f"{type(error).__name__}: {error}"
            )
            phrase_embedding = None

    ranked: list[tuple[float, dict[str, Any]]] = []

    for event in catalog:
        lexical_score = lexical_similarity(
            phrase,
            event_search_text(event),
        )
        embedding_score = 0.0

        if phrase_embedding is not None:
            try:
                event_embedding = get_event_embedding(
                    event,
                    args,
                    embedding_cache,
                )
                if event_embedding is not None:
                    embedding_score = cosine_similarity(
                        phrase_embedding,
                        event_embedding,
                    )
            except Exception:
                embedding_score = 0.0

        if phrase_embedding is not None:
            score = 0.75 * embedding_score + 0.25 * lexical_score
        else:
            score = lexical_score

        candidate = dict(event)
        candidate["_candidate_score"] = round(score, 6)
        ranked.append((score, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        candidate
        for _, candidate in ranked[: max(args.top_k, 1)]
    ]


def compact_candidate(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "event_type": event.get("event_type"),
        "description": event.get("description"),
        "aliases": event.get("aliases", [])[:8],
        "similarity_score": event.get("_candidate_score", 0.0),
    }


def build_normalization_prompt(
    phrase: str,
    role: str,
    rule: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    context = {
        "article_id": rule.get("article_id"),
        "article_title": rule.get("article_title"),
        "legal_subject": rule.get("legal_subject"),
        "condition": rule.get("condition"),
        "effect": rule.get("effect"),
        "rule_text": rule.get("rule_text"),
        "normalizing_role": role,
        "phrase_to_normalize": phrase,
    }

    candidate_payload = [
        compact_candidate(candidate)
        for candidate in candidates
    ]

    return f"""
{NORMALIZATION_INSTRUCTION}

RULE CONTEXT:
{json.dumps(context, ensure_ascii=False, indent=2)}

CANDIDATE EVENTS:
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

Hãy trả về quyết định chuẩn hóa dưới dạng JSON.
""".strip()


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


def call_llm_normalizer(
    prompt: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    last_response = ""

    for attempt in range(1, args.max_attempts + 1):
        try:
            response = requests.post(
                args.ollama_generate_url,
                json={
                    "model": args.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": NORMALIZATION_SCHEMA,
                    "think": False,
                    "options": {
                        "temperature": args.temperature,
                        "num_predict": args.num_predict,
                    },
                },
                timeout=args.timeout,
            )
            response.raise_for_status()

            payload = response.json()
            last_response = str(payload.get("response") or "")
            cleaned = clean_model_response(last_response)
            decision = json.loads(cleaned)

            if not isinstance(decision, dict):
                raise ValueError(
                    "Kết quả chuẩn hóa không phải JSON object."
                )

            return decision, last_response

        except Exception as error:
            last_error = error
            print(
                f"    Attempt {attempt}/{args.max_attempts} failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < args.max_attempts:
                time.sleep(args.retry_delay)

    raise ValueError(
        f"Chuẩn hóa thất bại sau {args.max_attempts} lần. "
        f"Lỗi cuối: {last_error}. Raw response: {last_response[:500]}"
    )


def generate_unique_event_id(
    event_name: str,
    catalog: list[dict[str, Any]],
) -> str:
    base = slugify(event_name)
    existing_ids = {
        str(event.get("event_id"))
        for event in catalog
    }

    if base not in existing_ids:
        return base

    suffix = 2

    while f"{base}_{suffix}" in existing_ids:
        suffix += 1

    return f"{base}_{suffix}"


def ensure_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    result: list[str] = []

    for item in value:
        text = normalize_spaces(item)
        if text and text not in result:
            result.append(text)

    return result


def add_alias(event: dict[str, Any], alias: str) -> None:
    alias = normalize_spaces(alias)

    if not alias:
        return

    aliases = ensure_string_list(event.get("aliases", []))
    normalized_aliases = {
        normalize_for_matching(value)
        for value in aliases
    }

    if normalize_for_matching(alias) not in normalized_aliases:
        aliases.append(alias)

    event["aliases"] = aliases


def add_source_reference(
    event: dict[str, Any],
    article_id: Any,
    rule_index: Any,
    role: str,
) -> None:
    article_ids = event.get("source_article_ids", [])

    if not isinstance(article_ids, list):
        article_ids = []

    try:
        article_value = int(article_id)
        if article_value not in article_ids:
            article_ids.append(article_value)
    except (TypeError, ValueError):
        pass

    event["source_article_ids"] = sorted(article_ids)

    occurrences = event.get("occurrences", [])

    if not isinstance(occurrences, list):
        occurrences = []

    occurrence = {
        "article_id": article_id,
        "rule_index": rule_index,
        "role": role,
    }

    if occurrence not in occurrences:
        occurrences.append(occurrence)

    event["occurrences"] = occurrences


def create_event(
    decision: dict[str, Any],
    phrase: str,
    rule: dict[str, Any],
    role: str,
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    event_name = normalize_spaces(decision.get("event_name"))

    if not event_name:
        event_name = normalize_spaces(phrase)

    description = normalize_spaces(decision.get("description"))

    if not description:
        description = (
            f"Sự kiện pháp lý chuẩn hóa từ cụm: {normalize_spaces(phrase)}."
        )

    event_type = normalize_spaces(
        decision.get("event_type")
    ).upper()

    if event_type not in VALID_EVENT_TYPES:
        event_type = (
            "LEGAL_CONDITION"
            if role == "condition"
            else "LEGAL_CONSEQUENCE"
        )

    event_id = generate_unique_event_id(
        event_name,
        catalog,
    )

    canonical_alias = normalize_spaces(
        decision.get("canonical_alias")
    ) or normalize_spaces(phrase)

    event = {
        "event_id": event_id,
        "event_name": event_name,
        "event_type": event_type,
        "description": description,
        "aliases": [],
        "source_article_ids": [],
        "occurrences": [],
        "status": "CANDIDATE",
    }

    add_alias(event, canonical_alias)
    add_alias(event, phrase)
    add_source_reference(
        event,
        rule.get("article_id"),
        rule.get("index"),
        role,
    )

    catalog.append(event)
    return event


def use_existing_event(
    event: dict[str, Any],
    decision: dict[str, Any],
    phrase: str,
    rule: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    canonical_alias = normalize_spaces(
        decision.get("canonical_alias")
    ) or normalize_spaces(phrase)

    add_alias(event, canonical_alias)
    add_alias(event, phrase)
    add_source_reference(
        event,
        rule.get("article_id"),
        rule.get("index"),
        role,
    )

    return event


def normalize_phrase(
    phrase: str,
    role: str,
    rule: dict[str, Any],
    catalog: list[dict[str, Any]],
    args: argparse.Namespace,
    embedding_cache: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    phrase = normalize_spaces(phrase)

    if not phrase:
        raise ValueError(
            f"Rule {rule.get('index')} có {role} rỗng."
        )

    if args.exact_alias_match:
        exact_event = find_exact_alias(phrase, catalog)

        if exact_event is not None:
            add_source_reference(
                exact_event,
                rule.get("article_id"),
                rule.get("index"),
                role,
            )

            decision_metadata = {
                "action": "USE_EXISTING",
                "method": "EXACT_ALIAS",
                "selected_event_id": exact_event["event_id"],
                "reason": "Alias trùng chính xác sau chuẩn hóa văn bản.",
            }
            return exact_event, decision_metadata

    candidates = rank_candidates(
        phrase=phrase,
        catalog=catalog,
        args=args,
        embedding_cache=embedding_cache,
    )

    prompt = build_normalization_prompt(
        phrase=phrase,
        role=role,
        rule=rule,
        candidates=candidates,
    )

    decision, _ = call_llm_normalizer(prompt, args)
    action = normalize_spaces(decision.get("action")).upper()

    candidate_by_id = {
        str(candidate.get("event_id")): candidate
        for candidate in candidates
    }

    selected_event_id = normalize_spaces(
        decision.get("selected_event_id")
    )

    if (
        action == "USE_EXISTING"
        and selected_event_id in candidate_by_id
    ):
        real_event = next(
            event
            for event in catalog
            if str(event.get("event_id")) == selected_event_id
        )

        event = use_existing_event(
            real_event,
            decision,
            phrase,
            rule,
            role,
        )

        metadata = {
            "action": "USE_EXISTING",
            "method": "LLM_CANDIDATE_MATCH",
            "selected_event_id": event["event_id"],
            "candidate_scores": {
                str(candidate.get("event_id")): candidate.get(
                    "_candidate_score",
                    0.0,
                )
                for candidate in candidates
            },
            "reason": normalize_spaces(decision.get("reason")),
        }
        return event, metadata

    event = create_event(
        decision=decision,
        phrase=phrase,
        rule=rule,
        role=role,
        catalog=catalog,
    )

    embedding_cache.pop(event["event_id"], None)

    metadata = {
        "action": "CREATE_NEW",
        "method": "LLM_NEW_EVENT",
        "selected_event_id": event["event_id"],
        "candidate_scores": {
            str(candidate.get("event_id")): candidate.get(
                "_candidate_score",
                0.0,
            )
            for candidate in candidates
        },
        "reason": normalize_spaces(decision.get("reason")),
    }

    return event, metadata


def validate_catalog(
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for event in catalog:
        if not isinstance(event, dict):
            continue

        event_id = normalize_spaces(event.get("event_id"))
        event_name = normalize_spaces(event.get("event_name"))

        if not event_id or not event_name or event_id in seen_ids:
            continue

        seen_ids.add(event_id)

        event["event_id"] = event_id
        event["event_name"] = event_name
        event["event_type"] = (
            normalize_spaces(event.get("event_type")).upper()
            if normalize_spaces(event.get("event_type")).upper()
            in VALID_EVENT_TYPES
            else "OTHER"
        )
        event["description"] = normalize_spaces(
            event.get("description")
        )
        event["aliases"] = ensure_string_list(
            event.get("aliases", [])
        )
        event["source_article_ids"] = sorted(
            {
                int(value)
                for value in event.get("source_article_ids", [])
                if str(value).isdigit()
            }
        )

        if not isinstance(event.get("occurrences"), list):
            event["occurrences"] = []

        event["status"] = normalize_spaces(
            event.get("status")
        ) or "CANDIDATE"

        result.append(event)

    return result


def rule_key(rule: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(rule.get("index", "")),
            str(rule.get("article_id", "")),
            normalize_spaces(rule.get("condition")),
            normalize_spaces(rule.get("effect")),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_state_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_state.json"


def get_error_directory(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_errors"


def save_error(
    error_directory: Path,
    rule: dict[str, Any],
    error: Exception,
) -> None:
    error_directory.mkdir(parents=True, exist_ok=True)

    rule_index = rule.get("index", "unknown")
    path = error_directory / f"rule_{rule_index}_error.json"

    atomic_save_json(
        {
            "rule_index": rule_index,
            "article_id": rule.get("article_id"),
            "condition": rule.get("condition"),
            "effect": rule.get("effect"),
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
        path,
    )


def merge_existing_output(
    raw_rules: list[dict[str, Any]],
    existing_output: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_by_key = {
        rule_key(rule): rule
        for rule in existing_output
        if isinstance(rule, dict)
    }

    result: list[dict[str, Any]] = []

    for raw_rule in raw_rules:
        key = rule_key(raw_rule)

        if key in existing_by_key:
            result.append(existing_by_key[key])
        else:
            result.append(dict(raw_rule))

    return result


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    catalog_path = Path(args.catalog)
    state_path = get_state_path(output_path)
    error_directory = get_error_directory(output_path)

    if args.overwrite:
        for path in (output_path, catalog_path, state_path):
            if path.exists():
                path.unlink()

    raw_rules = load_json(input_path)

    if not isinstance(raw_rules, list):
        raise ValueError("Input phải là JSON array các rule thô.")

    raw_rules = [
        rule
        for rule in raw_rules
        if isinstance(rule, dict)
    ]

    selected_rule_keys: set[str] = set()
    selected_count = 0

    for rule in raw_rules:
        try:
            index = int(rule.get("index"))
        except (TypeError, ValueError):
            continue

        if args.start_index is not None and index < args.start_index:
            continue

        if args.end_index is not None and index > args.end_index:
            continue

        if args.limit is not None and selected_count >= args.limit:
            continue

        selected_rule_keys.add(rule_key(rule))
        selected_count += 1

    existing_output: list[dict[str, Any]] = []

    if args.resume and output_path.exists():
        payload = load_json(output_path, default=[])
        if isinstance(payload, list):
            existing_output = [
                item for item in payload if isinstance(item, dict)
            ]

    output_rules = merge_existing_output(
        raw_rules,
        existing_output,
    )

    catalog_payload = load_json(catalog_path, default=[])

    if not isinstance(catalog_payload, list):
        raise ValueError("Event catalog phải là JSON array.")

    catalog = validate_catalog(catalog_payload)
    embedding_cache: dict[str, list[float]] = {}

    state_payload = load_json(
        state_path,
        default={
            "processed_rule_keys": [],
            "failed_rule_keys": [],
        },
    )

    processed_rule_keys = set(
        state_payload.get("processed_rule_keys", [])
        if isinstance(state_payload, dict)
        else []
    )
    failed_rule_keys = set(
        state_payload.get("failed_rule_keys", [])
        if isinstance(state_payload, dict)
        else []
    )

    success_count = 0
    skipped_count = 0
    failed_count = 0
    created_before = len(catalog)

    print(f"Input rules: {len(raw_rules)}")
    print(f"Selected rules: {len(selected_rule_keys)}")
    print(f"Existing events: {len(catalog)}")
    print(f"Model: {args.model}")
    print(
        "Embedding model: "
        f"{args.embedding_model or 'disabled (lexical fallback)'}"
    )
    print("-" * 72)

    for position, rule in enumerate(output_rules, start=1):
        key = rule_key(rule)

        if key not in selected_rule_keys:
            continue

        already_normalized = (
            normalize_spaces(rule.get("condition_event"))
            and normalize_spaces(rule.get("effect_event"))
        )

        if args.resume and (
            key in processed_rule_keys or already_normalized
        ):
            skipped_count += 1
            print(
                f"[{position}/{len(output_rules)}] "
                f"Skip rule {rule.get('index')}: đã chuẩn hóa"
            )
            continue

        print(
            f"[{position}/{len(output_rules)}] "
            f"Normalizing rule {rule.get('index')} "
            f"(Điều {rule.get('article_id')})"
        )

        started_at = time.time()

        try:
            condition_event, condition_metadata = normalize_phrase(
                phrase=normalize_spaces(rule.get("condition")),
                role="condition",
                rule=rule,
                catalog=catalog,
                args=args,
                embedding_cache=embedding_cache,
            )

            effect_event, effect_metadata = normalize_phrase(
                phrase=normalize_spaces(rule.get("effect")),
                role="effect",
                rule=rule,
                catalog=catalog,
                args=args,
                embedding_cache=embedding_cache,
            )

            rule["condition_event"] = condition_event["event_id"]
            rule["condition_event_name"] = condition_event["event_name"]
            rule["effect_event"] = effect_event["event_id"]
            rule["effect_event_name"] = effect_event["event_name"]

            rule["normalization_metadata"] = {
                "condition": condition_metadata,
                "effect": effect_metadata,
            }

            processed_rule_keys.add(key)
            failed_rule_keys.discard(key)
            success_count += 1

            atomic_save_json(output_rules, output_path)
            atomic_save_json(catalog, catalog_path)
            atomic_save_json(
                {
                    "processed_rule_keys": sorted(
                        processed_rule_keys
                    ),
                    "failed_rule_keys": sorted(failed_rule_keys),
                },
                state_path,
            )

            print(
                f"  condition_event: {condition_event['event_id']}"
            )
            print(
                f"  effect_event:    {effect_event['event_id']}"
            )
            print(
                f"  -> completed ({time.time() - started_at:.2f}s)"
            )

        except Exception as error:
            failed_count += 1
            failed_rule_keys.add(key)
            processed_rule_keys.discard(key)

            save_error(
                error_directory=error_directory,
                rule=rule,
                error=error,
            )

            atomic_save_json(catalog, catalog_path)
            atomic_save_json(
                {
                    "processed_rule_keys": sorted(
                        processed_rule_keys
                    ),
                    "failed_rule_keys": sorted(failed_rule_keys),
                },
                state_path,
            )

            print(
                f"  -> ERROR: {type(error).__name__}: {error}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)

    atomic_save_json(output_rules, output_path)
    atomic_save_json(catalog, catalog_path)
    atomic_save_json(
        {
            "processed_rule_keys": sorted(processed_rule_keys),
            "failed_rule_keys": sorted(failed_rule_keys),
        },
        state_path,
    )

    print("\n" + "=" * 72)
    print("COMPLETED")
    print("=" * 72)
    print(f"Successfully normalized rules: {success_count}")
    print(f"Skipped rules: {skipped_count}")
    print(f"Failed rules: {failed_count}")
    print(f"Total catalog events: {len(catalog)}")
    print(f"New events created: {len(catalog) - created_before}")
    print(f"Output: {output_path}")
    print(f"Catalog: {catalog_path}")
    print(f"State: {state_path}")
    print(f"Error directory: {error_directory}")


if __name__ == "__main__":
    main()
