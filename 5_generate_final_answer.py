from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, request


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

VERIFICATION_RESULT_PATH = (
    "data/counterfactual_verification_result.json"
)
RETRIEVAL_RESULT_PATH = "data/retrieval_result.json"
OUTPUT_PATH = "data/final_answer_result.json"

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_MAX_EVIDENCE = 8
DEFAULT_MAX_PATHS = 6
DEFAULT_MAX_CONTEXT_CHARS = 18000
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 180

DEFAULT_MIN_VERIFICATION_SCORE = 0.45

# Trọng số xếp hạng evidence trước khi đưa vào LLM.
FINAL_EVIDENCE_SCORE_WEIGHT = 0.55
ORIGINAL_RETRIEVAL_SCORE_WEIGHT = 0.30
COUNTERFACTUAL_SUPPORT_WEIGHT = 0.15


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FinalEvidence:
    evidence_index: int
    rule_id: str
    article_id: str
    article_title: str
    legal_subject: str
    condition: str
    effect: str
    condition_event: str
    condition_event_name: str
    effect_event: str
    effect_event_name: str
    causal_type: str

    verification_score: float
    original_final_score: float
    counterfactual_support_score: float
    final_selection_score: float

    decision: str
    path_ids: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class FinalPath:
    path_index: int
    original_path_id: int
    status: str
    consistency_score: float
    seed_event_id: str
    seed_event_name: str
    outcome_event_id: str
    outcome_event_name: str
    explanation: str
    event_names: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    provider: str
    model: str

    selected_evidence: list[dict[str, Any]]
    selected_paths: list[dict[str, Any]]

    citations_used: list[str]
    confidence: float
    consistency_score: float

    generation_metadata: dict[str, Any]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number != number:
        return default

    return number


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    return max(lower, min(upper, value))


def unique_preserve_order(
    values: Iterable[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = safe_string(value)

        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def load_json(path: str) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: str,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved final answer: {file_path}")


def truncate_text(
    text: str,
    max_chars: int,
) -> str:
    text = safe_string(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[truncated]"


def normalize_article_label(
    article_id: str,
    article_title: str,
) -> str:
    article_id = safe_string(article_id)
    article_title = safe_string(article_title)

    if article_id and article_title:
        return f"Điều {article_id} – {article_title}"

    if article_id:
        return f"Điều {article_id}"

    if article_title:
        return article_title

    return "Không xác định điều luật"


# ============================================================
# INPUT STORE
# ============================================================

class FinalAnswerInputStore:
    def __init__(
        self,
        *,
        verification_result_path: str,
        retrieval_result_path: str,
    ) -> None:
        self.verification_result_path = (
            Path(verification_result_path)
        )
        self.retrieval_result_path = (
            Path(retrieval_result_path)
        )

        self.verification_result = (
            self._load_verification_result()
        )
        self.retrieval_result = (
            self._load_retrieval_result_optional()
        )

        self._validate()

    def _load_verification_result(
        self,
    ) -> dict[str, Any]:
        print(
            "Loading verification result:",
            self.verification_result_path,
        )
        return load_json(
            str(self.verification_result_path)
        )

    def _load_retrieval_result_optional(
        self,
    ) -> dict[str, Any]:
        if not self.retrieval_result_path.exists():
            print(
                "Warning: retrieval result không tồn tại. "
                "Chỉ dùng dữ liệu trong verification result."
            )
            return {}

        print(
            "Loading retrieval result:",
            self.retrieval_result_path,
        )
        return load_json(
            str(self.retrieval_result_path)
        )

    def _validate(self) -> None:
        required = {
            "query",
            "verified_evidence",
            "uncertain_evidence",
            "removed_evidence",
            "path_verifications",
        }

        missing = required - set(
            self.verification_result
        )

        if missing:
            raise ValueError(
                "Verification result thiếu trường: "
                f"{sorted(missing)}"
            )

    @property
    def query(self) -> str:
        return safe_string(
            self.verification_result.get("query")
        )

    @property
    def confidence(self) -> float:
        return safe_float(
            self.verification_result.get(
                "confidence"
            )
        )

    @property
    def consistency_score(self) -> float:
        return safe_float(
            self.verification_result.get(
                "consistency_score"
            )
        )


# ============================================================
# EVIDENCE AND PATH SELECTION
# ============================================================

class FinalContextSelector:
    def __init__(
        self,
        store: FinalAnswerInputStore,
    ) -> None:
        self.store = store

    def select_evidence(
        self,
        *,
        max_evidence: int,
        min_verification_score: float,
        include_uncertain: bool,
    ) -> list[FinalEvidence]:
        candidates: list[FinalEvidence] = []

        verified_items = (
            self.store.verification_result.get(
                "verified_evidence",
                [],
            )
        )

        uncertain_items = (
            self.store.verification_result.get(
                "uncertain_evidence",
                [],
            )
            if include_uncertain
            else []
        )

        for item in verified_items:
            evidence = self._convert_evidence(
                item,
                default_decision="KEEP",
            )

            if (
                evidence.verification_score
                >= min_verification_score
            ):
                candidates.append(evidence)

        for item in uncertain_items:
            evidence = self._convert_evidence(
                item,
                default_decision="UNCERTAIN",
            )

            if (
                evidence.verification_score
                >= min_verification_score
            ):
                candidates.append(evidence)

        # Fallback: nếu không có evidence đạt ngưỡng, lấy evidence
        # có điểm cao nhất để LLM vẫn có ngữ cảnh nhưng prompt sẽ
        # yêu cầu nêu rõ mức độ không chắc chắn.
        if not candidates:
            fallback_items = (
                list(verified_items)
                + list(uncertain_items)
            )

            if fallback_items:
                fallback_items.sort(
                    key=lambda item: safe_float(
                        item.get(
                            "verification_score"
                        )
                    ),
                    reverse=True,
                )
                candidates.append(
                    self._convert_evidence(
                        fallback_items[0],
                        default_decision=(
                            safe_string(
                                fallback_items[0].get(
                                    "decision"
                                )
                            )
                            or "UNCERTAIN"
                        ),
                    )
                )

        candidates.sort(
            key=lambda item: (
                item.final_selection_score,
                item.verification_score,
                item.original_final_score,
            ),
            reverse=True,
        )

        selected = candidates[:max_evidence]

        for index, evidence in enumerate(
            selected,
            start=1,
        ):
            evidence.evidence_index = index

        return selected

    def _convert_evidence(
        self,
        item: dict[str, Any],
        *,
        default_decision: str,
    ) -> FinalEvidence:
        original = item.get(
            "original_evidence",
            {},
        )

        verification_score = safe_float(
            item.get("verification_score")
        )
        original_final_score = safe_float(
            item.get("original_final_score")
        )
        counterfactual_support_score = (
            safe_float(
                item.get(
                    "counterfactual_support_score"
                )
            )
        )

        final_selection_score = clamp(
            FINAL_EVIDENCE_SCORE_WEIGHT
            * verification_score
            + ORIGINAL_RETRIEVAL_SCORE_WEIGHT
            * original_final_score
            + COUNTERFACTUAL_SUPPORT_WEIGHT
            * counterfactual_support_score
        )

        path_ids = unique_preserve_order(
            [
                str(path_id)
                for path_id in (
                    item.get(
                        "verified_path_ids",
                        [],
                    )
                    + item.get(
                        "unresolved_path_ids",
                        [],
                    )
                )
            ]
        )

        return FinalEvidence(
            evidence_index=0,
            rule_id=safe_string(
                item.get("rule_id")
                or original.get("rule_id")
            ),
            article_id=safe_string(
                item.get("article_id")
                or original.get("article_id")
            ),
            article_title=safe_string(
                original.get("article_title")
            ),
            legal_subject=safe_string(
                original.get("legal_subject")
            ),
            condition=safe_string(
                original.get("condition")
            ),
            effect=safe_string(
                original.get("effect")
            ),
            condition_event=safe_string(
                original.get("condition_event")
            ),
            condition_event_name=safe_string(
                original.get(
                    "condition_event_name"
                )
            ),
            effect_event=safe_string(
                original.get("effect_event")
            ),
            effect_event_name=safe_string(
                original.get(
                    "effect_event_name"
                )
            ),
            causal_type=safe_string(
                original.get("causal_type")
            ),
            verification_score=(
                verification_score
            ),
            original_final_score=(
                original_final_score
            ),
            counterfactual_support_score=(
                counterfactual_support_score
            ),
            final_selection_score=(
                final_selection_score
            ),
            decision=(
                safe_string(
                    item.get("decision")
                )
                or default_decision
            ),
            path_ids=[
                safe_int(path_id)
                for path_id in path_ids
            ],
            reasons=[
                safe_string(reason)
                for reason in item.get(
                    "reasons",
                    [],
                )
                if safe_string(reason)
            ],
        )

    def select_paths(
        self,
        *,
        selected_evidence: list[FinalEvidence],
        max_paths: int,
    ) -> list[FinalPath]:
        relevant_path_ids = {
            path_id
            for evidence in selected_evidence
            for path_id in evidence.path_ids
        }

        path_items = (
            self.store.verification_result.get(
                "path_verifications",
                [],
            )
        )

        retrieval_paths = {
            index: path
            for index, path in enumerate(
                self.store.retrieval_result.get(
                    "causal_paths",
                    [],
                )
            )
        }

        selected: list[FinalPath] = []

        for item in path_items:
            path_id = safe_int(
                item.get("original_path_id"),
                -1,
            )

            if (
                relevant_path_ids
                and path_id not in relevant_path_ids
            ):
                continue

            original_path = retrieval_paths.get(
                path_id,
                {},
            )

            event_names = self._extract_event_names(
                original_path
            )
            rule_ids = [
                safe_string(rule_id)
                for rule_id in original_path.get(
                    "rule_ids",
                    [],
                )
                if safe_string(rule_id)
            ]

            selected.append(
                FinalPath(
                    path_index=0,
                    original_path_id=path_id,
                    status=safe_string(
                        item.get("status")
                    ),
                    consistency_score=safe_float(
                        item.get(
                            "consistency_score"
                        )
                    ),
                    seed_event_id=safe_string(
                        item.get("seed_event_id")
                    ),
                    seed_event_name=safe_string(
                        item.get(
                            "seed_event_name"
                        )
                    ),
                    outcome_event_id=safe_string(
                        item.get(
                            "original_outcome_event_id"
                        )
                    ),
                    outcome_event_name=safe_string(
                        item.get(
                            "original_outcome_event_name"
                        )
                    ),
                    explanation=safe_string(
                        item.get("explanation")
                    ),
                    event_names=event_names,
                    rule_ids=rule_ids,
                )
            )

        selected.sort(
            key=lambda item: (
                item.status == "SUPPORTED",
                item.consistency_score,
            ),
            reverse=True,
        )

        selected = selected[:max_paths]

        for index, path in enumerate(
            selected,
            start=1,
        ):
            path.path_index = index

        return selected

    @staticmethod
    def _extract_event_names(
        original_path: dict[str, Any],
    ) -> list[str]:
        steps = original_path.get(
            "steps",
            [],
        )

        if not steps:
            return []

        names: list[str] = []

        for step in steps:
            source_name = safe_string(
                step.get("source_event_name")
            )
            target_name = safe_string(
                step.get("target_event_name")
            )

            if source_name:
                names.append(source_name)

            if target_name:
                names.append(target_name)

        return unique_preserve_order(names)


# ============================================================
# PROMPT BUILDER
# ============================================================

class LegalAnswerPromptBuilder:
    SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.

Nhiệm vụ của bạn là trả lời câu hỏi chỉ dựa trên evidence và causal path được cung cấp.

Quy tắc bắt buộc:
1. Không tự bổ sung điều luật, hình phạt, điều kiện hoặc ngoại lệ không có trong evidence.
2. Không suy đoán vượt quá quan hệ điều kiện → hệ quả được cung cấp.
3. Mỗi nhận định pháp lý quan trọng phải gắn trích dẫn dạng [E1], [E2], ...
4. Chỉ sử dụng mã trích dẫn evidence có trong ngữ cảnh.
5. Không trích dẫn causal path như nguồn luật độc lập; causal path chỉ hỗ trợ giải thích chuỗi suy luận.
6. Nếu evidence không đủ để kết luận, phải nói rõ “Chưa đủ căn cứ từ dữ liệu được cung cấp”.
7. Evidence có nhãn UNCERTAIN phải được diễn đạt thận trọng.
8. Không khẳng định đây là tư vấn pháp lý chính thức.
9. Trả lời bằng tiếng Việt, rõ ràng, trực tiếp, ưu tiên cấu trúc:
   - Kết luận
   - Căn cứ và lập luận
   - Lưu ý về độ chắc chắn
10. Không tạo danh mục tài liệu ngoài danh sách evidence."""

    def build(
        self,
        *,
        query: str,
        evidence: list[FinalEvidence],
        paths: list[FinalPath],
        global_confidence: float,
        consistency_score: float,
        max_context_chars: int,
    ) -> tuple[str, str]:
        evidence_context = (
            self._build_evidence_context(evidence)
        )
        path_context = (
            self._build_path_context(paths)
        )

        user_prompt = f"""CÂU HỎI:
{query}

ĐỘ TIN CẬY TOÀN CỤC:
- confidence = {global_confidence:.4f}
- consistency_score = {consistency_score:.4f}

EVIDENCE ĐÃ CHỌN:
{evidence_context}

CAUSAL PATH HỖ TRỢ:
{path_context}

YÊU CẦU TRẢ LỜI:
- Trả lời trực tiếp câu hỏi.
- Dùng đúng citation [E1], [E2], ... tương ứng với evidence.
- Không dùng citation không tồn tại.
- Nếu có nhiều điều kiện hoặc hệ quả, trình bày theo đúng thứ tự logic.
- Nếu evidence mâu thuẫn hoặc chưa chắc chắn, nêu rõ giới hạn.
- Không nhắc tới tên file, JSON, pipeline hoặc điểm số kỹ thuật trừ phần “Lưu ý về độ chắc chắn”.
"""

        return (
            self.SYSTEM_PROMPT,
            truncate_text(
                user_prompt,
                max_context_chars,
            ),
        )

    @staticmethod
    def _build_evidence_context(
        evidence: list[FinalEvidence],
    ) -> str:
        if not evidence:
            return (
                "Không có evidence pháp lý đạt ngưỡng."
            )

        blocks: list[str] = []

        for item in evidence:
            citation = (
                f"[E{item.evidence_index}]"
            )
            article_label = (
                normalize_article_label(
                    item.article_id,
                    item.article_title,
                )
            )

            block = f"""{citation}
- Rule ID: {item.rule_id}
- Căn cứ: {article_label}
- Chủ thể pháp lý: {item.legal_subject or "Không nêu rõ"}
- Điều kiện: {item.condition or "Không nêu rõ"}
- Hệ quả pháp lý: {item.effect or "Không nêu rõ"}
- Sự kiện điều kiện: {item.condition_event_name or item.condition_event or "Không nêu rõ"}
- Sự kiện hệ quả: {item.effect_event_name or item.effect_event or "Không nêu rõ"}
- Loại quan hệ: {item.causal_type or "Không nêu rõ"}
- Trạng thái xác minh: {item.decision}
- Verification score: {item.verification_score:.4f}
"""

            if item.reasons:
                block += (
                    "- Ghi chú xác minh: "
                    + " | ".join(item.reasons)
                    + "\n"
                )

            blocks.append(block.rstrip())

        return "\n\n".join(blocks)

    @staticmethod
    def _build_path_context(
        paths: list[FinalPath],
    ) -> str:
        if not paths:
            return (
                "Không có causal path phù hợp."
            )

        blocks: list[str] = []

        for item in paths:
            path_label = (
                f"[P{item.path_index}]"
            )

            chain = (
                " → ".join(item.event_names)
                if item.event_names
                else (
                    f"{item.seed_event_name or item.seed_event_id}"
                    " → "
                    f"{item.outcome_event_name or item.outcome_event_id}"
                )
            )

            blocks.append(
                f"""{path_label}
- Trạng thái: {item.status}
- Consistency score: {item.consistency_score:.4f}
- Chuỗi sự kiện: {chain}
- Rule IDs trên path: {", ".join(item.rule_ids) or "Không nêu rõ"}
- Giải thích xác minh: {item.explanation or "Không có"}
""".rstrip()
            )

        return "\n\n".join(blocks)


# ============================================================
# LLM PROVIDERS
# ============================================================

class BaseLLMProvider:
    provider_name = "base"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = _http_post_json(
            url=url,
            payload=payload,
            timeout=timeout,
        )

        message = response.get("message", {})
        answer = safe_string(
            message.get("content")
        )

        if not answer:
            raise RuntimeError(
                "Ollama không trả về nội dung."
            )

        return answer


class OpenAICompatibleProvider(
    BaseLLMProvider
):
    provider_name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Thiếu OPENAI_API_KEY hoặc --api-key."
            )

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
        }

        response = _http_post_json(
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
        )

        choices = response.get("choices", [])

        if not choices:
            raise RuntimeError(
                "OpenAI-compatible API không trả choices."
            )

        answer = safe_string(
            choices[0]
            .get("message", {})
            .get("content")
        )

        if not answer:
            raise RuntimeError(
                "OpenAI-compatible API trả nội dung rỗng."
            )

        return answer


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
    ) -> None:
        self.api_key = api_key

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "Thiếu GEMINI_API_KEY hoặc --api-key."
            )

        encoded_model = model.strip()
        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{encoded_model}:generateContent"
            f"?key={self.api_key}"
        )

        payload = {
            "system_instruction": {
                "parts": [
                    {
                        "text": system_prompt,
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_prompt,
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        response = _http_post_json(
            url=url,
            payload=payload,
            timeout=timeout,
        )

        candidates = response.get(
            "candidates",
            [],
        )

        if not candidates:
            raise RuntimeError(
                "Gemini không trả candidates."
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        answer = "\n".join(
            safe_string(part.get("text"))
            for part in parts
            if safe_string(part.get("text"))
        )

        if not answer:
            raise RuntimeError(
                "Gemini trả nội dung rỗng."
            )

        return answer


class ExtractiveFallbackProvider(
    BaseLLMProvider
):
    provider_name = "extractive"

    def __init__(
        self,
        *,
        evidence: list[FinalEvidence],
        confidence: float,
    ) -> None:
        self.evidence = evidence
        self.confidence = confidence

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> str:
        del (
            system_prompt,
            user_prompt,
            model,
            temperature,
            max_tokens,
            timeout,
        )

        if not self.evidence:
            return (
                "### Kết luận\n"
                "Chưa đủ căn cứ từ dữ liệu được cung cấp "
                "để trả lời câu hỏi.\n\n"
                "### Lưu ý về độ chắc chắn\n"
                "Hệ thống không tìm được evidence đã xác minh."
            )

        lines = [
            "### Kết luận",
        ]

        first = self.evidence[0]

        if first.condition and first.effect:
            lines.append(
                f"Khi {first.condition}, hệ quả pháp lý là "
                f"{first.effect} [E1]."
            )
        elif first.effect:
            lines.append(
                f"Hệ quả pháp lý được ghi nhận là "
                f"{first.effect} [E1]."
            )
        else:
            lines.append(
                "Chưa đủ căn cứ từ dữ liệu được cung cấp "
                "để đưa ra kết luận đầy đủ."
            )

        lines.extend(
            [
                "",
                "### Căn cứ và lập luận",
            ]
        )

        for item in self.evidence:
            citation = (
                f"[E{item.evidence_index}]"
            )
            article = normalize_article_label(
                item.article_id,
                item.article_title,
            )

            sentence = (
                f"{citation} {article}: "
                f"điều kiện “{item.condition or 'không nêu rõ'}” "
                f"dẫn tới hệ quả “{item.effect or 'không nêu rõ'}”."
            )
            lines.append(sentence)

        lines.extend(
            [
                "",
                "### Lưu ý về độ chắc chắn",
                (
                    "Câu trả lời được tổng hợp trực tiếp từ "
                    "evidence đã truy hồi và xác minh. "
                    f"Độ tin cậy toàn cục: {self.confidence:.2f}."
                ),
                (
                    "Nội dung này không thay thế tư vấn pháp lý "
                    "chính thức."
                ),
            ]
        )

        return "\n".join(lines)


def _http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: Optional[
        dict[str, str]
    ] = None,
) -> dict[str, Any]:
    request_headers = {
        "Content-Type": "application/json",
    }

    if headers:
        request_headers.update(headers)

    data = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    http_request = request.Request(
        url=url,
        data=data,
        headers=request_headers,
        method="POST",
    )

    try:
        with request.urlopen(
            http_request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8"
            )
    except error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"HTTP {exc.code} từ {url}: {body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Không kết nối được tới {url}: "
            f"{exc.reason}"
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "API trả response không phải JSON hợp lệ."
        ) from exc


# ============================================================
# ANSWER VALIDATION
# ============================================================

class FinalAnswerValidator:
    CITATION_PATTERN = re.compile(
        r"\[E(\d+)\]"
    )

    def validate_and_repair(
        self,
        *,
        answer: str,
        evidence: list[FinalEvidence],
    ) -> tuple[str, list[str], list[str]]:
        allowed_ids = {
            item.evidence_index
            for item in evidence
        }

        cited_ids = [
            safe_int(match)
            for match in self.CITATION_PATTERN.findall(
                answer
            )
        ]

        invalid_ids = sorted(
            {
                citation_id
                for citation_id in cited_ids
                if citation_id not in allowed_ids
            }
        )

        warnings: list[str] = []

        repaired = answer

        if invalid_ids:
            for citation_id in invalid_ids:
                repaired = repaired.replace(
                    f"[E{citation_id}]",
                    "",
                )

            warnings.append(
                "Đã loại citation không tồn tại: "
                + ", ".join(
                    f"E{citation_id}"
                    for citation_id in invalid_ids
                )
            )

        used_ids = sorted(
            {
                safe_int(match)
                for match in (
                    self.CITATION_PATTERN.findall(
                        repaired
                    )
                )
                if safe_int(match) in allowed_ids
            }
        )

        if evidence and not used_ids:
            warnings.append(
                "LLM không tạo citation; đã thêm danh sách "
                "căn cứ evidence ở cuối câu trả lời."
            )

            citations = ", ".join(
                f"[E{item.evidence_index}]"
                for item in evidence
            )

            repaired = (
                repaired.rstrip()
                + "\n\n"
                + f"Căn cứ evidence: {citations}."
            )

            used_ids = [
                item.evidence_index
                for item in evidence
            ]

        if (
            "không thay thế tư vấn pháp lý"
            not in repaired.lower()
        ):
            repaired = (
                repaired.rstrip()
                + "\n\n"
                + "Lưu ý: Nội dung này không thay thế "
                "tư vấn pháp lý chính thức."
            )

        citations_used = [
            f"E{citation_id}"
            for citation_id in used_ids
        ]

        return (
            repaired.strip(),
            citations_used,
            warnings,
        )


# ============================================================
# PIPELINE
# ============================================================

class FinalAnswerPipeline:
    def __init__(
        self,
        store: FinalAnswerInputStore,
    ) -> None:
        self.store = store
        self.selector = FinalContextSelector(
            store
        )
        self.prompt_builder = (
            LegalAnswerPromptBuilder()
        )
        self.validator = FinalAnswerValidator()

    def run(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        max_evidence: int,
        max_paths: int,
        max_context_chars: int,
        max_tokens: int,
        temperature: float,
        timeout: int,
        min_verification_score: float,
        include_uncertain: bool,
        fallback_to_extractive: bool,
    ) -> GeneratedAnswer:
        selected_evidence = (
            self.selector.select_evidence(
                max_evidence=max_evidence,
                min_verification_score=(
                    min_verification_score
                ),
                include_uncertain=(
                    include_uncertain
                ),
            )
        )

        selected_paths = (
            self.selector.select_paths(
                selected_evidence=(
                    selected_evidence
                ),
                max_paths=max_paths,
            )
        )

        (
            system_prompt,
            user_prompt,
        ) = self.prompt_builder.build(
            query=self.store.query,
            evidence=selected_evidence,
            paths=selected_paths,
            global_confidence=(
                self.store.confidence
            ),
            consistency_score=(
                self.store.consistency_score
            ),
            max_context_chars=(
                max_context_chars
            ),
        )

        provider = self._create_provider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
            evidence=selected_evidence,
        )

        started_at = time.time()
        generation_error = ""

        try:
            raw_answer = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            actual_provider = (
                provider.provider_name
            )
        except Exception as exc:
            if not fallback_to_extractive:
                raise

            generation_error = str(exc)

            print(
                "Warning: LLM generation failed. "
                "Đang dùng extractive fallback."
            )
            print("Reason:", generation_error)

            fallback = ExtractiveFallbackProvider(
                evidence=selected_evidence,
                confidence=self.store.confidence,
            )

            raw_answer = fallback.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model="extractive",
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            actual_provider = (
                fallback.provider_name
            )

        elapsed = time.time() - started_at

        (
            final_answer,
            citations_used,
            validation_warnings,
        ) = self.validator.validate_and_repair(
            answer=raw_answer,
            evidence=selected_evidence,
        )

        return GeneratedAnswer(
            query=self.store.query,
            answer=final_answer,
            provider=actual_provider,
            model=(
                model
                if actual_provider
                != "extractive"
                else "extractive"
            ),
            selected_evidence=[
                asdict(item)
                for item in selected_evidence
            ],
            selected_paths=[
                asdict(item)
                for item in selected_paths
            ],
            citations_used=citations_used,
            confidence=self.store.confidence,
            consistency_score=(
                self.store.consistency_score
            ),
            generation_metadata={
                "requested_provider": (
                    provider_name
                ),
                "requested_model": model,
                "elapsed_seconds": round(
                    elapsed,
                    4,
                ),
                "max_evidence": max_evidence,
                "max_paths": max_paths,
                "max_context_chars": (
                    max_context_chars
                ),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "min_verification_score": (
                    min_verification_score
                ),
                "include_uncertain": (
                    include_uncertain
                ),
                "fallback_to_extractive": (
                    fallback_to_extractive
                ),
                "generation_error": (
                    generation_error
                ),
                "validation_warnings": (
                    validation_warnings
                ),
                "system_prompt_chars": len(
                    system_prompt
                ),
                "user_prompt_chars": len(
                    user_prompt
                ),
            },
        )

    @staticmethod
    def _create_provider(
        *,
        provider_name: str,
        api_key: str,
        base_url: str,
        evidence: list[FinalEvidence],
    ) -> BaseLLMProvider:
        provider_name = provider_name.lower()

        if provider_name == "ollama":
            return OllamaProvider(
                base_url=(
                    base_url
                    or DEFAULT_OLLAMA_URL
                )
            )

        if provider_name == "openai":
            return OpenAICompatibleProvider(
                base_url=(
                    base_url
                    or DEFAULT_OPENAI_BASE_URL
                ),
                api_key=(
                    api_key
                    or os.getenv(
                        "OPENAI_API_KEY",
                        "",
                    )
                ),
            )

        if provider_name == "gemini":
            return GeminiProvider(
                api_key=(
                    api_key
                    or os.getenv(
                        "GEMINI_API_KEY",
                        ""
                    )
                )
            )

        if provider_name == "extractive":
            return ExtractiveFallbackProvider(
                evidence=evidence,
                confidence=0.0,
            )

        raise ValueError(
            "Provider không hợp lệ. "
            "Chọn: ollama, openai, gemini, extractive."
        )


# ============================================================
# DISPLAY
# ============================================================

def print_summary(
    result: GeneratedAnswer,
) -> None:
    print("\n" + "=" * 76)
    print("FINAL LEGAL ANSWER")
    print("=" * 76)

    print("Query:", result.query)
    print(
        "Provider:",
        result.provider,
        "| Model:",
        result.model,
    )
    print(
        "Confidence:",
        f"{result.confidence:.4f}",
        "| Consistency:",
        f"{result.consistency_score:.4f}",
    )

    print("\n" + result.answer)

    print("\nSelected evidence:")
    for item in result.selected_evidence:
        print(
            f"- [E{item['evidence_index']}] "
            f"Rule {item['rule_id']} | "
            f"Điều {item['article_id']} | "
            f"verification="
            f"{item['verification_score']:.4f}"
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final grounded legal answer from "
            "counterfactually verified evidence."
        )
    )

    parser.add_argument(
        "--verification-result",
        default=VERIFICATION_RESULT_PATH,
    )
    parser.add_argument(
        "--retrieval-result",
        default=RETRIEVAL_RESULT_PATH,
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
    )

    parser.add_argument(
        "--provider",
        choices=[
            "ollama",
            "openai",
            "gemini",
            "extractive",
        ],
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--api-key",
        default="",
    )
    parser.add_argument(
        "--base-url",
        default="",
    )

    parser.add_argument(
        "--max-evidence",
        type=int,
        default=DEFAULT_MAX_EVIDENCE,
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=DEFAULT_MAX_PATHS,
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "--min-verification-score",
        type=float,
        default=DEFAULT_MIN_VERIFICATION_SCORE,
    )

    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help=(
            "Cho phép dùng evidence UNCERTAIN nếu đạt "
            "min-verification-score."
        ),
    )
    parser.add_argument(
        "--no-extractive-fallback",
        action="store_true",
        help=(
            "Không fallback sang câu trả lời extractive "
            "khi LLM bị lỗi."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.max_evidence < 1:
        raise ValueError(
            "--max-evidence phải lớn hơn 0."
        )

    if args.max_paths < 0:
        raise ValueError(
            "--max-paths không được âm."
        )

    if args.max_tokens < 1:
        raise ValueError(
            "--max-tokens phải lớn hơn 0."
        )

    if not (
        0.0
        <= args.min_verification_score
        <= 1.0
    ):
        raise ValueError(
            "--min-verification-score phải "
            "nằm trong [0, 1]."
        )

    model = args.model

    if (
        args.provider == "openai"
        and model == DEFAULT_MODEL
    ):
        model = DEFAULT_OPENAI_MODEL

    if (
        args.provider == "gemini"
        and model == DEFAULT_MODEL
    ):
        model = DEFAULT_GEMINI_MODEL

    base_url = args.base_url

    if (
        args.provider == "ollama"
        and not base_url
    ):
        base_url = DEFAULT_OLLAMA_URL

    if (
        args.provider == "openai"
        and not base_url
    ):
        base_url = DEFAULT_OPENAI_BASE_URL

    store = FinalAnswerInputStore(
        verification_result_path=(
            args.verification_result
        ),
        retrieval_result_path=(
            args.retrieval_result
        ),
    )

    pipeline = FinalAnswerPipeline(store)

    result = pipeline.run(
        provider_name=args.provider,
        model=model,
        api_key=args.api_key,
        base_url=base_url,
        max_evidence=args.max_evidence,
        max_paths=args.max_paths,
        max_context_chars=(
            args.max_context_chars
        ),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        min_verification_score=(
            args.min_verification_score
        ),
        include_uncertain=(
            args.include_uncertain
        ),
        fallback_to_extractive=(
            not args.no_extractive_fallback
        ),
    )

    print_summary(result)

    save_json(
        asdict(result),
        args.output,
    )


if __name__ == "__main__":
    main()
