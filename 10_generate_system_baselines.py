#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate controlled system-level baselines for the BLHS benchmark.

The four modes differ only in how context and an optional authoritative
verification decision are prepared. Every mode uses the same Ollama request,
system prompt, JSON schema, parser, retry policy, context budget and decoding
settings.

Modes
-----
llm_only
    Question -> qwen3:8b. No retrieval, graph or verifier.
vanilla_rag
    Question -> BGE-M3 raw-article retrieval -> qwen3:8b.
causal_path_rag
    Question -> Step-3 causal path/rule retrieval -> qwen3:8b. Step 4 is not
    imported or called in this mode.
full_legal_scm
    Question -> Step 3 -> query-aware Step 4 with LegalSCM -> qwen3:8b.
    Step-4 decisions are authoritative. Silent structural fallbacks are
    converted to UNCERTAIN unless --allow-scm-fallback is explicitly set.

This runner never reads gold/evaluation fields while generating predictions
and never falls back to an extractive answer generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib import error, request


RUNNER_VERSION = "1.0-system-level-common-generation"
GENERATION_CONTRACT_VERSION = "1.0-common-ollama-json"
PROMPT_VERSION = "1.0-legal-system-comparison"

MODES = (
    "llm_only",
    "vanilla_rag",
    "causal_path_rag",
    "full_legal_scm",
)

METHOD_NAMES = {
    "llm_only": "LLM_ONLY",
    "vanilla_rag": "VANILLA_RAG",
    "causal_path_rag": "CAUSAL_PATH_RAG",
    "full_legal_scm": "FULL_LEGAL_SCM",
}

SUPPORTED = "SUPPORTED"
REJECT_DIRECT_CLAIM = "REJECT_DIRECT_CLAIM"
UNCERTAIN = "UNCERTAIN"

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BENCHMARK = "data/blhs_multihop_benchmark_250.json"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1200
DEFAULT_NUM_CTX = 32768
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 1
DEFAULT_SEED = 42
DEFAULT_MAX_CONTEXT_CHARS = 18000
DEFAULT_MAX_EVIDENCE = 8

DEFAULT_CORPUS = "data/1_raw_data.json"
DEFAULT_ARTICLE_INDEX = "data/baselines/vanilla_rag_articles.index"
DEFAULT_ARTICLE_INDEX_META = "data/baselines/vanilla_rag_articles_index_meta.json"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_GRAPH = "data/legal_causal_knowledge_graph.graphml"
DEFAULT_MEMORY = "data/causal_memory.csv"
DEFAULT_CAUSAL_INDEX = "data/causal_memory.index"
DEFAULT_EMBEDDINGS = "data/causal_memory_embeddings.npy"
DEFAULT_RULES = "data/blhs_rules_final_all_normalized.json"

DEFAULT_RETRIEVER_SCRIPT = "3_multi_hop_causal_retriever.py"
DEFAULT_VERIFIER_SCRIPT = "4_counterfactual_verification.py"
DEFAULT_VANILLA_SCRIPT = "9_generate_vanilla_rag_predictions.py"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    question_type: str = ""


@dataclass
class PreparedContext:
    mode: str
    policy: str
    context_blocks: list[str] = field(default_factory=list)
    allowed_article_ids: list[str] = field(default_factory=list)
    retrieved_rule_ids: list[str] = field(default_factory=list)
    retrieved_event_ids: list[str] = field(default_factory=list)
    retrieved_article_ids: list[str] = field(default_factory=list)
    reasoning_path: Optional[list[dict[str, Any]]] = None
    reasoning_path_id: int = -1
    primary_path_ids: list[int] = field(default_factory=list)
    retrieval_payload: dict[str, Any] = field(default_factory=dict)
    verification_payload: dict[str, Any] = field(default_factory=dict)
    authoritative_decision: str = ""
    authoritative_score: Optional[float] = None
    preparation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedGeneration:
    decision: str
    decision_score: float
    answer: str
    citations: list[str]
    raw_response: str
    response_metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(value: Any, default: float = 0.5) -> float:
    number = safe_float(value, default)
    assert number is not None
    return max(0.0, min(1.0, number))


def normalize_id(value: Any) -> str:
    text = safe_string(value)
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def normalize_event_id(value: Any) -> str:
    text = normalize_id(value)
    return text[len("EVENT::"):] if text.upper().startswith("EVENT::") else text


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)
    match = re.search(
        r"(?:điều|dieu)\s*([0-9]+(?:\.[0-9]+)?[A-Za-z]?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_id(match.group(1))
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[A-Za-z]?", text):
        return normalize_id(text)
    return ""


def unique(values: Iterable[Any], normalizer=safe_string) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalizer(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def unique_ints(values: Iterable[Any]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = safe_int(value, -1)
        if number >= 0 and number not in seen:
            seen.add(number)
            output.append(number)
    return output


def normalize_decision(value: Any) -> str:
    key = re.sub(r"[^A-Z0-9]+", "_", safe_string(value).upper()).strip("_")
    aliases = {
        "SUPPORTED": SUPPORTED,
        "SUPPORT": SUPPORTED,
        "YES": SUPPORTED,
        "TRUE": SUPPORTED,
        "CORRECT": SUPPORTED,
        "ENTAILED": SUPPORTED,
        "REJECT_DIRECT_CLAIM": REJECT_DIRECT_CLAIM,
        "REJECT": REJECT_DIRECT_CLAIM,
        "NO": REJECT_DIRECT_CLAIM,
        "FALSE": REJECT_DIRECT_CLAIM,
        "CONTRADICTED": REJECT_DIRECT_CLAIM,
        "NOT_SUPPORTED": REJECT_DIRECT_CLAIM,
        "REFUTED": REJECT_DIRECT_CLAIM,
        "UNCERTAIN": UNCERTAIN,
        "UNKNOWN": UNCERTAIN,
        "UNRESOLVED": UNCERTAIN,
        "INCONCLUSIVE": UNCERTAIN,
    }
    return aliases.get(key, UNCERTAIN)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_serializable(dict(payload)), ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def rewrite_jsonl(
    path: Path,
    predictions: Mapping[str, Mapping[str, Any]],
    sample_order: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample_id in sample_order:
            row = predictions.get(sample_id)
            if row is not None:
                file.write(json.dumps(to_serializable(dict(row)), ensure_ascii=False) + "\n")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_module(name: str, script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(f"Không tìm thấy source module: {script_path}")
    module_name = f"{name}_{hashlib.sha1(str(script_path).encode()).hexdigest()[:10]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể import source module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sanitize_filename(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", safe_string(value)).strip("._")
    return text or "sample"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def model_slug(model: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", model).strip("_").lower()
    return slug or "model"


# ---------------------------------------------------------------------------
# Benchmark loading: intentionally only expose id/question/question_type
# ---------------------------------------------------------------------------

def load_samples(
    benchmark_path: Path,
    *,
    start_index: int,
    limit: Optional[int],
) -> tuple[dict[str, Any], list[Sample], list[str]]:
    payload = read_json(benchmark_path)
    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        rows = payload
    elif isinstance(payload, Mapping):
        metadata = dict(payload.get("metadata") or {})
        rows = payload.get("questions") or payload.get("samples") or payload.get("data")
    else:
        raise ValueError("Benchmark phải là JSON list hoặc object.")
    if not isinstance(rows, list):
        raise ValueError("Benchmark thiếu questions/samples/data.")

    all_samples: list[Sample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        sample_id = normalize_id(
            row.get("id") or row.get("sample_id") or row.get("question_id")
            or f"question_{index + 1:04d}"
        )
        question = safe_string(row.get("question") or row.get("query"))
        if sample_id and question:
            all_samples.append(
                Sample(
                    sample_id=sample_id,
                    question=question,
                    question_type=safe_string(row.get("question_type")),
                )
            )

    if not all_samples:
        raise ValueError("Benchmark không có câu hỏi hợp lệ.")
    if start_index < 0:
        raise ValueError("--start-index không được âm.")
    selected = all_samples[start_index:]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit phải lớn hơn 0.")
        selected = selected[:limit]
    if not selected:
        raise ValueError("Không có sample nào trong phạm vi đã chọn.")
    return metadata, selected, [sample.sample_id for sample in all_samples]


# ---------------------------------------------------------------------------
# Common prompt and structured-output parser
# ---------------------------------------------------------------------------

COMMON_SYSTEM_PROMPT = """Bạn là hệ thống hỏi đáp pháp luật Việt Nam trong một thí nghiệm so sánh có kiểm soát.

Mọi cấu hình đều phải tuân theo cùng các quy tắc sau:
1. Đọc CONTEXT_POLICY trong yêu cầu. Với CLOSED_BOOK_INTERNAL_KNOWLEDGE, chỉ dùng tri thức nội tại của mô hình. Với EVIDENCE_ONLY, chỉ dùng evidence được cung cấp. Với VERIFIED_EVIDENCE_AND_AUTHORITATIVE_DECISION, chỉ dùng evidence và kết quả xác minh được cung cấp.
2. Nếu AUTHORITATIVE_DECISION khác NONE, verification_decision phải sao chép chính xác giá trị đó và câu trả lời không được mâu thuẫn với nó.
3. Nếu căn cứ không đủ, trả UNCERTAIN; không bịa quy định, causal path, số điều luật hoặc kết quả can thiệp.
4. Với cấu hình có evidence, citations chỉ được lấy từ AVAILABLE_CITATIONS. Với cấu hình closed-book, chỉ nêu số điều nếu thực sự chắc chắn.
5. Không tạo rule_id, event_id hoặc citation dạng [E1] trong final_answer.
6. Trả lời ngắn gọn, trực tiếp bằng tiếng Việt.
7. Chỉ trả về một JSON object hợp lệ, không markdown và không văn bản ngoài JSON.

Schema bắt buộc:
{
  "verification_decision": "SUPPORTED | REJECT_DIRECT_CLAIM | UNCERTAIN",
  "decision_score": 0.0,
  "final_answer": "câu trả lời ngắn gọn",
  "citations": ["Điều 15", "Điều 57"]
}
""".strip()


class CommonPromptBuilder:
    def build(
        self,
        *,
        sample: Sample,
        prepared: PreparedContext,
        max_context_chars: int,
    ) -> tuple[str, str, dict[str, Any]]:
        packed_context, included_blocks, truncated = self._pack_blocks(
            prepared.context_blocks,
            max_context_chars,
        )
        authoritative = prepared.authoritative_decision or "NONE"
        authoritative_score = (
            f"{prepared.authoritative_score:.6f}"
            if prepared.authoritative_score is not None
            else "NONE"
        )
        available_citations = (
            ", ".join(f"Điều {item}" for item in prepared.allowed_article_ids)
            if prepared.allowed_article_ids
            else (
                "MODEL_MEMORY_ONLY"
                if prepared.mode == "llm_only"
                else "NONE"
            )
        )
        context_value = packed_context or "Không có context truy xuất."
        user_prompt = f"""CONTEXT_POLICY: {prepared.policy}
AUTHORITATIVE_DECISION: {authoritative}
AUTHORITATIVE_DECISION_SCORE: {authoritative_score}
AVAILABLE_CITATIONS: {available_citations}

CÂU HỎI:
{sample.question}

CONTEXT:
{context_value}

Hãy trả về đúng JSON schema đã quy định."""
        metadata = {
            "prompt_version": PROMPT_VERSION,
            "system_prompt_sha256": sha256_text(COMMON_SYSTEM_PROMPT),
            "system_prompt_chars": len(COMMON_SYSTEM_PROMPT),
            "user_prompt_chars": len(user_prompt),
            "context_chars": len(packed_context),
            "context_block_count": len(prepared.context_blocks),
            "included_context_block_count": included_blocks,
            "context_truncated": truncated,
            "max_context_chars": max_context_chars,
        }
        return COMMON_SYSTEM_PROMPT, user_prompt, metadata

    @staticmethod
    def _pack_blocks(blocks: Sequence[str], max_chars: int) -> tuple[str, int, bool]:
        if max_chars < 1:
            return "", 0, bool(blocks)
        output: list[str] = []
        remaining = max_chars
        truncated = False
        for block in blocks:
            text = safe_string(block)
            if not text:
                continue
            separator_cost = 2 if output else 0
            if remaining <= separator_cost:
                truncated = True
                break
            available = remaining - separator_cost
            if len(text) <= available:
                output.append(text)
                remaining -= len(text) + separator_cost
                continue
            marker = "\n...[truncated]"
            if available > len(marker):
                output.append(text[: available - len(marker)].rstrip() + marker)
            truncated = True
            break
        return "\n\n".join(output), len(output), truncated


def extract_json_object(text: str) -> dict[str, Any]:
    value = safe_string(text)
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    if start < 0:
        raise ValueError("Ollama response không chứa JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(value[start:index + 1])
                if isinstance(payload, dict):
                    return payload
    raise ValueError("Không parse được JSON object từ Ollama response.")


def normalize_citations(value: Any, allowed_ids: Optional[Sequence[str]]) -> list[str]:
    raw_values = value if isinstance(value, list) else ([] if value is None else [value])
    article_ids: list[str] = []
    for raw in raw_values:
        if isinstance(raw, Mapping):
            raw = (
                raw.get("article_id")
                or raw.get("article")
                or raw.get("citation")
                or raw.get("label")
            )
        article_id = normalize_article_id(raw)
        if article_id:
            article_ids.append(article_id)
    article_ids = unique(article_ids, normalizer=normalize_id)
    if allowed_ids is not None:
        allowed = set(unique(allowed_ids, normalizer=normalize_id))
        article_ids = [item for item in article_ids if item in allowed]
    return [f"Điều {item}" for item in article_ids]


def parse_generation(
    raw_response: str,
    *,
    allowed_article_ids: Optional[Sequence[str]],
    response_metadata: Mapping[str, Any],
) -> ParsedGeneration:
    payload = extract_json_object(raw_response)
    answer = safe_string(
        payload.get("final_answer") or payload.get("answer") or payload.get("response")
    )
    if not answer:
        raise ValueError("Ollama JSON thiếu final_answer.")
    return ParsedGeneration(
        decision=normalize_decision(
            payload.get("verification_decision")
            or payload.get("final_decision")
            or payload.get("decision")
        ),
        decision_score=clamp(
            payload.get("decision_score", payload.get("confidence", 0.5)),
            0.5,
        ),
        answer=answer,
        citations=normalize_citations(payload.get("citations"), allowed_article_ids),
        raw_response=raw_response,
        response_metadata=dict(response_metadata),
    )


# ---------------------------------------------------------------------------
# Common Ollama generator
# ---------------------------------------------------------------------------

class CommonOllamaJSONGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        timeout: int,
        retries: int,
        seed: int,
        thinking: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = retries
        self.seed = seed
        self.thinking = thinking

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allowed_article_ids: Optional[Sequence[str]],
    ) -> ParsedGeneration:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx,
                        "seed": self.seed,
                    },
                }
                if self.thinking != "server_default":
                    payload["think"] = self.thinking == "enabled"
                response = self._post_json(
                    f"{self.base_url}/api/chat",
                    payload,
                )
                raw = safe_string((response.get("message") or {}).get("content"))
                if not raw:
                    raise RuntimeError("Ollama trả content rỗng.")
                response_metadata = {
                    key: response.get(key)
                    for key in (
                        "model",
                        "created_at",
                        "done",
                        "done_reason",
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    )
                    if response.get(key) is not None
                }
                response_metadata["attempt"] = attempt + 1
                return parse_generation(
                    raw,
                    allowed_article_ids=allowed_article_ids,
                    response_metadata=response_metadata,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 5))
        assert last_error is not None
        raise last_error

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        http_request = request.Request(
            url=url,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Không kết nối được Ollama: {exc.reason}") from exc
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError("Ollama response không phải JSON object.")
        return result


# ---------------------------------------------------------------------------
# Context construction helpers
# ---------------------------------------------------------------------------

def normalize_path_steps(path: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(path.get("steps") or [], start=1):
        if not isinstance(raw, Mapping):
            continue
        source_id = normalize_event_id(
            raw.get("source_event_id") or raw.get("source_event_node")
        )
        target_id = normalize_event_id(
            raw.get("target_event_id") or raw.get("target_event_node")
        )
        if not source_id or not target_id:
            continue
        rule_ids = unique(
            list(raw.get("rule_ids") or [])
            + ([raw.get("rule_id")] if raw.get("rule_id") is not None else [])
        )
        article_ids = unique(
            list(raw.get("article_ids") or [])
            + ([raw.get("article_id")] if raw.get("article_id") is not None else []),
            normalizer=normalize_id,
        )
        output.append({
            "hop": safe_int(raw.get("hop"), index),
            "source_event_id": source_id,
            "source_event_name": safe_string(raw.get("source_event_name")),
            "target_event_id": target_id,
            "target_event_name": safe_string(raw.get("target_event_name")),
            "rule_ids": rule_ids,
            "article_ids": article_ids,
        })
    output.sort(key=lambda item: safe_int(item.get("hop"), 0))
    for index, item in enumerate(output, start=1):
        item["hop"] = index
    return output


def path_event_ids(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    if not steps:
        return []
    return unique(
        [steps[0].get("source_event_id")]
        + [step.get("target_event_id") for step in steps],
        normalizer=normalize_event_id,
    )


def path_rule_ids(path: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[str]:
    return unique(
        [rule_id for step in steps for rule_id in step.get("rule_ids") or []]
        + list(path.get("rule_ids") or [])
    )


def path_article_ids(path: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> list[str]:
    return unique(
        [article_id for step in steps for article_id in step.get("article_ids") or []]
        + list(path.get("article_ids") or []),
        normalizer=normalize_id,
    )


def select_primary_path(
    retrieval: Mapping[str, Any],
    verification: Optional[Mapping[str, Any]],
) -> tuple[int, dict[str, Any], list[int]]:
    paths = retrieval.get("causal_paths") or []
    if not isinstance(paths, list):
        paths = []
    primary_ids = unique_ints((verification or {}).get("primary_path_ids") or [])
    path_id = primary_ids[0] if primary_ids else (0 if paths else -1)
    if not (0 <= path_id < len(paths)):
        path_id = 0 if paths else -1
    path = paths[path_id] if path_id >= 0 and isinstance(paths[path_id], Mapping) else {}
    if path_id >= 0 and not primary_ids:
        primary_ids = [path_id]
    return path_id, dict(path), primary_ids


def unwrap_verified_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    original = item.get("original_evidence")
    merged = dict(original) if isinstance(original, Mapping) else {}
    for key in (
        "rule_id",
        "article_id",
        "verification_score",
        "decision",
        "verified_path_ids",
        "unresolved_path_ids",
        "rejected_path_ids",
    ):
        if item.get(key) is not None:
            merged[key] = item.get(key)
    return merged


def select_causal_evidence(
    *,
    retrieval: Mapping[str, Any],
    primary_path: Mapping[str, Any],
    verification: Optional[Mapping[str, Any]],
    max_evidence: int,
    include_uncertain: bool,
) -> list[dict[str, Any]]:
    raw_evidence = [
        dict(item)
        for item in retrieval.get("evidence") or []
        if isinstance(item, Mapping)
    ]
    candidates: list[dict[str, Any]] = []
    if verification is not None:
        candidates.extend(
            unwrap_verified_evidence(item)
            for item in verification.get("verified_evidence") or []
            if isinstance(item, Mapping)
        )
        if include_uncertain:
            candidates.extend(
                unwrap_verified_evidence(item)
                for item in verification.get("uncertain_evidence") or []
                if isinstance(item, Mapping)
            )
    if not candidates:
        candidates = raw_evidence

    by_rule: dict[str, dict[str, Any]] = {}
    for item in candidates + raw_evidence:
        rule_id = safe_string(item.get("rule_id"))
        if rule_id and rule_id not in by_rule:
            by_rule[rule_id] = item

    steps = normalize_path_steps(primary_path)
    preferred_rules = path_rule_ids(primary_path, steps)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rule_id in preferred_rules:
        item = by_rule.get(rule_id)
        if item is not None:
            selected.append(dict(item))
            seen.add(rule_id)
    for item in candidates:
        rule_id = safe_string(item.get("rule_id"))
        if rule_id and rule_id in seen:
            continue
        selected.append(dict(item))
        if rule_id:
            seen.add(rule_id)
        if len(selected) >= max_evidence:
            break
    return selected[:max_evidence]


def format_causal_context(
    *,
    primary_path_id: int,
    primary_path: Mapping[str, Any],
    selected_evidence: Sequence[Mapping[str, Any]],
    verification: Optional[Mapping[str, Any]],
    scm_guard: Optional[Mapping[str, Any]],
) -> list[str]:
    blocks: list[str] = []
    if verification is not None:
        analysis = verification.get("query_analysis") or {}
        guard = scm_guard or {}
        blocks.append(
            "[VERIFICATION]\n"
            f"Effective decision: {safe_string(guard.get('effective_decision') or verification.get('final_decision'))}\n"
            f"Claim type: {safe_string(analysis.get('claim_type')) or 'STANDARD_CAUSAL'}\n"
            f"Decision explanation: {safe_string(verification.get('decision_explanation'))}\n"
            f"Mediator: {safe_string(analysis.get('matched_mediator_name') or analysis.get('matched_mediator_id')) or 'N/A'}\n"
            f"Factual outcome: {safe_string(analysis.get('factual_outcome')) or 'N/A'}\n"
            f"Counterfactual outcome: {safe_string(analysis.get('counterfactual_outcome')) or 'N/A'}\n"
            f"Signal source: {safe_string(analysis.get('counterfactual_signal_source') or analysis.get('path_verification_method')) or 'N/A'}\n"
            f"SCM guard status: {safe_string(guard.get('status')) or 'N/A'}"
        )

    steps = normalize_path_steps(primary_path)
    if steps:
        chain_names = [
            safe_string(steps[0].get("source_event_name"))
            or safe_string(steps[0].get("source_event_id"))
        ]
        chain_names.extend(
            safe_string(step.get("target_event_name"))
            or safe_string(step.get("target_event_id"))
            for step in steps
        )
        step_lines = []
        for step in steps:
            step_lines.append(
                f"Hop {step['hop']}: "
                f"{step.get('source_event_name') or step.get('source_event_id')} -> "
                f"{step.get('target_event_name') or step.get('target_event_id')} | "
                f"rules={','.join(step.get('rule_ids') or []) or 'N/A'} | "
                f"articles={','.join(step.get('article_ids') or []) or 'N/A'}"
            )
        blocks.append(
            f"[CAUSAL PATH P{primary_path_id}]\n"
            f"Chain: {' -> '.join(chain_names)}\n"
            + "\n".join(step_lines)
        )

    for index, item in enumerate(selected_evidence, start=1):
        blocks.append(
            f"[EVIDENCE E{index}]\n"
            f"Rule ID: {safe_string(item.get('rule_id')) or 'N/A'}\n"
            f"Căn cứ: Điều {safe_string(item.get('article_id')) or 'N/A'}"
            f"{(' - ' + safe_string(item.get('article_title'))) if safe_string(item.get('article_title')) else ''}\n"
            f"Chủ thể: {safe_string(item.get('legal_subject')) or 'N/A'}\n"
            f"Điều kiện: {safe_string(item.get('condition')) or 'N/A'}\n"
            f"Hệ quả: {safe_string(item.get('effect')) or 'N/A'}\n"
            f"Condition event: {safe_string(item.get('condition_event_name') or item.get('condition_event')) or 'N/A'}\n"
            f"Effect event: {safe_string(item.get('effect_event_name') or item.get('effect_event')) or 'N/A'}"
        )
    return blocks


def detect_scm_guard(
    verification: Mapping[str, Any],
    *,
    allow_fallback: bool,
) -> dict[str, Any]:
    configuration = verification.get("configuration") or {}
    analysis = verification.get("query_analysis") or {}
    primary_ids = unique_ints(verification.get("primary_path_ids") or [])
    legal_scm_loaded = bool(configuration.get("legal_scm_loaded"))
    fallback_used = bool(analysis.get("structural_fallback_used"))
    fallback_reasons: list[str] = []
    if safe_string(analysis.get("structural_fallback_reason")):
        fallback_reasons.append(safe_string(analysis.get("structural_fallback_reason")))

    primary_set = set(primary_ids)
    for item in verification.get("path_verifications") or []:
        if not isinstance(item, Mapping):
            continue
        path_id = safe_int(item.get("original_path_id"), -1)
        if primary_set and path_id not in primary_set:
            continue
        summary = item.get("counterfactual_summary") or {}
        if isinstance(summary, Mapping) and summary.get("fallback_used"):
            fallback_used = True
            reason = safe_string(summary.get("fallback_reason"))
            if reason:
                fallback_reasons.append(reason)

    original_decision = normalize_decision(verification.get("final_decision"))
    original_score = clamp(verification.get("decision_score"), 0.35)
    if not legal_scm_loaded:
        status = "SCM_NOT_LOADED"
        effective_decision = UNCERTAIN
        effective_score = 0.0
    elif not primary_ids:
        status = "NO_PRIMARY_PATH"
        effective_decision = UNCERTAIN
        effective_score = min(original_score, 0.35)
    elif fallback_used and not allow_fallback:
        status = "STRUCTURAL_FALLBACK_BLOCKED"
        effective_decision = UNCERTAIN
        effective_score = min(original_score, 0.35)
    elif fallback_used:
        status = "STRUCTURAL_FALLBACK_ALLOWED"
        effective_decision = original_decision
        effective_score = original_score
    else:
        status = "STRUCTURAL_SCM_VERIFIED"
        effective_decision = original_decision
        effective_score = original_score

    return {
        "status": status,
        "legal_scm_loaded": legal_scm_loaded,
        "primary_path_ids": primary_ids,
        "structural_fallback_used": fallback_used,
        "structural_fallback_reasons": unique(fallback_reasons),
        "allow_scm_fallback": allow_fallback,
        "original_decision": original_decision,
        "original_decision_score": original_score,
        "effective_decision": effective_decision,
        "effective_decision_score": effective_score,
    }


# ---------------------------------------------------------------------------
# Unified mode runner
# ---------------------------------------------------------------------------

class SystemBaselineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.mode
        self.prompt_builder = CommonPromptBuilder()
        self.generator = CommonOllamaJSONGenerator(
            base_url=args.ollama_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
            retries=args.retries,
            seed=args.seed,
            thinking=args.thinking,
        )
        self._vanilla_module: Any = None
        self._article_retriever: Any = None
        self._causal_module: Any = None
        self._causal_retriever: Any = None
        self._verifier_module: Any = None

    def run_one(self, sample: Sample) -> dict[str, Any]:
        started = time.time()
        prepared = self.prepare(sample)
        system_prompt, user_prompt, prompt_metadata = self.prompt_builder.build(
            sample=sample,
            prepared=prepared,
            max_context_chars=self.args.max_context_chars,
        )
        allowed_ids: Optional[Sequence[str]] = (
            None if self.mode == "llm_only" else prepared.allowed_article_ids
        )
        generation = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_article_ids=allowed_ids,
        )
        return self.build_prediction(
            sample=sample,
            prepared=prepared,
            generation=generation,
            prompt_metadata=prompt_metadata,
            elapsed_seconds=time.time() - started,
        )

    def prepare(self, sample: Sample) -> PreparedContext:
        if self.mode == "llm_only":
            return self._prepare_llm_only()
        if self.mode == "vanilla_rag":
            return self._prepare_vanilla_rag(sample)
        if self.mode == "causal_path_rag":
            retrieval = self._run_causal_retrieval(sample.question)
            return self._prepare_causal(sample, retrieval, verification=None)
        if self.mode == "full_legal_scm":
            retrieval = self._run_causal_retrieval(sample.question)
            verification = self._run_structural_verification(sample, retrieval)
            return self._prepare_causal(sample, retrieval, verification=verification)
        raise ValueError(f"Mode không hợp lệ: {self.mode}")

    def _prepare_llm_only(self) -> PreparedContext:
        return PreparedContext(
            mode=self.mode,
            policy="CLOSED_BOOK_INTERNAL_KNOWLEDGE",
            retrieval_payload={
                "retrieval_type": "none",
                "retrieved_rule_ids": [],
                "retrieved_event_ids": [],
                "retrieved_article_ids": [],
                "causal_paths": [],
            },
            verification_payload={
                "verification_method": "none",
                "counterfactual_verification_enabled": False,
            },
            preparation_metadata={
                "retrieval_enabled": False,
                "causal_graph_enabled": False,
                "counterfactual_verification_enabled": False,
            },
        )

    def _ensure_vanilla_retriever(self) -> None:
        if self._article_retriever is not None:
            return
        self._vanilla_module = load_module(
            "causalrag_vanilla",
            Path(self.args.vanilla_script),
        )
        articles = self._vanilla_module.load_articles(self.args.corpus)
        self._article_retriever = self._vanilla_module.DenseRetriever(
            articles,
            self.args.embedding_model,
            self.args.embedding_batch_size,
            self.args.article_index,
            self.args.article_index_meta,
            self.args.rebuild_article_index,
        )

    def _prepare_vanilla_rag(self, sample: Sample) -> PreparedContext:
        self._ensure_vanilla_retriever()
        hits = self._article_retriever.retrieve(sample.question, self.args.vanilla_top_k)
        hit_payloads: list[dict[str, Any]] = []
        blocks: list[str] = []
        per_article_budget = max(
            600,
            self.args.max_context_chars // max(1, len(hits)) - 180,
        )
        for hit in hits:
            article_id = normalize_id(hit.article_id)
            content = safe_string(hit.content)
            if len(content) > per_article_budget:
                content = content[:per_article_budget].rstrip() + "\n...[document truncated]"
            blocks.append(
                f"[EVIDENCE ARTICLE {hit.rank}]\n"
                f"Căn cứ: Điều {article_id}"
                f"{(' - ' + safe_string(hit.title)) if safe_string(hit.title) else ''}\n"
                f"Nội dung: {content}"
            )
            hit_payloads.append({
                "rank": hit.rank,
                "article_id": article_id,
                "article_title": safe_string(hit.title),
                "score": safe_float(hit.score, 0.0),
            })
        article_ids = [item["article_id"] for item in hit_payloads]
        return PreparedContext(
            mode=self.mode,
            policy="EVIDENCE_ONLY",
            context_blocks=blocks,
            allowed_article_ids=article_ids,
            retrieved_article_ids=article_ids,
            retrieval_payload={
                "retrieval_type": "dense_raw_article",
                "retrieval_unit": "article",
                "embedding_model": self.args.embedding_model,
                "top_k": self.args.vanilla_top_k,
                "retrieved_rule_ids": [],
                "retrieved_event_ids": [],
                "retrieved_article_ids": article_ids,
                "retrieved_articles": hit_payloads,
                "causal_paths": [],
            },
            verification_payload={
                "verification_method": "none",
                "counterfactual_verification_enabled": False,
            },
            preparation_metadata={
                "retrieval_enabled": True,
                "retrieval_type": "dense_raw_article",
                "retrieval_unit": "article",
                "structured_rule_retrieval_enabled": False,
                "event_retrieval_enabled": False,
                "causal_graph_enabled": False,
                "causal_path_enabled": False,
                "counterfactual_verification_enabled": False,
            },
        )

    def _ensure_causal_retriever(self) -> None:
        if self._causal_retriever is not None:
            return
        self._causal_module = load_module(
            "causalrag_step3",
            Path(self.args.retriever_script),
        )
        store = self._causal_module.CausalResourceStore(
            graph_path=self.args.graph,
            memory_path=self.args.memory,
            index_path=self.args.causal_index,
            embeddings_path=self.args.embeddings,
            model_name=self.args.retriever_model,
        )
        self._causal_retriever = self._causal_module.MultiHopCausalRetriever(store)

    def _run_causal_retrieval(self, question: str) -> dict[str, Any]:
        self._ensure_causal_retriever()
        result = self._causal_retriever.retrieve(
            question,
            event_top_k=self.args.event_top_k,
            direct_rule_top_k=self.args.direct_rule_top_k,
            semantic_pool_size=self.args.semantic_pool_size,
            max_hops=self.args.max_hops,
            max_paths_per_event=self.args.max_paths_per_event,
            max_candidate_rules=self.args.max_candidate_rules,
            final_top_k=self.args.final_top_k,
            min_event_score=self.args.min_event_score,
            min_rule_score=self.args.min_rule_score,
            direction=self.args.direction,
        )
        return to_serializable(result)

    def _ensure_verifier(self) -> None:
        if self._verifier_module is not None:
            return
        self._verifier_module = load_module(
            "causalrag_step4",
            Path(self.args.verifier_script),
        )
        version = safe_string(getattr(self._verifier_module, "STEP4_VERSION", ""))
        if "query-aware" not in version.lower():
            raise RuntimeError(
                "full_legal_scm yêu cầu Step 4 query-aware; "
                f"nhận STEP4_VERSION={version or 'missing'}."
            )

    def _run_structural_verification(
        self,
        sample: Sample,
        retrieval: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_verifier()
        sample_dir = Path(self.args.work_dir) / sanitize_filename(sample.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        retrieval_path = sample_dir / "retrieval_result.json"
        verification_path = sample_dir / "verification_result.json"
        write_json(retrieval_path, retrieval)
        store = self._verifier_module.CounterfactualResourceStore(
            graph_path=self.args.graph,
            memory_path=self.args.memory,
            retrieval_result_path=str(retrieval_path),
            rules_path=self.args.rules,
            enable_semantic_mapping=False,
        )
        pipeline = self._verifier_module.CounterfactualVerificationPipeline(store)
        result = pipeline.run(
            counterfactual_mode="structural_scm",
            max_cf_hops=self.args.max_cf_hops,
            max_cf_paths=self.args.max_cf_paths,
            verified_top_k=self.args.verified_top_k,
            keep_threshold=self.args.keep_threshold,
            reject_threshold=self.args.reject_threshold,
        )
        payload = to_serializable(result)
        write_json(verification_path, payload)
        return payload

    def _prepare_causal(
        self,
        sample: Sample,
        retrieval: Mapping[str, Any],
        verification: Optional[Mapping[str, Any]],
    ) -> PreparedContext:
        path_id, primary_path, primary_ids = select_primary_path(retrieval, verification)
        steps = normalize_path_steps(primary_path)
        selected_evidence = select_causal_evidence(
            retrieval=retrieval,
            primary_path=primary_path,
            verification=verification,
            max_evidence=self.args.max_evidence,
            include_uncertain=self.args.include_uncertain,
        )
        scm_guard: Optional[dict[str, Any]] = None
        authoritative_decision = ""
        authoritative_score: Optional[float] = None
        verification_payload: dict[str, Any]
        if verification is None:
            verification_payload = {
                "verification_method": "none",
                "counterfactual_verification_enabled": False,
                "final_decision": "",
                "primary_path_ids": primary_ids,
            }
        else:
            scm_guard = detect_scm_guard(
                verification,
                allow_fallback=self.args.allow_scm_fallback,
            )
            authoritative_decision = safe_string(scm_guard["effective_decision"])
            authoritative_score = safe_float(
                scm_guard["effective_decision_score"],
                0.0,
            )
            verification_payload = dict(verification)
            verification_payload["runner_scm_guard"] = scm_guard
            verification_payload["effective_final_decision"] = authoritative_decision
            verification_payload["effective_decision_score"] = authoritative_score

        context_blocks = format_causal_context(
            primary_path_id=path_id,
            primary_path=primary_path,
            selected_evidence=selected_evidence,
            verification=verification,
            scm_guard=scm_guard,
        )
        path_rules = path_rule_ids(primary_path, steps)
        path_articles = path_article_ids(primary_path, steps)
        path_events = path_event_ids(steps)

        raw_events = retrieval.get("retrieved_events") or []
        retrieved_event_ids = unique(
            path_events
            + [
                item.get("event_id") or item.get("graph_node_id")
                for item in raw_events
                if isinstance(item, Mapping)
            ],
            normalizer=normalize_event_id,
        )
        raw_evidence = retrieval.get("evidence") or []
        retrieved_rule_ids = unique(
            path_rules
            + [item.get("rule_id") for item in raw_evidence if isinstance(item, Mapping)]
        )
        retrieved_article_ids = unique(
            path_articles
            + [item.get("article_id") for item in raw_evidence if isinstance(item, Mapping)],
            normalizer=normalize_id,
        )
        allowed_articles = unique(
            [item.get("article_id") for item in selected_evidence]
            + path_articles,
            normalizer=normalize_id,
        )
        retrieval_payload = {
            "retrieval_type": "causal_rule_event_path",
            "retrieval_unit": "rule_event_path",
            "retrieved_events": raw_events,
            "direct_rule_hits": retrieval.get("direct_rule_hits") or [],
            "retrieved_rules": raw_evidence,
            "retrieved_rule_ids": retrieved_rule_ids,
            "retrieved_event_ids": retrieved_event_ids,
            "retrieved_article_ids": retrieved_article_ids,
            "causal_paths": retrieval.get("causal_paths") or [],
            "selected_path_id": path_id,
            "selected_path_event_ids": path_events,
            "selected_path_rule_ids": path_rules,
            "statistics": retrieval.get("statistics") or {},
            "configuration": retrieval.get("configuration") or {},
        }
        full_mode = verification is not None
        return PreparedContext(
            mode=self.mode,
            policy=(
                "VERIFIED_EVIDENCE_AND_AUTHORITATIVE_DECISION"
                if full_mode
                else "EVIDENCE_ONLY"
            ),
            context_blocks=context_blocks,
            allowed_article_ids=allowed_articles,
            retrieved_rule_ids=retrieved_rule_ids,
            retrieved_event_ids=retrieved_event_ids,
            retrieved_article_ids=retrieved_article_ids,
            reasoning_path=steps or None,
            reasoning_path_id=path_id,
            primary_path_ids=primary_ids,
            retrieval_payload=retrieval_payload,
            verification_payload=verification_payload,
            authoritative_decision=authoritative_decision,
            authoritative_score=authoritative_score,
            preparation_metadata={
                "retrieval_enabled": True,
                "retrieval_type": "causal_rule_event_path",
                "retrieval_unit": "rule_event_path",
                "structured_rule_retrieval_enabled": True,
                "event_retrieval_enabled": True,
                "causal_graph_enabled": True,
                "causal_path_enabled": True,
                "counterfactual_verification_enabled": full_mode,
                "legal_scm_enabled": full_mode,
                "step4_bypassed": not full_mode,
                "scm_guard": scm_guard or {},
                "selected_evidence_count": len(selected_evidence),
            },
        )

    def build_prediction(
        self,
        *,
        sample: Sample,
        prepared: PreparedContext,
        generation: ParsedGeneration,
        prompt_metadata: Mapping[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        decision_overridden = bool(
            prepared.authoritative_decision
            and generation.decision != prepared.authoritative_decision
        )
        final_decision = (
            prepared.authoritative_decision or generation.decision
        )
        final_score = (
            prepared.authoritative_score
            if prepared.authoritative_score is not None
            else generation.decision_score
        )
        applicability = metric_applicability(self.mode)
        generation_settings = {
            "provider": "ollama",
            "model": self.args.model,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "num_ctx": self.args.num_ctx,
            "timeout": self.args.timeout,
            "retries": self.args.retries,
            "seed": self.args.seed,
            "thinking": self.args.thinking,
            "format": "json",
            "extractive_fallback": False,
        }
        return {
            "id": sample.sample_id,
            "question": sample.question,
            "question_type": sample.question_type,
            "method": METHOD_NAMES[self.mode],
            "baseline_version": RUNNER_VERSION,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "retrieved_rule_ids": prepared.retrieved_rule_ids,
            "retrieved_event_ids": prepared.retrieved_event_ids,
            "retrieved_article_ids": prepared.retrieved_article_ids,
            "reasoning_path": prepared.reasoning_path,
            "reasoning_path_id": prepared.reasoning_path_id,
            "primary_path_ids": prepared.primary_path_ids,
            "verification_decision": final_decision,
            "decision_score": final_score,
            "final_answer": generation.answer,
            "citations": generation.citations,
            "retrieval": prepared.retrieval_payload,
            "verification": {
                **prepared.verification_payload,
                "final_decision": final_decision,
                "decision_score": final_score,
                "counterfactual_verification_enabled": (
                    self.mode == "full_legal_scm"
                ),
            },
            "generation": {
                "provider": "ollama",
                "model": self.args.model,
                "answer": generation.answer,
                "citations": generation.citations,
                "llm_decision": generation.decision,
                "llm_decision_score": generation.decision_score,
                "final_decision": final_decision,
                "decision_score": final_score,
                "authoritative_decision_applied": bool(
                    prepared.authoritative_decision
                ),
                "decision_overridden": decision_overridden,
                "raw_response": generation.raw_response,
                "response_metadata": generation.response_metadata,
                "settings": generation_settings,
                "prompt_metadata": dict(prompt_metadata),
            },
            "runtime_seconds": round(elapsed_seconds, 6),
            "pipeline_metadata": {
                "status": "SUCCESS",
                "method": METHOD_NAMES[self.mode],
                "mode": self.mode,
                "runner_version": RUNNER_VERSION,
                "generation_contract_version": GENERATION_CONTRACT_VERSION,
                "prompt_version": PROMPT_VERSION,
                "metric_applicability": applicability,
                **prepared.preparation_metadata,
                "provider": "ollama",
                "model": self.args.model,
                "generation_settings": generation_settings,
                "elapsed_seconds": round(elapsed_seconds, 6),
                "completed_at_utc": utc_now_iso(),
            },
        }

    @staticmethod
    def remove_intermediate(work_dir: Path, sample_id: str) -> None:
        sample_dir = work_dir / sanitize_filename(sample_id)
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prediction/output helpers
# ---------------------------------------------------------------------------

def metric_applicability(mode: str) -> dict[str, bool]:
    causal = mode in {"causal_path_rag", "full_legal_scm"}
    return {
        "rule_retrieval": causal,
        "event_retrieval": causal,
        "article_retrieval": mode != "llm_only",
        "causal_path": causal,
        "counterfactual_verifier": mode == "full_legal_scm",
        "decision": True,
        "answer": True,
        "citation": True,
        "runtime": True,
    }


def build_error_prediction(
    *,
    sample: Sample,
    args: argparse.Namespace,
    elapsed_seconds: float,
    message: str,
) -> dict[str, Any]:
    return {
        "id": sample.sample_id,
        "question": sample.question,
        "question_type": sample.question_type,
        "method": METHOD_NAMES[args.mode],
        "baseline_version": RUNNER_VERSION,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "retrieved_rule_ids": [],
        "retrieved_event_ids": [],
        "retrieved_article_ids": [],
        "reasoning_path": None,
        "reasoning_path_id": -1,
        "primary_path_ids": [],
        "verification_decision": "",
        "decision_score": None,
        "final_answer": "",
        "citations": [],
        "runtime_seconds": round(elapsed_seconds, 6),
        "error": message,
        "pipeline_metadata": {
            "status": "ERROR",
            "method": METHOD_NAMES[args.mode],
            "mode": args.mode,
            "runner_version": RUNNER_VERSION,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "metric_applicability": metric_applicability(args.mode),
            "provider": "ollama",
            "model": args.model,
            "error": message,
            "elapsed_seconds": round(elapsed_seconds, 6),
        },
    }


def load_existing_predictions(output_path: Path, jsonl_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        payload = read_json(output_path)
        if isinstance(payload, Mapping):
            metadata = dict(payload.get("metadata") or {})
            rows = payload.get("predictions") or []
        else:
            rows = payload
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    sample_id = normalize_id(row.get("id"))
                    if sample_id:
                        predictions[sample_id] = dict(row)
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping):
                    sample_id = normalize_id(row.get("id"))
                    if sample_id:
                        predictions[sample_id] = dict(row)
    return metadata, predictions


def configuration_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "provider": "ollama",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "timeout": args.timeout,
        "retries": args.retries,
        "seed": args.seed,
        "thinking": args.thinking,
        "format": "json",
        "max_context_chars": args.max_context_chars,
        "max_evidence": args.max_evidence,
        "vanilla_top_k": args.vanilla_top_k,
        "embedding_model": args.embedding_model,
        "event_top_k": args.event_top_k,
        "direct_rule_top_k": args.direct_rule_top_k,
        "semantic_pool_size": args.semantic_pool_size,
        "max_hops": args.max_hops,
        "max_paths_per_event": args.max_paths_per_event,
        "max_candidate_rules": args.max_candidate_rules,
        "final_top_k": args.final_top_k,
        "min_event_score": args.min_event_score,
        "min_rule_score": args.min_rule_score,
        "direction": args.direction,
        "max_cf_hops": args.max_cf_hops,
        "max_cf_paths": args.max_cf_paths,
        "verified_top_k": args.verified_top_k,
        "keep_threshold": args.keep_threshold,
        "reject_threshold": args.reject_threshold,
        "allow_scm_fallback": args.allow_scm_fallback,
        "include_uncertain": args.include_uncertain,
    }


def configuration_fingerprint(args: argparse.Namespace) -> str:
    encoded = json.dumps(
        configuration_payload(args),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(encoded)


def save_predictions(
    *,
    args: argparse.Namespace,
    benchmark_metadata: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    sample_order: Sequence[str],
) -> None:
    ordered = [predictions[item] for item in sample_order if item in predictions]
    successful = sum(not safe_string(row.get("error")) for row in ordered)
    write_json(Path(args.output), {
        "metadata": {
            "name": "BLHS Controlled System-Level Baseline Predictions",
            "method": METHOD_NAMES[args.mode],
            "mode": args.mode,
            "runner_version": RUNNER_VERSION,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "created_at_utc": utc_now_iso(),
            "benchmark": args.benchmark,
            "benchmark_metadata": dict(benchmark_metadata),
            "configuration": configuration_payload(args),
            "configuration_fingerprint": configuration_fingerprint(args),
            "metric_applicability": metric_applicability(args.mode),
            "prediction_count": len(ordered),
            "successful_prediction_count": successful,
            "failed_prediction_count": len(ordered) - successful,
            "evaluation_compatible": True,
            "gold_access_during_generation": False,
            "extractive_fallback": False,
        },
        "predictions": ordered,
    })


def save_errors(args: argparse.Namespace, predictions: Mapping[str, Mapping[str, Any]], sample_order: Sequence[str]) -> None:
    errors = [
        {
            "id": sample_id,
            "question": predictions[sample_id].get("question"),
            "error": predictions[sample_id].get("error"),
            "runtime_seconds": predictions[sample_id].get("runtime_seconds"),
        }
        for sample_id in sample_order
        if sample_id in predictions and safe_string(predictions[sample_id].get("error"))
    ]
    write_json(Path(args.errors_output), {
        "method": METHOD_NAMES[args.mode],
        "runner_version": RUNNER_VERSION,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# CLI and validation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one controlled BLHS system-level baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", default="")
    parser.add_argument("--jsonl-output", default="")
    parser.add_argument("--errors-output", default="")
    parser.add_argument("--run-log", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--validate-only", action="store_true")

    common = parser.add_argument_group("common Ollama generation contract")
    common.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    common.add_argument("--model", default=DEFAULT_MODEL)
    common.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    common.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    common.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    common.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    common.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument(
        "--thinking",
        choices=("disabled", "enabled", "server_default"),
        default="disabled",
    )
    common.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
    )
    common.add_argument("--max-evidence", type=int, default=DEFAULT_MAX_EVIDENCE)

    vanilla = parser.add_argument_group("vanilla RAG")
    vanilla.add_argument("--vanilla-script", default=DEFAULT_VANILLA_SCRIPT)
    vanilla.add_argument("--corpus", default=DEFAULT_CORPUS)
    vanilla.add_argument("--article-index", default=DEFAULT_ARTICLE_INDEX)
    vanilla.add_argument("--article-index-meta", default=DEFAULT_ARTICLE_INDEX_META)
    vanilla.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    vanilla.add_argument("--embedding-batch-size", type=int, default=16)
    vanilla.add_argument("--vanilla-top-k", type=int, default=5)
    vanilla.add_argument("--rebuild-article-index", action="store_true")

    causal = parser.add_argument_group("causal retrieval")
    causal.add_argument("--retriever-script", default=DEFAULT_RETRIEVER_SCRIPT)
    causal.add_argument("--graph", default=DEFAULT_GRAPH)
    causal.add_argument("--memory", default=DEFAULT_MEMORY)
    causal.add_argument("--causal-index", default=DEFAULT_CAUSAL_INDEX)
    causal.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    causal.add_argument("--retriever-model", default=DEFAULT_EMBEDDING_MODEL)
    causal.add_argument("--event-top-k", type=int, default=8)
    causal.add_argument("--direct-rule-top-k", type=int, default=8)
    causal.add_argument("--semantic-pool-size", type=int, default=100)
    causal.add_argument("--max-hops", type=int, default=2)
    causal.add_argument("--max-paths-per-event", type=int, default=30)
    causal.add_argument("--max-candidate-rules", type=int, default=200)
    causal.add_argument("--final-top-k", type=int, default=12)
    causal.add_argument("--min-event-score", type=float, default=0.20)
    causal.add_argument("--min-rule-score", type=float, default=0.15)
    causal.add_argument(
        "--direction",
        choices=("forward", "backward", "both"),
        default="both",
    )

    verifier = parser.add_argument_group("full LegalSCM")
    verifier.add_argument("--verifier-script", default=DEFAULT_VERIFIER_SCRIPT)
    verifier.add_argument("--rules", default=DEFAULT_RULES)
    verifier.add_argument("--max-cf-hops", type=int, default=3)
    verifier.add_argument("--max-cf-paths", type=int, default=30)
    verifier.add_argument("--verified-top-k", type=int, default=10)
    verifier.add_argument("--keep-threshold", type=float, default=0.52)
    verifier.add_argument("--reject-threshold", type=float, default=0.34)
    verifier.add_argument("--include-uncertain", action="store_true")
    verifier.add_argument(
        "--allow-scm-fallback",
        action="store_true",
        help=(
            "Cho phép Step 4 dùng node-deletion result khi LegalSCM fallback. "
            "Mặc định runner chuyển trường hợp này thành UNCERTAIN."
        ),
    )
    return parser.parse_args()


def derive_output_paths(args: argparse.Namespace) -> None:
    slug = model_slug(args.model)
    base_dir = REPO_ROOT / "data" / "baselines" / "system_level"
    base_name = f"{args.mode}_{slug}"
    output = resolve_repo_path(args.output) if args.output else base_dir / f"{base_name}_predictions.json"
    args.output = str(output)
    args.jsonl_output = str(
        resolve_repo_path(args.jsonl_output)
        if args.jsonl_output
        else base_dir / f"{base_name}_predictions.jsonl"
    )
    args.errors_output = str(
        resolve_repo_path(args.errors_output)
        if args.errors_output
        else base_dir / f"{base_name}_errors.json"
    )
    args.run_log = str(
        resolve_repo_path(args.run_log)
        if args.run_log
        else base_dir / f"{base_name}_run_log.json"
    )
    args.work_dir = str(
        resolve_repo_path(args.work_dir)
        if args.work_dir
        else base_dir / "intermediate" / base_name
    )


def resolve_input_paths(args: argparse.Namespace) -> None:
    for name in (
        "benchmark",
        "vanilla_script",
        "corpus",
        "article_index",
        "article_index_meta",
        "retriever_script",
        "graph",
        "memory",
        "causal_index",
        "embeddings",
        "verifier_script",
        "rules",
    ):
        setattr(args, name, str(resolve_repo_path(getattr(args, name))))


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index không được âm.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit phải lớn hơn 0.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải lớn hơn 0.")
    if args.max_tokens < 1 or args.num_ctx < 1 or args.timeout < 1:
        raise ValueError("max-tokens, num-ctx và timeout phải lớn hơn 0.")
    if args.retries < 0:
        raise ValueError("--retries không được âm.")
    if args.max_context_chars < 1 or args.max_evidence < 1:
        raise ValueError("max-context-chars và max-evidence phải lớn hơn 0.")
    if args.vanilla_top_k < 1:
        raise ValueError("--vanilla-top-k phải lớn hơn 0.")
    if args.max_hops < 1 or args.max_cf_hops < 1 or args.max_cf_paths < 1:
        raise ValueError("Các giới hạn hop/path phải lớn hơn 0.")
    if not 0.0 <= args.reject_threshold <= args.keep_threshold <= 1.0:
        raise ValueError(
            "Cần 0 <= reject-threshold <= keep-threshold <= 1."
        )

    required = [Path(args.benchmark)]
    if args.mode == "vanilla_rag":
        required.extend((Path(args.vanilla_script), Path(args.corpus)))
    if args.mode in {"causal_path_rag", "full_legal_scm"}:
        required.extend((
            Path(args.retriever_script),
            Path(args.graph),
            Path(args.memory),
            Path(args.causal_index),
            Path(args.embeddings),
        ))
    if args.mode == "full_legal_scm":
        required.extend((Path(args.verifier_script), Path(args.rules)))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Thiếu resource bắt buộc: " + ", ".join(missing))


def print_configuration(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("CONTROLLED SYSTEM-LEVEL BASELINE")
    print("=" * 80)
    print(f"Mode               : {args.mode}")
    print(f"Method             : {METHOD_NAMES[args.mode]}")
    print(f"Runner             : {RUNNER_VERSION}")
    print(f"Generation contract: {GENERATION_CONTRACT_VERSION}")
    print(f"Model              : {args.model}")
    print(f"Ollama             : {args.ollama_url}")
    print(f"Temperature        : {args.temperature}")
    print(f"Max tokens         : {args.max_tokens}")
    print(f"Context chars      : {args.max_context_chars}")
    print(f"Seed / thinking    : {args.seed} / {args.thinking}")
    print(f"Output             : {args.output}")
    print("Extractive fallback: DISABLED")
    print("Gold access        : DISABLED")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    resolve_input_paths(args)
    derive_output_paths(args)
    validate_args(args)
    print_configuration(args)

    benchmark_metadata, samples, all_sample_order = load_samples(
        Path(args.benchmark),
        start_index=args.start_index,
        limit=args.limit,
    )
    selected_order = [sample.sample_id for sample in samples]
    if args.validate_only:
        print(f"Configuration valid; selected samples: {len(samples)}")
        return 0

    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl_output)
    errors_path = Path(args.errors_output)
    run_log_path = Path(args.run_log)
    work_dir = Path(args.work_dir)

    if args.overwrite:
        for path in (output_path, jsonl_path, errors_path, run_log_path):
            path.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
    elif not args.resume and any(
        path.exists() for path in (output_path, jsonl_path)
    ):
        raise FileExistsError(
            "Output đã tồn tại. Dùng --resume hoặc --overwrite để tránh trộn run."
        )

    predictions: dict[str, dict[str, Any]] = {}
    if args.resume:
        existing_metadata, predictions = load_existing_predictions(
            output_path,
            jsonl_path,
        )
        old_fingerprint = safe_string(existing_metadata.get("configuration_fingerprint"))
        current_fingerprint = configuration_fingerprint(args)
        if old_fingerprint and old_fingerprint != current_fingerprint:
            raise ValueError(
                "Không thể resume vì configuration fingerprint khác run cũ."
            )
        print(f"Resume: loaded {len(predictions)} existing predictions.")
    else:
        jsonl_path.unlink(missing_ok=True)

    runner = SystemBaselineRunner(args)
    run_started_at = utc_now_iso()
    wall_started = time.time()
    generated = 0
    skipped = 0

    for position, sample in enumerate(samples, start=1):
        existing = predictions.get(sample.sample_id)
        if args.resume and existing is not None:
            has_error = bool(safe_string(existing.get("error")))
            if not (has_error and args.retry_errors):
                skipped += 1
                print(f"[{position}/{len(samples)}] {sample.sample_id} SKIP")
                continue

        print(f"[{position}/{len(samples)}] {sample.sample_id} ...", flush=True)
        started = time.time()
        try:
            prediction = runner.run_one(sample)
            predictions[sample.sample_id] = prediction
            append_jsonl(jsonl_path, prediction)
            generated += 1
            print(
                f"  decision={prediction['verification_decision']} | "
                f"citations={len(prediction['citations'])} | "
                f"{prediction['runtime_seconds']:.2f}s"
            )
        except KeyboardInterrupt:
            print("Interrupted; saving checkpoint...", file=sys.stderr)
            save_predictions(
                args=args,
                benchmark_metadata=benchmark_metadata,
                predictions=predictions,
                sample_order=selected_order,
            )
            save_errors(args, predictions, selected_order)
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            prediction = build_error_prediction(
                sample=sample,
                args=args,
                elapsed_seconds=time.time() - started,
                message=message,
            )
            prediction["traceback"] = traceback.format_exc()
            predictions[sample.sample_id] = prediction
            append_jsonl(jsonl_path, prediction)
            generated += 1
            print(f"  ERROR: {message}", file=sys.stderr)
            if args.fail_fast:
                raise
        finally:
            if args.mode == "full_legal_scm" and not args.keep_intermediate:
                runner.remove_intermediate(work_dir, sample.sample_id)

        if generated % args.checkpoint_every == 0:
            save_predictions(
                args=args,
                benchmark_metadata=benchmark_metadata,
                predictions=predictions,
                sample_order=selected_order,
            )
            save_errors(args, predictions, selected_order)

    save_predictions(
        args=args,
        benchmark_metadata=benchmark_metadata,
        predictions=predictions,
        sample_order=selected_order,
    )
    save_errors(args, predictions, selected_order)
    rewrite_jsonl(jsonl_path, predictions, selected_order)

    ordered = [predictions[item] for item in selected_order if item in predictions]
    failed = sum(bool(safe_string(row.get("error"))) for row in ordered)
    write_json(run_log_path, {
        "status": "COMPLETED_WITH_ERRORS" if failed else "COMPLETED",
        "method": METHOD_NAMES[args.mode],
        "mode": args.mode,
        "runner_version": RUNNER_VERSION,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "run_started_at_utc": run_started_at,
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.time() - wall_started, 6),
        "benchmark": args.benchmark,
        "selected_samples": len(samples),
        "prediction_count": len(ordered),
        "successful_prediction_count": len(ordered) - failed,
        "failed_prediction_count": failed,
        "generated_this_run": generated,
        "skipped_this_run": skipped,
        "configuration": configuration_payload(args),
        "configuration_fingerprint": configuration_fingerprint(args),
        "output": args.output,
        "jsonl_output": args.jsonl_output,
    })

    print("=" * 80)
    print(f"Done: {len(ordered)} predictions, {failed} errors")
    print(f"Output: {output_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
