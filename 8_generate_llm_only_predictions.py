from __future__ import annotations

"""
8_generate_llm_only_predictions.py

LLM-only baseline cho luận văn CausalRAG + Counterfactual Verification.

Nguyên tắc baseline:
- Chỉ đưa QUESTION cho LLM.
- KHÔNG truy xuất văn bản pháp luật.
- KHÔNG dùng causal graph / causal memory / embeddings.
- KHÔNG dùng gold_rule_ids / gold_event_ids / gold_path / gold_answer /
  gold_decision / gold_citations để sinh dự đoán.
- Output giữ cùng schema với pipeline_predictions.json để có thể dùng trực tiếp
  với 7_compute_evaluation_metrics.py.

Ví dụ:
    python 8_generate_llm_only_predictions.py \
      --benchmark data/blhs_multihop_benchmark_250.json \
      --output data/baselines/llm_only_predictions.json \
      --provider ollama \
      --model qwen3:8b \
      --limit 5

Chạy đủ 250 mẫu:
    python 8_generate_llm_only_predictions.py \
      --benchmark data/blhs_multihop_benchmark_250.json \
      --output data/baselines/llm_only_predictions.json \
      --provider ollama \
      --model qwen3:8b
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib import error, request


# ============================================================
# VERSION / DEFAULTS
# ============================================================

BASELINE_VERSION = "1.0-llm-only"

DEFAULT_BENCHMARK_PATH = "data/blhs_multihop_benchmark_250.json"
DEFAULT_OUTPUT_PATH = "data/baselines/llm_only_predictions.json"
DEFAULT_JSONL_PATH = "data/baselines/llm_only_predictions.jsonl"
DEFAULT_ERRORS_PATH = "data/baselines/llm_only_errors.json"
DEFAULT_RUN_LOG_PATH = "data/baselines/llm_only_run_log.json"

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen3:8b"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TIMEOUT = 180
DEFAULT_RETRIES = 1
DEFAULT_CHECKPOINT_EVERY = 1


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_id(value: Any) -> str:
    text = safe_string(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def unique_preserve_order(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = safe_string(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def load_json(path_value: str | Path) -> Any:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path_value: str | Path, payload: Any) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path_value: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def load_jsonl_if_exists(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


# ============================================================
# BENCHMARK LOADING
# ============================================================

def extract_benchmark_rows(payload: Any) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """
    Chỉ lấy metadata + rows.
    Hàm sinh baseline phía dưới chỉ đọc:
        - id / sample_id / question_id
        - question

    Các trường evaluation/gold tuyệt đối không được đưa vào prompt.
    """

    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, Mapping)]

    if isinstance(payload, Mapping):
        rows = (
            payload.get("questions")
            or payload.get("samples")
            or payload.get("data")
        )
        if not isinstance(rows, list):
            raise ValueError(
                "Benchmark phải chứa list tại questions/samples/data."
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return dict(metadata), [
            row for row in rows if isinstance(row, Mapping)
        ]

    raise ValueError(
        "Benchmark phải là list hoặc JSON object chứa questions/samples/data."
    )


@dataclass
class BaselineSample:
    sample_id: str
    question: str


def parse_baseline_sample(raw: Mapping[str, Any]) -> BaselineSample:
    sample_id = normalize_id(
        raw.get("id")
        or raw.get("sample_id")
        or raw.get("question_id")
    )
    question = safe_string(raw.get("question"))

    if not sample_id:
        raise ValueError("Một benchmark sample thiếu id/sample_id/question_id.")
    if not question:
        raise ValueError(f"Sample {sample_id} thiếu question.")

    # QUAN TRỌNG:
    # Không đọc evaluation.*, gold_*, answer... tại đây.
    return BaselineSample(
        sample_id=sample_id,
        question=question,
    )


def load_baseline_samples(
    benchmark_path: str | Path,
    *,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> tuple[dict[str, Any], list[BaselineSample]]:
    payload = load_json(benchmark_path)
    metadata, raw_rows = extract_benchmark_rows(payload)

    samples = [parse_baseline_sample(row) for row in raw_rows]

    if start_index < 0:
        raise ValueError("--start-index không được âm.")
    samples = samples[start_index:]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit phải lớn hơn 0.")
        samples = samples[:limit]

    return metadata, samples


# ============================================================
# HTTP
# ============================================================

def http_post_json(
    *,
    url: str,
    payload: Mapping[str, Any],
    timeout: int,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    http_request = request.Request(
        url=url,
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} từ {url}: {body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Không kết nối được tới {url}: {exc.reason}"
        ) from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("API trả response không phải JSON.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("API trả JSON không phải object.")

    return result


# ============================================================
# LLM PROVIDERS
# ============================================================

class BaseProvider:
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


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str) -> None:
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
        response = http_post_json(
            url=f"{self.base_url}/api/chat",
            payload={
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )

        answer = safe_string(
            (response.get("message") or {}).get("content")
        )
        if not answer:
            raise RuntimeError("Ollama không trả nội dung.")
        return answer


class OpenAICompatibleProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, *, base_url: str, api_key: str) -> None:
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
            raise ValueError("Thiếu OPENAI_API_KEY hoặc --api-key.")

        response = http_post_json(
            url=f"{self.base_url}/chat/completions",
            payload={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible API không trả choices.")

        answer = safe_string(
            ((choices[0] or {}).get("message") or {}).get("content")
        )
        if not answer:
            raise RuntimeError("OpenAI-compatible API trả nội dung rỗng.")
        return answer


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(self, api_key: str) -> None:
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
            raise ValueError("Thiếu GEMINI_API_KEY hoặc --api-key.")

        response = http_post_json(
            url=(
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model}:generateContent?key={self.api_key}"
            ),
            payload={
                "system_instruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout,
        )

        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini không trả candidates.")

        parts = (
            (candidates[0].get("content") or {}).get("parts")
            or []
        )
        answer = "\n".join(
            safe_string(part.get("text"))
            for part in parts
            if isinstance(part, Mapping)
            and safe_string(part.get("text"))
        )
        if not answer:
            raise RuntimeError("Gemini trả nội dung rỗng.")
        return answer


def build_provider(
    *,
    provider_name: str,
    base_url: str,
    api_key: str,
) -> BaseProvider:
    provider_name = safe_string(provider_name).lower()

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=base_url or DEFAULT_OLLAMA_URL
        )

    if provider_name == "openai":
        return OpenAICompatibleProvider(
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=api_key or os.getenv("GEMINI_API_KEY", "")
        )

    raise ValueError(
        "Provider không hợp lệ. Chọn ollama, openai hoặc gemini."
    )


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """Bạn là một hệ thống hỏi đáp pháp luật Việt Nam.

Đây là thí nghiệm LLM-only:
- Bạn KHÔNG được cung cấp văn bản pháp luật truy xuất.
- Bạn KHÔNG được cung cấp đồ thị nhân quả.
- Bạn phải trả lời chỉ dựa trên tri thức có sẵn trong mô hình.
- Không được giả vờ rằng bạn đã tra cứu tài liệu bên ngoài.
- Nếu không đủ chắc chắn, hãy thể hiện sự không chắc chắn thay vì bịa đặt.

Bạn phải trả về DUY NHẤT một JSON object hợp lệ, không markdown, không giải thích ngoài JSON.

Schema bắt buộc:
{
  "verification_decision": "SUPPORTED | REJECT_DIRECT_CLAIM | UNCERTAIN",
  "decision_score": 0.0,
  "final_answer": "câu trả lời ngắn gọn",
  "citations": ["Điều 123", "Điều 15"]
}

Quy tắc:
1. verification_decision:
   - SUPPORTED: nội dung/khẳng định trong câu hỏi được bạn đánh giá là phù hợp.
   - REJECT_DIRECT_CLAIM: nội dung/khẳng định trực tiếp hoặc phản thực tế trong câu hỏi không phù hợp.
   - UNCERTAIN: không đủ chắc chắn để kết luận.
2. decision_score nằm trong [0,1].
3. final_answer phải ngắn gọn, đi thẳng vào hệ quả/kết luận; không thêm tiêu đề.
4. Không chèn citation dạng [E1], [E2] vào final_answer.
5. citations chỉ ghi điều luật nếu bạn thực sự nhớ chắc số điều. Nếu không chắc, trả [].
6. Tuyệt đối không tự tạo rule_id, event_id hay causal path.
"""


def build_user_prompt(question: str) -> str:
    return (
        "Hãy trả lời câu hỏi pháp lý sau chỉ bằng tri thức nội tại của mô hình.\n\n"
        f"Câu hỏi:\n{question}\n\n"
        "Trả về đúng JSON object theo schema đã yêu cầu."
    )


# ============================================================
# RESPONSE PARSING
# ============================================================

DECISION_ALIASES = {
    "SUPPORTED": "SUPPORTED",
    "SUPPORT": "SUPPORTED",
    "TRUE": "SUPPORTED",
    "YES": "SUPPORTED",
    "CORRECT": "SUPPORTED",

    "REJECT_DIRECT_CLAIM": "REJECT_DIRECT_CLAIM",
    "REJECT": "REJECT_DIRECT_CLAIM",
    "REJECTED": "REJECT_DIRECT_CLAIM",
    "CONTRADICTED": "REJECT_DIRECT_CLAIM",
    "CONTRADICT": "REJECT_DIRECT_CLAIM",
    "FALSE": "REJECT_DIRECT_CLAIM",
    "NO": "REJECT_DIRECT_CLAIM",
    "NOT_SUPPORTED": "REJECT_DIRECT_CLAIM",

    "UNCERTAIN": "UNCERTAIN",
    "UNKNOWN": "UNCERTAIN",
    "UNRESOLVED": "UNCERTAIN",
    "INCONCLUSIVE": "UNCERTAIN",
}


def normalize_decision(value: Any) -> str:
    text = safe_string(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text).strip("_")
    return DECISION_ALIASES.get(text, "UNCERTAIN")


def strip_code_fence(text: str) -> str:
    value = safe_string(text)
    value = re.sub(
        r"^```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: lấy JSON object ngoài cùng đầu tiên.
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("LLM response không chứa JSON object.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(cleaned)):
        ch = cleaned[index]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:index + 1]
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ValueError("Không parse được JSON object từ LLM response.")


def normalize_citations(value: Any) -> list[str]:
    raw_values = ensure_list(value)
    citations: list[str] = []

    for raw in raw_values:
        if isinstance(raw, Mapping):
            raw = (
                raw.get("article_id")
                or raw.get("article")
                or raw.get("citation")
                or raw.get("text")
            )

        text = safe_string(raw)
        if not text:
            continue

        # Ưu tiên các số điều xuất hiện sau chữ "Điều".
        matches = re.findall(
            r"\b(?:điều|dieu)\s*(\d+[a-zA-Z]?)\b",
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            citations.extend(f"Điều {match}" for match in matches)
            continue

        # Nếu model trả thẳng "123" thì coi là article id.
        if re.fullmatch(r"\d+[a-zA-Z]?", text):
            citations.append(f"Điều {text}")
            continue

        # Giữ nguyên chuỗi nếu không nhận dạng được để evaluator tự normalize.
        citations.append(text)

    return unique_preserve_order(citations)


@dataclass
class ParsedLLMOutput:
    verification_decision: str
    decision_score: float
    final_answer: str
    citations: list[str]
    raw_response: str


def parse_llm_output(raw_response: str) -> ParsedLLMOutput:
    payload = extract_json_object(raw_response)

    decision = normalize_decision(
        payload.get("verification_decision")
        or payload.get("final_decision")
        or payload.get("decision")
    )

    decision_score = safe_float(
        payload.get("decision_score", payload.get("confidence", 0.5)),
        0.5,
    )
    decision_score = max(0.0, min(1.0, decision_score))

    final_answer = safe_string(
        payload.get("final_answer")
        or payload.get("answer")
        or payload.get("response")
    )

    if not final_answer:
        raise ValueError("LLM JSON thiếu final_answer.")

    citations = normalize_citations(
        payload.get("citations")
        or payload.get("predicted_citations")
        or []
    )

    return ParsedLLMOutput(
        verification_decision=decision,
        decision_score=decision_score,
        final_answer=final_answer,
        citations=citations,
        raw_response=raw_response,
    )


# ============================================================
# PREDICTION CONSTRUCTION
# ============================================================

def build_prediction(
    *,
    sample: BaselineSample,
    output: ParsedLLMOutput,
    provider_name: str,
    model: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """
    Schema tương thích Step 7.

    Các trường retrieval/path bắt buộc để rỗng vì đây là LLM-only.
    """

    return {
        "id": sample.sample_id,
        "question": sample.question,
        "method": "LLM_ONLY",
        "baseline_version": BASELINE_VERSION,

        # Không retrieval.
        "retrieved_rule_ids": [],
        "retrieved_event_ids": [],
        "retrieved_article_ids": [],

        # Không causal reasoning path.
        "reasoning_path": None,
        "reasoning_path_id": -1,

        # Cho phép Step 7 đánh giá decision nếu benchmark có gold decision.
        "verification_decision": output.verification_decision,
        "decision_score": output.decision_score,

        "final_answer": output.final_answer,
        "citations": output.citations,

        "retrieval": {
            "retrieved_rule_ids": [],
            "retrieved_event_ids": [],
            "retrieved_article_ids": [],
            "causal_paths": [],
        },

        "verification": {
            "final_decision": output.verification_decision,
            "decision_score": output.decision_score,
            "verification_method": "llm_internal_knowledge_only",
        },

        "generation": {
            "provider": provider_name,
            "model": model,
            "answer": output.final_answer,
            "citations": output.citations,
            "final_decision": output.verification_decision,
            "decision_score": output.decision_score,
        },

        "runtime_seconds": round(elapsed_seconds, 6),

        "pipeline_metadata": {
            "method": "LLM_ONLY",
            "baseline_version": BASELINE_VERSION,
            "retrieval_enabled": False,
            "causal_graph_enabled": False,
            "counterfactual_verification_enabled": False,
            "provider": provider_name,
            "model": model,
        },
    }


def build_error_prediction(
    *,
    sample: BaselineSample,
    provider_name: str,
    model: str,
    elapsed_seconds: float,
    message: str,
) -> dict[str, Any]:
    return {
        "id": sample.sample_id,
        "question": sample.question,
        "method": "LLM_ONLY",
        "baseline_version": BASELINE_VERSION,
        "retrieved_rule_ids": [],
        "retrieved_event_ids": [],
        "retrieved_article_ids": [],
        "reasoning_path": None,
        "reasoning_path_id": -1,
        "verification_decision": "",
        "decision_score": None,
        "final_answer": "",
        "citations": [],
        "runtime_seconds": round(elapsed_seconds, 6),
        "error": message,
        "pipeline_metadata": {
            "method": "LLM_ONLY",
            "baseline_version": BASELINE_VERSION,
            "retrieval_enabled": False,
            "causal_graph_enabled": False,
            "counterfactual_verification_enabled": False,
            "provider": provider_name,
            "model": model,
            "error": message,
        },
    }


# ============================================================
# GENERATION
# ============================================================

def call_with_retry(
    *,
    provider: BaseProvider,
    sample: BaselineSample,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> ParsedLLMOutput:
    last_exception: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            raw = provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(sample.question),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return parse_llm_output(raw)
        except Exception as exc:
            last_exception = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 5))

    assert last_exception is not None
    raise last_exception


# ============================================================
# RESUME / OUTPUT
# ============================================================

def load_existing_predictions(
    *,
    output_path: str | Path,
    jsonl_path: str | Path,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    output = Path(output_path)
    if output.exists():
        try:
            payload = load_json(output)
            rows = (
                payload.get("predictions")
                if isinstance(payload, Mapping)
                else payload
            )
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    sample_id = normalize_id(
                        row.get("id")
                        or row.get("sample_id")
                        or row.get("question_id")
                    )
                    if sample_id:
                        indexed[sample_id] = dict(row)
        except Exception:
            pass

    # JSONL dùng như checkpoint; record cuối cùng thắng.
    for row in load_jsonl_if_exists(jsonl_path):
        sample_id = normalize_id(
            row.get("id")
            or row.get("sample_id")
            or row.get("question_id")
        )
        if sample_id:
            indexed[sample_id] = row

    return indexed


def build_output_payload(
    *,
    benchmark_path: str,
    provider_name: str,
    model: str,
    benchmark_metadata: Mapping[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = sum(
        1 for item in predictions
        if not safe_string(item.get("error"))
    )

    return {
        "metadata": {
            "method": "LLM_ONLY",
            "baseline_version": BASELINE_VERSION,
            "benchmark": benchmark_path,
            "provider": provider_name,
            "model": model,
            "retrieval_enabled": False,
            "causal_graph_enabled": False,
            "counterfactual_verification_enabled": False,
            "benchmark_metadata": dict(benchmark_metadata),
            "prediction_count": len(predictions),
            "successful_prediction_count": successful,
            "failed_prediction_count": len(predictions) - successful,
        },
        "predictions": predictions,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sinh prediction cho baseline LLM-only. "
            "LLM chỉ nhìn thấy question, không retrieval/graph/gold."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_PATH,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--jsonl-output",
        default=DEFAULT_JSONL_PATH,
    )
    parser.add_argument(
        "--errors-output",
        default=DEFAULT_ERRORS_PATH,
    )
    parser.add_argument(
        "--run-log",
        default=DEFAULT_RUN_LOG_PATH,
    )

    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "gemini"],
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
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua các sample đã có prediction thành công.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Khi --resume, chạy lại cả các sample đã có error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Xóa output/jsonl cũ trước khi chạy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens phải lớn hơn 0.")
    if args.timeout < 1:
        raise ValueError("--timeout phải lớn hơn 0.")
    if args.retries < 0:
        raise ValueError("--retries không được âm.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải lớn hơn 0.")

    model = args.model
    if args.provider == "openai" and model == DEFAULT_MODEL:
        model = DEFAULT_OPENAI_MODEL
    elif args.provider == "gemini" and model == DEFAULT_MODEL:
        model = DEFAULT_GEMINI_MODEL

    base_url = args.base_url
    if args.provider == "ollama" and not base_url:
        base_url = DEFAULT_OLLAMA_URL
    elif args.provider == "openai" and not base_url:
        base_url = DEFAULT_OPENAI_BASE_URL

    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl_output)
    errors_path = Path(args.errors_output)
    run_log_path = Path(args.run_log)

    if args.overwrite:
        for path in (
            output_path,
            jsonl_path,
            errors_path,
            run_log_path,
        ):
            if path.exists():
                path.unlink()

    benchmark_metadata, samples = load_baseline_samples(
        args.benchmark,
        start_index=args.start_index,
        limit=args.limit,
    )

    provider = build_provider(
        provider_name=args.provider,
        base_url=base_url,
        api_key=args.api_key,
    )

    existing: dict[str, dict[str, Any]] = {}
    if args.resume:
        existing = load_existing_predictions(
            output_path=output_path,
            jsonl_path=jsonl_path,
        )

    predictions_by_id: dict[str, dict[str, Any]] = dict(existing)
    new_errors: list[dict[str, Any]] = []

    print("=" * 80)
    print("LLM-ONLY BASELINE PREDICTION")
    print("=" * 80)
    print(f"Version           : {BASELINE_VERSION}")
    print(f"Benchmark         : {args.benchmark}")
    print(f"Samples selected  : {len(samples)}")
    print(f"Provider          : {args.provider}")
    print(f"Model             : {model}")
    print(f"Temperature       : {args.temperature}")
    print(f"Retrieval         : DISABLED")
    print(f"Causal graph      : DISABLED")
    print(f"Gold access       : DISABLED")
    print("=" * 80)

    run_started = time.time()
    generated_count = 0
    skipped_count = 0

    for position, sample in enumerate(samples, start=1):
        previous = predictions_by_id.get(sample.sample_id)

        if args.resume and previous is not None:
            previous_error = safe_string(previous.get("error"))
            should_retry = bool(previous_error) and args.retry_errors
            if not should_retry:
                skipped_count += 1
                print(
                    f"[{position}/{len(samples)}] {sample.sample_id} "
                    f"SKIP"
                )
                continue

        print(
            f"[{position}/{len(samples)}] {sample.sample_id} ...",
            flush=True,
        )

        started = time.time()

        try:
            parsed = call_with_retry(
                provider=provider,
                sample=sample,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )

            elapsed = time.time() - started
            prediction = build_prediction(
                sample=sample,
                output=parsed,
                provider_name=args.provider,
                model=model,
                elapsed_seconds=elapsed,
            )

            predictions_by_id[sample.sample_id] = prediction
            append_jsonl(jsonl_path, prediction)
            generated_count += 1

            print(
                f"  decision={prediction['verification_decision']} | "
                f"citations={len(prediction['citations'])} | "
                f"{elapsed:.2f}s"
            )

        except Exception as exc:
            elapsed = time.time() - started
            message = f"{type(exc).__name__}: {exc}"

            prediction = build_error_prediction(
                sample=sample,
                provider_name=args.provider,
                model=model,
                elapsed_seconds=elapsed,
                message=message,
            )
            predictions_by_id[sample.sample_id] = prediction
            append_jsonl(jsonl_path, prediction)
            new_errors.append(
                {
                    "id": sample.sample_id,
                    "question": sample.question,
                    "error": message,
                    "runtime_seconds": round(elapsed, 6),
                }
            )

            print(f"  ERROR: {message}")

        if generated_count % args.checkpoint_every == 0:
            ordered_predictions = [
                predictions_by_id[s.sample_id]
                for s in samples
                if s.sample_id in predictions_by_id
            ]
            write_json(
                output_path,
                build_output_payload(
                    benchmark_path=args.benchmark,
                    provider_name=args.provider,
                    model=model,
                    benchmark_metadata=benchmark_metadata,
                    predictions=ordered_predictions,
                ),
            )

    ordered_predictions = [
        predictions_by_id[s.sample_id]
        for s in samples
        if s.sample_id in predictions_by_id
    ]

    output_payload = build_output_payload(
        benchmark_path=args.benchmark,
        provider_name=args.provider,
        model=model,
        benchmark_metadata=benchmark_metadata,
        predictions=ordered_predictions,
    )
    write_json(output_path, output_payload)

    all_errors = [
        {
            "id": item.get("id"),
            "question": item.get("question"),
            "error": item.get("error"),
            "runtime_seconds": item.get("runtime_seconds"),
        }
        for item in ordered_predictions
        if safe_string(item.get("error"))
    ]
    write_json(
        errors_path,
        {
            "method": "LLM_ONLY",
            "baseline_version": BASELINE_VERSION,
            "errors": all_errors,
        },
    )

    elapsed_total = time.time() - run_started
    run_log = {
        "method": "LLM_ONLY",
        "baseline_version": BASELINE_VERSION,
        "benchmark": args.benchmark,
        "output": str(output_path),
        "jsonl_output": str(jsonl_path),
        "provider": args.provider,
        "model": model,
        "selected_samples": len(samples),
        "prediction_count": len(ordered_predictions),
        "generated_this_run": generated_count,
        "skipped_this_run": skipped_count,
        "failed_prediction_count": len(all_errors),
        "elapsed_seconds": round(elapsed_total, 6),
        "retrieval_enabled": False,
        "causal_graph_enabled": False,
        "counterfactual_verification_enabled": False,
    }
    write_json(run_log_path, run_log)

    success_count = len(ordered_predictions) - len(all_errors)

    print("\n" + "=" * 80)
    print("LLM-ONLY BASELINE COMPLETE")
    print("=" * 80)
    print(f"Predictions       : {len(ordered_predictions)}")
    print(f"Success           : {success_count}")
    print(f"Failed            : {len(all_errors)}")
    print(f"Generated now     : {generated_count}")
    print(f"Skipped           : {skipped_count}")
    print(f"Elapsed           : {elapsed_total:.2f}s")
    print(f"Output            : {output_path}")
    print(f"Checkpoint JSONL  : {jsonl_path}")
    print(f"Errors            : {errors_path}")
    print(f"Run log           : {run_log_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
