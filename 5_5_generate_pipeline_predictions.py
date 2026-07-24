#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_5_generate_pipeline_predictions.py

Batch runner cho pipeline CausalRAG BLHS:

    Benchmark
        -> Bước 3: Multi-hop causal retrieval
        -> Bước 4: Counterfactual verification
        -> Bước 5: Final answer generation
        -> data/pipeline_predictions.json

Script được thiết kế tương thích với:
- 3_multi_hop_causal_retriever.py
- 4_counterfactual_verification.py
- 5_generate_final_answer.py
- 6_run_evaluation_pipeline.py

Đặc điểm:
- Chỉ khởi tạo retriever và embedding model của bước 3 một lần.
- Hỗ trợ resume khi quá trình chạy bị gián đoạn.
- Ghi JSONL ngay sau từng câu để hạn chế mất kết quả.
- Lưu lỗi riêng, không làm dừng toàn bộ benchmark mặc định.
- Có thể chạy một phần benchmark bằng --limit/--start-index.
- Có chế độ --provider extractive để kiểm thử không cần LLM.
- Có thể lưu hoặc không lưu intermediate result của từng câu.

Ví dụ chạy nhanh 5 câu bằng extractive:

python 5_5_generate_pipeline_predictions.py \
    --benchmark data/blhs_multihop_benchmark_250.json \
    --provider extractive \
    --limit 5

Chạy toàn bộ bằng Ollama:

python 5_5_generate_pipeline_predictions.py \
    --benchmark data/blhs_multihop_benchmark_250.json \
    --provider ollama \
    --answer-model qwen3:8b \
    --resume

Sau khi sinh prediction:

python 6_run_evaluation_pipeline.py \
    --benchmark data/blhs_multihop_benchmark_250.json \
    --predictions data/pipeline_predictions.json \
    --output-dir evaluation_results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_BENCHMARK = "data/blhs_multihop_benchmark_250.json"

DEFAULT_RETRIEVER_SCRIPT = "3_multi_hop_causal_retriever.py"
DEFAULT_VERIFIER_SCRIPT = "4_counterfactual_verification.py"
DEFAULT_ANSWER_SCRIPT = "5_generate_final_answer.py"

DEFAULT_GRAPH = "data/legal_causal_knowledge_graph.graphml"
DEFAULT_MEMORY = "data/causal_memory.csv"
DEFAULT_INDEX = "data/causal_memory.index"
DEFAULT_EMBEDDINGS = "data/causal_memory_embeddings.npy"
DEFAULT_CF_MAP = "data/counterfactual_event_map.json"

DEFAULT_OUTPUT = "data/pipeline_predictions.json"
DEFAULT_JSONL_OUTPUT = "data/pipeline_predictions.jsonl"
DEFAULT_ERROR_OUTPUT = "data/pipeline_errors.json"
DEFAULT_RUN_LOG = "data/pipeline_run_log.json"
DEFAULT_WORK_DIR = "data/pipeline_intermediate"

DEFAULT_RETRIEVER_MODEL = "BAAI/bge-m3"
DEFAULT_VERIFIER_MODEL = "BAAI/bge-m3"
DEFAULT_PROVIDER = "extractive"
DEFAULT_ANSWER_MODEL = "qwen3:8b"


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = safe_string(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            key: to_serializable(child)
            for key, child in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_serializable(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_serializable(child)
            for child in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            to_serializable(data),
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                to_serializable(dict(row)),
                ensure_ascii=False,
            )
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def load_module(module_name: str, script_path: Path) -> Any:
    if not script_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy module {module_name}: {script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(script_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể import: {script_path}")

    module = importlib.util.module_from_spec(spec)

    # Cần đăng ký trước khi exec để dataclass xử lý đúng module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# BENCHMARK
# ============================================================

def load_benchmark(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = read_json(path)

    if isinstance(data, list):
        metadata: dict[str, Any] = {}
        rows = data
    elif isinstance(data, Mapping):
        metadata = dict(data.get("metadata") or {})
        rows = (
            data.get("questions")
            or data.get("samples")
            or data.get("data")
        )
    else:
        raise ValueError("Benchmark phải là JSON object hoặc JSON list.")

    if not isinstance(rows, list):
        raise ValueError(
            "Benchmark không có danh sách questions/samples/data."
        )

    valid_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue

        sample_id = safe_string(
            row.get("id")
            or row.get("sample_id")
            or row.get("question_id")
            or f"question_{index + 1:04d}"
        )
        question = safe_string(
            row.get("question")
            or row.get("query")
        )

        if not question:
            continue

        normalized = dict(row)
        normalized["id"] = sample_id
        normalized["question"] = question
        valid_rows.append(normalized)

    if not valid_rows:
        raise ValueError("Benchmark không có câu hỏi hợp lệ.")

    return metadata, valid_rows


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_existing_predictions(
    json_path: Path,
    jsonl_path: Path,
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}

    if json_path.exists():
        try:
            data = read_json(json_path)

            if isinstance(data, Mapping):
                rows = (
                    data.get("predictions")
                    or data.get("results")
                    or []
                )
            else:
                rows = data

            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    sample_id = safe_string(
                        row.get("id")
                        or row.get("sample_id")
                        or row.get("question_id")
                    )
                    if sample_id:
                        predictions[sample_id] = dict(row)
        except Exception as exc:
            print(
                "Warning: không đọc được prediction JSON cũ:",
                exc,
                file=sys.stderr,
            )

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: bỏ qua JSONL lỗi ở dòng {line_number}.",
                        file=sys.stderr,
                    )
                    continue

                if not isinstance(row, Mapping):
                    continue

                sample_id = safe_string(
                    row.get("id")
                    or row.get("sample_id")
                    or row.get("question_id")
                )
                if sample_id:
                    predictions[sample_id] = dict(row)

    return predictions


def rewrite_jsonl(
    path: Path,
    predictions: Mapping[str, Mapping[str, Any]],
    benchmark_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for sample_id in benchmark_order:
            row = predictions.get(sample_id)
            if row is None:
                continue
            file.write(
                json.dumps(
                    to_serializable(dict(row)),
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# PREDICTION EXTRACTION
# ============================================================

def derive_verification_decision(
    verification: Mapping[str, Any],
) -> str:
    """
    Bước 4 không xuất trực tiếp decision toàn cục nên runner suy ra:
    - CONTRADICTED chiếm ưu thế và không có SUPPORTED -> REJECT_DIRECT_CLAIM
    - Có SUPPORTED hoặc có verified evidence -> SUPPORTED
    - Còn lại -> UNCERTAIN
    """
    statistics = verification.get("statistics") or {}
    status_counts = statistics.get("path_status_counts") or {}

    supported = safe_int(status_counts.get("SUPPORTED"))
    contradicted = safe_int(status_counts.get("CONTRADICTED"))
    unresolved = safe_int(status_counts.get("UNRESOLVED"))

    verified_evidence = verification.get("verified_evidence") or []
    removed_evidence = verification.get("removed_evidence") or []

    if (
        contradicted > 0
        and supported == 0
        and contradicted >= unresolved
    ):
        return "REJECT_DIRECT_CLAIM"

    if supported > 0 or len(verified_evidence) > 0:
        return "SUPPORTED"

    if contradicted > 0 and len(removed_evidence) > 0:
        return "REJECT_DIRECT_CLAIM"

    return "UNCERTAIN"


def extract_reasoning_path(
    retrieval: Mapping[str, Any],
) -> list[dict[str, Any]]:
    causal_paths = retrieval.get("causal_paths") or []

    if not causal_paths:
        return []

    # Bước 3 đã sắp xếp path theo graph_score giảm dần.
    best_path = causal_paths[0]

    if not isinstance(best_path, Mapping):
        return []

    result: list[dict[str, Any]] = []

    for step in best_path.get("steps") or []:
        if not isinstance(step, Mapping):
            continue

        source_id = safe_string(
            step.get("source_event_id")
            or step.get("source_event_node")
        )
        target_id = safe_string(
            step.get("target_event_id")
            or step.get("target_event_node")
        )

        if not source_id or not target_id:
            continue

        result.append(
            {
                "hop": safe_int(step.get("hop"), len(result) + 1),
                "source_event_id": source_id,
                "source_event_name": safe_string(
                    step.get("source_event_name")
                ),
                "target_event_id": target_id,
                "target_event_name": safe_string(
                    step.get("target_event_name")
                ),
                "rule_ids": unique_preserve_order(
                    step.get("rule_ids") or []
                ),
                "article_ids": unique_preserve_order(
                    step.get("article_ids") or []
                ),
            }
        )

    return result


def extract_citations(
    final_answer: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> list[str]:
    citations: list[str] = []

    # Citation phục vụ evaluator phải là "Điều X", không chỉ E1/E2.
    for evidence in final_answer.get("selected_evidence") or []:
        if not isinstance(evidence, Mapping):
            continue
        article_id = safe_string(evidence.get("article_id"))
        if article_id:
            citations.append(f"Điều {article_id}")

    if not citations:
        for group_name in ("verified_evidence", "uncertain_evidence"):
            for evidence in verification.get(group_name) or []:
                if not isinstance(evidence, Mapping):
                    continue
                article_id = safe_string(
                    evidence.get("article_id")
                    or (evidence.get("original_evidence") or {}).get(
                        "article_id"
                    )
                )
                if article_id:
                    citations.append(f"Điều {article_id}")

    return unique_preserve_order(citations)


def build_prediction(
    sample: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    verification: Mapping[str, Any],
    final_answer: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    retrieved_events = retrieval.get("retrieved_events") or []
    evidence = retrieval.get("evidence") or []

    retrieved_event_ids = unique_preserve_order(
        item.get("event_id") or item.get("graph_node_id")
        for item in retrieved_events
        if isinstance(item, Mapping)
    )

    retrieved_rule_ids = unique_preserve_order(
        item.get("rule_id")
        for item in evidence
        if isinstance(item, Mapping)
    )

    retrieved_article_ids = unique_preserve_order(
        item.get("article_id")
        for item in evidence
        if isinstance(item, Mapping)
    )

    reasoning_path = extract_reasoning_path(retrieval)
    decision = derive_verification_decision(verification)
    answer_text = safe_string(
        final_answer.get("answer")
        or final_answer.get("final_answer")
    )
    citations = extract_citations(
        final_answer,
        verification,
    )

    return {
        "id": safe_string(sample.get("id")),
        "question": safe_string(sample.get("question")),
        "question_type": safe_string(sample.get("question_type")),
        "retrieved_rule_ids": retrieved_rule_ids,
        "retrieved_event_ids": retrieved_event_ids,
        "retrieved_article_ids": retrieved_article_ids,
        "reasoning_path": reasoning_path,
        "verification_decision": decision,
        "final_answer": answer_text,
        "citations": citations,
        "retrieval": {
            "retrieved_events": retrieved_events,
            "direct_rule_hits": retrieval.get("direct_rule_hits") or [],
            "retrieved_rules": evidence,
            "causal_paths": retrieval.get("causal_paths") or [],
            "statistics": retrieval.get("statistics") or {},
            "configuration": retrieval.get("configuration") or {},
        },
        "verification": {
            "final_decision": decision,
            "confidence": safe_float(verification.get("confidence")),
            "consistency_score": safe_float(
                verification.get("consistency_score")
            ),
            "path_verifications": (
                verification.get("path_verifications") or []
            ),
            "verified_evidence": (
                verification.get("verified_evidence") or []
            ),
            "uncertain_evidence": (
                verification.get("uncertain_evidence") or []
            ),
            "removed_evidence": (
                verification.get("removed_evidence") or []
            ),
            "statistics": verification.get("statistics") or {},
            "configuration": verification.get("configuration") or {},
        },
        "generation": {
            "answer": answer_text,
            "provider": safe_string(final_answer.get("provider")),
            "model": safe_string(final_answer.get("model")),
            "citations_used": (
                final_answer.get("citations_used") or []
            ),
            "selected_evidence": (
                final_answer.get("selected_evidence") or []
            ),
            "selected_paths": (
                final_answer.get("selected_paths") or []
            ),
            "confidence": safe_float(final_answer.get("confidence")),
            "consistency_score": safe_float(
                final_answer.get("consistency_score")
            ),
            "metadata": (
                final_answer.get("generation_metadata") or {}
            ),
        },
        "pipeline_metadata": {
            "status": "SUCCESS",
            "elapsed_seconds": round(elapsed_seconds, 4),
            "completed_at_utc": now_iso(),
        },
    }


# ============================================================
# PIPELINE RUNNER
# ============================================================

class BatchPipelineRunner:
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        self.args = args

        self.retriever_module = load_module(
            "causalrag_step3",
            Path(args.retriever_script),
        )
        self.verifier_module = load_module(
            "causalrag_step4",
            Path(args.verifier_script),
        )
        self.answer_module = load_module(
            "causalrag_step5",
            Path(args.answer_script),
        )

        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_retriever()

    def _initialize_retriever(self) -> None:
        print("\nKhởi tạo tài nguyên bước 3...")

        store = self.retriever_module.CausalResourceStore(
            graph_path=self.args.graph,
            memory_path=self.args.memory,
            index_path=self.args.index,
            embeddings_path=self.args.embeddings,
            model_name=self.args.retriever_model,
        )

        self.retriever = (
            self.retriever_module.MultiHopCausalRetriever(store)
        )

        print("Khởi tạo bước 3 hoàn tất.")

    def run_one(
        self,
        sample: Mapping[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        sample_id = safe_string(sample.get("id"))
        question = safe_string(sample.get("question"))

        started_at = time.time()

        # ---------------- STEP 3 ----------------
        retrieval_result = self.retriever.retrieve(
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
        retrieval_data = to_serializable(retrieval_result)

        sample_dir = self.work_dir / sample_id
        retrieval_path = sample_dir / "retrieval_result.json"
        verification_path = (
            sample_dir / "counterfactual_verification_result.json"
        )
        final_answer_path = sample_dir / "final_answer_result.json"

        # Bước 4 hiện đọc retrieval từ file nên luôn ghi file tạm.
        write_json(retrieval_path, retrieval_data)

        # ---------------- STEP 4 ----------------
        verifier_store = (
            self.verifier_module.CounterfactualResourceStore(
                graph_path=self.args.graph,
                memory_path=self.args.memory,
                embeddings_path=self.args.embeddings,
                retrieval_result_path=str(retrieval_path),
                counterfactual_map_path=self.args.counterfactual_map,
                model_name=self.args.verifier_model,
                enable_semantic_mapping=(
                    not self.args.disable_semantic_mapping
                ),
            )
        )

        verifier_pipeline = (
            self.verifier_module.CounterfactualVerificationPipeline(
                verifier_store
            )
        )

        verification_result = verifier_pipeline.run(
            cf_top_k=self.args.cf_top_k,
            mapping_top_k=self.args.mapping_top_k,
            mapping_threshold=self.args.mapping_threshold,
            max_cf_hops=self.args.max_cf_hops,
            max_cf_paths=self.args.max_cf_paths,
            verified_top_k=self.args.verified_top_k,
            keep_threshold=self.args.keep_threshold,
            reject_threshold=self.args.reject_threshold,
        )
        verification_data = to_serializable(
            verification_result
        )
        write_json(verification_path, verification_data)

        # ---------------- STEP 5 ----------------
        answer_store = (
            self.answer_module.FinalAnswerInputStore(
                verification_result_path=str(verification_path),
                retrieval_result_path=str(retrieval_path),
            )
        )

        answer_pipeline = (
            self.answer_module.FinalAnswerPipeline(answer_store)
        )

        answer_model = self._resolve_answer_model()

        final_answer_result = answer_pipeline.run(
            provider_name=self.args.provider,
            model=answer_model,
            api_key=self.args.api_key,
            base_url=self._resolve_base_url(),
            max_evidence=self.args.max_evidence,
            max_paths=self.args.answer_max_paths,
            max_context_chars=self.args.max_context_chars,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            timeout=self.args.timeout,
            min_verification_score=(
                self.args.min_verification_score
            ),
            include_uncertain=self.args.include_uncertain,
            fallback_to_extractive=(
                not self.args.no_extractive_fallback
            ),
        )
        final_answer_data = to_serializable(
            final_answer_result
        )
        write_json(final_answer_path, final_answer_data)

        prediction = build_prediction(
            sample=sample,
            retrieval=retrieval_data,
            verification=verification_data,
            final_answer=final_answer_data,
            elapsed_seconds=time.time() - started_at,
        )

        if not self.args.keep_intermediate:
            self._remove_intermediate(sample_dir)

        return (
            prediction,
            retrieval_data,
            verification_data,
            final_answer_data,
        )

    def _resolve_answer_model(self) -> str:
        default_model = getattr(
            self.answer_module,
            "DEFAULT_MODEL",
            DEFAULT_ANSWER_MODEL,
        )
        model = self.args.answer_model or default_model

        if (
            self.args.provider == "openai"
            and model == default_model
        ):
            model = getattr(
                self.answer_module,
                "DEFAULT_OPENAI_MODEL",
                model,
            )

        if (
            self.args.provider == "gemini"
            and model == default_model
        ):
            model = getattr(
                self.answer_module,
                "DEFAULT_GEMINI_MODEL",
                model,
            )

        if self.args.provider == "extractive":
            return "extractive"

        return model

    def _resolve_base_url(self) -> str:
        if self.args.base_url:
            return self.args.base_url

        if self.args.provider == "ollama":
            return getattr(
                self.answer_module,
                "DEFAULT_OLLAMA_URL",
                "http://localhost:11434",
            )

        if self.args.provider == "openai":
            return getattr(
                self.answer_module,
                "DEFAULT_OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )

        return ""

    @staticmethod
    def _remove_intermediate(sample_dir: Path) -> None:
        try:
            for file_path in sample_dir.iterdir():
                file_path.unlink(missing_ok=True)
            sample_dir.rmdir()
        except OSError:
            pass


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BLHS CausalRAG steps 3, 4 and 5 over a benchmark "
            "and generate pipeline_predictions.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output.
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL_OUTPUT)
    parser.add_argument("--errors-output", default=DEFAULT_ERROR_OUTPUT)
    parser.add_argument("--run-log", default=DEFAULT_RUN_LOG)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)

    # Source scripts.
    parser.add_argument(
        "--retriever-script",
        default=DEFAULT_RETRIEVER_SCRIPT,
    )
    parser.add_argument(
        "--verifier-script",
        default=DEFAULT_VERIFIER_SCRIPT,
    )
    parser.add_argument(
        "--answer-script",
        default=DEFAULT_ANSWER_SCRIPT,
    )

    # Shared resources.
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--counterfactual-map",
        default=DEFAULT_CF_MAP,
    )

    # Range/resume/error.
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Cập nhật JSON tổng sau mỗi N prediction thành công.",
    )

    # Step 3.
    parser.add_argument(
        "--retriever-model",
        default=DEFAULT_RETRIEVER_MODEL,
    )
    parser.add_argument("--event-top-k", type=int, default=8)
    parser.add_argument("--direct-rule-top-k", type=int, default=8)
    parser.add_argument("--semantic-pool-size", type=int, default=100)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-paths-per-event", type=int, default=30)
    parser.add_argument("--max-candidate-rules", type=int, default=200)
    parser.add_argument("--final-top-k", type=int, default=12)
    parser.add_argument("--min-event-score", type=float, default=0.20)
    parser.add_argument("--min-rule-score", type=float, default=0.15)
    parser.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="both",
    )

    # Step 4.
    parser.add_argument(
        "--verifier-model",
        default=DEFAULT_VERIFIER_MODEL,
    )
    parser.add_argument("--cf-top-k", type=int, default=5)
    parser.add_argument("--mapping-top-k", type=int, default=5)
    parser.add_argument("--mapping-threshold", type=float, default=0.42)
    parser.add_argument("--max-cf-hops", type=int, default=3)
    parser.add_argument("--max-cf-paths", type=int, default=30)
    parser.add_argument("--verified-top-k", type=int, default=10)
    parser.add_argument("--keep-threshold", type=float, default=0.52)
    parser.add_argument("--reject-threshold", type=float, default=0.34)
    parser.add_argument(
        "--disable-semantic-mapping",
        action="store_true",
        help=(
            "Không load SentenceTransformer lần nữa ở bước 4. "
            "Khuyến nghị bật để batch chạy nhanh hơn."
        ),
    )

    # Step 5.
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "gemini", "extractive"],
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument(
        "--answer-model",
        default=DEFAULT_ANSWER_MODEL,
    )
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-evidence", type=int, default=8)
    parser.add_argument(
        "--answer-max-paths",
        type=int,
        default=6,
    )
    parser.add_argument("--max-context-chars", type=int, default=18000)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--min-verification-score",
        type=float,
        default=0.45,
    )
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--no-extractive-fallback", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start-index không được âm.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit phải lớn hơn 0.")

    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every phải lớn hơn 0.")

    if args.max_hops < 1 or args.max_cf_hops < 1:
        raise ValueError("max hops phải lớn hơn 0.")

    if not (
        0.0
        <= args.reject_threshold
        <= args.keep_threshold
        <= 1.0
    ):
        raise ValueError(
            "Cần thỏa mãn 0 <= reject-threshold "
            "<= keep-threshold <= 1."
        )

    for script_path in (
        args.retriever_script,
        args.verifier_script,
        args.answer_script,
    ):
        if not Path(script_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy source script: {script_path}"
            )


# ============================================================
# REPORTING
# ============================================================

def save_predictions(
    output_path: Path,
    benchmark_path: Path,
    benchmark_metadata: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
    benchmark_order: list[str],
    errors: Mapping[str, Mapping[str, Any]],
    run_started_at: str,
) -> None:
    ordered_predictions = [
        predictions[sample_id]
        for sample_id in benchmark_order
        if sample_id in predictions
    ]

    payload = {
        "metadata": {
            "name": "BLHS CausalRAG Pipeline Predictions",
            "version": "1.0",
            "created_at_utc": now_iso(),
            "run_started_at_utc": run_started_at,
            "benchmark": str(benchmark_path),
            "benchmark_metadata": dict(benchmark_metadata),
            "successful_predictions": len(ordered_predictions),
            "failed_questions": len(errors),
            "evaluation_compatible": True,
        },
        "predictions": ordered_predictions,
    }

    write_json(output_path, payload)


def save_errors(
    path: Path,
    errors: Mapping[str, Mapping[str, Any]],
) -> None:
    write_json(
        path,
        {
            "metadata": {
                "created_at_utc": now_iso(),
                "count": len(errors),
            },
            "errors": list(errors.values()),
        },
    )


def save_run_log(
    path: Path,
    *,
    args: argparse.Namespace,
    benchmark_path: Path,
    total_selected: int,
    predictions: Mapping[str, Any],
    errors: Mapping[str, Any],
    run_started_at: str,
    status: str,
    elapsed_seconds: float,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "run_started_at_utc": run_started_at,
            "updated_at_utc": now_iso(),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "benchmark": str(benchmark_path),
            "selected_questions": total_selected,
            "successful_predictions": len(predictions),
            "failed_questions": len(errors),
            "configuration": vars(args),
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()
    validate_args(args)

    run_started_at = now_iso()
    wall_started = time.time()

    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)
    jsonl_path = Path(args.jsonl_output)
    errors_path = Path(args.errors_output)
    run_log_path = Path(args.run_log)

    benchmark_metadata, all_samples = load_benchmark(
        benchmark_path
    )

    selected_samples = all_samples[args.start_index:]

    if args.limit is not None:
        selected_samples = selected_samples[:args.limit]

    if not selected_samples:
        raise ValueError("Không có câu hỏi nào trong phạm vi đã chọn.")

    benchmark_order = [
        safe_string(sample.get("id"))
        for sample in all_samples
    ]

    predictions: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, Any]] = {}

    if args.resume:
        predictions = load_existing_predictions(
            output_path,
            jsonl_path,
        )
        print(
            f"Resume: đã tìm thấy {len(predictions)} prediction."
        )
    else:
        # Bắt đầu run mới: xóa JSONL cũ để tránh duplicate.
        jsonl_path.unlink(missing_ok=True)

    if args.retry_errors and errors_path.exists():
        old_errors = read_json(errors_path)
        for item in old_errors.get("errors") or []:
            if isinstance(item, Mapping):
                sample_id = safe_string(item.get("id"))
                if sample_id:
                    errors[sample_id] = dict(item)

    runner = BatchPipelineRunner(args)

    total = len(selected_samples)
    success_since_checkpoint = 0

    print("\n" + "=" * 78)
    print("GENERATE BLHS PIPELINE PREDICTIONS")
    print("=" * 78)
    print("Benchmark :", benchmark_path)
    print("Questions :", total)
    print("Provider  :", args.provider)
    print("Output    :", output_path)
    print("=" * 78)

    for position, sample in enumerate(
        selected_samples,
        start=1,
    ):
        sample_id = safe_string(sample.get("id"))
        question = safe_string(sample.get("question"))

        if args.resume and sample_id in predictions:
            print(
                f"[{position}/{total}] SKIP {sample_id}: đã hoàn thành."
            )
            continue

        if (
            not args.retry_errors
            and sample_id in errors
        ):
            print(
                f"[{position}/{total}] SKIP {sample_id}: đã lỗi trước đó."
            )
            continue

        print("\n" + "-" * 78)
        print(f"[{position}/{total}] {sample_id}")
        print(question)
        print("-" * 78)

        question_started = time.time()

        try:
            prediction, _, _, _ = runner.run_one(sample)
            predictions[sample_id] = prediction
            errors.pop(sample_id, None)

            append_jsonl(jsonl_path, prediction)
            success_since_checkpoint += 1

            elapsed = time.time() - question_started
            print(
                f"SUCCESS {sample_id} | {elapsed:.2f}s | "
                f"rules={len(prediction['retrieved_rule_ids'])} | "
                f"events={len(prediction['retrieved_event_ids'])} | "
                f"decision={prediction['verification_decision']}"
            )

            if (
                success_since_checkpoint
                >= args.checkpoint_every
            ):
                save_predictions(
                    output_path=output_path,
                    benchmark_path=benchmark_path,
                    benchmark_metadata=benchmark_metadata,
                    predictions=predictions,
                    benchmark_order=benchmark_order,
                    errors=errors,
                    run_started_at=run_started_at,
                )
                save_errors(errors_path, errors)
                success_since_checkpoint = 0

        except KeyboardInterrupt:
            print("\nĐang lưu checkpoint trước khi dừng...")
            save_predictions(
                output_path=output_path,
                benchmark_path=benchmark_path,
                benchmark_metadata=benchmark_metadata,
                predictions=predictions,
                benchmark_order=benchmark_order,
                errors=errors,
                run_started_at=run_started_at,
            )
            save_errors(errors_path, errors)
            save_run_log(
                run_log_path,
                args=args,
                benchmark_path=benchmark_path,
                total_selected=total,
                predictions=predictions,
                errors=errors,
                run_started_at=run_started_at,
                status="INTERRUPTED",
                elapsed_seconds=time.time() - wall_started,
            )
            raise

        except Exception as exc:
            error_row = {
                "id": sample_id,
                "question": question,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": now_iso(),
                "elapsed_seconds": round(
                    time.time() - question_started,
                    4,
                ),
            }
            errors[sample_id] = error_row
            save_errors(errors_path, errors)

            print(
                f"ERROR {sample_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

            if args.fail_fast:
                raise

    save_predictions(
        output_path=output_path,
        benchmark_path=benchmark_path,
        benchmark_metadata=benchmark_metadata,
        predictions=predictions,
        benchmark_order=benchmark_order,
        errors=errors,
        run_started_at=run_started_at,
    )
    save_errors(errors_path, errors)

    # Chuẩn hóa lại JSONL, loại duplicate có thể sinh khi resume.
    rewrite_jsonl(
        jsonl_path,
        predictions,
        benchmark_order,
    )

    status = (
        "COMPLETED_WITH_ERRORS"
        if errors
        else "COMPLETED"
    )

    save_run_log(
        run_log_path,
        args=args,
        benchmark_path=benchmark_path,
        total_selected=total,
        predictions=predictions,
        errors=errors,
        run_started_at=run_started_at,
        status=status,
        elapsed_seconds=time.time() - wall_started,
    )

    print("\n" + "=" * 78)
    print("PIPELINE FINISHED")
    print("=" * 78)
    print("Status               :", status)
    print("Successful prediction:", len(predictions))
    print("Errors               :", len(errors))
    print("Prediction JSON      :", output_path)
    print("Prediction JSONL     :", jsonl_path)
    print("Errors JSON          :", errors_path)
    print("Run log              :", run_log_path)
    print("=" * 78)

    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nLỖI: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1)
