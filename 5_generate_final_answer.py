#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_generate_final_answer.py

Sinh câu trả lời cuối cùng từ:
    - kết quả retrieval của bước 3;
    - kết quả counterfactual verification của bước 4.

Phiên bản 5.1 tiêu thụ trực tiếp schema query-aware của Step 4 v4.1,
đồng thời giữ fallback có kiểm soát cho artifact cũ.

Các nguyên tắc chính:
    - Evidence thuộc primary path được chọn trước và giữ đúng thứ tự rule trên path.
    - Answer intent được suy ra từ chính câu hỏi, không đọc question_type/gold label:
      hệ quả cuối, nguyên nhân khởi đầu, cầu nối, giải thích chuỗi, yes/no,
      phân nhánh, hội tụ hoặc xác minh claim direct/counterfactual.
    - Claim phản thực tế dùng world-state/intervention semantics của Step 4;
      node-deletion không được trình bày như hard do-intervention.
    - Prompt truyền final_decision, query_analysis và answer_intent để LLM không
      kết luận trái Step 4 hoặc trả sai loại thông tin.
    - Citation extractive được khử trùng theo causal edge và ưu tiên primary path.
    - Giữ nguyên API class/hàm mà 5_5_generate_pipeline_predictions.py đang gọi.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib import error, request


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

VERIFICATION_RESULT_PATH = "data/counterfactual_verification_result.json"
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

# Ranking evidence bổ sung. Primary-path evidence luôn được ưu tiên trước.
FINAL_EVIDENCE_SCORE_WEIGHT = 0.50
ORIGINAL_RETRIEVAL_SCORE_WEIGHT = 0.25
COUNTERFACTUAL_SUPPORT_WEIGHT = 0.15
PRIMARY_PATH_BONUS_WEIGHT = 0.10

SUPPORTED = "SUPPORTED"
REJECT_DIRECT_CLAIM = "REJECT_DIRECT_CLAIM"
UNCERTAIN = "UNCERTAIN"

STEP5_VERSION = "5.1-query-aware-grounded-answer"

ANSWER_INTENT_FINAL_OUTCOME = "FINAL_OUTCOME"
ANSWER_INTENT_INITIAL_CAUSE = "INITIAL_CAUSE"
ANSWER_INTENT_BRIDGE_EVENT = "BRIDGE_EVENT"
ANSWER_INTENT_CHAIN_EXPLANATION = "CHAIN_EXPLANATION"
ANSWER_INTENT_YES_NO = "YES_NO"
ANSWER_INTENT_BRANCH = "BRANCH"
ANSWER_INTENT_CONVERGENCE = "CONVERGENCE"
ANSWER_INTENT_CLAIM_VERIFICATION = "CLAIM_VERIFICATION"

COUNTERFACTUAL_CLAIM_TYPES = {
    "DIRECT_CAUSAL_CLAIM",
    "REMOVE_MEDIATOR_OUTCOME_REMAINS",
    "REMOVE_MEDIATOR_OUTCOME_DISAPPEARS",
    "MEDIATOR_NECESSARY_CLAIM",
    "COUNTERFACTUAL_UNSPECIFIED",
}


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

    is_primary_path_evidence: bool = False
    primary_path_position: int = -1


@dataclass
class FinalPath:
    path_index: int
    original_path_id: int
    status: str
    consistency_score: float
    path_score: float
    hop_count: int

    seed_event_id: str
    seed_event_name: str
    outcome_event_id: str
    outcome_event_name: str

    explanation: str
    event_ids: list[str] = field(default_factory=list)
    event_names: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    article_ids: list[str] = field(default_factory=list)

    is_primary: bool = False


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

    final_decision: str
    decision_score: float
    decision_explanation: str
    primary_path_ids: list[int]
    query_analysis: dict[str, Any]
    answer_intent: str

    generation_metadata: dict[str, Any]


# ============================================================
# GENERAL HELPERS
# ============================================================

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


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def unique_preserve_order(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = safe_string(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def unique_ints(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()

    for value in values:
        number = safe_int(value, -1)
        if number >= 0 and number not in seen:
            seen.add(number)
            result.append(number)

    return result


def normalize_decision(value: Any) -> str:
    text = safe_string(value).upper().replace(" ", "_")

    aliases = {
        "KEEP": SUPPORTED,
        "VERIFIED": SUPPORTED,
        "ACCEPT": SUPPORTED,
        "ACCEPTED": SUPPORTED,
        "YES": SUPPORTED,
        "TRUE": SUPPORTED,
        "SUPPORTED": SUPPORTED,
        "REMOVE": REJECT_DIRECT_CLAIM,
        "REJECT": REJECT_DIRECT_CLAIM,
        "REJECTED": REJECT_DIRECT_CLAIM,
        "CONTRADICTED": REJECT_DIRECT_CLAIM,
        "NO": REJECT_DIRECT_CLAIM,
        "FALSE": REJECT_DIRECT_CLAIM,
        "REJECT_DIRECT_CLAIM": REJECT_DIRECT_CLAIM,
        "UNRESOLVED": UNCERTAIN,
        "INSUFFICIENT": UNCERTAIN,
        "UNCERTAIN": UNCERTAIN,
    }

    return aliases.get(text, text or UNCERTAIN)


def infer_answer_intent(
    query: Any,
    query_analysis: Optional[Mapping[str, Any]] = None,
) -> str:
    """Suy ra dạng câu trả lời chỉ từ query và phân tích claim của Step 4.

    Không dùng ``question_type`` hoặc gold label của benchmark. Claim phản
    thực tế/direct-edge luôn ưu tiên semantics Step 4; các intent còn lại chỉ
    quyết định cách trình bày cùng một primary causal path.
    """

    analysis = as_mapping(query_analysis)
    claim_type = safe_string(
        analysis.get("claim_type")
    ).upper() or "STANDARD_CAUSAL"
    if claim_type in COUNTERFACTUAL_CLAIM_TYPES:
        return ANSWER_INTENT_CLAIM_VERIFICATION

    normalized = " ".join(safe_string(query).lower().split())

    if any(
        marker in normalized
        for marker in (
            "phân nhánh",
            "những hệ quả pháp lý nào",
            "các nhánh",
        )
    ):
        return ANSWER_INTENT_BRANCH

    if any(
        marker in normalized
        for marker in (
            "hội tụ",
            "những điều kiện ban đầu nào",
            "các điều kiện hội tụ",
        )
    ):
        return ANSWER_INTENT_CONVERGENCE

    if any(
        marker in normalized
        for marker in (
            "điều kiện ban đầu nào",
            "sự kiện khởi đầu",
            "suy ngược",
            "bắt nguồn từ sự kiện nào",
            "truy ngược",
            "sự kiện đầu tiên",
        )
    ):
        return ANSWER_INTENT_INITIAL_CAUSE

    if any(
        marker in normalized
        for marker in (
            "sự kiện nào là cầu nối",
            "bước pháp lý trung gian",
            "sự kiện trung gian nào",
            "mắt xích còn thiếu",
            "vừa là kết quả của bước thứ nhất",
        )
    ):
        return ANSWER_INTENT_BRIDGE_EVENT

    if any(
        marker in normalized
        for marker in (
            "hãy giải thích chuỗi",
            "trình bày hai bước",
            "mô tả đường suy luận",
            "vì sao theo graph",
            "nêu đầy đủ mắt xích",
        )
    ):
        return ANSWER_INTENT_CHAIN_EXPLANATION

    if any(
        marker in normalized
        for marker in (
            "hay không",
            "đúng hay sai",
            "có tồn tại chuỗi",
            "nhận định sau",
        )
    ):
        return ANSWER_INTENT_YES_NO

    return ANSWER_INTENT_FINAL_OUTCOME


def load_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"File phải chứa JSON object: {file_path}")

    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(file_path)
    print(f"Saved final answer: {file_path}")


def truncate_text(text: str, max_chars: int) -> str:
    text = safe_string(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def normalize_article_label(article_id: str, article_title: str) -> str:
    article_id = safe_string(article_id)
    article_title = safe_string(article_title)

    if article_id and article_title:
        return f"Điều {article_id} – {article_title}"
    if article_id:
        return f"Điều {article_id}"
    if article_title:
        return article_title
    return "Không xác định điều luật"


def as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


# ============================================================
# INPUT STORE
# ============================================================

class FinalAnswerInputStore:
    """Nạp và chuẩn hóa output của Step 3/Step 4.

    Lớp hỗ trợ cả schema Step 4 mới và schema cũ. Khi Step 4 chưa có
    ``final_decision`` hoặc ``primary_path_ids``, các giá trị này được suy ra
    có kiểm soát từ path_verifications.
    """

    def __init__(
        self,
        *,
        verification_result_path: str,
        retrieval_result_path: str,
    ) -> None:
        self.verification_result_path = Path(verification_result_path)
        self.retrieval_result_path = Path(retrieval_result_path)

        self.verification_result = self._load_verification_result()
        self.retrieval_result = self._load_retrieval_result_optional()
        self._validate()

    def _load_verification_result(self) -> dict[str, Any]:
        print("Loading verification result:", self.verification_result_path)
        return load_json(self.verification_result_path)

    def _load_retrieval_result_optional(self) -> dict[str, Any]:
        if not self.retrieval_result_path.exists():
            print(
                "Warning: retrieval result không tồn tại. "
                "Chỉ dùng dữ liệu trong verification result."
            )
            return {}

        print("Loading retrieval result:", self.retrieval_result_path)
        return load_json(self.retrieval_result_path)

    def _validate(self) -> None:
        required = {
            "query",
            "verified_evidence",
            "uncertain_evidence",
            "removed_evidence",
            "path_verifications",
        }
        missing = required - set(self.verification_result)
        if missing:
            raise ValueError(
                "Verification result thiếu trường: "
                f"{sorted(missing)}"
            )

        for key in (
            "verified_evidence",
            "uncertain_evidence",
            "removed_evidence",
            "path_verifications",
        ):
            if not isinstance(self.verification_result.get(key), list):
                raise ValueError(f"`{key}` phải là list.")

    @property
    def query(self) -> str:
        return safe_string(self.verification_result.get("query"))

    @property
    def confidence(self) -> float:
        return safe_float(self.verification_result.get("confidence"))

    @property
    def consistency_score(self) -> float:
        return safe_float(self.verification_result.get("consistency_score"))

    @property
    def final_decision(self) -> str:
        explicit = self.verification_result.get("final_decision")
        if explicit:
            return normalize_decision(explicit)

        query_analysis = as_mapping(
            self.verification_result.get("query_analysis")
        )
        if query_analysis.get("decision"):
            return normalize_decision(query_analysis.get("decision"))

        # Fallback tương thích Step 4 cũ.
        primary_paths = self.primary_path_ids
        path_by_id = {
            safe_int(item.get("original_path_id"), -1): item
            for item in self.path_verifications
        }

        primary_statuses = [
            safe_string(path_by_id[path_id].get("status")).upper()
            for path_id in primary_paths
            if path_id in path_by_id
        ]

        if primary_statuses:
            if all(status == "CONTRADICTED" for status in primary_statuses):
                return REJECT_DIRECT_CLAIM
            if any(status == "SUPPORTED" for status in primary_statuses):
                return SUPPORTED
            return UNCERTAIN

        status_counts = as_mapping(
            as_mapping(self.verification_result.get("statistics")).get(
                "path_status_counts"
            )
            or as_mapping(self.verification_result.get("statistics")).get(
                "status_counts"
            )
        )

        supported = safe_int(status_counts.get("SUPPORTED"))
        contradicted = safe_int(status_counts.get("CONTRADICTED"))

        if supported > 0:
            return SUPPORTED
        if contradicted > 0:
            return REJECT_DIRECT_CLAIM
        return UNCERTAIN

    @property
    def decision_score(self) -> float:
        explicit = self.verification_result.get("decision_score")
        if explicit is not None:
            return clamp(safe_float(explicit))

        query_analysis = as_mapping(
            self.verification_result.get("query_analysis")
        )
        for key in ("decision_score", "score", "confidence"):
            if query_analysis.get(key) is not None:
                return clamp(safe_float(query_analysis.get(key)))

        return clamp(max(self.confidence, self.consistency_score))

    @property
    def decision_explanation(self) -> str:
        explicit = safe_string(
            self.verification_result.get("decision_explanation")
        )
        if explicit:
            return explicit

        query_analysis = as_mapping(
            self.verification_result.get("query_analysis")
        )
        explanation = safe_string(query_analysis.get("explanation"))
        if explanation:
            return explanation

        primary_ids = set(self.primary_path_ids)
        primary_explanations = [
            safe_string(item.get("explanation"))
            for item in self.path_verifications
            if safe_int(item.get("original_path_id"), -1) in primary_ids
            and safe_string(item.get("explanation"))
        ]
        return " | ".join(primary_explanations)

    @property
    def query_analysis(self) -> dict[str, Any]:
        analysis = as_mapping(self.verification_result.get("query_analysis"))
        if analysis:
            return analysis
        return {
            "claim_type": "STANDARD_CAUSAL",
            "decision": self.final_decision,
            "decision_score": self.decision_score,
            "explanation": self.decision_explanation,
            "inferred_by_step5": True,
        }

    @property
    def answer_intent(self) -> str:
        return infer_answer_intent(self.query, self.query_analysis)

    @property
    def step4_version(self) -> str:
        configuration = as_mapping(
            self.verification_result.get("configuration")
        )
        return safe_string(configuration.get("step4_version"))

    @property
    def path_verifications(self) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.verification_result.get("path_verifications", [])
            if isinstance(item, Mapping)
        ]

    @property
    def primary_path_ids(self) -> list[int]:
        explicit = unique_ints(
            self.verification_result.get("primary_path_ids", [])
        )
        if explicit:
            return explicit

        query_analysis = as_mapping(
            self.verification_result.get("query_analysis")
        )
        explicit = unique_ints(query_analysis.get("primary_path_ids", []))
        if explicit:
            return explicit

        candidates = self.path_verifications
        if not candidates:
            return []

        # Ưu tiên SUPPORTED, path đủ 2 hop, consistency cao, graph score cao.
        def rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
            status = safe_string(item.get("status")).upper()
            hop_count = safe_int(
                item.get("original_hop_count"),
                len(item.get("original_event_ids", [])) - 1,
            )
            consistency = safe_float(item.get("consistency_score"))
            path_score = safe_float(item.get("original_path_score"))
            return (
                status == "SUPPORTED",
                hop_count == 2,
                hop_count,
                consistency,
                path_score,
                -safe_int(item.get("original_path_id"), 10**9),
            )

        best = max(candidates, key=rank_key)
        best_id = safe_int(best.get("original_path_id"), -1)
        return [best_id] if best_id >= 0 else []

    def retrieval_path(self, path_id: int) -> dict[str, Any]:
        paths = self.retrieval_result.get("causal_paths", [])
        if isinstance(paths, list) and 0 <= path_id < len(paths):
            item = paths[path_id]
            return dict(item) if isinstance(item, Mapping) else {}
        return {}

    def verification_path(self, path_id: int) -> dict[str, Any]:
        for item in self.path_verifications:
            if safe_int(item.get("original_path_id"), -1) == path_id:
                return item
        return {}


# ============================================================
# EVIDENCE AND PATH SELECTION
# ============================================================

class FinalContextSelector:
    def __init__(self, store: FinalAnswerInputStore) -> None:
        self.store = store

    def select_paths(
        self,
        *,
        selected_evidence: Optional[list[FinalEvidence]] = None,
        max_paths: int,
    ) -> list[FinalPath]:
        if max_paths <= 0:
            return []

        selected_evidence = selected_evidence or []
        primary_ids = self.store.primary_path_ids
        evidence_path_ids = unique_ints(
            path_id
            for evidence in selected_evidence
            for path_id in evidence.path_ids
        )

        ordered_ids = unique_ints(
            primary_ids
            + evidence_path_ids
            + [
                item.get("original_path_id")
                for item in self.store.path_verifications
            ]
        )

        primary_set = set(primary_ids)
        paths: list[FinalPath] = []

        for path_id in ordered_ids:
            verification = self.store.verification_path(path_id)
            retrieval = self.store.retrieval_path(path_id)
            if not verification and not retrieval:
                continue

            event_ids = self._extract_event_ids(verification, retrieval)
            event_names = self._extract_event_names(verification, retrieval)
            rule_ids = self._extract_rule_ids(verification, retrieval)
            article_ids = self._extract_article_ids(verification, retrieval)

            hop_count = safe_int(
                verification.get("original_hop_count"),
                max(0, len(event_ids) - 1),
            )

            path = FinalPath(
                path_index=0,
                original_path_id=path_id,
                status=safe_string(verification.get("status")) or "UNRESOLVED",
                consistency_score=safe_float(
                    verification.get("consistency_score")
                ),
                path_score=safe_float(
                    verification.get("original_path_score"),
                    safe_float(retrieval.get("graph_score")),
                ),
                hop_count=hop_count,
                seed_event_id=safe_string(
                    verification.get("seed_event_id")
                ) or (event_ids[0] if event_ids else ""),
                seed_event_name=safe_string(
                    verification.get("seed_event_name")
                ) or (event_names[0] if event_names else ""),
                outcome_event_id=safe_string(
                    verification.get("original_outcome_event_id")
                ) or (event_ids[-1] if event_ids else ""),
                outcome_event_name=safe_string(
                    verification.get("original_outcome_event_name")
                ) or (event_names[-1] if event_names else ""),
                explanation=safe_string(verification.get("explanation")),
                event_ids=event_ids,
                event_names=event_names,
                rule_ids=rule_ids,
                article_ids=article_ids,
                is_primary=path_id in primary_set,
            )
            paths.append(path)

        primary_path = next(
            (item for item in paths if item.is_primary),
            paths[0] if paths else None,
        )
        answer_intent = self.store.answer_intent

        def structure_key(path: FinalPath, *, prefix: bool) -> tuple[str, ...]:
            values = path.event_ids or path.event_names
            normalized = tuple(safe_string(value).lower() for value in values)
            return normalized[:2] if prefix else normalized[-2:]

        def matches_answer_structure(path: FinalPath) -> bool:
            if primary_path is None:
                return False
            if answer_intent == ANSWER_INTENT_BRANCH:
                candidate = structure_key(path, prefix=True)
                reference = structure_key(primary_path, prefix=True)
                return (
                    len(candidate) == 2
                    and len(reference) == 2
                    and candidate == reference
                )
            if answer_intent == ANSWER_INTENT_CONVERGENCE:
                candidate = structure_key(path, prefix=False)
                reference = structure_key(primary_path, prefix=False)
                return (
                    len(candidate) == 2
                    and len(reference) == 2
                    and candidate == reference
                )
            return False

        paths.sort(
            key=lambda item: (
                item.is_primary,
                matches_answer_structure(item),
                item.status.upper() == "SUPPORTED",
                item.hop_count == 2,
                item.hop_count,
                item.consistency_score,
                item.path_score,
            ),
            reverse=True,
        )

        paths = paths[:max_paths]
        for index, path in enumerate(paths, start=1):
            path.path_index = index
        return paths

    def select_evidence(
        self,
        *,
        max_evidence: int,
        min_verification_score: float,
        include_uncertain: bool,
        selected_paths: Optional[list[FinalPath]] = None,
    ) -> list[FinalEvidence]:
        if max_evidence < 1:
            return []

        selected_paths = selected_paths or self.select_paths(max_paths=1)
        primary_paths = [path for path in selected_paths if path.is_primary]
        if not primary_paths and selected_paths:
            primary_paths = [selected_paths[0]]

        required_paths = list(primary_paths)
        if (
            primary_paths
            and self.store.answer_intent
            in {ANSWER_INTENT_BRANCH, ANSWER_INTENT_CONVERGENCE}
        ):
            reference = primary_paths[0]
            use_prefix = self.store.answer_intent == ANSWER_INTENT_BRANCH

            def structure_key(path: FinalPath) -> tuple[str, ...]:
                values = path.event_ids or path.event_names
                normalized = tuple(
                    safe_string(value).lower() for value in values
                )
                return normalized[:2] if use_prefix else normalized[-2:]

            reference_key = structure_key(reference)
            if len(reference_key) == 2:
                required_paths = [
                    path
                    for path in selected_paths
                    if (
                        safe_string(path.status).upper() == SUPPORTED
                        and structure_key(path) == reference_key
                    )
                ]
                if reference not in required_paths:
                    required_paths.insert(0, reference)

        primary_rule_order = unique_preserve_order(
            rule_id
            for path in required_paths
            for rule_id in path.rule_ids
        )
        primary_path_ids = {
            path.original_path_id for path in required_paths
        }

        verification_items = self._verification_evidence_items(
            include_uncertain=include_uncertain
        )
        item_by_rule: dict[str, dict[str, Any]] = {}

        for item in verification_items:
            rule_id = safe_string(
                item.get("rule_id")
                or as_mapping(item.get("original_evidence")).get("rule_id")
            )
            if not rule_id:
                continue

            existing = item_by_rule.get(rule_id)
            if existing is None or safe_float(item.get("verification_score")) > safe_float(
                existing.get("verification_score")
            ):
                item_by_rule[rule_id] = item

        retrieval_by_rule = {
            safe_string(item.get("rule_id")): dict(item)
            for item in self.store.retrieval_result.get("evidence", [])
            if isinstance(item, Mapping) and safe_string(item.get("rule_id"))
        }

        selected: list[FinalEvidence] = []
        selected_rules: set[str] = set()

        # 1. Luôn lấy rule thuộc primary path theo đúng thứ tự causal.
        for position, rule_id in enumerate(primary_rule_order):
            verification_item = item_by_rule.get(rule_id)
            if verification_item is not None:
                evidence = self._convert_verified_evidence(
                    verification_item,
                    default_decision="KEEP",
                )
            else:
                raw = retrieval_by_rule.get(rule_id)
                if raw is None:
                    continue
                evidence = self._convert_retrieval_evidence(
                    raw,
                    primary_path_ids=primary_path_ids,
                )

            evidence.is_primary_path_evidence = True
            evidence.primary_path_position = position
            evidence.final_selection_score = clamp(
                evidence.final_selection_score + PRIMARY_PATH_BONUS_WEIGHT
            )
            selected.append(evidence)
            selected_rules.add(rule_id)

        # 2. Evidence bổ sung đã xác minh.
        supplements: list[FinalEvidence] = []
        for item in verification_items:
            evidence = self._convert_verified_evidence(
                item,
                default_decision="KEEP",
            )
            if not evidence.rule_id or evidence.rule_id in selected_rules:
                continue
            if evidence.verification_score < min_verification_score:
                continue
            if evidence.decision.upper() == "REMOVE":
                continue
            supplements.append(evidence)

        supplements.sort(
            key=lambda item: (
                item.final_selection_score,
                item.verification_score,
                item.original_final_score,
            ),
            reverse=True,
        )

        for evidence in supplements:
            if len(selected) >= max_evidence:
                break
            selected.append(evidence)
            selected_rules.add(evidence.rule_id)

        # 3. Fallback nếu Step 4 không giữ được evidence nào.
        if not selected:
            raw_candidates = [
                dict(item)
                for item in self.store.retrieval_result.get("evidence", [])
                if isinstance(item, Mapping)
            ]
            raw_candidates.sort(
                key=lambda item: safe_float(item.get("final_score")),
                reverse=True,
            )
            for raw in raw_candidates[:max_evidence]:
                selected.append(
                    self._convert_retrieval_evidence(
                        raw,
                        primary_path_ids=primary_path_ids,
                    )
                )

        # Primary-path evidence có thể vượt max_evidence nếu path dài hơn giới hạn.
        # Với benchmark hiện tại path thường 2 hop; vẫn bảo toàn tất cả rule chính.
        if len(selected) > max_evidence and len(primary_rule_order) <= max_evidence:
            selected = selected[:max_evidence]

        for index, evidence in enumerate(selected, start=1):
            evidence.evidence_index = index

        return selected

    def _verification_evidence_items(
        self,
        *,
        include_uncertain: bool,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        for raw in self.store.verification_result.get("verified_evidence", []):
            if isinstance(raw, Mapping):
                item = dict(raw)
                item.setdefault("decision", "KEEP")
                items.append(item)

        if include_uncertain:
            for raw in self.store.verification_result.get("uncertain_evidence", []):
                if isinstance(raw, Mapping):
                    item = dict(raw)
                    item.setdefault("decision", "UNCERTAIN")
                    items.append(item)

        return items

    def _convert_verified_evidence(
        self,
        item: dict[str, Any],
        *,
        default_decision: str,
    ) -> FinalEvidence:
        original = as_mapping(item.get("original_evidence"))

        verification_score = safe_float(item.get("verification_score"))
        original_final_score = safe_float(
            item.get("original_final_score"),
            safe_float(original.get("final_score")),
        )
        counterfactual_support_score = safe_float(
            item.get("counterfactual_support_score")
        )

        path_ids = unique_ints(
            list(item.get("verified_path_ids", []))
            + list(item.get("unresolved_path_ids", []))
            + list(original.get("path_ids", []))
        )

        primary_set = set(self.store.primary_path_ids)
        is_primary = bool(primary_set & set(path_ids))

        final_selection_score = clamp(
            FINAL_EVIDENCE_SCORE_WEIGHT * verification_score
            + ORIGINAL_RETRIEVAL_SCORE_WEIGHT * original_final_score
            + COUNTERFACTUAL_SUPPORT_WEIGHT * counterfactual_support_score
            + PRIMARY_PATH_BONUS_WEIGHT * float(is_primary)
        )

        return FinalEvidence(
            evidence_index=0,
            rule_id=safe_string(item.get("rule_id") or original.get("rule_id")),
            article_id=safe_string(
                item.get("article_id") or original.get("article_id")
            ),
            article_title=safe_string(original.get("article_title")),
            legal_subject=safe_string(original.get("legal_subject")),
            condition=safe_string(original.get("condition")),
            effect=safe_string(original.get("effect")),
            condition_event=safe_string(original.get("condition_event")),
            condition_event_name=safe_string(
                original.get("condition_event_name")
            ),
            effect_event=safe_string(original.get("effect_event")),
            effect_event_name=safe_string(original.get("effect_event_name")),
            causal_type=safe_string(original.get("causal_type")),
            verification_score=verification_score,
            original_final_score=original_final_score,
            counterfactual_support_score=counterfactual_support_score,
            final_selection_score=final_selection_score,
            decision=safe_string(item.get("decision")) or default_decision,
            path_ids=path_ids,
            reasons=[
                safe_string(reason)
                for reason in item.get("reasons", [])
                if safe_string(reason)
            ],
            is_primary_path_evidence=is_primary,
        )

    def _convert_retrieval_evidence(
        self,
        raw: dict[str, Any],
        *,
        primary_path_ids: set[int],
    ) -> FinalEvidence:
        path_ids = unique_ints(raw.get("path_ids", []))
        is_primary = bool(primary_path_ids & set(path_ids))
        original_score = safe_float(raw.get("final_score"))
        graph_score = safe_float(raw.get("graph_score"))

        return FinalEvidence(
            evidence_index=0,
            rule_id=safe_string(raw.get("rule_id")),
            article_id=safe_string(raw.get("article_id")),
            article_title=safe_string(raw.get("article_title")),
            legal_subject=safe_string(raw.get("legal_subject")),
            condition=safe_string(raw.get("condition")),
            effect=safe_string(raw.get("effect")),
            condition_event=safe_string(raw.get("condition_event")),
            condition_event_name=safe_string(raw.get("condition_event_name")),
            effect_event=safe_string(raw.get("effect_event")),
            effect_event_name=safe_string(raw.get("effect_event_name")),
            causal_type=safe_string(raw.get("causal_type")),
            verification_score=graph_score,
            original_final_score=original_score,
            counterfactual_support_score=self.store.consistency_score,
            final_selection_score=clamp(
                0.50 * original_score
                + 0.30 * graph_score
                + PRIMARY_PATH_BONUS_WEIGHT * float(is_primary)
            ),
            decision="UNCERTAIN",
            path_ids=path_ids,
            reasons=["Fallback từ retrieval evidence vì Step 4 không có record tương ứng."],
            is_primary_path_evidence=is_primary,
        )

    @staticmethod
    def _extract_event_ids(
        verification: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[str]:
        values = verification.get("original_event_ids") or retrieval.get("event_ids")
        if values:
            return unique_preserve_order(values)

        event_nodes = retrieval.get("event_nodes", [])
        return unique_preserve_order(
            safe_string(node).removeprefix("EVENT::") for node in event_nodes
        )

    @staticmethod
    def _extract_event_names(
        verification: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[str]:
        values = verification.get("original_event_names")
        if values:
            return unique_preserve_order(values)

        steps = retrieval.get("steps", [])
        names: list[str] = []
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, Mapping):
                    continue
                names.extend(
                    [
                        step.get("source_event_name"),
                        step.get("target_event_name"),
                    ]
                )
        return unique_preserve_order(names)

    @staticmethod
    def _extract_rule_ids(
        verification: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[str]:
        return unique_preserve_order(
            verification.get("original_rule_ids")
            or retrieval.get("rule_ids", [])
        )

    @staticmethod
    def _extract_article_ids(
        verification: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> list[str]:
        values = verification.get("original_article_ids") or retrieval.get(
            "article_ids", []
        )
        if values:
            return unique_preserve_order(values)

        article_ids: list[Any] = []
        for step in retrieval.get("steps", []):
            if isinstance(step, Mapping):
                article_ids.extend(step.get("article_ids", []))
        return unique_preserve_order(article_ids)


# ============================================================
# PROMPT BUILDER
# ============================================================

class LegalAnswerPromptBuilder:
    SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp pháp luật Việt Nam.

Chỉ trả lời dựa trên evidence, causal path và kết quả xác minh được cung cấp.

Quy tắc bắt buộc:
1. Không bổ sung điều luật, hình phạt, điều kiện hoặc ngoại lệ không có trong evidence.
2. Không đưa ra kết luận trái với FINAL_DECISION.
3. Tuân thủ ANSWER_INTENT: trả đúng loại thông tin được hỏi (hệ quả, nguyên nhân khởi đầu, sự kiện cầu nối, chuỗi, yes/no, phân nhánh, hội tụ hoặc xác minh claim).
4. Với claim direct/counterfactual, diễn đạt theo query_analysis và world-state; không thay bằng hệ quả thực tế chung chung.
5. Không gọi node-deletion reachability là can thiệp do hoặc kết quả cấu trúc.
6. Với FINAL_DECISION=REJECT_DIRECT_CLAIM, nêu rõ claim trong câu hỏi không được dữ liệu hỗ trợ.
7. Với FINAL_DECISION=UNCERTAIN, phải nói rõ chưa đủ căn cứ và không suy đoán nhị phân.
8. Mỗi nhận định pháp lý quan trọng phải gắn citation [E1], [E2], ... có trong ngữ cảnh.
9. Ưu tiên evidence thuộc PRIMARY PATH và giữ đúng thứ tự chuỗi.
10. Không sử dụng causal path như nguồn luật độc lập; path chỉ thể hiện thứ tự suy luận.
11. Trả lời ngắn gọn, trực tiếp bằng tiếng Việt. Không nhắc tên file, JSON, pipeline hay điểm số kỹ thuật.
12. Không tạo citation hoặc tài liệu ngoài danh sách evidence."""

    def build(
        self,
        *,
        query: str,
        evidence: list[FinalEvidence],
        paths: list[FinalPath],
        final_decision: str,
        decision_score: float,
        decision_explanation: str,
        query_analysis: dict[str, Any],
        answer_intent: str,
        global_confidence: float,
        consistency_score: float,
        max_context_chars: int,
    ) -> tuple[str, str]:
        user_prompt = f"""CÂU HỎI:
{query}

KẾT QUẢ XÁC MINH CLAIM:
- FINAL_DECISION: {final_decision}
- ANSWER_INTENT: {answer_intent}
- Decision score: {decision_score:.4f}
- Claim type: {safe_string(query_analysis.get('claim_type')) or 'STANDARD_CAUSAL'}
- Giải thích: {decision_explanation or 'Không có giải thích bổ sung'}

NGỮ NGHĨA CAN THIỆP/CLAIM TỪ STEP 4:
- Intent detector: {safe_string(query_analysis.get('intent_detection_version')) or 'legacy/unknown'}
- Mediator: {safe_string(query_analysis.get('matched_mediator_name') or query_analysis.get('matched_mediator_id')) or 'Không áp dụng'}
- Trạng thái mediator thực tế: {safe_string(query_analysis.get('factual_mediator_state')) or 'Không áp dụng'}
- Outcome thực tế: {safe_string(query_analysis.get('factual_outcome')) or 'Không áp dụng'}
- Outcome sau can thiệp: {safe_string(query_analysis.get('counterfactual_outcome')) or 'Không áp dụng'}
- Trạng thái intervention: {safe_string(query_analysis.get('intervention_status') or query_analysis.get('structural_status')) or 'Không áp dụng'}
- Nguồn tín hiệu: {safe_string(query_analysis.get('counterfactual_signal_source') or query_analysis.get('path_verification_method')) or 'Không áp dụng'}
- Structural result used: {bool(query_analysis.get('structural_result_used'))}
- Structural fallback used: {bool(query_analysis.get('structural_fallback_used'))}

ĐỘ TIN CẬY TOÀN CỤC:
- confidence = {global_confidence:.4f}
- consistency_score = {consistency_score:.4f}

EVIDENCE ĐÃ CHỌN:
{self._build_evidence_context(evidence)}

CAUSAL PATH ĐÃ CHỌN:
{self._build_path_context(paths)}

YÊU CẦU ĐẦU RA:
- Trả đúng ANSWER_INTENT, không mặc định mọi câu hỏi đều hỏi hệ quả cuối.
- INITIAL_CAUSE: chỉ ra sự kiện khởi đầu và chuỗi đầy đủ theo chiều nhân quả.
- BRIDGE_EVENT: chỉ ra sự kiện trung gian của PRIMARY PATH.
- CHAIN_EXPLANATION: trình bày từng hop theo đúng thứ tự và căn cứ tương ứng.
- YES_NO hoặc CLAIM_VERIFICATION: mở đầu bằng Có/Không/Chưa đủ căn cứ phù hợp FINAL_DECISION.
- BRANCH/CONVERGENCE: chỉ liệt kê các nhánh/path thực sự có trong context.
- FINAL_OUTCOME: ưu tiên tên outcome event của PRIMARY PATH; dùng effect của rule cuối để bổ sung nếu cần.
- Chỉ dùng citation của evidence thực sự được dùng; không liệt kê toàn bộ evidence nếu không cần thiết.
"""

        return self.SYSTEM_PROMPT, truncate_text(user_prompt, max_context_chars)

    @staticmethod
    def _build_evidence_context(evidence: list[FinalEvidence]) -> str:
        if not evidence:
            return "Không có evidence pháp lý đạt yêu cầu."

        blocks: list[str] = []
        for item in evidence:
            marker = "PRIMARY_PATH" if item.is_primary_path_evidence else "SUPPLEMENTAL"
            blocks.append(
                f"""[E{item.evidence_index}] ({marker})
- Rule ID: {item.rule_id}
- Căn cứ: {normalize_article_label(item.article_id, item.article_title)}
- Điều kiện: {item.condition or 'Không nêu rõ'}
- Hệ quả: {item.effect or 'Không nêu rõ'}
- Condition event: {item.condition_event_name or item.condition_event or 'Không nêu rõ'}
- Effect event: {item.effect_event_name or item.effect_event or 'Không nêu rõ'}
- Trạng thái evidence: {item.decision}"""
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_path_context(paths: list[FinalPath]) -> str:
        if not paths:
            return "Không có causal path phù hợp."

        blocks: list[str] = []
        for item in paths:
            chain = " → ".join(item.event_names or item.event_ids)
            blocks.append(
                f"""[P{item.path_index}] {'PRIMARY' if item.is_primary else 'SUPPORTING'}
- Path ID: {item.original_path_id}
- Trạng thái: {item.status}
- Số hop: {item.hop_count}
- Chuỗi: {chain or item.seed_event_name + ' → ' + item.outcome_event_name}
- Rule IDs theo thứ tự: {', '.join(item.rule_ids) or 'Không nêu rõ'}
- Giải thích: {item.explanation or 'Không có'}"""
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

    def __init__(self, *, base_url: str) -> None:
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
        response = _http_post_json(
            url=f"{self.base_url}/api/chat",
            payload={
                "model": model,
                "stream": False,
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
        answer = safe_string(response.get("message", {}).get("content"))
        if not answer:
            raise RuntimeError("Ollama không trả về nội dung.")
        return answer


class OpenAICompatibleProvider(BaseLLMProvider):
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

        response = _http_post_json(
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

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI-compatible API không trả choices.")
        answer = safe_string(choices[0].get("message", {}).get("content"))
        if not answer:
            raise RuntimeError("OpenAI-compatible API trả nội dung rỗng.")
        return answer


class GeminiProvider(BaseLLMProvider):
    provider_name = "gemini"

    def __init__(self, *, api_key: str) -> None:
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

        response = _http_post_json(
            url=(
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model.strip()}:generateContent?key={self.api_key}"
            ),
            payload={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]}
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            timeout=timeout,
        )

        candidates = response.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini không trả candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(
            safe_string(part.get("text"))
            for part in parts
            if safe_string(part.get("text"))
        )
        if not answer:
            raise RuntimeError("Gemini trả nội dung rỗng.")
        return answer


class ExtractiveFallbackProvider(BaseLLMProvider):
    provider_name = "extractive"

    def __init__(
        self,
        *,
        evidence: list[FinalEvidence],
        paths: list[FinalPath],
        final_decision: str,
        decision_explanation: str,
        confidence: float,
        query_analysis: Mapping[str, Any],
        answer_intent: str,
    ) -> None:
        self.evidence = evidence
        self.paths = paths
        self.final_decision = normalize_decision(final_decision)
        self.decision_explanation = safe_string(decision_explanation)
        self.confidence = confidence
        self.query_analysis = as_mapping(query_analysis)
        self.answer_intent = safe_string(answer_intent).upper()

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
        del system_prompt, user_prompt, model, temperature, max_tokens, timeout

        primary_path = next(
            (path for path in self.paths if path.is_primary),
            self.paths[0] if self.paths else None,
        )
        primary_evidence = [
            item for item in self.evidence if item.is_primary_path_evidence
        ]
        if not primary_evidence:
            primary_evidence = list(self.evidence)

        chain_evidence = self._deduplicate_path_evidence(primary_evidence)
        citation_evidence = (
            self._deduplicate_path_evidence(self.evidence)
            if self.answer_intent
            in {ANSWER_INTENT_BRANCH, ANSWER_INTENT_CONVERGENCE}
            else chain_evidence
        )
        citation_text = self._citation_text(citation_evidence)

        if self.final_decision == UNCERTAIN:
            detail = self.decision_explanation
            answer = "Chưa đủ căn cứ từ dữ liệu được cung cấp để kết luận chắc chắn."
            if detail:
                answer = f"{answer} {detail}"
            return self._append_citations(answer, citation_text)

        if self.final_decision == REJECT_DIRECT_CLAIM:
            return self._claim_answer(
                supported=False,
                primary_path=primary_path,
                citation_text=citation_text,
            )

        if self.answer_intent == ANSWER_INTENT_CLAIM_VERIFICATION:
            return self._claim_answer(
                supported=True,
                primary_path=primary_path,
                citation_text=citation_text,
            )

        if primary_path is None:
            return self._append_citations(
                "Chưa đủ căn cứ từ dữ liệu được cung cấp để trả lời câu hỏi.",
                citation_text,
            )

        if self.answer_intent == ANSWER_INTENT_INITIAL_CAUSE:
            answer = self._initial_cause_answer(primary_path)
        elif self.answer_intent == ANSWER_INTENT_BRIDGE_EVENT:
            answer = self._bridge_answer(primary_path)
        elif self.answer_intent == ANSWER_INTENT_CHAIN_EXPLANATION:
            answer = self._chain_explanation(primary_path, chain_evidence)
        elif self.answer_intent == ANSWER_INTENT_YES_NO:
            answer = self._yes_no_answer(primary_path)
        elif self.answer_intent == ANSWER_INTENT_BRANCH:
            answer = self._branch_answer(primary_path)
        elif self.answer_intent == ANSWER_INTENT_CONVERGENCE:
            answer = self._convergence_answer(primary_path)
        else:
            answer = self._final_outcome_answer(primary_path)

        if not answer:
            final_rule = self._find_final_rule(primary_path, primary_evidence)
            fallback_outcome = (
                self._clean_label(final_rule.effect)
                if final_rule is not None
                else ""
            )
            answer = (
                f"Hệ quả cuối cùng là {self._quote(fallback_outcome)}."
                if fallback_outcome
                else "Chưa đủ căn cứ để xác định nội dung được hỏi."
            )

        if self.answer_intent == ANSWER_INTENT_CHAIN_EXPLANATION:
            return answer
        return self._append_citations(answer, citation_text)

    def _claim_answer(
        self,
        *,
        supported: bool,
        primary_path: Optional[FinalPath],
        citation_text: str,
    ) -> str:
        claim_type = safe_string(
            self.query_analysis.get("claim_type")
        ).upper()
        names = self._path_names(primary_path)

        if claim_type == "DIRECT_CAUSAL_CLAIM" and len(names) >= 2:
            start = self._quote(names[0])
            outcome = self._quote(names[-1])
            if supported:
                answer = (
                    f"Có. Graph có cạnh CAUSES trực tiếp từ {start} "
                    f"đến {outcome}."
                )
            else:
                answer = (
                    f"Không. Graph không thể hiện một cạnh trực tiếp từ "
                    f"{start} đến {outcome}."
                )
                mediators = names[1:-1]
                if mediators:
                    mediator_text = ", ".join(
                        self._quote(item) for item in mediators
                    )
                    answer += (
                        " Kết quả này chỉ được hỗ trợ qua sự kiện "
                        f"trung gian {mediator_text}."
                    )
            return self._append_citations(answer, citation_text)

        prefix = "Có." if supported else "Không."
        explanation = self.decision_explanation
        if not explanation:
            mediator = safe_string(
                self.query_analysis.get("matched_mediator_name")
                or self.query_analysis.get("matched_mediator_id")
            )
            factual_outcome = safe_string(
                self.query_analysis.get("factual_outcome")
            )
            counterfactual_outcome = safe_string(
                self.query_analysis.get("counterfactual_outcome")
            )
            if mediator and counterfactual_outcome:
                explanation = (
                    f"Sau khi đặt {self._quote(mediator)} vắng mặt, "
                    f"outcome chuyển từ {factual_outcome or 'không rõ'} "
                    f"sang {counterfactual_outcome}."
                )
            else:
                explanation = "Claim được đối chiếu với primary causal path."

        return self._append_citations(
            f"{prefix} {explanation}",
            citation_text,
        )

    def _initial_cause_answer(self, primary_path: FinalPath) -> str:
        names = self._path_names(primary_path)
        if not names:
            return ""
        answer = f"Sự kiện khởi đầu là {self._quote(names[0])}."
        if len(names) >= 2:
            answer += " Chuỗi đầy đủ: " + " → ".join(
                self._quote(item) for item in names
            ) + "."
        return answer

    def _bridge_answer(self, primary_path: FinalPath) -> str:
        names = self._path_names(primary_path)
        if len(names) < 3:
            return ""
        mediators = names[1:-1]
        if len(mediators) == 1:
            return f"Sự kiện trung gian là {self._quote(mediators[0])}."
        return "Các sự kiện trung gian lần lượt là " + " → ".join(
            self._quote(item) for item in mediators
        ) + "."

    def _chain_explanation(
        self,
        primary_path: FinalPath,
        chain_evidence: list[FinalEvidence],
    ) -> str:
        names = self._path_names(primary_path)
        event_ids = list(primary_path.event_ids)
        if len(names) < 2:
            return ""

        steps: list[str] = []
        for index, (source, target) in enumerate(
            zip(names, names[1:]),
            start=1,
        ):
            evidence = self._evidence_for_hop(
                hop_index=index - 1,
                source_id=(
                    event_ids[index - 1]
                    if index - 1 < len(event_ids)
                    else ""
                ),
                target_id=(
                    event_ids[index]
                    if index < len(event_ids)
                    else ""
                ),
                source_name=source,
                target_name=target,
                evidence=chain_evidence,
            )
            article = ""
            citation = ""
            if evidence is not None:
                if evidence.article_id:
                    article = f" theo Điều {evidence.article_id}"
                citation = f" [E{evidence.evidence_index}]"
            steps.append(
                f"Bước {index}: {self._quote(source)} dẫn đến "
                f"{self._quote(target)}{article}.{citation}"
            )
        return " ".join(steps)

    def _yes_no_answer(self, primary_path: FinalPath) -> str:
        names = self._path_names(primary_path)
        if len(names) < 2:
            return "Có. Dữ liệu hỗ trợ quan hệ được hỏi."
        return (
            "Có. Dữ liệu hỗ trợ chuỗi "
            + " → ".join(self._quote(item) for item in names)
            + "."
        )

    def _branch_answer(self, primary_path: FinalPath) -> str:
        primary_ids = list(primary_path.event_ids)
        primary_names = self._path_names(primary_path)
        if len(primary_names) < 3:
            return ""

        outcomes: list[str] = []
        for path in self.paths:
            if safe_string(path.status).upper() != SUPPORTED:
                continue
            names = self._path_names(path)
            if len(names) < 3:
                continue
            same_prefix = (
                len(primary_ids) >= 2
                and len(path.event_ids) >= 2
                and path.event_ids[:2] == primary_ids[:2]
            ) or names[:2] == primary_names[:2]
            if same_prefix:
                outcomes.append(names[-1])

        outcomes = unique_preserve_order(outcomes)
        if not outcomes:
            outcomes = [primary_names[-1]]
        return (
            f"Sau bước {self._quote(primary_names[1])}, các nhánh được "
            "ghi nhận gồm: "
            + "; ".join(self._quote(item) for item in outcomes)
            + "."
        )

    def _convergence_answer(self, primary_path: FinalPath) -> str:
        primary_ids = list(primary_path.event_ids)
        primary_names = self._path_names(primary_path)
        if len(primary_names) < 3:
            return ""

        conditions: list[str] = []
        for path in self.paths:
            if safe_string(path.status).upper() != SUPPORTED:
                continue
            names = self._path_names(path)
            if len(names) < 3:
                continue
            same_suffix = (
                len(primary_ids) >= 2
                and len(path.event_ids) >= 2
                and path.event_ids[-2:] == primary_ids[-2:]
            ) or names[-2:] == primary_names[-2:]
            if same_suffix:
                conditions.append(names[0])

        conditions = unique_preserve_order(conditions)
        if not conditions:
            conditions = [primary_names[0]]
        return (
            "Các điều kiện hội tụ gồm: "
            + "; ".join(self._quote(item) for item in conditions)
            + f". Chúng cùng dẫn đến {self._quote(primary_names[-2])}, "
            f"sau đó đến {self._quote(primary_names[-1])}."
        )

    def _final_outcome_answer(self, primary_path: FinalPath) -> str:
        names = self._path_names(primary_path)
        primary_evidence = [
            item for item in self.evidence if item.is_primary_path_evidence
        ]
        if not primary_evidence:
            primary_evidence = list(self.evidence)

        final_rule = self._find_final_rule(primary_path, primary_evidence)
        legal_effect = self._clean_label(
            final_rule.effect if final_rule is not None else ""
        )
        outcome = legal_effect or self._clean_label(
            primary_path.outcome_event_name
            or primary_path.outcome_event_id
            or (names[-1] if names else "")
        )
        if not outcome:
            return ""

        answer = f"Hệ quả cuối cùng là {self._quote(outcome)}."
        if len(names) >= 2:
            answer += " Chuỗi suy luận: " + " → ".join(
                self._quote(item) for item in names
            ) + "."
        return answer

    @staticmethod
    def _path_names(path: Optional[FinalPath]) -> list[str]:
        if path is None:
            return []
        values = path.event_names or path.event_ids
        return [
            ExtractiveFallbackProvider._clean_label(value)
            for value in values
            if ExtractiveFallbackProvider._clean_label(value)
        ]

    @staticmethod
    def _clean_label(value: Any) -> str:
        text = safe_string(value).strip("“”\"")
        return text.rstrip(" .:;")

    @staticmethod
    def _quote(value: Any) -> str:
        text = ExtractiveFallbackProvider._clean_label(value)
        return f"“{text}”" if text else "“không xác định”"

    @staticmethod
    def _deduplicate_path_evidence(
        evidence: list[FinalEvidence],
    ) -> list[FinalEvidence]:
        ordered = sorted(
            evidence,
            key=lambda item: (
                item.primary_path_position < 0,
                item.primary_path_position,
                item.evidence_index,
            ),
        )
        result: list[FinalEvidence] = []
        seen_edges: set[tuple[str, str]] = set()
        for item in ordered:
            edge = (
                safe_string(item.condition_event or item.condition_event_name),
                safe_string(item.effect_event or item.effect_event_name),
            )
            if edge != ("", "") and edge in seen_edges:
                continue
            seen_edges.add(edge)
            result.append(item)
        return result

    @staticmethod
    def _citation_text(evidence: list[FinalEvidence]) -> str:
        return " ".join(
            f"[E{item.evidence_index}]" for item in evidence
        )

    @staticmethod
    def _append_citations(text: str, citation_text: str) -> str:
        answer = safe_string(text)
        if answer and answer[-1] not in ".!?":
            answer += "."
        return f"{answer} {citation_text}".strip()

    @staticmethod
    def _evidence_for_hop(
        *,
        hop_index: int,
        source_id: str,
        target_id: str,
        source_name: str,
        target_name: str,
        evidence: list[FinalEvidence],
    ) -> Optional[FinalEvidence]:
        for item in evidence:
            if (
                source_id
                and target_id
                and safe_string(item.condition_event) == source_id
                and safe_string(item.effect_event) == target_id
            ):
                return item

        normalized_source = safe_string(source_name).lower()
        normalized_target = safe_string(target_name).lower()
        for item in evidence:
            if (
                safe_string(item.condition_event_name).lower()
                == normalized_source
                and safe_string(item.effect_event_name).lower()
                == normalized_target
            ):
                return item

        return evidence[hop_index] if hop_index < len(evidence) else None

    @staticmethod
    def _find_final_rule(
        primary_path: Optional[FinalPath],
        primary_evidence: list[FinalEvidence],
    ) -> Optional[FinalEvidence]:
        if not primary_evidence:
            return None

        if primary_path is not None and primary_path.rule_ids:
            last_rule_id = primary_path.rule_ids[-1]
            for evidence in primary_evidence:
                if evidence.rule_id == last_rule_id:
                    return evidence

        if primary_path is not None:
            outcome_id = safe_string(primary_path.outcome_event_id)
            outcome_name = safe_string(primary_path.outcome_event_name).lower()

            for evidence in reversed(primary_evidence):
                if outcome_id and safe_string(evidence.effect_event) == outcome_id:
                    return evidence
                if (
                    outcome_name
                    and safe_string(evidence.effect_event_name).lower() == outcome_name
                ):
                    return evidence

        ordered = sorted(
            primary_evidence,
            key=lambda item: item.primary_path_position,
        )
        return ordered[-1]


def _http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: int,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    http_request = request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} từ {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Không kết nối được tới {url}: {exc.reason}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("API trả response không phải JSON hợp lệ.") from exc

    if not isinstance(result, dict):
        raise RuntimeError("API trả JSON không phải object.")
    return result


# ============================================================
# ANSWER VALIDATION
# ============================================================

class FinalAnswerValidator:
    CITATION_PATTERN = re.compile(r"\[E(\d+)\]")

    def validate_and_repair(
        self,
        *,
        answer: str,
        evidence: list[FinalEvidence],
        preferred_evidence_ids: list[int],
    ) -> tuple[str, list[str], list[str]]:
        allowed_ids = {item.evidence_index for item in evidence}
        warnings: list[str] = []
        repaired = safe_string(answer)

        cited_ids = [
            safe_int(match)
            for match in self.CITATION_PATTERN.findall(repaired)
        ]
        invalid_ids = sorted(
            citation_id
            for citation_id in set(cited_ids)
            if citation_id not in allowed_ids
        )

        for citation_id in invalid_ids:
            repaired = repaired.replace(f"[E{citation_id}]", "")

        if invalid_ids:
            warnings.append(
                "Đã loại citation không tồn tại: "
                + ", ".join(f"E{citation_id}" for citation_id in invalid_ids)
            )

        used_ids = unique_ints(
            self.CITATION_PATTERN.findall(repaired)
        )
        used_ids = [item for item in used_ids if item in allowed_ids]

        if evidence and not used_ids:
            fallback_ids = [
                evidence_id
                for evidence_id in unique_ints(preferred_evidence_ids)
                if evidence_id in allowed_ids
            ]
            if not fallback_ids:
                fallback_ids = [evidence[0].evidence_index]

            citations = " ".join(f"[E{item}]" for item in fallback_ids)
            repaired = f"{repaired.rstrip()} {citations}".strip()
            used_ids = fallback_ids
            warnings.append(
                "Câu trả lời không có citation; đã thêm citation của primary path."
            )

        # Dọn khoảng trắng sinh ra sau khi loại citation lỗi.
        repaired = re.sub(r"[ \t]+", " ", repaired)
        repaired = re.sub(r" +([.,;:!?])", r"\1", repaired)
        repaired = re.sub(r"\n{3,}", "\n\n", repaired).strip()

        citations_used = [f"E{citation_id}" for citation_id in used_ids]
        return repaired, citations_used, warnings


# ============================================================
# PIPELINE
# ============================================================

class FinalAnswerPipeline:
    def __init__(self, store: FinalAnswerInputStore) -> None:
        self.store = store
        self.selector = FinalContextSelector(store)
        self.prompt_builder = LegalAnswerPromptBuilder()
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
        # Chọn path trước để evidence được xếp đúng theo primary causal chain.
        selected_paths = self.selector.select_paths(max_paths=max_paths)
        selected_evidence = self.selector.select_evidence(
            max_evidence=max_evidence,
            min_verification_score=min_verification_score,
            include_uncertain=include_uncertain,
            selected_paths=selected_paths,
        )

        # Chọn lại supporting paths theo evidence, nhưng luôn giữ primary path.
        selected_paths = self.selector.select_paths(
            selected_evidence=selected_evidence,
            max_paths=max_paths,
        )

        system_prompt, user_prompt = self.prompt_builder.build(
            query=self.store.query,
            evidence=selected_evidence,
            paths=selected_paths,
            final_decision=self.store.final_decision,
            decision_score=self.store.decision_score,
            decision_explanation=self.store.decision_explanation,
            query_analysis=self.store.query_analysis,
            answer_intent=self.store.answer_intent,
            global_confidence=self.store.confidence,
            consistency_score=self.store.consistency_score,
            max_context_chars=max_context_chars,
        )

        provider = self._create_provider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
            evidence=selected_evidence,
            paths=selected_paths,
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
            actual_provider = provider.provider_name
        except Exception as exc:
            if not fallback_to_extractive:
                raise

            generation_error = f"{type(exc).__name__}: {exc}"
            print(
                "Warning: LLM generation failed. Đang dùng extractive fallback."
            )
            print("Reason:", generation_error)

            fallback = self._create_extractive_provider(
                evidence=selected_evidence,
                paths=selected_paths,
            )
            raw_answer = fallback.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model="extractive",
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            actual_provider = fallback.provider_name

        elapsed = time.time() - started_at

        preferred_evidence_ids = [
            item.evidence_index
            for item in selected_evidence
            if item.is_primary_path_evidence
        ]

        final_answer, citations_used, validation_warnings = (
            self.validator.validate_and_repair(
                answer=raw_answer,
                evidence=selected_evidence,
                preferred_evidence_ids=preferred_evidence_ids,
            )
        )

        return GeneratedAnswer(
            query=self.store.query,
            answer=final_answer,
            provider=actual_provider,
            model=model if actual_provider != "extractive" else "extractive",
            selected_evidence=[asdict(item) for item in selected_evidence],
            selected_paths=[asdict(item) for item in selected_paths],
            citations_used=citations_used,
            confidence=self.store.confidence,
            consistency_score=self.store.consistency_score,
            final_decision=self.store.final_decision,
            decision_score=self.store.decision_score,
            decision_explanation=self.store.decision_explanation,
            primary_path_ids=self.store.primary_path_ids,
            query_analysis=self.store.query_analysis,
            answer_intent=self.store.answer_intent,
            generation_metadata={
                "step5_version": STEP5_VERSION,
                "step4_version": self.store.step4_version,
                "intent_detection_version": safe_string(
                    self.store.query_analysis.get(
                        "intent_detection_version"
                    )
                ),
                "answer_intent": self.store.answer_intent,
                "counterfactual_signal_source": safe_string(
                    self.store.query_analysis.get(
                        "counterfactual_signal_source"
                    )
                ),
                "structural_result_used": bool(
                    self.store.query_analysis.get("structural_result_used")
                ),
                "structural_fallback_used": bool(
                    self.store.query_analysis.get("structural_fallback_used")
                ),
                "requested_provider": provider_name,
                "requested_model": model,
                "elapsed_seconds": round(elapsed, 4),
                "max_evidence": max_evidence,
                "max_paths": max_paths,
                "max_context_chars": max_context_chars,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "min_verification_score": min_verification_score,
                "include_uncertain": include_uncertain,
                "fallback_to_extractive": fallback_to_extractive,
                "generation_error": generation_error,
                "validation_warnings": validation_warnings,
                "system_prompt_chars": len(system_prompt),
                "user_prompt_chars": len(user_prompt),
                "primary_path_rule_ids": [
                    item.rule_id
                    for item in selected_evidence
                    if item.is_primary_path_evidence
                ],
            },
        )

    def _create_extractive_provider(
        self,
        *,
        evidence: list[FinalEvidence],
        paths: list[FinalPath],
    ) -> ExtractiveFallbackProvider:
        return ExtractiveFallbackProvider(
            evidence=evidence,
            paths=paths,
            final_decision=self.store.final_decision,
            decision_explanation=self.store.decision_explanation,
            confidence=self.store.confidence,
            query_analysis=self.store.query_analysis,
            answer_intent=self.store.answer_intent,
        )

    def _create_provider(
        self,
        *,
        provider_name: str,
        api_key: str,
        base_url: str,
        evidence: list[FinalEvidence],
        paths: list[FinalPath],
    ) -> BaseLLMProvider:
        provider_name = provider_name.lower()

        if provider_name == "ollama":
            return OllamaProvider(base_url=base_url or DEFAULT_OLLAMA_URL)
        if provider_name == "openai":
            return OpenAICompatibleProvider(
                base_url=base_url or DEFAULT_OPENAI_BASE_URL,
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            )
        if provider_name == "gemini":
            return GeminiProvider(
                api_key=api_key or os.getenv("GEMINI_API_KEY", "")
            )
        if provider_name == "extractive":
            return self._create_extractive_provider(
                evidence=evidence,
                paths=paths,
            )

        raise ValueError(
            "Provider không hợp lệ. Chọn: ollama, openai, gemini, extractive."
        )


# ============================================================
# DISPLAY
# ============================================================

def print_summary(result: GeneratedAnswer) -> None:
    print("\n" + "=" * 76)
    print("FINAL LEGAL ANSWER")
    print("=" * 76)
    print("Query:", result.query)
    print("Provider:", result.provider, "| Model:", result.model)
    print(
        "Step 5 version:",
        safe_string(result.generation_metadata.get("step5_version"))
        or STEP5_VERSION,
        "| Answer intent:",
        result.answer_intent,
    )
    print(
        "Decision:",
        result.final_decision,
        "| Score:",
        f"{result.decision_score:.4f}",
    )
    print(
        "Confidence:",
        f"{result.confidence:.4f}",
        "| Consistency:",
        f"{result.consistency_score:.4f}",
    )
    print("Primary path IDs:", result.primary_path_ids)
    print("\n" + result.answer)

    print("\nSelected evidence:")
    for item in result.selected_evidence:
        marker = "PRIMARY" if item.get("is_primary_path_evidence") else "SUPPLEMENT"
        print(
            f"- [E{item['evidence_index']}] {marker} | "
            f"Rule {item['rule_id']} | Điều {item['article_id']} | "
            f"verification={item['verification_score']:.4f}"
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded legal answer using the primary causal path "
            "and Step-4 verification decision."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verification-result", default=VERIFICATION_RESULT_PATH)
    parser.add_argument("--retrieval-result", default=RETRIEVAL_RESULT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)

    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "gemini", "extractive"],
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")

    parser.add_argument("--max-evidence", type=int, default=DEFAULT_MAX_EVIDENCE)
    parser.add_argument("--max-paths", type=int, default=DEFAULT_MAX_PATHS)
    parser.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--min-verification-score",
        type=float,
        default=DEFAULT_MIN_VERIFICATION_SCORE,
    )
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Cho phép dùng evidence UNCERTAIN nếu đạt ngưỡng.",
    )
    parser.add_argument(
        "--no-extractive-fallback",
        action="store_true",
        help="Không fallback sang extractive khi LLM lỗi.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_evidence < 1:
        raise ValueError("--max-evidence phải lớn hơn 0.")
    if args.max_paths < 0:
        raise ValueError("--max-paths không được âm.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens phải lớn hơn 0.")
    if not 0.0 <= args.min_verification_score <= 1.0:
        raise ValueError("--min-verification-score phải nằm trong [0, 1].")

    model = args.model
    if args.provider == "openai" and model == DEFAULT_MODEL:
        model = DEFAULT_OPENAI_MODEL
    if args.provider == "gemini" and model == DEFAULT_MODEL:
        model = DEFAULT_GEMINI_MODEL

    base_url = args.base_url
    if args.provider == "ollama" and not base_url:
        base_url = DEFAULT_OLLAMA_URL
    if args.provider == "openai" and not base_url:
        base_url = DEFAULT_OPENAI_BASE_URL

    store = FinalAnswerInputStore(
        verification_result_path=args.verification_result,
        retrieval_result_path=args.retrieval_result,
    )
    pipeline = FinalAnswerPipeline(store)

    result = pipeline.run(
        provider_name=args.provider,
        model=model,
        api_key=args.api_key,
        base_url=base_url,
        max_evidence=args.max_evidence,
        max_paths=args.max_paths,
        max_context_chars=args.max_context_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        min_verification_score=args.min_verification_score,
        include_uncertain=args.include_uncertain,
        fallback_to_extractive=not args.no_extractive_fallback,
    )

    print_summary(result)
    save_json(asdict(result), args.output)


if __name__ == "__main__":
    main()
