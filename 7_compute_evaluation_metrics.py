#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_compute_evaluation_metrics_pipeline_v2.py

Tính bộ chỉ số đánh giá chi tiết cho pipeline BLHS CausalRAG từ:

    benchmark BLHS
    + prediction do 5_5_generate_pipeline_predictions_query_aware_v2.py sinh ra

Script này bổ sung cho Step 6. Step 6 tạo báo cáo tổng quan; Step 7 tập trung vào:

1. Retrieval metrics cho rule, event và article.
2. Top-1 causal path metrics và oracle-path diagnostics.
3. Verification confusion matrix, macro-F1 và balanced accuracy.
4. Answer EM, token P/R/F1 và ROUGE-L.
5. Citation precision, recall, F1 và exact match.
6. Báo cáo theo question type, difficulty và counterfactual subset.
7. Error analysis ở mức từng câu.

Quan trọng
----------
- Metric top-1 chỉ đánh giá reasoning_path thực sự được pipeline chọn.
- Oracle path chỉ là diagnostic: chọn path khớp gold tốt nhất trong danh sách
  retrieval.causal_paths. Không dùng oracle để thay thế metric top-1.
- Không sử dụng gold field để thay đổi prediction.

Ví dụ
-----
python 7_compute_evaluation_metrics_pipeline_v2.py \
  --benchmark data/blhs_multihop_benchmark_250.json \
  --predictions data/pipeline_predictions.json \
  --output-dir evaluation_metrics_detailed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


STEP7_VERSION = "2.0-current-pipeline-schema"

DEFAULT_BENCHMARK_PATH = "data/blhs_multihop_benchmark_250.json"
DEFAULT_PREDICTION_PATH = "data/pipeline_predictions.json"
DEFAULT_OUTPUT_DIR = "evaluation_metrics_detailed"
DEFAULT_K_VALUES = (1, 3, 5, 10)

DECISION_LABELS = (
    "SUPPORTED",
    "REJECT_DIRECT_CLAIM",
    "UNCERTAIN",
)

TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ỹĐđ]+(?:['’-][0-9A-Za-zÀ-ỹĐđ]+)?",
    flags=re.UNICODE,
)
ARTICLE_RE = re.compile(
    r"(?:Điều|điều)\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.UNICODE,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class GoldSample:
    sample_id: str
    question: str
    answer: str
    question_type: str
    difficulty: str
    requires_counterfactual: bool
    gold_decision: str
    gold_rule_ids: list[str]
    gold_article_ids: list[str]
    gold_event_ids: list[str]
    gold_path_edges: list[tuple[str, str]]
    gold_path_rule_ids: list[str]
    gold_citations: list[str]
    raw: dict[str, Any] = field(repr=False)


@dataclass
class CandidatePath:
    rank: int
    path_id: int
    edges: list[tuple[str, str]]
    event_ids: list[str]
    rule_ids: list[str]
    article_ids: list[str]
    score: float


@dataclass
class Prediction:
    sample_id: str
    retrieved_rule_ids: list[str]
    retrieved_event_ids: list[str]
    retrieved_article_ids: list[str]
    selected_path: CandidatePath
    candidate_paths: list[CandidatePath]
    verification_decision: str
    decision_score: Optional[float]
    final_answer: str
    citations: list[str]
    runtime_seconds: Optional[float]
    pipeline_error: str
    raw: dict[str, Any] = field(repr=False)


# ============================================================
# GENERIC HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = remove_accents(safe_string(value)).lower()
    if text in {"true", "1", "yes", "y", "co"}:
        return True
    if text in {"false", "0", "no", "n", "khong"}:
        return False
    return default


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic_f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def remove_accents(value: Any) -> str:
    text = safe_string(value).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


def normalize_id(value: Any) -> str:
    text = safe_string(value)
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def normalize_event_id(value: Any) -> str:
    text = normalize_id(value)
    if text.upper().startswith("EVENT::"):
        text = text[len("EVENT::"):]
    return text


def normalize_label(value: Any) -> str:
    text = remove_accents(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_decision(value: Any) -> str:
    text = normalize_label(value)
    aliases = {
        "YES": "SUPPORTED",
        "TRUE": "SUPPORTED",
        "KEEP": "SUPPORTED",
        "VERIFIED": "SUPPORTED",
        "ACCEPT": "SUPPORTED",
        "ACCEPTED": "SUPPORTED",
        "ENTAILED": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "NO": "REJECT_DIRECT_CLAIM",
        "FALSE": "REJECT_DIRECT_CLAIM",
        "REJECT": "REJECT_DIRECT_CLAIM",
        "REJECTED": "REJECT_DIRECT_CLAIM",
        "CONTRADICTED": "REJECT_DIRECT_CLAIM",
        "CONTRADICTION": "REJECT_DIRECT_CLAIM",
        "REFUTED": "REJECT_DIRECT_CLAIM",
        "NO_DIRECT_EDGE": "REJECT_DIRECT_CLAIM",
        "REJECT_DIRECT_CLAIM": "REJECT_DIRECT_CLAIM",
        "UNCERTAIN": "UNCERTAIN",
        "UNRESOLVED": "UNCERTAIN",
        "UNKNOWN": "UNCERTAIN",
        "INSUFFICIENT": "UNCERTAIN",
        "NOT_ENOUGH_EVIDENCE": "UNCERTAIN",
    }
    return aliases.get(text, text)


def unique_preserve_order(
    values: Iterable[Any],
    normalizer=normalize_id,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = normalizer(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def recursive_round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, dict):
        return {
            key: recursive_round(item, digits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [recursive_round(item, digits) for item in value]
    if isinstance(value, tuple):
        return [recursive_round(item, digits) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    return value


def flatten_mapping(data: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(data, Mapping):
        return result

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        result[full_key] = value
        if isinstance(value, Mapping):
            result.update(flatten_mapping(value, full_key))
    return result


def find_first(
    data: Mapping[str, Any],
    candidates: Sequence[str],
    default: Any = None,
) -> Any:
    flat = flatten_mapping(data)

    for candidate in candidates:
        if candidate in flat:
            return flat[candidate]

    for candidate in candidates:
        for key, value in flat.items():
            if key == candidate or key.endswith(f".{candidate}"):
                return value
    return default


# ============================================================
# I/O
# ============================================================

def load_json_or_jsonl(path_value: str | Path) -> Any:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL lỗi tại dòng {line_number}: {exc}"
                    ) from exc
        return rows

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path_value: str | Path) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(recursive_round(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_csv(
    rows: list[dict[str, Any]],
    path_value: str | Path,
    fieldnames: Optional[list[str]] = None,
) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# TEXT AND SET METRICS
# ============================================================

def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", safe_string(value)).lower()
    text = re.sub(r"[“”\"`]", "", text)
    text = re.sub(r"[^\wÀ-ỹĐđ\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(value: Any) -> list[str]:
    text = unicodedata.normalize("NFC", safe_string(value)).lower()
    return TOKEN_RE.findall(text)


def token_overlap_metrics(
    gold_text: Any,
    predicted_text: Any,
) -> tuple[float, float, float]:
    gold_tokens = tokenize(gold_text)
    predicted_tokens = tokenize(predicted_text)

    if not gold_tokens and not predicted_tokens:
        return 1.0, 1.0, 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0, 0.0, 0.0

    overlap = sum(
        (Counter(gold_tokens) & Counter(predicted_tokens)).values()
    )
    precision = safe_div(overlap, len(predicted_tokens))
    recall = safe_div(overlap, len(gold_tokens))
    return precision, recall, harmonic_f1(precision, recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(gold_text: Any, predicted_text: Any) -> float:
    gold_tokens = tokenize(gold_text)
    predicted_tokens = tokenize(predicted_text)

    if not gold_tokens and not predicted_tokens:
        return 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0

    lcs = lcs_length(gold_tokens, predicted_tokens)
    precision = safe_div(lcs, len(predicted_tokens))
    recall = safe_div(lcs, len(gold_tokens))
    return harmonic_f1(precision, recall)


def normalized_exact_match(gold_text: Any, predicted_text: Any) -> float:
    return float(normalize_text(gold_text) == normalize_text(predicted_text))


def set_metrics(
    gold_values: Iterable[str],
    predicted_values: Iterable[str],
) -> dict[str, float]:
    gold = set(gold_values)
    predicted = set(predicted_values)
    true_positive = len(gold & predicted)

    precision = (
        true_positive / len(predicted)
        if predicted
        else (1.0 if not gold else 0.0)
    )
    recall = (
        true_positive / len(gold)
        if gold
        else (1.0 if not predicted else 0.0)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
        "exact": float(gold == predicted),
        "true_positive": float(true_positive),
    }


def recall_at_k(
    gold_values: Iterable[str],
    ranked_predictions: Sequence[str],
    k: int,
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0
    predicted = set(list(ranked_predictions)[:k])
    return safe_div(len(gold & predicted), len(gold))


def precision_at_k(
    gold_values: Iterable[str],
    ranked_predictions: Sequence[str],
    k: int,
) -> float:
    predicted = list(ranked_predictions)[:k]
    if not predicted:
        return 0.0
    gold = set(gold_values)
    return safe_div(sum(item in gold for item in predicted), len(predicted))


def hit_at_k(
    gold_values: Iterable[str],
    ranked_predictions: Sequence[str],
    k: int,
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0
    return float(bool(gold & set(list(ranked_predictions)[:k])))


def reciprocal_rank(
    gold_values: Iterable[str],
    ranked_predictions: Sequence[str],
) -> float:
    gold = set(gold_values)
    for rank, value in enumerate(ranked_predictions, start=1):
        if value in gold:
            return 1.0 / rank
    return 0.0


def average_precision(
    gold_values: Iterable[str],
    ranked_predictions: Sequence[str],
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0

    hits = 0
    total = 0.0
    seen: set[str] = set()

    for rank, value in enumerate(ranked_predictions, start=1):
        if value in seen:
            continue
        seen.add(value)
        if value in gold:
            hits += 1
            total += hits / rank

    return total / len(gold)


# ============================================================
# CITATION NORMALIZATION
# ============================================================

def normalize_article_id(value: Any) -> str:
    text = safe_string(value)
    match = ARTICLE_RE.search(text)
    if match:
        return normalize_id(match.group(1))
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return normalize_id(text)
    return normalize_id(text)


def normalize_citations(values: Any, answer_text: str = "") -> list[str]:
    citations: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            candidate = (
                value.get("article_id")
                or value.get("article")
                or value.get("citation")
                or value.get("label")
            )
            if candidate is not None:
                add(candidate)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return

        article_id = normalize_article_id(value)
        if article_id:
            citations.append(article_id)

    add(values)

    for match in ARTICLE_RE.finditer(answer_text):
        citations.append(normalize_id(match.group(1)))

    return unique_preserve_order(citations, normalizer=normalize_id)


# ============================================================
# PATH PARSING
# ============================================================

def edge_from_mapping(value: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    source = (
        value.get("source_event_id")
        or value.get("source_event_node")
        or value.get("source")
        or value.get("from")
        or value.get("condition_event")
    )
    target = (
        value.get("target_event_id")
        or value.get("target_event_node")
        or value.get("target")
        or value.get("to")
        or value.get("effect_event")
    )

    source_id = normalize_event_id(source)
    target_id = normalize_event_id(target)
    if source_id and target_id:
        return source_id, target_id
    return None


def edges_to_event_ids(edges: Sequence[tuple[str, str]]) -> list[str]:
    if not edges:
        return []
    result = [edges[0][0]]
    result.extend(target for _, target in edges)
    return unique_preserve_order(result, normalizer=normalize_event_id)


def order_edges(edges: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Dựng lại một directed chain khi input edge bị đảo thứ tự danh sách."""
    cleaned = [
        (normalize_event_id(source), normalize_event_id(target))
        for source, target in edges
        if normalize_event_id(source) and normalize_event_id(target)
    ]
    if len(cleaned) <= 1:
        return cleaned

    by_source: dict[str, list[tuple[int, tuple[str, str]]]] = defaultdict(list)
    targets: set[str] = set()
    for index, edge in enumerate(cleaned):
        by_source[edge[0]].append((index, edge))
        targets.add(edge[1])

    starts = [source for source in by_source if source not in targets]
    if len(starts) != 1:
        return cleaned

    current = starts[0]
    ordered: list[tuple[str, str]] = []
    used: set[int] = set()

    while current in by_source:
        candidates = [item for item in by_source[current] if item[0] not in used]
        if len(candidates) != 1:
            break
        index, edge = candidates[0]
        ordered.append(edge)
        used.add(index)
        current = edge[1]

    return ordered if len(ordered) == len(cleaned) else cleaned


def parse_path_steps(value: Any) -> tuple[list[tuple[str, str]], list[str], list[str], list[str]]:
    edges: list[tuple[str, str]] = []
    event_ids: list[str] = []
    rule_ids: list[str] = []
    article_ids: list[str] = []

    if value is None:
        return edges, event_ids, rule_ids, article_ids

    if isinstance(value, str):
        if "->" in value or "→" in value:
            nodes = [
                normalize_event_id(part)
                for part in re.split(r"\s*(?:->|→)\s*", value)
                if normalize_event_id(part)
            ]
            edges = list(zip(nodes, nodes[1:]))
            return edges, nodes, [], []
        event_id = normalize_event_id(value)
        return [], [event_id] if event_id else [], [], []

    if isinstance(value, Mapping):
        direct_edge = edge_from_mapping(value)
        if direct_edge:
            edges.append(direct_edge)
            rule_ids.extend(
                unique_preserve_order(value.get("rule_ids") or [])
            )
            article_ids.extend(
                unique_preserve_order(value.get("article_ids") or [])
            )
            return edges, edges_to_event_ids(edges), rule_ids, article_ids

        nested = (
            value.get("steps")
            or value.get("edges")
            or value.get("path")
            or value.get("event_chain")
            or value.get("events")
            or value.get("nodes")
        )
        if nested is not None:
            edges, event_ids, rule_ids, article_ids = parse_path_steps(nested)
            if not rule_ids:
                rule_ids = unique_preserve_order(value.get("rule_ids") or [])
            if not article_ids:
                article_ids = unique_preserve_order(value.get("article_ids") or [])
            return edges, event_ids, rule_ids, article_ids

        event_id = normalize_event_id(
            value.get("event_id")
            or value.get("event_node")
            or value.get("graph_node_id")
            or value.get("id")
        )
        return [], [event_id] if event_id else [], [], []

    if isinstance(value, (list, tuple)):
        if not value:
            return [], [], [], []

        # Danh sách node/event id.
        if all(
            isinstance(item, (str, int, float))
            or (
                isinstance(item, Mapping)
                and not edge_from_mapping(item)
                and any(
                    key in item
                    for key in ("event_id", "event_node", "graph_node_id", "id")
                )
            )
            for item in value
        ):
            nodes: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    node = normalize_event_id(
                        item.get("event_id")
                        or item.get("event_node")
                        or item.get("graph_node_id")
                        or item.get("id")
                    )
                else:
                    node = normalize_event_id(item)
                if node:
                    nodes.append(node)
            nodes = unique_preserve_order(nodes, normalizer=normalize_event_id)
            return list(zip(nodes, nodes[1:])), nodes, [], []

        for item in value:
            child_edges, child_events, child_rules, child_articles = parse_path_steps(item)
            edges.extend(child_edges)
            event_ids.extend(child_events)
            rule_ids.extend(child_rules)
            article_ids.extend(child_articles)

        edges = order_edges(edges)
        if edges:
            event_ids = edges_to_event_ids(edges)
        else:
            event_ids = unique_preserve_order(event_ids, normalizer=normalize_event_id)

        return (
            edges,
            event_ids,
            unique_preserve_order(rule_ids),
            unique_preserve_order(article_ids),
        )

    return [], [], [], []


def build_candidate_path(
    value: Any,
    rank: int,
    path_id: int,
    score: float = 0.0,
) -> CandidatePath:
    edges, event_ids, rule_ids, article_ids = parse_path_steps(value)
    return CandidatePath(
        rank=rank,
        path_id=path_id,
        edges=edges,
        event_ids=event_ids,
        rule_ids=rule_ids,
        article_ids=article_ids,
        score=float(score or 0.0),
    )


# ============================================================
# BENCHMARK LOADING
# ============================================================

def extract_benchmark_rows(payload: Any) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if isinstance(payload, list):
        return {}, [row for row in payload if isinstance(row, Mapping)]

    if isinstance(payload, Mapping):
        rows = (
            payload.get("questions")
            or payload.get("samples")
            or payload.get("data")
        )
        metadata = dict(payload.get("metadata") or {})
        if isinstance(rows, list):
            return metadata, [row for row in rows if isinstance(row, Mapping)]

    raise ValueError(
        "Benchmark phải là list hoặc JSON object chứa questions/samples/data."
    )


def parse_gold_sample(raw: Mapping[str, Any]) -> GoldSample:
    evaluation = raw.get("evaluation") or {}
    if not isinstance(evaluation, Mapping):
        evaluation = {}

    sample_id = normalize_id(
        raw.get("id")
        or raw.get("sample_id")
        or raw.get("question_id")
    )

    gold_rule_ids = unique_preserve_order(
        evaluation.get("gold_rule_ids")
        or raw.get("gold_rule_ids")
        or raw.get("supporting_rule_ids")
        or []
    )
    gold_article_ids = unique_preserve_order(
        evaluation.get("gold_article_ids")
        or raw.get("gold_article_ids")
        or raw.get("supporting_article_ids")
        or []
    )

    raw_gold_path = (
        evaluation.get("gold_path")
        or raw.get("gold_path")
        or []
    )
    gold_edges, path_event_ids, path_rule_ids, path_article_ids = parse_path_steps(
        raw_gold_path
    )

    supporting_chain = raw.get("supporting_event_chain") or []
    supporting_event_ids = [
        event.get("event_id")
        for event in supporting_chain
        if isinstance(event, Mapping)
    ]

    gold_event_ids = unique_preserve_order(
        evaluation.get("gold_event_ids")
        or raw.get("gold_event_ids")
        or supporting_event_ids
        or path_event_ids,
        normalizer=normalize_event_id,
    )

    if not gold_article_ids and path_article_ids:
        gold_article_ids = unique_preserve_order(path_article_ids)
    if not gold_rule_ids and path_rule_ids:
        gold_rule_ids = unique_preserve_order(path_rule_ids)

    requires_counterfactual = bool(
        evaluation.get(
            "requires_counterfactual",
            raw.get("question_type") == "yes_no_counterexample",
        )
    )

    gold_decision = normalize_decision(
        evaluation.get("gold_decision")
        or raw.get("gold_decision")
        or raw.get("expected_label")
        or (
            "REJECT_DIRECT_CLAIM"
            if requires_counterfactual
            else "SUPPORTED"
        )
    )

    gold_answer = safe_string(
        evaluation.get("gold_answer")
        or raw.get("gold_answer")
        or raw.get("answer")
    )

    gold_citations = normalize_citations(
        evaluation.get("gold_citations")
        or raw.get("gold_citations")
        or raw.get("expected_citations")
        or gold_article_ids
    )

    difficulty_value = evaluation.get("difficulty") or raw.get("difficulty")
    if isinstance(difficulty_value, Mapping):
        difficulty = safe_string(
            difficulty_value.get("overall_level")
            or difficulty_value.get("level")
            or difficulty_value.get("difficulty")
        )
    else:
        difficulty = safe_string(difficulty_value)

    return GoldSample(
        sample_id=sample_id,
        question=safe_string(raw.get("question")),
        answer=gold_answer,
        question_type=safe_string(raw.get("question_type") or "unknown"),
        difficulty=difficulty or "unknown",
        requires_counterfactual=requires_counterfactual,
        gold_decision=gold_decision,
        gold_rule_ids=gold_rule_ids,
        gold_article_ids=gold_article_ids,
        gold_event_ids=gold_event_ids,
        gold_path_edges=gold_edges,
        gold_path_rule_ids=path_rule_ids or gold_rule_ids,
        gold_citations=gold_citations,
        raw=dict(raw),
    )


def load_benchmark(
    path_value: str | Path,
    limit: Optional[int] = None,
) -> tuple[dict[str, Any], list[GoldSample]]:
    payload = load_json_or_jsonl(path_value)
    metadata, rows = extract_benchmark_rows(payload)

    samples = [parse_gold_sample(row) for row in rows]
    samples = [sample for sample in samples if sample.sample_id]

    if limit is not None:
        samples = samples[:limit]

    if not samples:
        raise ValueError("Benchmark không có sample hợp lệ.")

    counts = Counter(sample.sample_id for sample in samples)
    duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Benchmark có id trùng: {duplicates[:10]}")

    return metadata, samples


# ============================================================
# PREDICTION LOADING
# ============================================================

def extract_prediction_rows(
    payload: Any,
) -> tuple[dict[str, Any], list[tuple[Optional[str], Mapping[str, Any]]]]:
    if isinstance(payload, list):
        return {}, [
            (None, row)
            for row in payload
            if isinstance(row, Mapping)
        ]

    if not isinstance(payload, Mapping):
        raise ValueError("Prediction phải là JSON object, list hoặc JSONL.")

    metadata = dict(payload.get("metadata") or {})
    explicit = (
        payload.get("predictions")
        or payload.get("results")
        or payload.get("questions")
        or payload.get("samples")
        or payload.get("data")
    )

    if isinstance(explicit, list):
        return metadata, [
            (None, row)
            for row in explicit
            if isinstance(row, Mapping)
        ]

    if isinstance(explicit, Mapping):
        return metadata, [
            (normalize_id(key), value)
            for key, value in explicit.items()
            if isinstance(value, Mapping)
        ]

    candidate_rows = [
        (normalize_id(key), value)
        for key, value in payload.items()
        if key != "metadata" and isinstance(value, Mapping)
    ]
    if candidate_rows:
        return metadata, candidate_rows

    raise ValueError("Không nhận diện được cấu trúc prediction JSON.")


def extract_id_list(
    value: Any,
    id_keys: Sequence[str],
    normalizer=normalize_id,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (str, int, float)):
        normalized = normalizer(value)
        return [normalized] if normalized else []

    if isinstance(value, Mapping):
        for key in id_keys:
            if key in value and value[key] is not None:
                normalized = normalizer(value[key])
                return [normalized] if normalized else []

        result: list[str] = []
        for child in value.values():
            if isinstance(child, (list, tuple, set)):
                result.extend(
                    extract_id_list(child, id_keys, normalizer=normalizer)
                )
        return unique_preserve_order(result, normalizer=normalizer)

    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(
                extract_id_list(item, id_keys, normalizer=normalizer)
            )
        return unique_preserve_order(result, normalizer=normalizer)

    return []


def parse_candidate_paths(raw: Mapping[str, Any]) -> list[CandidatePath]:
    candidate_paths: list[CandidatePath] = []
    retrieval = raw.get("retrieval") or {}
    if not isinstance(retrieval, Mapping):
        retrieval = {}

    raw_paths = retrieval.get("causal_paths") or raw.get("predicted_paths") or []
    if not isinstance(raw_paths, list):
        return []

    for rank, path in enumerate(raw_paths, start=1):
        if not isinstance(path, Mapping):
            continue
        path_id = safe_int(path.get("path_id"), rank - 1)
        score = safe_float(
            path.get("graph_score", path.get("path_score", path.get("score"))),
            0.0,
        ) or 0.0
        candidate_paths.append(
            build_candidate_path(path, rank=rank, path_id=path_id, score=score)
        )

    return candidate_paths


def parse_prediction(
    raw: Mapping[str, Any],
    fallback_id: Optional[str] = None,
) -> Prediction:
    sample_id = normalize_id(
        raw.get("id")
        or raw.get("sample_id")
        or raw.get("question_id")
        or fallback_id
    )

    retrieved_rule_ids = extract_id_list(
        find_first(
            raw,
            (
                "retrieved_rule_ids",
                "retrieval.retrieved_rule_ids",
                "retrieved_rules",
                "retrieval.retrieved_rules",
                "rule_ids",
            ),
            [],
        ),
        id_keys=("rule_id", "id", "memory_id"),
    )

    retrieved_event_ids = extract_id_list(
        find_first(
            raw,
            (
                "retrieved_event_ids",
                "retrieval.retrieved_event_ids",
                "retrieved_events",
                "retrieval.retrieved_events",
                "event_ids",
            ),
            [],
        ),
        id_keys=("event_id", "graph_node_id", "id", "memory_id"),
        normalizer=normalize_event_id,
    )

    retrieved_article_ids = extract_id_list(
        find_first(
            raw,
            (
                "retrieved_article_ids",
                "retrieval.retrieved_article_ids",
                "article_ids",
            ),
            [],
        ),
        id_keys=("article_id", "article", "id"),
        normalizer=normalize_article_id,
    )

    selected_path_value = find_first(
        raw,
        (
            "reasoning_path",
            "retrieval.reasoning_path",
            "selected_path",
            "predicted_path",
            "verified_path",
        ),
        [],
    )
    selected_path_id = safe_int(
        raw.get("reasoning_path_id")
        or find_first(raw, ("retrieval.selected_path_id",), -1),
        -1,
    )
    selected_path = build_candidate_path(
        selected_path_value,
        rank=1,
        path_id=selected_path_id,
        score=float(safe_float(raw.get("decision_score"), 0.0) or 0.0),
    )

    retrieved_event_ids = unique_preserve_order(
        selected_path.event_ids + retrieved_event_ids,
        normalizer=normalize_event_id,
    )
    retrieved_rule_ids = unique_preserve_order(
        selected_path.rule_ids + retrieved_rule_ids
    )
    retrieved_article_ids = unique_preserve_order(
        selected_path.article_ids + retrieved_article_ids,
        normalizer=normalize_article_id,
    )

    candidate_paths = parse_candidate_paths(raw)

    # Oracle diagnostics phải bao gồm ít nhất path mà pipeline thực sự chọn.
    # Điều này ngăn trường hợp oracle thấp hơn top-1 chỉ vì prediction không
    # lưu retrieval.causal_paths hoặc lưu danh sách rỗng.
    if selected_path.edges and not any(
        candidate.edges == selected_path.edges
        for candidate in candidate_paths
    ):
        candidate_paths.insert(0, selected_path)

    verification_decision = normalize_decision(
        find_first(
            raw,
            (
                "verification_decision",
                "verification.final_decision",
                "generation.final_decision",
                "final_decision",
                "decision",
            ),
            "",
        )
    )

    decision_score = safe_float(
        find_first(
            raw,
            (
                "decision_score",
                "verification.decision_score",
                "generation.decision_score",
            ),
            None,
        ),
        None,
    )

    final_answer = safe_string(
        find_first(
            raw,
            (
                "final_answer",
                "generation.answer",
                "answer",
                "generated_answer",
            ),
            "",
        )
    )

    citations = normalize_citations(
        find_first(
            raw,
            (
                "citations",
                "generation.citations",
                "used_citations",
                "predicted_citations",
            ),
            [],
        ),
        answer_text=final_answer,
    )

    runtime_seconds = safe_float(
        find_first(
            raw,
            (
                "runtime_seconds",
                "pipeline_metadata.elapsed_seconds",
                "generation.metadata.elapsed_seconds",
                "elapsed_seconds",
            ),
            None,
        ),
        None,
    )

    pipeline_error = safe_string(
        raw.get("error")
        or find_first(raw, ("pipeline_metadata.error",), "")
    )

    return Prediction(
        sample_id=sample_id,
        retrieved_rule_ids=retrieved_rule_ids,
        retrieved_event_ids=retrieved_event_ids,
        retrieved_article_ids=retrieved_article_ids,
        selected_path=selected_path,
        candidate_paths=candidate_paths,
        verification_decision=verification_decision,
        decision_score=decision_score,
        final_answer=final_answer,
        citations=citations,
        runtime_seconds=runtime_seconds,
        pipeline_error=pipeline_error,
        raw=dict(raw),
    )


def load_predictions(
    path_value: str | Path,
) -> tuple[dict[str, Any], dict[str, Prediction]]:
    payload = load_json_or_jsonl(path_value)
    metadata, rows = extract_prediction_rows(payload)

    predictions: dict[str, Prediction] = {}
    for fallback_id, row in rows:
        prediction = parse_prediction(row, fallback_id=fallback_id)
        if not prediction.sample_id:
            continue
        if prediction.sample_id in predictions:
            raise ValueError(
                f"Prediction có sample id trùng: {prediction.sample_id}"
            )
        predictions[prediction.sample_id] = prediction

    if not predictions:
        raise ValueError("Không đọc được prediction hợp lệ.")

    return metadata, predictions


# ============================================================
# PATH METRICS
# ============================================================

def path_metric_values(
    gold_edges: Sequence[tuple[str, str]],
    predicted_path: CandidatePath,
    gold_rule_ids: Sequence[str],
) -> dict[str, float]:
    predicted_edges = predicted_path.edges
    gold_edge_labels = [f"{source}>>{target}" for source, target in gold_edges]
    predicted_edge_labels = [
        f"{source}>>{target}" for source, target in predicted_edges
    ]

    gold_events = edges_to_event_ids(gold_edges)
    predicted_events = predicted_path.event_ids or edges_to_event_ids(predicted_edges)

    edge_metrics = set_metrics(gold_edge_labels, predicted_edge_labels)
    event_metrics = set_metrics(gold_events, predicted_events)
    path_rule_metrics = set_metrics(gold_rule_ids, predicted_path.rule_ids)

    exact_ordered = float(list(gold_edges) == list(predicted_edges))
    reverse_exact = float(
        bool(gold_edges)
        and list(reversed([(target, source) for source, target in predicted_edges]))
        == list(gold_edges)
    )

    hop_correct = sum(
        index < len(predicted_edges) and predicted_edges[index] == gold_edge
        for index, gold_edge in enumerate(gold_edges)
    )
    hop_accuracy = (
        safe_div(hop_correct, len(gold_edges))
        if gold_edges
        else float(not predicted_edges)
    )

    final_event_accuracy = 0.0
    if gold_events:
        final_event_accuracy = float(
            bool(predicted_events)
            and predicted_events[-1] == gold_events[-1]
        )

    return {
        "exact_path_match": exact_ordered,
        "reverse_exact_path_match": reverse_exact,
        "hop_accuracy": hop_accuracy,
        "path_length_accuracy": float(len(predicted_edges) == len(gold_edges)),
        "edge_precision": edge_metrics["precision"],
        "edge_recall": edge_metrics["recall"],
        "edge_f1": edge_metrics["f1"],
        "event_precision": event_metrics["precision"],
        "event_recall": event_metrics["recall"],
        "event_f1": event_metrics["f1"],
        "final_event_accuracy": final_event_accuracy,
        "path_rule_precision": path_rule_metrics["precision"],
        "path_rule_recall": path_rule_metrics["recall"],
        "path_rule_f1": path_rule_metrics["f1"],
    }


def choose_oracle_candidate(
    gold: GoldSample,
    candidates: Sequence[CandidatePath],
) -> tuple[CandidatePath, dict[str, float]]:
    empty = CandidatePath(
        rank=0,
        path_id=-1,
        edges=[],
        event_ids=[],
        rule_ids=[],
        article_ids=[],
        score=0.0,
    )
    if not candidates:
        return empty, path_metric_values(
            gold.gold_path_edges,
            empty,
            gold.gold_path_rule_ids,
        )

    scored: list[tuple[tuple[float, ...], CandidatePath, dict[str, float]]] = []
    for candidate in candidates:
        metrics = path_metric_values(
            gold.gold_path_edges,
            candidate,
            gold.gold_path_rule_ids,
        )
        sort_key = (
            metrics["exact_path_match"],
            metrics["edge_f1"],
            metrics["event_f1"],
            metrics["path_rule_f1"],
            metrics["final_event_accuracy"],
            -float(candidate.rank),
        )
        scored.append((sort_key, candidate, metrics))

    _, candidate, metrics = max(scored, key=lambda item: item[0])
    return candidate, metrics


# ============================================================
# PER-SAMPLE EVALUATION
# ============================================================

def missing_prediction(sample_id: str) -> Prediction:
    empty_path = CandidatePath(
        rank=0,
        path_id=-1,
        edges=[],
        event_ids=[],
        rule_ids=[],
        article_ids=[],
        score=0.0,
    )
    return Prediction(
        sample_id=sample_id,
        retrieved_rule_ids=[],
        retrieved_event_ids=[],
        retrieved_article_ids=[],
        selected_path=empty_path,
        candidate_paths=[],
        verification_decision="",
        decision_score=None,
        final_answer="",
        citations=[],
        runtime_seconds=None,
        pipeline_error="MISSING_PREDICTION",
        raw={},
    )


def evaluate_sample(
    gold: GoldSample,
    prediction: Prediction,
    k_values: Sequence[int],
) -> dict[str, Any]:
    successful = not bool(prediction.pipeline_error)

    rule_set = set_metrics(gold.gold_rule_ids, prediction.retrieved_rule_ids)
    event_set = set_metrics(gold.gold_event_ids, prediction.retrieved_event_ids)
    article_set = set_metrics(
        gold.gold_article_ids,
        prediction.retrieved_article_ids,
    )
    citation_set = set_metrics(gold.gold_citations, prediction.citations)

    top1_path_metrics = path_metric_values(
        gold.gold_path_edges,
        prediction.selected_path,
        gold.gold_path_rule_ids,
    )
    oracle_path, oracle_path_metrics = choose_oracle_candidate(
        gold,
        prediction.candidate_paths,
    )

    predicted_decision = normalize_decision(prediction.verification_decision)
    verification_correct = float(
        bool(predicted_decision)
        and predicted_decision == gold.gold_decision
    )

    answer_precision, answer_recall, answer_f1 = token_overlap_metrics(
        gold.answer,
        prediction.final_answer,
    )
    answer_rouge_l = rouge_l_f1(gold.answer, prediction.final_answer)
    answer_em = normalized_exact_match(gold.answer, prediction.final_answer)

    row: dict[str, Any] = {
        "id": gold.sample_id,
        "question": gold.question,
        "question_type": gold.question_type,
        "difficulty": gold.difficulty,
        "requires_counterfactual": int(gold.requires_counterfactual),
        "prediction_present": int(prediction.pipeline_error != "MISSING_PREDICTION"),
        "successful": int(successful),
        "pipeline_error": prediction.pipeline_error,

        "gold_rule_count": len(gold.gold_rule_ids),
        "predicted_rule_count": len(prediction.retrieved_rule_ids),
        "rule_true_positive": rule_set["true_positive"],
        "rule_set_precision": rule_set["precision"],
        "rule_set_recall": rule_set["recall"],
        "rule_set_f1": rule_set["f1"],
        "rule_set_exact": rule_set["exact"],
        "rule_mrr": reciprocal_rank(
            gold.gold_rule_ids,
            prediction.retrieved_rule_ids,
        ),
        "rule_map": average_precision(
            gold.gold_rule_ids,
            prediction.retrieved_rule_ids,
        ),

        "gold_event_count": len(gold.gold_event_ids),
        "predicted_event_count": len(prediction.retrieved_event_ids),
        "event_true_positive": event_set["true_positive"],
        "event_set_precision": event_set["precision"],
        "event_set_recall": event_set["recall"],
        "event_set_f1": event_set["f1"],
        "event_set_exact": event_set["exact"],
        "event_mrr": reciprocal_rank(
            gold.gold_event_ids,
            prediction.retrieved_event_ids,
        ),

        "gold_article_count": len(gold.gold_article_ids),
        "predicted_article_count": len(prediction.retrieved_article_ids),
        "article_true_positive": article_set["true_positive"],
        "article_set_precision": article_set["precision"],
        "article_set_recall": article_set["recall"],
        "article_set_f1": article_set["f1"],
        "article_set_exact": article_set["exact"],

        "top1_path_id": prediction.selected_path.path_id,
        "top1_path_rank": prediction.selected_path.rank,
        "top1_path_edges": "|".join(
            f"{source}->{target}"
            for source, target in prediction.selected_path.edges
        ),
        "top1_path_event_ids": "|".join(prediction.selected_path.event_ids),
        "top1_path_rule_ids": "|".join(prediction.selected_path.rule_ids),
        "top1_exact_path_match": top1_path_metrics["exact_path_match"],
        "top1_reverse_exact_path_match": top1_path_metrics[
            "reverse_exact_path_match"
        ],
        "top1_hop_accuracy": top1_path_metrics["hop_accuracy"],
        "top1_path_length_accuracy": top1_path_metrics[
            "path_length_accuracy"
        ],
        "top1_edge_precision": top1_path_metrics["edge_precision"],
        "top1_edge_recall": top1_path_metrics["edge_recall"],
        "top1_edge_f1": top1_path_metrics["edge_f1"],
        "top1_event_precision": top1_path_metrics["event_precision"],
        "top1_event_recall": top1_path_metrics["event_recall"],
        "top1_event_f1": top1_path_metrics["event_f1"],
        "top1_final_event_accuracy": top1_path_metrics[
            "final_event_accuracy"
        ],
        "top1_path_rule_precision": top1_path_metrics[
            "path_rule_precision"
        ],
        "top1_path_rule_recall": top1_path_metrics["path_rule_recall"],
        "top1_path_rule_f1": top1_path_metrics["path_rule_f1"],

        "candidate_path_count": len(prediction.candidate_paths),
        "oracle_best_path_id": oracle_path.path_id,
        "oracle_best_path_rank": oracle_path.rank,
        "oracle_best_path_edges": "|".join(
            f"{source}->{target}" for source, target in oracle_path.edges
        ),
        "oracle_exact_path_match": oracle_path_metrics["exact_path_match"],
        "oracle_edge_f1": oracle_path_metrics["edge_f1"],
        "oracle_event_f1": oracle_path_metrics["event_f1"],
        "oracle_path_rule_f1": oracle_path_metrics["path_rule_f1"],
        "oracle_final_event_accuracy": oracle_path_metrics[
            "final_event_accuracy"
        ],
        "path_ranking_error": float(
            oracle_path_metrics["exact_path_match"] == 1.0
            and top1_path_metrics["exact_path_match"] == 0.0
        ),
        "path_retrieval_error": float(
            oracle_path_metrics["exact_path_match"] == 0.0
        ),

        "gold_decision": gold.gold_decision,
        "predicted_decision": predicted_decision,
        "verification_correct": verification_correct,
        "decision_score": prediction.decision_score,

        "answer_token_precision": answer_precision,
        "answer_token_recall": answer_recall,
        "answer_token_f1": answer_f1,
        "answer_rouge_l_f1": answer_rouge_l,
        "answer_exact_match": answer_em,
        "answer_length_tokens": len(tokenize(prediction.final_answer)),
        "gold_answer_length_tokens": len(tokenize(gold.answer)),
        "answer_present": float(bool(prediction.final_answer)),

        "citation_true_positive": citation_set["true_positive"],
        "citation_precision": citation_set["precision"],
        "citation_recall": citation_set["recall"],
        "citation_f1": citation_set["f1"],
        "citation_exact": citation_set["exact"],

        "runtime_seconds": prediction.runtime_seconds,

        "gold_rule_ids": "|".join(gold.gold_rule_ids),
        "predicted_rule_ids": "|".join(prediction.retrieved_rule_ids),
        "gold_event_ids": "|".join(gold.gold_event_ids),
        "predicted_event_ids": "|".join(prediction.retrieved_event_ids),
        "gold_article_ids": "|".join(gold.gold_article_ids),
        "predicted_article_ids": "|".join(prediction.retrieved_article_ids),
        "gold_path_edges": "|".join(
            f"{source}->{target}" for source, target in gold.gold_path_edges
        ),
        "gold_citations": "|".join(gold.gold_citations),
        "predicted_citations": "|".join(prediction.citations),
        "gold_answer": gold.answer,
        "predicted_answer": prediction.final_answer,
    }

    for k in k_values:
        row[f"rule_recall_at_{k}"] = recall_at_k(
            gold.gold_rule_ids,
            prediction.retrieved_rule_ids,
            k,
        )
        row[f"rule_precision_at_{k}"] = precision_at_k(
            gold.gold_rule_ids,
            prediction.retrieved_rule_ids,
            k,
        )
        row[f"rule_hit_at_{k}"] = hit_at_k(
            gold.gold_rule_ids,
            prediction.retrieved_rule_ids,
            k,
        )

        row[f"event_recall_at_{k}"] = recall_at_k(
            gold.gold_event_ids,
            prediction.retrieved_event_ids,
            k,
        )
        row[f"event_precision_at_{k}"] = precision_at_k(
            gold.gold_event_ids,
            prediction.retrieved_event_ids,
            k,
        )
        row[f"event_hit_at_{k}"] = hit_at_k(
            gold.gold_event_ids,
            prediction.retrieved_event_ids,
            k,
        )

    errors: list[str] = []
    primary_k = 5 if 5 in k_values else max(k_values)

    if not successful:
        errors.append("PIPELINE_ERROR")
    if row[f"rule_recall_at_{primary_k}"] < 1.0:
        errors.append("RULE_RETRIEVAL_MISS")
    if row[f"event_recall_at_{primary_k}"] < 1.0:
        errors.append("EVENT_RETRIEVAL_MISS")
    if gold.gold_path_edges and row["top1_exact_path_match"] < 1.0:
        errors.append("TOP1_PATH_MISMATCH")
    if row["path_ranking_error"] == 1.0:
        errors.append("PATH_RANKING_ERROR")
    if row["path_retrieval_error"] == 1.0:
        errors.append("PATH_NOT_RETRIEVED")
    if verification_correct < 1.0:
        errors.append("VERIFICATION_ERROR")
    if answer_f1 < 0.50:
        errors.append("LOW_ANSWER_TOKEN_F1")
    if citation_set["recall"] < 1.0:
        errors.append("CITATION_MISS")
    if citation_set["precision"] < 1.0:
        errors.append("EXTRA_CITATION")

    row["error_types"] = "|".join(errors)
    return recursive_round(row)


# ============================================================
# AGGREGATION
# ============================================================

def numeric_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def numeric_mean(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Optional[float]:
    values = numeric_values(rows, key)
    return statistics.fmean(values) if values else None


def numeric_sum(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> float:
    return sum(numeric_values(rows, key))


def micro_set_metrics(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> dict[str, float]:
    true_positive = numeric_sum(rows, f"{prefix}_true_positive")
    predicted_total = numeric_sum(rows, f"predicted_{prefix}_count")
    gold_total = numeric_sum(rows, f"gold_{prefix}_count")

    precision = safe_div(true_positive, predicted_total)
    recall = safe_div(true_positive, gold_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
    }


def build_confusion_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    labels = list(DECISION_LABELS)
    observed = {
        normalize_decision(row.get("gold_decision"))
        for row in rows
        if normalize_decision(row.get("gold_decision"))
    }
    predicted = {
        normalize_decision(row.get("predicted_decision"))
        for row in rows
        if normalize_decision(row.get("predicted_decision"))
    }

    for label in sorted(observed | predicted):
        if label and label not in labels:
            labels.append(label)

    matrix = {
        gold: {prediction: 0 for prediction in labels}
        for gold in labels
    }

    for row in rows:
        gold = normalize_decision(row.get("gold_decision"))
        prediction = normalize_decision(row.get("predicted_decision"))
        if not gold:
            continue
        if gold not in matrix:
            matrix[gold] = {label: 0 for label in labels}
        if prediction not in matrix[gold]:
            for values in matrix.values():
                values.setdefault(prediction, 0)
        matrix[gold][prediction] += 1

    return matrix


def classification_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix = build_confusion_matrix(rows)
    labels = list(matrix)
    all_predicted_labels = set()
    for values in matrix.values():
        all_predicted_labels.update(values)
    for label in sorted(all_predicted_labels):
        if label not in labels:
            labels.append(label)

    per_class: dict[str, dict[str, float | int]] = {}
    total = sum(sum(values.values()) for values in matrix.values())
    correct = sum(matrix.get(label, {}).get(label, 0) for label in labels)

    class_f1_values: list[float] = []
    class_recall_values: list[float] = []

    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(
            matrix.get(other, {}).get(label, 0)
            for other in labels
            if other != label
        )
        fn = sum(
            count
            for predicted_label, count in matrix.get(label, {}).items()
            if predicted_label != label
        )
        support = sum(matrix.get(label, {}).values())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = harmonic_f1(precision, recall)

        if support > 0:
            class_f1_values.append(f1)
            class_recall_values.append(recall)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "sample_count": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": statistics.fmean(class_f1_values) if class_f1_values else None,
        "balanced_accuracy": (
            statistics.fmean(class_recall_values)
            if class_recall_values
            else None
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def aggregate_group(
    rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    rows = list(rows)
    successful_rows = [
        row for row in rows if safe_int(row.get("successful"), 0) == 1
    ]
    counterfactual_rows = [
        row
        for row in successful_rows
        if safe_int(row.get("requires_counterfactual"), 0) == 1
    ]

    rule_micro = micro_set_metrics(successful_rows, "rule")
    event_micro = micro_set_metrics(successful_rows, "event")
    article_micro = micro_set_metrics(successful_rows, "article")

    retrieval: dict[str, Any] = {
        "rule_macro_precision": numeric_mean(successful_rows, "rule_set_precision"),
        "rule_macro_recall": numeric_mean(successful_rows, "rule_set_recall"),
        "rule_macro_f1": numeric_mean(successful_rows, "rule_set_f1"),
        "rule_micro_precision": rule_micro["precision"],
        "rule_micro_recall": rule_micro["recall"],
        "rule_micro_f1": rule_micro["f1"],
        "rule_exact_set_match": numeric_mean(successful_rows, "rule_set_exact"),
        "rule_mrr": numeric_mean(successful_rows, "rule_mrr"),
        "rule_map": numeric_mean(successful_rows, "rule_map"),
        "event_macro_precision": numeric_mean(successful_rows, "event_set_precision"),
        "event_macro_recall": numeric_mean(successful_rows, "event_set_recall"),
        "event_macro_f1": numeric_mean(successful_rows, "event_set_f1"),
        "event_micro_precision": event_micro["precision"],
        "event_micro_recall": event_micro["recall"],
        "event_micro_f1": event_micro["f1"],
        "event_exact_set_match": numeric_mean(successful_rows, "event_set_exact"),
        "event_mrr": numeric_mean(successful_rows, "event_mrr"),
        "article_macro_precision": numeric_mean(successful_rows, "article_set_precision"),
        "article_macro_recall": numeric_mean(successful_rows, "article_set_recall"),
        "article_macro_f1": numeric_mean(successful_rows, "article_set_f1"),
        "article_micro_precision": article_micro["precision"],
        "article_micro_recall": article_micro["recall"],
        "article_micro_f1": article_micro["f1"],
        "article_exact_set_match": numeric_mean(successful_rows, "article_set_exact"),
    }

    for k in k_values:
        retrieval[f"rule_recall_at_{k}"] = numeric_mean(
            successful_rows,
            f"rule_recall_at_{k}",
        )
        retrieval[f"rule_precision_at_{k}"] = numeric_mean(
            successful_rows,
            f"rule_precision_at_{k}",
        )
        retrieval[f"rule_hit_at_{k}"] = numeric_mean(
            successful_rows,
            f"rule_hit_at_{k}",
        )
        retrieval[f"event_recall_at_{k}"] = numeric_mean(
            successful_rows,
            f"event_recall_at_{k}",
        )
        retrieval[f"event_precision_at_{k}"] = numeric_mean(
            successful_rows,
            f"event_precision_at_{k}",
        )
        retrieval[f"event_hit_at_{k}"] = numeric_mean(
            successful_rows,
            f"event_hit_at_{k}",
        )

    runtime_values = numeric_values(successful_rows, "runtime_seconds")

    result = {
        "sample_count": len(rows),
        "successful_sample_count": len(successful_rows),
        "failed_sample_count": len(rows) - len(successful_rows),
        "prediction_coverage": numeric_mean(rows, "prediction_present"),
        "success_rate": safe_div(len(successful_rows), len(rows)),
        "average_runtime_seconds": (
            statistics.fmean(runtime_values) if runtime_values else None
        ),
        "median_runtime_seconds": (
            statistics.median(runtime_values) if runtime_values else None
        ),
        "retrieval": retrieval,
        "causal_path": {
            "top1_exact_path_match": numeric_mean(
                successful_rows,
                "top1_exact_path_match",
            ),
            "top1_reverse_exact_path_match": numeric_mean(
                successful_rows,
                "top1_reverse_exact_path_match",
            ),
            "top1_hop_accuracy": numeric_mean(
                successful_rows,
                "top1_hop_accuracy",
            ),
            "top1_path_length_accuracy": numeric_mean(
                successful_rows,
                "top1_path_length_accuracy",
            ),
            "top1_edge_precision": numeric_mean(
                successful_rows,
                "top1_edge_precision",
            ),
            "top1_edge_recall": numeric_mean(successful_rows, "top1_edge_recall"),
            "top1_edge_f1": numeric_mean(successful_rows, "top1_edge_f1"),
            "top1_event_precision": numeric_mean(
                successful_rows,
                "top1_event_precision",
            ),
            "top1_event_recall": numeric_mean(successful_rows, "top1_event_recall"),
            "top1_event_f1": numeric_mean(successful_rows, "top1_event_f1"),
            "top1_final_event_accuracy": numeric_mean(
                successful_rows,
                "top1_final_event_accuracy",
            ),
            "top1_path_rule_f1": numeric_mean(
                successful_rows,
                "top1_path_rule_f1",
            ),
            "oracle_exact_path_match": numeric_mean(
                successful_rows,
                "oracle_exact_path_match",
            ),
            "oracle_edge_f1": numeric_mean(successful_rows, "oracle_edge_f1"),
            "oracle_event_f1": numeric_mean(successful_rows, "oracle_event_f1"),
            "oracle_path_rule_f1": numeric_mean(
                successful_rows,
                "oracle_path_rule_f1",
            ),
            "oracle_final_event_accuracy": numeric_mean(
                successful_rows,
                "oracle_final_event_accuracy",
            ),
            "path_ranking_error_rate": numeric_mean(
                successful_rows,
                "path_ranking_error",
            ),
            "path_retrieval_error_rate": numeric_mean(
                successful_rows,
                "path_retrieval_error",
            ),
        },
        "verification": classification_metrics(successful_rows),
        "counterfactual_subset": {
            "sample_count": len(counterfactual_rows),
            "verification_accuracy": numeric_mean(
                counterfactual_rows,
                "verification_correct",
            ),
            "answer_token_f1": numeric_mean(
                counterfactual_rows,
                "answer_token_f1",
            ),
            "citation_f1": numeric_mean(counterfactual_rows, "citation_f1"),
        },
        "answer": {
            "token_precision": numeric_mean(
                successful_rows,
                "answer_token_precision",
            ),
            "token_recall": numeric_mean(successful_rows, "answer_token_recall"),
            "token_f1": numeric_mean(successful_rows, "answer_token_f1"),
            "rouge_l_f1": numeric_mean(successful_rows, "answer_rouge_l_f1"),
            "normalized_exact_match": numeric_mean(
                successful_rows,
                "answer_exact_match",
            ),
            "answer_present": numeric_mean(successful_rows, "answer_present"),
            "average_prediction_length_tokens": numeric_mean(
                successful_rows,
                "answer_length_tokens",
            ),
            "average_gold_length_tokens": numeric_mean(
                successful_rows,
                "gold_answer_length_tokens",
            ),
        },
        "citation": {
            "precision": numeric_mean(successful_rows, "citation_precision"),
            "recall": numeric_mean(successful_rows, "citation_recall"),
            "f1": numeric_mean(successful_rows, "citation_f1"),
            "exact_match": numeric_mean(successful_rows, "citation_exact"),
        },
    }

    return recursive_round(result)


def group_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = safe_string(row.get(key)) or "UNKNOWN"
        grouped[group_name].append(row)
    return dict(grouped)


# ============================================================
# ERROR ANALYSIS
# ============================================================

def build_error_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    error_rows: list[dict[str, Any]] = []

    for row in rows:
        if not safe_string(row.get("error_types")):
            continue

        error_rows.append({
            "id": row.get("id"),
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "requires_counterfactual": row.get("requires_counterfactual"),
            "error_types": row.get("error_types"),
            "pipeline_error": row.get("pipeline_error"),
            "rule_recall_at_5": row.get("rule_recall_at_5"),
            "event_recall_at_5": row.get("event_recall_at_5"),
            "top1_exact_path_match": row.get("top1_exact_path_match"),
            "top1_edge_f1": row.get("top1_edge_f1"),
            "oracle_exact_path_match": row.get("oracle_exact_path_match"),
            "oracle_best_path_rank": row.get("oracle_best_path_rank"),
            "gold_decision": row.get("gold_decision"),
            "predicted_decision": row.get("predicted_decision"),
            "answer_token_f1": row.get("answer_token_f1"),
            "answer_rouge_l_f1": row.get("answer_rouge_l_f1"),
            "citation_precision": row.get("citation_precision"),
            "citation_recall": row.get("citation_recall"),
            "gold_rule_ids": row.get("gold_rule_ids"),
            "predicted_rule_ids": row.get("predicted_rule_ids"),
            "gold_event_ids": row.get("gold_event_ids"),
            "predicted_event_ids": row.get("predicted_event_ids"),
            "gold_path_edges": row.get("gold_path_edges"),
            "top1_path_edges": row.get("top1_path_edges"),
            "gold_citations": row.get("gold_citations"),
            "predicted_citations": row.get("predicted_citations"),
            "gold_answer": row.get("gold_answer"),
            "predicted_answer": row.get("predicted_answer"),
        })

    return error_rows


def confusion_matrix_rows(
    matrix: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    labels: list[str] = list(matrix)
    for values in matrix.values():
        for label in values:
            if label not in labels:
                labels.append(label)

    rows: list[dict[str, Any]] = []
    for gold_label in labels:
        row: dict[str, Any] = {"gold_label": gold_label}
        for predicted_label in labels:
            row[f"pred_{predicted_label}"] = matrix.get(
                gold_label,
                {},
            ).get(predicted_label, 0)
        rows.append(row)
    return rows


def group_summary_rows(
    metrics_by_group: Mapping[str, Mapping[str, Any]],
    group_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, metrics in metrics_by_group.items():
        retrieval = metrics.get("retrieval") or {}
        path = metrics.get("causal_path") or {}
        verification = metrics.get("verification") or {}
        answer = metrics.get("answer") or {}
        citation = metrics.get("citation") or {}

        rows.append({
            "group_type": group_type,
            "group_name": group_name,
            "sample_count": metrics.get("sample_count"),
            "prediction_coverage": metrics.get("prediction_coverage"),
            "rule_recall_at_5": retrieval.get("rule_recall_at_5"),
            "event_recall_at_5": retrieval.get("event_recall_at_5"),
            "top1_exact_path_match": path.get("top1_exact_path_match"),
            "oracle_exact_path_match": path.get("oracle_exact_path_match"),
            "verification_accuracy": verification.get("accuracy"),
            "verification_macro_f1": verification.get("macro_f1"),
            "answer_token_f1": answer.get("token_f1"),
            "answer_rouge_l_f1": answer.get("rouge_l_f1"),
            "citation_f1": citation.get("f1"),
        })
    return rows


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def format_percentage(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def print_summary(report: Mapping[str, Any], k_values: Sequence[int]) -> None:
    overall = report["overall"]
    retrieval = overall["retrieval"]
    path = overall["causal_path"]
    verification = overall["verification"]
    answer = overall["answer"]
    citation = overall["citation"]
    primary_k = 5 if 5 in k_values else max(k_values)

    print("\n" + "=" * 80)
    print("BLHS CAUSALRAG - DETAILED EVALUATION METRICS")
    print("=" * 80)
    print("Step 7 version            :", STEP7_VERSION)
    print("Samples                   :", overall["sample_count"])
    print("Prediction coverage       :", format_percentage(overall["prediction_coverage"]))
    print("Success rate              :", format_percentage(overall["success_rate"]))

    print("\n[Retrieval]")
    print(
        f"Rule Recall@{primary_k} / Event Recall@{primary_k}: "
        f"{format_percentage(retrieval[f'rule_recall_at_{primary_k}'])} / "
        f"{format_percentage(retrieval[f'event_recall_at_{primary_k}'])}"
    )
    print(
        "Rule Macro P/R/F1       : "
        f"{format_percentage(retrieval['rule_macro_precision'])} / "
        f"{format_percentage(retrieval['rule_macro_recall'])} / "
        f"{format_percentage(retrieval['rule_macro_f1'])}"
    )
    print(
        f"Rule MRR / MAP            : "
        f"{retrieval['rule_mrr'] or 0:.4f} / "
        f"{retrieval['rule_map'] or 0:.4f}"
    )

    print("\n[Causal Path]")
    print(
        "Top-1 Exact / Edge F1   : "
        f"{format_percentage(path['top1_exact_path_match'])} / "
        f"{format_percentage(path['top1_edge_f1'])}"
    )
    print(
        "Oracle Exact / Edge F1  : "
        f"{format_percentage(path['oracle_exact_path_match'])} / "
        f"{format_percentage(path['oracle_edge_f1'])}"
    )
    print(
        "Ranking / Retrieval err : "
        f"{format_percentage(path['path_ranking_error_rate'])} / "
        f"{format_percentage(path['path_retrieval_error_rate'])}"
    )

    print("\n[Verification]")
    print(
        "Accuracy / Macro-F1     : "
        f"{format_percentage(verification['accuracy'])} / "
        f"{format_percentage(verification['macro_f1'])}"
    )
    print(
        "Balanced Accuracy       : "
        f"{format_percentage(verification['balanced_accuracy'])}"
    )

    print("\n[Answer]")
    print(
        "Token F1 / ROUGE-L / EM : "
        f"{format_percentage(answer['token_f1'])} / "
        f"{format_percentage(answer['rouge_l_f1'])} / "
        f"{format_percentage(answer['normalized_exact_match'])}"
    )

    print("\n[Citation]")
    print(
        "Precision / Recall / F1 : "
        f"{format_percentage(citation['precision'])} / "
        f"{format_percentage(citation['recall'])} / "
        f"{format_percentage(citation['f1'])}"
    )
    print("=" * 80)


def build_summary_markdown(report: Mapping[str, Any], k_values: Sequence[int]) -> str:
    overall = report["overall"]
    retrieval = overall["retrieval"]
    path = overall["causal_path"]
    verification = overall["verification"]
    answer = overall["answer"]
    citation = overall["citation"]
    primary_k = 5 if 5 in k_values else max(k_values)

    lines = [
        "# BLHS CausalRAG — Detailed Evaluation Metrics",
        "",
        f"- Step 7 version: `{STEP7_VERSION}`",
        f"- Samples: **{overall['sample_count']}**",
        f"- Prediction coverage: **{format_percentage(overall['prediction_coverage'])}**",
        f"- Success rate: **{format_percentage(overall['success_rate'])}**",
        "",
        "## Main metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rule Recall@{primary_k} | {format_percentage(retrieval[f'rule_recall_at_{primary_k}'])} |",
        f"| Event Recall@{primary_k} | {format_percentage(retrieval[f'event_recall_at_{primary_k}'])} |",
        f"| Top-1 Exact Path | {format_percentage(path['top1_exact_path_match'])} |",
        f"| Oracle Exact Path | {format_percentage(path['oracle_exact_path_match'])} |",
        f"| Verification Accuracy | {format_percentage(verification['accuracy'])} |",
        f"| Verification Macro-F1 | {format_percentage(verification['macro_f1'])} |",
        f"| Verification Balanced Accuracy | {format_percentage(verification['balanced_accuracy'])} |",
        f"| Answer Token F1 | {format_percentage(answer['token_f1'])} |",
        f"| Answer ROUGE-L F1 | {format_percentage(answer['rouge_l_f1'])} |",
        f"| Citation F1 | {format_percentage(citation['f1'])} |",
        "",
        "## Interpretation notes",
        "",
        "- Top-1 path evaluates the path actually selected by the pipeline.",
        "- Oracle path is a diagnostic over returned causal paths; it must not replace top-1 results.",
        "- The gap between oracle and top-1 exact path indicates a path-ranking problem.",
        "- Verification Macro-F1 and Balanced Accuracy should be reported with Accuracy when classes are imbalanced.",
    ]
    return "\n".join(lines) + "\n"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tính detailed evaluation metrics cho benchmark BLHS và "
            "pipeline_predictions.json của CausalRAG."
        )
    )
    parser.add_argument(
        "--benchmark",
        "--gold",
        dest="benchmark",
        default=DEFAULT_BENCHMARK_PATH,
        help="Benchmark BLHS JSON/JSONL.",
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTION_PATH,
        help="Prediction JSON/JSONL do Step 5.5 sinh ra.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Dừng nếu thiếu prediction hoặc có prediction id ngoài benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.k_values or any(k < 1 for k in args.k_values):
        raise ValueError("--k-values phải chứa các số nguyên dương.")
    k_values = sorted(set(args.k_values))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_metadata, samples = load_benchmark(
        args.benchmark,
        limit=args.limit,
    )
    prediction_metadata, predictions = load_predictions(args.predictions)

    gold_ids = [sample.sample_id for sample in samples]
    gold_id_set = set(gold_ids)
    prediction_id_set = set(predictions)

    missing_prediction_ids = sorted(gold_id_set - prediction_id_set)
    unexpected_prediction_ids = sorted(prediction_id_set - gold_id_set)

    if args.strict and missing_prediction_ids:
        raise ValueError(
            "Thiếu prediction cho: " + ", ".join(missing_prediction_ids[:20])
        )
    if args.strict and unexpected_prediction_ids:
        raise ValueError(
            "Prediction id không thuộc benchmark: "
            + ", ".join(unexpected_prediction_ids[:20])
        )

    rows: list[dict[str, Any]] = []
    for sample in samples:
        prediction = predictions.get(sample.sample_id) or missing_prediction(
            sample.sample_id
        )
        rows.append(evaluate_sample(sample, prediction, k_values))

    overall = aggregate_group(rows, k_values)

    by_question_type = {
        group_name: aggregate_group(group_values, k_values)
        for group_name, group_values in sorted(
            group_rows(rows, "question_type").items()
        )
    }
    by_difficulty = {
        group_name: aggregate_group(group_values, k_values)
        for group_name, group_values in sorted(
            group_rows(rows, "difficulty").items()
        )
    }
    by_counterfactual = {
        group_name: aggregate_group(group_values, k_values)
        for group_name, group_values in sorted(
            group_rows(rows, "requires_counterfactual").items()
        )
    }

    report = {
        "version": STEP7_VERSION,
        "created_at_utc": utc_now_iso(),
        "input": {
            "benchmark_path": args.benchmark,
            "prediction_path": args.predictions,
            "benchmark_metadata": benchmark_metadata,
            "prediction_metadata": prediction_metadata,
            "k_values": k_values,
            "limit": args.limit,
            "missing_prediction_ids": missing_prediction_ids,
            "unexpected_prediction_ids": unexpected_prediction_ids,
        },
        "overall": overall,
        "by_question_type": by_question_type,
        "by_difficulty": by_difficulty,
        "by_requires_counterfactual": by_counterfactual,
        "metric_notes": {
            "top1_path": (
                "Đánh giá reasoning_path thực sự được pipeline chọn."
            ),
            "oracle_path": (
                "Diagnostic chọn path khớp gold tốt nhất trong retrieval.causal_paths; "
                "không phải metric deploy và không thay thế top-1."
            ),
            "verification": (
                "Báo cáo Accuracy cùng Macro-F1 và Balanced Accuracy để tránh "
                "đánh giá sai khi benchmark mất cân bằng lớp."
            ),
            "answer_token_f1": (
                "Bag-of-token overlap F1, giữ tiếng Việt và bỏ dấu câu."
            ),
            "rouge_l_f1": (
                "ROUGE-L F1 dựa trên longest common subsequence ở mức token."
            ),
            "citation": (
                "So sánh tập article id sau khi chuẩn hóa nhãn Điều X."
            ),
        },
    }

    error_rows = build_error_rows(rows)
    confusion_rows = confusion_matrix_rows(
        overall["verification"]["confusion_matrix"]
    )

    group_rows_output: list[dict[str, Any]] = []
    group_rows_output.extend(
        group_summary_rows(by_question_type, "question_type")
    )
    group_rows_output.extend(
        group_summary_rows(by_difficulty, "difficulty")
    )
    group_rows_output.extend(
        group_summary_rows(by_counterfactual, "requires_counterfactual")
    )

    metrics_path = output_dir / "evaluation_metrics_report.json"
    by_sample_path = output_dir / "evaluation_metrics_by_sample.csv"
    errors_path = output_dir / "evaluation_metric_errors.csv"
    confusion_path = output_dir / "verification_confusion_matrix.csv"
    groups_path = output_dir / "evaluation_metrics_by_group.csv"
    summary_path = output_dir / "evaluation_metrics_summary.md"

    save_json(report, metrics_path)
    save_csv(rows, by_sample_path)
    save_csv(error_rows, errors_path)
    save_csv(confusion_rows, confusion_path)
    save_csv(group_rows_output, groups_path)
    summary_path.write_text(
        build_summary_markdown(report, k_values),
        encoding="utf-8",
    )

    print_summary(report, k_values)
    print("\nMetrics JSON             :", metrics_path)
    print("Per-sample CSV           :", by_sample_path)
    print("Error analysis CSV       :", errors_path)
    print("Confusion matrix CSV     :", confusion_path)
    print("Group metrics CSV        :", groups_path)
    print("Summary Markdown         :", summary_path)

    if missing_prediction_ids:
        print(
            "\nWarning - missing predictions:",
            ", ".join(missing_prediction_ids[:20]),
        )
    if unexpected_prediction_ids:
        print(
            "\nWarning - unexpected prediction ids:",
            ", ".join(unexpected_prediction_ids[:20]),
        )


if __name__ == "__main__":
    main()
