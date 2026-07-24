#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_run_evaluation_pipeline.py

Đánh giá toàn bộ pipeline CausalRAG trên benchmark BLHS v2.0:

    Event retrieval
    -> Graph expansion / path retrieval
    -> Rule retrieval
    -> Counterfactual verification
    -> Final answer generation
    -> Citation evaluation

Đầu vào
-------
1. Benchmark JSON:
   data/blhs_multihop_benchmark_250_updated.json

2. Prediction JSON hoặc JSONL được sinh bởi pipeline, ở một trong các dạng:

   A. Danh sách:
      [
        {
          "id": "BLHS_MH_0001",
          "retrieved_rule_ids": ["176", "203"],
          "retrieved_event_ids": ["...", "..."],
          "reasoning_path": [...],
          "verification_decision": "SUPPORTED",
          "final_answer": "... [E1] [E2]",
          "citations": ["Điều 62", "Điều 73"]
        }
      ]

   B. Object:
      {"predictions": [...]}

   C. Dictionary theo id:
      {
        "BLHS_MH_0001": {...},
        "BLHS_MH_0002": {...}
      }

Script có cơ chế alias linh hoạt, nên cũng đọc được nhiều field lồng nhau như:
- retrieval.retrieved_rules
- retrieved_rules
- rule_ids
- verification.final_decision
- answer
- final_answer
- evidence
- citations

Chế độ chạy
-----------
1. Đánh giá file dự đoán đã có:

   python 6_run_evaluation_pipeline.py \
       --benchmark data/blhs_multihop_benchmark_250_updated.json \
       --predictions data/pipeline_predictions.json \
       --output-dir evaluation_results

2. Kiểm tra evaluator bằng oracle predictions:

   python 6_run_evaluation_pipeline.py \
       --benchmark data/blhs_multihop_benchmark_250_updated.json \
       --oracle \
       --output-dir evaluation_results_oracle

3. Chạy thử chỉ N câu:

   python 6_run_evaluation_pipeline.py \
       --benchmark data/blhs_multihop_benchmark_250_updated.json \
       --predictions data/pipeline_predictions.json \
       --limit 20

Đầu ra
-----
- evaluation_report.json
- evaluation_per_question.csv
- evaluation_summary.md
- evaluation_errors.json
- normalized_predictions.json

Phụ thuộc tùy chọn
------------------
- rouge-score: tính ROUGE-L chuẩn
- bert-score: tính BERTScore

Nếu chưa cài, script vẫn chạy và dùng:
- ROUGE-L tự cài đặt bằng LCS
- BERTScore = null

Cài thêm:
    pip install rouge-score bert-score
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


# ============================================================
# Utilities
# ============================================================

VIETNAMESE_TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ỹĐđ]+(?:['’-][0-9A-Za-zÀ-ỹĐđ]+)?",
    flags=re.UNICODE,
)
CITATION_RE = re.compile(
    r"(?:Điều|điều)\s*([0-9]+(?:\.[0-9]+)?)"
    r"|\[(?:E|R|A)?\s*([0-9]+)\]",
    flags=re.UNICODE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.fmean(cleaned) if cleaned else None


def harmonic_f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def dedupe(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        normalized = normalize_id(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_text(text: Any) -> str:
    """Chuẩn hóa để tính EM/F1 nhưng vẫn giữ chữ tiếng Việt."""
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[“”\"`]", "", text)
    text = re.sub(r"[^\wÀ-ỹĐđ\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> List[str]:
    normalized = unicodedata.normalize("NFC", "" if text is None else str(text)).lower()
    return VIETNAMESE_TOKEN_RE.findall(normalized)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    precision = safe_div(overlap, len(pred_tokens))
    recall = safe_div(overlap, len(gold_tokens))
    return harmonic_f1(precision, recall)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """LCS với bộ nhớ O(min(n,m))."""
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1_fallback(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = safe_div(lcs, len(pred_tokens))
    recall = safe_div(lcs, len(gold_tokens))
    return harmonic_f1(precision, recall)


def set_metrics(predicted: Sequence[str], gold: Sequence[str]) -> Dict[str, float]:
    pred_set = set(dedupe(predicted))
    gold_set = set(dedupe(gold))
    true_positive = len(pred_set & gold_set)
    precision = safe_div(true_positive, len(pred_set))
    recall = safe_div(true_positive, len(gold_set))
    return {
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
        "exact": float(pred_set == gold_set),
        "intersection": float(true_positive),
    }


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(dedupe(gold_ids))
    if not gold:
        return 1.0
    top_k = set(dedupe(ranked_ids)[:k])
    return safe_div(len(top_k & gold), len(gold))


def precision_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    top_k = dedupe(ranked_ids)[:k]
    if not top_k:
        return 0.0
    gold = set(dedupe(gold_ids))
    return safe_div(sum(item in gold for item in top_k), len(top_k))


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    return float(bool(set(dedupe(ranked_ids)[:k]) & set(dedupe(gold_ids))))


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(dedupe(gold_ids))
    for index, item in enumerate(dedupe(ranked_ids), start=1):
        if item in gold:
            return 1.0 / index
    return 0.0


def average_precision(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(dedupe(gold_ids))
    if not gold:
        return 1.0

    hits = 0
    accumulated = 0.0
    for rank, item in enumerate(dedupe(ranked_ids), start=1):
        if item in gold:
            hits += 1
            accumulated += hits / rank
    return accumulated / len(gold)


def flatten_dict(data: Any, prefix: str = "") -> Dict[str, Any]:
    """Làm phẳng dictionary để tìm alias trong output nhiều tầng."""
    result: Dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            result[full_key] = value
            result.update(flatten_dict(value, full_key))
    return result


def find_first(data: Mapping[str, Any], candidates: Sequence[str], default: Any = None) -> Any:
    """
    Tìm field theo:
    - exact path
    - exact key ở dictionary đã flatten
    - suffix key
    """
    flat = flatten_dict(data)

    for candidate in candidates:
        if candidate in flat:
            return flat[candidate]

    for candidate in candidates:
        matches = [
            value for key, value in flat.items()
            if key == candidate or key.endswith(f".{candidate}")
        ]
        if matches:
            return matches[0]

    return default


def extract_ids_from_objects(
    value: Any,
    id_keys: Sequence[str],
) -> List[str]:
    if value is None:
        return []

    if isinstance(value, (str, int, float)):
        return [normalize_id(value)]

    if isinstance(value, Mapping):
        for key in id_keys:
            if key in value and value[key] is not None:
                return [normalize_id(value[key])]
        result: List[str] = []
        for child in value.values():
            if isinstance(child, (list, tuple)):
                result.extend(extract_ids_from_objects(child, id_keys))
        return dedupe(result)

    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for item in value:
            result.extend(extract_ids_from_objects(item, id_keys))
        return dedupe(result)

    return []


def normalize_decision(value: Any) -> str:
    text = normalize_text(value).replace(" ", "_").upper()

    aliases = {
        "YES": "SUPPORTED",
        "TRUE": "SUPPORTED",
        "KEEP": "SUPPORTED",
        "VERIFIED": "SUPPORTED",
        "ACCEPT": "SUPPORTED",
        "ACCEPTED": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "NO": "REJECT_DIRECT_CLAIM",
        "FALSE": "REJECT_DIRECT_CLAIM",
        "REJECT": "REJECT_DIRECT_CLAIM",
        "REJECTED": "REJECT_DIRECT_CLAIM",
        "CONTRADICTED": "REJECT_DIRECT_CLAIM",
        "NO_DIRECT_EDGE": "REJECT_DIRECT_CLAIM",
        "REJECT_DIRECT_CLAIM": "REJECT_DIRECT_CLAIM",
        "UNCERTAIN": "UNCERTAIN",
        "INSUFFICIENT": "UNCERTAIN",
        "NOT_ENOUGH_EVIDENCE": "UNCERTAIN",
    }
    return aliases.get(text, text)


def normalize_citations(values: Any, answer_text: str = "") -> List[str]:
    citations: List[str] = []

    def add_citation(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            article = (
                value.get("article_id")
                or value.get("article")
                or value.get("citation")
                or value.get("label")
            )
            if article is not None:
                add_citation(article)
            return
        text = str(value).strip()
        if not text:
            return
        match = re.search(r"(?:Điều|điều)\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            citations.append(f"Điều {match.group(1)}")
        elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
            citations.append(f"Điều {text}")
        else:
            citations.append(text)

    if isinstance(values, (list, tuple)):
        for value in values:
            add_citation(value)
    elif values is not None:
        add_citation(values)

    # Trích citation "Điều X" từ answer.
    for match in re.finditer(r"(?:Điều|điều)\s*([0-9]+(?:\.[0-9]+)?)", answer_text):
        citations.append(f"Điều {match.group(1)}")

    return dedupe(citations)


# ============================================================
# Data models
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
    gold_rule_ids: List[str]
    gold_article_ids: List[str]
    gold_event_ids: List[str]
    gold_path: List[Dict[str, str]]
    gold_citations: List[str]
    raw: Dict[str, Any] = field(repr=False)


@dataclass
class Prediction:
    sample_id: str
    retrieved_rule_ids: List[str]
    retrieved_event_ids: List[str]
    path_edges: List[Tuple[str, str]]
    path_event_ids: List[str]
    verification_decision: str
    final_answer: str
    citations: List[str]
    raw: Dict[str, Any] = field(repr=False)


# ============================================================
# Load benchmark
# ============================================================

def load_json_or_jsonl(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
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


def parse_gold_sample(raw: Mapping[str, Any]) -> GoldSample:
    evaluation = raw.get("evaluation") or {}

    gold_rule_ids = dedupe(
        evaluation.get("gold_rule_ids")
        or raw.get("supporting_rule_ids")
        or []
    )
    gold_article_ids = dedupe(
        evaluation.get("gold_article_ids")
        or raw.get("supporting_article_ids")
        or []
    )
    gold_event_ids = dedupe(
        evaluation.get("gold_event_ids")
        or [
            event.get("event_id")
            for event in raw.get("supporting_event_chain", [])
            if isinstance(event, Mapping)
        ]
    )
    gold_path = evaluation.get("gold_path") or []
    gold_citations = normalize_citations(
        evaluation.get("gold_citations")
        or raw.get("expected_citations")
        or []
    )

    requires_counterfactual = bool(
        evaluation.get(
            "requires_counterfactual",
            raw.get("question_type") == "yes_no_counterexample",
        )
    )

    gold_decision = normalize_decision(
        evaluation.get("gold_decision")
        or raw.get("expected_label")
        or ("REJECT_DIRECT_CLAIM" if requires_counterfactual else "SUPPORTED")
    )

    return GoldSample(
        sample_id=normalize_id(raw.get("id")),
        question=str(raw.get("question", "")).strip(),
        answer=str(evaluation.get("gold_answer") or raw.get("answer", "")).strip(),
        question_type=str(raw.get("question_type", "unknown")).strip(),
        difficulty=str(
            (evaluation.get("difficulty") or {}).get("overall_level")
            or raw.get("difficulty", "unknown")
        ).strip(),
        requires_counterfactual=requires_counterfactual,
        gold_decision=gold_decision,
        gold_rule_ids=gold_rule_ids,
        gold_article_ids=gold_article_ids,
        gold_event_ids=gold_event_ids,
        gold_path=list(gold_path),
        gold_citations=gold_citations,
        raw=dict(raw),
    )


def load_benchmark(path: Path, limit: Optional[int] = None) -> Tuple[Dict[str, Any], List[GoldSample]]:
    data = load_json_or_jsonl(path)

    if isinstance(data, Mapping):
        rows = data.get("questions") or data.get("samples") or data.get("data")
        metadata = dict(data.get("metadata") or {})
    elif isinstance(data, list):
        rows = data
        metadata = {}
    else:
        raise ValueError("Benchmark phải là JSON object hoặc list.")

    if not isinstance(rows, list):
        raise ValueError("Không tìm thấy danh sách questions/samples/data trong benchmark.")

    samples = [parse_gold_sample(row) for row in rows]
    samples = [sample for sample in samples if sample.sample_id]

    if limit is not None:
        samples = samples[:limit]

    if not samples:
        raise ValueError("Benchmark không có mẫu hợp lệ.")

    duplicate_ids = [
        sample_id for sample_id, count in Counter(s.sample_id for s in samples).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Benchmark có id trùng: {duplicate_ids[:10]}")

    return metadata, samples


# ============================================================
# Prediction normalization
# ============================================================

RULE_FIELD_ALIASES = [
    "retrieved_rule_ids",
    "rule_ids",
    "retrieval.rule_ids",
    "retrieval.retrieved_rule_ids",
    "retrieved_rules",
    "retrieval.retrieved_rules",
    "candidate_rules",
    "ranked_rules",
    "evidence_rules",
    "verification.verified_rule_ids",
    "verified_rule_ids",
]

EVENT_FIELD_ALIASES = [
    "retrieved_event_ids",
    "event_ids",
    "retrieval.event_ids",
    "retrieval.retrieved_event_ids",
    "retrieved_events",
    "retrieval.retrieved_events",
    "expanded_events",
    "graph.expanded_events",
    "candidate_events",
]

PATH_FIELD_ALIASES = [
    "reasoning_path",
    "retrieved_path",
    "path",
    "paths",
    "graph_path",
    "causal_path",
    "retrieval.reasoning_path",
    "verification.reasoning_path",
    "verified_path",
]

DECISION_FIELD_ALIASES = [
    "verification_decision",
    "gold_decision",
    "decision",
    "verdict",
    "status",
    "verification.final_decision",
    "verification.decision",
    "verification.verdict",
    "counterfactual_verification_result.decision",
]

ANSWER_FIELD_ALIASES = [
    "final_answer",
    "answer",
    "generated_answer",
    "generation.answer",
    "final.answer",
    "response",
    "output_text",
]

CITATION_FIELD_ALIASES = [
    "citations",
    "expected_citations",
    "generation.citations",
    "final.citations",
    "used_citations",
    "article_ids",
    "supporting_article_ids",
]


def parse_path(value: Any) -> Tuple[List[Tuple[str, str]], List[str]]:
    edges: List[Tuple[str, str]] = []
    event_ids: List[str] = []

    if value is None:
        return edges, event_ids

    if isinstance(value, Mapping):
        # Một path object có thể chứa events/edges/nodes.
        nested = (
            value.get("edges")
            or value.get("path")
            or value.get("events")
            or value.get("nodes")
        )
        if nested is not None:
            return parse_path(nested)

        source = (
            value.get("source_event_id")
            or value.get("source")
            or value.get("from")
            or value.get("condition_event")
        )
        target = (
            value.get("target_event_id")
            or value.get("target")
            or value.get("to")
            or value.get("effect_event")
        )
        if source is not None and target is not None:
            source_id, target_id = normalize_id(source), normalize_id(target)
            return [(source_id, target_id)], [source_id, target_id]

        event = value.get("event_id") or value.get("id")
        if event is not None:
            return [], [normalize_id(event)]
        return [], []

    if isinstance(value, str):
        # Hỗ trợ "A -> B -> C"
        if "->" in value or "→" in value:
            parts = [
                normalize_id(part)
                for part in re.split(r"\s*(?:->|→)\s*", value)
                if normalize_id(part)
            ]
            return list(zip(parts, parts[1:])), parts
        return [], [normalize_id(value)]

    if isinstance(value, (list, tuple)):
        if not value:
            return [], []

        # List event ids hoặc list event objects.
        if all(
            isinstance(item, (str, int, float))
            or (isinstance(item, Mapping) and ("event_id" in item or "id" in item))
            for item in value
        ):
            ids = []
            for item in value:
                if isinstance(item, Mapping):
                    ids.append(normalize_id(item.get("event_id") or item.get("id")))
                else:
                    ids.append(normalize_id(item))
            ids = [item for item in ids if item]
            return list(zip(ids, ids[1:])), ids

        # List edge objects.
        for item in value:
            child_edges, child_events = parse_path(item)
            edges.extend(child_edges)
            event_ids.extend(child_events)

        # Dedupe event IDs theo thứ tự nhưng giữ edges.
        return edges, dedupe(event_ids)

    return [], []


def parse_prediction(raw: Mapping[str, Any], fallback_id: Optional[str] = None) -> Prediction:
    sample_id = normalize_id(
        raw.get("id")
        or raw.get("sample_id")
        or raw.get("question_id")
        or fallback_id
    )

    rule_value = find_first(raw, RULE_FIELD_ALIASES, [])
    retrieved_rule_ids = extract_ids_from_objects(
        rule_value,
        id_keys=("rule_id", "id", "memory_id"),
    )

    event_value = find_first(raw, EVENT_FIELD_ALIASES, [])
    retrieved_event_ids = extract_ids_from_objects(
        event_value,
        id_keys=("event_id", "id", "memory_id"),
    )

    path_value = find_first(raw, PATH_FIELD_ALIASES, [])
    path_edges, path_event_ids = parse_path(path_value)

    # Event trong path cũng được xem là event đã retrieve.
    retrieved_event_ids = dedupe(retrieved_event_ids + path_event_ids)

    decision = normalize_decision(
        find_first(raw, DECISION_FIELD_ALIASES, "")
    )

    final_answer = str(
        find_first(raw, ANSWER_FIELD_ALIASES, "") or ""
    ).strip()

    citation_value = find_first(raw, CITATION_FIELD_ALIASES, [])
    citations = normalize_citations(citation_value, final_answer)

    return Prediction(
        sample_id=sample_id,
        retrieved_rule_ids=retrieved_rule_ids,
        retrieved_event_ids=retrieved_event_ids,
        path_edges=path_edges,
        path_event_ids=path_event_ids,
        verification_decision=decision,
        final_answer=final_answer,
        citations=citations,
        raw=dict(raw),
    )


def load_predictions(path: Path) -> Dict[str, Prediction]:
    data = load_json_or_jsonl(path)
    rows: List[Tuple[Optional[str], Mapping[str, Any]]] = []

    if isinstance(data, list):
        rows = [(None, row) for row in data if isinstance(row, Mapping)]

    elif isinstance(data, Mapping):
        explicit = (
            data.get("predictions")
            or data.get("results")
            or data.get("questions")
            or data.get("samples")
            or data.get("data")
        )

        if isinstance(explicit, list):
            rows = [(None, row) for row in explicit if isinstance(row, Mapping)]
        elif isinstance(explicit, Mapping):
            rows = [
                (normalize_id(key), value)
                for key, value in explicit.items()
                if isinstance(value, Mapping)
            ]
        else:
            # Có thể là dictionary theo sample id.
            candidate_rows = [
                (normalize_id(key), value)
                for key, value in data.items()
                if isinstance(value, Mapping)
            ]
            if candidate_rows:
                rows = candidate_rows
            else:
                raise ValueError("Không nhận diện được cấu trúc prediction JSON.")
    else:
        raise ValueError("Prediction phải là JSON object, JSON list hoặc JSONL.")

    predictions: Dict[str, Prediction] = {}
    for fallback_id, row in rows:
        prediction = parse_prediction(row, fallback_id=fallback_id)
        if prediction.sample_id:
            predictions[prediction.sample_id] = prediction

    if not predictions:
        raise ValueError("Không đọc được prediction hợp lệ.")

    return predictions


def build_oracle_predictions(samples: Sequence[GoldSample]) -> Dict[str, Prediction]:
    predictions: Dict[str, Prediction] = {}
    for sample in samples:
        edges = [
            (
                normalize_id(edge.get("source_event_id")),
                normalize_id(edge.get("target_event_id")),
            )
            for edge in sample.gold_path
            if edge.get("source_event_id") and edge.get("target_event_id")
        ]
        predictions[sample.sample_id] = Prediction(
            sample_id=sample.sample_id,
            retrieved_rule_ids=list(sample.gold_rule_ids),
            retrieved_event_ids=list(sample.gold_event_ids),
            path_edges=edges,
            path_event_ids=list(sample.gold_event_ids),
            verification_decision=sample.gold_decision,
            final_answer=sample.answer,
            citations=list(sample.gold_citations),
            raw={"oracle": True},
        )
    return predictions


# ============================================================
# Metric evaluators
# ============================================================

def gold_path_edges(sample: GoldSample) -> List[Tuple[str, str]]:
    return [
        (
            normalize_id(edge.get("source_event_id")),
            normalize_id(edge.get("target_event_id")),
        )
        for edge in sample.gold_path
        if edge.get("source_event_id") and edge.get("target_event_id")
    ]


def ordered_path_accuracy(
    predicted: Sequence[Tuple[str, str]],
    gold: Sequence[Tuple[str, str]],
) -> float:
    return float(list(predicted) == list(gold))


def hop_accuracy(
    predicted: Sequence[Tuple[str, str]],
    gold: Sequence[Tuple[str, str]],
) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0
    correct = sum(
        index < len(predicted) and predicted[index] == gold_edge
        for index, gold_edge in enumerate(gold)
    )
    return safe_div(correct, len(gold))


def evaluate_one(
    sample: GoldSample,
    prediction: Prediction,
    k_values: Sequence[int],
    rouge_scorer: Any = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": sample.sample_id,
        "question": sample.question,
        "question_type": sample.question_type,
        "difficulty": sample.difficulty,
        "requires_counterfactual": sample.requires_counterfactual,
        "prediction_present": True,
        "gold_rule_count": len(sample.gold_rule_ids),
        "predicted_rule_count": len(prediction.retrieved_rule_ids),
        "gold_event_count": len(sample.gold_event_ids),
        "predicted_event_count": len(prediction.retrieved_event_ids),
    }

    # Rule retrieval.
    for k in k_values:
        row[f"rule_hit@{k}"] = hit_at_k(
            prediction.retrieved_rule_ids, sample.gold_rule_ids, k
        )
        row[f"rule_recall@{k}"] = recall_at_k(
            prediction.retrieved_rule_ids, sample.gold_rule_ids, k
        )
        row[f"rule_precision@{k}"] = precision_at_k(
            prediction.retrieved_rule_ids, sample.gold_rule_ids, k
        )

    row["rule_mrr"] = reciprocal_rank(
        prediction.retrieved_rule_ids, sample.gold_rule_ids
    )
    row["rule_map"] = average_precision(
        prediction.retrieved_rule_ids, sample.gold_rule_ids
    )
    rule_full = set_metrics(
        prediction.retrieved_rule_ids, sample.gold_rule_ids
    )
    row["rule_set_precision"] = rule_full["precision"]
    row["rule_set_recall"] = rule_full["recall"]
    row["rule_set_f1"] = rule_full["f1"]
    row["rule_set_exact"] = rule_full["exact"]

    # Event retrieval.
    for k in k_values:
        row[f"event_hit@{k}"] = hit_at_k(
            prediction.retrieved_event_ids, sample.gold_event_ids, k
        )
        row[f"event_recall@{k}"] = recall_at_k(
            prediction.retrieved_event_ids, sample.gold_event_ids, k
        )
        row[f"event_precision@{k}"] = precision_at_k(
            prediction.retrieved_event_ids, sample.gold_event_ids, k
        )

    row["event_mrr"] = reciprocal_rank(
        prediction.retrieved_event_ids, sample.gold_event_ids
    )
    event_full = set_metrics(
        prediction.retrieved_event_ids, sample.gold_event_ids
    )
    row["event_set_precision"] = event_full["precision"]
    row["event_set_recall"] = event_full["recall"]
    row["event_set_f1"] = event_full["f1"]
    row["event_set_exact"] = event_full["exact"]

    # Path.
    gold_edges = gold_path_edges(sample)
    pred_edges = prediction.path_edges
    path_applicable = bool(gold_edges)
    row["path_applicable"] = path_applicable

    if path_applicable:
        edge_metrics = set_metrics(
            [f"{u}→{v}" for u, v in pred_edges],
            [f"{u}→{v}" for u, v in gold_edges],
        )
        row["exact_path_accuracy"] = ordered_path_accuracy(pred_edges, gold_edges)
        row["hop_accuracy"] = hop_accuracy(pred_edges, gold_edges)
        row["path_edge_precision"] = edge_metrics["precision"]
        row["path_edge_recall"] = edge_metrics["recall"]
        row["path_edge_f1"] = edge_metrics["f1"]
        row["path_length_accuracy"] = float(len(pred_edges) == len(gold_edges))

        path_events = set_metrics(
            prediction.path_event_ids,
            sample.gold_event_ids,
        )
        row["path_event_precision"] = path_events["precision"]
        row["path_event_recall"] = path_events["recall"]
        row["path_event_f1"] = path_events["f1"]
    else:
        for key in (
            "exact_path_accuracy",
            "hop_accuracy",
            "path_edge_precision",
            "path_edge_recall",
            "path_edge_f1",
            "path_length_accuracy",
            "path_event_precision",
            "path_event_recall",
            "path_event_f1",
        ):
            row[key] = None

    # Verification / counterfactual.
    pred_decision = normalize_decision(prediction.verification_decision)
    gold_decision = normalize_decision(sample.gold_decision)
    row["gold_decision"] = gold_decision
    row["predicted_decision"] = pred_decision
    row["verification_correct"] = float(
        bool(pred_decision) and pred_decision == gold_decision
    )

    # Answer generation.
    row["answer_exact_match"] = exact_match(
        prediction.final_answer, sample.answer
    )
    row["answer_token_f1"] = token_f1(
        prediction.final_answer, sample.answer
    )

    if rouge_scorer is not None:
        try:
            row["answer_rouge_l"] = rouge_scorer.score(
                sample.answer, prediction.final_answer
            )["rougeL"].fmeasure
        except Exception:
            row["answer_rouge_l"] = rouge_l_f1_fallback(
                prediction.final_answer, sample.answer
            )
    else:
        row["answer_rouge_l"] = rouge_l_f1_fallback(
            prediction.final_answer, sample.answer
        )

    row["answer_length_tokens"] = len(tokenize(prediction.final_answer))
    row["gold_answer_length_tokens"] = len(tokenize(sample.answer))
    row["answer_present"] = float(bool(prediction.final_answer.strip()))

    # Citation.
    citation_metrics = set_metrics(
        prediction.citations, sample.gold_citations
    )
    row["citation_precision"] = citation_metrics["precision"]
    row["citation_recall"] = citation_metrics["recall"]
    row["citation_f1"] = citation_metrics["f1"]
    row["citation_exact"] = citation_metrics["exact"]
    row["evidence_coverage"] = citation_metrics["recall"]

    # Error flags.
    errors: List[str] = []
    primary_k = 5 if 5 in k_values else max(k_values)
    if row[f"rule_recall@{primary_k}"] < 1.0:
        errors.append("RULE_RETRIEVAL_MISS")
    if row[f"event_recall@{primary_k}"] < 1.0:
        errors.append("EVENT_RETRIEVAL_MISS")
    if path_applicable and row["exact_path_accuracy"] < 1.0:
        errors.append("PATH_MISMATCH")
    if row["verification_correct"] < 1.0:
        errors.append("VERIFICATION_ERROR")
    if row["answer_token_f1"] < 0.5:
        errors.append("LOW_ANSWER_F1")
    if row["citation_recall"] < 1.0:
        errors.append("CITATION_MISS")

    row["error_types"] = errors
    row["predicted_rule_ids"] = prediction.retrieved_rule_ids
    row["gold_rule_ids"] = sample.gold_rule_ids
    row["predicted_event_ids"] = prediction.retrieved_event_ids
    row["gold_event_ids"] = sample.gold_event_ids
    row["predicted_path"] = [
        {"source_event_id": u, "target_event_id": v}
        for u, v in prediction.path_edges
    ]
    row["gold_path"] = sample.gold_path
    row["predicted_answer"] = prediction.final_answer
    row["gold_answer"] = sample.answer
    row["predicted_citations"] = prediction.citations
    row["gold_citations"] = sample.gold_citations

    return row


def missing_prediction_row(
    sample: GoldSample,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    empty = Prediction(
        sample_id=sample.sample_id,
        retrieved_rule_ids=[],
        retrieved_event_ids=[],
        path_edges=[],
        path_event_ids=[],
        verification_decision="",
        final_answer="",
        citations=[],
        raw={},
    )
    row = evaluate_one(sample, empty, k_values)
    row["prediction_present"] = False
    row["error_types"] = ["MISSING_PREDICTION"]
    return row


# ============================================================
# BERTScore
# ============================================================

def apply_bertscore(
    rows: List[Dict[str, Any]],
    enabled: bool,
    model_type: Optional[str],
    batch_size: int,
) -> Dict[str, Any]:
    info = {
        "enabled": enabled,
        "available": False,
        "model_type": model_type,
        "error": None,
    }

    for row in rows:
        row["answer_bertscore_precision"] = None
        row["answer_bertscore_recall"] = None
        row["answer_bertscore_f1"] = None

    if not enabled:
        return info

    try:
        from bert_score import score as bert_score  # type: ignore
    except ImportError:
        info["error"] = (
            "Chưa cài bert-score. Chạy: pip install bert-score"
        )
        return info

    candidates = [str(row["predicted_answer"]) for row in rows]
    references = [str(row["gold_answer"]) for row in rows]

    try:
        kwargs: Dict[str, Any] = {
            "cands": candidates,
            "refs": references,
            "batch_size": batch_size,
            "verbose": True,
        }
        if model_type:
            kwargs["model_type"] = model_type
        else:
            kwargs["lang"] = "vi"

        precision, recall, f1 = bert_score(**kwargs)

        for index, row in enumerate(rows):
            row["answer_bertscore_precision"] = float(precision[index])
            row["answer_bertscore_recall"] = float(recall[index])
            row["answer_bertscore_f1"] = float(f1[index])

        info["available"] = True
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    return info


# ============================================================
# Aggregation
# ============================================================

CORE_NUMERIC_FIELDS = [
    "rule_mrr",
    "rule_map",
    "rule_set_precision",
    "rule_set_recall",
    "rule_set_f1",
    "rule_set_exact",
    "event_mrr",
    "event_set_precision",
    "event_set_recall",
    "event_set_f1",
    "event_set_exact",
    "exact_path_accuracy",
    "hop_accuracy",
    "path_edge_precision",
    "path_edge_recall",
    "path_edge_f1",
    "path_length_accuracy",
    "path_event_precision",
    "path_event_recall",
    "path_event_f1",
    "verification_correct",
    "answer_exact_match",
    "answer_token_f1",
    "answer_rouge_l",
    "answer_bertscore_precision",
    "answer_bertscore_recall",
    "answer_bertscore_f1",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "citation_exact",
    "evidence_coverage",
]


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "count": len(rows),
        "prediction_coverage": mean(
            float(bool(row.get("prediction_present"))) for row in rows
        ),
    }

    for k in k_values:
        for prefix in ("rule", "event"):
            for metric in ("hit", "recall", "precision"):
                key = f"{prefix}_{metric}@{k}"
                summary[key] = mean(row.get(key) for row in rows)

    for field_name in CORE_NUMERIC_FIELDS:
        summary[field_name] = mean(row.get(field_name) for row in rows)

    summary["path_applicable_count"] = sum(
        bool(row.get("path_applicable")) for row in rows
    )
    summary["counterfactual_count"] = sum(
        bool(row.get("requires_counterfactual")) for row in rows
    )
    summary["error_rate"] = mean(
        float(bool(row.get("error_types"))) for row in rows
    )
    return summary


def confusion_matrix(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = ["SUPPORTED", "REJECT_DIRECT_CLAIM", "UNCERTAIN", "MISSING"]
    matrix = {
        gold: {pred: 0 for pred in labels}
        for gold in labels
    }

    for row in rows:
        gold = normalize_decision(row.get("gold_decision")) or "MISSING"
        pred = normalize_decision(row.get("predicted_decision")) or "MISSING"
        if gold not in matrix:
            matrix[gold] = {label: 0 for label in labels}
        if pred not in matrix[gold]:
            for gold_label in matrix:
                matrix[gold_label].setdefault(pred, 0)
        matrix[gold][pred] += 1

    return matrix


def binary_counterfactual_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Positive class = REJECT_DIRECT_CLAIM.
    Báo cáo cả trên toàn bộ benchmark và riêng subset counterfactual.
    """
    def calculate(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        tp = fp = tn = fn = 0
        for row in subset:
            gold_positive = normalize_decision(
                row.get("gold_decision")
            ) == "REJECT_DIRECT_CLAIM"
            pred_positive = normalize_decision(
                row.get("predicted_decision")
            ) == "REJECT_DIRECT_CLAIM"

            if gold_positive and pred_positive:
                tp += 1
            elif not gold_positive and pred_positive:
                fp += 1
            elif not gold_positive and not pred_positive:
                tn += 1
            else:
                fn += 1

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        return {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": harmonic_f1(precision, recall),
            "accuracy": accuracy,
            "count": len(subset),
        }

    cf_subset = [
        row for row in rows
        if bool(row.get("requires_counterfactual"))
    ]
    return {
        "all_questions": calculate(rows),
        "counterfactual_subset": calculate(cf_subset),
    }


def group_report(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field_name, "unknown"))].append(row)

    return {
        key: aggregate_rows(group_rows, k_values)
        for key, group_rows in sorted(groups.items())
    }


def error_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    question_ids: Dict[str, List[str]] = defaultdict(list)

    for row in rows:
        for error_type in row.get("error_types", []):
            counts[error_type] += 1
            question_ids[error_type].append(str(row.get("id")))

    return {
        "counts": dict(counts.most_common()),
        "question_ids": dict(question_ids),
    }


# ============================================================
# Output
# ============================================================

def csv_safe(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred_fields = [
        "id",
        "question_type",
        "difficulty",
        "requires_counterfactual",
        "prediction_present",
        "rule_recall@5",
        "rule_mrr",
        "event_recall@5",
        "event_mrr",
        "exact_path_accuracy",
        "hop_accuracy",
        "path_edge_f1",
        "verification_correct",
        "answer_exact_match",
        "answer_token_f1",
        "answer_rouge_l",
        "answer_bertscore_f1",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "evidence_coverage",
        "error_types",
        "question",
        "predicted_answer",
        "gold_answer",
        "predicted_rule_ids",
        "gold_rule_ids",
        "predicted_event_ids",
        "gold_event_ids",
        "predicted_path",
        "gold_path",
        "predicted_citations",
        "gold_citations",
    ]

    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())

    fieldnames = [
        field for field in preferred_fields if field in all_fields
    ] + sorted(all_fields - set(preferred_fields))

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: csv_safe(row.get(key))
                for key in fieldnames
            })


def percentage(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def metric_value(summary: Mapping[str, Any], key: str) -> str:
    return percentage(summary.get(key))


def build_markdown_summary(report: Mapping[str, Any]) -> str:
    overall = report["overall"]
    k_values = report["configuration"]["k_values"]
    primary_k = 5 if 5 in k_values else max(k_values)
    cf = report["counterfactual_classification"]["all_questions"]
    bert_info = report["bertscore"]

    lines = [
        "# Báo cáo đánh giá CausalRAG trên BLHS Multi-hop Benchmark",
        "",
        f"- Thời điểm chạy: `{report['created_at_utc']}`",
        f"- Tổng số câu benchmark: **{report['dataset']['evaluated_questions']}**",
        f"- Prediction coverage: **{percentage(overall['prediction_coverage'])}**",
        "",
        "## 1. Retrieval",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Rule Hit@{primary_k} | {metric_value(overall, f'rule_hit@{primary_k}')} |",
        f"| Rule Recall@{primary_k} | {metric_value(overall, f'rule_recall@{primary_k}')} |",
        f"| Rule Precision@{primary_k} | {metric_value(overall, f'rule_precision@{primary_k}')} |",
        f"| Rule MRR | {metric_value(overall, 'rule_mrr')} |",
        f"| Rule MAP | {metric_value(overall, 'rule_map')} |",
        f"| Event Hit@{primary_k} | {metric_value(overall, f'event_hit@{primary_k}')} |",
        f"| Event Recall@{primary_k} | {metric_value(overall, f'event_recall@{primary_k}')} |",
        f"| Event Precision@{primary_k} | {metric_value(overall, f'event_precision@{primary_k}')} |",
        f"| Event MRR | {metric_value(overall, 'event_mrr')} |",
        "",
        "## 2. Causal path",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Exact path accuracy | {metric_value(overall, 'exact_path_accuracy')} |",
        f"| Hop accuracy | {metric_value(overall, 'hop_accuracy')} |",
        f"| Path edge F1 | {metric_value(overall, 'path_edge_f1')} |",
        f"| Path event F1 | {metric_value(overall, 'path_event_f1')} |",
        "",
        "## 3. Counterfactual verification",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Verification accuracy | {metric_value(overall, 'verification_correct')} |",
        f"| Counterfactual precision | {percentage(cf['precision'])} |",
        f"| Counterfactual recall | {percentage(cf['recall'])} |",
        f"| Counterfactual F1 | {percentage(cf['f1'])} |",
        "",
        "## 4. Answer generation",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Exact Match | {metric_value(overall, 'answer_exact_match')} |",
        f"| Token F1 | {metric_value(overall, 'answer_token_f1')} |",
        f"| ROUGE-L | {metric_value(overall, 'answer_rouge_l')} |",
        f"| BERTScore F1 | {metric_value(overall, 'answer_bertscore_f1')} |",
        "",
        "## 5. Citation và evidence",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---:|",
        f"| Citation precision | {metric_value(overall, 'citation_precision')} |",
        f"| Citation recall | {metric_value(overall, 'citation_recall')} |",
        f"| Citation F1 | {metric_value(overall, 'citation_f1')} |",
        f"| Evidence coverage | {metric_value(overall, 'evidence_coverage')} |",
        "",
        "## 6. Ghi chú",
        "",
        f"- Số câu có gold path tuyến tính: **{overall['path_applicable_count']}**.",
        f"- Số câu counterfactual: **{overall['counterfactual_count']}**.",
    ]

    if bert_info.get("enabled") and not bert_info.get("available"):
        lines.append(
            f"- BERTScore chưa được tính: `{bert_info.get('error')}`."
        )
    elif not bert_info.get("enabled"):
        lines.append(
            "- BERTScore đang tắt. Thêm `--bertscore` để bật."
        )

    lines.extend([
        "",
        "## 7. Kết quả theo loại câu hỏi",
        "",
        "| Loại câu hỏi | Số câu | Rule Recall@5 | Path Accuracy | Token F1 | Citation F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for question_type, metrics in report["by_question_type"].items():
        lines.append(
            f"| {question_type} | {metrics['count']} | "
            f"{percentage(metrics.get('rule_recall@5'))} | "
            f"{percentage(metrics.get('exact_path_accuracy'))} | "
            f"{percentage(metrics.get('answer_token_f1'))} | "
            f"{percentage(metrics.get('citation_f1'))} |"
        )

    return "\n".join(lines) + "\n"


def normalized_prediction_payload(
    prediction: Prediction,
) -> Dict[str, Any]:
    return {
        "id": prediction.sample_id,
        "retrieved_rule_ids": prediction.retrieved_rule_ids,
        "retrieved_event_ids": prediction.retrieved_event_ids,
        "reasoning_path": [
            {
                "source_event_id": source,
                "target_event_id": target,
            }
            for source, target in prediction.path_edges
        ],
        "path_event_ids": prediction.path_event_ids,
        "verification_decision": prediction.verification_decision,
        "final_answer": prediction.final_answer,
        "citations": prediction.citations,
    }


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá end-to-end CausalRAG trên benchmark BLHS v2.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/blhs_multihop_benchmark_250_updated.json"),
        help="Đường dẫn benchmark JSON.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Prediction JSON/JSONL đã sinh từ pipeline.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Dùng gold làm prediction để kiểm thử evaluator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Thư mục lưu báo cáo.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Các giá trị k cho retrieval metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ đánh giá N câu đầu.",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Bật BERTScore (cần cài bert-score).",
    )
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default=None,
        help="Model Hugging Face cho BERTScore; bỏ trống để dùng lang=vi.",
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=8,
        help="Batch size BERTScore.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Dừng nếu thiếu prediction cho bất kỳ câu nào.",
    )
    return parser.parse_args()


def load_rouge_scorer() -> Tuple[Any, str]:
    try:
        from rouge_score import rouge_scorer  # type: ignore
        return (
            rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False),
            "rouge_score",
        )
    except ImportError:
        return None, "internal_lcs_fallback"


def main() -> int:
    args = parse_args()

    if args.oracle and args.predictions is not None:
        raise ValueError("Chỉ dùng một trong hai: --oracle hoặc --predictions.")
    if not args.oracle and args.predictions is None:
        raise ValueError(
            "Cần truyền --predictions hoặc dùng --oracle để kiểm thử."
        )

    k_values = sorted(set(k for k in args.k_values if k > 0))
    if not k_values:
        raise ValueError("--k-values phải có ít nhất một số nguyên dương.")

    benchmark_metadata, samples = load_benchmark(
        args.benchmark,
        limit=args.limit,
    )

    if args.oracle:
        predictions = build_oracle_predictions(samples)
        prediction_source = "oracle"
    else:
        assert args.predictions is not None
        predictions = load_predictions(args.predictions)
        prediction_source = str(args.predictions)

    sample_ids = {sample.sample_id for sample in samples}
    extra_prediction_ids = sorted(set(predictions) - sample_ids)
    missing_prediction_ids = sorted(sample_ids - set(predictions))

    if args.strict and missing_prediction_ids:
        raise ValueError(
            f"Thiếu {len(missing_prediction_ids)} prediction. "
            f"Ví dụ: {missing_prediction_ids[:10]}"
        )

    rouge_scorer, rouge_backend = load_rouge_scorer()

    rows: List[Dict[str, Any]] = []
    for sample in samples:
        prediction = predictions.get(sample.sample_id)
        if prediction is None:
            row = missing_prediction_row(sample, k_values)
        else:
            row = evaluate_one(
                sample,
                prediction,
                k_values,
                rouge_scorer=rouge_scorer,
            )
        rows.append(row)

    bertscore_info = apply_bertscore(
        rows,
        enabled=args.bertscore,
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch_size,
    )

    overall = aggregate_rows(rows, k_values)
    report = {
        "metadata": {
            "name": "BLHS CausalRAG End-to-End Evaluation",
            "version": "1.0",
        },
        "created_at_utc": utc_now_iso(),
        "configuration": {
            "benchmark": str(args.benchmark),
            "prediction_source": prediction_source,
            "output_dir": str(args.output_dir),
            "k_values": k_values,
            "limit": args.limit,
            "strict": args.strict,
            "rouge_backend": rouge_backend,
        },
        "dataset": {
            "benchmark_metadata": benchmark_metadata,
            "evaluated_questions": len(samples),
            "available_predictions": len(predictions),
            "matched_predictions": len(samples) - len(missing_prediction_ids),
            "missing_prediction_count": len(missing_prediction_ids),
            "missing_prediction_ids": missing_prediction_ids,
            "extra_prediction_count": len(extra_prediction_ids),
            "extra_prediction_ids": extra_prediction_ids,
        },
        "overall": overall,
        "counterfactual_classification": binary_counterfactual_metrics(rows),
        "verification_confusion_matrix": confusion_matrix(rows),
        "by_question_type": group_report(
            rows, "question_type", k_values
        ),
        "by_difficulty": group_report(
            rows, "difficulty", k_values
        ),
        "error_analysis": error_analysis(rows),
        "bertscore": bertscore_info,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "evaluation_report.json"
    per_question_path = args.output_dir / "evaluation_per_question.csv"
    summary_path = args.output_dir / "evaluation_summary.md"
    errors_path = args.output_dir / "evaluation_errors.json"
    normalized_predictions_path = (
        args.output_dir / "normalized_predictions.json"
    )

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(per_question_path, rows)
    summary_path.write_text(
        build_markdown_summary(report),
        encoding="utf-8",
    )

    error_rows = [
        {
            "id": row["id"],
            "question": row["question"],
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "error_types": row["error_types"],
            "predicted_rule_ids": row["predicted_rule_ids"],
            "gold_rule_ids": row["gold_rule_ids"],
            "predicted_event_ids": row["predicted_event_ids"],
            "gold_event_ids": row["gold_event_ids"],
            "predicted_path": row["predicted_path"],
            "gold_path": row["gold_path"],
            "predicted_answer": row["predicted_answer"],
            "gold_answer": row["gold_answer"],
            "predicted_citations": row["predicted_citations"],
            "gold_citations": row["gold_citations"],
        }
        for row in rows
        if row.get("error_types")
    ]
    errors_path.write_text(
        json.dumps(
            {
                "count": len(error_rows),
                "errors": error_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    normalized_predictions_path.write_text(
        json.dumps(
            {
                "predictions": [
                    normalized_prediction_payload(predictions[sample.sample_id])
                    for sample in samples
                    if sample.sample_id in predictions
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    primary_k = 5 if 5 in k_values else max(k_values)
    print("=" * 72)
    print("BLHS CAUSALRAG EVALUATION")
    print("=" * 72)
    print(f"Questions                 : {len(samples)}")
    print(
        f"Prediction coverage        : "
        f"{percentage(overall['prediction_coverage'])}"
    )
    print(
        f"Rule Recall@{primary_k:<2}             : "
        f"{percentage(overall.get(f'rule_recall@{primary_k}'))}"
    )
    print(
        f"Event Recall@{primary_k:<2}            : "
        f"{percentage(overall.get(f'event_recall@{primary_k}'))}"
    )
    print(
        f"Exact Path Accuracy        : "
        f"{percentage(overall.get('exact_path_accuracy'))}"
    )
    print(
        f"Verification Accuracy      : "
        f"{percentage(overall.get('verification_correct'))}"
    )
    print(
        f"Answer Token F1            : "
        f"{percentage(overall.get('answer_token_f1'))}"
    )
    print(
        f"Answer ROUGE-L             : "
        f"{percentage(overall.get('answer_rouge_l'))}"
    )
    print(
        f"Citation F1                : "
        f"{percentage(overall.get('citation_f1'))}"
    )
    print("-" * 72)
    print(f"Report JSON                : {report_path}")
    print(f"Per-question CSV           : {per_question_path}")
    print(f"Summary Markdown           : {summary_path}")
    print(f"Error analysis             : {errors_path}")
    print(f"Normalized predictions     : {normalized_predictions_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nLỖI: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
