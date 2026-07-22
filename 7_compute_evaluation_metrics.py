from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_GOLD_PATH = "data/evaluation_dataset.json"
DEFAULT_PREDICTION_PATH = "data/evaluation_predictions.json"

DEFAULT_METRICS_PATH = "data/evaluation_metrics.json"
DEFAULT_BY_SAMPLE_PATH = "data/evaluation_metrics_by_sample.csv"
DEFAULT_ERRORS_PATH = "data/evaluation_errors.csv"


# ============================================================
# BASIC I/O
# ============================================================

def load_json(path_value: str) -> Any:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path_value: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_csv(
    rows: list[dict[str, Any]],
    path_value: str,
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
# NORMALIZATION
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_id(value: Any) -> str:
    text = safe_string(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def remove_vietnamese_accents(text: str) -> str:
    text = safe_string(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


def normalize_event(value: Any) -> str:
    text = remove_vietnamese_accents(safe_string(value)).upper()
    text = re.sub(r"^EVENT::", "", text)
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_label(value: Any) -> str:
    text = remove_vietnamese_accents(safe_string(value)).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_text(value: Any) -> str:
    text = remove_vietnamese_accents(safe_string(value)).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def unique_preserve_order(values: Iterable[Any], normalizer=normalize_id) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = normalizer(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = safe_string(value).lower()
    if text in {"true", "1", "yes", "y", "co", "có"}:
        return True
    if text in {"false", "0", "no", "n", "khong", "không"}:
        return False
    return None


def round_metric(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def extract_gold_samples(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(payload, list):
        samples = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        samples = payload.get("samples")
        metadata = payload.get("metadata", {})
    else:
        raise ValueError("Gold dataset phải là list hoặc JSON object.")

    if not isinstance(samples, list):
        raise ValueError("Gold dataset thiếu danh sách `samples`.")

    return metadata, samples


def extract_predictions(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(payload, list):
        predictions = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        predictions = payload.get("predictions")
        metadata = payload.get("metadata", {})
    else:
        raise ValueError("Prediction file phải là list hoặc JSON object.")

    if not isinstance(predictions, list):
        raise ValueError("Prediction file thiếu danh sách `predictions`.")

    return metadata, predictions


def index_by_question_id(
    rows: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for position, row in enumerate(rows, start=1):
        question_id = safe_string(row.get("question_id"))
        if not question_id:
            raise ValueError(
                f"{source_name}: dòng {position} thiếu question_id."
            )
        if question_id in indexed:
            raise ValueError(
                f"{source_name}: question_id bị trùng: {question_id}"
            )
        indexed[question_id] = row

    return indexed


# ============================================================
# GENERIC METRICS
# ============================================================

def precision_recall_f1(
    gold_values: Iterable[str],
    predicted_values: Iterable[str],
) -> tuple[float, float, float, int]:
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1, true_positive


def exact_set_match(
    gold_values: Iterable[str],
    predicted_values: Iterable[str],
) -> float:
    return float(set(gold_values) == set(predicted_values))


def exact_sequence_match(
    gold_values: Iterable[str],
    predicted_values: Iterable[str],
) -> float:
    return float(list(gold_values) == list(predicted_values))


def recall_at_k(
    gold_values: Iterable[str],
    ranked_predictions: list[str],
    k: int,
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0
    predicted_at_k = set(ranked_predictions[:k])
    return len(gold & predicted_at_k) / len(gold)


def hit_at_k(
    gold_values: Iterable[str],
    ranked_predictions: list[str],
    k: int,
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0
    return float(bool(gold & set(ranked_predictions[:k])))


def reciprocal_rank(
    gold_values: Iterable[str],
    ranked_predictions: list[str],
) -> float:
    gold = set(gold_values)
    for rank, value in enumerate(ranked_predictions, start=1):
        if value in gold:
            return 1.0 / rank
    return 0.0


def average_precision(
    gold_values: Iterable[str],
    ranked_predictions: list[str],
) -> float:
    gold = set(gold_values)
    if not gold:
        return 1.0

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()

    for rank, value in enumerate(ranked_predictions, start=1):
        if value in seen:
            continue
        seen.add(value)

        if value in gold:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(gold)


def accuracy(gold: Any, predicted: Any) -> Optional[float]:
    if gold is None:
        return None
    return float(gold == predicted)


# ============================================================
# ANSWER METRICS
# ============================================================

def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


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

    gold_counts: dict[str, int] = defaultdict(int)
    predicted_counts: dict[str, int] = defaultdict(int)

    for token in gold_tokens:
        gold_counts[token] += 1
    for token in predicted_tokens:
        predicted_counts[token] += 1

    overlap = sum(
        min(gold_counts[token], predicted_counts[token])
        for token in gold_counts
    )

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def lcs_length(left: list[str], right: list[str]) -> int:
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
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(gold_tokens)

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def normalized_exact_match(gold_text: Any, predicted_text: Any) -> float:
    return float(normalize_text(gold_text) == normalize_text(predicted_text))


# ============================================================
# PATH EXTRACTION AND METRICS
# ============================================================

def normalize_edges(edges: Any) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if not isinstance(edges, list):
        return result

    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        source = normalize_event(edge[0])
        target = normalize_event(edge[1])
        if source and target:
            result.append((source, target))

    return result


def edges_from_nodes(nodes: list[str]) -> list[tuple[str, str]]:
    return [
        (nodes[index], nodes[index + 1])
        for index in range(len(nodes) - 1)
    ]


def extract_gold_path(sample: dict[str, Any]) -> dict[str, Any]:
    gold_path = sample.get("gold_path") or {}

    nodes = unique_preserve_order(
        gold_path.get("nodes", []),
        normalizer=normalize_event,
    )
    edges = normalize_edges(gold_path.get("edges", []))

    if not edges and len(nodes) >= 2:
        edges = edges_from_nodes(nodes)

    return {
        "nodes": nodes,
        "edges": edges,
        "rule_ids": unique_preserve_order(
            sample.get("gold_rule_ids", []),
            normalizer=normalize_id,
        ),
    }


def extract_predicted_paths(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    raw_paths = prediction.get("predicted_paths")

    if not isinstance(raw_paths, list) or not raw_paths:
        single_path = prediction.get("predicted_path")
        raw_paths = [single_path] if isinstance(single_path, dict) else []

    paths: list[dict[str, Any]] = []

    for rank, path in enumerate(raw_paths, start=1):
        if not isinstance(path, dict):
            continue

        nodes = unique_preserve_order(
            path.get("nodes", path.get("event_chain", [])),
            normalizer=normalize_event,
        )
        edges = normalize_edges(path.get("edges", []))

        if not edges and len(nodes) >= 2:
            edges = edges_from_nodes(nodes)

        rule_ids = unique_preserve_order(
            path.get("rule_ids", []),
            normalizer=normalize_id,
        )

        paths.append({
            "rank": int(path.get("rank", rank) or rank),
            "nodes": nodes,
            "edges": edges,
            "rule_ids": rule_ids,
            "score": path.get("score", 0.0),
        })

    return sorted(paths, key=lambda item: item["rank"])


def path_metrics(
    gold_path: dict[str, Any],
    predicted_path: dict[str, Any],
) -> dict[str, float]:
    node_precision, node_recall, node_f1, _ = precision_recall_f1(
        gold_path["nodes"],
        predicted_path.get("nodes", []),
    )
    edge_precision, edge_recall, edge_f1, _ = precision_recall_f1(
        [f"{source}>>{target}" for source, target in gold_path["edges"]],
        [
            f"{source}>>{target}"
            for source, target in predicted_path.get("edges", [])
        ],
    )
    path_rule_precision, path_rule_recall, path_rule_f1, _ = (
        precision_recall_f1(
            gold_path["rule_ids"],
            predicted_path.get("rule_ids", []),
        )
    )

    final_event_correct = 0.0
    if gold_path["nodes"]:
        predicted_nodes = predicted_path.get("nodes", [])
        final_event_correct = float(
            bool(predicted_nodes)
            and predicted_nodes[-1] == gold_path["nodes"][-1]
        )

    exact_node_sequence = exact_sequence_match(
        gold_path["nodes"],
        predicted_path.get("nodes", []),
    )
    exact_edge_sequence = exact_sequence_match(
        gold_path["edges"],
        predicted_path.get("edges", []),
    )
    exact_rule_sequence = exact_sequence_match(
        gold_path["rule_ids"],
        predicted_path.get("rule_ids", []),
    )

    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "path_rule_precision": path_rule_precision,
        "path_rule_recall": path_rule_recall,
        "path_rule_f1": path_rule_f1,
        "final_event_accuracy": final_event_correct,
        "exact_node_path_match": exact_node_sequence,
        "exact_edge_path_match": exact_edge_sequence,
        "exact_rule_path_match": exact_rule_sequence,
        "exact_path_match": float(
            exact_node_sequence == 1.0
            and exact_edge_sequence == 1.0
        ),
    }


def select_best_predicted_path(
    gold_path: dict[str, Any],
    predicted_paths: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float], int]:
    empty_path = {
        "rank": 0,
        "nodes": [],
        "edges": [],
        "rule_ids": [],
        "score": 0.0,
    }

    if not predicted_paths:
        metrics = path_metrics(gold_path, empty_path)
        return empty_path, metrics, 0

    scored: list[tuple[tuple[float, ...], dict[str, Any], dict[str, float]]] = []

    for path in predicted_paths:
        metrics = path_metrics(gold_path, path)
        sort_key = (
            metrics["exact_path_match"],
            metrics["edge_f1"],
            metrics["node_f1"],
            metrics["path_rule_f1"],
            metrics["final_event_accuracy"],
            -float(path.get("rank", 999999)),
        )
        scored.append((sort_key, path, metrics))

    _, best_path, best_metrics = max(scored, key=lambda item: item[0])
    return best_path, best_metrics, int(best_path.get("rank", 0))


# ============================================================
# PER-SAMPLE EVALUATION
# ============================================================

def evaluate_sample(
    gold: dict[str, Any],
    prediction: Optional[dict[str, Any]],
) -> dict[str, Any]:
    question_id = safe_string(gold.get("question_id"))
    task_type = normalize_label(gold.get("task_type"))
    difficulty = normalize_label(gold.get("difficulty"))

    if prediction is None:
        prediction = {
            "question_id": question_id,
            "error": "MISSING_PREDICTION",
        }

    pipeline_error = safe_string(prediction.get("error"))
    successful = not bool(pipeline_error)

    gold_rule_ids = unique_preserve_order(
        gold.get("gold_rule_ids", []),
        normalizer=normalize_id,
    )
    predicted_rule_ids = unique_preserve_order(
        prediction.get("predicted_rule_ids", []),
        normalizer=normalize_id,
    )

    gold_article_ids = unique_preserve_order(
        gold.get("gold_article_ids", []),
        normalizer=normalize_id,
    )
    predicted_article_ids = unique_preserve_order(
        prediction.get("predicted_article_ids", []),
        normalizer=normalize_id,
    )

    rule_precision, rule_recall, rule_f1, rule_tp = precision_recall_f1(
        gold_rule_ids,
        predicted_rule_ids,
    )
    article_precision, article_recall, article_f1, article_tp = (
        precision_recall_f1(
            gold_article_ids,
            predicted_article_ids,
        )
    )

    gold_path = extract_gold_path(gold)
    predicted_paths = extract_predicted_paths(prediction)
    top1_path = (
        predicted_paths[0]
        if predicted_paths
        else {
            "rank": 0,
            "nodes": [],
            "edges": [],
            "rule_ids": [],
            "score": 0.0,
        }
    )
    top1_path_metrics = path_metrics(gold_path, top1_path)

    best_path, oracle_path_metrics, oracle_path_rank = (
        select_best_predicted_path(gold_path, predicted_paths)
    )

    gold_cf_status = normalize_label(
        gold.get("gold_counterfactual_status")
    ) or None
    predicted_cf_status = normalize_label(
        prediction.get("predicted_counterfactual_status")
    ) or None

    gold_reachable = safe_bool(
        gold.get("gold_final_effect_reachable")
    )
    predicted_reachable = safe_bool(
        prediction.get("predicted_final_effect_reachable")
    )

    gold_answer_label = normalize_label(
        gold.get("gold_answer_label")
    ) or None
    predicted_answer_label = normalize_label(
        prediction.get("predicted_answer_label")
    ) or None

    cf_status_accuracy = accuracy(
        gold_cf_status,
        predicted_cf_status,
    )
    reachability_accuracy = accuracy(
        gold_reachable,
        predicted_reachable,
    )
    answer_label_accuracy = accuracy(
        gold_answer_label,
        predicted_answer_label,
    )

    gold_answer = safe_string(gold.get("gold_answer"))
    predicted_answer = safe_string(prediction.get("predicted_answer"))

    answer_precision, answer_recall, answer_token_f1 = (
        token_overlap_metrics(gold_answer, predicted_answer)
    )
    answer_rouge_l = rouge_l_f1(gold_answer, predicted_answer)
    answer_exact_match = normalized_exact_match(
        gold_answer,
        predicted_answer,
    )

    row: dict[str, Any] = {
        "question_id": question_id,
        "source_chain_id": gold.get("source_chain_id"),
        "task_type": task_type,
        "difficulty": difficulty,
        "successful": int(successful),
        "pipeline_error": pipeline_error,

        "gold_rule_count": len(gold_rule_ids),
        "predicted_rule_count": len(predicted_rule_ids),
        "rule_true_positive": rule_tp,
        "rule_precision": rule_precision,
        "rule_recall": rule_recall,
        "rule_f1": rule_f1,
        "rule_exact_set_match": exact_set_match(
            gold_rule_ids,
            predicted_rule_ids,
        ),
        "rule_recall_at_1": recall_at_k(
            gold_rule_ids, predicted_rule_ids, 1
        ),
        "rule_recall_at_3": recall_at_k(
            gold_rule_ids, predicted_rule_ids, 3
        ),
        "rule_recall_at_5": recall_at_k(
            gold_rule_ids, predicted_rule_ids, 5
        ),
        "rule_recall_at_10": recall_at_k(
            gold_rule_ids, predicted_rule_ids, 10
        ),
        "rule_hit_at_1": hit_at_k(
            gold_rule_ids, predicted_rule_ids, 1
        ),
        "rule_hit_at_3": hit_at_k(
            gold_rule_ids, predicted_rule_ids, 3
        ),
        "rule_hit_at_5": hit_at_k(
            gold_rule_ids, predicted_rule_ids, 5
        ),
        "rule_hit_at_10": hit_at_k(
            gold_rule_ids, predicted_rule_ids, 10
        ),
        "rule_mrr": reciprocal_rank(
            gold_rule_ids,
            predicted_rule_ids,
        ),
        "rule_average_precision": average_precision(
            gold_rule_ids,
            predicted_rule_ids,
        ),

        "gold_article_count": len(gold_article_ids),
        "predicted_article_count": len(predicted_article_ids),
        "article_true_positive": article_tp,
        "article_precision": article_precision,
        "article_recall": article_recall,
        "article_f1": article_f1,
        "article_exact_set_match": exact_set_match(
            gold_article_ids,
            predicted_article_ids,
        ),

        "predicted_path_count": len(predicted_paths),
        "top1_path_rule_ids": "|".join(top1_path["rule_ids"]),
        "top1_path_nodes": "|".join(top1_path["nodes"]),
        "top1_path_edges": "|".join(
            f"{source}->{target}"
            for source, target in top1_path["edges"]
        ),
        "top1_node_precision": top1_path_metrics["node_precision"],
        "top1_node_recall": top1_path_metrics["node_recall"],
        "top1_node_f1": top1_path_metrics["node_f1"],
        "top1_edge_precision": top1_path_metrics["edge_precision"],
        "top1_edge_recall": top1_path_metrics["edge_recall"],
        "top1_edge_f1": top1_path_metrics["edge_f1"],
        "top1_path_rule_f1": top1_path_metrics["path_rule_f1"],
        "top1_final_event_accuracy": top1_path_metrics[
            "final_event_accuracy"
        ],
        "top1_exact_node_path_match": top1_path_metrics[
            "exact_node_path_match"
        ],
        "top1_exact_edge_path_match": top1_path_metrics[
            "exact_edge_path_match"
        ],
        "top1_exact_rule_path_match": top1_path_metrics[
            "exact_rule_path_match"
        ],
        "top1_exact_path_match": top1_path_metrics[
            "exact_path_match"
        ],

        # Oracle/best-of-returned-paths: dùng để biết retriever có đưa
        # gold path vào danh sách hay không, dù chưa xếp nó ở top 1.
        "oracle_best_path_rank": oracle_path_rank,
        "oracle_best_path_rule_ids": "|".join(best_path["rule_ids"]),
        "oracle_best_path_nodes": "|".join(best_path["nodes"]),
        "oracle_node_f1": oracle_path_metrics["node_f1"],
        "oracle_edge_f1": oracle_path_metrics["edge_f1"],
        "oracle_path_rule_f1": oracle_path_metrics["path_rule_f1"],
        "oracle_final_event_accuracy": oracle_path_metrics[
            "final_event_accuracy"
        ],
        "oracle_exact_path_match": oracle_path_metrics[
            "exact_path_match"
        ],

        "gold_counterfactual_status": gold_cf_status or "",
        "predicted_counterfactual_status": predicted_cf_status or "",
        "counterfactual_status_accuracy": cf_status_accuracy,

        "gold_final_effect_reachable": gold_reachable,
        "predicted_final_effect_reachable": predicted_reachable,
        "reachability_accuracy": reachability_accuracy,

        "gold_answer_label": gold_answer_label or "",
        "predicted_answer_label": predicted_answer_label or "",
        "answer_label_accuracy": answer_label_accuracy,

        "answer_token_precision": answer_precision,
        "answer_token_recall": answer_recall,
        "answer_token_f1": answer_token_f1,
        "answer_rouge_l_f1": answer_rouge_l,
        "answer_exact_match": answer_exact_match,

        "runtime_seconds": prediction.get("runtime_seconds"),
        "gold_rule_ids": "|".join(gold_rule_ids),
        "predicted_rule_ids": "|".join(predicted_rule_ids),
        "gold_article_ids": "|".join(gold_article_ids),
        "predicted_article_ids": "|".join(predicted_article_ids),
        "gold_path_nodes": "|".join(gold_path["nodes"]),
        "gold_path_edges": "|".join(
            f"{source}->{target}"
            for source, target in gold_path["edges"]
        ),
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
    }

    return {
        key: round_metric(value)
        for key, value in row.items()
    }


# ============================================================
# AGGREGATION
# ============================================================

def numeric_mean(
    rows: list[dict[str, Any]],
    key: str,
) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
        and isinstance(row.get(key), (int, float))
    ]
    if not values:
        return None
    return statistics.fmean(values)


def numeric_sum(
    rows: list[dict[str, Any]],
    key: str,
) -> float:
    return sum(
        float(row[key])
        for row in rows
        if row.get(key) is not None
        and isinstance(row.get(key), (int, float))
    )


def aggregate_group(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [
        row for row in rows
        if int(row.get("successful", 0)) == 1
    ]
    counterfactual_rows = [
        row for row in successful_rows
        if row.get("gold_counterfactual_status")
    ]

    rule_tp = numeric_sum(successful_rows, "rule_true_positive")
    predicted_rule_total = numeric_sum(
        successful_rows,
        "predicted_rule_count",
    )
    gold_rule_total = numeric_sum(
        successful_rows,
        "gold_rule_count",
    )

    rule_micro_precision = (
        rule_tp / predicted_rule_total
        if predicted_rule_total > 0
        else 0.0
    )
    rule_micro_recall = (
        rule_tp / gold_rule_total
        if gold_rule_total > 0
        else 0.0
    )
    rule_micro_f1 = (
        2
        * rule_micro_precision
        * rule_micro_recall
        / (rule_micro_precision + rule_micro_recall)
        if rule_micro_precision + rule_micro_recall > 0
        else 0.0
    )

    article_tp = numeric_sum(
        successful_rows,
        "article_true_positive",
    )
    predicted_article_total = numeric_sum(
        successful_rows,
        "predicted_article_count",
    )
    gold_article_total = numeric_sum(
        successful_rows,
        "gold_article_count",
    )

    article_micro_precision = (
        article_tp / predicted_article_total
        if predicted_article_total > 0
        else 0.0
    )
    article_micro_recall = (
        article_tp / gold_article_total
        if gold_article_total > 0
        else 0.0
    )
    article_micro_f1 = (
        2
        * article_micro_precision
        * article_micro_recall
        / (article_micro_precision + article_micro_recall)
        if article_micro_precision + article_micro_recall > 0
        else 0.0
    )

    runtime_values = [
        float(row["runtime_seconds"])
        for row in successful_rows
        if isinstance(row.get("runtime_seconds"), (int, float))
    ]

    result = {
        "sample_count": len(rows),
        "successful_sample_count": len(successful_rows),
        "failed_sample_count": len(rows) - len(successful_rows),
        "success_rate": (
            len(successful_rows) / len(rows)
            if rows
            else 0.0
        ),
        "average_runtime_seconds": (
            statistics.fmean(runtime_values)
            if runtime_values
            else None
        ),
        "median_runtime_seconds": (
            statistics.median(runtime_values)
            if runtime_values
            else None
        ),

        "retrieval": {
            "rule_macro_precision": numeric_mean(
                successful_rows, "rule_precision"
            ),
            "rule_macro_recall": numeric_mean(
                successful_rows, "rule_recall"
            ),
            "rule_macro_f1": numeric_mean(
                successful_rows, "rule_f1"
            ),
            "rule_micro_precision": rule_micro_precision,
            "rule_micro_recall": rule_micro_recall,
            "rule_micro_f1": rule_micro_f1,
            "rule_exact_set_match": numeric_mean(
                successful_rows,
                "rule_exact_set_match",
            ),
            "rule_recall_at_1": numeric_mean(
                successful_rows,
                "rule_recall_at_1",
            ),
            "rule_recall_at_3": numeric_mean(
                successful_rows,
                "rule_recall_at_3",
            ),
            "rule_recall_at_5": numeric_mean(
                successful_rows,
                "rule_recall_at_5",
            ),
            "rule_recall_at_10": numeric_mean(
                successful_rows,
                "rule_recall_at_10",
            ),
            "rule_hit_at_1": numeric_mean(
                successful_rows,
                "rule_hit_at_1",
            ),
            "rule_hit_at_3": numeric_mean(
                successful_rows,
                "rule_hit_at_3",
            ),
            "rule_hit_at_5": numeric_mean(
                successful_rows,
                "rule_hit_at_5",
            ),
            "rule_hit_at_10": numeric_mean(
                successful_rows,
                "rule_hit_at_10",
            ),
            "mrr": numeric_mean(
                successful_rows,
                "rule_mrr",
            ),
            "map": numeric_mean(
                successful_rows,
                "rule_average_precision",
            ),
            "article_macro_precision": numeric_mean(
                successful_rows,
                "article_precision",
            ),
            "article_macro_recall": numeric_mean(
                successful_rows,
                "article_recall",
            ),
            "article_macro_f1": numeric_mean(
                successful_rows,
                "article_f1",
            ),
            "article_micro_precision": article_micro_precision,
            "article_micro_recall": article_micro_recall,
            "article_micro_f1": article_micro_f1,
            "article_exact_set_match": numeric_mean(
                successful_rows,
                "article_exact_set_match",
            ),
        },

        "causal_path": {
            "top1_node_precision": numeric_mean(
                successful_rows,
                "top1_node_precision",
            ),
            "top1_node_recall": numeric_mean(
                successful_rows,
                "top1_node_recall",
            ),
            "top1_node_f1": numeric_mean(
                successful_rows,
                "top1_node_f1",
            ),
            "top1_edge_precision": numeric_mean(
                successful_rows,
                "top1_edge_precision",
            ),
            "top1_edge_recall": numeric_mean(
                successful_rows,
                "top1_edge_recall",
            ),
            "top1_edge_f1": numeric_mean(
                successful_rows,
                "top1_edge_f1",
            ),
            "top1_path_rule_f1": numeric_mean(
                successful_rows,
                "top1_path_rule_f1",
            ),
            "top1_final_event_accuracy": numeric_mean(
                successful_rows,
                "top1_final_event_accuracy",
            ),
            "top1_exact_node_path_match": numeric_mean(
                successful_rows,
                "top1_exact_node_path_match",
            ),
            "top1_exact_edge_path_match": numeric_mean(
                successful_rows,
                "top1_exact_edge_path_match",
            ),
            "top1_exact_rule_path_match": numeric_mean(
                successful_rows,
                "top1_exact_rule_path_match",
            ),
            "top1_exact_path_match": numeric_mean(
                successful_rows,
                "top1_exact_path_match",
            ),

            # Best path trong toàn bộ danh sách trả về.
            "oracle_node_f1": numeric_mean(
                successful_rows,
                "oracle_node_f1",
            ),
            "oracle_edge_f1": numeric_mean(
                successful_rows,
                "oracle_edge_f1",
            ),
            "oracle_path_rule_f1": numeric_mean(
                successful_rows,
                "oracle_path_rule_f1",
            ),
            "oracle_final_event_accuracy": numeric_mean(
                successful_rows,
                "oracle_final_event_accuracy",
            ),
            "oracle_exact_path_match": numeric_mean(
                successful_rows,
                "oracle_exact_path_match",
            ),
        },

        "counterfactual": {
            "evaluated_sample_count": len(counterfactual_rows),
            "status_accuracy": numeric_mean(
                counterfactual_rows,
                "counterfactual_status_accuracy",
            ),
            "final_effect_reachability_accuracy": numeric_mean(
                counterfactual_rows,
                "reachability_accuracy",
            ),
        },

        "answer": {
            "answer_label_accuracy": numeric_mean(
                successful_rows,
                "answer_label_accuracy",
            ),
            "token_precision": numeric_mean(
                successful_rows,
                "answer_token_precision",
            ),
            "token_recall": numeric_mean(
                successful_rows,
                "answer_token_recall",
            ),
            "token_f1": numeric_mean(
                successful_rows,
                "answer_token_f1",
            ),
            "rouge_l_f1": numeric_mean(
                successful_rows,
                "answer_rouge_l_f1",
            ),
            "normalized_exact_match": numeric_mean(
                successful_rows,
                "answer_exact_match",
            ),
        },
    }

    def recursive_round(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: recursive_round(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [recursive_round(item) for item in value]
        return round_metric(value)

    return recursive_round(result)


def group_rows(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = safe_string(row.get(key)) or "UNKNOWN"
        grouped[group_name].append(row)
    return dict(grouped)


# ============================================================
# ERROR ANALYSIS
# ============================================================

def build_error_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []

        if not int(row.get("successful", 0)):
            reasons.append("PIPELINE_ERROR")
        else:
            if float(row.get("rule_recall", 0.0)) < 1.0:
                reasons.append("MISSED_GOLD_RULE")
            if float(row.get("top1_exact_path_match", 0.0)) < 1.0:
                reasons.append("TOP1_PATH_MISMATCH")
            if (
                row.get("counterfactual_status_accuracy") is not None
                and float(row["counterfactual_status_accuracy"]) < 1.0
            ):
                reasons.append("COUNTERFACTUAL_STATUS_MISMATCH")
            if (
                row.get("reachability_accuracy") is not None
                and float(row["reachability_accuracy"]) < 1.0
            ):
                reasons.append("REACHABILITY_MISMATCH")
            if (
                row.get("answer_label_accuracy") is not None
                and float(row["answer_label_accuracy"]) < 1.0
            ):
                reasons.append("ANSWER_LABEL_MISMATCH")
            if float(row.get("answer_token_f1", 0.0)) < 0.50:
                reasons.append("LOW_ANSWER_TOKEN_F1")

        if not reasons:
            continue

        errors.append({
            "question_id": row["question_id"],
            "source_chain_id": row.get("source_chain_id"),
            "task_type": row.get("task_type"),
            "difficulty": row.get("difficulty"),
            "error_types": "|".join(reasons),
            "pipeline_error": row.get("pipeline_error", ""),
            "rule_recall": row.get("rule_recall"),
            "top1_node_f1": row.get("top1_node_f1"),
            "top1_edge_f1": row.get("top1_edge_f1"),
            "top1_exact_path_match": row.get(
                "top1_exact_path_match"
            ),
            "oracle_exact_path_match": row.get(
                "oracle_exact_path_match"
            ),
            "gold_counterfactual_status": row.get(
                "gold_counterfactual_status"
            ),
            "predicted_counterfactual_status": row.get(
                "predicted_counterfactual_status"
            ),
            "gold_final_effect_reachable": row.get(
                "gold_final_effect_reachable"
            ),
            "predicted_final_effect_reachable": row.get(
                "predicted_final_effect_reachable"
            ),
            "gold_answer_label": row.get("gold_answer_label"),
            "predicted_answer_label": row.get(
                "predicted_answer_label"
            ),
            "answer_token_f1": row.get("answer_token_f1"),
            "answer_rouge_l_f1": row.get("answer_rouge_l_f1"),
            "gold_rule_ids": row.get("gold_rule_ids"),
            "predicted_rule_ids": row.get("predicted_rule_ids"),
            "gold_path_nodes": row.get("gold_path_nodes"),
            "top1_path_nodes": row.get("top1_path_nodes"),
            "gold_answer": row.get("gold_answer"),
            "predicted_answer": row.get("predicted_answer"),
        })

    return errors


# ============================================================
# DISPLAY
# ============================================================

def format_percentage(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def print_summary(metrics: dict[str, Any]) -> None:
    overall = metrics["overall"]
    retrieval = overall["retrieval"]
    causal_path = overall["causal_path"]
    counterfactual = overall["counterfactual"]
    answer = overall["answer"]

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"Samples: {overall['sample_count']} | "
        f"Success: {overall['successful_sample_count']} | "
        f"Failed: {overall['failed_sample_count']}"
    )

    print("\n[Retrieval]")
    print(
        "Rule Macro P/R/F1: "
        f"{format_percentage(retrieval['rule_macro_precision'])} / "
        f"{format_percentage(retrieval['rule_macro_recall'])} / "
        f"{format_percentage(retrieval['rule_macro_f1'])}"
    )
    print(
        "Recall@1/3/5/10: "
        f"{format_percentage(retrieval['rule_recall_at_1'])} / "
        f"{format_percentage(retrieval['rule_recall_at_3'])} / "
        f"{format_percentage(retrieval['rule_recall_at_5'])} / "
        f"{format_percentage(retrieval['rule_recall_at_10'])}"
    )
    print(
        f"MRR: {retrieval['mrr'] or 0:.4f} | "
        f"MAP: {retrieval['map'] or 0:.4f}"
    )

    print("\n[Causal Path]")
    print(
        "Top-1 Node F1 / Edge F1 / Exact Path: "
        f"{format_percentage(causal_path['top1_node_f1'])} / "
        f"{format_percentage(causal_path['top1_edge_f1'])} / "
        f"{format_percentage(causal_path['top1_exact_path_match'])}"
    )
    print(
        "Oracle Node F1 / Edge F1 / Exact Path: "
        f"{format_percentage(causal_path['oracle_node_f1'])} / "
        f"{format_percentage(causal_path['oracle_edge_f1'])} / "
        f"{format_percentage(causal_path['oracle_exact_path_match'])}"
    )

    print("\n[Counterfactual]")
    print(
        f"Evaluated samples: "
        f"{counterfactual['evaluated_sample_count']}"
    )
    print(
        "Status Accuracy / Reachability Accuracy: "
        f"{format_percentage(counterfactual['status_accuracy'])} / "
        f"{format_percentage(counterfactual['final_effect_reachability_accuracy'])}"
    )

    print("\n[Answer]")
    print(
        "Label Accuracy / Token F1 / ROUGE-L F1: "
        f"{format_percentage(answer['answer_label_accuracy'])} / "
        f"{format_percentage(answer['token_f1'])} / "
        f"{format_percentage(answer['rouge_l_f1'])}"
    )
    print("=" * 72)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tính evaluation metrics cho CausalRAG từ gold dataset "
            "và evaluation predictions."
        )
    )
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD_PATH,
        help="Đường dẫn evaluation_dataset.json.",
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTION_PATH,
        help="Đường dẫn evaluation_predictions.json.",
    )
    parser.add_argument(
        "--metrics-output",
        default=DEFAULT_METRICS_PATH,
    )
    parser.add_argument(
        "--by-sample-output",
        default=DEFAULT_BY_SAMPLE_PATH,
    )
    parser.add_argument(
        "--errors-output",
        default=DEFAULT_ERRORS_PATH,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Dừng nếu thiếu prediction hoặc prediction có question_id "
            "không tồn tại trong gold dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gold_payload = load_json(args.gold)
    prediction_payload = load_json(args.predictions)

    gold_metadata, gold_samples = extract_gold_samples(
        gold_payload
    )
    prediction_metadata, predictions = extract_predictions(
        prediction_payload
    )

    gold_by_id = index_by_question_id(
        gold_samples,
        source_name="Gold dataset",
    )
    prediction_by_id = index_by_question_id(
        predictions,
        source_name="Predictions",
    )

    missing_prediction_ids = sorted(
        set(gold_by_id) - set(prediction_by_id)
    )
    unexpected_prediction_ids = sorted(
        set(prediction_by_id) - set(gold_by_id)
    )

    if args.strict and missing_prediction_ids:
        raise ValueError(
            "Thiếu predictions cho: "
            + ", ".join(missing_prediction_ids)
        )
    if args.strict and unexpected_prediction_ids:
        raise ValueError(
            "Predictions có question_id không thuộc gold dataset: "
            + ", ".join(unexpected_prediction_ids)
        )

    rows: list[dict[str, Any]] = []

    for gold_sample in gold_samples:
        question_id = safe_string(
            gold_sample.get("question_id")
        )
        prediction = prediction_by_id.get(question_id)
        rows.append(
            evaluate_sample(
                gold=gold_sample,
                prediction=prediction,
            )
        )

    overall_metrics = aggregate_group(rows)

    by_task_type = {
        group_name: aggregate_group(group_rows_value)
        for group_name, group_rows_value in sorted(
            group_rows(rows, "task_type").items()
        )
    }
    by_difficulty = {
        group_name: aggregate_group(group_rows_value)
        for group_name, group_rows_value in sorted(
            group_rows(rows, "difficulty").items()
        )
    }

    metrics = {
        "input": {
            "gold_path": args.gold,
            "prediction_path": args.predictions,
            "gold_metadata": gold_metadata,
            "prediction_metadata": prediction_metadata,
            "missing_prediction_ids": missing_prediction_ids,
            "unexpected_prediction_ids": unexpected_prediction_ids,
        },
        "overall": overall_metrics,
        "by_task_type": by_task_type,
        "by_difficulty": by_difficulty,
        "metric_notes": {
            "top1_path_metrics": (
                "Đánh giá causal path được xếp hạng đầu tiên."
            ),
            "oracle_path_metrics": (
                "Chọn path khớp gold tốt nhất trong toàn bộ predicted_paths; "
                "dùng để phân biệt lỗi retrieval và lỗi ranking."
            ),
            "answer_token_f1": (
                "Token overlap F1 sau khi lowercase, bỏ dấu tiếng Việt "
                "và bỏ dấu câu."
            ),
            "rouge_l_f1": (
                "F1 dựa trên longest common subsequence ở mức token."
            ),
            "normalized_exact_match": (
                "Chỉ dùng tham khảo; không nên là metric chính cho đáp án "
                "sinh tự do."
            ),
        },
    }

    error_rows = build_error_rows(rows)

    save_json(metrics, args.metrics_output)
    save_csv(rows, args.by_sample_output)
    save_csv(error_rows, args.errors_output)

    print_summary(metrics)
    print(f"\nMetrics JSON: {args.metrics_output}")
    print(f"Per-sample CSV: {args.by_sample_output}")
    print(f"Error analysis CSV: {args.errors_output}")

    if missing_prediction_ids:
        print(
            "\nWarning - missing predictions: "
            + ", ".join(missing_prediction_ids)
        )
    if unexpected_prediction_ids:
        print(
            "\nWarning - unexpected predictions: "
            + ", ".join(unexpected_prediction_ids)
        )


if __name__ == "__main__":
    main()
