#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare controlled system-level baselines on the same BLHS samples.

Use this script for the paper's main LLM-only vs Vanilla RAG vs Causal-path
RAG vs Full LegalSCM table. Step 7 remains useful for detailed diagnostics of
one causal pipeline; this comparator makes cross-system metric applicability
explicit and penalizes missing/failed predictions in all shared end-to-end
metrics.

Example
-------
python 11_compare_system_baselines.py \
  --predictions \
    llm_only=data/baselines/system_level/llm_only_qwen3_8b_predictions.json \
    vanilla_rag=data/baselines/system_level/vanilla_rag_qwen3_8b_predictions.json \
    causal_path_rag=data/baselines/system_level/causal_path_rag_qwen3_8b_predictions.json \
    full_legal_scm=data/baselines/system_level/full_legal_scm_qwen3_8b_predictions.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


COMPARATOR_VERSION = "1.0-system-level-paired-comparison"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = "data/blhs_multihop_benchmark_250.json"
DEFAULT_EVALUATOR = "7_compute_evaluation_metrics.py"
DEFAULT_OUTPUT_DIR = "evaluation_results/system_level_comparison"
DEFAULT_REFERENCE = "full_legal_scm"
DEFAULT_K = 5
DEFAULT_BOOTSTRAP_ITERATIONS = 5000
DEFAULT_PERMUTATION_ITERATIONS = 5000
DEFAULT_RANDOM_SEED = 42

COMMON_METRICS = (
    "decision_accuracy",
    "answer_token_f1",
    "answer_rouge_l_f1",
    "citation_f1",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = safe_string(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def normalize_name(value: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", safe_string(value)).strip("_").lower()
    aliases = {
        "llm": "llm_only",
        "llm_only": "llm_only",
        "vanilla": "vanilla_rag",
        "rag": "vanilla_rag",
        "vanilla_rag": "vanilla_rag",
        "causal": "causal_path_rag",
        "causal_rag": "causal_path_rag",
        "causal_path": "causal_path_rag",
        "causal_path_rag": "causal_path_rag",
        "full": "full_legal_scm",
        "proposed": "full_legal_scm",
        "full_legal_scm": "full_legal_scm",
    }
    return aliases.get(text, text)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def recursive_round(value: Any, digits: int = 6) -> Any:
    if isinstance(value, Mapping):
        return {str(key): recursive_round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_round(item, digits) for item in value]
    if isinstance(value, tuple):
        return [recursive_round(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(recursive_round(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_module(script_path: Path) -> Any:
    module_name = "causalrag_step7_system_comparison"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể import evaluator: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(numbers) if numbers else None


def format_percentage(value: Any) -> str:
    number = safe_float(value)
    return "N/A" if number is None else f"{number * 100:.2f}%"


def format_number(value: Any, digits: int = 3) -> str:
    number = safe_float(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


# ---------------------------------------------------------------------------
# Input specifications and fairness contract
# ---------------------------------------------------------------------------

def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = resolve_repo_path(value)
        return normalize_name(path.stem), path
    name, raw_path = value.split("=", 1)
    method = normalize_name(name)
    if not method:
        raise ValueError(f"Tên method rỗng trong --predictions {value!r}.")
    return method, resolve_repo_path(raw_path)


def generation_signature(metadata: Mapping[str, Any]) -> dict[str, Any]:
    configuration = metadata.get("configuration") or {}
    if not isinstance(configuration, Mapping):
        configuration = {}
    return {
        "generation_contract_version": safe_string(
            metadata.get("generation_contract_version")
        ),
        "prompt_version": safe_string(metadata.get("prompt_version")),
        "provider": safe_string(configuration.get("provider") or metadata.get("provider")),
        "model": safe_string(configuration.get("model") or metadata.get("model")),
        "temperature": safe_float(configuration.get("temperature")),
        "max_tokens": safe_float(configuration.get("max_tokens")),
        "num_ctx": safe_float(configuration.get("num_ctx")),
        "seed": safe_float(configuration.get("seed")),
        "thinking": safe_string(configuration.get("thinking")),
        "format": safe_string(configuration.get("format")),
        "max_context_chars": safe_float(configuration.get("max_context_chars")),
        "extractive_fallback": bool(metadata.get("extractive_fallback", False)),
    }


def validate_generation_contracts(
    metadata_by_method: Mapping[str, Mapping[str, Any]],
    *,
    allow_mismatch: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    signatures = {
        method: generation_signature(metadata)
        for method, metadata in metadata_by_method.items()
    }
    warnings: list[str] = []
    methods = list(signatures)
    if not methods:
        return signatures, warnings
    reference_method = methods[0]
    reference = signatures[reference_method]
    required = (
        "generation_contract_version",
        "prompt_version",
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "num_ctx",
        "seed",
        "thinking",
        "format",
        "max_context_chars",
    )
    missing = {
        method: [key for key in required if signature.get(key) in {None, ""}]
        for method, signature in signatures.items()
    }
    missing = {method: keys for method, keys in missing.items() if keys}
    if missing:
        warnings.append(f"Thiếu generation contract fields: {missing}")
    for method in methods[1:]:
        differences = {
            key: (reference.get(key), signatures[method].get(key))
            for key in required
            if reference.get(key) != signatures[method].get(key)
        }
        if differences:
            warnings.append(
                f"Generation contract mismatch {reference_method} vs {method}: {differences}"
            )
    fallback_methods = [
        method for method, signature in signatures.items()
        if signature.get("extractive_fallback")
    ]
    if fallback_methods:
        warnings.append(
            "Extractive fallback được bật ở: " + ", ".join(fallback_methods)
        )
    if warnings and not allow_mismatch:
        raise ValueError(
            "Các prediction không dùng cùng generation contract. "
            "Chạy lại bằng 10_generate_system_baselines.py hoặc dùng "
            "--allow-contract-mismatch chỉ cho phân tích chẩn đoán.\n- "
            + "\n- ".join(warnings)
        )
    return signatures, warnings


def infer_applicability(
    method: str,
    metadata: Mapping[str, Any],
) -> dict[str, bool]:
    explicit = metadata.get("metric_applicability") or {}
    if isinstance(explicit, Mapping) and explicit:
        return {
            "rule_retrieval": safe_bool(explicit.get("rule_retrieval")),
            "event_retrieval": safe_bool(explicit.get("event_retrieval")),
            "article_retrieval": safe_bool(explicit.get("article_retrieval")),
            "causal_path": safe_bool(explicit.get("causal_path")),
            "counterfactual_verifier": safe_bool(explicit.get("counterfactual_verifier")),
        }
    normalized = normalize_name(method)
    causal = normalized in {"causal_path_rag", "full_legal_scm"}
    return {
        "rule_retrieval": causal,
        "event_retrieval": causal,
        "article_retrieval": normalized != "llm_only",
        "causal_path": causal,
        "counterfactual_verifier": normalized == "full_legal_scm",
    }


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    *,
    evaluator: Any,
    method: str,
    applicability: Mapping[str, bool],
    gold_samples: Sequence[Any],
    predictions: Mapping[str, Any],
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold in gold_samples:
        prediction = predictions.get(gold.sample_id)
        missing = prediction is None
        if missing:
            prediction = evaluator.missing_prediction(gold.sample_id)
        successful = not bool(prediction.pipeline_error)

        predicted_decision = (
            evaluator.normalize_decision(prediction.verification_decision)
            if successful
            else "NO_PREDICTION"
        )
        decision_correct = float(
            successful and predicted_decision == gold.gold_decision
        )
        answer_precision, answer_recall, answer_f1 = evaluator.token_overlap_metrics(
            gold.answer,
            prediction.final_answer if successful else "",
        )
        answer_rouge = evaluator.rouge_l_f1(
            gold.answer,
            prediction.final_answer if successful else "",
        )
        citation_metrics = evaluator.set_metrics(
            gold.gold_citations,
            prediction.citations if successful else [],
        )

        rule_applicable = bool(applicability.get("rule_retrieval"))
        event_applicable = bool(applicability.get("event_retrieval"))
        article_applicable = bool(applicability.get("article_retrieval"))
        path_applicable = bool(
            applicability.get("causal_path") and gold.gold_path_edges
        )

        if path_applicable:
            path_metrics = evaluator.path_metric_values(
                gold.gold_path_edges,
                prediction.selected_path,
                gold.gold_path_rule_ids,
            )
            _, oracle_metrics = evaluator.choose_oracle_candidate(
                gold,
                prediction.candidate_paths,
            )
        else:
            path_metrics = {}
            oracle_metrics = {}

        row = {
            "method": method,
            "id": gold.sample_id,
            "question_type": gold.question_type,
            "difficulty": gold.difficulty,
            "requires_counterfactual": int(gold.requires_counterfactual),
            "prediction_present": int(not missing),
            "successful": int(successful),
            "pipeline_error": prediction.pipeline_error,
            "gold_decision": gold.gold_decision,
            "predicted_decision": predicted_decision,
            "decision_accuracy": decision_correct,
            "answer_token_precision": answer_precision,
            "answer_token_recall": answer_recall,
            "answer_token_f1": answer_f1,
            "answer_rouge_l_f1": answer_rouge,
            "citation_precision": citation_metrics["precision"],
            "citation_recall": citation_metrics["recall"],
            "citation_f1": citation_metrics["f1"],
            "runtime_seconds": prediction.runtime_seconds,
            "rule_metric_applicable": int(rule_applicable),
            "event_metric_applicable": int(event_applicable),
            "article_metric_applicable": int(article_applicable),
            "path_metric_applicable": int(path_applicable),
            f"rule_recall_at_{k}": (
                evaluator.recall_at_k(
                    gold.gold_rule_ids,
                    prediction.retrieved_rule_ids,
                    k,
                )
                if rule_applicable else None
            ),
            "rule_mrr": (
                evaluator.reciprocal_rank(
                    gold.gold_rule_ids,
                    prediction.retrieved_rule_ids,
                )
                if rule_applicable else None
            ),
            "rule_map": (
                evaluator.average_precision(
                    gold.gold_rule_ids,
                    prediction.retrieved_rule_ids,
                )
                if rule_applicable else None
            ),
            f"event_recall_at_{k}": (
                evaluator.recall_at_k(
                    gold.gold_event_ids,
                    prediction.retrieved_event_ids,
                    k,
                )
                if event_applicable else None
            ),
            "event_mrr": (
                evaluator.reciprocal_rank(
                    gold.gold_event_ids,
                    prediction.retrieved_event_ids,
                )
                if event_applicable else None
            ),
            f"article_recall_at_{k}": (
                evaluator.recall_at_k(
                    gold.gold_article_ids,
                    prediction.retrieved_article_ids,
                    k,
                )
                if article_applicable else None
            ),
            "article_mrr": (
                evaluator.reciprocal_rank(
                    gold.gold_article_ids,
                    prediction.retrieved_article_ids,
                )
                if article_applicable else None
            ),
            "article_map": (
                evaluator.average_precision(
                    gold.gold_article_ids,
                    prediction.retrieved_article_ids,
                )
                if article_applicable else None
            ),
            "top1_exact_path_match": (
                path_metrics.get("exact_path_match") if path_applicable else None
            ),
            "top1_edge_f1": (
                path_metrics.get("edge_f1") if path_applicable else None
            ),
            "oracle_exact_path_match": (
                oracle_metrics.get("exact_path_match") if path_applicable else None
            ),
            "oracle_edge_f1": (
                oracle_metrics.get("edge_f1") if path_applicable else None
            ),
        }
        rows.append(recursive_round(row))
    return rows


def classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("SUPPORTED", "REJECT_DIRECT_CLAIM", "UNCERTAIN")
    matrix: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for row in rows:
        gold = safe_string(row.get("gold_decision"))
        predicted = safe_string(row.get("predicted_decision")) or "NO_PREDICTION"
        matrix.setdefault(gold, Counter())[predicted] += 1

    per_class: dict[str, dict[str, Any]] = {}
    class_f1: list[float] = []
    class_recall: list[float] = []
    total = len(rows)
    correct = sum(
        1 for row in rows
        if row.get("gold_decision") == row.get("predicted_decision")
    )
    predicted_labels = set(labels)
    for values in matrix.values():
        predicted_labels.update(values)

    for label in labels:
        tp = matrix.get(label, Counter()).get(label, 0)
        fp = sum(
            matrix.get(other, Counter()).get(label, 0)
            for other in matrix if other != label
        )
        fn = sum(
            count for predicted, count in matrix.get(label, Counter()).items()
            if predicted != label
        )
        support = sum(matrix.get(label, Counter()).values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        if support:
            class_f1.append(f1)
            class_recall.append(recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    matrix_payload = {
        gold: {predicted: values.get(predicted, 0) for predicted in sorted(predicted_labels)}
        for gold, values in matrix.items()
    }
    return {
        "sample_count": total,
        "accuracy": correct / total if total else None,
        "macro_f1": statistics.fmean(class_f1) if class_f1 else None,
        "balanced_accuracy": statistics.fmean(class_recall) if class_recall else None,
        "per_class": per_class,
        "confusion_matrix": matrix_payload,
    }


def aggregate_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    applicability: Mapping[str, bool],
    k: int,
) -> dict[str, Any]:
    rows = list(rows)
    successful_rows = [row for row in rows if row.get("successful") == 1]
    counterfactual_rows = [
        row for row in rows if row.get("requires_counterfactual") == 1
    ]
    runtime_values = [
        float(row["runtime_seconds"])
        for row in successful_rows
        if safe_float(row.get("runtime_seconds")) is not None
    ]
    return recursive_round({
        "sample_count": len(rows),
        "prediction_coverage": mean_or_none(row.get("prediction_present") for row in rows),
        "success_rate": mean_or_none(row.get("successful") for row in rows),
        "failed_or_missing_count": sum(row.get("successful") != 1 for row in rows),
        "quality_scope": "all_gold_samples_failures_score_zero",
        "verification": classification_metrics(rows),
        "answer": {
            "token_precision": mean_or_none(row.get("answer_token_precision") for row in rows),
            "token_recall": mean_or_none(row.get("answer_token_recall") for row in rows),
            "token_f1": mean_or_none(row.get("answer_token_f1") for row in rows),
            "rouge_l_f1": mean_or_none(row.get("answer_rouge_l_f1") for row in rows),
        },
        "citation": {
            "precision": mean_or_none(row.get("citation_precision") for row in rows),
            "recall": mean_or_none(row.get("citation_recall") for row in rows),
            "f1": mean_or_none(row.get("citation_f1") for row in rows),
        },
        "retrieval": {
            "rule_recall_at_k": mean_or_none(row.get(f"rule_recall_at_{k}") for row in rows),
            "rule_mrr": mean_or_none(row.get("rule_mrr") for row in rows),
            "rule_map": mean_or_none(row.get("rule_map") for row in rows),
            "event_recall_at_k": mean_or_none(row.get(f"event_recall_at_{k}") for row in rows),
            "event_mrr": mean_or_none(row.get("event_mrr") for row in rows),
            "article_recall_at_k": mean_or_none(row.get(f"article_recall_at_{k}") for row in rows),
            "article_mrr": mean_or_none(row.get("article_mrr") for row in rows),
            "article_map": mean_or_none(row.get("article_map") for row in rows),
        },
        "causal_path": {
            "applicable_sample_count": sum(row.get("path_metric_applicable") == 1 for row in rows),
            "top1_exact_path_match": mean_or_none(row.get("top1_exact_path_match") for row in rows),
            "top1_edge_f1": mean_or_none(row.get("top1_edge_f1") for row in rows),
            "oracle_exact_path_match": mean_or_none(row.get("oracle_exact_path_match") for row in rows),
            "oracle_edge_f1": mean_or_none(row.get("oracle_edge_f1") for row in rows),
        },
        "counterfactual_subset": {
            "sample_count": len(counterfactual_rows),
            "decision_accuracy": mean_or_none(row.get("decision_accuracy") for row in counterfactual_rows),
            "answer_token_f1": mean_or_none(row.get("answer_token_f1") for row in counterfactual_rows),
            "citation_f1": mean_or_none(row.get("citation_f1") for row in counterfactual_rows),
        },
        "runtime": {
            "average_seconds_successful_only": (
                statistics.fmean(runtime_values) if runtime_values else None
            ),
            "median_seconds_successful_only": (
                statistics.median(runtime_values) if runtime_values else None
            ),
        },
        "metric_applicability": dict(applicability),
    })


# ---------------------------------------------------------------------------
# Paired uncertainty and significance
# ---------------------------------------------------------------------------

def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    size = len(values)
    estimates = [
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(iterations)
    ]
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_bootstrap_delta_ci(
    reference: Sequence[float],
    comparator: Sequence[float],
    *,
    iterations: int,
    rng: random.Random,
) -> tuple[float, float]:
    differences = [left - right for left, right in zip(reference, comparator)]
    return bootstrap_mean_ci(differences, iterations=iterations, rng=rng)


def paired_randomization_p_value(
    reference: Sequence[float],
    comparator: Sequence[float],
    *,
    iterations: int,
    rng: random.Random,
) -> float:
    differences = [left - right for left, right in zip(reference, comparator)]
    if not differences:
        return 1.0
    observed = abs(statistics.fmean(differences))
    extreme = 0
    for _ in range(iterations):
        randomized = statistics.fmean(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        )
        if abs(randomized) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def exact_mcnemar_p_value(
    reference_correct: Sequence[float],
    comparator_correct: Sequence[float],
) -> tuple[int, int, float]:
    reference_only = sum(
        bool(left) and not bool(right)
        for left, right in zip(reference_correct, comparator_correct)
    )
    comparator_only = sum(
        bool(right) and not bool(left)
        for left, right in zip(reference_correct, comparator_correct)
    )
    discordant = reference_only + comparator_only
    if discordant == 0:
        return reference_only, comparator_only, 1.0
    lower = min(reference_only, comparator_only)
    probability = sum(
        math.comb(discordant, index)
        for index in range(lower + 1)
    ) / (2 ** discordant)
    return reference_only, comparator_only, min(1.0, 2.0 * probability)


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed: list[tuple[int, float]] = []
    for index, row in enumerate(rows):
        parsed = safe_float(row.get("p_value"), 1.0)
        indexed.append((index, 1.0 if parsed is None else parsed))
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    running = 0.0
    adjusted: dict[int, float] = {}
    for rank, (index, p_value) in enumerate(indexed):
        candidate = min(1.0, (total - rank) * p_value)
        running = max(running, candidate)
        adjusted[index] = running
    for index, value in adjusted.items():
        rows[index]["p_value_holm"] = value


def build_significance(
    *,
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_method: str,
    bootstrap_iterations: int,
    permutation_iterations: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if reference_method not in rows_by_method:
        raise ValueError(f"Không tìm thấy reference method: {reference_method}")
    rng = random.Random(random_seed)
    reference_rows = rows_by_method[reference_method]
    output: list[dict[str, Any]] = []
    for method, rows in rows_by_method.items():
        if method == reference_method:
            continue
        if len(rows) != len(reference_rows):
            raise ValueError("Paired comparison yêu cầu cùng số sample.")
        for metric in COMMON_METRICS:
            reference_values = [float(row[metric]) for row in reference_rows]
            comparator_values = [float(row[metric]) for row in rows]
            delta = statistics.fmean(
                left - right
                for left, right in zip(reference_values, comparator_values)
            )
            ci_low, ci_high = paired_bootstrap_delta_ci(
                reference_values,
                comparator_values,
                iterations=bootstrap_iterations,
                rng=rng,
            )
            if metric == "decision_accuracy":
                ref_only, cmp_only, p_value = exact_mcnemar_p_value(
                    reference_values,
                    comparator_values,
                )
                test = "exact_mcnemar"
                details = {
                    "reference_only_correct": ref_only,
                    "comparator_only_correct": cmp_only,
                }
            else:
                p_value = paired_randomization_p_value(
                    reference_values,
                    comparator_values,
                    iterations=permutation_iterations,
                    rng=rng,
                )
                test = "paired_randomization"
                details = {}
            output.append({
                "reference_method": reference_method,
                "comparator_method": method,
                "metric": metric,
                "delta_reference_minus_comparator": delta,
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "test": test,
                "p_value": p_value,
                **details,
            })
    holm_adjust(output)
    return recursive_round(output)


def build_method_confidence_intervals(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    output: dict[str, Any] = {}
    for method, rows in rows_by_method.items():
        output[method] = {}
        for metric in COMMON_METRICS:
            values = [float(row[metric]) for row in rows]
            low, high = bootstrap_mean_ci(values, iterations=iterations, rng=rng)
            output[method][metric] = {
                "mean": statistics.fmean(values),
                "bootstrap_ci_95_low": low,
                "bootstrap_ci_95_high": high,
            }
    return recursive_round(output)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary_markdown(
    *,
    aggregates: Mapping[str, Mapping[str, Any]],
    significance_rows: Sequence[Mapping[str, Any]],
    reference_method: str,
    k: int,
    contract_warnings: Sequence[str],
) -> str:
    lines = [
        "# BLHS System-Level Baseline Comparison",
        "",
        f"- Comparator version: `{COMPARATOR_VERSION}`",
        f"- Reference method: `{reference_method}`",
        "- Shared end-to-end quality metrics include every gold sample; a missing or failed prediction receives zero.",
        "- Retrieval/path cells are `N/A` when a method does not expose that component.",
        "- Top-1 path is computed only on samples with a linear gold path; branch/convergence samples are excluded.",
        "",
        "## Main comparison",
        "",
        f"| Method | Success | Decision Acc. | Macro-F1 | Answer F1 | ROUGE-L | Citation F1 | Article R@{k} | Rule R@{k} | Event R@{k} | Top-1 Path | Avg. sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, aggregate in aggregates.items():
        verification = aggregate["verification"]
        answer = aggregate["answer"]
        citation = aggregate["citation"]
        retrieval = aggregate["retrieval"]
        path = aggregate["causal_path"]
        runtime = aggregate["runtime"]
        lines.append(
            f"| {method} | {format_percentage(aggregate['success_rate'])} | "
            f"{format_percentage(verification['accuracy'])} | "
            f"{format_percentage(verification['macro_f1'])} | "
            f"{format_percentage(answer['token_f1'])} | "
            f"{format_percentage(answer['rouge_l_f1'])} | "
            f"{format_percentage(citation['f1'])} | "
            f"{format_percentage(retrieval['article_recall_at_k'])} | "
            f"{format_percentage(retrieval['rule_recall_at_k'])} | "
            f"{format_percentage(retrieval['event_recall_at_k'])} | "
            f"{format_percentage(path['top1_exact_path_match'])} | "
            f"{format_number(runtime['average_seconds_successful_only'])} |"
        )

    lines.extend([
        "",
        "## Paired significance against the reference",
        "",
        "Positive delta means the reference method scores higher.",
        "",
        "| Comparator | Metric | Delta | 95% CI | Test | p | Holm p |",
        "|---|---|---:|---:|---|---:|---:|",
    ])
    for row in significance_rows:
        lines.append(
            f"| {row['comparator_method']} | {row['metric']} | "
            f"{format_number(row['delta_reference_minus_comparator'], 4)} | "
            f"[{format_number(row['bootstrap_ci_95_low'], 4)}, "
            f"{format_number(row['bootstrap_ci_95_high'], 4)}] | "
            f"{row['test']} | {format_number(row['p_value'], 4)} | "
            f"{format_number(row['p_value_holm'], 4)} |"
        )

    if contract_warnings:
        lines.extend(["", "## Contract warnings", ""])
        lines.extend(f"- {warning}" for warning in contract_warnings)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare controlled BLHS system-level baseline predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--evaluator-script", default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        metavar="METHOD=PATH",
    )
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=DEFAULT_PERMUTATION_ITERATIONS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--allow-contract-mismatch", action="store_true")
    parser.add_argument(
        "--require-complete-success",
        action="store_true",
        help="Dừng nếu bất kỳ method nào thiếu sample hoặc có prediction error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k phải lớn hơn 0.")
    if args.bootstrap_iterations < 100 or args.permutation_iterations < 100:
        raise ValueError("Số iteration phải ít nhất 100.")

    benchmark_path = resolve_repo_path(args.benchmark)
    evaluator_path = resolve_repo_path(args.evaluator_script)
    output_dir = resolve_repo_path(args.output_dir)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Không tìm thấy benchmark: {benchmark_path}")
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Không tìm thấy evaluator: {evaluator_path}")

    specifications: list[tuple[str, Path]] = [
        parse_prediction_spec(value) for value in args.predictions
    ]
    methods = [method for method, _ in specifications]
    duplicates = [method for method, count in Counter(methods).items() if count > 1]
    if duplicates:
        raise ValueError(f"Method bị trùng: {duplicates}")
    if len(specifications) < 2:
        raise ValueError("Cần ít nhất hai prediction files để so sánh.")
    missing_files = [str(path) for _, path in specifications if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Thiếu prediction files: " + ", ".join(missing_files))

    evaluator = load_module(evaluator_path)
    benchmark_metadata, gold_samples = evaluator.load_benchmark(benchmark_path)

    metadata_by_method: dict[str, dict[str, Any]] = {}
    predictions_by_method: dict[str, dict[str, Any]] = {}
    applicability_by_method: dict[str, dict[str, bool]] = {}
    input_paths: dict[str, str] = {}
    for method, path in specifications:
        metadata, predictions = evaluator.load_predictions(path)
        metadata_by_method[method] = dict(metadata)
        predictions_by_method[method] = predictions
        applicability_by_method[method] = infer_applicability(method, metadata)
        input_paths[method] = str(path)

    signatures, contract_warnings = validate_generation_contracts(
        metadata_by_method,
        allow_mismatch=args.allow_contract_mismatch,
    )

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    completeness_errors: list[str] = []
    gold_ids = {sample.sample_id for sample in gold_samples}
    for method in methods:
        prediction_ids = set(predictions_by_method[method])
        missing_ids = sorted(gold_ids - prediction_ids)
        unexpected_ids = sorted(prediction_ids - gold_ids)
        error_ids = sorted(
            sample_id for sample_id, prediction in predictions_by_method[method].items()
            if prediction.pipeline_error
        )
        if missing_ids or unexpected_ids or error_ids:
            completeness_errors.append(
                f"{method}: missing={len(missing_ids)}, unexpected={len(unexpected_ids)}, errors={len(error_ids)}"
            )
        rows = evaluate_method(
            evaluator=evaluator,
            method=method,
            applicability=applicability_by_method[method],
            gold_samples=gold_samples,
            predictions=predictions_by_method[method],
            k=args.k,
        )
        rows_by_method[method] = rows
        all_rows.extend(rows)

    if completeness_errors and args.require_complete_success:
        raise ValueError(
            "Prediction sets chưa hoàn chỉnh:\n- " + "\n- ".join(completeness_errors)
        )

    reference = normalize_name(args.reference)
    if reference not in rows_by_method:
        raise ValueError(
            f"Reference {reference!r} không có trong methods: {methods}"
        )

    aggregates = {
        method: aggregate_method(
            rows_by_method[method],
            applicability=applicability_by_method[method],
            k=args.k,
        )
        for method in methods
    }
    confidence_intervals = build_method_confidence_intervals(
        rows_by_method,
        iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    significance_rows = build_significance(
        rows_by_method=rows_by_method,
        reference_method=reference,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        random_seed=args.random_seed,
    )

    report = {
        "version": COMPARATOR_VERSION,
        "created_at_utc": utc_now_iso(),
        "benchmark": str(benchmark_path),
        "benchmark_metadata": benchmark_metadata,
        "prediction_paths": input_paths,
        "reference_method": reference,
        "k": args.k,
        "quality_scope": "all_gold_samples_failures_score_zero",
        "generation_signatures": signatures,
        "generation_contract_warnings": contract_warnings,
        "completeness_warnings": completeness_errors,
        "metric_applicability": applicability_by_method,
        "aggregates": aggregates,
        "bootstrap_confidence_intervals": confidence_intervals,
        "paired_significance": significance_rows,
        "statistical_settings": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "random_seed": args.random_seed,
            "multiple_comparison_correction": "Holm",
            "decision_test": "exact McNemar",
            "continuous_metric_test": "paired randomization/sign-flip",
        },
        "metric_notes": {
            "shared_metrics": (
                "Decision, answer and citation metrics use all gold samples; "
                "missing/failed predictions receive zero."
            ),
            "retrieval_na": (
                "Retrieval/path metrics are null when the component is not "
                "available for a method, not zero."
            ),
            "path_scope": (
                "Linear path metrics exclude branch/convergence samples whose "
                "gold_path is empty."
            ),
            "oracle_path": "Oracle path remains diagnostic and is not a deployable score.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "system_comparison_report.json"
    sample_path = output_dir / "system_comparison_by_sample.csv"
    significance_path = output_dir / "system_comparison_significance.csv"
    summary_path = output_dir / "system_comparison_summary.md"
    write_json(report_path, report)
    write_csv(sample_path, all_rows)
    write_csv(significance_path, significance_rows)
    summary_path.write_text(
        build_summary_markdown(
            aggregates=aggregates,
            significance_rows=significance_rows,
            reference_method=reference,
            k=args.k,
            contract_warnings=contract_warnings,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("BLHS SYSTEM-LEVEL BASELINE COMPARISON")
    print("=" * 80)
    for method, aggregate in aggregates.items():
        print(
            f"{method:22s} | success={format_percentage(aggregate['success_rate'])} "
            f"| decision={format_percentage(aggregate['verification']['accuracy'])} "
            f"| answer_f1={format_percentage(aggregate['answer']['token_f1'])} "
            f"| citation_f1={format_percentage(aggregate['citation']['f1'])}"
        )
    print("Report      :", report_path)
    print("Per sample  :", sample_path)
    print("Significance:", significance_path)
    print("Summary     :", summary_path)
    if completeness_errors:
        print("Warnings:")
        for warning in completeness_errors:
            print("-", warning)
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
