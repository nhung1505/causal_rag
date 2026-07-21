from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/4_blhs_merged.json"
MEMORY_PATH = "data/causal_memory.csv"
FAISS_INDEX_PATH = "data/causal_memory.index"
RETRIEVAL_RESULT_PATH = "data/retrieval_result.json"

OUTPUT_PATH = "data/counterfactual_result.json"
CONTEXT_OUTPUT_PATH = "data/counterfactual_result_context.txt"

MODEL_NAME = "BAAI/bge-m3"

DEFAULT_TOP_K_ALTERNATIVES = 10
DEFAULT_SEMANTIC_SEARCH_K = 40
DEFAULT_MAX_PATHS = 12
MIN_SEMANTIC_SCORE = 0.30

ALTERNATIVE_WEIGHTS = {
    "REPLACEMENT_CONDITION": 0.55,
    "CAUSES_TARGET_EFFECT": 0.45,
    "SAME_ARTICLE": 0.32,
    "ARTICLE_REFERENCE": 0.30,
    "SEMANTIC": 0.28,
    "SAME_SUBJECT": 0.22,
    "SAME_EFFECT": 0.15,
    "SAME_CONDITION": 0.10,
}

CONTRADICTION_KEYWORDS = {
    "KHONG", "CAM", "MIEN", "LOAI_TRU", "DINH_CHI",
    "HUY", "TU_CHOI", "CHAM_DUT",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Rule:
    rule_id: str
    row_id: int
    article_id: str
    legal_subject: str
    subject_norm: str
    condition: str
    effect: str
    condition_norm: str
    effect_norm: str
    condition_event_id: str
    effect_event_id: str
    article_title: str
    content: str


@dataclass
class Intervention:
    intervention_type: str
    target_event_norm: str
    target_event_text: str
    replacement_event_norm: Optional[str] = None
    replacement_event_text: Optional[str] = None
    description: str = ""


@dataclass
class AlternativeRule:
    rule_id: str
    score: float
    relations: list[str]
    semantic_score: Optional[float]
    explanation: str


# ============================================================
# HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def remove_vietnamese_accents(text: str) -> str:
    text = safe_string(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def normalize_identifier(text: Any) -> str:
    value = remove_vietnamese_accents(safe_string(text)).upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def event_id(norm: str) -> str:
    return f"EVENT::{normalize_identifier(norm)}"


def tokenize_norm(value: str) -> set[str]:
    return {token for token in normalize_identifier(value).split("_") if token}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def extract_article_references(text: str) -> set[str]:
    return {
        normalize_article_id(number)
        for number in re.findall(r"\b(?:Điều|điều)\s+(\d+)\b", safe_string(text))
    }


def build_counterfactual_query(original_query: str, intervention: Intervention) -> str:
    if intervention.intervention_type == "NEGATE":
        return (
            f"{original_query}\n"
            f"Giả sử sự kiện pháp lý sau không xảy ra: "
            f"{intervention.target_event_text}.\n"
            "Trong trường hợp đó, hậu quả pháp lý nào còn có thể đạt được?"
        )

    return (
        f"{original_query}\n"
        f"Thay sự kiện '{intervention.target_event_text}' bằng "
        f"'{intervention.replacement_event_text}'.\n"
        "Các hậu quả pháp lý thay đổi như thế nào?"
    )


# ============================================================
# RULE REPOSITORY
# ============================================================

class RuleRepository:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu: {self.data_path}")

        self.df = pd.read_json(self.data_path)
        self._validate_columns()

        self.rules: dict[str, Rule] = {}
        self.article_to_rules: dict[str, set[str]] = defaultdict(set)
        self.subject_to_rules: dict[str, set[str]] = defaultdict(set)
        self.condition_to_rules: dict[str, set[str]] = defaultdict(set)
        self.effect_to_rules: dict[str, set[str]] = defaultdict(set)
        self.event_as_condition_to_rules: dict[str, set[str]] = defaultdict(set)
        self.event_as_effect_to_rules: dict[str, set[str]] = defaultdict(set)
        self.event_texts: dict[str, list[str]] = defaultdict(list)

        self._build_repository()

    def _validate_columns(self) -> None:
        required = {
            "index", "article_id", "legal_subject", "condition", "effect",
            "condition_norm", "effect_norm", "article_title", "content",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Thiếu các cột: {sorted(missing)}")

    def _build_repository(self) -> None:
        duplicate_counts: dict[str, int] = defaultdict(int)

        for row_position, row in self.df.iterrows():
            condition_norm = normalize_identifier(row["condition_norm"])
            effect_norm = normalize_identifier(row["effect_norm"])
            if not condition_norm or not effect_norm:
                continue

            original_rule_id = safe_string(row["index"]) or str(row_position + 1)
            duplicate_counts[original_rule_id] += 1
            count = duplicate_counts[original_rule_id]
            rule_id = original_rule_id if count == 1 else f"{original_rule_id}_{count}"

            article_id = normalize_article_id(row["article_id"])
            legal_subject = safe_string(row["legal_subject"])
            condition_text = safe_string(row["condition"])
            effect_text = safe_string(row["effect"])

            rule = Rule(
                rule_id=rule_id,
                row_id=int(row_position),
                article_id=article_id,
                legal_subject=legal_subject,
                subject_norm=normalize_identifier(legal_subject),
                condition=condition_text,
                effect=effect_text,
                condition_norm=condition_norm,
                effect_norm=effect_norm,
                condition_event_id=event_id(condition_norm),
                effect_event_id=event_id(effect_norm),
                article_title=safe_string(row["article_title"]),
                content=safe_string(row["content"]),
            )

            self.rules[rule_id] = rule
            self.article_to_rules[article_id].add(rule_id)
            self.subject_to_rules[rule.subject_norm].add(rule_id)
            self.condition_to_rules[condition_norm].add(rule_id)
            self.effect_to_rules[effect_norm].add(rule_id)
            self.event_as_condition_to_rules[condition_norm].add(rule_id)
            self.event_as_effect_to_rules[effect_norm].add(rule_id)

            if condition_text and condition_text not in self.event_texts[condition_norm]:
                self.event_texts[condition_norm].append(condition_text)
            if effect_text and effect_text not in self.event_texts[effect_norm]:
                self.event_texts[effect_norm].append(effect_text)

        print(f"Loaded valid rules: {len(self.rules)}")
        print(f"Unique events: {len(set(self.event_as_condition_to_rules) | set(self.event_as_effect_to_rules))}")

    def get(self, rule_id: str) -> Rule:
        return self.rules[rule_id]

    def has(self, rule_id: str) -> bool:
        return rule_id in self.rules

    def get_event_text(self, event_norm: str) -> str:
        texts = self.event_texts.get(normalize_identifier(event_norm), [])
        return " || ".join(texts[:3]) if texts else event_norm

    def build_event_chain(self, rule_ids: list[str]) -> list[str]:
        if not rule_ids:
            return []
        rules = [self.get(rule_id) for rule_id in rule_ids]
        chain = [rules[0].condition_norm]
        chain.extend(rule.effect_norm for rule in rules)
        return chain

    def is_pure_causal_rule_path(self, rule_ids: list[str]) -> bool:
        for left_id, right_id in zip(rule_ids, rule_ids[1:]):
            if self.get(left_id).effect_norm != self.get(right_id).condition_norm:
                return False
        return True


# ============================================================
# DENSE SEARCH
# ============================================================

class DenseRuleSearch:
    def __init__(
        self,
        repository: RuleRepository,
        memory_path: str,
        index_path: str,
        model_name: str,
    ):
        self.repo = repository
        self.memory_path = Path(memory_path)
        self.index_path = Path(index_path)

        if not self.memory_path.exists():
            raise FileNotFoundError(f"Không tìm thấy memory: {self.memory_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy FAISS index: {self.index_path}")

        self.memory_df = pd.read_csv(self.memory_path)
        self.index = faiss.read_index(str(self.index_path))
        if len(self.memory_df) != self.index.ntotal:
            raise ValueError("causal_memory.csv và FAISS index không cùng số phần tử.")

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.position_to_rule_id = self._build_position_mapping()

    def _build_position_mapping(self) -> dict[int, Optional[str]]:
        mapping: dict[int, Optional[str]] = {}
        row_id_to_rule = {rule.row_id: rule_id for rule_id, rule in self.repo.rules.items()}

        for position, row in self.memory_df.iterrows():
            rule_id = safe_string(row.get("rule_id", row.get("index", "")))
            if self.repo.has(rule_id):
                mapping[int(position)] = rule_id
                continue

            source_row_id = row.get("source_row_id", row.get("row_id"))
            matched: Optional[str] = None
            if source_row_id is not None and not pd.isna(source_row_id):
                matched = row_id_to_rule.get(int(source_row_id))

            if matched is None:
                required = {"article_id", "condition_norm", "effect_norm"}
                if required.issubset(self.memory_df.columns):
                    article_id = normalize_article_id(row["article_id"])
                    condition_norm = normalize_identifier(row["condition_norm"])
                    effect_norm = normalize_identifier(row["effect_norm"])
                    for candidate_id in self.repo.condition_to_rules.get(condition_norm, set()):
                        candidate = self.repo.get(candidate_id)
                        if candidate.article_id == article_id and candidate.effect_norm == effect_norm:
                            matched = candidate_id
                            break

            mapping[int(position)] = matched

        unmatched = sum(value is None for value in mapping.values())
        if unmatched:
            print(f"Warning: {unmatched} vector chưa khớp rule.")
        return mapping

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        embedding = self.model.encode(
            [safe_string(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        search_k = min(max(top_k * 2, top_k), self.index.ntotal)
        scores, positions = self.index.search(embedding, search_k)

        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            rule_id = self.position_to_rule_id.get(int(position))
            if rule_id is None or rule_id in seen:
                continue
            seen.add(rule_id)
            results.append((rule_id, float(score)))
            if len(results) >= top_k:
                break
        return results


# ============================================================
# INTERVENTION PARSER
# ============================================================

class InterventionParser:
    @staticmethod
    def create(
        repository: RuleRepository,
        target_event: str,
        intervention_type: str,
        replacement_event: Optional[str] = None,
    ) -> Intervention:
        intervention_type = intervention_type.upper().strip()
        if intervention_type not in {"NEGATE", "REPLACE"}:
            raise ValueError("intervention-type phải là NEGATE hoặc REPLACE.")

        target_norm = normalize_identifier(target_event)
        if not target_norm:
            raise ValueError("target-event không được để trống.")
        target_text = repository.get_event_text(target_norm)

        if intervention_type == "NEGATE":
            return Intervention(
                intervention_type="NEGATE",
                target_event_norm=target_norm,
                target_event_text=target_text,
                description=f"do(NOT {target_norm})",
            )

        if not replacement_event:
            raise ValueError("REPLACE yêu cầu --replacement-event.")

        replacement_norm = normalize_identifier(replacement_event)
        replacement_text = repository.get_event_text(replacement_norm)
        return Intervention(
            intervention_type="REPLACE",
            target_event_norm=target_norm,
            target_event_text=target_text,
            replacement_event_norm=replacement_norm,
            replacement_event_text=replacement_text,
            description=f"do({target_norm} := {replacement_norm})",
        )


# ============================================================
# COUNTERFACTUAL VERIFIER
# ============================================================

class CounterfactualVerifier:
    def __init__(self, repository: RuleRepository, dense_search: DenseRuleSearch):
        self.repo = repository
        self.dense = dense_search

    @staticmethod
    def load_retrieval_result(path_value: str) -> dict[str, Any]:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy retrieval result: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_factual_paths(
        self,
        retrieval_result: dict[str, Any],
        max_paths: int,
    ) -> list[dict[str, Any]]:
        factual_paths: list[dict[str, Any]] = []

        for path_index, path in enumerate(retrieval_result.get("paths", []), start=1):
            rule_ids = [
                safe_string(rule_id)
                for rule_id in path.get("rule_ids", [])
                if self.repo.has(safe_string(rule_id))
            ]
            if not rule_ids:
                continue

            pure_causal = self.repo.is_pure_causal_rule_path(rule_ids)
            if not pure_causal:
                continue

            computed_event_chain = self.repo.build_event_chain(rule_ids)
            supplied_event_chain = [
                normalize_identifier(value)
                for value in path.get("event_chain", [])
                if normalize_identifier(value)
            ]
            event_chain = supplied_event_chain if supplied_event_chain == computed_event_chain else computed_event_chain

            factual_paths.append({
                "path_id": path_index,
                "rule_ids": rule_ids,
                "event_chain": event_chain,
                "score": float(path.get("score", 0.0)),
                "is_pure_causal": True,
            })
            if len(factual_paths) >= max_paths:
                break

        if factual_paths:
            return factual_paths

        # Fallback cho retrieval_result cũ không có path hợp lệ.
        for evidence_index, evidence in enumerate(retrieval_result.get("evidence", []), start=1):
            rule_id = safe_string(evidence.get("rule_id"))
            if self.repo.has(rule_id):
                factual_paths.append({
                    "path_id": evidence_index,
                    "rule_ids": [rule_id],
                    "event_chain": self.repo.build_event_chain([rule_id]),
                    "score": float(evidence.get("evidence_score", 0.0)),
                    "is_pure_causal": True,
                })
                if len(factual_paths) >= max_paths:
                    break

        return factual_paths

    def evaluate_path_under_intervention(
        self,
        path: dict[str, Any],
        intervention: Intervention,
    ) -> dict[str, Any]:
        rule_ids = path["rule_ids"]
        factual_events = path["event_chain"]

        target = intervention.target_event_norm
        replacement = intervention.replacement_event_norm
        counterfactual_events = list(factual_events)

        target_positions = [i for i, event in enumerate(factual_events) if event == target]
        intervention_position: Optional[int] = target_positions[0] if target_positions else None

        if intervention.intervention_type == "REPLACE" and intervention_position is not None:
            counterfactual_events[intervention_position] = replacement or target

        rule_evaluations: list[dict[str, Any]] = []
        blocked = False
        blocked_from_rule_index: Optional[int] = None
        direct_status = "TARGET_EVENT_NOT_IN_PATH"

        for index, rule_id in enumerate(rule_ids):
            rule = self.repo.get(rule_id)
            input_event = factual_events[index]
            output_event = factual_events[index + 1]

            if blocked:
                status = "BLOCKED_BY_UPSTREAM"
                reason = (
                    f"Rule {rule_id} bị chặn vì event đầu vào {input_event} "
                    "không còn reachable sau can thiệp ở bước trước."
                )
            elif intervention_position is None:
                status = "PRESERVED"
                reason = "Event bị can thiệp không xuất hiện trong causal path này."
            elif intervention_position == index:
                # Can thiệp đúng vào input event của rule hiện tại.
                if intervention.intervention_type == "NEGATE":
                    status = "INVALIDATED"
                    reason = f"Input event {input_event} của Rule {rule_id} đã bị phủ định trực tiếp."
                    blocked = True
                    blocked_from_rule_index = index
                    direct_status = "INPUT_EVENT_NEGATED"
                else:
                    if replacement == input_event:
                        status = "PRESERVED"
                        reason = "Event thay thế giống event ban đầu nên rule vẫn giữ nguyên."
                    else:
                        status = "INVALIDATED"
                        reason = (
                            f"Input event {input_event} đã được thay bằng {replacement}; "
                            f"Rule {rule_id} không còn được kích hoạt theo path factual."
                        )
                        blocked = True
                        blocked_from_rule_index = index
                        direct_status = "INPUT_EVENT_REPLACED"
            elif intervention_position == index + 1:
                # Can thiệp trực tiếp vào output event của rule hiện tại.
                if intervention.intervention_type == "NEGATE":
                    status = "OUTPUT_INTERVENED"
                    reason = (
                        f"Output event {output_event} của Rule {rule_id} bị đặt thành không xảy ra; "
                        "các rule downstream bị chặn."
                    )
                    blocked = True
                    blocked_from_rule_index = index + 1
                    direct_status = "OUTPUT_EVENT_NEGATED"
                else:
                    status = "OUTPUT_REPLACED"
                    reason = (
                        f"Output event {output_event} được thay bằng {replacement}; "
                        "causal path factual bị ngắt tại đây."
                    )
                    blocked = True
                    blocked_from_rule_index = index + 1
                    direct_status = "OUTPUT_EVENT_REPLACED"
            else:
                status = "PRESERVED"
                reason = "Rule nằm trước vị trí can thiệp và vẫn có thể tạo output factual của nó."

            rule_evaluations.append({
                "path_position": index,
                "rule": asdict(rule),
                "input_event": input_event,
                "output_event": output_event,
                "counterfactual_status": status,
                "reason": reason,
            })

        final_effect_reachable = intervention_position is None
        if intervention_position is not None:
            final_effect_reachable = False
            if intervention.intervention_type == "REPLACE" and replacement == target:
                final_effect_reachable = True

        if final_effect_reachable:
            path_status = "CAUSAL_PATH_PRESERVED"
            surviving_events = factual_events
            removed_events: list[str] = []
            surviving_rule_ids = rule_ids
            removed_rule_ids: list[str] = []
        else:
            path_status = "CAUSAL_PATH_BROKEN"
            cut_event_position = intervention_position if intervention_position is not None else len(factual_events)
            surviving_events = factual_events[:cut_event_position]
            if intervention.intervention_type == "REPLACE" and replacement:
                surviving_events = surviving_events + [replacement]
            removed_events = factual_events[cut_event_position:]

            cut_rule_position = min(cut_event_position, len(rule_ids))
            surviving_rule_ids = rule_ids[:cut_rule_position]
            removed_rule_ids = rule_ids[cut_rule_position:]

        return {
            "path_id": path["path_id"],
            "path_score": path["score"],
            "path_status": path_status,
            "direct_intervention_status": direct_status,
            "intervention_event_position": intervention_position,
            "blocked_from_rule_index": blocked_from_rule_index,
            "factual_rule_ids": rule_ids,
            "factual_event_chain": factual_events,
            "counterfactual_event_chain_prefix": surviving_events,
            "surviving_rule_ids": surviving_rule_ids,
            "removed_rule_ids": removed_rule_ids,
            "removed_events": removed_events,
            "factual_final_effect": factual_events[-1] if factual_events else "",
            "counterfactual_final_effect_reachable": final_effect_reachable,
            "rule_evaluations": rule_evaluations,
        }

    def _collect_structural_candidates(
        self,
        anchor_rule: Rule,
        intervention: Intervention,
        target_effect: str,
    ) -> dict[str, set[str]]:
        candidates: dict[str, set[str]] = defaultdict(set)

        for rule_id in self.repo.article_to_rules.get(anchor_rule.article_id, set()):
            candidates[rule_id].add("SAME_ARTICLE")
        for rule_id in self.repo.subject_to_rules.get(anchor_rule.subject_norm, set()):
            candidates[rule_id].add("SAME_SUBJECT")
        for rule_id in self.repo.effect_to_rules.get(anchor_rule.effect_norm, set()):
            candidates[rule_id].add("SAME_EFFECT")
        for rule_id in self.repo.condition_to_rules.get(anchor_rule.condition_norm, set()):
            candidates[rule_id].add("SAME_CONDITION")
        for rule_id in self.repo.effect_to_rules.get(target_effect, set()):
            candidates[rule_id].add("CAUSES_TARGET_EFFECT")

        references = extract_article_references(
            anchor_rule.condition + "\n" + anchor_rule.effect + "\n" + anchor_rule.content
        )
        for article_id in references:
            for rule_id in self.repo.article_to_rules.get(article_id, set()):
                candidates[rule_id].add("ARTICLE_REFERENCE")

        if intervention.intervention_type == "REPLACE" and intervention.replacement_event_norm:
            for rule_id in self.repo.condition_to_rules.get(intervention.replacement_event_norm, set()):
                candidates[rule_id].add("REPLACEMENT_CONDITION")

        return candidates

    def _score_alternative(
        self,
        anchor_rule: Rule,
        candidate: Rule,
        relations: set[str],
        semantic_score: Optional[float],
        intervention: Intervention,
        target_effect: str,
    ) -> float:
        score = sum(ALTERNATIVE_WEIGHTS.get(relation, 0.15) for relation in relations)
        if semantic_score is not None:
            score += ALTERNATIVE_WEIGHTS["SEMANTIC"] * max(semantic_score, 0.0)

        score += 0.10 * jaccard_similarity(tokenize_norm(anchor_rule.subject_norm), tokenize_norm(candidate.subject_norm))
        score += 0.08 * jaccard_similarity(tokenize_norm(anchor_rule.condition_norm), tokenize_norm(candidate.condition_norm))
        score += 0.07 * jaccard_similarity(tokenize_norm(target_effect), tokenize_norm(candidate.effect_norm))

        if candidate.condition_norm == intervention.target_event_norm:
            score -= 0.80
        if intervention.intervention_type == "REPLACE" and candidate.condition_norm == intervention.replacement_event_norm:
            score += 0.50
        if candidate.effect_norm == target_effect:
            score += 0.35
        if candidate.rule_id == anchor_rule.rule_id:
            score -= 1.00
        return float(score)

    def find_alternative_rules(
        self,
        original_query: str,
        anchor_rule: Rule,
        intervention: Intervention,
        target_effect: str,
        semantic_search_k: int,
        top_k: int,
    ) -> list[AlternativeRule]:
        candidate_relations = self._collect_structural_candidates(
            anchor_rule, intervention, target_effect
        )

        semantic_results = self.dense.search(
            build_counterfactual_query(original_query, intervention),
            semantic_search_k,
        )
        semantic_scores = {
            rule_id: score for rule_id, score in semantic_results if score >= MIN_SEMANTIC_SCORE
        }
        for rule_id in semantic_scores:
            candidate_relations[rule_id].add("SEMANTIC")

        alternatives: list[AlternativeRule] = []
        for candidate_id, relations in candidate_relations.items():
            if not self.repo.has(candidate_id):
                continue
            candidate = self.repo.get(candidate_id)

            # Rule vẫn đòi đúng target event bị phủ định thì không phải alternative hợp lệ.
            if (
                intervention.intervention_type == "NEGATE"
                and candidate.condition_norm == intervention.target_event_norm
            ):
                continue

            score = self._score_alternative(
                anchor_rule,
                candidate,
                relations,
                semantic_scores.get(candidate_id),
                intervention,
                target_effect,
            )
            alternatives.append(AlternativeRule(
                rule_id=candidate_id,
                score=score,
                relations=sorted(relations),
                semantic_score=semantic_scores.get(candidate_id),
                explanation=f"Ứng viên được tìm qua: {', '.join(sorted(relations))}.",
            ))

        alternatives.sort(key=lambda item: item.score, reverse=True)
        return alternatives[:top_k]

    def compare_effects(
        self,
        target_effect: str,
        alternatives: list[AlternativeRule],
    ) -> dict[str, Any]:
        same_effect: list[dict[str, Any]] = []
        different_effect: list[dict[str, Any]] = []
        contradictory: list[dict[str, Any]] = []
        target_tokens = tokenize_norm(target_effect)

        for alternative in alternatives:
            rule = self.repo.get(alternative.rule_id)
            item = {
                "rule_id": rule.rule_id,
                "article_id": rule.article_id,
                "condition": rule.condition,
                "condition_norm": rule.condition_norm,
                "effect": rule.effect,
                "effect_norm": rule.effect_norm,
                "score": alternative.score,
                "relations": alternative.relations,
            }
            if rule.effect_norm == target_effect:
                same_effect.append(item)
                continue

            tokens = tokenize_norm(rule.effect_norm)
            if tokens & CONTRADICTION_KEYWORDS and jaccard_similarity(target_tokens, tokens) > 0:
                contradictory.append(item)
            else:
                different_effect.append(item)

        return {
            "target_effect_norm": target_effect,
            "same_effect_rules": same_effect,
            "different_effect_rules": different_effect,
            "possibly_contradictory_rules": contradictory,
        }

    @staticmethod
    def infer_path_conclusion(
        path_evaluation: dict[str, Any],
        effect_comparison: dict[str, Any],
    ) -> dict[str, str]:
        if path_evaluation["counterfactual_final_effect_reachable"]:
            return {
                "label": "FACTUAL_PATH_PRESERVED",
                "conclusion": "Can thiệp không làm mất khả năng đạt tới hệ quả cuối của path này.",
            }
        if effect_comparison["same_effect_rules"]:
            return {
                "label": "POTENTIAL_ALTERNATIVE_SUPPORT_FOUND",
                "conclusion": (
                    "Factual path đã bị phá vỡ.Hệ thống tìm thấy rule khác có thể liên quan đến cùng hệ quả cuối,"
                    "nhưng chưa đủ căn cứ xác định hệ quả đó vẫn áp dụng trong tình huống phản thực."
                ),
            }
        if effect_comparison["possibly_contradictory_rules"]:
            return {
                "label": "POSSIBLE_EFFECT_REVERSAL",
                "conclusion": "Path factual bị phá vỡ và có ứng viên mang dấu hiệu hệ quả đối lập.",
            }
        if effect_comparison["different_effect_rules"]:
            return {
                "label": "ALTERNATIVE_EFFECT_FOUND",
                "conclusion": "Path factual bị phá vỡ và tìm thấy các hệ quả pháp lý thay thế.",
            }
        return {
            "label": "FINAL_EFFECT_NOT_REACHABLE",
            "conclusion": (
                "Sau can thiệp, hệ quả cuối của causal path factual không còn reachable "
                "và chưa tìm thấy căn cứ thay thế đủ rõ."
            ),
        }

    def verify(
        self,
        retrieval_result_path: str,
        intervention: Intervention,
        top_k_alternatives: int,
        semantic_search_k: int,
        max_paths: int,
    ) -> dict[str, Any]:
        retrieval_result = self.load_retrieval_result(retrieval_result_path)
        original_query = safe_string(retrieval_result.get("query"))
        factual_paths = self.get_factual_paths(retrieval_result, max_paths=max_paths)
        if not factual_paths:
            raise ValueError("Không tìm thấy causal path hợp lệ trong retrieval_result.json.")

        verification_paths: list[dict[str, Any]] = []
        for path in factual_paths:
            path_evaluation = self.evaluate_path_under_intervention(path, intervention)
            rule_ids = path["rule_ids"]

            anchor_index = path_evaluation.get("intervention_event_position")
            if anchor_index is None:
                anchor_rule = self.repo.get(rule_ids[-1])
            else:
                anchor_rule = self.repo.get(rule_ids[min(anchor_index, len(rule_ids) - 1)])

            target_effect = path_evaluation["factual_final_effect"]
            alternatives = self.find_alternative_rules(
                original_query,
                anchor_rule,
                intervention,
                target_effect,
                semantic_search_k,
                top_k_alternatives,
            )
            effect_comparison = self.compare_effects(target_effect, alternatives)
            conclusion = self.infer_path_conclusion(path_evaluation, effect_comparison)

            verification_paths.append({
                **path_evaluation,
                "alternative_rules": [
                    {
                        **asdict(alternative),
                        "rule": asdict(self.repo.get(alternative.rule_id)),
                    }
                    for alternative in alternatives
                ],
                "effect_comparison": effect_comparison,
                "verification": conclusion,
            })

        overall = self._aggregate_overall_result(verification_paths)
        return {
            "original_query": original_query,
            "intervention": asdict(intervention),
            "factual_path_count": len(factual_paths),
            "verification_paths": verification_paths,
            "overall_verification": overall,
        }

    @staticmethod
    def _aggregate_overall_result(paths: list[dict[str, Any]]) -> dict[str, Any]:
        broken = sum(path["path_status"] == "CAUSAL_PATH_BROKEN" for path in paths)
        preserved = len(paths) - broken
        final_reachable = sum(path["counterfactual_final_effect_reachable"] for path in paths)
        alternative_supported = sum(
            path["verification"]["label"] == "POTENTIAL_ALTERNATIVE_SUPPORT_FOUND"
            for path in paths
        )
        labels: dict[str, int] = defaultdict(int)
        for path in paths:
            labels[path["verification"]["label"]] += 1

        if broken == 0:
            label = "INTERVENTION_HAS_NO_PATH_EFFECT"
            conclusion = "Can thiệp không làm phá vỡ causal path factual nào được truy hồi."
        elif preserved > 0:
            label = "PARTIAL_PATH_EFFECT"
            conclusion = "Can thiệp phá vỡ một số path nhưng vẫn còn path factual khác được bảo toàn."
        elif alternative_supported > 0:
            label = "FACTUAL_PATHS_BROKEN_BUT_ALTERNATIVE_SUPPORT_EXISTS"
            conclusion = (
                "Tất cả causal path factual bị phá vỡ, nhưng có rule thay thế cùng hỗ trợ "
                "ít nhất một hệ quả cuối."
            )
        else:
            label = "FACTUAL_SUPPORT_CHANGED"
            conclusion = (
                "Can thiệp làm phá vỡ toàn bộ causal path factual; không nên giữ nguyên "
                "câu trả lời ban đầu nếu chưa có causal path thay thế hợp lệ."
            )

        return {
            "label": label,
            "conclusion": conclusion,
            "broken_path_count": broken,
            "preserved_path_count": preserved,
            "reachable_final_effect_count": final_reachable,
            "alternative_supported_path_count": alternative_supported,
            "verification_label_counts": dict(labels),
        }


# ============================================================
# OUTPUT FORMATTERS
# ============================================================

def format_counterfactual_context(
    result: dict[str, Any],
    max_paths: int = 5,
    max_alternatives: int = 4,
) -> str:
    intervention = result["intervention"]
    lines = [
        "CÂU HỎI GỐC:",
        result["original_query"],
        "",
        "CAN THIỆP PHẢN THỰC:",
        intervention["description"],
        "",
        "LƯU Ý:",
        "Phủ định một event không đồng nghĩa trực tiếp với phủ định mọi effect; phải kiểm tra khả năng reachable theo causal path.",
        "",
        "KẾT QUẢ THEO CAUSAL PATH:",
    ]

    for index, path in enumerate(result["verification_paths"][:max_paths], start=1):
        lines.extend([
            "",
            f"[Path {index}]",
            "Factual events: " + " -> ".join(path["factual_event_chain"]),
            f"Trạng thái: {path['path_status']}",
            f"Hệ quả cuối: {path['factual_final_effect']}",
            "Hệ quả cuối còn reachable: " + str(path["counterfactual_final_effect_reachable"]),
            "Prefix còn tồn tại: " + " -> ".join(path["counterfactual_event_chain_prefix"]),
            "Các event bị loại: " + (" -> ".join(path["removed_events"]) or "Không có"),
        ])

        for evaluation in path["rule_evaluations"]:
            rule = evaluation["rule"]
            lines.append(
                f"- Rule {rule['rule_id']} (Điều {rule['article_id']}): "
                f"{evaluation['input_event']} -> {evaluation['output_event']} | "
                f"{evaluation['counterfactual_status']}"
            )

        alternatives = path["alternative_rules"][:max_alternatives]
        if alternatives:
            lines.append("Rule thay thế nổi bật:")
            for alternative in alternatives:
                rule = alternative["rule"]
                lines.append(
                    f"- Rule {rule['rule_id']}: {rule['condition_norm']} -> "
                    f"{rule['effect_norm']} (score={alternative['score']:.4f})"
                )

        lines.append("Kết luận: " + path["verification"]["conclusion"])

    overall = result["overall_verification"]
    lines.extend([
        "",
        "KẾT LUẬN TỔNG THỂ:",
        overall["label"],
        overall["conclusion"],
        "",
        "YÊU CẦU SINH CÂU TRẢ LỜI:",
        "Chỉ sử dụng causal path còn hợp lệ. Không suy diễn NOT condition thành NOT effect nếu không có causal verification. Nếu factual path bị phá vỡ và chưa có path thay thế hợp lệ, hãy nói chưa đủ căn cứ giữ nguyên kết luận ban đầu.",
    ])
    return "\n".join(lines)


def print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("PATH-AWARE COUNTERFACTUAL VERIFICATION")
    print("=" * 100)
    print("\nOriginal query:")
    print(result["original_query"])
    print("\nIntervention:")
    print(result["intervention"]["description"])

    for index, path in enumerate(result["verification_paths"], start=1):
        print("\n" + "-" * 100)
        print(f"Path {index} | {path['path_status']} | score={path['path_score']:.4f}")
        print("Factual:", " -> ".join(path["factual_event_chain"]))
        print("Remaining:", " -> ".join(path["counterfactual_event_chain_prefix"]) or "(empty)")
        print("Final effect reachable:", path["counterfactual_final_effect_reachable"])
        for evaluation in path["rule_evaluations"]:
            rule = evaluation["rule"]
            print(
                f"  Rule {rule['rule_id']}: {evaluation['input_event']} -> "
                f"{evaluation['output_event']} | {evaluation['counterfactual_status']}"
            )
        print("Verification:", path["verification"]["label"])

    overall = result["overall_verification"]
    print("\nOverall:")
    print("  Label:", overall["label"])
    print("  Conclusion:", overall["conclusion"])


# ============================================================
# ARGUMENTS + MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Path-aware Counterfactual Verification cho CausalRAG luật hình sự Việt Nam."
    )
    parser.add_argument("--retrieval-result", default=RETRIEVAL_RESULT_PATH)
    parser.add_argument(
        "--target-event",
        "--target-condition",
        dest="target_event",
        required=True,
        help="EVENT norm cần can thiệp, ví dụ TAI_PHAM_NGUY_HIEM.",
    )
    parser.add_argument(
        "--intervention-type",
        choices=["NEGATE", "REPLACE"],
        default="NEGATE",
    )
    parser.add_argument(
        "--replacement-event",
        "--replacement-condition",
        dest="replacement_event",
        default=None,
    )
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--memory", default=MEMORY_PATH)
    parser.add_argument("--index", default=FAISS_INDEX_PATH)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--top-k-alternatives", type=int, default=DEFAULT_TOP_K_ALTERNATIVES)
    parser.add_argument("--semantic-search-k", type=int, default=DEFAULT_SEMANTIC_SEARCH_K)
    parser.add_argument("--max-paths", type=int, default=DEFAULT_MAX_PATHS)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--context-output", default=CONTEXT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_paths <= 0:
        raise ValueError("--max-paths phải lớn hơn 0.")

    repository = RuleRepository(args.data)
    dense_search = DenseRuleSearch(repository, args.memory, args.index, args.model)
    intervention = InterventionParser.create(
        repository,
        args.target_event,
        args.intervention_type,
        args.replacement_event,
    )

    verifier = CounterfactualVerifier(repository, dense_search)
    result = verifier.verify(
        args.retrieval_result,
        intervention,
        args.top_k_alternatives,
        args.semantic_search_k,
        args.max_paths,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    context_path = Path(args.context_output)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(format_counterfactual_context(result), encoding="utf-8")

    print_summary(result)
    print("\nSaved:")
    print(f"- JSON: {output_path}")
    print(f"- LLM context: {context_path}")


if __name__ == "__main__":
    main()