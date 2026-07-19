from __future__ import annotations

import argparse
import json
import math
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

MIN_SEMANTIC_SCORE = 0.30


# ============================================================
# SCORING WEIGHTS
# ============================================================

ALTERNATIVE_WEIGHTS = {
    "SAME_ARTICLE": 0.32,
    "SAME_SUBJECT": 0.22,
    "SAME_EFFECT": 0.15,
    "SAME_CONDITION": 0.10,
    "SEMANTIC": 0.28,
    "ARTICLE_REFERENCE": 0.30,
}

CONTRADICTION_KEYWORDS = {
    "KHONG",
    "CAM",
    "MIEN",
    "LOAI_TRU",
    "DINH_CHI",
    "HUY",
    "TU_CHOI",
}

MODALITY_KEYWORDS = {
    "CO_THE",
    "PHAI",
    "DUOC",
    "BI",
    "KHONG_DUOC",
    "BUOC",
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

    article_title: str
    content: str


@dataclass
class Intervention:
    intervention_type: str

    target_condition_norm: str
    target_condition_text: str

    replacement_condition_norm: Optional[str] = None
    replacement_condition_text: Optional[str] = None

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
    text = safe_string(text)
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def normalize_identifier(text: str) -> str:
    text = remove_vietnamese_accents(text).upper()

    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def extract_article_references(text: str) -> set[str]:
    text = safe_string(text)

    return {
        normalize_article_id(number)
        for number in re.findall(
            r"\b(?:Điều|điều)\s+(\d+)\b",
            text,
        )
    }


def tokenize_norm(norm_value: str) -> set[str]:
    return {
        token
        for token in normalize_identifier(norm_value).split("_")
        if token
    }


def jaccard_similarity(
    left_tokens: set[str],
    right_tokens: set[str],
) -> float:
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)

    return intersection / union if union else 0.0


def build_counterfactual_query(
    original_query: str,
    intervention: Intervention,
) -> str:
    if intervention.intervention_type == "NEGATE":
        return (
            f"{original_query}\n"
            f"Giả sử điều kiện sau không xảy ra: "
            f"{intervention.target_condition_text}.\n"
            f"Trong trường hợp đó hậu quả pháp lý nào còn áp dụng?"
        )

    if intervention.intervention_type == "REPLACE":
        return (
            f"{original_query}\n"
            f"Thay điều kiện "
            f"'{intervention.target_condition_text}' "
            f"bằng "
            f"'{intervention.replacement_condition_text}'.\n"
            f"Hậu quả pháp lý thay đổi như thế nào?"
        )

    raise ValueError(
        f"Intervention type không hợp lệ: "
        f"{intervention.intervention_type}"
    )


# ============================================================
# RULE REPOSITORY
# ============================================================

class RuleRepository:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu: {self.data_path}"
            )

        self.df = pd.read_json(self.data_path)

        self._validate_columns()

        self.rules: dict[str, Rule] = {}

        self.article_to_rules: dict[str, set[str]] = defaultdict(set)
        self.subject_to_rules: dict[str, set[str]] = defaultdict(set)
        self.condition_to_rules: dict[str, set[str]] = defaultdict(set)
        self.effect_to_rules: dict[str, set[str]] = defaultdict(set)

        self._build_repository()

    def _validate_columns(self) -> None:
        required_columns = {
            "index",
            "article_id",
            "legal_subject",
            "condition",
            "effect",
            "condition_norm",
            "effect_norm",
            "article_title",
            "content",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Thiếu các cột: {sorted(missing)}"
            )

    def _build_repository(self) -> None:
        duplicate_counts: dict[str, int] = defaultdict(int)

        for row_position, row in self.df.iterrows():
            condition_norm = normalize_identifier(
                row["condition_norm"]
            )
            effect_norm = normalize_identifier(
                row["effect_norm"]
            )

            if not condition_norm or not effect_norm:
                continue

            original_rule_id = safe_string(row["index"])

            if not original_rule_id:
                original_rule_id = str(row_position + 1)

            duplicate_counts[original_rule_id] += 1

            if duplicate_counts[original_rule_id] == 1:
                rule_id = original_rule_id
            else:
                rule_id = (
                    f"{original_rule_id}_"
                    f"{duplicate_counts[original_rule_id]}"
                )

            article_id = normalize_article_id(
                row["article_id"]
            )

            legal_subject = safe_string(
                row["legal_subject"]
            )

            subject_norm = normalize_identifier(
                legal_subject
            )

            rule = Rule(
                rule_id=rule_id,
                row_id=int(row_position),
                article_id=article_id,
                legal_subject=legal_subject,
                subject_norm=subject_norm,
                condition=safe_string(row["condition"]),
                effect=safe_string(row["effect"]),
                condition_norm=condition_norm,
                effect_norm=effect_norm,
                article_title=safe_string(
                    row["article_title"]
                ),
                content=safe_string(row["content"]),
            )

            self.rules[rule_id] = rule

            self.article_to_rules[article_id].add(rule_id)
            self.subject_to_rules[subject_norm].add(rule_id)
            self.condition_to_rules[condition_norm].add(rule_id)
            self.effect_to_rules[effect_norm].add(rule_id)

        print(f"Loaded valid rules: {len(self.rules)}")

    def get(self, rule_id: str) -> Rule:
        return self.rules[rule_id]

    def has(self, rule_id: str) -> bool:
        return rule_id in self.rules


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
            raise FileNotFoundError(
                f"Không tìm thấy memory: {self.memory_path}"
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index: "
                f"{self.index_path}"
            )

        self.memory_df = pd.read_csv(self.memory_path)
        self.index = faiss.read_index(str(self.index_path))

        if len(self.memory_df) != self.index.ntotal:
            raise ValueError(
                "causal_memory.csv và FAISS index "
                "không cùng số phần tử."
            )

        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.position_to_rule_id = (
            self._build_position_mapping()
        )

    def _build_position_mapping(
        self,
    ) -> dict[int, Optional[str]]:
        mapping: dict[int, Optional[str]] = {}

        for position, row in self.memory_df.iterrows():
            rule_id = safe_string(
                row.get(
                    "rule_id",
                    row.get("index", ""),
                )
            )

            if self.repo.has(rule_id):
                mapping[int(position)] = rule_id
                continue

            row_id = row.get("row_id")

            matched_rule_id: Optional[str] = None

            if row_id is not None and not pd.isna(row_id):
                row_id = int(row_id)

                for candidate_id, rule in self.repo.rules.items():
                    if rule.row_id == row_id:
                        matched_rule_id = candidate_id
                        break

            mapping[int(position)] = matched_rule_id

        unmatched = sum(
            rule_id is None
            for rule_id in mapping.values()
        )

        if unmatched:
            print(
                f"Warning: {unmatched} vector chưa khớp rule."
            )

        return mapping

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        search_k = min(
            max(top_k * 2, top_k),
            self.index.ntotal,
        )

        scores, positions = self.index.search(
            embedding,
            search_k,
        )

        results: list[tuple[str, float]] = []
        seen: set[str] = set()

        for score, position in zip(
            scores[0],
            positions[0],
        ):
            if position < 0:
                continue

            rule_id = self.position_to_rule_id.get(
                int(position)
            )

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
    """
    Phiên bản đầu tiên dùng explicit intervention.

    Người dùng truyền:
        --target-condition PHAM_TOI_LAN_DAU_IT_NGHIEM_TRONG
        --intervention-type NEGATE

    hoặc:
        --intervention-type REPLACE
        --replacement-condition TAI_PHAM_NGUY_HIEM
    """

    @staticmethod
    def create(
        repository: RuleRepository,
        target_condition: str,
        intervention_type: str,
        replacement_condition: Optional[str] = None,
    ) -> Intervention:
        intervention_type = intervention_type.upper().strip()

        target_norm = normalize_identifier(target_condition)

        if intervention_type not in {"NEGATE", "REPLACE"}:
            raise ValueError(
                "intervention-type phải là NEGATE hoặc REPLACE."
            )

        target_rules = repository.condition_to_rules.get(
            target_norm,
            set(),
        )

        if target_rules:
            example_rule = repository.get(
                next(iter(target_rules))
            )
            target_text = example_rule.condition
        else:
            target_text = target_condition

        if intervention_type == "NEGATE":
            return Intervention(
                intervention_type="NEGATE",
                target_condition_norm=target_norm,
                target_condition_text=target_text,
                description=(
                    f"Can thiệp do(NOT {target_norm})"
                ),
            )

        if not replacement_condition:
            raise ValueError(
                "REPLACE yêu cầu --replacement-condition."
            )

        replacement_norm = normalize_identifier(
            replacement_condition
        )

        replacement_rules = (
            repository.condition_to_rules.get(
                replacement_norm,
                set(),
            )
        )

        if replacement_rules:
            replacement_example = repository.get(
                next(iter(replacement_rules))
            )
            replacement_text = (
                replacement_example.condition
            )
        else:
            replacement_text = replacement_condition

        return Intervention(
            intervention_type="REPLACE",
            target_condition_norm=target_norm,
            target_condition_text=target_text,
            replacement_condition_norm=replacement_norm,
            replacement_condition_text=replacement_text,
            description=(
                f"Can thiệp do("
                f"{target_norm} := {replacement_norm})"
            ),
        )


# ============================================================
# COUNTERFACTUAL VERIFIER
# ============================================================

class CounterfactualVerifier:
    def __init__(
        self,
        repository: RuleRepository,
        dense_search: DenseRuleSearch,
    ):
        self.repo = repository
        self.dense = dense_search

    def load_retrieval_result(
        self,
        retrieval_result_path: str,
    ) -> dict[str, Any]:
        path = Path(retrieval_result_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy retrieval result: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_factual_rule_ids(
        self,
        retrieval_result: dict[str, Any],
    ) -> list[str]:
        factual_ids: list[str] = []

        for evidence in retrieval_result.get(
            "evidence",
            [],
        ):
            rule_id = safe_string(
                evidence.get("rule_id")
            )

            if (
                rule_id
                and self.repo.has(rule_id)
                and rule_id not in factual_ids
            ):
                factual_ids.append(rule_id)

        if factual_ids:
            return factual_ids

        for path in retrieval_result.get("paths", []):
            for rule_id in path.get("rule_ids", []):
                rule_id = safe_string(rule_id)

                if (
                    self.repo.has(rule_id)
                    and rule_id not in factual_ids
                ):
                    factual_ids.append(rule_id)

        return factual_ids

    def classify_rule_under_intervention(
        self,
        rule: Rule,
        intervention: Intervention,
    ) -> tuple[str, str]:
        """
        Trả về:
            INVALIDATED
            PRESERVED
            ACTIVATED
            UNCERTAIN
        """

        if (
            rule.condition_norm
            == intervention.target_condition_norm
        ):
            if intervention.intervention_type == "NEGATE":
                return (
                    "INVALIDATED",
                    (
                        "Rule bị loại vì condition của rule "
                        "chính là condition đã bị phủ định."
                    ),
                )

            if intervention.intervention_type == "REPLACE":
                return (
                    "INVALIDATED",
                    (
                        "Rule factual bị loại vì condition "
                        "đã được thay bằng condition khác."
                    ),
                )

        if intervention.intervention_type == "REPLACE":
            if (
                rule.condition_norm
                == intervention.replacement_condition_norm
            ):
                return (
                    "ACTIVATED",
                    (
                        "Rule được kích hoạt vì condition của rule "
                        "khớp condition thay thế."
                    ),
                )

        return (
            "PRESERVED",
            (
                "Condition của rule không trùng với "
                "condition bị can thiệp."
            ),
        )

    def evaluate_factual_rules(
        self,
        factual_rule_ids: list[str],
        intervention: Intervention,
    ) -> list[dict[str, Any]]:
        evaluations = []

        for rule_id in factual_rule_ids:
            rule = self.repo.get(rule_id)

            status, reason = (
                self.classify_rule_under_intervention(
                    rule,
                    intervention,
                )
            )

            evaluations.append({
                "rule": asdict(rule),
                "counterfactual_status": status,
                "reason": reason,
            })

        return evaluations

    def _collect_structural_candidates(
        self,
        factual_rule: Rule,
        intervention: Intervention,
    ) -> dict[str, set[str]]:
        candidates: dict[str, set[str]] = defaultdict(set)

        # Cùng article
        for rule_id in self.repo.article_to_rules.get(
            factual_rule.article_id,
            set(),
        ):
            candidates[rule_id].add("SAME_ARTICLE")

        # Cùng chủ thể
        for rule_id in self.repo.subject_to_rules.get(
            factual_rule.subject_norm,
            set(),
        ):
            candidates[rule_id].add("SAME_SUBJECT")

        # Cùng effect
        for rule_id in self.repo.effect_to_rules.get(
            factual_rule.effect_norm,
            set(),
        ):
            candidates[rule_id].add("SAME_EFFECT")

        # Cùng condition
        for rule_id in self.repo.condition_to_rules.get(
            factual_rule.condition_norm,
            set(),
        ):
            candidates[rule_id].add("SAME_CONDITION")

        # Rule thuộc điều luật được tham chiếu
        references = extract_article_references(
            factual_rule.condition
            + "\n"
            + factual_rule.effect
            + "\n"
            + factual_rule.content
        )

        for article_id in references:
            for rule_id in self.repo.article_to_rules.get(
                article_id,
                set(),
            ):
                candidates[rule_id].add(
                    "ARTICLE_REFERENCE"
                )

        # Condition thay thế
        if (
            intervention.intervention_type == "REPLACE"
            and intervention.replacement_condition_norm
        ):
            for rule_id in self.repo.condition_to_rules.get(
                intervention.replacement_condition_norm,
                set(),
            ):
                candidates[rule_id].add(
                    "REPLACEMENT_CONDITION"
                )

        return candidates

    def _score_alternative(
        self,
        factual_rule: Rule,
        candidate_rule: Rule,
        relations: set[str],
        semantic_score: Optional[float],
        intervention: Intervention,
    ) -> float:
        score = 0.0

        for relation in relations:
            score += ALTERNATIVE_WEIGHTS.get(
                relation,
                0.20,
            )

        if semantic_score is not None:
            score += (
                ALTERNATIVE_WEIGHTS["SEMANTIC"]
                * max(semantic_score, 0.0)
            )

        subject_similarity = jaccard_similarity(
            tokenize_norm(factual_rule.subject_norm),
            tokenize_norm(candidate_rule.subject_norm),
        )

        condition_similarity = jaccard_similarity(
            tokenize_norm(factual_rule.condition_norm),
            tokenize_norm(candidate_rule.condition_norm),
        )

        effect_similarity = jaccard_similarity(
            tokenize_norm(factual_rule.effect_norm),
            tokenize_norm(candidate_rule.effect_norm),
        )

        score += 0.10 * subject_similarity
        score += 0.08 * condition_similarity
        score += 0.07 * effect_similarity

        if (
            candidate_rule.condition_norm
            == intervention.target_condition_norm
        ):
            score -= 0.80

        if (
            intervention.intervention_type == "REPLACE"
            and candidate_rule.condition_norm
            == intervention.replacement_condition_norm
        ):
            score += 0.50

        if candidate_rule.rule_id == factual_rule.rule_id:
            score -= 1.00

        return float(score)

    def find_alternative_rules(
        self,
        original_query: str,
        factual_rule: Rule,
        intervention: Intervention,
        semantic_search_k: int,
        top_k: int,
    ) -> list[AlternativeRule]:
        candidate_relations = (
            self._collect_structural_candidates(
                factual_rule,
                intervention,
            )
        )

        counterfactual_query = build_counterfactual_query(
            original_query,
            intervention,
        )

        semantic_results = self.dense.search(
            query=counterfactual_query,
            top_k=semantic_search_k,
        )

        semantic_scores = {
            rule_id: score
            for rule_id, score in semantic_results
            if score >= MIN_SEMANTIC_SCORE
        }

        for rule_id in semantic_scores:
            candidate_relations[rule_id].add("SEMANTIC")

        alternatives: list[AlternativeRule] = []

        for candidate_id, relations in (
            candidate_relations.items()
        ):
            if not self.repo.has(candidate_id):
                continue

            candidate_rule = self.repo.get(candidate_id)

            status, _ = self.classify_rule_under_intervention(
                candidate_rule,
                intervention,
            )

            if status == "INVALIDATED":
                continue

            semantic_score = semantic_scores.get(candidate_id)

            score = self._score_alternative(
                factual_rule=factual_rule,
                candidate_rule=candidate_rule,
                relations=relations,
                semantic_score=semantic_score,
                intervention=intervention,
            )

            explanation = (
                f"Rule thay thế được tìm qua các quan hệ: "
                f"{', '.join(sorted(relations))}."
            )

            alternatives.append(
                AlternativeRule(
                    rule_id=candidate_id,
                    score=score,
                    relations=sorted(relations),
                    semantic_score=semantic_score,
                    explanation=explanation,
                )
            )

        alternatives.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return alternatives[:top_k]

    def compare_effects(
        self,
        factual_rule: Rule,
        alternative_rules: list[AlternativeRule],
    ) -> dict[str, Any]:
        factual_effect = factual_rule.effect_norm

        same_effect_rules = []
        different_effect_rules = []
        possibly_contradictory_rules = []

        factual_tokens = tokenize_norm(factual_effect)

        for alternative in alternative_rules:
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

            if rule.effect_norm == factual_effect:
                same_effect_rules.append(item)
                continue

            alternative_tokens = tokenize_norm(
                rule.effect_norm
            )

            contains_contradiction_marker = bool(
                alternative_tokens
                & CONTRADICTION_KEYWORDS
            )

            semantic_overlap = jaccard_similarity(
                factual_tokens,
                alternative_tokens,
            )

            if (
                contains_contradiction_marker
                and semantic_overlap > 0
            ):
                possibly_contradictory_rules.append(item)
            else:
                different_effect_rules.append(item)

        return {
            "factual_effect": {
                "text": factual_rule.effect,
                "norm": factual_rule.effect_norm,
            },
            "same_effect_rules": same_effect_rules,
            "different_effect_rules": different_effect_rules,
            "possibly_contradictory_rules": (
                possibly_contradictory_rules
            ),
        }

    def infer_verification_conclusion(
        self,
        factual_evaluation: dict[str, Any],
        effect_comparison: dict[str, Any],
    ) -> dict[str, str]:
        status = factual_evaluation[
            "counterfactual_status"
        ]

        same_effect = effect_comparison[
            "same_effect_rules"
        ]

        different_effect = effect_comparison[
            "different_effect_rules"
        ]

        contradictory = effect_comparison[
            "possibly_contradictory_rules"
        ]

        if status == "PRESERVED":
            return {
                "label": "FACTUAL_RULE_PRESERVED",
                "conclusion": (
                    "Can thiệp không trực tiếp làm mất hiệu lực "
                    "của factual rule này. Hệ quả factual vẫn có "
                    "thể được giữ, nhưng cần kiểm tra thêm các "
                    "điều kiện pháp lý khác."
                ),
            }

        if status == "ACTIVATED":
            return {
                "label": "REPLACEMENT_RULE_ACTIVATED",
                "conclusion": (
                    "Condition thay thế trực tiếp kích hoạt rule. "
                    "Hệ quả của rule này là ứng viên "
                    "counterfactual chính."
                ),
            }

        if contradictory:
            return {
                "label": "POSSIBLE_EFFECT_REVERSAL",
                "conclusion": (
                    "Factual rule bị loại và tồn tại rule có hệ "
                    "quả có dấu hiệu đối lập. Tuy nhiên cần kiểm "
                    "tra bằng văn bản luật hoặc LLM trước khi "
                    "kết luận đảo ngược hậu quả."
                ),
            }

        if different_effect:
            return {
                "label": "ALTERNATIVE_EFFECT_FOUND",
                "conclusion": (
                    "Factual rule bị loại và tìm thấy một hoặc "
                    "nhiều hệ quả pháp lý thay thế. Không nên "
                    "coi các hệ quả này là phủ định trực tiếp của "
                    "hệ quả factual."
                ),
            }

        if same_effect:
            return {
                "label": "EFFECT_SUPPORTED_BY_OTHER_RULES",
                "conclusion": (
                    "Factual rule bị loại nhưng cùng hệ quả vẫn "
                    "được hỗ trợ bởi rule khác. Vì vậy việc phủ "
                    "định condition hiện tại chưa đủ để phủ định "
                    "hệ quả pháp lý."
                ),
            }

        return {
            "label": "INSUFFICIENT_COUNTERFACTUAL_EVIDENCE",
            "conclusion": (
                "Factual rule không còn áp dụng sau can thiệp, "
                "nhưng chưa tìm thấy rule đủ mạnh để xác định hệ "
                "quả thay thế. Kết luận an toàn là không còn đủ "
                "căn cứ áp dụng hệ quả factual theo rule này."
            ),
        }

    def verify(
        self,
        retrieval_result_path: str,
        intervention: Intervention,
        top_k_alternatives: int,
        semantic_search_k: int,
    ) -> dict[str, Any]:
        retrieval_result = self.load_retrieval_result(
            retrieval_result_path
        )

        original_query = retrieval_result.get(
            "query",
            "",
        )

        factual_rule_ids = self.get_factual_rule_ids(
            retrieval_result
        )

        if not factual_rule_ids:
            raise ValueError(
                "Không tìm thấy factual rules trong "
                "retrieval_result.json."
            )

        factual_evaluations = (
            self.evaluate_factual_rules(
                factual_rule_ids,
                intervention,
            )
        )

        verification_items = []

        for evaluation in factual_evaluations:
            factual_rule_data = evaluation["rule"]
            factual_rule = self.repo.get(
                factual_rule_data["rule_id"]
            )

            alternatives = self.find_alternative_rules(
                original_query=original_query,
                factual_rule=factual_rule,
                intervention=intervention,
                semantic_search_k=semantic_search_k,
                top_k=top_k_alternatives,
            )

            effect_comparison = self.compare_effects(
                factual_rule=factual_rule,
                alternative_rules=alternatives,
            )

            conclusion = (
                self.infer_verification_conclusion(
                    factual_evaluation=evaluation,
                    effect_comparison=effect_comparison,
                )
            )

            verification_items.append({
                "factual_rule": asdict(factual_rule),
                "counterfactual_status": evaluation[
                    "counterfactual_status"
                ],
                "status_reason": evaluation["reason"],
                "alternative_rules": [
                    {
                        **asdict(alternative),
                        "rule": asdict(
                            self.repo.get(
                                alternative.rule_id
                            )
                        ),
                    }
                    for alternative in alternatives
                ],
                "effect_comparison": effect_comparison,
                "verification": conclusion,
            })

        overall = self._aggregate_overall_result(
            verification_items
        )

        return {
            "original_query": original_query,
            "intervention": asdict(intervention),
            "factual_rule_count": len(
                factual_rule_ids
            ),
            "verification_items": verification_items,
            "overall_verification": overall,
        }

    @staticmethod
    def _aggregate_overall_result(
        verification_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        label_counts: dict[str, int] = defaultdict(int)

        invalidated_count = 0
        preserved_count = 0
        activated_count = 0

        for item in verification_items:
            status = item["counterfactual_status"]

            if status == "INVALIDATED":
                invalidated_count += 1
            elif status == "PRESERVED":
                preserved_count += 1
            elif status == "ACTIVATED":
                activated_count += 1

            label = item["verification"]["label"]
            label_counts[label] += 1

        if invalidated_count == 0:
            overall_label = "INTERVENTION_HAS_NO_DIRECT_EFFECT"
            overall_conclusion = (
                "Không factual rule nào có condition trùng với "
                "condition bị can thiệp. Can thiệp hiện tại chưa "
                "tác động trực tiếp đến bằng chứng được truy hồi."
            )

        elif invalidated_count > 0 and preserved_count > 0:
            overall_label = "PARTIAL_EFFECT"
            overall_conclusion = (
                "Can thiệp làm mất hiệu lực một phần bằng chứng, "
                "nhưng một số factual rule vẫn được giữ. Câu trả "
                "lời counterfactual cần dựa trên các rule còn hợp lệ."
            )

        elif invalidated_count > 0:
            overall_label = "FACTUAL_SUPPORT_CHANGED"
            overall_conclusion = (
                "Can thiệp làm thay đổi trực tiếp tập bằng chứng "
                "factual. Không được giữ nguyên câu trả lời ban đầu "
                "nếu chưa kiểm tra các rule thay thế."
            )

        else:
            overall_label = "UNCERTAIN"
            overall_conclusion = (
                "Chưa đủ thông tin để đánh giá tác động của "
                "can thiệp."
            )

        return {
            "label": overall_label,
            "conclusion": overall_conclusion,
            "invalidated_rule_count": invalidated_count,
            "preserved_rule_count": preserved_count,
            "activated_rule_count": activated_count,
            "verification_label_counts": dict(
                label_counts
            ),
        }


# ============================================================
# LLM CONTEXT
# ============================================================

def format_counterfactual_context(
    result: dict[str, Any],
    max_items: int = 5,
    max_alternatives: int = 5,
) -> str:
    intervention = result["intervention"]

    lines = [
        "CÂU HỎI GỐC:",
        result["original_query"],
        "",
        "CAN THIỆP PHẢN THỰC:",
        intervention["description"],
        "",
        (
            "Lưu ý: phủ định condition không đồng nghĩa "
            "với phủ định effect."
        ),
        "",
        "KẾT QUẢ KIỂM CHỨNG:",
    ]

    for index, item in enumerate(
        result["verification_items"][:max_items],
        start=1,
    ):
        factual = item["factual_rule"]

        lines.extend([
            "",
            f"[Factual rule {index}]",
            (
                f"Rule {factual['rule_id']} - "
                f"Điều {factual['article_id']}"
            ),
            f"Điều kiện: {factual['condition']}",
            f"Hệ quả: {factual['effect']}",
            (
                "Trạng thái sau can thiệp: "
                f"{item['counterfactual_status']}"
            ),
            f"Lý do: {item['status_reason']}",
            (
                "Kết luận kiểm chứng: "
                f"{item['verification']['conclusion']}"
            ),
        ])

        alternatives = item["alternative_rules"][
            :max_alternatives
        ]

        if alternatives:
            lines.append("Các rule thay thế:")

        for alternative in alternatives:
            rule = alternative["rule"]

            lines.extend([
                (
                    f"- Rule {rule['rule_id']} - "
                    f"Điều {rule['article_id']}"
                ),
                f"  Condition: {rule['condition']}",
                f"  Effect: {rule['effect']}",
                (
                    f"  Score: "
                    f"{alternative['score']:.4f}"
                ),
                (
                    f"  Relations: "
                    f"{', '.join(alternative['relations'])}"
                ),
            ])

    overall = result["overall_verification"]

    lines.extend([
        "",
        "KẾT LUẬN TỔNG THỂ:",
        overall["label"],
        overall["conclusion"],
        "",
        "YÊU CẦU SINH CÂU TRẢ LỜI:",
        (
            "Chỉ sử dụng các rule còn hợp lệ sau can thiệp. "
            "Không suy diễn rằng NOT condition dẫn trực tiếp "
            "đến NOT effect. Nếu không có rule thay thế rõ ràng, "
            "hãy nói rằng chưa đủ căn cứ xác định hậu quả pháp lý."
        ),
    ])

    return "\n".join(lines)


# ============================================================
# DISPLAY
# ============================================================

def print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL VERIFICATION")
    print("=" * 80)

    print("\nOriginal query:")
    print(result["original_query"])

    print("\nIntervention:")
    print(result["intervention"]["description"])

    print("\nVerification items:")

    for index, item in enumerate(
        result["verification_items"],
        start=1,
    ):
        factual = item["factual_rule"]

        print(
            f"\n{index}. Rule {factual['rule_id']} "
            f"- Điều {factual['article_id']}"
        )

        print(
            f"   Condition: "
            f"{factual['condition_norm']}"
        )

        print(
            f"   Effect: "
            f"{factual['effect_norm']}"
        )

        print(
            f"   Status: "
            f"{item['counterfactual_status']}"
        )

        print(
            f"   Verification: "
            f"{item['verification']['label']}"
        )

        print(
            f"   Alternatives: "
            f"{len(item['alternative_rules'])}"
        )

        for alternative in (
            item["alternative_rules"][:3]
        ):
            rule = alternative["rule"]

            print(
                f"      - Rule {rule['rule_id']}: "
                f"{rule['condition_norm']} "
                f"-> {rule['effect_norm']} "
                f"(score={alternative['score']:.4f})"
            )

    overall = result["overall_verification"]

    print("\nOverall:")
    print("  Label:", overall["label"])
    print("  Conclusion:", overall["conclusion"])


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual Verification cho "
            "CausalRAG luật hình sự Việt Nam."
        )
    )

    parser.add_argument(
        "--retrieval-result",
        type=str,
        default=RETRIEVAL_RESULT_PATH,
    )

    parser.add_argument(
        "--target-condition",
        type=str,
        required=True,
        help=(
            "condition_norm cần can thiệp, ví dụ "
            "PHAM_TOI_LAN_DAU_IT_NGHIEM_TRONG"
        ),
    )

    parser.add_argument(
        "--intervention-type",
        type=str,
        choices=["NEGATE", "REPLACE"],
        default="NEGATE",
    )

    parser.add_argument(
        "--replacement-condition",
        type=str,
        default=None,
        help=(
            "Condition thay thế khi dùng "
            "--intervention-type REPLACE."
        ),
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
    )

    parser.add_argument(
        "--memory",
        type=str,
        default=MEMORY_PATH,
    )

    parser.add_argument(
        "--index",
        type=str,
        default=FAISS_INDEX_PATH,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--top-k-alternatives",
        type=int,
        default=DEFAULT_TOP_K_ALTERNATIVES,
    )

    parser.add_argument(
        "--semantic-search-k",
        type=int,
        default=DEFAULT_SEMANTIC_SEARCH_K,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
    )

    parser.add_argument(
        "--context-output",
        type=str,
        default=CONTEXT_OUTPUT_PATH,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    repository = RuleRepository(
        data_path=args.data
    )

    dense_search = DenseRuleSearch(
        repository=repository,
        memory_path=args.memory,
        index_path=args.index,
        model_name=args.model,
    )

    intervention = InterventionParser.create(
        repository=repository,
        target_condition=args.target_condition,
        intervention_type=args.intervention_type,
        replacement_condition=(
            args.replacement_condition
        ),
    )

    verifier = CounterfactualVerifier(
        repository=repository,
        dense_search=dense_search,
    )

    result = verifier.verify(
        retrieval_result_path=args.retrieval_result,
        intervention=intervention,
        top_k_alternatives=args.top_k_alternatives,
        semantic_search_k=args.semantic_search_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    context = format_counterfactual_context(
        result
    )

    context_output_path = Path(
        args.context_output
    )

    context_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_output_path.write_text(
        context,
        encoding="utf-8",
    )

    print_summary(result)

    print("\nSaved:")
    print(f"- JSON: {output_path}")
    print(f"- LLM context: {context_output_path}")


if __name__ == "__main__":
    main()