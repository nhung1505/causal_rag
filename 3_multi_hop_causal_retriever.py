from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
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

FAISS_INDEX_PATH = "data/causal_memory.index"

# File CSV này phải giữ đúng thứ tự các dòng đã dùng khi tạo FAISS.
MEMORY_PATH = "data/causal_memory.csv"

EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_SEED_TOP_K = 8
DEFAULT_SEMANTIC_POOL_SIZE = 50

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_EXPANSIONS_PER_RULE = 15
DEFAULT_FINAL_TOP_K = 12

OUTPUT_PATH = "data/retrieval_result.json"


# ============================================================
# RELATION WEIGHTS
# ============================================================

RELATION_WEIGHTS = {
    # Quan hệ causal bridge quan trọng nhất:
    # effect của rule hiện tại trở thành condition của rule tiếp theo.
    "EFFECT_TO_CONDITION": 0.42,

    # Hai rule tạo cùng hậu quả pháp lý.
    "SAME_EFFECT": 0.30,

    # Hai rule dùng cùng điều kiện chuẩn hóa.
    "SAME_CONDITION": 0.26,

    # Cùng điều luật thường có tính bổ sung rất cao.
    "SAME_ARTICLE": 0.24,

    # Cùng nhóm chủ thể pháp lý.
    "SAME_SUBJECT": 0.18,

    # Điều luật hiện tại tham chiếu trực tiếp đến điều luật khác.
    "ARTICLE_REFERENCE": 0.34,

    # Quan hệ semantic từ dense retrieval.
    "SEMANTIC": 0.20,
}

DEPTH_PENALTY = 0.08

# Giảm điểm khi các rule trong path lặp lại cùng condition/effect.
REDUNDANCY_PENALTY = 0.05


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
class PathStep:
    from_rule_id: str
    to_rule_id: str
    relation: str
    relation_score: float
    explanation: str


@dataclass
class RetrievalPath:
    seed_rule_id: str
    rule_ids: list[str]
    steps: list[PathStep]

    seed_similarity: float
    score: float


# ============================================================
# TEXT HELPERS
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


def normalize_norm_field(value: Any) -> str:
    """
    Chuẩn hóa condition_norm/effect_norm để so khớp ổn định.

    Ví dụ:
        "CHIU_TRACH_NHIEM_HINH_SU"
        " chiu_trach_nhiem_hinh_su "
    cùng trở thành:
        CHIU_TRACH_NHIEM_HINH_SU
    """
    return normalize_identifier(safe_string(value))


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)

    # Pandas đôi khi đọc 2 thành 2.0
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def extract_article_references(text: str) -> set[str]:
    """
    Trích xuất các tham chiếu như:
        Điều 76
        Điều 123 của Bộ luật này
        các điều 123, 124 và 125

    Hàm ưu tiên độ chính xác, không cố bắt mọi cấu trúc pháp lý.
    """
    text = safe_string(text)

    references: set[str] = set()

    patterns = [
        r"\bĐiều\s+(\d+)\b",
        r"\bđiều\s+(\d+)\b",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text):
            references.add(normalize_article_id(match))

    return references


def build_embedding_text(rule: Rule) -> str:
    """
    Phải gần với định dạng đã dùng lúc xây FAISS index.
    """
    return (
        f"Legal Subject:\n{rule.legal_subject}\n\n"
        f"Condition:\n{rule.condition}\n\n"
        f"Effect:\n{rule.effect}\n\n"
        f"Article:\n"
        f"Điều {rule.article_id}. {rule.article_title}\n\n"
        f"Normalized condition: {rule.condition_norm}\n"
        f"Normalized effect: {rule.effect_norm}"
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

        # Cho causal chaining:
        # effect_norm của rule A == condition_norm của rule B.
        self.norm_as_condition_to_rules: dict[str, set[str]] = defaultdict(set)
        self.norm_as_effect_to_rules: dict[str, set[str]] = defaultdict(set)

        self.article_reference_to_rules: dict[str, set[str]] = defaultdict(set)

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
                f"File dữ liệu thiếu các cột: {sorted(missing)}"
            )

    def _build_repository(self) -> None:
        duplicate_rule_ids: dict[str, int] = defaultdict(int)

        for row_position, row in self.df.iterrows():
            original_rule_id = safe_string(row["index"])

            if not original_rule_id:
                original_rule_id = str(row_position + 1)

            duplicate_rule_ids[original_rule_id] += 1

            if duplicate_rule_ids[original_rule_id] == 1:
                rule_id = original_rule_id
            else:
                rule_id = (
                    f"{original_rule_id}_"
                    f"{duplicate_rule_ids[original_rule_id]}"
                )

            article_id = normalize_article_id(row["article_id"])

            legal_subject = safe_string(row["legal_subject"])
            subject_norm = normalize_identifier(legal_subject)

            condition_norm = normalize_norm_field(
                row["condition_norm"]
            )
            effect_norm = normalize_norm_field(
                row["effect_norm"]
            )

            if not condition_norm or not effect_norm:
                continue

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
                article_title=safe_string(row["article_title"]),
                content=safe_string(row["content"]),
            )

            self.rules[rule_id] = rule

            self.article_to_rules[article_id].add(rule_id)
            self.subject_to_rules[subject_norm].add(rule_id)
            self.condition_to_rules[condition_norm].add(rule_id)
            self.effect_to_rules[effect_norm].add(rule_id)

            self.norm_as_condition_to_rules[condition_norm].add(
                rule_id
            )
            self.norm_as_effect_to_rules[effect_norm].add(
                rule_id
            )

            references = extract_article_references(
                rule.condition + "\n" + rule.effect
            )

            references.discard(article_id)

            for referenced_article_id in references:
                self.article_reference_to_rules[
                    referenced_article_id
                ].add(rule_id)

        print(f"Loaded rules: {len(self.rules)}")
        print(
            "Unique articles:",
            len(self.article_to_rules),
        )
        print(
            "Unique subjects:",
            len(self.subject_to_rules),
        )
        print(
            "Unique conditions:",
            len(self.condition_to_rules),
        )
        print(
            "Unique effects:",
            len(self.effect_to_rules),
        )

    def get(self, rule_id: str) -> Rule:
        return self.rules[rule_id]

    def has(self, rule_id: str) -> bool:
        return rule_id in self.rules


# ============================================================
# DENSE RETRIEVER
# ============================================================

class DenseRuleRetriever:
    def __init__(
        self,
        repository: RuleRepository,
        index_path: str,
        memory_path: str,
        model_name: str,
    ):
        self.repository = repository

        self.index_path = Path(index_path)
        self.memory_path = Path(memory_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index: {self.index_path}"
            )

        if not self.memory_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy memory file: {self.memory_path}"
            )

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print(f"Loading FAISS index: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))

        self.memory_df = pd.read_csv(self.memory_path)

        if len(self.memory_df) != self.index.ntotal:
            raise ValueError(
                "Số dòng causal_memory.csv không khớp FAISS index: "
                f"memory={len(self.memory_df)}, "
                f"index={self.index.ntotal}"
            )

        self.memory_position_to_rule_id = (
            self._align_memory_with_repository()
        )

    def _align_memory_with_repository(
        self,
    ) -> dict[int, Optional[str]]:
        """
        Liên kết từng vector trong FAISS với rule_id trong repository.

        Ưu tiên:
        1. cột index/rule_id;
        2. article + condition_norm + effect_norm;
        3. vị trí dòng nếu memory và JSON cùng thứ tự.
        """
        mapping: dict[int, Optional[str]] = {}

        memory_columns = set(self.memory_df.columns)

        rule_key_to_ids: dict[
            tuple[str, str, str], list[str]
        ] = defaultdict(list)

        for rule_id, rule in self.repository.rules.items():
            key = (
                rule.article_id,
                rule.condition_norm,
                rule.effect_norm,
            )
            rule_key_to_ids[key].append(rule_id)

        used_rule_ids: set[str] = set()

        for position, row in self.memory_df.iterrows():
            matched_rule_id: Optional[str] = None

            # ------------------------------------------------
            # Cách 1: khớp trực tiếp bằng index/rule_id
            # ------------------------------------------------

            for candidate_column in [
                "rule_id",
                "index",
                "id",
            ]:
                if candidate_column not in memory_columns:
                    continue

                candidate_id = safe_string(row[candidate_column])

                if self.repository.has(candidate_id):
                    matched_rule_id = candidate_id
                    break

            # ------------------------------------------------
            # Cách 2: khớp bằng bộ ba causal
            # ------------------------------------------------

            if matched_rule_id is None:
                required = {
                    "article_id",
                    "condition_norm",
                    "effect_norm",
                }

                if required.issubset(memory_columns):
                    key = (
                        normalize_article_id(row["article_id"]),
                        normalize_norm_field(
                            row["condition_norm"]
                        ),
                        normalize_norm_field(
                            row["effect_norm"]
                        ),
                    )

                    possible_ids = rule_key_to_ids.get(key, [])

                    for possible_id in possible_ids:
                        if possible_id not in used_rule_ids:
                            matched_rule_id = possible_id
                            break

                    if matched_rule_id is None and possible_ids:
                        matched_rule_id = possible_ids[0]

            # ------------------------------------------------
            # Cách 3: dựa vào thứ tự dòng
            # ------------------------------------------------

            if matched_rule_id is None:
                for rule_id, rule in self.repository.rules.items():
                    if rule.row_id == position:
                        matched_rule_id = rule_id
                        break

            mapping[int(position)] = matched_rule_id

            if matched_rule_id is not None:
                used_rule_ids.add(matched_rule_id)

        unmatched = sum(
            rule_id is None
            for rule_id in mapping.values()
        )

        if unmatched:
            print(
                f"Warning: {unmatched} vectors không khớp được rule."
            )

        return mapping

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        query = safe_string(query)

        if not query:
            raise ValueError("Query không được để trống.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        search_k = min(
            max(top_k * 3, top_k),
            self.index.ntotal,
        )

        scores, positions = self.index.search(
            query_embedding,
            search_k,
        )

        results: list[tuple[str, float]] = []
        seen_rule_ids: set[str] = set()

        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue

            rule_id = self.memory_position_to_rule_id.get(
                int(position)
            )

            if rule_id is None:
                continue

            if rule_id in seen_rule_ids:
                continue

            seen_rule_ids.add(rule_id)

            # Với IndexFlatIP + normalized embeddings,
            # score thường là cosine similarity.
            normalized_score = float(score)

            results.append((rule_id, normalized_score))

            if len(results) >= top_k:
                break

        return results


# ============================================================
# CAUSAL GRAPH EXPANSION
# ============================================================

class MultiHopCausalRetriever:
    def __init__(
        self,
        repository: RuleRepository,
        dense_retriever: DenseRuleRetriever,
    ):
        self.repo = repository
        self.dense = dense_retriever

    def _relation_explanation(
        self,
        source: Rule,
        target: Rule,
        relation: str,
    ) -> str:
        if relation == "EFFECT_TO_CONDITION":
            return (
                f"Hậu quả {source.effect_norm} của Rule "
                f"{source.rule_id} trở thành điều kiện "
                f"{target.condition_norm} của Rule "
                f"{target.rule_id}."
            )

        if relation == "SAME_EFFECT":
            return (
                f"Hai rule cùng dẫn đến hậu quả pháp lý "
                f"{source.effect_norm}."
            )

        if relation == "SAME_CONDITION":
            return (
                f"Hai rule có cùng điều kiện chuẩn hóa "
                f"{source.condition_norm}."
            )

        if relation == "SAME_ARTICLE":
            return (
                f"Hai rule cùng thuộc Điều "
                f"{source.article_id}."
            )

        if relation == "SAME_SUBJECT":
            return (
                f"Hai rule cùng áp dụng cho chủ thể "
                f"{source.legal_subject}."
            )

        if relation == "ARTICLE_REFERENCE":
            return (
                f"Rule {source.rule_id} có tham chiếu đến "
                f"Điều {target.article_id}."
            )

        if relation == "SEMANTIC":
            return "Hai rule có nội dung gần nhau về ngữ nghĩa."

        return relation

    def _add_candidates(
        self,
        candidate_relations: dict[
            str, tuple[str, float, str]
        ],
        source: Rule,
        candidate_ids: set[str],
        relation: str,
    ) -> None:
        relation_score = RELATION_WEIGHTS[relation]

        for candidate_id in candidate_ids:
            if candidate_id == source.rule_id:
                continue

            if not self.repo.has(candidate_id):
                continue

            target = self.repo.get(candidate_id)

            explanation = self._relation_explanation(
                source,
                target,
                relation,
            )

            current = candidate_relations.get(candidate_id)

            # Nếu cùng candidate có nhiều quan hệ,
            # giữ quan hệ có trọng số lớn hơn.
            if current is None or relation_score > current[1]:
                candidate_relations[candidate_id] = (
                    relation,
                    relation_score,
                    explanation,
                )

    def expand_rule(
        self,
        source_rule_id: str,
        semantic_rule_ids: Optional[set[str]] = None,
        max_candidates: int = DEFAULT_MAX_EXPANSIONS_PER_RULE,
    ) -> list[tuple[str, str, float, str]]:
        source = self.repo.get(source_rule_id)

        candidate_relations: dict[
            str, tuple[str, float, str]
        ] = {}

        # ----------------------------------------------------
        # 1. Causal bridge:
        # effect của source trùng condition của target.
        # ----------------------------------------------------

        causal_targets = (
            self.repo.norm_as_condition_to_rules.get(
                source.effect_norm,
                set(),
            )
        )

        self._add_candidates(
            candidate_relations,
            source,
            causal_targets,
            "EFFECT_TO_CONDITION",
        )

        # ----------------------------------------------------
        # 2. Các rule có cùng effect.
        # ----------------------------------------------------

        same_effect = self.repo.effect_to_rules.get(
            source.effect_norm,
            set(),
        )

        self._add_candidates(
            candidate_relations,
            source,
            same_effect,
            "SAME_EFFECT",
        )

        # ----------------------------------------------------
        # 3. Các rule có cùng condition.
        # ----------------------------------------------------

        same_condition = self.repo.condition_to_rules.get(
            source.condition_norm,
            set(),
        )

        self._add_candidates(
            candidate_relations,
            source,
            same_condition,
            "SAME_CONDITION",
        )

        # ----------------------------------------------------
        # 4. Các rule cùng article.
        # ----------------------------------------------------

        same_article = self.repo.article_to_rules.get(
            source.article_id,
            set(),
        )

        self._add_candidates(
            candidate_relations,
            source,
            same_article,
            "SAME_ARTICLE",
        )

        # ----------------------------------------------------
        # 5. Các rule cùng subject.
        # ----------------------------------------------------

        same_subject = self.repo.subject_to_rules.get(
            source.subject_norm,
            set(),
        )

        self._add_candidates(
            candidate_relations,
            source,
            same_subject,
            "SAME_SUBJECT",
        )

        # ----------------------------------------------------
        # 6. Điều luật được source tham chiếu.
        # ----------------------------------------------------

        referenced_articles = extract_article_references(
            source.condition + "\n" + source.effect
        )

        for article_id in referenced_articles:
            target_ids = self.repo.article_to_rules.get(
                article_id,
                set(),
            )

            self._add_candidates(
                candidate_relations,
                source,
                target_ids,
                "ARTICLE_REFERENCE",
            )

        # ----------------------------------------------------
        # 7. Các rule semantic gần query.
        # Chỉ thêm các seed/global semantic candidates đã lấy.
        # ----------------------------------------------------

        if semantic_rule_ids:
            self._add_candidates(
                candidate_relations,
                source,
                semantic_rule_ids,
                "SEMANTIC",
            )

        sorted_candidates = sorted(
            (
                (
                    candidate_id,
                    relation,
                    relation_score,
                    explanation,
                )
                for candidate_id, (
                    relation,
                    relation_score,
                    explanation,
                ) in candidate_relations.items()
            ),
            key=lambda item: item[2],
            reverse=True,
        )

        return sorted_candidates[:max_candidates]

    @staticmethod
    def _compute_path_score(
        seed_similarity: float,
        rule_ids: list[str],
        steps: list[PathStep],
        repository: RuleRepository,
    ) -> float:
        score = seed_similarity

        for depth, step in enumerate(steps, start=1):
            depth_discount = 1.0 / math.sqrt(depth)

            score += (
                step.relation_score
                * depth_discount
            )

            score -= DEPTH_PENALTY * depth

        rules = [
            repository.get(rule_id)
            for rule_id in rule_ids
        ]

        condition_norms = [
            rule.condition_norm
            for rule in rules
        ]

        effect_norms = [
            rule.effect_norm
            for rule in rules
        ]

        repeated_conditions = (
            len(condition_norms)
            - len(set(condition_norms))
        )

        repeated_effects = (
            len(effect_norms)
            - len(set(effect_norms))
        )

        score -= REDUNDANCY_PENALTY * (
            repeated_conditions + repeated_effects
        )

        # Thưởng nhẹ nếu path đi qua nhiều article,
        # vì có khả năng thực sự là multi-hop liên điều luật.
        distinct_articles = len({
            rule.article_id
            for rule in rules
        })

        if distinct_articles > 1:
            score += min(
                0.05 * (distinct_articles - 1),
                0.15,
            )

        return float(score)

    def retrieve(
        self,
        query: str,
        seed_top_k: int = DEFAULT_SEED_TOP_K,
        semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_expansions_per_rule: int = (
            DEFAULT_MAX_EXPANSIONS_PER_RULE
        ),
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> dict[str, Any]:
        # ----------------------------------------------------
        # Dense retrieval pool
        # ----------------------------------------------------

        semantic_results = self.dense.retrieve(
            query=query,
            top_k=semantic_pool_size,
        )

        semantic_scores = {
            rule_id: score
            for rule_id, score in semantic_results
        }

        semantic_rule_ids = set(semantic_scores)

        seeds = semantic_results[:seed_top_k]

        if not seeds:
            return {
                "query": query,
                "paths": [],
                "evidence": [],
            }

        all_paths: list[RetrievalPath] = []

        # ----------------------------------------------------
        # BFS từ mỗi seed rule
        # ----------------------------------------------------

        for seed_rule_id, seed_similarity in seeds:
            initial_path = RetrievalPath(
                seed_rule_id=seed_rule_id,
                rule_ids=[seed_rule_id],
                steps=[],
                seed_similarity=seed_similarity,
                score=seed_similarity,
            )

            all_paths.append(initial_path)

            queue = deque([initial_path])

            while queue:
                current_path = queue.popleft()

                current_depth = len(current_path.steps)

                if current_depth >= max_depth:
                    continue

                current_rule_id = current_path.rule_ids[-1]

                expansions = self.expand_rule(
                    source_rule_id=current_rule_id,
                    semantic_rule_ids=semantic_rule_ids,
                    max_candidates=max_expansions_per_rule,
                )

                for (
                    target_rule_id,
                    relation,
                    relation_score,
                    explanation,
                ) in expansions:
                    # Tránh cycle ở path level.
                    if target_rule_id in current_path.rule_ids:
                        continue

                    new_rule_ids = (
                        current_path.rule_ids
                        + [target_rule_id]
                    )

                    new_step = PathStep(
                        from_rule_id=current_rule_id,
                        to_rule_id=target_rule_id,
                        relation=relation,
                        relation_score=relation_score,
                        explanation=explanation,
                    )

                    new_steps = (
                        current_path.steps
                        + [new_step]
                    )

                    new_score = self._compute_path_score(
                        seed_similarity=seed_similarity,
                        rule_ids=new_rule_ids,
                        steps=new_steps,
                        repository=self.repo,
                    )

                    new_path = RetrievalPath(
                        seed_rule_id=seed_rule_id,
                        rule_ids=new_rule_ids,
                        steps=new_steps,
                        seed_similarity=seed_similarity,
                        score=new_score,
                    )

                    all_paths.append(new_path)
                    queue.append(new_path)

        # ----------------------------------------------------
        # Loại path trùng
        # ----------------------------------------------------

        unique_paths: dict[
            tuple[str, ...], RetrievalPath
        ] = {}

        for path in all_paths:
            key = tuple(path.rule_ids)

            current = unique_paths.get(key)

            if current is None or path.score > current.score:
                unique_paths[key] = path

        ranked_paths = sorted(
            unique_paths.values(),
            key=lambda path: (
                path.score,
                len(path.rule_ids),
            ),
            reverse=True,
        )

        selected_paths = ranked_paths[:final_top_k]

        # ----------------------------------------------------
        # Gom evidence rule từ các path tốt nhất
        # ----------------------------------------------------

        evidence_scores: dict[str, float] = defaultdict(float)
        evidence_path_count: dict[str, int] = defaultdict(int)

        for path in selected_paths:
            path_contribution = path.score / max(
                len(path.rule_ids),
                1,
            )

            for rule_id in path.rule_ids:
                evidence_scores[rule_id] = max(
                    evidence_scores[rule_id],
                    path_contribution,
                )

                evidence_path_count[rule_id] += 1

        ranked_evidence_ids = sorted(
            evidence_scores,
            key=lambda rule_id: (
                evidence_scores[rule_id],
                evidence_path_count[rule_id],
                semantic_scores.get(rule_id, -1.0),
            ),
            reverse=True,
        )

        # Giữ evidence gọn để không làm tràn context LLM.
        ranked_evidence_ids = ranked_evidence_ids[
            :final_top_k
        ]

        return {
            "query": query,
            "config": {
                "seed_top_k": seed_top_k,
                "semantic_pool_size": semantic_pool_size,
                "max_depth": max_depth,
                "max_expansions_per_rule": (
                    max_expansions_per_rule
                ),
                "final_top_k": final_top_k,
            },
            "seed_rules": [
                self._serialize_rule(
                    rule_id,
                    semantic_score=score,
                )
                for rule_id, score in seeds
            ],
            "paths": [
                self._serialize_path(path)
                for path in selected_paths
            ],
            "evidence": [
                {
                    **self._serialize_rule(
                        rule_id,
                        semantic_score=semantic_scores.get(
                            rule_id
                        ),
                    ),
                    "evidence_score": evidence_scores[
                        rule_id
                    ],
                    "path_count": evidence_path_count[
                        rule_id
                    ],
                }
                for rule_id in ranked_evidence_ids
            ],
        }

    def _serialize_rule(
        self,
        rule_id: str,
        semantic_score: Optional[float] = None,
    ) -> dict[str, Any]:
        rule = self.repo.get(rule_id)

        result = asdict(rule)

        if semantic_score is not None:
            result["semantic_score"] = float(
                semantic_score
            )

        return result

    def _serialize_path(
        self,
        path: RetrievalPath,
    ) -> dict[str, Any]:
        return {
            "seed_rule_id": path.seed_rule_id,
            "rule_ids": path.rule_ids,
            "seed_similarity": path.seed_similarity,
            "score": path.score,
            "steps": [
                asdict(step)
                for step in path.steps
            ],
            "rules": [
                self._serialize_rule(rule_id)
                for rule_id in path.rule_ids
            ],
        }


# ============================================================
# LLM CONTEXT FORMATTER
# ============================================================

def format_context_for_llm(
    retrieval_result: dict[str, Any],
    max_evidence: int = 8,
) -> str:
    """
    Biến retrieval result thành context gọn để truyền vào LLM.
    """
    query = retrieval_result["query"]

    lines = [
        "CÂU HỎI:",
        query,
        "",
        "BẰNG CHỨNG PHÁP LUẬT:",
    ]

    evidence = retrieval_result.get(
        "evidence",
        [],
    )[:max_evidence]

    for position, item in enumerate(evidence, start=1):
        lines.extend([
            "",
            f"[Bằng chứng {position}]",
            (
                f"Rule ID: {item['rule_id']} | "
                f"Điều {item['article_id']}"
            ),
            f"Tên điều: {item['article_title']}",
            f"Chủ thể: {item['legal_subject']}",
            (
                f"Điều kiện: {item['condition']} "
                f"({item['condition_norm']})"
            ),
            (
                f"Hệ quả: {item['effect']} "
                f"({item['effect_norm']})"
            ),
        ])

    lines.extend([
        "",
        "CÁC ĐƯỜNG SUY LUẬN ĐƯỢC TRUY HỒI:",
    ])

    for path_index, path in enumerate(
        retrieval_result.get("paths", [])[:5],
        start=1,
    ):
        lines.append("")
        lines.append(
            f"[Đường {path_index}] "
            f"score={path['score']:.4f}"
        )

        for step in path["steps"]:
            lines.append(
                f"- Rule {step['from_rule_id']} "
                f"--{step['relation']}--> "
                f"Rule {step['to_rule_id']}"
            )
            lines.append(
                f"  {step['explanation']}"
            )

    return "\n".join(lines)


# ============================================================
# DISPLAY
# ============================================================

def print_result_summary(
    retrieval_result: dict[str, Any],
) -> None:
    print("\n" + "=" * 80)
    print("MULTI-HOP CAUSAL RETRIEVAL")
    print("=" * 80)

    print("\nQuery:")
    print(retrieval_result["query"])

    print("\nTop paths:")

    for index, path in enumerate(
        retrieval_result.get("paths", [])[:10],
        start=1,
    ):
        print(
            f"\nPath {index}: "
            f"score={path['score']:.4f}"
        )

        print(
            "  "
            + " -> ".join(path["rule_ids"])
        )

        if not path["steps"]:
            print("  Seed rule trực tiếp.")
            continue

        for step in path["steps"]:
            print(
                f"  [{step['relation']}] "
                f"{step['from_rule_id']} "
                f"-> {step['to_rule_id']}"
            )

    print("\nTop evidence:")

    for index, item in enumerate(
        retrieval_result.get("evidence", [])[:10],
        start=1,
    ):
        print(
            f"\n{index}. Rule {item['rule_id']} "
            f"- Điều {item['article_id']}: "
            f"{item['article_title']}"
        )
        print(
            f"   Chủ thể: {item['legal_subject']}"
        )
        print(
            f"   Condition: {item['condition']}"
        )
        print(
            f"   Effect: {item['effect']}"
        )
        print(
            f"   Evidence score: "
            f"{item['evidence_score']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-hop Causal Retriever cho dữ liệu "
            "Bộ luật Hình sự Việt Nam."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Câu hỏi pháp luật cần truy hồi.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
    )

    parser.add_argument(
        "--index",
        type=str,
        default=FAISS_INDEX_PATH,
    )

    parser.add_argument(
        "--memory",
        type=str,
        default=MEMORY_PATH,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
    )

    parser.add_argument(
        "--seed-top-k",
        type=int,
        default=DEFAULT_SEED_TOP_K,
    )

    parser.add_argument(
        "--semantic-pool-size",
        type=int,
        default=DEFAULT_SEMANTIC_POOL_SIZE,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
    )

    parser.add_argument(
        "--max-expansions",
        type=int,
        default=DEFAULT_MAX_EXPANSIONS_PER_RULE,
    )

    parser.add_argument(
        "--final-top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_depth < 0:
        raise ValueError(
            "--max-depth phải lớn hơn hoặc bằng 0."
        )

    repository = RuleRepository(
        data_path=args.data
    )

    dense_retriever = DenseRuleRetriever(
        repository=repository,
        index_path=args.index,
        memory_path=args.memory,
        model_name=args.model,
    )

    retriever = MultiHopCausalRetriever(
        repository=repository,
        dense_retriever=dense_retriever,
    )

    result = retriever.retrieve(
        query=args.query,
        seed_top_k=args.seed_top_k,
        semantic_pool_size=args.semantic_pool_size,
        max_depth=args.max_depth,
        max_expansions_per_rule=args.max_expansions,
        final_top_k=args.final_top_k,
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

    print_result_summary(result)

    llm_context = format_context_for_llm(
        result,
        max_evidence=8,
    )

    context_output_path = output_path.with_name(
        output_path.stem + "_context.txt"
    )

    context_output_path.write_text(
        llm_context,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(f"Saved retrieval JSON: {output_path}")
    print(f"Saved LLM context: {context_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()