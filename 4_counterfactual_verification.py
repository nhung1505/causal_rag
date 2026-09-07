#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4 v4.1 - Query-aware structural counterfactual verification.

Mặc định, Step 4 dùng ``causal_core.LegalSCM`` để:
1. Xác nhận factual causal chain bằng các legal rule mechanisms.
2. Thực hiện hard ``do(mediator=FALSE)`` và suy luận lại outcome.
3. Đánh giá claim cụ thể trong query từ factual/counterfactual worlds.

Node-deletion reachability được giữ dưới mode ``path_ablation`` và luôn được
lưu như baseline diagnostic khi chạy mode ``structural_scm``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from causal_core import (
    CounterfactualStatus,
    EventState,
    LegalSCM,
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

GRAPH_PATH = "data/legal_causal_knowledge_graph.graphml"
MEMORY_PATH = "data/causal_memory.csv"
RULES_PATH = "data/blhs_rules_final_all_normalized.json"
RETRIEVAL_RESULT_PATH = "data/retrieval_result.json"
OUTPUT_PATH = "data/counterfactual_verification_result.json"

STRUCTURAL_SCM_MODE = "structural_scm"
PATH_ABLATION_MODE = "path_ablation"
DEFAULT_COUNTERFACTUAL_MODE = STRUCTURAL_SCM_MODE

# Giữ `query-aware` để Step 5.5 nhận diện compatibility.
STEP4_VERSION = "4.1-query-aware-legal-scm"

# Cache model theo file + mtime để batch không parse 2.884 rules mỗi câu.
_LEGAL_SCM_CACHE: dict[tuple[str, int], LegalSCM] = {}

# Điểm thưởng cho evidence thuộc primary path.
PRIMARY_PATH_EVIDENCE_BONUS = 0.12

# Mục tiêu của benchmark hiện tại chủ yếu là chuỗi hai hop.
DEFAULT_TARGET_HOPS = 2

# Số hop tối đa khi tìm đường thay thế sau intervention.
DEFAULT_MAX_CF_HOPS = 3

# Số đường thay thế tối đa được lưu cho mỗi intervention.
DEFAULT_MAX_CF_PATHS = 30

# Số evidence KEEP tối đa trả về.
DEFAULT_VERIFIED_TOP_K = 10

# Ngưỡng phân loại evidence.
DEFAULT_KEEP_THRESHOLD = 0.52
DEFAULT_REJECT_THRESHOLD = 0.34

# Ngưỡng dùng để kết luận mediator có cần thiết hay không.
#
# Nếu điểm đường thay thế >= ngưỡng này:
#   mediator không cần thiết hoàn toàn -> CONTRADICTED
#
# Nếu không tồn tại đường thay thế:
#   mediator cần thiết trong graph -> SUPPORTED
DEFAULT_ALTERNATIVE_PATH_THRESHOLD = 0.35

# Giới hạn số mediator được kiểm tra trên một path.
# Với graph hiện tại chủ yếu là path hai hop nên giá trị 5 là đủ.
DEFAULT_MAX_MEDIATORS_PER_PATH = 5

# Hệ số giảm điểm theo số hop.
HOP_DECAY = 0.84

# Trọng số đánh giá evidence.
PATH_SUPPORT_WEIGHT = 0.38
COUNTERFACTUAL_SUPPORT_WEIGHT = 0.32
SEMANTIC_EVIDENCE_WEIGHT = 0.16
GRAPH_EVIDENCE_WEIGHT = 0.14

# Điểm cơ sở cho các trường hợp intervention.
NECESSARY_MEDIATOR_SCORE = 0.85
PARTIALLY_NECESSARY_SCORE = 0.60
NON_NECESSARY_SCORE = 0.25
DIRECT_PATH_SUPPORT_SCORE = 0.70
UNRESOLVED_BASE_SCORE = 0.35


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class AlternativeCausalPath:
    """
    Một causal path thay thế được tìm thấy sau khi loại mediator.

    Ví dụ original path:

        A -> B -> C

    Intervention:

        remove(B)

    Alternative path có thể là:

        A -> D -> C
    """

    start_event_node: str
    start_event_id: str
    start_event_name: str

    end_event_node: str
    end_event_id: str
    end_event_name: str

    event_nodes: list[str]
    event_ids: list[str]
    event_names: list[str]

    rule_ids: list[str]
    article_ids: list[str]

    hop_count: int
    path_score: float


@dataclass
class MediatorIntervention:
    """
    Kết quả một phép can thiệp do(remove mediator).

    intervention_status:
        NECESSARY:
            Sau khi loại mediator không còn đường từ seed đến outcome.

        PARTIALLY_NECESSARY:
            Có đường thay thế nhưng yếu hơn đáng kể so với original path.

        NON_NECESSARY:
            Có đường thay thế đủ mạnh; mediator không phải mắt xích bắt buộc.

        UNRESOLVED:
            Không đủ dữ liệu để thực hiện intervention.
    """

    mediator_index: int

    mediator_event_node: str
    mediator_event_id: str
    mediator_event_name: str

    removed_nodes: list[str]

    alternative_paths: list[dict[str, Any]]
    best_alternative_path_score: float

    intervention_status: str
    necessity_score: float
    explanation: str

    # Structural SCM metadata. Baseline records keep the defaults below.
    verification_method: str = "node_deletion_reachability"
    intervention_assignment: dict[str, str] = field(default_factory=dict)
    structural_status: str = ""
    factual_mediator_state: str = ""
    factual_outcome: str = ""
    counterfactual_outcome: str = ""
    outcome_changed: Optional[bool] = None
    disabled_rule_ids: list[str] = field(default_factory=list)
    newly_activated_rule_ids: list[str] = field(default_factory=list)
    alternative_outcome_rule_ids: list[str] = field(default_factory=list)
    recomputed_context_event_ids: list[str] = field(default_factory=list)


@dataclass
class PathVerification:
    """
    Kết quả xác minh một causal path từ Step 3.

    Giữ các trường cũ như:
        seed_event_id
        original_outcome_event_id
        status
        consistency_score

    để Step 5 và Step 5.5 tiếp tục sử dụng được.
    """

    original_path_id: int

    seed_event_id: str
    seed_event_name: str

    original_outcome_event_id: str
    original_outcome_event_name: str

    original_event_nodes: list[str]
    original_event_ids: list[str]
    original_event_names: list[str]

    original_rule_ids: list[str]
    original_article_ids: list[str]

    original_path_score: float
    original_hop_count: int

    intervention_type: str
    mediator_interventions: list[dict[str, Any]]

    status: str
    consistency_score: float
    explanation: str

    # Các trường tương thích với output cũ.
    # Bản mới không sinh counterfactual event phủ định.
    counterfactual_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    counterfactual_to_same_outcome: list[dict[str, Any]] = field(
        default_factory=list
    )
    counterfactual_to_opposite_outcome: list[dict[str, Any]] = field(
        default_factory=list
    )
    opposite_outcome_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )

    # Compact audit payload; downstream Step 5 ignores unknown additive fields.
    counterfactual_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceVerification:
    original_rank: int
    rule_id: str
    article_id: str

    original_final_score: float
    semantic_score: float
    graph_score: float

    path_support_score: float
    counterfactual_support_score: float

    verification_score: float
    decision: str

    verified_path_ids: list[int] = field(
        default_factory=list
    )
    rejected_path_ids: list[int] = field(
        default_factory=list
    )
    unresolved_path_ids: list[int] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )

    original_evidence: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class VerificationResult:
    query: str

    configuration: dict[str, Any]
    statistics: dict[str, Any]

    path_verifications: list[dict[str, Any]]

    verified_evidence: list[dict[str, Any]]
    uncertain_evidence: list[dict[str, Any]]
    removed_evidence: list[dict[str, Any]]

    consistency_score: float
    confidence: float

    # Kết luận toàn cục ở cấp độ claim/câu hỏi.
    final_decision: str = "UNCERTAIN"
    decision_score: float = 0.0
    decision_explanation: str = ""

    # Path đại diện được Step 5 và Step 5.5 ưu tiên sử dụng.
    primary_path_ids: list[int] = field(default_factory=list)

    # Kết quả phân tích câu hỏi phản thực tế.
    query_analysis: dict[str, Any] = field(default_factory=dict)

    verification_method: str = (
        "query_aware_graph_intervention_v3"
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    """Chuyển giá trị sang chuỗi và xử lý NaN/None."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Chuyển giá trị sang float an toàn."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Chuyển giá trị sang int an toàn."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def split_csv_values(value: Any) -> list[str]:
    """
    Chuyển chuỗi phân cách bằng dấu phẩy thành list.

    Ví dụ:
        "R1,R2,R3" -> ["R1", "R2", "R3"]
    """

    text = safe_string(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def ensure_string_list(value: Any) -> list[str]:
    """
    Chuẩn hóa dữ liệu về list[str].

    Chấp nhận:
        - list
        - tuple
        - set
        - chuỗi CSV
        - chuỗi JSON list
        - None
    """

    if value is None:
        return []

    if isinstance(value, list):
        return unique_preserve_order(
            safe_string(item)
            for item in value
        )

    if isinstance(value, (tuple, set)):
        return unique_preserve_order(
            safe_string(item)
            for item in value
        )

    text = safe_string(value)

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return unique_preserve_order(
                    safe_string(item)
                    for item in parsed
                )
        except json.JSONDecodeError:
            pass

    return split_csv_values(text)


def unique_preserve_order(
    values: Iterable[Any],
) -> list[str]:
    """Loại trùng nhưng giữ nguyên thứ tự xuất hiện."""

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = safe_string(value)

        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def clamp(
    value: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Đưa giá trị vào đoạn [lower, upper]."""

    return max(
        lower,
        min(upper, value),
    )


def event_id_from_node(
    graph: nx.Graph,
    node_id: str,
) -> str:
    """Lấy event_id từ EVENT node."""

    if node_id not in graph:
        return safe_string(node_id).removeprefix(
            "EVENT::"
        )

    data = graph.nodes[node_id]

    return (
        safe_string(data.get("event_id"))
        or safe_string(node_id).removeprefix(
            "EVENT::"
        )
    )


def event_name_from_node(
    graph: nx.Graph,
    node_id: str,
) -> str:
    """Lấy event_name hiển thị của EVENT node."""

    if node_id not in graph:
        return event_id_from_node(
            graph,
            node_id,
        )

    data = graph.nodes[node_id]

    return (
        safe_string(data.get("event_name"))
        or safe_string(data.get("label"))
        or safe_string(data.get("name"))
        or event_id_from_node(
            graph,
            node_id,
        )
    )


def is_event_node(
    graph: nx.Graph,
    node_id: str,
) -> bool:
    """Kiểm tra node có phải EVENT hay không."""

    if node_id not in graph:
        return False

    node_type = safe_string(
        graph.nodes[node_id].get("node_type")
    ).upper()

    return (
        node_type == "EVENT"
        or safe_string(node_id).startswith("EVENT::")
    )


def json_serializable(data: Any) -> Any:
    """Chuyển numpy và các object lồng nhau sang JSON-compatible."""

    if isinstance(data, np.generic):
        return data.item()

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, dict):
        return {
            str(key): json_serializable(value)
            for key, value in data.items()
        }

    if isinstance(data, (list, tuple, set)):
        return [
            json_serializable(value)
            for value in data
        ]

    return data


# ============================================================
# RESOURCE STORE
# ============================================================

class CounterfactualResourceStore:
    """
    Nạp dữ liệu cần thiết cho graph intervention.

    Bản v2 chỉ cần:
        - legal causal graph
        - causal memory
        - retrieval result của Step 3

    Không còn cần:
        - sentence-transformers
        - embeddings
        - counterfactual_event_map.json
        - semantic mapping
    """

    def __init__(
        self,
        *,
        graph_path: str,
        memory_path: str,
        retrieval_result_path: str,
        rules_path: Optional[str] = RULES_PATH,

        # Giữ lại các tham số dưới đây để tương thích với
        # 5_5_generate_pipeline_predictions.py và CLI cũ.
        embeddings_path: Optional[str] = None,
        counterfactual_map_path: Optional[str] = None,
        model_name: Optional[str] = None,
        enable_semantic_mapping: bool = False,
        **_: Any,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.memory_path = Path(memory_path)
        self.retrieval_result_path = Path(
            retrieval_result_path
        )
        self.rules_path = Path(rules_path) if rules_path else None
        self.legal_scm: Optional[LegalSCM] = None
        self.scm_load_error = ""

        # Chỉ lưu để compatibility, không dùng trong thuật toán.
        self.embeddings_path = (
            Path(embeddings_path)
            if embeddings_path
            else None
        )
        self.counterfactual_map_path = (
            Path(counterfactual_map_path)
            if counterfactual_map_path
            else None
        )
        self.model_name = safe_string(model_name)
        self.enable_semantic_mapping = False

        self.graph = self._load_graph()
        self.memory_df = self._load_memory()
        self.retrieval_result = (
            self._load_retrieval_result()
        )
        self.legal_scm = self._load_legal_scm()

        self._validate_resources()
        self._build_lookup_tables()

        self.causal_event_graph = (
            self._build_causal_event_graph()
        )

        self._validate_retrieval_result()

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    def _load_legal_scm(self) -> Optional[LegalSCM]:
        if self.rules_path is None:
            self.scm_load_error = "Không cấu hình rules_path."
            return None
        if not self.rules_path.exists():
            self.scm_load_error = (
                f"Không tìm thấy normalized rules: {self.rules_path}"
            )
            print("Warning:", self.scm_load_error)
            return None

        resolved = self.rules_path.resolve()
        cache_key = (str(resolved), resolved.stat().st_mtime_ns)
        cached = _LEGAL_SCM_CACHE.get(cache_key)
        if cached is not None:
            print(f"Using cached LegalSCM: {resolved}")
            return cached

        print(f"Loading LegalSCM rules: {resolved}")
        try:
            with resolved.open("r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError("Normalized rules phải là JSON array.")
            valid_records = [
                item for item in records if isinstance(item, dict)
            ]
            if len(valid_records) != len(records):
                raise ValueError(
                    "Normalized rules chứa record không phải JSON object."
                )
            model = LegalSCM.from_legacy_records(valid_records)
        except Exception as error:
            self.scm_load_error = f"{type(error).__name__}: {error}"
            print("Warning: không thể load LegalSCM:", self.scm_load_error)
            return None

        # Xóa cache cũ của cùng file sau khi nội dung thay đổi.
        for old_key in list(_LEGAL_SCM_CACHE):
            if old_key[0] == str(resolved) and old_key != cache_key:
                _LEGAL_SCM_CACHE.pop(old_key, None)
        _LEGAL_SCM_CACHE[cache_key] = model
        print(
            "LegalSCM loaded:",
            len(model.rules),
            "rules,",
            len(model.event_ids),
            "events",
        )
        return model

    def _load_graph(
        self,
    ) -> nx.MultiDiGraph | nx.DiGraph:
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy graph: "
                f"{self.graph_path}"
            )

        print(
            f"Loading graph: {self.graph_path}"
        )

        graph = nx.read_graphml(
            self.graph_path
        )

        if not graph.is_directed():
            raise ValueError(
                "Legal causal graph phải là đồ thị "
                "có hướng."
            )

        return graph

    def _load_memory(
        self,
    ) -> pd.DataFrame:
        if not self.memory_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy memory: "
                f"{self.memory_path}"
            )

        print(
            f"Loading memory: {self.memory_path}"
        )

        memory_df = pd.read_csv(
            self.memory_path,
            dtype=str,
            keep_default_na=False,
        )

        required_columns = {
            "memory_id",
            "memory_type",
            "graph_node_id",
        }

        missing_columns = (
            required_columns
            - set(memory_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Causal memory thiếu cột: "
                f"{sorted(missing_columns)}"
            )

        memory_df["memory_id"] = pd.to_numeric(
            memory_df["memory_id"],
            errors="raise",
        ).astype(np.int64)

        memory_df["memory_type"] = (
            memory_df["memory_type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        memory_df["graph_node_id"] = (
            memory_df["graph_node_id"]
            .astype(str)
            .str.strip()
        )

        return memory_df

    def _load_retrieval_result(
        self,
    ) -> dict[str, Any]:
        if not self.retrieval_result_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy retrieval result: "
                f"{self.retrieval_result_path}"
            )

        print(
            "Loading retrieval result: "
            f"{self.retrieval_result_path}"
        )

        with self.retrieval_result_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

        if not isinstance(result, dict):
            raise ValueError(
                "Retrieval result phải là một JSON object."
            )

        required_keys = {
            "query",
            "causal_paths",
            "evidence",
        }

        missing_keys = (
            required_keys
            - set(result.keys())
        )

        if missing_keys:
            raise ValueError(
                "Retrieval result thiếu trường: "
                f"{sorted(missing_keys)}"
            )

        if not isinstance(
            result.get("causal_paths"),
            list,
        ):
            raise ValueError(
                "`causal_paths` phải là list."
            )

        if not isinstance(
            result.get("evidence"),
            list,
        ):
            raise ValueError(
                "`evidence` phải là list."
            )

        return result

    # --------------------------------------------------------
    # RESOURCE VALIDATION
    # --------------------------------------------------------

    def _validate_resources(
        self,
    ) -> None:
        graph_nodes = set(
            self.graph.nodes
        )

        memory_nodes = {
            node_id
            for node_id in self.memory_df[
                "graph_node_id"
            ].tolist()
            if node_id
        }

        missing_nodes = (
            memory_nodes - graph_nodes
        )

        if missing_nodes:
            examples = sorted(
                missing_nodes
            )[:10]

            raise ValueError(
                "Memory chứa graph_node_id không tồn tại "
                "trong graph. Ví dụ: "
                f"{examples}"
            )

        memory_types = set(
            self.memory_df[
                "memory_type"
            ].tolist()
        )

        if "EVENT" not in memory_types:
            raise ValueError(
                "Causal memory không có EVENT record."
            )

        if "RULE" not in memory_types:
            raise ValueError(
                "Causal memory không có RULE record."
            )

        event_nodes = [
            node_id
            for node_id in self.graph.nodes
            if is_event_node(
                self.graph,
                node_id,
            )
        ]

        if not event_nodes:
            raise ValueError(
                "Graph không chứa EVENT node."
            )

        print("Resource validation: OK")

    def _validate_retrieval_result(
        self,
    ) -> None:
        """
        Kiểm tra nhẹ retrieval result.

        Không dừng toàn bộ pipeline nếu có một số path lỗi;
        PathVerifier sẽ đánh dấu các path đó là UNRESOLVED.
        """

        causal_paths = (
            self.retrieval_result.get(
                "causal_paths",
                [],
            )
        )

        invalid_count = 0

        for path in causal_paths:
            if not isinstance(path, dict):
                invalid_count += 1
                continue

            event_nodes = self.get_path_event_nodes(
                path
            )

            if len(event_nodes) < 2:
                invalid_count += 1

        if invalid_count:
            print(
                "Warning:",
                invalid_count,
                "causal path không có đủ hai EVENT node."
            )

    # --------------------------------------------------------
    # LOOKUP TABLES
    # --------------------------------------------------------

    def _build_lookup_tables(
        self,
    ) -> None:
        self.memory_by_id = (
            self.memory_df.set_index(
                "memory_id",
                drop=False,
            )
        )

        self.event_df = self.memory_df[
            self.memory_df["memory_type"]
            == "EVENT"
        ].copy()

        self.rule_df = self.memory_df[
            self.memory_df["memory_type"]
            == "RULE"
        ].copy()

        self.event_by_node: dict[
            str,
            pd.Series,
        ] = {}

        self.event_node_by_id: dict[
            str,
            str,
        ] = {}

        self.rule_by_id: dict[
            str,
            pd.Series,
        ] = {}

        for _, row in self.event_df.iterrows():
            node_id = safe_string(
                row.get("graph_node_id")
            )

            if not node_id:
                continue

            event_id = (
                safe_string(
                    row.get("event_id")
                )
                or event_id_from_node(
                    self.graph,
                    node_id,
                )
            )

            self.event_by_node[node_id] = row

            if event_id:
                self.event_node_by_id[
                    event_id
                ] = node_id

        for _, row in self.rule_df.iterrows():
            rule_id = safe_string(
                row.get("rule_id")
            )

            if rule_id:
                self.rule_by_id[
                    rule_id
                ] = row

    # --------------------------------------------------------
    # EVENT GRAPH
    # --------------------------------------------------------

    def _build_causal_event_graph(
        self,
    ) -> nx.DiGraph:
        """
        Tạo graph chỉ gồm EVENT node và CAUSES edge.

        Multi-edge giữa cùng hai event được gộp thành một edge,
        đồng thời hợp nhất:
            - rule_ids
            - article_ids
            - support_count
        """

        causal_graph = nx.DiGraph()

        for node_id, data in self.graph.nodes(
            data=True
        ):
            if is_event_node(
                self.graph,
                node_id,
            ):
                causal_graph.add_node(
                    node_id,
                    **dict(data),
                )

        if self.graph.is_multigraph():
            edge_iterator = self.graph.edges(
                keys=True,
                data=True,
            )

            for (
                source,
                target,
                _,
                edge_data,
            ) in edge_iterator:
                self._merge_causal_edge(
                    graph=causal_graph,
                    source=source,
                    target=target,
                    data=edge_data,
                )
        else:
            for (
                source,
                target,
                edge_data,
            ) in self.graph.edges(
                data=True
            ):
                self._merge_causal_edge(
                    graph=causal_graph,
                    source=source,
                    target=target,
                    data=edge_data,
                )

        print(
            "Causal event graph:",
            causal_graph.number_of_nodes(),
            "events,",
            causal_graph.number_of_edges(),
            "causal edges",
        )

        return causal_graph

    @staticmethod
    def _merge_causal_edge(
        *,
        graph: nx.DiGraph,
        source: str,
        target: str,
        data: dict[str, Any],
    ) -> None:
        relation = safe_string(
            data.get("relation")
        ).upper()

        if relation != "CAUSES":
            return

        if (
            source not in graph
            or target not in graph
        ):
            return

        rule_ids = ensure_string_list(
            data.get("rule_ids")
        )

        if not rule_ids:
            rule_id = safe_string(
                data.get("rule_id")
            )

            if rule_id:
                rule_ids = [rule_id]

        article_ids = ensure_string_list(
            data.get("article_ids")
        )

        if not article_ids:
            article_id = safe_string(
                data.get("article_id")
            )

            if article_id:
                article_ids = [article_id]

        support_count = max(
            1,
            safe_int(
                data.get("support_count"),
                default=1,
            ),
        )

        if graph.has_edge(
            source,
            target,
        ):
            existing = graph[
                source
            ][target]

            existing["rule_ids"] = (
                unique_preserve_order(
                    list(
                        existing.get(
                            "rule_ids",
                            [],
                        )
                    )
                    + rule_ids
                )
            )

            existing["article_ids"] = (
                unique_preserve_order(
                    list(
                        existing.get(
                            "article_ids",
                            [],
                        )
                    )
                    + article_ids
                )
            )

            existing["support_count"] = (
                safe_int(
                    existing.get(
                        "support_count"
                    ),
                    default=0,
                )
                + support_count
            )
        else:
            graph.add_edge(
                source,
                target,
                relation="CAUSES",
                rule_ids=rule_ids,
                article_ids=article_ids,
                support_count=support_count,
            )

    # --------------------------------------------------------
    # PATH NORMALIZATION
    # --------------------------------------------------------

    def get_path_event_nodes(
        self,
        path: dict[str, Any],
    ) -> list[str]:
        """
        Chuẩn hóa EVENT node từ causal path của Step 3.

        Ưu tiên:
            1. path["event_nodes"]
            2. path["event_ids"]
            3. path["events"]

        Chỉ giữ node thực sự tồn tại trong causal_event_graph.
        """

        raw_event_nodes = ensure_string_list(
            path.get("event_nodes")
        )

        valid_event_nodes = [
            node_id
            for node_id in raw_event_nodes
            if node_id in self.causal_event_graph
        ]

        if len(valid_event_nodes) >= 2:
            return self._orient_path_event_nodes(
                path,
                valid_event_nodes,
            )

        raw_event_ids = ensure_string_list(
            path.get("event_ids")
        )

        mapped_nodes = [
            self.event_node_by_id[event_id]
            for event_id in raw_event_ids
            if event_id in self.event_node_by_id
        ]

        mapped_nodes = [
            node_id
            for node_id in mapped_nodes
            if node_id in self.causal_event_graph
        ]

        if len(mapped_nodes) >= 2:
            return self._orient_path_event_nodes(
                path,
                mapped_nodes,
            )

        raw_events = path.get("events")

        if isinstance(raw_events, list):
            fallback_nodes: list[str] = []

            for event in raw_events:
                if not isinstance(event, dict):
                    continue

                node_id = safe_string(
                    event.get("event_node")
                    or event.get("node_id")
                    or event.get("graph_node_id")
                )

                if (
                    node_id
                    and node_id
                    in self.causal_event_graph
                ):
                    fallback_nodes.append(
                        node_id
                    )
                    continue

                event_id = safe_string(
                    event.get("event_id")
                )

                mapped_node = (
                    self.event_node_by_id.get(
                        event_id
                    )
                )

                if (
                    mapped_node
                    and mapped_node
                    in self.causal_event_graph
                ):
                    fallback_nodes.append(
                        mapped_node
                    )

            return self._orient_path_event_nodes(
                path,
                unique_preserve_order(
                    fallback_nodes
                ),
            )

        return self._orient_path_event_nodes(
            path,
            valid_event_nodes,
        )

    def _orient_path_event_nodes(
        self,
        path: dict[str, Any],
        event_nodes: list[str],
    ) -> list[str]:
        """Đưa path về đúng chiều nhân quả: cause -> ... -> effect.

        Step 3 lưu path backward theo thứ tự seed -> predecessor. Vì vậy
        ``event_nodes`` của path backward cần được đảo trước khi Step 4 dùng
        node đầu làm seed/cause và node cuối làm outcome/effect.

        Hàm cũng tự kiểm tra hai chiều để tương thích với retrieval result cũ
        hoặc dữ liệu được tạo từ phiên bản khác của Step 3.
        """

        nodes = unique_preserve_order(event_nodes)

        if len(nodes) < 2:
            return nodes

        direction = safe_string(
            path.get("direction")
        ).lower()

        preferred = (
            list(reversed(nodes))
            if direction == "backward"
            else list(nodes)
        )

        def is_valid_chain(candidate: list[str]) -> bool:
            return all(
                self.causal_event_graph.has_edge(source, target)
                for source, target in zip(
                    candidate[:-1],
                    candidate[1:],
                )
            )

        if is_valid_chain(preferred):
            return preferred

        reversed_preferred = list(reversed(preferred))

        if is_valid_chain(reversed_preferred):
            return reversed_preferred

        # Giữ thứ tự ưu tiên để verifier có thể đánh dấu path sai cấu trúc.
        return preferred

    def get_path_rule_ids(
        self,
        path: dict[str, Any],
        event_nodes: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Lấy rule_ids từ path.

        Nếu Step 3 không cung cấp rule_ids, suy ra từ các CAUSES edge.
        """

        rule_ids = ensure_string_list(
            path.get("rule_ids")
        )

        if rule_ids:
            return rule_ids

        nodes = (
            event_nodes
            if event_nodes is not None
            else self.get_path_event_nodes(path)
        )

        inferred_rule_ids: list[str] = []

        for source, target in zip(
            nodes[:-1],
            nodes[1:],
        ):
            if self.causal_event_graph.has_edge(
                source,
                target,
            ):
                edge_data = (
                    self.causal_event_graph[
                        source
                    ][target]
                )

                inferred_rule_ids.extend(
                    ensure_string_list(
                        edge_data.get(
                            "rule_ids"
                        )
                    )
                )

        return unique_preserve_order(
            inferred_rule_ids
        )

    def get_path_article_ids(
        self,
        path: dict[str, Any],
        event_nodes: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Lấy article_ids từ path.

        Nếu Step 3 không cung cấp, suy ra từ các CAUSES edge.
        """

        article_ids = ensure_string_list(
            path.get("article_ids")
        )

        if article_ids:
            return article_ids

        nodes = (
            event_nodes
            if event_nodes is not None
            else self.get_path_event_nodes(path)
        )

        inferred_article_ids: list[str] = []

        for source, target in zip(
            nodes[:-1],
            nodes[1:],
        ):
            if self.causal_event_graph.has_edge(
                source,
                target,
            ):
                edge_data = (
                    self.causal_event_graph[
                        source
                    ][target]
                )

                inferred_article_ids.extend(
                    ensure_string_list(
                        edge_data.get(
                            "article_ids"
                        )
                    )
                )

        return unique_preserve_order(
            inferred_article_ids
        )

    def calculate_path_score(
        self,
        event_nodes: list[str],
    ) -> float:
        """
        Tính điểm structural support của path.

        Điểm dựa trên:
            - support_count của từng CAUSES edge
            - độ dài path
        """

        if len(event_nodes) < 2:
            return 0.0

        edge_support_scores: list[float] = []

        for source, target in zip(
            event_nodes[:-1],
            event_nodes[1:],
        ):
            if not self.causal_event_graph.has_edge(
                source,
                target,
            ):
                return 0.0

            edge_data = (
                self.causal_event_graph[
                    source
                ][target]
            )

            support_count = max(
                1,
                safe_int(
                    edge_data.get(
                        "support_count"
                    ),
                    default=1,
                ),
            )

            normalized_support = min(
                1.0,
                math.log1p(
                    support_count
                )
                / math.log(4.0),
            )

            edge_support_scores.append(
                normalized_support
            )

        average_support = (
            sum(edge_support_scores)
            / len(edge_support_scores)
        )

        hop_count = len(event_nodes) - 1

        return clamp(
            average_support
            * (
                HOP_DECAY
                ** max(
                    0,
                    hop_count - 1,
                )
            )
        )   

# ============================================================
# MEDIATOR INTERVENTION SEARCH
# ============================================================

class CounterfactualGraphSearcher:
    """Tìm đường thay thế từ seed tới outcome sau do(remove mediator)."""

    def __init__(self, store: CounterfactualResourceStore) -> None:
        self.store = store

    def find_alternative_paths(
        self,
        *,
        start_node: str,
        end_node: str,
        removed_nodes: Iterable[str],
        max_hops: int,
        max_paths: int,
    ) -> list[AlternativeCausalPath]:
        graph = self.store.causal_event_graph
        removed = {safe_string(node) for node in removed_nodes if safe_string(node)}

        if start_node not in graph or end_node not in graph:
            return []
        if start_node in removed or end_node in removed:
            return []

        view = nx.subgraph_view(
            graph,
            filter_node=lambda node: node not in removed,
        )

        results: list[AlternativeCausalPath] = []
        try:
            paths = nx.all_simple_paths(
                view,
                source=start_node,
                target=end_node,
                cutoff=max_hops,
            )
            for event_nodes in paths:
                results.append(self._build_path(list(event_nodes)))
                if len(results) >= max_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        results.sort(key=lambda item: item.path_score, reverse=True)
        return results[:max_paths]

    def _build_path(self, event_nodes: list[str]) -> AlternativeCausalPath:
        graph = self.store.causal_event_graph
        rule_ids: list[str] = []
        article_ids: list[str] = []

        for source, target in zip(event_nodes[:-1], event_nodes[1:]):
            edge = graph[source][target]
            rule_ids.extend(ensure_string_list(edge.get("rule_ids")))
            article_ids.extend(ensure_string_list(edge.get("article_ids")))

        return AlternativeCausalPath(
            start_event_node=event_nodes[0],
            start_event_id=event_id_from_node(graph, event_nodes[0]),
            start_event_name=event_name_from_node(graph, event_nodes[0]),
            end_event_node=event_nodes[-1],
            end_event_id=event_id_from_node(graph, event_nodes[-1]),
            end_event_name=event_name_from_node(graph, event_nodes[-1]),
            event_nodes=event_nodes,
            event_ids=[event_id_from_node(graph, node) for node in event_nodes],
            event_names=[event_name_from_node(graph, node) for node in event_nodes],
            rule_ids=unique_preserve_order(rule_ids),
            article_ids=unique_preserve_order(article_ids),
            hop_count=len(event_nodes) - 1,
            path_score=self.store.calculate_path_score(event_nodes),
        )


# ============================================================
# PATH VERIFICATION
# ============================================================

class CounterfactualPathVerifier:
    """
    Xác minh path bằng can thiệp cấu trúc.

    Điểm sửa quan trọng: path hai node (một hop) là path hợp lệ và được
    SUPPORTED trực tiếp; không còn bị đánh UNRESOLVED chỉ vì không có mediator.
    """

    def __init__(
        self,
        *,
        store: CounterfactualResourceStore,
        searcher: Optional[CounterfactualGraphSearcher] = None,
        **_: Any,
    ) -> None:
        self.store = store
        self.searcher = searcher or CounterfactualGraphSearcher(store)

    def verify_path(
        self,
        *,
        path_id: int,
        original_path: dict[str, Any],
        max_hops: int = DEFAULT_MAX_CF_HOPS,
        max_paths: int = DEFAULT_MAX_CF_PATHS,
        max_mediators: int = DEFAULT_MAX_MEDIATORS_PER_PATH,
        alternative_path_threshold: float = DEFAULT_ALTERNATIVE_PATH_THRESHOLD,
        **_: Any,
    ) -> PathVerification:
        event_nodes = self.store.get_path_event_nodes(original_path)
        rule_ids = self.store.get_path_rule_ids(original_path, event_nodes)
        article_ids = self.store.get_path_article_ids(original_path, event_nodes)

        if len(event_nodes) < 2:
            return self._unresolved(
                path_id=path_id,
                explanation=(
                    "Không chuẩn hóa được ít nhất hai EVENT node từ path. "
                    "Hãy kiểm tra event_nodes/event_ids/events trong retrieval_result.json."
                ),
            )

        seed_node = event_nodes[0]
        outcome_node = event_nodes[-1]
        path_score = safe_float(
            original_path.get(
                "graph_score",
                original_path.get(
                    "path_score",
                    original_path.get("score"),
                ),
            ),
            default=self.store.calculate_path_score(event_nodes),
        )
        structural_path_score = self.store.calculate_path_score(
            event_nodes
        )

        if structural_path_score <= 0.0:
            return PathVerification(
                original_path_id=path_id,
                seed_event_id=event_id_from_node(
                    self.store.graph,
                    event_nodes[0],
                ),
                seed_event_name=event_name_from_node(
                    self.store.graph,
                    event_nodes[0],
                ),
                original_outcome_event_id=event_id_from_node(
                    self.store.graph,
                    event_nodes[-1],
                ),
                original_outcome_event_name=event_name_from_node(
                    self.store.graph,
                    event_nodes[-1],
                ),
                original_event_nodes=event_nodes,
                original_event_ids=[
                    event_id_from_node(self.store.graph, node)
                    for node in event_nodes
                ],
                original_event_names=[
                    event_name_from_node(self.store.graph, node)
                    for node in event_nodes
                ],
                original_rule_ids=rule_ids,
                original_article_ids=article_ids,
                original_path_score=0.0,
                original_hop_count=len(event_nodes) - 1,
                intervention_type="STRUCTURAL_PATH_VALIDATION",
                mediator_interventions=[],
                status="CONTRADICTED",
                consistency_score=0.10,
                explanation=(
                    "Chuỗi EVENT không tạo thành các CAUSES edge liên tiếp "
                    "trong causal event graph, kể cả sau khi chuẩn hóa hướng."
                ),
            )

        if path_score <= 0.0:
            path_score = structural_path_score
        else:
            # graph_score của Step 3 có chứa semantic seed score. Giữ lại tín
            # hiệu đó nhưng không cho phép nó vượt qua kiểm tra cấu trúc.
            path_score = clamp(
                0.55 * path_score
                + 0.45 * structural_path_score
            )

        base_kwargs = dict(
            original_path_id=path_id,
            seed_event_id=event_id_from_node(self.store.graph, seed_node),
            seed_event_name=event_name_from_node(self.store.graph, seed_node),
            original_outcome_event_id=event_id_from_node(self.store.graph, outcome_node),
            original_outcome_event_name=event_name_from_node(self.store.graph, outcome_node),
            original_event_nodes=event_nodes,
            original_event_ids=[event_id_from_node(self.store.graph, n) for n in event_nodes],
            original_event_names=[event_name_from_node(self.store.graph, n) for n in event_nodes],
            original_rule_ids=rule_ids,
            original_article_ids=article_ids,
            original_path_score=path_score,
            original_hop_count=len(event_nodes) - 1,
        )

        # Một hop không có mediator để loại, nhưng đây vẫn là causal edge hợp lệ.
        if len(event_nodes) == 2:
            edge_exists = self.store.causal_event_graph.has_edge(seed_node, outcome_node)
            score = clamp(max(DIRECT_PATH_SUPPORT_SCORE, path_score)) if edge_exists else 0.10
            return PathVerification(
                **base_kwargs,
                intervention_type="DIRECT_EDGE_VALIDATION",
                mediator_interventions=[],
                status="SUPPORTED" if edge_exists else "CONTRADICTED",
                consistency_score=score,
                explanation=(
                    "Path trực tiếp một hop; không có mediator để can thiệp. "
                    "CAUSES edge tồn tại trong causal event graph."
                    if edge_exists else
                    "Path có hai EVENT node nhưng không tồn tại CAUSES edge tương ứng."
                ),
            )

        mediator_nodes = event_nodes[1:-1][:max_mediators]
        interventions: list[MediatorIntervention] = []

        for mediator_index, mediator_node in enumerate(mediator_nodes, start=1):
            alternatives = self.searcher.find_alternative_paths(
                start_node=seed_node,
                end_node=outcome_node,
                removed_nodes=[mediator_node],
                max_hops=max_hops,
                max_paths=max_paths,
            )
            best_score = alternatives[0].path_score if alternatives else 0.0

            if not alternatives:
                status = "NECESSARY"
                necessity_score = NECESSARY_MEDIATOR_SCORE
                explanation = "Loại mediator làm outcome không còn reachable trong giới hạn tìm kiếm."
            elif best_score >= alternative_path_threshold:
                status = "NON_NECESSARY"
                necessity_score = NON_NECESSARY_SCORE
                explanation = "Tồn tại đường thay thế đủ mạnh sau khi loại mediator."
            else:
                status = "PARTIALLY_NECESSARY"
                necessity_score = PARTIALLY_NECESSARY_SCORE
                explanation = "Có đường thay thế nhưng structural support thấp hơn ngưỡng."

            interventions.append(MediatorIntervention(
                mediator_index=mediator_index,
                mediator_event_node=mediator_node,
                mediator_event_id=event_id_from_node(self.store.graph, mediator_node),
                mediator_event_name=event_name_from_node(self.store.graph, mediator_node),
                removed_nodes=[mediator_node],
                alternative_paths=[asdict(path) for path in alternatives],
                best_alternative_path_score=best_score,
                intervention_status=status,
                necessity_score=necessity_score,
                explanation=explanation,
            ))

        if not interventions:
            return PathVerification(
                **base_kwargs,
                intervention_type="REMOVE_MEDIATOR",
                mediator_interventions=[],
                status="UNRESOLVED",
                consistency_score=UNRESOLVED_BASE_SCORE,
                explanation="Không tìm được mediator hợp lệ để thực hiện intervention.",
            )

        non_necessary = sum(item.intervention_status == "NON_NECESSARY" for item in interventions)
        partially = sum(item.intervention_status == "PARTIALLY_NECESSARY" for item in interventions)
        necessary = sum(item.intervention_status == "NECESSARY" for item in interventions)
        average_necessity = sum(item.necessity_score for item in interventions) / len(interventions)
        consistency = clamp(0.55 * average_necessity + 0.45 * path_score)

        # Một đường thay thế không phủ định sự tồn tại của path gốc.
        # Vì vậy path có các CAUSES edge hợp lệ luôn được xem là SUPPORTED.
        # Trạng thái NECESSARY/NON_NECESSARY chỉ được dùng ở tầng claim-aware
        # để đánh giá phát biểu như “mediator là bắt buộc” hoặc
        # “bỏ mediator nhưng outcome vẫn xảy ra”.
        status = "SUPPORTED"

        if non_necessary > 0:
            explanation = (
                f"Path gốc hợp lệ; {non_necessary}/{len(interventions)} mediator "
                "không phải mắt xích duy nhất vì tồn tại đường thay thế."
            )
        elif necessary > 0 or partially > 0:
            explanation = (
                f"Path gốc hợp lệ; intervention cho thấy {necessary} mediator "
                f"cần thiết và {partially} mediator cần thiết một phần."
            )
        else:
            explanation = (
                "Path gốc hợp lệ nhưng tín hiệu về tính cần thiết của mediator "
                "không đủ mạnh."
            )

        # Không để đường thay thế kéo consistency của một path hợp lệ xuống quá thấp.
        consistency = clamp(max(0.52, consistency))

        return PathVerification(
            **base_kwargs,
            intervention_type="REMOVE_MEDIATOR",
            mediator_interventions=[asdict(item) for item in interventions],
            status=status,
            consistency_score=consistency,
            explanation=explanation,
        )

    @staticmethod
    def _unresolved(*, path_id: int, explanation: str) -> PathVerification:
        return PathVerification(
            original_path_id=path_id,
            seed_event_id="",
            seed_event_name="",
            original_outcome_event_id="",
            original_outcome_event_name="",
            original_event_nodes=[],
            original_event_ids=[],
            original_event_names=[],
            original_rule_ids=[],
            original_article_ids=[],
            original_path_score=0.0,
            original_hop_count=0,
            intervention_type="REMOVE_MEDIATOR",
            mediator_interventions=[],
            status="UNRESOLVED",
            consistency_score=UNRESOLVED_BASE_SCORE,
            explanation=explanation,
        )


# ============================================================
# LEGAL SCM STRUCTURAL VERIFICATION
# ============================================================

class StructuralCounterfactualVerifier:
    """Run LegalSCM while preserving node-deletion output as diagnostics."""

    METHOD = "legal_scm_do_intervention"

    def __init__(
        self,
        *,
        store: CounterfactualResourceStore,
        baseline_verifier: CounterfactualPathVerifier,
    ) -> None:
        self.store = store
        self.baseline_verifier = baseline_verifier

    def verify_path(
        self,
        *,
        path_id: int,
        original_path: dict[str, Any],
        max_hops: int = DEFAULT_MAX_CF_HOPS,
        max_paths: int = DEFAULT_MAX_CF_PATHS,
        max_mediators: int = DEFAULT_MAX_MEDIATORS_PER_PATH,
        **_: Any,
    ) -> PathVerification:
        baseline = self.baseline_verifier.verify_path(
            path_id=path_id,
            original_path=original_path,
            max_hops=max_hops,
            max_paths=max_paths,
            max_mediators=max_mediators,
        )
        summary = {
            "engine": "LegalSCM",
            "mode": STRUCTURAL_SCM_MODE,
            "factual_context": {},
            "outcome_event_id": baseline.original_outcome_event_id,
            "factual_outcome": "unknown",
            "path_rule_coverage": {},
            "interventions": [],
            "baseline": self._baseline_summary(baseline),
            "fallback_used": False,
            "fallback_reason": "",
        }

        scm = self.store.legal_scm
        if scm is None:
            return self._fallback(
                baseline,
                summary,
                self.store.scm_load_error or "LegalSCM chưa được khởi tạo.",
            )
        if baseline.status == "CONTRADICTED":
            return self._fallback(
                baseline,
                summary,
                "Path không tạo thành chuỗi CAUSES hợp lệ trong graph.",
            )
        if len(baseline.original_event_ids) < 2:
            return self._fallback(
                baseline,
                summary,
                "Path không có đủ hai event để chạy structural inference.",
            )

        event_nodes = baseline.original_event_nodes
        event_ids = baseline.original_event_ids
        seed_event_id = event_ids[0]
        outcome_event_id = event_ids[-1]
        factual_context = {seed_event_id: EventState.TRUE}
        summary["factual_context"] = {seed_event_id: EventState.TRUE.value}
        summary["outcome_event_id"] = outcome_event_id

        try:
            factual = scm.infer(factual_context)
        except Exception as error:
            return self._fallback(
                baseline,
                summary,
                f"Factual inference lỗi: {type(error).__name__}: {error}",
            )

        coverage = self._path_rule_coverage(
            event_nodes=event_nodes,
            activated_rule_ids=set(factual.activated_rule_ids),
        )
        summary["factual_outcome"] = factual.state_of(
            outcome_event_id
        ).value
        summary["factual_states"] = {
            event_id: factual.state_of(event_id).value
            for event_id in event_ids
        }
        summary["factual_iterations"] = factual.iterations
        summary["factual_conflicting_event_ids"] = [
            event_id
            for event_id in factual.conflicting_event_ids
            if event_id in set(event_ids)
        ]
        summary["path_rule_coverage"] = coverage

        factual_path_supported = (
            factual.state_of(outcome_event_id) is EventState.TRUE
            and bool(coverage.get("all_hops_covered"))
            and not summary["factual_conflicting_event_ids"]
        )

        if len(event_ids) == 2:
            baseline.intervention_type = "LEGAL_SCM_FACTUAL_VALIDATION"
            baseline.counterfactual_summary = summary
            if factual_path_supported:
                baseline.status = "SUPPORTED"
                baseline.consistency_score = clamp(
                    max(
                        DIRECT_PATH_SUPPORT_SCORE,
                        baseline.original_path_score,
                    )
                )
                baseline.explanation = (
                    "LegalSCM kích hoạt rule mechanism của path một hop và "
                    "suy ra factual outcome=TRUE; không có mediator để do-intervention."
                )
            else:
                baseline.status = "UNRESOLVED"
                baseline.consistency_score = UNRESOLVED_BASE_SCORE
                baseline.explanation = (
                    "Graph có direct edge nhưng LegalSCM không xác nhận được "
                    "factual outcome hoặc rule mechanism của hop."
                )
            return baseline

        baseline_by_mediator = {
            safe_string(item.get("mediator_event_id")): item
            for item in baseline.mediator_interventions
            if isinstance(item, dict)
        }
        mediator_ids = event_ids[1:-1][:max_mediators]
        mediator_nodes = event_nodes[1:-1][:max_mediators]

        if not factual_path_supported:
            baseline.intervention_type = "LEGAL_SCM_DO_INTERVENTION"
            baseline.status = "UNRESOLVED"
            baseline.consistency_score = UNRESOLVED_BASE_SCORE
            baseline.explanation = (
                "LegalSCM không xác nhận được factual chain trước intervention; "
                "không gán nhãn necessity từ topology đơn thuần."
            )
            baseline.mediator_interventions = [
                asdict(
                    self._unresolved_intervention(
                        mediator_index=index,
                        mediator_node=mediator_node,
                        mediator_id=mediator_id,
                        baseline_item=baseline_by_mediator.get(
                            mediator_id,
                            {},
                        ),
                        factual=factual,
                        outcome_event_id=outcome_event_id,
                        explanation=(
                            "Factual outcome/rule coverage không đủ để thực hiện "
                            "structural necessity test."
                        ),
                    )
                )
                for index, (mediator_node, mediator_id) in enumerate(
                    zip(mediator_nodes, mediator_ids),
                    start=1,
                )
            ]
            summary["interventions"] = [
                self._compact_intervention(item)
                for item in baseline.mediator_interventions
            ]
            baseline.counterfactual_summary = summary
            return baseline

        structural_interventions: list[MediatorIntervention] = []
        for mediator_index, (mediator_node, mediator_id) in enumerate(
            zip(mediator_nodes, mediator_ids),
            start=1,
        ):
            baseline_item = baseline_by_mediator.get(mediator_id, {})
            try:
                result = scm.counterfactual(
                    factual_context,
                    interventions={mediator_id: EventState.FALSE},
                    outcome_event_id=outcome_event_id,
                    recompute_endogenous_context=False,
                    recompute_outcome=True,
                )
                intervention = self._from_scm_result(
                    mediator_index=mediator_index,
                    mediator_node=mediator_node,
                    mediator_id=mediator_id,
                    baseline_item=baseline_item,
                    result=result,
                )
            except Exception as error:
                intervention = self._unresolved_intervention(
                    mediator_index=mediator_index,
                    mediator_node=mediator_node,
                    mediator_id=mediator_id,
                    baseline_item=baseline_item,
                    factual=factual,
                    outcome_event_id=outcome_event_id,
                    explanation=(
                        "LegalSCM intervention lỗi: "
                        f"{type(error).__name__}: {error}"
                    ),
                )
            structural_interventions.append(intervention)

        if not structural_interventions:
            return self._fallback(
                baseline,
                summary,
                "Không xác định được mediator để chạy LegalSCM.",
            )

        baseline.intervention_type = "LEGAL_SCM_DO_INTERVENTION"
        baseline.mediator_interventions = [
            asdict(item) for item in structural_interventions
        ]
        baseline.status = "SUPPORTED"
        average_necessity = sum(
            item.necessity_score for item in structural_interventions
        ) / len(structural_interventions)
        baseline.consistency_score = clamp(
            max(
                0.52,
                0.55 * average_necessity
                + 0.45 * baseline.original_path_score,
            )
        )

        counts = {
            status: sum(
                item.intervention_status == status
                for item in structural_interventions
            )
            for status in ("NECESSARY", "NON_NECESSARY", "UNRESOLVED")
        }
        baseline.explanation = (
            "Factual path được LegalSCM xác nhận; hard do-intervention cho thấy "
            f"{counts['NECESSARY']} mediator cần thiết, "
            f"{counts['NON_NECESSARY']} không cần thiết và "
            f"{counts['UNRESOLVED']} chưa xác định."
        )
        summary["interventions"] = [
            self._compact_intervention(item)
            for item in baseline.mediator_interventions
        ]
        baseline.counterfactual_summary = summary
        return baseline

    def _path_rule_coverage(
        self,
        *,
        event_nodes: list[str],
        activated_rule_ids: set[str],
    ) -> dict[str, Any]:
        scm = self.store.legal_scm
        per_hop: list[dict[str, Any]] = []
        all_expected: list[str] = []
        all_activated: list[str] = []
        all_missing: list[str] = []

        for hop, (source, target) in enumerate(
            zip(event_nodes[:-1], event_nodes[1:]),
            start=1,
        ):
            edge = self.store.causal_event_graph[source][target]
            expected = ensure_string_list(edge.get("rule_ids"))
            known = [
                rule_id
                for rule_id in expected
                if scm is not None and rule_id in scm.rule_by_id
            ]
            missing = [
                rule_id
                for rule_id in expected
                if scm is None or rule_id not in scm.rule_by_id
            ]
            activated = [
                rule_id for rule_id in known if rule_id in activated_rule_ids
            ]
            per_hop.append({
                "hop": hop,
                "source_event_id": event_id_from_node(
                    self.store.graph,
                    source,
                ),
                "target_event_id": event_id_from_node(
                    self.store.graph,
                    target,
                ),
                "expected_rule_ids": expected,
                "activated_rule_ids": activated,
                "missing_rule_ids": missing,
                "covered": bool(activated),
            })
            all_expected.extend(expected)
            all_activated.extend(activated)
            all_missing.extend(missing)

        return {
            "all_hops_covered": bool(per_hop)
            and all(item["covered"] for item in per_hop),
            "expected_rule_ids": unique_preserve_order(all_expected),
            "activated_rule_ids": unique_preserve_order(all_activated),
            "missing_rule_ids": unique_preserve_order(all_missing),
            "per_hop": per_hop,
        }

    def _from_scm_result(
        self,
        *,
        mediator_index: int,
        mediator_node: str,
        mediator_id: str,
        baseline_item: dict[str, Any],
        result: Any,
    ) -> MediatorIntervention:
        factual_mediator = result.factual.state_of(mediator_id)
        factual_outcome = result.factual_outcome
        counterfactual_outcome = result.counterfactual_outcome

        if (
            result.status is CounterfactualStatus.NECESSARY
            and factual_mediator is EventState.TRUE
            and factual_outcome is EventState.TRUE
            and counterfactual_outcome is EventState.FALSE
        ):
            status = "NECESSARY"
            necessity_score = NECESSARY_MEDIATOR_SCORE
            explanation = (
                "LegalSCM: do(mediator=FALSE) làm outcome đổi TRUE→FALSE."
            )
        elif (
            result.status is CounterfactualStatus.NON_NECESSARY
            and factual_mediator is EventState.TRUE
            and factual_outcome is EventState.TRUE
            and counterfactual_outcome is EventState.TRUE
        ):
            status = "NON_NECESSARY"
            necessity_score = NON_NECESSARY_SCORE
            explanation = (
                "LegalSCM: sau do(mediator=FALSE), outcome vẫn TRUE nhờ "
                "mechanism khác đang hoạt động."
            )
        else:
            status = "UNRESOLVED"
            necessity_score = UNRESOLVED_BASE_SCORE
            explanation = (
                "LegalSCM không đủ điều kiện gán necessity: "
                f"status={result.status.value}, "
                f"mediator={factual_mediator.value}, "
                f"outcome={factual_outcome.value}→{counterfactual_outcome.value}."
            )

        return MediatorIntervention(
            mediator_index=mediator_index,
            mediator_event_node=mediator_node,
            mediator_event_id=mediator_id,
            mediator_event_name=event_name_from_node(
                self.store.graph,
                mediator_node,
            ),
            removed_nodes=[],
            alternative_paths=list(
                baseline_item.get("alternative_paths") or []
            ),
            best_alternative_path_score=safe_float(
                baseline_item.get("best_alternative_path_score")
            ),
            intervention_status=status,
            necessity_score=necessity_score,
            explanation=explanation,
            verification_method=self.METHOD,
            intervention_assignment={mediator_id: EventState.FALSE.value},
            structural_status=result.status.value,
            factual_mediator_state=factual_mediator.value,
            factual_outcome=factual_outcome.value,
            counterfactual_outcome=counterfactual_outcome.value,
            outcome_changed=result.outcome_changed,
            disabled_rule_ids=list(result.disabled_rule_ids),
            newly_activated_rule_ids=list(result.newly_activated_rule_ids),
            alternative_outcome_rule_ids=list(
                result.alternative_outcome_rule_ids
            ),
            recomputed_context_event_ids=list(
                result.recomputed_context_event_ids
            ),
        )

    def _unresolved_intervention(
        self,
        *,
        mediator_index: int,
        mediator_node: str,
        mediator_id: str,
        baseline_item: dict[str, Any],
        factual: Any,
        outcome_event_id: str,
        explanation: str,
    ) -> MediatorIntervention:
        return MediatorIntervention(
            mediator_index=mediator_index,
            mediator_event_node=mediator_node,
            mediator_event_id=mediator_id,
            mediator_event_name=event_name_from_node(
                self.store.graph,
                mediator_node,
            ),
            removed_nodes=[],
            alternative_paths=list(
                baseline_item.get("alternative_paths") or []
            ),
            best_alternative_path_score=safe_float(
                baseline_item.get("best_alternative_path_score")
            ),
            intervention_status="UNRESOLVED",
            necessity_score=UNRESOLVED_BASE_SCORE,
            explanation=explanation,
            verification_method=self.METHOD,
            intervention_assignment={mediator_id: EventState.FALSE.value},
            structural_status=CounterfactualStatus.INDETERMINATE.value,
            factual_mediator_state=factual.state_of(mediator_id).value,
            factual_outcome=factual.state_of(outcome_event_id).value,
            counterfactual_outcome=EventState.UNKNOWN.value,
        )

    @staticmethod
    def _compact_intervention(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "mediator_event_id": safe_string(item.get("mediator_event_id")),
            "assignment": dict(item.get("intervention_assignment") or {}),
            "status": safe_string(item.get("structural_status")),
            "mapped_status": safe_string(item.get("intervention_status")),
            "factual_mediator_state": safe_string(
                item.get("factual_mediator_state")
            ),
            "factual_outcome": safe_string(item.get("factual_outcome")),
            "counterfactual_outcome": safe_string(
                item.get("counterfactual_outcome")
            ),
            "outcome_changed": item.get("outcome_changed"),
            "disabled_rule_ids": list(item.get("disabled_rule_ids") or []),
            "newly_activated_rule_ids": list(
                item.get("newly_activated_rule_ids") or []
            ),
            "alternative_outcome_rule_ids": list(
                item.get("alternative_outcome_rule_ids") or []
            ),
        }

    @staticmethod
    def _baseline_summary(baseline: PathVerification) -> dict[str, Any]:
        return {
            "method": "node_deletion_reachability",
            "path_status": baseline.status,
            "interventions": [
                {
                    "mediator_event_id": safe_string(
                        item.get("mediator_event_id")
                    ),
                    "status": safe_string(
                        item.get("intervention_status")
                    ),
                    "alternative_path_count": len(
                        item.get("alternative_paths") or []
                    ),
                    "best_alternative_path_score": safe_float(
                        item.get("best_alternative_path_score")
                    ),
                }
                for item in baseline.mediator_interventions
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _fallback(
        baseline: PathVerification,
        summary: dict[str, Any],
        reason: str,
    ) -> PathVerification:
        summary["fallback_used"] = True
        summary["fallback_reason"] = reason
        baseline.counterfactual_summary = summary
        baseline.explanation = (
            baseline.explanation
            + " Structural SCM fallback: "
            + reason
        ).strip()
        return baseline


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

class EvidenceVerifier:
    def __init__(self, store: CounterfactualResourceStore) -> None:
        self.store = store

    def verify_all(
        self,
        *,
        path_verifications: list[PathVerification],
        keep_threshold: float,
        reject_threshold: float,
        verified_top_k: int,
    ) -> tuple[list[EvidenceVerification], list[EvidenceVerification], list[EvidenceVerification]]:
        verified: list[EvidenceVerification] = []
        uncertain: list[EvidenceVerification] = []
        removed: list[EvidenceVerification] = []

        for index, evidence in enumerate(self.store.retrieval_result.get("evidence", []), start=1):
            result = self._verify_one(index, evidence, path_verifications, keep_threshold, reject_threshold)
            if result.decision == "KEEP":
                verified.append(result)
            elif result.decision == "REMOVE":
                removed.append(result)
            else:
                uncertain.append(result)

        verified.sort(key=lambda item: item.verification_score, reverse=True)
        uncertain.sort(key=lambda item: item.verification_score, reverse=True)
        removed.sort(key=lambda item: item.verification_score)
        return verified[:verified_top_k], uncertain, removed

    def _verify_one(
        self,
        rank: int,
        evidence: dict[str, Any],
        paths: list[PathVerification],
        keep_threshold: float,
        reject_threshold: float,
    ) -> EvidenceVerification:
        rule_id = safe_string(evidence.get("rule_id"))
        related = [p for p in paths if rule_id and rule_id in p.original_rule_ids]
        if not related:
            raw_path_ids = {safe_int(x, -1) for x in ensure_string_list(evidence.get("path_ids"))}
            related = [p for p in paths if p.original_path_id in raw_path_ids]

        supported = [p.original_path_id for p in related if p.status == "SUPPORTED"]
        contradicted = [p.original_path_id for p in related if p.status == "CONTRADICTED"]
        unresolved = [p.original_path_id for p in related if p.status == "UNRESOLVED"]

        path_support = (
            sum(p.original_path_score for p in related) / len(related)
            if related else safe_float(evidence.get("path_score", evidence.get("graph_score")), 0.0)
        )
        cf_support = (
            sum(p.consistency_score for p in related) / len(related)
            if related else UNRESOLVED_BASE_SCORE
        )
        semantic = safe_float(evidence.get("semantic_score", evidence.get("similarity_score")), 0.0)
        graph_score = safe_float(evidence.get("graph_score", evidence.get("causal_score")), 0.0)
        original_final = safe_float(evidence.get("final_score", evidence.get("evidence_score")), 0.0)

        score = clamp(
            PATH_SUPPORT_WEIGHT * path_support
            + COUNTERFACTUAL_SUPPORT_WEIGHT * cf_support
            + SEMANTIC_EVIDENCE_WEIGHT * semantic
            + GRAPH_EVIDENCE_WEIGHT * graph_score
        )

        reasons: list[str] = []
        if related and len(contradicted) / len(related) >= 0.5:
            decision = "REMOVE"
            reasons.append("Ít nhất một nửa path liên quan bị CONTRADICTED.")
        elif supported and score >= keep_threshold:
            decision = "KEEP"
            reasons.append("Có path SUPPORTED và verification score đạt keep threshold.")
        elif score < reject_threshold and not supported:
            decision = "REMOVE"
            reasons.append("Verification score thấp hơn reject threshold và không có path SUPPORTED.")
        else:
            decision = "UNCERTAIN"
            reasons.append("Chưa đủ điều kiện KEEP hoặc REMOVE.")

        if not related:
            reasons.append("Evidence không ánh xạ được tới causal path bằng rule_id/path_ids.")

        return EvidenceVerification(
            original_rank=safe_int(evidence.get("rank"), rank),
            rule_id=rule_id,
            article_id=safe_string(evidence.get("article_id")),
            original_final_score=original_final,
            semantic_score=semantic,
            graph_score=graph_score,
            path_support_score=path_support,
            counterfactual_support_score=cf_support,
            verification_score=score,
            decision=decision,
            verified_path_ids=supported,
            rejected_path_ids=contradicted,
            unresolved_path_ids=unresolved,
            reasons=reasons,
            original_evidence=evidence,
        )


# ============================================================
# PRIMARY PATH SELECTION + QUERY-AWARE CLAIM VERIFICATION
# ============================================================

class PrimaryPathSelector:
    """Chọn path đại diện thay vì lấy trung bình mọi candidate path."""

    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    def select(
        self,
        path_results: list[PathVerification],
        *,
        target_hops: int = DEFAULT_TARGET_HOPS,
        top_k: int = 1,
    ) -> list[int]:
        if not path_results:
            return []

        def status_priority(item: PathVerification) -> int:
            return {
                "SUPPORTED": 2,
                "UNRESOLVED": 1,
                "CONTRADICTED": 0,
            }.get(item.status, 0)

        ranked = sorted(
            path_results,
            key=lambda item: (
                status_priority(item),
                item.original_hop_count == target_hops,
                min(item.original_hop_count, target_hops),
                item.consistency_score,
                item.original_path_score,
                -item.original_path_id,
            ),
            reverse=True,
        )

        selected = [
            item.original_path_id
            for item in ranked
            if item.status != "CONTRADICTED"
        ][:max(1, top_k)]

        if selected:
            return selected

        return [ranked[0].original_path_id]


class QueryAwareClaimVerifier:
    """Đánh giá claim của câu hỏi dựa trên primary path và intervention.

    Đây là tầng còn thiếu trong bản cũ. Path verification chỉ trả lời
    “chuỗi edge có hợp lệ không”; lớp này trả lời phát biểu cụ thể trong query.
    """

    INTENT_DETECTION_VERSION = "2.0-explicit-intervention"

    DIRECT_TERMS = (
        "trực tiếp",
        "không qua trung gian",
        "không cần qua",
        "ngay lập tức dẫn",
    )
    REMOVE_TERMS = (
        "nếu loại bỏ",
        "nếu loại",
        "nếu bỏ",
        "khi loại bỏ",
        "khi bỏ",
        "nếu không có",
        "khi không có",
        "giả sử không có",
        "trong trường hợp không có",
        "nếu không xảy ra",
    )
    REMOVE_PATTERNS = (
        r"\b(?:nếu|khi|giả sử)\s+(?:loại\s+bỏ|loại|bỏ)\b",
        r"\b(?:nếu|khi|giả sử|trong\s+trường\s+hợp)\s+không\s+có\b",
        # Match “Nếu M không xảy ra ...” nhưng không match
        # “Nếu A thì Y không xảy ra ...”.
        r"\b(?:nếu|giả\s+sử)\s+(?:(?!\bthì\b).){1,160}?\bkhông\s+(?:xảy\s+ra|tồn\s+tại|hiện\s+diện|được\s+thực\s+hiện)\b",
        r"\bdo\s*\([^)]*=\s*(?:false|0)\s*\)",
    )
    REMAINS_TERMS = (
        "vẫn xảy ra",
        "vẫn dẫn đến",
        "vẫn xuất hiện",
        "vẫn có",
        "có còn xảy ra",
        "tiếp tục xảy ra",
    )
    DISAPPEARS_TERMS = (
        "không còn xảy ra",
        "không còn dẫn đến",
        "không xảy ra nữa",
        "sẽ không xảy ra",
        "biến mất",
        "chấm dứt",
    )
    NECESSITY_CLAIM_TERMS = (
        "có cần thiết",
        "có thực sự cần thiết",
        "có bắt buộc",
        "có phải là điều kiện",
        "có phải là mắt xích",
        "là điều kiện cần thiết",
        "là điều kiện bắt buộc",
        "là mắt xích bắt buộc",
        "là mắt xích duy nhất",
        "điều kiện duy nhất để",
        "không thể thiếu để",
    )
    NECESSITY_CLAIM_PATTERNS = (
        r"\bcó\s+(?:thực\s+sự\s+)?(?:cần\s+thiết|bắt\s+buộc)\b",
        r"\bcó\s+phải\s+(?:là\s+)?[^?.!]{0,120}\b(?:điều\s+kiện|mắt\s+xích)\b[^?.!]{0,80}\b(?:cần\s+thiết|bắt\s+buộc|duy\s+nhất)\b",
        r"\blà\s+(?:một\s+)?(?:điều\s+kiện|mắt\s+xích)\s+(?:cần\s+thiết|bắt\s+buộc|duy\s+nhất)\b",
        r"\b(?:điều\s+kiện|mắt\s+xích)\s+duy\s+nhất\b",
        r"\bkhông\s+thể\s+thiếu\s+để\b",
    )
    EXPLICIT_COUNTERFACTUAL_TERMS = (
        "phản thực tế",
        "counterfactual",
        "can thiệp",
        "intervention",
    )
    EXPLICIT_COUNTERFACTUAL_PATTERNS = (
        r"\bdo\s*\(",
    )
    CONDITIONAL_TERMS = (
        "nếu",
        "giả sử",
        "trong trường hợp",
    )

    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    @staticmethod
    def _normalize(text: Any) -> str:
        value = safe_string(text).lower()
        return " ".join(value.split())

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _matches_any_pattern(
        text: str,
        patterns: Iterable[str],
    ) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def analyze_query(
        self,
        query: str,
    ) -> dict[str, Any]:
        normalized = self._normalize(query)

        has_direct = self._contains_any(normalized, self.DIRECT_TERMS)
        has_remove = (
            self._contains_any(normalized, self.REMOVE_TERMS)
            or self._matches_any_pattern(
                normalized,
                self.REMOVE_PATTERNS,
            )
        )
        has_disappears = self._contains_any(
            normalized,
            self.DISAPPEARS_TERMS,
        )
        # “không còn xảy ra” chứa substring “còn xảy ra”; tín hiệu phủ định
        # phải thắng để không đảo claim thành OUTCOME_REMAINS.
        has_remains = (
            not has_disappears
            and self._contains_any(normalized, self.REMAINS_TERMS)
        )
        has_necessary = (
            self._contains_any(
                normalized,
                self.NECESSITY_CLAIM_TERMS,
            )
            or self._matches_any_pattern(
                normalized,
                self.NECESSITY_CLAIM_PATTERNS,
            )
        )
        has_explicit_counterfactual = (
            self._contains_any(
                normalized,
                self.EXPLICIT_COUNTERFACTUAL_TERMS,
            )
            or self._matches_any_pattern(
                normalized,
                self.EXPLICIT_COUNTERFACTUAL_PATTERNS,
            )
        )
        has_conditional = self._contains_any(
            normalized,
            self.CONDITIONAL_TERMS,
        )
        has_counterfactual = has_remove or has_explicit_counterfactual
        conditional_antecedent_only = (
            has_conditional and not has_counterfactual
        )

        # Explicit intervention có ưu tiên cao nhất. Một antecedent “Nếu A”
        # đơn thuần chỉ mô tả factual context và vẫn là STANDARD_CAUSAL.
        if has_remove and has_disappears:
            claim_type = "REMOVE_MEDIATOR_OUTCOME_DISAPPEARS"
        elif has_remove and has_remains:
            claim_type = "REMOVE_MEDIATOR_OUTCOME_REMAINS"
        elif has_necessary:
            claim_type = "MEDIATOR_NECESSARY_CLAIM"
        elif has_remove:
            claim_type = "COUNTERFACTUAL_UNSPECIFIED"
        elif has_direct:
            claim_type = "DIRECT_CAUSAL_CLAIM"
        elif has_explicit_counterfactual:
            claim_type = "COUNTERFACTUAL_UNSPECIFIED"
        else:
            claim_type = "STANDARD_CAUSAL"

        return {
            "normalized_query": normalized,
            "claim_type": claim_type,
            "intent_detection_version": self.INTENT_DETECTION_VERSION,
            "has_direct_indicator": has_direct,
            "has_remove_indicator": has_remove,
            "has_remains_indicator": has_remains,
            "has_disappears_indicator": has_disappears,
            "has_necessary_indicator": has_necessary,
            "has_counterfactual_indicator": has_counterfactual,
            "has_explicit_counterfactual_indicator": (
                has_explicit_counterfactual
            ),
            "has_conditional_indicator": has_conditional,
            "conditional_antecedent_only": conditional_antecedent_only,
        }

    def _select_mediator_intervention(
        self,
        query: str,
        primary: PathVerification,
    ) -> Optional[dict[str, Any]]:
        interventions = primary.mediator_interventions or []
        if not interventions:
            return None

        normalized_query = self._normalize(query)

        for item in interventions:
            mediator_name = self._normalize(
                item.get("mediator_event_name")
            )
            mediator_id = self._normalize(
                item.get("mediator_event_id")
            )

            if (
                mediator_name
                and mediator_name in normalized_query
            ) or (
                mediator_id
                and mediator_id in normalized_query
            ):
                return item

        # Benchmark hai hop thường chỉ có một mediator.
        return interventions[0]

    def verify(
        self,
        *,
        query: str,
        path_results: list[PathVerification],
        primary_path_ids: list[int],
    ) -> tuple[str, float, str, dict[str, Any]]:
        analysis = self.analyze_query(query)
        claim_type = analysis["claim_type"]

        by_id = {
            item.original_path_id: item
            for item in path_results
        }
        primary = next(
            (
                by_id[path_id]
                for path_id in primary_path_ids
                if path_id in by_id
            ),
            None,
        )

        if primary is None:
            explanation = (
                "Không chọn được primary path hợp lệ để xác minh claim."
            )
            analysis.update({
                "matched_mediator_id": "",
                "matched_mediator_name": "",
            })
            return "UNCERTAIN", UNRESOLVED_BASE_SCORE, explanation, analysis

        primary_summary = (
            primary.counterfactual_summary
            if isinstance(primary.counterfactual_summary, dict)
            else {}
        )
        structural_summary = (
            primary_summary.get("engine") == "LegalSCM"
        )
        structural_fallback = bool(
            primary_summary.get("fallback_used")
        )
        path_verification_method = (
            StructuralCounterfactualVerifier.METHOD
            if structural_summary and not structural_fallback
            else "node_deletion_reachability"
        )
        analysis.update({
            "counterfactual_mode": safe_string(
                primary_summary.get("mode")
            ) or PATH_ABLATION_MODE,
            "path_verification_method": path_verification_method,
            "structural_fallback_used": structural_fallback,
            "structural_fallback_reason": safe_string(
                primary_summary.get("fallback_reason")
            ),
            "factual_outcome": safe_string(
                primary_summary.get("factual_outcome")
            ),
        })

        base_score = clamp(
            max(primary.consistency_score, primary.original_path_score)
        )

        if primary.status == "CONTRADICTED":
            explanation = (
                "Primary path không tạo thành chuỗi CAUSES hợp lệ trong graph."
            )
            return "REJECT_DIRECT_CLAIM", max(0.60, 1.0 - base_score), explanation, analysis

        if claim_type == "STANDARD_CAUSAL":
            if primary.status == "SUPPORTED":
                explanation = (
                    "Primary multi-hop path hợp lệ và hỗ trợ quan hệ "
                    "nguyên nhân → hệ quả được hỏi."
                )
                return "SUPPORTED", max(0.55, base_score), explanation, analysis

            return (
                "UNCERTAIN",
                max(UNRESOLVED_BASE_SCORE, base_score),
                "Primary path chưa được xác minh đầy đủ.",
                analysis,
            )

        if claim_type == "DIRECT_CAUSAL_CLAIM":
            start_node = (
                primary.original_event_nodes[0]
                if primary.original_event_nodes
                else ""
            )
            end_node = (
                primary.original_event_nodes[-1]
                if primary.original_event_nodes
                else ""
            )
            direct_edge = bool(
                start_node
                and end_node
                and self.store.causal_event_graph.has_edge(
                    start_node,
                    end_node,
                )
            )

            if direct_edge:
                return (
                    "SUPPORTED",
                    max(0.65, base_score),
                    "Graph có CAUSES edge trực tiếp giữa nguyên nhân và hệ quả.",
                    analysis,
                )

            return (
                "REJECT_DIRECT_CLAIM",
                max(0.65, base_score),
                "Graph chỉ hỗ trợ chuỗi qua mediator, không hỗ trợ quan hệ trực tiếp.",
                analysis,
            )

        intervention = self._select_mediator_intervention(
            query,
            primary,
        )

        if intervention is None:
            analysis.update({
                "matched_mediator_id": "",
                "matched_mediator_name": "",
            })
            return (
                "UNCERTAIN",
                UNRESOLVED_BASE_SCORE,
                "Không xác định được mediator để thực hiện counterfactual intervention.",
                analysis,
            )

        intervention_status = safe_string(
            intervention.get("intervention_status")
        ).upper()
        alternative_paths = intervention.get("alternative_paths") or []
        verification_method = safe_string(
            intervention.get("verification_method")
        ) or "node_deletion_reachability"
        is_structural = (
            verification_method
            == StructuralCounterfactualVerifier.METHOD
        )
        structural_status = safe_string(
            intervention.get("structural_status")
        ).upper()
        factual_mediator_state = safe_string(
            intervention.get("factual_mediator_state")
        ).lower()
        factual_outcome = safe_string(
            intervention.get("factual_outcome")
        ).lower()
        counterfactual_outcome = safe_string(
            intervention.get("counterfactual_outcome")
        ).lower()

        # Với LegalSCM, topology chỉ là diagnostic. Tín hiệu quyết định là
        # trạng thái outcome trong thế giới được suy luận lại sau hard do().
        # Baseline cũ tiếp tục dùng sự tồn tại của alternative path.
        has_alternative = (
            counterfactual_outcome == EventState.TRUE.value
            if is_structural
            else bool(alternative_paths)
        )
        outcome_disappears = (
            counterfactual_outcome == EventState.FALSE.value
            if is_structural
            else (
                intervention_status == "NECESSARY"
                and not has_alternative
            )
        )

        analysis.update({
            "matched_mediator_id": safe_string(
                intervention.get("mediator_event_id")
            ),
            "matched_mediator_name": safe_string(
                intervention.get("mediator_event_name")
            ),
            "intervention_status": intervention_status,
            "verification_method": verification_method,
            "structural_result_used": is_structural,
            "structural_status": structural_status,
            "factual_mediator_state": factual_mediator_state,
            "factual_outcome": factual_outcome,
            "counterfactual_outcome": counterfactual_outcome,
            "outcome_changed": intervention.get("outcome_changed"),
            "counterfactual_outcome_remains": has_alternative,
            "counterfactual_signal_source": (
                "legal_scm_world_state"
                if is_structural
                else "node_deletion_reachability"
            ),
            # Các alternative path vẫn được báo cáo để làm ablation audit.
            "alternative_path_count": len(alternative_paths),
            "baseline_alternative_path_count": len(alternative_paths),
            "best_alternative_path_score": safe_float(
                intervention.get("best_alternative_path_score")
            ),
        })

        if claim_type == "REMOVE_MEDIATOR_OUTCOME_REMAINS":
            if has_alternative or intervention_status == "NON_NECESSARY":
                explanation = (
                    "LegalSCM suy luận outcome vẫn TRUE sau "
                    "do(mediator=FALSE); mediator không cần thiết trong "
                    "factual context này."
                    if is_structural
                    else (
                        "Sau khi xóa mediator khỏi graph vẫn tồn tại causal "
                        "path thay thế tới outcome."
                    )
                )
                return (
                    "SUPPORTED",
                    max(0.65, base_score),
                    explanation,
                    analysis,
                )
            if outcome_disappears or intervention_status == "NECESSARY":
                explanation = (
                    "LegalSCM suy luận outcome đổi TRUE→FALSE sau "
                    "do(mediator=FALSE); mediator là structurally necessary "
                    "trong factual context này."
                    if is_structural
                    else (
                        "Sau khi xóa mediator, outcome không còn reachable "
                        "trong giới hạn tìm kiếm."
                    )
                )
                return (
                    "REJECT_DIRECT_CLAIM",
                    max(0.65, base_score),
                    explanation,
                    analysis,
                )

        if claim_type in {
            "REMOVE_MEDIATOR_OUTCOME_DISAPPEARS",
            "MEDIATOR_NECESSARY_CLAIM",
        }:
            if outcome_disappears and intervention_status == "NECESSARY":
                explanation = (
                    "LegalSCM xác nhận do(mediator=FALSE) làm outcome đổi "
                    "TRUE→FALSE; mediator cần thiết trong factual context."
                    if is_structural
                    else (
                        "Node-deletion không tìm thấy đường thay thế; mediator "
                        "cần thiết theo reachability baseline."
                    )
                )
                return (
                    "SUPPORTED",
                    max(0.65, base_score),
                    explanation,
                    analysis,
                )
            if has_alternative or intervention_status == "NON_NECESSARY":
                explanation = (
                    "LegalSCM suy luận outcome vẫn TRUE sau "
                    "do(mediator=FALSE), nên mediator không phải điều kiện "
                    "cần thiết trong factual context."
                    if is_structural
                    else (
                        "Node-deletion tìm thấy đường thay thế nên mediator "
                        "không phải mắt xích bắt buộc."
                    )
                )
                return (
                    "REJECT_DIRECT_CLAIM",
                    max(0.65, base_score),
                    explanation,
                    analysis,
                )

        return (
            "UNCERTAIN",
            max(UNRESOLVED_BASE_SCORE, 0.5 * base_score),
            "Câu hỏi có yếu tố phản thực tế nhưng chưa đủ tín hiệu để kết luận nhị phân.",
            analysis,
        )


def promote_primary_path_evidence(
    *,
    verified: list[EvidenceVerification],
    uncertain: list[EvidenceVerification],
    removed: list[EvidenceVerification],
    primary_path_ids: list[int],
    verified_top_k: int,
) -> tuple[
    list[EvidenceVerification],
    list[EvidenceVerification],
    list[EvidenceVerification],
]:
    """Ưu tiên KEEP các rule thuộc primary path sau khi xác minh cấu trúc."""

    primary_set = set(primary_path_ids)
    if not primary_set:
        return verified, uncertain, removed

    all_items = verified + uncertain + removed
    new_verified: list[EvidenceVerification] = []
    new_uncertain: list[EvidenceVerification] = []
    new_removed: list[EvidenceVerification] = []

    for item in all_items:
        related_ids = set(
            item.verified_path_ids
            + item.unresolved_path_ids
            + item.rejected_path_ids
        )
        belongs_to_primary = bool(related_ids & primary_set)

        if belongs_to_primary and not item.rejected_path_ids:
            item.verification_score = clamp(
                item.verification_score
                + PRIMARY_PATH_EVIDENCE_BONUS
            )
            item.decision = "KEEP"
            item.reasons.append(
                "Evidence thuộc primary multi-hop path."
            )
            new_verified.append(item)
        elif item.decision == "KEEP":
            new_verified.append(item)
        elif item.decision == "REMOVE":
            new_removed.append(item)
        else:
            new_uncertain.append(item)

    def sort_key(item: EvidenceVerification) -> tuple[float, float, int]:
        belongs = bool(
            set(item.verified_path_ids + item.unresolved_path_ids)
            & primary_set
        )
        return (
            float(belongs),
            item.verification_score,
            -item.original_rank,
        )

    new_verified.sort(key=sort_key, reverse=True)
    new_uncertain.sort(
        key=lambda item: item.verification_score,
        reverse=True,
    )
    new_removed.sort(
        key=lambda item: item.verification_score,
    )

    return (
        new_verified[:verified_top_k],
        new_uncertain,
        new_removed,
    )


# ============================================================
# END-TO-END PIPELINE
# ============================================================

class CounterfactualVerificationPipeline:
    def __init__(self, store: CounterfactualResourceStore) -> None:
        self.store = store
        self.searcher = CounterfactualGraphSearcher(store)
        self.path_verifier = CounterfactualPathVerifier(
            store=store,
            searcher=self.searcher,
        )
        self.structural_verifier = StructuralCounterfactualVerifier(
            store=store,
            baseline_verifier=self.path_verifier,
        )
        self.evidence_verifier = EvidenceVerifier(store)
        self.primary_path_selector = PrimaryPathSelector(store)
        self.claim_verifier = QueryAwareClaimVerifier(store)

    def run(
        self,
        *,
        counterfactual_mode: str = DEFAULT_COUNTERFACTUAL_MODE,
        max_cf_hops: int = DEFAULT_MAX_CF_HOPS,
        max_cf_paths: int = DEFAULT_MAX_CF_PATHS,
        verified_top_k: int = DEFAULT_VERIFIED_TOP_K,
        keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        cf_top_k: int = 5,
        mapping_top_k: int = 5,
        mapping_threshold: float = 0.42,
        **_: Any,
    ) -> VerificationResult:
        valid_modes = {STRUCTURAL_SCM_MODE, PATH_ABLATION_MODE}
        if counterfactual_mode not in valid_modes:
            choices = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"counterfactual_mode phải thuộc {{{choices}}}, "
                f"nhận được: {counterfactual_mode!r}."
            )

        selected_path_verifier = (
            self.structural_verifier
            if counterfactual_mode == STRUCTURAL_SCM_MODE
            else self.path_verifier
        )
        verification_method = (
            StructuralCounterfactualVerifier.METHOD
            if counterfactual_mode == STRUCTURAL_SCM_MODE
            else "node_deletion_reachability"
        )

        original_paths = self.store.retrieval_result.get("causal_paths", [])
        path_results: list[PathVerification] = []
        print(f"\nVerifying {len(original_paths)} causal paths...")

        for path_id, path in enumerate(original_paths):
            if not isinstance(path, dict):
                verification = self.path_verifier._unresolved(
                    path_id=path_id,
                    explanation="Causal path không phải JSON object.",
                )
            else:
                verification = selected_path_verifier.verify_path(
                    path_id=path_id,
                    original_path=path,
                    max_hops=max_cf_hops,
                    max_paths=max_cf_paths,
                )
            path_results.append(verification)
            print(f"- Path {path_id}: {verification.status} score={verification.consistency_score:.4f}")

        primary_path_ids = self.primary_path_selector.select(
            path_results,
            target_hops=safe_int(
                self.store.retrieval_result
                .get("configuration", {})
                .get("max_hops"),
                DEFAULT_TARGET_HOPS,
            ),
            top_k=1,
        )

        verified, uncertain, removed = self.evidence_verifier.verify_all(
            path_verifications=path_results,
            keep_threshold=keep_threshold,
            reject_threshold=reject_threshold,
            verified_top_k=max(verified_top_k, 1),
        )

        verified, uncertain, removed = promote_primary_path_evidence(
            verified=verified,
            uncertain=uncertain,
            removed=removed,
            primary_path_ids=primary_path_ids,
            verified_top_k=verified_top_k,
        )

        query = safe_string(
            self.store.retrieval_result.get("query")
        )
        (
            final_decision,
            decision_score,
            decision_explanation,
            query_analysis,
        ) = self.claim_verifier.verify(
            query=query,
            path_results=path_results,
            primary_path_ids=primary_path_ids,
        )

        status_counts = {
            status: sum(item.status == status for item in path_results)
            for status in ("SUPPORTED", "CONTRADICTED", "UNRESOLVED")
        }
        structural_summaries = [
            item.counterfactual_summary
            for item in path_results
            if (
                isinstance(item.counterfactual_summary, dict)
                and item.counterfactual_summary.get("engine") == "LegalSCM"
            )
        ]
        structural_paths = sum(
            not bool(summary.get("fallback_used"))
            for summary in structural_summaries
        )
        structural_fallback_paths = sum(
            bool(summary.get("fallback_used"))
            for summary in structural_summaries
        )

        primary_results = [
            item
            for item in path_results
            if item.original_path_id in set(primary_path_ids)
        ]
        consistency = (
            sum(item.consistency_score for item in primary_results)
            / len(primary_results)
            if primary_results
            else (
                sum(item.consistency_score for item in path_results)
                / len(path_results)
                if path_results
                else 0.0
            )
        )

        # Confidence bám theo quyết định claim và primary path, không còn bị
        # làm loãng bởi hàng chục path phụ.
        confidence = clamp(
            0.65 * decision_score
            + 0.35 * consistency
        )

        print(
            "Primary path IDs:",
            primary_path_ids,
        )
        print(
            "Final decision:",
            final_decision,
            f"score={decision_score:.4f}",
        )

        return VerificationResult(
            query=query,
            configuration={
                "cf_top_k": cf_top_k,
                "mapping_top_k": mapping_top_k,
                "mapping_threshold": mapping_threshold,
                "max_cf_hops": max_cf_hops,
                "max_cf_paths": max_cf_paths,
                "verified_top_k": verified_top_k,
                "keep_threshold": keep_threshold,
                "reject_threshold": reject_threshold,
                "alternative_path_threshold": DEFAULT_ALTERNATIVE_PATH_THRESHOLD,
                "counterfactual_mode": counterfactual_mode,
                "verification_method": verification_method,
                "rules_path": (
                    str(self.store.rules_path)
                    if self.store.rules_path is not None
                    else ""
                ),
                "legal_scm_loaded": self.store.legal_scm is not None,
                "scm_load_error": self.store.scm_load_error,
                "step4_version": STEP4_VERSION,
                "semantic_mapping_enabled": False,
                "model_name": self.store.model_name,
            },
            statistics={
                # Giữ cả tên trường cũ và mới để tương thích với Step 5/5.5.
                "original_paths": len(path_results),
                "total_paths": len(path_results),
                "path_status_counts": status_counts,
                "status_counts": status_counts,
                "counterfactual_mode": counterfactual_mode,
                "structural_paths": structural_paths,
                "structural_fallback_paths": structural_fallback_paths,
                "path_ablation_paths": (
                    len(path_results)
                    if counterfactual_mode == PATH_ABLATION_MODE
                    else 0
                ),
                "legal_scm_loaded": self.store.legal_scm is not None,
                "original_evidence": len(self.store.retrieval_result.get("evidence", [])),
                "total_evidence": len(self.store.retrieval_result.get("evidence", [])),
                "verified_evidence": len(verified),
                "uncertain_evidence": len(uncertain),
                "removed_evidence": len(removed),
                "primary_path_ids": primary_path_ids,
                "final_decision": final_decision,
                "decision_score": decision_score,
            },
            path_verifications=[asdict(item) for item in path_results],
            verified_evidence=[asdict(item) for item in verified],
            uncertain_evidence=[asdict(item) for item in uncertain],
            removed_evidence=[asdict(item) for item in removed],
            consistency_score=consistency,
            confidence=confidence,
            final_decision=final_decision,
            decision_score=decision_score,
            decision_explanation=decision_explanation,
            primary_path_ids=primary_path_ids,
            query_analysis=query_analysis,
            verification_method=verification_method,
        )


def run_counterfactual_verification(
    *,
    graph_path: str = GRAPH_PATH,
    memory_path: str = MEMORY_PATH,
    rules_path: Optional[str] = RULES_PATH,
    retrieval_result_path: str = RETRIEVAL_RESULT_PATH,
    output_path: str = OUTPUT_PATH,
    counterfactual_mode: str = DEFAULT_COUNTERFACTUAL_MODE,
    embeddings_path: Optional[str] = None,
    counterfactual_map_path: Optional[str] = None,
    model_name: Optional[str] = None,
    enable_semantic_mapping: bool = False,
    max_cf_hops: int = DEFAULT_MAX_CF_HOPS,
    max_cf_paths: int = DEFAULT_MAX_CF_PATHS,
    verified_top_k: int = DEFAULT_VERIFIED_TOP_K,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    **kwargs: Any,
) -> dict[str, Any]:
    store = CounterfactualResourceStore(
        graph_path=graph_path,
        memory_path=memory_path,
        retrieval_result_path=retrieval_result_path,
        rules_path=rules_path,
        embeddings_path=embeddings_path,
        counterfactual_map_path=counterfactual_map_path,
        model_name=model_name,
        enable_semantic_mapping=enable_semantic_mapping,
    )
    result = CounterfactualVerificationPipeline(store).run(
        counterfactual_mode=counterfactual_mode,
        max_cf_hops=max_cf_hops,
        max_cf_paths=max_cf_paths,
        verified_top_k=verified_top_k,
        keep_threshold=keep_threshold,
        reject_threshold=reject_threshold,
        **kwargs,
    )
    payload = json_serializable(asdict(result))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return payload


def save_result(result: VerificationResult | dict[str, Any], output_path: str) -> None:
    payload = asdict(result) if isinstance(result, VerificationResult) else result
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved verification result: {path}")


def print_summary(result: VerificationResult | dict[str, Any]) -> None:
    payload = asdict(result) if isinstance(result, VerificationResult) else result
    stats = payload.get("statistics", {})
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL VERIFICATION - MEDIATOR INTERVENTION")
    print("=" * 80)
    print("Query:", payload.get("query", ""))
    print("Step 4 version:", payload.get("configuration", {}).get("step4_version", ""))
    print("Path status:", stats.get("status_counts", {}))
    print("Primary path IDs:", payload.get("primary_path_ids", []))
    print("Final decision:", payload.get("final_decision", "UNCERTAIN"))
    print("Decision score:", f"{safe_float(payload.get('decision_score')):.4f}")
    print("Decision explanation:", payload.get("decision_explanation", ""))
    print("Consistency:", f"{safe_float(payload.get('consistency_score')):.4f}")
    print("Confidence:", f"{safe_float(payload.get('confidence')):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual verification bằng LegalSCM do-intervention "
            "hoặc node-deletion ablation."
        )
    )
    parser.add_argument("--graph", default=GRAPH_PATH)
    parser.add_argument("--memory", default=MEMORY_PATH)
    parser.add_argument("--rules", default=RULES_PATH)
    parser.add_argument("--retrieval-result", default=RETRIEVAL_RESULT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument(
        "--counterfactual-mode",
        choices=(STRUCTURAL_SCM_MODE, PATH_ABLATION_MODE),
        default=DEFAULT_COUNTERFACTUAL_MODE,
    )
    parser.add_argument("--max-cf-hops", type=int, default=DEFAULT_MAX_CF_HOPS)
    parser.add_argument("--max-cf-paths", type=int, default=DEFAULT_MAX_CF_PATHS)
    parser.add_argument("--verified-top-k", type=int, default=DEFAULT_VERIFIED_TOP_K)
    parser.add_argument("--keep-threshold", type=float, default=DEFAULT_KEEP_THRESHOLD)
    parser.add_argument("--reject-threshold", type=float, default=DEFAULT_REJECT_THRESHOLD)
    # Tham số cũ được giữ để script gọi ngoài không bị vỡ.
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--counterfactual-map", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--disable-semantic-mapping", action="store_true")
    parser.add_argument("--cf-top-k", type=int, default=5)
    parser.add_argument("--mapping-top-k", type=int, default=5)
    parser.add_argument("--mapping-threshold", type=float, default=0.42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cf_hops < 1 or args.max_cf_paths < 1:
        raise ValueError("max-cf-hops và max-cf-paths phải lớn hơn 0.")
    if not 0.0 <= args.reject_threshold <= args.keep_threshold <= 1.0:
        raise ValueError("Cần thỏa mãn 0 <= reject-threshold <= keep-threshold <= 1.")

    payload = run_counterfactual_verification(
        graph_path=args.graph,
        memory_path=args.memory,
        rules_path=args.rules,
        retrieval_result_path=args.retrieval_result,
        output_path=args.output,
        counterfactual_mode=args.counterfactual_mode,
        max_cf_hops=args.max_cf_hops,
        max_cf_paths=args.max_cf_paths,
        verified_top_k=args.verified_top_k,
        keep_threshold=args.keep_threshold,
        reject_threshold=args.reject_threshold,
        cf_top_k=args.cf_top_k,
        mapping_top_k=args.mapping_top_k,
        mapping_threshold=args.mapping_threshold,
    )
    print_summary(payload)


if __name__ == "__main__":
    main()
