from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

GRAPH_PATH = "data/legal_causal_knowledge_graph.graphml"
MEMORY_PATH = "data/causal_memory.csv"
RETRIEVAL_RESULT_PATH = "data/retrieval_result.json"
OUTPUT_PATH = "data/counterfactual_verification_result.json"

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

    verification_method: str = (
        "graph_intervention_remove_mediator"
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

        self._validate_resources()
        self._build_lookup_tables()

        self.causal_event_graph = (
            self._build_causal_event_graph()
        )

        self._validate_retrieval_result()

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

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

        # Đường thay thế không chứng minh path gốc sai; nó chỉ cho thấy mediator
        # không phải điều kiện cần duy nhất. Vì vậy trường hợp này là UNCERTAIN,
        # không phải contradiction cứng. CONTRADICTED chỉ dành cho path sai cấu
        # trúc ở phần kiểm tra phía trên.
        if non_necessary > len(interventions) / 2:
            status = "UNRESOLVED"
            explanation = (
                f"{non_necessary}/{len(interventions)} mediator không cần thiết; "
                "tồn tại đường nhân quả thay thế nên chưa thể kết luận path gốc "
                "là chuỗi bắt buộc, nhưng điều này không phủ định path gốc."
            )
        elif necessary > 0 or partially > 0:
            status = "SUPPORTED"
            explanation = (
                f"Intervention cho thấy {necessary} mediator cần thiết và "
                f"{partially} mediator cần thiết một phần."
            )
        else:
            status = "UNRESOLVED"
            consistency = UNRESOLVED_BASE_SCORE
            explanation = "Không đủ tín hiệu cấu trúc để kết luận."

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
# END-TO-END PIPELINE
# ============================================================

class CounterfactualVerificationPipeline:
    def __init__(self, store: CounterfactualResourceStore) -> None:
        self.store = store
        self.searcher = CounterfactualGraphSearcher(store)
        self.path_verifier = CounterfactualPathVerifier(store=store, searcher=self.searcher)
        self.evidence_verifier = EvidenceVerifier(store)

    def run(
        self,
        *,
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
                verification = self.path_verifier.verify_path(
                    path_id=path_id,
                    original_path=path,
                    max_hops=max_cf_hops,
                    max_paths=max_cf_paths,
                )
            path_results.append(verification)
            print(f"- Path {path_id}: {verification.status} score={verification.consistency_score:.4f}")

        verified, uncertain, removed = self.evidence_verifier.verify_all(
            path_verifications=path_results,
            keep_threshold=keep_threshold,
            reject_threshold=reject_threshold,
            verified_top_k=verified_top_k,
        )

        status_counts = {
            status: sum(item.status == status for item in path_results)
            for status in ("SUPPORTED", "CONTRADICTED", "UNRESOLVED")
        }
        consistency = (
            sum(item.consistency_score for item in path_results) / len(path_results)
            if path_results else 0.0
        )
        resolved_ratio = (
            (status_counts["SUPPORTED"] + status_counts["CONTRADICTED"]) / len(path_results)
            if path_results else 0.0
        )
        confidence = clamp(0.65 * resolved_ratio + 0.35 * consistency)

        return VerificationResult(
            query=safe_string(self.store.retrieval_result.get("query")),
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
                "verification_method": "graph_intervention_remove_mediator",
                "semantic_mapping_enabled": False,
                "model_name": self.store.model_name,
            },
            statistics={
                # Giữ cả tên trường cũ và mới để tương thích với Step 5/5.5.
                "original_paths": len(path_results),
                "total_paths": len(path_results),
                "path_status_counts": status_counts,
                "status_counts": status_counts,
                "original_evidence": len(self.store.retrieval_result.get("evidence", [])),
                "total_evidence": len(self.store.retrieval_result.get("evidence", [])),
                "verified_evidence": len(verified),
                "uncertain_evidence": len(uncertain),
                "removed_evidence": len(removed),
            },
            path_verifications=[asdict(item) for item in path_results],
            verified_evidence=[asdict(item) for item in verified],
            uncertain_evidence=[asdict(item) for item in uncertain],
            removed_evidence=[asdict(item) for item in removed],
            consistency_score=consistency,
            confidence=confidence,
        )


def run_counterfactual_verification(
    *,
    graph_path: str = GRAPH_PATH,
    memory_path: str = MEMORY_PATH,
    retrieval_result_path: str = RETRIEVAL_RESULT_PATH,
    output_path: str = OUTPUT_PATH,
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
        embeddings_path=embeddings_path,
        counterfactual_map_path=counterfactual_map_path,
        model_name=model_name,
        enable_semantic_mapping=enable_semantic_mapping,
    )
    result = CounterfactualVerificationPipeline(store).run(
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
    print("Path status:", stats.get("status_counts", {}))
    print("Consistency:", f"{safe_float(payload.get('consistency_score')):.4f}")
    print("Confidence:", f"{safe_float(payload.get('confidence')):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual verification bằng do(remove mediator).")
    parser.add_argument("--graph", default=GRAPH_PATH)
    parser.add_argument("--memory", default=MEMORY_PATH)
    parser.add_argument("--retrieval-result", default=RETRIEVAL_RESULT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
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
        retrieval_result_path=args.retrieval_result,
        output_path=args.output,
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
