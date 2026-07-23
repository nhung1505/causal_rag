from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

GRAPH_PATH = "data/legal_causal_knowledge_graph.graphml"
MEMORY_PATH = "data/causal_memory.csv"
EMBEDDINGS_PATH = "data/causal_memory_embeddings.npy"
RETRIEVAL_RESULT_PATH = "data/retrieval_result.json"

# File ánh xạ phản thực tế do người dùng tự xây dựng.
#
# Ví dụ:
# {
#   "PHAM_TOI_CHUA_DAT": ["KHONG_PHAM_TOI_CHUA_DAT"],
#   "CO_AN_TICH": ["KHONG_CO_AN_TICH"]
# }
COUNTERFACTUAL_MAP_PATH = "data/counterfactual_event_map.json"

OUTPUT_PATH = "data/counterfactual_verification_result.json"

MODEL_NAME = "BAAI/bge-m3"

DEFAULT_CF_TOP_K = 5
DEFAULT_MAPPING_TOP_K = 5
DEFAULT_MAPPING_THRESHOLD = 0.42
DEFAULT_MAX_CF_HOPS = 3
DEFAULT_MAX_CF_PATHS = 30
DEFAULT_VERIFIED_TOP_K = 10

# Ngưỡng dùng để giữ hoặc loại evidence.
DEFAULT_KEEP_THRESHOLD = 0.52
DEFAULT_REJECT_THRESHOLD = 0.34

# Trọng số điểm xác minh cuối cùng.
PATH_SUPPORT_WEIGHT = 0.34
COUNTERFACTUAL_SUPPORT_WEIGHT = 0.30
SEMANTIC_EVIDENCE_WEIGHT = 0.20
GRAPH_EVIDENCE_WEIGHT = 0.16

# Thành phần đánh giá counterfactual.
OPPOSITE_OUTCOME_BONUS = 0.55
SAME_OUTCOME_PENALTY = 0.45
NO_PATH_BASE_SCORE = 0.45
UNRESOLVED_BASE_SCORE = 0.35

# Phạt khi counterfactual event chỉ được map bằng semantic similarity thấp.
CF_MAPPING_UNCERTAINTY_WEIGHT = 0.25

# Giảm điểm theo độ dài path.
HOP_DECAY = 0.84


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CounterfactualCandidate:
    source_event_node: str
    source_event_id: str
    source_event_name: str

    counterfactual_event_id: str
    counterfactual_event_name: str
    counterfactual_event_node: str

    generation_method: str
    mapping_method: str
    mapping_score: float

    confidence: float


@dataclass
class CounterfactualPath:
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
class PathVerification:
    original_path_id: int

    seed_event_id: str
    seed_event_name: str
    original_outcome_event_id: str
    original_outcome_event_name: str

    counterfactual_candidates: list[dict[str, Any]]
    counterfactual_to_same_outcome: list[dict[str, Any]]
    counterfactual_to_opposite_outcome: list[dict[str, Any]]

    opposite_outcome_candidates: list[dict[str, Any]]

    status: str
    consistency_score: float
    explanation: str


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

    verified_path_ids: list[int] = field(default_factory=list)
    rejected_path_ids: list[int] = field(default_factory=list)
    unresolved_path_ids: list[int] = field(default_factory=list)

    reasons: list[str] = field(default_factory=list)
    original_evidence: dict[str, Any] = field(default_factory=dict)


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


# ============================================================
# GENERAL HELPERS
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


def safe_float(value: Any, default: float = 0.0) -> float:
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


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return safe_string(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def split_csv_values(value: Any) -> list[str]:
    text = safe_string(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def unique_preserve_order(
    values: Iterable[str],
) -> list[str]:
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
    return max(lower, min(upper, value))


def remove_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFD",
        safe_string(text),
    )

    result = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )

    return result.replace("đ", "d").replace("Đ", "D")


def normalize_event_token(text: str) -> str:
    text = remove_vietnamese_accents(text).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_text_for_match(text: str) -> str:
    text = remove_vietnamese_accents(text).lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def event_id_from_node(
    graph: nx.Graph,
    node_id: str,
) -> str:
    data = graph.nodes[node_id]

    return (
        safe_string(data.get("event_id"))
        or safe_string(node_id).removeprefix("EVENT::")
    )


def event_name_from_node(
    graph: nx.Graph,
    node_id: str,
) -> str:
    data = graph.nodes[node_id]

    return (
        safe_string(data.get("event_name"))
        or safe_string(data.get("label"))
        or event_id_from_node(graph, node_id)
    )


def make_event_node_id(event_id: str) -> str:
    return f"EVENT::{safe_string(event_id)}"


def json_serializable(data: Any) -> Any:
    if isinstance(data, np.generic):
        return data.item()

    if isinstance(data, dict):
        return {
            key: json_serializable(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            json_serializable(value)
            for value in data
        ]

    return data


# ============================================================
# RESOURCE STORE
# ============================================================

class CounterfactualResourceStore:
    """Nạp graph, causal memory, embeddings và kết quả retrieval."""

    def __init__(
        self,
        *,
        graph_path: str,
        memory_path: str,
        embeddings_path: str,
        retrieval_result_path: str,
        counterfactual_map_path: str,
        model_name: str,
        enable_semantic_mapping: bool,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.memory_path = Path(memory_path)
        self.embeddings_path = Path(embeddings_path)
        self.retrieval_result_path = Path(
            retrieval_result_path
        )
        self.counterfactual_map_path = Path(
            counterfactual_map_path
        )

        self.model_name = model_name
        self.enable_semantic_mapping = (
            enable_semantic_mapping
        )

        self.graph = self._load_graph()
        self.memory_df = self._load_memory()
        self.embeddings = self._load_embeddings()
        self.retrieval_result = (
            self._load_retrieval_result()
        )
        self.counterfactual_map = (
            self._load_counterfactual_map()
        )

        self.model = self._load_model()

        self._validate_resources()
        self._build_lookup_tables()
        self.causal_event_graph = (
            self._build_causal_event_graph()
        )

    def _load_graph(self) -> nx.MultiDiGraph:
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy graph: {self.graph_path}"
            )

        print(f"Loading graph: {self.graph_path}")
        graph = nx.read_graphml(self.graph_path)

        if not graph.is_directed():
            raise ValueError(
                "Legal causal graph phải là đồ thị có hướng."
            )

        return graph

    def _load_memory(self) -> pd.DataFrame:
        if not self.memory_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy memory: {self.memory_path}"
            )

        print(f"Loading memory: {self.memory_path}")

        memory_df = pd.read_csv(
            self.memory_path,
            dtype={
                "rule_id": str,
                "article_id": str,
                "event_id": str,
                "graph_node_id": str,
            },
            keep_default_na=False,
        )

        required_columns = {
            "memory_id",
            "memory_type",
            "graph_node_id",
        }

        missing = required_columns - set(
            memory_df.columns
        )

        if missing:
            raise ValueError(
                "Causal memory thiếu cột: "
                f"{sorted(missing)}"
            )

        memory_df["memory_id"] = pd.to_numeric(
            memory_df["memory_id"],
            errors="raise",
        ).astype(np.int64)

        return memory_df

    def _load_embeddings(
        self,
    ) -> Optional[np.ndarray]:
        if not self.enable_semantic_mapping:
            return None

        if not self.embeddings_path.exists():
            print(
                "Warning: không tìm thấy embeddings. "
                "Semantic mapping sẽ bị tắt."
            )
            self.enable_semantic_mapping = False
            return None

        print(
            f"Loading embeddings: {self.embeddings_path}"
        )

        return np.asarray(
            np.load(self.embeddings_path),
            dtype=np.float32,
        )

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

        required_keys = {
            "query",
            "causal_paths",
            "evidence",
        }

        missing = required_keys - set(result)

        if missing:
            raise ValueError(
                "Retrieval result thiếu trường: "
                f"{sorted(missing)}"
            )

        return result

    def _load_counterfactual_map(
        self,
    ) -> dict[str, list[str]]:
        if not self.counterfactual_map_path.exists():
            print(
                "Counterfactual map không tồn tại; "
                "sử dụng bộ sinh heuristic."
            )
            return {}

        print(
            "Loading counterfactual map: "
            f"{self.counterfactual_map_path}"
        )

        with self.counterfactual_map_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_map = json.load(file)

        normalized_map: dict[str, list[str]] = {}

        for source, targets in raw_map.items():
            source_id = safe_string(source)

            if isinstance(targets, str):
                targets = [targets]

            normalized_map[source_id] = (
                unique_preserve_order(targets)
            )

        return normalized_map

    def _load_model(
        self,
    ) -> Optional[SentenceTransformer]:
        if not self.enable_semantic_mapping:
            return None

        if SentenceTransformer is None:
            print(
                "Warning: sentence-transformers chưa được "
                "cài đặt. Semantic mapping sẽ bị tắt."
            )
            self.enable_semantic_mapping = False
            return None

        print(
            "Loading embedding model: "
            f"{self.model_name}"
        )

        return SentenceTransformer(self.model_name)

    def _validate_resources(self) -> None:
        graph_nodes = set(self.graph.nodes)

        memory_nodes = set(
            self.memory_df["graph_node_id"]
        )

        missing_nodes = memory_nodes - graph_nodes

        if missing_nodes:
            raise ValueError(
                "Memory chứa graph_node_id không tồn tại "
                f"trong graph. Ví dụ: "
                f"{sorted(missing_nodes)[:10]}"
            )

        if self.embeddings is not None:
            if len(self.embeddings) != len(
                self.memory_df
            ):
                raise ValueError(
                    "Số embedding không khớp số dòng memory."
                )

        memory_types = set(
            self.memory_df["memory_type"]
            .astype(str)
            .str.upper()
        )

        if "EVENT" not in memory_types:
            raise ValueError(
                "Memory không có EVENT record."
            )

        if "RULE" not in memory_types:
            raise ValueError(
                "Memory không có RULE record."
            )

        print("Resource validation: OK")

    def _build_lookup_tables(self) -> None:
        self.memory_by_id = self.memory_df.set_index(
            "memory_id",
            drop=False,
        )

        event_df = self.memory_df[
            self.memory_df["memory_type"].str.upper()
            == "EVENT"
        ].copy()

        rule_df = self.memory_df[
            self.memory_df["memory_type"].str.upper()
            == "RULE"
        ].copy()

        self.event_df = event_df
        self.rule_df = rule_df

        self.event_memory_ids = event_df[
            "memory_id"
        ].to_numpy(dtype=np.int64)

        self.event_by_node: dict[str, pd.Series] = {}
        self.event_node_by_id: dict[str, str] = {}
        self.event_nodes_by_normalized_name: dict[
            str,
            list[str],
        ] = {}

        for _, row in event_df.iterrows():
            node_id = safe_string(
                row.get("graph_node_id")
            )
            event_id = (
                safe_string(row.get("event_id"))
                or event_id_from_node(
                    self.graph,
                    node_id,
                )
            )
            event_name = (
                safe_string(row.get("event_name"))
                or event_name_from_node(
                    self.graph,
                    node_id,
                )
            )

            self.event_by_node[node_id] = row
            self.event_node_by_id[event_id] = node_id

            normalized_name = normalize_text_for_match(
                event_name
            )

            if normalized_name:
                self.event_nodes_by_normalized_name.setdefault(
                    normalized_name,
                    [],
                ).append(node_id)

        self.rule_by_id: dict[str, pd.Series] = {}

        for _, row in rule_df.iterrows():
            rule_id = safe_string(
                row.get("rule_id")
            )

            if rule_id:
                self.rule_by_id[rule_id] = row

    def _build_causal_event_graph(
        self,
    ) -> nx.DiGraph:
        causal_graph = nx.DiGraph()

        for node_id, data in self.graph.nodes(
            data=True
        ):
            if (
                safe_string(data.get("node_type"))
                == "EVENT"
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

            for source, target, _, data in (
                edge_iterator
            ):
                self._merge_causal_edge(
                    causal_graph,
                    source,
                    target,
                    data,
                )
        else:
            for source, target, data in (
                self.graph.edges(data=True)
            ):
                self._merge_causal_edge(
                    causal_graph,
                    source,
                    target,
                    data,
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
        graph: nx.DiGraph,
        source: str,
        target: str,
        data: dict[str, Any],
    ) -> None:
        if safe_string(data.get("relation")) != "CAUSES":
            return

        rule_ids = split_csv_values(
            data.get("rule_ids")
        )

        if not rule_ids:
            rule_id = safe_string(
                data.get("rule_id")
            )
            if rule_id:
                rule_ids = [rule_id]

        article_ids = split_csv_values(
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
                1,
            ),
        )

        if graph.has_edge(source, target):
            existing = graph[source][target]

            existing["rule_ids"] = (
                unique_preserve_order(
                    list(existing["rule_ids"])
                    + rule_ids
                )
            )

            existing["article_ids"] = (
                unique_preserve_order(
                    list(existing["article_ids"])
                    + article_ids
                )
            )

            existing["support_count"] += (
                support_count
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

    def encode_texts(
        self,
        texts: list[str],
    ) -> Optional[np.ndarray]:
        if (
            not self.enable_semantic_mapping
            or self.model is None
        ):
            return None

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


# ============================================================
# COUNTERFACTUAL GENERATION
# ============================================================

class CounterfactualEventGenerator:
    """Sinh counterfactual event bằng map tường minh và heuristic."""

    NEGATIVE_PREFIXES = (
        "NOT_",
        "NO_",
        "WITHOUT_",
        "KHONG_",
        "CHUA_",
    )

    POSITIVE_NEGATIVE_PAIRS = (
        ("CO_", "KHONG_CO_"),
        ("DA_", "CHUA_"),
        ("DUOC_", "KHONG_DUOC_"),
        ("PHAI_", "KHONG_PHAI_"),
        ("BI_", "KHONG_BI_"),
    )

    VIETNAMESE_NEGATION_PATTERNS = (
        (r"^không\s+", ""),
        (r"^chưa\s+", "đã "),
        (r"^được\s+", "không được "),
        (r"^bị\s+", "không bị "),
        (r"^phải\s+", "không phải "),
        (r"^có\s+", "không có "),
    )

    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    def generate_raw_candidates(
        self,
        *,
        event_id: str,
        event_name: str,
    ) -> list[tuple[str, str, str]]:
        """Trả về (candidate_id, candidate_name, method)."""

        candidates: list[
            tuple[str, str, str]
        ] = []

        explicit_targets = (
            self.store.counterfactual_map.get(
                event_id,
                [],
            )
        )

        for target_id in explicit_targets:
            node_id = (
                self.store.event_node_by_id.get(
                    target_id
                )
            )

            target_name = (
                event_name_from_node(
                    self.store.graph,
                    node_id,
                )
                if node_id
                else target_id
            )

            candidates.append(
                (
                    target_id,
                    target_name,
                    "explicit_map",
                )
            )

        normalized_id = normalize_event_token(
            event_id
        )

        for candidate_id in self._toggle_event_id(
            normalized_id
        ):
            candidates.append(
                (
                    candidate_id,
                    self._event_id_to_name(
                        candidate_id
                    ),
                    "id_negation",
                )
            )

        for candidate_name in self._negate_name(
            event_name
        ):
            candidate_id = normalize_event_token(
                candidate_name
            )

            candidates.append(
                (
                    candidate_id,
                    candidate_name,
                    "name_negation",
                )
            )

        # Loại chính event ban đầu và loại trùng.
        deduplicated: list[
            tuple[str, str, str]
        ] = []
        seen: set[tuple[str, str]] = set()

        for candidate_id, candidate_name, method in (
            candidates
        ):
            if candidate_id == normalized_id:
                continue

            key = (
                normalize_event_token(candidate_id),
                normalize_text_for_match(
                    candidate_name
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(
                (
                    candidate_id,
                    candidate_name,
                    method,
                )
            )

        return deduplicated

    def _toggle_event_id(
        self,
        event_id: str,
    ) -> list[str]:
        candidates: list[str] = []

        for prefix in self.NEGATIVE_PREFIXES:
            if event_id.startswith(prefix):
                positive = event_id[len(prefix):]

                if positive:
                    candidates.append(positive)

        for positive, negative in (
            self.POSITIVE_NEGATIVE_PAIRS
        ):
            if event_id.startswith(negative):
                remainder = event_id[
                    len(negative):
                ]
                candidates.append(
                    positive + remainder
                )
            elif event_id.startswith(positive):
                remainder = event_id[
                    len(positive):
                ]
                candidates.append(
                    negative + remainder
                )

        if not any(
            event_id.startswith(prefix)
            for prefix in self.NEGATIVE_PREFIXES
        ):
            candidates.extend(
                [
                    f"KHONG_{event_id}",
                    f"NOT_{event_id}",
                ]
            )

        return unique_preserve_order(candidates)

    def _negate_name(
        self,
        event_name: str,
    ) -> list[str]:
        name = safe_string(event_name)
        lowered = name.lower()

        if not name:
            return []

        candidates: list[str] = []

        if lowered.startswith("không "):
            candidates.append(name[6:].strip())
        elif lowered.startswith("chưa "):
            candidates.append(
                "đã " + name[5:].strip()
            )
        else:
            candidates.append(
                "không " + name
            )

        for pattern, replacement in (
            self.VIETNAMESE_NEGATION_PATTERNS
        ):
            if re.search(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            ):
                candidate = re.sub(
                    pattern,
                    replacement,
                    name,
                    count=1,
                    flags=re.IGNORECASE,
                ).strip()

                if candidate:
                    candidates.append(candidate)

        return unique_preserve_order(candidates)

    @staticmethod
    def _event_id_to_name(
        event_id: str,
    ) -> str:
        text = event_id.lower().replace(
            "_",
            " ",
        )
        return text.strip()


# ============================================================
# COUNTERFACTUAL EVENT MAPPING
# ============================================================

class CounterfactualEventMapper:
    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    def map_candidates(
        self,
        *,
        source_event_node: str,
        source_event_id: str,
        source_event_name: str,
        raw_candidates: list[
            tuple[str, str, str]
        ],
        top_k: int,
        min_score: float,
    ) -> list[CounterfactualCandidate]:
        mapped: list[CounterfactualCandidate] = []

        for (
            candidate_id,
            candidate_name,
            generation_method,
        ) in raw_candidates:
            exact = self._exact_match(
                candidate_id,
                candidate_name,
            )

            for (
                node_id,
                mapping_method,
                mapping_score,
            ) in exact:
                mapped.append(
                    self._make_candidate(
                        source_event_node=(
                            source_event_node
                        ),
                        source_event_id=(
                            source_event_id
                        ),
                        source_event_name=(
                            source_event_name
                        ),
                        candidate_id=candidate_id,
                        candidate_name=candidate_name,
                        node_id=node_id,
                        generation_method=(
                            generation_method
                        ),
                        mapping_method=mapping_method,
                        mapping_score=mapping_score,
                    )
                )

        if (
            len(mapped) < top_k
            and self.store.enable_semantic_mapping
        ):
            semantic_candidates = (
                self._semantic_map(
                    source_event_node=(
                        source_event_node
                    ),
                    source_event_id=(
                        source_event_id
                    ),
                    source_event_name=(
                        source_event_name
                    ),
                    raw_candidates=raw_candidates,
                    top_k=top_k,
                    min_score=min_score,
                )
            )
            mapped.extend(semantic_candidates)

        # Không cho phép map ngược lại chính event nguồn.
        mapped = [
            candidate
            for candidate in mapped
            if (
                candidate.counterfactual_event_node
                != source_event_node
            )
        ]

        best_by_node: dict[
            str,
            CounterfactualCandidate,
        ] = {}

        for candidate in mapped:
            node_id = (
                candidate.counterfactual_event_node
            )

            existing = best_by_node.get(node_id)

            if (
                existing is None
                or candidate.confidence
                > existing.confidence
            ):
                best_by_node[node_id] = candidate

        result = list(best_by_node.values())
        result.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        return result[:top_k]

    def _exact_match(
        self,
        candidate_id: str,
        candidate_name: str,
    ) -> list[tuple[str, str, float]]:
        matches: list[
            tuple[str, str, float]
        ] = []

        node_id = self.store.event_node_by_id.get(
            candidate_id
        )

        if node_id:
            matches.append(
                (
                    node_id,
                    "exact_event_id",
                    1.0,
                )
            )

        normalized_candidate_id = (
            normalize_event_token(candidate_id)
        )

        for event_id, event_node in (
            self.store.event_node_by_id.items()
        ):
            if (
                normalize_event_token(event_id)
                == normalized_candidate_id
            ):
                matches.append(
                    (
                        event_node,
                        "normalized_event_id",
                        0.96,
                    )
                )

        normalized_name = normalize_text_for_match(
            candidate_name
        )

        for event_node in (
            self.store
            .event_nodes_by_normalized_name
            .get(normalized_name, [])
        ):
            matches.append(
                (
                    event_node,
                    "exact_event_name",
                    0.94,
                )
            )

        deduplicated: dict[
            str,
            tuple[str, str, float],
        ] = {}

        for match in matches:
            node_id = match[0]
            existing = deduplicated.get(node_id)

            if (
                existing is None
                or match[2] > existing[2]
            ):
                deduplicated[node_id] = match

        return list(deduplicated.values())

    def _semantic_map(
        self,
        *,
        source_event_node: str,
        source_event_id: str,
        source_event_name: str,
        raw_candidates: list[
            tuple[str, str, str]
        ],
        top_k: int,
        min_score: float,
    ) -> list[CounterfactualCandidate]:
        texts = [
            candidate_name
            for _, candidate_name, _ in raw_candidates
        ]

        if not texts:
            return []

        query_vectors = self.store.encode_texts(
            texts
        )

        if query_vectors is None:
            return []

        event_vectors = self.store.embeddings[
            self.store.event_memory_ids
        ]

        candidates: list[
            CounterfactualCandidate
        ] = []

        for raw_index, query_vector in enumerate(
            query_vectors
        ):
            scores = event_vectors @ query_vector
            order = np.argsort(-scores)

            (
                candidate_id,
                candidate_name,
                generation_method,
            ) = raw_candidates[raw_index]

            for local_index in order[:top_k]:
                memory_id = int(
                    self.store.event_memory_ids[
                        local_index
                    ]
                )
                score = float(
                    scores[local_index]
                )

                if score < min_score:
                    continue

                row = self.store.memory_by_id.loc[
                    memory_id
                ]
                node_id = safe_string(
                    row.get("graph_node_id")
                )

                if node_id == source_event_node:
                    continue

                candidates.append(
                    self._make_candidate(
                        source_event_node=(
                            source_event_node
                        ),
                        source_event_id=(
                            source_event_id
                        ),
                        source_event_name=(
                            source_event_name
                        ),
                        candidate_id=candidate_id,
                        candidate_name=candidate_name,
                        node_id=node_id,
                        generation_method=(
                            generation_method
                        ),
                        mapping_method=(
                            "semantic_event_mapping"
                        ),
                        mapping_score=score,
                    )
                )

        return candidates

    def _make_candidate(
        self,
        *,
        source_event_node: str,
        source_event_id: str,
        source_event_name: str,
        candidate_id: str,
        candidate_name: str,
        node_id: str,
        generation_method: str,
        mapping_method: str,
        mapping_score: float,
    ) -> CounterfactualCandidate:
        mapped_id = event_id_from_node(
            self.store.graph,
            node_id,
        )
        mapped_name = event_name_from_node(
            self.store.graph,
            node_id,
        )

        generation_reliability = {
            "explicit_map": 1.0,
            "id_negation": 0.82,
            "name_negation": 0.70,
        }.get(
            generation_method,
            0.60,
        )

        confidence = clamp(
            0.55 * mapping_score
            + 0.45 * generation_reliability
        )

        return CounterfactualCandidate(
            source_event_node=source_event_node,
            source_event_id=source_event_id,
            source_event_name=source_event_name,
            counterfactual_event_id=mapped_id,
            counterfactual_event_name=mapped_name,
            counterfactual_event_node=node_id,
            generation_method=generation_method,
            mapping_method=mapping_method,
            mapping_score=mapping_score,
            confidence=confidence,
        )


# ============================================================
# GRAPH SEARCH
# ============================================================

class CounterfactualGraphSearcher:
    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    def find_paths(
        self,
        *,
        start_node: str,
        target_nodes: set[str],
        max_hops: int,
        max_paths: int,
    ) -> list[CounterfactualPath]:
        graph = self.store.causal_event_graph

        if start_node not in graph:
            return []

        if not target_nodes:
            return []

        queue: list[
            tuple[
                str,
                list[str],
                list[str],
                list[str],
            ]
        ] = [
            (
                start_node,
                [start_node],
                [],
                [],
            )
        ]

        queue_index = 0
        results: list[
            CounterfactualPath
        ] = []

        while (
            queue_index < len(queue)
            and len(results) < max_paths
        ):
            (
                current_node,
                event_nodes,
                rule_ids,
                article_ids,
            ) = queue[queue_index]
            queue_index += 1

            hop_count = len(event_nodes) - 1

            if hop_count >= max_hops:
                continue

            for neighbor in graph.successors(
                current_node
            ):
                if neighbor in event_nodes:
                    continue

                edge_data = graph[
                    current_node
                ][neighbor]

                next_event_nodes = (
                    event_nodes + [neighbor]
                )

                next_rule_ids = (
                    unique_preserve_order(
                        rule_ids
                        + edge_data.get(
                            "rule_ids",
                            [],
                        )
                    )
                )

                next_article_ids = (
                    unique_preserve_order(
                        article_ids
                        + edge_data.get(
                            "article_ids",
                            [],
                        )
                    )
                )

                next_hop_count = (
                    len(next_event_nodes) - 1
                )

                if neighbor in target_nodes:
                    results.append(
                        self._build_path(
                            event_nodes=(
                                next_event_nodes
                            ),
                            rule_ids=next_rule_ids,
                            article_ids=(
                                next_article_ids
                            ),
                        )
                    )

                    if len(results) >= max_paths:
                        break

                if next_hop_count < max_hops:
                    queue.append(
                        (
                            neighbor,
                            next_event_nodes,
                            next_rule_ids,
                            next_article_ids,
                        )
                    )

        results.sort(
            key=lambda item: item.path_score,
            reverse=True,
        )

        return results[:max_paths]

    def reachable_nodes(
        self,
        *,
        start_node: str,
        max_hops: int,
    ) -> dict[str, int]:
        graph = self.store.causal_event_graph

        if start_node not in graph:
            return {}

        distances = nx.single_source_shortest_path_length(
            graph,
            start_node,
            cutoff=max_hops,
        )

        return {
            node_id: distance
            for node_id, distance in distances.items()
            if distance > 0
        }

    def _build_path(
        self,
        *,
        event_nodes: list[str],
        rule_ids: list[str],
        article_ids: list[str],
    ) -> CounterfactualPath:
        graph = self.store.causal_event_graph

        hop_count = len(event_nodes) - 1

        support_values: list[float] = []

        for source, target in zip(
            event_nodes[:-1],
            event_nodes[1:],
        ):
            support_count = max(
                1,
                safe_int(
                    graph[source][target].get(
                        "support_count"
                    ),
                    1,
                ),
            )
            support_values.append(
                math.log1p(support_count)
            )

        average_support = (
            sum(support_values)
            / len(support_values)
            if support_values
            else 0.0
        )

        normalized_support = min(
            1.0,
            average_support / math.log(4.0),
        )

        path_score = clamp(
            normalized_support
            * (HOP_DECAY ** max(0, hop_count - 1))
        )

        return CounterfactualPath(
            start_event_node=event_nodes[0],
            start_event_id=event_id_from_node(
                graph,
                event_nodes[0],
            ),
            start_event_name=event_name_from_node(
                graph,
                event_nodes[0],
            ),
            end_event_node=event_nodes[-1],
            end_event_id=event_id_from_node(
                graph,
                event_nodes[-1],
            ),
            end_event_name=event_name_from_node(
                graph,
                event_nodes[-1],
            ),
            event_nodes=event_nodes,
            event_ids=[
                event_id_from_node(
                    graph,
                    node_id,
                )
                for node_id in event_nodes
            ],
            event_names=[
                event_name_from_node(
                    graph,
                    node_id,
                )
                for node_id in event_nodes
            ],
            rule_ids=rule_ids,
            article_ids=article_ids,
            hop_count=hop_count,
            path_score=path_score,
        )


# ============================================================
# PATH VERIFICATION
# ============================================================

class CounterfactualPathVerifier:
    def __init__(
        self,
        *,
        store: CounterfactualResourceStore,
        generator: CounterfactualEventGenerator,
        mapper: CounterfactualEventMapper,
        searcher: CounterfactualGraphSearcher,
    ) -> None:
        self.store = store
        self.generator = generator
        self.mapper = mapper
        self.searcher = searcher

    def verify_path(
        self,
        *,
        path_id: int,
        original_path: dict[str, Any],
        cf_top_k: int,
        mapping_top_k: int,
        mapping_threshold: float,
        max_hops: int,
        max_paths: int,
    ) -> PathVerification:
        event_nodes = [
            safe_string(node_id)
            for node_id in original_path.get(
                "event_nodes",
                [],
            )
            if safe_string(node_id)
        ]

        if len(event_nodes) < 2:
            return PathVerification(
                original_path_id=path_id,
                seed_event_id="",
                seed_event_name="",
                original_outcome_event_id="",
                original_outcome_event_name="",
                counterfactual_candidates=[],
                counterfactual_to_same_outcome=[],
                counterfactual_to_opposite_outcome=[],
                opposite_outcome_candidates=[],
                status="UNRESOLVED",
                consistency_score=0.0,
                explanation=(
                    "Original path không đủ hai event."
                ),
            )

        seed_node = event_nodes[0]
        outcome_node = event_nodes[-1]

        seed_id = event_id_from_node(
            self.store.graph,
            seed_node,
        )
        seed_name = event_name_from_node(
            self.store.graph,
            seed_node,
        )

        outcome_id = event_id_from_node(
            self.store.graph,
            outcome_node,
        )
        outcome_name = event_name_from_node(
            self.store.graph,
            outcome_node,
        )

        raw_seed_counterfactuals = (
            self.generator.generate_raw_candidates(
                event_id=seed_id,
                event_name=seed_name,
            )
        )

        seed_counterfactuals = (
            self.mapper.map_candidates(
                source_event_node=seed_node,
                source_event_id=seed_id,
                source_event_name=seed_name,
                raw_candidates=(
                    raw_seed_counterfactuals
                ),
                top_k=mapping_top_k,
                min_score=mapping_threshold,
            )
        )[:cf_top_k]

        raw_outcome_counterfactuals = (
            self.generator.generate_raw_candidates(
                event_id=outcome_id,
                event_name=outcome_name,
            )
        )

        outcome_counterfactuals = (
            self.mapper.map_candidates(
                source_event_node=outcome_node,
                source_event_id=outcome_id,
                source_event_name=outcome_name,
                raw_candidates=(
                    raw_outcome_counterfactuals
                ),
                top_k=mapping_top_k,
                min_score=mapping_threshold,
            )
        )[:cf_top_k]

        opposite_outcome_nodes = {
            candidate.counterfactual_event_node
            for candidate in outcome_counterfactuals
        }

        same_outcome_paths: list[
            CounterfactualPath
        ] = []

        opposite_outcome_paths: list[
            CounterfactualPath
        ] = []

        best_mapping_confidence = 0.0

        for candidate in seed_counterfactuals:
            best_mapping_confidence = max(
                best_mapping_confidence,
                candidate.confidence,
            )

            same_paths = self.searcher.find_paths(
                start_node=(
                    candidate
                    .counterfactual_event_node
                ),
                target_nodes={outcome_node},
                max_hops=max_hops,
                max_paths=max_paths,
            )
            same_outcome_paths.extend(same_paths)

            opposite_paths = self.searcher.find_paths(
                start_node=(
                    candidate
                    .counterfactual_event_node
                ),
                target_nodes=(
                    opposite_outcome_nodes
                ),
                max_hops=max_hops,
                max_paths=max_paths,
            )
            opposite_outcome_paths.extend(
                opposite_paths
            )

        (
            status,
            score,
            explanation,
        ) = self._classify_counterfactual_result(
            seed_counterfactuals=(
                seed_counterfactuals
            ),
            outcome_counterfactuals=(
                outcome_counterfactuals
            ),
            same_outcome_paths=same_outcome_paths,
            opposite_outcome_paths=(
                opposite_outcome_paths
            ),
            best_mapping_confidence=(
                best_mapping_confidence
            ),
        )

        return PathVerification(
            original_path_id=path_id,
            seed_event_id=seed_id,
            seed_event_name=seed_name,
            original_outcome_event_id=outcome_id,
            original_outcome_event_name=outcome_name,
            counterfactual_candidates=[
                asdict(candidate)
                for candidate in seed_counterfactuals
            ],
            counterfactual_to_same_outcome=[
                asdict(path)
                for path in same_outcome_paths
            ],
            counterfactual_to_opposite_outcome=[
                asdict(path)
                for path in opposite_outcome_paths
            ],
            opposite_outcome_candidates=[
                asdict(candidate)
                for candidate in outcome_counterfactuals
            ],
            status=status,
            consistency_score=score,
            explanation=explanation,
        )

    def _classify_counterfactual_result(
        self,
        *,
        seed_counterfactuals: list[
            CounterfactualCandidate
        ],
        outcome_counterfactuals: list[
            CounterfactualCandidate
        ],
        same_outcome_paths: list[
            CounterfactualPath
        ],
        opposite_outcome_paths: list[
            CounterfactualPath
        ],
        best_mapping_confidence: float,
    ) -> tuple[str, float, str]:
        if not seed_counterfactuals:
            return (
                "UNRESOLVED",
                UNRESOLVED_BASE_SCORE,
                (
                    "Không map được counterfactual của "
                    "seed event vào graph."
                ),
            )

        if not outcome_counterfactuals:
            if same_outcome_paths:
                best_same_score = max(
                    path.path_score
                    for path in same_outcome_paths
                )

                score = clamp(
                    NO_PATH_BASE_SCORE
                    - SAME_OUTCOME_PENALTY
                    * best_same_score
                )

                return (
                    "CONTRADICTED",
                    score,
                    (
                        "Counterfactual seed vẫn dẫn tới "
                        "outcome ban đầu; chưa tìm được "
                        "opposite outcome rõ ràng."
                    ),
                )

            score = clamp(
                NO_PATH_BASE_SCORE
                * best_mapping_confidence
            )

            return (
                "UNRESOLVED",
                score,
                (
                    "Không tìm được opposite outcome trong "
                    "graph; counterfactual seed cũng không "
                    "dẫn tới outcome ban đầu."
                ),
            )

        best_same_score = (
            max(
                path.path_score
                for path in same_outcome_paths
            )
            if same_outcome_paths
            else 0.0
        )

        best_opposite_score = (
            max(
                path.path_score
                for path in opposite_outcome_paths
            )
            if opposite_outcome_paths
            else 0.0
        )

        uncertainty_penalty = (
            CF_MAPPING_UNCERTAINTY_WEIGHT
            * (1.0 - best_mapping_confidence)
        )

        score = (
            NO_PATH_BASE_SCORE
            + OPPOSITE_OUTCOME_BONUS
            * best_opposite_score
            - SAME_OUTCOME_PENALTY
            * best_same_score
            - uncertainty_penalty
        )

        score = clamp(score)

        if (
            best_opposite_score > 0.0
            and best_opposite_score
            >= best_same_score
        ):
            return (
                "SUPPORTED",
                score,
                (
                    "Counterfactual seed dẫn tới opposite "
                    "outcome và không ưu thế hơn đối với "
                    "outcome ban đầu."
                ),
            )

        if (
            best_same_score > 0.0
            and best_same_score
            > best_opposite_score
        ):
            return (
                "CONTRADICTED",
                score,
                (
                    "Counterfactual seed vẫn dẫn tới outcome "
                    "ban đầu mạnh hơn opposite outcome."
                ),
            )

        return (
            "UNRESOLVED",
            score,
            (
                "Không tìm được causal path đủ rõ để kết luận "
                "phản thực tế."
            ),
        )


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

class EvidenceVerifier:
    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

    def verify_all(
        self,
        *,
        path_verifications: list[
            PathVerification
        ],
        keep_threshold: float,
        reject_threshold: float,
        verified_top_k: int,
    ) -> tuple[
        list[EvidenceVerification],
        list[EvidenceVerification],
        list[EvidenceVerification],
    ]:
        evidence_items = (
            self.store.retrieval_result.get(
                "evidence",
                [],
            )
        )

        path_verification_by_id = {
            item.original_path_id: item
            for item in path_verifications
        }

        verified: list[
            EvidenceVerification
        ] = []
        uncertain: list[
            EvidenceVerification
        ] = []
        removed: list[
            EvidenceVerification
        ] = []

        for evidence in evidence_items:
            verification = self._verify_one(
                evidence=evidence,
                path_verification_by_id=(
                    path_verification_by_id
                ),
                keep_threshold=keep_threshold,
                reject_threshold=reject_threshold,
            )

            if verification.decision == "KEEP":
                verified.append(verification)
            elif verification.decision == "REMOVE":
                removed.append(verification)
            else:
                uncertain.append(verification)

        verified.sort(
            key=lambda item: item.verification_score,
            reverse=True,
        )
        uncertain.sort(
            key=lambda item: item.verification_score,
            reverse=True,
        )
        removed.sort(
            key=lambda item: item.verification_score,
        )

        return (
            verified[:verified_top_k],
            uncertain,
            removed,
        )

    def _verify_one(
        self,
        *,
        evidence: dict[str, Any],
        path_verification_by_id: dict[
            int,
            PathVerification,
        ],
        keep_threshold: float,
        reject_threshold: float,
    ) -> EvidenceVerification:
        path_ids = [
            safe_int(path_id)
            for path_id in evidence.get(
                "path_ids",
                [],
            )
        ]

        supported_path_ids: list[int] = []
        rejected_path_ids: list[int] = []
        unresolved_path_ids: list[int] = []

        path_scores: list[float] = []
        counterfactual_scores: list[float] = []

        reasons: list[str] = []

        for path_id in path_ids:
            verification = (
                path_verification_by_id.get(
                    path_id
                )
            )

            if verification is None:
                continue

            path_scores.append(
                verification.consistency_score
            )

            if verification.status == "SUPPORTED":
                supported_path_ids.append(path_id)
                counterfactual_scores.append(
                    verification.consistency_score
                )
            elif (
                verification.status
                == "CONTRADICTED"
            ):
                rejected_path_ids.append(path_id)
                counterfactual_scores.append(
                    verification.consistency_score
                )
            else:
                unresolved_path_ids.append(
                    path_id
                )
                counterfactual_scores.append(
                    verification.consistency_score
                )

        semantic_score = safe_float(
            evidence.get("semantic_score")
        )
        graph_score = safe_float(
            evidence.get("graph_score")
        )
        original_final_score = safe_float(
            evidence.get("final_score")
        )

        if path_scores:
            path_support_score = (
                sum(path_scores)
                / len(path_scores)
            )
        else:
            # Evidence được semantic retrieval trực tiếp nhưng
            # không nằm trên path vẫn được giữ một mức cơ sở.
            path_support_score = 0.40

        if counterfactual_scores:
            counterfactual_support_score = (
                sum(counterfactual_scores)
                / len(counterfactual_scores)
            )
        else:
            counterfactual_support_score = (
                UNRESOLVED_BASE_SCORE
            )

        verification_score = clamp(
            PATH_SUPPORT_WEIGHT
            * path_support_score
            + COUNTERFACTUAL_SUPPORT_WEIGHT
            * counterfactual_support_score
            + SEMANTIC_EVIDENCE_WEIGHT
            * semantic_score
            + GRAPH_EVIDENCE_WEIGHT
            * graph_score
        )

        contradicted_ratio = (
            len(rejected_path_ids)
            / len(path_ids)
            if path_ids
            else 0.0
        )

        supported_ratio = (
            len(supported_path_ids)
            / len(path_ids)
            if path_ids
            else 0.0
        )

        if contradicted_ratio >= 0.5:
            decision = "REMOVE"
            reasons.append(
                "Ít nhất một nửa causal path liên quan bị "
                "counterfactual contradiction."
            )
        elif (
            verification_score >= keep_threshold
            and (
                supported_ratio > 0.0
                or not path_ids
            )
        ):
            decision = "KEEP"
            reasons.append(
                "Evidence đạt ngưỡng xác minh và có causal "
                "path được counterfactual support."
            )
        elif verification_score < reject_threshold:
            decision = "REMOVE"
            reasons.append(
                "Điểm xác minh thấp hơn reject threshold."
            )
        else:
            decision = "UNCERTAIN"
            reasons.append(
                "Evidence chưa đủ mạnh để giữ nhưng cũng "
                "chưa đủ bằng chứng để loại."
            )

        if not path_ids:
            reasons.append(
                "Evidence không gắn với causal path; quyết "
                "định chủ yếu dựa vào semantic và graph score."
            )

        if unresolved_path_ids:
            reasons.append(
                f"{len(unresolved_path_ids)} path chưa xác "
                "định được phản thực tế."
            )

        return EvidenceVerification(
            original_rank=safe_int(
                evidence.get("rank")
            ),
            rule_id=safe_string(
                evidence.get("rule_id")
            ),
            article_id=safe_string(
                evidence.get("article_id")
            ),
            original_final_score=(
                original_final_score
            ),
            semantic_score=semantic_score,
            graph_score=graph_score,
            path_support_score=(
                path_support_score
            ),
            counterfactual_support_score=(
                counterfactual_support_score
            ),
            verification_score=verification_score,
            decision=decision,
            verified_path_ids=(
                supported_path_ids
            ),
            rejected_path_ids=(
                rejected_path_ids
            ),
            unresolved_path_ids=(
                unresolved_path_ids
            ),
            reasons=reasons,
            original_evidence=evidence,
        )


# ============================================================
# END-TO-END VERIFICATION PIPELINE
# ============================================================

class CounterfactualVerificationPipeline:
    def __init__(
        self,
        store: CounterfactualResourceStore,
    ) -> None:
        self.store = store

        self.generator = (
            CounterfactualEventGenerator(store)
        )
        self.mapper = CounterfactualEventMapper(
            store
        )
        self.searcher = (
            CounterfactualGraphSearcher(store)
        )

        self.path_verifier = (
            CounterfactualPathVerifier(
                store=store,
                generator=self.generator,
                mapper=self.mapper,
                searcher=self.searcher,
            )
        )

        self.evidence_verifier = (
            EvidenceVerifier(store)
        )

    def run(
        self,
        *,
        cf_top_k: int,
        mapping_top_k: int,
        mapping_threshold: float,
        max_cf_hops: int,
        max_cf_paths: int,
        verified_top_k: int,
        keep_threshold: float,
        reject_threshold: float,
    ) -> VerificationResult:
        original_paths = (
            self.store.retrieval_result.get(
                "causal_paths",
                [],
            )
        )

        path_verifications: list[
            PathVerification
        ] = []

        print(
            "\nVerifying",
            len(original_paths),
            "causal paths...",
        )

        for path_id, path in enumerate(
            original_paths
        ):
            verification = (
                self.path_verifier.verify_path(
                    path_id=path_id,
                    original_path=path,
                    cf_top_k=cf_top_k,
                    mapping_top_k=(
                        mapping_top_k
                    ),
                    mapping_threshold=(
                        mapping_threshold
                    ),
                    max_hops=max_cf_hops,
                    max_paths=max_cf_paths,
                )
            )

            path_verifications.append(
                verification
            )

            print(
                f"- Path {path_id}: "
                f"{verification.status} "
                f"score="
                f"{verification.consistency_score:.4f}"
            )

        (
            verified_evidence,
            uncertain_evidence,
            removed_evidence,
        ) = self.evidence_verifier.verify_all(
            path_verifications=(
                path_verifications
            ),
            keep_threshold=keep_threshold,
            reject_threshold=reject_threshold,
            verified_top_k=verified_top_k,
        )

        consistency_score = (
            self._calculate_global_consistency(
                path_verifications
            )
        )

        confidence = (
            self._calculate_global_confidence(
                path_verifications=(
                    path_verifications
                ),
                verified_evidence=(
                    verified_evidence
                ),
                uncertain_evidence=(
                    uncertain_evidence
                ),
                removed_evidence=(
                    removed_evidence
                ),
            )
        )

        status_counts = {
            "SUPPORTED": sum(
                item.status == "SUPPORTED"
                for item in path_verifications
            ),
            "CONTRADICTED": sum(
                item.status == "CONTRADICTED"
                for item in path_verifications
            ),
            "UNRESOLVED": sum(
                item.status == "UNRESOLVED"
                for item in path_verifications
            ),
        }

        return VerificationResult(
            query=safe_string(
                self.store.retrieval_result.get(
                    "query"
                )
            ),
            configuration={
                "cf_top_k": cf_top_k,
                "mapping_top_k": mapping_top_k,
                "mapping_threshold": (
                    mapping_threshold
                ),
                "max_cf_hops": max_cf_hops,
                "max_cf_paths": max_cf_paths,
                "verified_top_k": verified_top_k,
                "keep_threshold": keep_threshold,
                "reject_threshold": (
                    reject_threshold
                ),
                "semantic_mapping_enabled": (
                    self.store
                    .enable_semantic_mapping
                ),
                "model_name": (
                    self.store.model_name
                ),
                "score_weights": {
                    "path_support": (
                        PATH_SUPPORT_WEIGHT
                    ),
                    "counterfactual_support": (
                        COUNTERFACTUAL_SUPPORT_WEIGHT
                    ),
                    "semantic_evidence": (
                        SEMANTIC_EVIDENCE_WEIGHT
                    ),
                    "graph_evidence": (
                        GRAPH_EVIDENCE_WEIGHT
                    ),
                },
            },
            statistics={
                "original_paths": len(
                    original_paths
                ),
                "path_status_counts": status_counts,
                "original_evidence": len(
                    self.store.retrieval_result.get(
                        "evidence",
                        [],
                    )
                ),
                "verified_evidence": len(
                    verified_evidence
                ),
                "uncertain_evidence": len(
                    uncertain_evidence
                ),
                "removed_evidence": len(
                    removed_evidence
                ),
            },
            path_verifications=[
                asdict(item)
                for item in path_verifications
            ],
            verified_evidence=[
                asdict(item)
                for item in verified_evidence
            ],
            uncertain_evidence=[
                asdict(item)
                for item in uncertain_evidence
            ],
            removed_evidence=[
                asdict(item)
                for item in removed_evidence
            ],
            consistency_score=consistency_score,
            confidence=confidence,
        )

    @staticmethod
    def _calculate_global_consistency(
        path_verifications: list[
            PathVerification
        ],
    ) -> float:
        if not path_verifications:
            return 0.0

        return clamp(
            sum(
                item.consistency_score
                for item in path_verifications
            )
            / len(path_verifications)
        )

    @staticmethod
    def _calculate_global_confidence(
        *,
        path_verifications: list[
            PathVerification
        ],
        verified_evidence: list[
            EvidenceVerification
        ],
        uncertain_evidence: list[
            EvidenceVerification
        ],
        removed_evidence: list[
            EvidenceVerification
        ],
    ) -> float:
        total_paths = len(path_verifications)

        if total_paths:
            resolved_ratio = (
                sum(
                    item.status != "UNRESOLVED"
                    for item in path_verifications
                )
                / total_paths
            )
        else:
            resolved_ratio = 0.0

        total_evidence = (
            len(verified_evidence)
            + len(uncertain_evidence)
            + len(removed_evidence)
        )

        if total_evidence:
            evidence_decision_ratio = (
                len(verified_evidence)
                + len(removed_evidence)
            ) / total_evidence
        else:
            evidence_decision_ratio = 0.0

        verified_score = (
            sum(
                item.verification_score
                for item in verified_evidence
            )
            / len(verified_evidence)
            if verified_evidence
            else 0.0
        )

        return clamp(
            0.40 * resolved_ratio
            + 0.30 * evidence_decision_ratio
            + 0.30 * verified_score
        )


# ============================================================
# OUTPUT
# ============================================================

def save_result(
    result: VerificationResult,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_serializable(asdict(result)),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nSaved verification result:",
        path,
    )


def print_summary(
    result: VerificationResult,
) -> None:
    print("\n" + "=" * 76)
    print("COUNTERFACTUAL VERIFICATION RESULT")
    print("=" * 76)

    print("Query:", result.query)

    print(
        "\nGlobal consistency score:",
        f"{result.consistency_score:.4f}",
    )
    print(
        "Global confidence:",
        f"{result.confidence:.4f}",
    )

    print("\nPath verification:")
    counts = result.statistics[
        "path_status_counts"
    ]

    for status, count in counts.items():
        print(f"- {status}: {count}")

    print("\nVerified evidence:")

    for item in result.verified_evidence:
        original = item["original_evidence"]

        print(
            f"- Rule {item['rule_id']} | "
            f"Điều {item['article_id']} | "
            f"verification="
            f"{item['verification_score']:.4f}"
        )
        print(
            "  Nếu:",
            original.get("condition", ""),
        )
        print(
            "  Thì:",
            original.get("effect", ""),
        )

    if not result.verified_evidence:
        print("- Không có evidence đạt KEEP threshold.")

    print("\nRemoved evidence:")

    for item in result.removed_evidence:
        print(
            f"- Rule {item['rule_id']} | "
            f"Điều {item['article_id']} | "
            f"verification="
            f"{item['verification_score']:.4f}"
        )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Graph-based counterfactual verification for "
            "Counterfactual-Aware CausalRAG."
        )
    )

    parser.add_argument(
        "--graph",
        default=GRAPH_PATH,
    )
    parser.add_argument(
        "--memory",
        default=MEMORY_PATH,
    )
    parser.add_argument(
        "--embeddings",
        default=EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--retrieval-result",
        default=RETRIEVAL_RESULT_PATH,
    )
    parser.add_argument(
        "--counterfactual-map",
        default=COUNTERFACTUAL_MAP_PATH,
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
    )

    parser.add_argument(
        "--cf-top-k",
        type=int,
        default=DEFAULT_CF_TOP_K,
    )
    parser.add_argument(
        "--mapping-top-k",
        type=int,
        default=DEFAULT_MAPPING_TOP_K,
    )
    parser.add_argument(
        "--mapping-threshold",
        type=float,
        default=DEFAULT_MAPPING_THRESHOLD,
    )
    parser.add_argument(
        "--max-cf-hops",
        type=int,
        default=DEFAULT_MAX_CF_HOPS,
    )
    parser.add_argument(
        "--max-cf-paths",
        type=int,
        default=DEFAULT_MAX_CF_PATHS,
    )
    parser.add_argument(
        "--verified-top-k",
        type=int,
        default=DEFAULT_VERIFIED_TOP_K,
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=DEFAULT_KEEP_THRESHOLD,
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=DEFAULT_REJECT_THRESHOLD,
    )

    parser.add_argument(
        "--disable-semantic-mapping",
        action="store_true",
        help=(
            "Chỉ dùng exact/heuristic mapping, không load "
            "SentenceTransformer và embeddings."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.cf_top_k < 1:
        raise ValueError(
            "--cf-top-k phải lớn hơn 0."
        )

    if args.mapping_top_k < 1:
        raise ValueError(
            "--mapping-top-k phải lớn hơn 0."
        )

    if args.max_cf_hops < 1:
        raise ValueError(
            "--max-cf-hops phải lớn hơn 0."
        )

    if not (
        0.0 <= args.reject_threshold
        <= args.keep_threshold
        <= 1.0
    ):
        raise ValueError(
            "Cần thỏa mãn: 0 <= reject-threshold "
            "<= keep-threshold <= 1."
        )

    store = CounterfactualResourceStore(
        graph_path=args.graph,
        memory_path=args.memory,
        embeddings_path=args.embeddings,
        retrieval_result_path=(
            args.retrieval_result
        ),
        counterfactual_map_path=(
            args.counterfactual_map
        ),
        model_name=args.model,
        enable_semantic_mapping=(
            not args.disable_semantic_mapping
        ),
    )

    pipeline = (
        CounterfactualVerificationPipeline(store)
    )

    result = pipeline.run(
        cf_top_k=args.cf_top_k,
        mapping_top_k=args.mapping_top_k,
        mapping_threshold=(
            args.mapping_threshold
        ),
        max_cf_hops=args.max_cf_hops,
        max_cf_paths=args.max_cf_paths,
        verified_top_k=args.verified_top_k,
        keep_threshold=args.keep_threshold,
        reject_threshold=(
            args.reject_threshold
        ),
    )

    print_summary(result)
    save_result(
        result,
        args.output,
    )


if __name__ == "__main__":
    main()
