from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

GRAPH_PATH = "data/legal_causal_knowledge_graph.graphml"
MEMORY_PATH = "data/causal_memory.csv"
FAISS_INDEX_PATH = "data/causal_memory.index"
EMBEDDINGS_PATH = "data/causal_memory_embeddings.npy"

MODEL_NAME = "BAAI/bge-m3"

OUTPUT_PATH = "data/retrieval_result.json"

DEFAULT_EVENT_TOP_K = 8
DEFAULT_DIRECT_RULE_TOP_K = 8
DEFAULT_SEMANTIC_POOL_SIZE = 100

DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_PATHS_PER_EVENT = 30
DEFAULT_MAX_CANDIDATE_RULES = 200
DEFAULT_FINAL_TOP_K = 12

DEFAULT_MIN_EVENT_SCORE = 0.20
DEFAULT_MIN_RULE_SCORE = 0.15

# Trọng số xếp hạng cuối cùng.
SEMANTIC_RULE_WEIGHT = 0.58
GRAPH_PATH_WEIGHT = 0.24
SEED_EVENT_WEIGHT = 0.12
DIRECT_RULE_BONUS_WEIGHT = 0.06

# Giảm dần đóng góp graph theo số hop.
HOP_DECAY = 0.82

# Thưởng nhẹ cho bridge event vì thường nối được nhiều bước suy luận.
BRIDGE_EVENT_BONUS = 0.05

# Phạt các path quá dài hoặc quá nhiều nhánh.
PATH_LENGTH_PENALTY = 0.03


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SemanticHit:
    memory_id: int
    score: float
    memory_type: str
    graph_node_id: str
    event_id: str = ""
    event_name: str = ""
    event_role: str = ""
    rule_id: str = ""
    article_id: str = ""


@dataclass
class CausalPathStep:
    hop: int
    source_event_node: str
    source_event_id: str
    source_event_name: str
    target_event_node: str
    target_event_id: str
    target_event_name: str
    rule_ids: list[str]
    article_ids: list[str]
    support_count: int


@dataclass
class CausalPath:
    seed_event_node: str
    seed_event_id: str
    seed_event_name: str
    seed_similarity: float
    direction: str
    event_nodes: list[str]
    rule_ids: list[str]
    steps: list[CausalPathStep]
    graph_score: float


@dataclass
class RuleEvidence:
    rank: int
    rule_id: str
    rule_node_id: str
    article_id: str
    article_title: str
    legal_subject: str
    condition: str
    effect: str
    condition_event: str
    condition_event_name: str
    effect_event: str
    effect_event_name: str
    causal_type: str
    semantic_score: float
    graph_score: float
    seed_event_score: float
    direct_rule_score: float
    final_score: float
    matched_seed_events: list[str] = field(default_factory=list)
    path_ids: list[int] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    configuration: dict[str, Any]
    retrieved_events: list[dict[str, Any]]
    direct_rule_hits: list[dict[str, Any]]
    causal_paths: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    statistics: dict[str, Any]


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
        "true", "1", "yes", "y"
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


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        value = safe_string(value)

        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def make_rule_node_id(rule_id: str) -> str:
    return f"RULE::{rule_id}"


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


# ============================================================
# RESOURCE STORE
# ============================================================

class CausalResourceStore:
    """Load và kiểm tra toàn bộ tài nguyên retrieval.

    File memory và embeddings phải giữ đúng thứ tự memory_id.
    """

    def __init__(
        self,
        *,
        graph_path: str,
        memory_path: str,
        index_path: str,
        embeddings_path: str,
        model_name: str,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.memory_path = Path(memory_path)
        self.index_path = Path(index_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name

        self.graph = self._load_graph()
        self.memory_df = self._load_memory()
        self.index = self._load_index()
        self.embeddings = self._load_embeddings()
        self.model = self._load_model()

        self._validate_resources()
        self._build_lookup_tables()
        self.causal_event_graph = self._build_causal_event_graph()

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
                f"Không tìm thấy memory CSV: {self.memory_path}"
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
            "embedding_text",
        }

        missing = required_columns - set(memory_df.columns)

        if missing:
            raise ValueError(
                "Causal memory thiếu các cột bắt buộc: "
                f"{sorted(missing)}"
            )

        memory_df["memory_id"] = pd.to_numeric(
            memory_df["memory_id"],
            errors="raise",
        ).astype(np.int64)

        return memory_df

    def _load_index(self) -> faiss.Index:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FAISS index: {self.index_path}"
            )

        print(f"Loading FAISS index: {self.index_path}")
        return faiss.read_index(str(self.index_path))

    def _load_embeddings(self) -> np.ndarray:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy embeddings: "
                f"{self.embeddings_path}"
            )

        print(f"Loading embeddings: {self.embeddings_path}")
        embeddings = np.load(self.embeddings_path)

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def _load_model(self) -> SentenceTransformer:
        print(f"Loading embedding model: {self.model_name}")
        return SentenceTransformer(self.model_name)

    def _validate_resources(self) -> None:
        expected_ids = np.arange(
            len(self.memory_df),
            dtype=np.int64,
        )
        actual_ids = self.memory_df[
            "memory_id"
        ].to_numpy(dtype=np.int64)

        if not np.array_equal(expected_ids, actual_ids):
            raise ValueError(
                "memory_id phải đúng bằng vị trí vector trong FAISS."
            )

        if self.index.ntotal != len(self.memory_df):
            raise ValueError(
                "Số vector trong FAISS không khớp số dòng memory."
            )

        if self.embeddings.shape != (
            self.index.ntotal,
            self.index.d,
        ):
            raise ValueError(
                "Shape embeddings không khớp FAISS index."
            )

        memory_types = set(
            self.memory_df["memory_type"]
            .astype(str)
            .str.upper()
        )

        if "RULE" not in memory_types:
            raise ValueError(
                "Causal memory không chứa RULE records."
            )

        if "EVENT" not in memory_types:
            raise ValueError(
                "Causal memory không chứa EVENT records. "
                "Hãy chạy lại file 2 không dùng --rule-only."
            )

        graph_nodes = set(self.graph.nodes)
        missing_nodes = (
            set(self.memory_df["graph_node_id"]) - graph_nodes
        )

        if missing_nodes:
            raise ValueError(
                "Một số graph_node_id trong memory không tồn tại "
                f"trong graph. Ví dụ: {sorted(missing_nodes)[:10]}"
            )

        print("Resource validation: OK")
        print("- Memory records:", len(self.memory_df))
        print("- FAISS dimension:", self.index.d)

    def _build_lookup_tables(self) -> None:
        self.memory_by_id = self.memory_df.set_index(
            "memory_id",
            drop=False,
        )

        rule_df = self.memory_df[
            self.memory_df["memory_type"].str.upper() == "RULE"
        ].copy()

        event_df = self.memory_df[
            self.memory_df["memory_type"].str.upper() == "EVENT"
        ].copy()

        self.rule_memory_ids = rule_df[
            "memory_id"
        ].to_numpy(dtype=np.int64)

        self.event_memory_ids = event_df[
            "memory_id"
        ].to_numpy(dtype=np.int64)

        self.rule_by_id: dict[str, pd.Series] = {}

        for _, row in rule_df.iterrows():
            rule_id = safe_string(row.get("rule_id"))

            if rule_id:
                self.rule_by_id[rule_id] = row

        self.rule_memory_id_by_rule_id = {
            rule_id: int(row["memory_id"])
            for rule_id, row in self.rule_by_id.items()
        }

        self.event_memory_id_by_node = {
            safe_string(row["graph_node_id"]): int(row["memory_id"])
            for _, row in event_df.iterrows()
        }

    def _build_causal_event_graph(self) -> nx.DiGraph:
        """Gộp các cạnh CAUSES song song thành một DiGraph event-level."""

        causal_graph = nx.DiGraph()

        for node_id, data in self.graph.nodes(data=True):
            if safe_string(data.get("node_type")) == "EVENT":
                causal_graph.add_node(
                    node_id,
                    **dict(data),
                )

        if self.graph.is_multigraph():
            edge_iterator = self.graph.edges(
                keys=True,
                data=True,
            )

            for source, target, _, data in edge_iterator:
                self._add_causal_edge(
                    causal_graph,
                    source,
                    target,
                    data,
                )
        else:
            for source, target, data in self.graph.edges(
                data=True
            ):
                self._add_causal_edge(
                    causal_graph,
                    source,
                    target,
                    data,
                )

        if causal_graph.number_of_edges() == 0:
            raise ValueError(
                "Không tìm thấy cạnh CAUSES trong graph."
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
    def _add_causal_edge(
        causal_graph: nx.DiGraph,
        source: str,
        target: str,
        edge_data: dict[str, Any],
    ) -> None:
        if safe_string(edge_data.get("relation")) != "CAUSES":
            return

        rule_ids = split_csv_values(
            edge_data.get("rule_ids")
        )

        if not rule_ids:
            rule_id = safe_string(
                edge_data.get("rule_id")
            )
            if rule_id:
                rule_ids = [rule_id]

        article_ids = split_csv_values(
            edge_data.get("article_ids")
        )

        if not article_ids:
            article_id = safe_string(
                edge_data.get("article_id")
            )
            if article_id:
                article_ids = [article_id]

        support_count = max(
            1,
            safe_int(edge_data.get("support_count"), 1),
        )

        if causal_graph.has_edge(source, target):
            existing = causal_graph[source][target]
            existing["rule_ids"] = unique_preserve_order(
                list(existing["rule_ids"]) + rule_ids
            )
            existing["article_ids"] = unique_preserve_order(
                list(existing["article_ids"]) + article_ids
            )
            existing["support_count"] += support_count
        else:
            causal_graph.add_edge(
                source,
                target,
                relation="CAUSES",
                rule_ids=rule_ids,
                article_ids=article_ids,
                support_count=support_count,
            )

    def encode_query(self, query: str) -> np.ndarray:
        query = safe_string(query)

        if not query:
            raise ValueError("Query không được để trống.")

        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vector,
            dtype=np.float32,
        )[0]


# ============================================================
# MULTI-HOP CAUSAL RETRIEVER
# ============================================================

class MultiHopCausalRetriever:
    def __init__(
        self,
        store: CausalResourceStore,
    ) -> None:
        self.store = store

    # --------------------------------------------------------
    # STAGE 1: SEMANTIC SEED RETRIEVAL
    # --------------------------------------------------------

    def semantic_search(
        self,
        query_vector: np.ndarray,
        *,
        pool_size: int,
    ) -> list[SemanticHit]:
        pool_size = min(
            max(1, pool_size),
            self.store.index.ntotal,
        )

        scores, indices = self.store.index.search(
            query_vector.reshape(1, -1),
            pool_size,
        )

        hits: list[SemanticHit] = []

        for score, memory_id in zip(
            scores[0],
            indices[0],
        ):
            if memory_id < 0:
                continue

            row = self.store.memory_by_id.loc[
                int(memory_id)
            ]

            hits.append(
                SemanticHit(
                    memory_id=int(memory_id),
                    score=float(score),
                    memory_type=safe_string(
                        row.get("memory_type")
                    ).upper(),
                    graph_node_id=safe_string(
                        row.get("graph_node_id")
                    ),
                    event_id=safe_string(
                        row.get("event_id")
                    ),
                    event_name=safe_string(
                        row.get("event_name")
                    ),
                    event_role=safe_string(
                        row.get("event_role")
                    ),
                    rule_id=safe_string(
                        row.get("rule_id")
                    ),
                    article_id=safe_string(
                        row.get("article_id")
                    ),
                )
            )

        return hits

    def select_event_seeds(
        self,
        hits: list[SemanticHit],
        *,
        top_k: int,
        min_score: float,
    ) -> list[SemanticHit]:
        selected = [
            hit
            for hit in hits
            if (
                hit.memory_type == "EVENT"
                and hit.score >= min_score
            )
        ][:top_k]

        # Nếu combined search chưa trả đủ EVENT, tính trực tiếp
        # cosine trên event vectors để bảo đảm recall.
        if len(selected) < top_k:
            selected_ids = {
                hit.memory_id for hit in selected
            }

            # Query vector sẽ được gắn tạm bởi retrieve().
            query_vector = self._active_query_vector

            event_vectors = self.store.embeddings[
                self.store.event_memory_ids
            ]
            scores = event_vectors @ query_vector

            order = np.argsort(-scores)

            for local_index in order:
                memory_id = int(
                    self.store.event_memory_ids[local_index]
                )
                score = float(scores[local_index])

                if memory_id in selected_ids:
                    continue
                if score < min_score:
                    break

                row = self.store.memory_by_id.loc[memory_id]

                selected.append(
                    SemanticHit(
                        memory_id=memory_id,
                        score=score,
                        memory_type="EVENT",
                        graph_node_id=safe_string(
                            row.get("graph_node_id")
                        ),
                        event_id=safe_string(
                            row.get("event_id")
                        ),
                        event_name=safe_string(
                            row.get("event_name")
                        ),
                        event_role=safe_string(
                            row.get("event_role")
                        ),
                    )
                )
                selected_ids.add(memory_id)

                if len(selected) >= top_k:
                    break

        return selected

    def select_direct_rule_hits(
        self,
        hits: list[SemanticHit],
        *,
        top_k: int,
        min_score: float,
    ) -> list[SemanticHit]:
        selected = [
            hit
            for hit in hits
            if (
                hit.memory_type == "RULE"
                and hit.score >= min_score
            )
        ][:top_k]

        if len(selected) < top_k:
            selected_ids = {
                hit.memory_id for hit in selected
            }
            query_vector = self._active_query_vector

            rule_vectors = self.store.embeddings[
                self.store.rule_memory_ids
            ]
            scores = rule_vectors @ query_vector
            order = np.argsort(-scores)

            for local_index in order:
                memory_id = int(
                    self.store.rule_memory_ids[local_index]
                )
                score = float(scores[local_index])

                if memory_id in selected_ids:
                    continue
                if score < min_score:
                    break

                row = self.store.memory_by_id.loc[memory_id]

                selected.append(
                    SemanticHit(
                        memory_id=memory_id,
                        score=score,
                        memory_type="RULE",
                        graph_node_id=safe_string(
                            row.get("graph_node_id")
                        ),
                        rule_id=safe_string(
                            row.get("rule_id")
                        ),
                        article_id=safe_string(
                            row.get("article_id")
                        ),
                    )
                )
                selected_ids.add(memory_id)

                if len(selected) >= top_k:
                    break

        return selected

    # --------------------------------------------------------
    # STAGE 2: MULTI-HOP GRAPH EXPANSION
    # --------------------------------------------------------

    def expand_event_seed(
        self,
        seed: SemanticHit,
        *,
        max_hops: int,
        max_paths: int,
        direction: str,
    ) -> list[CausalPath]:
        graph = self.store.causal_event_graph
        seed_node = seed.graph_node_id

        if seed_node not in graph:
            return []

        directions = (
            ["forward", "backward"]
            if direction == "both"
            else [direction]
        )

        all_paths: list[CausalPath] = []

        for one_direction in directions:
            all_paths.extend(
                self._bounded_path_search(
                    seed=seed,
                    max_hops=max_hops,
                    max_paths=max_paths,
                    direction=one_direction,
                )
            )

        all_paths.sort(
            key=lambda path: path.graph_score,
            reverse=True,
        )

        return all_paths[:max_paths]

    def _bounded_path_search(
        self,
        *,
        seed: SemanticHit,
        max_hops: int,
        max_paths: int,
        direction: str,
    ) -> list[CausalPath]:
        graph = self.store.causal_event_graph
        seed_node = seed.graph_node_id

        # queue item:
        # current node, event path, rule ids, steps
        queue: list[
            tuple[
                str,
                list[str],
                list[str],
                list[CausalPathStep],
            ]
        ] = [
            (seed_node, [seed_node], [], [])
        ]

        completed_paths: list[CausalPath] = []
        queue_index = 0

        while (
            queue_index < len(queue)
            and len(completed_paths) < max_paths
        ):
            (
                current_node,
                event_nodes,
                path_rule_ids,
                steps,
            ) = queue[queue_index]
            queue_index += 1

            hop = len(steps)

            if hop >= max_hops:
                continue

            if direction == "forward":
                neighbors = graph.successors(current_node)
            elif direction == "backward":
                neighbors = graph.predecessors(current_node)
            else:
                raise ValueError(
                    "direction phải là forward, backward hoặc both."
                )

            for neighbor in neighbors:
                if neighbor in event_nodes:
                    continue

                if direction == "forward":
                    source = current_node
                    target = neighbor
                else:
                    source = neighbor
                    target = current_node

                edge_data = graph[source][target]

                edge_rule_ids = unique_preserve_order(
                    edge_data.get("rule_ids", [])
                )
                edge_article_ids = unique_preserve_order(
                    edge_data.get("article_ids", [])
                )

                next_step = CausalPathStep(
                    hop=hop + 1,
                    source_event_node=source,
                    source_event_id=event_id_from_node(
                        graph,
                        source,
                    ),
                    source_event_name=event_name_from_node(
                        graph,
                        source,
                    ),
                    target_event_node=target,
                    target_event_id=event_id_from_node(
                        graph,
                        target,
                    ),
                    target_event_name=event_name_from_node(
                        graph,
                        target,
                    ),
                    rule_ids=edge_rule_ids,
                    article_ids=edge_article_ids,
                    support_count=safe_int(
                        edge_data.get("support_count"),
                        1,
                    ),
                )

                next_event_nodes = (
                    event_nodes + [neighbor]
                )
                next_rule_ids = unique_preserve_order(
                    path_rule_ids + edge_rule_ids
                )
                next_steps = steps + [next_step]

                graph_score = self._score_path(
                    seed_similarity=seed.score,
                    event_nodes=next_event_nodes,
                    steps=next_steps,
                )

                completed_paths.append(
                    CausalPath(
                        seed_event_node=seed_node,
                        seed_event_id=seed.event_id,
                        seed_event_name=(
                            seed.event_name or seed.event_id
                        ),
                        seed_similarity=seed.score,
                        direction=direction,
                        event_nodes=next_event_nodes,
                        rule_ids=next_rule_ids,
                        steps=next_steps,
                        graph_score=graph_score,
                    )
                )

                if len(next_steps) < max_hops:
                    queue.append(
                        (
                            neighbor,
                            next_event_nodes,
                            next_rule_ids,
                            next_steps,
                        )
                    )

                if len(completed_paths) >= max_paths:
                    break

        completed_paths.sort(
            key=lambda path: path.graph_score,
            reverse=True,
        )
        return completed_paths

    def _score_path(
        self,
        *,
        seed_similarity: float,
        event_nodes: list[str],
        steps: list[CausalPathStep],
    ) -> float:
        hop_count = len(steps)

        if hop_count == 0:
            return 0.0

        support_score = sum(
            math.log1p(max(1, step.support_count))
            for step in steps
        ) / hop_count

        support_score = min(
            1.0,
            support_score / math.log(4.0),
        )

        bridge_count = 0

        # Không tính seed và endpoint, chỉ tính event trung gian.
        for node_id in event_nodes[1:-1]:
            data = self.store.causal_event_graph.nodes[
                node_id
            ]

            if (
                safe_bool(data.get("is_condition"))
                and safe_bool(data.get("is_effect"))
            ):
                bridge_count += 1

        bridge_bonus = min(
            0.15,
            bridge_count * BRIDGE_EVENT_BONUS,
        )

        depth_factor = HOP_DECAY ** (hop_count - 1)
        length_penalty = PATH_LENGTH_PENALTY * max(
            0,
            hop_count - 1,
        )

        score = (
            0.62 * seed_similarity
            + 0.28 * support_score
            + bridge_bonus
        )

        score = score * depth_factor - length_penalty

        return max(0.0, min(1.0, score))

    # --------------------------------------------------------
    # STAGE 3: CANDIDATE RULE GENERATION
    # --------------------------------------------------------

    def collect_candidate_rules(
        self,
        *,
        event_seeds: list[SemanticHit],
        direct_rule_hits: list[SemanticHit],
        paths: list[CausalPath],
        max_candidates: int,
    ) -> tuple[
        list[str],
        dict[str, float],
        dict[str, float],
        dict[str, list[str]],
        dict[str, list[int]],
    ]:
        graph_score_by_rule: dict[str, float] = {}
        seed_score_by_rule: dict[str, float] = {}
        seed_events_by_rule: dict[str, list[str]] = {}
        path_ids_by_rule: dict[str, list[int]] = {}

        event_seed_by_node = {
            seed.graph_node_id: seed
            for seed in event_seeds
        }

        # Rule trực tiếp nối với seed event, kể cả seed không sinh path.
        for seed in event_seeds:
            event_node = seed.graph_node_id

            for rule_id in self._rules_touching_event(
                event_node
            ):
                seed_score_by_rule[rule_id] = max(
                    seed_score_by_rule.get(rule_id, 0.0),
                    seed.score,
                )
                seed_events_by_rule.setdefault(
                    rule_id,
                    [],
                ).append(
                    seed.event_name or seed.event_id
                )

        # Rule xuất hiện trên causal path.
        for path_id, path in enumerate(paths):
            for rule_id in path.rule_ids:
                graph_score_by_rule[rule_id] = max(
                    graph_score_by_rule.get(rule_id, 0.0),
                    path.graph_score,
                )
                seed_score_by_rule[rule_id] = max(
                    seed_score_by_rule.get(rule_id, 0.0),
                    path.seed_similarity,
                )
                seed_events_by_rule.setdefault(
                    rule_id,
                    [],
                ).append(
                    path.seed_event_name
                    or path.seed_event_id
                )
                path_ids_by_rule.setdefault(
                    rule_id,
                    [],
                ).append(path_id)

        direct_rule_ids = [
            hit.rule_id
            for hit in direct_rule_hits
            if hit.rule_id
        ]

        candidate_ids = unique_preserve_order(
            list(graph_score_by_rule)
            + list(seed_score_by_rule)
            + direct_rule_ids
        )

        # Ưu tiên sơ bộ theo graph + seed trước khi semantic rerank.
        candidate_ids.sort(
            key=lambda rule_id: (
                graph_score_by_rule.get(rule_id, 0.0)
                + seed_score_by_rule.get(rule_id, 0.0)
            ),
            reverse=True,
        )

        candidate_ids = candidate_ids[:max_candidates]

        for rule_id in seed_events_by_rule:
            seed_events_by_rule[rule_id] = (
                unique_preserve_order(
                    seed_events_by_rule[rule_id]
                )
            )

        for rule_id in path_ids_by_rule:
            path_ids_by_rule[rule_id] = sorted(
                set(path_ids_by_rule[rule_id])
            )

        return (
            candidate_ids,
            graph_score_by_rule,
            seed_score_by_rule,
            seed_events_by_rule,
            path_ids_by_rule,
        )

    def _rules_touching_event(
        self,
        event_node: str,
    ) -> list[str]:
        graph = self.store.causal_event_graph
        rule_ids: list[str] = []

        if event_node not in graph:
            return []

        for _, target, edge_data in graph.out_edges(
            event_node,
            data=True,
        ):
            rule_ids.extend(
                edge_data.get("rule_ids", [])
            )

        for source, _, edge_data in graph.in_edges(
            event_node,
            data=True,
        ):
            rule_ids.extend(
                edge_data.get("rule_ids", [])
            )

        return unique_preserve_order(rule_ids)

    # --------------------------------------------------------
    # STAGE 4: RULE SEMANTIC RE-RANKING
    # --------------------------------------------------------

    def rerank_rules(
        self,
        *,
        query_vector: np.ndarray,
        candidate_rule_ids: list[str],
        direct_rule_hits: list[SemanticHit],
        graph_score_by_rule: dict[str, float],
        seed_score_by_rule: dict[str, float],
        seed_events_by_rule: dict[str, list[str]],
        path_ids_by_rule: dict[str, list[int]],
        final_top_k: int,
    ) -> list[RuleEvidence]:
        direct_score_by_rule = {
            hit.rule_id: hit.score
            for hit in direct_rule_hits
            if hit.rule_id
        }

        evidences: list[RuleEvidence] = []

        for rule_id in candidate_rule_ids:
            row = self.store.rule_by_id.get(rule_id)

            if row is None:
                continue

            memory_id = int(row["memory_id"])
            rule_vector = self.store.embeddings[
                memory_id
            ]

            semantic_score = float(
                np.dot(query_vector, rule_vector)
            )
            graph_score = graph_score_by_rule.get(
                rule_id,
                0.0,
            )
            seed_event_score = seed_score_by_rule.get(
                rule_id,
                0.0,
            )
            direct_rule_score = direct_score_by_rule.get(
                rule_id,
                0.0,
            )

            final_score = (
                SEMANTIC_RULE_WEIGHT * semantic_score
                + GRAPH_PATH_WEIGHT * graph_score
                + SEED_EVENT_WEIGHT * seed_event_score
                + DIRECT_RULE_BONUS_WEIGHT
                * direct_rule_score
            )

            evidences.append(
                RuleEvidence(
                    rank=0,
                    rule_id=rule_id,
                    rule_node_id=safe_string(
                        row.get("rule_node_id")
                    )
                    or make_rule_node_id(rule_id),
                    article_id=safe_string(
                        row.get("article_id")
                    ),
                    article_title=safe_string(
                        row.get("article_title")
                    ),
                    legal_subject=safe_string(
                        row.get("legal_subject")
                    ),
                    condition=safe_string(
                        row.get("condition")
                    ),
                    effect=safe_string(
                        row.get("effect")
                    ),
                    condition_event=safe_string(
                        row.get("condition_event")
                    ),
                    condition_event_name=safe_string(
                        row.get("condition_event_name")
                    ),
                    effect_event=safe_string(
                        row.get("effect_event")
                    ),
                    effect_event_name=safe_string(
                        row.get("effect_event_name")
                    ),
                    causal_type=safe_string(
                        row.get("causal_type")
                    ),
                    semantic_score=semantic_score,
                    graph_score=graph_score,
                    seed_event_score=seed_event_score,
                    direct_rule_score=direct_rule_score,
                    final_score=final_score,
                    matched_seed_events=seed_events_by_rule.get(
                        rule_id,
                        [],
                    ),
                    path_ids=path_ids_by_rule.get(
                        rule_id,
                        [],
                    ),
                )
            )

        evidences.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        evidences = evidences[:final_top_k]

        for rank, evidence in enumerate(
            evidences,
            start=1,
        ):
            evidence.rank = rank

        return evidences

    # --------------------------------------------------------
    # END-TO-END RETRIEVAL
    # --------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        event_top_k: int = DEFAULT_EVENT_TOP_K,
        direct_rule_top_k: int = DEFAULT_DIRECT_RULE_TOP_K,
        semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_paths_per_event: int = DEFAULT_MAX_PATHS_PER_EVENT,
        max_candidate_rules: int = DEFAULT_MAX_CANDIDATE_RULES,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        min_event_score: float = DEFAULT_MIN_EVENT_SCORE,
        min_rule_score: float = DEFAULT_MIN_RULE_SCORE,
        direction: str = "both",
    ) -> RetrievalResult:
        if event_top_k < 1:
            raise ValueError("event_top_k phải lớn hơn 0.")
        if direct_rule_top_k < 0:
            raise ValueError(
                "direct_rule_top_k không được âm."
            )
        if max_hops < 1:
            raise ValueError("max_hops phải lớn hơn 0.")
        if direction not in {
            "forward",
            "backward",
            "both",
        }:
            raise ValueError(
                "direction phải là forward, backward hoặc both."
            )

        query = safe_string(query)
        query_vector = self.store.encode_query(query)
        self._active_query_vector = query_vector

        semantic_hits = self.semantic_search(
            query_vector,
            pool_size=semantic_pool_size,
        )

        event_seeds = self.select_event_seeds(
            semantic_hits,
            top_k=event_top_k,
            min_score=min_event_score,
        )

        direct_rule_hits = self.select_direct_rule_hits(
            semantic_hits,
            top_k=direct_rule_top_k,
            min_score=min_rule_score,
        )

        all_paths: list[CausalPath] = []

        for seed in event_seeds:
            seed_paths = self.expand_event_seed(
                seed,
                max_hops=max_hops,
                max_paths=max_paths_per_event,
                direction=direction,
            )
            all_paths.extend(seed_paths)

        # Xếp hạng toàn cục và loại path trùng.
        all_paths = self._deduplicate_paths(all_paths)
        all_paths.sort(
            key=lambda path: path.graph_score,
            reverse=True,
        )

        (
            candidate_rule_ids,
            graph_score_by_rule,
            seed_score_by_rule,
            seed_events_by_rule,
            path_ids_by_rule,
        ) = self.collect_candidate_rules(
            event_seeds=event_seeds,
            direct_rule_hits=direct_rule_hits,
            paths=all_paths,
            max_candidates=max_candidate_rules,
        )

        evidence = self.rerank_rules(
            query_vector=query_vector,
            candidate_rule_ids=candidate_rule_ids,
            direct_rule_hits=direct_rule_hits,
            graph_score_by_rule=graph_score_by_rule,
            seed_score_by_rule=seed_score_by_rule,
            seed_events_by_rule=seed_events_by_rule,
            path_ids_by_rule=path_ids_by_rule,
            final_top_k=final_top_k,
        )

        result = RetrievalResult(
            query=query,
            configuration={
                "event_top_k": event_top_k,
                "direct_rule_top_k": direct_rule_top_k,
                "semantic_pool_size": semantic_pool_size,
                "max_hops": max_hops,
                "max_paths_per_event": max_paths_per_event,
                "max_candidate_rules": max_candidate_rules,
                "final_top_k": final_top_k,
                "min_event_score": min_event_score,
                "min_rule_score": min_rule_score,
                "direction": direction,
                "model_name": self.store.model_name,
                "score_weights": {
                    "semantic_rule": SEMANTIC_RULE_WEIGHT,
                    "graph_path": GRAPH_PATH_WEIGHT,
                    "seed_event": SEED_EVENT_WEIGHT,
                    "direct_rule_bonus": (
                        DIRECT_RULE_BONUS_WEIGHT
                    ),
                },
            },
            retrieved_events=[
                asdict(hit) for hit in event_seeds
            ],
            direct_rule_hits=[
                asdict(hit) for hit in direct_rule_hits
            ],
            causal_paths=[
                asdict(path) for path in all_paths
            ],
            evidence=[
                asdict(item) for item in evidence
            ],
            statistics={
                "semantic_hits": len(semantic_hits),
                "retrieved_event_seeds": len(event_seeds),
                "direct_rule_hits": len(
                    direct_rule_hits
                ),
                "causal_paths": len(all_paths),
                "candidate_rules": len(
                    candidate_rule_ids
                ),
                "final_evidence": len(evidence),
                "unique_path_rules": len(
                    {
                        rule_id
                        for path in all_paths
                        for rule_id in path.rule_ids
                    }
                ),
            },
        )

        del self._active_query_vector
        return result

    @staticmethod
    def _deduplicate_paths(
        paths: list[CausalPath],
    ) -> list[CausalPath]:
        best_by_key: dict[
            tuple[str, str, tuple[str, ...]],
            CausalPath,
        ] = {}

        for path in paths:
            key = (
                path.seed_event_node,
                path.direction,
                tuple(path.event_nodes),
            )

            existing = best_by_key.get(key)

            if (
                existing is None
                or path.graph_score > existing.graph_score
            ):
                best_by_key[key] = path

        return list(best_by_key.values())


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_result(
    result: RetrievalResult,
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
            asdict(result),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved retrieval result: {path}")


def print_result_summary(
    result: RetrievalResult,
) -> None:
    print("\n" + "=" * 72)
    print("MULTI-HOP CAUSAL RETRIEVAL RESULT")
    print("=" * 72)
    print("Query:", result.query)

    print("\nTop retrieved events:")
    for index, event in enumerate(
        result.retrieved_events,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{event.get('event_name') or event.get('event_id')} "
            f"[{event.get('event_role')}] "
            f"score={event.get('score', 0.0):.4f}"
        )

    print("\nTop evidence:")
    for item in result.evidence:
        print(
            f"{item['rank']:>2}. "
            f"Rule {item['rule_id']} - "
            f"Điều {item['article_id']} | "
            f"final={item['final_score']:.4f} | "
            f"semantic={item['semantic_score']:.4f} | "
            f"graph={item['graph_score']:.4f}"
        )
        print(
            f"    Nếu: {item['condition']}"
        )
        print(
            f"    Thì: {item['effect']}"
        )

    print("\nStatistics:")
    for key, value in result.statistics.items():
        print(f"- {key}: {value}")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage multi-hop CausalRAG retriever: "
            "Event semantic retrieval -> causal graph expansion "
            "-> candidate rule generation -> rule re-ranking."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Câu hỏi pháp lý cần truy hồi.",
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
        "--index",
        default=FAISS_INDEX_PATH,
    )
    parser.add_argument(
        "--embeddings",
        default=EMBEDDINGS_PATH,
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
        "--event-top-k",
        type=int,
        default=DEFAULT_EVENT_TOP_K,
    )
    parser.add_argument(
        "--direct-rule-top-k",
        type=int,
        default=DEFAULT_DIRECT_RULE_TOP_K,
    )
    parser.add_argument(
        "--semantic-pool-size",
        type=int,
        default=DEFAULT_SEMANTIC_POOL_SIZE,
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=DEFAULT_MAX_HOPS,
    )
    parser.add_argument(
        "--max-paths-per-event",
        type=int,
        default=DEFAULT_MAX_PATHS_PER_EVENT,
    )
    parser.add_argument(
        "--max-candidate-rules",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_RULES,
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
    )
    parser.add_argument(
        "--min-event-score",
        type=float,
        default=DEFAULT_MIN_EVENT_SCORE,
    )
    parser.add_argument(
        "--min-rule-score",
        type=float,
        default=DEFAULT_MIN_RULE_SCORE,
    )
    parser.add_argument(
        "--direction",
        choices=[
            "forward",
            "backward",
            "both",
        ],
        default="both",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    query = args.query

    if not query:
        query = input(
            "Nhập câu hỏi pháp lý: "
        ).strip()

    store = CausalResourceStore(
        graph_path=args.graph,
        memory_path=args.memory,
        index_path=args.index,
        embeddings_path=args.embeddings,
        model_name=args.model,
    )

    retriever = MultiHopCausalRetriever(store)

    result = retriever.retrieve(
        query,
        event_top_k=args.event_top_k,
        direct_rule_top_k=args.direct_rule_top_k,
        semantic_pool_size=args.semantic_pool_size,
        max_hops=args.max_hops,
        max_paths_per_event=args.max_paths_per_event,
        max_candidate_rules=args.max_candidate_rules,
        final_top_k=args.final_top_k,
        min_event_score=args.min_event_score,
        min_rule_score=args.min_rule_score,
        direction=args.direction,
    )

    print_result_summary(result)
    save_result(result, args.output)


if __name__ == "__main__":
    main()
