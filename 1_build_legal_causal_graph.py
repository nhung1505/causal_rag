from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_INPUT_PATH = "data/4_blhs_merged.json"
DEFAULT_GRAPHML_PATH = "data/legal_causal_knowledge_graph.graphml"
DEFAULT_GEXF_PATH = "data/legal_causal_knowledge_graph.gexf"
DEFAULT_STATS_PATH = "data/legal_causal_graph_stats.json"
DEFAULT_CHAINS_PATH = "data/legal_causal_two_hop_chains.csv"


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    """
    Chuyển một giá trị thành chuỗi an toàn.

    Trả về chuỗi rỗng nếu value là:
    - None
    - NaN
    - pd.NA
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def remove_vietnamese_accents(text: str) -> str:
    """
    Bỏ dấu tiếng Việt.

    Ví dụ:
        "chịu trách nhiệm" -> "chiu trach nhiem"
    """

    text = safe_string(text)
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)

    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )


def normalize_identifier(text: Any) -> str:
    """
    Chuẩn hóa condition_norm/effect_norm thành ID ổn định.

    Ví dụ:
        "Chịu trách nhiệm hình sự"
        -> "CHIU_TRACH_NHIEM_HINH_SU"
    """

    text = remove_vietnamese_accents(
        safe_string(text)
    ).upper()

    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_article_id(value: Any) -> str:
    """
    Chuẩn hóa article_id.

    Ví dụ:
        2.0 -> "2"
        2   -> "2"
    """

    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def normalize_rule_id(value: Any, fallback: int) -> str:
    """
    Chuẩn hóa index thành rule_id.
    """

    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    if not text:
        text = str(fallback)

    return text


def merge_unique_text(
    existing_text: str,
    new_text: str,
    separator: str = " || ",
) -> str:
    """
    Ghép các mô tả khác nhau của cùng một event.

    Ví dụ cùng event_norm có thể xuất hiện với nhiều cách diễn đạt:
        "chịu trách nhiệm hình sự"
        "phải chịu trách nhiệm hình sự"
    """

    existing_text = safe_string(existing_text)
    new_text = safe_string(new_text)

    if not existing_text:
        return new_text

    if not new_text:
        return existing_text

    existing_items = {
        item.strip()
        for item in existing_text.split(separator)
        if item.strip()
    }

    if new_text in existing_items:
        return existing_text

    return existing_text + separator + new_text


def increment_csv_attribute(
    existing_value: str,
    new_value: str,
) -> str:
    """
    Lưu một tập giá trị thành chuỗi phân tách bằng dấu phẩy.

    GraphML không hỗ trợ trực tiếp list/set nên cần chuyển sang string.
    """

    existing_value = safe_string(existing_value)
    new_value = safe_string(new_value)

    values = {
        item.strip()
        for item in existing_value.split(",")
        if item.strip()
    }

    if new_value:
        values.add(new_value)

    return ",".join(sorted(values))


# ============================================================
# INPUT LOADING
# ============================================================

def load_input_dataframe(input_path: str) -> pd.DataFrame:
    """
    Đọc JSON hoặc CSV thành DataFrame.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đầu vào: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            dataframe = pd.read_json(path)
        except ValueError:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            dataframe = pd.DataFrame(data)

    elif suffix == ".csv":
        dataframe = pd.read_csv(path)

    else:
        raise ValueError(
            "Chỉ hỗ trợ file JSON hoặc CSV."
        )

    required_columns = {
        "article_id",
        "legal_subject",
        "condition",
        "effect",
        "condition_norm",
        "effect_norm",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dữ liệu thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )

    optional_columns = {
        "index": "",
        "article_title": "",
        "content": "",
        "causal_type": "",
    }

    for column_name, default_value in (
        optional_columns.items()
    ):
        if column_name not in dataframe.columns:
            dataframe[column_name] = default_value

    print(f"Input rows: {len(dataframe)}")

    return dataframe


# ============================================================
# NODE ID BUILDERS
# ============================================================

def make_article_node_id(article_id: str) -> str:
    return f"ARTICLE::{article_id}"


def make_rule_node_id(rule_id: str) -> str:
    return f"RULE::{rule_id}"


def make_event_node_id(event_norm: str) -> str:
    return f"EVENT::{event_norm}"


def make_subject_node_id(subject_norm: str) -> str:
    return f"SUBJECT::{subject_norm}"


# ============================================================
# GRAPH BUILDING HELPERS
# ============================================================

def add_or_update_article_node(
    graph: nx.MultiDiGraph,
    article_id: str,
    article_title: str,
    content: str,
) -> str:
    node_id = make_article_node_id(article_id)

    if node_id not in graph:
        graph.add_node(
            node_id,
            node_type="ARTICLE",
            article_id=article_id,
            article_title=article_title,
            content=content,
            label=(
                f"Điều {article_id}"
                if article_id
                else "Không xác định điều luật"
            ),
        )
    else:
        node = graph.nodes[node_id]

        node["article_title"] = merge_unique_text(
            node.get("article_title", ""),
            article_title,
        )

        if not safe_string(node.get("content")):
            node["content"] = content

    return node_id


def add_or_update_rule_node(
    graph: nx.MultiDiGraph,
    rule_id: str,
    article_id: str,
    legal_subject: str,
    condition: str,
    effect: str,
    condition_norm: str,
    effect_norm: str,
    article_title: str,
    causal_type: str,
) -> str:
    node_id = make_rule_node_id(rule_id)

    if node_id in graph:
        raise ValueError(
            f"Trùng rule_id: {rule_id}. "
            "Mỗi index trong dữ liệu phải là duy nhất."
        )

    graph.add_node(
        node_id,
        node_type="RULE",
        rule_id=rule_id,
        article_id=article_id,
        legal_subject=legal_subject,
        condition=condition,
        effect=effect,
        condition_norm=condition_norm,
        effect_norm=effect_norm,
        article_title=article_title,
        causal_type=causal_type,
        label=f"Rule {rule_id}",
    )

    return node_id


def add_or_update_event_node(
    graph: nx.MultiDiGraph,
    event_norm: str,
    event_text: str,
    role: str,
    rule_id: str,
    article_id: str,
) -> str:
    """
    Tạo hoặc cập nhật một EVENT node.

    Điểm quan trọng:
    - condition_norm và effect_norm dùng chung node ID.
    - Một event có thể vừa là condition, vừa là effect.
    """

    if role not in {"CONDITION", "EFFECT"}:
        raise ValueError(
            f"Event role không hợp lệ: {role}"
        )

    node_id = make_event_node_id(event_norm)

    if node_id not in graph:
        graph.add_node(
            node_id,
            node_type="EVENT",
            event_norm=event_norm,
            label=event_norm,
            texts=event_text,
            condition_texts=(
                event_text
                if role == "CONDITION"
                else ""
            ),
            effect_texts=(
                event_text
                if role == "EFFECT"
                else ""
            ),
            is_condition=(
                role == "CONDITION"
            ),
            is_effect=(
                role == "EFFECT"
            ),
            condition_count=(
                1 if role == "CONDITION" else 0
            ),
            effect_count=(
                1 if role == "EFFECT" else 0
            ),
            rule_ids=rule_id,
            article_ids=article_id,
        )

        return node_id

    node = graph.nodes[node_id]

    node["texts"] = merge_unique_text(
        node.get("texts", ""),
        event_text,
    )

    node["rule_ids"] = increment_csv_attribute(
        node.get("rule_ids", ""),
        rule_id,
    )

    node["article_ids"] = increment_csv_attribute(
        node.get("article_ids", ""),
        article_id,
    )

    if role == "CONDITION":
        node["is_condition"] = True
        node["condition_count"] = (
            int(node.get("condition_count", 0)) + 1
        )

        node["condition_texts"] = merge_unique_text(
            node.get("condition_texts", ""),
            event_text,
        )

    if role == "EFFECT":
        node["is_effect"] = True
        node["effect_count"] = (
            int(node.get("effect_count", 0)) + 1
        )

        node["effect_texts"] = merge_unique_text(
            node.get("effect_texts", ""),
            event_text,
        )

    return node_id


def add_or_update_subject_node(
    graph: nx.MultiDiGraph,
    legal_subject: str,
    subject_norm: str,
    rule_id: str,
    article_id: str,
) -> str:
    node_id = make_subject_node_id(subject_norm)

    if node_id not in graph:
        graph.add_node(
            node_id,
            node_type="SUBJECT",
            subject_norm=subject_norm,
            label=legal_subject,
            texts=legal_subject,
            rule_ids=rule_id,
            article_ids=article_id,
        )

        return node_id

    node = graph.nodes[node_id]

    node["texts"] = merge_unique_text(
        node.get("texts", ""),
        legal_subject,
    )

    node["rule_ids"] = increment_csv_attribute(
        node.get("rule_ids", ""),
        rule_id,
    )

    node["article_ids"] = increment_csv_attribute(
        node.get("article_ids", ""),
        article_id,
    )

    return node_id


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_legal_causal_graph(
    dataframe: pd.DataFrame,
) -> tuple[nx.MultiDiGraph, dict[str, int]]:
    """
    Xây heterogeneous legal causal knowledge graph.

    Node types:
        ARTICLE
        RULE
        EVENT
        SUBJECT

    Edge types:
        HAS_RULE
        HAS_SUBJECT
        HAS_CONDITION
        HAS_EFFECT
        CAUSES

    Causal path thực tế nằm trên các EVENT node:

        EVENT(condition_norm)
            --CAUSES-->
        EVENT(effect_norm)

    Nếu effect_norm của rule A bằng condition_norm của rule B,
    cả hai tự động dùng chung EVENT node.
    """

    graph = nx.MultiDiGraph()

    graph.graph["name"] = (
        "Vietnamese Legal Causal Knowledge Graph"
    )
    graph.graph["version"] = "2.0"
    graph.graph["event_node_strategy"] = (
        "condition_norm_and_effect_norm_share_event_nodes"
    )

    skipped_missing_norm = 0
    skipped_self_loop = 0
    duplicate_rule_counts: defaultdict[
        str,
        int
    ] = defaultdict(int)

    for row_position, row in dataframe.iterrows():
        condition_norm = normalize_identifier(
            row.get("condition_norm")
        )

        effect_norm = normalize_identifier(
            row.get("effect_norm")
        )

        if not condition_norm or not effect_norm:
            skipped_missing_norm += 1
            continue

        base_rule_id = normalize_rule_id(
            row.get("index"),
            fallback=int(row_position) + 1,
        )

        duplicate_rule_counts[base_rule_id] += 1

        if duplicate_rule_counts[base_rule_id] == 1:
            rule_id = base_rule_id
        else:
            rule_id = (
                f"{base_rule_id}_"
                f"{duplicate_rule_counts[base_rule_id]}"
            )

        article_id = normalize_article_id(
            row.get("article_id")
        )

        legal_subject = safe_string(
            row.get("legal_subject")
        )

        subject_norm = normalize_identifier(
            legal_subject
        )

        condition = safe_string(
            row.get("condition")
        )

        effect = safe_string(
            row.get("effect")
        )

        article_title = safe_string(
            row.get("article_title")
        )

        content = safe_string(
            row.get("content")
        )

        causal_type = safe_string(
            row.get("causal_type")
        )

        article_node = add_or_update_article_node(
            graph=graph,
            article_id=article_id,
            article_title=article_title,
            content=content,
        )

        rule_node = add_or_update_rule_node(
            graph=graph,
            rule_id=rule_id,
            article_id=article_id,
            legal_subject=legal_subject,
            condition=condition,
            effect=effect,
            condition_norm=condition_norm,
            effect_norm=effect_norm,
            article_title=article_title,
            causal_type=causal_type,
        )

        condition_node = add_or_update_event_node(
            graph=graph,
            event_norm=condition_norm,
            event_text=condition,
            role="CONDITION",
            rule_id=rule_id,
            article_id=article_id,
        )

        effect_node = add_or_update_event_node(
            graph=graph,
            event_norm=effect_norm,
            event_text=effect,
            role="EFFECT",
            rule_id=rule_id,
            article_id=article_id,
        )

        subject_node = None

        if subject_norm:
            subject_node = add_or_update_subject_node(
                graph=graph,
                legal_subject=legal_subject,
                subject_norm=subject_norm,
                rule_id=rule_id,
                article_id=article_id,
            )

        # ARTICLE -> RULE
        graph.add_edge(
            article_node,
            rule_node,
            key=f"HAS_RULE::{rule_id}",
            relation="HAS_RULE",
            rule_id=rule_id,
            article_id=article_id,
        )

        # RULE -> SUBJECT
        if subject_node is not None:
            graph.add_edge(
                rule_node,
                subject_node,
                key=f"HAS_SUBJECT::{rule_id}",
                relation="HAS_SUBJECT",
                rule_id=rule_id,
                article_id=article_id,
            )

        # RULE -> CONDITION EVENT
        graph.add_edge(
            rule_node,
            condition_node,
            key=f"HAS_CONDITION::{rule_id}",
            relation="HAS_CONDITION",
            rule_id=rule_id,
            article_id=article_id,
        )

        # RULE -> EFFECT EVENT
        graph.add_edge(
            rule_node,
            effect_node,
            key=f"HAS_EFFECT::{rule_id}",
            relation="HAS_EFFECT",
            rule_id=rule_id,
            article_id=article_id,
        )

        # CONDITION EVENT -> EFFECT EVENT
        if condition_node == effect_node:
            skipped_self_loop += 1
        else:
            graph.add_edge(
                condition_node,
                effect_node,
                key=f"CAUSES::{rule_id}",
                relation="CAUSES",
                rule_id=rule_id,
                article_id=article_id,
                legal_subject=legal_subject,
                condition=condition,
                effect=effect,
                causal_type=causal_type,
                weight=1.0,
            )

    build_stats = {
        "input_rows": int(len(dataframe)),
        "valid_rule_rows": int(
            len(dataframe) - skipped_missing_norm
        ),
        "skipped_missing_norm": int(
            skipped_missing_norm
        ),
        "skipped_causal_self_loop": int(
            skipped_self_loop
        ),
    }

    return graph, build_stats


# ============================================================
# CAUSAL SUBGRAPH
# ============================================================

def extract_causal_event_graph(
    graph: nx.MultiDiGraph,
) -> nx.DiGraph:
    """
    Trích riêng graph EVENT --CAUSES--> EVENT.

    Dùng DiGraph thay vì MultiDiGraph để:
    - tìm path dễ hơn;
    - kiểm tra DAG;
    - đếm degree;
    - gộp nhiều rule cùng tạo một quan hệ nhân quả.
    """

    causal_graph = nx.DiGraph()

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") == "EVENT":
            causal_graph.add_node(
                node_id,
                **dict(node_data),
            )

    for source, target, edge_data in (
        graph.edges(data=True)
    ):
        if edge_data.get("relation") != "CAUSES":
            continue

        rule_id = safe_string(
            edge_data.get("rule_id")
        )

        article_id = safe_string(
            edge_data.get("article_id")
        )

        if causal_graph.has_edge(source, target):
            existing = causal_graph[source][target]

            existing["rule_ids"] = (
                increment_csv_attribute(
                    existing.get("rule_ids", ""),
                    rule_id,
                )
            )

            existing["article_ids"] = (
                increment_csv_attribute(
                    existing.get("article_ids", ""),
                    article_id,
                )
            )

            existing["support_count"] = (
                int(existing.get("support_count", 1))
                + 1
            )

        else:
            causal_graph.add_edge(
                source,
                target,
                relation="CAUSES",
                rule_ids=rule_id,
                article_ids=article_id,
                support_count=1,
                weight=1.0,
            )

    return causal_graph


# ============================================================
# TWO-HOP CHAIN EXTRACTION
# ============================================================

def extract_two_hop_causal_chains(
    causal_graph: nx.DiGraph,
) -> pd.DataFrame:
    """
    Tìm tất cả chain:

        Event A -> Event B -> Event C

    Event B chính là:
        effect_norm của rule trước
        và condition_norm của rule sau.
    """

    chains: list[dict[str, Any]] = []

    for event_a in causal_graph.nodes:
        for event_b in causal_graph.successors(event_a):
            for event_c in causal_graph.successors(event_b):
                if len({event_a, event_b, event_c}) < 3:
                    continue

                edge_ab = causal_graph[
                    event_a
                ][event_b]

                edge_bc = causal_graph[
                    event_b
                ][event_c]

                node_a = causal_graph.nodes[event_a]
                node_b = causal_graph.nodes[event_b]
                node_c = causal_graph.nodes[event_c]

                chains.append({
                    "event_a_id": event_a,
                    "event_a_norm": node_a.get(
                        "event_norm",
                        event_a,
                    ),
                    "event_a_texts": node_a.get(
                        "texts",
                        "",
                    ),
                    "event_b_id": event_b,
                    "event_b_norm": node_b.get(
                        "event_norm",
                        event_b,
                    ),
                    "event_b_texts": node_b.get(
                        "texts",
                        "",
                    ),
                    "event_c_id": event_c,
                    "event_c_norm": node_c.get(
                        "event_norm",
                        event_c,
                    ),
                    "event_c_texts": node_c.get(
                        "texts",
                        "",
                    ),
                    "rule_ids_a_to_b": edge_ab.get(
                        "rule_ids",
                        "",
                    ),
                    "article_ids_a_to_b": edge_ab.get(
                        "article_ids",
                        "",
                    ),
                    "rule_ids_b_to_c": edge_bc.get(
                        "rule_ids",
                        "",
                    ),
                    "article_ids_b_to_c": edge_bc.get(
                        "article_ids",
                        "",
                    ),
                    "support_a_to_b": edge_ab.get(
                        "support_count",
                        1,
                    ),
                    "support_b_to_c": edge_bc.get(
                        "support_count",
                        1,
                    ),
                })

    chain_df = pd.DataFrame(chains)

    if chain_df.empty:
        return chain_df

    chain_df = chain_df.drop_duplicates(
        subset=[
            "event_a_norm",
            "event_b_norm",
            "event_c_norm",
        ]
    )

    chain_df = chain_df.sort_values(
        by=[
            "support_a_to_b",
            "support_b_to_c",
            "event_a_norm",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return chain_df


# ============================================================
# GRAPH STATISTICS
# ============================================================

def calculate_graph_statistics(
    graph: nx.MultiDiGraph,
    causal_graph: nx.DiGraph,
    build_stats: dict[str, int],
    two_hop_chain_count: int,
) -> dict[str, Any]:
    node_type_counts = Counter(
        data.get("node_type", "UNKNOWN")
        for _, data in graph.nodes(data=True)
    )

    edge_type_counts = Counter(
        data.get("relation", "UNKNOWN")
        for _, _, data in graph.edges(data=True)
    )

    event_role_counts = {
        "condition_only": 0,
        "effect_only": 0,
        "both_condition_and_effect": 0,
        "neither": 0,
    }

    for _, data in graph.nodes(data=True):
        if data.get("node_type") != "EVENT":
            continue

        is_condition = bool(
            data.get("is_condition", False)
        )

        is_effect = bool(
            data.get("is_effect", False)
        )

        if is_condition and is_effect:
            event_role_counts[
                "both_condition_and_effect"
            ] += 1

        elif is_condition:
            event_role_counts[
                "condition_only"
            ] += 1

        elif is_effect:
            event_role_counts[
                "effect_only"
            ] += 1

        else:
            event_role_counts["neither"] += 1

    if causal_graph.number_of_nodes() > 0:
        is_dag = nx.is_directed_acyclic_graph(
            causal_graph
        )

        weak_components = (
            nx.number_weakly_connected_components(
                causal_graph
            )
        )
    else:
        is_dag = True
        weak_components = 0

    bridge_event_count = sum(
        1
        for _, data in causal_graph.nodes(data=True)
        if (
            bool(data.get("is_condition", False))
            and bool(data.get("is_effect", False))
        )
    )

    statistics = {
        **build_stats,
        "total_nodes": int(
            graph.number_of_nodes()
        ),
        "total_edges": int(
            graph.number_of_edges()
        ),
        "node_type_counts": dict(
            node_type_counts
        ),
        "edge_type_counts": dict(
            edge_type_counts
        ),
        "event_role_counts": event_role_counts,
        "causal_event_nodes": int(
            causal_graph.number_of_nodes()
        ),
        "causal_edges": int(
            causal_graph.number_of_edges()
        ),
        "bridge_event_nodes": int(
            bridge_event_count
        ),
        "two_hop_causal_chains": int(
            two_hop_chain_count
        ),
        "causal_graph_is_dag": bool(
            is_dag
        ),
        "causal_weak_components": int(
            weak_components
        ),
    }

    return statistics


# ============================================================
# GRAPHML COMPATIBILITY
# ============================================================

def sanitize_graphml_value(value: Any) -> Any:
    """
    GraphML chỉ hỗ trợ các kiểu dữ liệu nguyên thủy.

    Hàm này chuyển:
    - None -> ""
    - numpy scalar -> Python scalar
    - list/set/dict -> JSON string
    """

    if value is None:
        return ""

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return float(value)

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    if hasattr(value, "item"):
        try:
            return sanitize_graphml_value(
                value.item()
            )
        except (ValueError, TypeError):
            pass

    return str(value)


def sanitize_graph_for_export(
    graph: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    sanitized = nx.MultiDiGraph()

    for key, value in graph.graph.items():
        sanitized.graph[str(key)] = (
            sanitize_graphml_value(value)
        )

    for node_id, node_data in graph.nodes(data=True):
        sanitized.add_node(
            str(node_id),
            **{
                str(key): sanitize_graphml_value(value)
                for key, value in node_data.items()
            },
        )

    for source, target, edge_key, edge_data in (
        graph.edges(
            keys=True,
            data=True,
        )
    ):
        sanitized.add_edge(
            str(source),
            str(target),
            key=str(edge_key),
            **{
                str(key): sanitize_graphml_value(value)
                for key, value in edge_data.items()
            },
        )

    return sanitized


# ============================================================
# PRINTING
# ============================================================

def print_graph_summary(
    statistics: dict[str, Any],
    chain_df: pd.DataFrame,
    preview_count: int,
) -> None:
    print("\n" + "=" * 80)
    print("LEGAL CAUSAL KNOWLEDGE GRAPH")
    print("=" * 80)

    print("\nBuild:")
    print(
        f"- Input rows: "
        f"{statistics['input_rows']}"
    )
    print(
        f"- Valid rules: "
        f"{statistics['valid_rule_rows']}"
    )
    print(
        f"- Skipped missing norm: "
        f"{statistics['skipped_missing_norm']}"
    )

    print("\nGraph:")
    print(
        f"- Total nodes: "
        f"{statistics['total_nodes']}"
    )
    print(
        f"- Total edges: "
        f"{statistics['total_edges']}"
    )
    print(
        f"- Node types: "
        f"{statistics['node_type_counts']}"
    )
    print(
        f"- Edge types: "
        f"{statistics['edge_type_counts']}"
    )

    print("\nCausal event graph:")
    print(
        f"- Event nodes: "
        f"{statistics['causal_event_nodes']}"
    )
    print(
        f"- Causal edges: "
        f"{statistics['causal_edges']}"
    )
    print(
        f"- Bridge event nodes: "
        f"{statistics['bridge_event_nodes']}"
    )
    print(
        f"- Two-hop chains: "
        f"{statistics['two_hop_causal_chains']}"
    )
    print(
        f"- DAG: "
        f"{statistics['causal_graph_is_dag']}"
    )

    if chain_df.empty:
        print(
            "\nKhông tìm thấy chain "
            "Event A -> Event B -> Event C."
        )
        print(
            "Điều này nghĩa là chưa có effect_norm nào "
            "trùng condition_norm của rule khác."
        )
        return

    print(
        f"\n{min(preview_count, len(chain_df))} "
        "causal chains đầu tiên:"
    )

    for index, row in chain_df.head(
        preview_count
    ).iterrows():
        print("\n" + "-" * 80)
        print(f"Chain {index + 1}")

        print(
            f"{row['event_a_norm']}"
            f"  -- Rule {row['rule_ids_a_to_b']} -->  "
            f"{row['event_b_norm']}"
            f"  -- Rule {row['rule_ids_b_to_c']} -->  "
            f"{row['event_c_norm']}"
        )

        print(
            "Articles: "
            f"{row['article_ids_a_to_b']} "
            "-> "
            f"{row['article_ids_b_to_c']}"
        )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Xây dựng Vietnamese Legal Causal "
            "Knowledge Graph từ condition_norm "
            "và effect_norm."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="File JSON hoặc CSV đầu vào.",
    )

    parser.add_argument(
        "--graphml-output",
        type=str,
        default=DEFAULT_GRAPHML_PATH,
    )

    parser.add_argument(
        "--gexf-output",
        type=str,
        default=DEFAULT_GEXF_PATH,
    )

    parser.add_argument(
        "--stats-output",
        type=str,
        default=DEFAULT_STATS_PATH,
    )

    parser.add_argument(
        "--chains-output",
        type=str,
        default=DEFAULT_CHAINS_PATH,
    )

    parser.add_argument(
        "--preview-chains",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--skip-gexf",
        action="store_true",
        help="Không xuất file GEXF.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    dataframe = load_input_dataframe(
        args.input
    )

    graph, build_stats = (
        build_legal_causal_graph(
            dataframe
        )
    )

    causal_graph = extract_causal_event_graph(
        graph
    )

    chain_df = extract_two_hop_causal_chains(
        causal_graph
    )

    statistics = calculate_graph_statistics(
        graph=graph,
        causal_graph=causal_graph,
        build_stats=build_stats,
        two_hop_chain_count=len(chain_df),
    )

    graphml_path = Path(
        args.graphml_output
    )

    graphml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sanitized_graph = sanitize_graph_for_export(
        graph
    )

    nx.write_graphml(
        sanitized_graph,
        graphml_path,
        encoding="utf-8",
        prettyprint=True,
    )

    stats_path = Path(
        args.stats_output
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    chains_path = Path(
        args.chains_output
    )

    chains_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if chain_df.empty:
        pd.DataFrame(
            columns=[
                "event_a_norm",
                "event_b_norm",
                "event_c_norm",
                "rule_ids_a_to_b",
                "rule_ids_b_to_c",
                "article_ids_a_to_b",
                "article_ids_b_to_c",
            ]
        ).to_csv(
            chains_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        chain_df.to_csv(
            chains_path,
            index=False,
            encoding="utf-8-sig",
        )

    if not args.skip_gexf:
        gexf_path = Path(
            args.gexf_output
        )

        gexf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        nx.write_gexf(
            sanitized_graph,
            gexf_path,
            encoding="utf-8",
            prettyprint=True,
        )

    print_graph_summary(
        statistics=statistics,
        chain_df=chain_df,
        preview_count=args.preview_chains,
    )

    print("\nSaved:")
    print(f"- GraphML: {graphml_path}")
    print(f"- Statistics: {stats_path}")
    print(f"- Two-hop chains: {chains_path}")

    if not args.skip_gexf:
        print(f"- GEXF: {args.gexf_output}")


if __name__ == "__main__":
    main()