from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_GRAPH_PATH = (
    "data/legal_causal_knowledge_graph.graphml"
)

DEFAULT_MIN_NODES = 4
DEFAULT_MAX_PATHS = 100
DEFAULT_MAX_DEPTH = 20


def safe_string(value: Any) -> str:
    """
    Chuyển giá trị thành chuỗi an toàn.
    """

    if value is None:
        return ""

    return str(value).strip()


def load_causal_graph(
    graph_path: str,
) -> tuple[nx.MultiDiGraph, nx.DiGraph]:
    """
    Đọc graph GraphML và trích riêng causal graph:

        EVENT --CAUSES--> EVENT

    Trả về:
        full_graph: toàn bộ graph gốc
        causal_graph: graph chỉ gồm EVENT và cạnh CAUSES
    """

    path = Path(graph_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file graph: {path}"
        )

    full_graph = nx.read_graphml(path)

    causal_graph = nx.DiGraph()

    for node_id, node_data in full_graph.nodes(
        data=True
    ):
        if node_data.get("node_type") != "EVENT":
            continue

        causal_graph.add_node(
            node_id,
            **dict(node_data),
        )

    for source, target, edge_data in (
        full_graph.edges(data=True)
    ):
        if edge_data.get("relation") != "CAUSES":
            continue

        if source not in causal_graph:
            causal_graph.add_node(
                source,
                **dict(full_graph.nodes[source]),
            )

        if target not in causal_graph:
            causal_graph.add_node(
                target,
                **dict(full_graph.nodes[target]),
            )

        rule_id = safe_string(
            edge_data.get("rule_id")
        )

        article_id = safe_string(
            edge_data.get("article_id")
        )

        if causal_graph.has_edge(source, target):
            existing = causal_graph[source][target]

            existing_rule_ids = {
                item.strip()
                for item in safe_string(
                    existing.get("rule_ids")
                ).split(",")
                if item.strip()
            }

            existing_article_ids = {
                item.strip()
                for item in safe_string(
                    existing.get("article_ids")
                ).split(",")
                if item.strip()
            }

            if rule_id:
                existing_rule_ids.add(rule_id)

            if article_id:
                existing_article_ids.add(
                    article_id
                )

            existing["rule_ids"] = ",".join(
                sorted(existing_rule_ids)
            )

            existing["article_ids"] = ",".join(
                sorted(existing_article_ids)
            )

            existing["support_count"] = (
                int(
                    existing.get(
                        "support_count",
                        1,
                    )
                )
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
            )

    return full_graph, causal_graph


def get_event_norm(
    graph: nx.DiGraph,
    node_id: str,
) -> str:
    """
    Lấy tên chuẩn hóa của EVENT node.
    """

    node_data = graph.nodes[node_id]

    return safe_string(
        node_data.get(
            "event_norm",
            node_id.replace("EVENT::", ""),
        )
    )


def get_event_text(
    graph: nx.DiGraph,
    node_id: str,
) -> str:
    """
    Lấy mô tả tự nhiên của EVENT node.
    """

    node_data = graph.nodes[node_id]

    condition_texts = safe_string(
        node_data.get("condition_texts")
    )

    effect_texts = safe_string(
        node_data.get("effect_texts")
    )

    texts = safe_string(
        node_data.get("texts")
    )

    if condition_texts:
        return condition_texts

    if effect_texts:
        return effect_texts

    return texts


def find_root_nodes(
    causal_graph: nx.DiGraph,
) -> list[str]:
    """
    Root node là node không có cạnh CAUSES đi vào.
    """

    roots = [
        node_id
        for node_id in causal_graph.nodes
        if causal_graph.in_degree(node_id) == 0
        and causal_graph.out_degree(node_id) > 0
    ]

    return sorted(
        roots,
        key=lambda node_id: get_event_norm(
            causal_graph,
            node_id,
        ),
    )


def find_long_causal_paths(
    causal_graph: nx.DiGraph,
    min_nodes: int = 4,
    max_depth: int = 20,
) -> list[list[str]]:
    """
    Tìm các causal path từ root đến leaf.

    Chỉ giữ path có ít nhất min_nodes node.

    max_depth giúp tránh path quá dài hoặc graph có lỗi.
    """

    roots = find_root_nodes(causal_graph)

    all_paths: list[list[str]] = []

    def dfs(
        current_node: str,
        current_path: list[str],
        visited_in_path: set[str],
    ) -> None:
        """
        DFS có kiểm soát chu trình.
        """

        if len(current_path) >= max_depth:
            if len(current_path) >= min_nodes:
                all_paths.append(
                    current_path.copy()
                )
            return

        children = list(
            causal_graph.successors(
                current_node
            )
        )

        valid_children = [
            child
            for child in children
            if child not in visited_in_path
        ]

        if not valid_children:
            if len(current_path) >= min_nodes:
                all_paths.append(
                    current_path.copy()
                )
            return

        for child in valid_children:
            current_path.append(child)
            visited_in_path.add(child)

            dfs(
                current_node=child,
                current_path=current_path,
                visited_in_path=visited_in_path,
            )

            visited_in_path.remove(child)
            current_path.pop()

    for root in roots:
        dfs(
            current_node=root,
            current_path=[root],
            visited_in_path={root},
        )

    return remove_duplicate_paths(
        all_paths
    )


def remove_duplicate_paths(
    paths: list[list[str]],
) -> list[list[str]]:
    """
    Loại các path trùng hoàn toàn.
    """

    unique_paths: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for path in paths:
        path_key = tuple(path)

        if path_key in seen:
            continue

        seen.add(path_key)
        unique_paths.append(path)

    unique_paths.sort(
        key=lambda path: (
            -len(path),
            tuple(path),
        )
    )

    return unique_paths


def get_edge_description(
    causal_graph: nx.DiGraph,
    source: str,
    target: str,
) -> str:
    """
    Lấy rule_id và article_id của cạnh CAUSES.
    """

    edge_data = causal_graph[
        source
    ][target]

    rule_ids = safe_string(
        edge_data.get("rule_ids")
    )

    article_ids = safe_string(
        edge_data.get("article_ids")
    )

    parts = []

    if rule_ids:
        parts.append(
            f"Rule: {rule_ids}"
        )

    if article_ids:
        parts.append(
            f"Điều: {article_ids}"
        )

    if not parts:
        return "CAUSES"

    return " | ".join(parts)


def print_path(
    causal_graph: nx.DiGraph,
    path: list[str],
    path_number: int,
    show_text: bool = True,
) -> None:
    """
    In một causal path theo định dạng dễ đọc.
    """

    hop_count = len(path) - 1

    print("\n" + "=" * 100)
    print(
        f"CHAIN {path_number} "
        f"| Nodes: {len(path)} "
        f"| Hops: {hop_count}"
    )
    print("=" * 100)

    for index, node_id in enumerate(path):
        event_norm = get_event_norm(
            causal_graph,
            node_id,
        )

        print(event_norm)

        if show_text:
            event_text = get_event_text(
                causal_graph,
                node_id,
            )

            if event_text:
                print(
                    f"  Nội dung: {event_text}"
                )

        if index < len(path) - 1:
            next_node = path[index + 1]

            edge_description = (
                get_edge_description(
                    causal_graph,
                    node_id,
                    next_node,
                )
            )

            print("    │")
            print(
                f"    │ {edge_description}"
            )
            print("    ▼")

    normalized_path = " → ".join(
        get_event_norm(
            causal_graph,
            node_id,
        )
        for node_id in path
    )

    print("\nChuỗi chuẩn hóa:")
    print(normalized_path)


def print_statistics(
    causal_graph: nx.DiGraph,
    paths: list[list[str]],
    min_nodes: int,
) -> None:
    """
    In thống kê tổng quan.
    """

    roots = find_root_nodes(
        causal_graph
    )

    leaf_nodes = [
        node_id
        for node_id in causal_graph.nodes
        if causal_graph.out_degree(node_id) == 0
        and causal_graph.in_degree(node_id) > 0
    ]

    bridge_nodes = [
        node_id
        for node_id, node_data
        in causal_graph.nodes(data=True)
        if (
            bool(
                node_data.get(
                    "is_condition",
                    False,
                )
            )
            and bool(
                node_data.get(
                    "is_effect",
                    False,
                )
            )
        )
    ]

    print("=" * 100)
    print("THỐNG KÊ CAUSAL GRAPH")
    print("=" * 100)

    print(
        f"Event nodes: "
        f"{causal_graph.number_of_nodes()}"
    )

    print(
        f"Causal edges: "
        f"{causal_graph.number_of_edges()}"
    )

    print(
        f"Root nodes: {len(roots)}"
    )

    print(
        f"Leaf nodes: {len(leaf_nodes)}"
    )

    print(
        f"Bridge nodes: "
        f"{len(bridge_nodes)}"
    )

    print(
        f"Chains có ít nhất "
        f"{min_nodes} node: {len(paths)}"
    )

    if paths:
        longest_path_length = max(
            len(path)
            for path in paths
        )

        longest_hops = (
            longest_path_length - 1
        )

        print(
            f"Chuỗi dài nhất: "
            f"{longest_path_length} node, "
            f"{longest_hops} hop"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "In các causal chain dài từ "
            "legal_causal_knowledge_graph.graphml"
        )
    )

    parser.add_argument(
        "--graph",
        type=str,
        default=DEFAULT_GRAPH_PATH,
        help="Đường dẫn file GraphML.",
    )

    parser.add_argument(
        "--min-nodes",
        type=int,
        default=DEFAULT_MIN_NODES,
        help=(
            "Số node tối thiểu của một chain. "
            "Mặc định: 4."
        ),
    )

    parser.add_argument(
        "--max-paths",
        type=int,
        default=DEFAULT_MAX_PATHS,
        help=(
            "Số chain tối đa được in. "
            "Dùng 0 để in tất cả."
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=(
            "Độ sâu tối đa khi duyệt graph."
        ),
    )

    parser.add_argument(
        "--hide-text",
        action="store_true",
        help=(
            "Chỉ in event_norm, "
            "không in nội dung tiếng Việt."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_nodes < 2:
        raise ValueError(
            "--min-nodes phải từ 2 trở lên."
        )

    if args.max_depth < args.min_nodes:
        raise ValueError(
            "--max-depth phải lớn hơn "
            "hoặc bằng --min-nodes."
        )

    _, causal_graph = load_causal_graph(
        args.graph
    )

    paths = find_long_causal_paths(
        causal_graph=causal_graph,
        min_nodes=args.min_nodes,
        max_depth=args.max_depth,
    )

    print_statistics(
        causal_graph=causal_graph,
        paths=paths,
        min_nodes=args.min_nodes,
    )

    if not paths:
        print(
            "\nKhông tìm thấy causal chain "
            f"có ít nhất {args.min_nodes} node."
        )
        return

    if args.max_paths == 0:
        selected_paths = paths
    else:
        selected_paths = paths[
            :args.max_paths
        ]

    print(
        f"\nĐang in "
        f"{len(selected_paths)}/{len(paths)} "
        "causal chains, sắp xếp dài nhất trước."
    )

    for path_number, path in enumerate(
        selected_paths,
        start=1,
    ):
        print_path(
            causal_graph=causal_graph,
            path=path,
            path_number=path_number,
            show_text=not args.hide_text,
        )


if __name__ == "__main__":
    main()


# python test.py \
#   --min-nodes 3 \
#   --max-paths 50
# ====================================================================================================
# THỐNG KÊ CAUSAL GRAPH
# ====================================================================================================
# Event nodes: 1914
# Causal edges: 1682
# Root nodes: 1487
# Leaf nodes: 416
# Bridge nodes: 12
# Chains có ít nhất 3 node: 35
# Chuỗi dài nhất: 4 node, 3 hop
