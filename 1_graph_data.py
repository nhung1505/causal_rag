import re
import unicodedata
from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/4_blhs_merged.json"
OUTPUT_PATH = "data/legal_causal_knowledge_graph.graphml"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_string(value) -> str:
    """
    GraphML không xử lý tốt NaN hoặc các kiểu dữ liệu phức tạp.
    Chuyển toàn bộ giá trị về chuỗi an toàn.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_identifier(text: str) -> str:
    """
    Chuẩn hóa một chuỗi thành ID node dạng UPPER_SNAKE_CASE.
    Dùng chủ yếu cho subject chưa có cột subject_norm.
    """
    text = safe_string(text)

    text = text.replace("Đ", "D").replace("đ", "d")

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def add_node_if_absent(
    graph: nx.MultiDiGraph,
    node_id: str,
    **attributes
) -> None:
    """
    Chỉ tạo node nếu node chưa tồn tại.
    Nếu đã tồn tại, giữ label ban đầu và cập nhật các thuộc tính còn thiếu.
    """
    if not graph.has_node(node_id):
        graph.add_node(node_id, **attributes)
        return

    for key, value in attributes.items():
        if key not in graph.nodes[node_id]:
            graph.nodes[node_id][key] = value


def add_unique_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: str,
    **attributes
) -> None:
    """
    Tránh thêm trùng một cạnh cùng source, target và relation.
    """
    edge_data = graph.get_edge_data(source, target, default={})

    for _, data in edge_data.items():
        if data.get("relation") == relation:
            return

    graph.add_edge(
        source,
        target,
        relation=relation,
        **attributes
    )


# ============================================================
# LOAD DATA
# ============================================================

input_file = Path(INPUT_PATH)

if not input_file.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu: {INPUT_PATH}"
    )

df = pd.read_json(INPUT_PATH)

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

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Thiếu các cột bắt buộc: {sorted(missing_columns)}"
    )

print(f"Total rules: {len(df)}")


# ============================================================
# BUILD GRAPH
# ============================================================

G = nx.MultiDiGraph()

for row_number, row in df.iterrows():

    rule_index = safe_string(row["index"])
    article_number = safe_string(row["article_id"])

    subject_text = safe_string(row["legal_subject"])
    condition_text = safe_string(row["condition"])
    effect_text = safe_string(row["effect"])

    condition_norm = safe_string(row["condition_norm"])
    effect_norm = safe_string(row["effect_norm"])

    article_title = safe_string(row["article_title"])
    content = safe_string(row["content"])

    if not rule_index:
        rule_index = str(row_number)

    if not condition_norm or not effect_norm:
        print(
            f"Skip row {row_number}: "
            "condition_norm hoặc effect_norm bị thiếu"
        )
        continue

    # --------------------------------------------------------
    # NODE IDS
    # Có prefix để CONDITION và EFFECT không bao giờ bị gộp nhầm.
    # --------------------------------------------------------

    article_node = f"ARTICLE::{article_number}"
    rule_node = f"RULE::{rule_index}"

    subject_norm = normalize_identifier(subject_text)
    subject_node = f"SUBJECT::{subject_norm}"

    condition_node = f"CONDITION::{condition_norm}"
    effect_node = f"EFFECT::{effect_norm}"

    # --------------------------------------------------------
    # ARTICLE NODE
    # --------------------------------------------------------

    add_node_if_absent(
        G,
        article_node,
        node_type="article",
        article_id=article_number,
        label=article_title,
        title=article_title,
        content=content
    )

    # --------------------------------------------------------
    # RULE NODE
    # Mỗi dòng dữ liệu là một causal rule riêng.
    # --------------------------------------------------------

    add_node_if_absent(
        G,
        rule_node,
        node_type="rule",
        rule_id=rule_index,
        article_id=article_number,
        label=f"Rule {rule_index}",
        legal_subject=subject_text,
        condition=condition_text,
        effect=effect_text,
        condition_norm=condition_norm,
        effect_norm=effect_norm
    )

    # --------------------------------------------------------
    # SUBJECT NODE
    # --------------------------------------------------------

    add_node_if_absent(
        G,
        subject_node,
        node_type="subject",
        norm=subject_norm,
        label=subject_text
    )

    # --------------------------------------------------------
    # CONDITION NODE
    # Các condition giống nghĩa dùng chung condition_norm
    # nên tự động hội tụ về cùng node.
    # --------------------------------------------------------

    add_node_if_absent(
        G,
        condition_node,
        node_type="condition",
        norm=condition_norm,
        label=condition_text
    )

    # --------------------------------------------------------
    # EFFECT NODE
    # Các effect giống nghĩa dùng chung effect_norm
    # nên tự động hội tụ về cùng node.
    # --------------------------------------------------------

    add_node_if_absent(
        G,
        effect_node,
        node_type="effect",
        norm=effect_norm,
        label=effect_text
    )

    # --------------------------------------------------------
    # STRUCTURAL RELATIONS
    # --------------------------------------------------------

    add_unique_edge(
        G,
        article_node,
        rule_node,
        relation="HAS_RULE"
    )

    add_unique_edge(
        G,
        rule_node,
        subject_node,
        relation="HAS_SUBJECT"
    )

    add_unique_edge(
        G,
        rule_node,
        condition_node,
        relation="HAS_CONDITION"
    )

    add_unique_edge(
        G,
        rule_node,
        effect_node,
        relation="HAS_EFFECT"
    )

    # --------------------------------------------------------
    # CAUSAL RELATION
    # Đây là cạnh trung tâm của CausalRAG.
    #
    # Không dùng add_unique_edge tại đây vì cùng một cặp
    # condition-effect có thể xuất hiện ở nhiều article/rule.
    # MultiDiGraph cho phép giữ riêng từng bằng chứng pháp lý.
    # --------------------------------------------------------

    G.add_edge(
        condition_node,
        effect_node,
        relation="CAUSES",
        rule_id=rule_index,
        article_id=article_number,
        subject=subject_text,
        condition=condition_text,
        effect=effect_text
    )


# ============================================================
# GRAPH STATISTICS
# ============================================================

node_type_counts = {}

for _, attributes in G.nodes(data=True):
    node_type = attributes.get("node_type", "unknown")
    node_type_counts[node_type] = (
        node_type_counts.get(node_type, 0) + 1
    )

relation_counts = {}

for _, _, attributes in G.edges(data=True):
    relation = attributes.get("relation", "unknown")
    relation_counts[relation] = (
        relation_counts.get(relation, 0) + 1
    )

print("\n" + "=" * 60)
print("LEGAL CAUSAL KNOWLEDGE GRAPH")
print("=" * 60)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nNode types:")
for node_type, count in sorted(node_type_counts.items()):
    print(f"  {node_type:12s}: {count}")

print("\nRelations:")
for relation, count in sorted(relation_counts.items()):
    print(f"  {relation:16s}: {count}")


# ============================================================
# IMPORTANT CAUSAL NODES
# ============================================================

condition_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "condition"
]

effect_nodes = [
    node
    for node, data in G.nodes(data=True)
    if data.get("node_type") == "effect"
]

top_conditions = sorted(
    condition_nodes,
    key=lambda node: G.out_degree(node),
    reverse=True
)[:20]

top_effects = sorted(
    effect_nodes,
    key=lambda node: G.in_degree(node),
    reverse=True
)[:20]

print("\nTop condition hubs:")

for node in top_conditions:
    print(
        f"  {node}: "
        f"out_degree={G.out_degree(node)}, "
        f"label={G.nodes[node].get('label', '')}"
    )

print("\nTop effect hubs:")

for node in top_effects:
    print(
        f"  {node}: "
        f"in_degree={G.in_degree(node)}, "
        f"label={G.nodes[node].get('label', '')}"
    )


# ============================================================
# SAVE GRAPH
# ============================================================

output_file = Path(OUTPUT_PATH)
output_file.parent.mkdir(parents=True, exist_ok=True)

nx.write_graphml(G, OUTPUT_PATH)

print(f"\nSaved graph to: {OUTPUT_PATH}")