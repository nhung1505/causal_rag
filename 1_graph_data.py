import pandas as pd
import networkx as nx

# =====================================
# LOAD DATA
# =====================================

df = pd.read_json("data/4_blhs_merged.json")

# =====================================
# GRAPH
# =====================================

G = nx.MultiDiGraph()

# =====================================
# BUILD GRAPH
# =====================================

for _, row in df.iterrows():

    article_node = f"ARTICLE_{row.article_id}"

    rule_node = f"RULE_{row['index']}"

    subject_node = (
        "SUBJECT_"
        + row.legal_subject.upper()
        .replace(" ", "_")
        .replace(",", "")
        .replace("Đ", "D")
    )

    condition_node = row.condition_norm

    effect_node = row.effect_norm

    # ======================================================
    # ARTICLE NODE
    # ======================================================

    if not G.has_node(article_node):

        G.add_node(
            article_node,
            node_type="article",
            article_id=row.article_id,
            title=row.article_title,
            content=row.content
        )

    # ======================================================
    # RULE NODE
    # ======================================================

    G.add_node(
        rule_node,
        node_type="rule",
        rule_id=row["index"]
    )

    # ======================================================
    # SUBJECT
    # ======================================================

    if not G.has_node(subject_node):

        G.add_node(

            subject_node,

            node_type="subject",

            label=row.legal_subject

        )

    # ======================================================
    # CONDITION
    # ======================================================

    if not G.has_node(condition_node):

        G.add_node(

            condition_node,

            node_type="condition",

            label=row.condition

        )

    # ======================================================
    # EFFECT
    # ======================================================

    if not G.has_node(effect_node):

        G.add_node(

            effect_node,

            node_type="effect",

            label=row.effect

        )

    # ======================================================
    # RELATIONS
    # ======================================================

    G.add_edge(
        article_node,
        rule_node,
        relation="HAS_RULE"
    )

    G.add_edge(
        rule_node,
        subject_node,
        relation="HAS_SUBJECT"
    )

    G.add_edge(
        rule_node,
        condition_node,
        relation="HAS_CONDITION"
    )

    G.add_edge(
        rule_node,
        effect_node,
        relation="HAS_EFFECT"
    )

    # ======================================================
    # CAUSAL RELATION
    # ======================================================

    G.add_edge(

        condition_node,

        effect_node,

        relation="CAUSES",

        article=row.article_id,

        rule=row["index"]

    )

    # ======================================================
    # SUBJECT -> CONDITION
    # ======================================================

    G.add_edge(

        subject_node,

        condition_node,

        relation="CAN_TRIGGER"

    )

print("=" * 50)
print("Knowledge Graph Statistics")
print("=" * 50)

print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())

# =====================================
# SAVE
# =====================================

nx.write_graphml(
    G,
    "data/legal_knowledge_graph.graphml"
)

print("\nSaved to:")
print("data/legal_knowledge_graph.graphml")