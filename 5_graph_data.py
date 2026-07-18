import pandas as pd
import networkx as nx

df = pd.read_json("data/4_blhs_merged.json")

G = nx.DiGraph()

for _, row in df.iterrows():

    G.add_node(
        row["condition_norm"],
        label=row["condition"]
    )

    G.add_node(
        row["effect_norm"],
        label=row["effect"]
    )

    G.add_edge(
        row["condition_norm"],
        row["effect_norm"],
        article_id=row["article_id"],
        legal_subject=row["legal_subject"],
        condition=row["condition"],
        effect=row["effect"]
    )

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

nx.write_graphml(
    G,
    "data/legal_causal_graph.graphml"
)

print("Saved to data/legal_causal_graph.graphml")