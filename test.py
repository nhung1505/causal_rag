import networkx as nx

G = nx.read_graphml(
    "data/legal_causal_knowledge_graph.graphml"
)

causal_graph = nx.DiGraph()

for source, target, data in G.edges(data=True):

    if data.get("relation") == "CAUSES":
        causal_graph.add_edge(source, target)

print("Causal nodes:", causal_graph.number_of_nodes())
print("Causal edges:", causal_graph.number_of_edges())
print(
    "Causal graph is DAG:",
    nx.is_directed_acyclic_graph(causal_graph)
)