import networkx as nx

G = nx.read_graphml("data/legal_knowledge_graph.graphml")

MIN_LENGTH = 4      # ít nhất 4 node
MAX_DEPTH = 8       # tìm tối đa 8 bước

all_paths = []

for source in G.nodes():

    for target in G.nodes():

        if source == target:
            continue

        try:
            for path in nx.all_simple_paths(
                    G,
                    source,
                    target,
                    cutoff=MAX_DEPTH):

                if len(path) >= MIN_LENGTH:
                    all_paths.append(path)

        except nx.NetworkXNoPath:
            pass

print("Total paths:", len(all_paths))

# Sắp xếp theo độ dài
all_paths.sort(key=len, reverse=True)

# In 20 path dài nhất
for i, p in enumerate(all_paths[:20], 1):

    print("=" * 80)
    print(f"Path {i} (length={len(p)})")

    for node in p:
        print(node, "->", G.nodes[node].get("label"))