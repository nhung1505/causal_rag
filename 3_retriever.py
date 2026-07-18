import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ===============================
# CONFIG
# ===============================

MODEL_NAME = "BAAI/bge-m3"

CSV_PATH = "data/causal_memory/causal_memory.csv"

INDEX_PATH = "data/causal_memory/causal_memory.index"


TOP_K = 10


# ===============================
# LOAD MODEL
# ===============================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Done.")


# ===============================
# LOAD MEMORY
# ===============================

memory = pd.read_csv(CSV_PATH)

print(f"Memory size: {len(memory)}")


# ===============================
# LOAD FAISS
# ===============================

index = faiss.read_index(INDEX_PATH)

print(f"FAISS vectors: {index.ntotal}")


# ===============================
# RETRIEVAL FUNCTION
# ===============================

def retrieve(question, top_k=TOP_K):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    query_embedding = np.array(
        [query_embedding],
        dtype=np.float32
    )

    scores, ids = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx == -1:
            continue

        row = memory.iloc[idx]

        results.append({

            "score": float(score),

            "article": row["article_id"],

            "subject": row["legal_subject"],

            "condition": row["condition"],

            "effect": row["effect"],

            "condition_norm": row["condition_norm"],

            "effect_norm": row["effect_norm"],

            "title": row["article_title"]

        })

    return results


# ===============================
# TEST
# ===============================

if __name__ == "__main__":

    question = input("\nQuestion: ")

    results = retrieve(question)

    print("\n==============================")

    print("Top Retrieval Results")

    print("==============================\n")

    for i, r in enumerate(results, 1):

        print(f"[{i}] score={r['score']:.4f}")

        print("Article:", r["article"])

        print("Title:", r["title"])

        print("Subject:", r["subject"])

        print("Condition:", r["condition"])

        print("Effect:", r["effect"])

        print("Condition Norm:", r["condition_norm"])

        print("Effect Norm:", r["effect_norm"])

        print("-" * 80)