import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =====================================
# CONFIG
# =====================================

INPUT_FILE = "data/4_blhs_merged.json"
OUTPUT_DIR = "data/causal_memory"
MODEL_NAME = "BAAI/bge-m3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_json(INPUT_FILE)
print("Total rules:", len(df))

# =====================================
# BUILD CAUSAL MEMORY
# =====================================

memory = []
for _, row in df.iterrows():

    text = f"""
Legal Subject:
{row['legal_subject']}

Condition:
{row['condition']}

Effect:
{row['effect']}

Article:
{row['article_title']}
""".strip()

    memory.append({
        "id": row["index"],
        "article_id": row["article_id"],
        "article_title": row["article_title"],
        "legal_subject": row["legal_subject"],
        "condition": row["condition"],
        "effect": row["effect"],
        "condition_norm": row["condition_norm"],
        "effect_norm": row["effect_norm"],
        "content": row["content"],
        "embedding_text": text

    })

memory_df = pd.DataFrame(memory)

memory_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "causal_memory.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("Saved causal_memory.csv")

# =====================================
# LOAD EMBEDDING MODEL
# =====================================

print("\nLoading model...")
model = SentenceTransformer(MODEL_NAME)
print("Done")

# =====================================
# EMBEDDING
# =====================================

texts = memory_df["embedding_text"].tolist()

embeddings = model.encode(

    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True

)

print("Embedding shape:", embeddings.shape)

# =====================================
# SAVE NUMPY
# =====================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "embeddings.npy"
    ),
    embeddings

)

# =====================================
# SAVE PICKLE
# =====================================

memory_dict = {}

for i in range(len(memory_df)):
    memory_dict[i] = {
        "article_id":
            memory_df.iloc[i]["article_id"],
        "condition_norm":
            memory_df.iloc[i]["condition_norm"],
        "effect_norm":
            memory_df.iloc[i]["effect_norm"],
        "embedding":
            embeddings[i]
    }

with open(
    os.path.join(
        OUTPUT_DIR,
        "embeddings.pkl"
    ),
    "wb"

) as f:
    pickle.dump(memory_dict, f)

# =====================================
# BUILD FAISS
# =====================================

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(
    embeddings.astype("float32")
)

faiss.write_index(
    index,
    os.path.join(
        OUTPUT_DIR,
        "causal_memory.index"
    )

)

print("\n=====================")
print("Finished")
print("=====================")

print("Total vectors:", index.ntotal)