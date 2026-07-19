

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/4_blhs_merged.json"

MEMORY_OUTPUT_PATH = "data/causal_memory.csv"
INDEX_OUTPUT_PATH = "data/causal_memory.index"
EMBEDDINGS_OUTPUT_PATH = "data/causal_memory_embeddings.npy"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 16


# ============================================================
# HELPERS
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


def remove_vietnamese_accents(text: str) -> str:
    text = safe_string(text)

    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def normalize_identifier(text: str) -> str:
    text = remove_vietnamese_accents(text).upper()

    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def build_embedding_text(row: pd.Series) -> str:
    """
    Văn bản dùng để sinh embedding cho mỗi causal rule.

    Nên giữ cùng format với truy vấn/rule representation
    trong retriever để kết quả ổn định.
    """

    article_id = normalize_article_id(row["article_id"])

    legal_subject = safe_string(row["legal_subject"])
    condition = safe_string(row["condition"])
    effect = safe_string(row["effect"])

    condition_norm = normalize_identifier(
        safe_string(row["condition_norm"])
    )

    effect_norm = normalize_identifier(
        safe_string(row["effect_norm"])
    )

    article_title = safe_string(row["article_title"])

    return (
        f"Legal Subject:\n"
        f"{legal_subject}\n\n"

        f"Condition:\n"
        f"{condition}\n\n"

        f"Effect:\n"
        f"{effect}\n\n"

        f"Article:\n"
        f"Điều {article_id}. {article_title}\n\n"

        f"Normalized condition:\n"
        f"{condition_norm}\n\n"

        f"Normalized effect:\n"
        f"{effect_norm}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
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
            f"Thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# BUILD MEMORY DATAFRAME
# ============================================================

def build_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for row_position, row in df.iterrows():

        rule_id = safe_string(row["index"])

        if not rule_id:
            rule_id = str(row_position + 1)

        condition_norm = normalize_identifier(
            safe_string(row["condition_norm"])
        )

        effect_norm = normalize_identifier(
            safe_string(row["effect_norm"])
        )

        if not condition_norm or not effect_norm:
            print(
                f"Skip row {row_position}: "
                f"condition_norm/effect_norm bị thiếu"
            )
            continue

        record = {
            "rule_id": rule_id,
            "row_id": int(row_position),

            "article_id": normalize_article_id(
                row["article_id"]
            ),

            "legal_subject": safe_string(
                row["legal_subject"]
            ),

            "subject_norm": normalize_identifier(
                row["legal_subject"]
            ),

            "condition": safe_string(
                row["condition"]
            ),

            "effect": safe_string(
                row["effect"]
            ),

            "condition_norm": condition_norm,
            "effect_norm": effect_norm,

            "article_title": safe_string(
                row["article_title"]
            ),

            "content": safe_string(
                row["content"]
            ),

            "embedding_text": build_embedding_text(row),
        }

        records.append(record)

    memory_df = pd.DataFrame(records)

    if memory_df.empty:
        raise ValueError(
            "Không có rule hợp lệ để tạo causal memory."
        )

    if memory_df["rule_id"].duplicated().any():
        duplicated_ids = (
            memory_df.loc[
                memory_df["rule_id"].duplicated(
                    keep=False
                ),
                "rule_id"
            ]
            .astype(str)
            .tolist()
        )

        print(
            "Warning: phát hiện rule_id trùng:",
            duplicated_ids[:20]
        )

    return memory_df


# ============================================================
# EMBEDDING + FAISS
# ============================================================

def build_faiss_index(
    memory_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> tuple[faiss.Index, np.ndarray]:

    print(f"Loading model: {model_name}")

    model = SentenceTransformer(model_name)

    texts = memory_df["embedding_text"].tolist()

    print(f"Encoding {len(texts)} rules...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding shape không hợp lệ: "
            f"{embeddings.shape}"
        )

    dimension = embeddings.shape[1]

    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", dimension)

    # IndexFlatIP + normalized embeddings
    # tương đương cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)

    return index, embeddings


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    memory_df: pd.DataFrame,
    index: faiss.Index,
    embeddings: np.ndarray,
    memory_output_path: str,
    index_output_path: str,
    embeddings_output_path: str,
) -> None:

    memory_path = Path(memory_output_path)
    index_path = Path(index_output_path)
    embeddings_path = Path(embeddings_output_path)

    memory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_df.to_csv(
        memory_path,
        index=False,
        encoding="utf-8-sig",
    )

    faiss.write_index(
        index,
        str(index_path),
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    print("\nSaved:")
    print(f"- Memory CSV: {memory_path}")
    print(f"- FAISS index: {index_path}")
    print(f"- Embeddings: {embeddings_path}")


# ============================================================
# VERIFY
# ============================================================

def verify_outputs(
    memory_output_path: str,
    index_output_path: str,
) -> None:

    memory_df = pd.read_csv(memory_output_path)

    index = faiss.read_index(
        index_output_path
    )

    print("\nVerification:")
    print("Memory rows:", len(memory_df))
    print("FAISS vectors:", index.ntotal)
    print("FAISS dimension:", index.d)

    if len(memory_df) != index.ntotal:
        raise ValueError(
            "Số dòng causal_memory.csv không khớp "
            "số vector trong FAISS index."
        )

    print("Memory và FAISS index khớp nhau.")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Tạo causal memory và FAISS index "
            "cho dữ liệu luật hình sự Việt Nam."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
    )

    parser.add_argument(
        "--memory-output",
        type=str,
        default=MEMORY_OUTPUT_PATH,
    )

    parser.add_argument(
        "--index-output",
        type=str,
        default=INDEX_OUTPUT_PATH,
    )

    parser.add_argument(
        "--embeddings-output",
        type=str,
        default=EMBEDDINGS_OUTPUT_PATH,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: "
            f"{input_path}"
        )

    print(f"Reading data: {input_path}")

    df = pd.read_json(input_path)

    validate_dataframe(df)

    print("Input rows:", len(df))

    memory_df = build_memory_dataframe(df)

    print("Valid memory rows:", len(memory_df))

    index, embeddings = build_faiss_index(
        memory_df=memory_df,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    save_outputs(
        memory_df=memory_df,
        index=index,
        embeddings=embeddings,
        memory_output_path=args.memory_output,
        index_output_path=args.index_output,
        embeddings_output_path=(
            args.embeddings_output
        ),
    )

    verify_outputs(
        memory_output_path=args.memory_output,
        index_output_path=args.index_output,
    )


if __name__ == "__main__":
    main()
