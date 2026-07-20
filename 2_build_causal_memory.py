from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/4_blhs_merged.json"
GRAPH_PATH = "data/legal_causal_knowledge_graph.graphml"

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


def normalize_identifier(text: Any) -> str:
    text = remove_vietnamese_accents(
        safe_string(text)
    ).upper()

    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_article_id(value: Any) -> str:
    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def normalize_rule_id(value: Any, fallback: int) -> str:
    """Chuẩn hóa rule ID giống file build graph."""
    text = safe_string(value)

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text or str(fallback)


def make_rule_node_id(rule_id: str) -> str:
    return f"RULE::{rule_id}"


def make_event_node_id(event_norm: str) -> str:
    return f"EVENT::{event_norm}"


def build_embedding_text(
    *,
    article_id: str,
    legal_subject: str,
    condition: str,
    effect: str,
    condition_norm: str,
    effect_norm: str,
    article_title: str,
) -> str:
    """Văn bản embedding của một causal rule.

    Memory vẫn được xây theo RULE. Hai trường *_event_id cho phép
    retriever chuyển từ kết quả semantic search sang graph EVENT mới.
    """

    return (
        f"Chủ thể pháp lý:\n{legal_subject}\n\n"
        f"Điều kiện pháp lý:\n{condition}\n\n"
        f"Hệ quả pháp lý:\n{effect}\n\n"
        f"Điều luật:\nĐiều {article_id}. {article_title}\n\n"
        f"Sự kiện điều kiện chuẩn hóa:\n{condition_norm}\n\n"
        f"Sự kiện hệ quả chuẩn hóa:\n{effect_norm}"
    )


# ============================================================
# INPUT + VALIDATION
# ============================================================

def load_dataframe(input_path: str) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {path}"
        )

    if path.suffix.lower() == ".json":
        return pd.read_json(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Chỉ hỗ trợ đầu vào JSON hoặc CSV.")


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
            "Thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )


# ============================================================
# BUILD RULE MEMORY
# ============================================================

def build_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo một memory record cho mỗi RULE hợp lệ.

    Quan trọng:
    - Không tạo memory riêng cho CONDITION/EFFECT cũ.
    - condition_event_id và effect_event_id trỏ thẳng tới EVENT node.
    - Cách xử lý rule_id trùng được giữ giống file build graph.
    """

    records: list[dict[str, Any]] = []
    duplicate_rule_counts: defaultdict[str, int] = defaultdict(int)
    skipped_rows = 0

    for row_position, row in df.iterrows():
        condition_norm = normalize_identifier(
            row.get("condition_norm")
        )
        effect_norm = normalize_identifier(
            row.get("effect_norm")
        )

        if not condition_norm or not effect_norm:
            skipped_rows += 1
            print(
                f"Skip row {row_position}: "
                "condition_norm/effect_norm bị thiếu"
            )
            continue

        base_rule_id = normalize_rule_id(
            row.get("index"),
            fallback=int(row_position) + 1,
        )

        duplicate_rule_counts[base_rule_id] += 1
        occurrence = duplicate_rule_counts[base_rule_id]

        rule_id = (
            base_rule_id
            if occurrence == 1
            else f"{base_rule_id}_{occurrence}"
        )

        article_id = normalize_article_id(
            row.get("article_id")
        )
        legal_subject = safe_string(
            row.get("legal_subject")
        )
        condition = safe_string(row.get("condition"))
        effect = safe_string(row.get("effect"))
        article_title = safe_string(
            row.get("article_title")
        )
        content = safe_string(row.get("content"))
        causal_type = safe_string(row.get("causal_type"))

        condition_event_id = make_event_node_id(
            condition_norm
        )
        effect_event_id = make_event_node_id(effect_norm)
        rule_node_id = make_rule_node_id(rule_id)

        embedding_text = build_embedding_text(
            article_id=article_id,
            legal_subject=legal_subject,
            condition=condition,
            effect=effect,
            condition_norm=condition_norm,
            effect_norm=effect_norm,
            article_title=article_title,
        )

        records.append({
            # FAISS position is the DataFrame row after reset_index.
            "memory_id": len(records),
            "memory_type": "RULE",
            "rule_id": rule_id,
            "rule_node_id": rule_node_id,
            "source_row_id": int(row_position),
            "article_id": article_id,
            "legal_subject": legal_subject,
            "subject_norm": normalize_identifier(
                legal_subject
            ),
            "condition": condition,
            "effect": effect,
            "condition_norm": condition_norm,
            "effect_norm": effect_norm,
            "condition_event_id": condition_event_id,
            "effect_event_id": effect_event_id,
            "article_title": article_title,
            "content": content,
            "causal_type": causal_type,
            "embedding_text": embedding_text,
        })

    memory_df = pd.DataFrame(records).reset_index(drop=True)

    if memory_df.empty:
        raise ValueError(
            "Không có rule hợp lệ để tạo causal memory."
        )

    # Đảm bảo memory_id luôn đúng bằng vị trí vector trong FAISS.
    memory_df["memory_id"] = np.arange(
        len(memory_df),
        dtype=np.int64,
    )

    if memory_df["rule_id"].duplicated().any():
        raise ValueError(
            "rule_id vẫn bị trùng sau khi chuẩn hóa."
        )

    print("Skipped rows:", skipped_rows)
    return memory_df


# ============================================================
# GRAPH CONSISTENCY CHECK
# ============================================================

def validate_memory_against_graph(
    memory_df: pd.DataFrame,
    graph_path: str,
) -> None:
    """Kiểm tra metadata memory có trỏ đúng node trong GraphML mới."""

    path = Path(graph_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy GraphML để kiểm tra: {path}"
        )

    print(f"Reading graph for validation: {path}")
    graph = nx.read_graphml(path)
    graph_nodes = set(graph.nodes)

    missing_rule_nodes = sorted(
        set(memory_df["rule_node_id"]) - graph_nodes
    )
    missing_condition_events = sorted(
        set(memory_df["condition_event_id"]) - graph_nodes
    )
    missing_effect_events = sorted(
        set(memory_df["effect_event_id"]) - graph_nodes
    )

    if missing_rule_nodes:
        raise ValueError(
            "Có rule_node_id không tồn tại trong GraphML. "
            f"Ví dụ: {missing_rule_nodes[:10]}"
        )

    if missing_condition_events:
        raise ValueError(
            "Có condition_event_id không tồn tại trong GraphML. "
            f"Ví dụ: {missing_condition_events[:10]}"
        )

    if missing_effect_events:
        raise ValueError(
            "Có effect_event_id không tồn tại trong GraphML. "
            f"Ví dụ: {missing_effect_events[:10]}"
        )

    invalid_node_types: list[str] = []

    for event_id in set(
        memory_df["condition_event_id"]
    ).union(memory_df["effect_event_id"]):
        if graph.nodes[event_id].get("node_type") != "EVENT":
            invalid_node_types.append(event_id)

    if invalid_node_types:
        raise ValueError(
            "Một số *_event_id không phải EVENT node: "
            f"{invalid_node_types[:10]}"
        )

    print("Graph consistency check: OK")
    print("- Rule nodes matched:", len(memory_df))
    print(
        "- Unique condition events:",
        memory_df["condition_event_id"].nunique(),
    )
    print(
        "- Unique effect events:",
        memory_df["effect_event_id"].nunique(),
    )


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
            "Embedding shape không hợp lệ: "
            f"{embeddings.shape}"
        )

    if len(embeddings) != len(memory_df):
        raise ValueError(
            "Số embeddings không khớp số memory records."
        )

    dimension = embeddings.shape[1]

    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", dimension)

    # Vector đã normalize + inner product = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)
    return index, embeddings


# ============================================================
# SAVE + VERIFY
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

    for path in (memory_path, index_path, embeddings_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    memory_df.to_csv(
        memory_path,
        index=False,
        encoding="utf-8-sig",
    )
    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)

    print("\nSaved:")
    print(f"- Memory CSV: {memory_path}")
    print(f"- FAISS index: {index_path}")
    print(f"- Embeddings: {embeddings_path}")


def verify_outputs(
    memory_output_path: str,
    index_output_path: str,
    embeddings_output_path: str,
) -> None:
    memory_df = pd.read_csv(
        memory_output_path,
        dtype={
            "rule_id": str,
            "article_id": str,
        },
    )
    index = faiss.read_index(index_output_path)
    embeddings = np.load(embeddings_output_path)

    print("\nVerification:")
    print("Memory rows:", len(memory_df))
    print("FAISS vectors:", index.ntotal)
    print("FAISS dimension:", index.d)
    print("Embeddings shape:", embeddings.shape)

    if len(memory_df) != index.ntotal:
        raise ValueError(
            "Số dòng causal_memory.csv không khớp "
            "số vector trong FAISS index."
        )

    if embeddings.shape != (index.ntotal, index.d):
        raise ValueError(
            "Shape file embeddings không khớp FAISS index."
        )

    expected_memory_ids = np.arange(len(memory_df))
    actual_memory_ids = memory_df["memory_id"].to_numpy()

    if not np.array_equal(
        expected_memory_ids,
        actual_memory_ids,
    ):
        raise ValueError(
            "memory_id không khớp vị trí vector trong FAISS."
        )

    print("Memory, embeddings và FAISS index khớp nhau.")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo rule-level causal memory và FAISS index, "
            "liên kết với EVENT nodes trong legal causal graph."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=GRAPH_PATH,
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
    parser.add_argument(
        "--skip-graph-validation",
        action="store_true",
        help="Bỏ qua kiểm tra ID memory với GraphML.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size phải lớn hơn 0.")

    print(f"Reading data: {args.input}")
    df = load_dataframe(args.input)
    validate_dataframe(df)
    print("Input rows:", len(df))

    memory_df = build_memory_dataframe(df)
    print("Valid memory rows:", len(memory_df))

    if not args.skip_graph_validation:
        validate_memory_against_graph(
            memory_df=memory_df,
            graph_path=args.graph,
        )

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
        embeddings_output_path=args.embeddings_output,
    )

    verify_outputs(
        memory_output_path=args.memory_output,
        index_output_path=args.index_output,
        embeddings_output_path=args.embeddings_output,
    )


if __name__ == "__main__":
    main()