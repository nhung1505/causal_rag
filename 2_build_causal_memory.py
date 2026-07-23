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

INPUT_PATH = "data/blhs_rules_final_all_normalized.json"
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


def make_event_node_id(event_id: str) -> str:
    return f"EVENT::{event_id}"


def build_rule_embedding_text(
    *,
    article_id: str,
    legal_subject: str,
    condition: str,
    effect: str,
    condition_event: str,
    condition_event_name: str,
    condition_event_modality: str,
    effect_event: str,
    effect_event_name: str,
    effect_event_modality: str,
    article_title: str,
    causal_type: str,
    rule_text: str,
) -> str:
    """Tạo văn bản embedding cho một rule pháp lý."""

    parts = [
        "Loại bộ nhớ: Quy tắc pháp lý",
        f"Chủ thể pháp lý: {legal_subject}",
        f"Điều kiện pháp lý: {condition}",
        f"Hệ quả pháp lý: {effect}",
        f"Sự kiện điều kiện: {condition_event_name}",
        f"Mã sự kiện điều kiện: {condition_event}",
        f"Tình thái điều kiện: {condition_event_modality}",
        f"Sự kiện hệ quả: {effect_event_name}",
        f"Mã sự kiện hệ quả: {effect_event}",
        f"Tình thái hệ quả: {effect_event_modality}",
        f"Loại quan hệ nhân quả: {causal_type}",
        f"Điều luật: Điều {article_id}. {article_title}",
    ]

    if rule_text:
        parts.append(f"Diễn đạt quy tắc: {rule_text}")

    return "\n".join(parts)


def build_event_embedding_text(
    *,
    event_id: str,
    event_name: str,
    event_role: str,
    event_texts: str,
    condition_texts: str,
    effect_texts: str,
    condition_count: int,
    effect_count: int,
    article_ids: str,
) -> str:
    """Tạo văn bản embedding cho một EVENT node trong causal graph."""

    role_label = {
        "CONDITION": "Sự kiện điều kiện",
        "EFFECT": "Sự kiện hệ quả",
        "BRIDGE": "Sự kiện cầu nối",
    }.get(event_role, "Sự kiện pháp lý")

    return (
        f"Loại bộ nhớ: {role_label}\n"
        f"Tên sự kiện: {event_name}\n"
        f"Mã sự kiện: {event_id}\n"
        f"Vai trò trong đồ thị: {event_role}\n"
        f"Các cách diễn đạt: {event_texts}\n"
        f"Ngữ cảnh điều kiện: {condition_texts}\n"
        f"Ngữ cảnh hệ quả: {effect_texts}\n"
        f"Số lần làm điều kiện: {condition_count}\n"
        f"Số lần làm hệ quả: {effect_count}\n"
        f"Các điều luật liên quan: {article_ids}"
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
        "condition_event",
        "effect_event",
        "article_title",
        "content",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Thiếu các cột bắt buộc: "
            f"{sorted(missing_columns)}"
        )

    optional_columns = {
        "condition_event_name": "",
        "effect_event_name": "",
        "condition_event_original": "",
        "effect_event_original": "",
        "condition_event_modality": "",
        "effect_event_modality": "",
        "event_normalization_version": "",
        "causal_type": "",
        "rule_text": "",
        "quality_status": "",
        "source_scope": "",
    }

    for column_name, default_value in optional_columns.items():
        if column_name not in df.columns:
            df[column_name] = default_value


# ============================================================
# BUILD RULE MEMORY
# ============================================================

def build_rule_memory_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo một memory record cho mỗi RULE hợp lệ."""

    records: list[dict[str, Any]] = []
    duplicate_rule_counts: defaultdict[str, int] = defaultdict(int)
    skipped_rows = 0

    for row_position, row in df.iterrows():
        condition_event = safe_string(
            row.get("condition_event")
        )
        effect_event = safe_string(
            row.get("effect_event")
        )

        if not condition_event or not effect_event:
            skipped_rows += 1
            print(
                f"Skip row {row_position}: "
                "condition_event/effect_event bị thiếu"
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
        rule_text = safe_string(row.get("rule_text"))

        condition_event_name = (
            safe_string(row.get("condition_event_name"))
            or condition_event
        )
        effect_event_name = (
            safe_string(row.get("effect_event_name"))
            or effect_event
        )

        condition_event_original = (
            safe_string(row.get("condition_event_original"))
            or condition_event
        )
        effect_event_original = (
            safe_string(row.get("effect_event_original"))
            or effect_event
        )

        condition_event_modality = safe_string(
            row.get("condition_event_modality")
        )
        effect_event_modality = safe_string(
            row.get("effect_event_modality")
        )

        condition_event_id = make_event_node_id(
            condition_event
        )
        effect_event_id = make_event_node_id(
            effect_event
        )
        rule_node_id = make_rule_node_id(rule_id)

        embedding_text = build_rule_embedding_text(
            article_id=article_id,
            legal_subject=legal_subject,
            condition=condition,
            effect=effect,
            condition_event=condition_event,
            condition_event_name=condition_event_name,
            condition_event_modality=condition_event_modality,
            effect_event=effect_event,
            effect_event_name=effect_event_name,
            effect_event_modality=effect_event_modality,
            article_title=article_title,
            causal_type=causal_type,
            rule_text=rule_text,
        )

        records.append({
            "memory_id": -1,
            "memory_type": "RULE",
            "graph_node_id": rule_node_id,
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
            "condition_event": condition_event,
            "effect_event": effect_event,
            "condition_event_name": condition_event_name,
            "effect_event_name": effect_event_name,
            "condition_event_original": condition_event_original,
            "effect_event_original": effect_event_original,
            "condition_event_modality": condition_event_modality,
            "effect_event_modality": effect_event_modality,
            "event_normalization_version": safe_string(
                row.get("event_normalization_version")
            ),
            "condition_event_id": condition_event_id,
            "effect_event_id": effect_event_id,
            "article_title": article_title,
            "content": content,
            "causal_type": causal_type,
            "rule_text": rule_text,
            "quality_status": safe_string(
                row.get("quality_status")
            ),
            "source_scope": safe_string(
                row.get("source_scope")
            ),
            "event_role": "",
            "event_id": "",
            "event_name": "",
            "is_bridge_event": False,
            "condition_count": 0,
            "effect_count": 0,
            "rule_ids": rule_id,
            "article_ids": article_id,
            "embedding_text": embedding_text,
        })

    memory_df = pd.DataFrame(records)

    if memory_df.empty:
        raise ValueError(
            "Không có rule hợp lệ để tạo rule memory."
        )

    if memory_df["rule_id"].duplicated().any():
        raise ValueError(
            "rule_id vẫn bị trùng sau khi chuẩn hóa."
        )

    print("Skipped rule rows:", skipped_rows)
    return memory_df


# ============================================================
# BUILD EVENT MEMORY
# ============================================================

def load_graph(graph_path: str) -> nx.Graph:
    path = Path(graph_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy GraphML: {path}"
        )

    print(f"Reading graph: {path}")
    return nx.read_graphml(path)


def to_int(value: Any, default: int = 0) -> int:
    text = safe_string(value)
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return safe_string(value).lower() in {
        "true", "1", "yes", "y"
    }


def build_event_memory_dataframe(
    graph: nx.Graph,
) -> pd.DataFrame:
    """Tạo memory record cho từng EVENT node trong causal graph."""

    records: list[dict[str, Any]] = []

    for node_id, data in graph.nodes(data=True):
        if safe_string(data.get("node_type")) != "EVENT":
            continue

        event_id = (
            safe_string(data.get("event_id"))
            or safe_string(node_id).removeprefix("EVENT::")
        )
        event_name = (
            safe_string(data.get("event_name"))
            or safe_string(data.get("label"))
            or event_id
        )

        is_condition = to_bool(data.get("is_condition"))
        is_effect = to_bool(data.get("is_effect"))

        if is_condition and is_effect:
            event_role = "BRIDGE"
        elif is_condition:
            event_role = "CONDITION"
        elif is_effect:
            event_role = "EFFECT"
        else:
            event_role = "EVENT"

        condition_count = to_int(
            data.get("condition_count")
        )
        effect_count = to_int(
            data.get("effect_count")
        )

        texts = safe_string(data.get("texts"))
        condition_texts = safe_string(
            data.get("condition_texts")
        )
        effect_texts = safe_string(
            data.get("effect_texts")
        )
        rule_ids = safe_string(data.get("rule_ids"))
        article_ids = safe_string(
            data.get("article_ids")
        )

        embedding_text = build_event_embedding_text(
            event_id=event_id,
            event_name=event_name,
            event_role=event_role,
            event_texts=texts,
            condition_texts=condition_texts,
            effect_texts=effect_texts,
            condition_count=condition_count,
            effect_count=effect_count,
            article_ids=article_ids,
        )

        records.append({
            "memory_id": -1,
            "memory_type": "EVENT",
            "graph_node_id": safe_string(node_id),
            "rule_id": "",
            "rule_node_id": "",
            "source_row_id": -1,
            "article_id": "",
            "legal_subject": "",
            "subject_norm": "",
            "condition": "",
            "effect": "",
            "condition_event": "",
            "effect_event": "",
            "condition_event_name": "",
            "effect_event_name": "",
            "condition_event_original": "",
            "effect_event_original": "",
            "condition_event_modality": "",
            "effect_event_modality": "",
            "event_normalization_version": "",
            "condition_event_id": "",
            "effect_event_id": "",
            "article_title": "",
            "content": "",
            "causal_type": "",
            "rule_text": "",
            "quality_status": "",
            "source_scope": "",
            "event_role": event_role,
            "event_id": event_id,
            "event_name": event_name,
            "is_bridge_event": event_role == "BRIDGE",
            "condition_count": condition_count,
            "effect_count": effect_count,
            "rule_ids": rule_ids,
            "article_ids": article_ids,
            "embedding_text": embedding_text,
        })

    event_df = pd.DataFrame(records)

    if event_df.empty:
        raise ValueError(
            "Không tìm thấy EVENT node để tạo event memory."
        )

    if event_df["graph_node_id"].duplicated().any():
        raise ValueError(
            "graph_node_id của event memory bị trùng."
        )

    return event_df


# ============================================================
# COMBINE + GRAPH CONSISTENCY CHECK
# ============================================================

def combine_memory_dataframes(
    rule_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> pd.DataFrame:
    """Hợp nhất RULE memory và EVENT memory vào một FAISS index."""

    all_columns = sorted(
        set(rule_df.columns) | set(event_df.columns)
    )

    rule_df = rule_df.reindex(columns=all_columns)
    event_df = event_df.reindex(columns=all_columns)

    memory_df = pd.concat(
        [rule_df, event_df],
        ignore_index=True,
    )

    memory_df["memory_id"] = np.arange(
        len(memory_df),
        dtype=np.int64,
    )

    if memory_df["graph_node_id"].isna().any():
        raise ValueError(
            "Có graph_node_id bị thiếu trong causal memory."
        )

    return memory_df


def validate_memory_against_graph(
    memory_df: pd.DataFrame,
    graph: nx.Graph,
) -> None:
    """Kiểm tra mọi graph_node_id và event reference trong memory."""

    graph_nodes = set(graph.nodes)

    missing_graph_nodes = sorted(
        set(memory_df["graph_node_id"]) - graph_nodes
    )

    if missing_graph_nodes:
        raise ValueError(
            "Có graph_node_id không tồn tại trong GraphML. "
            f"Ví dụ: {missing_graph_nodes[:10]}"
        )

    rule_df = memory_df[
        memory_df["memory_type"] == "RULE"
    ]

    missing_condition_events = sorted(
        set(rule_df["condition_event_id"]) - graph_nodes
    )
    missing_effect_events = sorted(
        set(rule_df["effect_event_id"]) - graph_nodes
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

    invalid_event_nodes = []

    event_node_ids = set(
        rule_df["condition_event_id"]
    ).union(rule_df["effect_event_id"])

    for event_node_id in event_node_ids:
        if graph.nodes[event_node_id].get("node_type") != "EVENT":
            invalid_event_nodes.append(event_node_id)

    if invalid_event_nodes:
        raise ValueError(
            "Một số event reference không phải EVENT node: "
            f"{invalid_event_nodes[:10]}"
        )

    print("Graph consistency check: OK")
    print("- Rule memories:", len(rule_df))
    print(
        "- Event memories:",
        int((memory_df["memory_type"] == "EVENT").sum()),
    )
    print(
        "- Bridge event memories:",
        int(memory_df["is_bridge_event"].fillna(False).sum()),
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

    texts = memory_df["embedding_text"].fillna("").tolist()
    print(f"Encoding {len(texts)} memory records...")

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
    print(
        "Rule memories:",
        int((memory_df["memory_type"] == "RULE").sum()),
    )
    print(
        "Event memories:",
        int((memory_df["memory_type"] == "EVENT").sum()),
    )
    print(
        "Bridge events:",
        int(
            memory_df["is_bridge_event"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .sum()
        ),
    )
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
            "Tạo combined causal memory gồm RULE và EVENT, "
            "sau đó xây một FAISS index dùng cho semantic retrieval "
            "và graph expansion."
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
        "--rule-only",
        action="store_true",
        help="Chỉ tạo RULE memory, không thêm EVENT memory.",
    )
    parser.add_argument(
        "--skip-graph-validation",
        action="store_true",
        help="Bỏ qua kiểm tra memory với GraphML.",
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

    rule_df = build_rule_memory_dataframe(df)
    print("Rule memory rows:", len(rule_df))

    graph = load_graph(args.graph)

    if args.rule_only:
        memory_df = rule_df.copy()
        memory_df["memory_id"] = np.arange(
            len(memory_df),
            dtype=np.int64,
        )
    else:
        event_df = build_event_memory_dataframe(graph)
        print("Event memory rows:", len(event_df))
        print(
            "Bridge event rows:",
            int(event_df["is_bridge_event"].sum()),
        )
        memory_df = combine_memory_dataframes(
            rule_df=rule_df,
            event_df=event_df,
        )

    print("Total memory rows:", len(memory_df))

    if not args.skip_graph_validation:
        validate_memory_against_graph(
            memory_df=memory_df,
            graph=graph,
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
