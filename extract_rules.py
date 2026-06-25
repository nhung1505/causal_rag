import os
import re
import json
import time
import pandas as pd

from tqdm import tqdm
from ollama import chat

# ==========================
# CONFIG
# ==========================

MODEL_NAME = "llama3:latest"

INPUT_FILE = "data/legal_truncated_corpus.csv"
OUTPUT_FILE = "data/outputs/extracted_rules.csv"

MAX_CONTEXT_CHARS = 1000
MAX_OUTPUT_TOKENS = 200

SYSTEM_PROMPT = """
Bạn là hệ thống trích xuất thông tin.

Chỉ trả về đúng JSON.

Ví dụ:

{
  "legal_subject":"",
  "condition":"",
  "effect":"",
  "trigger_event":"",
  "consequence_event":""
}

Không được viết bất kỳ chữ nào ngoài JSON.
"""

# ==========================
# JSON PARSER
# ==========================

def parse_json(text):

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        return None

    try:
        return json.loads(match.group())
    except Exception:
        return None


# ==========================
# EXTRACT RULE
# ==========================

def extract_rule(text):

    if pd.isna(text):
        return None

    text = str(text)[:MAX_CONTEXT_CHARS]

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ],
        options={
            "temperature": 0,
            "num_predict": MAX_OUTPUT_TOKENS
        }
    )
    print(type(response))
    print(response)

    output = response["message"]["content"]
    print(output)

    return parse_json(output)


# ==========================
# LOAD DATA
# ==========================

print("Loading dataset...")
keywords = [
    "thì",
    "phải",
    "bị",
    "được",
    "có trách nhiệm",
    "không được",
    "xử phạt"
]

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-16"
)
df = df[
    df["context"].str.contains(
        "|".join(keywords),
        case=False,
        na=False
    )
]

df = df.head(100)
print(f"Total rows: {len(df)}")

# ==========================
# RESUME SUPPORT
# ==========================

processed_ids = set()

if os.path.exists(OUTPUT_FILE):

    try:

        old_df = pd.read_csv(OUTPUT_FILE)

        if "article_id" in old_df.columns:

            processed_ids = set(
                old_df["article_id"].astype(str)
            )

        print(
            f"Found {len(processed_ids)} processed records"
        )

    except Exception as e:

        print(
            f"Cannot load previous output: {e}"
        )

# ==========================
# CREATE OUTPUT FILE
# ==========================

if not os.path.exists(OUTPUT_FILE):

    pd.DataFrame(columns=[
        "legal_subject",
        "condition",
        "effect",
        "trigger_event",
        "consequence_event",
        "article_id",
        "title"
    ]).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

# ==========================
# MAIN LOOP
# ==========================

start_all = time.time()

for idx, row in enumerate(
    tqdm(
        df.itertuples(index=False),
        total=len(df)
    )
):

    article_id = str(row.id)

    if article_id in processed_ids:
        continue

    try:

        start_row = time.time()

        rule = extract_rule(row.context)

        elapsed = time.time() - start_row

        print(
            f"[{idx+1}/{len(df)}] "
            f"{article_id} "
            f"| {elapsed:.2f}s"
        )

        if rule is None:

            print(
                f"FAILED PARSE: {article_id}"
            )

            continue

        rule["article_id"] = article_id
        rule["title"] = row.title

        pd.DataFrame([rule]).to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as e:

        print(
            f"ERROR {article_id}: {e}"
        )

# ==========================
# SUMMARY
# ==========================

total_time = time.time() - start_all

print("\nDone")
print(
    f"Total runtime: {total_time:.2f}s"
)
print(
    f"Output saved to: {OUTPUT_FILE}"
)