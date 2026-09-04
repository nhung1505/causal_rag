#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Base/Vanilla RAG baseline for BLHS multi-hop QA.

- Retrieval corpus: raw legal articles in data/1_raw_data.json
- Retrieval unit: one legal article
- Dense retriever: BAAI/bge-m3 + normalized embeddings + FAISS IndexFlatIP
- Generator: Ollama (default qwen3:8b)
- No structured-rule retrieval, no event retrieval, no causal graph/path,
  no Counterfactual Verification, and no access to gold fields.
- Output schema is compatible with 7_compute_evaluation_metrics.py.
"""

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib import error, request

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VERSION = "1.0-vanilla-rag-raw-article"
DEFAULT_BENCHMARK = "data/blhs_multihop_benchmark_250.json"
DEFAULT_CORPUS = "data/1_raw_data.json"
DEFAULT_OUTPUT = "data/baselines/vanilla_rag_predictions.json"
DEFAULT_JSONL = "data/baselines/vanilla_rag_predictions.jsonl"
DEFAULT_ERRORS = "data/baselines/vanilla_rag_errors.json"
DEFAULT_RUN_LOG = "data/baselines/vanilla_rag_run_log.json"
DEFAULT_INDEX = "data/baselines/vanilla_rag_articles.index"
DEFAULT_INDEX_META = "data/baselines/vanilla_rag_articles_index_meta.json"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def safe_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_id(value: Any) -> str:
    text = safe_string(value)
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def unique(values: list[Any]) -> list[str]:
    out, seen = [], set()
    for value in values:
        text = safe_string(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    content: str

    @property
    def embedding_text(self) -> str:
        return f"Điều {self.article_id}. {self.title}\n{self.content}".strip()


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str


@dataclass(frozen=True)
class Hit:
    rank: int
    article_id: str
    title: str
    content: str
    score: float


def load_articles(path: str | Path) -> list[Article]:
    payload = load_json(path)
    rows = payload if isinstance(payload, list) else (
        payload.get("articles") or payload.get("data") or payload.get("documents")
    )
    if not isinstance(rows, list):
        raise ValueError("Corpus phải là list hoặc chứa articles/data/documents.")

    articles, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        article_id = normalize_id(row.get("article_id") or row.get("id"))
        title = safe_string(row.get("article_title") or row.get("title"))
        content = safe_string(row.get("content") or row.get("text"))
        if not article_id or not content:
            continue
        if article_id in seen:
            raise ValueError(f"article_id bị trùng: {article_id}")
        seen.add(article_id)
        articles.append(Article(article_id, title, content))

    if not articles:
        raise ValueError("Không đọc được article nào từ corpus.")
    return articles


def load_samples(path: str | Path, start_index: int = 0, limit: Optional[int] = None):
    payload = load_json(path)
    if isinstance(payload, list):
        metadata, rows = {}, payload
    elif isinstance(payload, Mapping):
        metadata = dict(payload.get("metadata") or {})
        rows = payload.get("questions") or payload.get("samples") or payload.get("data")
    else:
        raise ValueError("Benchmark không hợp lệ.")
    if not isinstance(rows, list):
        raise ValueError("Benchmark thiếu questions/samples/data.")

    samples = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        # Baseline chỉ đọc ID + QUESTION; không đọc evaluation/gold_*.
        sample_id = normalize_id(row.get("id") or row.get("sample_id") or row.get("question_id"))
        question = safe_string(row.get("question"))
        if sample_id and question:
            samples.append(Sample(sample_id, question))

    samples = samples[start_index:]
    if limit is not None:
        samples = samples[:limit]
    return metadata, samples


def fingerprint(articles: list[Article], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for a in articles:
        h.update(a.article_id.encode())
        h.update(b"\0")
        h.update(a.title.encode())
        h.update(b"\0")
        h.update(a.content.encode())
        h.update(b"\x1e")
    return h.hexdigest()


class DenseRetriever:
    def __init__(self, articles, model_name, batch_size, index_path, meta_path, rebuild=False):
        self.articles = articles
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.fp = fingerprint(articles, model_name)
        self.index = self._load_or_build(batch_size, rebuild)

    def _load_or_build(self, batch_size: int, rebuild: bool):
        if not rebuild and self.index_path.exists() and self.meta_path.exists():
            try:
                meta = load_json(self.meta_path)
                idx = faiss.read_index(str(self.index_path))
                ids = [a.article_id for a in self.articles]
                if (
                    meta.get("fingerprint") == self.fp
                    and meta.get("embedding_model") == self.model_name
                    and meta.get("article_ids") == ids
                    and idx.ntotal == len(ids)
                ):
                    print(f"Loaded cached FAISS index: {self.index_path}")
                    return idx
            except Exception as exc:
                print(f"Cache invalid, rebuild index: {exc}")

        texts = [a.embedding_text for a in self.articles]
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        idx = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(self.index_path))
        write_json(self.meta_path, {
            "version": VERSION,
            "embedding_model": self.model_name,
            "similarity": "cosine_via_normalized_inner_product",
            "article_ids": [a.article_id for a in self.articles],
            "fingerprint": self.fp,
        })
        return idx

    def retrieve(self, question: str, top_k: int) -> list[Hit]:
        q = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        scores, ids = self.index.search(q, min(top_k, len(self.articles)))
        hits = []
        for rank, (score, pos) in enumerate(zip(scores[0], ids[0]), start=1):
            if pos < 0:
                continue
            a = self.articles[int(pos)]
            hits.append(Hit(rank, a.article_id, a.title, a.content, float(score)))
        return hits


def http_post_json(url: str, payload: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    req = request.Request(
        url=url,
        data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Không kết nối Ollama: {exc.reason}") from exc
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("Ollama response không phải JSON object.")
    return result


SYSTEM_PROMPT = """Bạn là hệ thống hỏi đáp pháp luật Việt Nam trong thí nghiệm Vanilla RAG.
Bạn chỉ được sử dụng các điều luật được truy xuất trong EVIDENCE.
Không sử dụng causal graph, causal path hoặc Counterfactual Verification.
Nếu evidence không đủ, trả decision UNCERTAIN. Không bịa thêm quy định.
Citation chỉ được lấy từ các điều luật xuất hiện trong evidence.

Trả về DUY NHẤT JSON object:
{
  "verification_decision": "SUPPORTED | REJECT_DIRECT_CLAIM | UNCERTAIN",
  "decision_score": 0.0,
  "final_answer": "câu trả lời ngắn gọn",
  "citations": ["Điều 15", "Điều 57"]
}
""".strip()


def build_prompt(question: str, hits: list[Hit], max_article_chars: int) -> str:
    blocks = []
    for hit in hits:
        content = hit.content
        if max_article_chars > 0 and len(content) > max_article_chars:
            content = content[:max_article_chars].rstrip() + "\n...[truncated]"
        blocks.append(
            f"[Tài liệu {hit.rank}]\nĐiều {hit.article_id}. {hit.title}\n{content}"
        )
    return (
        f"CÂU HỎI:\n{question}\n\n"
        f"EVIDENCE:\n" + "\n\n".join(blocks) +
        "\n\nTrả lời dựa duy nhất trên evidence và đúng JSON schema."
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", safe_string(text), flags=re.I)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response không có JSON object.")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(text[start:i + 1])
                if isinstance(obj, dict):
                    return obj
    raise ValueError("Không parse được JSON từ LLM.")


def normalize_decision(value: Any) -> str:
    key = re.sub(r"[^A-Z0-9]+", "_", safe_string(value).upper()).strip("_")
    if key in {"SUPPORTED", "SUPPORT", "TRUE", "YES", "CORRECT"}:
        return "SUPPORTED"
    if key in {"REJECT_DIRECT_CLAIM", "REJECT", "FALSE", "NO", "CONTRADICTED", "NOT_SUPPORTED"}:
        return "REJECT_DIRECT_CLAIM"
    return "UNCERTAIN"


def parse_output(raw: str, allowed_ids: list[str]):
    obj = extract_json_object(raw)
    decision = normalize_decision(
        obj.get("verification_decision") or obj.get("decision") or obj.get("final_decision")
    )
    try:
        score = max(0.0, min(1.0, float(obj.get("decision_score", obj.get("confidence", 0.5)))))
    except (TypeError, ValueError):
        score = 0.5
    answer = safe_string(obj.get("final_answer") or obj.get("answer") or obj.get("response"))
    if not answer:
        raise ValueError("LLM JSON thiếu final_answer.")

    raw_citations = obj.get("citations") or []
    if not isinstance(raw_citations, list):
        raw_citations = [raw_citations]
    cited_ids = []
    for item in raw_citations:
        text = safe_string(item)
        matches = re.findall(r"\b(?:điều|dieu)\s*(\d+(?:\.\d+)?)\b", text, flags=re.I)
        if matches:
            cited_ids.extend(normalize_id(x) for x in matches)
        elif re.fullmatch(r"\d+(?:\.\d+)?", text):
            cited_ids.append(normalize_id(text))
    allowed = set(allowed_ids)
    cited_ids = [x for x in unique(cited_ids) if x in allowed]
    return decision, score, answer, [f"Điều {x}" for x in cited_ids]


def ollama_generate(base_url, model, question, hits, temperature, max_tokens, timeout, retries, max_article_chars):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = http_post_json(
                f"{base_url.rstrip('/')}/api/chat",
                {
                    "model": model,
                    "stream": False,
                    "format": "json",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(question, hits, max_article_chars)},
                    ],
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout,
            )
            raw = safe_string((response.get("message") or {}).get("content"))
            if not raw:
                raise RuntimeError("Ollama trả content rỗng.")
            return parse_output(raw, [h.article_id for h in hits])
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 5))
    raise last_exc


def prediction_row(sample: Sample, hits: list[Hit], result, model, embedding_model, top_k, runtime):
    decision, score, answer, citations = result
    article_ids = [h.article_id for h in hits]
    hit_meta = [
        {"rank": h.rank, "article_id": h.article_id, "article_title": h.title, "score": round(h.score, 8)}
        for h in hits
    ]
    return {
        "id": sample.sample_id,
        "question": sample.question,
        "method": "VANILLA_RAG",
        "baseline_version": VERSION,
        "retrieved_rule_ids": [],
        "retrieved_event_ids": [],
        "retrieved_article_ids": article_ids,
        "reasoning_path": None,
        "reasoning_path_id": -1,
        "verification_decision": decision,
        "decision_score": score,
        "final_answer": answer,
        "citations": citations,
        "retrieval": {
            "retrieval_type": "dense_raw_article",
            "embedding_model": embedding_model,
            "top_k": top_k,
            "retrieved_rule_ids": [],
            "retrieved_event_ids": [],
            "retrieved_article_ids": article_ids,
            "retrieved_articles": hit_meta,
            "causal_paths": [],
        },
        "verification": {
            "final_decision": decision,
            "decision_score": score,
            "verification_method": "llm_on_retrieved_raw_articles",
            "counterfactual_verification_enabled": False,
        },
        "generation": {
            "provider": "ollama",
            "model": model,
            "answer": answer,
            "citations": citations,
            "final_decision": decision,
            "decision_score": score,
        },
        "runtime_seconds": round(runtime, 6),
        "pipeline_metadata": {
            "method": "VANILLA_RAG",
            "baseline_version": VERSION,
            "retrieval_enabled": True,
            "retrieval_type": "dense_raw_article",
            "retrieval_unit": "article",
            "structured_rule_retrieval_enabled": False,
            "event_retrieval_enabled": False,
            "causal_graph_enabled": False,
            "causal_path_enabled": False,
            "counterfactual_verification_enabled": False,
            "embedding_model": embedding_model,
            "top_k": top_k,
            "provider": "ollama",
            "model": model,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--jsonl-output", default=DEFAULT_JSONL)
    p.add_argument("--errors-output", default=DEFAULT_ERRORS)
    p.add_argument("--run-log", default=DEFAULT_RUN_LOG)
    p.add_argument("--index-path", default=DEFAULT_INDEX)
    p.add_argument("--index-meta", default=DEFAULT_INDEX_META)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--embedding-batch-size", type=int, default=16)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--max-article-chars", type=int, default=6000)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_existing(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = load_json(p)
        rows = payload.get("predictions") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            return {}
        return {normalize_id(r.get("id")): dict(r) for r in rows if isinstance(r, Mapping) and normalize_id(r.get("id"))}
    except Exception:
        return {}


def main():
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k phải >= 1")

    if args.overwrite:
        for path in (args.output, args.jsonl_output, args.errors_output, args.run_log):
            p = Path(path)
            if p.exists():
                p.unlink()

    benchmark_meta, samples = load_samples(args.benchmark, args.start_index, args.limit)
    articles = load_articles(args.corpus)
    retriever = DenseRetriever(
        articles,
        args.embedding_model,
        args.embedding_batch_size,
        args.index_path,
        args.index_meta,
        args.rebuild_index,
    )

    predictions = load_existing(args.output) if args.resume else {}
    errors = []
    started_all = time.time()

    print("=" * 80)
    print("VANILLA RAG BASELINE")
    print(f"Corpus articles  : {len(articles)}")
    print(f"Samples          : {len(samples)}")
    print(f"Retriever        : {args.embedding_model}, top_k={args.top_k}")
    print(f"Generator        : {args.model}")
    print("Graph/Path/CF    : DISABLED")
    print("=" * 80)

    for i, sample in enumerate(samples, start=1):
        if args.resume and sample.sample_id in predictions and not safe_string(predictions[sample.sample_id].get("error")):
            print(f"[{i}/{len(samples)}] {sample.sample_id} SKIP")
            continue

        t0 = time.time()
        hits = []
        try:
            hits = retriever.retrieve(sample.question, args.top_k)
            result = ollama_generate(
                args.ollama_url,
                args.model,
                sample.question,
                hits,
                args.temperature,
                args.max_tokens,
                args.timeout,
                args.retries,
                args.max_article_chars,
            )
            row = prediction_row(
                sample, hits, result, args.model, args.embedding_model,
                args.top_k, time.time() - t0,
            )
            predictions[sample.sample_id] = row
            append_jsonl(args.jsonl_output, row)
            print(
                f"[{i}/{len(samples)}] {sample.sample_id} "
                f"articles={','.join(h.article_id for h in hits)} "
                f"decision={row['verification_decision']}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            row = {
                "id": sample.sample_id,
                "question": sample.question,
                "method": "VANILLA_RAG",
                "baseline_version": VERSION,
                "retrieved_rule_ids": [],
                "retrieved_event_ids": [],
                "retrieved_article_ids": [h.article_id for h in hits],
                "reasoning_path": None,
                "reasoning_path_id": -1,
                "verification_decision": "",
                "decision_score": None,
                "final_answer": "",
                "citations": [],
                "runtime_seconds": round(time.time() - t0, 6),
                "error": message,
                "pipeline_metadata": {
                    "method": "VANILLA_RAG",
                    "baseline_version": VERSION,
                    "retrieval_enabled": True,
                    "causal_graph_enabled": False,
                    "counterfactual_verification_enabled": False,
                },
            }
            predictions[sample.sample_id] = row
            append_jsonl(args.jsonl_output, row)
            errors.append({"id": sample.sample_id, "error": message})
            print(f"[{i}/{len(samples)}] {sample.sample_id} ERROR: {message}")

        ordered = [predictions[s.sample_id] for s in samples if s.sample_id in predictions]
        write_json(args.output, {
            "metadata": {
                "method": "VANILLA_RAG",
                "baseline_version": VERSION,
                "benchmark": args.benchmark,
                "corpus": args.corpus,
                "retrieval_enabled": True,
                "retrieval_type": "dense_raw_article",
                "retrieval_unit": "article",
                "structured_rule_retrieval_enabled": False,
                "event_retrieval_enabled": False,
                "causal_graph_enabled": False,
                "causal_path_enabled": False,
                "counterfactual_verification_enabled": False,
                "embedding_model": args.embedding_model,
                "top_k": args.top_k,
                "provider": "ollama",
                "model": args.model,
                "benchmark_metadata": benchmark_meta,
                "prediction_count": len(ordered),
                "successful_prediction_count": sum(not safe_string(x.get("error")) for x in ordered),
                "failed_prediction_count": sum(bool(safe_string(x.get("error"))) for x in ordered),
            },
            "predictions": ordered,
        })

    ordered = [predictions[s.sample_id] for s in samples if s.sample_id in predictions]
    all_errors = [
        {"id": r.get("id"), "question": r.get("question"), "error": r.get("error")}
        for r in ordered if safe_string(r.get("error"))
    ]
    write_json(args.errors_output, {"method": "VANILLA_RAG", "version": VERSION, "errors": all_errors})
    write_json(args.run_log, {
        "method": "VANILLA_RAG",
        "version": VERSION,
        "benchmark": args.benchmark,
        "corpus": args.corpus,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "model": args.model,
        "prediction_count": len(ordered),
        "failed_prediction_count": len(all_errors),
        "elapsed_seconds": round(time.time() - started_all, 6),
    })

    print("=" * 80)
    print(f"Done: {len(ordered)} predictions, {len(all_errors)} errors")
    print(f"Output: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
