from __future__ import annotations

import argparse
import importlib.util
import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional


DEFAULT_DATASET = "data/evaluation_dataset.json"
DEFAULT_OUTPUT = "data/evaluation_predictions.json"

DEFAULT_RETRIEVER_SCRIPT = "3_multi_hop_causal_retriever.py"
DEFAULT_COUNTERFACTUAL_SCRIPT = "4_counterfactual_verification.py"
DEFAULT_ANSWER_SCRIPT = "5_generate_final_answer.py"

DEFAULT_DATA = "data/4_blhs_merged.json"
DEFAULT_MEMORY = "data/causal_memory.csv"
DEFAULT_INDEX = "data/causal_memory.index"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


def load_module(module_name: str, file_path: str):
    import sys

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy script: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể load module từ {path}")

    module = importlib.util.module_from_spec(spec)

    # Cần đăng ký module trước khi exec_module.
    # Python 3.12 dataclass sử dụng sys.modules để đọc namespace.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Xóa module bị load dở nếu import thất bại.
        sys.modules.pop(module_name, None)
        raise

    return module


def load_dataset(path_value: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy evaluation dataset: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return {}, payload

    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(
            "evaluation_dataset.json phải là list hoặc object có trường `samples`."
        )

    return payload.get("metadata", {}), payload["samples"]


def save_json(data: Any, path_value: str) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def extract_retrieval_prediction(result: dict[str, Any]) -> dict[str, Any]:
    paths = result.get("paths", [])
    evidence = result.get("evidence", [])

    predicted_rule_ids = unique([
        str(item.get("rule_id", ""))
        for item in evidence
        if item.get("rule_id") is not None
    ])

    predicted_article_ids = unique([
        str(item.get("article_id", ""))
        for item in evidence
        if item.get("article_id") is not None
    ])

    predicted_paths: list[dict[str, Any]] = []

    for rank, path in enumerate(paths, start=1):
        rule_ids = [
            str(value).strip()
            for value in path.get("rule_ids", [])
            if str(value).strip()
        ]
        event_chain = [
            str(value).strip()
            for value in path.get("event_chain", [])
            if str(value).strip()
        ]

        if not event_chain:
            # Một số phiên bản retriever chỉ ghi event chain trong evidence/path details.
            rules = path.get("rules", [])
            if rules:
                first_condition = str(rules[0].get("condition_norm", "")).strip()
                effects = [
                    str(rule.get("effect_norm", "")).strip()
                    for rule in rules
                    if str(rule.get("effect_norm", "")).strip()
                ]
                if first_condition:
                    event_chain = [first_condition] + effects

        predicted_paths.append({
            "rank": rank,
            "seed_rule_id": path.get("seed_rule_id"),
            "rule_ids": rule_ids,
            "nodes": event_chain,
            "edges": [
                [event_chain[i], event_chain[i + 1]]
                for i in range(max(0, len(event_chain) - 1))
            ],
            "score": float(path.get("score", 0.0)),
        })

    return {
        "predicted_rule_ids": predicted_rule_ids,
        "predicted_article_ids": predicted_article_ids,
        "predicted_paths": predicted_paths,
        "predicted_path": predicted_paths[0] if predicted_paths else {
            "rank": None,
            "rule_ids": [],
            "nodes": [],
            "edges": [],
            "score": 0.0,
        },
    }


def build_factual_template_answer(
    retrieval_result: dict[str, Any],
    retrieval_prediction: dict[str, Any],
) -> str:
    best_path = retrieval_prediction["predicted_path"]
    rule_ids = best_path.get("rule_ids", [])
    nodes = best_path.get("nodes", [])

    evidence_by_id = {
        str(item.get("rule_id")): item
        for item in retrieval_result.get("evidence", [])
    }

    effects: list[str] = []
    citations: list[str] = []

    for rule_id in rule_ids:
        evidence = evidence_by_id.get(str(rule_id), {})
        effect = str(evidence.get("effect", "")).strip()
        article_id = str(evidence.get("article_id", "")).strip()

        if effect:
            effects.append(effect)
        if article_id:
            citations.append(f"Điều {article_id}")

    effects = unique(effects)
    citations = unique(citations)

    if effects:
        answer = (
            "Theo causal path được truy hồi, chuỗi hậu quả pháp lý là: "
            + " → ".join(effects)
            + "."
        )
        if citations:
            answer += " Căn cứ được truy hồi gồm " + ", ".join(citations) + "."
        return answer

    if nodes:
        return (
            "Theo causal path được truy hồi, hậu quả cuối cùng là "
            + nodes[-1].replace("_", " ").lower()
            + "."
        )

    return "Không tìm thấy causal path đủ căn cứ để trả lời câu hỏi."


def build_factual_ollama_prompt(
    question: str,
    retrieval_result: dict[str, Any],
    template_answer: str,
    max_evidence: int = 8,
) -> str:
    lines = [
        "Bạn là hệ thống hỏi đáp pháp luật hình sự Việt Nam.",
        "Hãy trả lời câu hỏi chỉ dựa trên causal path và evidence được cung cấp.",
        "Không thêm điều luật hoặc kết luận không có trong dữ liệu.",
        "Nếu bằng chứng không đủ, hãy nói chưa đủ căn cứ.",
        "Trả lời bằng tiếng Việt trong 1-3 đoạn.",
        "",
        "CÂU HỎI:",
        question,
        "",
        "CAUSAL PATHS:",
    ]

    for index, path in enumerate(retrieval_result.get("paths", [])[:5], start=1):
        chain = path.get("event_chain", [])
        lines.append(
            f"- Path {index}: "
            + (" -> ".join(chain) if chain else str(path.get("rule_ids", [])))
        )

    lines.extend(["", "EVIDENCE:"])

    for item in retrieval_result.get("evidence", [])[:max_evidence]:
        lines.extend([
            f"- Rule {item.get('rule_id')} - Điều {item.get('article_id')}",
            f"  Điều kiện: {item.get('condition', '')}",
            f"  Hệ quả: {item.get('effect', '')}",
        ])

    lines.extend([
        "",
        "BẢN NHÁP DETERMINISTIC:",
        template_answer,
        "",
        "Chỉ trả về câu trả lời hoàn chỉnh.",
    ])
    return "\n".join(lines)


def normalize_intervention(sample: dict[str, Any]) -> tuple[str, str, Optional[str]]:
    intervention = sample.get("intervention") or {}

    raw_type = str(
        intervention.get("type")
        or intervention.get("intervention_type")
        or "NEGATE"
    ).upper()

    type_mapping = {
        "NEGATE": "NEGATE",
        "NEGATE_EVENT": "NEGATE",
        "REMOVE_EVENT": "NEGATE",
        "REMOVE_CONDITION": "NEGATE",
        "REPLACE": "REPLACE",
        "REPLACE_EVENT": "REPLACE",
        "REPLACE_CONDITION": "REPLACE",
    }
    intervention_type = type_mapping.get(raw_type)

    if intervention_type is None:
        raise ValueError(f"Không hỗ trợ intervention type: {raw_type}")

    target_event = str(
        intervention.get("target_event")
        or intervention.get("target_event_norm")
        or intervention.get("target_condition")
        or ""
    ).strip()

    replacement_event = str(
        intervention.get("replacement_event")
        or intervention.get("replacement_event_norm")
        or intervention.get("replacement_condition")
        or ""
    ).strip() or None

    if not target_event:
        raise ValueError("Mẫu counterfactual thiếu target_event.")

    if intervention_type == "REPLACE" and not replacement_event:
        raise ValueError("Intervention REPLACE thiếu replacement_event.")

    return intervention_type, target_event, replacement_event


def select_samples(
    samples: list[dict[str, Any]],
    limit: Optional[int],
    question_ids: Optional[list[str]],
    task_type: Optional[str],
) -> list[dict[str, Any]]:
    selected = samples

    if question_ids:
        wanted = set(question_ids)
        selected = [
            sample for sample in selected
            if str(sample.get("question_id")) in wanted
        ]

    if task_type:
        selected = [
            sample for sample in selected
            if str(sample.get("task_type", "")).upper() == task_type.upper()
        ]

    if limit is not None:
        selected = selected[:limit]

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chạy batch evaluation qua Multi-hop Retriever, "
            "Counterfactual Verification và Final Answer Generation."
        )
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", default="data/evaluation_runs")

    parser.add_argument("--retriever-script", default=DEFAULT_RETRIEVER_SCRIPT)
    parser.add_argument(
        "--counterfactual-script",
        default=DEFAULT_COUNTERFACTUAL_SCRIPT,
    )
    parser.add_argument("--answer-script", default=DEFAULT_ANSWER_SCRIPT)

    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)

    parser.add_argument("--seed-top-k", type=int, default=8)
    parser.add_argument("--semantic-pool-size", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-expansions", type=int, default=15)
    parser.add_argument("--final-top-k", type=int, default=12)
    parser.add_argument("--min-event-nodes", type=int, default=3)

    parser.add_argument("--top-k-alternatives", type=int, default=10)
    parser.add_argument("--semantic-search-k", type=int, default=40)
    parser.add_argument("--max-paths", type=int, default=12)

    parser.add_argument(
        "--generator",
        choices=["template", "ollama"],
        default="template",
    )
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--min-alternative-score", type=float, default=0.80)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-ids",
        nargs="*",
        default=None,
        help="Ví dụ: --question-ids F001 F002 CF001",
    )
    parser.add_argument(
        "--task-type",
        default=None,
        help="Chỉ chạy một task_type cụ thể.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua các question_id đã có kết quả thành công.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Lưu retrieval/counterfactual/final JSON riêng cho từng mẫu.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata, all_samples = load_dataset(args.dataset)
    samples = select_samples(
        samples=all_samples,
        limit=args.limit,
        question_ids=args.question_ids,
        task_type=args.task_type,
    )

    if not samples:
        raise ValueError("Không có mẫu nào phù hợp để chạy.")

    print("Loading pipeline modules...")

    retriever_module = load_module(
        "multi_hop_retriever_module",
        args.retriever_script,
    )
    counterfactual_module = load_module(
        "counterfactual_module",
        args.counterfactual_script,
    )
    answer_module = load_module(
        "answer_module",
        args.answer_script,
    )

    print("Initializing shared repository and embedding model...")

    retrieval_repository = retriever_module.RuleRepository(args.data)
    dense_retriever = retriever_module.DenseRuleRetriever(
        repository=retrieval_repository,
        index_path=args.index,
        memory_path=args.memory,
        model_name=args.embedding_model,
    )
    retriever = retriever_module.MultiHopCausalRetriever(
        repository=retrieval_repository,
        dense_retriever=dense_retriever,
    )

    # Bước 4 đang định nghĩa repository/dense search riêng, nên khởi tạo một lần
    # và tái sử dụng cho toàn bộ các mẫu counterfactual.
    counterfactual_repository = counterfactual_module.RuleRepository(args.data)
    counterfactual_dense = counterfactual_module.DenseRuleSearch(
        repository=counterfactual_repository,
        memory_path=args.memory,
        index_path=args.index,
        model_name=args.embedding_model,
    )
    verifier = counterfactual_module.CounterfactualVerifier(
        repository=counterfactual_repository,
        dense_search=counterfactual_dense,
    )

    refiner = answer_module.EvidenceRefiner(
        min_alternative_score=args.min_alternative_score,
        max_alternatives=args.max_alternatives,
    )
    counterfactual_template_generator = answer_module.TemplateAnswerGenerator()

    output_path = Path(args.output)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    existing_predictions: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            existing_payload = json.load(file)
        existing_predictions = existing_payload.get("predictions", [])
        completed_ids = {
            str(item.get("question_id"))
            for item in existing_predictions
            if not item.get("error")
        }

    predictions = list(existing_predictions)
    total = len(samples)

    for position, sample in enumerate(samples, start=1):
        question_id = str(sample.get("question_id", f"sample_{position}"))
        question = str(sample.get("question", "")).strip()
        task_type = str(sample.get("task_type", "")).upper()

        if question_id in completed_ids:
            print(f"[{position}/{total}] Skip {question_id}: đã hoàn thành.")
            continue

        print(f"\n[{position}/{total}] Running {question_id} | {task_type}")
        started = time.perf_counter()

        prediction: dict[str, Any] = {
            "question_id": question_id,
            "source_chain_id": sample.get("source_chain_id"),
            "task_type": task_type,
            "question": question,
            "predicted_rule_ids": [],
            "predicted_article_ids": [],
            "predicted_paths": [],
            "predicted_path": {
                "rank": None,
                "rule_ids": [],
                "nodes": [],
                "edges": [],
                "score": 0.0,
            },
            "predicted_counterfactual_status": None,
            "predicted_overall_verification_label": None,
            "predicted_final_effect_reachable": None,
            "predicted_answer_label": None,
            "predicted_answer": "",
            "generator": args.generator,
            "runtime_seconds": None,
            "error": None,
        }

        try:
            retrieval_result = retriever.retrieve(
                query=question,
                seed_top_k=args.seed_top_k,
                semantic_pool_size=args.semantic_pool_size,
                max_depth=args.max_depth,
                max_expansions_per_rule=args.max_expansions,
                final_top_k=args.final_top_k,
                causal_only=True,
                min_event_nodes=args.min_event_nodes,
            )

            retrieval_prediction = extract_retrieval_prediction(retrieval_result)
            prediction.update(retrieval_prediction)

            sample_dir = work_dir / question_id
            if args.save_intermediate:
                save_json(
                    retrieval_result,
                    str(sample_dir / "retrieval_result.json"),
                )

            is_counterfactual = (
                task_type.startswith("COUNTERFACTUAL")
                or sample.get("intervention") is not None
            )

            if not is_counterfactual:
                template_answer = build_factual_template_answer(
                    retrieval_result=retrieval_result,
                    retrieval_prediction=retrieval_prediction,
                )

                if args.generator == "ollama":
                    prompt = build_factual_ollama_prompt(
                        question=question,
                        retrieval_result=retrieval_result,
                        template_answer=template_answer,
                    )
                    final_answer = answer_module.call_ollama(
                        prompt=prompt,
                        model=args.ollama_model,
                        url=args.ollama_url,
                        timeout=args.timeout,
                    )
                else:
                    prompt = ""
                    final_answer = template_answer

                has_path = bool(retrieval_prediction["predicted_paths"])
                prediction["predicted_final_effect_reachable"] = has_path
                prediction["predicted_answer_label"] = (
                    "SUPPORTED" if has_path else "NOT_ENOUGH_LEGAL_BASIS"
                )
                prediction["predicted_answer"] = final_answer

                if args.save_intermediate:
                    save_json(
                        {
                            "question": question,
                            "template_answer": template_answer,
                            "final_answer": final_answer,
                            "generator": args.generator,
                        },
                        str(sample_dir / "final_answer.json"),
                    )

                    (sample_dir / "final_answer.txt").write_text(
                        final_answer + "\n",
                        encoding="utf-8",
                    )

                    if prompt:
                        (sample_dir / "final_answer_prompt.txt").write_text(
                            prompt + "\n",
                            encoding="utf-8",
                        )

            else:
                intervention_type, target_event, replacement_event = (
                    normalize_intervention(sample)
                )

                intervention = counterfactual_module.InterventionParser.create(
                    repository=counterfactual_repository,
                    target_event=target_event,
                    intervention_type=intervention_type,
                    replacement_event=replacement_event,
                )

                sample_dir.mkdir(parents=True, exist_ok=True)
                retrieval_file = sample_dir / "retrieval_result.json"
                save_json(retrieval_result, str(retrieval_file))

                counterfactual_result = verifier.verify(
                    retrieval_result_path=str(retrieval_file),
                    intervention=intervention,
                    top_k_alternatives=args.top_k_alternatives,
                    semantic_search_k=args.semantic_search_k,
                    max_paths=args.max_paths,
                )

                if args.save_intermediate:
                    save_json(
                        counterfactual_result,
                        str(sample_dir / "counterfactual_result.json"),
                    )

                refined = refiner.refine(counterfactual_result)
                template_answer = counterfactual_template_generator.generate(
                    source=counterfactual_result,
                    refined=refined,
                )
                prompt = answer_module.build_llm_prompt(
                    source=counterfactual_result,
                    refined=refined,
                    template_answer=template_answer,
                )

                if args.generator == "ollama":
                    final_answer = answer_module.call_ollama(
                        prompt=prompt,
                        model=args.ollama_model,
                        url=args.ollama_url,
                        timeout=args.timeout,
                    )
                else:
                    final_answer = template_answer

                final_output = answer_module.build_output(
                    source=counterfactual_result,
                    refined=refined,
                    template_answer=template_answer,
                    final_answer=final_answer,
                    generator=args.generator,
                    model=args.ollama_model if args.generator == "ollama" else None,
                )

                overall = counterfactual_result.get(
                    "overall_verification", {}
                )
                verification_paths = counterfactual_result.get(
                    "verification_paths", []
                )

                reachable = any(
                    bool(path.get(
                        "counterfactual_final_effect_reachable", False
                    ))
                    for path in verification_paths
                )

                path_statuses = unique([
                    str(path.get("path_status", ""))
                    for path in verification_paths
                ])

                if path_statuses == ["CAUSAL_PATH_BROKEN"]:
                    predicted_cf_status = "INVALIDATED"
                elif "CAUSAL_PATH_BROKEN" in path_statuses:
                    predicted_cf_status = "PARTIALLY_INVALIDATED"
                elif verification_paths:
                    predicted_cf_status = "PRESERVED"
                else:
                    predicted_cf_status = "NO_PATH_FOUND"

                prediction["predicted_counterfactual_status"] = (
                    predicted_cf_status
                )
                prediction["predicted_overall_verification_label"] = (
                    overall.get("label")
                )
                prediction["predicted_final_effect_reachable"] = reachable
                prediction["predicted_answer_label"] = (
                    "SUPPORTED"
                    if reachable
                    else "NOT_ENOUGH_LEGAL_BASIS"
                )
                prediction["predicted_answer"] = final_answer
                prediction["evidence_summary"] = final_output.get(
                    "evidence_summary", {}
                )

                if args.save_intermediate:
                    save_json(
                        final_output,
                        str(sample_dir / "final_answer.json"),
                    )
                    (sample_dir / "final_answer.txt").write_text(
                        final_answer + "\n",
                        encoding="utf-8",
                    )
                    (sample_dir / "final_answer_prompt.txt").write_text(
                        prompt + "\n",
                        encoding="utf-8",
                    )

        except Exception as exc:
            prediction["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            prediction["traceback"] = traceback.format_exc()
            print(f"ERROR {question_id}: {prediction['error']}")

        prediction["runtime_seconds"] = round(
            time.perf_counter() - started,
            4,
        )

        # Xóa bản ghi cũ cùng question_id khi resume/chạy lại.
        predictions = [
            item for item in predictions
            if str(item.get("question_id")) != question_id
        ]
        predictions.append(prediction)

        payload = {
            "metadata": {
                "source_dataset": args.dataset,
                "source_dataset_metadata": metadata,
                "generator": args.generator,
                "ollama_model": (
                    args.ollama_model
                    if args.generator == "ollama"
                    else None
                ),
                "number_of_requested_samples": total,
                "number_of_predictions": len(predictions),
                "successful_predictions": sum(
                    not item.get("error") for item in predictions
                ),
                "failed_predictions": sum(
                    bool(item.get("error")) for item in predictions
                ),
            },
            "predictions": sorted(
                predictions,
                key=lambda item: str(item.get("question_id", "")),
            ),
        }
        save_json(payload, args.output)

        print(
            f"Completed {question_id} in "
            f"{prediction['runtime_seconds']:.2f}s"
        )

    print("\nBatch evaluation finished.")
    print(f"Predictions: {args.output}")
    print(f"Intermediate files: {args.work_dir}")


if __name__ == "__main__":
    main()
