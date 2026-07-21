from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


# ============================================================
# CONFIG
# ============================================================

COUNTERFACTUAL_RESULT_PATH = "data/counterfactual_result.json"
OUTPUT_JSON_PATH = "data/final_answer.json"
OUTPUT_TEXT_PATH = "data/final_answer.txt"
PROMPT_OUTPUT_PATH = "data/final_answer_prompt.txt"

DEFAULT_GENERATOR = "template"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_ALTERNATIVES = 3
DEFAULT_MIN_ALTERNATIVE_SCORE = 0.80

VALID_RULE_STATUSES = {"PRESERVED", "ACTIVATED"}
INVALID_RULE_STATUSES = {
    "INVALIDATED",
    "BLOCKED_BY_UPSTREAM",
    "OUTPUT_INTERVENED",
    "OUTPUT_REPLACED",
}

# Các quan hệ đủ mạnh để alternative rule đáng được nêu như ứng viên.
# Alternative vẫn KHÔNG được xem là bằng chứng hợp lệ nếu condition chưa được xác nhận.
STRONG_ALTERNATIVE_RELATIONS = {
    "SAME_EFFECT",
    "CAUSES_TARGET_EFFECT",
    "REPLACEMENT_CONDITION",
    "SAME_ARTICLE",
    "ARTICLE_REFERENCE",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EvidenceRule:
    rule_id: str
    article_id: str
    article_title: str
    legal_subject: str
    condition: str
    effect: str
    condition_norm: str
    effect_norm: str
    status: str
    path_ids: list[int]


@dataclass
class RemovedRule:
    rule_id: str
    article_id: str
    article_title: str
    condition: str
    effect: str
    condition_norm: str
    effect_norm: str
    status: str
    reason: str
    path_ids: list[int]


@dataclass
class AlternativeCandidate:
    rule_id: str
    article_id: str
    article_title: str
    condition: str
    effect: str
    condition_norm: str
    effect_norm: str
    score: float
    relations: list[str]
    path_ids: list[int]
    validation_status: str = "CONDITION_NOT_VERIFIED"


# ============================================================
# HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        value = safe_string(value)
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def humanize_event(event_norm: str) -> str:
    text = safe_string(event_norm)
    if not text:
        return ""
    return text.replace("_", " ").lower()


def clean_generated_text(text: str) -> str:
    text = safe_string(text)
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def load_json(path: str) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required = {
        "original_query",
        "intervention",
        "verification_paths",
        "overall_verification",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(
            "counterfactual_result.json thiếu các trường: "
            f"{sorted(missing)}"
        )

    if not isinstance(data.get("verification_paths"), list):
        raise ValueError("verification_paths phải là một danh sách.")

    return data


# ============================================================
# EVIDENCE REFINEMENT
# ============================================================

class EvidenceRefiner:
    def __init__(
        self,
        min_alternative_score: float,
        max_alternatives: int,
    ) -> None:
        self.min_alternative_score = min_alternative_score
        self.max_alternatives = max_alternatives

    def collect_valid_evidence(
        self,
        result: dict[str, Any],
    ) -> list[EvidenceRule]:
        by_rule_id: dict[str, EvidenceRule] = {}

        for path in result.get("verification_paths", []):
            path_id = safe_int(path.get("path_id"))

            for evaluation in path.get("rule_evaluations", []):
                status = safe_string(
                    evaluation.get("counterfactual_status")
                ).upper()

                if status not in VALID_RULE_STATUSES:
                    continue

                rule = evaluation.get("rule", {})
                rule_id = safe_string(rule.get("rule_id"))
                if not rule_id:
                    continue

                if rule_id not in by_rule_id:
                    by_rule_id[rule_id] = EvidenceRule(
                        rule_id=rule_id,
                        article_id=safe_string(rule.get("article_id")),
                        article_title=safe_string(rule.get("article_title")),
                        legal_subject=safe_string(rule.get("legal_subject")),
                        condition=safe_string(rule.get("condition")),
                        effect=safe_string(rule.get("effect")),
                        condition_norm=safe_string(rule.get("condition_norm")),
                        effect_norm=safe_string(rule.get("effect_norm")),
                        status=status,
                        path_ids=[path_id],
                    )
                elif path_id not in by_rule_id[rule_id].path_ids:
                    by_rule_id[rule_id].path_ids.append(path_id)

        return list(by_rule_id.values())

    def collect_removed_evidence(
        self,
        result: dict[str, Any],
    ) -> list[RemovedRule]:
        by_rule_id: dict[str, RemovedRule] = {}

        for path in result.get("verification_paths", []):
            path_id = safe_int(path.get("path_id"))

            for evaluation in path.get("rule_evaluations", []):
                status = safe_string(
                    evaluation.get("counterfactual_status")
                ).upper()

                if status not in INVALID_RULE_STATUSES:
                    continue

                rule = evaluation.get("rule", {})
                rule_id = safe_string(rule.get("rule_id"))
                if not rule_id:
                    continue

                if rule_id not in by_rule_id:
                    by_rule_id[rule_id] = RemovedRule(
                        rule_id=rule_id,
                        article_id=safe_string(rule.get("article_id")),
                        article_title=safe_string(rule.get("article_title")),
                        condition=safe_string(rule.get("condition")),
                        effect=safe_string(rule.get("effect")),
                        condition_norm=safe_string(rule.get("condition_norm")),
                        effect_norm=safe_string(rule.get("effect_norm")),
                        status=status,
                        reason=safe_string(evaluation.get("reason")),
                        path_ids=[path_id],
                    )
                else:
                    existing = by_rule_id[rule_id]
                    if path_id not in existing.path_ids:
                        existing.path_ids.append(path_id)

                    # Ưu tiên trạng thái tác động trực tiếp hơn BLOCKED_BY_UPSTREAM.
                    if (
                        existing.status == "BLOCKED_BY_UPSTREAM"
                        and status != "BLOCKED_BY_UPSTREAM"
                    ):
                        existing.status = status
                        existing.reason = safe_string(evaluation.get("reason"))

        return list(by_rule_id.values())

    def collect_alternative_candidates(
        self,
        result: dict[str, Any],
        removed_rule_ids: set[str],
        valid_rule_ids: set[str],
    ) -> list[AlternativeCandidate]:
        by_rule_id: dict[str, AlternativeCandidate] = {}

        for path in result.get("verification_paths", []):
            path_id = safe_int(path.get("path_id"))

            for alternative in path.get("alternative_rules", []):
                rule = alternative.get("rule", {})
                rule_id = safe_string(
                    alternative.get("rule_id", rule.get("rule_id"))
                )

                if not rule_id:
                    continue

                # Không gọi lại chính rule factual đã bị loại là alternative support.
                if rule_id in removed_rule_ids or rule_id in valid_rule_ids:
                    continue

                score = safe_float(alternative.get("score"))
                relations = unique_preserve_order(
                    [safe_string(x) for x in alternative.get("relations", [])]
                )

                has_strong_relation = bool(
                    set(relations) & STRONG_ALTERNATIVE_RELATIONS
                )

                if score < self.min_alternative_score or not has_strong_relation:
                    continue

                candidate = AlternativeCandidate(
                    rule_id=rule_id,
                    article_id=safe_string(rule.get("article_id")),
                    article_title=safe_string(rule.get("article_title")),
                    condition=safe_string(rule.get("condition")),
                    effect=safe_string(rule.get("effect")),
                    condition_norm=safe_string(rule.get("condition_norm")),
                    effect_norm=safe_string(rule.get("effect_norm")),
                    score=score,
                    relations=relations,
                    path_ids=[path_id],
                )

                if rule_id not in by_rule_id:
                    by_rule_id[rule_id] = candidate
                else:
                    existing = by_rule_id[rule_id]
                    existing.score = max(existing.score, score)
                    existing.relations = unique_preserve_order(
                        existing.relations + relations
                    )
                    if path_id not in existing.path_ids:
                        existing.path_ids.append(path_id)

        candidates = sorted(
            by_rule_id.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return candidates[: self.max_alternatives]

    @staticmethod
    def collect_path_summary(
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []

        for path in result.get("verification_paths", []):
            summaries.append({
                "path_id": safe_int(path.get("path_id")),
                "path_score": safe_float(path.get("path_score")),
                "path_status": safe_string(path.get("path_status")),
                "factual_event_chain": path.get("factual_event_chain", []),
                "counterfactual_event_chain_prefix": path.get(
                    "counterfactual_event_chain_prefix", []
                ),
                "factual_final_effect": safe_string(
                    path.get("factual_final_effect")
                ),
                "counterfactual_final_effect_reachable": bool(
                    path.get("counterfactual_final_effect_reachable", False)
                ),
                "verification_label": safe_string(
                    path.get("verification", {}).get("label")
                ),
            })

        return summaries

    def refine(self, result: dict[str, Any]) -> dict[str, Any]:
        valid = self.collect_valid_evidence(result)
        removed = self.collect_removed_evidence(result)

        valid_ids = {item.rule_id for item in valid}
        removed_ids = {item.rule_id for item in removed}

        alternatives = self.collect_alternative_candidates(
            result=result,
            removed_rule_ids=removed_ids,
            valid_rule_ids=valid_ids,
        )

        return {
            "valid_evidence": [asdict(item) for item in valid],
            "removed_evidence": [asdict(item) for item in removed],
            "alternative_candidates": [asdict(item) for item in alternatives],
            "path_summary": self.collect_path_summary(result),
        }


# ============================================================
# TEMPLATE ANSWER GENERATION
# ============================================================

class TemplateAnswerGenerator:
    @staticmethod
    def _article_citations(rules: list[dict[str, Any]]) -> list[str]:
        citations: list[str] = []

        for rule in rules:
            article_id = safe_string(rule.get("article_id"))
            title = safe_string(rule.get("article_title"))

            if not article_id:
                continue

            citation = f"Điều {article_id}"
            if title:
                citation += f" ({title})"
            citations.append(citation)

        return unique_preserve_order(citations)

    @staticmethod
    def _removed_effects(removed: list[dict[str, Any]]) -> list[str]:
        return unique_preserve_order([
            safe_string(item.get("effect"))
            for item in removed
            if safe_string(item.get("effect"))
        ])

    @staticmethod
    def _final_effects(
        result: dict[str, Any],
        reachable: bool,
    ) -> list[str]:
        effects: list[str] = []

        for path in result.get("verification_paths", []):
            is_reachable = bool(
                path.get("counterfactual_final_effect_reachable", False)
            )
            if is_reachable != reachable:
                continue

            norm = safe_string(path.get("factual_final_effect"))
            if norm:
                effects.append(humanize_event(norm))

        return unique_preserve_order(effects)

    def generate(
        self,
        source: dict[str, Any],
        refined: dict[str, Any],
    ) -> str:
        intervention = source.get("intervention", {})
        intervention_type = safe_string(
            intervention.get("intervention_type")
        ).upper()
        target_text = safe_string(
            intervention.get("target_event_text")
        ) or humanize_event(
            safe_string(intervention.get("target_event_norm"))
        )
        replacement_text = safe_string(
            intervention.get("replacement_event_text")
        ) or humanize_event(
            safe_string(intervention.get("replacement_event_norm"))
        )

        valid = refined["valid_evidence"]
        removed = refined["removed_evidence"]
        alternatives = refined["alternative_candidates"]

        broken_paths = [
            path for path in refined["path_summary"]
            if path["path_status"] == "CAUSAL_PATH_BROKEN"
        ]
        preserved_paths = [
            path for path in refined["path_summary"]
            if path["path_status"] != "CAUSAL_PATH_BROKEN"
        ]

        lines: list[str] = []

        if intervention_type == "NEGATE":
            lines.append(
                f"Trong giả định phản thực rằng “{target_text}” không xảy ra, "
                "cần đánh giá lại chuỗi suy luận pháp lý ban đầu."
            )
        elif intervention_type == "REPLACE":
            lines.append(
                f"Trong giả định phản thực thay sự kiện “{target_text}” "
                f"bằng “{replacement_text}”, cần đánh giá lại chuỗi suy luận "
                "pháp lý ban đầu."
            )
        else:
            lines.append(
                "Sau can thiệp phản thực, cần đánh giá lại chuỗi suy luận "
                "pháp lý ban đầu."
            )

        if broken_paths:
            lines.append(
                f"Có {len(broken_paths)} causal path factual bị phá vỡ. "
                "Các rule bị tác động trực tiếp hoặc bị chặn bởi sự kiện phía "
                "trước không còn được sử dụng để duy trì kết luận ban đầu."
            )

        if valid:
            lines.append(
                "Những bằng chứng vẫn còn hợp lệ sau can thiệp gồm:"
            )
            for item in valid:
                lines.append(
                    f"- Điều {item['article_id']}: nếu {item['condition']} "
                    f"thì {item['effect']} ({item['status']})."
                )

            reachable_effects = self._final_effects(source, reachable=True)
            if reachable_effects:
                lines.append(
                    "Vì vậy, các hệ quả cuối vẫn còn reachable gồm: "
                    + "; ".join(reachable_effects)
                    + "."
                )
        else:
            lines.append(
                "Không còn rule nào trong các factual path được xác nhận là "
                "PRESERVED hoặc ACTIVATED. Vì vậy, chưa đủ căn cứ từ các "
                "bằng chứng đã truy hồi để giữ nguyên kết luận ban đầu."
            )

            unreachable_effects = self._final_effects(source, reachable=False)
            if unreachable_effects:
                lines.append(
                    "Cụ thể, hệ thống không còn suy ra được qua các path factual: "
                    + "; ".join(unreachable_effects)
                    + "."
                )

        if removed:
            citations = self._article_citations(removed)
            removed_effects = self._removed_effects(removed)

            if citations:
                lines.append(
                    "Chuỗi bị loại trước đây dựa trên "
                    + ", ".join(citations)
                    + "."
                )

            if removed_effects:
                lines.append(
                    "Các kết luận không còn được hỗ trợ bởi chuỗi factual gồm: "
                    + "; ".join(removed_effects)
                    + "."
                )

        if alternatives:
            lines.append(
                "Hệ thống có tìm thấy một số căn cứ thay thế tiềm năng, nhưng "
                "điều kiện của chúng chưa được xác nhận trong tình huống phản thực:"
            )
            for candidate in alternatives:
                lines.append(
                    f"- Điều {candidate['article_id']}: nếu "
                    f"{candidate['condition']} thì {candidate['effect']}."
                )

            lines.append(
                "Do đó, các rule thay thế này chỉ là ứng viên cần kiểm tra thêm, "
                "không phải bằng chứng đủ để khẳng định hệ quả cuối vẫn áp dụng."
            )

        if not valid:
            lines.append(
                "Kết luận: chưa đủ căn cứ pháp lý từ tập bằng chứng hiện có để "
                "xác định người này vẫn thuộc trường hợp tái phạm nguy hiểm, bị "
                "áp dụng quản chế hoặc phải chịu các hệ quả tiếp theo. Việc phủ "
                "định điều kiện ban đầu cũng không tự động chứng minh rằng mọi "
                "hệ quả đó chắc chắn không thể phát sinh từ một căn cứ độc lập khác."
            )
        elif preserved_paths:
            lines.append(
                "Kết luận trên chỉ dựa vào những causal path còn hợp lệ sau can thiệp."
            )

        return "\n\n".join(lines)


# ============================================================
# LLM PROMPT + OLLAMA
# ============================================================

def build_llm_prompt(
    source: dict[str, Any],
    refined: dict[str, Any],
    template_answer: str,
) -> str:
    intervention = source.get("intervention", {})

    lines = [
        "Bạn là hệ thống hỏi đáp pháp luật hình sự Việt Nam.",
        "Hãy viết lại câu trả lời cuối cùng bằng tiếng Việt rõ ràng, thận trọng và có căn cứ.",
        "",
        "QUY TẮC BẮT BUỘC:",
        "1. Chỉ coi rule có trạng thái PRESERVED hoặc ACTIVATED là bằng chứng hợp lệ.",
        "2. Không sử dụng rule INVALIDATED, BLOCKED_BY_UPSTREAM, OUTPUT_INTERVENED hoặc OUTPUT_REPLACED để khẳng định kết luận.",
        "3. Alternative candidate chỉ là ứng viên; không coi là đúng nếu condition của nó chưa được xác nhận.",
        "4. Không suy luận NOT condition đồng nghĩa với NOT effect.",
        "5. Không thêm điều luật, sự kiện hoặc kết luận không có trong context.",
        "6. Khi không còn bằng chứng hợp lệ, phải nói 'chưa đủ căn cứ' thay vì khẳng định chắc chắn không có hệ quả.",
        "7. Trả lời trong 2-5 đoạn, không dùng markdown table.",
        "",
        "CÂU HỎI GỐC:",
        safe_string(source.get("original_query")),
        "",
        "CAN THIỆP PHẢN THỰC:",
        safe_string(intervention.get("description")),
        "",
        "KẾT QUẢ TỔNG THỂ:",
        safe_string(source.get("overall_verification", {}).get("label")),
        safe_string(source.get("overall_verification", {}).get("conclusion")),
        "",
        "BẰNG CHỨNG HỢP LỆ:",
    ]

    valid = refined.get("valid_evidence", [])
    if not valid:
        lines.append("Không có rule PRESERVED hoặc ACTIVATED.")
    else:
        for item in valid:
            lines.extend([
                f"- Rule {item['rule_id']} - Điều {item['article_id']}",
                f"  Điều kiện: {item['condition']}",
                f"  Hệ quả: {item['effect']}",
                f"  Trạng thái: {item['status']}",
            ])

    lines.extend(["", "BẰNG CHỨNG BỊ LOẠI:"])
    for item in refined.get("removed_evidence", []):
        lines.extend([
            f"- Rule {item['rule_id']} - Điều {item['article_id']}",
            f"  Điều kiện: {item['condition']}",
            f"  Hệ quả: {item['effect']}",
            f"  Trạng thái: {item['status']}",
            f"  Lý do: {item['reason']}",
        ])

    lines.extend(["", "ALTERNATIVE CANDIDATES CHƯA XÁC NHẬN:"])
    alternatives = refined.get("alternative_candidates", [])
    if not alternatives:
        lines.append("Không có ứng viên đủ ngưỡng.")
    else:
        for item in alternatives:
            lines.extend([
                f"- Rule {item['rule_id']} - Điều {item['article_id']}",
                f"  Điều kiện cần kiểm tra: {item['condition']}",
                f"  Hệ quả: {item['effect']}",
                f"  Trạng thái: {item['validation_status']}",
            ])

    lines.extend([
        "",
        "BẢN NHÁP DETERMINISTIC:",
        template_answer,
        "",
        "Hãy trả về duy nhất câu trả lời hoàn chỉnh, không thêm tiêu đề như 'Câu trả lời'.",
    ])

    return "\n".join(lines)


def call_ollama(
    prompt: str,
    model: str,
    url: str,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Không gọi được Ollama. Hãy kiểm tra `ollama serve`, URL và model. "
            f"Chi tiết: {exc}"
        ) from exc

    generated = clean_generated_text(response_data.get("response", ""))
    if not generated:
        raise RuntimeError("Ollama trả về nội dung rỗng.")

    return generated


# ============================================================
# OUTPUT
# ============================================================

def build_output(
    source: dict[str, Any],
    refined: dict[str, Any],
    template_answer: str,
    final_answer: str,
    generator: str,
    model: Optional[str],
) -> dict[str, Any]:
    return {
        "original_query": source.get("original_query", ""),
        "intervention": source.get("intervention", {}),
        "source_overall_verification": source.get(
            "overall_verification", {}
        ),
        "evidence_summary": {
            "valid_rule_count": len(refined["valid_evidence"]),
            "removed_rule_count": len(refined["removed_evidence"]),
            "alternative_candidate_count": len(
                refined["alternative_candidates"]
            ),
            "valid_rule_ids": [
                item["rule_id"] for item in refined["valid_evidence"]
            ],
            "removed_rule_ids": [
                item["rule_id"] for item in refined["removed_evidence"]
            ],
            "alternative_candidate_rule_ids": [
                item["rule_id"]
                for item in refined["alternative_candidates"]
            ],
        },
        "refined_evidence": refined,
        "generation": {
            "generator": generator,
            "model": model,
            "template_answer": template_answer,
            "final_answer": final_answer,
        },
    }


def save_outputs(
    output: dict[str, Any],
    prompt: str,
    output_json: str,
    output_text: str,
    prompt_output: str,
) -> None:
    json_path = Path(output_json)
    text_path = Path(output_text)
    prompt_path = Path(prompt_output)

    for path in (json_path, text_path, prompt_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    text_path.write_text(
        safe_string(output["generation"]["final_answer"]) + "\n",
        encoding="utf-8",
    )
    prompt_path.write_text(prompt + "\n", encoding="utf-8")


# ============================================================
# DISPLAY
# ============================================================

def print_summary(output: dict[str, Any]) -> None:
    evidence = output["evidence_summary"]
    generation = output["generation"]

    print("\n" + "=" * 100)
    print("FINAL ANSWER GENERATION")
    print("=" * 100)

    print("\nQuestion:")
    print(output["original_query"])

    print("\nIntervention:")
    print(output["intervention"].get("description", ""))

    print("\nEvidence refinement:")
    print(f"  Valid rules: {evidence['valid_rule_count']}")
    print(f"  Removed rules: {evidence['removed_rule_count']}")
    print(
        "  Unverified alternative candidates: "
        f"{evidence['alternative_candidate_count']}"
    )

    if evidence["valid_rule_ids"]:
        print("  Valid rule IDs:", ", ".join(evidence["valid_rule_ids"]))
    if evidence["removed_rule_ids"]:
        print("  Removed rule IDs:", ", ".join(evidence["removed_rule_ids"]))
    if evidence["alternative_candidate_rule_ids"]:
        print(
            "  Alternative IDs:",
            ", ".join(evidence["alternative_candidate_rule_ids"]),
        )

    print("\nGenerator:", generation["generator"])
    if generation.get("model"):
        print("Model:", generation["model"])

    print("\nFinal answer:")
    print(generation["final_answer"])


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence refinement và sinh câu trả lời cuối cho "
            "path-aware Counterfactual CausalRAG."
        )
    )

    parser.add_argument(
        "--counterfactual-result",
        type=str,
        default=COUNTERFACTUAL_RESULT_PATH,
    )
    parser.add_argument(
        "--generator",
        choices=["template", "ollama"],
        default=DEFAULT_GENERATOR,
        help=(
            "template: deterministic, dễ đánh giá; "
            "ollama: dùng LLM để viết lại bản template."
        ),
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=DEFAULT_MAX_ALTERNATIVES,
    )
    parser.add_argument(
        "--min-alternative-score",
        type=float,
        default=DEFAULT_MIN_ALTERNATIVE_SCORE,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=OUTPUT_JSON_PATH,
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default=OUTPUT_TEXT_PATH,
    )
    parser.add_argument(
        "--prompt-output",
        type=str,
        default=PROMPT_OUTPUT_PATH,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    source = load_json(args.counterfactual_result)

    refiner = EvidenceRefiner(
        min_alternative_score=args.min_alternative_score,
        max_alternatives=args.max_alternatives,
    )
    refined = refiner.refine(source)

    template_generator = TemplateAnswerGenerator()
    template_answer = template_generator.generate(
        source=source,
        refined=refined,
    )

    prompt = build_llm_prompt(
        source=source,
        refined=refined,
        template_answer=template_answer,
    )

    if args.generator == "ollama":
        final_answer = call_ollama(
            prompt=prompt,
            model=args.ollama_model,
            url=args.ollama_url,
            timeout=args.timeout,
        )
        model: Optional[str] = args.ollama_model
    else:
        final_answer = template_answer
        model = None

    output = build_output(
        source=source,
        refined=refined,
        template_answer=template_answer,
        final_answer=final_answer,
        generator=args.generator,
        model=model,
    )

    save_outputs(
        output=output,
        prompt=prompt,
        output_json=args.output_json,
        output_text=args.output_text,
        prompt_output=args.prompt_output,
    )

    print_summary(output)

    print("\nSaved:")
    print(f"- JSON: {args.output_json}")
    print(f"- Text: {args.output_text}")
    print(f"- Prompt: {args.prompt_output}")


if __name__ == "__main__":
    main()