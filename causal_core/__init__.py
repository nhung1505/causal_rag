"""Core types for causal graph construction and inference."""

from causal_core.legal_scm import (
    CounterfactualResult,
    CounterfactualStatus,
    InferenceResult,
    InferenceTraceStep,
    Intervention,
    LegalSCM,
    RuleEvaluation,
    RuleEvaluationStatus,
)
from causal_core.schema import (
    CausalLiteral,
    CausalRule,
    EventState,
    EventType,
    LegalEvent,
    LegacyLegalRule,
    NormalizationAction,
)

__all__ = [
    "CausalLiteral",
    "CausalRule",
    "CounterfactualResult",
    "CounterfactualStatus",
    "EventState",
    "EventType",
    "InferenceResult",
    "InferenceTraceStep",
    "Intervention",
    "LegalEvent",
    "LegalSCM",
    "LegacyLegalRule",
    "NormalizationAction",
    "RuleEvaluation",
    "RuleEvaluationStatus",
]
