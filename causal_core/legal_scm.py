"""Deterministic structural inference for normative legal causal rules.

``LegalSCM`` is intentionally independent from NetworkX and the retrieval
pipeline.  It executes canonical :class:`~causal_core.schema.CausalRule`
mechanisms, supports explicit ``do(X=x)`` interventions, and returns auditable
factual/counterfactual traces with rule provenance.

This is a normative Boolean structural model, not causal discovery or causal
effect estimation from observational data.  Events produced by at least one
rule are treated as endogenous and default to ``FALSE`` when no mechanism
produces them.  Root events remain ``UNKNOWN`` until supplied in the context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

from causal_core.schema import CausalLiteral, CausalRule, EventState


AssignmentValue = Union[EventState, bool, str, None]
SupportMap = dict[str, set[EventState]]


class RuleEvaluationStatus(str, Enum):
    ACTIVATED = "ACTIVATED"
    UNSATISFIED = "UNSATISFIED"
    BLOCKED_BY_EXCEPTION = "BLOCKED_BY_EXCEPTION"
    BLOCKED_BY_INTERVENTION = "BLOCKED_BY_INTERVENTION"


class CounterfactualStatus(str, Enum):
    NECESSARY = "NECESSARY"
    NON_NECESSARY = "NON_NECESSARY"
    SUFFICIENT = "SUFFICIENT"
    NO_EFFECT = "NO_EFFECT"
    OUTCOME_CHANGED = "OUTCOME_CHANGED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Intervention:
    """A hard intervention that fixes one event to TRUE or FALSE."""

    event_id: str
    state: EventState

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        state = EventState.from_value(self.state)
        if not event_id:
            raise ValueError("Intervention.event_id không được để trống.")
        if state is EventState.UNKNOWN:
            raise ValueError("Không thể do-intervention một event thành UNKNOWN.")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, str]:
        return {"event_id": self.event_id, "state": self.state.value}


@dataclass(frozen=True)
class RuleEvaluation:
    rule_id: str
    status: RuleEvaluationStatus
    effect: CausalLiteral
    unsatisfied_conditions: tuple[CausalLiteral, ...] = ()
    matched_exceptions: tuple[CausalLiteral, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "effect": self.effect.to_dict(),
            "unsatisfied_conditions": [
                literal.to_dict() for literal in self.unsatisfied_conditions
            ],
            "matched_exceptions": [
                literal.to_dict() for literal in self.matched_exceptions
            ],
        }


@dataclass(frozen=True)
class InferenceTraceStep:
    iteration: int
    rule_id: str
    article_id: int | str | None
    conditions: tuple[CausalLiteral, ...]
    effect: CausalLiteral

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "rule_id": self.rule_id,
            "article_id": self.article_id,
            "conditions": [literal.to_dict() for literal in self.conditions],
            "effect": self.effect.to_dict(),
        }


@dataclass(frozen=True)
class InferenceResult:
    """Auditable result of one structural-world evaluation."""

    states: dict[str, EventState]
    supports: dict[str, tuple[EventState, ...]]
    context: dict[str, EventState]
    interventions: dict[str, EventState]
    activated_rule_ids: tuple[str, ...]
    blocked_rule_ids: tuple[str, ...]
    pending_rule_ids: tuple[str, ...]
    rule_evaluations: tuple[RuleEvaluation, ...]
    trace: tuple[InferenceTraceStep, ...]
    supporting_rule_ids: dict[str, tuple[str, ...]]
    conflicting_event_ids: tuple[str, ...]
    iterations: int
    converged: bool
    closed_world: bool

    def state_of(self, event_id: str) -> EventState:
        return self.states.get(str(event_id).strip(), EventState.UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": {
                event_id: state.value
                for event_id, state in sorted(self.states.items())
            },
            "supports": {
                event_id: [state.value for state in states]
                for event_id, states in sorted(self.supports.items())
            },
            "context": {
                event_id: state.value
                for event_id, state in sorted(self.context.items())
            },
            "interventions": {
                event_id: state.value
                for event_id, state in sorted(self.interventions.items())
            },
            "activated_rule_ids": list(self.activated_rule_ids),
            "blocked_rule_ids": list(self.blocked_rule_ids),
            "pending_rule_ids": list(self.pending_rule_ids),
            "rule_evaluations": [
                evaluation.to_dict() for evaluation in self.rule_evaluations
            ],
            "trace": [step.to_dict() for step in self.trace],
            "supporting_rule_ids": {
                event_id: list(rule_ids)
                for event_id, rule_ids in sorted(
                    self.supporting_rule_ids.items()
                )
            },
            "conflicting_event_ids": list(self.conflicting_event_ids),
            "iterations": self.iterations,
            "converged": self.converged,
            "closed_world": self.closed_world,
        }


@dataclass(frozen=True)
class CounterfactualResult:
    outcome_event_id: str
    interventions: dict[str, EventState]
    factual_outcome: EventState
    counterfactual_outcome: EventState
    outcome_changed: bool | None
    status: CounterfactualStatus
    factual: InferenceResult
    counterfactual: InferenceResult
    disabled_rule_ids: tuple[str, ...]
    newly_activated_rule_ids: tuple[str, ...]
    alternative_outcome_rule_ids: tuple[str, ...]
    recomputed_context_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_event_id": self.outcome_event_id,
            "interventions": {
                event_id: state.value
                for event_id, state in sorted(self.interventions.items())
            },
            "factual_outcome": self.factual_outcome.value,
            "counterfactual_outcome": self.counterfactual_outcome.value,
            "outcome_changed": self.outcome_changed,
            "status": self.status.value,
            "disabled_rule_ids": list(self.disabled_rule_ids),
            "newly_activated_rule_ids": list(
                self.newly_activated_rule_ids
            ),
            "alternative_outcome_rule_ids": list(
                self.alternative_outcome_rule_ids
            ),
            "recomputed_context_event_ids": list(
                self.recomputed_context_event_ids
            ),
            "factual": self.factual.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
        }


class LegalSCM:
    """Forward structural model over legal rule mechanisms.

    Context assignments are factual inputs.  A hard intervention takes
    precedence over context and disables every mechanism whose effect is the
    intervened event, implementing the graph operation that cuts incoming
    causal edges before fixing the event value.
    """

    def __init__(
        self,
        rules: Iterable[CausalRule],
        *,
        validate: bool = True,
        closed_world: bool = True,
    ) -> None:
        self.rules = tuple(rules)
        if not self.rules:
            raise ValueError("LegalSCM cần ít nhất một CausalRule.")

        self.closed_world = bool(closed_world)
        self.rule_by_id: dict[str, CausalRule] = {}
        self.rules_by_effect: dict[str, list[CausalRule]] = {}
        event_ids: set[str] = set()

        for rule in self.rules:
            if not isinstance(rule, CausalRule):
                raise TypeError("LegalSCM chỉ nhận các CausalRule.")
            if validate:
                rule.validate()
            if rule.rule_id in self.rule_by_id:
                raise ValueError(f"Trùng rule_id trong LegalSCM: {rule.rule_id}")

            self.rule_by_id[rule.rule_id] = rule
            self.rules_by_effect.setdefault(
                rule.effect.event_id,
                [],
            ).append(rule)
            event_ids.add(rule.effect.event_id)
            event_ids.update(item.event_id for item in rule.conditions)
            event_ids.update(item.event_id for item in rule.exceptions)

        self.event_ids = frozenset(event_ids)
        self.endogenous_event_ids = frozenset(self.rules_by_effect)

    @classmethod
    def from_legacy_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        validate: bool = True,
        closed_world: bool = True,
    ) -> LegalSCM:
        rules = (
            CausalRule.from_legacy_rule(record, validate=validate)
            for record in records
        )
        return cls(
            rules,
            validate=validate,
            closed_world=closed_world,
        )

    @staticmethod
    def _normalize_assignments(
        assignments: Mapping[str, AssignmentValue] | None,
        *,
        label: str,
        allow_unknown: bool,
    ) -> dict[str, EventState]:
        if assignments is None:
            return {}
        if not isinstance(assignments, Mapping):
            raise TypeError(f"{label} phải là một mapping event_id -> state.")

        normalized: dict[str, EventState] = {}
        for raw_event_id, raw_state in assignments.items():
            event_id = str(raw_event_id).strip()
            if not event_id:
                raise ValueError(f"{label} chứa event_id rỗng.")
            state = EventState.from_value(raw_state)
            if not allow_unknown and state is EventState.UNKNOWN:
                raise ValueError(f"{label}[{event_id!r}] không thể là UNKNOWN.")
            if event_id in normalized and normalized[event_id] is not state:
                raise ValueError(
                    f"{label} chứa assignment xung đột cho {event_id}."
                )
            normalized[event_id] = state
        return normalized

    @staticmethod
    def _copy_supports(supports: SupportMap) -> SupportMap:
        return {
            event_id: set(states)
            for event_id, states in supports.items()
        }

    @staticmethod
    def _signature(supports: SupportMap) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (
                event_id,
                tuple(sorted(state.value for state in states)),
            )
            for event_id, states in sorted(supports.items())
            if states
        )

    @staticmethod
    def _literal_is_supported(
        literal: CausalLiteral,
        supports: SupportMap,
    ) -> bool:
        """Match only a settled state; conflicting support satisfies neither."""

        effective_state = LegalSCM._effective_state(
            supports.get(literal.event_id, set())
        )
        return effective_state is literal.state

    @staticmethod
    def _effective_state(states: set[EventState]) -> EventState:
        concrete = {
            state for state in states if state is not EventState.UNKNOWN
        }
        if len(concrete) == 1:
            return next(iter(concrete))
        return EventState.UNKNOWN

    def _base_supports(
        self,
        context: Mapping[str, EventState],
        interventions: Mapping[str, EventState],
    ) -> SupportMap:
        all_event_ids = self.event_ids | context.keys() | interventions.keys()
        supports: SupportMap = {event_id: set() for event_id in all_event_ids}

        for event_id, state in context.items():
            if event_id in interventions or state is EventState.UNKNOWN:
                continue
            supports[event_id].add(state)

        for event_id, state in interventions.items():
            supports[event_id] = {state}

        return supports

    def _evaluate_rules(
        self,
        supports: SupportMap,
        interventions: Mapping[str, EventState],
    ) -> tuple[tuple[RuleEvaluation, ...], tuple[CausalRule, ...]]:
        evaluations: list[RuleEvaluation] = []
        activated: list[CausalRule] = []

        for rule in self.rules:
            if rule.effect.event_id in interventions:
                evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.rule_id,
                        status=(
                            RuleEvaluationStatus.BLOCKED_BY_INTERVENTION
                        ),
                        effect=rule.effect,
                    )
                )
                continue

            unsatisfied = tuple(
                literal
                for literal in rule.conditions
                if not self._literal_is_supported(literal, supports)
            )
            if unsatisfied:
                evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.rule_id,
                        status=RuleEvaluationStatus.UNSATISFIED,
                        effect=rule.effect,
                        unsatisfied_conditions=unsatisfied,
                    )
                )
                continue

            matched_exceptions = tuple(
                literal
                for literal in rule.exceptions
                if self._literal_is_supported(literal, supports)
            )
            if matched_exceptions:
                evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.rule_id,
                        status=(
                            RuleEvaluationStatus.BLOCKED_BY_EXCEPTION
                        ),
                        effect=rule.effect,
                        matched_exceptions=matched_exceptions,
                    )
                )
                continue

            activated.append(rule)
            evaluations.append(
                RuleEvaluation(
                    rule_id=rule.rule_id,
                    status=RuleEvaluationStatus.ACTIVATED,
                    effect=rule.effect,
                )
            )

        return tuple(evaluations), tuple(activated)

    def _derive_next_supports(
        self,
        base_supports: SupportMap,
        activated_rules: Iterable[CausalRule],
        interventions: Mapping[str, EventState],
    ) -> SupportMap:
        supports = self._copy_supports(base_supports)
        derived_events: set[str] = set()

        for rule in activated_rules:
            effect = rule.effect
            supports.setdefault(effect.event_id, set()).add(effect.state)
            derived_events.add(effect.event_id)

        if self.closed_world:
            for event_id in self.endogenous_event_ids:
                if event_id in interventions:
                    continue
                event_supports = supports.setdefault(event_id, set())
                if not event_supports and event_id not in derived_events:
                    event_supports.add(EventState.FALSE)

        return supports

    def infer(
        self,
        context: Mapping[str, AssignmentValue] | None = None,
        *,
        interventions: Mapping[str, AssignmentValue] | None = None,
        max_iterations: int | None = None,
        raise_on_non_convergence: bool = True,
    ) -> InferenceResult:
        """Evaluate one world until its structural assignments stabilize."""

        normalized_context = self._normalize_assignments(
            context,
            label="context",
            allow_unknown=True,
        )
        normalized_interventions = self._normalize_assignments(
            interventions,
            label="interventions",
            allow_unknown=False,
        )

        if max_iterations is None:
            max_iterations = max(2, len(self.event_ids) + 1)
        if max_iterations < 1:
            raise ValueError("max_iterations phải lớn hơn 0.")

        base_supports = self._base_supports(
            normalized_context,
            normalized_interventions,
        )
        current = self._copy_supports(base_supports)
        seen_signatures = {self._signature(current)}
        activation_starts: dict[str, int] = {}
        previously_active_rule_ids: set[str] = set()
        converged = False
        iterations = 0

        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            _, activated = self._evaluate_rules(
                current,
                normalized_interventions,
            )
            active_rule_ids = {rule.rule_id for rule in activated}
            for rule_id in active_rule_ids - previously_active_rule_ids:
                activation_starts[rule_id] = iteration
            for rule_id in previously_active_rule_ids - active_rule_ids:
                activation_starts.pop(rule_id, None)
            previously_active_rule_ids = active_rule_ids

            next_supports = self._derive_next_supports(
                base_supports,
                activated,
                normalized_interventions,
            )
            next_signature = self._signature(next_supports)
            current_signature = self._signature(current)

            if next_signature == current_signature:
                current = next_supports
                converged = True
                break

            if next_signature in seen_signatures:
                current = next_supports
                break

            seen_signatures.add(next_signature)
            current = next_supports

        final_evaluations, final_activated = self._evaluate_rules(
            current,
            normalized_interventions,
        )

        if not converged and raise_on_non_convergence:
            raise RuntimeError(
                "LegalSCM không hội tụ; có thể tồn tại chu kỳ chứa phủ định "
                "hoặc exception. Hãy kiểm tra rule graph."
            )

        final_active_ids = {rule.rule_id for rule in final_activated}
        supporting_rules: dict[str, list[str]] = {}
        for rule in final_activated:
            supporting_rules.setdefault(rule.effect.event_id, []).append(
                rule.rule_id
            )

        all_event_ids = (
            self.event_ids
            | normalized_context.keys()
            | normalized_interventions.keys()
        )
        states = {
            event_id: self._effective_state(current.get(event_id, set()))
            for event_id in sorted(all_event_ids)
        }
        conflicts = tuple(
            event_id
            for event_id in sorted(all_event_ids)
            if len(
                {
                    state
                    for state in current.get(event_id, set())
                    if state is not EventState.UNKNOWN
                }
            )
            > 1
        )

        evaluation_by_rule = {
            evaluation.rule_id: evaluation
            for evaluation in final_evaluations
        }
        blocked_ids = tuple(
            rule.rule_id
            for rule in self.rules
            if evaluation_by_rule[rule.rule_id].status
            in {
                RuleEvaluationStatus.BLOCKED_BY_EXCEPTION,
                RuleEvaluationStatus.BLOCKED_BY_INTERVENTION,
            }
        )
        pending_ids = tuple(
            rule.rule_id
            for rule in self.rules
            if evaluation_by_rule[rule.rule_id].status
            is RuleEvaluationStatus.UNSATISFIED
        )
        activated_ids = tuple(
            rule.rule_id
            for rule in self.rules
            if rule.rule_id in final_active_ids
        )
        rule_order = {
            rule.rule_id: position
            for position, rule in enumerate(self.rules)
        }
        stable_rules = sorted(
            (
                rule
                for rule in self.rules
                if rule.rule_id in final_active_ids
            ),
            key=lambda rule: (
                activation_starts.get(rule.rule_id, iterations),
                rule_order[rule.rule_id],
            ),
        )
        trace = tuple(
            InferenceTraceStep(
                iteration=activation_starts.get(rule.rule_id, iterations),
                rule_id=rule.rule_id,
                article_id=rule.article_id,
                conditions=rule.conditions,
                effect=rule.effect,
            )
            for rule in stable_rules
        )

        return InferenceResult(
            states=states,
            supports={
                event_id: tuple(
                    sorted(
                        current.get(event_id, set()),
                        key=lambda state: state.value,
                    )
                )
                for event_id in sorted(all_event_ids)
            },
            context=normalized_context,
            interventions=normalized_interventions,
            activated_rule_ids=activated_ids,
            blocked_rule_ids=blocked_ids,
            pending_rule_ids=pending_ids,
            rule_evaluations=final_evaluations,
            trace=trace,
            supporting_rule_ids={
                event_id: tuple(rule_ids)
                for event_id, rule_ids in supporting_rules.items()
            },
            conflicting_event_ids=conflicts,
            iterations=iterations,
            converged=converged,
            closed_world=self.closed_world,
        )

    @staticmethod
    def _classify_counterfactual(
        factual: EventState,
        counterfactual: EventState,
        interventions: Mapping[str, EventState],
        factual_world: InferenceResult,
    ) -> tuple[CounterfactualStatus, bool | None]:
        treatment_states = {
            event_id: factual_world.state_of(event_id)
            for event_id in interventions
        }
        if (
            factual is EventState.UNKNOWN
            or counterfactual is EventState.UNKNOWN
            or any(
                state is EventState.UNKNOWN
                for state in treatment_states.values()
            )
        ):
            return CounterfactualStatus.INDETERMINATE, None

        changed = factual is not counterfactual
        removal = bool(interventions) and all(
            state is EventState.FALSE for state in interventions.values()
        )
        activation = bool(interventions) and all(
            state is EventState.TRUE for state in interventions.values()
        )
        factual_treatments_present = all(
            state is EventState.TRUE for state in treatment_states.values()
        )
        factual_treatments_absent = all(
            state is EventState.FALSE for state in treatment_states.values()
        )

        if (
            removal
            and factual_treatments_present
            and factual is EventState.TRUE
        ):
            if counterfactual is EventState.FALSE:
                return CounterfactualStatus.NECESSARY, True
            return CounterfactualStatus.NON_NECESSARY, False

        if (
            activation
            and factual_treatments_absent
            and factual is EventState.FALSE
            and counterfactual is EventState.TRUE
        ):
            return CounterfactualStatus.SUFFICIENT, True

        if not changed:
            return CounterfactualStatus.NO_EFFECT, False
        return CounterfactualStatus.OUTCOME_CHANGED, True

    def counterfactual(
        self,
        context: Mapping[str, AssignmentValue] | None,
        *,
        interventions: Mapping[str, AssignmentValue],
        outcome_event_id: str,
        recompute_outcome: bool = True,
        recompute_endogenous_context: bool = True,
        max_iterations: int | None = None,
    ) -> CounterfactualResult:
        """Compare factual and intervened worlds from the same root context.

        By default, concrete assignments for endogenous variables are removed
        before both worlds are evaluated, because this engine has no latent
        variable abduction step that could reconcile observed intermediates.
        The removed IDs are returned in ``recomputed_context_event_ids``.
        ``recompute_outcome`` retains the narrower legacy control when full
        endogenous recomputation is explicitly disabled.
        """

        outcome_event_id = str(outcome_event_id).strip()
        if not outcome_event_id:
            raise ValueError("outcome_event_id không được để trống.")
        if outcome_event_id not in self.event_ids:
            raise ValueError(
                f"Outcome event không tồn tại trong LegalSCM: {outcome_event_id}"
            )

        normalized_context = self._normalize_assignments(
            context,
            label="context",
            allow_unknown=True,
        )
        normalized_interventions = self._normalize_assignments(
            interventions,
            label="interventions",
            allow_unknown=False,
        )
        if not normalized_interventions:
            raise ValueError("Counterfactual cần ít nhất một intervention.")
        if outcome_event_id in normalized_interventions:
            raise ValueError(
                "Không được can thiệp trực tiếp lên outcome; thao tác đó "
                "không chứng minh necessity hoặc sufficiency của nguyên nhân."
            )

        factual_context = dict(normalized_context)
        recomputed_context_event_ids: tuple[str, ...] = ()
        if recompute_endogenous_context:
            recomputed_context_event_ids = tuple(
                sorted(
                    event_id
                    for event_id in factual_context
                    if event_id in self.endogenous_event_ids
                )
            )
            for event_id in recomputed_context_event_ids:
                factual_context.pop(event_id, None)
        elif recompute_outcome:
            factual_context.pop(outcome_event_id, None)

        factual = self.infer(
            factual_context,
            max_iterations=max_iterations,
        )
        counterfactual = self.infer(
            factual_context,
            interventions=normalized_interventions,
            max_iterations=max_iterations,
        )

        factual_outcome = factual.state_of(outcome_event_id)
        counterfactual_outcome = counterfactual.state_of(
            outcome_event_id
        )
        status, changed = self._classify_counterfactual(
            factual_outcome,
            counterfactual_outcome,
            normalized_interventions,
            factual,
        )

        factual_active = set(factual.activated_rule_ids)
        counterfactual_active = set(counterfactual.activated_rule_ids)
        alternative_rules = tuple(
            rule.rule_id
            for rule in self.rules
            if (
                rule.rule_id in counterfactual_active
                and rule.effect.event_id == outcome_event_id
                and rule.effect.state is counterfactual_outcome
            )
        )

        return CounterfactualResult(
            outcome_event_id=outcome_event_id,
            interventions=normalized_interventions,
            factual_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            outcome_changed=changed,
            status=status,
            factual=factual,
            counterfactual=counterfactual,
            disabled_rule_ids=tuple(
                rule.rule_id
                for rule in self.rules
                if rule.rule_id in factual_active - counterfactual_active
            ),
            newly_activated_rule_ids=tuple(
                rule.rule_id
                for rule in self.rules
                if rule.rule_id in counterfactual_active - factual_active
            ),
            alternative_outcome_rule_ids=alternative_rules,
            recomputed_context_event_ids=(
                recomputed_context_event_ids
            ),
        )


__all__ = [
    "CounterfactualResult",
    "CounterfactualStatus",
    "InferenceResult",
    "InferenceTraceStep",
    "Intervention",
    "LegalSCM",
    "RuleEvaluation",
    "RuleEvaluationStatus",
]
