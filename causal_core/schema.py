"""Core schemas for legal causal rules and events.

The current pipeline exchanges flat dictionaries between scripts.  This
module deliberately keeps that format available through ``LegacyLegalRule``
while providing a canonical ``CausalRule`` representation for later
structural inference.  Importing this module does not change the existing
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from typing import Any


LEGACY_RULE_FIELDS: tuple[str, ...] = (
    "index",
    "article_id",
    "legal_subject",
    "condition",
    "effect",
    "condition_event",
    "condition_event_name",
    "effect_event",
    "effect_event_name",
    "rule_text",
    "article_title",
    "content",
    "quality_status",
    "source_scope",
    "condition_event_original",
    "effect_event_original",
    "condition_event_modality",
    "effect_event_modality",
    "event_normalization_version",
    "causal_type",
    "normalization_metadata",
)

LEGAL_EVENT_FIELDS: tuple[str, ...] = (
    "event_id",
    "event_name",
    "event_type",
    "description",
    "aliases",
    "source_article_ids",
    "occurrences",
    "status",
)


class EventState(str, Enum):
    """Three-valued state used by factual and counterfactual worlds."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> EventState:
        if isinstance(value, cls):
            return value
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE
        if value is None:
            return cls.UNKNOWN

        normalized = str(value).strip().lower()
        aliases = {
            "true": cls.TRUE,
            "1": cls.TRUE,
            "yes": cls.TRUE,
            "present": cls.TRUE,
            "false": cls.FALSE,
            "0": cls.FALSE,
            "no": cls.FALSE,
            "absent": cls.FALSE,
            "unknown": cls.UNKNOWN,
            "none": cls.UNKNOWN,
            "": cls.UNKNOWN,
        }
        if normalized not in aliases:
            raise ValueError(f"Trạng thái sự kiện không hợp lệ: {value!r}")
        return aliases[normalized]


class EventType(str, Enum):
    """Event types currently produced by ``a2_normalize_events_fixed.py``."""

    ACTION = "ACTION"
    STATE = "STATE"
    LEGAL_CONDITION = "LEGAL_CONDITION"
    LEGAL_CONSEQUENCE = "LEGAL_CONSEQUENCE"
    SANCTION = "SANCTION"
    OBLIGATION = "OBLIGATION"
    PERMISSION = "PERMISSION"
    PROHIBITION = "PROHIBITION"
    EXEMPTION = "EXEMPTION"
    CLASSIFICATION = "CLASSIFICATION"
    SCOPE_APPLICATION = "SCOPE_APPLICATION"
    PROCEDURE = "PROCEDURE"
    OTHER = "OTHER"


class NormalizationAction(str, Enum):
    USE_EXISTING = "USE_EXISTING"
    CREATE_NEW = "CREATE_NEW"


def _field_default(instance: Any, field_name: str) -> Any:
    for item in fields(instance):
        if item.name != field_name:
            continue
        if item.default is not MISSING:
            return item.default
        if item.default_factory is not MISSING:
            return item.default_factory()
        return MISSING
    raise KeyError(field_name)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_scalar_id(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(
            f"{name} phải là int, str hoặc None; nhận {type(value).__name__}."
        )


def _serialize_adapter(
    instance: Any,
    known_fields: tuple[str, ...],
    *,
    preserve_missing: bool,
    preserve_null: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    for name in known_fields:
        value = getattr(instance, name)
        default = _field_default(instance, name)
        should_emit = (
            not preserve_missing
            or name in instance._present_fields
            or value != default
        )
        if not should_emit:
            continue
        if value is None and not preserve_null:
            continue
        if isinstance(value, Enum):
            value = value.value
        payload[name] = deepcopy(value)

    for name, value in instance._extra.items():
        if value is None and not preserve_null:
            continue
        payload[name] = deepcopy(value)

    return payload


@dataclass
class LegacyLegalRule:
    """Lossless adapter for rule dictionaries used by the existing scripts.

    Missing keys, explicit ``None`` values, and unknown future fields are
    tracked separately so loading and serializing a record does not silently
    alter its contract.
    """

    index: int | str | None = None
    article_id: int | str | None = None
    legal_subject: str | None = ""
    condition: str | None = ""
    effect: str | None = ""
    condition_event: str | None = ""
    condition_event_name: str | None = ""
    effect_event: str | None = ""
    effect_event_name: str | None = ""
    rule_text: str | None = ""
    article_title: str | None = ""
    content: str | None = ""
    quality_status: str | None = ""
    source_scope: str | None = ""
    condition_event_original: str | None = ""
    effect_event_original: str | None = ""
    condition_event_modality: str | None = ""
    effect_event_modality: str | None = ""
    event_normalization_version: str | None = ""
    causal_type: str | None = ""
    normalization_metadata: dict[str, Any] | None = None
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)
    _present_fields: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_legacy_dict(
        cls,
        data: Mapping[str, Any],
        *,
        validate: bool = False,
        require_normalized: bool = False,
    ) -> LegacyLegalRule:
        if not isinstance(data, Mapping):
            raise TypeError("Rule đầu vào phải là một mapping.")

        known = set(LEGACY_RULE_FIELDS)
        values = {
            name: deepcopy(data[name])
            for name in LEGACY_RULE_FIELDS
            if name in data
        }
        rule = cls(**values)
        rule._present_fields = set(data).intersection(known)
        rule._extra = {
            str(name): deepcopy(value)
            for name, value in data.items()
            if name not in known
        }

        if validate:
            rule.validate(require_normalized=require_normalized)
        return rule

    def validate(
        self,
        *,
        require_normalized: bool = False,
        allow_self_loop: bool = False,
    ) -> None:
        errors: list[str] = []

        try:
            _validate_scalar_id("index", self.index)
            _validate_scalar_id("article_id", self.article_id)
        except ValueError as error:
            errors.append(str(error))

        if not _non_empty_string(self.condition):
            errors.append("condition phải là chuỗi không rỗng.")
        if not _non_empty_string(self.effect):
            errors.append("effect phải là chuỗi không rỗng.")

        optional_strings = (
            "legal_subject",
            "condition_event",
            "condition_event_name",
            "effect_event",
            "effect_event_name",
            "rule_text",
            "article_title",
            "content",
            "quality_status",
            "source_scope",
            "condition_event_original",
            "effect_event_original",
            "condition_event_modality",
            "effect_event_modality",
            "event_normalization_version",
            "causal_type",
        )
        for name in optional_strings:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                errors.append(f"{name} phải là str hoặc None.")

        if require_normalized:
            if not _non_empty_string(self.condition_event):
                errors.append("condition_event là bắt buộc với normalized rule.")
            if not _non_empty_string(self.effect_event):
                errors.append("effect_event là bắt buộc với normalized rule.")

        if (
            not allow_self_loop
            and _non_empty_string(self.condition_event)
            and self.condition_event == self.effect_event
        ):
            errors.append("condition_event trùng effect_event (causal self-loop).")

        if (
            self.normalization_metadata is not None
            and not isinstance(self.normalization_metadata, Mapping)
        ):
            errors.append("normalization_metadata phải là mapping hoặc None.")

        if errors:
            identity = self.index if self.index is not None else "unknown"
            raise ValueError(f"Rule {identity} không hợp lệ: " + " ".join(errors))

    def to_legacy_dict(
        self,
        *,
        preserve_missing: bool = True,
        preserve_null: bool = True,
    ) -> dict[str, Any]:
        return _serialize_adapter(
            self,
            LEGACY_RULE_FIELDS,
            preserve_missing=preserve_missing,
            preserve_null=preserve_null,
        )

    def to_dict(
        self,
        *,
        preserve_missing: bool = True,
        preserve_null: bool = True,
    ) -> dict[str, Any]:
        return self.to_legacy_dict(
            preserve_missing=preserve_missing,
            preserve_null=preserve_null,
        )

    def to_causal_rule(self, *, validate: bool = True) -> CausalRule:
        return CausalRule.from_legacy_rule(self, validate=validate)


@dataclass
class LegalEvent:
    """Lossless adapter for records in the event catalog."""

    event_id: str | None = ""
    event_name: str | None = ""
    event_type: str | None = EventType.OTHER.value
    description: str | None = ""
    aliases: list[str] | None = field(default_factory=list)
    source_article_ids: list[int] | None = field(default_factory=list)
    occurrences: list[dict[str, Any]] | None = field(default_factory=list)
    status: str | None = "CANDIDATE"
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)
    _present_fields: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def from_legacy_dict(
        cls,
        data: Mapping[str, Any],
        *,
        validate: bool = False,
    ) -> LegalEvent:
        if not isinstance(data, Mapping):
            raise TypeError("Event đầu vào phải là một mapping.")

        known = set(LEGAL_EVENT_FIELDS)
        values = {
            name: deepcopy(data[name])
            for name in LEGAL_EVENT_FIELDS
            if name in data
        }
        event = cls(**values)
        event._present_fields = set(data).intersection(known)
        event._extra = {
            str(name): deepcopy(value)
            for name, value in data.items()
            if name not in known
        }

        if validate:
            event.validate()
        return event

    @property
    def event_type_enum(self) -> EventType:
        try:
            return EventType(str(self.event_type).strip().upper())
        except ValueError:
            return EventType.OTHER

    def validate(self) -> None:
        errors: list[str] = []
        if not _non_empty_string(self.event_id):
            errors.append("event_id phải là chuỗi không rỗng.")
        if not _non_empty_string(self.event_name):
            errors.append("event_name phải là chuỗi không rỗng.")

        if self.event_type is not None:
            try:
                EventType(str(self.event_type).strip().upper())
            except ValueError:
                errors.append(f"event_type không hợp lệ: {self.event_type!r}.")

        if self.aliases is not None and (
            not isinstance(self.aliases, list)
            or any(not isinstance(value, str) for value in self.aliases)
        ):
            errors.append("aliases phải là list[str] hoặc None.")

        if self.source_article_ids is not None:
            if not isinstance(self.source_article_ids, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.source_article_ids
            ):
                errors.append("source_article_ids phải là list[int] hoặc None.")

        if self.occurrences is not None:
            if not isinstance(self.occurrences, list) or any(
                not isinstance(value, Mapping) for value in self.occurrences
            ):
                errors.append("occurrences phải là list[mapping] hoặc None.")
            else:
                for occurrence in self.occurrences:
                    role = occurrence.get("role")
                    if role is not None and role not in {"condition", "effect"}:
                        errors.append(f"Occurrence role không hợp lệ: {role!r}.")

        if errors:
            raise ValueError("LegalEvent không hợp lệ: " + " ".join(errors))

    def to_dict(
        self,
        *,
        preserve_missing: bool = True,
        preserve_null: bool = True,
    ) -> dict[str, Any]:
        return _serialize_adapter(
            self,
            LEGAL_EVENT_FIELDS,
            preserve_missing=preserve_missing,
            preserve_null=preserve_null,
        )


@dataclass(frozen=True)
class CausalLiteral:
    """An event assignment used as a condition, exception, or effect."""

    event_id: str
    state: EventState = EventState.TRUE

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("CausalLiteral.event_id không được để trống.")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "state", EventState.from_value(self.state))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CausalLiteral:
        if not isinstance(data, Mapping):
            raise TypeError("Causal literal phải là một mapping.")
        state = data.get("state", data.get("value", EventState.TRUE))
        return cls(event_id=str(data.get("event_id", "")), state=state)

    def to_dict(self) -> dict[str, str]:
        return {"event_id": self.event_id, "state": self.state.value}


@dataclass(frozen=True)
class CausalRule:
    """Canonical rule mechanism used by the future causal inference engine.

    A rule fires when all ``conditions`` match and none of its ``exceptions``
    match.  The existing dataset maps to one positive condition, no exception,
    and one positive effect without losing its legacy payload.
    """

    rule_id: str
    article_id: int | str | None
    legal_subject: str
    conditions: tuple[CausalLiteral, ...]
    exceptions: tuple[CausalLiteral, ...]
    effect: CausalLiteral
    condition_texts: tuple[str, ...] = ()
    effect_text: str = ""
    rule_text: str = ""
    article_title: str = ""
    condition_modalities: tuple[str, ...] = ()
    effect_modality: str = ""
    causal_type: str = ""
    quality_status: str = ""
    source_scope: str = ""
    event_normalization_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    _legacy_rule: LegacyLegalRule | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_legacy_rule(
        cls,
        rule: LegacyLegalRule | Mapping[str, Any],
        *,
        validate: bool = True,
    ) -> CausalRule:
        legacy = (
            rule
            if isinstance(rule, LegacyLegalRule)
            else LegacyLegalRule.from_legacy_dict(rule)
        )
        if validate:
            legacy.validate(require_normalized=True)

        rule_id = "" if legacy.index is None else str(legacy.index).strip()
        if not rule_id:
            raise ValueError("Normalized rule cần index để tạo rule_id.")
        if not _non_empty_string(legacy.condition_event):
            raise ValueError("Normalized rule thiếu condition_event.")
        if not _non_empty_string(legacy.effect_event):
            raise ValueError("Normalized rule thiếu effect_event.")

        result = cls(
            rule_id=rule_id,
            article_id=legacy.article_id,
            legal_subject=legacy.legal_subject or "",
            conditions=(
                CausalLiteral(legacy.condition_event, EventState.TRUE),
            ),
            exceptions=(),
            effect=CausalLiteral(legacy.effect_event, EventState.TRUE),
            condition_texts=(legacy.condition or "",),
            effect_text=legacy.effect or "",
            rule_text=legacy.rule_text or "",
            article_title=legacy.article_title or "",
            condition_modalities=(
                legacy.condition_event_modality or "",
            ),
            effect_modality=legacy.effect_event_modality or "",
            causal_type=legacy.causal_type or "",
            quality_status=legacy.quality_status or "",
            source_scope=legacy.source_scope or "",
            event_normalization_version=(
                legacy.event_normalization_version or ""
            ),
            metadata={
                "condition_event_name": legacy.condition_event_name or "",
                "effect_event_name": legacy.effect_event_name or "",
                "condition_event_original": (
                    legacy.condition_event_original
                    or legacy.condition_event
                ),
                "effect_event_original": (
                    legacy.effect_event_original or legacy.effect_event
                ),
                "normalization_metadata": deepcopy(
                    legacy.normalization_metadata
                ),
            },
            _legacy_rule=legacy,
        )
        if validate:
            result.validate()
        return result

    def validate(self, *, allow_self_loop: bool = False) -> None:
        errors: list[str] = []
        if not self.rule_id.strip():
            errors.append("rule_id không được để trống.")
        try:
            _validate_scalar_id("article_id", self.article_id)
        except ValueError as error:
            errors.append(str(error))
        if not self.conditions:
            errors.append("CausalRule phải có ít nhất một condition.")

        all_literals = (*self.conditions, *self.exceptions, self.effect)
        if any(item.state is EventState.UNKNOWN for item in all_literals):
            errors.append(
                "Rule mechanism không được dùng UNKNOWN làm trạng thái kỳ vọng."
            )

        condition_keys = {
            (item.event_id, item.state) for item in self.conditions
        }
        if len(condition_keys) != len(self.conditions):
            errors.append("CausalRule có condition trùng lặp.")

        exception_keys = {
            (item.event_id, item.state) for item in self.exceptions
        }
        if len(exception_keys) != len(self.exceptions):
            errors.append("CausalRule có exception trùng lặp.")

        if condition_keys.intersection(exception_keys):
            errors.append("Một literal không thể vừa là condition vừa là exception.")

        if (
            not allow_self_loop
            and any(
                item.event_id == self.effect.event_id
                and item.state == self.effect.state
                for item in self.conditions
            )
        ):
            errors.append("CausalRule tạo self-loop condition/effect.")

        if self.condition_texts and (
            len(self.condition_texts) != len(self.conditions)
        ):
            errors.append(
                "condition_texts phải rỗng hoặc cùng số phần tử với conditions."
            )
        if self.condition_modalities and (
            len(self.condition_modalities) != len(self.conditions)
        ):
            errors.append(
                "condition_modalities phải rỗng hoặc cùng số phần tử với conditions."
            )

        if errors:
            raise ValueError(
                f"CausalRule {self.rule_id} không hợp lệ: " + " ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "article_id": self.article_id,
            "legal_subject": self.legal_subject,
            "conditions": [item.to_dict() for item in self.conditions],
            "exceptions": [item.to_dict() for item in self.exceptions],
            "effect": self.effect.to_dict(),
            "condition_texts": list(self.condition_texts),
            "effect_text": self.effect_text,
            "rule_text": self.rule_text,
            "article_title": self.article_title,
            "condition_modalities": list(self.condition_modalities),
            "effect_modality": self.effect_modality,
            "causal_type": self.causal_type,
            "quality_status": self.quality_status,
            "source_scope": self.source_scope,
            "event_normalization_version": self.event_normalization_version,
            "metadata": deepcopy(self.metadata),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a flat legacy record when the rule can be represented losslessly."""

        if len(self.conditions) != 1 or self.exceptions:
            raise ValueError(
                "Rule nhiều condition/exception không thể chuyển về schema "
                "legacy một-condition mà không mất dữ liệu."
            )

        if self._legacy_rule is not None:
            payload = self._legacy_rule.to_legacy_dict()
        else:
            payload = {}

        payload.update(
            {
                "index": self.rule_id,
                "article_id": self.article_id,
                "legal_subject": self.legal_subject,
                "condition": (
                    self.condition_texts[0] if self.condition_texts else ""
                ),
                "effect": self.effect_text,
                "condition_event": self.conditions[0].event_id,
                "effect_event": self.effect.event_id,
                "rule_text": self.rule_text,
                "article_title": self.article_title,
                "causal_type": self.causal_type,
                "quality_status": self.quality_status,
                "source_scope": self.source_scope,
                "condition_event_modality": (
                    self.condition_modalities[0]
                    if self.condition_modalities
                    else ""
                ),
                "effect_event_modality": self.effect_modality,
                "event_normalization_version": (
                    self.event_normalization_version
                ),
            }
        )
        return payload


__all__ = [
    "CausalLiteral",
    "CausalRule",
    "EventState",
    "EventType",
    "LegalEvent",
    "LegacyLegalRule",
    "NormalizationAction",
]
