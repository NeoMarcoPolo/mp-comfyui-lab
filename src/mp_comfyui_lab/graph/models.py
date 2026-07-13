from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from math import isfinite
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    field_validator,
)

from mp_comfyui_lab.json_types import (
    FrozenJsonObject,
    freeze_json_object,
    thaw_json,
)


NonBlankStr = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]

_STRENGTH_LIMITS = (
    (0.25, "subtle"),
    (0.75, "light"),
    (1.25, "standard"),
    (2.0, "strong"),
)
MAX_SERIALIZABLE_JSON_DEPTH = 48
MAX_SERIALIZABLE_JSON_VALUES = 100_000


def strength_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("strength must be a finite JSON number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("strength must be a finite JSON number")
    if value == 0:
        return "zero"
    sign = "negative" if value < 0 else "positive"
    magnitude = abs(value)
    for upper_bound, label in _STRENGTH_LIMITS:
        if magnitude <= upper_bound:
            return f"{sign}_{label}"
    return f"{sign}_extreme"


def _validate_node_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("node ID must be a nonblank string")
    if value != value.strip():
        raise ValueError("node ID must not have leading or trailing whitespace")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("node ID must not contain control or surrogate characters")
    return value


def validate_serializable_json(
    value: Any,
    path: str,
    *,
    max_depth: int = MAX_SERIALIZABLE_JSON_DEPTH,
    max_values: int = MAX_SERIALIZABLE_JSON_VALUES,
) -> Any:
    stack = [(value, path, 0)]
    value_count = 0
    while stack:
        item, item_path, depth = stack.pop()
        value_count += 1
        if value_count > max_values:
            raise ValueError(f"JSON value limit of {max_values} exceeded")
        if depth > max_depth:
            raise ValueError(f"JSON depth limit of {max_depth} exceeded")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, str):
            if any(0xD800 <= ord(character) <= 0xDFFF for character in item):
                raise ValueError(f"{item_path} contains a lone UTF-16 surrogate")
            continue
        if isinstance(item, float):
            if not isfinite(item):
                raise ValueError(f"{item_path} JSON numbers must be finite")
            continue
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{item_path} object keys must be strings")
                if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                    raise ValueError(
                        f"{item_path} object key contains a lone UTF-16 surrogate"
                    )
                stack.append((nested, f"{item_path}.{key}", depth + 1))
            continue
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                stack.append((nested, f"{item_path}[{index}]", depth + 1))
            continue
        raise ValueError(
            f"{item_path} has unsupported JSON value type {type(item).__name__}"
        )
    return value


ExactNodeId = Annotated[str, BeforeValidator(_validate_node_id)]


def _freeze_json_value(value: Any) -> Any:
    return freeze_json_object({"value": value})["value"]


FrozenJsonValue = Annotated[
    Any,
    BeforeValidator(_freeze_json_value),
    PlainSerializer(thaw_json, when_used="json"),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class SemanticRole(StrEnum):
    CHECKPOINT = "checkpoint"
    MODEL = "model"
    UNET = "unet"
    LORA = "lora"
    PROMPT_SOURCE = "prompt_source"
    PROMPT_FANOUT = "prompt_fanout"
    REFERENCE_IMAGE = "reference_image"
    SEED = "seed"
    SAMPLER = "sampler"
    SCHEDULER = "scheduler"
    DIMENSIONS = "dimensions"
    VAE = "vae"
    OUTPUT = "output"
    CONDITIONING = "conditioning"
    ENCODER = "encoder"


class NormalizedNode(_FrozenModel):
    node_id: ExactNodeId
    class_type: NonBlankStr
    inputs: FrozenJsonObject
    input_links: FrozenJsonObject

    @field_validator("inputs", mode="before")
    @classmethod
    def validate_inputs_before_freezing(cls, value: Any) -> Any:
        return validate_serializable_json(value, "normalized node inputs")

    @field_validator("input_links", mode="before")
    @classmethod
    def validate_input_links_before_freezing(cls, value: Any) -> Any:
        return validate_serializable_json(value, "normalized node input links")

    @field_validator("input_links")
    @classmethod
    def validate_input_links(cls, links: FrozenJsonObject) -> FrozenJsonObject:
        for field_name, link in links.items():
            if not field_name.strip():
                raise ValueError("input link field names must be nonblank")
            if (
                not isinstance(link, tuple)
                or len(link) != 2
                or not isinstance(link[1], int)
                or isinstance(link[1], bool)
                or link[1] < 0
            ):
                raise ValueError(
                    "input links must contain [nonblank node ID, nonnegative output index]"
                )
            _validate_node_id(link[0])
        return links


class RoleBinding(_FrozenModel):
    node_id: ExactNodeId
    field_name: NonBlankStr
    value: FrozenJsonValue
    confidence: float = Field(ge=0.0, le=1.0)
    reason: NonBlankStr

    @field_validator("value", mode="before")
    @classmethod
    def validate_value_before_freezing(cls, value: Any) -> Any:
        return validate_serializable_json(value, "role binding value")


class RoleBindings(_FrozenModel):
    role: SemanticRole
    bindings: tuple[RoleBinding, ...] = ()


def _node_sort_key(node_id: str) -> tuple[int, int, str, str]:
    if node_id.isdecimal():
        significant = node_id.lstrip("0") or "0"
        return (0, len(significant), significant, node_id)
    return (1, 0, node_id, node_id)


def _binding_description(binding: RoleBinding) -> dict[str, Any]:
    return {
        "confidence": binding.confidence,
        "field_name": binding.field_name,
        "node_id": binding.node_id,
        "reason": binding.reason,
        "value": thaw_json(binding.value),
    }


class RoleResolutionError(LookupError):
    def __init__(
        self,
        role: SemanticRole,
        bindings: tuple[RoleBinding, ...],
        message: str,
    ) -> None:
        self.role = role
        self.bindings = bindings
        candidates = json.dumps(
            [_binding_description(binding) for binding in bindings],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        super().__init__(f"{message} for role {role.value}; bindings={candidates}")


class MissingRoleError(RoleResolutionError):
    def __init__(
        self,
        role: SemanticRole,
        bindings: tuple[RoleBinding, ...],
    ) -> None:
        super().__init__(role, bindings, "no binding has confidence above 0.8")


class AmbiguousRoleError(RoleResolutionError):
    def __init__(
        self,
        role: SemanticRole,
        bindings: tuple[RoleBinding, ...],
    ) -> None:
        super().__init__(role, bindings, "multiple bindings have confidence above 0.8")


class NormalizedGraph(_FrozenModel):
    nodes: tuple[NormalizedNode, ...]
    role_bindings: tuple[RoleBindings, ...] = ()

    @field_validator("nodes")
    @classmethod
    def sort_nodes(
        cls,
        nodes: tuple[NormalizedNode, ...],
    ) -> tuple[NormalizedNode, ...]:
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("normalized node IDs must be unique")
        return tuple(sorted(nodes, key=lambda node: _node_sort_key(node.node_id)))

    @field_validator("role_bindings")
    @classmethod
    def sort_role_groups(
        cls,
        groups: tuple[RoleBindings, ...],
    ) -> tuple[RoleBindings, ...]:
        roles = [group.role for group in groups]
        if len(roles) != len(set(roles)):
            raise ValueError("semantic roles must have at most one binding group")
        return tuple(
            sorted(groups, key=lambda group: list(SemanticRole).index(group.role))
        )

    def bindings(self, role: SemanticRole) -> tuple[RoleBinding, ...]:
        if not isinstance(role, SemanticRole):
            raise TypeError("role must be a SemanticRole")
        for group in self.role_bindings:
            if group.role is role:
                return group.bindings
        return ()

    def require_one(self, role: SemanticRole) -> RoleBinding:
        bindings = self.bindings(role)
        confident = tuple(binding for binding in bindings if binding.confidence > 0.8)
        if len(confident) == 1:
            return confident[0]
        if len(confident) > 1:
            raise AmbiguousRoleError(role, bindings)
        raise MissingRoleError(role, bindings)
