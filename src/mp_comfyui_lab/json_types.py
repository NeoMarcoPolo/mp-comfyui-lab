from __future__ import annotations

from collections.abc import Iterator, Mapping
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import BeforeValidator, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from mp_comfyui_lab.security import (
    sanitize_for_persistence,
    sanitize_planning_for_persistence,
)


class FrozenDict(Mapping[str, Any]):
    """A recursively immutable, JSON-serializable object mapping."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        source = {} if values is None else values
        if not isinstance(source, Mapping):
            raise ValueError("JSON object must be a mapping")

        frozen: dict[str, Any] = {}
        for key, value in source.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            frozen[key] = _freeze_json_value(value)
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("FrozenDict attributes cannot be changed")

    def __delattr__(self, name: str) -> None:
        raise TypeError("FrozenDict attributes cannot be deleted")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.no_info_plain_validator_function(
            freeze_json_object,
            json_schema_input_schema=core_schema.dict_schema(
                keys_schema=core_schema.str_schema(strict=True),
                values_schema=core_schema.any_schema(),
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                thaw_json,
                return_schema=core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(),
                    values_schema=core_schema.any_schema(),
                ),
                when_used="json",
            ),
        )


def _freeze_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def freeze_json_object(value: Any) -> FrozenDict:
    if isinstance(value, FrozenDict):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("JSON object must be a mapping")
    return FrozenDict(value)


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def _sanitize_json_object(value: Any) -> Any:
    validated = freeze_json_object(value)
    return sanitize_for_persistence(thaw_json(validated))


def _sanitize_planning_json_object(value: Any) -> Any:
    validated = freeze_json_object(value)
    return sanitize_planning_for_persistence(thaw_json(validated))


FrozenJsonObject = FrozenDict
SanitizedJsonObject = Annotated[
    FrozenDict,
    BeforeValidator(_sanitize_json_object),
]
PlanningSanitizedJsonObject = Annotated[
    FrozenDict,
    BeforeValidator(_sanitize_planning_json_object),
]
