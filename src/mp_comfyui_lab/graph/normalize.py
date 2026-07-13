from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from math import isfinite
import re
from typing import Any

from pydantic import ValidationError

from mp_comfyui_lab.graph.models import (
    NormalizedGraph,
    NormalizedNode,
    RoleBinding,
    RoleBindings,
    SemanticRole,
    strength_bucket,
)
from mp_comfyui_lab.json_types import FrozenDict


class GraphNormalizationError(ValueError):
    pass


class GraphValidationError(GraphNormalizationError):
    pass


class GraphTopologyError(GraphNormalizationError):
    pass


_MODEL_NAME_FIELDS = (
    "ckpt_name",
    "checkpoint_name",
    "diffusion_model_name",
    "model_name",
    "unet_name",
)
_LORA_NAME_FIELDS = ("lora_name", "lora_file", "lora")
_MODEL_LINK_FIELDS = ("model", "unet")
_PROMPT_FIELDS = (
    "prompt",
    "text",
    "value",
    "promptLine",
    "positive_prompt",
    "negative_prompt",
)
_REFERENCE_FIELDS = ("image", "image_name", "filename", "path")
_SEED_FIELDS = ("noise_seed", "random_seed", "seed")
_DIMENSION_FIELDS = ("height", "width")
_VAE_NAME_FIELDS = ("vae_name", "model_name")
_ENCODER_NAME_FIELDS = ("clip_name", "encoder_name", "text_encoder_name")
_LITERAL_COLLECTION_FIELDS = frozenset(
    {
        "choices",
        "colors",
        "indices",
        "list",
        "ranges",
        "values",
    }
)
_POWER_LORA_SLOT = re.compile(r"^lora_(\d+)$", re.IGNORECASE)
_MAX_GRAPH_NODES = 4_096
_MAX_JSON_DEPTH = 256
_MAX_JSON_VALUES = 100_000


def _node_sort_key(node_id: str) -> tuple[int, int, str, str]:
    if node_id.isdecimal():
        significant = node_id.lstrip("0") or "0"
        return (0, len(significant), significant, node_id)
    return (1, 0, node_id, node_id)


def _compact_class(class_type: str) -> str:
    return "".join(
        character for character in class_type.casefold() if character.isalnum()
    )


def _canonical_field_name(field_name: str) -> str:
    return "".join(
        character for character in field_name.casefold() if character.isalnum()
    )


def _validate_unicode(value: str, path: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise GraphValidationError(f"{path} contains a lone UTF-16 surrogate")


def _validate_identifier(value: str, path: str) -> None:
    _validate_unicode(value, path)
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise GraphValidationError(f"{path} must not contain control characters")


def _validate_json(value: Any, path: str, *, allow_frozen: bool) -> None:
    stack = [(value, path, 0)]
    value_count = 0
    while stack:
        item, item_path, depth = stack.pop()
        value_count += 1
        if value_count > _MAX_JSON_VALUES:
            raise GraphValidationError(
                f"graph JSON value limit of {_MAX_JSON_VALUES} exceeded"
            )
        if depth > _MAX_JSON_DEPTH:
            raise GraphValidationError(
                f"graph JSON depth limit of {_MAX_JSON_DEPTH} exceeded"
            )
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, str):
            _validate_unicode(item, item_path)
            continue
        if isinstance(item, float):
            if not isfinite(item):
                raise GraphValidationError(f"{item_path} JSON numbers must be finite")
            continue
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise GraphValidationError(
                        f"{item_path} object keys must be strings"
                    )
                _validate_unicode(key, f"{item_path} object key")
                stack.append((nested, f"{item_path}.{key}", depth + 1))
            continue
        if isinstance(item, list) or (
            allow_frozen and isinstance(item, tuple)
        ):
            for index, nested in enumerate(item):
                stack.append((nested, f"{item_path}[{index}]", depth + 1))
            continue
        raise GraphValidationError(
            f"{item_path} has unsupported JSON value type {type(item).__name__}"
        )


def _without_meta(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_meta(item)
            for key, item in value.items()
            if key != "_meta"
        }
    if isinstance(value, (list, tuple)):
        return [_without_meta(item) for item in value]
    return value


def semantic_graph_projection(graph: Any) -> dict[str, Any]:
    """Validate an API graph and return a detached JSON projection without `_meta`."""
    if not isinstance(graph, Mapping):
        raise GraphValidationError("API graph must be an object")
    if not graph:
        raise GraphValidationError("API graph must be a nonempty object")

    node_count = 0
    for node_id, node in graph.items():
        if not isinstance(node_id, str):
            raise GraphValidationError("graph node IDs must be strings")
        if node_id == "_meta":
            continue
        node_count += 1
        if node_count > _MAX_GRAPH_NODES:
            raise GraphValidationError(
                f"graph node limit of {_MAX_GRAPH_NODES} exceeded"
            )
        _validate_identifier(node_id, "graph node ID")
        if node_id != node_id.strip():
            raise GraphValidationError(
                f"graph node ID {node_id!r} must not have leading or trailing whitespace"
            )
        if not node_id.strip():
            raise GraphValidationError("graph node IDs must not be blank")
        if not isinstance(node, Mapping):
            raise GraphValidationError(f"node '{node_id}' must be an object")
        if "class_type" not in node:
            raise GraphValidationError(f"node '{node_id}' requires class_type")
        class_type = node["class_type"]
        if not isinstance(class_type, str):
            raise GraphValidationError(f"node '{node_id}' class_type must be a string")
        _validate_identifier(class_type, f"node '{node_id}' class_type")
        if not class_type.strip():
            raise GraphValidationError(f"node '{node_id}' class_type must be nonblank")
        if "inputs" not in node:
            raise GraphValidationError(f"node '{node_id}' requires inputs")
        inputs = node["inputs"]
        if not isinstance(inputs, Mapping):
            raise GraphValidationError(f"node '{node_id}' inputs must be an object")
        if any(not isinstance(field_name, str) for field_name in inputs):
            raise GraphValidationError(
                f"node '{node_id}' input names must be strings"
            )
        for field_name in inputs:
            _validate_identifier(field_name, f"node '{node_id}' input name")
    if node_count == 0:
        raise GraphValidationError("API graph must contain a nonempty node object")

    allow_frozen = isinstance(graph, FrozenDict)
    _validate_json(graph, "graph", allow_frozen=allow_frozen)
    projected = _without_meta(graph)
    return projected


def _input_link(
    node_id: str,
    field_name: str,
    value: Any,
    node_ids: frozenset[str],
) -> tuple[str, int] | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
    ):
        return None

    source_id, output_index = value
    canonical_field = _canonical_field_name(field_name)
    if canonical_field in _LITERAL_COLLECTION_FIELDS:
        return None
    if (
        not isinstance(output_index, int)
        or isinstance(output_index, bool)
        or output_index < 0
    ):
        raise GraphValidationError(
            f"node '{node_id}' input '{field_name}' link output index "
            "must be a nonnegative integer"
        )
    if source_id not in node_ids:
        raise GraphValidationError(
            f"node '{node_id}' input '{field_name}' links to unknown node '{source_id}'"
        )
    return (source_id, output_index)


def _binding(
    node: NormalizedNode,
    field_name: str,
    value: Any,
    confidence: float,
    reason: str,
) -> RoleBinding:
    return RoleBinding(
        node_id=node.node_id,
        field_name=field_name,
        value=value,
        confidence=confidence,
        reason=reason,
    )


def _is_literal(node: NormalizedNode, field_name: str) -> bool:
    return field_name in node.inputs and field_name not in node.input_links


def _nonblank_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_model_loader(node: NormalizedNode) -> bool:
    compact = _compact_class(node.class_type)
    if any(token in compact for token in ("clip", "lora", "textencoder", "vae")):
        return False
    has_identity = any(
        _is_literal(node, field_name)
        and _nonblank_string(node.inputs[field_name])
        for field_name in _MODEL_NAME_FIELDS
    )
    known_loader = any(
        token in compact
        for token in (
            "checkpointloader",
            "loaddiffusionmodel",
            "modelload",
            "unetloader",
        )
    )
    return known_loader or (
        has_identity
        and any(
            token in compact
            for token in ("checkpoint", "diffusion", "loader", "model", "unet")
        )
    )


def _is_unet_loader(node: NormalizedNode) -> bool:
    compact = _compact_class(node.class_type)
    return "unetloader" in compact or "loaddiffusionmodel" in compact


def _is_lora_node(node: NormalizedNode) -> bool:
    if "lora" not in _compact_class(node.class_type):
        return False
    if any(
        _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name])
        for field_name in _LORA_NAME_FIELDS
    ):
        return True
    return any(
        _POWER_LORA_SLOT.fullmatch(field_name)
        and isinstance(value, Mapping)
        and value.get("on", True) is not False
        and _nonblank_string(value.get("lora"))
        for field_name, value in node.inputs.items()
    )


def _lora_binding_sort_key(binding: RoleBinding) -> tuple[int, int, str]:
    match = _POWER_LORA_SLOT.fullmatch(binding.field_name)
    if match is not None:
        slot = match.group(1).lstrip("0") or "0"
        return (0, len(slot), f"{slot}:{binding.field_name}")
    return (1, 0, binding.field_name)


def _lora_order(
    nodes_by_id: Mapping[str, NormalizedNode],
    bindings: list[RoleBinding],
) -> list[RoleBinding]:
    if not bindings:
        return []

    bindings_by_node: dict[str, list[RoleBinding]] = defaultdict(list)
    for binding in bindings:
        bindings_by_node[binding.node_id].append(binding)
    lora_ids = frozenset(bindings_by_node)
    parent_by_node: dict[str, str] = {}

    for node_id in sorted(lora_ids, key=_node_sort_key):
        node = nodes_by_id[node_id]
        model_links = [
            node.input_links[field_name]
            for field_name in _MODEL_LINK_FIELDS
            if field_name in node.input_links
        ]
        if len(model_links) != 1:
            raise GraphTopologyError(
                f"disconnected LoRA node '{node_id}' must have exactly one model input edge"
            )
        source_id = model_links[0][0]
        path = [node_id]
        # Model-preserving intermediates do not break the effective LoRA chain.
        while source_id not in lora_ids and not _is_model_loader(
            nodes_by_id[source_id]
        ):
            if source_id in path:
                path.append(source_id)
                raise GraphTopologyError(
                    f"LoRA model-edge cycle detected: {' -> '.join(path)}"
                )
            path.append(source_id)
            source = nodes_by_id[source_id]
            upstream_links = [
                source.input_links[field_name]
                for field_name in _MODEL_LINK_FIELDS
                if field_name in source.input_links
            ]
            if len(upstream_links) != 1:
                raise GraphTopologyError(
                    f"disconnected LoRA node '{node_id}' cannot trace its model "
                    f"input through '{source_id}'"
                )
            source_id = upstream_links[0][0]
        parent_by_node[node_id] = source_id

    for start in sorted(lora_ids, key=_node_sort_key):
        path: list[str] = []
        cursor = start
        while cursor in lora_ids:
            if cursor in path:
                cycle = path[path.index(cursor) :] + [cursor]
                raise GraphTopologyError(f"LoRA cycle detected: {' -> '.join(cycle)}")
            path.append(cursor)
            cursor = parent_by_node[cursor]

    roots = [node_id for node_id in lora_ids if parent_by_node[node_id] not in lora_ids]
    roots.sort(key=_node_sort_key)
    if len(roots) != 1:
        raise GraphTopologyError(
            "disconnected LoRA chains: expected one model-edge root, found "
            f"{roots}"
        )
    root_parent = nodes_by_id[parent_by_node[roots[0]]]
    if not _is_model_loader(root_parent):
        raise GraphTopologyError(
            f"disconnected LoRA chain root '{roots[0]}' does not follow a model loader"
        )

    children: dict[str, list[str]] = defaultdict(list)
    for node_id, parent_id in parent_by_node.items():
        if parent_id in lora_ids:
            children[parent_id].append(node_id)
    for parent_id, child_ids in children.items():
        child_ids.sort(key=_node_sort_key)
        if len(child_ids) > 1:
            raise GraphTopologyError(
                f"ambiguous LoRA branch at '{parent_id}': {child_ids}"
            )

    ordered_node_ids: list[str] = []
    cursor: str | None = roots[0]
    while cursor is not None:
        ordered_node_ids.append(cursor)
        cursor = children.get(cursor, [None])[0]
    if len(ordered_node_ids) != len(lora_ids):
        missing = sorted(lora_ids - set(ordered_node_ids), key=_node_sort_key)
        raise GraphTopologyError(f"disconnected LoRA nodes: {missing}")

    ordered: list[RoleBinding] = []
    for node_id in ordered_node_ids:
        ordered.extend(
            sorted(bindings_by_node[node_id], key=_lora_binding_sort_key)
        )

    model_consumers: dict[str, list[str]] = defaultdict(list)
    for consumer in nodes_by_id.values():
        for field_name in _MODEL_LINK_FIELDS:
            if field_name in consumer.input_links:
                model_consumers[consumer.input_links[field_name][0]].append(
                    consumer.node_id
                )

    def downstream_outcomes(start: str) -> tuple[set[str], set[str]]:
        pending = [(start, frozenset({start}))]
        reached_loras: set[str] = set()
        terminal_consumers: set[str] = set()
        while pending:
            cursor, path = pending.pop()
            targets = set(model_consumers.get(cursor, []))
            if not targets:
                if cursor != start:
                    terminal_consumers.add(cursor)
                continue
            for target in targets:
                if target in path:
                    raise GraphTopologyError(
                        f"LoRA downstream cycle detected after '{start}'"
                    )
                if target in lora_ids:
                    reached_loras.add(target)
                    continue
                pending.append((target, path | {target}))
        return reached_loras, terminal_consumers

    for index, node_id in enumerate(ordered_node_ids):
        expected_next = (
            ordered_node_ids[index + 1]
            if index + 1 < len(ordered_node_ids)
            else None
        )
        reached_loras, terminals = downstream_outcomes(node_id)
        if expected_next is None:
            if reached_loras:
                raise GraphTopologyError(
                    f"LoRA '{node_id}' has competing downstream LoRA order: "
                    f"{sorted(reached_loras, key=_node_sort_key)}"
                )
            if not terminals:
                raise GraphTopologyError(
                    f"LoRA '{node_id}' has no downstream model consumer"
                )
            continue
        if reached_loras != {expected_next} or terminals:
            raise GraphTopologyError(
                f"LoRA '{node_id}' model flow bypasses or branches around "
                f"'{expected_next}'"
            )
    return ordered


def _role_bindings(nodes: tuple[NormalizedNode, ...]) -> tuple[RoleBindings, ...]:
    grouped: dict[SemanticRole, list[RoleBinding]] = defaultdict(list)
    nodes_by_id = {node.node_id: node for node in nodes}
    consumers: dict[str, list[tuple[NormalizedNode, str]]] = defaultdict(list)
    for node in nodes:
        for field_name, link in node.input_links.items():
            consumers[link[0]].append((node, field_name))

    for node in nodes:
        compact = _compact_class(node.class_type)

        if _is_model_loader(node):
            for field_name in _MODEL_NAME_FIELDS:
                if _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name]):
                    reason = f"{node.class_type} exposes model identity input '{field_name}'"
                    grouped[SemanticRole.CHECKPOINT].append(
                        _binding(node, field_name, node.inputs[field_name], 0.99, reason)
                    )
                    grouped[SemanticRole.MODEL].append(
                        _binding(node, field_name, node.inputs[field_name], 0.99, reason)
                    )
                    if _is_unet_loader(node):
                        grouped[SemanticRole.UNET].append(
                            _binding(
                                node,
                                field_name,
                                node.inputs[field_name],
                                0.99,
                                reason,
                            )
                        )

        if "powerloraloader" in compact:
            for field_name, record in node.inputs.items():
                if (
                    _POWER_LORA_SLOT.fullmatch(field_name) is None
                    or not isinstance(record, Mapping)
                    or record.get("on", True) is False
                    or not _nonblank_string(record.get("lora"))
                ):
                    continue
                value: dict[str, Any] = {"name": record["lora"]}
                strength = record.get("strength")
                if isinstance(strength, (int, float)) and not isinstance(
                    strength, bool
                ):
                    value["strength_bucket"] = strength_bucket(strength)
                elif isinstance(strength, bool):
                    value["strength"] = strength
                grouped[SemanticRole.LORA].append(
                    _binding(
                        node,
                        field_name,
                        value,
                        0.99,
                        f"{node.class_type} exposes nested LoRA slot '{field_name}'",
                    )
                )
        elif _is_lora_node(node):
            for field_name in _LORA_NAME_FIELDS:
                if _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name]):
                    grouped[SemanticRole.LORA].append(
                        _binding(
                            node,
                            field_name,
                            node.inputs[field_name],
                            0.99,
                            f"{node.class_type} exposes LoRA input '{field_name}'",
                        )
                    )

        textish = any(
            token in compact
            for token in ("cliptext", "conditioning", "llm", "prompt", "string", "text")
        )
        for field_name in _PROMPT_FIELDS:
            if not _is_literal(node, field_name):
                continue
            value = node.inputs[field_name]
            if not _nonblank_string(value):
                continue
            prompt_consumers = [
                (consumer, consumer_field)
                for consumer, consumer_field in consumers[node.node_id]
                if _canonical_field_name(consumer_field)
                in {"prompt", "promptline", "text", "value"}
                and any(
                    token in _compact_class(consumer.class_type)
                    for token in (
                        "cliptext",
                        "conditioning",
                        "prompt",
                        "string",
                        "textencode",
                    )
                )
            ]
            if (
                field_name.casefold() not in {"prompt", "text"}
                and not textish
                and not prompt_consumers
            ):
                continue
            confidence = (
                0.97
                if prompt_consumers
                else 0.95
                if field_name.casefold() in {"prompt", "text"}
                else 0.86
            )
            reason = (
                "literal value feeds recognized text consumer topology"
                if prompt_consumers
                else f"literal {field_name} input on {node.class_type}"
            )
            grouped[SemanticRole.PROMPT_SOURCE].append(
                _binding(
                    node,
                    field_name,
                    value,
                    confidence,
                    reason,
                )
            )

        linked_prompt_fields = [
            field_name for field_name in _PROMPT_FIELDS if field_name in node.input_links
        ]
        if "promptline" in compact or (textish and linked_prompt_fields):
            field_name = linked_prompt_fields[0] if linked_prompt_fields else "$class_type"
            value = {
                "consumer_count": len(consumers[node.node_id]),
                "kind": "prompt_line" if "promptline" in compact else "linked_text",
            }
            grouped[SemanticRole.PROMPT_FANOUT].append(
                _binding(
                    node,
                    field_name,
                    value,
                    0.98 if "promptline" in compact else 0.9,
                    f"{node.class_type} participates in linked prompt topology",
                )
            )

        is_output = (
            compact.startswith("save")
            or compact.startswith("preview")
            or "videocombine" in compact
            or compact.endswith("output")
            or compact in {"imagesave", "imagesaversimple"}
        )
        is_reference_provider = not is_output and (
            "loadimage" in compact
            or "reference" in compact
            or "imageprovider" in compact
            or "imageinput" in compact
            or "inputimage" in compact
        )
        if is_reference_provider:
            for field_name in _REFERENCE_FIELDS:
                if _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name]):
                    grouped[SemanticRole.REFERENCE_IMAGE].append(
                        _binding(
                            node,
                            field_name,
                            node.inputs[field_name],
                            0.99,
                            f"unlinked image-name input on {node.class_type}",
                        )
                    )

        for field_name in _SEED_FIELDS:
            if _is_literal(node, field_name):
                value = node.inputs[field_name]
                if isinstance(value, int) and not isinstance(value, bool):
                    generation_seed = (
                        field_name == "noise_seed"
                        or "noise" in compact
                        or "sampler" in compact
                    )
                    grouped[SemanticRole.SEED].append(
                        _binding(
                            node,
                            field_name,
                            value,
                            0.99 if generation_seed else 0.75,
                            (
                                f"generation {field_name} input"
                                if generation_seed
                                else f"generic {field_name} input"
                            ),
                        )
                    )

        if _is_literal(node, "sampler_name"):
            grouped[SemanticRole.SAMPLER].append(
                _binding(
                    node,
                    "sampler_name",
                    node.inputs["sampler_name"],
                    0.99,
                    "explicit sampler_name input",
                )
            )
        elif "sampler" in compact:
            grouped[SemanticRole.SAMPLER].append(
                _binding(
                    node,
                    "$class_type",
                    node.class_type,
                    0.85,
                    "sampler node class",
                )
            )

        for field_name in ("scheduler", "scheduler_name"):
            if _is_literal(node, field_name):
                grouped[SemanticRole.SCHEDULER].append(
                    _binding(
                        node,
                        field_name,
                        node.inputs[field_name],
                        0.99,
                        f"explicit {field_name} input",
                    )
                )
        if (
            "scheduler" in compact
            and not any(
                _is_literal(node, field_name)
                for field_name in ("scheduler", "scheduler_name")
            )
        ):
            grouped[SemanticRole.SCHEDULER].append(
                _binding(
                    node,
                    "$class_type",
                    node.class_type,
                    0.9,
                    "scheduler node class",
                )
            )

        for field_name in _DIMENSION_FIELDS:
            if _is_literal(node, field_name):
                value = node.inputs[field_name]
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    grouped[SemanticRole.DIMENSIONS].append(
                        _binding(
                            node,
                            field_name,
                            value,
                            0.95,
                            f"positive {field_name} input",
                        )
                    )

        if "vaeloader" in compact:
            for field_name in _VAE_NAME_FIELDS:
                if _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name]):
                    grouped[SemanticRole.VAE].append(
                        _binding(
                            node,
                            field_name,
                            node.inputs[field_name],
                            0.99,
                            f"{node.class_type} exposes VAE identity input '{field_name}'",
                        )
                    )

        is_encoder = any(
            token in compact
            for token in ("cliploader", "encoder", "textencode")
        )
        if is_encoder:
            identity_fields = [
                field_name
                for field_name in _ENCODER_NAME_FIELDS
                if _is_literal(node, field_name) and _nonblank_string(node.inputs[field_name])
            ]
            if identity_fields:
                for field_name in identity_fields:
                    grouped[SemanticRole.ENCODER].append(
                        _binding(
                            node,
                            field_name,
                            node.inputs[field_name],
                            0.99,
                            f"{node.class_type} exposes encoder identity input '{field_name}'",
                        )
                    )
            else:
                confidence = 0.8 if "textencode" in compact else 0.85
                grouped[SemanticRole.ENCODER].append(
                    _binding(
                        node,
                        "$class_type",
                        node.class_type,
                        confidence,
                        "encoder node class",
                    )
                )

        if "conditioning" in compact or "textencode" in compact:
            grouped[SemanticRole.CONDITIONING].append(
                _binding(
                    node,
                    "$class_type",
                    node.class_type,
                    0.95,
                    "conditioning-producing node class",
                )
            )

        if is_output:
            grouped[SemanticRole.OUTPUT].append(
                _binding(
                    node,
                    "$class_type",
                    node.class_type,
                    0.99,
                    "terminal output node class",
                )
            )

    grouped[SemanticRole.LORA] = _lora_order(
        nodes_by_id,
        grouped.get(SemanticRole.LORA, []),
    )

    role_groups: list[RoleBindings] = []
    for role in SemanticRole:
        role_values = grouped.get(role, [])
        if role is not SemanticRole.LORA:
            role_values.sort(
                key=lambda binding: (
                    _node_sort_key(binding.node_id),
                    binding.field_name,
                )
            )
        if role_values:
            role_groups.append(RoleBindings(role=role, bindings=tuple(role_values)))
    return tuple(role_groups)


def normalize_graph(graph: Any) -> NormalizedGraph:
    projection = semantic_graph_projection(graph)
    node_ids = frozenset(projection)
    normalized_nodes: list[NormalizedNode] = []

    for node_id in sorted(node_ids, key=_node_sort_key):
        node = projection[node_id]
        inputs = node["inputs"]
        input_links: dict[str, tuple[str, int]] = {}
        for field_name, value in inputs.items():
            link = _input_link(node_id, field_name, value, node_ids)
            if link is not None:
                input_links[field_name] = link
        try:
            normalized_nodes.append(
                NormalizedNode(
                    node_id=node_id,
                    class_type=node["class_type"],
                    inputs=inputs,
                    input_links=input_links,
                )
            )
        except (RecursionError, UnicodeError, ValidationError) as error:
            raise GraphValidationError(
                f"node '{node_id}' failed normalized model validation: {error}"
            ) from error

    nodes = tuple(normalized_nodes)
    try:
        return NormalizedGraph(nodes=nodes, role_bindings=_role_bindings(nodes))
    except GraphNormalizationError:
        raise
    except (RecursionError, UnicodeError, ValidationError) as error:
        raise GraphValidationError(
            f"graph failed normalized model validation: {error}"
        ) from error
