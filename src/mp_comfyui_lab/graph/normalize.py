from __future__ import annotations

import heapq
import json
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
import re
from typing import Any

from pydantic import ValidationError

from mp_comfyui_lab.graph.models import (
    MAX_SERIALIZABLE_JSON_DEPTH,
    MAX_SERIALIZABLE_JSON_VALUES,
    NormalizedGraph,
    NormalizedNode,
    RoleBinding,
    RoleBindings,
    SemanticRole,
    strength_bucket,
)
from mp_comfyui_lab.json_types import FrozenDict, thaw_json


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
_PROMPT_FIELD_KEYS = frozenset(
    {"negativeprompt", "positiveprompt", "prompt", "promptline", "text", "value"}
)
_SELECTOR_PARENT_FIELDS = frozenset(
    {"lorastack", "previouslora", "prevlora"}
)
_SELECTOR_FLOW_FIELDS = _SELECTOR_PARENT_FIELDS | {"lora", "loras"}
_MAX_GRAPH_NODES = 4_096
_MAX_JSON_DEPTH = MAX_SERIALIZABLE_JSON_DEPTH
_MAX_JSON_VALUES = MAX_SERIALIZABLE_JSON_VALUES


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


def _validate_json(
    value: Any,
    path: str,
    *,
    allow_frozen: bool,
    depth_limit: int | None = None,
) -> None:
    effective_depth_limit = _MAX_JSON_DEPTH if depth_limit is None else depth_limit
    stack = [(value, path, 0)]
    value_count = 0
    while stack:
        item, item_path, depth = stack.pop()
        value_count += 1
        if value_count > _MAX_JSON_VALUES:
            raise GraphValidationError(
                f"graph JSON value limit of {_MAX_JSON_VALUES} exceeded"
            )
        if depth > effective_depth_limit:
            raise GraphValidationError(
                f"graph JSON depth limit of {effective_depth_limit} exceeded"
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
            _validate_json(
                node,
                "graph._meta",
                allow_frozen=isinstance(graph, FrozenDict),
            )
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
        for node_field, node_value in node.items():
            if not isinstance(node_field, str):
                raise GraphValidationError(
                    f"node '{node_id}' object keys must be strings"
                )
            _validate_unicode(node_field, f"node '{node_id}' object key")
            if node_field not in {"class_type", "inputs"}:
                _validate_json(
                    node_value,
                    f"node '{node_id}'.{node_field}",
                    allow_frozen=isinstance(graph, FrozenDict),
                )
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
        _validate_json(
            inputs,
            f"node '{node_id}' inputs",
            allow_frozen=isinstance(graph, FrozenDict),
        )
    if node_count == 0:
        raise GraphValidationError("API graph must contain a nonempty node object")

    _validate_json(
        graph,
        "graph",
        allow_frozen=isinstance(graph, FrozenDict),
        depth_limit=_MAX_JSON_DEPTH + 2,
    )
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


def _is_model_flow_field(field_name: str) -> bool:
    canonical = _canonical_field_name(field_name)
    return canonical in {"model", "unet"} or canonical.endswith(("model", "unet"))


def _model_links(node: NormalizedNode) -> list[tuple[str, tuple[str, int]]]:
    return [
        (field_name, link)
        for field_name, link in node.input_links.items()
        if _is_model_flow_field(field_name)
    ]


def _lora_binding_sort_key(binding: RoleBinding) -> tuple[int, int, str]:
    match = _POWER_LORA_SLOT.fullmatch(binding.field_name)
    if match is not None:
        slot = match.group(1).lstrip("0") or "0"
        return (0, len(slot), f"{slot}:{binding.field_name}")
    return (1, 0, binding.field_name)


@dataclass(frozen=True)
class _LoraChain:
    bindings: tuple[RoleBinding, ...]
    consumer_node_id: str
    consumer_field: str


def _lora_chains(
    nodes_by_id: Mapping[str, NormalizedNode],
    bindings: list[RoleBinding],
) -> tuple[_LoraChain, ...]:
    if not bindings:
        return ()

    bindings_by_node: dict[str, list[RoleBinding]] = defaultdict(list)
    for binding in bindings:
        bindings_by_node[binding.node_id].append(binding)
    for node_bindings in bindings_by_node.values():
        node_bindings.sort(key=_lora_binding_sort_key)
    lora_ids = frozenset(bindings_by_node)
    transform_ids = frozenset(
        node_id for node_id in lora_ids if _model_links(nodes_by_id[node_id])
    )
    selector_ids = lora_ids - transform_ids

    model_consumers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for consumer in nodes_by_id.values():
        for field_name, link in _model_links(consumer):
            model_consumers[link[0]].append((consumer.node_id, field_name))

    transform_parent: dict[str, str] = {}
    resolved_model_anchors: dict[str, str] = {}
    for node_id in sorted(transform_ids, key=_node_sort_key):
        links = _model_links(nodes_by_id[node_id])
        if len(links) != 1:
            raise GraphTopologyError(
                f"LoRA node '{node_id}' must have exactly one model input edge"
            )
        source_id = links[0][1][0]
        trail: list[str] = []
        trail_positions: dict[str, int] = {}
        while source_id not in transform_ids and not _is_model_loader(
            nodes_by_id[source_id]
        ):
            if source_id in resolved_model_anchors:
                source_id = resolved_model_anchors[source_id]
                break
            if source_id in trail_positions:
                cycle = trail[trail_positions[source_id] :] + [source_id]
                raise GraphTopologyError(
                    f"LoRA model-edge cycle detected: {' -> '.join(cycle)}"
                )
            if source_id in selector_ids:
                raise GraphTopologyError(
                    f"LoRA model transform '{node_id}' follows selector '{source_id}' "
                    "through a model edge"
                )
            trail_positions[source_id] = len(trail)
            trail.append(source_id)
            upstream = _model_links(nodes_by_id[source_id])
            if len(upstream) != 1:
                raise GraphTopologyError(
                    f"disconnected LoRA node '{node_id}' cannot trace its model "
                    f"input through '{source_id}'"
                )
            source_id = upstream[0][1][0]
        for intermediate_id in trail:
            resolved_model_anchors[intermediate_id] = source_id
        transform_parent[node_id] = source_id

    def parent_topology(
        chain_ids: frozenset[str],
        parent_by_node: Mapping[str, str],
        kind: str,
    ) -> tuple[list[str], dict[str, list[str]], list[str]]:
        children: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in chain_ids}
        for node_id, parent_id in parent_by_node.items():
            if parent_id in chain_ids:
                children[parent_id].append(node_id)
                indegree[node_id] += 1

        ready = [
            (_node_sort_key(node_id), node_id)
            for node_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        roots = [node_id for _, node_id in sorted(ready)]
        ordered: list[str] = []
        while ready:
            _, node_id = heapq.heappop(ready)
            ordered.append(node_id)
            for child_id in children.get(node_id, []):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    heapq.heappush(ready, (_node_sort_key(child_id), child_id))

        if len(ordered) != len(chain_ids):
            remaining = sorted(
                (node_id for node_id, degree in indegree.items() if degree > 0),
                key=_node_sort_key,
            )
            label = (
                "LoRA cycle detected (model)"
                if kind == "model"
                else "LoRA selector cycle detected"
            )
            raise GraphTopologyError(f"{label}: {remaining}")
        return ordered, children, roots

    transform_order, _, transform_roots = parent_topology(
        transform_ids,
        transform_parent,
        "model",
    )

    def downstream_model_outcomes(
        start: str,
    ) -> tuple[set[str], set[tuple[str, str]]]:
        pending = [(start, frozenset({start}), "model")]
        reached_loras: set[str] = set()
        terminal_consumers: set[tuple[str, str]] = set()
        while pending:
            cursor, path, incoming_field = pending.pop()
            targets = model_consumers.get(cursor, [])
            if not targets:
                if cursor != start:
                    terminal_consumers.add((cursor, incoming_field))
                continue
            for target, field_name in targets:
                if target in path:
                    raise GraphTopologyError(
                        f"LoRA downstream cycle detected after '{start}'"
                    )
                if target in transform_ids:
                    reached_loras.add(target)
                    continue
                pending.append((target, path | {target}, field_name))
        return reached_loras, terminal_consumers

    transform_outcomes = {
        node_id: downstream_model_outcomes(node_id)
        for node_id in transform_ids
    }
    for parent_id, (child_ids, _) in transform_outcomes.items():
        for child_id in child_ids:
            if transform_parent.get(child_id) != parent_id:
                raise GraphTopologyError(
                    f"LoRA '{parent_id}' has competing downstream order through "
                    f"'{child_id}'"
                )
        if len(child_ids) > 1 and any(
            not any(transform_outcomes[child_id]) for child_id in child_ids
        ):
            raise GraphTopologyError(
                f"ambiguous LoRA branch at '{parent_id}': "
                f"{sorted(child_ids, key=_node_sort_key)}"
            )

    chains: list[_LoraChain] = []

    def expanded_bindings(path: list[str]) -> tuple[RoleBinding, ...]:
        return tuple(
            binding
            for node_id in path
            for binding in bindings_by_node[node_id]
        )

    def chain_bindings(
        terminal_node_id: str,
        chain_ids: frozenset[str],
        parent_by_node: Mapping[str, str],
    ) -> tuple[RoleBinding, ...]:
        path: list[str] = []
        cursor = terminal_node_id
        while cursor in chain_ids:
            path.append(cursor)
            cursor = parent_by_node.get(cursor, "")
        path.reverse()
        return expanded_bindings(path)

    for node_id in transform_order:
        child_ids, terminals = transform_outcomes[node_id]
        if not child_ids and not terminals:
            prefix = "disconnected LoRA chains: " if len(transform_roots) > 1 else ""
            raise GraphTopologyError(
                f"{prefix}LoRA '{node_id}' has no downstream model consumer"
            )
        for consumer_id, field_name in sorted(terminals):
            chains.append(
                _LoraChain(
                    bindings=chain_bindings(
                        node_id,
                        transform_ids,
                        transform_parent,
                    ),
                    consumer_node_id=consumer_id,
                    consumer_field=field_name,
                )
            )

    selector_parent: dict[str, str] = {}
    for node_id in sorted(selector_ids, key=_node_sort_key):
        parent_links = [
            link
            for field_name, link in nodes_by_id[node_id].input_links.items()
            if _canonical_field_name(field_name) in _SELECTOR_PARENT_FIELDS
        ]
        if len(parent_links) > 1:
            raise GraphTopologyError(
                f"LoRA selector '{node_id}' has competing stack parents"
            )
        if parent_links:
            parent_id = parent_links[0][0]
            if parent_id not in selector_ids:
                raise GraphTopologyError(
                    f"LoRA selector '{node_id}' follows non-selector '{parent_id}'"
                )
            selector_parent[node_id] = parent_id

    selector_order, _, _ = parent_topology(
        selector_ids,
        selector_parent,
        "selector",
    )
    selector_consumers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for consumer in nodes_by_id.values():
        for field_name, link in consumer.input_links.items():
            if _canonical_field_name(field_name) in _SELECTOR_FLOW_FIELDS:
                selector_consumers[link[0]].append((consumer.node_id, field_name))

    def has_downstream_model_consumer(start: str) -> bool:
        pending = [(start, frozenset({start}))]
        while pending:
            cursor, path = pending.pop()
            targets = model_consumers.get(cursor, [])
            if not targets:
                if cursor != start:
                    return True
                continue
            for target, _ in targets:
                if target in path:
                    raise GraphTopologyError(
                        f"LoRA selector downstream cycle detected after '{start}'"
                    )
                pending.append((target, path | {target}))
        return False

    selector_outcomes: dict[
        str, tuple[set[str], set[tuple[str, str]]]
    ] = {}
    for node_id in selector_ids:
        children: set[str] = set()
        terminals: set[tuple[str, str]] = set()
        for target_id, field_name in selector_consumers.get(node_id, []):
            if target_id in selector_ids:
                if selector_parent.get(target_id) != node_id:
                    raise GraphTopologyError(
                        f"LoRA selector '{node_id}' has competing order through "
                        f"'{target_id}'"
                    )
                children.add(target_id)
                continue
            target = nodes_by_id[target_id]
            applies_to_model = _is_model_loader(target) or bool(_model_links(target))
            if applies_to_model and has_downstream_model_consumer(target_id):
                terminals.add((target_id, field_name))
        selector_outcomes[node_id] = (children, terminals)

    for node_id in selector_order:
        child_ids, terminals = selector_outcomes[node_id]
        if not child_ids and not terminals:
            raise GraphTopologyError(
                f"LoRA selector '{node_id}' has no downstream model consumer"
            )
        for consumer_id, field_name in sorted(terminals):
            chains.append(
                _LoraChain(
                    bindings=chain_bindings(
                        node_id,
                        selector_ids,
                        selector_parent,
                    ),
                    consumer_node_id=consumer_id,
                    consumer_field=field_name,
                )
            )

    chains_by_consumer: dict[tuple[str, str], list[_LoraChain]] = defaultdict(list)
    for chain in chains:
        chains_by_consumer[
            (
                chain.consumer_node_id,
                _canonical_field_name(chain.consumer_field),
            )
        ].append(chain)
    for consumer, consumer_chains in chains_by_consumer.items():
        semantic_orders = {
            tuple(
                (
                    nodes_by_id[binding.node_id].class_type,
                    binding.field_name,
                    json.dumps(
                        thaw_json(binding.value),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
                for binding in chain.bindings
            )
            for chain in consumer_chains
        }
        if len(semantic_orders) > 1:
            raise GraphTopologyError(
                "competing LoRA orders reach model consumer "
                f"'{consumer[0]}' field '{consumer[1]}'"
            )

    used = {
        binding.node_id
        for chain in chains
        for binding in chain.bindings
    }
    if used != lora_ids:
        missing = sorted(lora_ids - used, key=_node_sort_key)
        raise GraphTopologyError(f"disconnected or unused LoRA nodes: {missing}")

    def chain_key(chain: _LoraChain) -> tuple[Any, ...]:
        return (
            nodes_by_id[chain.consumer_node_id].class_type,
            _canonical_field_name(chain.consumer_field),
            tuple(
                (
                    nodes_by_id[binding.node_id].class_type,
                    binding.field_name,
                    json.dumps(
                        thaw_json(binding.value),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
                for binding in chain.bindings
            ),
        )

    return tuple(sorted(chains, key=chain_key))


def _lora_order(
    nodes_by_id: Mapping[str, NormalizedNode],
    bindings: list[RoleBinding],
) -> list[RoleBinding]:
    ordered: list[RoleBinding] = []
    seen: set[tuple[str, str]] = set()
    for chain in _lora_chains(nodes_by_id, bindings):
        for binding in chain.bindings:
            key = (binding.node_id, binding.field_name)
            if key not in seen:
                seen.add(key)
                ordered.append(binding)
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
                    value: Any = node.inputs[field_name]
                    if not _model_links(node):
                        selector_value: dict[str, Any] = {"name": value}
                        for strength_field, strength in node.inputs.items():
                            if not _canonical_field_name(strength_field).startswith(
                                "strength"
                            ):
                                continue
                            if isinstance(strength, (int, float)) and not isinstance(
                                strength, bool
                            ):
                                selector_value[f"{strength_field}_bucket"] = (
                                    strength_bucket(strength)
                                )
                            elif isinstance(strength, bool):
                                selector_value[strength_field] = strength
                        value = selector_value
                    grouped[SemanticRole.LORA].append(
                        _binding(
                            node,
                            field_name,
                            value,
                            0.99,
                            f"{node.class_type} exposes LoRA input '{field_name}'",
                        )
                    )

        textish = any(
            token in compact
            for token in ("cliptext", "conditioning", "llm", "prompt", "string", "text")
        )
        prompt_fields = [
            field_name
            for field_name in node.inputs
            if _canonical_field_name(field_name) in _PROMPT_FIELD_KEYS
        ]
        for field_name in prompt_fields:
            if not _is_literal(node, field_name):
                continue
            value = node.inputs[field_name]
            if not _nonblank_string(value):
                continue
            prompt_consumers = [
                (consumer, consumer_field)
                for consumer, consumer_field in consumers[node.node_id]
                if _canonical_field_name(consumer_field)
                in _PROMPT_FIELD_KEYS
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
                _canonical_field_name(field_name) not in {"prompt", "text"}
                and not textish
                and not prompt_consumers
            ):
                continue
            confidence = (
                0.97
                if prompt_consumers
                else 0.95
                if _canonical_field_name(field_name) in {"prompt", "text"}
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
            field_name
            for field_name in node.input_links
            if _canonical_field_name(field_name) in _PROMPT_FIELD_KEYS
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
