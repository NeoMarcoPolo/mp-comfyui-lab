from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import isfinite
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mp_comfyui_lab.graph.models import (
    NormalizedGraph,
    NormalizedNode,
    RoleBinding,
    SemanticRole,
    strength_bucket,
)
from mp_comfyui_lab.graph.normalize import (
    GraphNormalizationError,
    _lora_chains,
    normalize_graph,
    semantic_graph_projection,
)
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import (
    CANONICAL_SECRET_KEYS,
    canonical_security_key,
)


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_PLACEHOLDER = re.compile(r"\{[^{}]*\}|<[^<>]*>|\[[^\[\]]*\]")
_OPERATIONAL_FIELDS = frozenset(
    {
        "completedat",
        "createdat",
        "filename",
        "filenameprefix",
        "filepath",
        "jobid",
        "outputfilename",
        "outputpath",
        "outputprefix",
        "providertaskid",
        "requestid",
        "submittedat",
        "taskid",
        "timestamp",
        "updatedat",
        "workflowid",
    }
)
_PROMPT_FIELDS = frozenset(
    {"negativeprompt", "positiveprompt", "prompt", "promptline", "text", "value"}
)
_REFERENCE_FIELDS = frozenset({"filename", "image", "imagename", "path"})
_OUTPUT_PREFIX_FIELDS = frozenset({"prefix", "saveprefix"})
_POWER_LORA_SLOT = re.compile(r"^lora_\d+$", re.IGNORECASE)
_SIGNED_CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "awsaccesskeyid",
        "xamzalgorithm",
        "xamzcredential",
        "xamzdate",
        "xamzexpires",
        "xamzsecuritytoken",
        "xamzsignature",
        "xamzsignedheaders",
        "xgoogalgorithm",
        "xgoogcredential",
        "xgoogdate",
        "xgoogexpires",
        "xgoogsignature",
        "xgoogsignedheaders",
    }
    | CANONICAL_SECRET_KEYS
)
_MAX_HASH_JSON_DEPTH = 48
_MAX_HASH_JSON_VALUES = 100_000
_MAX_CANONICALIZATION_STATES = 4_096
_MAX_INDIVIDUALIZATION_BRANCHES = 8


class GraphHashError(ValueError):
    pass


class HashValidationError(GraphHashError):
    pass


class CapabilityCanonicalizationError(GraphHashError):
    pass


def _validate_hash_unicode(value: str, path: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise HashValidationError(f"{path} contains a lone UTF-16 surrogate")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    try:
        return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    except (RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise GraphHashError(f"canonical JSON hashing failed: {error}") from error


def _validate_hash_json(value: Any, path: str) -> None:
    stack = [(value, path, 0)]
    value_count = 0
    while stack:
        item, item_path, depth = stack.pop()
        value_count += 1
        if value_count > _MAX_HASH_JSON_VALUES:
            raise HashValidationError(
                f"{path} JSON value limit of {_MAX_HASH_JSON_VALUES} exceeded"
            )
        if depth > _MAX_HASH_JSON_DEPTH:
            raise HashValidationError(
                f"{path} JSON depth limit of {_MAX_HASH_JSON_DEPTH} exceeded"
            )
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, str):
            _validate_hash_unicode(item, item_path)
            continue
        if isinstance(item, float):
            if not isfinite(item):
                raise HashValidationError(
                    f"{item_path} JSON numbers must be finite"
                )
            continue
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise HashValidationError(
                        f"{item_path} object keys must be strings"
                    )
                _validate_hash_unicode(key, f"{item_path} object key")
                stack.append((nested, f"{item_path}.{key}", depth + 1))
            continue
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                stack.append((nested, f"{item_path}[{index}]", depth + 1))
            continue
        raise HashValidationError(
            f"{item_path} has unsupported JSON value type {type(item).__name__}"
        )


def _copy_validated_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Mapping):
        return {
            key: _copy_validated_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_copy_validated_json(item) for item in value]
    raise AssertionError("hash JSON was copied without validation")


def _strict_json_copy(value: Any, path: str) -> Any:
    _validate_hash_json(value, path)
    try:
        return _copy_validated_json(value)
    except (RecursionError, UnicodeError) as error:
        raise HashValidationError(f"{path} JSON copy failed: {error}") from error


def source_graph_hash(graph: Any) -> str:
    normalize_graph(graph)
    return _sha256_json(semantic_graph_projection(graph))


def _validate_fixtures(fixtures: Any) -> dict[str, str]:
    if not isinstance(fixtures, Mapping):
        raise HashValidationError(
            "fixtures must be a strict mapping of names to SHA-256 hashes"
        )
    validated: dict[str, str] = {}
    for name, digest in fixtures.items():
        if not isinstance(name, str) or not name.strip():
            raise HashValidationError("fixture names must be nonblank strings")
        _validate_hash_unicode(name, "fixture name")
        if isinstance(digest, str):
            _validate_hash_unicode(digest, f"fixture '{name}' hash")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise HashValidationError(
                f"fixture '{name}' must have a lowercase SHA-256 hash"
            )
        validated[name] = digest
    return validated


def _validate_prompts(prompts: Any) -> list[str]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise HashValidationError(
            "prompts must be an ordered list or tuple of strings"
        )
    validated: list[str] = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise HashValidationError(f"prompt at index {index} must be a string")
        _validate_hash_unicode(prompt, f"prompt at index {index}")
        validated.append(prompt)
    return validated


def _validate_seed(seed: Any) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise HashValidationError("seed must be an integer")
    if not 0 <= seed < 2**64:
        raise HashValidationError("seed must be between 0 and 2^64 - 1")
    return seed


def exact_run_hash(
    graph: Any,
    fixtures: Mapping[str, str],
    prompts: Sequence[str],
    seed: int,
    execution_parameters: Mapping[str, Any] | None = None,
) -> str:
    normalize_graph(graph)
    if execution_parameters is None:
        execution_parameters = {}
    if not isinstance(execution_parameters, Mapping):
        raise HashValidationError("execution_parameters must be a mapping")

    projection = {
        "execution_parameters": _strict_json_copy(
            execution_parameters,
            "execution_parameters",
        ),
        "fixtures": _validate_fixtures(fixtures),
        "graph": semantic_graph_projection(graph),
        "prompts": _validate_prompts(prompts),
        "schema": "mp-comfyui-lab/exact-run/v1",
        "seed": _validate_seed(seed),
    }
    return _sha256_json(projection)


def _canonical_field_name(field_name: str) -> str:
    return canonical_security_key(field_name)


def _compact_class(class_type: str) -> str:
    return "".join(
        character for character in class_type.casefold() if character.isalnum()
    )


def _prompt_grammar(value: str) -> dict[str, Any]:
    lines = value.splitlines() or [""]
    placeholders = _PROMPT_PLACEHOLDER.findall(value)
    return {
        "colon_led_lines": sum(
            1 for line in lines if ":" in line and line.index(":") > 0
        ),
        "comma_segments": min(value.count(",") + 1, 8),
        "line_count": len(lines),
        "placeholder_count": len(placeholders),
        "placeholder_kinds": sorted(
            {
                "brace" if item.startswith("{") else
                "angle" if item.startswith("<") else
                "bracket"
                for item in placeholders
            }
        ),
    }


def _is_operational_field(field_name: str) -> bool:
    canonical = _canonical_field_name(field_name)
    return (
        canonical in _OPERATIONAL_FIELDS
        or canonical.endswith("seed")
        or canonical.endswith("taskid")
        or canonical.endswith("timestamp")
    )


def _canonicalize_capability_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return value
    try:
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
        return value
    canonical_keys = {canonical_security_key(key) for key, _ in query_items}
    if canonical_keys & _SIGNED_CREDENTIAL_QUERY_KEYS:
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment)
        )
    retained = sorted(
        query_items,
        key=lambda pair: (pair[0], pair[1]),
    )
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(retained, doseq=True),
            parsed.fragment,
        )
    )


def _capability_value(value: Any) -> Any:
    if isinstance(value, str):
        return _canonicalize_capability_url(value)
    if isinstance(value, Mapping):
        projected = {}
        for key, item in value.items():
            if (
                _canonical_field_name(key).startswith("strength")
                and isinstance(item, (int, float))
                and not isinstance(item, bool)
            ):
                projected[key] = {"bucket": strength_bucket(item)}
            else:
                projected[key] = _capability_value(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [_capability_value(item) for item in value]
    return value


def _capability_inputs(
    node: NormalizedNode,
    prompt_fields: frozenset[tuple[str, str]],
    reference_nodes: frozenset[str],
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    compact_class = _compact_class(node.class_type)
    is_output = (
        compact_class.startswith("save")
        or compact_class.startswith("preview")
        or "videocombine" in compact_class
        or compact_class.endswith("output")
        or compact_class in {"imagesave", "imagesaversimple"}
    )
    for field_name, frozen_value in node.inputs.items():
        if field_name in node.input_links or _is_operational_field(field_name):
            continue
        value = thaw_json(frozen_value)
        if (
            "powerloraloader" in compact_class
            and _POWER_LORA_SLOT.fullmatch(field_name)
            and isinstance(value, Mapping)
            and value.get("on") is False
        ):
            continue
        canonical = _canonical_field_name(field_name)
        if is_output and canonical in _OUTPUT_PREFIX_FIELDS:
            continue
        if (
            (node.node_id, field_name) in prompt_fields
            and canonical in _PROMPT_FIELDS
            and isinstance(value, str)
        ):
            projected[field_name] = {"prompt_grammar": _prompt_grammar(value)}
        elif node.node_id in reference_nodes and canonical in _REFERENCE_FIELDS:
            projected[field_name] = {"strategy": "unlinked_literal"}
        elif (
            canonical.startswith("strength")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            projected[field_name] = {"bucket": strength_bucket(value)}
        else:
            projected[field_name] = _capability_value(value)
    return projected


def _refine_partition(
    partition: tuple[tuple[str, ...], ...],
    incoming: Mapping[str, tuple[tuple[str, int, str], ...]],
    outgoing: Mapping[str, tuple[tuple[str, int, str], ...]],
) -> tuple[tuple[str, ...], ...]:
    while True:
        colors = {
            node_id: color
            for color, cell in enumerate(partition)
            for node_id in cell
        }
        refined: list[tuple[str, ...]] = []
        for cell in partition:
            groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
            for node_id in cell:
                signature = (
                    tuple(
                        sorted(
                            (field, output_index, colors[source_id])
                            for field, output_index, source_id in incoming[node_id]
                        )
                    ),
                    tuple(
                        sorted(
                            (field, output_index, colors[target_id])
                            for field, output_index, target_id in outgoing[node_id]
                        )
                    ),
                )
                groups[signature].append(node_id)
            refined.extend(
                tuple(sorted(groups[signature]))
                for signature in sorted(groups)
            )
        next_partition = tuple(refined)
        if next_partition == partition:
            return partition
        partition = next_partition


def _dag_signature_partition(
    node_ids: tuple[str, ...],
    bases: Mapping[str, dict[str, Any]],
    incoming: Mapping[str, tuple[tuple[str, int, str], ...]],
    outgoing: Mapping[str, tuple[tuple[str, int, str], ...]],
) -> tuple[tuple[str, ...], ...] | None:
    indegree = {node_id: len(incoming[node_id]) for node_id in node_ids}
    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    layers: list[tuple[str, ...]] = []
    visited = 0
    while ready:
        layer = tuple(ready)
        layers.append(layer)
        visited += len(layer)
        next_ready: list[str] = []
        for node_id in layer:
            for _, _, target_id in outgoing[node_id]:
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    next_ready.append(target_id)
        ready = next_ready
    if visited != len(node_ids):
        return None

    base_labels = {node_id: _canonical_json(bases[node_id]) for node_id in node_ids}

    def assign_layer_colors(
        layer: tuple[str, ...],
        signatures: Mapping[str, tuple[Any, ...]],
        offset: int,
    ) -> tuple[dict[str, int], int]:
        distinct = sorted({signatures[node_id] for node_id in layer})
        colors_by_signature = {
            signature: offset + index
            for index, signature in enumerate(distinct)
        }
        return (
            {
                node_id: colors_by_signature[signatures[node_id]]
                for node_id in layer
            },
            offset + len(distinct),
        )

    forward: dict[str, int] = {}
    color_offset = 0
    for layer in layers:
        signatures = {
            node_id: (
                base_labels[node_id],
                tuple(
                    sorted(
                        (field, output_index, forward[source_id])
                        for field, output_index, source_id in incoming[node_id]
                    )
                ),
            )
            for node_id in layer
        }
        colors, color_offset = assign_layer_colors(layer, signatures, color_offset)
        forward.update(colors)

    backward: dict[str, int] = {}
    color_offset = 0
    for layer in reversed(layers):
        signatures = {
            node_id: (
                base_labels[node_id],
                tuple(
                    sorted(
                        (field, output_index, backward[target_id])
                        for field, output_index, target_id in outgoing[node_id]
                    )
                ),
            )
            for node_id in layer
        }
        colors, color_offset = assign_layer_colors(layer, signatures, color_offset)
        backward.update(colors)

    groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for node_id in node_ids:
        groups[(base_labels[node_id], forward[node_id], backward[node_id])].append(
            node_id
        )
    return tuple(
        tuple(sorted(groups[signature]))
        for signature in sorted(groups)
    )


def _are_exact_twins(
    left: str,
    right: str,
    node_ids: tuple[str, ...],
    adjacency: Mapping[tuple[str, str], tuple[tuple[str, int], ...]],
) -> bool:
    if adjacency.get((left, left), ()) != adjacency.get((right, right), ()):
        return False
    if adjacency.get((left, right), ()) != adjacency.get((right, left), ()):
        return False
    for other in node_ids:
        if other in {left, right}:
            continue
        if adjacency.get((left, other), ()) != adjacency.get((right, other), ()):
            return False
        if adjacency.get((other, left), ()) != adjacency.get((other, right), ()):
            return False
    return True


def _collapse_exact_twin_cells(
    partition: tuple[tuple[str, ...], ...],
    node_ids: tuple[str, ...],
    adjacency: Mapping[tuple[str, str], tuple[tuple[str, int], ...]],
) -> tuple[tuple[str, ...], ...]:
    collapsed: list[tuple[str, ...]] = []
    for cell in partition:
        if len(cell) > 1 and all(
            _are_exact_twins(cell[0], node_id, node_ids, adjacency)
            for node_id in cell[1:]
        ):
            collapsed.extend((node_id,) for node_id in sorted(cell))
        else:
            collapsed.append(cell)
    return tuple(collapsed)


def _branch_representatives(
    cell: tuple[str, ...],
    node_ids: tuple[str, ...],
    adjacency: Mapping[tuple[str, str], tuple[tuple[str, int], ...]],
) -> tuple[str, ...]:
    representatives: list[str] = []
    for node_id in sorted(cell):
        if any(
            _are_exact_twins(node_id, representative, node_ids, adjacency)
            for representative in representatives
        ):
            continue
        representatives.append(node_id)
    return tuple(representatives)


def _individualize(
    partition: tuple[tuple[str, ...], ...],
    cell_index: int,
    node_id: str,
) -> tuple[tuple[str, ...], ...]:
    cell = partition[cell_index]
    remainder = tuple(item for item in cell if item != node_id)
    return (
        partition[:cell_index]
        + ((node_id,), remainder)
        + partition[cell_index + 1 :]
    )


def _canonical_serialization(
    order: tuple[str, ...],
    bases: Mapping[str, dict[str, Any]],
    raw_edges: tuple[tuple[str, str, str, int], ...],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    positions = {node_id: index for index, node_id in enumerate(order)}
    nodes = [
        {"base": bases[node_id], "index": positions[node_id]}
        for node_id in order
    ]
    edges = sorted(
        [
            {
                "field": field_name,
                "output_index": output_index,
                "source": positions[source_id],
                "target": positions[target_id],
            }
            for source_id, target_id, field_name, output_index in raw_edges
        ],
        key=_canonical_json,
    )
    certificate = _canonical_json({"edges": edges, "nodes": nodes})
    return certificate, nodes, edges, positions


def _canonicalize_topology(
    graph: NormalizedGraph,
    bases: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    incoming_lists: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    outgoing_lists: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    adjacency_lists: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    raw_edges: list[tuple[str, str, str, int]] = []
    for node in graph.nodes:
        for field_name, link in node.input_links.items():
            source_id, output_index = link
            target_id = node.node_id
            incoming_lists[target_id].append((field_name, output_index, source_id))
            outgoing_lists[source_id].append((field_name, output_index, target_id))
            adjacency_lists[(source_id, target_id)].append(
                (field_name, output_index)
            )
            raw_edges.append((source_id, target_id, field_name, output_index))

    node_ids = tuple(node.node_id for node in graph.nodes)
    incoming = {
        node_id: tuple(sorted(incoming_lists[node_id]))
        for node_id in node_ids
    }
    outgoing = {
        node_id: tuple(sorted(outgoing_lists[node_id]))
        for node_id in node_ids
    }
    adjacency = {
        endpoints: tuple(sorted(attributes))
        for endpoints, attributes in adjacency_lists.items()
    }
    initial = _dag_signature_partition(
        node_ids,
        bases,
        incoming,
        outgoing,
    )
    if initial is None:
        base_groups: dict[str, list[str]] = defaultdict(list)
        for node_id in node_ids:
            base_groups[_canonical_json(bases[node_id])].append(node_id)
        initial = tuple(
            tuple(sorted(base_groups[base]))
            for base in sorted(base_groups)
        )

    pending = [_refine_partition(initial, incoming, outgoing)]
    seen: set[tuple[tuple[str, ...], ...]] = set()
    best: tuple[
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
    ] | None = None
    while pending:
        partition = pending.pop()
        if partition in seen:
            continue
        seen.add(partition)
        if len(seen) > _MAX_CANONICALIZATION_STATES:
            raise CapabilityCanonicalizationError(
                "capability canonicalization state bound exceeded"
            )
        partition = _collapse_exact_twin_cells(partition, node_ids, adjacency)
        unresolved = next(
            (
                (index, cell)
                for index, cell in enumerate(partition)
                if len(cell) > 1
            ),
            None,
        )
        if unresolved is None:
            order = tuple(cell[0] for cell in partition)
            candidate = _canonical_serialization(
                order,
                bases,
                tuple(raw_edges),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
            continue

        cell_index, cell = unresolved
        representatives = _branch_representatives(cell, node_ids, adjacency)
        if len(representatives) > _MAX_INDIVIDUALIZATION_BRANCHES:
            raise CapabilityCanonicalizationError(
                "capability canonicalization branch bound exceeded"
            )
        for node_id in reversed(representatives):
            individualized = _individualize(partition, cell_index, node_id)
            pending.append(_refine_partition(individualized, incoming, outgoing))

    if best is None:
        raise CapabilityCanonicalizationError(
            "capability canonicalization produced no exact serialization"
        )
    _, nodes, edges, positions = best
    return positions, edges, nodes


def _sorted_role_values(
    graph: NormalizedGraph,
    role: SemanticRole,
    nodes_by_id: Mapping[str, NormalizedNode],
) -> list[dict[str, Any]]:
    values = [
        {
            "class_type": nodes_by_id[binding.node_id].class_type,
            "field": binding.field_name,
            "value": _capability_value(thaw_json(binding.value)),
        }
        for binding in graph.bindings(role)
    ]
    return sorted(values, key=_canonical_json)


def _capability_projection(graph: NormalizedGraph) -> dict[str, Any]:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    prompt_fields = frozenset(
        (binding.node_id, binding.field_name)
        for binding in graph.bindings(SemanticRole.PROMPT_SOURCE)
    )
    reference_nodes = frozenset(
        binding.node_id
        for binding in graph.bindings(SemanticRole.REFERENCE_IMAGE)
    )
    bases = {
        node.node_id: {
            "class_type": node.class_type,
            "inputs": _capability_inputs(node, prompt_fields, reference_nodes),
        }
        for node in graph.nodes
    }
    labels, edges, topology_nodes = _canonicalize_topology(graph, bases)

    def lora_capability(binding: RoleBinding) -> dict[str, Any]:
        node = nodes_by_id[binding.node_id]
        binding_value = thaw_json(binding.value)
        if isinstance(binding_value, Mapping) and isinstance(
            binding_value.get("name"), str
        ):
            identity = binding_value["name"]
            strengths = {
                key: value
                for key, value in binding_value.items()
                if key != "name"
            }
        else:
            identity = binding_value
            strengths = {
                field_name: strength_bucket(value)
                for field_name, value in node.inputs.items()
                if _canonical_field_name(field_name).startswith("strength")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        return {
            "class_type": node.class_type,
            "identity": _capability_value(identity),
            "slot": binding.field_name,
            "strengths": strengths,
        }

    loras = sorted(
        [
            {
                "chain": [
                    lora_capability(binding)
                    for binding in chain.bindings
                ],
                "consumer_topology": {
                    "class_type": nodes_by_id[chain.consumer_node_id].class_type,
                    "field": _canonical_field_name(chain.consumer_field),
                    "node": labels[chain.consumer_node_id],
                },
            }
            for chain in _lora_chains(
                nodes_by_id,
                list(graph.bindings(SemanticRole.LORA)),
            )
        ],
        key=_canonical_json,
    )

    prompt_sources = sorted(
        [
            {
                "class_type": nodes_by_id[binding.node_id].class_type,
                "field": binding.field_name,
                "grammar": _prompt_grammar(binding.value),
            }
            for binding in graph.bindings(SemanticRole.PROMPT_SOURCE)
            if isinstance(binding.value, str)
        ],
        key=_canonical_json,
    )
    fanout = sorted(
        [
            {
                "class_type": nodes_by_id[binding.node_id].class_type,
                "field": binding.field_name,
                "node_topology": labels[binding.node_id],
                "strategy": _capability_value(thaw_json(binding.value)),
            }
            for binding in graph.bindings(SemanticRole.PROMPT_FANOUT)
        ],
        key=_canonical_json,
    )

    consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph.nodes:
        for field_name, link in node.input_links.items():
            consumers[link[0]].append(
                {"class_type": node.class_type, "field": field_name}
            )
    references = sorted(
        [
            {
                "class_type": nodes_by_id[binding.node_id].class_type,
                "consumers": sorted(consumers[binding.node_id], key=_canonical_json),
                "field": binding.field_name,
                "strategy": "unlinked_literal",
            }
            for binding in graph.bindings(SemanticRole.REFERENCE_IMAGE)
        ],
        key=_canonical_json,
    )

    model_values = {
        _canonical_json(
            {
                "class_type": nodes_by_id[binding.node_id].class_type,
                "field": binding.field_name,
                "identity": _capability_value(thaw_json(binding.value)),
            }
        )
        for role in (SemanticRole.CHECKPOINT, SemanticRole.MODEL, SemanticRole.UNET)
        for binding in graph.bindings(role)
    }

    return {
        "conditioning": _sorted_role_values(
            graph, SemanticRole.CONDITIONING, nodes_by_id
        ),
        "dimensions": _sorted_role_values(graph, SemanticRole.DIMENSIONS, nodes_by_id),
        "encoders": _sorted_role_values(graph, SemanticRole.ENCODER, nodes_by_id),
        "loras": loras,
        "models": [json.loads(value) for value in sorted(model_values)],
        "outputs": {
            "cardinality": len(graph.bindings(SemanticRole.OUTPUT)),
            "types": sorted(
                (
                    _capability_value(thaw_json(binding.value))
                    for binding in graph.bindings(SemanticRole.OUTPUT)
                ),
                key=_canonical_json,
            ),
        },
        "prompt": {"fanout": fanout, "sources": prompt_sources},
        "references": references,
        "samplers": _sorted_role_values(graph, SemanticRole.SAMPLER, nodes_by_id),
        "schedulers": _sorted_role_values(
            graph, SemanticRole.SCHEDULER, nodes_by_id
        ),
        "schema": "mp-comfyui-lab/capability/v1",
        "topology": {
            "edges": edges,
            "nodes": topology_nodes,
        },
        "vae": _sorted_role_values(graph, SemanticRole.VAE, nodes_by_id),
    }


def capability_hash(graph: Any) -> str:
    try:
        normalized = normalize_graph(graph)
        return _sha256_json(_capability_projection(normalized))
    except (GraphHashError, GraphNormalizationError):
        raise
    except (RecursionError, UnicodeError) as error:
        raise GraphHashError(
            f"capability hash projection failed: {error}"
        ) from error
