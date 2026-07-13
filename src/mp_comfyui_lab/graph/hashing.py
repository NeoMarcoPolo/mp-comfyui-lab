from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import isfinite
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mp_comfyui_lab.graph.models import NormalizedGraph, NormalizedNode, SemanticRole
from mp_comfyui_lab.graph.normalize import normalize_graph, semantic_graph_projection
from mp_comfyui_lab.json_types import thaw_json


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
_PROMPT_FIELDS = frozenset({"prompt", "promptline", "text", "value"})
_REFERENCE_FIELDS = frozenset({"filename", "image", "imagename", "path"})
_OUTPUT_PREFIX_FIELDS = frozenset({"prefix", "saveprefix"})
_SIGNED_CREDENTIAL_QUERY_KEYS = frozenset(
    {
        "accesstoken",
        "authorization",
        "sig",
        "signature",
        "token",
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
)
_CONDITIONAL_SIGNED_QUERY_KEYS = frozenset({"expires", "keypairid", "policy"})
_STRENGTH_LIMITS = (
    (0.25, "subtle"),
    (0.75, "light"),
    (1.25, "standard"),
    (2.0, "strong"),
)


class HashValidationError(ValueError):
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
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_copy(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        _validate_hash_unicode(value, path)
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise HashValidationError(f"{path} JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HashValidationError(f"{path} object keys must be strings")
            _validate_hash_unicode(key, f"{path} object key")
            copied[key] = _strict_json_copy(item, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_copy(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise HashValidationError(
        f"{path} has unsupported JSON value type {type(value).__name__}"
    )


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


def _canonical_field_name(field_name: str) -> str:
    return "".join(
        character for character in field_name.casefold() if character.isalnum()
    )


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
    canonical_keys = {
        _canonical_field_name(key)
        for key, _ in query_items
    }
    has_signed_credentials = bool(canonical_keys & _SIGNED_CREDENTIAL_QUERY_KEYS)
    retained = sorted(
        (
            (key, item)
            for key, item in query_items
            if _canonical_field_name(key) not in _SIGNED_CREDENTIAL_QUERY_KEYS
            and not (
                has_signed_credentials
                and _canonical_field_name(key) in _CONDITIONAL_SIGNED_QUERY_KEYS
            )
        ),
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
        return {key: _capability_value(item) for key, item in value.items()}
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
    )
    for field_name, frozen_value in node.inputs.items():
        if field_name in node.input_links or _is_operational_field(field_name):
            continue
        value = thaw_json(frozen_value)
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
        elif canonical.startswith("strength") and isinstance(value, (int, float)):
            projected[field_name] = {"bucket": strength_bucket(value)}
        else:
            projected[field_name] = _capability_value(value)
    return projected


def _topology_labels(
    graph: NormalizedGraph,
    bases: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    incoming: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    outgoing: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for node in graph.nodes:
        for field_name, link in node.input_links.items():
            source_id, output_index = link
            incoming[node.node_id].append((field_name, output_index, source_id))
            outgoing[source_id].append((field_name, output_index, node.node_id))

    labels = {node_id: _sha256_json(base) for node_id, base in bases.items()}
    for _ in range(max(1, len(graph.nodes))):
        labels = {
            node.node_id: _sha256_json(
                {
                    "base": bases[node.node_id],
                    "incoming": sorted(
                        [
                            {
                                "field": field_name,
                                "output_index": output_index,
                                "source": labels[source_id],
                            }
                            for field_name, output_index, source_id in incoming[node.node_id]
                        ],
                        key=_canonical_json,
                    ),
                    "outgoing": sorted(
                        [
                            {
                                "field": field_name,
                                "output_index": output_index,
                                "target": labels[target_id],
                            }
                            for field_name, output_index, target_id in outgoing[node.node_id]
                        ],
                        key=_canonical_json,
                    ),
                }
            )
            for node in graph.nodes
        }

    edges = sorted(
        [
            {
                "field": field_name,
                "output_index": output_index,
                "source": labels[source_id],
                "target": labels[node.node_id],
            }
            for node in graph.nodes
            for field_name, (source_id, output_index) in node.input_links.items()
        ],
        key=_canonical_json,
    )
    return labels, edges


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
    labels, edges = _topology_labels(graph, bases)

    loras = []
    for binding in graph.bindings(SemanticRole.LORA):
        node = nodes_by_id[binding.node_id]
        strengths = {
            field_name: strength_bucket(value)
            for field_name, value in node.inputs.items()
            if _canonical_field_name(field_name).startswith("strength")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        loras.append(
            {
                "class_type": node.class_type,
                "identity": _capability_value(thaw_json(binding.value)),
                "strengths": strengths,
            }
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
            "nodes": sorted(
                [
                    {"base": bases[node.node_id], "label": labels[node.node_id]}
                    for node in graph.nodes
                ],
                key=_canonical_json,
            ),
        },
        "vae": _sorted_role_values(graph, SemanticRole.VAE, nodes_by_id),
    }


def capability_hash(graph: Any) -> str:
    return _sha256_json(_capability_projection(normalize_graph(graph)))
