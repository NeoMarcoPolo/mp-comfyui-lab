"""Deterministic, atomic materialization of explicit semantic graph patches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mp_comfyui_lab.evidence.models import ConfidenceLabel
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import (
    GraphNormalizationError,
    GraphTopologyError,
    GraphValidationError,
    _canonical_field_name,
    _compact_class,
    _input_link,
    _lora_chains,
    _model_links,
    _node_sort_key,
    normalize_graph,
    semantic_graph_projection,
)
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations.models import (
    BoundaryInputMapping,
    CanonicalDiff,
    FieldDiff,
    MaterializationResult,
    MutationDiff,
    NodeDiff,
    ReplaceLoraStack,
    ReplaceSubgraph,
    SemanticPatch,
    SetNodeInput,
    SetRoleInput,
    ValidationIssue,
    ValidationReport,
)


class PatchError(ValueError):
    """Base class for patch materialization failures."""


class PatchValidationError(PatchError):
    """The patch does not name a valid explicit graph surface."""


class PatchTopologyError(PatchError):
    """The graph topology is unsupported or would become invalid."""


@dataclass(frozen=True)
class _OperationScope:
    writes: frozenset[tuple[str, str]] = frozenset()
    controlled_nodes: frozenset[str] = frozenset()
    added_nodes: tuple[str, ...] = ()
    deleted_nodes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _LoraChainPlan:
    node_ids: tuple[str, ...]
    model_fields: tuple[str, ...]
    model_base: tuple[str, int]
    model_terminal: tuple[str, str]
    clip_fields: tuple[str, ...] | None
    clip_base: tuple[str, int] | None
    clip_terminals: tuple[tuple[str, str], ...]


def _plain_json_copy(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {key: _plain_json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_copy(item) for item in value]
    raise PatchValidationError(
        f"validated graph contained unsupported JSON type {type(value).__name__}"
    )


def _normalize(graph: Mapping[str, Any]):
    try:
        # Projection is intentionally called even though the working copy keeps metadata.
        semantic_graph_projection(graph)
        return normalize_graph(graph)
    except GraphTopologyError as error:
        raise PatchTopologyError(str(error)) from error
    except GraphValidationError as error:
        message = str(error)
        error_type = PatchTopologyError if "links to unknown" in message else PatchValidationError
        raise error_type(message) from error
    except GraphNormalizationError as error:
        raise PatchValidationError(str(error)) from error


def _next_numeric_ids_for_node_ids(
    node_ids: Mapping[str, Any] | set[str] | frozenset[str],
    count: int,
) -> list[str]:
    maximum = max((int(node_id) for node_id in node_ids if node_id.isdecimal()), default=0)
    return [str(maximum + offset) for offset in range(1, count + 1)]


def _semantic_link(
    node_id: str,
    field_name: str,
    value: Any,
    node_ids: frozenset[str],
) -> tuple[str, int] | None:
    try:
        return _input_link(node_id, field_name, value, node_ids)
    except GraphValidationError as error:
        raise PatchValidationError(str(error)) from error


def _ordered_donors(
    mutation: ReplaceSubgraph,
    node_ids: frozenset[str],
) -> tuple[Any, ...]:
    donors = {donor.node_id: donor for donor in mutation.donor_nodes}
    parents: dict[str, set[str]] = {node_id: set() for node_id in donors}
    for donor in donors.values():
        for field_name, value in thaw_json(donor.inputs).items():
            link = _semantic_link(donor.node_id, field_name, value, node_ids)
            if link is not None and link[0] in donors:
                parents[donor.node_id].add(link[0])
    ordered: list[Any] = []
    ready = sorted((node_id for node_id, values in parents.items() if not values), key=_node_sort_key)
    while ready:
        node_id = ready.pop(0)
        ordered.append(donors[node_id])
        for child_id, values in parents.items():
            if node_id in values:
                values.remove(node_id)
                if not values:
                    ready.append(child_id)
        ready.sort(key=_node_sort_key)
    if len(ordered) != len(donors):
        raise PatchTopologyError("donor subgraph contains an internal cycle")
    return tuple(ordered)


def _set_role_input(
    graph: dict[str, Any],
    mutation: SetRoleInput,
) -> None:
    binding = _role_binding_target(graph, mutation)
    if binding.node_id not in graph or mutation.field_name not in graph[binding.node_id]["inputs"]:
        raise PatchValidationError("role binding does not identify an existing node input")
    graph[binding.node_id]["inputs"][mutation.field_name] = _plain_json_copy(
        thaw_json(mutation.value)
    )
    bindings = _normalize(graph).bindings(mutation.role)
    if (
        mutation.binding_index >= len(bindings)
        or bindings[mutation.binding_index].node_id != binding.node_id
        or bindings[mutation.binding_index].field_name != binding.field_name
    ):
        raise PatchValidationError(
            "selected role binding is no longer valid after assignment"
        )


def _role_binding_target(graph: Mapping[str, Any], mutation: SetRoleInput):
    bindings = _normalize(graph).bindings(mutation.role)
    if mutation.binding_index >= len(bindings):
        raise PatchValidationError(
            f"role {mutation.role.value!r} has no binding at index "
            f"{mutation.binding_index}"
        )
    binding = bindings[mutation.binding_index]
    if binding.field_name != mutation.field_name:
        raise PatchValidationError(
            f"role {mutation.role.value!r} binding {mutation.binding_index} is "
            f"field {binding.field_name!r}, not {mutation.field_name!r}"
        )
    return binding


def _set_node_input(graph: dict[str, Any], mutation: SetNodeInput) -> None:
    node = graph.get(mutation.node_id)
    if not isinstance(node, dict):
        raise PatchValidationError(f"node {mutation.node_id!r} does not exist")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or mutation.field_name not in inputs:
        raise PatchValidationError(
            f"node {mutation.node_id!r} has no input {mutation.field_name!r}"
        )
    inputs[mutation.field_name] = _plain_json_copy(thaw_json(mutation.value))


def _lora_name_field(inputs: Mapping[str, Any]) -> str | None:
    for field_name in ("lora_name", "lora_file", "lora"):
        if field_name in inputs:
            return field_name
    return None


def _clip_link(node: Any) -> tuple[str, tuple[str, int]] | None:
    links = [
        (field_name, link)
        for field_name, link in node.input_links.items()
        if _canonical_field_name(field_name) == "clip"
    ]
    if len(links) > 1:
        raise PatchTopologyError("LoRA loader has multiple clip inputs")
    return links[0] if links else None


def _assert_linear_lora_chain(graph: dict[str, Any]) -> _LoraChainPlan:
    normalized = _normalize(graph)
    bindings = list(normalized.bindings(SemanticRole.LORA))
    if not bindings:
        raise PatchTopologyError("replace_lora_stack requires an existing LoRA chain")
    nodes = {node.node_id: node for node in normalized.nodes}
    try:
        chains = _lora_chains(nodes, bindings)
    except GraphTopologyError as error:
        raise PatchTopologyError(str(error)) from error
    if len(chains) != 1:
        raise PatchTopologyError("replace_lora_stack requires one unambiguous LoRA chain")
    chain = chains[0]
    if len({binding.node_id for binding in chain.bindings}) != len(chain.bindings):
        raise PatchTopologyError("multi-slot and selector LoRA nodes are unsupported")
    if {binding.node_id for binding in chain.bindings} != {binding.node_id for binding in bindings}:
        raise PatchTopologyError("replace_lora_stack requires a single complete LoRA chain")
    for binding in chain.bindings:
        node = nodes[binding.node_id]
        model_links = _model_links(node)
        if (
            len(model_links) != 1
            or model_links[0][1][1] != 0
            or _lora_name_field(node.inputs) != binding.field_name
        ):
            raise PatchTopologyError("replace_lora_stack supports standard model LoRA loaders only")

    chain_ids = tuple(binding.node_id for binding in chain.bindings)
    chain_set = set(chain_ids)
    model_fields = tuple(_model_links(nodes[node_id])[0][0] for node_id in chain_ids)
    model_base = _model_links(nodes[chain_ids[0]])[0][1]
    terminal_model_link = nodes[chain.consumer_node_id].input_links.get(chain.consumer_field)
    if terminal_model_link != (chain_ids[-1], 0):
        raise PatchTopologyError("LoRA terminal model consumer has an unknown output index")
    clip_links = tuple(_clip_link(nodes[node_id]) for node_id in chain_ids)
    has_clip_path = any(link is not None for link in clip_links)
    clip_fields: tuple[str, ...] | None = None
    clip_base: tuple[str, int] | None = None
    if has_clip_path:
        if any(link is None for link in clip_links):
            raise PatchTopologyError("LoRA model and clip paths must use matching loaders")
        resolved_clips = tuple(link for link in clip_links if link is not None)
        if any(link[1][1] != 1 for link in resolved_clips):
            raise PatchTopologyError("LoRA clip path has an unknown output index")
        for index, (_, link) in enumerate(resolved_clips[1:], start=1):
            if link != (chain_ids[index - 1], 1):
                raise PatchTopologyError("LoRA clip path order does not match model path")
        clip_fields = tuple(link[0] for link in resolved_clips)
        clip_base = resolved_clips[0][1]

    permitted_consumers = {
        (chain_ids[index + 1], model_fields[index + 1])
        for index in range(len(chain_ids) - 1)
    }
    permitted_consumers.add((chain.consumer_node_id, chain.consumer_field))
    clip_terminals: list[tuple[str, str]] = []
    if clip_fields is not None:
        permitted_consumers.update(
            (chain_ids[index + 1], clip_fields[index + 1])
            for index in range(len(chain_ids) - 1)
        )
    for consumer in normalized.nodes:
        for field_name, link in consumer.input_links.items():
            if link[0] not in chain_set:
                continue
            if (consumer.node_id, field_name) in permitted_consumers:
                continue
            if (
                clip_fields is not None
                and link == (chain_ids[-1], 1)
                and _canonical_field_name(field_name) == "clip"
                and any(token in _compact_class(consumer.class_type) for token in ("clip", "textencoder"))
            ):
                clip_terminals.append((consumer.node_id, field_name))
                continue
            if link[0] in chain_set:
                raise PatchTopologyError(
                    "replace_lora_stack would orphan a non-chain LoRA consumer"
                )
    return _LoraChainPlan(
        node_ids=chain_ids,
        model_fields=model_fields,
        model_base=model_base,
        model_terminal=(chain.consumer_node_id, chain.consumer_field),
        clip_fields=clip_fields,
        clip_base=clip_base,
        clip_terminals=tuple(sorted(clip_terminals, key=lambda item: (_node_sort_key(item[0]), item[1]))),
    )


def _template_lora_node(mutation: ReplaceLoraStack) -> dict[str, Any]:
    template = mutation.loader_template
    if template is None:
        raise PatchTopologyError(
            "growing a LoRA stack requires a provenance-bearing loader template"
        )
    inputs = _plain_json_copy(thaw_json(template.inputs))
    if _lora_name_field(inputs) is None or "strength_model" not in inputs:
        raise PatchTopologyError("LoRA loader template lacks standard LoRA inputs")
    node: dict[str, Any] = {
        "class_type": template.class_type,
        "inputs": inputs,
    }
    node.update(_plain_json_copy(thaw_json(template.metadata)))
    return node


def _replace_lora_stack(
    graph: dict[str, Any],
    mutation: ReplaceLoraStack,
    reserved_added_ids: tuple[str, ...],
) -> None:
    plan = _assert_linear_lora_chain(graph)
    normalized = _normalize(graph)
    bindings = list(normalized.bindings(SemanticRole.LORA))
    bindings_by_id = {binding.node_id: binding for binding in bindings}
    old_ids = list(plan.node_ids)
    old_by_name = {str(bindings_by_id[node_id].value): bindings_by_id[node_id] for node_id in old_ids}
    if len(old_by_name) != len(old_ids):
        raise PatchTopologyError("replace_lora_stack cannot disambiguate duplicate LoRAs")

    requested = list(mutation.loras)
    additional = max(0, len(requested) - len(old_ids))
    if additional and mutation.loader_template is None:
        raise PatchTopologyError(
            "growing a LoRA stack requires a provenance-bearing loader template"
        )
    if len(reserved_added_ids) != additional or any(
        node_id in graph for node_id in reserved_added_ids
    ):
        raise PatchValidationError("LoRA allocation does not match its reserved node IDs")
    new_ids = iter(reserved_added_ids)
    assigned_ids: list[str] = []
    reusable = iter(node_id for node_id in old_ids if node_id not in {
        old_by_name[lora.name].node_id for lora in requested if lora.name in old_by_name
    })
    # Existing identities retain their node metadata; otherwise consume a spare node first.
    for lora in requested:
        binding = old_by_name.get(lora.name)
        if binding is not None:
            assigned_ids.append(binding.node_id)
        else:
            reused_id = next(reusable, None)
            assigned_ids.append(reused_id if reused_id is not None else next(new_ids, ""))

    if "" in assigned_ids:
        raise PatchTopologyError("LoRA stack allocation failed")
    retained = set(assigned_ids)
    for node_id in old_ids:
        if node_id not in retained:
            del graph[node_id]
    for node_id in assigned_ids:
        if node_id not in graph:
            graph[node_id] = _template_lora_node(mutation)

    previous_model = plan.model_base
    previous_clip = plan.clip_base
    for index, (node_id, lora) in enumerate(zip(assigned_ids, requested, strict=True)):
        inputs = graph[node_id]["inputs"]
        name_field = _lora_name_field(inputs)
        if name_field is None or "strength_model" not in inputs:
            raise PatchTopologyError("LoRA loader does not expose standard LoRA inputs")
        model_field = plan.model_fields[old_ids.index(node_id)] if node_id in old_ids else plan.model_fields[0]
        if model_field not in inputs:
            raise PatchTopologyError("LoRA loader template lacks a model input")
        inputs[name_field] = lora.name
        inputs["strength_model"] = lora.strength_model
        if lora.strength_clip is not None:
            if "strength_clip" not in inputs:
                raise PatchTopologyError("LoRA loader lacks strength_clip input")
            inputs["strength_clip"] = lora.strength_clip
        inputs[model_field] = [previous_model[0], previous_model[1]]
        previous_model = (node_id, 0)
        if plan.clip_fields is not None:
            clip_field = plan.clip_fields[old_ids.index(node_id)] if node_id in old_ids else plan.clip_fields[0]
            if clip_field not in inputs or previous_clip is None:
                raise PatchTopologyError("LoRA loader template lacks a clip input")
            inputs[clip_field] = [previous_clip[0], previous_clip[1]]
            previous_clip = (node_id, 1)

    consumer_id, consumer_field = plan.model_terminal
    consumer = graph.get(consumer_id)
    if not isinstance(consumer, dict) or consumer_field not in consumer["inputs"]:
        raise PatchTopologyError("LoRA terminal consumer is no longer available")
    consumer["inputs"][consumer_field] = [previous_model[0], previous_model[1]]
    if previous_clip is not None:
        for terminal_id, terminal_field in plan.clip_terminals:
            graph[terminal_id]["inputs"][terminal_field] = [previous_clip[0], previous_clip[1]]


def _replace_subgraph(
    graph: dict[str, Any],
    mutation: ReplaceSubgraph,
    reserved_added_ids: tuple[str, ...],
) -> None:
    _normalize(graph)
    targets = set(mutation.target_node_ids)
    missing = sorted(targets.difference(graph), key=_node_sort_key)
    if missing:
        raise PatchValidationError(f"replacement target nodes do not exist: {missing}")
    donors = {node.node_id: node for node in mutation.donor_nodes}
    donor_ids = set(donors)
    link_universe = frozenset((set(graph) - {"_meta"}) | donor_ids)
    input_maps = {
        (item.donor_node_id, item.donor_field_name): item
        for item in mutation.boundary_inputs
    }
    output_maps = {
        (item.consumer_node_id, item.consumer_field_name): item
        for item in mutation.boundary_outputs
    }

    external_links: set[tuple[str, str]] = set()
    for consumer_id, consumer in graph.items():
        if consumer_id in targets or not isinstance(consumer, dict):
            continue
        for field_name, value in consumer["inputs"].items():
            link = _semantic_link(
                consumer_id,
                field_name,
                value,
                frozenset(set(graph) - {"_meta"}),
            )
            if link is not None and link[0] in targets:
                external_links.add((consumer_id, field_name))
    if external_links != set(output_maps):
        raise PatchValidationError("boundary output mappings must cover exactly every target consumer")

    for (consumer_id, field_name), mapping in output_maps.items():
        if consumer_id in targets or consumer_id not in graph:
            raise PatchValidationError("boundary output consumer must be an external existing node")
        if field_name not in graph[consumer_id]["inputs"]:
            raise PatchValidationError("boundary output names a missing consumer field")
        if mapping.donor_node_id not in donor_ids:
            raise PatchValidationError("boundary output names an unknown donor node")
        if mapping.donor_output_index != 0:
            raise PatchValidationError("boundary output names an unknown donor output")

    consumed_inputs: set[tuple[str, str]] = set()
    donor_parents: dict[str, set[str]] = {node_id: set() for node_id in donor_ids}
    for (donor_id, field_name), mapping in input_maps.items():
        donor = donors.get(donor_id)
        if donor is None:
            raise PatchValidationError("boundary input names an unknown donor node")
        if field_name not in donor.inputs:
            raise PatchValidationError("boundary input names an unknown donor field")
        if mapping.source_node_id not in graph or mapping.source_node_id in targets:
            raise PatchValidationError("boundary input source must be an existing external node")
        original = thaw_json(donor.inputs)[field_name]
        original_link = _semantic_link(
            donor_id,
            field_name,
            original,
            link_universe,
        )
        if original is not None and original_link != (
            mapping.source_node_id,
            mapping.source_output_index,
        ):
            raise PatchValidationError("boundary input mapping is unused by its donor field")
        materialized_link = _semantic_link(
            donor_id,
            field_name,
            [mapping.source_node_id, mapping.source_output_index],
            link_universe,
        )
        if materialized_link != (
            mapping.source_node_id,
            mapping.source_output_index,
        ):
            raise PatchValidationError(
                "boundary input mapping does not materialize as a graph link"
            )

    for donor in mutation.donor_nodes:
        inputs = thaw_json(donor.inputs)
        for field_name, value in inputs.items():
            mapping = input_maps.get((donor.node_id, field_name))
            link = _semantic_link(donor.node_id, field_name, value, link_universe)
            if mapping is not None:
                consumed_inputs.add((donor.node_id, field_name))
                continue
            if link is not None and link[0] not in donor_ids:
                raise PatchValidationError(
                    "donor external link requires exactly one boundary input mapping"
                )
            if link is not None and link[0] in donor_ids:
                donor_parents[donor.node_id].add(link[0])
    if consumed_inputs != set(input_maps):
        raise PatchValidationError("boundary input mapping was not consumed exactly once")

    reached = {mapping.donor_node_id for mapping in output_maps.values()}
    pending = list(reached)
    while pending:
        node_id = pending.pop()
        for parent_id in donor_parents[node_id]:
            if parent_id not in reached:
                reached.add(parent_id)
                pending.append(parent_id)
    if reached != donor_ids:
        raise PatchValidationError("every donor node must lie on a path to a boundary output")

    ordered_donors = _ordered_donors(mutation, link_universe)
    if len(reserved_added_ids) != len(ordered_donors) or any(
        node_id in graph for node_id in reserved_added_ids
    ):
        raise PatchValidationError("subgraph allocation does not match its reserved node IDs")
    allocated = dict(zip(
        (node.node_id for node in ordered_donors),
        reserved_added_ids,
        strict=True,
    ))
    for target in targets:
        del graph[target]
    for donor in ordered_donors:
        inputs = _plain_json_copy(thaw_json(donor.inputs))
        for field_name, value in list(inputs.items()):
            mapping: BoundaryInputMapping | None = input_maps.get((donor.node_id, field_name))
            if mapping is not None:
                inputs[field_name] = [mapping.source_node_id, mapping.source_output_index]
            else:
                link = _semantic_link(donor.node_id, field_name, value, link_universe)
                if link is not None and link[0] in allocated:
                    inputs[field_name] = [allocated[link[0]], link[1]]
        node: dict[str, Any] = {"class_type": donor.class_type, "inputs": inputs}
        node.update(_plain_json_copy(thaw_json(donor.metadata)))
        graph[allocated[donor.node_id]] = node
    for mapping in output_maps.values():
        graph[mapping.consumer_node_id]["inputs"][mapping.consumer_field_name] = [
            allocated[mapping.donor_node_id],
            mapping.donor_output_index,
        ]


def _operation_scope(
    graph: Mapping[str, Any],
    mutation: Any,
    reserved_node_ids: frozenset[str] = frozenset(),
) -> _OperationScope:
    if isinstance(mutation, SetRoleInput):
        binding = _role_binding_target(graph, mutation)
        return _OperationScope(writes=frozenset({(binding.node_id, binding.field_name)}))
    if isinstance(mutation, SetNodeInput):
        node = graph.get(mutation.node_id)
        if not isinstance(node, Mapping) or mutation.field_name not in node.get("inputs", {}):
            raise PatchValidationError(
                f"node {mutation.node_id!r} has no input {mutation.field_name!r}"
            )
        return _OperationScope(writes=frozenset({(mutation.node_id, mutation.field_name)}))
    if isinstance(mutation, ReplaceLoraStack):
        plan = _assert_linear_lora_chain(dict(graph))
        added = max(0, len(mutation.loras) - len(plan.node_ids))
        if added and mutation.loader_template is None:
            raise PatchTopologyError("growing a LoRA stack requires a provenance-bearing loader template")
        writes = {(node_id, field_name) for node_id in plan.node_ids for field_name in ("model", "clip", "lora_name", "lora_file", "lora", "strength_model", "strength_clip")}
        writes.update(zip(plan.node_ids, plan.model_fields, strict=True))
        if plan.clip_fields is not None:
            writes.update(zip(plan.node_ids, plan.clip_fields, strict=True))
        writes.add(plan.model_terminal)
        writes.update(plan.clip_terminals)
        return _OperationScope(
            writes=frozenset(writes),
            controlled_nodes=frozenset(plan.node_ids),
            added_nodes=tuple(
                _next_numeric_ids_for_node_ids(
                    (set(graph) - {"_meta"}) | set(reserved_node_ids), added
                )
            ),
            deleted_nodes=frozenset(plan.node_ids),
        )
    if isinstance(mutation, ReplaceSubgraph):
        targets = frozenset(mutation.target_node_ids)
        return _OperationScope(
            writes=frozenset(
                (item.consumer_node_id, item.consumer_field_name)
                for item in mutation.boundary_outputs
            ),
            controlled_nodes=targets,
            added_nodes=tuple(
                _next_numeric_ids_for_node_ids(
                    (set(graph) - {"_meta"}) | set(reserved_node_ids),
                    len(mutation.donor_nodes),
                )
            ),
            deleted_nodes=targets,
        )
    raise TypeError(f"unsupported mutation type {type(mutation).__name__}")


def _preflight_patch(graph: Mapping[str, Any], patch: SemanticPatch) -> tuple[_OperationScope, ...]:
    reserved_node_ids: set[str] = set()
    scopes: list[_OperationScope] = []
    for mutation in patch.mutations:
        scope = _operation_scope(graph, mutation, frozenset(reserved_node_ids))
        scopes.append(scope)
        reserved_node_ids.update(scope.added_nodes)
    scopes = tuple(scopes)
    for index, left in enumerate(scopes):
        left_nodes = left.controlled_nodes | left.deleted_nodes
        for right in scopes[index + 1 :]:
            right_nodes = right.controlled_nodes | right.deleted_nodes
            if (
                left.writes.intersection(right.writes)
                or left_nodes.intersection(right_nodes)
                or any(node_id in right_nodes for node_id, _ in left.writes)
                or any(node_id in left_nodes for node_id, _ in right.writes)
            ):
                raise PatchValidationError("cross-operation conflict in declared graph scopes")
    return scopes


def _assert_operation_scope(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    scope: _OperationScope,
) -> None:
    before_ids = set(before) - {"_meta"}
    after_ids = set(after) - {"_meta"}
    added = after_ids - before_ids
    deleted = before_ids - after_ids
    if not added.issubset(set(scope.added_nodes)) or not deleted.issubset(scope.deleted_nodes):
        raise PatchValidationError("operation changed nodes outside its declared scope")
    for node_id in sorted(before_ids.intersection(after_ids), key=_node_sort_key):
        original = before[node_id]
        updated = after[node_id]
        if original == updated:
            continue
        if not isinstance(original, Mapping) or not isinstance(updated, Mapping):
            raise PatchValidationError("operation changed a node outside its declared scope")
        if {
            key: value for key, value in original.items() if key != "inputs"
        } != {
            key: value for key, value in updated.items() if key != "inputs"
        }:
            raise PatchValidationError("operation changed node metadata outside its declared scope")
        original_inputs = original.get("inputs", {})
        updated_inputs = updated.get("inputs", {})
        for field_name in set(original_inputs) | set(updated_inputs):
            if original_inputs.get(field_name) != updated_inputs.get(field_name) and (
                node_id,
                field_name,
            ) not in scope.writes:
                raise PatchValidationError("operation changed an input outside its declared scope")


def _field_diffs(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[FieldDiff]:
    diffs: list[FieldDiff] = []
    before_inputs = before.get("inputs", {})
    after_inputs = after.get("inputs", {})
    for field_name in sorted(set(before_inputs) | set(after_inputs)):
        if field_name not in before_inputs:
            diffs.append(FieldDiff(path=("inputs", field_name), before_present=False, after_present=True, after=after_inputs[field_name]))
        elif field_name not in after_inputs:
            diffs.append(FieldDiff(path=("inputs", field_name), before_present=True, after_present=False, before=before_inputs[field_name]))
        elif before_inputs[field_name] != after_inputs[field_name]:
            diffs.append(FieldDiff(path=("inputs", field_name), before_present=True, after_present=True, before=before_inputs[field_name], after=after_inputs[field_name]))
    for field_name in sorted((set(before) | set(after)) - {"inputs"}):
        if before.get(field_name) != after.get(field_name):
            diffs.append(FieldDiff(path=(field_name,), before_present=field_name in before, after_present=field_name in after, before=before.get(field_name), after=after.get(field_name)))
    return diffs


def _canonical_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    patch: SemanticPatch,
    operation_added_nodes: tuple[tuple[str, ...], ...],
) -> CanonicalDiff:
    nodes: list[NodeDiff] = []
    node_ids = sorted((set(before) | set(after)) - {"_meta"}, key=_node_sort_key)
    for node_id in node_ids:
        if node_id not in before:
            fields = (FieldDiff(path=(), before_present=False, after_present=True, after=after[node_id]),)
            nodes.append(NodeDiff(node_id=node_id, change="added", fields=fields))
        elif node_id not in after:
            fields = (FieldDiff(path=(), before_present=True, after_present=False, before=before[node_id]),)
            nodes.append(NodeDiff(node_id=node_id, change="removed", fields=fields))
        else:
            fields = tuple(_field_diffs(before[node_id], after[node_id]))
            if fields:
                nodes.append(NodeDiff(node_id=node_id, change="modified", fields=fields))
    mutations = []
    for index, mutation in enumerate(patch.mutations):
        evidence_ids = getattr(mutation, "evidence_ids", ())
        confidence = getattr(mutation, "confidence", None)
        donor_workflow_id = getattr(mutation, "donor_workflow_id", None)
        template = mutation.loader_template if isinstance(mutation, ReplaceLoraStack) else None
        mutations.append(MutationDiff(
            mutation_index=index,
            op=mutation.op,
            evidence_ids=evidence_ids,
            confidence=confidence,
            donor_workflow_id=donor_workflow_id,
            loader_template_donor_node_id=(template.donor_node_id if template is not None else None),
            loader_template_donor_workflow_id=(template.donor_workflow_id if template is not None else None),
            loader_template_evidence_ids=(template.evidence_ids if template is not None else ()),
            added_node_ids=operation_added_nodes[index],
        ))
    return CanonicalDiff(nodes=tuple(nodes), mutations=tuple(mutations))


def _materialize(
    graph: Any,
    patch: SemanticPatch,
) -> tuple[dict[str, Any], ValidationReport, tuple[tuple[str, ...], ...]]:
    if not isinstance(patch, SemanticPatch):
        raise TypeError("patch must be a SemanticPatch")
    # Validate first, then clone all metadata and JSON values into mutable plain JSON.
    semantic_graph_projection(graph)
    working = _plain_json_copy(graph)
    _normalize(working)
    scopes = _preflight_patch(working, patch)
    warnings: list[ValidationIssue] = []
    operation_added_nodes: list[tuple[str, ...]] = []
    for index, mutation in enumerate(patch.mutations):
        before_operation = _plain_json_copy(working)
        if isinstance(mutation, SetRoleInput):
            _set_role_input(working, mutation)
        elif isinstance(mutation, SetNodeInput):
            _set_node_input(working, mutation)
        elif isinstance(mutation, ReplaceLoraStack):
            _replace_lora_stack(working, mutation, scopes[index].added_nodes)
        elif isinstance(mutation, ReplaceSubgraph):
            _replace_subgraph(working, mutation, scopes[index].added_nodes)
        else:  # pragma: no cover - discriminated DTO prevents this branch.
            raise TypeError(f"unsupported mutation type {type(mutation).__name__}")
        _normalize(working)
        _assert_operation_scope(before_operation, working, scopes[index])
        operation_added_nodes.append(
            tuple(sorted((set(working) - set(before_operation)) - {"_meta"}, key=_node_sort_key))
        )
        if getattr(mutation, "confidence", None) is ConfidenceLabel.UNEXPLORED:
            warnings.append(ValidationIssue(
                code="unexplored_compatibility",
                message="mutation compatibility is unexplored",
                mutation_index=index,
            ))
    _normalize(working)
    return working, ValidationReport(warnings=tuple(warnings)), tuple(operation_added_nodes)


def apply_patch(graph: Any, patch: SemanticPatch) -> dict[str, Any]:
    """Apply a patch atomically and return a detached, plain JSON graph."""
    materialized, _, _ = _materialize(graph, patch)
    return _plain_json_copy(materialized)


def materialize_variant(graph: Any, patch: SemanticPatch) -> MaterializationResult:
    """Materialize a patch with deterministic graph diff and validation report."""
    semantic_graph_projection(graph)
    before = _plain_json_copy(graph)
    materialized, validation, operation_added_nodes = _materialize(graph, patch)
    return MaterializationResult(
        graph=materialized,
        diff=_canonical_diff(before, materialized, patch, operation_added_nodes),
        validation=validation,
    )
