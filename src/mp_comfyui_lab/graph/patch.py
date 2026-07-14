"""Deterministic, atomic materialization of explicit semantic graph patches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mp_comfyui_lab.evidence.models import ConfidenceLabel
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import (
    GraphNormalizationError,
    GraphTopologyError,
    GraphValidationError,
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


def _link(value: Any) -> tuple[str, int] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and value[1] >= 0
    ):
        return (value[0], value[1])
    return None


def _next_numeric_ids(graph: Mapping[str, Any], count: int) -> list[str]:
    maximum = max((int(node_id) for node_id in graph if node_id.isdecimal()), default=0)
    return [str(maximum + offset) for offset in range(1, count + 1)]


def _ordered_donors(mutation: ReplaceSubgraph) -> tuple[Any, ...]:
    donors = {donor.node_id: donor for donor in mutation.donor_nodes}
    parents: dict[str, set[str]] = {node_id: set() for node_id in donors}
    for donor in donors.values():
        for value in thaw_json(donor.inputs).values():
            link = _link(value)
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
    normalized = _normalize(graph)
    bindings = normalized.bindings(mutation.role)
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
    if binding.node_id not in graph or mutation.field_name not in graph[binding.node_id]["inputs"]:
        raise PatchValidationError("role binding does not identify an existing node input")
    graph[binding.node_id]["inputs"][mutation.field_name] = _plain_json_copy(
        thaw_json(mutation.value)
    )


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


def _assert_linear_lora_chain(graph: dict[str, Any]):
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
        if len(_model_links(node)) != 1 or _lora_name_field(node.inputs) != binding.field_name:
            raise PatchTopologyError("replace_lora_stack supports standard model LoRA loaders only")

    chain_ids = [binding.node_id for binding in chain.bindings]
    chain_set = set(chain_ids)
    permitted_consumers = {
        (chain_ids[index + 1], _model_links(nodes[chain_ids[index + 1]])[0][0])
        for index in range(len(chain_ids) - 1)
    }
    permitted_consumers.add((chain.consumer_node_id, chain.consumer_field))
    for consumer in normalized.nodes:
        for field_name, link in consumer.input_links.items():
            if link[0] in chain_set and (consumer.node_id, field_name) not in permitted_consumers:
                raise PatchTopologyError(
                    "replace_lora_stack would orphan a non-chain LoRA consumer"
                )
    return normalized, chain


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


def _replace_lora_stack(graph: dict[str, Any], mutation: ReplaceLoraStack) -> None:
    normalized, chain = _assert_linear_lora_chain(graph)
    bindings = list(chain.bindings)
    nodes = {node.node_id: node for node in normalized.nodes}
    old_ids = [binding.node_id for binding in bindings]
    old_by_name = {str(binding.value): binding for binding in bindings}
    if len(old_by_name) != len(bindings):
        raise PatchTopologyError("replace_lora_stack cannot disambiguate duplicate LoRAs")

    requested = list(mutation.loras)
    additional = max(0, len(requested) - len(old_ids))
    if additional and mutation.loader_template is None:
        raise PatchTopologyError(
            "growing a LoRA stack requires a provenance-bearing loader template"
        )
    new_ids = iter(_next_numeric_ids(graph, additional))
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

    first_model_link = _model_links(nodes[old_ids[0]])[0][1]
    previous = first_model_link
    for node_id, lora in zip(assigned_ids, requested, strict=True):
        inputs = graph[node_id]["inputs"]
        name_field = _lora_name_field(inputs)
        if name_field is None or "strength_model" not in inputs:
            raise PatchTopologyError("LoRA loader does not expose standard LoRA inputs")
        model_fields = [field_name for field_name, _ in _model_links(nodes.get(node_id, nodes[old_ids[0]]))]
        model_field = model_fields[0] if model_fields else "model"
        if model_field not in inputs:
            raise PatchTopologyError("LoRA loader template lacks a model input")
        inputs[name_field] = lora.name
        inputs["strength_model"] = lora.strength_model
        if lora.strength_clip is not None:
            if "strength_clip" not in inputs:
                raise PatchTopologyError("LoRA loader lacks strength_clip input")
            inputs["strength_clip"] = lora.strength_clip
        inputs[model_field] = [previous[0], previous[1]]
        previous = (node_id, 0)

    consumer = graph.get(chain.consumer_node_id)
    if not isinstance(consumer, dict) or chain.consumer_field not in consumer["inputs"]:
        raise PatchTopologyError("LoRA terminal consumer is no longer available")
    consumer["inputs"][chain.consumer_field] = [previous[0], previous[1]]


def _replace_subgraph(graph: dict[str, Any], mutation: ReplaceSubgraph) -> None:
    _normalize(graph)
    targets = set(mutation.target_node_ids)
    missing = sorted(targets.difference(graph), key=_node_sort_key)
    if missing:
        raise PatchValidationError(f"replacement target nodes do not exist: {missing}")
    donor_ids = {node.node_id for node in mutation.donor_nodes}
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
            link = _link(value)
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

    for donor in mutation.donor_nodes:
        inputs = thaw_json(donor.inputs)
        for field_name, value in inputs.items():
            mapping = input_maps.get((donor.node_id, field_name))
            link = _link(value)
            if mapping is not None:
                if mapping.source_node_id not in graph or mapping.source_node_id in targets:
                    raise PatchValidationError("boundary input source must be an existing external node")
                continue
            if link is not None and link[0] not in donor_ids:
                raise PatchValidationError(
                    "donor external links require explicit boundary input mappings"
                )
            if link is not None and link[0] in donor_ids:
                continue
        for mapped_id, mapped_field in input_maps:
            if mapped_id == donor.node_id and mapped_field not in inputs:
                raise PatchValidationError("boundary input names a missing donor field")

    ordered_donors = _ordered_donors(mutation)
    allocated = dict(zip(
        (node.node_id for node in ordered_donors),
        _next_numeric_ids(graph, len(ordered_donors)),
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
                link = _link(value)
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


def _field_diffs(before: Any, after: Any, path: tuple[str, ...] = ()) -> list[FieldDiff]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        diffs: list[FieldDiff] = []
        for key in sorted(set(before) | set(after)):
            if key not in before:
                diffs.append(FieldDiff(path=path + (key,), before_present=False, after_present=True, after=after[key]))
            elif key not in after:
                diffs.append(FieldDiff(path=path + (key,), before_present=True, after_present=False, before=before[key]))
            else:
                diffs.extend(_field_diffs(before[key], after[key], path + (key,)))
        return diffs
    if before != after:
        return [FieldDiff(path=path, before_present=True, after_present=True, before=before, after=after)]
    return []


def _canonical_diff(before: Mapping[str, Any], after: Mapping[str, Any], patch: SemanticPatch) -> CanonicalDiff:
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
        mutations.append(MutationDiff(
            mutation_index=index,
            op=mutation.op,
            evidence_ids=evidence_ids,
            confidence=confidence,
            donor_workflow_id=donor_workflow_id,
        ))
    return CanonicalDiff(nodes=tuple(nodes), mutations=tuple(mutations))


def _materialize(graph: Any, patch: SemanticPatch) -> tuple[dict[str, Any], ValidationReport]:
    if not isinstance(patch, SemanticPatch):
        raise TypeError("patch must be a SemanticPatch")
    # Validate first, then clone all metadata and JSON values into mutable plain JSON.
    semantic_graph_projection(graph)
    working = _plain_json_copy(graph)
    _normalize(working)
    warnings: list[ValidationIssue] = []
    touched_inputs: set[tuple[str, str]] = set()
    for index, mutation in enumerate(patch.mutations):
        if isinstance(mutation, SetRoleInput):
            current = _normalize(working)
            bindings = current.bindings(mutation.role)
            if mutation.binding_index < len(bindings):
                target = (bindings[mutation.binding_index].node_id, mutation.field_name)
                if target in touched_inputs:
                    raise PatchValidationError("semantic patch has conflicting mutations")
                touched_inputs.add(target)
            _set_role_input(working, mutation)
        elif isinstance(mutation, SetNodeInput):
            target = (mutation.node_id, mutation.field_name)
            if target in touched_inputs:
                raise PatchValidationError("semantic patch has conflicting mutations")
            touched_inputs.add(target)
            _set_node_input(working, mutation)
        elif isinstance(mutation, ReplaceLoraStack):
            _replace_lora_stack(working, mutation)
        elif isinstance(mutation, ReplaceSubgraph):
            _replace_subgraph(working, mutation)
        else:  # pragma: no cover - discriminated DTO prevents this branch.
            raise TypeError(f"unsupported mutation type {type(mutation).__name__}")
        _normalize(working)
        if getattr(mutation, "confidence", None) is ConfidenceLabel.UNEXPLORED:
            warnings.append(ValidationIssue(
                code="unexplored_compatibility",
                message="mutation compatibility is unexplored",
                mutation_index=index,
            ))
    _normalize(working)
    return working, ValidationReport(warnings=tuple(warnings))


def apply_patch(graph: Any, patch: SemanticPatch) -> dict[str, Any]:
    """Apply a patch atomically and return a detached, plain JSON graph."""
    materialized, _ = _materialize(graph, patch)
    return _plain_json_copy(materialized)


def materialize_variant(graph: Any, patch: SemanticPatch) -> MaterializationResult:
    """Materialize a patch with deterministic graph diff and validation report."""
    semantic_graph_projection(graph)
    before = _plain_json_copy(graph)
    materialized, validation = _materialize(graph, patch)
    return MaterializationResult(
        graph=materialized,
        diff=_canonical_diff(before, materialized, patch),
        validation=validation,
    )
