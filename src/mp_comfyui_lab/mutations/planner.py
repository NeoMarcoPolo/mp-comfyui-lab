"""Deterministic evidence candidate selection for dry-run planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations.models import (
    ReplaceLoraStack,
    ReplaceSubgraph,
    SemanticPatch,
    SetNodeInput,
    SetRoleInput,
)

if TYPE_CHECKING:
    from mp_comfyui_lab.experiments.models import (
        Candidate,
        DimensionName,
        ExperimentBrief,
        PlanningSource,
    )


_CONFIDENCE_ORDER = {
    "observed": 0,
    "strongly_inferred": 1,
    "weakly_inferred": 2,
    "unexplored": 3,
}
_CLAIM_ROLES = {
    "checkpoint": SemanticRole.CHECKPOINT,
    "prompt_grammar": SemanticRole.PROMPT_SOURCE,
    "sampler": SemanticRole.SAMPLER,
    "scheduler": SemanticRole.SCHEDULER,
    "encoder": SemanticRole.ENCODER,
}
_ROLE_DIMENSIONS = {
    SemanticRole.CHECKPOINT: "checkpoint",
    SemanticRole.MODEL: "checkpoint",
    SemanticRole.UNET: "checkpoint",
    SemanticRole.LORA: "lora_stack",
    SemanticRole.PROMPT_SOURCE: "prompt_grammar",
    SemanticRole.PROMPT_FANOUT: "prompt_grammar",
    SemanticRole.SAMPLER: "sampler",
    SemanticRole.SCHEDULER: "scheduler",
    SemanticRole.ENCODER: "encoder",
    SemanticRole.REFERENCE_IMAGE: "reference_strategy",
    SemanticRole.DIMENSIONS: "output_dimensions",
    SemanticRole.SEED: "seed_variance",
}


@dataclass(frozen=True)
class CandidateSelection:
    candidates: tuple[Candidate, ...]
    warnings: tuple[str, ...]


def select_candidates(
    brief: ExperimentBrief,
    candidates: tuple[Candidate, ...],
) -> CandidateSelection:
    """Pick a single canonical representative for each novel capability."""
    grouped: dict[str, list[Candidate]] = {}
    warnings: list[str] = []
    for candidate in candidates:
        if brief.policy_for(candidate.hypothesis).value != "approved":
            continue
        if candidate.hypothesis.value == "seed_variance" and (
            brief.policy_for(candidate.hypothesis).value != "approved"
        ):
            continue
        grouped.setdefault(candidate.capability_signature, []).append(candidate)

    selected: list[Candidate] = []
    for signature in sorted(grouped):
        representative = min(
            grouped[signature],
            key=lambda item: (
                _CONFIDENCE_ORDER[item.confidence.value],
                -item.provenance_rank,
                -len(item.evidence_ids),
                -item.engagement,
                json.dumps(
                    item.model_dump(mode="json"),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                item.candidate_id,
            ),
        )
        selected.append(representative)
        if representative.confidence.value == "unexplored":
            warnings.append(
                f"candidate {representative.candidate_id} has unexplored compatibility"
            )

    return CandidateSelection(
        candidates=tuple(
            sorted(
                selected,
                key=lambda item: (
                    item.capability_signature,
                    json.dumps(
                        item.model_dump(mode="json"),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            )
        ),
        warnings=tuple(sorted(warnings)),
    )


def propose_patch(candidate: Candidate, source: PlanningSource) -> SemanticPatch | None:
    """Return an exact semantic patch or ``None`` when evidence is insufficient."""
    if candidate.semantic_patch is not None:
        if not isinstance(candidate.semantic_patch, SemanticPatch):
            raise TypeError("candidate semantic_patch must be a SemanticPatch")
        return candidate.semantic_patch
    if len(candidate.claims) != 1:
        return None

    claim = candidate.claims[0]
    if claim.dimension is not candidate.hypothesis:
        return None
    role = _CLAIM_ROLES.get(claim.dimension.value)
    if role is None:
        return None
    bindings = source.normalized_graph.bindings(role)
    if len(bindings) != 1 or claim.binding_index != 0:
        return None
    binding = bindings[0]
    if binding.field_name == "$class_type":
        return None
    return SemanticPatch(
        mutations=(
            SetRoleInput(
                role=role,
                binding_index=0,
                field_name=binding.field_name,
                value=claim.value,
            ),
        )
    )


def validate_candidate_patch(
    candidate: Candidate,
    patch: SemanticPatch,
    source: PlanningSource,
    brief: ExperimentBrief,
) -> str | None:
    """Classify every candidate mutation from graph bindings; never guess."""
    dimensions: set[str] = set()
    for mutation in patch.mutations:
        affected = _mutation_dimensions(mutation, source)
        if affected is None:
            return "affected dimension cannot be proven"
        dimensions.update(affected)
    if "seed_variance" in dimensions and brief.policy_for(
        type(candidate.hypothesis).SEED_VARIANCE
    ).value != "approved":
        return "seed_variance is not approved"
    if len(dimensions) != 1 or candidate.hypothesis.value not in dimensions:
        return "candidate patch changes a dimension other than its hypothesis"
    if brief.policy_for(candidate.hypothesis).value != "approved":
        return f"{candidate.hypothesis.value} is not approved"
    return None


def _mutation_dimensions(mutation, source: PlanningSource) -> set[str] | None:
    if isinstance(mutation, SetRoleInput):
        dimension = _ROLE_DIMENSIONS.get(mutation.role)
        return None if dimension is None else {dimension}
    if isinstance(mutation, SetNodeInput):
        matches = {
            _ROLE_DIMENSIONS[role]
            for role in SemanticRole
            for binding in source.normalized_graph.bindings(role)
            if binding.node_id == mutation.node_id
            and binding.field_name == mutation.field_name
            and role in _ROLE_DIMENSIONS
        }
        return matches if len(matches) == 1 else None
    if isinstance(mutation, ReplaceLoraStack):
        existing = _lora_effective_values(source)
        requested = tuple((item.name, item.strength_model, item.strength_clip) for item in mutation.loras)
        if len(existing) != len(requested):
            dimensions = {"lora_stack"}
            existing_names = {item[0] for item in existing}
            for name, model, clip in requested:
                if name not in existing_names and (model != 1.0 or clip not in {None, 1.0}):
                    dimensions.add("lora_strength")
            return dimensions
        identity_changed = tuple(item[0] for item in existing) != tuple(item[0] for item in requested)
        strength_changed = any(left[1:] != right[1:] for left, right in zip(existing, requested, strict=True))
        dimensions: set[str] = set()
        if identity_changed:
            dimensions.add("lora_stack")
        if strength_changed:
            dimensions.add("lora_strength")
        return dimensions or set()
    if isinstance(mutation, ReplaceSubgraph):
        target_dimensions = {
            _ROLE_DIMENSIONS[role]
            for role in SemanticRole
            for binding in source.normalized_graph.bindings(role)
            if binding.node_id in mutation.target_node_ids and role in _ROLE_DIMENSIONS
        }
        if len(target_dimensions) != 1:
            return None
        dimension = next(iter(target_dimensions))
        donor_classes = tuple(node.class_type.casefold() for node in mutation.donor_nodes)
        required = {
            "encoder": "encode",
            "prompt_grammar": "prompt",
            "reference_strategy": "image",
            "checkpoint": "load",
        }.get(dimension)
        if required is None or not any(required in class_type for class_type in donor_classes):
            return None
        return {dimension}
    return None


def _lora_effective_values(source: PlanningSource) -> tuple[tuple[str, float | None, float | None], ...]:
    nodes = {node.node_id: node for node in source.normalized_graph.nodes}
    values = []
    for binding in source.normalized_graph.bindings(SemanticRole.LORA):
        raw = thaw_json(binding.value)
        name = str(raw.get("name", raw)) if isinstance(raw, Mapping) else str(raw)
        node = nodes[binding.node_id]
        model = node.inputs.get("strength_model")
        clip = node.inputs.get("strength_clip")
        values.append((name, float(model) if isinstance(model, (int, float)) and not isinstance(model, bool) else None, float(clip) if isinstance(clip, (int, float)) and not isinstance(clip, bool) else None))
    return tuple(values)
