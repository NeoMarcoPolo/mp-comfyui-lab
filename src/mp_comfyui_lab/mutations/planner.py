"""Deterministic evidence candidate selection for dry-run planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations.models import (
    ReplaceLoraStack,
    LoraLoaderTemplate,
    LoraSpec,
    ReplaceSubgraph,
    SemanticPatch,
    SetNodeInput,
    SetRoleInput,
)
from mp_comfyui_lab.experiments.models import Candidate, DimensionName, ExperimentBrief, PlanningSource



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


def compile_observed_candidates(
    brief: ExperimentBrief,
    baseline: PlanningSource,
    sources: tuple[PlanningSource, ...],
) -> CandidateSelection:
    roles = {
        DimensionName.CHECKPOINT: SemanticRole.CHECKPOINT,
        DimensionName.SAMPLER: SemanticRole.SAMPLER,
        DimensionName.SCHEDULER: SemanticRole.SCHEDULER,
        DimensionName.ENCODER: SemanticRole.ENCODER,
        DimensionName.PROMPT_GRAMMAR: SemanticRole.PROMPT_SOURCE,
    }
    candidates: list[Candidate] = []
    warnings: list[str] = []
    for donor in sources:
        if donor.workflow_id == baseline.workflow_id:
            continue
        for dimension, role in roles.items():
            if brief.policy_for(dimension).value != "approved":
                continue
            left = baseline.normalized_graph.bindings(role)
            right = donor.normalized_graph.bindings(role)
            if len(left) != 1 or len(right) != 1:
                warnings.append(f"candidate {dimension.value}-{donor.workflow_id} omitted: ambiguous scalar bindings")
                continue
            if left[0].field_name != right[0].field_name or left[0].field_name == "$class_type":
                warnings.append(f"candidate {dimension.value}-{donor.workflow_id} omitted: mismatched scalar bindings")
                continue
            value = thaw_json(right[0].value)
            if value == thaw_json(left[0].value):
                continue
            digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:16]
            candidates.append(Candidate(
                candidate_id=f"{dimension.value}-{donor.workflow_id}"[:96],
                hypothesis=dimension, capability_signature=f"{dimension.value}:{digest}",
                confidence="observed", evidence_ids=(donor.workflow_id,),
                semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=role, binding_index=0, field_name=left[0].field_name, value=value),)),
                engagement=donor.engagement, provenance_rank=1,
            ))
        lora_candidate, lora_warning = _compile_lora_candidate(brief, baseline, donor)
        if lora_candidate is not None:
            candidates.append(lora_candidate)
        if lora_warning is not None:
            warnings.append(lora_warning)
    candidates.sort(key=lambda item: (item.capability_signature, item.candidate_id))
    return CandidateSelection(candidates=tuple(candidates), warnings=tuple(sorted(set(warnings))))


def _compile_lora_candidate(brief, baseline, donor):
    try:
        before = _lora_specs(baseline)
        after = _lora_specs(donor)
    except ValueError as error:
        return None, f"candidate lora-{donor.workflow_id} omitted: {error}"
    if before == after:
        return None, None
    before_names = tuple(item.name for item in before)
    after_names = tuple(item.name for item in after)
    strength_changed = tuple((item.strength_model, item.strength_clip) for item in before) != tuple((item.strength_model, item.strength_clip) for item in after)
    identity_changed = before_names != after_names
    if identity_changed and any(item.strength_model != 1.0 or item.strength_clip not in {None, 1.0} for item in after if item.name not in before_names):
        return None, f"candidate lora-{donor.workflow_id} omitted: lora_stack and lora_strength changed"
    if identity_changed:
        dimension = DimensionName.LORA_STACK
    elif strength_changed:
        dimension = DimensionName.LORA_STRENGTH
    else:
        return None, None
    if brief.policy_for(dimension).value != "approved":
        return None, None
    template = _lora_template(donor) if len(after) > len(before) else None
    if len(after) > len(before) and template is None:
        return None, f"candidate lora-{donor.workflow_id} omitted: exact donor loader template unavailable"
    mutation = ReplaceLoraStack(loras=after, evidence_ids=(donor.workflow_id,), loader_template=template)
    candidate = Candidate(candidate_id=f"{dimension.value}-{donor.workflow_id}"[:96], hypothesis=dimension, capability_signature=f"{dimension.value}:{_lora_signature(after)}", confidence="observed", evidence_ids=(donor.workflow_id,), semantic_patch=SemanticPatch(mutations=(mutation,)), engagement=donor.engagement, provenance_rank=1)
    if _mutation_dimensions(mutation, baseline) != {dimension.value}:
        return None, f"candidate lora-{donor.workflow_id} omitted: compiled patch affects multiple dimensions"
    return candidate, None


def _lora_specs(source):
    nodes = {node.node_id: node for node in source.normalized_graph.nodes}
    specs = []
    for binding in source.normalized_graph.bindings(SemanticRole.LORA):
        node = nodes[binding.node_id]
        name = node.inputs.get("lora_name")
        model = node.inputs.get("strength_model", 1.0)
        clip = node.inputs.get("strength_clip")
        if not isinstance(name, str) or not isinstance(model, (int, float)) or isinstance(model, bool):
            raise ValueError("malformed LoRA binding")
        if clip is not None and (not isinstance(clip, (int, float)) or isinstance(clip, bool)):
            raise ValueError("malformed LoRA binding")
        specs.append(LoraSpec(name=name, strength_model=float(model), strength_clip=None if clip is None else float(clip)))
    return tuple(specs)


def _lora_template(source):
    bindings = source.normalized_graph.bindings(SemanticRole.LORA)
    if not bindings:
        return None
    node_id = bindings[-1].node_id
    raw = thaw_json(source.raw_graph).get(node_id)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("class_type"), str) or not isinstance(raw.get("inputs"), Mapping):
        return None
    return LoraLoaderTemplate(donor_node_id=node_id, class_type=raw["class_type"], inputs=raw["inputs"], donor_workflow_id=source.workflow_id, evidence_ids=(source.workflow_id,))


def _lora_signature(specs):
    return hashlib.sha256(json.dumps([item.model_dump(mode="json") for item in specs], sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


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
