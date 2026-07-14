"""Deterministic evidence candidate selection for dry-run planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

from mp_comfyui_lab.graph.models import SemanticRole
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
        dimension = _mutation_dimension(mutation, source)
        if dimension is None:
            return "affected dimension cannot be proven"
        dimensions.add(dimension)
    if "seed_variance" in dimensions and brief.policy_for(
        type(candidate.hypothesis).SEED_VARIANCE
    ).value != "approved":
        return "seed_variance is not approved"
    if len(dimensions) != 1 or candidate.hypothesis.value not in dimensions:
        return "candidate patch changes a dimension other than its hypothesis"
    if brief.policy_for(candidate.hypothesis).value != "approved":
        return f"{candidate.hypothesis.value} is not approved"
    return None


def _mutation_dimension(mutation, source: PlanningSource) -> str | None:
    if isinstance(mutation, SetRoleInput):
        return _ROLE_DIMENSIONS.get(mutation.role)
    if isinstance(mutation, SetNodeInput):
        matches = {
            _ROLE_DIMENSIONS[role]
            for role in SemanticRole
            for binding in source.normalized_graph.bindings(role)
            if binding.node_id == mutation.node_id
            and binding.field_name == mutation.field_name
            and role in _ROLE_DIMENSIONS
        }
        return next(iter(matches)) if len(matches) == 1 else None
    if isinstance(mutation, ReplaceLoraStack):
        existing = tuple(
            str(binding.value.get("name", binding.value))
            if isinstance(binding.value, Mapping)
            else str(binding.value)
            for binding in source.normalized_graph.bindings(SemanticRole.LORA)
        )
        requested = tuple(item.name for item in mutation.loras)
        return "lora_strength" if requested == existing else "lora_stack"
    if isinstance(mutation, ReplaceSubgraph):
        return None
    return None
