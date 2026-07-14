"""Deterministic evidence candidate selection for dry-run planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.mutations.models import SemanticPatch, SetRoleInput

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
                -item.engagement,
                -item.provenance_rank,
                item.candidate_id,
            ),
        )
        selected.append(representative)
        if representative.confidence.value == "unexplored":
            warnings.append(
                f"candidate {representative.candidate_id} has unexplored compatibility"
            )

    return CandidateSelection(
        candidates=tuple(sorted(selected, key=lambda item: item.candidate_id)),
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
