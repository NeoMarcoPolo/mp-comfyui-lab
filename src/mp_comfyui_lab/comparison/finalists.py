from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from mp_comfyui_lab.experiments.models import (
    ExperimentPlan,
    PlannedVariant,
    PlanningConfidence,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.json_types import thaw_json

from .judgments import (
    JudgmentSubmission,
    VariantJudgment,
    judgment_document_bytes,
)
from .metrics import ComparisonReport, VariantComparison, compare_outputs
from .render import comparison_facts_sha256, render_comparison


class FinalistSelectionError(ValueError):
    """Raised when a human selection is absent, stale, or invalid."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class FinalistSupportingFacts(_FrozenModel):
    objective_completion: bool
    perceptually_unique_output_count: int = Field(ge=0)
    available_output_count: int = Field(ge=0)
    capability_covered: bool
    capability: str
    evidence_confidence: str
    evidence_ids: tuple[str, ...]
    partial_batch: bool
    partial_behavior_approved: bool
    actual_output_count: int = Field(ge=0)
    expected_output_count: int = Field(ge=1)


class FinalistProposal(_FrozenModel):
    variant_id: str
    label: str
    supporting_facts: FinalistSupportingFacts
    factor_scores: dict[str, float]
    judgment_reasons: tuple[str, ...] = ()
    agent_judgment: VariantJudgment | None = None


class FinalistsArtifact(_FrozenModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    plan_hash: str
    comparison_facts_sha256: str
    requires_human_selection: Literal[True] = True
    finalists: tuple[FinalistProposal, ...]
    excluded: dict[str, str]

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(item.variant_id for item in self.finalists)


class HumanSelection(_FrozenModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    plan_hash: str
    finalists_artifact_sha256: str
    variant_id: str
    timestamp: AwareDatetime
    selected_by: Literal["human_cli_input"] = "human_cli_input"

    @property
    def selected_variant_id(self) -> str:
        return self.variant_id

    @property
    def selected_at(self) -> AwareDatetime:
        return self.timestamp


class SelectionState(StrEnum):
    MISSING = "missing"
    VALID = "valid"
    STALE = "stale"


class SelectionStatus(_FrozenModel):
    state: SelectionState
    reason: str | None = None
    selection: HumanSelection | None = None


class ReviewArtifacts(_FrozenModel):
    comparison_path: Path
    finalists_path: Path
    comparison: ComparisonReport
    finalists: FinalistsArtifact


class ReviewArtifactPaths(_FrozenModel):
    comparison_path: Path
    finalists_path: Path


_CONFIDENCE_SCORES = {
    PlanningConfidence.OBSERVED.value: 1.0,
    PlanningConfidence.STRONGLY_INFERRED.value: 0.75,
    PlanningConfidence.WEAKLY_INFERRED.value: 0.5,
    PlanningConfidence.UNEXPLORED.value: 0.0,
}


def propose_finalists(
    report: ComparisonReport,
    plan: ExperimentPlan,
    *,
    judgments: JudgmentSubmission | None = None,
) -> FinalistsArtifact:
    if report.experiment_id.strip() == "":
        raise ValueError("comparison experiment ID must not be blank")
    planned = {item.variant_id: item for item in plan.variants}
    unknown = set(report.variants) - set(planned)
    if unknown:
        raise ValueError(
            f"comparison contains variants absent from the plan: {sorted(unknown)}"
        )
    judgment_by_variant = _validated_judgments(report, judgments)
    partial_approved = _partial_behavior_is_approved(plan)
    duplicate_output_ids = {
        output_id for group in report.duplicate_groups for output_id in group
    }
    proposals: list[FinalistProposal] = []
    excluded: dict[str, str] = {}
    for variant_id, comparison in sorted(report.variants.items()):
        exclusion = _exclusion_reason(comparison, partial_approved)
        if exclusion is not None:
            excluded[variant_id] = exclusion
            continue
        proposal = _proposal(
            comparison,
            planned[variant_id],
            plan,
            partial_approved=partial_approved,
            duplicate_output_ids=duplicate_output_ids,
            judgment=judgment_by_variant.get(variant_id),
        )
        proposals.append(proposal)
    proposals.sort(key=_proposal_order)
    return FinalistsArtifact(
        experiment_id=report.experiment_id,
        plan_hash=plan.plan_hash,
        comparison_facts_sha256=comparison_facts_sha256(report),
        finalists=tuple(proposals),
        excluded=dict(sorted(excluded.items())),
    )


def write_finalists(
    store: ExperimentStore,
    finalists: FinalistsArtifact,
) -> Path:
    with store.run_lock(finalists.experiment_id, "finalist-selection"):
        return store.write_json(
            finalists.experiment_id,
            "finalists.json",
            finalists.model_dump(mode="json"),
        ).resolve()


def regenerate_review_artifacts(
    store: ExperimentStore,
    experiment_id: str,
    plan: ExperimentPlan,
    runs: tuple[RunRecord, ...],
    *,
    judgments: JudgmentSubmission | None = None,
) -> ReviewArtifacts:
    report = compare_outputs(
        runs,
        store.root(experiment_id),
        experiment_id=experiment_id,
    )
    if judgments is not None:
        rendered = render_comparison(
            report,
            store.root(experiment_id),
            judgment_document=judgment_document_bytes(judgments),
            replace_existing=True,
        )
    else:
        rendered = render_comparison(
            report,
            store.root(experiment_id),
            replace_existing=True,
        )
    finalists = propose_finalists(report, plan, judgments=judgments)
    finalists_path = write_finalists(store, finalists)
    return ReviewArtifacts(
        comparison_path=rendered.facts.resolve(),
        finalists_path=finalists_path,
        comparison=report,
        finalists=finalists,
    )


def select_finalist(
    store: ExperimentStore,
    experiment_id: str,
    variant_id: str,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> HumanSelection:
    with store.run_lock(experiment_id, "finalist-selection"):
        finalists, finalists_bytes = _load_finalists_document(
            store,
            experiment_id,
        )
        if variant_id not in finalists.variant_ids:
            raise FinalistSelectionError(
                f"variant {variant_id!r} is not a current finalist"
            )
        selection = HumanSelection(
            experiment_id=experiment_id,
            plan_hash=finalists.plan_hash,
            finalists_artifact_sha256=hashlib.sha256(
                finalists_bytes
            ).hexdigest(),
            variant_id=variant_id,
            timestamp=now().astimezone(UTC),
        )
        store.write_json(
            experiment_id,
            "selection.json",
            selection.model_dump(mode="json"),
        )
    return selection


def selection_status(
    store: ExperimentStore,
    experiment_id: str,
) -> SelectionStatus:
    selection_path = store.root(experiment_id) / "selection.json"
    if not selection_path.exists():
        return SelectionStatus(
            state=SelectionState.MISSING,
            reason="selection.json does not exist",
        )
    try:
        selection = HumanSelection.model_validate_json(
            store.read_bytes(experiment_id, "selection.json")
        )
    except (OSError, ValueError) as error:
        return SelectionStatus(
            state=SelectionState.STALE,
            reason=f"selection artifact is invalid: {error}",
        )
    if selection.experiment_id != experiment_id:
        return _stale(selection, "selection experiment ID changed")
    try:
        finalists, finalists_bytes = _load_finalists_document(
            store,
            experiment_id,
        )
    except (OSError, ValueError) as error:
        return _stale(selection, f"current finalists artifact is invalid: {error}")
    if selection.plan_hash != finalists.plan_hash:
        return _stale(selection, "finalists plan hash changed")
    if selection.finalists_artifact_sha256 != hashlib.sha256(
        finalists_bytes
    ).hexdigest():
        return _stale(selection, "finalists artifact SHA-256 changed")
    if selection.selected_variant_id not in finalists.variant_ids:
        return _stale(selection, "selected variant is not a current finalist")
    return SelectionStatus(state=SelectionState.VALID, selection=selection)


def require_valid_selection(
    store: ExperimentStore,
    experiment_id: str,
) -> HumanSelection:
    status = selection_status(store, experiment_id)
    if status.state is not SelectionState.VALID or status.selection is None:
        raise FinalistSelectionError(
            status.reason or "a current human selection is required"
        )
    return status.selection


def _proposal(
    comparison: VariantComparison,
    variant: PlannedVariant,
    plan: ExperimentPlan,
    *,
    partial_approved: bool,
    duplicate_output_ids: set[str],
    judgment: VariantJudgment | None,
) -> FinalistProposal:
    partial_batch = comparison.status == RunState.INCOMPLETE.value
    available = tuple(
        output for output in comparison.outputs if output.availability == "available"
    )
    unique_count = sum(
        output.output_id not in duplicate_output_ids for output in available
    )
    capability = (
        "baseline" if variant.hypothesis is None else variant.hypothesis.value
    )
    capability_covered = (
        variant.hypothesis is None or variant.hypothesis in plan.capability_coverage
    )
    confidence = (
        variant.confidence.value
        if variant.confidence is not None
        else PlanningConfidence.UNEXPLORED.value
    )
    uniqueness_score = unique_count / max(1, len(available))
    judgment_score = _judgment_score(judgment)
    factor_scores = {
        "objective_completion": 1.0 if not partial_batch else 0.5,
        "perceptual_uniqueness": uniqueness_score,
        "capability_coverage": 1.0 if capability_covered else 0.0,
        "evidence_confidence": _CONFIDENCE_SCORES[confidence],
        "agent_judgment": judgment_score,
    }
    label = comparison.label
    if partial_batch:
        label = f"PARTIAL TEST CAPABILITY: {label}"
    return FinalistProposal(
        variant_id=variant.variant_id,
        label=label,
        supporting_facts=FinalistSupportingFacts(
            objective_completion=not partial_batch,
            perceptually_unique_output_count=unique_count,
            available_output_count=len(available),
            capability_covered=capability_covered,
            capability=capability,
            evidence_confidence=confidence,
            evidence_ids=variant.evidence_ids,
            partial_batch=partial_batch,
            partial_behavior_approved=partial_batch and partial_approved,
            actual_output_count=comparison.actual_output_count,
            expected_output_count=comparison.expected_output_count,
        ),
        factor_scores=factor_scores,
        judgment_reasons=() if judgment is None else judgment.reasons,
        agent_judgment=judgment,
    )


def _proposal_order(proposal: FinalistProposal) -> tuple[float, str]:
    return (-sum(proposal.factor_scores.values()), proposal.variant_id)


def _exclusion_reason(
    comparison: VariantComparison,
    partial_approved: bool,
) -> str | None:
    if comparison.status == RunState.SUCCESS.value:
        return None
    if comparison.status == RunState.INCOMPLETE.value:
        if _is_partial_batch(comparison):
            return None if partial_approved else "partial_batch_not_approved"
        return "run_status_incomplete"
    return f"run_status_{comparison.status.casefold()}"


def _is_partial_batch(comparison: VariantComparison) -> bool:
    return (
        comparison.status == RunState.INCOMPLETE.value
        and 0 < comparison.actual_output_count < comparison.expected_output_count
        and isinstance(comparison.error, Mapping)
        and comparison.error.get("kind") == "partial_outputs"
    )


def _partial_behavior_is_approved(plan: ExperimentPlan) -> bool:
    payload = thaw_json(plan.approval_payload)
    if not isinstance(payload, Mapping):
        return False
    brief = payload.get("brief")
    return (
        isinstance(brief, Mapping)
        and brief.get("partial_behavior_is_tested_capability") is True
    )


def _validated_judgments(
    report: ComparisonReport,
    judgments: JudgmentSubmission | None,
) -> dict[str, VariantJudgment]:
    if judgments is None:
        return {}
    if judgments.experiment_id != report.experiment_id:
        raise ValueError("judgment experiment ID does not match comparison")
    if judgments.facts_sha256 != comparison_facts_sha256(report):
        raise ValueError("judgment facts SHA-256 does not match comparison")
    by_variant: dict[str, VariantJudgment] = {}
    available_outputs = {
        output.output_id: variant.variant_id
        for variant in report.variants.values()
        for output in variant.outputs
        if output.availability == "available"
    }
    for judgment in judgments.judgments:
        if judgment.variant_id not in report.variants:
            raise ValueError("judgment references an unknown variant")
        if judgment.variant_id in by_variant:
            raise ValueError("judgments contain a duplicate variant")
        if any(
            available_outputs.get(output_id) != judgment.variant_id
            for output_id in judgment.referenced_output_ids
        ):
            raise ValueError("judgment references an unavailable or foreign output")
        by_variant[judgment.variant_id] = judgment
    if set(by_variant) != set(report.variants):
        raise ValueError("judgments must cover every comparison variant")
    return by_variant


def _judgment_score(judgment: VariantJudgment | None) -> float:
    if judgment is None:
        return 0.0
    return (
        judgment.identity_fidelity
        + judgment.prompt_adherence
        + judgment.visual_quality
        + judgment.creative_usefulness
    ) / 20.0


def _load_finalists(
    store: ExperimentStore,
    experiment_id: str,
) -> FinalistsArtifact:
    finalists, _ = _load_finalists_document(store, experiment_id)
    return finalists


def _load_finalists_document(
    store: ExperimentStore,
    experiment_id: str,
) -> tuple[FinalistsArtifact, bytes]:
    try:
        document = store.read_bytes(experiment_id, "finalists.json")
        finalists = FinalistsArtifact.model_validate_json(
            document,
            strict=True,
        )
    except (OSError, ValueError) as error:
        raise FinalistSelectionError(
            "a valid current finalists.json artifact is required"
        ) from error
    if finalists.experiment_id != experiment_id:
        raise FinalistSelectionError(
            "finalists experiment ID does not match the requested experiment"
        )
    return finalists, document


def _stale(selection: HumanSelection, reason: str) -> SelectionStatus:
    return SelectionStatus(
        state=SelectionState.STALE,
        reason=reason,
        selection=selection,
    )
