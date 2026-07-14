from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

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
from .paths import hash_stored_artifact, path_component_is_link_or_junction
from .render import (
    _rename_directory_new,
    comparison_facts_sha256,
    render_comparison,
)


class FinalistSelectionError(ValueError):
    """Raised when a human selection is absent, stale, or invalid."""


class SelectionRequiredError(FinalistSelectionError):
    """Raised when no explicit human selection exists."""


class StaleSelectionError(FinalistSelectionError):
    """Raised when an existing selection no longer binds current evidence."""


class ArtifactIntegrityError(FinalistSelectionError):
    """Raised when review evidence differs from its canonical durable inputs."""


class ReviewGenerationConflict(ArtifactIntegrityError):
    """Raised when an immutable generation path has different content."""


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


class FinalistOutputEvidence(_FrozenModel):
    output_id: str
    ordinal: int = Field(ge=0)
    sha256: str
    bytes: int = Field(ge=0)
    media_type: str
    artifact_relative_path: str
    node_id: str
    provider_order: int = Field(ge=0)


class FinalistRunEvidence(_FrozenModel):
    run_id: str
    run_record_path: str
    run_record_sha256: str
    state: str
    provider_task_id: str | None
    exact_run_hash: str
    outputs: tuple[FinalistOutputEvidence, ...]


class FinalistProposal(_FrozenModel):
    variant_id: str
    label: str
    supporting_facts: FinalistSupportingFacts
    factor_scores: dict[str, float]
    comparison_report_sha256: str
    run_evidence: FinalistRunEvidence
    judgment_reasons: tuple[str, ...] = ()
    agent_judgment: VariantJudgment | None = None


class FinalistsArtifact(_FrozenModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    plan_hash: str
    comparison_facts_sha256: str
    review_generation_id: str | None = None
    review_generation_path: str | None = None
    comparison_facts_path: str | None = None
    judgments_path: str | None = None
    judgments_sha256: str | None = None
    requires_human_selection: Literal[True] = True
    finalists: tuple[FinalistProposal, ...]
    excluded: dict[str, str]

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(item.variant_id for item in self.finalists)


class ReviewGenerationManifest(_FrozenModel):
    schema_version: Literal["1"] = "1"
    generation_id: str
    experiment_id: str
    plan_hash: str
    comparison_facts_path: Literal["comparisons/facts.json"]
    comparison_facts_sha256: str
    judgments_path: Literal["comparisons/judgments.json"] | None = None
    judgments_sha256: str | None = None
    finalists_path: Literal["finalists.json"]
    finalists_sha256: str


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
    generation_manifest_path: Path
    comparison: ComparisonReport
    finalists: FinalistsArtifact


class ReviewArtifactPaths(_FrozenModel):
    comparison_path: Path
    finalists_path: Path


@dataclass(frozen=True)
class CanonicalReview:
    root: Path
    finalists: FinalistsArtifact
    finalists_bytes: bytes
    manifest: ReviewGenerationManifest
    manifest_bytes: bytes
    manifest_path: Path
    comparison: ComparisonReport
    comparison_bytes: bytes
    judgments: JudgmentSubmission | None
    plan: ExperimentPlan
    runs: tuple[RunRecord, ...]


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
    runs: tuple[RunRecord, ...],
    run_record_sha256: Mapping[str, str] | None = None,
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
    report_sha256 = comparison_facts_sha256(report)
    run_by_id = {run.run_id: run for run in runs}
    if len(run_by_id) != len(runs):
        raise ValueError("comparison runs contain duplicate run IDs")
    selected_runs: dict[str, RunRecord] = {}
    for variant_id, comparison in report.variants.items():
        run = run_by_id.get(comparison.run_id)
        if run is None or run.variant_id != variant_id:
            raise ValueError("comparison run evidence is missing or mismatched")
        selected_runs[variant_id] = run
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
            selected_runs[variant_id],
            planned[variant_id],
            plan,
            partial_approved=partial_approved,
            duplicate_output_ids=duplicate_output_ids,
            judgment=judgment_by_variant.get(variant_id),
            comparison_report_sha256=report_sha256,
            run_record_sha256=(
                None
                if run_record_sha256 is None
                else run_record_sha256.get(comparison.run_id)
            ),
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
    fault_injector: Callable[[str], None] | None = None,
) -> ReviewArtifacts:
    with store.run_lock(experiment_id, "finalist-selection"):
        return _regenerate_review_artifacts_locked(
            store,
            experiment_id,
            plan,
            runs,
            judgments=judgments,
            fault_injector=fault_injector,
        )


def _regenerate_review_artifacts_locked(
    store: ExperimentStore,
    experiment_id: str,
    plan: ExperimentPlan,
    runs: tuple[RunRecord, ...],
    *,
    judgments: JudgmentSubmission | None,
    fault_injector: Callable[[str], None] | None,
) -> ReviewArtifacts:
    root = store.root(experiment_id)
    durable_runs, run_hashes = _durable_run_evidence(
        store,
        experiment_id,
        plan,
        runs,
    )
    report = compare_outputs(
        durable_runs,
        root,
        experiment_id=experiment_id,
    )
    judgment_bytes = (
        None if judgments is None else judgment_document_bytes(judgments)
    )
    rendered = render_comparison(
        report,
        root,
        judgment_document=judgment_bytes,
        replace_existing=True,
    )
    facts_sha256 = comparison_facts_sha256(report)
    judgments_sha256 = (
        None
        if judgment_bytes is None
        else hashlib.sha256(judgment_bytes).hexdigest()
    )
    generation_id = _review_generation_id(
        experiment_id=experiment_id,
        plan_hash=plan.plan_hash,
        comparison_facts_sha256=facts_sha256,
        judgments_sha256=judgments_sha256,
    )
    generation_relative = Path("reviews", generation_id)
    comparison_relative = generation_relative / "comparisons" / "facts.json"
    judgments_relative = (
        None
        if judgment_bytes is None
        else generation_relative / "comparisons" / "judgments.json"
    )
    finalists = propose_finalists(
        report,
        plan,
        runs=durable_runs,
        run_record_sha256=run_hashes,
        judgments=judgments,
    ).model_copy(
        update={
            "review_generation_id": generation_id,
            "review_generation_path": generation_relative.as_posix(),
            "comparison_facts_path": comparison_relative.as_posix(),
            "judgments_path": (
                None
                if judgments_relative is None
                else judgments_relative.as_posix()
            ),
            "judgments_sha256": judgments_sha256,
        }
    )
    finalists_bytes = _json_document_bytes(finalists.model_dump(mode="json"))
    manifest = {
        "schema_version": "1",
        "generation_id": generation_id,
        "experiment_id": experiment_id,
        "plan_hash": plan.plan_hash,
        "comparison_facts_path": "comparisons/facts.json",
        "comparison_facts_sha256": facts_sha256,
        "judgments_path": (
            None if judgment_bytes is None else "comparisons/judgments.json"
        ),
        "judgments_sha256": judgments_sha256,
        "finalists_path": "finalists.json",
        "finalists_sha256": hashlib.sha256(finalists_bytes).hexdigest(),
    }
    generation = _publish_review_generation(
        root,
        generation_id,
        rendered.experiment_root / "comparisons",
        finalists_bytes,
        _json_document_bytes(manifest),
        include_judgments=judgment_bytes is not None,
        fault_injector=fault_injector,
    )
    _inject(fault_injector, "current:before_publish")
    current = root / "finalists.json"
    if not current.exists() or current.read_bytes() != finalists_bytes:
        store.write_json(
            experiment_id,
            "finalists.json",
            finalists.model_dump(mode="json"),
        )
    _inject(fault_injector, "current:published")
    return ReviewArtifacts(
        comparison_path=(generation / "comparisons" / "facts.json").resolve(),
        finalists_path=(generation / "finalists.json").resolve(),
        generation_manifest_path=(generation / "manifest.json").resolve(),
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
        review = load_canonical_review(
            store,
            experiment_id,
        )
        if variant_id not in review.finalists.variant_ids:
            raise FinalistSelectionError(
                f"variant {variant_id!r} is not a current finalist"
            )
        selection = HumanSelection(
            experiment_id=experiment_id,
            plan_hash=review.finalists.plan_hash,
            finalists_artifact_sha256=hashlib.sha256(
                review.finalists_bytes
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


def load_canonical_review(
    store: ExperimentStore,
    experiment_id: str,
) -> CanonicalReview:
    root = store.root(experiment_id)
    try:
        finalists, finalists_bytes = _load_finalists_document(
            store,
            experiment_id,
        )
    except FinalistSelectionError as error:
        raise ArtifactIntegrityError(
            "current finalists artifact is missing or invalid"
        ) from error
    generation_id = finalists.review_generation_id
    generation_relative = finalists.review_generation_path
    facts_relative = finalists.comparison_facts_path
    if (
        generation_id is None
        or generation_relative != f"reviews/{generation_id}"
        or facts_relative
        != f"reviews/{generation_id}/comparisons/facts.json"
    ):
        raise ArtifactIntegrityError(
            "current finalists artifact lacks a canonical review generation"
        )
    manifest_relative = f"reviews/{generation_id}/manifest.json"
    generation_finalists_relative = f"reviews/{generation_id}/finalists.json"
    try:
        manifest_bytes = store.read_bytes(experiment_id, manifest_relative)
        manifest = ReviewGenerationManifest.model_validate_json(
            manifest_bytes,
            strict=True,
        )
        generation_finalists_bytes = store.read_bytes(
            experiment_id,
            generation_finalists_relative,
        )
        comparison_bytes = store.read_bytes(experiment_id, facts_relative)
        comparison = ComparisonReport.model_validate_json(
            comparison_bytes,
            strict=True,
        )
        plan = ExperimentPlan.model_validate_json(
            store.read_bytes(
                experiment_id,
                f"plans/{finalists.plan_hash}.json",
            ),
            strict=True,
        )
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactIntegrityError(
            "review generation contains a missing or invalid artifact"
        ) from error
    if (
        manifest.generation_id != generation_id
        or manifest.experiment_id != experiment_id
        or manifest.plan_hash != finalists.plan_hash
        or manifest.comparison_facts_sha256
        != finalists.comparison_facts_sha256
        or manifest.finalists_sha256
        != hashlib.sha256(generation_finalists_bytes).hexdigest()
        or generation_finalists_bytes != finalists_bytes
        or hashlib.sha256(comparison_bytes).hexdigest()
        != finalists.comparison_facts_sha256
        or comparison_facts_sha256(comparison)
        != finalists.comparison_facts_sha256
        or plan.plan_hash != finalists.plan_hash
    ):
        raise ArtifactIntegrityError(
            "review generation does not match canonical finalist bindings"
        )
    judgments = _load_generation_judgments(
        store,
        experiment_id,
        finalists,
        manifest,
        comparison,
    )
    expected_generation_id = _review_generation_id(
        experiment_id=experiment_id,
        plan_hash=plan.plan_hash,
        comparison_facts_sha256=finalists.comparison_facts_sha256,
        judgments_sha256=manifest.judgments_sha256,
    )
    if expected_generation_id != generation_id:
        raise ArtifactIntegrityError("review generation ID is not canonical")
    runs, run_hashes = _load_current_plan_runs(
        store,
        experiment_id,
        plan,
    )
    recomputed_report = compare_outputs(
        runs,
        root,
        experiment_id=experiment_id,
    )
    if recomputed_report != comparison:
        raise ArtifactIntegrityError(
            "current run or output evidence differs from the review generation"
        )
    canonical = propose_finalists(
        comparison,
        plan,
        runs=runs,
        run_record_sha256=run_hashes,
        judgments=judgments,
    ).model_copy(
        update={
            "review_generation_id": generation_id,
            "review_generation_path": generation_relative,
            "comparison_facts_path": facts_relative,
            "judgments_path": finalists.judgments_path,
            "judgments_sha256": finalists.judgments_sha256,
        }
    )
    if canonical != finalists:
        raise ArtifactIntegrityError(
            "current finalists content is not canonical"
        )
    return CanonicalReview(
        root=root,
        finalists=finalists,
        finalists_bytes=finalists_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_path=(root / manifest_relative).resolve(),
        comparison=comparison,
        comparison_bytes=comparison_bytes,
        judgments=judgments,
        plan=plan,
        runs=runs,
    )


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
    if status.state is SelectionState.MISSING:
        raise SelectionRequiredError(
            status.reason or "a current human selection is required"
        )
    if status.state is not SelectionState.VALID or status.selection is None:
        raise StaleSelectionError(
            status.reason or "the human selection is stale"
        )
    return status.selection


def _durable_run_evidence(
    store: ExperimentStore,
    experiment_id: str,
    plan: ExperimentPlan,
    runs: tuple[RunRecord, ...],
) -> tuple[tuple[RunRecord, ...], dict[str, str]]:
    durable, hashes = _load_current_plan_runs(store, experiment_id, plan)
    durable_by_id = {record.run_id: record for record in durable}
    for expected in runs:
        record = durable_by_id.get(expected.run_id)
        if record is None:
            raise ArtifactIntegrityError(
                f"durable run record is missing or invalid: {expected.run_id}"
            )
        if record.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ArtifactIntegrityError(
                f"durable run record changed: {expected.run_id}"
            )
    return durable, hashes


def _load_generation_judgments(
    store: ExperimentStore,
    experiment_id: str,
    finalists: FinalistsArtifact,
    manifest: ReviewGenerationManifest,
    comparison: ComparisonReport,
) -> JudgmentSubmission | None:
    generation_id = manifest.generation_id
    expected_relative = f"reviews/{generation_id}/comparisons/judgments.json"
    values = (
        finalists.judgments_path,
        finalists.judgments_sha256,
        manifest.judgments_path,
        manifest.judgments_sha256,
    )
    if all(value is None for value in values):
        return None
    if (
        finalists.judgments_path != expected_relative
        or manifest.judgments_path != "comparisons/judgments.json"
        or finalists.judgments_sha256 is None
        or manifest.judgments_sha256 != finalists.judgments_sha256
    ):
        raise ArtifactIntegrityError(
            "review judgment bindings are inconsistent"
        )
    try:
        document = store.read_bytes(experiment_id, expected_relative)
        submission = JudgmentSubmission.model_validate_json(
            document,
            strict=True,
        )
    except (OSError, ValueError, ValidationError) as error:
        raise ArtifactIntegrityError(
            "review judgments are missing or invalid"
        ) from error
    if hashlib.sha256(document).hexdigest() != finalists.judgments_sha256:
        raise ArtifactIntegrityError("review judgment hash changed")
    if (
        submission.experiment_id != experiment_id
        or submission.facts_sha256 != comparison_facts_sha256(comparison)
    ):
        raise ArtifactIntegrityError(
            "review judgments do not match comparison facts"
        )
    return submission


def _load_current_plan_runs(
    store: ExperimentStore,
    experiment_id: str,
    plan: ExperimentPlan,
) -> tuple[tuple[RunRecord, ...], dict[str, str]]:
    variants = {variant.variant_id: variant for variant in plan.variants}
    records: list[RunRecord] = []
    hashes: dict[str, str] = {}
    seen_run_ids: set[str] = set()
    runs_root = store.root(experiment_id) / "runs"
    if path_component_is_link_or_junction(runs_root) or not runs_root.is_dir():
        raise ArtifactIntegrityError("runs path is not a safe local directory")
    try:
        paths = tuple(sorted(runs_root.glob("*.json"), key=lambda path: path.name))
    except OSError as error:
        raise ArtifactIntegrityError("durable runs cannot be listed") from error
    for path in paths:
        if path_component_is_link_or_junction(path) or not path.is_file():
            raise ArtifactIntegrityError(
                "durable run artifact is not a regular local file"
            )
        try:
            document = store.read_bytes(
                experiment_id,
                f"runs/{path.name}",
            )
            record = RunRecord.model_validate_json(document, strict=True)
        except (OSError, ValueError, ValidationError) as error:
            raise ArtifactIntegrityError(
                f"review run record is missing or invalid: {path.stem}"
            ) from error
        if record.run_id != path.stem or record.run_id in seen_run_ids:
            raise ArtifactIntegrityError(
                "durable run ID does not match its unique artifact path"
            )
        seen_run_ids.add(record.run_id)
        variant = variants.get(record.variant_id)
        if (
            variant is None
            or record.plan_hash != plan.plan_hash
            or record.exact_run_hash != variant.exact_run_hash
        ):
            continue
        if (
            record.experiment_id != experiment_id
            or (
                record.variant_snapshot is not None
                and thaw_json(record.variant_snapshot)
                != variant.model_dump(mode="json")
            )
        ):
            raise ArtifactIntegrityError(
                f"review run provenance changed: {record.run_id}"
            )
        records.append(record)
        hashes[record.run_id] = hashlib.sha256(document).hexdigest()
    return tuple(records), hashes


def _review_generation_id(
    *,
    experiment_id: str,
    plan_hash: str,
    comparison_facts_sha256: str,
    judgments_sha256: str | None,
) -> str:
    preimage = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "plan_hash": plan_hash,
        "comparison_facts_sha256": comparison_facts_sha256,
        "judgments_sha256": judgments_sha256,
    }
    return hashlib.sha256(_canonical_json_bytes(preimage)).hexdigest()


def _publish_review_generation(
    root: Path,
    generation_id: str,
    comparison_source: Path,
    finalists_bytes: bytes,
    manifest_bytes: bytes,
    *,
    include_judgments: bool,
    fault_injector: Callable[[str], None] | None,
) -> Path:
    reviews = root / "reviews"
    if reviews.exists() and (
        path_component_is_link_or_junction(reviews) or not reviews.is_dir()
    ):
        raise ArtifactIntegrityError("reviews path is not a safe directory")
    reviews.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".review-stage-", dir=reviews))
    destination = reviews / generation_id
    try:
        shutil.copytree(comparison_source, stage / "comparisons")
        staged_judgments = stage / "comparisons" / "judgments.json"
        if not include_judgments and staged_judgments.exists():
            staged_judgments.unlink()
        _write_new_bytes(stage / "finalists.json", finalists_bytes)
        _write_new_bytes(stage / "manifest.json", manifest_bytes)
        _fsync_tree(stage)
        _inject(fault_injector, "generation:before_publish")
        if destination.exists() or destination.is_symlink():
            _verify_generation_tree(stage, destination)
        else:
            try:
                _rename_directory_new(stage, destination)
            except FileExistsError:
                _verify_generation_tree(stage, destination)
        _fsync_directory(reviews)
        _inject(fault_injector, "generation:published")
        return destination
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _verify_generation_tree(expected: Path, actual: Path) -> None:
    if path_component_is_link_or_junction(actual) or not actual.is_dir():
        raise ReviewGenerationConflict(
            "review generation exists as an unsafe path"
        )
    if _tree_manifest(expected) != _tree_manifest(actual):
        raise ReviewGenerationConflict(
            "review generation already exists with different content"
        )


def _tree_manifest(root: Path) -> tuple[tuple[str, str, int], ...]:
    entries: list[tuple[str, str, int]] = []
    try:
        paths = tuple(root.rglob("*"))
    except OSError as error:
        raise ReviewGenerationConflict(
            "review generation cannot be read safely"
        ) from error
    for path in paths:
        if path_component_is_link_or_junction(path):
            raise ReviewGenerationConflict(
                "review generation contains a symlink or junction"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReviewGenerationConflict(
                "review generation contains a non-file entry"
            )
        try:
            digest, size = hash_stored_artifact(path)
        except OSError as error:
            raise ReviewGenerationConflict(
                "review generation changed while being verified"
            ) from error
        entries.append((path.relative_to(root).as_posix(), digest, size))
    return tuple(sorted(entries))


def _write_new_bytes(path: Path, content: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("review artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_document_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _inject(
    injector: Callable[[str], None] | None,
    point: str,
) -> None:
    if injector is not None:
        injector(point)


def _proposal(
    comparison: VariantComparison,
    run: RunRecord,
    variant: PlannedVariant,
    plan: ExperimentPlan,
    *,
    partial_approved: bool,
    duplicate_output_ids: set[str],
    judgment: VariantJudgment | None,
    comparison_report_sha256: str,
    run_record_sha256: str | None,
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
    outputs_by_ordinal = {output.ordinal: output for output in run.outputs}
    output_evidence = []
    for output in available:
        recorded = outputs_by_ordinal.get(output.output_ordinal)
        if (
            recorded is None
            or output.sha256 is None
            or output.bytes is None
            or output.artifact_relative_path is None
            or recorded.content_sha256 != output.sha256
            or recorded.size != output.bytes
            or recorded.media_type != output.declared_mime_type
            or recorded.local_relative_path != output.artifact_relative_path
        ):
            raise ValueError("comparison output evidence differs from its run record")
        output_evidence.append(
            FinalistOutputEvidence(
                output_id=output.output_id,
                ordinal=output.output_ordinal,
                sha256=output.sha256,
                bytes=output.bytes,
                media_type=recorded.media_type,
                artifact_relative_path=recorded.local_relative_path,
                node_id=recorded.node_id,
                provider_order=recorded.provider_order,
            )
        )
    record_sha256 = run_record_sha256 or hashlib.sha256(
        _run_record_bytes(run)
    ).hexdigest()
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
        comparison_report_sha256=comparison_report_sha256,
        run_evidence=FinalistRunEvidence(
            run_id=run.run_id,
            run_record_path=f"runs/{run.run_id}.json",
            run_record_sha256=record_sha256,
            state=run.state.value,
            provider_task_id=run.provider_task_id,
            exact_run_hash=variant.exact_run_hash,
            outputs=tuple(output_evidence),
        ),
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
        unavailable = next(
            (
                output.availability
                for output in comparison.outputs
                if output.availability != "available"
            ),
            None,
        )
        if unavailable is not None:
            return f"output_evidence_{unavailable}"
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
        and all(
            output.availability == "available"
            for output in comparison.outputs[: comparison.actual_output_count]
        )
    )


def _partial_behavior_is_approved(plan: ExperimentPlan) -> bool:
    return plan.partial_behavior_is_tested_capability


def _run_record_bytes(run: RunRecord) -> bytes:
    return (
        json.dumps(
            run.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


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
