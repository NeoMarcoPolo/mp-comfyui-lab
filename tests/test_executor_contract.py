import importlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest

from helpers import FakeExecutor
from mp_comfyui_lab.discovery.models import (
    ContentClassification,
    CoverAsset,
    Engagement,
    WorkflowCandidate,
    WorkflowDetail,
)
from mp_comfyui_lab.executors.base import (
    AmbiguousSubmissionError,
    CancelResult,
    CapabilityReport,
    Executor,
    FetchedWorkflow,
    PollResult,
    ProviderOutput,
    ProviderTaskState,
    SearchPage,
    SearchRequest,
    SubmissionReceipt,
    ValidationIssue,
    ValidationReport,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.models import (
    MaterializedVariant,
    ResolvedFixture,
    ResolvedFixtures,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.store import ExperimentStore


@pytest.fixture
def sample_variant() -> MaterializedVariant:
    return MaterializedVariant(
        variant_id="variant-1",
        source_workflow_id="workflow-1",
        graph={"9": {"class_type": "SaveImage", "inputs": {}}},
        exact_run_hash="a" * 64,
        capability_hash="b" * 64,
        expected_output_count=1,
        expected_output_node_ids=("9",),
    )


@pytest.fixture
def resolved_fixtures(tmp_path: Path) -> ResolvedFixtures:
    return ResolvedFixtures(
        items=(
            ResolvedFixture(
                uri="vault://posted/input.png",
                path=tmp_path / "input.png",
                sha256="c" * 64,
                media_type="image/png",
                binding_role="reference",
            ),
        )
    )


def _candidate(**updates) -> WorkflowCandidate:
    values = {
        "executor": "fake",
        "workflow_id": "workflow-1",
        "canonical_url": "memory://workflow-1",
        "name": "Fixture workflow",
        "engagement": Engagement(),
        "content_classification": ContentClassification.UNKNOWN,
    }
    values.update(updates)
    return WorkflowCandidate(**values)


def _json_bearing_record(kind, value):
    if kind == "fetched_graph":
        return FetchedWorkflow(
            ref=WorkflowRef(
                executor="fake",
                workflow_id="workflow-1",
                canonical_url="memory://workflow-1",
            ),
            detail=WorkflowDetail(candidate=_candidate()),
            graph=value,
        )
    if kind == "variant_graph":
        return MaterializedVariant(
            variant_id="variant-json",
            source_workflow_id="workflow-json",
            graph=value,
            exact_run_hash="a" * 64,
            capability_hash="b" * 64,
            expected_output_count=1,
        )
    if kind == "candidate_metadata":
        return _candidate(provider_metadata=value)
    if kind == "capability_metadata":
        return CapabilityReport(provider_metadata=value)
    if kind == "receipt_metadata":
        return SubmissionReceipt(
            provider_task_id="fake-task-json",
            provider_metadata=value,
        )
    if kind == "poll_metadata":
        return PollResult(
            provider_task_id="fake-task-json",
            state=ProviderTaskState.RUNNING,
            provider_metadata=value,
        )
    if kind == "output_metadata":
        return ProviderOutput(
            url="memory://output",
            provider_metadata=value,
        )
    if kind == "run_error":
        return RunRecord(
            run_id="run-json",
            experiment_id="experiment-json",
            variant_id="variant-json",
            state=RunState.FAILED,
            expected_output_count=1,
            error=value,
        )
    raise AssertionError(f"unknown JSON-bearing record: {kind}")


def test_fake_executor_exposes_complete_lifecycle(
    sample_variant,
    resolved_fixtures,
):
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://one.png",
                media_type="image/png",
            ),
        )
    )

    assert executor.name == "fake"
    assert executor.validate_variant(sample_variant, resolved_fixtures).errors == ()
    receipt = executor.submit(sample_variant, resolved_fixtures)
    assert receipt.provider_task_id == "fake-task-1"
    assert executor.poll(receipt.provider_task_id).state is ProviderTaskState.SUCCESS
    assert len(executor.collect_outputs(receipt.provider_task_id)) == 1
    assert executor.cancel(receipt.provider_task_id).supported is True
    assert executor.calls == [
        "validate_variant",
        "submit",
        "poll",
        "collect_outputs",
        "cancel",
    ]


def test_fake_executor_uses_task_keyed_poll_sequences_in_call_order(
    sample_variant,
    resolved_fixtures,
):
    executor = FakeExecutor(
        poll_sequences={
            "fake-task-1": (
                ProviderTaskState.QUEUED,
                ProviderTaskState.SUCCESS,
            ),
            "fake-task-2": (
                ProviderTaskState.RUNNING,
                ProviderTaskState.FAILED,
            ),
        }
    )

    first = executor.submit(sample_variant, resolved_fixtures)
    second = executor.submit(sample_variant, resolved_fixtures)
    states = (
        executor.poll(first.provider_task_id).state,
        executor.poll(second.provider_task_id).state,
        executor.poll(first.provider_task_id).state,
        executor.poll(second.provider_task_id).state,
    )

    assert states == (
        ProviderTaskState.QUEUED,
        ProviderTaskState.RUNNING,
        ProviderTaskState.SUCCESS,
        ProviderTaskState.FAILED,
    )
    assert second.provider_task_id == "fake-task-2"
    assert executor.calls == ["submit", "submit", "poll", "poll", "poll", "poll"]
    assert executor.polls == [
        "fake-task-1",
        "fake-task-2",
        "fake-task-1",
        "fake-task-2",
    ]


def test_fake_executor_raises_ambiguous_submission_without_task_id(
    sample_variant,
    resolved_fixtures,
):
    executor = FakeExecutor(ambiguous_submission=True)

    with pytest.raises(AmbiguousSubmissionError) as raised:
        executor.submit(sample_variant, resolved_fixtures)

    assert not hasattr(raised.value, "provider_task_id")
    assert executor.calls == ["submit"]


def test_fake_executor_preserves_task_keyed_partial_output_scripts():
    first = ProviderOutput(node_id="9", url="memory://one.png", provider_order=0)
    second = ProviderOutput(node_id="9", url="memory://two.png", provider_order=1)
    executor = FakeExecutor(
        output_scripts={
            "fake-task-1": ((first,), (first, second)),
            "fake-task-2": ((second,),),
        }
    )

    assert executor.collect_outputs("fake-task-1") == (first,)
    assert executor.collect_outputs("fake-task-2") == (second,)
    assert executor.collect_outputs("fake-task-1") == (first, second)
    assert executor.collections == [
        "fake-task-1",
        "fake-task-2",
        "fake-task-1",
    ]


def test_fake_executor_rejects_mismatched_configured_poll_result():
    mismatched = PollResult(
        provider_task_id="fake-task-2",
        state=ProviderTaskState.RUNNING,
    )

    with pytest.raises(ValueError, match="fake-task-1"):
        FakeExecutor(poll_sequences={"fake-task-1": (mismatched,)})


def test_fake_executor_supports_configurable_cancel_result():
    configured = CancelResult(
        supported=False,
        cancelled=False,
        diagnostic="provider has no cancellation endpoint",
    )
    executor = FakeExecutor(cancel_result=configured)

    assert executor.cancel("fake-task-1") is configured
    assert executor.calls == ["cancel"]


def test_fake_executor_supports_retry_style_method_exceptions(
    sample_variant,
    resolved_fixtures,
):
    output = ProviderOutput(node_id="9", url="memory://one.png")
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"fake-task-1": ((output,),)},
        method_exceptions={
            "poll": (TimeoutError("poll timed out"),),
            "collect_outputs": (OSError("download failed"),),
        },
    )
    receipt = executor.submit(sample_variant, resolved_fixtures)

    with pytest.raises(TimeoutError, match="poll timed out"):
        executor.poll(receipt.provider_task_id)
    assert executor.poll(receipt.provider_task_id).state is ProviderTaskState.SUCCESS
    with pytest.raises(OSError, match="download failed"):
        executor.collect_outputs(receipt.provider_task_id)
    assert executor.collect_outputs(receipt.provider_task_id) == (output,)
    assert executor.calls == [
        "submit",
        "poll",
        "poll",
        "collect_outputs",
        "collect_outputs",
    ]


def test_fake_executor_success_scripts_every_submitted_task(
    sample_variant,
    resolved_fixtures,
):
    output = ProviderOutput(node_id="9", url="memory://one.png")
    executor = FakeExecutor.success(outputs=(output,))

    first = executor.submit(sample_variant, resolved_fixtures)
    second = executor.submit(sample_variant, resolved_fixtures)

    assert executor.poll(first.provider_task_id).state is ProviderTaskState.SUCCESS
    assert executor.poll(second.provider_task_id).state is ProviderTaskState.SUCCESS
    assert executor.collect_outputs(first.provider_task_id) == (output,)
    assert executor.collect_outputs(second.provider_task_id) == (output,)


def test_discovery_models_preserve_absent_engagement_and_compose_detail():
    candidate = _candidate(
        covers=(CoverAsset(url="memory://cover.png", width=640, height=480),)
    )
    detail = WorkflowDetail(
        candidate=candidate,
        node_count=1,
        primitive_nodes=("LoadImage",),
        custom_nodes=("CustomSampler",),
        used_models=("model.safetensors",),
        preview=candidate.covers[0],
        webapp_id="app-1",
    )

    assert candidate.engagement.likes is None
    assert candidate.tags == ()
    assert detail.candidate is candidate
    assert detail.node_count == 1


@pytest.mark.parametrize(
    ("ref_updates", "candidate_updates"),
    [
        ({"executor": "other"}, {}),
        ({"workflow_id": "other-workflow"}, {}),
        ({"canonical_url": "memory://other-workflow"}, {}),
    ],
)
def test_fetched_workflow_rejects_ref_detail_identity_mismatch(
    ref_updates,
    candidate_updates,
):
    ref_values = {
        "executor": "fake",
        "workflow_id": "workflow-1",
        "canonical_url": "memory://workflow-1",
        **ref_updates,
    }
    candidate = _candidate(**candidate_updates)

    with pytest.raises(ValidationError, match="match"):
        FetchedWorkflow(
            ref=WorkflowRef(**ref_values),
            detail=WorkflowDetail(candidate=candidate),
            graph={"9": {"class_type": "SaveImage", "inputs": {}}},
        )


def test_fetched_workflow_allows_absent_ref_canonical_url():
    fetched = FetchedWorkflow(
        ref=WorkflowRef(executor="fake", workflow_id="workflow-1"),
        detail=WorkflowDetail(candidate=_candidate()),
        graph={"9": {"class_type": "SaveImage", "inputs": {}}},
    )

    assert fetched.ref.canonical_url is None


def test_resolved_fixtures_by_role_preserves_item_order(tmp_path):
    fixtures = ResolvedFixtures(
        items=tuple(
            ResolvedFixture(
                uri=f"memory://{index}",
                path=tmp_path / f"{index}.png",
                sha256=f"{index:x}" * 64,
                binding_role=role,
                binding_index=index,
            )
            for index, role in ((1, "reference"), (2, "mask"), (3, "reference"))
        )
    )

    assert [item.binding_index for item in fixtures.by_role("reference")] == [1, 3]


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (Engagement, {"likes": -1}),
        (CoverAsset, {"url": "memory://cover.png", "width": 0}),
        (WorkflowDetail, {"candidate": _candidate(), "node_count": -1}),
        (
            ResolvedFixture,
            {
                "uri": "memory://input",
                "path": Path.cwd() / "input.png",
                "sha256": "a" * 64,
                "binding_index": -1,
            },
        ),
        (
            SearchPage,
            {"records": (), "total": -1, "page": 1, "size": 1, "has_next": False},
        ),
        (ProviderOutput, {"url": "memory://output", "provider_order": -1}),
    ],
)
def test_models_reject_invalid_counts(model, values):
    with pytest.raises(ValidationError):
        model(**values)


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (Engagement, {"likes": "1"}),
        (SearchRequest, {"query": "qwen", "page": "1"}),
        (ProviderOutput, {"url": "memory://output", "provider_order": "1"}),
        (
            RunRecord,
            {
                "run_id": "run-1",
                "experiment_id": "experiment-1",
                "variant_id": "variant-1",
                "state": RunState.PLANNED,
                "expected_output_count": "1",
            },
        ),
        (
            RunRecord,
            {
                "run_id": "run-1",
                "experiment_id": "experiment-1",
                "variant_id": "variant-1",
                "state": RunState.PLANNED,
                "expected_output_count": 1,
                "actual_output_count": "0",
            },
        ),
    ],
)
def test_python_constructors_reject_numeric_string_coercion(model, values):
    with pytest.raises(ValidationError):
        model(**values)


def test_materialized_variant_rejects_string_output_count(sample_variant):
    values = sample_variant.model_dump()
    values["expected_output_count"] = "1"

    with pytest.raises(ValidationError):
        MaterializedVariant(**values)


@pytest.mark.parametrize("bad_hash", ["a" * 63, "A" * 64, "g" * 64])
def test_materialized_variant_rejects_invalid_hashes(sample_variant, bad_hash):
    with pytest.raises(ValidationError):
        sample_variant.model_copy(update={"exact_run_hash": bad_hash}).model_validate(
            {
                **sample_variant.model_dump(),
                "exact_run_hash": bad_hash,
            }
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"variant_id": ""},
        {"source_workflow_id": "   "},
        {"graph": {}},
        {"expected_output_count": 0},
    ],
)
def test_materialized_variant_rejects_empty_identity_or_graph(
    sample_variant,
    updates,
):
    with pytest.raises(ValidationError):
        MaterializedVariant(**{**sample_variant.model_dump(), **updates})


@pytest.mark.parametrize(
    "state",
    [
        RunState.SUBMITTED,
        RunState.RUNNING,
        RunState.SUCCESS,
        RunState.CANCELLED,
    ],
)
def test_active_run_states_require_provider_task_id(state):
    with pytest.raises(ValidationError, match="provider_task_id"):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=state,
            expected_output_count=1,
        )


def test_planned_run_rejects_provider_task_id():
    with pytest.raises(ValidationError, match="provider_task_id"):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.PLANNED,
            provider_task_id="fake-task-1",
            expected_output_count=1,
        )


@pytest.mark.parametrize("state", [RunState.FAILED, RunState.INCOMPLETE])
@pytest.mark.parametrize("provider_task_id", [None, "fake-task-1"])
def test_failed_and_incomplete_runs_allow_optional_task_id(
    state,
    provider_task_id,
):
    record = RunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        variant_id="variant-1",
        state=state,
        provider_task_id=provider_task_id,
        expected_output_count=1,
    )

    assert record.provider_task_id == provider_task_id


def test_success_run_requires_at_least_expected_output_count():
    with pytest.raises(ValidationError, match="actual_output_count"):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.SUCCESS,
            provider_task_id="fake-task-1",
            expected_output_count=2,
            actual_output_count=1,
        )


def test_run_transition_revalidates_complete_record():
    created_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    planned = RunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        variant_id="variant-1",
        state=RunState.PLANNED,
        expected_output_count=1,
        created_at=created_at,
    )

    submitted = planned.transition(
        RunState.SUBMITTED,
        provider_task_id="fake-task-1",
        submitted_at=created_at + timedelta(seconds=1),
    )
    running = submitted.transition(
        RunState.RUNNING,
        started_at=created_at + timedelta(seconds=2),
    )
    success = running.transition(
        RunState.SUCCESS,
        actual_output_count=1,
        completed_at=created_at + timedelta(seconds=3),
        updated_at=created_at + timedelta(seconds=4),
    )

    assert planned.state is RunState.PLANNED
    assert submitted.state is RunState.SUBMITTED
    assert running.state is RunState.RUNNING
    assert success.state is RunState.SUCCESS
    assert success.actual_output_count == 1
    with pytest.raises(ValueError):
        planned.transition(RunState.RUNNING)


def _run_for_transition(state: RunState, **updates) -> RunRecord:
    values = {
        "run_id": "run-1",
        "experiment_id": "experiment-1",
        "variant_id": "variant-1",
        "state": state,
        "expected_output_count": 1,
    }
    if state in {
        RunState.SUBMITTED,
        RunState.RUNNING,
        RunState.SUCCESS,
        RunState.CANCELLED,
    }:
        values["provider_task_id"] = "fake-task-1"
    if state is RunState.SUCCESS:
        values["actual_output_count"] = 1
    values.update(updates)
    return RunRecord(**values)


@pytest.mark.parametrize(
    ("source_state", "destination", "updates"),
    [
        (RunState.SUCCESS, RunState.RUNNING, {}),
        (
            RunState.FAILED,
            RunState.SUBMITTED,
            {"provider_task_id": "fake-task-1"},
        ),
    ],
)
def test_run_transition_rejects_reverse_edges(source_state, destination, updates):
    record = _run_for_transition(source_state)

    with pytest.raises(ValueError, match="transition"):
        record.transition(destination, **updates)


@pytest.mark.parametrize("source_state", [RunState.SUCCESS, RunState.CANCELLED])
@pytest.mark.parametrize("destination", list(RunState))
def test_success_and_cancelled_runs_are_terminal(source_state, destination):
    record = _run_for_transition(source_state, actual_output_count=1)

    with pytest.raises(ValueError, match="transition"):
        record.transition(destination)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "run-2"),
        ("experiment_id", "experiment-2"),
        ("variant_id", "variant-2"),
        ("expected_output_count", 2),
        ("created_at", datetime(2026, 7, 13, 12, 1, tzinfo=UTC)),
    ],
)
def test_run_transition_cannot_change_durable_identity(field, replacement):
    record = _run_for_transition(
        RunState.PLANNED,
        created_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match=field):
        record.transition(
            RunState.SUBMITTED,
            provider_task_id="fake-task-1",
            **{field: replacement},
        )


def test_run_transition_cannot_replace_provider_task_id():
    record = _run_for_transition(RunState.SUBMITTED)

    with pytest.raises(ValueError, match="provider_task_id"):
        record.transition(
            RunState.RUNNING,
            provider_task_id="fake-task-2",
        )


@pytest.mark.parametrize(
    ("source_state", "destination", "updates"),
    [
        (
            RunState.PLANNED,
            RunState.SUBMITTED,
            {"provider_task_id": "fake-task-1"},
        ),
        (
            RunState.PLANNED,
            RunState.INCOMPLETE,
            {"error": {"kind": "ambiguous_submission"}},
        ),
        (
            RunState.PLANNED,
            RunState.FAILED,
            {"error": {"kind": "provider_rejection"}},
        ),
        (RunState.SUBMITTED, RunState.RUNNING, {}),
        (RunState.SUBMITTED, RunState.SUCCESS, {"actual_output_count": 1}),
        (RunState.SUBMITTED, RunState.FAILED, {}),
        (RunState.SUBMITTED, RunState.INCOMPLETE, {}),
        (RunState.SUBMITTED, RunState.CANCELLED, {}),
        (RunState.RUNNING, RunState.SUCCESS, {"actual_output_count": 1}),
        (RunState.RUNNING, RunState.FAILED, {}),
        (RunState.RUNNING, RunState.INCOMPLETE, {}),
        (RunState.RUNNING, RunState.CANCELLED, {}),
    ],
)
def test_run_transition_accepts_normal_edges(source_state, destination, updates):
    record = _run_for_transition(source_state)

    transitioned = record.transition(destination, **updates)

    assert transitioned.state is destination


def test_planned_run_can_record_confirmed_failure_without_provider_task_id():
    record = _run_for_transition(RunState.PLANNED)

    failed = record.transition(
        RunState.FAILED,
        error={"kind": "provider_rejection"},
    )

    assert failed.state is RunState.FAILED
    assert failed.provider_task_id is None
    assert failed.error == {"kind": "provider_rejection"}


@pytest.mark.parametrize("destination", list(RunState))
def test_confirmed_planned_failure_is_terminal(destination):
    record = _run_for_transition(RunState.PLANNED)
    failed = record.transition(
        RunState.FAILED,
        error={"kind": "provider_rejection"},
    )

    with pytest.raises(ValueError, match="transition"):
        failed.transition(destination)


def test_planned_to_failed_rejects_provider_task_id():
    record = _run_for_transition(RunState.PLANNED)

    with pytest.raises(ValueError, match="provider_task_id"):
        record.transition(
            RunState.FAILED,
            provider_task_id="fake-task-1",
            error={"kind": "provider_rejection"},
        )


def test_planned_to_failed_rejects_ambiguous_submission_error():
    record = _run_for_transition(RunState.PLANNED)

    with pytest.raises(ValueError, match="ambiguous_submission"):
        record.transition(
            RunState.FAILED,
            error={"kind": "ambiguous_submission"},
        )


@pytest.mark.parametrize("error", [None, {"kind": "partial_outputs"}])
def test_planned_to_incomplete_requires_ambiguous_submission_error(error):
    record = _run_for_transition(RunState.PLANNED)

    with pytest.raises(ValueError, match="ambiguous_submission"):
        record.transition(RunState.INCOMPLETE, error=error)


def test_planned_ambiguous_incomplete_rejects_known_task_id():
    record = _run_for_transition(RunState.PLANNED)

    with pytest.raises(ValueError, match="provider_task_id"):
        record.transition(
            RunState.INCOMPLETE,
            provider_task_id="fake-task-1",
            error={"kind": "ambiguous_submission"},
        )


@pytest.mark.parametrize(
    "destination",
    [
        RunState.SUBMITTED,
        RunState.RUNNING,
        RunState.SUCCESS,
        RunState.FAILED,
        RunState.CANCELLED,
    ],
)
def test_partial_output_incomplete_run_cannot_reopen(destination):
    record = _run_for_transition(
        RunState.INCOMPLETE,
        provider_task_id="fake-task-1",
        expected_output_count=2,
        actual_output_count=1,
        error={"kind": "partial_outputs"},
    )
    updates = {"actual_output_count": 2} if destination is RunState.SUCCESS else {}

    with pytest.raises(ValueError, match="transition"):
        record.transition(destination, **updates)


@pytest.mark.parametrize(
    ("destination", "updates"),
    [
        (RunState.SUBMITTED, {}),
        (RunState.RUNNING, {}),
        (RunState.SUCCESS, {"actual_output_count": 1}),
        (RunState.FAILED, {}),
        (RunState.CANCELLED, {}),
    ],
)
def test_ambiguous_incomplete_run_can_reconcile(destination, updates):
    record = _run_for_transition(
        RunState.INCOMPLETE,
        error={"kind": "ambiguous_submission"},
    )

    reconciled = record.transition(
        destination,
        provider_task_id="fake-task-1",
        **updates,
    )

    assert reconciled.state is destination
    assert reconciled.provider_task_id == "fake-task-1"


@pytest.mark.parametrize("error", [None, {"kind": "partial_outputs"}])
def test_incomplete_reconciliation_requires_ambiguous_submission_error(error):
    record = _run_for_transition(RunState.INCOMPLETE, error=error)

    with pytest.raises(ValueError, match="ambiguous_submission"):
        record.transition(
            RunState.RUNNING,
            provider_task_id="fake-task-1",
        )


def test_ambiguous_incomplete_reconciliation_requires_attached_task_id():
    record = _run_for_transition(
        RunState.INCOMPLETE,
        error={"kind": "ambiguous_submission"},
    )

    with pytest.raises(ValueError, match="provider_task_id"):
        record.transition(RunState.FAILED)


def test_identified_ambiguous_incomplete_run_cannot_reconcile_again():
    record = _run_for_transition(
        RunState.INCOMPLETE,
        provider_task_id="fake-task-1",
        error={"kind": "ambiguous_submission"},
    )

    with pytest.raises(ValueError, match="provider_task_id"):
        record.transition(
            RunState.RUNNING,
            provider_task_id="fake-task-1",
        )


def test_ambiguous_incomplete_reconciliation_revalidates_destination():
    record = _run_for_transition(
        RunState.INCOMPLETE,
        expected_output_count=2,
        actual_output_count=1,
        error={"kind": "ambiguous_submission"},
    )

    with pytest.raises(ValidationError, match="actual_output_count"):
        record.transition(
            RunState.SUCCESS,
            provider_task_id="fake-task-1",
        )


@pytest.mark.parametrize("state", list(RunState))
def test_run_transition_rejects_same_state(state):
    record = _run_for_transition(state)

    with pytest.raises(ValueError, match="same state"):
        record.transition(state)


def test_run_transition_allows_state_related_updates():
    created_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    record = _run_for_transition(RunState.PLANNED, created_at=created_at)

    submitted = record.transition(
        RunState.SUBMITTED,
        provider_task_id="fake-task-1",
        actual_output_count=1,
        error={"kind": "accepted"},
        submitted_at=created_at + timedelta(seconds=1),
    )
    running = submitted.transition(
        RunState.RUNNING,
        error=None,
        started_at=created_at + timedelta(seconds=2),
    )
    failed = running.transition(
        RunState.FAILED,
        error={"kind": "provider_failure"},
        completed_at=created_at + timedelta(seconds=3),
    )

    assert submitted.actual_output_count == 1
    assert submitted.error == {"kind": "accepted"}
    assert submitted.submitted_at == created_at + timedelta(seconds=1)
    assert running.error is None
    assert running.started_at == created_at + timedelta(seconds=2)
    assert failed.error == {"kind": "provider_failure"}
    assert failed.completed_at == created_at + timedelta(seconds=3)


def test_state_changes_must_not_use_unvalidated_model_copy():
    planned = RunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        variant_id="variant-1",
        state=RunState.PLANNED,
        expected_output_count=1,
    )

    invalid_copy = planned.model_copy(update={"state": RunState.RUNNING})

    assert invalid_copy.state is RunState.RUNNING
    assert invalid_copy.provider_task_id is None
    assert "model_copy(update=...)" in RunRecord.transition.__doc__


def test_run_record_allows_more_actual_outputs_than_expected():
    record = RunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        variant_id="variant-1",
        state=RunState.SUCCESS,
        provider_task_id="fake-task-1",
        expected_output_count=1,
        actual_output_count=2,
    )

    assert record.actual_output_count == 2


def test_workflow_and_run_datetimes_normalize_to_utc():
    offset = timezone(timedelta(hours=-4))
    source = datetime(2026, 7, 13, 8, 0, tzinfo=offset)
    candidate = _candidate(published_at=source)
    run = RunRecord(
        run_id="run-1",
        experiment_id="experiment-1",
        variant_id="variant-1",
        state=RunState.FAILED,
        expected_output_count=1,
        created_at=source,
        updated_at=source + timedelta(minutes=1),
    )

    assert candidate.published_at == datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    assert candidate.published_at.tzinfo is UTC
    assert run.created_at.tzinfo is UTC
    assert run.updated_at.tzinfo is UTC


def test_workflow_published_at_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        _candidate(published_at=datetime(2026, 7, 13, 12, 0))


@pytest.mark.parametrize(
    "field",
    ["created_at", "submitted_at", "started_at", "completed_at", "updated_at"],
)
def test_run_timestamps_reject_naive_datetimes(field):
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.FAILED,
            expected_output_count=1,
            **{field: datetime(2026, 7, 13, 12, 0)},
        )


@pytest.mark.parametrize(
    ("earlier_field", "later_field"),
    [
        ("created_at", "submitted_at"),
        ("submitted_at", "started_at"),
        ("started_at", "completed_at"),
        ("completed_at", "updated_at"),
    ],
)
def test_run_timestamps_must_be_chronologically_nondecreasing(
    earlier_field,
    later_field,
):
    later = datetime(2026, 7, 13, 12, 1, tzinfo=UTC)
    earlier = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="chronological"):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.FAILED,
            expected_output_count=1,
            **{earlier_field: later, later_field: earlier},
        )


def test_submission_receipt_requires_provider_task_state_enum():
    receipt = SubmissionReceipt(
        provider_task_id="fake-task-1",
        provider_state=ProviderTaskState.SUBMITTED,
    )

    assert receipt.provider_state is ProviderTaskState.SUBMITTED
    with pytest.raises(ValidationError):
        SubmissionReceipt(
            provider_task_id="fake-task-1",
            provider_state="NOT_A_PROVIDER_STATE",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"path": Path("relative.png")},
        {"binding_role": "   "},
        {"uri": "   "},
        {"sha256": "A" * 64},
    ],
)
def test_resolved_fixture_rejects_invalid_identity_fields(tmp_path, updates):
    values = {
        "uri": "memory://input",
        "path": tmp_path / "missing.png",
        "sha256": "a" * 64,
    }

    with pytest.raises(ValidationError):
        ResolvedFixture(**{**values, **updates})


def test_resolved_fixture_strips_role_without_requiring_existence(tmp_path):
    missing_path = tmp_path / "missing.png"

    fixture = ResolvedFixture(
        uri="memory://input",
        path=missing_path,
        sha256="a" * 64,
        binding_role="  reference  ",
    )

    assert fixture.path == missing_path
    assert fixture.binding_role == "reference"
    assert not fixture.path.exists()


def test_cancel_result_rejects_cancelled_when_unsupported():
    with pytest.raises(ValidationError):
        CancelResult(supported=False, cancelled=True, diagnostic="not available")


def test_validation_report_is_valid_only_without_errors():
    warning = ValidationIssue(code="preview", message="Preview may differ")
    error = ValidationIssue(code="missing_node", message="Node is unavailable")

    assert ValidationReport(warnings=(warning,)).is_valid is True
    assert ValidationReport(errors=(error,)).is_valid is False


def test_contract_models_are_frozen_and_reject_unknown_fields():
    request = SearchRequest(query="qwen")

    with pytest.raises(ValidationError):
        request.page = 2
    with pytest.raises(ValidationError):
        SearchRequest(query="qwen", unexpected=True)
    with pytest.raises(ValidationError):
        _candidate(unexpected=True)
    with pytest.raises(ValidationError):
        ResolvedFixtures(items=(), unexpected=True)
    with pytest.raises(ValidationError):
        ProviderOutput(url="memory://output", unexpected=True)
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.PLANNED,
            expected_output_count=1,
            unexpected=True,
        )


def test_strict_records_round_trip_durable_json_values(tmp_path):
    timestamp = datetime(2026, 7, 13, 12, 30, tzinfo=UTC)
    records = (
        _candidate(
            published_at=timestamp,
            tags=("qwen", "angles"),
            covers=(CoverAsset(url="memory://cover.png", width=640, height=480),),
        ),
        ResolvedFixtures(
            items=(
                ResolvedFixture(
                    uri="memory://input",
                    path=tmp_path / "missing.png",
                    sha256="a" * 64,
                    binding_role="reference",
                ),
            )
        ),
        RunRecord(
            run_id="run-1",
            experiment_id="experiment-1",
            variant_id="variant-1",
            state=RunState.SUBMITTED,
            provider_task_id="fake-task-1",
            expected_output_count=1,
            created_at=timestamp,
        ),
        SubmissionReceipt(
            provider_task_id="fake-task-1",
            provider_state=ProviderTaskState.QUEUED,
        ),
    )

    for record in records:
        restored = type(record).model_validate_json(record.model_dump_json())
        assert restored == record


@pytest.mark.parametrize(
    "record_kind",
    [
        "fetched_graph",
        "variant_graph",
        "candidate_metadata",
        "capability_metadata",
        "receipt_metadata",
        "poll_metadata",
        "output_metadata",
        "run_error",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        Path("not-json.txt"),
        object(),
        {1: "non-string-key"},
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=["path", "object", "non-string-key", "nan", "positive-inf", "negative-inf"],
)
def test_json_bearing_records_reject_non_json_or_non_finite_values(
    record_kind,
    invalid_value,
):
    with pytest.raises(ValidationError):
        _json_bearing_record(record_kind, {"nested": invalid_value})


@pytest.mark.parametrize(
    "record_kind",
    [
        "candidate_metadata",
        "capability_metadata",
        "receipt_metadata",
        "poll_metadata",
        "output_metadata",
        "run_error",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        Path("not-json.txt"),
        object(),
        {1: "non-string-key"},
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=["path", "object", "non-string-key", "nan", "positive-inf", "negative-inf"],
)
def test_sanitization_does_not_hide_invalid_json_values(
    record_kind,
    invalid_value,
):
    with pytest.raises(ValidationError):
        _json_bearing_record(record_kind, {"api_key": invalid_value})


def test_json_bearing_records_are_deeply_frozen_and_detached_from_sources():
    fetched_graph_source = {
        "9": {
            "class_type": "SaveImage",
            "inputs": {"weights": [1, 2]},
        }
    }
    graph_source = {
        "9": {
            "class_type": "SaveImage",
            "inputs": {"weights": [1, 2]},
        }
    }
    metadata_source = {"nested": {"labels": ["one", "two"]}}
    error_source = {"context": {"attempts": [1, 2]}}

    fetched = _json_bearing_record("fetched_graph", fetched_graph_source)
    variant = _json_bearing_record("variant_graph", graph_source)
    candidate = _json_bearing_record("candidate_metadata", metadata_source)
    run = _json_bearing_record("run_error", error_source)
    fetched_graph_source["9"]["inputs"]["weights"].append(3)
    graph_source["9"]["inputs"]["weights"].append(3)
    metadata_source["nested"]["labels"].append("three")
    error_source["context"]["attempts"].append(3)

    assert fetched.graph["9"]["inputs"]["weights"] == (1, 2)
    assert variant.graph["9"]["inputs"]["weights"] == (1, 2)
    assert candidate.provider_metadata["nested"]["labels"] == ("one", "two")
    assert run.error["context"]["attempts"] == (1, 2)
    for value in (
        fetched.graph,
        fetched.graph["9"]["inputs"],
        variant.graph,
        variant.graph["9"]["inputs"],
        candidate.provider_metadata,
        candidate.provider_metadata["nested"],
        run.error,
        run.error["context"],
    ):
        assert isinstance(value, Mapping)
        with pytest.raises(TypeError):
            value["mutate"] = True
    with pytest.raises(TypeError):
        variant.graph._data = {}


def test_frozen_json_serializes_to_plain_json_and_has_schema():
    variant = _json_bearing_record(
        "variant_graph",
        {"9": {"inputs": {"weights": [1, 2]}}},
    )

    dumped = variant.model_dump(mode="json")
    assert type(dumped["graph"]) is dict
    assert type(dumped["graph"]["9"]["inputs"]["weights"]) is list
    assert json.loads(variant.model_dump_json()) == dumped
    assert MaterializedVariant.model_validate_json(variant.model_dump_json()) == variant
    for model, field in (
        (FetchedWorkflow, "graph"),
        (MaterializedVariant, "graph"),
        (WorkflowCandidate, "provider_metadata"),
        (CapabilityReport, "provider_metadata"),
        (SubmissionReceipt, "provider_metadata"),
        (PollResult, "provider_metadata"),
        (ProviderOutput, "provider_metadata"),
        (RunRecord, "error"),
    ):
        field_schema = model.model_json_schema()["properties"][field]
        assert field_schema.get("type") == "object" or any(
            option.get("type") == "object"
            for option in field_schema.get("anyOf", ())
        )


def test_json_model_configs_disallow_non_finite_numbers():
    for model in (WorkflowCandidate, MaterializedVariant, ProviderOutput, RunRecord):
        assert model.model_config["allow_inf_nan"] is False


def test_error_and_provider_metadata_serialize_sanitized_and_persist_safely(
    tmp_path,
):
    signed_url = "https://cdn.test/output.png?X-Amz-Signature=secret"
    output = ProviderOutput(
        url="memory://output",
        provider_metadata={
            "api_key": "provider-secret",
            "signedUrl": signed_url,
        },
    )
    run = RunRecord(
        run_id="run-json",
        experiment_id="experiment-json",
        variant_id="variant-json",
        state=RunState.FAILED,
        expected_output_count=1,
        error={
            "Authorization": "Bearer secret",
            "outputUrl": signed_url,
        },
    )

    assert json.loads(output.model_dump_json())["provider_metadata"] == {
        "api_key": "[REDACTED]",
        "signedUrl": "https://cdn.test/output.png",
    }
    assert json.loads(run.model_dump_json())["error"] == {
        "Authorization": "[REDACTED]",
        "outputUrl": "https://cdn.test/output.png",
    }
    store = ExperimentStore(tmp_path / "experiments")
    store.create("json-001")
    store.write_json(
        "json-001",
        "runs/run-json.json",
        {
            "run": run.model_dump(mode="json"),
            "output": output.model_dump(mode="json"),
        },
    )
    persisted = store.read_json("json-001", "runs/run-json.json")
    assert persisted["run"]["error"] == {
        "Authorization": "[REDACTED]",
        "outputUrl": "https://cdn.test/output.png",
    }
    assert persisted["output"]["provider_metadata"] == {
        "api_key": "[REDACTED]",
        "signedUrl": "https://cdn.test/output.png",
    }


def test_mutable_defaults_do_not_leak_between_records():
    first_candidate = _candidate()
    second_candidate = _candidate(workflow_id="workflow-2")
    first_report = CapabilityReport()
    second_report = CapabilityReport()

    assert second_candidate.provider_metadata == {}
    assert second_report.provider_metadata == {}
    assert first_candidate.provider_metadata is not second_candidate.provider_metadata
    assert first_report.provider_metadata is not second_report.provider_metadata
    with pytest.raises(TypeError):
        first_candidate.provider_metadata["provider"] = "first"
    with pytest.raises(TypeError):
        first_report.provider_metadata["provider"] = "first"


def test_provider_metadata_is_sanitized_and_json_like():
    candidate = _candidate(
        provider_metadata={
            "api_key": "secret",
            "previewUrl": "https://cdn.test/preview.png?token=secret",
        }
    )
    output = ProviderOutput(
        url="memory://output",
        provider_metadata={"Authorization": "Bearer secret"},
    )

    assert candidate.provider_metadata == {
        "api_key": "[REDACTED]",
        "previewUrl": "https://cdn.test/preview.png",
    }
    assert output.provider_metadata == {"Authorization": "[REDACTED]"}
    with pytest.raises(ValidationError):
        CapabilityReport(provider_metadata={"not_json": Path("local.txt")})


def test_exact_run_state_and_provider_state_values():
    assert tuple(RunState) == (
        RunState.PLANNED,
        RunState.SUBMITTED,
        RunState.RUNNING,
        RunState.SUCCESS,
        RunState.FAILED,
        RunState.INCOMPLETE,
        RunState.CANCELLED,
    )
    assert tuple(ProviderTaskState) == (
        ProviderTaskState.SUBMITTED,
        ProviderTaskState.QUEUED,
        ProviderTaskState.RUNNING,
        ProviderTaskState.SUCCESS,
        ProviderTaskState.FAILED,
        ProviderTaskState.CANCELLED,
        ProviderTaskState.UNKNOWN,
    )


def test_core_imports_do_not_load_runninghub_even_when_it_becomes_available():
    script = """
import importlib.abc
import sys

class RejectRunningHubImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'mp_comfyui_lab.executors.runninghub':
            raise AssertionError('core imported RunningHub')
        return None

sys.meta_path.insert(0, RejectRunningHubImport())
from mp_comfyui_lab.discovery.models import WorkflowCandidate
from mp_comfyui_lab.experiments.models import MaterializedVariant
from mp_comfyui_lab.executors.base import Executor
assert WorkflowCandidate.__module__ == 'mp_comfyui_lab.discovery.models'
assert MaterializedVariant.__module__ == 'mp_comfyui_lab.experiments.models'
assert Executor.__module__ == 'mp_comfyui_lab.executors.base'
assert 'mp_comfyui_lab.executors.runninghub' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_executor_abc_cannot_instantiate_incomplete_implementation():
    class IncompleteExecutor(Executor):
        name = "incomplete"

    with pytest.raises(TypeError, match="abstract"):
        IncompleteExecutor()


def test_executor_name_is_abstract_and_fake_implements_property():
    class MissingNameExecutor(Executor):
        def search(self, request):
            raise NotImplementedError

        def fetch_workflow(self, ref):
            raise NotImplementedError

        def inspect_capabilities(self, workflow):
            raise NotImplementedError

        def validate_variant(self, variant, fixtures):
            raise NotImplementedError

        def submit(self, variant, fixtures):
            raise NotImplementedError

        def poll(self, provider_task_id):
            raise NotImplementedError

        def collect_outputs(self, provider_task_id):
            raise NotImplementedError

        def cancel(self, provider_task_id):
            raise NotImplementedError

    assert "name" in Executor.__abstractmethods__
    with pytest.raises(TypeError, match="name"):
        MissingNameExecutor()
    assert isinstance(FakeExecutor.name, property)
    assert FakeExecutor().name == "fake"


def test_executor_contract_modules_are_importable_by_name():
    for module_name in (
        "mp_comfyui_lab.discovery.models",
        "mp_comfyui_lab.experiments.models",
        "mp_comfyui_lab.executors.base",
    ):
        assert importlib.import_module(module_name).__name__ == module_name
