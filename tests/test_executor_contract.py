import importlib
from datetime import UTC, datetime
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


def test_fake_executor_uses_configurable_poll_sequence_in_call_order(
    sample_variant,
    resolved_fixtures,
):
    executor = FakeExecutor(
        poll_sequence=(
            ProviderTaskState.QUEUED,
            ProviderTaskState.RUNNING,
            ProviderTaskState.SUCCESS,
        )
    )

    receipt = executor.submit(sample_variant, resolved_fixtures)
    states = tuple(executor.poll(receipt.provider_task_id).state for _ in range(3))
    second_receipt = executor.submit(sample_variant, resolved_fixtures)

    assert states == (
        ProviderTaskState.QUEUED,
        ProviderTaskState.RUNNING,
        ProviderTaskState.SUCCESS,
    )
    assert second_receipt.provider_task_id == "fake-task-2"
    assert executor.calls == ["submit", "poll", "poll", "poll", "submit"]
    assert executor.polls == ["fake-task-1", "fake-task-1", "fake-task-1"]


def test_fake_executor_raises_ambiguous_submission_without_task_id(
    sample_variant,
    resolved_fixtures,
):
    executor = FakeExecutor(ambiguous_submission=True)

    with pytest.raises(AmbiguousSubmissionError) as raised:
        executor.submit(sample_variant, resolved_fixtures)

    assert not hasattr(raised.value, "provider_task_id")
    assert executor.calls == ["submit"]


def test_fake_executor_preserves_partial_output_batches():
    first = ProviderOutput(node_id="9", url="memory://one.png", provider_order=0)
    second = ProviderOutput(node_id="9", url="memory://two.png", provider_order=1)
    executor = FakeExecutor(output_batches=((first,), (first, second)))

    assert executor.collect_outputs("fake-task-1") == (first,)
    assert executor.collect_outputs("fake-task-1") == (first, second)
    assert executor.collections == ["fake-task-1", "fake-task-1"]


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


def test_mutable_defaults_do_not_leak_between_records():
    first_candidate = _candidate()
    second_candidate = _candidate(workflow_id="workflow-2")
    first_report = CapabilityReport()
    second_report = CapabilityReport()

    first_candidate.provider_metadata["provider"] = "first"
    first_report.provider_metadata["provider"] = "first"

    assert second_candidate.provider_metadata == {}
    assert second_report.provider_metadata == {}
    assert first_candidate.provider_metadata is not second_candidate.provider_metadata
    assert first_report.provider_metadata is not second_report.provider_metadata


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


def test_executor_contract_modules_are_importable_by_name():
    for module_name in (
        "mp_comfyui_lab.discovery.models",
        "mp_comfyui_lab.experiments.models",
        "mp_comfyui_lab.executors.base",
    ):
        assert importlib.import_module(module_name).__name__ == module_name
