from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from mp_runninghub_api.errors import RunningHubApiError
import pytest
import requests

from helpers import FakeExecutor, valid_media_bytes
from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.executors.base import (
    ExecutorError,
    PollResult,
    ProviderOutput,
    ProviderSchemaError,
    ProviderTaskState,
    ValidationIssue,
    ValidationReport,
)
from mp_comfyui_lab.executors.runninghub import RunningHubExecutor
from mp_comfyui_lab.experiments.models import (
    ExperimentPlan,
    FixtureFingerprint,
    OutputRecord,
    PlannedVariant,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.json_types import thaw_json


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _persist_plan(
    tmp_path: Path,
    *,
    experiment_id: str = "plan-approved",
    with_fixture: bool = False,
    expected_output_count: int = 1,
    output_node_ids: tuple[str, ...] = ("9",),
    expected_media_types: tuple[str, ...] = ("image/png",),
) -> tuple[LabConfig, ExperimentPlan]:
    config = LabConfig.model_validate(
        {"experiments_root": tmp_path / "experiments"},
        context={"base_dir": tmp_path},
    )
    store = ExperimentStore(config.experiments_root)
    store.create(experiment_id)
    fixture_fingerprints = ()
    graph_inputs = {"filename_prefix": "out"}
    if with_fixture:
        fixture_bytes = b"planned fixture bytes"
        (tmp_path / "portrait.png").write_bytes(fixture_bytes)
        fixture_fingerprints = (
            FixtureFingerprint(
                uri="portrait.png",
                binding_index=0,
                size=len(fixture_bytes),
                media_type="image/png",
                sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            ),
        )
        graph_inputs["image"] = "fixture://0"
    draft = PlannedVariant(
        variant_id="baseline",
        source_workflow_id="workflow-1",
        source_hash="a" * 64,
        materialized_graph={
            "9": {
                "class_type": "SaveImage",
                "inputs": graph_inputs,
            }
        },
        exact_run_hash="0" * 64,
        capability_hash="b" * 64,
        expected_output_count=expected_output_count,
        output_node_ids=output_node_ids,
        expected_media_types=expected_media_types,
        execution_parameters={
            "prompt_lines": ["one"],
            "expected_output_count": expected_output_count,
            "output_node_ids": list(output_node_ids),
        },
        fixture_fingerprints=fixture_fingerprints,
    )
    variant = PlannedVariant.model_validate(
        {
            **draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(draft),
        }
    )
    approval_payload = {
        "source_hashes": ["a" * 64],
        "variants": [variant.model_dump(mode="json")],
        "capability_coverage": [],
        "warnings": [],
    }
    plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=("a" * 64,),
        variants=(variant,),
        capability_coverage=(),
        warnings=(),
        approval_payload=approval_payload,
    )
    store.write_json_new(
        experiment_id,
        f"variants/{variant.variant_id}.json",
        variant.model_dump(mode="json"),
    )
    store.write_json_new(
        experiment_id,
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    return config, plan


def test_wrong_approved_hash_has_zero_paid_calls_and_creates_no_run(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    module_spec = importlib.util.find_spec("mp_comfyui_lab.experiments.engine")
    assert module_spec is not None, "execution engine is not implemented"
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    paid_constructions: list[str] = []

    def executor_factory():
        paid_constructions.append("executor")
        raise AssertionError("wrong approval touched the paid boundary")

    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=executor_factory,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError):
        engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash="f" * 64,
        )

    assert plan.plan_hash != "f" * 64
    assert paid_constructions == []
    assert list((config.experiments_root / "plan-approved" / "runs").iterdir()) == []


def test_fixture_hash_mismatch_fails_before_executor_construction(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, with_fixture=True)
    (tmp_path / "portrait.png").write_bytes(b"changed after planning")
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    paid_constructions: list[str] = []
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: paid_constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.FixtureIntegrityError):
        engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    assert paid_constructions == []
    runs_root = config.experiments_root / "plan-approved" / "runs"
    assert list(runs_root.glob("*.json")) == []
    assert list(runs_root.glob("*/fixtures/*")) == []


def test_fixture_source_replacement_after_approval_uses_run_snapshot(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, with_fixture=True)
    original = tmp_path / "portrait.png"
    approved_bytes = original.read_bytes()
    observed: list[tuple[Path, bytes]] = []

    class SnapshotProbeExecutor(FakeExecutor):
        def validate_variant(self, variant, fixtures):
            fixture = fixtures.items[0]
            observed.append((fixture.path, fixture.path.read_bytes()))
            return super().validate_variant(variant, fixtures)

        def submit(self, variant, fixtures):
            fixture = fixtures.items[0]
            observed.append((fixture.path, fixture.path.read_bytes()))
            return super().submit(variant, fixtures)

    executor = SnapshotProbeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.FAILED,)}
    )

    def executor_factory() -> SnapshotProbeExecutor:
        original.write_bytes(b"replacement after approval")
        return executor

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=executor_factory,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert [content for _, content in observed] == [approved_bytes, approved_bytes]
    assert all(path != original for path, _ in observed)
    assert all(
        path.is_relative_to(
            config.experiments_root / "plan-approved" / "runs" / record.run_id
        )
        for path, _ in observed
    )


def test_tampered_variant_artifact_fails_before_paid_boundary(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    variant_path = (
        config.experiments_root
        / "plan-approved"
        / "variants"
        / "baseline.json"
    )
    tampered = json.loads(variant_path.read_text(encoding="utf-8"))
    tampered["materialized_graph"]["9"]["inputs"]["filename_prefix"] = "changed"
    variant_path.write_text(json.dumps(tampered), encoding="utf-8")
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    paid_constructions: list[str] = []
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: paid_constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError):
        engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    assert paid_constructions == []
    assert list((config.experiments_root / "plan-approved" / "runs").iterdir()) == []


def test_task_id_is_durable_before_first_poll(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    observed_records: list[RunRecord] = []

    class PersistProbeExecutor(FakeExecutor):
        def poll(self, provider_task_id: str) -> PollResult:
            run_paths = tuple(
                path
                for path in (
                    config.experiments_root / "plan-approved" / "runs"
                ).glob("*.json")
            )
            assert len(run_paths) == 1
            observed_records.append(
                RunRecord.model_validate_json(
                    run_paths[0].read_text(encoding="utf-8"),
                    strict=True,
                )
            )
            return PollResult(
                provider_task_id=provider_task_id,
                state=ProviderTaskState.FAILED,
                diagnostic="provider rejected graph",
            )

    executor = PersistProbeExecutor()
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert observed_records[0].state is RunState.SUBMITTED
    assert observed_records[0].provider_task_id == "fake-task-1"
    assert record.state is RunState.FAILED
    assert executor.calls == ["validate_variant", "submit"]


def test_validation_errors_are_durable_and_prevent_submission(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        validation=ValidationReport(
            errors=(
                ValidationIssue(
                    code="missing_node",
                    message="required node is unavailable",
                ),
            )
        )
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    persisted = RunRecord.model_validate_json(
        (
            config.experiments_root
            / "plan-approved"
            / "runs"
            / f"{record.run_id}.json"
        ).read_text(encoding="utf-8"),
        strict=True,
    )
    assert persisted == record
    assert record.state is RunState.FAILED
    assert thaw_json(record.error) == {
        "kind": "validation_failed",
        "errors": [
            {
                "code": "missing_node",
                "message": "required node is unavailable",
            }
        ],
    }
    assert executor.calls == ["validate_variant"]


def test_all_run_artifacts_sanitize_untrusted_diagnostics(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    untrusted = (
        "api_key=durable-secret Authorization: Bearer bearer-secret "
        "https://cdn.example/output.png?sig=signed-secret"
    )
    executors = (
        FakeExecutor(
            validation=ValidationReport(
                errors=(ValidationIssue(code="invalid", message=untrusted),)
            )
        ),
        FakeExecutor(
            poll_sequences={
                "fake-task-1": (
                    PollResult(
                        provider_task_id="fake-task-1",
                        state=ProviderTaskState.FAILED,
                        diagnostic=untrusted,
                    ),
                )
            }
        ),
        FakeExecutor(
            method_exceptions={"poll": (TimeoutError(untrusted),)},
            poll_sequences={"fake-task-1": (ProviderTaskState.FAILED,)},
        ),
        FakeExecutor(method_exceptions={"submit": ExecutorError(untrusted)}),
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")

    for executor in executors:
        engine = engine_module.ExperimentEngine(
            config=config,
            executor_factory=lambda executor=executor: executor,
            downloader=lambda _url, _target: None,
            sleep=lambda _seconds: None,
        )
        engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    durable_bytes = b"\n".join(
        path.read_bytes()
        for path in (config.experiments_root / "plan-approved").rglob("*")
        if path.is_file()
    )
    for secret in (
        b"durable-secret",
        b"bearer-secret",
        b"signed-secret",
        b"?sig=",
    ):
        assert secret not in durable_bytes


def test_ambiguous_submission_is_incomplete_and_never_retried(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(ambiguous_submission=True)
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.provider_task_id is None
    assert thaw_json(record.error)["kind"] == "ambiguous_submission"
    assert executor.calls == ["validate_variant", "submit"]
    persisted = RunRecord.model_validate_json(
        (
            config.experiments_root
            / "plan-approved"
            / "runs"
            / f"{record.run_id}.json"
        ).read_text(encoding="utf-8"),
        strict=True,
    )
    assert persisted == record


@pytest.mark.parametrize(
    "create_result",
    [
        pytest.param(SimpleNamespace(taskStatus="QUEUED"), id="missing"),
        pytest.param(
            SimpleNamespace(taskId="", taskStatus="QUEUED"),
            id="blank",
        ),
        pytest.param(
            SimpleNamespace(taskId="None", taskStatus="QUEUED"),
            id="none-string",
        ),
        pytest.param(
            SimpleNamespace(taskId="malformed task", taskStatus="QUEUED"),
            id="malformed",
        ),
    ],
)
def test_real_runninghub_invalid_task_id_is_durable_ambiguous_without_retry(
    tmp_path: Path,
    create_result: SimpleNamespace,
) -> None:
    config, plan = _persist_plan(tmp_path, with_fixture=True)
    creates: list[str] = []

    def create_task_call(**_kwargs):
        creates.append("create")
        return create_result

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: pytest.fail("ambiguous run polled"),
        info_call=lambda **_kwargs: pytest.fail("ambiguous run fetched info"),
        outputs_call=lambda **_kwargs: pytest.fail("ambiguous run collected"),
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.provider_task_id is None
    assert thaw_json(record.error)["kind"] == "ambiguous_submission"
    assert creates == ["create"]
    persisted = (
        config.experiments_root
        / "plan-approved"
        / "runs"
        / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "ambiguous_submission" in persisted


def test_default_runninghub_create_lost_response_is_durable_ambiguous_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, plan = _persist_plan(tmp_path, with_fixture=True)
    posts: list[str] = []

    def lost_response(url, **_kwargs):
        posts.append(url)
        raise requests.Timeout("response lost after paid create transmission")

    runninghub_module = importlib.import_module(
        "mp_comfyui_lab.executors.runninghub"
    )
    monkeypatch.setattr(runninghub_module.requests, "post", lost_response)
    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        status_call=lambda **_kwargs: pytest.fail("ambiguous task polled"),
        info_call=lambda **_kwargs: pytest.fail("ambiguous task info fetched"),
        outputs_call=lambda **_kwargs: pytest.fail("ambiguous outputs collected"),
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "ambiguous_submission"
    assert posts == [runninghub_module.CREATE_TASK_ENDPOINT]


def test_real_runninghub_valid_id_is_persisted_before_malformed_status_poll(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, with_fixture=True)
    observed: list[RunRecord] = []

    def status_call(**_kwargs):
        run_path = next(
            (config.experiments_root / "plan-approved" / "runs").glob("*.json")
        )
        observed.append(
            RunRecord.model_validate_json(
                run_path.read_text(encoding="utf-8"),
                strict=True,
            )
        )
        return "FAILED"

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=lambda **_kwargs: SimpleNamespace(
            taskId="provider-task-1",
            taskStatus=7,
        ),
        status_call=status_call,
        info_call=lambda **_kwargs: {"taskId": "provider-task-1"},
        outputs_call=lambda **_kwargs: pytest.fail("failed task collected outputs"),
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert record.provider_task_id == "provider-task-1"
    assert len(observed) == 1
    assert observed[0].state is RunState.SUBMITTED
    assert observed[0].provider_task_id == "provider-task-1"


def test_submit_intent_recovers_crash_after_remote_return_without_resubmit(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)

    class CrashReceipt:
        @property
        def provider_task_id(self):
            raise SystemExit("crash before provider ID persistence")

    class CrashAfterSubmitExecutor(FakeExecutor):
        def submit(self, variant, fixtures):
            self.calls.append("submit")
            self.submissions.append((variant, fixtures))
            return CrashReceipt()

    crashing_executor = CrashAfterSubmitExecutor()
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    crashing_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: crashing_executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(SystemExit, match="crash before provider ID persistence"):
        crashing_engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    run_path = next(
        (config.experiments_root / "plan-approved" / "runs").glob("*.json")
    )
    planned = RunRecord.model_validate_json(
        run_path.read_text(encoding="utf-8"),
        strict=True,
    )
    assert planned.state is RunState.PLANNED
    assert planned.submission_intent_at is not None
    assert planned.provider_task_id is None
    assert crashing_executor.calls == ["validate_variant", "submit"]

    unexpected_factories: list[str] = []
    recovery_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: unexpected_factories.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    recovered = recovery_engine.resume("plan-approved", planned.run_id)

    assert recovered.state is RunState.INCOMPLETE
    assert thaw_json(recovered.error)["kind"] == "ambiguous_submission"
    assert unexpected_factories == []

    reconcile_executor = FakeExecutor(
        poll_sequences={"operator-task-7": (ProviderTaskState.FAILED,)}
    )
    reconcile_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: reconcile_executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    reconciled = reconcile_engine.reconcile(
        "plan-approved",
        planned.run_id,
        "operator-task-7",
    )

    assert reconciled.state is RunState.FAILED
    assert reconciled.provider_task_id == "operator-task-7"
    assert reconcile_executor.calls == ["poll"]


def test_resume_polls_existing_task_id_without_validation_or_submission(
    tmp_path: Path,
) -> None:
    config, _ = _persist_plan(tmp_path)
    existing = RunRecord(
        run_id="run-existing",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUBMITTED,
        plan_hash="a" * 64,
        exact_run_hash="b" * 64,
        provider_task_id="provider-task-1",
        expected_output_count=1,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "plan-approved",
        "runs/run-existing.json",
        existing.model_dump(mode="json"),
    )
    executor = FakeExecutor(
        poll_sequences={"provider-task-1": (ProviderTaskState.FAILED,)}
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.resume("plan-approved", "run-existing")

    assert record.state is RunState.FAILED
    assert record.provider_task_id == "provider-task-1"
    assert executor.calls == ["poll"]
    assert executor.polls == ["provider-task-1"]


@pytest.mark.parametrize(
    ("state", "error"),
    [
        pytest.param(RunState.SUCCESS, None, id="success"),
        pytest.param(RunState.FAILED, None, id="failed"),
        pytest.param(RunState.CANCELLED, None, id="cancelled"),
        pytest.param(
            RunState.INCOMPLETE,
            {"kind": "ambiguous_submission"},
            id="ambiguous-incomplete",
        ),
        pytest.param(
            RunState.INCOMPLETE,
            {"kind": "output_count_mismatch"},
            id="nonambiguous-incomplete",
        ),
    ],
)
def test_terminal_resume_is_read_only_without_executor(
    tmp_path: Path,
    state: RunState,
    error: dict[str, str] | None,
) -> None:
    config, _ = _persist_plan(tmp_path)
    error_kind = "none" if error is None else error["kind"]
    record = RunRecord(
        run_id=f"run-terminal-{state.value.casefold()}-{error_kind}",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=state,
        provider_task_id=(
            "provider-task-1"
            if state in {RunState.SUCCESS, RunState.CANCELLED}
            or error_kind == "output_count_mismatch"
            else None
        ),
        expected_output_count=1,
        actual_output_count=1 if state is RunState.SUCCESS else 0,
        error=error,
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "plan-approved",
        f"runs/{record.run_id}.json",
        record.model_dump(mode="json"),
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    resumed = engine.resume("plan-approved", record.run_id)

    assert resumed == record
    assert constructions == []


def test_transient_poll_retries_use_backoff_without_resubmission(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    transient = (
        TimeoutError("poll https://provider.example/status?token=one timed out"),
        TimeoutError("poll timed out twice"),
        TimeoutError("poll timed out three times"),
    )
    executor = FakeExecutor(
        method_exceptions={"poll": transient},
        poll_sequences={"fake-task-1": (ProviderTaskState.FAILED,)},
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert waits == [1, 2, 4]
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("poll") == 4
    assert len(record.diagnostics) == 3
    persisted = (
        config.experiments_root
        / "plan-approved"
        / "runs"
        / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "token=one" not in persisted


def test_provider_running_then_cancelled_maps_to_durable_states(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        poll_sequences={
            "fake-task-1": (
                ProviderTaskState.RUNNING,
                ProviderTaskState.CANCELLED,
            )
        }
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.CANCELLED
    assert executor.calls == [
        "validate_variant",
        "submit",
        "poll",
        "poll",
    ]
    assert waits == [1]
    assert thaw_json(record.error)["kind"] == "provider_cancelled"


def test_partial_output_batch_is_incomplete_and_retains_verified_output(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    output_bytes = valid_media_bytes("image/png")
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?token=live-secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(url: str, target: Path) -> Path:
        assert url.endswith("?token=live-secret")
        target.write_bytes(output_bytes)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == 1
    assert len(record.outputs) == 1
    durable_output = record.outputs[0]
    assert durable_output.provider_order == 0
    assert durable_output.node_id == "9"
    assert durable_output.media_type == "image/png"
    assert durable_output.content_sha256 == hashlib.sha256(output_bytes).hexdigest()
    assert durable_output.size == len(output_bytes)
    output_path = (
        config.experiments_root
        / "plan-approved"
        / Path(durable_output.local_relative_path)
    )
    assert output_path.read_bytes() == output_bytes
    persisted = (
        config.experiments_root
        / "plan-approved"
        / "runs"
        / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "live-secret" not in persisted
    assert "https://cdn.example" not in persisted
    assert thaw_json(record.error)["kind"] == "partial_outputs"
    assert executor.calls.count("collect_outputs") == 4
    assert len(record.diagnostics) == 4


@pytest.mark.parametrize("first_count", [0, 1])
def test_incomplete_success_batch_retries_until_exact_without_resubmit(
    tmp_path: Path,
    first_count: int,
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    first = ProviderOutput(
        node_id="9",
        url="https://cdn.example/first.png?sig=one",
        media_type="image/png",
        provider_order=0,
    )
    second = ProviderOutput(
        node_id="9",
        url="https://cdn.example/second.png?sig=two",
        media_type="image/png",
        provider_order=1,
    )
    first_batch = () if first_count == 0 else (first,)
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"fake-task-1": (first_batch, (first, second))},
    )

    def downloader(url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert record.actual_output_count == 2
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("collect_outputs") == 2
    assert waits == [1]
    assert thaw_json(record.diagnostics[0]) == {
        "kind": "incomplete_output_batch",
        "operation": "collect_outputs",
        "attempt": 1,
        "observed_output_count": first_count,
        "expected_output_count": 2,
    }


@pytest.mark.parametrize("observed_count", [0, 1])
def test_persistent_incomplete_success_batch_keeps_final_evidence(
    tmp_path: Path,
    observed_count: int,
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/partial.png",
        media_type="image/png",
        provider_order=0,
    )
    batch = () if observed_count == 0 else (output,)
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"fake-task-1": (batch,)},
    )

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == observed_count
    assert len(record.outputs) == observed_count
    assert thaw_json(record.error)["kind"] == "partial_outputs"
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("collect_outputs") == 4
    assert waits == [1, 2, 4]
    assert len(record.diagnostics) == 4


def test_partial_batch_survives_later_transient_collection_exhaustion(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/partial.png",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
    )
    attempts = 0

    def collect_outputs(task_id: str) -> tuple[ProviderOutput, ...]:
        nonlocal attempts
        executor.calls.append("collect_outputs")
        executor.collections.append(task_id)
        attempts += 1
        if attempts == 1:
            return (output,)
        raise TimeoutError("collection response was lost")

    executor.collect_outputs = collect_outputs  # type: ignore[method-assign]

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == 1
    assert len(record.outputs) == 1
    assert thaw_json(record.error)["kind"] == "partial_outputs"
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("collect_outputs") == 4
    assert waits == [1, 2, 4]


def test_complete_verified_output_batch_is_success(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert record.actual_output_count == record.expected_output_count == 1
    assert len(record.outputs) == 1
    assert record.error is None
    assert executor.calls == [
        "validate_variant",
        "submit",
        "poll",
        "collect_outputs",
    ]


def test_resume_reuses_orphan_published_before_output_record_persistence(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    content = valid_media_bytes("image/png")
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=secret",
        media_type="image/png",
        provider_order=0,
    )
    first_executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(content)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    crashing_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: first_executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )
    original_write = crashing_engine._store.publish_file_new

    def crash_after_output_publish(experiment_id, relative_path, value, **kwargs):
        published = original_write(
            experiment_id,
            relative_path,
            value,
            **kwargs,
        )
        if Path(relative_path).parts[0] == "outputs":
            raise SystemExit("crash after final output publish")
        return published

    crashing_engine._store.publish_file_new = crash_after_output_publish

    with pytest.raises(SystemExit, match="crash after final output publish"):
        crashing_engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    run_path = next(
        (config.experiments_root / "plan-approved" / "runs").glob("*.json")
    )
    interrupted = RunRecord.model_validate_json(
        run_path.read_text(encoding="utf-8"),
        strict=True,
    )
    assert interrupted.state is RunState.SUBMITTED
    assert interrupted.outputs == ()
    orphan = config.experiments_root / "plan-approved" / "outputs/baseline/0000-9.png"
    assert orphan.read_bytes() == content

    resume_executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"fake-task-1": ((output,),)},
    )
    resume_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: resume_executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    resumed = resume_engine.resume("plan-approved", interrupted.run_id)

    assert resumed.state is RunState.SUCCESS
    assert len(resumed.outputs) == 1
    assert resumed.outputs[0].content_sha256 == hashlib.sha256(content).hexdigest()
    assert orphan.read_bytes() == content
    assert resume_executor.calls == ["poll", "collect_outputs"]


def test_identical_second_run_reuses_matching_variant_output(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    content = valid_media_bytes("image/png")
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=secret",
        media_type="image/png",
        provider_order=0,
    )
    executors = [
        FakeExecutor.success(outputs=(output,)),
        FakeExecutor.success(outputs=(output,)),
    ]

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(content)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executors.pop(0),
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    first = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )
    second = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert first.state is second.state is RunState.SUCCESS
    assert first.run_id != second.run_id
    assert first.outputs == second.outputs
    output_path = (
        config.experiments_root
        / "plan-approved"
        / first.outputs[0].local_relative_path
    )
    assert output_path.read_bytes() == content


def test_extra_provider_output_is_incomplete(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    outputs = tuple(
        ProviderOutput(
            node_id="9",
            url=f"https://cdn.example/output-{index}.png?sig=secret-{index}",
            media_type="image/png",
            provider_order=index,
        )
        for index in range(2)
    )
    executor = FakeExecutor.success(outputs=outputs)

    def downloader(url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == 2
    assert len(record.outputs) == 2
    assert thaw_json(record.error)["kind"] == "extra_outputs"


def test_resume_provider_batch_shrink_replaces_stale_output_records(
    tmp_path: Path,
) -> None:
    config, _ = _persist_plan(tmp_path)
    first_content = valid_media_bytes("image/png")
    stale_content = valid_media_bytes("image/png")
    durable_outputs = (
        OutputRecord(
            ordinal=0,
            provider_order=0,
            node_id="9",
            media_type="image/png",
            local_relative_path="outputs/baseline/0000-9.png",
            content_sha256=hashlib.sha256(first_content).hexdigest(),
            size=len(first_content),
        ),
        OutputRecord(
            ordinal=1,
            provider_order=1,
            node_id="9",
            media_type="image/png",
            local_relative_path="outputs/baseline/0001-9.png",
            content_sha256=hashlib.sha256(stale_content).hexdigest(),
            size=len(stale_content),
        ),
    )
    existing = RunRecord(
        run_id="run-shrunk-batch",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUBMITTED,
        provider_task_id="provider-task-1",
        expected_output_count=2,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
        actual_output_count=2,
        outputs=durable_outputs,
    )
    store = ExperimentStore(config.experiments_root)
    store.write_bytes_new(
        "plan-approved",
        durable_outputs[0].local_relative_path,
        first_content,
    )
    store.write_bytes_new(
        "plan-approved",
        durable_outputs[1].local_relative_path,
        stale_content,
    )
    store.write_json_new(
        "plan-approved",
        "runs/run-shrunk-batch.json",
        existing.model_dump(mode="json"),
    )
    current_output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/current.png?sig=live",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor(
        poll_sequences={"provider-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"provider-task-1": ((current_output,),)},
    )
    downloads: list[str] = []

    def downloader(url: str, target: Path) -> Path:
        downloads.append(url)
        target.write_bytes(first_content)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.resume("plan-approved", "run-shrunk-batch")

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == 1
    assert record.outputs == (durable_outputs[0],)
    assert thaw_json(record.error)["kind"] == "partial_outputs"
    assert downloads == [current_output.url]


def test_malformed_output_is_incomplete_and_preserves_prior_output(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    outputs = (
        ProviderOutput(
            node_id="9",
            url="https://cdn.example/first.png?sig=first-secret",
            media_type="image/png",
            provider_order=0,
        ),
        ProviderOutput(
            node_id="../escape",
            url="https://cdn.example/second.png?sig=second-secret",
            media_type="image/png",
            provider_order=1,
        ),
    )
    executor = FakeExecutor.success(outputs=outputs)

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.actual_output_count == 1
    assert len(record.outputs) == 1
    assert thaw_json(record.error)["kind"] == "malformed_output"
    experiment_root = config.experiments_root / "plan-approved"
    assert not (experiment_root / "escape.png").exists()
    persisted = (
        experiment_root / "runs" / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "first-secret" not in persisted
    assert "second-secret" not in persisted


def test_collection_and_download_retries_never_resubmit(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?token=secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"fake-task-1": ((output,),)},
        method_exceptions={
            "collect_outputs": (
                TimeoutError("collect one"),
                TimeoutError("collect two"),
                TimeoutError("collect three"),
            )
        },
    )
    download_attempts: list[str] = []

    def downloader(url: str, target: Path) -> Path:
        download_attempts.append(url)
        if len(download_attempts) <= 3:
            raise TimeoutError(f"download {len(download_attempts)}")
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert waits == [1, 2, 4, 1, 2, 4]
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("collect_outputs") == 4
    assert len(download_attempts) == 4
    assert len(record.diagnostics) == 6


def test_production_downloader_retries_runninghub_error_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.xiaoyaoyou.com/output.png?sig=live-signed-secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))
    attempts: list[int] = []
    output_bytes = valid_media_bytes("image/png")

    def download_file(_url: str, target: Path, **_kwargs) -> Path:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= 3:
            raise RunningHubApiError(
                "download failed api_key=durable-secret "
                "Bearer bearer-secret "
                "https://cdn.example/output.png?sig=signed-secret"
            )
        target.write_bytes(output_bytes)
        return target

    cli_app = importlib.import_module("mp_comfyui_lab.cli.app")
    monkeypatch.setattr(
        cli_app,
        "_download_runninghub_output_http",
        download_file,
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=cli_app._download_runninghub_output,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert attempts == [1, 2, 3, 4]
    assert waits == [1, 2, 4]
    assert len(record.diagnostics) == 3
    assert executor.calls.count("submit") == 1
    durable_bytes = b"\n".join(
        path.read_bytes()
        for path in (config.experiments_root / "plan-approved").rglob("*")
        if path.is_file()
    )
    for secret in (
        b"durable-secret",
        b"bearer-secret",
        b"signed-secret",
        b"live-signed-secret",
    ):
        assert secret not in durable_bytes


def test_resume_reuses_matching_durable_output_without_overwrite(
    tmp_path: Path,
) -> None:
    config, _ = _persist_plan(tmp_path)
    content = valid_media_bytes("image/png")
    durable_output = OutputRecord(
        ordinal=0,
        provider_order=0,
        node_id="9",
        media_type="image/png",
        local_relative_path="outputs/baseline/0000-9.png",
        content_sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    existing = RunRecord(
        run_id="run-existing-output",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUBMITTED,
        provider_task_id="provider-task-1",
        expected_output_count=1,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
        actual_output_count=1,
        outputs=(durable_output,),
    )
    store = ExperimentStore(config.experiments_root)
    store.write_bytes_new(
        "plan-approved",
        durable_output.local_relative_path,
        content,
    )
    store.write_json_new(
        "plan-approved",
        "runs/run-existing-output.json",
        existing.model_dump(mode="json"),
    )
    provider_output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=live",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor(
        poll_sequences={"provider-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"provider-task-1": ((provider_output,),)},
    )
    downloads: list[str] = []

    def downloader(url: str, target: Path) -> Path:
        downloads.append(url)
        target.write_bytes(content)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.resume("plan-approved", "run-existing-output")

    assert record.state is RunState.SUCCESS
    assert record.outputs == (durable_output,)
    assert store.read_bytes(
        "plan-approved",
        durable_output.local_relative_path,
    ) == content
    assert executor.calls == ["poll", "collect_outputs"]
    assert downloads == [provider_output.url]


def test_reconcile_attaches_operator_task_id_then_polls_without_submit(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    ambiguous_executor = FakeExecutor(ambiguous_submission=True)
    provider_output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/reconciled.png?sig=secret",
        media_type="image/png",
        provider_order=0,
    )
    polling_executor = FakeExecutor(
        poll_sequences={"operator-task-7": (ProviderTaskState.SUCCESS,)},
        output_scripts={"operator-task-7": ((provider_output,),)},
    )
    executors = iter((ambiguous_executor, polling_executor))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: next(executors),
        downloader=downloader,
        sleep=lambda _seconds: None,
    )
    ambiguous = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    record = engine.reconcile(
        "plan-approved",
        ambiguous.run_id,
        "operator-task-7",
    )

    assert record.state is RunState.SUCCESS
    assert record.provider_task_id == "operator-task-7"
    assert ambiguous_executor.calls == ["validate_variant", "submit"]
    assert polling_executor.calls == ["poll", "collect_outputs"]


def test_reconcile_rejects_nonambiguous_incomplete_run(tmp_path: Path) -> None:
    config, _ = _persist_plan(tmp_path)
    record = RunRecord(
        run_id="run-partial",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.INCOMPLETE,
        provider_task_id="provider-task-1",
        expected_output_count=2,
        actual_output_count=1,
        error={"kind": "partial_outputs"},
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "plan-approved",
        "runs/run-partial.json",
        record.model_dump(mode="json"),
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(ValueError, match="only an ambiguous INCOMPLETE"):
        engine.reconcile("plan-approved", "run-partial", "operator-task-1")

    assert constructions == []


def test_status_is_store_only_and_returns_full_run_record(tmp_path: Path) -> None:
    config, _ = _persist_plan(tmp_path)
    existing = RunRecord(
        run_id="run-status",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.RUNNING,
        provider_task_id="provider-task-1",
        expected_output_count=2,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
        actual_output_count=1,
        error={"kind": "prior_error"},
        diagnostics=({"kind": "transient_error", "operation": "poll"},),
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "plan-approved",
        "runs/run-status.json",
        existing.model_dump(mode="json"),
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: pytest.fail("status constructed executor"),
        downloader=lambda _url, _target: pytest.fail("status downloaded output"),
        sleep=lambda _seconds: pytest.fail("status slept"),
    )

    assert engine.status("plan-approved", "run-status") == existing


def test_provider_task_id_mismatch_is_durable_safety_failure(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)

    class MismatchedExecutor(FakeExecutor):
        def poll(self, provider_task_id: str) -> PollResult:
            self.calls.append("poll")
            self.polls.append(provider_task_id)
            return PollResult(
                provider_task_id="foreign-task",
                state=ProviderTaskState.RUNNING,
            )

    executor = MismatchedExecutor()
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(Exception) as raised:
        engine.run(
            "plan-approved",
            "baseline",
            approved_plan_hash=plan.plan_hash,
        )

    assert isinstance(raised.value, engine_module.SafetyError)
    run_path = next(
        (config.experiments_root / "plan-approved" / "runs").glob("*.json")
    )
    record = RunRecord.model_validate_json(
        run_path.read_text(encoding="utf-8"),
        strict=True,
    )
    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "provider_task_id_mismatch"
    assert executor.calls.count("submit") == 1


def test_provider_submitted_and_queued_remain_submitted_until_terminal(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        poll_sequences={
            "fake-task-1": (
                ProviderTaskState.SUBMITTED,
                ProviderTaskState.QUEUED,
                ProviderTaskState.FAILED,
            )
        }
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert waits == [1, 1]
    assert executor.calls.count("poll") == 3


def test_unknown_provider_state_is_incomplete_without_collection(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        poll_sequences={
            "fake-task-1": (
                PollResult(
                    provider_task_id="fake-task-1",
                    state=ProviderTaskState.UNKNOWN,
                    diagnostic="future provider state",
                ),
            )
        }
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error) == {
        "kind": "unknown_provider_state",
        "diagnostic": "future provider state",
    }
    assert executor.calls == ["validate_variant", "submit", "poll"]


def test_confirmed_submission_rejection_is_failed_without_retry(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        method_exceptions={
            "submit": ExecutorError(
                "provider rejected https://provider.example?token=secret"
            )
        }
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert record.provider_task_id is None
    assert thaw_json(record.error)["kind"] == "provider_rejection"
    assert executor.calls == ["validate_variant", "submit"]
    persisted = (
        config.experiments_root
        / "plan-approved"
        / "runs"
        / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "token=secret" not in persisted


def test_validation_exception_is_durable_failed_without_submission(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        method_exceptions={
            "validate_variant": ExecutorError(
                "validation failed https://provider.example?token=secret"
            )
        }
    )
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.FAILED
    assert thaw_json(record.error)["kind"] == "validation_error"
    assert executor.calls == ["validate_variant"]
    persisted = (
        config.experiments_root
        / "plan-approved"
        / "runs"
        / f"{record.run_id}.json"
    ).read_text(encoding="utf-8")
    assert "token=secret" not in persisted


def test_exhausted_poll_retries_are_durable_incomplete(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        method_exceptions={
            "poll": tuple(
                TimeoutError(f"poll timeout {index}")
                for index in range(1, 5)
            )
        }
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "retry_exhausted"
    assert thaw_json(record.error)["operation"] == "poll"
    assert len(record.diagnostics) == 4
    assert waits == [1, 2, 4]
    assert executor.calls.count("submit") == 1
    assert executor.calls.count("poll") == 4


def test_exhausted_download_retries_are_durable_incomplete(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))
    attempts: list[int] = []

    def downloader(_url: str, _target: Path) -> None:
        attempts.append(len(attempts) + 1)
        raise TimeoutError(f"download timeout {attempts[-1]}")

    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error) == {
        "kind": "retry_exhausted",
        "operation": "download_output",
    }
    assert record.outputs == ()
    assert len(record.diagnostics) == 4
    assert waits == [1, 2, 4]
    assert attempts == [1, 2, 3, 4]


def test_production_downloader_exhaustion_is_durable_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.xiaoyaoyou.com/output.png?sig=live-signed-secret",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))
    attempts: list[int] = []

    def download_file(_url: str, _target: Path, **_kwargs) -> None:
        attempts.append(len(attempts) + 1)
        raise RunningHubApiError(
            "download failed api_key=durable-secret "
            "Bearer bearer-secret "
            "https://cdn.example/output.png?sig=signed-secret"
        )

    cli_app = importlib.import_module("mp_comfyui_lab.cli.app")
    monkeypatch.setattr(
        cli_app,
        "_download_runninghub_output_http",
        download_file,
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=cli_app._download_runninghub_output,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error) == {
        "kind": "retry_exhausted",
        "operation": "download_output",
    }
    assert attempts == [1, 2, 3, 4]
    assert waits == [1, 2, 4]
    assert len(record.diagnostics) == 4
    assert executor.calls.count("submit") == 1
    durable_bytes = b"\n".join(
        path.read_bytes()
        for path in (config.experiments_root / "plan-approved").rglob("*")
        if path.is_file()
    )
    for secret in (
        b"durable-secret",
        b"bearer-secret",
        b"signed-secret",
        b"live-signed-secret",
    ):
        assert secret not in durable_bytes


def test_nontransient_poll_contract_error_is_not_retried(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        method_exceptions={
            "poll": ProviderSchemaError("malformed provider status")
        }
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "poll_error"
    assert executor.calls.count("poll") == 1
    assert waits == []


def test_nontransient_collection_contract_error_is_not_retried(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.SUCCESS,)},
        method_exceptions={
            "collect_outputs": ProviderSchemaError("malformed provider outputs")
        },
    )
    waits: list[float] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: None,
        sleep=waits.append,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "collection_error"
    assert executor.calls.count("collect_outputs") == 1
    assert waits == []


def test_output_media_mismatch_is_incomplete_without_download(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.mp4?sig=secret",
        media_type="video/mp4",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: pytest.fail("mismatched media downloaded"),
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert record.outputs == ()
    assert thaw_json(record.error)["kind"] == "malformed_output"


def test_html_declared_as_png_is_incomplete_without_publication(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(b"<html><body>not an image</body></html>")
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"
    assert not list(
        (config.experiments_root / "plan-approved" / "outputs").rglob("*.png")
    )


@pytest.mark.parametrize(
    "media_type",
    [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "video/mp4",
        "video/webm",
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "application/zip",
    ],
)
def test_supported_media_types_require_valid_minimal_content(
    tmp_path: Path,
    media_type: str,
) -> None:
    config, plan = _persist_plan(
        tmp_path,
        expected_media_types=(media_type,),
    )
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output",
        media_type=media_type,
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes(media_type))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert record.outputs[0].media_type == media_type


@pytest.mark.parametrize(
    ("declared", "actual"),
    [
        ("image/png", "image/jpeg"),
        ("video/mp4", "video/webm"),
        ("audio/wav", "audio/mpeg"),
    ],
)
def test_mismatched_download_signature_is_incomplete(
    tmp_path: Path,
    declared: str,
    actual: str,
) -> None:
    config, plan = _persist_plan(
        tmp_path,
        expected_media_types=(declared,),
    )
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output",
        media_type=declared,
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes(actual))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"
    assert record.outputs == ()


def test_injected_downloader_output_cannot_exceed_engine_byte_limit(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(tmp_path)
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor.success(outputs=(output,))

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
        max_output_bytes=8,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"
    assert record.outputs == ()


@pytest.mark.parametrize(
    ("first_pair", "second_pair"),
    [
        (("9", "image/png"), ("9", "image/png")),
        (("9", "image/jpeg"), ("10", "image/png")),
        (("9", "image/png"), ("9", "image/jpeg")),
    ],
    ids=["duplicate-node", "swapped-media", "missing-pair"],
)
def test_exact_output_pairs_reject_independent_allowlist_matches(
    tmp_path: Path,
    first_pair: tuple[str, str],
    second_pair: tuple[str, str],
) -> None:
    config, plan = _persist_plan(
        tmp_path,
        expected_output_count=2,
        output_node_ids=("9", "10"),
        expected_media_types=("image/png", "image/jpeg"),
    )
    outputs = tuple(
        ProviderOutput(
            node_id=node_id,
            url=f"https://cdn.example/{order}.png",
            media_type=media_type,
            provider_order=order,
        )
        for order, (node_id, media_type) in enumerate(
            (first_pair, second_pair)
        )
    )
    executor = FakeExecutor.success(outputs=outputs)

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(b"provider output")
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"


def test_provider_outputs_are_canonicalized_by_provider_order(
    tmp_path: Path,
) -> None:
    config, plan = _persist_plan(
        tmp_path,
        expected_output_count=2,
        output_node_ids=("9", "10"),
        expected_media_types=("image/png", "image/jpeg"),
    )
    outputs = (
        ProviderOutput(
            node_id="10",
            url="https://cdn.example/second.jpg",
            media_type="image/jpeg",
            provider_order=1,
        ),
        ProviderOutput(
            node_id="9",
            url="https://cdn.example/first.png",
            media_type="image/png",
            provider_order=0,
        ),
    )
    executor = FakeExecutor.success(outputs=outputs)

    def downloader(url: str, target: Path) -> Path:
        media_type = "image/jpeg" if url.endswith(".jpg") else "image/png"
        target.write_bytes(valid_media_bytes(media_type))
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.SUCCESS
    assert [item.provider_order for item in record.outputs] == [0, 1]
    assert [item.node_id for item in record.outputs] == ["9", "10"]
    assert [item.local_relative_path for item in record.outputs] == [
        "outputs/baseline/0000-9.png",
        "outputs/baseline/0001-10.jpg",
    ]


@pytest.mark.parametrize("orders", [(0, 0), (0, 2)])
def test_provider_output_order_must_be_unique_and_contiguous(
    tmp_path: Path,
    orders: tuple[int, int],
) -> None:
    config, plan = _persist_plan(tmp_path, expected_output_count=2)
    outputs = tuple(
        ProviderOutput(
            node_id="9",
            url=f"https://cdn.example/{index}.png",
            media_type="image/png",
            provider_order=order,
        )
        for index, order in enumerate(orders)
    )
    executor = FakeExecutor.success(outputs=outputs)
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: pytest.fail("invalid order downloaded"),
        sleep=lambda _seconds: None,
    )

    record = engine.run(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
    )

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"
    assert record.outputs == ()


@pytest.mark.parametrize(
    ("count", "node_ids", "media_types"),
    [
        (2, ("9", "10"), ("image/png",)),
        (2, ("9", "9"), ("image/png", "image/jpeg")),
        (3, ("9", "10"), ("image/png", "image/jpeg")),
    ],
    ids=["unaligned", "duplicate-node", "indivisible"],
)
def test_planned_output_expectations_require_aligned_divisible_unique_pairs(
    tmp_path: Path,
    count: int,
    node_ids: tuple[str, ...],
    media_types: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _persist_plan(
            tmp_path,
            expected_output_count=count,
            output_node_ids=node_ids,
            expected_media_types=media_types,
        )


def test_durable_output_hash_mismatch_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    config, _ = _persist_plan(tmp_path)
    approved = b"approved bytes"
    tampered = b"tampered bytes"
    durable_output = OutputRecord(
        ordinal=0,
        provider_order=0,
        node_id="9",
        media_type="image/png",
        local_relative_path="outputs/baseline/0000-9.png",
        content_sha256=hashlib.sha256(approved).hexdigest(),
        size=len(approved),
    )
    existing = RunRecord(
        run_id="run-mismatched-output",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUBMITTED,
        provider_task_id="provider-task-1",
        expected_output_count=1,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
        actual_output_count=1,
        outputs=(durable_output,),
    )
    store = ExperimentStore(config.experiments_root)
    store.write_bytes_new(
        "plan-approved",
        durable_output.local_relative_path,
        tampered,
    )
    store.write_json_new(
        "plan-approved",
        "runs/run-mismatched-output.json",
        existing.model_dump(mode="json"),
    )
    provider_output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?sig=live",
        media_type="image/png",
        provider_order=0,
    )
    executor = FakeExecutor(
        poll_sequences={"provider-task-1": (ProviderTaskState.SUCCESS,)},
        output_scripts={"provider-task-1": ((provider_output,),)},
    )

    def downloader(_url: str, target: Path) -> Path:
        target.write_bytes(approved)
        return target

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )

    record = engine.resume("plan-approved", "run-mismatched-output")

    assert record.state is RunState.INCOMPLETE
    assert thaw_json(record.error)["kind"] == "malformed_output"
    assert store.read_bytes(
        "plan-approved",
        durable_output.local_relative_path,
    ) == tampered


def test_run_lock_allows_only_one_concurrent_resume_poll(tmp_path: Path) -> None:
    config, _ = _persist_plan(tmp_path)
    existing = RunRecord(
        run_id="run-concurrent",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUBMITTED,
        provider_task_id="provider-task-1",
        expected_output_count=1,
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "plan-approved",
        "runs/run-concurrent.json",
        existing.model_dump(mode="json"),
    )
    poll_entered = threading.Event()
    release_poll = threading.Event()

    class BlockingExecutor(FakeExecutor):
        def poll(self, provider_task_id: str) -> PollResult:
            self.calls.append("poll")
            self.polls.append(provider_task_id)
            poll_entered.set()
            assert release_poll.wait(timeout=5)
            return PollResult(
                provider_task_id=provider_task_id,
                state=ProviderTaskState.FAILED,
            )

    executor = BlockingExecutor()
    constructions: list[str] = []

    def executor_factory():
        constructions.append("executor")
        return executor

    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=executor_factory,
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )
    results: list[RunRecord] = []
    failures: list[BaseException] = []

    def resume() -> None:
        try:
            results.append(engine.resume("plan-approved", "run-concurrent"))
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=resume)
    second = threading.Thread(target=resume)
    first.start()
    assert poll_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert second.is_alive()
    release_poll.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert failures == []
    assert [record.state for record in results] == [RunState.FAILED, RunState.FAILED]
    assert executor.calls == ["poll"]
    assert constructions == ["executor"]


def test_run_lock_serializes_separate_processes(tmp_path: Path) -> None:
    config, _ = _persist_plan(tmp_path)
    store = ExperimentStore(config.experiments_root)
    marker = tmp_path / "child-acquired.txt"
    script = """
from pathlib import Path
import sys
from mp_comfyui_lab.experiments.store import ExperimentStore

store = ExperimentStore(Path(sys.argv[1]))
with store.run_lock("plan-approved", "run-process-lock"):
    Path(sys.argv[2]).write_text("acquired", encoding="utf-8")
"""
    with store.run_lock("plan-approved", "run-process-lock"):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(config.experiments_root),
                str(marker),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.25)
        assert process.poll() is None
        assert not marker.exists()

    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (stdout, stderr)
    assert marker.read_text(encoding="utf-8") == "acquired"
