from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import threading
import time

import pytest

from helpers import FakeExecutor
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
        output_node_ids=("9",),
        expected_media_types=("image/png",),
        execution_parameters={
            "prompt_lines": ["one"],
            "expected_output_count": expected_output_count,
            "output_node_ids": ["9"],
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
    assert list((config.experiments_root / "plan-approved" / "runs").iterdir()) == []


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
    output_bytes = b"verified png output"
    output = ProviderOutput(
        node_id="9",
        url="https://cdn.example/output.png?token=live-secret",
        media_type="image/png",
        provider_order=7,
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
    assert durable_output.provider_order == 7
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
        target.write_bytes(b"complete output")
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
        target.write_bytes(b"one")
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
        target.write_bytes(b"retried output")
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


def test_resume_reuses_matching_durable_output_without_overwrite(
    tmp_path: Path,
) -> None:
    config, _ = _persist_plan(tmp_path)
    content = b"already published"
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
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: pytest.fail("matching output redownloaded"),
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
        target.write_bytes(b"reconciled")
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
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=lambda _url, _target: pytest.fail("mismatch redownloaded"),
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
