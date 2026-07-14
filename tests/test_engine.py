from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib
from io import BytesIO
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
from PIL import Image

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
from mp_comfyui_lab.graph.hashing import capability_hash as graph_capability_hash
from mp_comfyui_lab.experiments.models import (
    DimensionName,
    ExperimentPlan,
    FixtureFingerprint,
    OutputRecord,
    PlannedVariant,
    PlanningConfidence,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.saturation import (
    RerunReasonRequired,
    WaveArtifact,
)
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import SafetyError


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
    graph = {
        "9": {
            "class_type": "SaveImage",
            "inputs": graph_inputs,
        }
    }
    draft = PlannedVariant(
        variant_id="baseline",
        source_workflow_id="workflow-1",
        source_hash="a" * 64,
        materialized_graph=graph,
        exact_run_hash="0" * 64,
        capability_hash=graph_capability_hash(graph),
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


def _persist_wave_plan(
    tmp_path: Path,
    *,
    experiment_id: str = "wave-approved",
    variants: tuple[tuple[str, str, PlanningConfidence, int, int], ...] = (
        ("strong", "b" * 64, PlanningConfidence.OBSERVED, 10, 2),
        (
            "weak-popular",
            "b" * 64,
            PlanningConfidence.WEAKLY_INFERRED,
            100_000,
            1,
        ),
        ("other", "c" * 64, PlanningConfidence.OBSERVED, 5, 1),
    ),
    hypothesis: DimensionName = DimensionName.CHECKPOINT,
    coverage: tuple[DimensionName, ...] = (DimensionName.CHECKPOINT,),
    corrections: dict[str, dict[str, object]] | None = None,
) -> tuple[LabConfig, ExperimentPlan]:
    config = LabConfig.model_validate(
        {"experiments_root": tmp_path / "experiments"},
        context={"base_dir": tmp_path},
    )
    store = ExperimentStore(config.experiments_root)
    store.create(experiment_id)
    planned: list[PlannedVariant] = []
    candidates: list[dict[str, object]] = []
    for index, (variant_id, capability_group, confidence, engagement, provenance) in enumerate(
        variants,
        start=1,
    ):
        execution_parameters: dict[str, object] = {
            "prompt_lines": ["one"],
            "expected_output_count": 1,
            "output_node_ids": ["9"],
        }
        if corrections is not None and variant_id in corrections:
            execution_parameters["correction"] = corrections[variant_id]
        graph = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": (
                        f"group-{capability_group[0]}.safetensors"
                    )
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": variant_id},
            },
        }
        draft = PlannedVariant(
            variant_id=variant_id,
            source_workflow_id="workflow-1",
            source_hash="a" * 64,
            materialized_graph=graph,
            hypothesis=hypothesis,
            evidence_ids=(f"workflow-{index}",),
            confidence=confidence,
            exact_run_hash="0" * 64,
            capability_hash=graph_capability_hash(graph),
            expected_output_count=1,
            output_node_ids=("9",),
            expected_media_types=("image/png",),
            execution_parameters=execution_parameters,
        )
        variant = PlannedVariant.model_validate(
            {
                **draft.model_dump(mode="python"),
                "exact_run_hash": recompute_exact_run_hash(draft),
            }
        )
        planned.append(variant)
        candidates.append(
            {
                "candidate_id": variant_id,
                "confidence": confidence.value,
                "evidence_ids": [f"workflow-{index}"],
                "engagement": engagement,
                "provenance_rank": provenance,
            }
        )
    approval_payload = {
        "source_hashes": ["a" * 64],
        "variants": [item.model_dump(mode="json") for item in planned],
        "capability_coverage": [item.value for item in coverage],
        "warnings": [],
        "candidates": candidates,
    }
    plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=("a" * 64,),
        variants=tuple(planned),
        capability_coverage=coverage,
        warnings=(),
        approval_payload=approval_payload,
    )
    for variant in plan.variants:
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


def _persist_plan_revision(
    config: LabConfig,
    original_plan: ExperimentPlan,
    variants: tuple[PlannedVariant, ...],
    *,
    experiment_id: str = "wave-approved",
    candidate_scores: dict[str, tuple[int, int]] | None = None,
) -> ExperimentPlan:
    scores = candidate_scores or {}
    approval_payload = {
        "source_hashes": list(original_plan.source_hashes),
        "variants": [item.model_dump(mode="json") for item in variants],
        "capability_coverage": [
            item.value for item in original_plan.capability_coverage
        ],
        "warnings": list(original_plan.warnings),
        "candidates": [
            {
                "candidate_id": item.variant_id,
                "confidence": (
                    item.confidence or PlanningConfidence.OBSERVED
                ).value,
                "evidence_ids": list(item.evidence_ids),
                "engagement": scores.get(item.variant_id, (0, 0))[0],
                "provenance_rank": scores.get(item.variant_id, (0, 0))[1],
            }
            for item in variants
        ],
    }
    plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=original_plan.source_hashes,
        variants=variants,
        capability_coverage=original_plan.capability_coverage,
        warnings=original_plan.warnings,
        approval_payload=approval_payload,
    )
    store = ExperimentStore(config.experiments_root)
    for variant in variants:
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
    return plan


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

    for index, executor in enumerate(executors):
        engine = engine_module.ExperimentEngine(
            config=config,
            executor_factory=lambda executor=executor: executor,
            downloader=lambda _url, _target: None,
            sleep=lambda _seconds: None,
        )
        if index == 0:
            engine.run(
                "plan-approved",
                "baseline",
                approved_plan_hash=plan.plan_hash,
            )
        else:
            engine.rerun(
                "plan-approved",
                "baseline",
                approved_plan_hash=plan.plan_hash,
                reason=f"exercise sanitized diagnostic path {index}",
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
    orphan = (
        config.experiments_root
        / "plan-approved"
        / f"outputs/baseline/{interrupted.run_id}/0000.png"
    )
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
    second = engine.rerun(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
        reason="verify exact-output reproducibility",
    )

    assert first.state is second.state is RunState.SUCCESS
    assert first.run_id != second.run_id
    assert first.outputs[0].content_sha256 == second.outputs[0].content_sha256
    assert first.outputs[0].local_relative_path != second.outputs[0].local_relative_path
    assert first.outputs[0].local_relative_path == (
        f"outputs/baseline/{first.run_id}/0000.png"
    )
    assert second.outputs[0].local_relative_path == (
        f"outputs/baseline/{second.run_id}/0000.png"
    )
    assert second.original_run_ids == (first.run_id,)
    for record in (first, second):
        output_path = (
            config.experiments_root
            / "plan-approved"
            / record.outputs[0].local_relative_path
        )
        assert output_path.read_bytes() == content


def test_different_rerun_output_bytes_coexist_without_clobber(tmp_path: Path) -> None:
    config, plan = _persist_plan(tmp_path)
    outputs = (
        ProviderOutput(
            node_id="9",
            url="memory://first.png",
            media_type="image/png",
            provider_order=0,
        ),
        ProviderOutput(
            node_id="9",
            url="memory://second.png",
            media_type="image/png",
            provider_order=0,
        ),
    )
    executors = [
        FakeExecutor.success(outputs=(outputs[0],)),
        FakeExecutor.success(outputs=(outputs[1],)),
    ]

    def png_bytes(color: tuple[int, int, int]) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (1, 1), color=color).save(buffer, format="PNG")
        return buffer.getvalue()

    contents = {
        outputs[0].url: png_bytes((12, 34, 56)),
        outputs[1].url: png_bytes((65, 43, 21)),
    }

    def downloader(url: str, target: Path) -> Path:
        target.write_bytes(contents[url])
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
    second = engine.rerun(
        "plan-approved",
        "baseline",
        approved_plan_hash=plan.plan_hash,
        reason="measure output variance",
    )

    assert first.outputs[0].local_relative_path == (
        f"outputs/baseline/{first.run_id}/0000.png"
    )
    assert second.outputs[0].local_relative_path == (
        f"outputs/baseline/{second.run_id}/0000.png"
    )
    assert first.outputs[0].content_sha256 != second.outputs[0].content_sha256
    first_path = (
        config.experiments_root
        / "plan-approved"
        / first.outputs[0].local_relative_path
    )
    second_path = (
        config.experiments_root
        / "plan-approved"
        / second.outputs[0].local_relative_path
    )
    assert first_path.read_bytes() == contents[outputs[0].url]
    assert second_path.read_bytes() == contents[outputs[1].url]


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
        f"outputs/baseline/{record.run_id}/0000.png",
        f"outputs/baseline/{record.run_id}/0001.jpg",
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


def _wave_engine(
    config: LabConfig,
    executor: FakeExecutor,
):
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")

    def downloader(_url: str, target: Path) -> None:
        target.write_bytes(valid_media_bytes("image/png"))

    return engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: executor,
        downloader=downloader,
        sleep=lambda _seconds: None,
    )


def _persist_active_wave_run(
    config: LabConfig,
    plan: ExperimentPlan,
    *,
    variant_id: str,
    state: RunState,
    run_id: str = "run-wave-active",
    provider_task_id: str = "provider-wave-active",
    record_variant_id: str | None = None,
) -> RunRecord:
    variant = next(item for item in plan.variants if item.variant_id == variant_id)
    record = RunRecord(
        run_id=run_id,
        experiment_id="wave-approved",
        variant_id=record_variant_id or variant.variant_id,
        state=state,
        plan_hash=plan.plan_hash,
        exact_run_hash=variant.exact_run_hash,
        provider_task_id=provider_task_id,
        expected_output_count=variant.expected_output_count,
        expected_output_node_ids=variant.output_node_ids,
        expected_media_types=variant.expected_media_types,
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        f"runs/{record.run_id}.json",
        record.model_dump(mode="json"),
    )
    return record


@pytest.mark.parametrize("state", [RunState.SUBMITTED, RunState.RUNNING])
def test_public_run_resumes_exact_active_fingerprint_without_resubmission(
    tmp_path: Path,
    state: RunState,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=state,
    )
    executor = FakeExecutor(
        poll_sequences={
            existing.provider_task_id: (ProviderTaskState.RUNNING,)
        }
    )
    engine = _wave_engine(config, executor)

    recovered = engine.run(
        "wave-approved",
        "active",
        approved_plan_hash=plan.plan_hash,
    )

    assert recovered.run_id == existing.run_id
    assert recovered.state is RunState.RUNNING
    assert executor.calls == ["poll"]
    assert executor.submissions == []


@pytest.mark.parametrize(
    "initial_state",
    [RunState.PLANNED, RunState.INCOMPLETE],
    ids=("planned-with-intent", "reconcile-submission"),
)
def test_unresolved_submission_is_pending_and_never_saturates(
    tmp_path: Path,
    initial_state: RunState,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(("pending", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),),
    )
    variant = plan.variants[0]
    values: dict[str, object] = {
        "run_id": "run-submission-uncertain",
        "experiment_id": "wave-approved",
        "variant_id": variant.variant_id,
        "state": initial_state,
        "plan_hash": plan.plan_hash,
        "exact_run_hash": variant.exact_run_hash,
        "expected_output_count": variant.expected_output_count,
        "expected_output_node_ids": variant.output_node_ids,
        "expected_media_types": variant.expected_media_types,
    }
    if initial_state is RunState.PLANNED:
        values["submission_intent_at"] = datetime(2026, 7, 14, 12, tzinfo=UTC)
    else:
        values["error"] = {"kind": "ambiguous_submission"}
    record = RunRecord.model_validate(values)
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
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

    wave = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    durable = engine.status("wave-approved", record.run_id)
    assert durable.state is RunState.INCOMPLETE
    assert durable.recovery_disposition == "reconcile_submission"
    assert wave.pending_variant_ids == ("pending",)
    assert wave.pending_run_ids == (record.run_id,)
    assert wave.objective_evidence["run_states"] == {"pending": "INCOMPLETE"}
    assert wave.saturation_reason is None
    assert constructions == []


def _persist_incomplete_wave_run(
    config: LabConfig,
    plan: ExperimentPlan,
    *,
    error: dict[str, object],
    actual_output_count: int = 0,
    expected_output_count: int = 1,
) -> RunRecord:
    variant = plan.variants[0]
    record = RunRecord(
        run_id="run-wave-incomplete",
        experiment_id="wave-approved",
        variant_id=variant.variant_id,
        state=RunState.INCOMPLETE,
        plan_hash=plan.plan_hash,
        exact_run_hash=variant.exact_run_hash,
        provider_task_id="provider-wave-incomplete",
        expected_output_count=expected_output_count,
        expected_output_node_ids=variant.output_node_ids,
        expected_media_types=variant.expected_media_types,
        actual_output_count=actual_output_count,
        error=error,
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        f"runs/{record.run_id}.json",
        record.model_dump(mode="json"),
    )
    return record


def _persist_tampered_incomplete_wave_run(
    config: LabConfig,
    plan: ExperimentPlan,
    *,
    error: dict[str, object],
    recovery_disposition: str,
    actual_output_count: int = 0,
    expected_output_count: int = 1,
) -> RunRecord:
    variant = plan.variants[0]
    record = RunRecord(
        run_id="run-wave-tampered",
        experiment_id="wave-approved",
        variant_id=variant.variant_id,
        state=RunState.INCOMPLETE,
        plan_hash=plan.plan_hash,
        exact_run_hash=variant.exact_run_hash,
        provider_task_id="provider-wave-tampered",
        expected_output_count=expected_output_count,
        expected_output_node_ids=variant.output_node_ids,
        expected_media_types=variant.expected_media_types,
        actual_output_count=actual_output_count,
        error=error,
    )
    payload = record.model_dump(mode="json")
    payload["recovery_disposition"] = recovery_disposition
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        f"runs/{record.run_id}.json",
        payload,
    )
    return record


def test_wave_artifact_write_never_clobbers_conflicting_ordinal(
    tmp_path: Path,
) -> None:
    config, _plan = _persist_wave_plan(tmp_path)
    store = ExperimentStore(config.experiments_root)
    original = {"ordinal": 1, "owner": "existing-wave"}
    store.write_json_new(
        "wave-approved",
        "plans/waves/1.json",
        original,
    )

    with pytest.raises(SafetyError, match="refusing to overwrite"):
        store.write_json_new(
            "wave-approved",
            "plans/waves/1.json",
            {"ordinal": 1, "owner": "conflicting-wave"},
        )

    assert json.loads(
        store.read_text("wave-approved", "plans/waves/1.json")
    ) == original


def test_fabricated_saturation_wave_without_runs_fails_closed(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    artifact = WaveArtifact(
        ordinal=1,
        plan_hash=plan.plan_hash,
        considered_variant_ids=tuple(item.variant_id for item in plan.variants),
        skips={
            item.variant_id: "exact_run_already_recorded"
            for item in plan.variants
        },
        candidate_evidence={item.variant_id: {} for item in plan.variants},
        objective_evidence={"run_states": {}},
        saturation_reason=(
            "no_approved_non_dominated_novel_capability_signature_remains"
        ),
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        "plans/waves/1.json",
        artifact.model_dump(mode="json"),
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(SafetyError, match="wave artifact.*conflict"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    assert constructions == []


def test_altered_wave_selection_fails_closed_against_durable_runs(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    engine = _wave_engine(config, executor)
    original = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )
    assert original.selected_variant_ids == ("strong", "other")

    artifact_path = (
        config.experiments_root
        / "wave-approved"
        / "plans"
        / "waves"
        / "1.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["selected_variant_ids"] = ["other", "strong"]
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    constructions: list[str] = []
    validating_engine = importlib.import_module(
        "mp_comfyui_lab.experiments.engine"
    ).ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(SafetyError, match="wave artifact.*conflict"):
        validating_engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    assert constructions == []


def test_run_wave_executes_all_novel_capabilities_and_persists_decision(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
                provider_order=0,
            ),
        )
    )
    engine = _wave_engine(config, executor)

    wave = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert wave.ordinal == 1
    assert wave.plan_hash == plan.plan_hash
    assert wave.selected_variant_ids == ("strong", "other")
    assert wave.saturation_reason is None
    assert [item.variant_id for item, _fixtures in executor.submissions] == [
        "strong",
        "other",
    ]
    artifact_path = (
        config.experiments_root
        / "wave-approved"
        / "plans"
        / "waves"
        / "1.json"
    )
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert persisted["completed"] is True
    assert persisted["considered_variant_ids"] == [
        "strong",
        "weak-popular",
        "other",
    ]
    assert persisted["selected_variant_ids"] == ["strong", "other"]
    assert persisted["skips"]["weak-popular"] == (
        "weaker_representative_of_novel_capability"
    )
    assert persisted["objective_evidence"]["run_states"] == {
        "other": "SUCCESS",
        "strong": "SUCCESS",
    }
    assert persisted["objective_evidence"]["output_duplicate_groups"] == [
        ["other", "strong"]
    ]
    assert persisted["objective_evidence"]["actual_output_counts"] == {
        "other": 1,
        "strong": 1,
    }
    assert "task_budget" not in persisted
    assert "max_tasks" not in persisted
    assert len(persisted["run_ids"]) == 2


def test_completed_saturation_wave_resumes_without_regenerating_runs(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    engine = _wave_engine(config, executor)
    first = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )
    submitted = len(executor.submissions)

    saturated = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )
    resumed = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert first.ordinal == 1
    assert saturated.ordinal == 2
    assert saturated.selected_variant_ids == ()
    assert saturated.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )
    assert resumed == saturated
    assert len(executor.submissions) == submitted
    waves_root = (
        config.experiments_root / "wave-approved" / "plans" / "waves"
    )
    assert sorted(path.name for path in waves_root.glob("*.json")) == [
        "1.json",
        "2.json",
    ]


def test_wave_resume_uses_durable_runs_even_without_a_wave_artifact(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    engine = _wave_engine(config, executor)

    original = engine.run(
        "wave-approved",
        "strong",
        approved_plan_hash=plan.plan_hash,
    )
    wave = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert original.state is RunState.SUCCESS
    assert wave.selected_variant_ids == ("other",)
    assert [item.variant_id for item, _fixtures in executor.submissions] == [
        "strong",
        "other",
    ]
    assert wave.skips["strong"] == "exact_run_already_recorded"
    assert wave.skips["weak-popular"] == "dominated_by_tested_capability"


@pytest.mark.parametrize("state", [RunState.SUBMITTED, RunState.RUNNING])
def test_wave_recovers_existing_provider_task_without_resubmission(
    tmp_path: Path,
    state: RunState,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=state,
    )
    executor = FakeExecutor(
        poll_sequences={existing.provider_task_id: (ProviderTaskState.FAILED,)}
    )
    engine = _wave_engine(config, executor)

    wave = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    recovered = engine.status("wave-approved", existing.run_id)
    assert recovered.state is RunState.FAILED
    assert wave.selected_variant_ids == ()
    assert wave.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )
    assert executor.calls == ["poll"]
    assert executor.polls == [existing.provider_task_id]
    assert executor.submissions == []


def test_stale_saturation_with_active_run_fails_closed_before_polling(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=RunState.RUNNING,
    )
    prior = WaveArtifact(
        ordinal=1,
        plan_hash=plan.plan_hash,
        considered_variant_ids=("active",),
        skips={"active": "exact_run_already_recorded"},
        candidate_evidence={"active": {}},
        objective_evidence={"run_states": {}},
        saturation_reason=(
            "no_approved_non_dominated_novel_capability_signature_remains"
        ),
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        "plans/waves/1.json",
        prior.model_dump(mode="json"),
    )
    executor = FakeExecutor(
        poll_sequences={existing.provider_task_id: (ProviderTaskState.FAILED,)}
    )
    engine = _wave_engine(config, executor)

    with pytest.raises(SafetyError, match="wave artifact.*conflict"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    assert engine.status("wave-approved", existing.run_id).state is RunState.RUNNING
    assert executor.calls == []
    assert executor.submissions == []


def test_wave_tracks_still_active_exact_fingerprint_after_variant_rename(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("renamed", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="renamed",
        record_variant_id="old-name",
        state=RunState.SUBMITTED,
    )
    executor = FakeExecutor(
        poll_sequences={existing.provider_task_id: (ProviderTaskState.RUNNING,)}
    )
    engine = _wave_engine(config, executor)

    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert pending.pending_variant_ids == ("renamed",)
    assert pending.pending_run_ids == (existing.run_id,)
    assert pending.saturation_reason is None
    assert executor.calls == ["poll"]
    assert executor.submissions == []


def test_wave_retains_all_active_run_ids_for_one_exact_fingerprint(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    submitted = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=RunState.SUBMITTED,
        run_id="run-a",
        provider_task_id="task-a",
    )
    running = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=RunState.RUNNING,
        run_id="run-b",
        provider_task_id="task-b",
    )
    executor = FakeExecutor(
        poll_sequences={
            submitted.provider_task_id: (ProviderTaskState.RUNNING,),
            running.provider_task_id: (ProviderTaskState.RUNNING,),
        }
    )
    engine = _wave_engine(config, executor)

    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert pending.pending_variant_ids == ("active",)
    assert pending.pending_run_ids == ("run-a", "run-b")
    assert pending.saturation_reason is None
    assert executor.calls == ["poll", "poll"]
    assert executor.submissions == []


def test_wave_persists_still_active_run_as_pending_not_saturation(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=RunState.SUBMITTED,
    )
    output = ProviderOutput(
        node_id="9",
        url="memory://result.png",
        media_type="image/png",
    )
    executor = FakeExecutor(
        poll_sequences={
            existing.provider_task_id: (
                ProviderTaskState.RUNNING,
                ProviderTaskState.SUCCESS,
            )
        },
        output_scripts={existing.provider_task_id: ((output,),)},
    )
    engine = _wave_engine(config, executor)

    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert pending.selected_variant_ids == ()
    assert pending.pending_variant_ids == ("active",)
    assert pending.pending_run_ids == (existing.run_id,)
    assert pending.saturation_reason is None
    assert pending.objective_evidence["run_states"] == {"active": "RUNNING"}
    assert engine.status("wave-approved", existing.run_id).state is RunState.RUNNING
    assert executor.calls == ["poll"]
    assert executor.submissions == []

    saturated = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert saturated.ordinal == pending.ordinal + 1
    assert saturated.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )
    assert engine.status("wave-approved", existing.run_id).state is RunState.SUCCESS
    assert executor.calls == ["poll", "poll", "collect_outputs"]
    assert executor.submissions == []


@pytest.mark.parametrize("state", [RunState.SUBMITTED, RunState.RUNNING])
def test_wave_poll_retry_exhaustion_keeps_paid_task_pending(
    tmp_path: Path,
    state: RunState,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_active_wave_run(
        config,
        plan,
        variant_id="active",
        state=state,
    )
    executor = FakeExecutor(
        method_exceptions={
            "poll": [TimeoutError("poll timed out") for _ in range(4)]
        }
    )
    engine = _wave_engine(config, executor)

    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert engine.status("wave-approved", existing.run_id).state is state
    assert pending.selected_variant_ids == ()
    assert pending.pending_variant_ids == ("active",)
    assert pending.pending_run_ids == (existing.run_id,)
    assert pending.saturation_reason is None
    assert executor.calls == ["poll"] * 4
    assert executor.submissions == []


def test_inconclusive_incomplete_run_repeatedly_resumes_without_saturation(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_incomplete_wave_run(
        config,
        plan,
        error={"kind": "poll_error", "diagnostic": "status unavailable"},
    )
    executor = FakeExecutor(
        method_exceptions={
            "poll": [TimeoutError("poll timed out") for _ in range(8)]
        }
    )
    engine = _wave_engine(config, executor)

    first = engine.resume("wave-approved", existing.run_id)
    second = engine.resume("wave-approved", existing.run_id)
    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert first.state is RunState.INCOMPLETE
    assert second.state is RunState.INCOMPLETE
    assert first.provider_task_id == existing.provider_task_id
    assert second.provider_task_id == existing.provider_task_id
    assert first.recovery_disposition == "resume_provider_task"
    assert second.recovery_disposition == "resume_provider_task"
    assert pending.selected_variant_ids == ()
    assert pending.pending_variant_ids == ("active",)
    assert pending.pending_run_ids == (existing.run_id,)
    assert pending.saturation_reason is None
    assert executor.calls == ["poll"] * 9
    assert executor.submissions == []


@pytest.mark.parametrize(
    "error",
    [
        {"kind": "collection_error", "diagnostic": "outputs unavailable"},
        {"kind": "retry_exhausted", "operation": "collect_outputs"},
    ],
    ids=("collection-error", "collection-retries-exhausted"),
)
def test_collection_uncertainty_incomplete_run_remains_pending(
    tmp_path: Path,
    error: dict[str, object],
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_incomplete_wave_run(config, plan, error=error)
    executor = FakeExecutor(
        poll_sequences={
            existing.provider_task_id: (ProviderTaskState.RUNNING,)
        }
    )
    engine = _wave_engine(config, executor)

    pending = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert pending.pending_variant_ids == ("active",)
    assert pending.pending_run_ids == (existing.run_id,)
    assert pending.saturation_reason is None
    assert executor.calls == ["poll"]
    assert executor.submissions == []


def test_conclusive_partial_incomplete_run_remains_terminal(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("partial", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    existing = _persist_incomplete_wave_run(
        config,
        plan,
        error={
            "kind": "partial_outputs",
            "expected_output_count": 2,
            "actual_output_count": 1,
        },
        actual_output_count=1,
        expected_output_count=2,
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    resumed = engine.resume("wave-approved", existing.run_id)
    saturated = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert resumed.state is RunState.INCOMPLETE
    assert resumed.recovery_disposition == "terminal"
    assert saturated.pending_variant_ids == ()
    assert saturated.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )
    assert constructions == []


def test_poll_error_explicitly_marked_terminal_cannot_saturate(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("active", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    _persist_tampered_incomplete_wave_run(
        config,
        plan,
        error={"kind": "poll_error", "diagnostic": "status unavailable"},
        recovery_disposition="terminal",
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(SafetyError, match="run artifact is missing or invalid"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    assert constructions == []
    assert not (
        config.experiments_root / "wave-approved" / "plans" / "waves"
    ).exists()


def test_partial_outputs_marked_resumable_cannot_be_polled_as_active(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("partial", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    _persist_tampered_incomplete_wave_run(
        config,
        plan,
        error={
            "kind": "partial_outputs",
            "expected_output_count": 2,
            "actual_output_count": 1,
        },
        recovery_disposition="resume_provider_task",
        actual_output_count=1,
        expected_output_count=2,
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(SafetyError, match="run artifact is missing or invalid"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    assert constructions == []
    assert not (
        config.experiments_root / "wave-approved" / "plans" / "waves"
    ).exists()


def test_wave_recovers_crash_after_submission_without_double_charge(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("crash", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    crashing_executor = FakeExecutor(
        method_exceptions={"poll": SystemExit("crash after submission")}
    )
    crashing_engine = _wave_engine(config, crashing_executor)

    with pytest.raises(SystemExit, match="crash after submission"):
        crashing_engine.run_wave(
            "wave-approved",
            approved_plan_hash=plan.plan_hash,
        )

    run_path = next(
        (config.experiments_root / "wave-approved" / "runs").glob("*.json")
    )
    submitted = RunRecord.model_validate_json(
        run_path.read_text(encoding="utf-8"),
        strict=True,
    )
    assert submitted.state is RunState.SUBMITTED
    assert submitted.provider_task_id == "fake-task-1"
    assert crashing_executor.calls == ["validate_variant", "submit", "poll"]
    assert len(crashing_executor.submissions) == 1
    assert not (
        config.experiments_root / "wave-approved" / "plans" / "waves"
    ).exists()

    recovery_executor = FakeExecutor(
        poll_sequences={"fake-task-1": (ProviderTaskState.FAILED,)}
    )
    recovery_engine = _wave_engine(config, recovery_executor)

    wave = recovery_engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert recovery_engine.status("wave-approved", submitted.run_id).state is RunState.FAILED
    assert wave.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )
    assert recovery_executor.calls == ["poll"]
    assert recovery_executor.submissions == []


def test_wave_approval_failure_constructs_no_executor(tmp_path: Path) -> None:
    config, plan = _persist_wave_plan(tmp_path)
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash="f" * 64,
        )

    assert plan.plan_hash != "f" * 64
    assert constructions == []


def test_wave_approval_rejects_tampered_capability_hash_preimage(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("tampered", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    original = plan.variants[0]
    fabricated_hash = (
        "f" * 64 if original.capability_hash != "f" * 64 else "e" * 64
    )
    tampered_variant = PlannedVariant.model_validate(
        {
            **original.model_dump(mode="python"),
            "capability_hash": fabricated_hash,
        }
    )
    approval_payload = thaw_json(plan.approval_payload)
    approval_payload["variants"] = [
        tampered_variant.model_dump(mode="json")
    ]
    tampered_plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=plan.source_hashes,
        variants=(tampered_variant,),
        capability_coverage=plan.capability_coverage,
        warnings=plan.warnings,
        approval_payload=approval_payload,
    )
    store = ExperimentStore(config.experiments_root)
    store.write_json_new(
        "wave-approved",
        f"plans/{tampered_plan.plan_hash}.json",
        tampered_plan.model_dump(mode="json"),
    )
    completed = RunRecord(
        run_id="run-tampered-complete",
        experiment_id="wave-approved",
        variant_id=tampered_variant.variant_id,
        state=RunState.FAILED,
        plan_hash=tampered_plan.plan_hash,
        exact_run_hash=tampered_variant.exact_run_hash,
        expected_output_count=tampered_variant.expected_output_count,
        expected_output_node_ids=tampered_variant.output_node_ids,
        expected_media_types=tampered_variant.expected_media_types,
        error={"kind": "provider_failed"},
    )
    store.write_json_new(
        "wave-approved",
        f"runs/{completed.run_id}.json",
        completed.model_dump(mode="json"),
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(
        engine_module.ApprovalError,
        match="capability hash",
    ):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=tampered_plan.plan_hash,
        )

    assert constructions == []
    assert not (
        config.experiments_root / "wave-approved" / "plans" / "waves"
    ).exists()


@pytest.mark.parametrize("failed_history", [False, True], ids=("unselected", "failed"))
def test_wave_approval_validates_every_variant_exact_hash_preimage(
    tmp_path: Path,
    failed_history: bool,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("selected", "d" * 64, PlanningConfidence.OBSERVED, 2, 2),
            ("unselected", "d" * 64, PlanningConfidence.WEAKLY_INFERRED, 1, 1),
        ),
    )
    original = plan.variants[1]
    fabricated_hash = (
        "f" * 64 if original.exact_run_hash != "f" * 64 else "e" * 64
    )
    tampered_variant = PlannedVariant.model_validate(
        {
            **original.model_dump(mode="python"),
            "exact_run_hash": fabricated_hash,
        }
    )
    variants = (plan.variants[0], tampered_variant)
    approval_payload = thaw_json(plan.approval_payload)
    approval_payload["variants"] = [
        item.model_dump(mode="json") for item in variants
    ]
    tampered_plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=plan.source_hashes,
        variants=variants,
        capability_coverage=plan.capability_coverage,
        warnings=plan.warnings,
        approval_payload=approval_payload,
    )
    store = ExperimentStore(config.experiments_root)
    store.write_json_new(
        "wave-approved",
        f"plans/{tampered_plan.plan_hash}.json",
        tampered_plan.model_dump(mode="json"),
    )
    if failed_history:
        failed = RunRecord(
            run_id="run-tampered-failed",
            experiment_id="wave-approved",
            variant_id=tampered_variant.variant_id,
            state=RunState.FAILED,
            plan_hash=tampered_plan.plan_hash,
            exact_run_hash=tampered_variant.exact_run_hash,
            expected_output_count=tampered_variant.expected_output_count,
            expected_output_node_ids=tampered_variant.output_node_ids,
            expected_media_types=tampered_variant.expected_media_types,
            error={"kind": "provider_failed"},
        )
        store.write_json_new(
            "wave-approved",
            f"runs/{failed.run_id}.json",
            failed.model_dump(mode="json"),
        )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError, match="exact run hash"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=tampered_plan.plan_hash,
        )

    assert constructions == []


def test_unapproved_hypothesis_dimension_requires_reapproval_before_executor(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("new-sampler", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
        hypothesis=DimensionName.SAMPLER,
        coverage=(DimensionName.CHECKPOINT,),
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError, match="approved dimensions"):
        engine.run(
            "wave-approved",
            "new-sampler",
            approved_plan_hash=plan.plan_hash,
        )

    assert constructions == []
    assert list(
        (config.experiments_root / "wave-approved" / "runs").glob("*.json")
    ) == []


def test_correction_outside_approved_dimensions_changes_hash_and_requires_reapproval(
    tmp_path: Path,
) -> None:
    _, baseline = _persist_wave_plan(
        tmp_path,
        experiment_id="baseline-plan",
        variants=(
            ("candidate", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    correction = {
        "schema_version": 1,
        "failure_signature": "missing-node:TextEncodeQwen",
        "failed_variant_id": "failed-baseline",
        "failed_exact_run_hash": "e" * 64,
        "failed_capability_hash": "f" * 64,
        "candidate_variant_id": "candidate",
        "changed_dimensions": [DimensionName.SAMPLER.value],
    }
    config, revised = _persist_wave_plan(
        tmp_path,
        variants=(
            ("candidate", "d" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
        corrections={"candidate": correction},
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError, match="approved dimensions"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=revised.plan_hash,
        )

    assert revised.plan_hash != baseline.plan_hash
    assert revised.variants[0].exact_run_hash != baseline.variants[0].exact_run_hash
    assert constructions == []
    assert list(
        (config.experiments_root / "wave-approved" / "runs").glob("*.json")
    ) == []


@pytest.mark.parametrize(
    "delta_kind",
    ["unapproved-sampler", "unclassified-node", "unclassified-lora-input"],
)
def test_actual_unapproved_correction_dimension_requires_reapproval(
    tmp_path: Path,
    delta_kind: str,
) -> None:
    approved_dimension = (
        DimensionName.LORA_STACK
        if delta_kind == "unclassified-lora-input"
        else DimensionName.CHECKPOINT
    )
    config, original_plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("failed", "d" * 64, PlanningConfidence.OBSERVED, 2, 2),
            ("candidate", "e" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
        hypothesis=approved_dimension,
        coverage=(approved_dimension,),
    )

    def graph(
        sampler: str,
        *,
        checkpoint: str = "base.safetensors",
        custom_node: bool = False,
        lora: str | None = None,
        lora_unapproved_mode: bool = False,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
        }
        model_node_id = "1"
        if lora is not None:
            result["2"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": lora,
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                },
            }
            if lora_unapproved_mode:
                result["2"]["inputs"]["unapproved_mode"] = "experimental"
            model_node_id = "2"
        result.update(
            {
                "8": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": [model_node_id, 0],
                        "sampler_name": sampler,
                        "scheduler": "normal",
                        "seed": 1,
                    },
                },
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["8", 0]},
                },
            }
        )
        if custom_node:
            result["90"] = {
                "class_type": "UnapprovedCustomNode",
                "inputs": {"mode": "checkpoint"},
            }
        return result

    failed_graph = graph(
        "euler",
        lora=(
            "old.safetensors"
            if delta_kind == "unclassified-lora-input"
            else None
        ),
    )
    failed_draft = PlannedVariant.model_validate(
        {
            **original_plan.variants[0].model_dump(mode="python"),
            "materialized_graph": failed_graph,
            "capability_hash": graph_capability_hash(failed_graph),
            "exact_run_hash": "0" * 64,
        }
    )
    failed = PlannedVariant.model_validate(
        {
            **failed_draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(failed_draft),
        }
    )
    correction = {
        "schema_version": 1,
        "failure_signature": "provider-failed:unsupported sampler",
        "failed_variant_id": failed.variant_id,
        "failed_exact_run_hash": failed.exact_run_hash,
        "failed_capability_hash": failed.capability_hash,
        "candidate_variant_id": "candidate",
        "changed_dimensions": [approved_dimension.value],
    }
    candidate_parameters = thaw_json(
        original_plan.variants[1].execution_parameters
    )
    candidate_parameters["correction"] = correction
    if delta_kind == "unapproved-sampler":
        candidate_graph = graph("dpmpp_2m")
    elif delta_kind == "unclassified-node":
        candidate_graph = graph(
            "euler",
            checkpoint="revised.safetensors",
            custom_node=True,
        )
    else:
        candidate_graph = graph(
            "euler",
            lora="new.safetensors",
            lora_unapproved_mode=True,
        )
    candidate_draft = PlannedVariant.model_validate(
        {
            **original_plan.variants[1].model_dump(mode="python"),
            "materialized_graph": candidate_graph,
            "capability_hash": graph_capability_hash(candidate_graph),
            "execution_parameters": candidate_parameters,
            "exact_run_hash": "0" * 64,
        }
    )
    candidate = PlannedVariant.model_validate(
        {
            **candidate_draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(candidate_draft),
        }
    )
    variants = (failed, candidate)
    approval_payload = thaw_json(original_plan.approval_payload)
    approval_payload["variants"] = [
        item.model_dump(mode="json") for item in variants
    ]
    revised_plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=original_plan.source_hashes,
        variants=variants,
        capability_coverage=original_plan.capability_coverage,
        warnings=original_plan.warnings,
        approval_payload=approval_payload,
    )
    ExperimentStore(config.experiments_root).write_json_new(
        "wave-approved",
        f"plans/{revised_plan.plan_hash}.json",
        revised_plan.model_dump(mode="json"),
    )
    for variant in variants:
        (
            config.experiments_root
            / "wave-approved"
            / "variants"
            / f"{variant.variant_id}.json"
        ).write_text(
            json.dumps(variant.model_dump(mode="json")),
            encoding="utf-8",
        )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(engine_module.ApprovalError, match="approved dimensions"):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=revised_plan.plan_hash,
        )

    assert constructions == []


@pytest.mark.parametrize(
    "legacy_record",
    [False, True],
    ids=("variant-snapshot", "referenced-plan"),
)
def test_revised_plan_uses_success_history_after_variant_rename_and_removal(
    tmp_path: Path,
    legacy_record: bool,
) -> None:
    config, original_plan = _persist_wave_plan(
        tmp_path,
        variants=(("old-name", "d" * 64, PlanningConfidence.OBSERVED, 20, 3),),
    )
    output = ProviderOutput(
        node_id="9",
        url="memory://old-success.png",
        media_type="image/png",
    )
    original_engine = _wave_engine(
        config,
        FakeExecutor.success(outputs=(output,)),
    )
    original_run = original_engine.run(
        "wave-approved",
        "old-name",
        approved_plan_hash=original_plan.plan_hash,
    )
    assert original_run.state is RunState.SUCCESS
    if legacy_record:
        run_path = (
            config.experiments_root
            / "wave-approved"
            / "runs"
            / f"{original_run.run_id}.json"
        )
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        payload.pop("variant_snapshot")
        run_path.write_text(json.dumps(payload), encoding="utf-8")

    original = original_plan.variants[0]
    parameters = thaw_json(original.execution_parameters)
    parameters["prompt_lines"] = ["revised exact input"]
    renamed_draft = PlannedVariant.model_validate(
        {
            **original.model_dump(mode="python"),
            "variant_id": "new-name",
            "confidence": PlanningConfidence.WEAKLY_INFERRED,
            "evidence_ids": ("new-evidence",),
            "execution_parameters": parameters,
            "exact_run_hash": "0" * 64,
        }
    )
    renamed = PlannedVariant.model_validate(
        {
            **renamed_draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(renamed_draft),
        }
    )
    assert renamed.exact_run_hash != original.exact_run_hash
    assert renamed.capability_hash == original.capability_hash
    revised_plan = _persist_plan_revision(
        config,
        original_plan,
        (renamed,),
        candidate_scores={"new-name": (1, 1)},
    )
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    revised_engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    wave = revised_engine.run_wave(
        "wave-approved",
        approved_plan_hash=revised_plan.plan_hash,
    )

    assert wave.selected_variant_ids == ()
    assert wave.skips[renamed.variant_id] == "dominated_by_tested_capability"
    assert constructions == []


@pytest.mark.parametrize("tamper_mode", ["missing", "altered-evidence"])
def test_revised_plan_fails_closed_when_historical_plan_evidence_is_untrusted(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    config, original_plan = _persist_wave_plan(
        tmp_path,
        variants=(("old-name", "d" * 64, PlanningConfidence.OBSERVED, 20, 3),),
    )
    original_run = _wave_engine(
        config,
        FakeExecutor.success(
            outputs=(
                ProviderOutput(
                    node_id="9",
                    url="memory://old-success.png",
                    media_type="image/png",
                ),
            )
        ),
    ).run(
        "wave-approved",
        "old-name",
        approved_plan_hash=original_plan.plan_hash,
    )
    assert original_run.state is RunState.SUCCESS

    original = original_plan.variants[0]
    parameters = thaw_json(original.execution_parameters)
    parameters["prompt_lines"] = ["revised exact input"]
    revised_draft = PlannedVariant.model_validate(
        {
            **original.model_dump(mode="python"),
            "variant_id": "new-name",
            "execution_parameters": parameters,
            "exact_run_hash": "0" * 64,
        }
    )
    revised = PlannedVariant.model_validate(
        {
            **revised_draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(revised_draft),
        }
    )
    revised_plan = _persist_plan_revision(
        config,
        original_plan,
        (revised,),
        candidate_scores={"new-name": (1, 1)},
    )
    historical_plan_path = (
        config.experiments_root
        / "wave-approved"
        / "plans"
        / f"{original_plan.plan_hash}.json"
    )
    if tamper_mode == "missing":
        historical_plan_path.unlink()
    else:
        payload = json.loads(historical_plan_path.read_text(encoding="utf-8"))
        payload["approval_payload"]["candidates"][0]["engagement"] = 999
        historical_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    constructions: list[str] = []
    engine_module = importlib.import_module("mp_comfyui_lab.experiments.engine")
    engine = engine_module.ExperimentEngine(
        config=config,
        executor_factory=lambda: constructions.append("executor"),
        downloader=lambda _url, _target: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(
        SafetyError,
        match="durable run references a missing or invalid approved plan",
    ):
        engine.run_wave(
            "wave-approved",
            approved_plan_hash=revised_plan.plan_hash,
        )

    assert constructions == []


def test_corrected_candidate_uses_two_failures_from_removed_old_plan(
    tmp_path: Path,
) -> None:
    config, original_plan = _persist_wave_plan(
        tmp_path,
        variants=(("old-failure", "d" * 64, PlanningConfidence.OBSERVED, 2, 2),),
    )
    signature = "missing-node:TextEncodeQwen"
    failing_executor = FakeExecutor(
        validation=ValidationReport(
            errors=(
                ValidationIssue(
                    code="missing-node",
                    message="TextEncodeQwen is unavailable",
                ),
            )
        )
    )
    failing_engine = _wave_engine(config, failing_executor)
    first = failing_engine.run(
        "wave-approved",
        "old-failure",
        approved_plan_hash=original_plan.plan_hash,
    )
    second = failing_engine.rerun(
        "wave-approved",
        "old-failure",
        approved_plan_hash=original_plan.plan_hash,
        reason="confirm reproducible provider capability failure",
    )
    assert first.state is second.state is RunState.FAILED

    original = original_plan.variants[0]
    corrected_graph = thaw_json(original.materialized_graph)
    corrected_graph["1"]["inputs"]["ckpt_name"] = "corrected.safetensors"
    correction = {
        "schema_version": 1,
        "failure_signature": signature,
        "failed_variant_id": original.variant_id,
        "failed_exact_run_hash": original.exact_run_hash,
        "failed_capability_hash": original.capability_hash,
        "candidate_variant_id": "corrected",
        "changed_dimensions": [DimensionName.CHECKPOINT.value],
    }
    parameters = thaw_json(original.execution_parameters)
    parameters["correction"] = correction
    corrected_draft = PlannedVariant.model_validate(
        {
            **original.model_dump(mode="python"),
            "variant_id": "corrected",
            "materialized_graph": corrected_graph,
            "capability_hash": graph_capability_hash(corrected_graph),
            "execution_parameters": parameters,
            "exact_run_hash": "0" * 64,
        }
    )
    corrected = PlannedVariant.model_validate(
        {
            **corrected_draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(corrected_draft),
        }
    )
    revised_plan = _persist_plan_revision(
        config,
        original_plan,
        (corrected,),
        candidate_scores={"corrected": (1, 1)},
    )
    success_executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://corrected.png",
                media_type="image/png",
            ),
        )
    )
    revised_engine = _wave_engine(config, success_executor)

    wave = revised_engine.run_wave(
        "wave-approved",
        approved_plan_hash=revised_plan.plan_hash,
    )

    assert wave.selected_variant_ids == ("corrected",)
    assert wave.objective_evidence["failure_signatures"][signature] == 2
    assert [item.variant_id for item, _fixtures in success_executor.submissions] == [
        "corrected"
    ]


def test_repeated_wave_failure_requires_a_new_plan_hashed_hypothesis(
    tmp_path: Path,
) -> None:
    same_capability = "d" * 64
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("first", same_capability, PlanningConfidence.OBSERVED, 3, 3),
            ("second", same_capability, PlanningConfidence.OBSERVED, 2, 2),
            ("third", same_capability, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    executor = FakeExecutor(
        validation=ValidationReport(
            errors=(
                ValidationIssue(
                    code="missing-node",
                    message="TextEncodeQwen is unavailable",
                ),
            )
        )
    )
    engine = _wave_engine(config, executor)

    first = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )
    second = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )
    saturated = engine.run_wave(
        "wave-approved",
        approved_plan_hash=plan.plan_hash,
    )

    assert first.selected_variant_ids == ("first",)
    assert second.selected_variant_ids == ("second",)
    assert saturated.selected_variant_ids == ()
    assert saturated.skips["third"] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )
    assert sum(second.objective_evidence["failure_signatures"].values()) == 2
    assert executor.submissions == []


def test_exact_repeat_is_skipped_without_explicit_rerun_request(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("repeat", "e" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    engine = _wave_engine(config, executor)
    original = engine.run(
        "wave-approved",
        "repeat",
        approved_plan_hash=plan.plan_hash,
    )
    submitted = len(executor.submissions)

    with pytest.raises(ValueError, match="already recorded"):
        engine.run(
            "wave-approved",
            "repeat",
            approved_plan_hash=plan.plan_hash,
        )

    assert original.state is RunState.SUCCESS
    assert len(executor.submissions) == submitted


def test_exact_rerun_requires_reason_and_persists_original_run_links(
    tmp_path: Path,
) -> None:
    config, plan = _persist_wave_plan(
        tmp_path,
        variants=(
            ("repeat", "e" * 64, PlanningConfidence.OBSERVED, 1, 1),
        ),
    )
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="9",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    engine = _wave_engine(config, executor)
    original = engine.run(
        "wave-approved",
        "repeat",
        approved_plan_hash=plan.plan_hash,
    )
    submitted = len(executor.submissions)

    with pytest.raises(RerunReasonRequired):
        engine.rerun(
            "wave-approved",
            "repeat",
            approved_plan_hash=plan.plan_hash,
            reason="  ",
        )

    rerun = engine.rerun(
        "wave-approved",
        "repeat",
        approved_plan_hash=plan.plan_hash,
        reason="measure output variance",
    )

    assert len(executor.submissions) == submitted + 1
    assert rerun.run_id != original.run_id
    assert rerun.exact_run_hash == original.exact_run_hash
    assert rerun.rerun_reason == "measure output variance"
    assert rerun.original_exact_run_hash == original.exact_run_hash
    assert rerun.original_run_ids == (original.run_id,)
    persisted = json.loads(
        (
            config.experiments_root
            / "wave-approved"
            / "runs"
            / f"{rerun.run_id}.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["original_run_ids"] == [original.run_id]
