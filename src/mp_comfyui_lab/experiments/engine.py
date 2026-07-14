from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import uuid

from pydantic import ValidationError
import requests

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.executors.base import (
    AmbiguousSubmissionError,
    ExecutorError,
    ProviderTaskState,
    TransientExecutorError,
    TransientOperationError,
)
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import SafetyError
from mp_comfyui_lab.security import safe_filename
from mp_comfyui_lab.security import sanitize_for_persistence
from mp_comfyui_lab.security import sanitize_untrusted_diagnostic

from .models import (
    ExperimentPlan,
    MaterializedVariant,
    OutputRecord,
    PlannedVariant,
    ResolvedFixture,
    ResolvedFixtures,
    RunRecord,
    RunState,
)
from .planning import recompute_exact_run_hash
from .store import ExperimentStore


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RETRY_DELAYS = (1, 2, 4)
_POLL_INTERVAL_SECONDS = 1
_OUTPUT_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "application/zip": "zip",
}


class ApprovalError(RuntimeError):
    """Raised before execution when exact durable approval cannot be proven."""


class FixtureIntegrityError(SafetyError):
    """Raised before execution when fixture bytes no longer match approval."""


class _RetryExhausted(RuntimeError):
    def __init__(self, record: RunRecord, operation: str) -> None:
        super().__init__(operation)
        self.record = record
        self.operation = operation


class ExperimentEngine:
    def __init__(
        self,
        *,
        config: LabConfig,
        executor_factory: Callable[[], Any],
        downloader: Callable[[str, Path], Any],
        sleep: Callable[[float], None],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._store = ExperimentStore(config.experiments_root)
        self._executor_factory = executor_factory
        self._downloader = downloader
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(UTC))

    def run(
        self,
        experiment_id: str,
        variant_id: str,
        *,
        approved_plan_hash: str,
    ) -> RunRecord:
        plan, variant = self._load_approved_plan(
            experiment_id,
            variant_id,
            approved_plan_hash,
        )
        materialized = MaterializedVariant(
            variant_id=variant.variant_id,
            source_workflow_id=variant.source_workflow_id,
            graph=thaw_json(variant.materialized_graph),
            exact_run_hash=variant.exact_run_hash,
            capability_hash=variant.capability_hash,
            instance_type=thaw_json(variant.execution_parameters).get(
                "instance_type"
            ),
            expected_output_count=variant.expected_output_count,
            expected_output_node_ids=variant.output_node_ids,
            expected_media_types=variant.expected_media_types,
            warnings=variant.warnings,
        )
        created_at = self._now()
        run_id = self._new_run_id(created_at)
        record = RunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            variant_id=variant.variant_id,
            state=RunState.PLANNED,
            plan_hash=plan.plan_hash,
            exact_run_hash=variant.exact_run_hash,
            expected_output_count=variant.expected_output_count,
            expected_output_node_ids=variant.output_node_ids,
            expected_media_types=variant.expected_media_types,
            created_at=created_at,
            updated_at=created_at,
        )
        with self._store.run_lock(experiment_id, run_id):
            fixtures = self._snapshot_fixtures(
                experiment_id,
                run_id,
                variant,
            )
            self._store.write_json_new(
                experiment_id,
                self._run_path(run_id),
                record.model_dump(mode="json"),
            )
            executor = self._executor_factory()
            try:
                validation = executor.validate_variant(materialized, fixtures)
            except Exception as error:
                failed_at = self._now()
                record = record.transition(
                    RunState.FAILED,
                    error=sanitize_for_persistence(
                        {
                            "kind": "validation_error",
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                    completed_at=failed_at,
                    updated_at=failed_at,
                )
                return self._persist(record)
            if not validation.is_valid:
                failed_at = self._now()
                record = record.transition(
                    RunState.FAILED,
                    error=sanitize_for_persistence(
                        {
                            "kind": "validation_failed",
                            "errors": [
                                item.model_dump(mode="json")
                                for item in validation.errors
                            ],
                        }
                    ),
                    completed_at=failed_at,
                    updated_at=failed_at,
                )
                return self._persist(record)
            intent_at = self._now()
            intent_values = record.model_dump(mode="python")
            intent_values["submission_intent_at"] = intent_at
            intent_values["updated_at"] = intent_at
            record = RunRecord.model_validate(intent_values)
            self._persist(record)
            try:
                receipt = executor.submit(materialized, fixtures)
            except AmbiguousSubmissionError as error:
                incomplete_at = self._now()
                record = record.transition(
                    RunState.INCOMPLETE,
                    error=sanitize_for_persistence(
                        {
                            "kind": "ambiguous_submission",
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                    updated_at=incomplete_at,
                )
                return self._persist(record)
            except ExecutorError as error:
                failed_at = self._now()
                record = record.transition(
                    RunState.FAILED,
                    error=sanitize_for_persistence(
                        {
                            "kind": "provider_rejection",
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                    completed_at=failed_at,
                    updated_at=failed_at,
                )
                return self._persist(record)
            submitted_at = self._now()
            record = record.transition(
                RunState.SUBMITTED,
                provider_task_id=receipt.provider_task_id,
                submitted_at=submitted_at,
                updated_at=submitted_at,
            )
            self._persist(record)
            return self._poll_current(record, executor)

    def resume(self, experiment_id: str, run_id: str) -> RunRecord:
        record = self._load_run(experiment_id, run_id)
        if record.state in {
            RunState.SUCCESS,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.INCOMPLETE,
        }:
            return record
        if record.state is not RunState.PLANNED and record.provider_task_id is None:
            raise SafetyError("resumable run has no durable provider task ID")
        with self._store.run_lock(experiment_id, run_id):
            record = self._load_run(experiment_id, run_id)
            if record.state in {
                RunState.SUCCESS,
                RunState.FAILED,
                RunState.CANCELLED,
                RunState.INCOMPLETE,
            }:
                return record
            if record.state is RunState.PLANNED:
                ambiguous_at = self._now()
                record = record.transition(
                    RunState.INCOMPLETE,
                    error=sanitize_for_persistence(
                        {
                            "kind": "ambiguous_submission",
                            "diagnostic": (
                                "planned run has no durable provider task ID; "
                                "submission cannot be proven absent"
                            ),
                        }
                    ),
                    updated_at=ambiguous_at,
                )
                return self._persist(record)
            if record.provider_task_id is None:
                raise SafetyError("resumable run has no durable provider task ID")
            return self._poll_current(record, self._executor_factory())

    def reconcile(
        self,
        experiment_id: str,
        run_id: str,
        provider_task_id: str,
    ) -> RunRecord:
        if not isinstance(provider_task_id, str) or not provider_task_id.strip():
            raise SafetyError("reconciliation task ID must be nonblank")
        task_id = provider_task_id.strip()
        with self._store.run_lock(experiment_id, run_id):
            record = self._load_run(experiment_id, run_id)
            error = thaw_json(record.error) if record.error is not None else None
            if (
                record.state is not RunState.INCOMPLETE
                or not isinstance(error, dict)
                or error.get("kind") != "ambiguous_submission"
                or record.provider_task_id is not None
            ):
                raise ValueError(
                    "only an ambiguous INCOMPLETE run can be reconciled"
                )
            reconciled_at = self._now()
            record = record.transition(
                RunState.SUBMITTED,
                provider_task_id=task_id,
                error=None,
                submitted_at=reconciled_at,
                updated_at=reconciled_at,
            )
            self._persist(record)
            return self._poll_current(record, self._executor_factory())

    def status(self, experiment_id: str, run_id: str) -> RunRecord:
        return self._load_run(experiment_id, run_id)

    def _load_approved_plan(
        self,
        experiment_id: str,
        variant_id: str,
        approved_plan_hash: str,
    ) -> tuple[ExperimentPlan, PlannedVariant]:
        if (
            not isinstance(approved_plan_hash, str)
            or _SHA256_PATTERN.fullmatch(approved_plan_hash) is None
        ):
            raise ApprovalError("approved plan hash must be an exact SHA-256 value")
        try:
            plan = ExperimentPlan.model_validate_json(
                self._store.read_text(
                    experiment_id,
                    f"plans/{approved_plan_hash}.json",
                ),
                strict=True,
            )
        except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
            raise ApprovalError("approved plan artifact is missing or invalid") from error
        if plan.plan_hash != approved_plan_hash:
            raise ApprovalError("approved plan hash does not match its artifact filename")
        selected = tuple(
            item for item in plan.variants if item.variant_id == variant_id
        )
        if len(selected) != 1:
            raise ApprovalError("selected variant is not present exactly once")
        selected_variant = selected[0]
        try:
            variant_artifact = PlannedVariant.model_validate_json(
                self._store.read_text(
                    experiment_id,
                    f"variants/{variant_id}.json",
                ),
                strict=True,
            )
        except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
            raise ApprovalError("selected variant artifact is missing or invalid") from error
        if variant_artifact != selected_variant:
            raise ApprovalError("selected variant artifact does not match approved plan")
        if recompute_exact_run_hash(selected_variant) != selected_variant.exact_run_hash:
            raise ApprovalError("selected variant exact run hash does not match")
        return plan, selected_variant

    def _snapshot_fixtures(
        self,
        experiment_id: str,
        run_id: str,
        variant: PlannedVariant,
    ) -> ResolvedFixtures:
        indexes = [item.binding_index for item in variant.fixture_fingerprints]
        if len(indexes) != len(set(indexes)):
            raise FixtureIntegrityError("approved fixture bindings are duplicated")

        resolved: list[ResolvedFixture] = []
        for fingerprint in variant.fixture_fingerprints:
            try:
                source_path = self._config.resolve_fixture(fingerprint.uri)
                descriptor = os.open(
                    source_path,
                    os.O_RDONLY
                    | getattr(os, "O_BINARY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
            except (OSError, SafetyError) as error:
                raise FixtureIntegrityError("approved fixture cannot be opened safely") from error
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise FixtureIntegrityError("approved fixture is not a regular file")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 65_536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    digest.update(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            stable_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            stable_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if stable_before != stable_after:
                raise FixtureIntegrityError("approved fixture changed during validation")
            content = b"".join(chunks)
            if (
                len(content) != fingerprint.size
                or digest.hexdigest() != fingerprint.sha256
            ):
                raise FixtureIntegrityError("approved fixture bytes do not match plan")
            suffix = source_path.suffix.casefold() or ".bin"
            snapshot_path = self._store.write_bytes_new(
                experiment_id,
                Path(
                    "runs",
                    run_id,
                    "fixtures",
                    (
                        f"{fingerprint.binding_index:04d}-"
                        f"{fingerprint.sha256[:16]}{suffix}"
                    ),
                ),
                content,
            )
            if (
                snapshot_path.stat().st_size != fingerprint.size
                or hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                != fingerprint.sha256
            ):
                raise FixtureIntegrityError("approved fixture snapshot is invalid")
            resolved.append(
                ResolvedFixture(
                    uri=fingerprint.uri,
                    path=snapshot_path,
                    sha256=fingerprint.sha256,
                    media_type=fingerprint.media_type,
                    binding_role="reference",
                    binding_index=fingerprint.binding_index,
                )
            )
        return ResolvedFixtures(items=tuple(resolved))

    @staticmethod
    def _new_run_id(created_at: datetime) -> str:
        timestamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"run-{timestamp}-{uuid.uuid4().hex}"

    @staticmethod
    def _run_path(run_id: str) -> str:
        return f"runs/{run_id}.json"

    def _persist(self, record: RunRecord) -> RunRecord:
        self._store.write_json(
            record.experiment_id,
            self._run_path(record.run_id),
            record.model_dump(mode="json"),
        )
        return record

    def _load_run(self, experiment_id: str, run_id: str) -> RunRecord:
        try:
            document = self._store.read_text(
                experiment_id,
                self._run_path(run_id),
            )
            record = RunRecord.model_validate_json(document, strict=True)
        except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
            raise SafetyError("run artifact is missing or invalid") from error
        if record.experiment_id != experiment_id or record.run_id != run_id:
            raise SafetyError("run artifact identity does not match its path")
        return record

    def _poll_current(self, record: RunRecord, executor: Any) -> RunRecord:
        task_id = record.provider_task_id
        if task_id is None:
            raise SafetyError("poll requires a durable provider task ID")
        while True:
            try:
                poll, record = self._retry_idempotent(
                    record,
                    "poll",
                    lambda: executor.poll(task_id),
                )
            except _RetryExhausted as error:
                return self._retry_incomplete(error)
            except ExecutorError as error:
                completed_at = self._now()
                record = record.transition(
                    RunState.INCOMPLETE,
                    error=sanitize_for_persistence(
                        {
                            "kind": "poll_error",
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                    completed_at=completed_at,
                    updated_at=completed_at,
                )
                return self._persist(record)
            if poll.provider_task_id != task_id:
                mismatch_at = self._now()
                record = record.transition(
                    RunState.INCOMPLETE,
                    error=sanitize_for_persistence(
                        {
                            "kind": "provider_task_id_mismatch",
                            "expected_task_id": task_id,
                            "observed_task_id": poll.provider_task_id,
                        }
                    ),
                    completed_at=mismatch_at,
                    updated_at=mismatch_at,
                )
                self._persist(record)
                raise SafetyError("provider poll task ID does not match durable run")
            observed_at = self._now()
            if poll.state in {
                ProviderTaskState.SUBMITTED,
                ProviderTaskState.QUEUED,
            }:
                if record.state is not RunState.SUBMITTED:
                    raise SafetyError("provider state regressed after RUNNING")
                values = record.model_dump(mode="python")
                values["diagnostics"] = (
                    *record.diagnostics,
                    sanitize_for_persistence(
                        {
                            "kind": "provider_state",
                            "state": poll.state.value,
                            "diagnostic": poll.diagnostic,
                        }
                    ),
                )
                values["updated_at"] = observed_at
                record = RunRecord.model_validate(values)
                self._persist(record)
                self._sleep(_POLL_INTERVAL_SECONDS)
                continue
            if poll.state is ProviderTaskState.RUNNING:
                if record.state is RunState.SUBMITTED:
                    record = record.transition(
                        RunState.RUNNING,
                        started_at=observed_at,
                        updated_at=observed_at,
                    )
                    self._persist(record)
                self._sleep(_POLL_INTERVAL_SECONDS)
                continue
            if poll.state is ProviderTaskState.CANCELLED:
                record = record.transition(
                    RunState.CANCELLED,
                    error=sanitize_for_persistence(
                        {
                            "kind": "provider_cancelled",
                            "diagnostic": poll.diagnostic,
                        }
                    ),
                    completed_at=observed_at,
                    updated_at=observed_at,
                )
                return self._persist(record)
            if poll.state is ProviderTaskState.FAILED:
                record = record.transition(
                    RunState.FAILED,
                    error=sanitize_for_persistence(
                        {
                            "kind": "provider_failed",
                            "diagnostic": poll.diagnostic,
                        }
                    ),
                    completed_at=observed_at,
                    updated_at=observed_at,
                )
                return self._persist(record)
            if poll.state is ProviderTaskState.SUCCESS:
                return self._collect_success(record, executor)
            record = record.transition(
                RunState.INCOMPLETE,
                error=sanitize_for_persistence(
                    {
                        "kind": "unknown_provider_state",
                        "diagnostic": poll.diagnostic,
                    }
                ),
                completed_at=observed_at,
                updated_at=observed_at,
            )
            return self._persist(record)

    def _retry_idempotent(
        self,
        record: RunRecord,
        operation: str,
        function: Callable[[], Any],
    ) -> tuple[Any, RunRecord]:
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                return function(), record
            except Exception as error:
                if not isinstance(
                    error,
                    (
                        TimeoutError,
                        ConnectionError,
                        OSError,
                        requests.RequestException,
                        TransientExecutorError,
                        TransientOperationError,
                    ),
                ):
                    raise
                observed_at = self._now()
                values = record.model_dump(mode="python")
                values["diagnostics"] = (
                    *record.diagnostics,
                    sanitize_for_persistence(
                        {
                            "kind": "transient_error",
                            "operation": operation,
                            "attempt": attempt + 1,
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                )
                values["updated_at"] = observed_at
                record = RunRecord.model_validate(values)
                self._persist(record)
                if attempt == len(_RETRY_DELAYS):
                    raise _RetryExhausted(record, operation) from error
                self._sleep(_RETRY_DELAYS[attempt])
        raise AssertionError("retry loop did not terminate")

    def _retry_incomplete(self, error: _RetryExhausted) -> RunRecord:
        completed_at = self._now()
        record = error.record.transition(
            RunState.INCOMPLETE,
            error=sanitize_for_persistence(
                {
                    "kind": "retry_exhausted",
                    "operation": error.operation,
                }
            ),
            completed_at=completed_at,
            updated_at=completed_at,
        )
        return self._persist(record)

    def _collect_success(self, record: RunRecord, executor: Any) -> RunRecord:
        task_id = record.provider_task_id
        if task_id is None:
            raise SafetyError("output collection requires a provider task ID")
        try:
            provider_outputs, record = self._retry_idempotent(
                record,
                "collect_outputs",
                lambda: executor.collect_outputs(task_id),
            )
        except _RetryExhausted as error:
            return self._retry_incomplete(error)
        except ExecutorError as error:
            completed_at = self._now()
            record = record.transition(
                RunState.INCOMPLETE,
                error=sanitize_for_persistence(
                    {
                        "kind": "collection_error",
                        "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                    }
                ),
                completed_at=completed_at,
                updated_at=completed_at,
            )
            return self._persist(record)
        reset_at = self._now()
        reset_values = record.model_dump(mode="python")
        reset_values["outputs"] = ()
        reset_values["actual_output_count"] = 0
        reset_values["updated_at"] = reset_at
        record = RunRecord.model_validate(reset_values)
        self._persist(record)
        for ordinal, output in enumerate(provider_outputs):
            try:
                durable, record = self._retry_idempotent(
                    record,
                    "download_output",
                    lambda output=output, ordinal=ordinal: self._download_output(
                        record,
                        output,
                        ordinal,
                    ),
                )
            except _RetryExhausted as error:
                return self._retry_incomplete(error)
            except SafetyError as error:
                completed_at = self._now()
                record = record.transition(
                    RunState.INCOMPLETE,
                    error=sanitize_for_persistence(
                        {
                            "kind": "malformed_output",
                            "ordinal": ordinal,
                            "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                        }
                    ),
                    completed_at=completed_at,
                    updated_at=completed_at,
                )
                return self._persist(record)
            observed_at = self._now()
            values = record.model_dump(mode="python")
            values["outputs"] = (*record.outputs, durable)
            values["actual_output_count"] = len(values["outputs"])
            values["updated_at"] = observed_at
            record = RunRecord.model_validate(values)
            self._persist(record)

        completed_at = self._now()
        if record.actual_output_count != record.expected_output_count:
            error_kind = (
                "partial_outputs"
                if record.actual_output_count < record.expected_output_count
                else "extra_outputs"
            )
            record = record.transition(
                RunState.INCOMPLETE,
                error=sanitize_for_persistence(
                    {
                        "kind": error_kind,
                        "expected_output_count": record.expected_output_count,
                        "actual_output_count": record.actual_output_count,
                    }
                ),
                completed_at=completed_at,
                updated_at=completed_at,
            )
            return self._persist(record)
        record = record.transition(
            RunState.SUCCESS,
            completed_at=completed_at,
            updated_at=completed_at,
        )
        return self._persist(record)

    def _download_output(
        self,
        record: RunRecord,
        output: Any,
        ordinal: int,
    ) -> OutputRecord:
        if output.node_id is None or output.node_id not in record.expected_output_node_ids:
            raise SafetyError("provider output node does not match approved expectation")
        node_id = safe_filename(output.node_id)
        media_type = output.media_type
        if not isinstance(media_type, str) or media_type not in _OUTPUT_EXTENSIONS:
            raise SafetyError("provider output media type is unsupported")
        if (
            record.expected_media_types
            and media_type not in record.expected_media_types
        ):
            raise SafetyError("provider output media type does not match approval")
        extension = _OUTPUT_EXTENSIONS[media_type]
        relative_path = Path(
            "outputs",
            record.variant_id,
            f"{ordinal:04d}-{node_id}.{extension}",
        )
        with tempfile.TemporaryDirectory(prefix="mp-comfyui-lab-output-") as directory:
            temporary_target = Path(directory) / f"output.{extension}"
            returned = self._downloader(output.url, temporary_target)
            downloaded_path = returned if isinstance(returned, Path) else temporary_target
            try:
                content = downloaded_path.read_bytes()
            except OSError as error:
                raise SafetyError("output downloader did not produce a readable file") from error
        self._store.write_bytes_new(
            record.experiment_id,
            relative_path,
            content,
            reuse_matching=True,
        )
        return OutputRecord(
            ordinal=ordinal,
            provider_order=output.provider_order,
            node_id=node_id,
            media_type=media_type,
            local_relative_path=relative_path.as_posix(),
            content_sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
