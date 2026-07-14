from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any
import uuid
import warnings
import zipfile

from PIL import Image, UnidentifiedImageError
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
from .saturation import (
    ExperimentHistory,
    RerunRequest,
    WaveArtifact,
    next_wave,
    request_rerun,
)
from .store import ExperimentStore


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RETRY_DELAYS = (1, 2, 4)
_POLL_INTERVAL_SECONDS = 1
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
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
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ) -> None:
        self._config = config
        self._store = ExperimentStore(config.experiments_root)
        self._executor_factory = executor_factory
        self._downloader = downloader
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(UTC))
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
        ):
            raise ValueError("max_output_bytes must be a positive integer")
        self._max_output_bytes = max_output_bytes

    def run(
        self,
        experiment_id: str,
        variant_id: str,
        *,
        approved_plan_hash: str,
    ) -> RunRecord:
        self._load_approved_plan(
            experiment_id,
            variant_id,
            approved_plan_hash,
        )
        with self._store.run_lock(experiment_id, "wave-execution"):
            return self._run_locked(
                experiment_id,
                variant_id,
                approved_plan_hash=approved_plan_hash,
            )

    def _run_locked(
        self,
        experiment_id: str,
        variant_id: str,
        *,
        approved_plan_hash: str,
        rerun_request: RerunRequest | None = None,
    ) -> RunRecord:
        plan, variant = self._load_approved_plan(
            experiment_id,
            variant_id,
            approved_plan_hash,
        )
        history = self._load_history(experiment_id, plan)
        original_run_ids = history.original_run_ids(variant.exact_run_hash)
        if rerun_request is None and original_run_ids:
            raise ValueError(
                "exact run fingerprint is already recorded; use --rerun with a reason"
            )
        if rerun_request is not None and (
            rerun_request.variant_id != variant.variant_id
            or rerun_request.original_exact_run_hash != variant.exact_run_hash
            or rerun_request.original_run_ids != original_run_ids
            or not original_run_ids
        ):
            raise ApprovalError("rerun request does not match approved history")
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
            rerun_reason=(
                None if rerun_request is None else rerun_request.reason
            ),
            original_run_ids=(
                ()
                if rerun_request is None
                else rerun_request.original_run_ids
            ),
            original_exact_run_hash=(
                None
                if rerun_request is None
                else rerun_request.original_exact_run_hash
            ),
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

    def rerun(
        self,
        experiment_id: str,
        variant_id: str,
        *,
        approved_plan_hash: str,
        reason: str,
    ) -> RunRecord:
        plan, variant = self._load_approved_plan(
            experiment_id,
            variant_id,
            approved_plan_hash,
        )
        request_rerun(variant, reason=reason)
        with self._store.run_lock(experiment_id, "wave-execution"):
            plan, variant = self._load_approved_plan(
                experiment_id,
                variant_id,
                approved_plan_hash,
            )
            history = self._load_history(experiment_id, plan)
            original_run_ids = history.original_run_ids(variant.exact_run_hash)
            if not original_run_ids:
                raise ValueError("exact rerun requires at least one original run")
            rerun_request = request_rerun(
                variant,
                reason=reason,
                original_run_ids=original_run_ids,
            )
            return self._run_locked(
                experiment_id,
                variant_id,
                approved_plan_hash=approved_plan_hash,
                rerun_request=rerun_request,
            )

    def run_wave(
        self,
        experiment_id: str,
        *,
        approved_plan_hash: str,
    ) -> WaveArtifact:
        self._load_approved_plan_document(experiment_id, approved_plan_hash)
        with self._store.run_lock(experiment_id, "wave-execution"):
            plan = self._load_approved_plan_document(
                experiment_id,
                approved_plan_hash,
            )
            completed_waves = self._load_completed_waves(experiment_id)
            if (
                completed_waves
                and completed_waves[-1].plan_hash == plan.plan_hash
                and completed_waves[-1].saturation_reason is not None
            ):
                return completed_waves[-1]

            history = self._load_history(experiment_id, plan)
            self._load_equivalent_groups(experiment_id, history)
            decision = next_wave(plan, history)
            records = tuple(
                self._run_locked(
                    experiment_id,
                    variant.variant_id,
                    approved_plan_hash=approved_plan_hash,
                )
                for variant in decision.variants
            )
            ordinal = completed_waves[-1].ordinal + 1 if completed_waves else 1
            refreshed_history = self._load_history(experiment_id, plan)
            self._load_equivalent_groups(experiment_id, refreshed_history)
            artifact = WaveArtifact(
                ordinal=ordinal,
                plan_hash=plan.plan_hash,
                considered_variant_ids=decision.considered_variant_ids,
                selected_variant_ids=tuple(
                    item.variant_id for item in decision.variants
                ),
                skips=dict(decision.reasons),
                candidate_evidence={
                    variant_id: dict(evidence)
                    for variant_id, evidence in decision.evidence.items()
                },
                run_ids=tuple(record.run_id for record in records),
                objective_evidence=self._wave_objective_evidence(
                    records,
                    refreshed_history,
                ),
                saturation_reason=decision.saturation_reason,
            )
            self._store.write_json_new(
                experiment_id,
                f"plans/waves/{ordinal}.json",
                artifact.model_dump(mode="json"),
            )
            return artifact

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
        plan = self._load_approved_plan_document(
            experiment_id,
            approved_plan_hash,
        )
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

    def _load_approved_plan_document(
        self,
        experiment_id: str,
        approved_plan_hash: str,
    ) -> ExperimentPlan:
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
        if any(
            variant.hypothesis is not None
            and variant.hypothesis not in plan.capability_coverage
            for variant in plan.variants
        ):
            raise ApprovalError(
                "plan contains a hypothesis outside approved dimensions; "
                "create and approve the revised plan hash"
            )
        return plan

    def _load_completed_waves(
        self,
        experiment_id: str,
    ) -> tuple[WaveArtifact, ...]:
        waves_root = self._store.root(experiment_id) / "plans" / "waves"
        if not waves_root.exists():
            return ()
        try:
            paths = tuple(waves_root.glob("*.json"))
        except OSError as error:
            raise SafetyError("wave artifacts cannot be listed") from error

        by_ordinal: dict[int, WaveArtifact] = {}
        for path in paths:
            if not path.stem.isdigit():
                raise SafetyError("wave artifact filename is not an ordinal")
            ordinal = int(path.stem)
            if ordinal < 1 or path.name != f"{ordinal}.json":
                raise SafetyError("wave artifact filename is not canonical")
            try:
                artifact = WaveArtifact.model_validate_json(
                    self._store.read_text(
                        experiment_id,
                        f"plans/waves/{path.name}",
                    ),
                    strict=True,
                )
            except (FileNotFoundError, OSError, ValueError, ValidationError) as error:
                raise SafetyError("wave artifact is missing or invalid") from error
            if artifact.ordinal != ordinal or ordinal in by_ordinal:
                raise SafetyError("wave artifact ordinal does not match its path")
            by_ordinal[ordinal] = artifact

        ordered = tuple(by_ordinal[index] for index in sorted(by_ordinal))
        if tuple(item.ordinal for item in ordered) != tuple(
            range(1, len(ordered) + 1)
        ):
            raise SafetyError("wave artifact ordinals must be contiguous")
        return ordered

    def _load_history(
        self,
        experiment_id: str,
        plan: ExperimentPlan,
    ) -> ExperimentHistory:
        history = ExperimentHistory()
        variants = {
            (variant.variant_id, variant.exact_run_hash): variant
            for variant in plan.variants
        }
        output_groups: dict[tuple[tuple[object, ...], ...], set[str]] = {}

        for record in self._load_run_records(experiment_id):
            if record.exact_run_hash is not None:
                history.record_exact_run(
                    record.exact_run_hash,
                    run_id=record.run_id,
                )
            variant = variants.get((record.variant_id, record.exact_run_hash))
            if variant is None:
                continue
            if record.state is RunState.SUCCESS:
                history.record_success(variant, run_id=record.run_id)
                output_signature = tuple(
                    (
                        output.ordinal,
                        output.provider_order,
                        output.node_id,
                        output.media_type,
                        output.content_sha256,
                        output.size,
                    )
                    for output in sorted(record.outputs, key=lambda item: item.ordinal)
                )
                if output_signature:
                    output_groups.setdefault(output_signature, set()).add(
                        variant.variant_id
                    )
            elif record.state is RunState.FAILED:
                for signature in self._failure_signatures(record):
                    history.record_failure(
                        variant,
                        signature=signature,
                        run_id=record.run_id,
                    )

        for variant_ids in output_groups.values():
            if len(variant_ids) >= 2:
                history.add_equivalent_group(variant_ids)
        return history

    def _load_run_records(
        self,
        experiment_id: str,
    ) -> tuple[RunRecord, ...]:
        runs_root = self._store.root(experiment_id) / "runs"
        try:
            paths = sorted(runs_root.glob("*.json"), key=lambda item: item.name)
        except OSError as error:
            raise SafetyError("run artifacts cannot be listed") from error
        return tuple(self._load_run(experiment_id, path.stem) for path in paths)

    def _load_equivalent_groups(
        self,
        experiment_id: str,
        history: ExperimentHistory,
    ) -> None:
        for wave in self._load_completed_waves(experiment_id):
            evidence = thaw_json(wave.objective_evidence)
            groups = evidence.get("output_duplicate_groups", [])
            if not isinstance(groups, list):
                raise SafetyError("wave output duplicate groups are invalid")
            for group in groups:
                if (
                    not isinstance(group, list)
                    or len(group) < 2
                    or any(not isinstance(item, str) or not item.strip() for item in group)
                ):
                    raise SafetyError("wave output duplicate group is invalid")
                history.add_equivalent_group(group)

    @staticmethod
    def _failure_signatures(record: RunRecord) -> tuple[str, ...]:
        if record.state is not RunState.FAILED or record.error is None:
            return ()
        error = thaw_json(record.error)
        if not isinstance(error, Mapping):
            return ()
        kind = error.get("kind")
        normalized_kind = (
            re.sub(r"[^a-z0-9]+", "-", kind.casefold()).strip("-")
            if isinstance(kind, str)
            else "failed"
        )
        signatures: set[str] = set()
        issues = error.get("errors")
        if normalized_kind == "validation-failed" and isinstance(issues, list):
            for issue in issues:
                if not isinstance(issue, Mapping):
                    continue
                raw_code = issue.get("code")
                code = (
                    re.sub(
                        r"[^a-z0-9]+",
                        "-",
                        raw_code.casefold(),
                    ).strip("-")
                    if isinstance(raw_code, str)
                    else "validation"
                )
                raw_message = issue.get("message")
                message = (
                    " ".join(raw_message.split())[:512]
                    if isinstance(raw_message, str)
                    else ""
                )
                if code.startswith("missing-node") and message:
                    requirement = re.search(
                        r"\b[A-Za-z][A-Za-z0-9_.-]{1,127}\b",
                        message,
                    )
                    signatures.add(
                        f"missing-node:{requirement.group(0) if requirement else message}"
                    )
                else:
                    signatures.add(f"{code}:{message}" if message else code)
        if not signatures:
            diagnostic = error.get("diagnostic")
            message = (
                " ".join(diagnostic.split())[:512]
                if isinstance(diagnostic, str)
                else ""
            )
            signatures.add(
                f"{normalized_kind}:{message}" if message else normalized_kind
            )
        return tuple(sorted(signatures))

    @staticmethod
    def _wave_objective_evidence(
        records: tuple[RunRecord, ...],
        history: ExperimentHistory,
    ) -> dict[str, object]:
        ordered_records = sorted(records, key=lambda item: item.variant_id)
        duplicate_groups = sorted(
            (sorted(group) for group in history.equivalent_groups),
            key=lambda group: tuple(group),
        )
        return {
            "run_states": {
                record.variant_id: record.state.value for record in ordered_records
            },
            "expected_output_counts": {
                record.variant_id: record.expected_output_count
                for record in ordered_records
            },
            "actual_output_counts": {
                record.variant_id: record.actual_output_count
                for record in ordered_records
            },
            "output_complete": {
                record.variant_id: (
                    record.state is RunState.SUCCESS
                    and record.actual_output_count == record.expected_output_count
                )
                for record in ordered_records
            },
            "output_duplicate_groups": duplicate_groups,
            "failure_signatures": dict(sorted(history.failure_signatures.items())),
        }

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
                if not self._is_transient(error):
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

    @staticmethod
    def _is_transient(error: BaseException) -> bool:
        return isinstance(
            error,
            (
                TimeoutError,
                ConnectionError,
                OSError,
                requests.RequestException,
                TransientExecutorError,
                TransientOperationError,
            ),
        )

    def _collect_output_batch(
        self,
        record: RunRecord,
        executor: Any,
        task_id: str,
    ) -> tuple[tuple[Any, ...], RunRecord]:
        latest: tuple[Any, ...] = ()
        observed_batch = False
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                latest = tuple(executor.collect_outputs(task_id))
            except Exception as error:
                if not self._is_transient(error):
                    raise
                diagnostic = {
                    "kind": "transient_error",
                    "operation": "collect_outputs",
                    "attempt": attempt + 1,
                    "diagnostic": sanitize_untrusted_diagnostic(str(error)),
                }
                exhausted = attempt == len(_RETRY_DELAYS)
            else:
                observed_batch = True
                if len(latest) >= record.expected_output_count:
                    return latest, record
                diagnostic = {
                    "kind": "incomplete_output_batch",
                    "operation": "collect_outputs",
                    "attempt": attempt + 1,
                    "observed_output_count": len(latest),
                    "expected_output_count": record.expected_output_count,
                }
                exhausted = attempt == len(_RETRY_DELAYS)

            observed_at = self._now()
            values = record.model_dump(mode="python")
            values["diagnostics"] = (
                *record.diagnostics,
                sanitize_for_persistence(diagnostic),
            )
            values["updated_at"] = observed_at
            record = RunRecord.model_validate(values)
            self._persist(record)
            if exhausted:
                if diagnostic["kind"] == "transient_error" and not observed_batch:
                    raise _RetryExhausted(record, "collect_outputs")
                return latest, record
            self._sleep(_RETRY_DELAYS[attempt])
        raise AssertionError("collection retry loop did not terminate")

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
            provider_outputs, record = self._collect_output_batch(
                record,
                executor,
                task_id,
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
        try:
            provider_outputs = self._ordered_provider_outputs(
                record,
                provider_outputs,
            )
        except SafetyError as error:
            completed_at = self._now()
            record = record.transition(
                RunState.INCOMPLETE,
                error=sanitize_for_persistence(
                    {
                        "kind": "malformed_output",
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
        allow_pair_overflow = (
            len(provider_outputs) > record.expected_output_count
        )
        for ordinal, output in enumerate(provider_outputs):
            try:
                durable, record = self._retry_idempotent(
                    record,
                    "download_output",
                    lambda output=output, ordinal=ordinal: self._download_output(
                        record,
                        output,
                        ordinal,
                        allow_pair_overflow=allow_pair_overflow,
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
        expected_pairs = Counter(
            {
                pair: record.expected_output_count
                // len(record.expected_output_node_ids)
                for pair in zip(
                    record.expected_output_node_ids,
                    record.expected_media_types,
                    strict=True,
                )
            }
        )
        actual_pairs = Counter(
            (item.node_id, item.media_type) for item in record.outputs
        )
        if actual_pairs != expected_pairs:
            record = record.transition(
                RunState.INCOMPLETE,
                error=sanitize_for_persistence(
                    {"kind": "malformed_output", "diagnostic": (
                        "provider output node/media cardinality does not match approval"
                    )}
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

    @staticmethod
    def _ordered_provider_outputs(
        record: RunRecord,
        provider_outputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        ordered = tuple(
            sorted(provider_outputs, key=lambda item: item.provider_order)
        )
        orders = [item.provider_order for item in ordered]
        if orders != list(range(len(ordered))):
            raise SafetyError(
                "provider output order must be unique and contiguous from zero"
            )
        if not record.expected_output_node_ids:
            raise SafetyError("approved output node/media pairs are missing")
        return ordered

    def _download_output(
        self,
        record: RunRecord,
        output: Any,
        ordinal: int,
        *,
        allow_pair_overflow: bool,
    ) -> OutputRecord:
        if output.node_id is None:
            raise SafetyError("provider output node does not match approved expectation")
        node_id = safe_filename(output.node_id)
        media_type = output.media_type
        if not isinstance(media_type, str) or media_type not in _OUTPUT_EXTENSIONS:
            raise SafetyError("provider output media type is unsupported")
        if (node_id, media_type) not in set(
            zip(
                record.expected_output_node_ids,
                record.expected_media_types,
                strict=True,
            )
        ):
            raise SafetyError(
                "provider output node/media pair does not match approval"
            )
        if not allow_pair_overflow:
            multiplicity = (
                record.expected_output_count
                // len(record.expected_output_node_ids)
            )
            existing_pair_count = sum(
                item.node_id == node_id and item.media_type == media_type
                for item in record.outputs
            )
            if existing_pair_count >= multiplicity:
                raise SafetyError(
                    "provider output node/media pair exceeds approved cardinality"
                )
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
            self._verify_downloaded_media(downloaded_path, media_type)
            _, content_sha256, size = self._store.publish_file_new(
                record.experiment_id,
                relative_path,
                downloaded_path,
                max_bytes=self._max_output_bytes,
                reuse_matching=True,
            )
        return OutputRecord(
            ordinal=ordinal,
            provider_order=output.provider_order,
            node_id=node_id,
            media_type=media_type,
            local_relative_path=relative_path.as_posix(),
            content_sha256=content_sha256,
            size=size,
        )

    def _verify_downloaded_media(self, path: Path, media_type: str) -> None:
        try:
            file_stat = path.lstat()
        except OSError as error:
            raise SafetyError(
                "output downloader did not produce a readable file"
            ) from error
        if not stat.S_ISREG(file_stat.st_mode):
            raise SafetyError("downloaded output is not a regular file")
        if file_stat.st_size == 0:
            raise SafetyError("downloaded output is empty")
        if file_stat.st_size > self._max_output_bytes:
            raise SafetyError("downloaded output exceeds byte limit")

        image_formats = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
            "image/gif": "GIF",
        }
        if media_type in image_formats:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.format != image_formats[media_type]:
                            raise SafetyError(
                                "downloaded image format does not match media type"
                            )
                        image.verify()
            except SafetyError:
                raise
            except (
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
                UnidentifiedImageError,
                OSError,
                SyntaxError,
                ValueError,
            ) as error:
                raise SafetyError("downloaded image failed verification") from error
            return

        try:
            with path.open("rb") as source:
                prefix = source.read(65_536)
        except OSError as error:
            raise SafetyError("downloaded output cannot be inspected") from error
        if media_type == "video/mp4":
            box_size = int.from_bytes(prefix[:4], "big") if len(prefix) >= 4 else 0
            valid = (
                len(prefix) >= 16
                and prefix[4:8] == b"ftyp"
                and 16 <= box_size <= file_stat.st_size
                and prefix[8:12] != b"\x00\x00\x00\x00"
            )
        elif media_type == "video/webm":
            valid = prefix.startswith(b"\x1aE\xdf\xa3") and b"webm" in prefix
        elif media_type == "audio/mpeg":
            valid = self._valid_mp3(prefix, file_stat.st_size)
        elif media_type == "audio/wav":
            valid = (
                len(prefix) >= 12
                and prefix[:4] == b"RIFF"
                and prefix[8:12] == b"WAVE"
                and int.from_bytes(prefix[4:8], "little") + 8
                <= file_stat.st_size
            )
        elif media_type == "audio/ogg":
            segment_count = prefix[26] if len(prefix) >= 27 else 0
            header_size = 27 + segment_count
            payload_size = (
                sum(prefix[27:header_size])
                if len(prefix) >= header_size
                else file_stat.st_size + 1
            )
            valid = (
                len(prefix) >= header_size
                and prefix[:4] == b"OggS"
                and prefix[4] == 0
                and header_size + payload_size <= file_stat.st_size
            )
        elif media_type == "application/zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    valid = archive.testzip() is None
            except (OSError, zipfile.BadZipFile, RuntimeError):
                valid = False
        else:
            valid = False
        if not valid:
            raise SafetyError(
                "downloaded output signature does not match media type"
            )

    @staticmethod
    def _valid_mp3(prefix: bytes, size: int) -> bool:
        if prefix.startswith(b"ID3") and len(prefix) >= 10:
            encoded_size = prefix[6:10]
            if any(value >= 128 for value in encoded_size):
                return False
            tag_size = (
                (encoded_size[0] << 21)
                | (encoded_size[1] << 14)
                | (encoded_size[2] << 7)
                | encoded_size[3]
            )
            return 10 + tag_size <= size
        return (
            len(prefix) >= 4
            and prefix[0] == 0xFF
            and prefix[1] & 0xE0 == 0xE0
            and prefix[1] & 0x18 != 0x08
            and prefix[2] & 0xF0 != 0xF0
        )
