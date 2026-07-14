from __future__ import annotations

from collections.abc import Mapping
import hashlib
from datetime import UTC
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    AfterValidator,
    AwareDatetime,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from mp_comfyui_lab.discovery.urls import UnsupportedSourceError, parse_runninghub_url
from mp_comfyui_lab.graph.models import FrozenJsonValue, NormalizedGraph
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash
from mp_comfyui_lab.graph.normalize import GraphNormalizationError, normalize_graph
from mp_comfyui_lab.json_types import (
    FrozenJsonObject,
    SanitizedJsonObject,
    thaw_json,
)
from mp_comfyui_lab.security import (
    SafetyError,
    safe_filename,
    sanitize_untrusted_diagnostic,
)


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
UtcDatetime = Annotated[
    AwareDatetime,
    AfterValidator(lambda value: value.astimezone(UTC)),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class RunState(StrEnum):
    PLANNED = "PLANNED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"
    CANCELLED = "CANCELLED"


class RecoveryDisposition(StrEnum):
    RESUME_PROVIDER_TASK = "resume_provider_task"
    RECONCILE_SUBMISSION = "reconcile_submission"
    TERMINAL = "terminal"


_NORMAL_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PLANNED: frozenset(
        {RunState.SUBMITTED, RunState.FAILED, RunState.INCOMPLETE}
    ),
    RunState.SUBMITTED: frozenset(
        {
            RunState.RUNNING,
            RunState.SUCCESS,
            RunState.FAILED,
            RunState.INCOMPLETE,
            RunState.CANCELLED,
        }
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.SUCCESS,
            RunState.FAILED,
            RunState.INCOMPLETE,
            RunState.CANCELLED,
        }
    ),
}
_AMBIGUOUS_RECONCILIATION_STATES = frozenset(
    {
        RunState.SUBMITTED,
        RunState.RUNNING,
        RunState.SUCCESS,
        RunState.FAILED,
        RunState.CANCELLED,
    }
)
_IMMUTABLE_TRANSITION_FIELDS = (
    "run_id",
    "experiment_id",
    "variant_id",
    "plan_hash",
    "exact_run_hash",
    "expected_output_count",
    "expected_output_node_ids",
    "expected_media_types",
    "created_at",
    "submission_intent_at",
    "rerun_reason",
    "original_run_ids",
    "original_exact_run_hash",
)


def _is_ambiguous_submission(error: Any) -> bool:
    return (
        isinstance(error, Mapping)
        and error.get("kind") == "ambiguous_submission"
    )


def _incomplete_recovery_disposition(
    error: Any,
    *,
    has_provider_task_id: bool,
) -> RecoveryDisposition:
    if _is_ambiguous_submission(error) and not has_provider_task_id:
        return RecoveryDisposition.RECONCILE_SUBMISSION
    if not has_provider_task_id or not isinstance(error, Mapping):
        return RecoveryDisposition.TERMINAL
    kind = error.get("kind")
    if kind in {
        "collection_error",
        "poll_error",
        "provider_task_id_mismatch",
        "unknown_provider_state",
    }:
        return RecoveryDisposition.RESUME_PROVIDER_TASK
    if kind == "retry_exhausted" and error.get("operation") in {
        "collect_outputs",
        "download_output",
        "poll",
    }:
        return RecoveryDisposition.RESUME_PROVIDER_TASK
    return RecoveryDisposition.TERMINAL


def _sanitize_diagnostic_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            (
                sanitize_untrusted_diagnostic(key)
                if isinstance(key, str)
                else key
            ): _sanitize_diagnostic_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_sanitize_diagnostic_json(item) for item in value)
    if isinstance(value, str):
        return sanitize_untrusted_diagnostic(value)
    return value


def _validate_output_expectations(
    expected_output_count: int,
    node_ids: tuple[str, ...],
    media_types: tuple[str, ...],
    *,
    required: bool,
) -> None:
    if not node_ids and not media_types:
        if required:
            raise ValueError("output node/media expectations are required")
        return
    if len(node_ids) != len(media_types):
        raise ValueError("output node IDs and media types must be aligned")
    if not node_ids:
        raise ValueError("output node/media expectations must not be empty")
    if any(not value.strip() for value in (*node_ids, *media_types)):
        raise ValueError("output node/media expectations must not be blank")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("output node IDs must be unique")
    if expected_output_count % len(node_ids) != 0:
        raise ValueError(
            "expected output count must be divisible by output expectation pairs"
        )


class ResolvedFixture(_FrozenModel):
    uri: NonBlankStr
    path: Path
    sha256: Sha256
    media_type: str | None = None
    binding_role: NonBlankStr | None = None
    binding_index: int = Field(default=0, ge=0)

    @field_validator("path")
    @classmethod
    def require_absolute_path(cls, path: Path) -> Path:
        if not path.is_absolute():
            raise ValueError("path must be absolute")
        return path


class ResolvedFixtures(_FrozenModel):
    items: tuple[ResolvedFixture, ...] = ()

    def by_role(self, role: str) -> tuple[ResolvedFixture, ...]:
        return tuple(item for item in self.items if item.binding_role == role)


class MaterializedVariant(_FrozenModel):
    variant_id: NonBlankStr
    source_workflow_id: NonBlankStr
    graph: FrozenJsonObject
    exact_run_hash: Sha256
    capability_hash: Sha256
    instance_type: str | None = None
    expected_output_count: int = Field(gt=0)
    expected_output_node_ids: tuple[str, ...] = ()
    expected_media_types: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator("graph")
    @classmethod
    def validate_graph(cls, graph: FrozenJsonObject) -> FrozenJsonObject:
        if not graph:
            raise ValueError("graph must not be empty")
        if any(not node_id.strip() for node_id in graph):
            raise ValueError("graph node IDs must not be blank")
        return graph

    @model_validator(mode="after")
    def validate_output_expectations(self) -> MaterializedVariant:
        _validate_output_expectations(
            self.expected_output_count,
            self.expected_output_node_ids,
            self.expected_media_types,
            required=False,
        )
        return self


class OutputRecord(_FrozenModel):
    ordinal: int = Field(ge=0)
    provider_order: int = Field(ge=0)
    node_id: NonBlankStr
    media_type: NonBlankStr
    local_relative_path: NonBlankStr
    content_sha256: Sha256
    size: int = Field(ge=0)

    @field_validator("local_relative_path")
    @classmethod
    def require_safe_relative_output_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("local_relative_path must be a safe relative path")
        return path.as_posix()


class RunRecord(_FrozenModel):
    run_id: NonBlankStr
    experiment_id: NonBlankStr
    variant_id: NonBlankStr
    state: RunState
    recovery_disposition: RecoveryDisposition | None = None
    plan_hash: Sha256 | None = None
    exact_run_hash: Sha256 | None = None
    provider_task_id: NonBlankStr | None = None
    expected_output_count: int = Field(gt=0)
    expected_output_node_ids: tuple[str, ...] = ()
    expected_media_types: tuple[str, ...] = ()
    actual_output_count: int = Field(default=0, ge=0)
    outputs: tuple[OutputRecord, ...] = ()
    error: SanitizedJsonObject | None = None
    diagnostics: tuple[SanitizedJsonObject, ...] = ()
    rerun_reason: NonBlankStr | None = None
    original_run_ids: tuple[NonBlankStr, ...] = ()
    original_exact_run_hash: Sha256 | None = None
    created_at: UtcDatetime | None = None
    submission_intent_at: UtcDatetime | None = None
    submitted_at: UtcDatetime | None = None
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None

    @field_validator(
        "expected_output_node_ids",
        "expected_media_types",
        "outputs",
        "original_run_ids",
        mode="before",
    )
    @classmethod
    def run_json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("error", mode="before")
    @classmethod
    def sanitize_error_diagnostics(cls, value: Any) -> Any:
        return None if value is None else _sanitize_diagnostic_json(value)

    @field_validator("diagnostics", mode="before")
    @classmethod
    def sanitize_diagnostic_history(cls, value: Any) -> Any:
        items = tuple(value) if isinstance(value, list) else value
        return tuple(_sanitize_diagnostic_json(item) for item in items)

    @model_validator(mode="after")
    def validate_provider_task_id(self) -> RunRecord:
        if (
            self.state is RunState.INCOMPLETE
            and self.recovery_disposition is None
        ):
            object.__setattr__(
                self,
                "recovery_disposition",
                _incomplete_recovery_disposition(
                    self.error,
                    has_provider_task_id=self.provider_task_id is not None,
                ),
            )
        _validate_output_expectations(
            self.expected_output_count,
            self.expected_output_node_ids,
            self.expected_media_types,
            required=False,
        )
        task_id_required_states = {
            RunState.SUBMITTED,
            RunState.RUNNING,
            RunState.SUCCESS,
            RunState.CANCELLED,
        }
        if self.state is RunState.PLANNED and self.provider_task_id is not None:
            raise ValueError("provider_task_id is not allowed for PLANNED")
        if self.state in task_id_required_states and self.provider_task_id is None:
            raise ValueError(f"provider_task_id is required for {self.state.value}")
        if self.state is not RunState.INCOMPLETE:
            if self.recovery_disposition is not None:
                raise ValueError(
                    "recovery_disposition is only allowed for INCOMPLETE"
                )
        elif self.recovery_disposition is None:
            raise ValueError("INCOMPLETE requires a recovery_disposition")
        elif (
            self.recovery_disposition
            is RecoveryDisposition.RESUME_PROVIDER_TASK
            and self.provider_task_id is None
        ):
            raise ValueError(
                "resumable INCOMPLETE requires a provider_task_id"
            )
        elif (
            self.recovery_disposition
            is RecoveryDisposition.RECONCILE_SUBMISSION
            and (
                self.provider_task_id is not None
                or not _is_ambiguous_submission(self.error)
            )
        ):
            raise ValueError(
                "reconcile disposition requires ambiguous submission "
                "without a provider_task_id"
            )
        if (
            self.state is RunState.SUCCESS
            and self.actual_output_count != self.expected_output_count
        ):
            raise ValueError(
                "actual_output_count must equal expected_output_count for SUCCESS"
            )

        rerun_fields_present = (
            self.rerun_reason is not None,
            bool(self.original_run_ids),
            self.original_exact_run_hash is not None,
        )
        if any(rerun_fields_present) and not all(rerun_fields_present):
            raise ValueError(
                "rerun reason, original run IDs, and original exact hash "
                "must be recorded together"
            )
        if self.rerun_reason is not None:
            if len(set(self.original_run_ids)) != len(self.original_run_ids):
                raise ValueError("original run IDs must be unique")
            if self.run_id in self.original_run_ids:
                raise ValueError("a rerun cannot link itself as an original run")
            if self.exact_run_hash is None:
                raise ValueError("a rerun requires an exact run hash")
            if self.original_exact_run_hash != self.exact_run_hash:
                raise ValueError(
                    "rerun original exact hash must equal the repeated fingerprint"
                )

        timestamps = [
            timestamp
            for timestamp in (
                self.created_at,
                self.submission_intent_at,
                self.submitted_at,
                self.started_at,
                self.completed_at,
                self.updated_at,
            )
            if timestamp is not None
        ]
        if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("run timestamps must be chronologically nondecreasing")
        return self

    def transition(self, new_state: RunState, **updates: Any) -> Self:
        """Revalidate a state change; never use model_copy(update=...) for transitions."""
        if not isinstance(new_state, RunState):
            raise ValueError("new state must be a RunState")
        if new_state is self.state:
            raise ValueError("same state transition is not allowed")

        for field_name in _IMMUTABLE_TRANSITION_FIELDS:
            if (
                field_name in updates
                and updates[field_name] != getattr(self, field_name)
            ):
                raise ValueError(f"transition cannot change {field_name}")

        if (
            self.provider_task_id is not None
            and "provider_task_id" in updates
            and updates["provider_task_id"] != self.provider_task_id
        ):
            raise ValueError("transition cannot replace provider_task_id")

        if self.state is RunState.INCOMPLETE:
            updates.setdefault("recovery_disposition", None)
            if (
                self.recovery_disposition
                is RecoveryDisposition.RECONCILE_SUBMISSION
            ):
                self._validate_ambiguous_reconciliation(new_state, updates)
            elif (
                self.recovery_disposition
                is RecoveryDisposition.RESUME_PROVIDER_TASK
            ):
                self._validate_provider_resume(new_state, updates)
            else:
                self._validate_ambiguous_reconciliation(new_state, updates)
        else:
            allowed_states = _NORMAL_RUN_TRANSITIONS.get(self.state, frozenset())
            if new_state not in allowed_states:
                raise ValueError(
                    f"transition from {self.state.value} to {new_state.value} "
                    "is not allowed"
                )
            if self.state is RunState.PLANNED:
                error = updates.get("error", self.error)
                provider_task_id = updates.get(
                    "provider_task_id",
                    self.provider_task_id,
                )
                if new_state is RunState.INCOMPLETE:
                    if not _is_ambiguous_submission(error):
                        raise ValueError(
                            "PLANNED to INCOMPLETE transition requires "
                            "error.kind == 'ambiguous_submission'"
                        )
                    if provider_task_id is not None:
                        raise ValueError(
                            "provider_task_id must be absent for ambiguous submission"
                        )
                elif new_state is RunState.FAILED:
                    if provider_task_id is not None:
                        raise ValueError(
                            "provider_task_id must be absent for confirmed rejection"
                        )
                    if _is_ambiguous_submission(error):
                        raise ValueError(
                            "error.kind == 'ambiguous_submission' must transition "
                            "from PLANNED to INCOMPLETE"
                        )

        values = self.model_dump()
        values.update(updates)
        values["state"] = new_state
        return type(self).model_validate(values)

    def _validate_ambiguous_reconciliation(
        self,
        new_state: RunState,
        updates: dict[str, Any],
    ) -> None:
        if new_state not in _AMBIGUOUS_RECONCILIATION_STATES:
            raise ValueError(
                f"transition from INCOMPLETE to {new_state.value} is not allowed"
            )
        if not _is_ambiguous_submission(self.error):
            raise ValueError(
                "INCOMPLETE transition requires "
                "error.kind == 'ambiguous_submission'"
            )
        if self.provider_task_id is not None:
            raise ValueError(
                "provider_task_id must be absent before ambiguous reconciliation"
            )

        provider_task_id = updates.get("provider_task_id")
        if not isinstance(provider_task_id, str) or not provider_task_id.strip():
            raise ValueError(
                "a nonblank provider_task_id must be attached during "
                "ambiguous reconciliation"
            )
        if updates.get("recovery_disposition", self.recovery_disposition) is not None:
            raise ValueError("reconciliation must clear recovery_disposition")

    def _validate_provider_resume(
        self,
        new_state: RunState,
        updates: dict[str, Any],
    ) -> None:
        if new_state not in {RunState.SUBMITTED, RunState.RUNNING}:
            raise ValueError(
                "resumable INCOMPLETE can only return to an active state"
            )
        if self.provider_task_id is None:
            raise ValueError("resumable INCOMPLETE requires a provider_task_id")
        if updates.get("recovery_disposition", self.recovery_disposition) is not None:
            raise ValueError("provider resume must clear recovery_disposition")


# Planning DTOs deliberately use tuples rather than mutable containers.  Plans are
# evidence records, so their input shape is part of their durable contract.
MAX_MANIFEST_BYTES = 1_048_576
MAX_MANIFEST_DEPTH = 48
MAX_MANIFEST_VALUES = 100_000
MAX_MANIFEST_SCALAR_LENGTH = 8_192
MAX_MANIFEST_SOURCES = 128
MAX_FIXTURES = 64
MAX_PROMPT_LINES = 512
MAX_PLAN_VARIANTS = 512
_ARTIFACT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_VAULT_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class DimensionName(StrEnum):
    CHECKPOINT = "checkpoint"
    LORA_STACK = "lora_stack"
    LORA_STRENGTH = "lora_strength"
    PROMPT_GRAMMAR = "prompt_grammar"
    SAMPLER = "sampler"
    SCHEDULER = "scheduler"
    ENCODER = "encoder"
    REFERENCE_STRATEGY = "reference_strategy"
    OUTPUT_DIMENSIONS = "output_dimensions"
    SEED_VARIANCE = "seed_variance"


class DimensionPolicy(StrEnum):
    APPROVED = "approved"
    BASELINE_ONLY = "baseline_only"
    DISABLED = "disabled"


class PlanningConfidence(StrEnum):
    OBSERVED = "observed"
    STRONGLY_INFERRED = "strongly_inferred"
    WEAKLY_INFERRED = "weakly_inferred"
    UNEXPLORED = "unexplored"


class SourceSpec(_FrozenModel):
    queries: tuple[Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_MANIFEST_SCALAR_LENGTH)], ...] = Field(default=(), max_length=MAX_MANIFEST_SOURCES)
    urls: tuple[Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2_048)], ...] = Field(default=(), max_length=MAX_MANIFEST_SOURCES)

    @field_validator("queries", "urls", mode="before")
    @classmethod
    def normalize_arrays(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("sources queries and urls must be explicit arrays")
        return tuple(value)

    @field_validator("urls")
    @classmethod
    def canonicalize_urls(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        canonical: list[str] = []
        for value in values:
            try:
                canonical.append(parse_runninghub_url(value).canonical_url)
            except UnsupportedSourceError as error:
                raise ValueError(str(error)) from error
        return tuple(canonical)

    @model_validator(mode="after")
    def require_one_source(self) -> Self:
        if not self.queries and not self.urls:
            raise ValueError("sources must declare at least one query or URL")
        return self


class ReferenceImageFixture(_FrozenModel):
    uri: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2_048)]
    binding_index: int = Field(ge=0, le=4_095)

    @field_validator("uri")
    @classmethod
    def canonical_logical_uri(cls, value: str) -> str:
        if any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("fixture URI contains control or surrogate text")
        if "\\" in value or "?" in value or "#" in value:
            raise ValueError("fixture URI must not contain a query, fragment, or backslash")
        parsed = urlsplit(value)
        if value.startswith("vault://"):
            alias, slash, relative = value.removeprefix("vault://").partition("/")
            if not slash or not _VAULT_ALIAS.fullmatch(alias):
                raise ValueError("vault fixture URI requires a safe alias and relative path")
            prefix = f"vault://{alias}/"
        else:
            if parsed.scheme or parsed.netloc or value.startswith(("/", "//")) or ":" in value:
                raise ValueError("fixture URI must be a safe relative logical path")
            relative = value
            prefix = ""
        parts = relative.split("/")
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("fixture URI contains an unsafe path component")
        for part in parts:
            try:
                safe_filename(part)
            except SafetyError as error:
                raise ValueError("fixture URI contains an unsafe path component") from error
        return prefix + "/".join(parts)


class FixtureSpec(_FrozenModel):
    reference_images: tuple[ReferenceImageFixture, ...] = Field(default=(), max_length=MAX_FIXTURES)
    prompt_lines: tuple[str, ...] = Field(min_length=1, max_length=MAX_PROMPT_LINES)

    @field_validator("reference_images", "prompt_lines", mode="before")
    @classmethod
    def explicit_arrays(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("fixtures must use explicit ordered arrays")
        return tuple(value)

    @field_validator("prompt_lines")
    @classmethod
    def validate_prompt_lines(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError("prompt_lines must contain nonempty strings")
            if len(value) > MAX_MANIFEST_SCALAR_LENGTH:
                raise ValueError("prompt line exceeds maximum length")
        return values

    @model_validator(mode="after")
    def unique_reference_bindings(self) -> Self:
        indexes = [item.binding_index for item in self.reference_images]
        if len(indexes) != len(set(indexes)):
            raise ValueError("reference fixture binding indexes must be unique")
        return self


class ExpectationSpec(_FrozenModel):
    output_count: int | Literal["auto"]
    media_types: tuple[Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)], ...] = Field(default=(), max_length=32)

    @field_validator("output_count")
    @classmethod
    def positive_output_count(cls, value: int | str) -> int | str:
        if value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("output_count must be a positive strict integer or 'auto'")
        return value

    @field_validator("media_types", mode="before")
    @classmethod
    def media_type_array(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("media_types must be an explicit array")
        return tuple(value)


class DimensionApproval(_FrozenModel):
    name: DimensionName
    policy: DimensionPolicy

    @field_validator("name", mode="before")
    @classmethod
    def parse_name(cls, value: Any) -> DimensionName:
        return DimensionName(value)

    @field_validator("policy", mode="before")
    @classmethod
    def parse_policy(cls, value: Any) -> DimensionPolicy:
        return DimensionPolicy(value)


class ExperimentBrief(_FrozenModel):
    goal: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=MAX_MANIFEST_SCALAR_LENGTH)]
    sources: SourceSpec
    fixtures: FixtureSpec
    expectations: ExpectationSpec
    dimensions: tuple[DimensionApproval, ...] = Field(min_length=1, max_length=len(DimensionName))

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimension_map(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("dimensions must be a known capability map")
        return tuple({"name": name, "policy": policy} for name, policy in value.items())

    @field_validator("dimensions")
    @classmethod
    def canonical_dimensions(
        cls,
        values: tuple[DimensionApproval, ...],
    ) -> tuple[DimensionApproval, ...]:
        if len({item.name for item in values}) != len(values):
            raise ValueError("dimensions must not repeat capability names")
        return tuple(sorted(values, key=lambda item: item.name.value))

    def policy_for(self, name: DimensionName) -> DimensionPolicy:
        for item in self.dimensions:
            if item.name is name:
                return item.policy
        return DimensionPolicy.DISABLED


class CandidateClaim(_FrozenModel):
    dimension: DimensionName
    value: FrozenJsonValue
    binding_index: int = Field(default=0, ge=0, le=4_095)

    @field_validator("dimension", mode="before")
    @classmethod
    def parse_dimension(cls, value: Any) -> DimensionName:
        return DimensionName(value)


class Candidate(_FrozenModel):
    candidate_id: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
    hypothesis: DimensionName
    capability_signature: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=8_192)]
    confidence: PlanningConfidence
    evidence_ids: tuple[Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)], ...] = Field(min_length=1, max_length=128)
    semantic_patch: Any | None = None
    claims: tuple[CandidateClaim, ...] = Field(default=(), max_length=32)
    engagement: int = Field(default=0, ge=0)
    provenance_rank: int = Field(default=0, ge=0)

    @field_validator("candidate_id")
    @classmethod
    def safe_candidate_id(cls, value: str) -> str:
        try:
            safe_filename(value)
        except SafetyError as error:
            raise ValueError("candidate_id must be a safe non-baseline artifact ID") from error
        if value == "baseline" or not _ARTIFACT_ID.fullmatch(value):
            raise ValueError("candidate_id must be a safe non-baseline artifact ID")
        return value

    @field_validator("hypothesis", mode="before")
    @classmethod
    def parse_hypothesis(cls, value: Any) -> DimensionName:
        return DimensionName(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def parse_confidence(cls, value: Any) -> PlanningConfidence:
        return PlanningConfidence(value)

    @field_validator("semantic_patch", mode="before")
    @classmethod
    def require_typed_patch(cls, value: Any) -> Any:
        if value is None:
            return value
        if (
            type(value).__name__ == "SemanticPatch"
            and type(value).__module__ == "mp_comfyui_lab.mutations.models"
        ):
            return value
        raise ValueError("semantic_patch must be a typed SemanticPatch")

    @field_validator("evidence_ids", "claims", mode="before")
    @classmethod
    def candidate_arrays(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("candidate evidence and claims must be arrays")
        return tuple(value)

    @field_validator("evidence_ids")
    @classmethod
    def canonical_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(values)))

    @model_validator(mode="after")
    def require_patch_or_claim(self) -> Self:
        if self.semantic_patch is None and not self.claims:
            raise ValueError("candidate requires an explicit semantic_patch or claims")
        return self


class PlanningSource(_FrozenModel):
    workflow_id: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
    canonical_url: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2_048)]
    raw_graph: FrozenJsonObject
    normalized_graph: NormalizedGraph
    source_hash: Sha256
    capability_hash: Sha256
    evidence_ids: tuple[Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)], ...] = Field(min_length=1, max_length=128)
    engagement: int = Field(default=0, ge=0)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def source_evidence_array(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("source evidence_ids must be an array")
        return tuple(value)

    @field_validator("canonical_url")
    @classmethod
    def canonical_source_url(cls, value: str) -> str:
        try:
            return parse_runninghub_url(value).canonical_url
        except UnsupportedSourceError as error:
            raise ValueError(str(error)) from error

    @model_validator(mode="after")
    def source_id_matches_url(self) -> Self:
        if parse_runninghub_url(self.canonical_url).workflow_id != self.workflow_id:
            raise ValueError("workflow_id does not match canonical_url")
        try:
            normalized = normalize_graph(self.raw_graph)
            actual_source_hash = source_graph_hash(self.raw_graph)
            actual_capability_hash = capability_hash(self.raw_graph)
        except (GraphNormalizationError, ValueError) as error:
            raise ValueError(f"raw_graph is not a valid normalized graph: {error}") from error
        if normalized != self.normalized_graph:
            raise ValueError("normalized_graph must exactly match raw_graph normalization")
        if actual_source_hash != self.source_hash:
            raise ValueError("source_hash must match raw_graph")
        if actual_capability_hash != self.capability_hash:
            raise ValueError("capability_hash must match raw_graph")
        return self


class SourceBundle(_FrozenModel):
    sources: tuple[PlanningSource, ...] = Field(min_length=1, max_length=MAX_MANIFEST_SOURCES)
    candidates: tuple[Candidate, ...] = Field(default=(), max_length=MAX_PLAN_VARIANTS)
    baseline_workflow_id: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)] | None = None
    evidence_graph: Any
    compiler_warnings: tuple[str, ...] = Field(default=(), max_length=256)

    @field_validator("compiler_warnings", mode="before")
    @classmethod
    def compiler_warning_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("compiler_warnings")
    @classmethod
    def canonical_compiler_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    @field_validator("sources", "candidates", mode="before")
    @classmethod
    def bundle_arrays(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("source bundle values must be arrays")
        return tuple(value)

    @field_validator("sources")
    @classmethod
    def canonical_sources(
        cls,
        values: tuple[PlanningSource, ...],
    ) -> tuple[PlanningSource, ...]:
        keys = [(item.workflow_id, item.source_hash) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("source bundle contains duplicate source identity")
        return tuple(sorted(values, key=lambda item: (item.workflow_id, item.source_hash)))

    @field_validator("candidates")
    @classmethod
    def canonical_candidates(
        cls,
        values: tuple[Candidate, ...],
    ) -> tuple[Candidate, ...]:
        ids = [item.candidate_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("source bundle candidate IDs must be unique")
        return tuple(
            sorted(
                values,
                key=lambda item: json.dumps(
                    item.model_dump(mode="json"),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        )

    @model_validator(mode="after")
    def validate_evidence_and_baseline(self) -> Self:
        workflow_ids = {item.workflow_id for item in self.sources}
        graph = self.evidence_graph
        if (
            type(graph).__name__ != "EvidenceGraph"
            or type(graph).__module__ != "mp_comfyui_lab.evidence.models"
        ):
            raise ValueError("source bundle requires a strict EvidenceGraph")
        graph_sources = {item.source_id for item in graph.sources}
        graph_observations = {item.source_id for item in graph.observations}
        workflow_ids_with_evidence = {item.workflow_id for item in self.sources}
        if not workflow_ids_with_evidence.issubset(graph_sources & graph_observations):
            raise ValueError("every planning source requires an evidence source and observation")
        evidence_ids = graph_sources
        if self.baseline_workflow_id is not None and self.baseline_workflow_id not in workflow_ids:
            raise ValueError("baseline_workflow_id must identify a bundle source")
        missing = sorted(
            evidence_id
            for candidate in self.candidates
            for evidence_id in candidate.evidence_ids
            if evidence_id not in evidence_ids
        )
        if missing:
            raise ValueError(f"candidates reference missing evidence IDs: {missing}")
        return self


class FixtureFingerprint(_FrozenModel):
    uri: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2_048)]
    binding_index: int = Field(ge=0, le=4_095)
    size: int = Field(ge=0, le=64 * 1024 * 1024)
    media_type: str | None = None
    sha256: Sha256


class PlannedVariant(_FrozenModel):
    variant_id: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
    source_workflow_id: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
    source_hash: Sha256
    semantic_patch: FrozenJsonObject | None = None
    diff: FrozenJsonObject | None = None
    materialized_graph: FrozenJsonObject
    hypothesis: DimensionName | None = None
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=128)
    confidence: PlanningConfidence | None = None
    warnings: tuple[str, ...] = Field(default=(), max_length=128)
    exact_run_hash: Sha256
    capability_hash: Sha256
    expected_output_count: int = Field(gt=0)
    output_node_ids: tuple[str, ...] = Field(min_length=1, max_length=128)
    expected_media_types: tuple[str, ...] = Field(default=(), max_length=32)
    execution_parameters: FrozenJsonObject
    fixture_fingerprints: tuple[FixtureFingerprint, ...] = Field(default=(), max_length=MAX_FIXTURES)

    @field_validator("evidence_ids", "warnings", "output_node_ids", "expected_media_types", "fixture_fingerprints", mode="before")
    @classmethod
    def json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_output_expectations(self) -> PlannedVariant:
        _validate_output_expectations(
            self.expected_output_count,
            self.output_node_ids,
            self.expected_media_types,
            required=True,
        )
        return self


class ExperimentPlan(_FrozenModel):
    plan_hash: Sha256
    source_hashes: tuple[Sha256, ...] = Field(min_length=1, max_length=MAX_MANIFEST_SOURCES)
    variants: tuple[PlannedVariant, ...] = Field(min_length=1, max_length=MAX_PLAN_VARIANTS)
    capability_coverage: tuple[DimensionName, ...] = Field(default=(), max_length=len(DimensionName))
    warnings: tuple[str, ...] = Field(default=(), max_length=512)
    approval_payload: SanitizedJsonObject

    @field_validator("source_hashes", "variants", "capability_coverage", "warnings", mode="before")
    @classmethod
    def json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_plan_hash(self) -> Self:
        if self.plan_hash != self.recompute_plan_hash():
            raise ValueError("plan_hash does not match durable approval_payload")
        payload = thaw_json(self.approval_payload)
        durable = {
            "source_hashes": list(self.source_hashes),
            "variants": [item.model_dump(mode="json") for item in self.variants],
            "capability_coverage": [item.value for item in self.capability_coverage],
            "warnings": list(self.warnings),
        }
        for field_name, value in durable.items():
            if payload.get(field_name) != value:
                raise ValueError(f"{field_name} does not match durable approval_payload")
        return self

    def recompute_plan_hash(self) -> str:
        payload = json.dumps(
            thaw_json(self.approval_payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
