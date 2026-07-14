from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from enum import StrEnum
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

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
from mp_comfyui_lab.json_types import (
    FrozenJsonObject,
    SanitizedJsonObject,
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
    "expected_output_count",
    "created_at",
)


def _is_ambiguous_submission(error: Any) -> bool:
    return (
        isinstance(error, Mapping)
        and error.get("kind") == "ambiguous_submission"
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
    warnings: tuple[str, ...] = ()

    @field_validator("graph")
    @classmethod
    def validate_graph(cls, graph: FrozenJsonObject) -> FrozenJsonObject:
        if not graph:
            raise ValueError("graph must not be empty")
        if any(not node_id.strip() for node_id in graph):
            raise ValueError("graph node IDs must not be blank")
        return graph


class RunRecord(_FrozenModel):
    run_id: NonBlankStr
    experiment_id: NonBlankStr
    variant_id: NonBlankStr
    state: RunState
    provider_task_id: NonBlankStr | None = None
    expected_output_count: int = Field(gt=0)
    actual_output_count: int = Field(default=0, ge=0)
    error: SanitizedJsonObject | None = None
    created_at: UtcDatetime | None = None
    submitted_at: UtcDatetime | None = None
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_provider_task_id(self) -> RunRecord:
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
        if (
            self.state is RunState.SUCCESS
            and self.actual_output_count < self.expected_output_count
        ):
            raise ValueError(
                "actual_output_count must meet expected_output_count for SUCCESS"
            )

        timestamps = [
            timestamp
            for timestamp in (
                self.created_at,
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
        return self


class SourceBundle(_FrozenModel):
    sources: tuple[PlanningSource, ...] = Field(min_length=1, max_length=MAX_MANIFEST_SOURCES)
    candidates: tuple[Candidate, ...] = Field(default=(), max_length=MAX_PLAN_VARIANTS)

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


class ExperimentPlan(_FrozenModel):
    plan_hash: Sha256
    source_hashes: tuple[Sha256, ...] = Field(min_length=1, max_length=MAX_MANIFEST_SOURCES)
    variants: tuple[PlannedVariant, ...] = Field(min_length=1, max_length=MAX_PLAN_VARIANTS)
    capability_coverage: tuple[DimensionName, ...] = Field(default=(), max_length=len(DimensionName))
    warnings: tuple[str, ...] = Field(default=(), max_length=512)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
