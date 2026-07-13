from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

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
    RunState.PLANNED: frozenset({RunState.SUBMITTED, RunState.INCOMPLETE}),
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
            if self.state is RunState.PLANNED and new_state is RunState.INCOMPLETE:
                error = updates.get("error", self.error)
                if not _is_ambiguous_submission(error):
                    raise ValueError(
                        "PLANNED to INCOMPLETE transition requires "
                        "error.kind == 'ambiguous_submission'"
                    )
                if updates.get("provider_task_id", self.provider_task_id) is not None:
                    raise ValueError(
                        "provider_task_id must be absent for ambiguous submission"
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
