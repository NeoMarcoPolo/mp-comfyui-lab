from __future__ import annotations

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
        values = self.model_dump()
        values.update(updates)
        values["state"] = new_state
        return type(self).model_validate(values)
