from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from mp_comfyui_lab.discovery.models import WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.experiments.models import MaterializedVariant, ResolvedFixtures
from mp_comfyui_lab.security import sanitize_for_persistence


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SanitizedMetadata = Annotated[
    dict[str, JsonValue],
    BeforeValidator(sanitize_for_persistence),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SearchRequest(_FrozenModel):
    query: NonBlankStr
    sort: str = ""
    page: int = Field(default=1, ge=1)
    size: int = Field(default=30, ge=1, le=100)


class SearchPage(_FrozenModel):
    records: tuple[WorkflowCandidate, ...] = ()
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    size: int = Field(ge=1)
    has_next: bool


class WorkflowRef(_FrozenModel):
    executor: NonBlankStr
    workflow_id: NonBlankStr
    canonical_url: NonBlankStr | None = None


class FetchedWorkflow(_FrozenModel):
    ref: WorkflowRef
    detail: WorkflowDetail
    graph: dict[str, Any]


class CapabilityReport(_FrozenModel):
    required_nodes: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    lora_names: tuple[str, ...] = ()
    output_node_ids: tuple[str, ...] = ()
    provider_metadata: SanitizedMetadata = Field(default_factory=dict)


class ValidationIssue(_FrozenModel):
    code: NonBlankStr
    message: NonBlankStr


class ValidationReport(_FrozenModel):
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


class SubmissionReceipt(_FrozenModel):
    provider_task_id: NonBlankStr
    provider_state: str | None = None
    provider_metadata: SanitizedMetadata = Field(default_factory=dict)


class ProviderTaskState(StrEnum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class PollResult(_FrozenModel):
    provider_task_id: NonBlankStr
    state: ProviderTaskState
    diagnostic: str | None = None
    provider_metadata: SanitizedMetadata = Field(default_factory=dict)


class ProviderOutput(_FrozenModel):
    node_id: str | None = None
    url: NonBlankStr
    media_type: str | None = None
    file_name: str | None = None
    provider_order: int = Field(default=0, ge=0)
    provider_metadata: SanitizedMetadata = Field(default_factory=dict)


class CancelResult(_FrozenModel):
    supported: bool
    cancelled: bool
    diagnostic: str | None = None

    @model_validator(mode="after")
    def reject_unsupported_cancellation(self) -> CancelResult:
        if self.cancelled and not self.supported:
            raise ValueError("cancelled cannot be true when cancellation is unsupported")
        return self


class ExecutorError(RuntimeError):
    pass


class ProviderSchemaError(ExecutorError):
    pass


class UnsupportedCapabilityError(ExecutorError):
    pass


class AmbiguousSubmissionError(ExecutorError):
    pass


class Executor(ABC):
    name: str

    @abstractmethod
    def search(self, request: SearchRequest) -> SearchPage:
        raise NotImplementedError

    @abstractmethod
    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        raise NotImplementedError

    @abstractmethod
    def inspect_capabilities(self, workflow: FetchedWorkflow) -> CapabilityReport:
        raise NotImplementedError

    @abstractmethod
    def validate_variant(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> ValidationReport:
        raise NotImplementedError

    @abstractmethod
    def submit(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> SubmissionReceipt:
        raise NotImplementedError

    @abstractmethod
    def poll(self, provider_task_id: str) -> PollResult:
        raise NotImplementedError

    @abstractmethod
    def collect_outputs(self, provider_task_id: str) -> tuple[ProviderOutput, ...]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, provider_task_id: str) -> CancelResult:
        raise NotImplementedError
