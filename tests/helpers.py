from collections.abc import Iterable

from mp_comfyui_lab.executors.base import (
    AmbiguousSubmissionError,
    CancelResult,
    CapabilityReport,
    Executor,
    FetchedWorkflow,
    PollResult,
    ProviderOutput,
    ProviderTaskState,
    SearchPage,
    SearchRequest,
    SubmissionReceipt,
    ValidationReport,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.models import MaterializedVariant, ResolvedFixtures


class FakeExecutor(Executor):
    name = "fake"

    def __init__(
        self,
        *,
        search_page: SearchPage | None = None,
        fetched_workflow: FetchedWorkflow | None = None,
        capabilities: CapabilityReport | None = None,
        validation: ValidationReport | None = None,
        poll_sequence: Iterable[ProviderTaskState | PollResult] = (),
        outputs: Iterable[ProviderOutput] = (),
        output_batches: Iterable[Iterable[ProviderOutput]] | None = None,
        ambiguous_submission: bool | AmbiguousSubmissionError = False,
    ) -> None:
        self.search_page = search_page or SearchPage(
            records=(), total=0, page=1, size=1, has_next=False
        )
        self.fetched_workflow = fetched_workflow
        self.capabilities = capabilities or CapabilityReport()
        self.validation = validation or ValidationReport()
        self.poll_sequence = list(poll_sequence)
        self.outputs = list(outputs)
        self.output_batches = (
            None
            if output_batches is None
            else [tuple(batch) for batch in output_batches]
        )
        self.ambiguous_submission = ambiguous_submission

        self.calls: list[str] = []
        self.search_requests: list[SearchRequest] = []
        self.fetches: list[WorkflowRef] = []
        self.inspections: list[FetchedWorkflow] = []
        self.validations: list[tuple[MaterializedVariant, ResolvedFixtures]] = []
        self.submissions: list[tuple[MaterializedVariant, ResolvedFixtures]] = []
        self.polls: list[str] = []
        self.collections: list[str] = []
        self.cancellations: list[str] = []
        self._task_number = 0
        self._poll_index = 0
        self._output_batch_index = 0

    @classmethod
    def success(
        cls,
        *,
        outputs: Iterable[ProviderOutput] = (),
    ) -> "FakeExecutor":
        return cls(
            poll_sequence=(ProviderTaskState.SUCCESS,),
            outputs=outputs,
        )

    def search(self, request: SearchRequest) -> SearchPage:
        self.calls.append("search")
        self.search_requests.append(request)
        return self.search_page

    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        self.calls.append("fetch_workflow")
        self.fetches.append(ref)
        if self.fetched_workflow is None:
            raise AssertionError("fake fetch result was not configured")
        return self.fetched_workflow

    def inspect_capabilities(self, workflow: FetchedWorkflow) -> CapabilityReport:
        self.calls.append("inspect_capabilities")
        self.inspections.append(workflow)
        return self.capabilities

    def validate_variant(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> ValidationReport:
        self.calls.append("validate_variant")
        self.validations.append((variant, fixtures))
        return self.validation

    def submit(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> SubmissionReceipt:
        self.calls.append("submit")
        self.submissions.append((variant, fixtures))
        if self.ambiguous_submission:
            if isinstance(self.ambiguous_submission, AmbiguousSubmissionError):
                raise self.ambiguous_submission
            raise AmbiguousSubmissionError("submission outcome is ambiguous")

        self._task_number += 1
        return SubmissionReceipt(provider_task_id=f"fake-task-{self._task_number}")

    def poll(self, provider_task_id: str) -> PollResult:
        self.calls.append("poll")
        self.polls.append(provider_task_id)
        if not self.poll_sequence:
            return PollResult(
                provider_task_id=provider_task_id,
                state=ProviderTaskState.UNKNOWN,
            )

        index = min(self._poll_index, len(self.poll_sequence) - 1)
        configured = self.poll_sequence[index]
        self._poll_index += 1
        if isinstance(configured, PollResult):
            return configured
        return PollResult(provider_task_id=provider_task_id, state=configured)

    def collect_outputs(self, provider_task_id: str) -> tuple[ProviderOutput, ...]:
        self.calls.append("collect_outputs")
        self.collections.append(provider_task_id)
        if self.output_batches is None:
            return tuple(self.outputs)
        if not self.output_batches:
            return ()

        index = min(self._output_batch_index, len(self.output_batches) - 1)
        batch = self.output_batches[index]
        self._output_batch_index += 1
        return batch

    def cancel(self, provider_task_id: str) -> CancelResult:
        self.calls.append("cancel")
        self.cancellations.append(provider_task_id)
        return CancelResult(supported=True, cancelled=True, diagnostic=None)
