from collections.abc import Iterable, Mapping

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
    def __init__(
        self,
        *,
        search_page: SearchPage | None = None,
        fetched_workflow: FetchedWorkflow | None = None,
        capabilities: CapabilityReport | None = None,
        validation: ValidationReport | None = None,
        poll_sequences: Mapping[
            str,
            Iterable[ProviderTaskState | PollResult],
        ]
        | None = None,
        output_scripts: Mapping[
            str,
            Iterable[Iterable[ProviderOutput]],
        ]
        | None = None,
        cancel_result: CancelResult | None = None,
        method_exceptions: Mapping[
            str,
            BaseException | Iterable[BaseException],
        ]
        | None = None,
        ambiguous_submission: bool | AmbiguousSubmissionError = False,
    ) -> None:
        self.search_page = search_page or SearchPage(
            records=(), total=0, page=1, size=1, has_next=False
        )
        self.fetched_workflow = fetched_workflow
        self.capabilities = capabilities or CapabilityReport()
        self.validation = validation or ValidationReport()
        self.poll_sequences = {
            task_id: list(sequence)
            for task_id, sequence in (poll_sequences or {}).items()
        }
        for task_id, sequence in self.poll_sequences.items():
            for result in sequence:
                if (
                    isinstance(result, PollResult)
                    and result.provider_task_id != task_id
                ):
                    raise ValueError(
                        f"configured PollResult must use task ID {task_id!r}"
                    )
        self.output_scripts = {
            task_id: [tuple(batch) for batch in script]
            for task_id, script in (output_scripts or {}).items()
        }
        self.cancel_result = cancel_result or CancelResult(
            supported=True,
            cancelled=True,
            diagnostic=None,
        )
        self.method_exceptions = self._prepare_method_exceptions(
            method_exceptions or {}
        )
        self.ambiguous_submission = ambiguous_submission
        self._default_poll_sequence: tuple[ProviderTaskState | PollResult, ...] = ()
        self._default_output_script: tuple[tuple[ProviderOutput, ...], ...] = ()

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
        self._poll_indices: dict[str, int] = {}
        self._output_script_indices: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "fake"

    @classmethod
    def success(
        cls,
        *,
        outputs: Iterable[ProviderOutput] = (),
    ) -> "FakeExecutor":
        executor = cls()
        executor._default_poll_sequence = (ProviderTaskState.SUCCESS,)
        executor._default_output_script = (tuple(outputs),)
        return executor

    @staticmethod
    def _prepare_method_exceptions(
        configured: Mapping[
            str,
            BaseException | Iterable[BaseException],
        ],
    ) -> dict[str, list[BaseException]]:
        supported_methods = {
            "search",
            "fetch_workflow",
            "inspect_capabilities",
            "validate_variant",
            "submit",
            "poll",
            "collect_outputs",
            "cancel",
        }
        prepared = {}
        for method, exceptions in configured.items():
            if method not in supported_methods:
                raise ValueError(f"unknown fake executor method: {method}")
            values = (
                [exceptions]
                if isinstance(exceptions, BaseException)
                else list(exceptions)
            )
            if not all(isinstance(error, BaseException) for error in values):
                raise ValueError(f"configured {method} failures must be exceptions")
            prepared[method] = values
        return prepared

    def _raise_configured(self, method: str) -> None:
        exceptions = self.method_exceptions.get(method)
        if exceptions:
            raise exceptions.pop(0)

    def search(self, request: SearchRequest) -> SearchPage:
        self.calls.append("search")
        self.search_requests.append(request)
        self._raise_configured("search")
        return self.search_page

    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        self.calls.append("fetch_workflow")
        self.fetches.append(ref)
        self._raise_configured("fetch_workflow")
        if self.fetched_workflow is None:
            raise AssertionError("fake fetch result was not configured")
        return self.fetched_workflow

    def inspect_capabilities(self, workflow: FetchedWorkflow) -> CapabilityReport:
        self.calls.append("inspect_capabilities")
        self.inspections.append(workflow)
        self._raise_configured("inspect_capabilities")
        return self.capabilities

    def validate_variant(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> ValidationReport:
        self.calls.append("validate_variant")
        self.validations.append((variant, fixtures))
        self._raise_configured("validate_variant")
        return self.validation

    def submit(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> SubmissionReceipt:
        self.calls.append("submit")
        self.submissions.append((variant, fixtures))
        self._raise_configured("submit")
        if self.ambiguous_submission:
            if isinstance(self.ambiguous_submission, AmbiguousSubmissionError):
                raise self.ambiguous_submission
            raise AmbiguousSubmissionError("submission outcome is ambiguous")

        self._task_number += 1
        task_id = f"fake-task-{self._task_number}"
        if self._default_poll_sequence:
            self.poll_sequences.setdefault(
                task_id,
                list(self._default_poll_sequence),
            )
        if self._default_output_script:
            self.output_scripts.setdefault(
                task_id,
                list(self._default_output_script),
            )
        return SubmissionReceipt(provider_task_id=task_id)

    def poll(self, provider_task_id: str) -> PollResult:
        self.calls.append("poll")
        self.polls.append(provider_task_id)
        self._raise_configured("poll")
        sequence = self.poll_sequences.get(provider_task_id, [])
        if not sequence:
            return PollResult(
                provider_task_id=provider_task_id,
                state=ProviderTaskState.UNKNOWN,
            )

        index = min(
            self._poll_indices.get(provider_task_id, 0),
            len(sequence) - 1,
        )
        configured = sequence[index]
        self._poll_indices[provider_task_id] = index + 1
        if isinstance(configured, PollResult):
            if configured.provider_task_id != provider_task_id:
                raise ValueError(
                    f"configured PollResult must use task ID {provider_task_id!r}"
                )
            return configured
        return PollResult(provider_task_id=provider_task_id, state=configured)

    def collect_outputs(self, provider_task_id: str) -> tuple[ProviderOutput, ...]:
        self.calls.append("collect_outputs")
        self.collections.append(provider_task_id)
        self._raise_configured("collect_outputs")
        script = self.output_scripts.get(provider_task_id, [])
        if not script:
            return ()

        index = min(
            self._output_script_indices.get(provider_task_id, 0),
            len(script) - 1,
        )
        batch = script[index]
        self._output_script_indices[provider_task_id] = index + 1
        return batch

    def cancel(self, provider_task_id: str) -> CancelResult:
        self.calls.append("cancel")
        self.cancellations.append(provider_task_id)
        self._raise_configured("cancel")
        return self.cancel_result
