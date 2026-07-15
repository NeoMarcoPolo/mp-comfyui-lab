from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.discovery.models import WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.discovery.query import (
    ExpandedQuery,
    ExpansionRecord,
    expand_query,
    query_tokens,
)
from mp_comfyui_lab.discovery.ranking import (
    Artifact,
    GraphEvidence,
    RankedWorkflow,
    coarse_duplicate_group,
    load_default_ranking_weights,
    rank_candidates,
    select_group_representatives,
)
from mp_comfyui_lab.discovery.urls import parse_runninghub_url
from mp_comfyui_lab.executors.base import (
    Executor,
    FetchedWorkflow,
    SearchPage,
    SearchRequest,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.graph.hashing import GraphHashError, capability_hash, source_graph_hash
from mp_comfyui_lab.graph.models import NormalizedGraph, SemanticRole
from mp_comfyui_lab.graph.normalize import GraphNormalizationError, normalize_graph
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import (
    sanitize_graph_for_persistence,
    sanitize_untrusted_diagnostic,
)


DEFAULT_SORTS = ("", "REPUTATION", "HOTTEST", "NEWEST")
MAX_DERIVATIONS = 16
MAX_DIAGNOSTIC_SAMPLE = 16
# RunningHub accepts compact free-text searches; 512 keeps category evidence
# representable while bounding every provider request and persisted model field.
MAX_DERIVATION_LENGTH = 512

BoundedText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
]
DerivationText = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_DERIVATION_LENGTH,
    ),
]


class DiscoveryNotFoundError(LookupError):
    def __init__(
        self,
        message: str,
        *,
        experiment_id: str | None = None,
        artifacts: tuple[Artifact, ...] = (),
    ) -> None:
        super().__init__(sanitize_untrusted_diagnostic(message)[:512])
        self.experiment_id = experiment_id
        self.artifacts = artifacts


class DiscoveryEvidenceError(ValueError):
    pass


class _ArtifactDiscoveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        experiment_id: str,
        artifacts: tuple[Artifact, ...],
        diagnostics: tuple[DiscoveryAttempt, ...],
    ) -> None:
        sanitized = sanitize_untrusted_diagnostic(message)[:512]
        super().__init__(sanitized or "discovery incomplete")
        self.experiment_id = experiment_id
        self.artifacts = artifacts
        self.diagnostics = diagnostics


class DiscoveryIncompleteError(_ArtifactDiscoveryError):
    pass


class DiscoveryExecutionError(_ArtifactDiscoveryError):
    pass


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class DerivationEvidence(_FrozenModel):
    source: BoundedText
    value: DerivationText
    omitted_count: int = Field(default=0, ge=0, le=1_000_000)


class DiscoveryAttempt(_FrozenModel):
    order: int = Field(ge=1)
    timestamp: BoundedText
    operation: Literal["search", "detail", "graph"]
    status: Literal["success", "failed"]
    query: str | None = None
    sort: str | None = None
    workflow_id: str | None = None
    error_kind: str | None = None
    message: str | None = None
    result_ids: tuple[BoundedText, ...] = Field(default=(), max_length=10_000)
    artifact_paths: tuple[BoundedText, ...] = Field(default=(), max_length=64)


class DiscoveryResult(_FrozenModel):
    experiment_id: BoundedText
    query_plan: ExpandedQuery
    results: tuple[RankedWorkflow, ...]
    derivations: tuple[DerivationEvidence, ...] = Field(
        default=(),
        max_length=MAX_DERIVATIONS,
    )
    artifacts: tuple[Artifact, ...] = ()
    incomplete: bool = False
    diagnostic_count: int = Field(default=0, ge=0)
    diagnostics_artifact: Artifact | None = None
    diagnostics: tuple[DiscoveryAttempt, ...] = Field(
        default=(),
        max_length=MAX_DIAGNOSTIC_SAMPLE,
    )


class InspectResult(_FrozenModel):
    experiment_id: BoundedText
    detail: WorkflowDetail
    source_hash: str | None = None
    capability_hash: str | None = None
    roles: tuple[BoundedText, ...] = ()
    canonicalization_error: BoundedText | None = None
    artifacts: tuple[Artifact, ...] = ()


class _Enriched:
    __slots__ = ("fetched", "evidence", "artifacts", "normalized")

    def __init__(
        self,
        fetched: FetchedWorkflow,
        evidence: GraphEvidence,
        artifacts: tuple[Artifact, ...],
        normalized: NormalizedGraph | None,
    ) -> None:
        self.fetched = fetched
        self.evidence = evidence
        self.artifacts = artifacts
        self.normalized = normalized

    @property
    def detail(self) -> WorkflowDetail:
        return self.fetched.detail


class _DiscoveryJournal:
    def __init__(
        self,
        service: DiscoveryService,
        experiment_id: str,
        *,
        operation: str,
        query_plan: ExpandedQuery | None = None,
        initial_query: str | None = None,
        derivations: tuple[DerivationEvidence, ...] = (),
    ) -> None:
        self.service = service
        self.experiment_id = experiment_id
        self.operation = operation
        self.query_plan = query_plan
        self.initial_query = initial_query
        self.derivations = derivations
        self.requests: list[SearchRequest] = []
        self.pages: list[SearchPage] = []
        self.attempt_count = 0
        self.diagnostic_count = 0
        self._diagnostics: list[DiscoveryAttempt] = []
        self.results: tuple[RankedWorkflow, ...] = ()
        self.status = "running"
        snapshot_path = service.store.root(experiment_id) / "search-results.json"
        attempts_path = service.store.root(experiment_id) / "attempts.jsonl"
        service.store.write_text(experiment_id, "attempts.jsonl", "")
        self.diagnostics_artifact = Artifact(kind="attempts", path=str(attempts_path))
        self.artifacts: list[Artifact] = [
            Artifact(kind="search", path=str(snapshot_path)),
            self.diagnostics_artifact,
        ]
        self.checkpoint()

    @property
    def diagnostics(self) -> tuple[DiscoveryAttempt, ...]:
        return tuple(self._diagnostics)

    def set_plan(
        self,
        query_plan: ExpandedQuery,
        derivations: tuple[DerivationEvidence, ...],
    ) -> None:
        self.query_plan = query_plan
        self.derivations = derivations
        self.checkpoint()

    def add_artifacts(self, artifacts: Sequence[Artifact]) -> None:
        self.artifacts.extend(artifacts)

    def record(
        self,
        operation: Literal["search", "detail", "graph"],
        status: Literal["success", "failed"],
        *,
        query: str | None = None,
        sort: str | None = None,
        workflow_id: str | None = None,
        result_ids: Sequence[str] = (),
        artifact_paths: Sequence[str] = (),
        error: Exception | None = None,
        error_kind: str | None = None,
        message: str | None = None,
    ) -> None:
        if error is not None:
            error_kind = type(error).__name__
            message = str(error)
        sanitized_kind = (
            re.sub(r"[^A-Za-z0-9_.-]", "_", error_kind)[:128]
            if error_kind
            else None
        )
        sanitized_message = None
        if message is not None:
            sanitized_message = sanitize_untrusted_diagnostic(message)[:512]
            if not sanitized_message:
                sanitized_message = f"{operation} failed"
        attempt = DiscoveryAttempt(
            order=self.attempt_count + 1,
            timestamp=(
                self.service.clock()
                .astimezone(UTC)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            operation=operation,
            status=status,
            query=query,
            sort=sort,
            workflow_id=workflow_id,
            error_kind=sanitized_kind,
            message=sanitized_message,
            result_ids=tuple(result_ids),
            artifact_paths=tuple(artifact_paths),
        )
        self.service.store.append_jsonl(
            self.experiment_id,
            "attempts.jsonl",
            attempt.model_dump(mode="json"),
        )
        self.attempt_count += 1
        if status == "failed":
            self.diagnostic_count += 1
            if len(self._diagnostics) < MAX_DIAGNOSTIC_SAMPLE:
                self._diagnostics.append(attempt)

    def phase(self) -> None:
        self.checkpoint()

    def finish(self, results: tuple[RankedWorkflow, ...]) -> None:
        self.results = results
        self.status = "partial" if self.diagnostic_count else "complete"
        self.checkpoint()

    def fail(self) -> None:
        self.status = "failed"
        self.checkpoint()

    def checkpoint(self) -> None:
        plan = self.query_plan
        unique_artifacts = self.service._unique_artifacts(self.artifacts)
        self.service.store.write_json(
            self.experiment_id,
            "search-results.json",
            {
                "operation": self.operation,
                "status": self.status,
                "incomplete": self.status != "complete",
                "query": plan.original if plan is not None else self.initial_query,
                "queries": list(plan.queries) if plan is not None else [],
                "source_queries": list(plan.source_queries) if plan is not None else [],
                "expansions": (
                    [item.model_dump(mode="json") for item in plan.expansions]
                    if plan is not None
                    else []
                ),
                "derivations": [
                    item.model_dump(mode="json") for item in self.derivations
                ],
                "provider_request_count": len(self.requests),
                "provider_page_count": len(self.pages),
                "attempt_count": self.attempt_count,
                "diagnostic_count": self.diagnostic_count,
                "diagnostics_artifact": self.diagnostics_artifact.path,
                "diagnostics_sample": [
                    item.model_dump(mode="json") for item in self.diagnostics
                ],
                "results": [
                    {
                        "workflow_id": item.candidate.workflow_id,
                        "url": item.candidate.canonical_url,
                        "score": item.score,
                    }
                    for item in self.results[:100]
                ],
                "result_count": len(self.results),
                "artifacts": [
                    item.model_dump(mode="json") for item in unique_artifacts
                ][:128],
                "artifact_count": len(unique_artifacts),
            },
        )


class DiscoveryService:
    def __init__(
        self,
        *,
        executor: Executor,
        config: LabConfig,
        store: ExperimentStore | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.executor = executor
        self.config = config
        self.store = store if store is not None else ExperimentStore(config.experiments_root)
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.id_factory = id_factory if id_factory is not None else lambda: uuid4().hex[:12]

    def search(
        self,
        query: str,
        *,
        sort: str | None = None,
        page: int = 1,
        size: int = 30,
        include_graphs: bool = False,
    ) -> DiscoveryResult:
        plan = expand_query(query)
        experiment_id = self.store.create_unique(self._new_experiment_id("search"))
        journal = _DiscoveryJournal(
            self,
            experiment_id,
            operation="search",
            query_plan=plan,
        )
        try:
            return self._discover(
                journal,
                sort=sort,
                page=page,
                size=size,
                include_graphs=include_graphs,
            )
        except (DiscoveryIncompleteError, DiscoveryExecutionError):
            raise
        except Exception as error:
            raise self._terminal_error(journal, error) from error

    def inspect(self, url: str) -> InspectResult:
        parsed = parse_runninghub_url(url)
        experiment_id = self.store.create_unique(self._new_experiment_id("inspect"))
        enriched = self._inspect_into(
            experiment_id,
            WorkflowRef(
                executor=self.executor.name,
                workflow_id=parsed.workflow_id,
                canonical_url=parsed.canonical_url,
            ),
        )
        evidence = enriched.evidence
        return InspectResult(
            experiment_id=experiment_id,
            detail=enriched.detail,
            source_hash=evidence.source_hash,
            capability_hash=evidence.capability_hash,
            roles=evidence.roles,
            canonicalization_error=evidence.canonicalization_error,
            artifacts=enriched.artifacts,
        )

    def similar(
        self,
        url: str,
        *,
        sort: str | None = None,
        page: int = 1,
        size: int = 30,
        include_graphs: bool = False,
    ) -> DiscoveryResult:
        parsed = parse_runninghub_url(url)
        experiment_id = self.store.create_unique(self._new_experiment_id("similar"))
        journal = _DiscoveryJournal(
            self,
            experiment_id,
            operation="similar",
            initial_query=parsed.canonical_url,
        )
        try:
            return self._similar_from_journal(
                journal,
                parsed.workflow_id,
                parsed.canonical_url,
                sort=sort,
                page=page,
                size=size,
                include_graphs=include_graphs,
            )
        except (
            DiscoveryIncompleteError,
            DiscoveryExecutionError,
            DiscoveryNotFoundError,
        ):
            raise
        except Exception as error:
            raise self._terminal_error(journal, error) from error

    def _similar_from_journal(
        self,
        journal: _DiscoveryJournal,
        workflow_id: str,
        canonical_url: str,
        *,
        sort: str | None,
        page: int,
        size: int,
        include_graphs: bool,
    ) -> DiscoveryResult:
        experiment_id = journal.experiment_id
        try:
            source = self._inspect_into(
                experiment_id,
                WorkflowRef(
                    executor=self.executor.name,
                    workflow_id=workflow_id,
                    canonical_url=canonical_url,
                ),
            )
        except Exception as error:
            journal.record(
                "graph",
                "failed",
                workflow_id=workflow_id,
                error=error,
            )
            journal.fail()
            raise DiscoveryIncompleteError(
                "source workflow inspection failed",
                experiment_id=experiment_id,
                artifacts=self._unique_artifacts(journal.artifacts),
                diagnostics=journal.diagnostics,
            ) from error
        journal.add_artifacts(source.artifacts)
        if source.evidence.canonicalization_error is not None:
            journal.record(
                "graph",
                "failed",
                workflow_id=workflow_id,
                error_kind="GraphCanonicalizationError",
                message=source.evidence.canonicalization_error,
                artifact_paths=tuple(item.path for item in source.artifacts),
            )
        else:
            journal.record(
                "graph",
                "success",
                workflow_id=workflow_id,
                artifact_paths=tuple(item.path for item in source.artifacts),
            )
        derivations = self._derive_searches(source)
        if not derivations:
            journal.fail()
            raise DiscoveryNotFoundError(
                f"workflow {workflow_id} has no derivable public capability evidence",
                experiment_id=experiment_id,
                artifacts=self._unique_artifacts(journal.artifacts),
            )
        query_plan = self._derivation_query_plan(derivations)
        journal.set_plan(query_plan, derivations)
        discovered = self._discover(
            journal,
            sort=sort,
            page=page,
            size=size,
            include_graphs=include_graphs,
            initial_enriched={workflow_id: source},
        )
        return discovered

    def _terminal_error(
        self,
        journal: _DiscoveryJournal,
        error: Exception,
    ) -> DiscoveryExecutionError:
        journal.fail()
        message = sanitize_untrusted_diagnostic(str(error))[:384]
        return DiscoveryExecutionError(
            f"discovery execution failed: {message or type(error).__name__}",
            experiment_id=journal.experiment_id,
            artifacts=self._unique_artifacts(journal.artifacts),
            diagnostics=journal.diagnostics,
        )

    def _discover(
        self,
        journal: _DiscoveryJournal,
        *,
        sort: str | None,
        page: int,
        size: int,
        include_graphs: bool,
        initial_enriched: Mapping[str, _Enriched] | None = None,
    ) -> DiscoveryResult:
        experiment_id = journal.experiment_id
        query_plan = journal.query_plan
        if query_plan is None:
            raise AssertionError("discovery journal is missing a query plan")
        sorts = (sort,) if sort is not None else DEFAULT_SORTS
        seen_requests: set[SearchRequest] = set()
        successful_searches = 0
        for provider_query in query_plan.queries:
            for provider_sort in sorts:
                request = SearchRequest(
                    query=provider_query,
                    sort=provider_sort,
                    page=page,
                    size=size,
                )
                if request in seen_requests:
                    continue
                seen_requests.add(request)
                journal.requests.append(request)
                try:
                    page_result = self.executor.search(request)
                except Exception as error:
                    journal.record(
                        "search",
                        "failed",
                        query=provider_query,
                        sort=provider_sort,
                        error=error,
                    )
                    continue
                successful_searches += 1
                journal.pages.append(page_result)
                journal.record(
                    "search",
                    "success",
                    query=provider_query,
                    sort=provider_sort,
                    result_ids=tuple(
                        sorted(
                            {
                                candidate.workflow_id
                                for candidate in page_result.records
                            }
                        )
                    ),
                )

        journal.phase()

        if successful_searches == 0:
            journal.fail()
            raise DiscoveryIncompleteError(
                "no provider search succeeded",
                experiment_id=experiment_id,
                artifacts=self._unique_artifacts(journal.artifacts),
                diagnostics=journal.diagnostics,
            )

        merged = self._merge_candidates(
            candidate
            for search_page in journal.pages
            for candidate in search_page.records
        )
        details: dict[str, WorkflowDetail] = {}
        evidence: dict[str, GraphEvidence] = {}
        enriched_cache = dict(initial_enriched or {})
        for workflow_id in sorted(merged):
            cached = enriched_cache.get(workflow_id)
            if cached is not None:
                details[workflow_id] = cached.detail
                evidence[workflow_id] = cached.evidence
            else:
                detail_method = getattr(self.executor, "detail", None)
                try:
                    if callable(detail_method):
                        detail = detail_method(workflow_id)
                    else:
                        detail = self.executor.fetch_workflow(
                            WorkflowRef(
                                executor=self.executor.name,
                                workflow_id=workflow_id,
                                canonical_url=merged[
                                    workflow_id
                                ].canonical_url,
                            )
                        ).detail
                except Exception as error:
                    journal.record(
                        "detail",
                        "failed",
                        workflow_id=workflow_id,
                        error=error,
                    )
                    continue
                details[workflow_id] = detail
                detail_artifact = self._persist_detail(experiment_id, detail)
                journal.add_artifacts((detail_artifact,))
                journal.record(
                    "detail",
                    "success",
                    workflow_id=workflow_id,
                    artifact_paths=(detail_artifact.path,),
                )

        journal.phase()

        weights = load_default_ranking_weights(
            self.config.ranking.model_dump(exclude_none=True)
        )
        if include_graphs:
            graph_groups: dict[str, list[str]] = {}
            for workflow_id, detail in details.items():
                group = coarse_duplicate_group(detail) or f"workflow:{workflow_id}"
                graph_groups.setdefault(group, []).append(workflow_id)
            group_keys = {
                workflow_id: group
                for group, workflow_ids in graph_groups.items()
                for workflow_id in workflow_ids
            }
            selected = select_group_representatives(
                query_plan,
                tuple(details.values()),
                weights,
                group_keys,
                graph_evidence=evidence,
            )
            graph_representatives: list[str] = []
            for group, workflow_ids in graph_groups.items():
                if any(workflow_id in enriched_cache for workflow_id in workflow_ids):
                    continue
                graph_representatives.append(selected[group])
            for representative in sorted(graph_representatives):
                try:
                    enriched = self._inspect_into(
                        experiment_id,
                        WorkflowRef(
                            executor=self.executor.name,
                            workflow_id=representative,
                            canonical_url=merged[representative].canonical_url,
                        ),
                    )
                except Exception as error:
                    message = (
                        sanitize_untrusted_diagnostic(str(error))[:512]
                        or "graph enrichment failed"
                    )
                    evidence[representative] = GraphEvidence(
                        canonicalization_error=message
                    )
                    journal.record(
                        "graph",
                        "failed",
                        workflow_id=representative,
                        error=error,
                    )
                    continue
                details[representative] = enriched.detail
                evidence[representative] = enriched.evidence
                journal.add_artifacts(enriched.artifacts)
                if enriched.evidence.canonicalization_error is not None:
                    journal.record(
                        "graph",
                        "failed",
                        workflow_id=representative,
                        error_kind="GraphCanonicalizationError",
                        message=enriched.evidence.canonicalization_error,
                        artifact_paths=tuple(
                            item.path for item in enriched.artifacts
                        ),
                    )
                else:
                    journal.record(
                        "graph",
                        "success",
                        workflow_id=representative,
                        artifact_paths=tuple(
                            item.path for item in enriched.artifacts
                        ),
                    )
            journal.phase()

        ranked = rank_candidates(
            query_plan,
            tuple(merged.values()),
            details,
            weights,
            graph_evidence=evidence,
        )
        journal.finish(ranked)
        return DiscoveryResult(
            experiment_id=experiment_id,
            query_plan=query_plan,
            results=ranked,
            derivations=journal.derivations,
            artifacts=self._unique_artifacts(journal.artifacts),
            incomplete=journal.diagnostic_count > 0,
            diagnostic_count=journal.diagnostic_count,
            diagnostics_artifact=journal.diagnostics_artifact,
            diagnostics=journal.diagnostics,
        )

    def _inspect_into(self, experiment_id: str, ref: WorkflowRef) -> _Enriched:
        fetched = self.executor.fetch_workflow(ref)
        safe_graph = sanitize_graph_for_persistence(thaw_json(fetched.graph))
        safe_fetched = FetchedWorkflow(
            ref=fetched.ref,
            detail=fetched.detail,
            graph=safe_graph,
        )
        artifacts: list[Artifact] = [self._persist_detail(experiment_id, fetched.detail)]
        graph_path = self.store._write_pre_sanitized_json(
            experiment_id,
            Path("sources") / ref.workflow_id / "graph.json",
            safe_graph,
        )
        artifacts.append(Artifact(kind="graph", path=str(graph_path)))

        normalized: NormalizedGraph | None = None
        try:
            normalized = normalize_graph(safe_graph)
            source_digest = source_graph_hash(safe_graph)
            capability_digest = capability_hash(safe_graph)
            roles = tuple(group.role.value for group in normalized.role_bindings)
            evidence = GraphEvidence(
                source_hash=source_digest,
                capability_hash=capability_digest,
                roles=roles,
            )
            normalized_path = self.store._write_pre_sanitized_json(
                experiment_id,
                Path("sources") / ref.workflow_id / "normalized.json",
                sanitize_graph_for_persistence(
                    normalized.model_dump(mode="json")
                ),
            )
            artifacts.append(Artifact(kind="normalized", path=str(normalized_path)))
        except (GraphHashError, GraphNormalizationError, RecursionError, UnicodeError) as error:
            message = (
                sanitize_untrusted_diagnostic(str(error))[:512]
                or "graph canonicalization failed"
            )
            evidence = GraphEvidence(canonicalization_error=message)

        hashes_path = self.store.write_json(
            experiment_id,
            Path("sources") / ref.workflow_id / "hashes.json",
            evidence.model_dump(mode="json"),
        )
        artifacts.append(Artifact(kind="hashes", path=str(hashes_path)))
        return _Enriched(
            safe_fetched,
            evidence,
            self._unique_artifacts(artifacts),
            normalized,
        )

    def _persist_detail(self, experiment_id: str, detail: WorkflowDetail) -> Artifact:
        path = self.store.write_json(
            experiment_id,
            Path("sources") / detail.candidate.workflow_id / "detail.json",
            detail.model_dump(mode="json"),
        )
        return Artifact(kind="detail", path=str(path))

    def _derive_searches(
        self,
        source: _Enriched,
    ) -> tuple[DerivationEvidence, ...]:
        report = self.executor.inspect_capabilities(source.fetched)
        candidates: list[tuple[str, str, int]] = []
        self._append_category(
            candidates,
            "model",
            tuple(self._inventory_search(name) for name in report.model_names),
        )
        self._append_category(
            candidates,
            "lora",
            tuple(self._inventory_search(name) for name in report.lora_names),
        )
        self._append_category(candidates, "tag", source.detail.candidate.tags)
        prompt_bindings = (
            source.normalized.bindings(SemanticRole.PROMPT_SOURCE)
            if source.normalized is not None
            else ()
        )
        fanout_bindings = (
            source.normalized.bindings(SemanticRole.PROMPT_FANOUT)
            if source.normalized is not None
            else ()
        )
        conditioning_bindings = (
            source.normalized.bindings(SemanticRole.CONDITIONING)
            if source.normalized is not None
            else ()
        )
        prompts = tuple(
            value
            for binding in prompt_bindings
            if isinstance((value := thaw_json(binding.value)), str)
        )
        fanout_count = len(fanout_bindings)
        conditioning_count = len(conditioning_bindings)
        line_count = max(
            (len(value.splitlines()) or 1 for value in prompts),
            default=0,
        )
        comma_count = max((value.count(",") for value in prompts), default=0)
        placeholder_pattern = re.compile(r"{[^{}]*}|<[^<>]*>|\[[^\[\]]*\]")
        placeholder_count = max(
            (len(placeholder_pattern.findall(value)) for value in prompts),
            default=0,
        )
        has_prompt_evidence = bool(
            prompt_bindings or fanout_bindings or conditioning_bindings
        )
        has_grammar_features = any(
            (
                line_count,
                comma_count,
                placeholder_count,
                fanout_count,
                conditioning_count,
            )
        )
        if has_prompt_evidence or has_grammar_features:
            self._append_category(
                candidates,
                "prompt_grammar",
                (
                    "prompt grammar "
                    f"lines {line_count} commas {comma_count} "
                    f"placeholders {placeholder_count} "
                    f"fanout {fanout_count} conditioning {conditioning_count}",
                ),
            )
        if source.normalized is not None:
            self._append_category(
                candidates,
                "capability_role",
                tuple(
                    sorted(
                        {group.role.value for group in source.normalized.role_bindings}
                    )
                ),
            )

        unique: list[DerivationEvidence] = []
        seen_term_sets: set[frozenset[str]] = set()
        for kind, raw_value, omitted_count in candidates:
            value = raw_value.strip()
            terms = frozenset(query_tokens(value))
            if not value or not terms or terms in seen_term_sets:
                continue
            seen_term_sets.add(terms)
            unique.append(
                DerivationEvidence(
                    source=kind,
                    value=value,
                    omitted_count=omitted_count,
                )
            )
        if len(unique) > 1:
            summary_tokens: list[str] = []
            for derivation in unique:
                for token in query_tokens(derivation.value):
                    if token not in summary_tokens:
                        summary_tokens.append(token)
                        break
            summary = " ".join(summary_tokens)
            summary_terms = frozenset(query_tokens(summary))
            if summary_terms and summary_terms not in seen_term_sets:
                self._require_derivation_length("summary", summary)
                unique.append(DerivationEvidence(source="summary", value=summary))
        return tuple(unique)

    @staticmethod
    def _append_category(
        candidates: list[tuple[str, str, int]],
        category: str,
        values: Sequence[str],
    ) -> None:
        unique: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            value = raw_value.strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                unique.append(value)
        if not unique:
            return
        included: list[str] = []
        omitted_count = 0
        for value in unique:
            if len(value) > MAX_DERIVATION_LENGTH:
                raise DiscoveryEvidenceError(
                    f"{category} evidence item cannot be represented within "
                    f"{MAX_DERIVATION_LENGTH} characters"
                )
            candidate = " ".join((*included, value))
            if len(candidate) <= MAX_DERIVATION_LENGTH:
                included.append(value)
            else:
                omitted_count += 1
        combined = " ".join(included)
        if not combined:
            raise DiscoveryEvidenceError(
                f"{category} evidence cannot be represented within "
                f"{MAX_DERIVATION_LENGTH} characters"
            )
        candidates.append((category, combined, omitted_count))

    @staticmethod
    def _require_derivation_length(category: str, value: str) -> None:
        if len(value) > MAX_DERIVATION_LENGTH:
            raise DiscoveryEvidenceError(
                f"{category} evidence cannot be represented within "
                f"{MAX_DERIVATION_LENGTH} characters"
            )

    @staticmethod
    def _inventory_search(value: str) -> str:
        try:
            parsed = urlsplit(value)
            candidate = Path(parsed.path).name if parsed.scheme else Path(value).name
        except ValueError:
            candidate = value
        candidate = re.sub(
            r"\.(?:ckpt|pt|pth|safetensors)$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        return candidate or value

    @staticmethod
    def _derivation_query_plan(
        derivations: tuple[DerivationEvidence, ...],
    ) -> ExpandedQuery:
        queries: list[str] = []
        expansions: list[ExpansionRecord] = []
        expansion_keys: set[tuple[str, str]] = set()
        seen_term_sets: set[frozenset[str]] = set()
        for derivation in derivations:
            term_set = frozenset(query_tokens(derivation.value))
            if not term_set or term_set in seen_term_sets:
                continue
            seen_term_sets.add(term_set)
            expanded = expand_query(derivation.value)
            queries.append(derivation.value)
            for record in expanded.expansions:
                key = (record.concept, record.source.casefold())
                if key not in expansion_keys:
                    expansion_keys.add(key)
                    expansions.append(record)
        unique_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            key = query.casefold()
            if key not in seen_queries:
                seen_queries.add(key)
                unique_queries.append(query)
        if len(unique_queries) > 64 or len(expansions) > 64:
            raise DiscoveryEvidenceError(
                "derived evidence exceeds query expansion resource limits"
            )
        return ExpandedQuery(
            original=derivations[0].value,
            source_queries=tuple(item.value for item in derivations),
            queries=tuple(unique_queries),
            expansions=tuple(expansions),
        )

    @staticmethod
    def _merge_candidates(
        candidates: Iterable[WorkflowCandidate],
    ) -> dict[str, WorkflowCandidate]:
        grouped: dict[str, list[WorkflowCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.workflow_id, []).append(candidate)
        return {
            workflow_id: min(values, key=lambda item: item.model_dump_json())
            for workflow_id, values in sorted(grouped.items())
        }

    @staticmethod
    def _unique_artifacts(artifacts: Sequence[Artifact]) -> tuple[Artifact, ...]:
        unique: list[Artifact] = []
        seen: set[tuple[str, str]] = set()
        for artifact in artifacts:
            key = (artifact.kind, artifact.path)
            if key not in seen:
                seen.add(key)
                unique.append(artifact)
        return tuple(unique)

    def _new_experiment_id(self, operation: str) -> str:
        timestamp = self.clock().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = re.sub(r"[^a-zA-Z0-9._-]", "-", self.id_factory()).strip(".-_")
        suffix = suffix[:32] or "generated"
        return f"{operation}-{timestamp}-{suffix}".lower()
