from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.discovery.models import WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.discovery.query import (
    ExpandedQuery,
    ExpansionRecord,
    expand_query,
)
from mp_comfyui_lab.discovery.ranking import (
    Artifact,
    GraphEvidence,
    RankedWorkflow,
    load_default_ranking_weights,
    rank_candidates,
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
from mp_comfyui_lab.security import sanitize_untrusted_diagnostic


DEFAULT_SORTS = ("", "REPUTATION", "HOTTEST", "NEWEST")
MAX_DERIVATIONS = 16
MAX_DERIVATION_LENGTH = 128

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


class DiscoveryResult(_FrozenModel):
    experiment_id: BoundedText
    query_plan: ExpandedQuery
    results: tuple[RankedWorkflow, ...]
    derivations: tuple[DerivationEvidence, ...] = Field(
        default=(),
        max_length=MAX_DERIVATIONS,
    )
    artifacts: tuple[Artifact, ...] = ()


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
        experiment_id = self._new_experiment_id("search")
        self.store.create(experiment_id)
        return self._discover(
            experiment_id,
            plan,
            sort=sort,
            page=page,
            size=size,
            include_graphs=include_graphs,
            derivations=(),
        )

    def inspect(self, url: str) -> InspectResult:
        parsed = parse_runninghub_url(url)
        experiment_id = self._new_experiment_id("inspect")
        self.store.create(experiment_id)
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
        experiment_id = self._new_experiment_id("similar")
        self.store.create(experiment_id)
        source = self._inspect_into(
            experiment_id,
            WorkflowRef(
                executor=self.executor.name,
                workflow_id=parsed.workflow_id,
                canonical_url=parsed.canonical_url,
            ),
        )
        derivations = self._derive_searches(source)
        if not derivations:
            raise DiscoveryNotFoundError(
                f"workflow {parsed.workflow_id} has no derivable public capability evidence"
            )
        query_plan = self._derivation_query_plan(derivations)
        discovered = self._discover(
            experiment_id,
            query_plan,
            sort=sort,
            page=page,
            size=size,
            include_graphs=include_graphs,
            derivations=derivations,
            initial_artifacts=source.artifacts,
            initial_enriched={parsed.workflow_id: source},
        )
        return discovered

    def _discover(
        self,
        experiment_id: str,
        query_plan: ExpandedQuery,
        *,
        sort: str | None,
        page: int,
        size: int,
        include_graphs: bool,
        derivations: tuple[DerivationEvidence, ...],
        initial_artifacts: tuple[Artifact, ...] = (),
        initial_enriched: Mapping[str, _Enriched] | None = None,
    ) -> DiscoveryResult:
        sorts = (sort,) if sort is not None else DEFAULT_SORTS
        requests: list[SearchRequest] = []
        pages: list[SearchPage] = []
        seen_requests: set[SearchRequest] = set()
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
                requests.append(request)
                pages.append(self.executor.search(request))

        merged = self._merge_candidates(
            candidate
            for search_page in pages
            for candidate in search_page.records
        )
        details: dict[str, WorkflowDetail] = {}
        evidence: dict[str, GraphEvidence] = {}
        artifacts = list(initial_artifacts)
        enriched_cache = dict(initial_enriched or {})
        for workflow_id in sorted(merged):
            cached = enriched_cache.get(workflow_id)
            if cached is not None:
                details[workflow_id] = cached.detail
                evidence[workflow_id] = cached.evidence
            elif include_graphs:
                enriched = self._inspect_into(
                    experiment_id,
                    WorkflowRef(
                        executor=self.executor.name,
                        workflow_id=workflow_id,
                        canonical_url=merged[workflow_id].canonical_url,
                    ),
                )
                details[workflow_id] = enriched.detail
                evidence[workflow_id] = enriched.evidence
                artifacts.extend(enriched.artifacts)
            else:
                detail_method = getattr(self.executor, "detail", None)
                if not callable(detail_method):
                    raise TypeError("discovery executor must expose public detail()")
                detail = detail_method(workflow_id)
                details[workflow_id] = detail
                artifacts.append(self._persist_detail(experiment_id, detail))

        weights = load_default_ranking_weights(
            self.config.ranking.model_dump(exclude_none=True)
        )
        ranked = rank_candidates(
            query_plan,
            tuple(merged.values()),
            details,
            weights,
            graph_evidence=evidence,
        )
        snapshot_path = self.store.write_json(
            experiment_id,
            "search-results.json",
            {
                "derivations": [item.model_dump(mode="json") for item in derivations],
                "expansions": [
                    item.model_dump(mode="json") for item in query_plan.expansions
                ],
                "provider_requests": [
                    request.model_dump(mode="json") for request in requests
                ],
                "provider_pages": [page.model_dump(mode="json") for page in pages],
                "query": query_plan.original,
                "queries": list(query_plan.queries),
                "source_queries": list(query_plan.source_queries),
                "results": [item.model_dump(mode="json") for item in ranked],
            },
        )
        artifacts.append(Artifact(kind="search", path=str(snapshot_path)))
        return DiscoveryResult(
            experiment_id=experiment_id,
            query_plan=query_plan,
            results=ranked,
            derivations=derivations,
            artifacts=self._unique_artifacts(artifacts),
        )

    def _inspect_into(self, experiment_id: str, ref: WorkflowRef) -> _Enriched:
        fetched = self.executor.fetch_workflow(ref)
        artifacts: list[Artifact] = [self._persist_detail(experiment_id, fetched.detail)]
        graph_path = self.store.write_json(
            experiment_id,
            Path("sources") / ref.workflow_id / "graph.json",
            thaw_json(fetched.graph),
        )
        artifacts.append(Artifact(kind="graph", path=str(graph_path)))

        normalized: NormalizedGraph | None = None
        try:
            normalized = normalize_graph(fetched.graph)
            source_digest = source_graph_hash(fetched.graph)
            capability_digest = capability_hash(fetched.graph)
            roles = tuple(group.role.value for group in normalized.role_bindings)
            evidence = GraphEvidence(
                source_hash=source_digest,
                capability_hash=capability_digest,
                roles=roles,
            )
            normalized_path = self.store.write_json(
                experiment_id,
                Path("sources") / ref.workflow_id / "normalized.json",
                normalized.model_dump(mode="json"),
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
            fetched,
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
        candidates: list[tuple[str, str]] = []
        candidates.extend(
            ("model", self._inventory_search(name))
            for name in report.model_names[:3]
        )
        candidates.extend(("lora", self._inventory_search(name)) for name in report.lora_names[:2])
        candidates.extend(("tag", tag) for tag in source.detail.candidate.tags[:3])
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
            candidates.append(
                (
                    "prompt_grammar",
                    "prompt grammar "
                    f"lines {line_count} commas {comma_count} "
                    f"placeholders {placeholder_count} "
                    f"fanout {fanout_count} conditioning {conditioning_count}",
                )
            )
        if source.normalized is not None:
            candidates.extend(
                ("capability_role", group.role.value)
                for group in source.normalized.role_bindings
            )

        unique: list[DerivationEvidence] = []
        seen: set[tuple[str, str]] = set()
        for kind, raw_value in candidates:
            value = raw_value.strip()[:MAX_DERIVATION_LENGTH]
            if not value:
                continue
            key = (kind, value.casefold())
            if key in seen:
                continue
            seen.add(key)
            unique.append(DerivationEvidence(source=kind, value=value))
            if len(unique) == MAX_DERIVATIONS:
                break
        return tuple(unique)

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
        for derivation in derivations:
            expanded = expand_query(derivation.value)
            queries.extend(expanded.queries)
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
        return ExpandedQuery(
            original=derivations[0].value,
            source_queries=tuple(item.value for item in derivations),
            queries=tuple(unique_queries[:64]),
            expansions=tuple(expansions[:64]),
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
