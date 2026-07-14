import argparse
import yaml
from ipaddress import ip_address
from pathlib import Path
import socket
import sys
import time
from enum import IntEnum
from typing import Any, NoReturn, Protocol, Sequence
from urllib.parse import urljoin, urlsplit

from mp_runninghub_api.errors import RunningHubApiError
from pydantic import ValidationError
import requests

from mp_comfyui_lab import __version__
from mp_comfyui_lab.cli.output import emit_error, emit_success
from mp_comfyui_lab.config import LabConfig, LabConfigError
from mp_comfyui_lab.discovery.query import QueryExpansionConfigError, expand_query
from mp_comfyui_lab.discovery.ranking import RankedWorkflow, RankingConfigError
from mp_comfyui_lab.discovery.service import (
    DiscoveryExecutionError,
    DiscoveryIncompleteError,
    DiscoveryNotFoundError,
    DiscoveryResult,
    DiscoveryService,
    InspectResult,
)
from mp_comfyui_lab.discovery.urls import UnsupportedSourceError, parse_runninghub_url
from mp_comfyui_lab.evidence.build import build_evidence_graph
from mp_comfyui_lab.evidence.models import CurrentEvidenceInput
from mp_comfyui_lab.executors.base import (
    Executor,
    ExecutorError,
    SearchRequest,
    TransientOperationError,
    WorkflowRef,
)
from mp_comfyui_lab.executors.runninghub import (
    DEFAULT_OUTPUT_HOST_SUFFIXES,
    RunningHubExecutor,
)
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.experiments.engine import (
    MAX_OUTPUT_BYTES,
    ApprovalError,
    ExperimentEngine,
)
from mp_comfyui_lab.experiments.models import (
    Candidate,
    DimensionName,
    ExperimentBrief,
    PlanningSource,
    RunRecord,
    RunState,
    SourceBundle,
)
from mp_comfyui_lab.experiments.planning import (
    PlanningValidationError,
    build_plan,
    evidence_graph_for_bundle,
    load_experiment_brief,
)
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash
from mp_comfyui_lab.graph.normalize import normalize_graph
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations import compile_observed_candidates
from mp_comfyui_lab.security import (
    SafetyError,
    sanitize_for_persistence,
    sanitize_untrusted_diagnostic,
)


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    NOT_FOUND = 3
    PROVIDER = 4
    APPROVAL = 5
    EXECUTION = 6
    INCOMPLETE = 7
    SELECTION = 8
    SAFETY = 9


class HelpRequested(Exception):
    pass


class UsageError(Exception):
    pass


class PlanningBundleLoader(Protocol):
    """Read-only source discovery seam for the dry-run planning command."""

    def load_source_bundle(self, brief: ExperimentBrief) -> SourceBundle: ...


class ReadOnlyPlanningBundleLoader:
    """Public discovery adapter for `plan`; it cannot submit or poll work."""

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    @staticmethod
    def _engagement(candidate) -> int:
        return sum(
            value or 0
            for value in candidate.engagement.model_dump(mode="json").values()
        )

    def _source_for_ref(self, ref: WorkflowRef) -> tuple[PlanningSource, CurrentEvidenceInput]:
        fetched = self._executor.fetch_workflow(ref)
        # Inspection is intentional evidence collection, not provider validation.
        capability_report = self._executor.inspect_capabilities(fetched)
        graph = sanitize_for_persistence(thaw_json(fetched.graph))
        source = PlanningSource(
            workflow_id=fetched.detail.candidate.workflow_id,
            canonical_url=fetched.detail.candidate.canonical_url,
            raw_graph=graph,
            normalized_graph=normalize_graph(graph),
            source_hash=source_graph_hash(graph),
            capability_hash=capability_hash(graph),
            evidence_ids=(fetched.detail.candidate.workflow_id,),
            engagement=self._engagement(fetched.detail.candidate),
        )
        return source, CurrentEvidenceInput(
            source_id=source.workflow_id,
            detail=fetched.detail,
            normalized_graph=source.normalized_graph,
            capability_report=capability_report,
            source_hash=source.source_hash,
            capability_hash=source.capability_hash,
        )

    def load_source_bundle(self, brief: ExperimentBrief) -> SourceBundle:
        refs: dict[str, WorkflowRef] = {}
        first_url_id: str | None = None
        for url in brief.sources.urls:
            parsed = parse_runninghub_url(url)
            first_url_id = first_url_id or parsed.workflow_id
            refs[parsed.workflow_id] = WorkflowRef(
                executor=self._executor.name,
                workflow_id=parsed.workflow_id,
                canonical_url=parsed.canonical_url,
            )
        for query in brief.sources.queries:
            for expanded in expand_query(query).queries:
                page = self._executor.search(SearchRequest(query=expanded))
                for candidate in page.records:
                    if candidate.executor != self._executor.name:
                        continue
                    refs.setdefault(
                        candidate.workflow_id,
                        WorkflowRef(
                            executor=candidate.executor,
                            workflow_id=candidate.workflow_id,
                            canonical_url=candidate.canonical_url,
                        ),
                    )
        records = tuple(
            self._source_for_ref(ref)
            for _, ref in sorted(refs.items())
        )
        sources = tuple(record[0] for record in records)
        if not sources:
            raise PlanningValidationError("public discovery returned no workflows")
        baseline = next((item for item in sources if item.workflow_id == first_url_id), None)
        if baseline is None:
            baseline = min(sources, key=lambda item: (-item.engagement, item.source_hash, item.workflow_id))
        compilation = compile_observed_candidates(brief, baseline, sources)
        return SourceBundle(
            sources=sources,
            baseline_workflow_id=first_url_id,
            candidates=compilation.candidates,
            compiler_warnings=compilation.warnings,
            evidence_graph=build_evidence_graph(
                tuple(record[1] for record in records),
                prior_experiment_roots=(),
            ),
        )

    @staticmethod
    def _compile_candidates(
        brief: ExperimentBrief,
        baseline: PlanningSource,
        sources: tuple[PlanningSource, ...],
    ) -> tuple[Candidate, ...]:
        roles = {
            DimensionName.CHECKPOINT: SemanticRole.CHECKPOINT,
            DimensionName.SAMPLER: SemanticRole.SAMPLER,
            DimensionName.SCHEDULER: SemanticRole.SCHEDULER,
            DimensionName.ENCODER: SemanticRole.ENCODER,
            DimensionName.PROMPT_GRAMMAR: SemanticRole.PROMPT_SOURCE,
        }
        candidates: list[Candidate] = []
        for donor in sources:
            if donor.workflow_id == baseline.workflow_id:
                continue
            for dimension, role in roles.items():
                if brief.policy_for(dimension).value != "approved":
                    continue
                source_bindings = baseline.normalized_graph.bindings(role)
                donor_bindings = donor.normalized_graph.bindings(role)
                if len(source_bindings) != 1 or len(donor_bindings) != 1:
                    continue
                source_binding, donor_binding = source_bindings[0], donor_bindings[0]
                if source_binding.field_name != donor_binding.field_name or source_binding.field_name == "$class_type":
                    continue
                value = thaw_json(donor_binding.value)
                if value == thaw_json(source_binding.value):
                    continue
                candidate_id = f"{dimension.value}-{donor.workflow_id}"[:96]
                candidates.append(Candidate(
                    candidate_id=candidate_id,
                    hypothesis=dimension,
                    capability_signature=f"{dimension.value}:{_canonical_candidate_value(value)}",
                    confidence="observed",
                    evidence_ids=(donor.workflow_id,),
                    semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=role, binding_index=0, field_name=source_binding.field_name, value=value),)),
                    engagement=donor.engagement,
                    provenance_rank=1,
                ))
        return tuple(candidates)


def _canonical_candidate_value(value) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()[:16]


class StructuredArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None) -> NoReturn:
        raise HelpRequested(self.format_help())

    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _build_parser() -> StructuredArgumentParser:
    parser = StructuredArgumentParser(prog="mp-comfy-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version")

    search = subparsers.add_parser(
        "search",
        usage=(
            "%(prog)s QUERY [--sort SORT] [--page N] [--size N] "
            "[--include-graphs]"
        ),
    )
    search.add_argument("query", metavar="QUERY")
    _add_discovery_options(search, include_sort=True)

    inspect = subparsers.add_parser("inspect", usage="%(prog)s URL")
    inspect.add_argument("url", metavar="URL")

    similar = subparsers.add_parser(
        "similar",
        usage=(
            "%(prog)s URL [--sort SORT] [--page N] [--size N] "
            "[--include-graphs]"
        ),
    )
    similar.add_argument("url", metavar="URL")
    _add_discovery_options(similar, include_sort=True)

    plan = subparsers.add_parser("plan", usage="%(prog)s MANIFEST")
    plan.add_argument("manifest", metavar="MANIFEST")

    run = subparsers.add_parser(
        "run",
        usage=(
            "%(prog)s EXPERIMENT_ID [VARIANT_ID | --rerun VARIANT_ID "
            "--reason TEXT] --approve-plan-hash HASH"
        ),
    )
    run.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    run.add_argument("variant_id", metavar="VARIANT_ID", nargs="?")
    run.add_argument("--rerun", dest="rerun_variant_id", metavar="VARIANT_ID")
    run.add_argument("--reason", metavar="TEXT")
    run.add_argument(
        "--approve-plan-hash",
        metavar="HASH",
        required=True,
    )

    resume = subparsers.add_parser(
        "resume",
        usage="%(prog)s EXPERIMENT_ID RUN_ID",
    )
    resume.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    resume.add_argument("run_id", metavar="RUN_ID")

    status = subparsers.add_parser(
        "status",
        usage="%(prog)s EXPERIMENT_ID RUN_ID",
    )
    status.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    status.add_argument("run_id", metavar="RUN_ID")

    reconcile = subparsers.add_parser(
        "reconcile",
        usage="%(prog)s EXPERIMENT_ID RUN_ID --task-id ID",
    )
    reconcile.add_argument("experiment_id", metavar="EXPERIMENT_ID")
    reconcile.add_argument("run_id", metavar="RUN_ID")
    reconcile.add_argument("--task-id", metavar="ID", required=True)
    return parser


def _validate_run_arguments(parsed: argparse.Namespace) -> None:
    has_variant = parsed.variant_id is not None
    has_rerun = parsed.rerun_variant_id is not None
    if has_variant == has_rerun:
        raise UsageError("run requires exactly one of VARIANT_ID or --rerun VARIANT_ID")
    if not has_rerun and parsed.reason is not None:
        raise UsageError("--reason is valid only with --rerun")
    if has_rerun and (
        not isinstance(parsed.reason, str) or not parsed.reason.strip()
    ):
        raise UsageError("--rerun requires a non-empty --reason")


def _bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer") from None
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _add_discovery_options(
    parser: StructuredArgumentParser,
    *,
    include_sort: bool,
) -> None:
    if include_sort:
        parser.add_argument("--sort")
    parser.add_argument("--page", type=_bounded_integer(1, 1_000_000), default=1)
    parser.add_argument("--size", type=_bounded_integer(1, 100), default=30)
    parser.add_argument("--include-graphs", action="store_true")


def _make_service(
    repository_root: Path,
    executor: Executor | None = None,
) -> DiscoveryService:
    config = LabConfig.load(repository_root)
    resolved_executor = executor if executor is not None else RunningHubExecutor()
    return DiscoveryService(
        executor=resolved_executor,
        config=config,
        store=ExperimentStore(config.experiments_root),
    )


def _make_experiment_engine(
    repository_root: Path,
    *,
    executor: Executor | None = None,
) -> ExperimentEngine:
    config = LabConfig.load(repository_root)
    return ExperimentEngine(
        config=config,
        executor_factory=(
            (lambda: executor)
            if executor is not None
            else (lambda: RunningHubExecutor())
        ),
        downloader=_download_runninghub_output,
        sleep=time.sleep,
    )


def _download_runninghub_output(
    url: str,
    target: Path,
    *,
    session: Any | None = None,
    resolver: Any = socket.getaddrinfo,
    max_redirects: int = 3,
    max_bytes: int = MAX_OUTPUT_BYTES,
) -> Path:
    try:
        return _download_runninghub_output_http(
            url,
            target,
            session=session,
            resolver=resolver,
            max_redirects=max_redirects,
            max_bytes=max_bytes,
        )
    except RunningHubApiError as error:
        raise TransientOperationError(
            "RunningHub output download transport failed"
        ) from error


def _download_runninghub_output_http(
    url: str,
    target: Path,
    *,
    session: Any | None = None,
    resolver: Any = socket.getaddrinfo,
    max_redirects: int = 3,
    max_bytes: int = MAX_OUTPUT_BYTES,
) -> Path:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    request_session = requests.Session() if session is None else session
    if hasattr(request_session, "trust_env"):
        request_session.trust_env = False
    current_url = url
    redirects = 0
    while True:
        _validate_runninghub_download_url(
            current_url,
            resolver=resolver,
        )
        if hasattr(request_session, "auth"):
            request_session.auth = None
        cookies = getattr(request_session, "cookies", None)
        if cookies is not None and hasattr(cookies, "clear"):
            cookies.clear()
        session_headers = getattr(request_session, "headers", None)
        if session_headers is not None:
            for header_name in tuple(session_headers):
                normalized_header = str(header_name).casefold().replace("-", "")
                if normalized_header in {
                    "authorization",
                    "proxyauthorization",
                    "cookie",
                    "apikey",
                    "xapikey",
                    "xauthtoken",
                    "accesstoken",
                }:
                    session_headers.pop(header_name, None)
        try:
            response = request_session.get(
                current_url,
                allow_redirects=False,
                stream=True,
                timeout=(10, 60),
                headers={"Accept": "*/*"},
            )
        except (
            requests.RequestException,
            TimeoutError,
            ConnectionError,
        ) as error:
            raise TransientOperationError(
                "RunningHub output download transport failed"
            ) from error

        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("Location")
            response.close()
            if not isinstance(location, str) or not location.strip():
                raise SafetyError("RunningHub output redirect is missing Location")
            if redirects >= max_redirects:
                raise SafetyError("RunningHub output redirect limit exceeded")
            current_url = urljoin(current_url, location)
            redirects += 1
            continue
        if response.status_code != 200:
            status_code = response.status_code
            response.close()
            if status_code >= 500 or status_code == 429:
                raise TransientOperationError(
                    "RunningHub output download transport failed"
                )
            raise SafetyError("RunningHub output download was rejected")

        declared_size = response.headers.get("Content-Length")
        if declared_size is not None:
            if (
                not isinstance(declared_size, str)
                or not declared_size.isdecimal()
            ):
                response.close()
                raise SafetyError("RunningHub output Content-Length is invalid")
            if int(declared_size) > max_bytes:
                response.close()
                raise SafetyError("RunningHub output exceeds byte limit")

        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with target.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=65_536):
                    if chunk:
                        total += len(chunk)
                        if total > max_bytes:
                            raise SafetyError(
                                "RunningHub output exceeds byte limit"
                            )
                        output_file.write(chunk)
            if total == 0:
                raise SafetyError("RunningHub output is empty")
        except SafetyError:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            raise
        except requests.RequestException as error:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            raise TransientOperationError(
                "RunningHub output download transport failed"
            ) from error
        except OSError as error:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            raise SafetyError("RunningHub output could not be written safely") from error
        finally:
            response.close()
        return target


def _validate_runninghub_download_url(
    url: str,
    *,
    resolver: Any = socket.getaddrinfo,
    resolve_dns: bool = True,
) -> None:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, UnicodeError, ValueError):
        raise SafetyError("RunningHub output URL is invalid") from None
    if (
        parsed.scheme.casefold() != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or "%" in hostname
        or "\\" in url
    ):
        raise SafetyError("RunningHub output URL is not allowed")
    try:
        normalized = hostname.encode("idna").decode("ascii").casefold().rstrip(".")
    except UnicodeError:
        raise SafetyError("RunningHub output URL hostname is invalid") from None
    if normalized.isdecimal() or not any(
        normalized == suffix or normalized.endswith(f".{suffix}")
        for suffix in DEFAULT_OUTPUT_HOST_SUFFIXES
    ):
        raise SafetyError("RunningHub output URL hostname is not provider-approved")
    try:
        literal = ip_address(normalized)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise SafetyError("RunningHub output URL resolves to a non-public address")
    if not resolve_dns:
        return
    try:
        addresses = resolver(
            normalized,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise SafetyError("RunningHub output hostname could not be resolved") from error
    if not addresses:
        raise SafetyError("RunningHub output hostname has no addresses")
    for address in addresses:
        try:
            resolved = ip_address(address[4][0])
        except (IndexError, ValueError):
            raise SafetyError("RunningHub output hostname resolution is invalid") from None
        if not resolved.is_global:
            raise SafetyError("RunningHub output URL resolves to a non-public address")


def _run_payload(record: RunRecord) -> dict[str, Any]:
    return {
        "experiment_id": record.experiment_id,
        "run_id": record.run_id,
        "variant_id": record.variant_id,
        "state": record.state.value,
        "provider_task_id": record.provider_task_id,
        "expected_output_count": record.expected_output_count,
        "actual_output_count": record.actual_output_count,
        "error": (
            thaw_json(record.error) if record.error is not None else None
        ),
        "diagnostics": [thaw_json(item) for item in record.diagnostics],
        "outputs": [item.model_dump(mode="json") for item in record.outputs],
    }


def _execution_exit(record: RunRecord) -> ExitCode:
    if record.state is RunState.SUCCESS:
        return ExitCode.OK
    if record.state in {RunState.FAILED, RunState.CANCELLED}:
        return ExitCode.EXECUTION
    return ExitCode.INCOMPLETE


def _ranked_payload(item: RankedWorkflow) -> dict[str, object]:
    return {
        "workflow_id": item.candidate.workflow_id,
        "url": item.candidate.canonical_url,
        "name": item.candidate.name,
        "score": item.score,
        "components": item.components.model_dump(mode="json"),
        "explanation": item.explanation,
        "duplicate_group": item.duplicate_group,
        "source_hash": item.source_hash,
        "capability_hash": item.capability_hash,
        "canonicalization_error": item.canonicalization_error,
    }


def _discovery_payload(result: DiscoveryResult) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "query": result.query_plan.original,
        "queries": list(result.query_plan.queries),
        "expansions": [
            item.model_dump(mode="json") for item in result.query_plan.expansions
        ],
        "derivations": [
            item.model_dump(mode="json") for item in result.derivations
        ],
        "incomplete": result.incomplete,
        "diagnostic_count": result.diagnostic_count,
        "diagnostics_artifact": (
            result.diagnostics_artifact.model_dump(mode="json")
            if result.diagnostics_artifact is not None
            else None
        ),
        "diagnostics": [
            item.model_dump(mode="json") for item in result.diagnostics
        ],
        "results": [_ranked_payload(item) for item in result.results],
    }


def _inspect_payload(result: InspectResult) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "workflow_id": result.detail.candidate.workflow_id,
        "url": result.detail.candidate.canonical_url,
        "name": result.detail.candidate.name,
        "source_hash": result.source_hash,
        "capability_hash": result.capability_hash,
        "roles": list(result.roles),
        "canonicalization_error": result.canonicalization_error,
    }


def _plan_artifacts(
    *,
    store: ExperimentStore,
    brief: ExperimentBrief,
    source_bundle: SourceBundle,
    plan,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Persist a completed plan last, so partial writes never claim a plan."""
    experiment_id = store.create_unique(f"plan-{plan.plan_hash[:16]}")
    brief_document = yaml.safe_dump(
        sanitize_for_persistence(brief.model_dump(mode="json")),
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=True,
    )
    discovery_snapshot = {
        "queries": list(brief.sources.queries),
        "urls": list(brief.sources.urls),
    }
    source_snapshot = {
        "sources": [
            {
                "workflow_id": item.workflow_id,
                "canonical_url": item.canonical_url,
                "source_hash": item.source_hash,
                "capability_hash": item.capability_hash,
                "evidence_ids": list(item.evidence_ids),
                "engagement": item.engagement,
                "raw_graph": thaw_json(item.raw_graph),
                "normalized_graph": item.normalized_graph.model_dump(mode="json"),
            }
            for item in source_bundle.sources
        ],
        "candidates": [item.model_dump(mode="json") for item in source_bundle.candidates],
    }
    for item in source_snapshot["sources"]:
        if source_graph_hash(item["raw_graph"]) != item["source_hash"] or capability_hash(item["raw_graph"]) != item["capability_hash"]:
            raise PlanningValidationError("source snapshot graph hashes do not match validated source")
    evidence_graph = evidence_graph_for_bundle(source_bundle)
    paths = [
        "brief.yaml",
        "search-results.json",
        "sources/source-bundle.json",
        "evidence/graph.json",
        *(f"variants/{variant.variant_id}.json" for variant in plan.variants),
        f"plans/{plan.plan_hash}.json",
    ]
    if len(paths) != len(set(paths)):
        raise PlanningValidationError("planned artifact paths must be unique")
    written = [
        store.write_text_new(experiment_id, "brief.yaml", brief_document),
        store.write_json_new(experiment_id, "search-results.json", discovery_snapshot),
        store.write_sanitized_json_new(experiment_id, "sources/source-bundle.json", source_snapshot),
        store.write_json_new(
            experiment_id,
            "evidence/graph.json",
            evidence_graph.model_dump(mode="json"),
        ),
    ]
    for variant in plan.variants:
        written.append(
            store.write_json_new(
                experiment_id,
                f"variants/{variant.variant_id}.json",
                variant.model_dump(mode="json"),
            )
        )
    # This final artifact is the durable success marker for the experiment.
    written.append(
        store.write_json_new(
            experiment_id,
            f"plans/{plan.plan_hash}.json",
            plan.model_dump(mode="json"),
        )
    )
    return experiment_id, tuple({"kind": "plan", "path": str(path)} for path in written)


def main(
    argv: Sequence[str] | None = None,
    *,
    service: DiscoveryService | None = None,
    planning_loader: PlanningBundleLoader | None = None,
    executor: Executor | None = None,
    experiment_engine: ExperimentEngine | None = None,
    cwd: str | Path | None = None,
) -> ExitCode:
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        parsed = _build_parser().parse_args(arguments)
        if parsed.command == "run":
            _validate_run_arguments(parsed)
    except HelpRequested as error:
        emit_success("help", {"text": str(error)})
        return ExitCode.OK
    except UsageError as error:
        command = arguments[0] if arguments and not arguments[0].startswith("-") else None
        emit_error(command, "usage", str(error))
        return ExitCode.USAGE

    if parsed.command == "version":
        emit_success("version", {"version": __version__})
        return ExitCode.OK

    if parsed.command in {"run", "resume", "status", "reconcile"}:
        try:
            root = Path.cwd() if cwd is None else Path(cwd)
            engine = experiment_engine or _make_experiment_engine(
                root,
                executor=executor,
            )
            if parsed.command == "run":
                if parsed.rerun_variant_id is not None:
                    record = engine.rerun(
                        parsed.experiment_id,
                        parsed.rerun_variant_id,
                        approved_plan_hash=parsed.approve_plan_hash,
                        reason=parsed.reason.strip(),
                    )
                else:
                    record = engine.run(
                        parsed.experiment_id,
                        parsed.variant_id,
                        approved_plan_hash=parsed.approve_plan_hash,
                    )
            elif parsed.command == "resume":
                record = engine.resume(parsed.experiment_id, parsed.run_id)
            elif parsed.command == "reconcile":
                record = engine.reconcile(
                    parsed.experiment_id,
                    parsed.run_id,
                    parsed.task_id,
                )
            else:
                record = engine.status(parsed.experiment_id, parsed.run_id)
            artifacts = [
                {
                    "kind": "run",
                    "path": f"runs/{record.run_id}.json",
                }
            ]
            emit_success(parsed.command, _run_payload(record), artifacts)
            return (
                ExitCode.OK
                if parsed.command == "status"
                else _execution_exit(record)
            )
        except ApprovalError as error:
            emit_error(
                parsed.command,
                "approval",
                sanitize_untrusted_diagnostic(str(error)),
            )
            return ExitCode.APPROVAL
        except SafetyError as error:
            emit_error(
                parsed.command,
                "safety",
                sanitize_untrusted_diagnostic(str(error)),
            )
            return ExitCode.SAFETY
        except Exception as error:
            emit_error(
                parsed.command,
                "execution",
                sanitize_untrusted_diagnostic(str(error)),
            )
            return ExitCode.EXECUTION

    if parsed.command == "plan":
        try:
            root = Path.cwd() if cwd is None else Path(cwd)
            config = LabConfig.load(root)
            brief = load_experiment_brief(parsed.manifest)
            loader = planning_loader or ReadOnlyPlanningBundleLoader(
                executor if executor is not None else RunningHubExecutor()
            )
            source_bundle = loader.load_source_bundle(brief)
            if not isinstance(source_bundle, SourceBundle):
                raise PlanningValidationError(
                    "planning source-bundle loader returned an invalid bundle"
                )
            plan = build_plan(brief, source_bundle, config=config)
            experiment_id, artifacts = _plan_artifacts(
                store=ExperimentStore(config.experiments_root),
                brief=brief,
                source_bundle=source_bundle,
                plan=plan,
            )
            emit_success(
                "plan",
                {
                    "experiment_id": experiment_id,
                    "plan_hash": plan.plan_hash,
                    "capability_coverage": [
                        item.value for item in plan.capability_coverage
                    ],
                    "warnings": list(plan.warnings),
                    "variant_ids": [item.variant_id for item in plan.variants],
                    "artifact_paths": [item["path"] for item in artifacts],
                },
                list(artifacts),
            )
            return ExitCode.OK
        except (ValidationError, LabConfigError, PlanningValidationError) as error:
            emit_error("plan", "usage", sanitize_untrusted_diagnostic(str(error)))
            return ExitCode.USAGE
        except SafetyError as error:
            emit_error("plan", "safety", sanitize_untrusted_diagnostic(str(error)))
            return ExitCode.SAFETY
        except Exception as error:
            emit_error("plan", "execution", sanitize_untrusted_diagnostic(str(error)))
            return ExitCode.EXECUTION

    try:
        discovery = (
            service
            if service is not None
            else _make_service(
                Path.cwd() if cwd is None else Path(cwd),
                executor=executor,
            )
        )
        if parsed.command == "search":
            result = discovery.search(
                parsed.query,
                sort=parsed.sort,
                page=parsed.page,
                size=parsed.size,
                include_graphs=parsed.include_graphs,
            )
            emit_success(
                "search",
                _discovery_payload(result),
                [item.model_dump(mode="json") for item in result.artifacts],
            )
            return ExitCode.OK
        if parsed.command == "inspect":
            inspected = discovery.inspect(parsed.url)
            emit_success(
                "inspect",
                _inspect_payload(inspected),
                [item.model_dump(mode="json") for item in inspected.artifacts],
            )
            return ExitCode.OK
        if parsed.command == "similar":
            similar = discovery.similar(
                parsed.url,
                sort=parsed.sort,
                page=parsed.page,
                size=parsed.size,
                include_graphs=parsed.include_graphs,
            )
            emit_success(
                "similar",
                _discovery_payload(similar),
                [item.model_dump(mode="json") for item in similar.artifacts],
            )
            return ExitCode.OK
    except (
        ValidationError,
        LabConfigError,
        QueryExpansionConfigError,
        RankingConfigError,
    ) as error:
        emit_error(parsed.command, "usage", sanitize_untrusted_diagnostic(str(error)))
        return ExitCode.USAGE
    except DiscoveryIncompleteError as error:
        emit_error(
            parsed.command,
            "incomplete",
            sanitize_untrusted_diagnostic(str(error)),
            [item.model_dump(mode="json") for item in error.artifacts],
        )
        return ExitCode.INCOMPLETE
    except DiscoveryExecutionError as error:
        emit_error(
            parsed.command,
            "execution",
            sanitize_untrusted_diagnostic(str(error)),
            [item.model_dump(mode="json") for item in error.artifacts],
        )
        return ExitCode.EXECUTION
    except DiscoveryNotFoundError as error:
        emit_error(
            parsed.command,
            "not_found",
            sanitize_untrusted_diagnostic(str(error)),
            [item.model_dump(mode="json") for item in error.artifacts],
        )
        return ExitCode.NOT_FOUND
    except (UnsupportedSourceError, SafetyError) as error:
        emit_error(parsed.command, "safety", sanitize_untrusted_diagnostic(str(error)))
        return ExitCode.SAFETY
    except ExecutorError as error:
        emit_error(parsed.command, "provider", sanitize_untrusted_diagnostic(str(error)))
        return ExitCode.PROVIDER
    except Exception as error:
        emit_error(parsed.command, "execution", sanitize_untrusted_diagnostic(str(error)))
        return ExitCode.EXECUTION

    emit_error(parsed.command, "usage", "invalid command line")
    return ExitCode.USAGE


def entrypoint() -> NoReturn:
    raise SystemExit(int(main()))
