import argparse
from pathlib import Path
import sys
from enum import IntEnum
from typing import NoReturn, Sequence

from pydantic import ValidationError

from mp_comfyui_lab import __version__
from mp_comfyui_lab.cli.output import emit_error, emit_success
from mp_comfyui_lab.config import LabConfig, LabConfigError
from mp_comfyui_lab.discovery.query import QueryExpansionConfigError
from mp_comfyui_lab.discovery.ranking import RankedWorkflow, RankingConfigError
from mp_comfyui_lab.discovery.service import (
    DiscoveryIncompleteError,
    DiscoveryNotFoundError,
    DiscoveryResult,
    DiscoveryService,
    InspectResult,
)
from mp_comfyui_lab.discovery.urls import UnsupportedSourceError
from mp_comfyui_lab.executors.base import Executor, ExecutorError
from mp_comfyui_lab.executors.runninghub import RunningHubExecutor
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import SafetyError, sanitize_untrusted_diagnostic


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
    return parser


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


def main(
    argv: Sequence[str] | None = None,
    *,
    service: DiscoveryService | None = None,
    executor: Executor | None = None,
    cwd: str | Path | None = None,
) -> ExitCode:
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        parsed = _build_parser().parse_args(arguments)
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
    except DiscoveryNotFoundError as error:
        emit_error(parsed.command, "not_found", sanitize_untrusted_diagnostic(str(error)))
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
