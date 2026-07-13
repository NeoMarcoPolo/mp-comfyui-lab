import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from enum import IntEnum
from io import StringIO
from typing import NoReturn, Sequence

from mp_comfyui_lab import __version__
from mp_comfyui_lab.cli.output import emit_error, emit_success


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mp-comfy-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version")
    return parser


def main(argv: Sequence[str] | None = None) -> ExitCode:
    arguments = list(sys.argv[1:] if argv is None else argv)

    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            parsed = _build_parser().parse_args(arguments)
    except SystemExit:
        command = arguments[0] if arguments else None
        emit_error(command, "usage", "invalid command line")
        return ExitCode.USAGE

    if parsed.command == "version":
        emit_success("version", {"version": __version__})
        return ExitCode.OK

    emit_error(parsed.command, "usage", "invalid command line")
    return ExitCode.USAGE


def entrypoint() -> NoReturn:
    raise SystemExit(int(main()))
