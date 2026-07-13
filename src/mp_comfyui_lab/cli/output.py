import json
import sys
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    document = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write(f"{document}\n")


def emit_success(
    command: str,
    data: Any,
    artifacts: list[Any] | None = None,
) -> None:
    _emit(
        {
            "schema_version": "1",
            "ok": True,
            "command": command,
            "data": data,
            "artifacts": artifacts or [],
        }
    )


def emit_error(command: str | None, kind: str, message: str) -> None:
    _emit(
        {
            "schema_version": "1",
            "ok": False,
            "command": command,
            "error": {"kind": kind, "message": message},
            "artifacts": [],
        }
    )


def progress(message: str) -> None:
    sys.stderr.write(f"{message}\n")
