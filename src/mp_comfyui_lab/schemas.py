from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from pydantic import BaseModel

from mp_comfyui_lab.experiments.models import ExperimentBrief, ExperimentPlan
from mp_comfyui_lab.export.bundle import HandoffManifest


_SCHEMA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("brief.schema.json", ExperimentBrief),
    ("handoff.schema.json", HandoffManifest),
    ("plan.schema.json", ExperimentPlan),
)


class SchemaCheckError(ValueError):
    def __init__(
        self,
        *,
        missing: tuple[Path, ...] = (),
        stale: tuple[Path, ...] = (),
    ) -> None:
        self.missing = missing
        self.stale = stale
        details = []
        if missing:
            details.append(
                "missing: " + ", ".join(path.name for path in missing)
            )
        if stale:
            details.append(
                "stale: " + ", ".join(path.name for path in stale)
            )
        super().__init__("schema check failed; " + "; ".join(details))


def render_schemas() -> dict[str, str]:
    return {
        filename: json.dumps(
            model.model_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
        for filename, model in _SCHEMA_MODELS
    }


def export_schemas(output: str | Path) -> tuple[Path, ...]:
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, document in render_schemas().items():
        path = output_root / filename
        _atomic_write_text(path, document)
        paths.append(path)
    return tuple(paths)


def check_schemas(output: str | Path) -> tuple[Path, ...]:
    output_root = Path(output).resolve()
    expected = render_schemas()
    paths = tuple(output_root / filename for filename in expected)
    missing = tuple(path for path in paths if not path.is_file())
    stale = tuple(
        path
        for path in paths
        if path.is_file()
        and path.read_bytes() != expected[path.name].encode("utf-8")
    )
    if missing or stale:
        raise SchemaCheckError(missing=missing, stale=stale)
    return paths


def _atomic_write_text(path: Path, document: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
