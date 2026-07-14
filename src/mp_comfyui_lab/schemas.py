from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from pydantic import BaseModel

from mp_comfyui_lab.experiments.models import ExperimentBrief, ExperimentPlan
from mp_comfyui_lab.experiments.store import advisory_file_lock
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


def export_schemas(
    output: str | Path,
    *,
    _replace: Callable[[Path, Path], None] = os.replace,
) -> tuple[Path, ...]:
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with advisory_file_lock(_schema_lock_path(output_root)):
        rendered = render_schemas()
        paths = tuple(output_root / filename for filename in rendered)
        transaction_root = Path(
            tempfile.mkdtemp(prefix=".schemas-transaction-", dir=output_root)
        )
        stage_root = transaction_root / "stage"
        backup_root = transaction_root / "backup"
        stage_root.mkdir()
        backup_root.mkdir()
        existed: dict[str, bool] = {}
        try:
            for filename, document in rendered.items():
                stage_path = stage_root / filename
                _write_durable_text(stage_path, document)
                if stage_path.read_bytes() != document.encode("utf-8"):
                    raise OSError(f"staged schema validation failed: {filename}")
                json.loads(stage_path.read_text(encoding="utf-8"))
            _fsync_directory(stage_root)

            for path in paths:
                if path.exists() and not path.is_file():
                    raise OSError(f"schema destination is not a file: {path.name}")
                existed[path.name] = path.is_file()
                if existed[path.name]:
                    _copy_durable(path, backup_root / path.name)
            _fsync_directory(backup_root)

            try:
                for path in paths:
                    _replace(stage_root / path.name, path)
                _fsync_directory(output_root)
            except BaseException:
                _restore_schema_set(
                    paths=paths,
                    backup_root=backup_root,
                    existed=existed,
                )
                _fsync_directory(output_root)
                raise
            return paths
        finally:
            shutil.rmtree(transaction_root)
            _fsync_directory(output_root)


def check_schemas(output: str | Path) -> tuple[Path, ...]:
    output_root = Path(output).resolve()
    with advisory_file_lock(_schema_lock_path(output_root)):
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


def _schema_lock_path(output_root: Path) -> Path:
    import hashlib

    key = os.path.normcase(str(output_root)).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return Path(tempfile.gettempdir()) / "mp-comfyui-lab-schema-locks" / f"{digest}.lock"


def _write_durable_text(path: Path, document: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_durable(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _restore_schema_set(
    *,
    paths: tuple[Path, ...],
    backup_root: Path,
    existed: dict[str, bool],
) -> None:
    for path in paths:
        if existed[path.name]:
            os.replace(backup_root / path.name, path)
        else:
            path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
