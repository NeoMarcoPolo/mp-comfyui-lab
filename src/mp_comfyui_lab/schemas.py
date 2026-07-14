from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Callable

from pydantic import BaseModel

from mp_comfyui_lab.comparison.paths import path_component_is_link_or_junction
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


class SchemaPublicationError(OSError):
    """Reject a schema path that cannot be used without following links."""


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
    output_root = _absolute_without_links(output)
    _validate_output_directory_path(output_root, allow_missing=True)
    with advisory_file_lock(_schema_lock_path(output_root)):
        _validate_output_directory_path(output_root, allow_missing=True)
        rendered = render_schemas()
        paths = tuple(output_root / filename for filename in rendered)
        _inspect_schema_targets(paths)
        output_root.mkdir(parents=True, exist_ok=True)
        _validate_output_directory_path(output_root, allow_missing=False)
        existed = _inspect_schema_targets(paths)
        transaction_root = Path(
            tempfile.mkdtemp(prefix=".schemas-transaction-", dir=output_root)
        )
        try:
            stage_root = transaction_root / "stage"
            backup_root = transaction_root / "backup"
            stage_root.mkdir()
            backup_root.mkdir()
            for filename, document in rendered.items():
                stage_path = stage_root / filename
                _write_durable_text(stage_path, document)
                if stage_path.read_bytes() != document.encode("utf-8"):
                    raise OSError(f"staged schema validation failed: {filename}")
                json.loads(stage_path.read_text(encoding="utf-8"))
            _fsync_directory(stage_root)

            for path in paths:
                if existed[path.name] is not None:
                    _copy_durable(path, backup_root / path.name)
            _fsync_directory(backup_root)
            _validate_output_directory_path(output_root, allow_missing=False)
            _require_unchanged_targets(paths, existed)

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
    output_root = _absolute_without_links(output)
    _validate_output_directory_path(output_root, allow_missing=True)
    with advisory_file_lock(_schema_lock_path(output_root)):
        _validate_output_directory_path(output_root, allow_missing=True)
        expected = render_schemas()
        paths = tuple(output_root / filename for filename in expected)
        states = _inspect_schema_targets(paths)
        missing = tuple(path for path in paths if states[path.name] is None)
        stale = tuple(
            path
            for path in paths
            if states[path.name] is not None
            and _read_regular_bytes(path) != expected[path.name].encode("utf-8")
        )
        if missing or stale:
            raise SchemaCheckError(missing=missing, stale=stale)
        return paths


def _schema_lock_path(output_root: Path) -> Path:
    import hashlib

    key = os.path.normcase(str(output_root)).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return Path(tempfile.gettempdir()) / "mp-comfyui-lab-schema-locks" / f"{digest}.lock"


def _absolute_without_links(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path)))


def _validate_output_directory_path(
    output_root: Path,
    *,
    allow_missing: bool,
) -> None:
    root_exists = False
    for component in (*reversed(output_root.parents), output_root):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise SchemaPublicationError(
                f"schema output path cannot be inspected safely: {component}"
            ) from error
        if path_component_is_link_or_junction(component, metadata):
            raise SchemaPublicationError(
                "schema output path contains a link, junction, or reparse "
                f"point: {component}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise SchemaPublicationError(
                f"schema output path component is not a directory: {component}"
            )
        if component == output_root:
            root_exists = True
    if not allow_missing and not root_exists:
        raise SchemaPublicationError(
            f"schema output directory was not created safely: {output_root}"
        )


def _inspect_schema_targets(
    paths: tuple[Path, ...],
) -> dict[str, os.stat_result | None]:
    states: dict[str, os.stat_result | None] = {}
    first_error: SchemaPublicationError | None = None
    first_cause: OSError | None = None
    for path in paths:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            states[path.name] = None
            continue
        except OSError as error:
            states[path.name] = None
            if first_error is None:
                first_error = SchemaPublicationError(
                    f"schema target cannot be inspected safely: {path.name}"
                )
                first_cause = error
            continue
        states[path.name] = metadata
        if path_component_is_link_or_junction(path, metadata):
            if first_error is None:
                first_error = SchemaPublicationError(
                    "schema target is a link, junction, or reparse point: "
                    f"{path.name}"
                )
        elif not stat.S_ISREG(metadata.st_mode) and first_error is None:
            first_error = SchemaPublicationError(
                f"schema target must be a regular file: {path.name}"
            )
    if first_error is not None:
        if first_cause is not None:
            raise first_error from first_cause
        raise first_error
    return states


def _open_regular_descriptor(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SchemaPublicationError(
            f"schema target cannot be opened safely: {path.name}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        metadata = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or path_component_is_link_or_junction(path, metadata)
            or not os.path.samestat(opened, metadata)
        ):
            raise SchemaPublicationError(
                f"schema target changed during safe read: {path.name}"
            )
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _verify_open_regular_descriptor(
    path: Path,
    descriptor: int,
    before: os.stat_result,
) -> None:
    after = os.fstat(descriptor)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SchemaPublicationError(
            f"schema target changed during safe read: {path.name}"
        ) from error
    if (
        _stat_snapshot(before) != _stat_snapshot(after)
        or path_component_is_link_or_junction(path, metadata)
        or not os.path.samestat(after, metadata)
    ):
        raise SchemaPublicationError(
            f"schema target changed during safe read: {path.name}"
        )


def _read_regular_bytes(path: Path) -> bytes:
    descriptor, before = _open_regular_descriptor(path)
    try:
        chunks = []
        while chunk := os.read(descriptor, 65_536):
            chunks.append(chunk)
        _verify_open_regular_descriptor(path, descriptor, before)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_unchanged_targets(
    paths: tuple[Path, ...],
    expected: dict[str, os.stat_result | None],
) -> None:
    current = _inspect_schema_targets(paths)
    for path in paths:
        before = expected[path.name]
        after = current[path.name]
        if before is None and after is None:
            continue
        if (
            before is None
            or after is None
            or _stat_snapshot(before) != _stat_snapshot(after)
        ):
            raise SchemaPublicationError(
                f"schema target changed before publication: {path.name}"
            )


def _stat_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _write_durable_text(path: Path, document: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_durable(source: Path, destination: Path) -> None:
    descriptor, before = _open_regular_descriptor(source)
    try:
        with destination.open("wb") as destination_handle:
            while chunk := os.read(descriptor, 65_536):
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        _verify_open_regular_descriptor(source, descriptor, before)
    finally:
        os.close(descriptor)


def _restore_schema_set(
    *,
    paths: tuple[Path, ...],
    backup_root: Path,
    existed: dict[str, os.stat_result | None],
) -> None:
    for path in paths:
        if existed[path.name] is not None:
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
