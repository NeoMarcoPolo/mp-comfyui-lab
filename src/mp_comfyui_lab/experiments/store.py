import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

from mp_comfyui_lab.security import (
    SafetyError,
    safe_filename,
    sanitize_for_persistence,
    sanitize_untrusted_diagnostic,
)


EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
MAX_EXPERIMENT_ID_LENGTH = 96
MAX_JSONL_RECORD_BYTES = 131_072
WINDOWS_MOUNT_POINT_REPARSE_TAG = 0xA0000003
EXPERIMENT_DIRECTORIES = (
    "sources",
    "evidence",
    "plans",
    "variants",
    "runs",
    "outputs",
    "comparisons",
    "export",
)


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self._store_root = Path(root).resolve()
        self._append_locks: dict[Path, threading.Lock] = {}
        self._append_locks_guard = threading.Lock()

    def root(self, experiment_id: str) -> Path:
        self._validate_experiment_id(experiment_id)
        experiment_path = self._store_root / experiment_id
        self._reject_existing_links(self._store_root, Path(experiment_id))
        experiment_root = experiment_path.resolve()
        try:
            experiment_root.relative_to(self._store_root)
        except ValueError as error:
            raise SafetyError("experiment root escapes store root") from error
        return experiment_root

    def create(self, experiment_id: str) -> Path:
        experiment_root = self.root(experiment_id)
        experiment_root.mkdir(parents=True, exist_ok=True)
        self._create_layout(experiment_root)
        return experiment_root

    def create_unique(self, base_id: str) -> str:
        self._validate_experiment_id(base_id)
        self._store_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            suffix = "" if attempt == 1 else f"-{attempt}"
            prefix_length = MAX_EXPERIMENT_ID_LENGTH - len(suffix)
            if prefix_length < 3:
                raise SafetyError("experiment collision suffix exceeds safe ID bounds")
            prefix = base_id[:prefix_length].rstrip(". ")
            candidate = f"{prefix}{suffix}"
            self._validate_experiment_id(candidate)
            experiment_root = self._store_root / candidate
            self._reject_existing_links(self._store_root, Path(candidate))
            try:
                experiment_root.mkdir(exist_ok=False)
            except FileExistsError:
                attempt += 1
                continue
            self._create_layout(experiment_root)
            return candidate

    def _create_layout(self, experiment_root: Path) -> None:
        for directory in EXPERIMENT_DIRECTORIES:
            self._reject_existing_links(experiment_root, Path(directory))
            (experiment_root / directory).mkdir(exist_ok=True)

    def write_json(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: Any,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        return self._write_json_value(destination, sanitize_for_persistence(value))

    def write_sanitized_json(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: Any,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        return self._write_json_value(destination, value)

    def append_jsonl(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: Any,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        sanitized = self._sanitize_jsonl_value(sanitize_for_persistence(value))
        serialized = json.dumps(
            sanitized,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        encoded = f"{serialized}\n".encode("utf-8")
        if len(encoded) > MAX_JSONL_RECORD_BYTES:
            raise SafetyError(
                f"JSONL record exceeds {MAX_JSONL_RECORD_BYTES} byte limit"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = self._artifact_path(experiment_id, relative_path)
        with self._append_lock(destination):
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags, 0o600)
            try:
                start = os.lseek(descriptor, 0, os.SEEK_END)
                try:
                    written = os.write(descriptor, encoded)
                    if written != len(encoded):
                        raise OSError("incomplete JSONL append")
                    os.fsync(descriptor)
                except BaseException:
                    os.ftruncate(descriptor, start)
                    os.fsync(descriptor)
                    raise
            finally:
                os.close(descriptor)
        return destination

    def _append_lock(self, destination: Path) -> threading.Lock:
        with self._append_locks_guard:
            return self._append_locks.setdefault(destination, threading.Lock())

    @classmethod
    def _sanitize_jsonl_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._sanitize_jsonl_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._sanitize_jsonl_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._sanitize_jsonl_value(item) for item in value)
        if isinstance(value, str):
            return sanitize_untrusted_diagnostic(value)
        return value

    def _write_json_value(self, destination: Path, value: Any) -> Path:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._atomic_write(destination, f"{serialized}\n")
        return destination

    def write_text(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: str,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        self._atomic_write(destination, value)
        return destination

    def read_json(self, experiment_id: str, relative_path: str | Path) -> Any:
        source = self._artifact_path(experiment_id, relative_path)
        return json.loads(source.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_experiment_id(experiment_id: str) -> None:
        if not isinstance(experiment_id, str) or not EXPERIMENT_ID_PATTERN.fullmatch(
            experiment_id
        ):
            raise SafetyError(f"invalid experiment ID: {experiment_id!r}")
        try:
            safe_filename(experiment_id)
        except SafetyError as error:
            raise SafetyError(f"invalid experiment ID: {experiment_id!r}") from error

    def _artifact_path(
        self,
        experiment_id: str,
        relative_path: str | Path,
    ) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise SafetyError("artifact path must be relative")
        if not path.parts or str(path) in {"", "."}:
            raise SafetyError("artifact path must name a file")
        if ".." in path.parts:
            raise SafetyError("artifact path contains traversal")
        for component in path.parts:
            try:
                safe_filename(component)
            except SafetyError as error:
                raise SafetyError(
                    f"unsafe artifact path component: {component!r}"
                ) from error

        experiment_root = self.root(experiment_id)
        self._reject_existing_links(experiment_root, path)
        resolved = (experiment_root / path).resolve()
        try:
            resolved.relative_to(experiment_root)
        except ValueError as error:
            raise SafetyError("artifact path escapes experiment root") from error
        return resolved

    @staticmethod
    def _reject_existing_links(base: Path, relative_path: Path) -> None:
        current = base
        for component in relative_path.parts:
            current /= component
            if current.is_symlink() or ExperimentStore._is_junction(current):
                raise SafetyError(
                    f"path component is a symlink or junction: {current}"
                )

    @staticmethod
    def _is_junction(path: Path) -> bool:
        path_is_junction = getattr(path, "is_junction", None)
        if callable(path_is_junction) and path_is_junction():
            return True

        os_path_is_junction = getattr(os.path, "isjunction", None)
        if callable(os_path_is_junction) and os_path_is_junction(path):
            return True

        try:
            return (
                getattr(path.lstat(), "st_reparse_tag", None)
                == WINDOWS_MOUNT_POINT_REPARSE_TAG
            )
        except OSError:
            return False

    @staticmethod
    def _atomic_write(destination: Path, value: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                text=True,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as temporary_file:
                temporary_file.write(value)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
