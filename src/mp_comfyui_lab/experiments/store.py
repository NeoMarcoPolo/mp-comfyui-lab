from collections.abc import Iterator
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

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
_JSONL_PATH_LOCKS: dict[str, threading.Lock] = {}
_JSONL_PATH_LOCKS_GUARD = threading.Lock()
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

    def write_json_new(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: Any,
    ) -> Path:
        """Publish a sanitized artifact only when its final name is unused."""
        destination = self._artifact_path(experiment_id, relative_path)
        serialized = json.dumps(
            sanitize_for_persistence(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._atomic_write_new(destination, f"{serialized}\n")
        return destination

    def write_sanitized_json_new(self, experiment_id: str, relative_path: str | Path, value: Any) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        self._atomic_write_new(destination, f"{serialized}\n")
        return destination

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
        if not isinstance(sanitized, dict):
            raise SafetyError("JSONL record must be an object")
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
        with self._jsonl_lock(experiment_id, relative_path, destination):
            flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
            flags |= getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags, 0o600)
            try:
                self._repair_jsonl_tail(descriptor)
                offset = 0
                while offset < len(encoded):
                    written = os.write(descriptor, encoded[offset:])
                    if written <= 0:
                        raise OSError("JSONL append made no progress")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return destination

    def read_jsonl(
        self,
        experiment_id: str,
        relative_path: str | Path,
    ) -> tuple[dict[str, Any], ...]:
        source = self._artifact_path(experiment_id, relative_path)
        if not source.parent.is_dir():
            return ()
        with self._jsonl_lock(experiment_id, relative_path, source):
            flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(source, flags)
            except FileNotFoundError:
                return ()
            try:
                self._repair_jsonl_tail(descriptor)
                size = os.fstat(descriptor).st_size
                content = self._read_descriptor_range(descriptor, 0, size)
            finally:
                os.close(descriptor)

        records: list[dict[str, Any]] = []
        for index, line in enumerate(content.splitlines(), start=1):
            try:
                records.append(self._parse_jsonl_object(line))
            except (UnicodeDecodeError, ValueError) as error:
                raise SafetyError(f"invalid JSONL record {index}: {error}") from error
        return tuple(records)

    def iter_jsonl(
        self,
        experiment_id: str,
        relative_path: str | Path,
    ) -> Iterator[dict[str, Any]]:
        return iter(self.read_jsonl(experiment_id, relative_path))

    @classmethod
    def _repair_jsonl_tail(cls, descriptor: int) -> None:
        """Keep the durable prefix; a crash may lose only its in-flight record."""
        size = os.fstat(descriptor).st_size
        if size == 0:
            return

        repaired = False
        if cls._read_descriptor_range(descriptor, size - 1, 1) != b"\n":
            newline = cls._find_previous_newline(descriptor, size)
            size = newline + 1 if newline >= 0 else 0
            os.ftruncate(descriptor, size)
            repaired = True
        else:
            start, line = cls._last_complete_jsonl_line(descriptor, size)
            try:
                cls._parse_jsonl_object(line)
            except (UnicodeDecodeError, ValueError):
                size = start
                os.ftruncate(descriptor, size)
                repaired = True

        if repaired:
            os.fsync(descriptor)
            if size:
                _, prior_line = cls._last_complete_jsonl_line(descriptor, size)
                try:
                    cls._parse_jsonl_object(prior_line)
                except (UnicodeDecodeError, ValueError) as error:
                    raise SafetyError(
                        "invalid JSONL record precedes recoverable tail"
                    ) from error

    @classmethod
    def _last_complete_jsonl_line(
        cls,
        descriptor: int,
        size: int,
    ) -> tuple[int, bytes]:
        previous_newline = cls._find_previous_newline(descriptor, size - 1)
        start = previous_newline + 1
        return (
            start,
            cls._read_descriptor_range(descriptor, start, size - start - 1),
        )

    @staticmethod
    def _find_previous_newline(descriptor: int, end: int) -> int:
        cursor = end
        while cursor > 0:
            start = max(0, cursor - 8_192)
            chunk = ExperimentStore._read_descriptor_range(
                descriptor,
                start,
                cursor - start,
            )
            index = chunk.rfind(b"\n")
            if index >= 0:
                return start + index
            cursor = start
        return -1

    @staticmethod
    def _read_descriptor_range(
        descriptor: int,
        start: int,
        length: int,
    ) -> bytes:
        os.lseek(descriptor, start, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                raise OSError("unexpected end of JSONL artifact")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _parse_jsonl_object(line: bytes) -> dict[str, Any]:
        if not line:
            raise ValueError("record must not be empty")
        if len(line) + 1 > MAX_JSONL_RECORD_BYTES:
            raise ValueError(
                f"record exceeds {MAX_JSONL_RECORD_BYTES} byte limit"
            )
        parsed = json.loads(
            line.decode("utf-8"),
            parse_constant=ExperimentStore._reject_json_constant,
        )
        if not isinstance(parsed, dict):
            raise ValueError("record must be an object")
        return parsed

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    @contextmanager
    def _jsonl_lock(
        self,
        experiment_id: str,
        relative_path: str | Path,
        artifact_path: Path,
    ) -> Iterator[None]:
        lock_relative_path = self._jsonl_lock_relative_path(relative_path)
        lock_path = self._artifact_path(experiment_id, lock_relative_path)
        process_lock = self._process_jsonl_lock(artifact_path)
        with process_lock:
            lock_path = self._artifact_path(experiment_id, lock_relative_path)
            with self._os_advisory_lock(lock_path):
                self._artifact_path(experiment_id, relative_path)
                yield

    @staticmethod
    def _jsonl_lock_relative_path(relative_path: str | Path) -> Path:
        path = Path(relative_path)
        return path.with_name(f"{path.name}.lock")

    @staticmethod
    def _process_jsonl_lock(artifact_path: Path) -> threading.Lock:
        key = os.path.normcase(str(artifact_path))
        with _JSONL_PATH_LOCKS_GUARD:
            return _JSONL_PATH_LOCKS.setdefault(key, threading.Lock())

    @contextmanager
    def _os_advisory_lock(self, lock_path: Path) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        locked = False
        try:
            self._verify_open_lock_file(lock_path, descriptor)
            if os.fstat(descriptor).st_size == 0:
                os.lseek(descriptor, 0, os.SEEK_SET)
                written = os.write(descriptor, b"\0")
                if written != 1:
                    raise OSError("journal lock initialization made no progress")
                os.fsync(descriptor)
            self._acquire_os_lock(descriptor)
            locked = True
            self._verify_open_lock_file(lock_path, descriptor)
            yield
        finally:
            if locked:
                self._release_os_lock(descriptor)
            os.close(descriptor)

    @classmethod
    def _verify_open_lock_file(cls, lock_path: Path, descriptor: int) -> None:
        cls._reject_existing_links(lock_path.parent, Path(lock_path.name))
        path_stat = os.stat(lock_path, follow_symlinks=False)
        if not os.path.samestat(path_stat, os.fstat(descriptor)):
            raise SafetyError("journal lock file changed while opening")

    @staticmethod
    def _acquire_os_lock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name != "nt":
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return

        retry_errors = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in retry_errors:
                    raise
                time.sleep(0.01)

    @staticmethod
    def _release_os_lock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)

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

    def write_text_new(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: str,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        self._atomic_write_new(destination, value)
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

    @staticmethod
    def _atomic_write_new(destination: Path, value: str) -> None:
        """Hard-link a fully fsynced temp file into place without replacement."""
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
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as temporary_file:
                temporary_file.write(value)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                raise SafetyError("artifact already exists; refusing to overwrite") from error
            except (AttributeError, NotImplementedError, OSError) as error:
                raise SafetyError("safe no-clobber artifact publish is unavailable") from error
            if os.name != "nt":
                directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
