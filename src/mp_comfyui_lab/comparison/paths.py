from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Literal


_READ_CHUNK_BYTES = 1024 * 1024
_WINDOWS_MOUNT_POINT_REPARSE_TAG = 0xA0000003


def path_component_is_link_or_junction(
    path: Path,
    metadata: os.stat_result | None = None,
) -> bool:
    try:
        current = path.lstat() if metadata is None else metadata
    except OSError:
        return True
    if stat.S_ISLNK(current.st_mode):
        return True
    path_is_junction = getattr(path, "is_junction", None)
    if callable(path_is_junction) and path_is_junction():
        return True
    os_path_is_junction = getattr(os.path, "isjunction", None)
    if callable(os_path_is_junction) and os_path_is_junction(path):
        return True
    return (
        getattr(current, "st_reparse_tag", None)
        == _WINDOWS_MOUNT_POINT_REPARSE_TAG
    )


def resolve_stored_artifact(
    experiment_root: Path,
    relative_path: str,
) -> tuple[Path | None, Literal["missing", "unsafe"]]:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None, "unsafe"
    current = experiment_root
    for component in path.parts:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None, "missing"
        except OSError:
            return None, "unsafe"
        if path_component_is_link_or_junction(current, metadata):
            return None, "unsafe"
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(experiment_root)
        metadata = resolved.stat()
    except (OSError, ValueError):
        return None, "unsafe"
    if not stat.S_ISREG(metadata.st_mode):
        return None, "unsafe"
    return resolved, "missing"


def hash_stored_artifact(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise OSError("stored artifact is not a regular file")
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(handle.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OSError("stored artifact changed while reading")
    return digest.hexdigest(), byte_count
