import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from mp_comfyui_lab.security import SafetyError, sanitize_for_persistence


EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,95}$")
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
        experiment_root = (self._store_root / experiment_id).resolve()
        try:
            experiment_root.relative_to(self._store_root)
        except ValueError as error:
            raise SafetyError("experiment root escapes store root") from error
        return experiment_root

    def create(self, experiment_id: str) -> Path:
        experiment_root = self.root(experiment_id)
        experiment_root.mkdir(parents=True, exist_ok=True)
        for directory in EXPERIMENT_DIRECTORIES:
            (experiment_root / directory).mkdir(exist_ok=True)
        return experiment_root

    def write_json(
        self,
        experiment_id: str,
        relative_path: str | Path,
        value: Any,
    ) -> Path:
        destination = self._artifact_path(experiment_id, relative_path)
        serialized = json.dumps(
            sanitize_for_persistence(value),
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

        experiment_root = self.root(experiment_id)
        resolved = (experiment_root / path).resolve()
        try:
            resolved.relative_to(experiment_root)
        except ValueError as error:
            raise SafetyError("artifact path escapes experiment root") from error
        return resolved

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
