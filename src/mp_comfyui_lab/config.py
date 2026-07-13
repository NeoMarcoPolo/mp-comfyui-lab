from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import yaml

from mp_comfyui_lab.security import SafetyError


def _resolve_from(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiments_root: Path
    vaults: dict[str, Path] = Field(default_factory=dict)
    _fixture_base: Path = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        base = Path(context["base_dir"] if context else Path.cwd()).resolve()
        self._fixture_base = base
        self.experiments_root = _resolve_from(base, self.experiments_root)
        self.vaults = {
            alias: _resolve_from(base, root) for alias, root in self.vaults.items()
        }

    @classmethod
    def load(cls, repository_root: str | Path | None = None) -> Self:
        base = Path.cwd() if repository_root is None else Path(repository_root)
        base = base.resolve()
        config_path = base / "config.local.yaml"
        if not config_path.is_file():
            return cls.model_validate(
                {"experiments_root": base / "experiments", "vaults": {}},
                context={"base_dir": base},
            )

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls.model_validate(data, context={"base_dir": base})

    def resolve_fixture(self, fixture: str | Path) -> Path:
        if isinstance(fixture, str) and fixture.startswith("vault://"):
            return self._resolve_vault_fixture(fixture)

        path = Path(fixture).expanduser()
        resolved = _resolve_from(self._fixture_base, path)
        return self._require_fixture(resolved)

    def _resolve_vault_fixture(self, fixture: str) -> Path:
        alias, separator, relative_value = fixture.removeprefix("vault://").partition(
            "/"
        )
        if alias not in self.vaults:
            raise SafetyError(f"unknown vault alias: {alias or '<empty>'}")
        if not separator or not relative_value:
            raise SafetyError(f"vault URI is missing path component: {fixture}")

        vault_root = self.vaults[alias].resolve()
        relative_path = Path(relative_value)
        resolved = (vault_root / relative_path).resolve()
        try:
            resolved.relative_to(vault_root)
        except ValueError as error:
            raise SafetyError(f"fixture escapes vault root: {fixture}") from error
        if ".." in relative_path.parts:
            raise SafetyError(f"vault fixture path contains traversal: {fixture}")
        return self._require_fixture(resolved)

    @staticmethod
    def _require_fixture(path: Path) -> Path:
        if not path.exists():
            raise SafetyError(f"fixture does not exist: {path}")
        return path
