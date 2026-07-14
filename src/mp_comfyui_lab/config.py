from pathlib import Path
from math import isfinite
import re
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr
import yaml

from mp_comfyui_lab.security import SafetyError


class LabConfigError(ValueError):
    pass


_PLAIN_PATH_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _format_yaml_path(path: tuple[str | int, ...]) -> str:
    if not path:
        return "<root>"
    result = ""
    for segment in path:
        if isinstance(segment, int):
            result += f"[{segment}]"
        elif _PLAIN_PATH_SEGMENT.fullmatch(segment):
            result += ("." if result else "") + segment
        else:
            result += f"[{segment!r}]"
    return result


def _yaml_key_segment(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _reject_duplicate_yaml_keys(
    loader: yaml.SafeLoader,
    node: yaml.Node,
    *,
    source: str,
    error_type: type[ValueError],
    path: tuple[str | int, ...] = (),
    visited: set[int] | None = None,
) -> None:
    checked = visited if visited is not None else set()
    identity = id(node)
    if identity in checked:
        return
    checked.add(identity)

    if isinstance(node, yaml.MappingNode):
        seen: dict[Any, yaml.Node] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            try:
                first = seen.get(key)
            except TypeError as error:
                raise error_type(
                    f"{source}: line {key_node.start_mark.line + 1}, "
                    f"column {key_node.start_mark.column + 1}: "
                    f"unhashable YAML key at {_format_yaml_path(path)}"
                ) from error
            key_path = (*path, _yaml_key_segment(key))
            if first is not None:
                raise error_type(
                    f"{source}: line {key_node.start_mark.line + 1}, "
                    f"column {key_node.start_mark.column + 1}: "
                    f"duplicate YAML key {key!r} at {_format_yaml_path(key_path)} "
                    f"in mapping {_format_yaml_path(path)}; first defined at "
                    f"line {first.start_mark.line + 1}, "
                    f"column {first.start_mark.column + 1}"
                )
            seen[key] = key_node
            _reject_duplicate_yaml_keys(
                loader,
                value_node,
                source=source,
                error_type=error_type,
                path=key_path,
                visited=checked,
            )
    elif isinstance(node, yaml.SequenceNode):
        for index, value_node in enumerate(node.value):
            _reject_duplicate_yaml_keys(
                loader,
                value_node,
                source=source,
                error_type=error_type,
                path=(*path, index),
                visited=checked,
            )


def safe_load_yaml(
    document: str,
    *,
    source: str,
    error_type: type[ValueError],
) -> Any:
    """Safely load YAML while rejecting duplicate keys at every mapping depth."""
    if not isinstance(document, str):
        raise error_type(f"{source}: YAML document must be text")

    loader = yaml.SafeLoader(document)
    try:
        root = loader.get_single_node()
        if root is None:
            return None
        _reject_duplicate_yaml_keys(
            loader,
            root,
            source=source,
            error_type=error_type,
        )
        return loader.construct_document(root)
    except error_type:
        raise
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f"line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else "unknown location"
        )
        problem = getattr(error, "problem", None) or str(error)
        raise error_type(f"{source}: malformed YAML at {location}: {problem}") from error
    finally:
        loader.dispose()


def _resolve_from(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _ranking_weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ranking weight must be a number")
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("ranking weight must be finite and between 0 and 1")
    return parsed


RankingWeight = Annotated[float, BeforeValidator(_ranking_weight)]


class RankingOverrides(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    relevance: RankingWeight | None = None
    engagement: RankingWeight | None = None
    output_evidence: RankingWeight | None = None
    recency: RankingWeight | None = None
    creator_track_record: RankingWeight | None = None
    experimentability: RankingWeight | None = None
    novelty: RankingWeight | None = None
    duplicate_penalty: RankingWeight | None = None


class LabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiments_root: Path
    vaults: dict[str, Path] = Field(default_factory=dict)
    ranking: RankingOverrides = Field(default_factory=RankingOverrides)
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

        data = safe_load_yaml(
            config_path.read_text(encoding="utf-8"),
            source=str(config_path),
            error_type=LabConfigError,
        )
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
