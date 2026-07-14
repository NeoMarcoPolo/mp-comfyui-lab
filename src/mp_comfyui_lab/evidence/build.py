from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import re
import stat
import unicodedata
from typing import Any

from pydantic import ValidationError

from mp_comfyui_lab.discovery.models import ContentClassification
from mp_comfyui_lab.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    ClaimAssessment,
    ConfidenceLabel,
    CurrentEvidenceInput,
    EvidenceDimension,
    EvidenceGraph,
    EvidenceObservation,
    EvidenceSource,
    EvidenceSourceKind,
    EvidenceWarning,
    LoraStrengthRange,
    MutationClaim,
    PRIOR_FINALISTS_SCHEMA_VERSION,
    PRIOR_HANDOFF_SCHEMA_VERSION,
    PriorFinalistsArtifact,
    PriorHandoffArtifact,
    PriorObservationRecord,
    VisibleOutputEvidence,
)
from mp_comfyui_lab.graph.models import (
    NormalizedGraph,
    RoleBinding,
    SemanticRole,
    strength_bucket,
)
from mp_comfyui_lab.json_types import thaw_json


MAX_PRIOR_ROOTS = 128
MAX_PRIOR_FILES = 1_024
MAX_PRIOR_FILE_BYTES = 2 * 1024 * 1024
MAX_PRIOR_JSON_DEPTH = 48
MAX_PRIOR_JSON_NODES = 100_000
MAX_PRIOR_STRING_LENGTH = 8_192


class EvidenceScanError(ValueError):
    pass


class _DuplicateJsonKeyError(ValueError):
    pass


_IDENTITY_EXTENSION = re.compile(
    r"\.(?:safetensors|ckpt|pt|pth|bin|gguf)$",
    re.IGNORECASE,
)
_IDENTITY_SEPARATOR = re.compile(r"[^a-z0-9]+")
_PRECISION_TOKEN = re.compile(
    r"^(?:bf16|fp(?:8|16|32)(?:e\d+m\d+fn)?|int\d+|q\d+|scaled|default)$"
)
_VERSION_TOKEN = re.compile(r"^(?:v?\d+(?:\.\d+)*|\d{4})$")
_PROMPT_PLACEHOLDER = re.compile(r"\{[^{}]+\}|\[[^\[\]]+\]|<[^<>]+>")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_identity(value: str) -> str:
    basename = value.replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()
    basename = _IDENTITY_EXTENSION.sub("", basename)
    return _IDENTITY_SEPARATOR.sub("_", basename).strip("_")


def _asset_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.strip().replace("\\", "/"))
    return unicodedata.normalize("NFC", normalized.casefold())


def _model_family(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    tokens = [
        token
        for token in _canonical_identity(checkpoint).split("_")
        if token and not _PRECISION_TOKEN.fullmatch(token)
    ]
    return "_".join(tokens) or None


def _model_lineage(family: str | None) -> str | None:
    if family is None:
        return None
    tokens = [
        token
        for token in _canonical_identity(family).split("_")
        if token and not _PRECISION_TOKEN.fullmatch(token) and not _VERSION_TOKEN.fullmatch(token)
    ]
    return "_".join(tokens) or None


def _binding_identity(binding: RoleBinding) -> str:
    value = thaw_json(binding.value)
    if isinstance(value, Mapping) and isinstance(value.get("name"), str):
        return value["name"]
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _node_lookup(graph: NormalizedGraph) -> dict[str, Any]:
    return {node.node_id: node for node in graph.nodes}


def _topology_for_role(
    graph: NormalizedGraph,
    role: SemanticRole,
    *,
    include_value: bool,
) -> tuple[str, ...]:
    nodes = _node_lookup(graph)
    records: list[str] = []
    for index, binding in enumerate(graph.bindings(role)):
        node = nodes[binding.node_id]
        base = f"{index}:{node.class_type}.{binding.field_name}"
        if include_value:
            base = f"{base}={_asset_identity(_binding_identity(binding))}"
        records.append(base)
    return tuple(records)


def _loader_topology(graph: NormalizedGraph) -> tuple[str, ...]:
    nodes = _node_lookup(graph)
    records: list[str] = []
    checkpoint_bindings = graph.bindings(SemanticRole.CHECKPOINT)
    for index, binding in enumerate(checkpoint_bindings):
        node = nodes[binding.node_id]
        records.append(f"checkpoint:{index}:{node.class_type}.{binding.field_name}")
    for index, binding in enumerate(graph.bindings(SemanticRole.LORA)):
        node = nodes[binding.node_id]
        identity = _asset_identity(_binding_identity(binding))
        model_inputs = sorted(
            field_name
            for field_name in node.input_links
            if "model" in field_name.casefold()
        )
        records.append(
            f"lora:{index}:{node.class_type}.{binding.field_name}:{identity}:"
            f"inputs={','.join(model_inputs) or 'slot'}"
        )
    return tuple(records)


def _prompt_grammar(value: str) -> str:
    separators = tuple(
        separator
        for separator in ("|", "\n", ",", ";", "::")
        if separator in value
    )
    words = len(value.split())
    if words == 0:
        word_bucket = "empty"
    elif words <= 8:
        word_bucket = "short"
    elif words <= 32:
        word_bucket = "medium"
    else:
        word_bucket = "long"
    return _canonical_json(
        {
            "line_count": value.count("\n") + 1,
            "placeholder_count": len(_PROMPT_PLACEHOLDER.findall(value)),
            "separators": separators,
            "word_bucket": word_bucket,
        }
    )


def _prompt_grammars(graph: NormalizedGraph) -> tuple[str, ...]:
    nodes = _node_lookup(graph)
    records = []
    for index, binding in enumerate(graph.bindings(SemanticRole.PROMPT_SOURCE)):
        value = thaw_json(binding.value)
        if not isinstance(value, str):
            continue
        node = nodes[binding.node_id]
        records.append(
            f"{index}:{node.class_type}.{binding.field_name}:{_prompt_grammar(value)}"
        )
    return tuple(records)


def _fanout(graph: NormalizedGraph) -> tuple[str, ...]:
    nodes = _node_lookup(graph)
    records = []
    for index, binding in enumerate(graph.bindings(SemanticRole.PROMPT_FANOUT)):
        node = nodes[binding.node_id]
        records.append(
            f"{index}:{node.class_type}.{binding.field_name}:"
            f"{_canonical_json(thaw_json(binding.value))}"
        )
    return tuple(records)


def _reference_routing(graph: NormalizedGraph) -> tuple[str, ...]:
    nodes = _node_lookup(graph)
    consumers: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for consumer in graph.nodes:
        for field_name, link in consumer.input_links.items():
            consumers[link[0]].append((consumer.node_id, field_name, link[1]))

    records: list[str] = []
    for index, binding in enumerate(graph.bindings(SemanticRole.REFERENCE_IMAGE)):
        provider = nodes[binding.node_id]
        destinations = sorted(
            (
                nodes[node_id].class_type,
                field_name,
                output_index,
            )
            for node_id, field_name, output_index in consumers.get(binding.node_id, [])
        )
        rendered = ",".join(
            f"{class_type}.{field_name}[{output_index}]"
            for class_type, field_name, output_index in destinations
        )
        records.append(
            f"{index}:{provider.class_type}.{binding.field_name}->{rendered or 'unrouted'}"
        )
    return tuple(records)


def _lora_strengths(graph: NormalizedGraph) -> tuple[LoraStrengthRange, ...]:
    nodes = _node_lookup(graph)
    records: list[LoraStrengthRange] = []
    for binding in graph.bindings(SemanticRole.LORA):
        identity = _binding_identity(binding)
        node = nodes[binding.node_id]
        values = [
            float(value)
            for field_name, value in sorted(node.inputs.items())
            if field_name.casefold().replace("_", "").startswith("strength")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        ]
        buckets: list[str] = []
        for value in values:
            bucket = strength_bucket(value)
            if bucket not in buckets:
                buckets.append(bucket)

        raw_binding = thaw_json(binding.value)
        if isinstance(raw_binding, Mapping):
            for key, value in sorted(raw_binding.items()):
                if "bucket" in key.casefold() and isinstance(value, str):
                    if value not in buckets:
                        buckets.append(value)
        records.append(
            LoraStrengthRange(
                identity=identity,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                buckets=tuple(buckets),
            )
        )
    return tuple(records)


def _first_checkpoint(graph: NormalizedGraph) -> str | None:
    bindings = graph.bindings(SemanticRole.CHECKPOINT)
    if not bindings:
        return None
    return _binding_identity(bindings[0])


def _extract_observation(source: CurrentEvidenceInput) -> EvidenceObservation:
    graph = source.normalized_graph
    checkpoint = _first_checkpoint(graph)
    loras = tuple(
        _binding_identity(binding)
        for binding in graph.bindings(SemanticRole.LORA)
    )
    outputs = graph.bindings(SemanticRole.OUTPUT)
    output_types = tuple(_binding_identity(binding) for binding in outputs)
    candidate = source.detail.candidate

    return EvidenceObservation(
        source_id=source.source_id,
        checkpoint=checkpoint,
        model_family=_model_family(checkpoint),
        loras=loras,
        lora_strengths=_lora_strengths(graph),
        loader_topology=_loader_topology(graph),
        encoder_topology=(
            *_topology_for_role(
                graph,
                SemanticRole.ENCODER,
                include_value=True,
            ),
            *_topology_for_role(
                graph,
                SemanticRole.CONDITIONING,
                include_value=False,
            ),
        ),
        sampler_topology=_topology_for_role(
            graph,
            SemanticRole.SAMPLER,
            include_value=True,
        ),
        scheduler_topology=_topology_for_role(
            graph,
            SemanticRole.SCHEDULER,
            include_value=True,
        ),
        vae_topology=_topology_for_role(
            graph,
            SemanticRole.VAE,
            include_value=True,
        ),
        prompt_grammar=_prompt_grammars(graph),
        reference_count=len(graph.bindings(SemanticRole.REFERENCE_IMAGE)),
        reference_routing=_reference_routing(graph),
        fanout=_fanout(graph),
        output_node_count=len(outputs),
        output_node_types=output_types,
        visible_output=VisibleOutputEvidence(
            cover_urls=tuple(cover.url for cover in candidate.covers),
            preview_url=(
                source.detail.preview.url if source.detail.preview is not None else None
            ),
            output_role_node_ids=tuple(binding.node_id for binding in outputs),
        ),
        engagement=candidate.engagement,
        content_classification=candidate.content_classification,
    )


@dataclass(frozen=True, slots=True)
class _EvidenceRecord:
    source: EvidenceSource
    observation: EvidenceObservation
    payload_fingerprint: str


def _identities_for_source(source: EvidenceSource) -> tuple[tuple[str, str], ...]:
    identities = [("source_id", source.source_id)]
    if source.source_hash is not None:
        identities.append(("source_hash", source.source_hash))
    if source.capability_hash is not None:
        identities.append(("capability_hash", source.capability_hash))
    return tuple(identities)


def _record_payload_fingerprint(
    source: EvidenceSource,
    observation: EvidenceObservation,
) -> str:
    del source
    observation_payload = observation.model_dump(mode="json")
    observation_payload.pop("source_id", None)
    return hashlib.sha256(
        _canonical_json(observation_payload).encode("utf-8")
    ).hexdigest()


def _current_record(source: CurrentEvidenceInput) -> _EvidenceRecord:
    provenance = EvidenceSource(
        source_id=source.source_id,
        kind=source.source_kind,
        canonical_url=source.detail.candidate.canonical_url,
        source_hash=source.source_hash,
        capability_hash=source.capability_hash,
    )
    observation = _extract_observation(source)
    return _EvidenceRecord(
        source=provenance,
        observation=observation,
        payload_fingerprint=_record_payload_fingerprint(provenance, observation),
    )


def _warning(code: str, reason: str, path: str | None = None) -> EvidenceWarning:
    return EvidenceWarning(code=code, reason=reason, path=path)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "outside-root"
    rendered = relative.as_posix()
    if len(rendered) > 500:
        return f".../{path.name[:128]}"
    return rendered


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _has_reparse_component(path: Path, root: Path) -> bool:
    current = path
    while True:
        if _is_reparse_point(current):
            return True
        if current == root:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _safe_under_root(path: Path, root: Path) -> bool:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError:
        return False
    return resolved_path.is_relative_to(resolved_root) and not _has_reparse_component(
        path, root
    )


def _directory_children(
    directory: Path,
    root: Path,
    warnings: list[EvidenceWarning],
) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    relative = _safe_relative(directory, root)
    if not _safe_under_root(directory, root):
        warnings.append(
            _warning("unsafe_path", "reparse point or path escape rejected", relative)
        )
        return ()
    try:
        return tuple(
            sorted(
                (child for child in directory.iterdir() if child.is_dir()),
                key=lambda path: path.name,
            )
        )
    except OSError:
        warnings.append(
            _warning("unsafe_path", "could not inspect allowed directory", relative)
        )
        return ()


def _experiment_directories(
    root: Path,
    warnings: list[EvidenceWarning],
) -> tuple[Path, ...]:
    candidates: list[Path] = [root]
    if root.name.casefold() == "experiments":
        candidates.extend(_directory_children(root, root, warnings))
    else:
        experiments = root / "experiments"
        if experiments.exists():
            candidates.extend(_directory_children(experiments, root, warnings))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        key = os.path.normcase(str(candidate.absolute()))
        unique[key] = candidate
    return tuple(sorted(unique.values(), key=lambda path: str(path).casefold()))


def _artifact_candidates(
    root: Path,
    experiment: Path,
    warnings: list[EvidenceWarning],
) -> tuple[tuple[Path, EvidenceSourceKind], ...]:
    if not experiment.exists() or not _safe_under_root(experiment, root):
        if experiment.exists():
            warnings.append(
                _warning(
                    "unsafe_path",
                    "reparse point or path escape rejected",
                    _safe_relative(experiment, root),
                )
            )
        return ()

    candidates: list[tuple[Path, EvidenceSourceKind]] = [
        (experiment / "evidence" / "graph.json", EvidenceSourceKind.PRIOR_EVIDENCE_GRAPH),
        (experiment / "finalists.json", EvidenceSourceKind.PRIOR_FINALISTS),
    ]
    export = experiment / "export"
    for export_child in _directory_children(export, root, warnings):
        if not _safe_under_root(export_child, root):
            warnings.append(
                _warning(
                    "unsafe_path",
                    "reparse point or path escape rejected",
                    _safe_relative(export_child, root),
                )
            )
            continue
        candidates.append(
            (export_child / "handoff.json", EvidenceSourceKind.PRIOR_HANDOFF)
        )
    return tuple(candidates)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate JSON object key")
        result[key] = value
    return result


def _validate_json_bounds(value: Any) -> None:
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > MAX_PRIOR_JSON_NODES:
            raise ValueError("JSON node limit exceeded")
        if depth > MAX_PRIOR_JSON_DEPTH:
            raise ValueError("JSON depth limit exceeded")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, float):
            if not isfinite(item):
                raise ValueError("JSON numbers must be finite")
            continue
        if isinstance(item, str):
            if len(item) > MAX_PRIOR_STRING_LENGTH:
                raise ValueError("JSON string limit exceeded")
            continue
        if isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
            continue
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str) or len(key) > MAX_PRIOR_STRING_LENGTH:
                    raise ValueError("JSON object key limit exceeded")
                pending.append((nested, depth + 1))
            continue
        raise ValueError("unsupported JSON value")


def _artifact_open_flags() -> int:
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, name, 0)
    return flags


def _descriptor_matches_path(descriptor_info: os.stat_result, path: Path) -> bool:
    try:
        path_info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(descriptor_info, "st_file_attributes", 0) & reparse_flag:
        return False
    if getattr(path_info, "st_file_attributes", 0) & reparse_flag:
        return False
    descriptor_identity = (descriptor_info.st_dev, descriptor_info.st_ino)
    path_identity = (path_info.st_dev, path_info.st_ino)
    if all(descriptor_identity) and all(path_identity):
        return descriptor_identity == path_identity
    return True


def _read_regular_artifact(
    path: Path,
    root: Path,
    relative: str,
    warnings: list[EvidenceWarning],
) -> bytes | None:
    try:
        descriptor = os.open(path, _artifact_open_flags())
    except OSError:
        warnings.append(
            _warning(
                "unsafe_path",
                "artifact is not a safe regular file",
                relative,
            )
        )
        return None

    try:
        descriptor_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_info.st_mode)
            or not _descriptor_matches_path(descriptor_info, path)
            or not _safe_under_root(path, root)
        ):
            warnings.append(
                _warning(
                    "unsafe_path",
                    "artifact is not a safe regular file",
                    relative,
                )
            )
            return None
        if descriptor_info.st_size > MAX_PRIOR_FILE_BYTES:
            warnings.append(
                _warning(
                    "oversized_artifact",
                    f"artifact exceeds {MAX_PRIOR_FILE_BYTES} byte limit",
                    relative,
                )
            )
            return None

        chunks: list[bytes] = []
        remaining = MAX_PRIOR_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError:
        warnings.append(
            _warning("malformed_artifact", "artifact could not be read", relative)
        )
        return None
    finally:
        os.close(descriptor)

    if len(raw) > MAX_PRIOR_FILE_BYTES:
        warnings.append(
            _warning(
                "oversized_artifact",
                f"artifact exceeds {MAX_PRIOR_FILE_BYTES} byte limit",
                relative,
            )
        )
        return None
    return raw


def _prior_records_from_graph(
    graph: EvidenceGraph,
    *,
    kind: EvidenceSourceKind,
    artifact_path: str,
    content_sha256: str,
) -> tuple[_EvidenceRecord, ...]:
    sources = {source.source_id: source for source in graph.sources}
    records: list[_EvidenceRecord] = []
    for observation in graph.observations:
        prior = sources[observation.source_id]
        source = EvidenceSource(
            source_id=observation.source_id,
            kind=kind,
            canonical_url=prior.canonical_url,
            artifact_path=artifact_path,
            source_hash=prior.source_hash,
            capability_hash=prior.capability_hash,
            content_sha256=content_sha256,
        )
        records.append(
            _EvidenceRecord(
                source=source,
                observation=observation,
                payload_fingerprint=_record_payload_fingerprint(source, observation),
            )
        )
    return tuple(records)


def _prior_records_from_observations(
    records_with_provenance: Sequence[PriorObservationRecord],
    *,
    kind: EvidenceSourceKind,
    artifact_path: str,
    content_sha256: str,
) -> tuple[_EvidenceRecord, ...]:
    records = []
    for record in records_with_provenance:
        observation = record.observation
        provenance = record.source
        source = EvidenceSource(
            source_id=observation.source_id,
            kind=kind,
            canonical_url=provenance.canonical_url,
            artifact_path=artifact_path,
            source_hash=provenance.source_hash,
            capability_hash=provenance.capability_hash,
            content_sha256=content_sha256,
        )
        records.append(
            _EvidenceRecord(
                source=source,
                observation=observation,
                payload_fingerprint=_record_payload_fingerprint(source, observation),
            )
        )
    return tuple(records)


def _read_prior_artifact(
    path: Path,
    root: Path,
    kind: EvidenceSourceKind,
    warnings: list[EvidenceWarning],
) -> tuple[_EvidenceRecord, ...]:
    relative = _safe_relative(path, root)
    if not _safe_under_root(path, root):
        warnings.append(
            _warning("unsafe_path", "reparse point or path escape rejected", relative)
        )
        return ()
    raw = _read_regular_artifact(path, root, relative, warnings)
    if raw is None:
        return ()

    try:
        text = raw.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError:
        warnings.append(
            _warning("malformed_artifact", "artifact is not valid JSON", relative)
        )
        return ()
    except _DuplicateJsonKeyError:
        warnings.append(
            _warning(
                "malformed_artifact",
                "artifact contains a duplicate JSON object key",
                relative,
            )
        )
        return ()
    except (UnicodeError, ValueError, RecursionError):
        warnings.append(
            _warning("unsafe_json", "artifact is not bounded finite JSON", relative)
        )
        return ()
    try:
        _validate_json_bounds(payload)
    except (ValueError, RecursionError):
        warnings.append(
            _warning("unsafe_json", "artifact is not bounded finite JSON", relative)
        )
        return ()
    if not isinstance(payload, dict):
        warnings.append(
            _warning("malformed_artifact", "artifact root must be an object", relative)
        )
        return ()
    expected_schema_version = {
        EvidenceSourceKind.PRIOR_EVIDENCE_GRAPH: EVIDENCE_SCHEMA_VERSION,
        EvidenceSourceKind.PRIOR_FINALISTS: PRIOR_FINALISTS_SCHEMA_VERSION,
        EvidenceSourceKind.PRIOR_HANDOFF: PRIOR_HANDOFF_SCHEMA_VERSION,
    }[kind]
    if payload.get("schema_version") != expected_schema_version:
        warnings.append(
            _warning(
                "unknown_schema",
                f"artifact schema_version must equal {expected_schema_version}",
                relative,
            )
        )
        return ()

    content_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        if kind is EvidenceSourceKind.PRIOR_EVIDENCE_GRAPH:
            graph = EvidenceGraph.model_validate_json(raw, strict=True)
            return _prior_records_from_graph(
                graph,
                kind=kind,
                artifact_path=relative,
                content_sha256=content_sha256,
            )
        if kind is EvidenceSourceKind.PRIOR_FINALISTS:
            finalists = PriorFinalistsArtifact.model_validate_json(raw, strict=True)
            return _prior_records_from_observations(
                finalists.finalists,
                kind=kind,
                artifact_path=relative,
                content_sha256=content_sha256,
            )
        handoff = PriorHandoffArtifact.model_validate_json(raw, strict=True)
        return _prior_records_from_observations(
            (handoff.selected,),
            kind=kind,
            artifact_path=relative,
            content_sha256=content_sha256,
        )
    except (ValidationError, ValueError, RecursionError):
        warnings.append(
            _warning(
                "malformed_artifact",
                "artifact does not match its strict known schema",
                relative,
            )
        )
        return ()


def _scan_prior_artifacts(
    roots: Sequence[Path],
) -> tuple[tuple[_EvidenceRecord, ...], tuple[EvidenceWarning, ...]]:
    if len(roots) > MAX_PRIOR_ROOTS:
        raise EvidenceScanError(f"prior experiment root limit is {MAX_PRIOR_ROOTS}")
    normalized_roots: dict[str, Path] = {}
    for value in roots:
        if not isinstance(value, Path):
            raise TypeError("prior_experiment_roots must contain Path values")
        key = os.path.normcase(str(value.absolute()))
        normalized_roots[key] = value

    records: list[_EvidenceRecord] = []
    warnings: list[EvidenceWarning] = []
    file_count = 0
    for root in sorted(normalized_roots.values(), key=lambda path: str(path).casefold()):
        if not root.exists():
            warnings.append(
                _warning("unsafe_path", "caller-provided root does not exist", root.name)
            )
            continue
        if not root.is_dir() or _is_reparse_point(root):
            warnings.append(
                _warning("unsafe_path", "caller-provided root is unsafe", root.name)
            )
            continue
        for experiment in _experiment_directories(root, warnings):
            for path, kind in _artifact_candidates(root, experiment, warnings):
                if not path.exists() and not path.is_symlink():
                    continue
                file_count += 1
                if file_count > MAX_PRIOR_FILES:
                    raise EvidenceScanError(
                        f"prior artifact file limit is {MAX_PRIOR_FILES}"
                    )
                records.extend(_read_prior_artifact(path, root, kind, warnings))
    return tuple(records), tuple(warnings)


def _deduplicate_records(
    records: Iterable[_EvidenceRecord],
    warnings: Iterable[EvidenceWarning],
) -> EvidenceGraph:
    ordered_records = tuple(
        sorted(
            records,
            key=lambda item: (
                item.source.source_id,
                item.payload_fingerprint,
                item.source.kind.value,
                item.source.artifact_path or "",
                item.source.canonical_url or "",
                item.source.source_hash or "",
                item.source.capability_hash or "",
            ),
        )
    )
    parents = list(range(len(ordered_records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, higher = sorted((left_root, right_root))
        parents[higher] = lower

    identity_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(ordered_records):
        for identity in _identities_for_source(record.source):
            owner = identity_owner.setdefault(identity, index)
            union(index, owner)

    grouped: dict[int, list[_EvidenceRecord]] = defaultdict(list)
    for index, record in enumerate(ordered_records):
        grouped[find(index)].append(record)

    selected: list[_EvidenceRecord] = []
    all_warnings = list(warnings)
    for component_index in sorted(grouped):
        candidates = grouped[component_index]
        winner = candidates[0]
        fingerprints = {candidate.payload_fingerprint for candidate in candidates}
        source_hashes = sorted(
            {
                candidate.source.source_hash
                for candidate in candidates
                if candidate.source.source_hash is not None
            }
        )
        capability_hashes = sorted(
            {
                candidate.source.capability_hash
                for candidate in candidates
                if candidate.source.capability_hash is not None
            }
        )
        conflict_fields = []
        if len(fingerprints) > 1:
            conflict_fields.append("observation payload")
        if len(source_hashes) > 1:
            conflict_fields.append("source_hash")
        if len(capability_hashes) > 1:
            conflict_fields.append("capability_hash")
        if conflict_fields:
            all_warnings.append(
                _warning(
                    "conflicting_duplicate",
                    "duplicate identity component has conflicting payloads "
                    f"({', '.join(conflict_fields)}); canonical source_id and "
                    "payload fingerprint order selected deterministically",
                    winner.source.artifact_path,
                )
            )
        canonical_id = min(candidate.source.source_id for candidate in candidates)
        source_payload = winner.source.model_dump()
        source_payload.update(
            {
                "source_id": canonical_id,
                "source_hash": source_hashes[0] if source_hashes else None,
                "capability_hash": (
                    capability_hashes[0] if capability_hashes else None
                ),
            }
        )
        reconciled_source = EvidenceSource(**source_payload)
        reconciled_observation = winner.observation.model_copy(
            update={"source_id": canonical_id}
        )
        winner = _EvidenceRecord(
            source=reconciled_source,
            observation=reconciled_observation,
            payload_fingerprint=winner.payload_fingerprint,
        )
        selected.append(winner)

    return EvidenceGraph(
        sources=tuple(item.source for item in selected),
        observations=tuple(item.observation for item in selected),
        warnings=tuple(all_warnings),
    )


def build_evidence_graph(
    current_sources: Sequence[CurrentEvidenceInput],
    *,
    prior_experiment_roots: Sequence[Path],
) -> EvidenceGraph:
    if not isinstance(current_sources, (list, tuple)):
        raise TypeError("current_sources must be an explicit sequence")
    if not isinstance(prior_experiment_roots, (list, tuple)):
        raise TypeError("prior_experiment_roots must be an explicit sequence")

    current_records = tuple(_current_record(source) for source in current_sources)
    prior_records, warnings = _scan_prior_artifacts(prior_experiment_roots)
    return _deduplicate_records((*current_records, *prior_records), warnings)


def _dimension_value_equal(
    dimension: EvidenceDimension,
    claim_value: Any,
    observation_value: Any,
) -> bool:
    if dimension is EvidenceDimension.CHECKPOINT:
        return _asset_identity(claim_value) == _asset_identity(observation_value)
    if dimension is EvidenceDimension.MODEL_FAMILY:
        return _canonical_identity(claim_value) == _canonical_identity(observation_value)
    if dimension is EvidenceDimension.LORAS:
        return tuple(_asset_identity(value) for value in claim_value) == tuple(
            _asset_identity(value) for value in observation_value
        )
    if dimension is EvidenceDimension.LORA_STRENGTHS:
        claim_records = tuple(
            (
                _asset_identity(record.identity),
                record.minimum,
                record.maximum,
                record.buckets,
            )
            for record in claim_value
        )
        observation_records = tuple(
            (
                _asset_identity(record.identity),
                record.minimum,
                record.maximum,
                record.buckets,
            )
            for record in observation_value
        )
        return claim_records == observation_records
    return claim_value == observation_value


@dataclass(frozen=True, slots=True)
class _ObservationAssessment:
    observation: EvidenceObservation
    label: ConfidenceLabel
    matched: tuple[EvidenceDimension, ...]
    differing: tuple[EvidenceDimension, ...]
    reason: str


def _family_support(
    claim: MutationClaim,
    observation: EvidenceObservation,
) -> tuple[bool, bool]:
    claim_family = claim.model_family or _model_family(claim.checkpoint)
    observation_family = observation.model_family or _model_family(observation.checkpoint)
    if claim_family is None or observation_family is None:
        return (False, False)
    exact = _canonical_identity(claim_family) == _canonical_identity(observation_family)
    claim_lineage = _model_lineage(claim_family)
    observation_lineage = _model_lineage(observation_family)
    partial = (
        claim_lineage is not None
        and observation_lineage is not None
        and claim_lineage == observation_lineage
    )
    return (exact, partial)


def _assess_observation(
    claim: MutationClaim,
    observation: EvidenceObservation,
) -> _ObservationAssessment:
    relevant = tuple(
        dimension
        for dimension in EvidenceDimension
        if getattr(claim, dimension.value) is not None
    )
    matched = tuple(
        dimension
        for dimension in relevant
        if _dimension_value_equal(
            dimension,
            getattr(claim, dimension.value),
            getattr(observation, dimension.value),
        )
    )
    differing = tuple(dimension for dimension in relevant if dimension not in matched)
    exact_family, partial_family = _family_support(claim, observation)
    has_topology = bool(
        observation.checkpoint
        and observation.loader_topology
        and observation.encoder_topology
        and observation.output_node_types
    )
    exact_relevant = bool(relevant) and not differing
    if exact_relevant and has_topology and observation.visible_output.has_visible_output:
        return _ObservationAssessment(
            observation,
            ConfidenceLabel.OBSERVED,
            matched,
            differing,
            "exact relevant combination and topology with visible output evidence",
        )

    topology_dimensions = {
        EvidenceDimension.LORAS,
        EvidenceDimension.LOADER_TOPOLOGY,
        EvidenceDimension.ENCODER_TOPOLOGY,
    }
    topology_matches = not any(
        dimension in topology_dimensions for dimension in differing
    )
    substantive_matching_topology = bool(
        claim.loader_topology
        and observation.loader_topology
        and claim.encoder_topology
        and observation.encoder_topology
        and claim.loras
        and observation.loras
        and topology_matches
    )
    declared = set(claim.changed_dimensions)
    if (
        exact_family
        and substantive_matching_topology
        and len(differing) == 1
        and set(differing) == declared
    ):
        return _ObservationAssessment(
            observation,
            ConfidenceLabel.STRONGLY_INFERRED,
            matched,
            differing,
            "same model family and matching loader/encoder/LoRA topology with one "
            "declared changed dimension",
        )

    if exact_family or partial_family:
        reason = (
            "exact combination lacks visible output evidence"
            if exact_relevant and not observation.visible_output.has_visible_output
            else "partial model-family evidence has meaningful topology or dimension differences"
        )
        return _ObservationAssessment(
            observation,
            ConfidenceLabel.WEAKLY_INFERRED,
            matched,
            differing,
            reason,
        )

    return _ObservationAssessment(
        observation,
        ConfidenceLabel.UNEXPLORED,
        matched,
        differing,
        "no matching model-family support",
    )


_LABEL_PRECEDENCE = {
    ConfidenceLabel.OBSERVED: 3,
    ConfidenceLabel.STRONGLY_INFERRED: 2,
    ConfidenceLabel.WEAKLY_INFERRED: 1,
    ConfidenceLabel.UNEXPLORED: 0,
}


def assess_claim(claim: MutationClaim, graph: EvidenceGraph) -> ClaimAssessment:
    graph_warnings = tuple(sorted({warning.reason for warning in graph.warnings}))
    assessments = tuple(
        _assess_observation(claim, observation)
        for observation in graph.observations
    )
    if not assessments:
        return ClaimAssessment(
            claim=claim,
            label=ConfidenceLabel.UNEXPLORED,
            explanation="No matching support; speculative claim retained.",
            warnings=graph_warnings,
        )

    best_label = max(assessments, key=lambda item: _LABEL_PRECEDENCE[item.label]).label
    if best_label is ConfidenceLabel.UNEXPLORED:
        sources = {source.source_id: source for source in graph.sources}
        explanations = []
        warnings = list(graph_warnings)
        for item in sorted(assessments, key=lambda value: value.observation.source_id):
            source = sources[item.observation.source_id]
            matches = ", ".join(dimension.value for dimension in item.matched) or "none"
            differences = (
                ", ".join(dimension.value for dimension in item.differing) or "none"
            )
            explanations.append(
                f"{item.observation.source_id} ({source.kind.value}) matched [{matches}]; "
                f"differed [{differences}]; {item.reason}."
            )
            if item.observation.content_classification is not ContentClassification.SFW:
                warnings.append(
                    f"{item.observation.source_id} content classification: "
                    f"{item.observation.content_classification.value}"
                )
        return ClaimAssessment(
            claim=claim,
            label=best_label,
            explanation=(
                " ".join(explanations)
                + " No matching support; speculative claim retained."
            ),
            warnings=tuple(sorted(set(warnings))),
        )

    best = tuple(
        sorted(
            (item for item in assessments if item.label is best_label),
            key=lambda item: item.observation.source_id,
        )
    )
    representative = best[0]
    supporting = tuple(
        sorted(
            (
                item
                for item in assessments
                if item.label is not ConfidenceLabel.UNEXPLORED
            ),
            key=lambda item: item.observation.source_id,
        )
    )
    sources = {source.source_id: source for source in graph.sources}
    explanations = []
    warnings = list(graph_warnings)
    for item in supporting:
        source = sources[item.observation.source_id]
        matches = ", ".join(dimension.value for dimension in item.matched) or "none"
        differences = (
            ", ".join(dimension.value for dimension in item.differing) or "none"
        )
        explanations.append(
            f"{item.observation.source_id} ({source.kind.value}, {item.label.value}) "
            f"matched [{matches}]; "
            f"differed [{differences}]; {item.reason}."
        )
        if item.observation.content_classification is not ContentClassification.SFW:
            warnings.append(
                f"{item.observation.source_id} content classification: "
                f"{item.observation.content_classification.value}"
            )

    return ClaimAssessment(
        claim=claim,
        label=best_label,
        supporting_source_ids=tuple(
            item.observation.source_id for item in supporting
        ),
        matched_dimensions=representative.matched,
        differing_dimensions=representative.differing,
        explanation=" ".join(explanations),
        warnings=tuple(sorted(set(warnings))),
    )
