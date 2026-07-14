"""Pure, deterministic, dry-run experiment planning."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import stat
from typing import Any

from pydantic import ValidationError

from mp_comfyui_lab.config import LabConfig, safe_load_yaml
from mp_comfyui_lab.evidence.models import EvidenceGraph, EvidenceSource, EvidenceSourceKind
from mp_comfyui_lab.graph.hashing import capability_hash
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import normalize_graph
from mp_comfyui_lab.graph.patch import (
    PatchTopologyError,
    PatchValidationError,
    materialize_variant,
)
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations import SemanticPatch, SetNodeInput, SetRoleInput
from mp_comfyui_lab.mutations.planner import (
    propose_patch,
    select_candidates,
    validate_candidate_patch,
)
from mp_comfyui_lab.security import SafetyError

from .models import (
    MAX_MANIFEST_BYTES,
    MAX_MANIFEST_DEPTH,
    MAX_MANIFEST_SCALAR_LENGTH,
    MAX_MANIFEST_VALUES,
    Candidate,
    DimensionName,
    ExperimentBrief,
    ExperimentPlan,
    FixtureFingerprint,
    PlannedVariant,
    PlanningSource,
    SourceBundle,
)


MAX_FIXTURE_BYTES = 64 * 1024 * 1024


class PlanningValidationError(ValueError):
    """Raised when a manifest or graph cannot produce a safe deterministic plan."""


def evidence_graph_for_bundle(source_bundle: SourceBundle) -> EvidenceGraph:
    """Return the already-validated Task 7 evidence artifact unchanged."""
    graph = source_bundle.evidence_graph
    if not isinstance(graph, EvidenceGraph):
        raise PlanningValidationError("source bundle requires a strict EvidenceGraph")
    return graph


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise PlanningValidationError(f"canonical JSON failed: {error}") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_manifest_value(value: Any) -> None:
    stack = [(value, "manifest", 0)]
    count = 0
    seen_containers: set[int] = set()
    while stack:
        item, path, depth = stack.pop()
        count += 1
        if count > MAX_MANIFEST_VALUES:
            raise PlanningValidationError("manifest YAML value limit exceeded")
        if depth > MAX_MANIFEST_DEPTH:
            raise PlanningValidationError("manifest YAML depth limit exceeded")
        if item is None or isinstance(item, (bool, int)):
            continue
        if isinstance(item, float):
            if not isfinite(item):
                raise PlanningValidationError("manifest YAML numbers must be finite")
            continue
        if isinstance(item, str):
            if len(item) > MAX_MANIFEST_SCALAR_LENGTH:
                raise PlanningValidationError("manifest YAML scalar exceeds maximum length")
            continue
        if isinstance(item, dict):
            if id(item) in seen_containers:
                raise PlanningValidationError("manifest YAML aliases are not allowed")
            seen_containers.add(id(item))
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise PlanningValidationError(f"{path} YAML keys must be strings")
                if len(key) > MAX_MANIFEST_SCALAR_LENGTH:
                    raise PlanningValidationError("manifest YAML key exceeds maximum length")
                stack.append((nested, f"{path}.{key}", depth + 1))
            continue
        if isinstance(item, list):
            if id(item) in seen_containers:
                raise PlanningValidationError("manifest YAML aliases are not allowed")
            seen_containers.add(id(item))
            for index, nested in enumerate(item):
                stack.append((nested, f"{path}[{index}]", depth + 1))
            continue
        raise PlanningValidationError(
            f"{path} contains unsupported YAML type {type(item).__name__}"
        )


def load_experiment_brief(path: str | Path) -> ExperimentBrief:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise PlanningValidationError(f"cannot read manifest: {error}") from error
    if size > MAX_MANIFEST_BYTES:
        raise PlanningValidationError("manifest exceeds maximum byte size")
    try:
        document = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise PlanningValidationError(f"cannot read manifest: {error}") from error
    data = safe_load_yaml(
        document,
        source=str(source),
        error_type=PlanningValidationError,
    )
    _validate_manifest_value(data)
    try:
        return ExperimentBrief.model_validate(data)
    except ValidationError as error:
        raise PlanningValidationError(str(error)) from error


def _fixture_root_and_parts(config: LabConfig, uri: str) -> tuple[Path, tuple[str, ...]]:
    """Return logical fixture containment without resolving the target pathname."""
    # Keep LabConfig as the authority for configured aliases and fixture roots.
    # Its resolved return value is intentionally not used for opening: resolving
    # a pathname before opening would reintroduce a symlink race.
    config.resolve_fixture(uri)
    if uri.startswith("vault://"):
        alias, separator, relative = uri.removeprefix("vault://").partition("/")
        if not separator or alias not in config.vaults:
            raise SafetyError("invalid vault fixture URI")
        containment = config.vaults[alias]
    else:
        containment = config._fixture_base
        relative = uri
    return containment, tuple(relative.split("/"))


def _is_reparse_point(path: Path) -> bool:
    try:
        return bool(getattr(path.lstat(), "st_reparse_tag", 0))
    except OSError:
        return False


def _safe_open_fixture(config: LabConfig, uri: str) -> int:
    """Open a regular fixture through containment-relative descriptors.

    POSIX walks each component under an already-open directory descriptor.  This
    prevents a parent rename or symlink replacement from redirecting the final
    open.  Windows lacks a portable ``openat`` equivalent in Python, so it uses
    non-following/non-blocking open plus pre/post reparse checks as a best effort.
    """
    root, parts = _fixture_root_and_parts(config, uri)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    binary = getattr(os, "O_BINARY", 0)
    if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | nofollow | nonblock | binary
        try:
            current = os.open(root, directory_flags)
        except OSError as error:
            raise SafetyError(f"cannot open fixture containment safely: {error}") from error
        try:
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise SafetyError("fixture containment must be a directory")
            for part in parts[:-1]:
                try:
                    child = os.open(part, directory_flags, dir_fd=current)
                except OSError as error:
                    raise SafetyError(f"cannot open fixture parent safely: {error}") from error
                os.close(current)
                current = child
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise SafetyError("fixture parent must be a directory")
            try:
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY | nofollow | nonblock | binary,
                    dir_fd=current,
                )
            except OSError as error:
                raise SafetyError(f"cannot open fixture safely: {error}") from error
            return descriptor
        finally:
            os.close(current)

    if os.name == "nt":
        return _windows_open_fixture(root, parts)

    # Platforms without descriptor-relative opens fail closed rather than using
    # a pathname-only fallback.
    raise SafetyError("safe fixture descriptor walking is unavailable")


def _windows_open_fixture(root: Path, parts: tuple[str, ...]) -> int:
    """Use Win32 handles, reparse-point opens, and final-handle containment."""
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
    except ImportError as error:  # pragma: no cover - Windows only
        raise SafetyError("required Windows handle APIs are unavailable") from error

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_attributes = kernel32.GetFileAttributesW
    get_attributes.argtypes = (wintypes.LPCWSTR,)
    get_attributes.restype = wintypes.DWORD
    get_final = kernel32.GetFinalPathNameByHandleW
    get_final.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
    get_final.restype = wintypes.DWORD
    get_type = kernel32.GetFileType
    get_type.argtypes = (wintypes.HANDLE,)
    get_type.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    invalid = wintypes.HANDLE(-1).value
    generic_read = 0x80000000
    share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    file_attribute_reparse_point = 0x00000400
    invalid_attributes = 0xFFFFFFFF
    file_flag_open_reparse_point = 0x00200000
    file_flag_backup_semantics = 0x02000000
    file_type_disk = 0x0001

    def open_handle(path: Path, *, directory: bool) -> int:
        attributes = get_attributes(str(path))
        if attributes == invalid_attributes or attributes & file_attribute_reparse_point:
            raise SafetyError("fixture path contains a reparse point or unavailable component")
        flags = file_flag_open_reparse_point
        if directory:
            flags |= file_flag_backup_semantics
        handle = create_file(
            str(path), generic_read, share_read_write_delete, None, open_existing, flags, None
        )
        if handle == invalid:
            raise SafetyError("cannot open fixture safely with Windows handle APIs")
        return handle

    def final_path(handle: int) -> Path:
        size = get_final(handle, None, 0, 0)
        if not size:
            raise SafetyError("cannot verify final fixture handle path")
        buffer = ctypes.create_unicode_buffer(size + 1)
        if not get_final(handle, buffer, len(buffer), 0):
            raise SafetyError("cannot verify final fixture handle path")
        return Path(buffer.value.removeprefix("\\\\?\\")).resolve()

    root_handle = open_handle(root, directory=True)
    try:
        root_final = final_path(root_handle)
    finally:
        close_handle(root_handle)
    current = root
    for part in parts[:-1]:
        current /= part
        handle = open_handle(current, directory=True)
        try:
            final_path(handle)
        finally:
            close_handle(handle)
    candidate = current / parts[-1]
    handle = open_handle(candidate, directory=False)
    try:
        if get_type(handle) != file_type_disk:
            raise SafetyError("fixture must be a regular disk file")
        actual = final_path(handle)
        try:
            actual.relative_to(root_final)
        except ValueError as error:
            raise SafetyError("fixture handle escapes configured containment") from error
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        handle = None
        return descriptor
    finally:
        if handle is not None:
            close_handle(handle)


def _media_type(initial: bytes) -> str | None:
    if initial.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if initial.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if initial.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if initial.startswith(b"RIFF") and initial[8:12] == b"WEBP":
        return "image/webp"
    return None


def _fingerprint_fixture(config: LabConfig, uri: str, binding_index: int) -> FixtureFingerprint:
    descriptor = _safe_open_fixture(config, uri)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SafetyError("fixture must be a regular file")
        if before.st_size > MAX_FIXTURE_BYTES:
            raise SafetyError("fixture exceeds maximum byte size")
        digest = hashlib.sha256()
        prefix = b""
        remaining = MAX_FIXTURE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            if len(prefix) < 16:
                prefix += chunk[: 16 - len(prefix)]
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise SafetyError("fixture exceeds maximum byte size")
        after = os.fstat(descriptor)
        stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        observed = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if stable != observed:
            raise SafetyError("fixture changed during read")
        return FixtureFingerprint(
            uri=uri,
            binding_index=binding_index,
            size=before.st_size,
            media_type=_media_type(prefix),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _fixture_fingerprints(
    brief: ExperimentBrief,
    config: LabConfig,
) -> tuple[FixtureFingerprint, ...]:
    return tuple(
        _fingerprint_fixture(config, item.uri, item.binding_index)
        for item in brief.fixtures.reference_images
    )


def _common_fixture_patch(
    brief: ExperimentBrief,
    source: PlanningSource,
) -> tuple[SemanticPatch, str]:
    mutations: list[Any] = []
    references = source.normalized_graph.bindings(SemanticRole.REFERENCE_IMAGE)
    for fixture in brief.fixtures.reference_images:
        if fixture.binding_index >= len(references):
            raise PlanningValidationError(
                f"reference binding index {fixture.binding_index} is not available"
            )
        binding = references[fixture.binding_index]
        if binding.field_name == "$class_type":
            raise PlanningValidationError("reference binding is not a concrete input")
        mutations.append(
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=fixture.binding_index,
                field_name=binding.field_name,
                value=f"fixture://{fixture.binding_index}",
            )
        )
    fanout = _prompt_fanout_binding(source)
    if fanout is None or fanout.field_name == "$class_type":
        raise PlanningValidationError(
            "prompt_lines require one unambiguous prompt/fanout binding; "
            "specify an explicit integer"
        )
    mutations.append(
        SetNodeInput(
            node_id=fanout.node_id,
            field_name=fanout.field_name,
            value=list(brief.fixtures.prompt_lines),
        )
    )
    return SemanticPatch(mutations=tuple(mutations)), fanout.node_id


def _prompt_fanout_binding(source: PlanningSource):
    fanouts = source.normalized_graph.bindings(SemanticRole.PROMPT_FANOUT)
    prompt_lines = tuple(
        binding
        for binding in fanouts
        if isinstance(thaw_json(binding.value), dict)
        and thaw_json(binding.value).get("kind") == "prompt_line"
    )
    if len(prompt_lines) == 1:
        return prompt_lines[0]
    return fanouts[0] if len(fanouts) == 1 else None


def _combine_patches(common: SemanticPatch, candidate: SemanticPatch | None) -> SemanticPatch:
    if candidate is None:
        return common
    try:
        return SemanticPatch(mutations=(*common.mutations, *candidate.mutations))
    except ValidationError as error:
        raise PlanningValidationError(f"candidate conflicts with fixture patch: {error}") from error


def _output_media_type(class_type: str) -> str | None:
    compact = "".join(character for character in class_type.casefold() if character.isalnum())
    if "jpeg" in compact or "jpg" in compact:
        return "image/jpeg"
    if "webp" in compact:
        return "image/webp"
    if "gif" in compact:
        return "image/gif"
    if "video" in compact:
        return "video/mp4"
    if compact.startswith(("save", "preview")) or compact.endswith("output"):
        return "image/png"
    return None


def _output_details(
    brief: ExperimentBrief,
    graph: dict[str, Any],
    prompt_fanout_node_id: str,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    normalized = normalize_graph(graph)
    outputs = normalized.bindings(SemanticRole.OUTPUT)
    if not outputs:
        raise PlanningValidationError(
            "output graph must contain at least one normalized output node"
        )
    node_by_id = {node.node_id: node for node in normalized.nodes}
    output_node_ids = tuple(sorted({binding.node_id for binding in outputs}))
    inferred = tuple(
        _output_media_type(node_by_id[node_id].class_type)
        for node_id in output_node_ids
    )
    if any(item is None for item in inferred):
        raise PlanningValidationError("output media type cannot be inferred")
    inferred_media = tuple(item for item in inferred if item is not None)
    declared = brief.expectations.media_types
    if declared:
        if len(declared) not in {1, len(inferred_media)}:
            raise PlanningValidationError("declared media types do not match output semantics")
        expected_media = (
            declared * len(inferred_media)
            if len(declared) == 1
            else declared
        )
        if expected_media != inferred_media:
            raise PlanningValidationError("declared media types do not match output semantics")
    else:
        expected_media = inferred_media
    if brief.expectations.output_count != "auto":
        return brief.expectations.output_count, output_node_ids, expected_media

    children: dict[str, set[str]] = {node.node_id: set() for node in normalized.nodes}
    for node in normalized.nodes:
        for parent_id, _ in node.input_links.values():
            children.setdefault(parent_id, set()).add(node.node_id)
    reachable = {prompt_fanout_node_id}
    pending = [prompt_fanout_node_id]
    while pending:
        node_id = pending.pop()
        for child_id in sorted(children.get(node_id, ())):
            if child_id not in reachable:
                reachable.add(child_id)
                pending.append(child_id)
    if not set(output_node_ids).issubset(reachable):
        raise PlanningValidationError(
            "auto output_count requires the changed prompt fanout to be reachable to every output; specify an explicit integer"
        )
    batches = [
        (node.node_id, node.inputs["batch_size"])
        for node in normalized.nodes
        if node.node_id in reachable and "batch_size" in node.inputs
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for _, value in batches
    ):
        raise PlanningValidationError("auto output_count has malformed batch size; specify an explicit integer")
    if len(batches) > 1:
        raise PlanningValidationError("auto output_count has ambiguous multiple batch branches; specify an explicit integer")
    batch = batches[0][1] if batches else 1
    return len(brief.fixtures.prompt_lines) * batch * len(output_node_ids), output_node_ids, expected_media


def _variant(
    *,
    brief: ExperimentBrief,
    variant_id: str,
    source: PlanningSource,
    patch: SemanticPatch,
    hypothesis: DimensionName | None,
    candidate: Candidate | None,
    fixture_fingerprints: tuple[FixtureFingerprint, ...],
    expected_output_count: int,
    output_node_ids: tuple[str, ...],
    expected_media_types: tuple[str, ...],
    prompt_lines: tuple[str, ...],
    prompt_fanout_node_id: str,
) -> PlannedVariant:
    try:
        materialized = materialize_variant(thaw_json(source.raw_graph), patch)
    except (PatchValidationError, PatchTopologyError, ValidationError, ValueError) as error:
        raise PlanningValidationError(f"structural materialization failed: {error}") from error
    graph = thaw_json(materialized.graph)
    expected_output_count, output_node_ids, expected_media_types = _output_details(
        brief=brief,
        graph=graph,
        prompt_fanout_node_id=prompt_fanout_node_id,
    )
    execution_parameters = {
        "prompt_lines": list(prompt_lines),
        "expected_output_count": expected_output_count,
        "output_node_ids": list(output_node_ids),
    }
    exact_run_hash = _hash(
        {
            "graph": graph,
            "source_hash": source.source_hash,
            "fixtures": [item.model_dump(mode="json") for item in fixture_fingerprints],
            "prompt_lines": list(prompt_lines),
            "execution_parameters": execution_parameters,
            "expected_output_count": expected_output_count,
            "output_node_ids": list(output_node_ids),
            "expected_media_types": list(expected_media_types),
        }
    )
    warnings = list(item.message for item in materialized.validation.warnings)
    if candidate is not None and candidate.confidence.value == "unexplored":
        warnings.append("candidate has unexplored compatibility")
    return PlannedVariant(
        variant_id=variant_id,
        source_workflow_id=source.workflow_id,
        source_hash=source.source_hash,
        semantic_patch=patch.model_dump(mode="json"),
        diff=materialized.diff.model_dump(mode="json"),
        materialized_graph=graph,
        hypothesis=hypothesis,
        evidence_ids=() if candidate is None else candidate.evidence_ids,
        confidence=None if candidate is None else candidate.confidence,
        warnings=tuple(sorted(set(warnings))),
        exact_run_hash=exact_run_hash,
        capability_hash=capability_hash(graph),
        expected_output_count=expected_output_count,
        output_node_ids=output_node_ids,
        expected_media_types=expected_media_types,
        execution_parameters=execution_parameters,
        fixture_fingerprints=fixture_fingerprints,
    )


def build_plan(
    brief: ExperimentBrief,
    source_bundle: SourceBundle,
    *,
    config: LabConfig,
) -> ExperimentPlan:
    """Build a pure plan; the only I/O is bounded fixture fingerprinting."""
    if not isinstance(brief, ExperimentBrief) or not isinstance(source_bundle, SourceBundle):
        raise TypeError("build_plan requires validated ExperimentBrief and SourceBundle")
    source = (
        next(
            item
            for item in source_bundle.sources
            if item.workflow_id == source_bundle.baseline_workflow_id
        )
        if source_bundle.baseline_workflow_id is not None
        else min(
            source_bundle.sources,
            key=lambda item: (-item.engagement, item.capability_hash, item.source_hash, item.workflow_id),
        )
    )
    common_patch, prompt_fanout_node_id = _common_fixture_patch(brief, source)
    fixture_fingerprints = _fixture_fingerprints(brief, config)
    variants = [
        _variant(
            brief=brief,
            variant_id="baseline",
            source=source,
            patch=common_patch,
            hypothesis=None,
            candidate=None,
            fixture_fingerprints=fixture_fingerprints,
            expected_output_count=0,
            output_node_ids=(),
            expected_media_types=(),
            prompt_lines=brief.fixtures.prompt_lines,
            prompt_fanout_node_id=prompt_fanout_node_id,
        )
    ]
    selection = select_candidates(brief, source_bundle.candidates)
    warnings = list(selection.warnings)
    coverage: set[DimensionName] = set()
    for candidate in selection.candidates:
        patch = propose_patch(candidate, source)
        if patch is None:
            warnings.append(
                f"candidate {candidate.candidate_id} omitted: insufficient exact binding or donor evidence"
            )
            continue
        reason = validate_candidate_patch(candidate, patch, source, brief)
        if reason is not None:
            warnings.append(f"candidate {candidate.candidate_id} omitted: {reason}")
            continue
        variants.append(
            _variant(
                brief=brief,
                variant_id=candidate.candidate_id,
                source=source,
                patch=_combine_patches(common_patch, patch),
                hypothesis=candidate.hypothesis,
                candidate=candidate,
                fixture_fingerprints=fixture_fingerprints,
                expected_output_count=0,
                output_node_ids=(),
                expected_media_types=(),
                prompt_lines=brief.fixtures.prompt_lines,
                prompt_fanout_node_id=prompt_fanout_node_id,
            )
        )
        coverage.add(candidate.hypothesis)
    variants_tuple = tuple(variants)
    plan_payload = {
        "source_hashes": sorted(item.source_hash for item in source_bundle.sources),
        "brief": brief.model_dump(mode="json"),
        "candidates": sorted(
            (item.model_dump(mode="json") for item in source_bundle.candidates),
            key=lambda item: (
                item["capability_signature"],
                item["candidate_id"],
            ),
        ),
        "variants": [item.model_dump(mode="json") for item in variants_tuple],
        "capability_coverage": sorted(item.value for item in coverage),
        "warnings": sorted(set(warnings)),
    }
    return ExperimentPlan(
        plan_hash=_hash(plan_payload),
        source_hashes=tuple(sorted(item.source_hash for item in source_bundle.sources)),
        variants=variants_tuple,
        capability_coverage=tuple(sorted(coverage, key=lambda item: item.value)),
        warnings=tuple(sorted(set(warnings))),
    )
