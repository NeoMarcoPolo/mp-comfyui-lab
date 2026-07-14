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
from mp_comfyui_lab.graph.hashing import capability_hash
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.patch import (
    PatchTopologyError,
    PatchValidationError,
    materialize_variant,
)
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.mutations import SemanticPatch, SetNodeInput, SetRoleInput
from mp_comfyui_lab.mutations.planner import propose_patch, select_candidates
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


def _fixture_candidate_path(config: LabConfig, uri: str) -> Path:
    # Resolve through LabConfig first; the raw path is checked separately so a
    # symlink cannot disappear from the evidence trail when resolve() follows it.
    resolved = config.resolve_fixture(uri)
    if uri.startswith("vault://"):
        alias, separator, relative = uri.removeprefix("vault://").partition("/")
        if not separator or alias not in config.vaults:
            raise SafetyError("invalid vault fixture URI")
        candidate = config.vaults[alias] / relative
        containment = config.vaults[alias]
    else:
        candidate = Path(uri).expanduser()
        if not candidate.is_absolute():
            candidate = config._fixture_base / candidate
        containment = config._fixture_base
    try:
        resolved.relative_to(containment.resolve())
    except ValueError as error:
        raise SafetyError("fixture escapes configured containment") from error
    current = containment
    relative_parts = candidate.relative_to(containment).parts
    for part in relative_parts:
        current /= part
        if current.is_symlink() or _is_reparse_point(current):
            raise SafetyError("fixture path contains a symlink or reparse point")
    return resolved


def _is_reparse_point(path: Path) -> bool:
    try:
        return bool(getattr(path.lstat(), "st_reparse_tag", 0))
    except OSError:
        return False


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
    resolved = _fixture_candidate_path(config, uri)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise SafetyError(f"cannot open fixture safely: {error}") from error
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
) -> SemanticPatch:
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
            "prompt_lines require one unambiguous prompt/fanout binding"
        )
    mutations.append(
        SetNodeInput(
            node_id=fanout.node_id,
            field_name=fanout.field_name,
            value=list(brief.fixtures.prompt_lines),
        )
    )
    return SemanticPatch(mutations=tuple(mutations))


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


def _output_details(
    brief: ExperimentBrief,
    source: PlanningSource,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    outputs = source.normalized_graph.bindings(SemanticRole.OUTPUT)
    if len(outputs) != 1:
        raise PlanningValidationError(
            "auto output_count requires one unambiguous normalized output node; use an explicit integer"
        )
    output_node_ids = tuple(binding.node_id for binding in outputs)
    media_types = brief.expectations.media_types
    if brief.expectations.output_count != "auto":
        return brief.expectations.output_count, output_node_ids, media_types
    if _prompt_fanout_binding(source) is None:
        raise PlanningValidationError(
            "auto output_count is unresolved; specify an explicit integer"
        )
    batch_values = [
        node.inputs.get("batch_size")
        for node in source.normalized_graph.nodes
        if "batch_size" in node.inputs
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in batch_values):
        raise PlanningValidationError(
            "auto output_count has malformed batch size; specify an explicit integer"
        )
    batch = batch_values[0] if len(batch_values) == 1 else 1
    if batch != 1:
        raise PlanningValidationError(
            "auto output_count cannot prove independent batch fanout; specify an explicit integer"
        )
    return len(brief.fixtures.prompt_lines), output_node_ids, media_types


def _variant(
    *,
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
) -> PlannedVariant:
    try:
        materialized = materialize_variant(thaw_json(source.raw_graph), patch)
    except (PatchValidationError, PatchTopologyError, ValidationError, ValueError) as error:
        raise PlanningValidationError(f"structural materialization failed: {error}") from error
    graph = thaw_json(materialized.graph)
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
    source = min(
        source_bundle.sources,
        key=lambda item: (item.capability_hash, -item.engagement, item.workflow_id),
    )
    expected_count, output_node_ids, media_types = _output_details(brief, source)
    common_patch = _common_fixture_patch(brief, source)
    fixture_fingerprints = _fixture_fingerprints(brief, config)
    variants = [
        _variant(
            variant_id="baseline",
            source=source,
            patch=common_patch,
            hypothesis=None,
            candidate=None,
            fixture_fingerprints=fixture_fingerprints,
            expected_output_count=expected_count,
            output_node_ids=output_node_ids,
            expected_media_types=media_types,
            prompt_lines=brief.fixtures.prompt_lines,
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
        variants.append(
            _variant(
                variant_id=candidate.candidate_id,
                source=source,
                patch=_combine_patches(common_patch, patch),
                hypothesis=candidate.hypothesis,
                candidate=candidate,
                fixture_fingerprints=fixture_fingerprints,
                expected_output_count=expected_count,
                output_node_ids=output_node_ids,
                expected_media_types=media_types,
                prompt_lines=brief.fixtures.prompt_lines,
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
