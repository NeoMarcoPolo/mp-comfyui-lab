from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

import imagehash
from pydantic import BaseModel, ConfigDict, Field

from mp_comfyui_lab.discovery.models import ContentClassification
from mp_comfyui_lab.experiments.models import OutputRecord, RunRecord
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import normalize_graph
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.media import (
    COMPARISON_MEDIA_POLICY,
    MediaFamily,
    inspect_media_file,
    media_family,
)

from .paths import hash_stored_artifact, resolve_stored_artifact


PHASH_HAMMING_THRESHOLD = 6


class _FactModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ProviderOutputMetadata(_FactModel):
    run_id: str
    provider_task_id: str | None = None
    node_id: str
    provider_order: int = Field(ge=0)
    recorded_media_type: str
    local_relative_path: str
    recorded_sha256: str
    recorded_bytes: int = Field(ge=0)
    integrity_matches_record: bool | None = None


class MediaErrorFacts(_FactModel):
    code: Literal[
        "unreadable_file",
        "empty_file",
        "byte_limit_exceeded",
        "unrecognized_signature",
        "declared_mime_mismatch",
        "invalid_image",
        "image_pixel_limit_exceeded",
        "invalid_archive",
    ]
    message: str


class MediaFacts(_FactModel):
    output_id: str
    variant_id: str
    output_ordinal: int = Field(ge=0)
    media_kind: Literal["image", "video", "audio", "unknown"]
    availability: Literal[
        "available",
        "integrity_mismatch",
        "missing",
        "not_recorded",
        "unsafe",
        "unreadable",
        "invalid",
    ]
    artifact_relative_path: str | None = None
    mime_type: str
    declared_mime_type: str
    declared_media_kind: MediaFamily
    detected_mime_type: str | None = None
    detected_media_kind: MediaFamily | None = None
    bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    mode: str | None = None
    phash: str | None = None
    prompt_label: str
    provider_metadata: ProviderOutputMetadata | None = None
    media_error: MediaErrorFacts | None = None


class VariantComparison(_FactModel):
    variant_id: str
    run_id: str
    label: str
    source_workflow_id: str | None = None
    author: str | None = None
    model_lora_summary: str
    prompt_label: str
    content_classification: ContentClassification
    status: str
    expected_output_count: int = Field(ge=1)
    actual_output_count: int = Field(ge=0)
    error: dict[str, Any] | None = None
    outputs: tuple[MediaFacts, ...]

    @property
    def facts(self) -> MediaFacts:
        """Compatibility shorthand for single-output comparisons."""
        return self.outputs[0]


class ComparisonReport(_FactModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    phash_hamming_threshold: Literal[6] = PHASH_HAMMING_THRESHOLD
    variants: dict[str, VariantComparison]
    duplicate_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def output_ids(self) -> tuple[str, ...]:
        return tuple(
            output.output_id
            for variant_id in sorted(self.variants)
            for variant in (self.variants[variant_id],)
            for output in variant.outputs
        )


def compare_outputs(
    runs: Iterable[RunRecord],
    experiment_root: str | Path,
    *,
    experiment_id: str | None = None,
) -> ComparisonReport:
    """Build objective facts from durable run records and their stored paths."""
    root = Path(experiment_root).resolve()
    selected = _select_variant_runs(tuple(runs))
    inferred_experiment_id = experiment_id or (
        next(iter(selected.values())).experiment_id if selected else root.name
    )
    variants: dict[str, VariantComparison] = {}
    for variant_id, run in sorted(selected.items()):
        if run.experiment_id != inferred_experiment_id:
            raise ValueError("comparison run experiment IDs must match")
        variants[variant_id] = _variant_facts(run, root)
    duplicate_groups = _duplicate_groups(
        output
        for variant in variants.values()
        for output in variant.outputs
    )
    return ComparisonReport(
        experiment_id=inferred_experiment_id,
        variants=variants,
        duplicate_groups=duplicate_groups,
    )


def _select_variant_runs(runs: tuple[RunRecord, ...]) -> dict[str, RunRecord]:
    selected: dict[str, RunRecord] = {}
    for run in runs:
        prior = selected.get(run.variant_id)
        if prior is None or _run_order(run) > _run_order(prior):
            selected[run.variant_id] = run
    return selected


def _run_order(run: RunRecord) -> tuple[str, str]:
    timestamp = next(
        (
            value
            for value in (run.updated_at, run.completed_at, run.created_at)
            if isinstance(value, datetime)
        ),
        None,
    )
    return ("" if timestamp is None else timestamp.isoformat(), run.run_id)


def _variant_facts(run: RunRecord, experiment_root: Path) -> VariantComparison:
    snapshot = (
        thaw_json(run.variant_snapshot)
        if run.variant_snapshot is not None
        else {}
    )
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    source_workflow_id = _nonblank_string(snapshot.get("source_workflow_id"))
    prompts = _prompt_lines(snapshot)
    records = {record.ordinal: record for record in run.outputs}
    ordinals = sorted(set(range(run.expected_output_count)).union(records))
    outputs = tuple(
        _media_facts(
            run,
            records.get(ordinal),
            ordinal,
            experiment_root,
            prompt_label=_prompt_label(
                prompts,
                ordinal,
                run.expected_output_count,
            ),
        )
        for ordinal in ordinals
    )
    return VariantComparison(
        variant_id=run.variant_id,
        run_id=run.run_id,
        label=source_workflow_id or run.variant_id,
        source_workflow_id=source_workflow_id,
        model_lora_summary=_model_lora_summary(snapshot),
        prompt_label=_prompt_summary(prompts),
        content_classification=_content_classification(snapshot),
        status=run.state.value,
        expected_output_count=run.expected_output_count,
        actual_output_count=run.actual_output_count,
        error=(
            None
            if run.error is None
            else dict(sorted(thaw_json(run.error).items()))
        ),
        outputs=outputs,
    )


def _media_facts(
    run: RunRecord,
    record: OutputRecord | None,
    ordinal: int,
    experiment_root: Path,
    *,
    prompt_label: str,
) -> MediaFacts:
    output_id = f"{run.variant_id}/{ordinal}"
    if record is None:
        mime_type = _expected_media_type(run, ordinal)
        declared_kind = _media_kind(mime_type)
        return MediaFacts(
            output_id=output_id,
            variant_id=run.variant_id,
            output_ordinal=ordinal,
            media_kind=declared_kind,
            availability="not_recorded",
            mime_type=mime_type,
            declared_mime_type=mime_type,
            declared_media_kind=declared_kind,
            prompt_label=prompt_label,
        )

    declared_kind = _media_kind(record.media_type)
    provider = ProviderOutputMetadata(
        run_id=run.run_id,
        provider_task_id=run.provider_task_id,
        node_id=record.node_id,
        provider_order=record.provider_order,
        recorded_media_type=record.media_type,
        local_relative_path=record.local_relative_path,
        recorded_sha256=record.content_sha256,
        recorded_bytes=record.size,
    )
    path, availability = resolve_stored_artifact(
        experiment_root,
        record.local_relative_path,
    )
    if path is None:
        return MediaFacts(
            output_id=output_id,
            variant_id=run.variant_id,
            output_ordinal=ordinal,
            media_kind=declared_kind,
            availability=availability,
            mime_type=record.media_type,
            declared_mime_type=record.media_type,
            declared_media_kind=declared_kind,
            prompt_label=prompt_label,
            provider_metadata=provider,
        )

    try:
        sha256, byte_count = hash_stored_artifact(path)
    except OSError:
        return MediaFacts(
            output_id=output_id,
            variant_id=run.variant_id,
            output_ordinal=ordinal,
            media_kind=declared_kind,
            availability="unreadable",
            mime_type=record.media_type,
            declared_mime_type=record.media_type,
            declared_media_kind=declared_kind,
            prompt_label=prompt_label,
            provider_metadata=provider,
        )

    provider = provider.model_copy(
        update={
            "integrity_matches_record": (
                sha256 == record.content_sha256 and byte_count == record.size
            )
        }
    )
    integrity_matches = provider.integrity_matches_record is True
    common: dict[str, Any] = {
        "output_id": output_id,
        "variant_id": run.variant_id,
        "output_ordinal": ordinal,
        "availability": (
            "available" if integrity_matches else "integrity_mismatch"
        ),
        "artifact_relative_path": (
            record.local_relative_path if integrity_matches else None
        ),
        "mime_type": record.media_type,
        "declared_mime_type": record.media_type,
        "declared_media_kind": declared_kind,
        "bytes": byte_count,
        "sha256": sha256,
        "prompt_label": prompt_label,
        "provider_metadata": provider,
    }
    if not integrity_matches:
        return MediaFacts(
            media_kind=declared_kind,
            **common,
        )

    inspection = inspect_media_file(
        path,
        record.media_type,
        policy=COMPARISON_MEDIA_POLICY,
    )
    detected = {
        "detected_mime_type": inspection.detected_mime_type,
        "detected_media_kind": inspection.detected_family,
    }
    if not inspection.valid:
        return MediaFacts(
            media_kind=declared_kind,
            **{
                **common,
                "availability": "invalid",
                "artifact_relative_path": None,
            },
            **detected,
            media_error=MediaErrorFacts(
                code=inspection.error_code,
                message=inspection.error_message,
            ),
        )
    if inspection.image is None:
        return MediaFacts(
            media_kind=declared_kind,
            **common,
            **detected,
        )
    display = inspection.image
    try:
        perceptual_hash = str(imagehash.phash(display))
        width, height = display.size
    finally:
        display.close()
    return MediaFacts(
        media_kind="image",
        width=width,
        height=height,
        mode=inspection.source_mode,
        phash=perceptual_hash,
        **common,
        **detected,
    )


def _media_kind(mime_type: str) -> MediaFamily:
    return media_family(mime_type)


def _expected_media_type(run: RunRecord, ordinal: int) -> str:
    if not run.expected_media_types:
        return "application/octet-stream"
    return run.expected_media_types[ordinal % len(run.expected_media_types)]


def _prompt_lines(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    parameters = snapshot.get("execution_parameters")
    if not isinstance(parameters, Mapping):
        return ()
    values = parameters.get("prompt_lines")
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(value for value in values if isinstance(value, str) and value)


def _content_classification(
    snapshot: Mapping[str, Any],
) -> ContentClassification:
    value = snapshot.get("content_classification")
    try:
        return ContentClassification(value)
    except (TypeError, ValueError):
        return ContentClassification.UNKNOWN


def _prompt_label(
    prompts: tuple[str, ...],
    ordinal: int,
    expected_output_count: int,
) -> str:
    if not prompts:
        return "Prompt unavailable"
    outputs_per_prompt = max(1, expected_output_count // len(prompts))
    prompt_index = min(ordinal // outputs_per_prompt, len(prompts) - 1)
    return f"Prompt {prompt_index + 1}: {prompts[prompt_index]}"


def _prompt_summary(prompts: tuple[str, ...]) -> str:
    if not prompts:
        return "Prompt unavailable"
    if len(prompts) == 1:
        return f"Prompt 1: {prompts[0]}"
    return f"{len(prompts)} prompts"


def _model_lora_summary(snapshot: Mapping[str, Any]) -> str:
    graph = snapshot.get("materialized_graph")
    if not isinstance(graph, Mapping):
        return "Model/LoRA unavailable"
    try:
        normalized = normalize_graph(graph)
    except ValueError:
        return _literal_model_lora_summary(graph)
    models = _binding_labels(
        normalized.bindings(SemanticRole.CHECKPOINT)
        or normalized.bindings(SemanticRole.MODEL)
    )
    loras = _binding_labels(normalized.bindings(SemanticRole.LORA))
    parts = []
    if models:
        parts.append(f"Model: {', '.join(models)}")
    if loras:
        parts.append(f"LoRA: {', '.join(loras)}")
    return " | ".join(parts) or _literal_model_lora_summary(graph)


def _literal_model_lora_summary(graph: Mapping[str, Any]) -> str:
    model_fields = {
        "checkpoint_name",
        "ckpt_name",
        "diffusion_model_name",
        "model_name",
        "unet_name",
    }
    lora_fields = {"lora", "lora_file", "lora_name"}
    models: list[str] = []
    loras: list[str] = []
    for node_id in sorted(graph, key=str):
        node = graph[node_id]
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        class_type = str(node.get("class_type", "")).casefold()
        for field_name, value in sorted(inputs.items(), key=lambda item: str(item[0])):
            canonical_name = str(field_name).casefold()
            if canonical_name in model_fields and isinstance(value, str):
                if value not in models:
                    models.append(value)
            if (
                canonical_name in lora_fields
                and "lora" in class_type
                and isinstance(value, str)
                and value not in loras
            ):
                loras.append(value)
            if "lora" not in class_type or not isinstance(value, Mapping):
                continue
            nested_name = value.get("lora")
            if isinstance(nested_name, str) and nested_name not in loras:
                loras.append(nested_name)
    parts = []
    if models:
        parts.append(f"Model: {', '.join(models)}")
    if loras:
        parts.append(f"LoRA: {', '.join(loras)}")
    return " | ".join(parts) or "Model/LoRA unavailable"


def _binding_labels(bindings: Iterable[Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for binding in bindings:
        value = thaw_json(binding.value)
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            label = value["name"]
        elif isinstance(value, str):
            label = value
        else:
            label = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _nonblank_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _duplicate_groups(outputs: Iterable[MediaFacts]) -> tuple[tuple[str, ...], ...]:
    return group_perceptual_duplicates(
        {
            output.output_id: output.phash
            for output in outputs
            if output.media_kind == "image"
            and output.availability == "available"
            and output.phash is not None
        }
    )


def group_perceptual_duplicates(
    phashes: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Return stable connected duplicate groups using the fixed threshold."""
    images = sorted(
        (
            (output_id, imagehash.hex_to_hash(perceptual_hash))
            for output_id, perceptual_hash in phashes.items()
        ),
        key=lambda item: item[0],
    )
    parents = {output_id: output_id for output_id, _ in images}

    def find(output_id: str) -> str:
        while parents[output_id] != output_id:
            parents[output_id] = parents[parents[output_id]]
            output_id = parents[output_id]
        return output_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for index, (left_id, left_hash) in enumerate(images):
        for right_id, right_hash in images[index + 1 :]:
            if left_hash - right_hash <= PHASH_HAMMING_THRESHOLD:
                union(left_id, right_id)

    grouped: dict[str, list[str]] = {}
    for output_id, _ in images:
        grouped.setdefault(find(output_id), []).append(output_id)
    return tuple(
        sorted(
            tuple(sorted(group))
            for group in grouped.values()
            if len(group) > 1
        )
    )
