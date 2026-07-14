from __future__ import annotations

import ctypes
import errno
from io import BytesIO
import hashlib
import html
from importlib import resources
import json
import math
import os
from pathlib import Path
import shutil
from string import Template
import tempfile
from typing import Any, Callable, Sequence
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.media import (
    COMPARISON_MEDIA_POLICY,
    load_normalized_image,
)
from mp_comfyui_lab.security import SafetyError

from .metrics import ComparisonReport, MediaFacts, VariantComparison
from .paths import (
    hash_stored_artifact,
    path_component_is_link_or_junction,
    resolve_stored_artifact,
)


_COMPARISON_DIRECTORY = Path("comparisons")
_TILE_WIDTH = 420
_TILE_HEIGHT = 360
_THUMBNAIL_SIZE = (380, 205)
_CONTACT_COLUMNS = 3
# 64 tiles cap each decoded page at 1,260 x 7,920 pixels.
CONTACT_SHEET_MAX_TILES = 64
_JUDGMENTS_NAME = "judgments.json"
FaultInjector = Callable[[str], None]
ArtifactProducer = Callable[[], bytes]
ArtifactValue = bytes | ArtifactProducer


class ComparisonArtifactConflict(SafetyError):
    """Raised when rendering would replace a different durable artifact."""


class ComparisonArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    experiment_root: Path
    facts: Path
    duplicate_groups: Path
    contact_sheet: Path
    contact_sheets: tuple[Path, ...]
    gallery: Path
    judgment_input: Path
    judgments: Path | None = None


def render_comparison(
    report: ComparisonReport,
    experiment_root: str | Path,
    *,
    judgment_document: bytes | None = None,
    fault_injector: FaultInjector | None = None,
    replace_existing: bool = False,
) -> ComparisonArtifacts:
    """Render a deterministic, local-only comparison artifact set."""
    root = Path(experiment_root).resolve()
    if report.experiment_id != root.name:
        raise SafetyError(
            "comparison report identity does not match experiment root"
        )
    facts_bytes = _json_bytes(report.model_dump(mode="json"))
    duplicate_groups_bytes = _json_bytes(
        {
            "schema_version": "1",
            "phash_hamming_threshold": report.phash_hamming_threshold,
            "duplicate_groups": [list(group) for group in report.duplicate_groups],
        }
    )
    contact_items = _contact_sheet_items(report)
    contact_page_names = _contact_sheet_page_names(len(contact_items))
    artifact_values: dict[str, ArtifactValue] = {
        "facts.json": facts_bytes,
        "duplicate-groups.json": duplicate_groups_bytes,
        "index.html": _gallery_html(
            report,
            root,
            contact_page_names,
        ).encode("utf-8"),
        "judgment-input.json": _json_bytes(
            _judgment_input(report, facts_bytes)
        ),
    }
    for page_index, page_name in enumerate(contact_page_names):
        start = page_index * CONTACT_SHEET_MAX_TILES
        page_items = contact_items[start : start + CONTACT_SHEET_MAX_TILES]
        artifact_values[page_name] = (
            lambda items=page_items: _contact_sheet_page_bytes(items, root)
        )
    _publish_artifact_set(
        root,
        artifact_values,
        judgment_document=judgment_document,
        fault_injector=fault_injector,
        replace_existing=replace_existing,
    )
    judgments_path = root / _COMPARISON_DIRECTORY / _JUDGMENTS_NAME
    contact_sheets = tuple(
        root / _COMPARISON_DIRECTORY / name for name in contact_page_names
    )
    return ComparisonArtifacts(
        experiment_root=root,
        facts=root / _COMPARISON_DIRECTORY / "facts.json",
        duplicate_groups=(
            root / _COMPARISON_DIRECTORY / "duplicate-groups.json"
        ),
        contact_sheet=contact_sheets[0],
        contact_sheets=contact_sheets,
        gallery=root / _COMPARISON_DIRECTORY / "index.html",
        judgment_input=root / _COMPARISON_DIRECTORY / "judgment-input.json",
        judgments=judgments_path if judgments_path.exists() else None,
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def comparison_facts_sha256(report: ComparisonReport) -> str:
    return hashlib.sha256(_json_bytes(report.model_dump(mode="json"))).hexdigest()


def _publish_artifact_set(
    root: Path,
    objective_artifacts: dict[str, ArtifactValue],
    *,
    judgment_document: bytes | None,
    fault_injector: FaultInjector | None,
    replace_existing: bool,
) -> None:
    destination = root / _COMPARISON_DIRECTORY
    if destination.exists() or destination.is_symlink():
        try:
            _verify_existing_artifact_set(
                destination,
                objective_artifacts,
                context="preexisting",
            )
        except ComparisonArtifactConflict:
            if not replace_existing:
                raise
            _replace_artifact_set(
                root,
                destination,
                objective_artifacts,
                judgment_document=judgment_document,
                fault_injector=fault_injector,
            )
            return
        if judgment_document is not None:
            _append_judgments(
                root,
                judgment_document,
                fault_injector=fault_injector,
            )
        return

    stage = Path(
        tempfile.mkdtemp(prefix=".comparison-stage-", dir=root)
    )
    staged_artifacts: dict[str, ArtifactValue] = dict(objective_artifacts)
    if judgment_document is not None:
        staged_artifacts[_JUDGMENTS_NAME] = judgment_document
    try:
        for name, content in staged_artifacts.items():
            _inject(fault_injector, f"stage:{name}")
            _write_staged_file(stage / name, _artifact_bytes(content))
        _inject(fault_injector, "fsync:staging")
        _fsync_directory(stage)
        _inject(fault_injector, "publish:comparisons")
        try:
            _rename_directory_new(stage, destination)
        except (FileExistsError, FileNotFoundError, OSError) as error:
            try:
                _verify_existing_artifact_set(
                    destination,
                    objective_artifacts,
                    context="concurrent",
                )
                if judgment_document is not None:
                    _append_judgments(
                        root,
                        judgment_document,
                        fault_injector=fault_injector,
                    )
            except ComparisonArtifactConflict as conflict:
                raise conflict from error
        else:
            _fsync_directory(root)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _replace_artifact_set(
    root: Path,
    destination: Path,
    objective_artifacts: dict[str, ArtifactValue],
    *,
    judgment_document: bytes | None,
    fault_injector: FaultInjector | None,
) -> None:
    if path_component_is_link_or_junction(destination) or not destination.is_dir():
        raise ComparisonArtifactConflict(
            "preexisting comparisons path is not a safe local directory"
        )
    stage = Path(tempfile.mkdtemp(prefix=".comparison-stage-", dir=root))
    backup = Path(tempfile.mkdtemp(prefix=".comparison-backup-", dir=root))
    backup.rmdir()
    staged_artifacts: dict[str, ArtifactValue] = dict(objective_artifacts)
    if judgment_document is not None:
        staged_artifacts[_JUDGMENTS_NAME] = judgment_document
    moved_existing = False
    published = False
    try:
        for name, content in staged_artifacts.items():
            _inject(fault_injector, f"stage:{name}")
            _write_staged_file(stage / name, _artifact_bytes(content))
        _inject(fault_injector, "fsync:staging")
        _fsync_directory(stage)
        _inject(fault_injector, "replace:backup")
        os.rename(destination, backup)
        moved_existing = True
        _inject(fault_injector, "replace:publish")
        _rename_directory_new(stage, destination)
        published = True
        _fsync_directory(root)
    except Exception:
        if moved_existing and not published and not destination.exists():
            os.rename(backup, destination)
            moved_existing = False
            _fsync_directory(root)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if published:
            shutil.rmtree(backup, ignore_errors=True)
            _fsync_directory(root)


def _verify_existing_artifact_set(
    directory: Path,
    objective_artifacts: dict[str, ArtifactValue],
    *,
    context: str,
) -> None:
    if path_component_is_link_or_junction(directory) or not directory.is_dir():
        raise ComparisonArtifactConflict(
            f"{context} comparisons path is not a safe local directory"
        )
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise ComparisonArtifactConflict(
            f"{context} comparison artifact set is not safely readable"
        ) from error
    names = {entry.name for entry in entries}
    objective_names = set(objective_artifacts)
    allowed_names = objective_names | {_JUDGMENTS_NAME}
    if not objective_names.issubset(names):
        raise ComparisonArtifactConflict(
            f"{context} comparison artifact set is incomplete"
        )
    if not names.issubset(allowed_names):
        raise ComparisonArtifactConflict(
            f"{context} comparison artifact set contains unexpected files"
        )
    expected = dict(objective_artifacts)
    if _JUDGMENTS_NAME in names:
        artifact = directory / _JUDGMENTS_NAME
        if path_component_is_link_or_junction(artifact) or not artifact.is_file():
            raise ComparisonArtifactConflict(
                f"{context} comparison artifact is a link or junction: "
                f"{_JUDGMENTS_NAME}"
            )
    for name, content in expected.items():
        artifact = directory / name
        if path_component_is_link_or_junction(artifact):
            raise ComparisonArtifactConflict(
                f"{context} comparison artifact is a link or junction: {name}"
            )
        try:
            if (
                not artifact.is_file()
                or artifact.read_bytes() != _artifact_bytes(content)
            ):
                raise ComparisonArtifactConflict(
                    f"{context} comparison artifact would clobber {name}"
                )
        except OSError as error:
            raise ComparisonArtifactConflict(
                f"{context} comparison artifact is not safely readable: {name}"
            ) from error


def _append_judgments(
    root: Path,
    judgment_document: bytes,
    *,
    fault_injector: FaultInjector | None,
) -> None:
    judgment_path = root / _COMPARISON_DIRECTORY / _JUDGMENTS_NAME
    if not judgment_path.exists():
        _inject(fault_injector, f"publish:{_JUDGMENTS_NAME}")
    store = ExperimentStore(root.parent)
    try:
        store.write_bytes_new(
            root.name,
            _COMPARISON_DIRECTORY / _JUDGMENTS_NAME,
            judgment_document,
            reuse_matching=True,
        )
    except SafetyError as error:
        raise ComparisonArtifactConflict(
            "comparison judgments conflict with an existing artifact"
        ) from error


def _artifact_bytes(value: ArtifactValue) -> bytes:
    return value() if callable(value) else value


def _write_staged_file(path: Path, content: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("comparison staging write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_new(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SafetyError(
            "atomic no-clobber comparison directory publish is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number == errno.ENOENT:
        raise FileNotFoundError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inject(injector: FaultInjector | None, point: str) -> None:
    if injector is not None:
        injector(point)


def _judgment_input(report: ComparisonReport, facts_bytes: bytes) -> dict[str, Any]:
    requested_fields = [
        "identity_fidelity",
        "prompt_adherence",
        "visual_quality",
        "creative_usefulness",
        "reasons",
        "referenced_output_ids",
    ]
    facts_sha256 = hashlib.sha256(facts_bytes).hexdigest()
    variant_ids = sorted(report.variants)
    output_ids = [
        output.output_id
        for variant_id in sorted(report.variants)
        for variant in (report.variants[variant_id],)
        for output in variant.outputs
        if output.availability == "available"
        and output.artifact_relative_path is not None
    ]
    variant_context = [
        {
            "variant_id": variant_id,
            "content_classification": report.variants[
                variant_id
            ].content_classification.value,
            "status": report.variants[variant_id].status,
            "output_ids": [
                output.output_id
                for output in report.variants[variant_id].outputs
                if output.availability == "available"
                and output.artifact_relative_path is not None
            ],
        }
        for variant_id in variant_ids
    ]
    score_schema = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    per_variant_schemas = [
        _variant_judgment_schema(
            variant_id,
            [
                output.output_id
                for output in report.variants[variant_id].outputs
                if output.availability == "available"
                and output.artifact_relative_path is not None
            ],
            requested_fields=requested_fields,
            score_schema=score_schema,
        )
        for variant_id in variant_ids
    ]
    judgment_items_schema: dict[str, Any] | bool = (
        {"oneOf": per_variant_schemas} if per_variant_schemas else False
    )
    judgments_schema: dict[str, Any] = {
        "type": "array",
        "minItems": len(variant_ids),
        "maxItems": len(variant_ids),
        "items": judgment_items_schema,
    }
    if variant_ids:
        judgments_schema["allOf"] = [
            {
                "type": "array",
                "contains": {
                    "type": "object",
                    "required": ["variant_id"],
                    "properties": {
                        "variant_id": {"const": variant_id},
                    },
                },
                "minContains": 1,
                "maxContains": 1,
            }
            for variant_id in variant_ids
        ]
    return {
        "schema_version": "1",
        "experiment_id": report.experiment_id,
        "facts_sha256": facts_sha256,
        "variant_ids": variant_ids,
        "output_ids": output_ids,
        "variant_context": variant_context,
        "requested_fields": requested_fields,
        "score_enum": [1, 2, 3, 4, 5],
        "submission_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "experiment_id",
                "facts_sha256",
                "judgments",
            ],
            "properties": {
                "schema_version": {"const": "1"},
                "experiment_id": {"const": report.experiment_id},
                "facts_sha256": {"const": facts_sha256},
                "judgments": judgments_schema,
            },
        },
    }


def _variant_judgment_schema(
    variant_id: str,
    output_ids: list[str],
    *,
    requested_fields: list[str],
    score_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["variant_id", *requested_fields],
        "properties": {
            "variant_id": {"type": "string", "const": variant_id},
            "identity_fidelity": score_schema,
            "prompt_adherence": score_schema,
            "visual_quality": score_schema,
            "creative_usefulness": score_schema,
            "reasons": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4096,
                    "pattern": r"\S",
                },
            },
            "referenced_output_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": 512,
                "uniqueItems": True,
                "items": (
                    {"type": "string", "enum": output_ids}
                    if output_ids
                    else False
                ),
            },
        },
    }


def _gallery_html(
    report: ComparisonReport,
    experiment_root: Path,
    contact_page_names: tuple[str, ...],
) -> str:
    template_text = (
        resources.files("mp_comfyui_lab.resources")
        .joinpath("comparison.html")
        .read_text(encoding="utf-8")
    )
    sections = "\n".join(
        _variant_html(report.variants[variant_id], experiment_root)
        for variant_id in sorted(report.variants)
    )
    if not sections:
        sections = '<p class="empty">No durable runs are available.</p>'
    return Template(template_text).substitute(
        experiment_id=_escape(report.experiment_id),
        contact_sheet_links=_contact_sheet_links(contact_page_names),
        variant_sections=sections,
    )


def _contact_sheet_links(page_names: tuple[str, ...]) -> str:
    links = "".join(
        f'<a href="{_escape(name)}">Page {index}</a>'
        for index, name in enumerate(page_names, start=1)
    )
    return (
        '<nav class="contact-sheets" aria-label="Contact sheets">'
        "<strong>Contact sheets</strong>"
        f"{links}"
        "</nav>"
    )


def _variant_html(variant: VariantComparison, experiment_root: Path) -> str:
    output_html = "\n".join(
        _output_html(output, variant, experiment_root)
        for output in sorted(variant.outputs, key=lambda item: item.output_ordinal)
    )
    if not output_html:
        output_html = '<p class="empty">No outputs were recorded.</p>'
    author = (
        ""
        if variant.author is None
        else f"<dt>Author</dt><dd>{_escape(variant.author)}</dd>"
    )
    error = (
        ""
        if variant.error is None
        else (
            "<dt>Error</dt><dd><code>"
            f"{_escape(json.dumps(variant.error, ensure_ascii=False, sort_keys=True))}"
            "</code></dd>"
        )
    )
    workflow = variant.source_workflow_id or "Unavailable"
    return (
        '<section class="variant">'
        f"<h2>{_escape(variant.variant_id)}</h2>"
        "<dl>"
        f"<dt>Label</dt><dd>{_escape(variant.label)}</dd>"
        f"<dt>Workflow</dt><dd>{_escape(workflow)}</dd>"
        f"{author}"
        f"<dt>Model stack</dt><dd>{_escape(variant.model_lora_summary)}</dd>"
        f"<dt>Prompt</dt><dd>{_escape(variant.prompt_label)}</dd>"
        "<dt>Content classification</dt>"
        f"<dd>{_escape(variant.content_classification.value)}</dd>"
        f"<dt>Status</dt><dd>{_escape(variant.status)}</dd>"
        f"{error}"
        "</dl>"
        f'<div class="outputs">{output_html}</div>'
        "</section>"
    )


def _output_html(
    output: MediaFacts,
    variant: VariantComparison,
    experiment_root: Path,
) -> str:
    href = _validated_artifact_href(experiment_root, output)
    unavailable_label = (
        "Artifact invalid"
        if output.availability == "invalid"
        else "Artifact unavailable"
    )
    media = f'<div class="missing">{unavailable_label}</div>'
    if href is not None:
        escaped_href = _escape(href)
        if output.media_kind == "image":
            media = (
                f'<img src="{escaped_href}" '
                f'alt="{_escape(output.output_id)}">'
            )
        elif output.media_kind == "video":
            media = (
                '<video controls preload="metadata" '
                f'src="{escaped_href}"></video>'
            )
        elif output.media_kind == "audio":
            media = (
                '<audio controls preload="metadata" '
                f'src="{escaped_href}"></audio>'
            )
        else:
            media = (
                f'<a class="artifact-link" href="{escaped_href}">'
                "Open local artifact</a>"
            )
    dimensions = (
        ""
        if output.width is None or output.height is None
        else f" | {output.width} x {output.height} {output.mode or ''}".rstrip()
    )
    detected = output.detected_mime_type or "unrecognized"
    error = (
        ""
        if output.media_error is None
        else f" | {_escape(output.media_error.code)}: {_escape(output.media_error.message)}"
    )
    return (
        f'<figure data-output-id="{_escape(output.output_id)}">'
        f'<div class="media">{media}</div>'
        "<figcaption>"
        f"<strong>Output {output.output_ordinal}</strong>"
        f"<span>{_escape(output.prompt_label)}</span>"
        f"<span>Status: {_escape(variant.status)} / {_escape(output.availability)}</span>"
        f"<span>Declared: {_escape(output.declared_mime_type)} | "
        f"Detected: {_escape(detected)} | "
        f"{output.bytes if output.bytes is not None else 'n/a'} bytes"
        f"{_escape(dimensions)}{error}</span>"
        "</figcaption>"
        "</figure>"
    )


def _validated_artifact_href(
    experiment_root: Path,
    output: MediaFacts,
) -> str | None:
    relative_path = output.artifact_relative_path
    if not isinstance(relative_path, str):
        return None
    provider = output.provider_metadata
    if (
        output.availability != "available"
        or provider is None
        or provider.integrity_matches_record is not True
        or provider.local_relative_path != relative_path
        or provider.recorded_sha256 != output.sha256
        or provider.recorded_bytes != output.bytes
    ):
        return None
    resolved, _ = resolve_stored_artifact(experiment_root, relative_path)
    if resolved is None:
        return None
    try:
        sha256, byte_count = hash_stored_artifact(resolved)
    except OSError:
        return None
    if output.sha256 != sha256 or output.bytes != byte_count:
        return None
    return quote(
        (Path("..") / Path(relative_path)).as_posix(),
        safe="/-._~",
    )


def _contact_sheet_items(
    report: ComparisonReport,
) -> tuple[tuple[VariantComparison, MediaFacts], ...]:
    return tuple(
        (variant, output)
        for variant_id in sorted(report.variants)
        for variant in (report.variants[variant_id],)
        for output in sorted(variant.outputs, key=lambda item: item.output_ordinal)
    )


def _contact_sheet_page_names(item_count: int) -> tuple[str, ...]:
    page_count = max(1, math.ceil(item_count / CONTACT_SHEET_MAX_TILES))
    return tuple(
        "contact-sheet.jpg"
        if page == 1
        else f"contact-sheet-{page:04d}.jpg"
        for page in range(1, page_count + 1)
    )


def _contact_sheet_page_bytes(
    items: Sequence[tuple[VariantComparison, MediaFacts]],
    experiment_root: Path,
) -> bytes:
    count = max(1, len(items))
    columns = min(_CONTACT_COLUMNS, count)
    rows = math.ceil(count / columns)
    canvas = Image.new(
        "RGB",
        (columns * _TILE_WIDTH, rows * _TILE_HEIGHT),
        color=(245, 247, 248),
    )
    try:
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        if not items:
            draw.text(
                (24, 24),
                "No durable comparison outputs",
                fill=(25, 30, 35),
                font=font,
            )
        for index, (variant, output) in enumerate(items):
            x = (index % columns) * _TILE_WIDTH
            y = (index // columns) * _TILE_HEIGHT
            _draw_contact_tile(
                canvas,
                draw,
                font,
                x,
                y,
                variant,
                output,
                experiment_root,
            )
        encoded = BytesIO()
        canvas.save(
            encoded,
            format="JPEG",
            quality=90,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        return encoded.getvalue()
    finally:
        canvas.close()


def _draw_contact_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    x: int,
    y: int,
    variant: VariantComparison,
    output: MediaFacts,
    experiment_root: Path,
) -> None:
    margin = 20
    draw.rectangle(
        (x + 8, y + 8, x + _TILE_WIDTH - 8, y + _TILE_HEIGHT - 8),
        fill=(255, 255, 255),
        outline=(202, 209, 214),
        width=1,
    )
    media_box = (
        x + margin,
        y + margin,
        x + margin + _THUMBNAIL_SIZE[0],
        y + margin + _THUMBNAIL_SIZE[1],
    )
    draw.rectangle(media_box, fill=(229, 233, 236))
    href = _validated_artifact_href(experiment_root, output)
    if output.media_kind == "image" and href is not None:
        source_path = experiment_root / output.artifact_relative_path
        try:
            image, _ = load_normalized_image(
                source_path,
                alpha_background="checkerboard",
                policy=COMPARISON_MEDIA_POLICY,
            )
            try:
                image.thumbnail(_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                left = media_box[0] + (_THUMBNAIL_SIZE[0] - image.width) // 2
                top = media_box[1] + (_THUMBNAIL_SIZE[1] - image.height) // 2
                canvas.paste(image, (left, top))
            finally:
                image.close()
        except (OSError, SyntaxError, ValueError):
            _draw_placeholder(draw, media_box, "invalid", font)
    else:
        placeholder = (
            "invalid" if output.availability == "invalid" else output.media_kind
        )
        _draw_placeholder(draw, media_box, placeholder, font)
    lines = (
        f"Variant: {variant.variant_id}",
        variant.model_lora_summary,
        output.prompt_label,
        f"Output {output.output_ordinal}",
        f"Classification: {variant.content_classification.value}",
        f"Status: {variant.status} / {output.availability}",
    )
    text_y = media_box[3] + 12
    for line in lines:
        draw.text(
            (x + margin, text_y),
            _contact_text(line, 64),
            fill=(25, 30, 35),
            font=font,
        )
        text_y += 21


def _draw_placeholder(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    media_kind: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    label = _contact_text(media_kind.upper(), 24)
    bounds = draw.textbbox((0, 0), label, font=font)
    x = box[0] + (box[2] - box[0] - (bounds[2] - bounds[0])) // 2
    y = box[1] + (box[3] - box[1] - (bounds[3] - bounds[1])) // 2
    draw.text((x, y), label, fill=(78, 88, 96), font=font)


def _contact_text(value: Any, maximum: int) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    if len(text) > maximum:
        text = f"{text[: maximum - 3]}..."
    return text.encode("latin-1", "replace").decode("latin-1")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
