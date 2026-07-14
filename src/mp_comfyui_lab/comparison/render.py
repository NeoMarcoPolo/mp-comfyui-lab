from __future__ import annotations

from io import BytesIO
import hashlib
import html
from importlib import resources
import json
import math
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict

from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import SafetyError

from .metrics import ComparisonReport, MediaFacts, VariantComparison
from .paths import hash_stored_artifact, resolve_stored_artifact


_COMPARISON_DIRECTORY = Path("comparisons")
_TILE_WIDTH = 420
_TILE_HEIGHT = 360
_THUMBNAIL_SIZE = (380, 205)
_CONTACT_COLUMNS = 3


class ComparisonArtifactConflict(SafetyError):
    """Raised when rendering would replace a different durable artifact."""


class ComparisonArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    facts: Path
    duplicate_groups: Path
    contact_sheet: Path
    gallery: Path
    judgment_input: Path


def render_comparison(
    report: ComparisonReport,
    experiment_root: str | Path,
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
    artifact_values = {
        _COMPARISON_DIRECTORY / "facts.json": facts_bytes,
        _COMPARISON_DIRECTORY / "duplicate-groups.json": duplicate_groups_bytes,
        _COMPARISON_DIRECTORY / "contact-sheet.jpg": _contact_sheet_bytes(
            report,
            root,
        ),
        _COMPARISON_DIRECTORY / "index.html": _gallery_html(
            report,
            root,
        ).encode("utf-8"),
        _COMPARISON_DIRECTORY / "judgment-input.json": _json_bytes(
            _judgment_input(report, facts_bytes)
        ),
    }
    _publish_artifacts(root, artifact_values)
    return ComparisonArtifacts(
        facts=root / _COMPARISON_DIRECTORY / "facts.json",
        duplicate_groups=(
            root / _COMPARISON_DIRECTORY / "duplicate-groups.json"
        ),
        contact_sheet=root / _COMPARISON_DIRECTORY / "contact-sheet.jpg",
        gallery=root / _COMPARISON_DIRECTORY / "index.html",
        judgment_input=root / _COMPARISON_DIRECTORY / "judgment-input.json",
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


def _publish_artifacts(root: Path, artifacts: dict[Path, bytes]) -> None:
    store = ExperimentStore(root.parent)
    experiment_id = root.name
    for relative_path, content in artifacts.items():
        destination = root / relative_path
        if destination.exists():
            try:
                existing = store.read_bytes(experiment_id, relative_path)
            except OSError as error:
                raise ComparisonArtifactConflict(
                    f"comparison artifact is not safely readable: {relative_path}"
                ) from error
            if existing != content:
                raise ComparisonArtifactConflict(
                    f"comparison render would clobber {relative_path.as_posix()}"
                )
    for relative_path, content in artifacts.items():
        try:
            store.write_bytes_new(
                experiment_id,
                relative_path,
                content,
                reuse_matching=True,
            )
        except SafetyError as error:
            raise ComparisonArtifactConflict(
                f"comparison render would clobber {relative_path.as_posix()}"
            ) from error


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
    score_schema = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
    judgment_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["variant_id", *requested_fields],
        "properties": {
            "variant_id": (
                {"type": "string", "enum": variant_ids}
                if variant_ids
                else {"not": {}}
            ),
            "identity_fidelity": score_schema,
            "prompt_adherence": score_schema,
            "visual_quality": score_schema,
            "creative_usefulness": score_schema,
            "reasons": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1, "maxLength": 4096},
            },
            "referenced_output_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 512,
                "uniqueItems": True,
                "items": (
                    {"type": "string", "enum": output_ids}
                    if output_ids
                    else {"not": {}}
                ),
            },
        },
    }
    return {
        "schema_version": "1",
        "experiment_id": report.experiment_id,
        "facts_sha256": facts_sha256,
        "variant_ids": variant_ids,
        "output_ids": output_ids,
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
                "judgments": {
                    "type": "array",
                    "minItems": 1 if variant_ids else 0,
                    "maxItems": len(variant_ids),
                    "items": judgment_schema,
                },
            },
        },
    }


def _gallery_html(report: ComparisonReport, experiment_root: Path) -> str:
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
        variant_sections=sections,
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
    media = '<div class="missing">Artifact unavailable</div>'
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
    return (
        f'<figure data-output-id="{_escape(output.output_id)}">'
        f'<div class="media">{media}</div>'
        "<figcaption>"
        f"<strong>Output {output.output_ordinal}</strong>"
        f"<span>{_escape(output.prompt_label)}</span>"
        f"<span>Status: {_escape(variant.status)} / {_escape(output.availability)}</span>"
        f"<span>{_escape(output.mime_type)} | {output.bytes if output.bytes is not None else 'n/a'} bytes{_escape(dimensions)}</span>"
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


def _contact_sheet_bytes(
    report: ComparisonReport,
    experiment_root: Path,
) -> bytes:
    items = [
        (variant, output)
        for variant_id in sorted(report.variants)
        for variant in (report.variants[variant_id],)
        for output in sorted(variant.outputs, key=lambda item: item.output_ordinal)
    ]
    count = max(1, len(items))
    columns = min(_CONTACT_COLUMNS, count)
    rows = math.ceil(count / columns)
    canvas = Image.new(
        "RGB",
        (columns * _TILE_WIDTH, rows * _TILE_HEIGHT),
        color=(245, 247, 248),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if not items:
        draw.text((24, 24), "No durable comparison outputs", fill=(25, 30, 35), font=font)
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
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                left = media_box[0] + (_THUMBNAIL_SIZE[0] - image.width) // 2
                top = media_box[1] + (_THUMBNAIL_SIZE[1] - image.height) // 2
                canvas.paste(image, (left, top))
        except OSError:
            _draw_placeholder(draw, media_box, output.media_kind, font)
    else:
        _draw_placeholder(draw, media_box, output.media_kind, font)
    lines = (
        f"Variant: {variant.variant_id}",
        variant.model_lora_summary,
        output.prompt_label,
        f"Output {output.output_ordinal}",
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
