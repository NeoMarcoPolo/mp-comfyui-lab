from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image
from PIL import ImageDraw
import pytest

from mp_comfyui_lab.cli import app
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.comparison import metrics as comparison_metrics
from mp_comfyui_lab.comparison import render as comparison_render
from mp_comfyui_lab.comparison.judgments import (
    JudgmentValidationError,
    load_and_validate_judgments,
)
from mp_comfyui_lab.comparison.metrics import (
    PHASH_HAMMING_THRESHOLD,
    compare_outputs,
    group_perceptual_duplicates,
)
from mp_comfyui_lab.comparison.render import (
    ComparisonArtifactConflict,
    comparison_facts_sha256,
    render_comparison,
)
from mp_comfyui_lab.discovery.models import ContentClassification
from mp_comfyui_lab.experiments.models import OutputRecord, RunRecord, RunState
from mp_comfyui_lab.experiments.models import ExperimentPlan, PlannedVariant
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.graph.hashing import capability_hash
from mp_comfyui_lab.media import MediaVerificationPolicy
from mp_comfyui_lab.security import SafetyError
from tests.helpers import valid_media_bytes


PLAN_HASH = "a" * 64


def _write_image(path: Path, *, color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path, format="PNG")
    return path.read_bytes()


def _output(
    relative_path: str,
    content: bytes,
    *,
    ordinal: int = 0,
    media_type: str = "image/png",
) -> OutputRecord:
    return OutputRecord(
        ordinal=ordinal,
        provider_order=ordinal,
        node_id=str(9 + ordinal),
        media_type=media_type,
        local_relative_path=relative_path,
        content_sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def _run(
    variant_id: str,
    outputs: tuple[OutputRecord, ...],
    *,
    state: RunState = RunState.SUCCESS,
    expected_output_count: int | None = None,
    error: dict[str, object] | None = None,
    content_classification: ContentClassification = ContentClassification.UNKNOWN,
) -> RunRecord:
    expected = len(outputs) if expected_output_count is None else expected_output_count
    exact_run_hash = hashlib.sha256(variant_id.encode("utf-8")).hexdigest()
    snapshot = {
        "variant_id": variant_id,
        "source_workflow_id": f"workflow-{variant_id}",
        "exact_run_hash": exact_run_hash,
        "content_classification": content_classification.value,
        "materialized_graph": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": f"{variant_id}.safetensors"},
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": f"{variant_id}.lora.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {"model": ["2", 0]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["3", 0],
                    "filename_prefix": variant_id,
                },
            },
        },
        "execution_parameters": {
            "prompt_lines": [f"portrait prompt for {variant_id}"],
        },
    }
    return RunRecord(
        run_id=f"run-{variant_id}",
        experiment_id="experiment-compare",
        variant_id=variant_id,
        state=state,
        plan_hash=PLAN_HASH,
        exact_run_hash=exact_run_hash,
        variant_snapshot=snapshot,
        provider_task_id=(
            f"task-{variant_id}"
            if state in {RunState.SUCCESS, RunState.CANCELLED}
            else None
        ),
        expected_output_count=expected,
        actual_output_count=len(outputs),
        outputs=outputs,
        error=error,
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def _json_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _schema_accepts(value: object, schema: object) -> bool:
    if schema is True:
        return True
    if schema is False or not isinstance(schema, dict):
        return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False
    if "enum" in schema and not any(
        _json_equal(value, item) for item in schema["enum"]
    ):
        return False
    if "not" in schema and _schema_accepts(value, schema["not"]):
        return False
    if "oneOf" in schema and sum(
        _schema_accepts(value, item) for item in schema["oneOf"]
    ) != 1:
        return False
    if "allOf" in schema and not all(
        _schema_accepts(value, item) for item in schema["allOf"]
    ):
        return False
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        if any(
            key in value and not _schema_accepts(value[key], child)
            for key, child in properties.items()
        ):
            return False
    elif expected_type == "array":
        if not isinstance(value, list):
            return False
        if len(value) < schema.get("minItems", 0):
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        if schema.get("uniqueItems") and len(
            {json.dumps(item, sort_keys=True) for item in value}
        ) != len(value):
            return False
        if "items" in schema and any(
            not _schema_accepts(item, schema["items"]) for item in value
        ):
            return False
        if "contains" in schema:
            matches = sum(
                _schema_accepts(item, schema["contains"]) for item in value
            )
            if matches < schema.get("minContains", 1):
                return False
            if "maxContains" in schema and matches > schema["maxContains"]:
                return False
    elif expected_type == "string":
        if not isinstance(value, str):
            return False
        if len(value) < schema.get("minLength", 0):
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return False
    elif expected_type == "integer" and type(value) is not int:
        return False
    return True


def test_compare_groups_perceptual_duplicates_and_keeps_facts_objective(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    first_path = experiment_root / "outputs/variant-a/run-a/0000.png"
    second_path = experiment_root / "outputs/variant-b/0000-9.png"
    first = _write_image(first_path)
    second = _write_image(second_path)
    runs = (
        _run(
            "variant-a",
            (_output("outputs/variant-a/run-a/0000.png", first),),
        ),
        _run(
            "variant-b",
            (_output("outputs/variant-b/0000-9.png", second),),
        ),
    )

    report = compare_outputs(runs, experiment_root)

    assert PHASH_HAMMING_THRESHOLD == 6
    assert report.phash_hamming_threshold == 6
    assert report.duplicate_groups == (("variant-a/0", "variant-b/0"),)
    facts = report.variants["variant-a"].facts
    assert facts.width == 64
    assert facts.height == 64
    assert facts.mode == "RGB"
    assert facts.bytes == len(first)
    assert facts.sha256 == hashlib.sha256(first).hexdigest()
    assert facts.mime_type == "image/png"
    assert len(facts.phash or "") == 16
    assert facts.provider_metadata.local_relative_path == (
        "outputs/variant-a/run-a/0000.png"
    )
    assert facts.provider_metadata.integrity_matches_record is True
    objective_keys = _all_keys(report.model_dump(mode="json"))
    assert not {
        "identity_fidelity",
        "prompt_adherence",
        "visual_quality",
        "creative_usefulness",
        "reasons",
    } & objective_keys


def test_perceptual_duplicate_threshold_includes_six_but_excludes_seven() -> None:
    assert group_perceptual_duplicates(
        {
            "base": "0000000000000000",
            "distance-six": "000000000000003f",
            "distance-seven": "fe00000000000000",
        }
    ) == (("base", "distance-six"),)


def test_compare_records_video_audio_and_unknown_without_phash(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    media = (
        (
            "outputs/mixed/run-mixed/clip.mp4",
            valid_media_bytes("video/mp4"),
            "video/mp4",
        ),
        (
            "outputs/mixed/run-mixed/sound.wav",
            valid_media_bytes("audio/wav"),
            "audio/wav",
        ),
        (
            "outputs/mixed/run-mixed/result.bin",
            b"unknown-bytes",
            "application/octet-stream",
        ),
    )
    outputs = []
    for ordinal, (relative_path, content, media_type) in enumerate(media):
        path = experiment_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        outputs.append(
            _output(
                relative_path,
                content,
                ordinal=ordinal,
                media_type=media_type,
            )
        )

    report = compare_outputs((_run("mixed", tuple(outputs)),), experiment_root)
    facts = report.variants["mixed"].outputs

    assert [item.media_kind for item in facts] == ["video", "audio", "unknown"]
    assert [item.mime_type for item in facts] == [
        "video/mp4",
        "audio/wav",
        "application/octet-stream",
    ]
    assert all(item.phash is None for item in facts)
    assert [item.bytes for item in facts] == [
        len(valid_media_bytes("video/mp4")),
        len(valid_media_bytes("audio/wav")),
        len(b"unknown-bytes"),
    ]
    assert all(item.provider_metadata.run_id == "run-mixed" for item in facts)


def _write_alpha_checkerboard(path: Path, *, alpha: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (64, 64))
    for y in range(64):
        for x in range(64):
            value = 0 if ((x // 32) + (y // 32)) % 2 == 0 else 255
            image.putpixel((x, y), (value, value, value, alpha))
    image.save(path, format="PNG")
    return path.read_bytes()


def test_phash_composites_alpha_on_white_before_luminance(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    transparent_path = (
        experiment_root / "outputs/transparent/run-transparent/frame.png"
    )
    opaque_path = experiment_root / "outputs/opaque/run-opaque/frame.png"
    transparent = _write_alpha_checkerboard(transparent_path, alpha=0)
    opaque = _write_alpha_checkerboard(opaque_path, alpha=255)

    report = compare_outputs(
        (
            _run(
                "transparent",
                (
                    _output(
                        "outputs/transparent/run-transparent/frame.png",
                        transparent,
                    ),
                ),
            ),
            _run(
                "opaque",
                (_output("outputs/opaque/run-opaque/frame.png", opaque),),
            ),
        ),
        experiment_root,
    )

    transparent_hash = report.variants["transparent"].outputs[0].phash
    opaque_hash = report.variants["opaque"].outputs[0].phash
    assert transparent_hash is not None
    assert opaque_hash is not None
    assert transparent_hash != opaque_hash
    assert report.duplicate_groups == ()


def test_contact_sheet_represents_transparency_on_checkerboard(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    path = experiment_root / "outputs/transparent/run-transparent/frame.png"
    content = _write_alpha_checkerboard(path, alpha=0)
    report = compare_outputs(
        (
            _run(
                "transparent",
                (
                    _output(
                        "outputs/transparent/run-transparent/frame.png",
                        content,
                    ),
                ),
            ),
        ),
        experiment_root,
    )

    artifacts = render_comparison(report, experiment_root)

    with Image.open(artifacts.contact_sheet) as sheet:
        thumbnail = sheet.crop((178, 90, 242, 154)).convert("L")
        extrema = thumbnail.getextrema()
        assert extrema[0] >= 150
        assert extrema[1] - extrema[0] >= 15


def test_invalid_images_are_objective_nonavailable_facts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    corrupt = b"not an image"
    valid_path = experiment_root / "valid.png"
    valid = _write_image(valid_path)
    truncated = valid[:32]
    payloads = (
        (
            "outputs/invalid/run-invalid/corrupt.png",
            corrupt,
            "unrecognized_signature",
        ),
        (
            "outputs/invalid/run-invalid/truncated.png",
            truncated,
            "invalid_image",
        ),
        (
            "outputs/invalid/run-invalid/too-many-pixels.png",
            valid,
            "image_pixel_limit_exceeded",
        ),
    )
    outputs = []
    for ordinal, (relative_path, content, _) in enumerate(payloads):
        path = experiment_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        outputs.append(_output(relative_path, content, ordinal=ordinal))
    monkeypatch.setattr(
        comparison_metrics,
        "COMPARISON_MEDIA_POLICY",
        MediaVerificationPolicy(max_image_pixels=1),
        raising=False,
    )

    report = compare_outputs((_run("invalid", tuple(outputs)),), experiment_root)
    facts = report.variants["invalid"].outputs

    assert [item.availability for item in facts] == ["invalid"] * 3
    assert [item.media_error.code for item in facts] == [
        item[2] for item in payloads
    ]
    assert all(item.artifact_relative_path is None for item in facts)
    assert all(item.bytes is not None and item.sha256 is not None for item in facts)
    assert all(item.phash is None for item in facts)
    artifacts = render_comparison(report, experiment_root)
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    html = artifacts.gallery.read_text(encoding="utf-8")
    assert judgment_input["output_ids"] == []
    assert "Status: SUCCESS / invalid" in html
    assert "Artifact invalid" in html
    assert "<img" not in html


def test_comparison_rejects_50mp_image_above_decoding_policy(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    image_path = experiment_root / "outputs/large/run-large/output.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("1", (10_000, 5_000)).save(image_path, format="PNG")
    content = image_path.read_bytes()
    relative_path = image_path.relative_to(experiment_root).as_posix()

    report = compare_outputs(
        (_run("large", (_output(relative_path, content),)),),
        experiment_root,
    )

    facts = report.variants["large"].outputs[0]
    assert facts.availability == "invalid"
    assert facts.media_error.code == "image_pixel_limit_exceeded"
    assert facts.artifact_relative_path is None


def test_comparison_verifies_declared_and_detected_media_types(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    cases = (
        ("video/mp4", "image/png"),
        ("image/png", "image/jpeg"),
        ("audio/wav", "audio/mpeg"),
        ("application/octet-stream", "image/png"),
    )
    outputs = []
    for ordinal, (declared, detected) in enumerate(cases):
        content = valid_media_bytes(detected)
        relative_path = f"outputs/mismatch/run-mismatch/{ordinal:04d}.bin"
        path = experiment_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        outputs.append(
            _output(
                relative_path,
                content,
                ordinal=ordinal,
                media_type=declared,
            )
        )

    report = compare_outputs((_run("mismatch", tuple(outputs)),), experiment_root)
    facts = report.variants["mismatch"].outputs

    assert [item.declared_mime_type for item in facts] == [
        item[0] for item in cases
    ]
    assert [item.detected_mime_type for item in facts] == [
        item[1] for item in cases
    ]
    assert [item.declared_media_kind for item in facts] == [
        "video",
        "image",
        "audio",
        "unknown",
    ]
    assert [item.detected_media_kind for item in facts] == [
        "image",
        "image",
        "audio",
        "image",
    ]
    assert [item.availability for item in facts] == [
        "invalid",
        "invalid",
        "invalid",
        "available",
    ]
    assert [item.media_error.code if item.media_error else None for item in facts] == [
        "declared_mime_mismatch",
        "declared_mime_mismatch",
        "declared_mime_mismatch",
        None,
    ]
    assert facts[3].media_kind == "unknown"
    assert facts[3].phash is None
    artifacts = render_comparison(report, experiment_root)
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    html = artifacts.gallery.read_text(encoding="utf-8")
    assert judgment_input["output_ids"] == ["mismatch/3"]
    assert "<video" not in html
    assert "<audio" not in html
    assert "<img" not in html
    assert html.count("Open local artifact") == 1


def test_compare_gracefully_records_missing_output_from_failed_run(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    recorded = b"bytes-that-are-no-longer-present"
    output = _output(
        "outputs/failed/run-failed/missing.png",
        recorded,
    )
    run = _run(
        "failed",
        (output,),
        state=RunState.FAILED,
        expected_output_count=2,
        error={"kind": "provider_error", "diagnostic": "failed cleanly"},
    )

    report = compare_outputs((run,), experiment_root)

    variant = report.variants["failed"]
    assert variant.status == "FAILED"
    assert variant.error == {
        "diagnostic": "failed cleanly",
        "kind": "provider_error",
    }
    assert variant.outputs[0].availability == "missing"
    assert variant.outputs[0].bytes is None
    assert variant.outputs[0].sha256 is None
    assert variant.outputs[0].phash is None
    assert variant.outputs[0].artifact_relative_path is None


def test_classification_is_objective_and_rendered_for_all_media_states(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    sfw_path = experiment_root / "outputs/sfw/run-sfw/frame.png"
    sfw_bytes = _write_image(sfw_path)
    partial_path = experiment_root / "outputs/partial/run-partial/frame.png"
    partial_bytes = _write_image(partial_path, color=(90, 80, 70))
    missing_bytes = b"missing image bytes"
    runs = (
        _run(
            "sfw",
            (_output("outputs/sfw/run-sfw/frame.png", sfw_bytes),),
            content_classification=ContentClassification.SFW,
        ),
        _run(
            "nsfw",
            (_output("outputs/nsfw/run-nsfw/missing.png", missing_bytes),),
            state=RunState.FAILED,
            error={"kind": "provider_error"},
            content_classification=ContentClassification.NSFW,
        ),
        _run(
            "partial",
            (_output("outputs/partial/run-partial/frame.png", partial_bytes),),
            state=RunState.FAILED,
            expected_output_count=2,
            error={"kind": "partial_output"},
            content_classification=ContentClassification.UNKNOWN,
        ),
    )
    report = compare_outputs(runs, experiment_root)

    assert {
        variant_id: variant.content_classification
        for variant_id, variant in report.variants.items()
    } == {
        "nsfw": ContentClassification.NSFW,
        "partial": ContentClassification.UNKNOWN,
        "sfw": ContentClassification.SFW,
    }
    assert report.variants["nsfw"].outputs[0].availability == "missing"
    assert report.variants["partial"].outputs[1].availability == "not_recorded"

    drawn_text: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(draw, xy, text, *args, **kwargs):
        drawn_text.append(str(text))
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    artifacts = render_comparison(report, experiment_root)
    facts = json.loads(artifacts.facts.read_text(encoding="utf-8"))
    html = artifacts.gallery.read_text(encoding="utf-8")
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )

    assert facts["variants"]["sfw"]["content_classification"] == "SFW"
    assert facts["variants"]["nsfw"]["content_classification"] == "NSFW"
    assert facts["variants"]["partial"]["content_classification"] == "UNKNOWN"
    assert "<dt>Content classification</dt><dd>SFW</dd>" in html
    assert "<dt>Content classification</dt><dd>NSFW</dd>" in html
    assert "<dt>Content classification</dt><dd>UNKNOWN</dd>" in html
    assert "Classification: SFW" in drawn_text
    assert "Classification: NSFW" in drawn_text
    assert "Classification: UNKNOWN" in drawn_text
    assert "Status: FAILED / missing" in drawn_text
    assert "Status: FAILED / not_recorded" in drawn_text
    assert judgment_input["variant_context"] == [
        {
            "content_classification": "NSFW",
            "output_ids": [],
            "status": "FAILED",
            "variant_id": "nsfw",
        },
        {
            "content_classification": "UNKNOWN",
            "output_ids": ["partial/0"],
            "status": "FAILED",
            "variant_id": "partial",
        },
        {
            "content_classification": "SFW",
            "output_ids": ["sfw/0"],
            "status": "SUCCESS",
            "variant_id": "sfw",
        },
    ]


def _mixed_report(tmp_path: Path):
    experiment_root = tmp_path / "experiment-compare"
    image_path = experiment_root / "outputs/mixed/run-mixed/frame.png"
    image_bytes = _write_image(image_path)
    media = (
        ("outputs/mixed/run-mixed/frame.png", image_bytes, "image/png"),
        (
            "outputs/mixed/run-mixed/clip.mp4",
            valid_media_bytes("video/mp4"),
            "video/mp4",
        ),
        (
            "outputs/mixed/run-mixed/sound.wav",
            valid_media_bytes("audio/wav"),
            "audio/wav",
        ),
        (
            "outputs/mixed/run-mixed/result.bin",
            b"unknown-bytes",
            "application/octet-stream",
        ),
    )
    outputs = []
    for ordinal, (relative_path, content, media_type) in enumerate(media):
        path = experiment_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        outputs.append(
            _output(
                relative_path,
                content,
                ordinal=ordinal,
                media_type=media_type,
            )
        )
    return experiment_root, compare_outputs(
        (_run("mixed", tuple(outputs)),),
        experiment_root,
    )


def _objective_artifact_bytes(artifacts) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in (
            artifacts.facts,
            artifacts.duplicate_groups,
            artifacts.contact_sheet,
            artifacts.gallery,
            artifacts.judgment_input,
        )
    }


def test_render_writes_stable_objective_artifacts_and_media_controls(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)

    artifacts = render_comparison(report, experiment_root)

    assert artifacts.facts == experiment_root / "comparisons/facts.json"
    assert artifacts.duplicate_groups == (
        experiment_root / "comparisons/duplicate-groups.json"
    )
    assert artifacts.contact_sheet == (
        experiment_root / "comparisons/contact-sheet.jpg"
    )
    assert artifacts.gallery == experiment_root / "comparisons/index.html"
    assert artifacts.judgment_input == (
        experiment_root / "comparisons/judgment-input.json"
    )
    facts = json.loads(artifacts.facts.read_text(encoding="utf-8"))
    assert facts == report.model_dump(mode="json")
    duplicate_groups = json.loads(
        artifacts.duplicate_groups.read_text(encoding="utf-8")
    )
    assert duplicate_groups == {
        "schema_version": "1",
        "phash_hamming_threshold": 6,
        "duplicate_groups": [],
    }
    html = artifacts.gallery.read_text(encoding="utf-8")
    assert '<img src="../outputs/mixed/run-mixed/frame.png"' in html
    assert '<video controls preload="metadata"' in html
    assert '<audio controls preload="metadata"' in html
    assert "Open local artifact" in html
    assert "Output 0" in html
    assert "Model:" in html
    assert "Prompt 1:" in html
    assert "Status: SUCCESS / available" in html
    with Image.open(artifacts.contact_sheet) as contact_sheet:
        assert contact_sheet.format == "JPEG"
        assert contact_sheet.width >= 320
        assert contact_sheet.height >= 240

    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    assert judgment_input["requested_fields"] == [
        "identity_fidelity",
        "prompt_adherence",
        "visual_quality",
        "creative_usefulness",
        "reasons",
        "referenced_output_ids",
    ]
    assert judgment_input["output_ids"] == [
        "mixed/0",
        "mixed/1",
        "mixed/2",
        "mixed/3",
    ]
    assert "judgments" not in judgment_input
    judgments_schema = judgment_input["submission_schema"]["properties"][
        "judgments"
    ]
    assert judgments_schema["minItems"] == 1
    assert judgments_schema["maxItems"] == 1
    judgment_schema = judgments_schema["items"]["oneOf"][0]
    assert judgment_schema["additionalProperties"] is False
    assert judgment_schema["properties"]["identity_fidelity"]["enum"] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert judgment_schema["properties"]["variant_id"]["const"] == "mixed"
    assert judgment_schema["properties"]["referenced_output_ids"]["items"][
        "enum"
    ] == ["mixed/0", "mixed/1", "mixed/2", "mixed/3"]
    assert judgments_schema["allOf"][0]["contains"]["properties"][
        "variant_id"
    ]["const"] == "mixed"


def test_contact_sheets_paginate_545_outputs_with_bounded_dimensions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    experiment_root.mkdir()
    report = compare_outputs(
        (
            _run(
                "large",
                (),
                state=RunState.FAILED,
                expected_output_count=545,
                error={"kind": "provider_error"},
            ),
        ),
        experiment_root,
    )
    original_new = comparison_render.Image.new
    canvas_sizes: list[tuple[int, int]] = []

    def bounded_canvas(mode, size, *args, **kwargs):
        assert size[0] <= 3 * 420
        assert size[1] <= 22 * 360
        canvas_sizes.append(size)
        return original_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(comparison_render.Image, "new", bounded_canvas)

    artifacts = render_comparison(report, experiment_root)

    expected_names = [
        "contact-sheet.jpg",
        *[f"contact-sheet-{page:04d}.jpg" for page in range(2, 10)],
    ]
    assert artifacts.contact_sheet == artifacts.contact_sheets[0]
    assert [path.name for path in artifacts.contact_sheets] == expected_names
    assert len(canvas_sizes) == 9
    for path in artifacts.contact_sheets:
        with Image.open(path) as page:
            assert page.width <= 3 * 420
            assert page.height <= 22 * 360
    html = artifacts.gallery.read_text(encoding="utf-8")
    assert all(f'href="{name}"' in html for name in expected_names)
    cli_artifacts = app._comparison_artifacts_payload(artifacts)
    assert [
        Path(item["path"]).name
        for item in cli_artifacts
        if item["kind"].startswith("contact_sheet")
    ] == expected_names


def test_later_contact_sheet_page_failure_is_atomic(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    experiment_root.mkdir()
    report = compare_outputs(
        (
            _run(
                "large",
                (),
                state=RunState.FAILED,
                expected_output_count=65,
                error={"kind": "provider_error"},
            ),
        ),
        experiment_root,
    )

    def inject(point: str) -> None:
        if point == "stage:contact-sheet-0002.jpg":
            raise RuntimeError("injected later page failure")

    with pytest.raises(RuntimeError, match="later page failure"):
        render_comparison(report, experiment_root, fault_injector=inject)

    assert not (experiment_root / "comparisons").exists()
    assert not list(experiment_root.glob(".comparison-stage-*"))


def _judgment_for(variant_id: str, output_ids: list[str]) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "identity_fidelity": 5,
        "prompt_adherence": 4,
        "visual_quality": 3,
        "creative_usefulness": 2,
        "reasons": [f"Objective review for {variant_id}"],
        "referenced_output_ids": output_ids,
    }


def _runtime_accepts_submission(
    tmp_path: Path,
    payload: dict[str, object],
    report,
    facts_sha256: str,
) -> bool:
    path = tmp_path / "schema-runtime-submission.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_and_validate_judgments(
            path,
            report,
            expected_facts_sha256=facts_sha256,
        )
    except JudgmentValidationError:
        return False
    return True


def test_judgment_schema_and_runtime_require_each_variant_exactly_once(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    first_path = experiment_root / "outputs/a/run-a/frame.png"
    second_path = experiment_root / "outputs/b/run-b/frame.png"
    first = _write_image(first_path, color=(1, 2, 3))
    second = _write_image(second_path, color=(30, 20, 10))
    report = compare_outputs(
        (
            _run("a", (_output("outputs/a/run-a/frame.png", first),)),
            _run("b", (_output("outputs/b/run-b/frame.png", second),)),
        ),
        experiment_root,
    )
    artifacts = render_comparison(report, experiment_root)
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    schema = judgment_input["submission_schema"]
    facts_sha256 = judgment_input["facts_sha256"]
    valid = {
        "schema_version": "1",
        "experiment_id": "experiment-compare",
        "facts_sha256": facts_sha256,
        "judgments": [
            _judgment_for("a", ["a/0"]),
            _judgment_for("b", ["b/0"]),
        ],
    }

    assert _schema_accepts(valid, schema)
    assert _runtime_accepts_submission(tmp_path, valid, report, facts_sha256)

    invalid_cases = []
    missing = json.loads(json.dumps(valid))
    missing["judgments"] = missing["judgments"][:1]
    invalid_cases.append(missing)
    duplicate = json.loads(json.dumps(valid))
    duplicate["judgments"] = [
        duplicate["judgments"][0],
        duplicate["judgments"][0],
    ]
    invalid_cases.append(duplicate)
    cross_variant = json.loads(json.dumps(valid))
    cross_variant["judgments"][0]["referenced_output_ids"] = ["b/0"]
    invalid_cases.append(cross_variant)
    padded_variant = json.loads(json.dumps(valid))
    padded_variant["judgments"][0]["variant_id"] = " a "
    invalid_cases.append(padded_variant)
    padded_output = json.loads(json.dumps(valid))
    padded_output["judgments"][0]["referenced_output_ids"] = [" a/0 "]
    invalid_cases.append(padded_output)
    duplicate_references = json.loads(json.dumps(valid))
    duplicate_references["judgments"][0]["referenced_output_ids"] = [
        "a/0",
        "a/0",
    ]
    invalid_cases.append(duplicate_references)
    empty = json.loads(json.dumps(valid))
    empty["judgments"] = []
    invalid_cases.append(empty)
    out_of_range = json.loads(json.dumps(valid))
    out_of_range["judgments"][0]["visual_quality"] = 6
    invalid_cases.append(out_of_range)
    unknown_field = json.loads(json.dumps(valid))
    unknown_field["judgments"][0]["surprise"] = True
    invalid_cases.append(unknown_field)

    for invalid in invalid_cases:
        assert not _schema_accepts(invalid, schema)
        assert not _runtime_accepts_submission(
            tmp_path,
            invalid,
            report,
            facts_sha256,
        )


def test_empty_report_schema_and_runtime_accept_exactly_zero_judgments(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-empty"
    experiment_root.mkdir()
    report = compare_outputs(
        (),
        experiment_root,
        experiment_id="experiment-empty",
    )
    artifacts = render_comparison(report, experiment_root)
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": "1",
        "experiment_id": "experiment-empty",
        "facts_sha256": judgment_input["facts_sha256"],
        "judgments": [],
    }

    assert _schema_accepts(payload, judgment_input["submission_schema"])
    assert _runtime_accepts_submission(
        tmp_path,
        payload,
        report,
        judgment_input["facts_sha256"],
    )


def test_variant_without_available_outputs_accepts_empty_references(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    experiment_root.mkdir()
    report = compare_outputs(
        (
            _run(
                "failed",
                (),
                state=RunState.FAILED,
                expected_output_count=1,
                error={"kind": "provider_error"},
            ),
        ),
        experiment_root,
    )
    artifacts = render_comparison(report, experiment_root)
    judgment_input = json.loads(
        artifacts.judgment_input.read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": "1",
        "experiment_id": "experiment-compare",
        "facts_sha256": judgment_input["facts_sha256"],
        "judgments": [_judgment_for("failed", [])],
    }

    assert _schema_accepts(payload, judgment_input["submission_schema"])
    assert _runtime_accepts_submission(
        tmp_path,
        payload,
        report,
        judgment_input["facts_sha256"],
    )


def test_gallery_escapes_all_untrusted_values_and_rejects_unsafe_links(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    variant = report.variants["mixed"]
    hostile = '<script>alert("x")</script>'
    output = variant.outputs[0].model_copy(
        update={
            "prompt_label": hostile,
            "artifact_relative_path": "https://evil.example/embed.png",
        }
    )
    malicious = variant.model_copy(
        update={
            "label": hostile,
            "source_workflow_id": hostile,
            "author": hostile,
            "model_lora_summary": hostile,
            "prompt_label": hostile,
            "error": {"diagnostic": hostile},
            "outputs": (output, *variant.outputs[1:]),
        }
    )
    unsafe_report = report.model_copy(
        update={"variants": {"mixed": malicious}}
    )

    artifacts = render_comparison(unsafe_report, experiment_root)
    html = artifacts.gallery.read_text(encoding="utf-8")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "https://evil.example" not in html
    assert "javascript:" not in html.casefold()
    assert "data:" not in html.casefold()
    assert "remote" not in html.casefold()
    assert "Artifact unavailable" in html


def test_gallery_rejects_traversal_even_when_outside_file_matches_facts(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    source_path = experiment_root / "outputs/safe/run-safe/frame.png"
    content = _write_image(source_path)
    outside_path = tmp_path / "outside.png"
    outside_path.write_bytes(content)
    report = compare_outputs(
        (_run("safe", (_output("outputs/safe/run-safe/frame.png", content),)),),
        experiment_root,
    )
    variant = report.variants["safe"]
    output = variant.outputs[0]
    provider = output.provider_metadata.model_copy(
        update={"local_relative_path": "../outside.png"}
    )
    traversal_output = output.model_copy(
        update={
            "artifact_relative_path": "../outside.png",
            "provider_metadata": provider,
        }
    )
    traversal_report = report.model_copy(
        update={
            "variants": {
                "safe": variant.model_copy(update={"outputs": (traversal_output,)})
            }
        }
    )

    html = render_comparison(traversal_report, experiment_root).gallery.read_text(
        encoding="utf-8"
    )

    assert "../outside.png" not in html
    assert "Artifact unavailable" in html


def test_stored_output_symlink_is_unsafe_and_never_rendered(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    outside_path = tmp_path / "outside.png"
    content = _write_image(outside_path)
    stored_path = experiment_root / "outputs/symlink/run-symlink/frame.png"
    stored_path.parent.mkdir(parents=True)
    try:
        os.symlink(outside_path, stored_path)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    report = compare_outputs(
        (
            _run(
                "symlink",
                (_output("outputs/symlink/run-symlink/frame.png", content),),
            ),
        ),
        experiment_root,
    )

    output = report.variants["symlink"].outputs[0]
    html = render_comparison(report, experiment_root).gallery.read_text(
        encoding="utf-8"
    )

    assert output.availability == "unsafe"
    assert output.artifact_relative_path is None
    assert "<img" not in html
    assert "Artifact unavailable" in html


def test_render_rejects_symlinked_comparisons_publish_target(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    outside = tmp_path / "outside-comparisons"
    outside.mkdir()
    try:
        os.symlink(
            outside,
            experiment_root / "comparisons",
            target_is_directory=True,
        )
    except OSError as error:
        pytest.skip(f"directory symlink creation unavailable: {error}")

    with pytest.raises(ComparisonArtifactConflict, match="safe local directory"):
        render_comparison(report, experiment_root)

    assert list(outside.iterdir()) == []


def test_render_rejects_symlinked_existing_judgments_artifact(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    artifacts = render_comparison(report, experiment_root)
    outside = tmp_path / "outside-judgments.json"
    outside.write_text("{}\n", encoding="utf-8")
    try:
        os.symlink(outside, artifacts.facts.parent / "judgments.json")
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(ComparisonArtifactConflict, match="link or junction"):
        render_comparison(report, experiment_root)


def test_integrity_mismatch_is_factual_but_never_linked_or_grouped(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    path = experiment_root / "outputs/tampered/run-tampered/frame.png"
    original = _write_image(path, color=(1, 2, 3))
    record = _output(
        "outputs/tampered/run-tampered/frame.png",
        original,
    )
    _write_image(path, color=(200, 100, 50))

    report = compare_outputs((_run("tampered", (record,)),), experiment_root)
    output = report.variants["tampered"].outputs[0]

    assert output.availability == "integrity_mismatch"
    assert output.artifact_relative_path is None
    assert output.provider_metadata.integrity_matches_record is False
    assert report.duplicate_groups == ()
    html = render_comparison(report, experiment_root).gallery.read_text(
        encoding="utf-8"
    )
    assert "<img" not in html
    assert "Artifact unavailable" in html


def test_render_is_stably_ordered_idempotent_and_never_clobbers(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiment-compare"
    first_path = experiment_root / "outputs/variant-a/run-a/0000.png"
    second_path = experiment_root / "outputs/variant-b/0000.png"
    first = _write_image(first_path, color=(1, 2, 3))
    second = _write_image(second_path, color=(200, 150, 100))
    report = compare_outputs(
        (
            _run(
                "variant-b",
                (_output("outputs/variant-b/0000.png", second),),
            ),
            _run(
                "variant-a",
                (_output("outputs/variant-a/run-a/0000.png", first),),
            ),
        ),
        experiment_root,
    )

    first_render = render_comparison(report, experiment_root)
    before = _objective_artifact_bytes(first_render)
    second_render = render_comparison(report, experiment_root)
    after = _objective_artifact_bytes(second_render)

    assert after == before
    html = first_render.gallery.read_text(encoding="utf-8")
    assert html.index("variant-a") < html.index("variant-b")

    changed_variant = report.variants["variant-a"].model_copy(
        update={"label": "changed label"}
    )
    changed_report = report.model_copy(
        update={
            "variants": {
                **report.variants,
                "variant-a": changed_variant,
            }
        }
    )
    with pytest.raises(ComparisonArtifactConflict, match="would clobber"):
        render_comparison(changed_report, experiment_root)
    assert _objective_artifact_bytes(first_render) == before


@pytest.mark.parametrize(
    "fault_point",
    [
        "stage:facts.json",
        "stage:duplicate-groups.json",
        "stage:contact-sheet.jpg",
        "stage:index.html",
        "stage:judgment-input.json",
        "stage:judgments.json",
        "fsync:staging",
        "publish:comparisons",
    ],
)
def test_fresh_render_faults_leave_no_comparison_directory(
    tmp_path: Path,
    fault_point: str,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)

    def inject(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(f"injected failure at {point}")

    with pytest.raises(RuntimeError, match="injected failure"):
        render_comparison(
            report,
            experiment_root,
            judgment_document=b'{"validated":true}\n',
            fault_injector=inject,
        )

    assert not (experiment_root / "comparisons").exists()
    assert not list(experiment_root.glob(".comparison-stage-*"))


def test_render_fails_closed_on_preexisting_partial_artifact_set(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    comparisons = experiment_root / "comparisons"
    comparisons.mkdir()
    partial = comparisons / "facts.json"
    partial.write_bytes(b"partial facts")

    with pytest.raises(ComparisonArtifactConflict, match="incomplete"):
        render_comparison(report, experiment_root)

    assert list(comparisons.iterdir()) == [partial]
    assert partial.read_bytes() == b"partial facts"


def test_render_accepts_byte_identical_concurrent_directory_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    def concurrent_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        concurrent_publish,
    )
    artifacts = render_comparison(report, experiment_root)

    assert artifacts.facts.read_bytes()
    assert set(path.name for path in artifacts.facts.parent.iterdir()) == {
        "contact-sheet.jpg",
        "duplicate-groups.json",
        "facts.json",
        "index.html",
        "judgment-input.json",
    }


def test_judged_render_appends_after_compatible_plain_concurrent_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    judgment_document = b'{"writer":"judged"}\n'

    def concurrent_plain_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        (destination / "judgments.json").unlink()
        raise FileExistsError("plain comparison won")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        concurrent_plain_publish,
    )

    artifacts = render_comparison(
        report,
        experiment_root,
        judgment_document=judgment_document,
    )

    assert artifacts.judgments is not None
    assert artifacts.judgments.read_bytes() == judgment_document
    assert not list(experiment_root.glob(".comparison-stage-*"))


def test_plain_render_accepts_compatible_judged_concurrent_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    judgment_document = b'{"writer":"judged"}\n'

    def concurrent_judged_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        (destination / "judgments.json").write_bytes(judgment_document)
        raise FileExistsError("judged comparison won")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        concurrent_judged_publish,
    )

    artifacts = render_comparison(report, experiment_root)

    assert artifacts.judgments is not None
    assert artifacts.judgments.read_bytes() == judgment_document


def test_identical_concurrent_judged_writer_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    judgment_document = b'{"writer":"same"}\n'

    def identical_judged_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        raise FileExistsError("identical judged comparison won")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        identical_judged_publish,
    )

    artifacts = render_comparison(
        report,
        experiment_root,
        judgment_document=judgment_document,
    )

    assert artifacts.judgments is not None
    assert artifacts.judgments.read_bytes() == judgment_document


def test_conflicting_concurrent_judged_writer_wins_without_clobber(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    submitted = b'{"writer":"submitted"}\n'
    winner = b'{"writer":"winner"}\n'

    def conflicting_judged_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        (destination / "judgments.json").write_bytes(winner)
        raise FileExistsError("different judged comparison won")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        conflicting_judged_publish,
    )

    with pytest.raises(ComparisonArtifactConflict, match="judgments conflict"):
        render_comparison(
            report,
            experiment_root,
            judgment_document=submitted,
        )

    assert (experiment_root / "comparisons/judgments.json").read_bytes() == winner


def test_render_rejects_conflicting_concurrent_directory_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)

    def concurrent_publish(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        (destination / "facts.json").write_bytes(b"conflict")
        raise FileExistsError("concurrent winner")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        concurrent_publish,
    )

    with pytest.raises(ComparisonArtifactConflict, match="concurrent"):
        render_comparison(report, experiment_root)

    assert (experiment_root / "comparisons/facts.json").read_bytes() == b"conflict"
    assert not list(experiment_root.glob(".comparison-stage-*"))


def test_render_rejects_concurrent_partial_directory_winner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)

    def concurrent_partial(source: Path, destination: Path) -> None:
        destination.mkdir()
        raise FileExistsError("concurrent partial winner")

    monkeypatch.setattr(
        comparison_render,
        "_rename_directory_new",
        concurrent_partial,
    )

    with pytest.raises(ComparisonArtifactConflict, match="incomplete"):
        render_comparison(report, experiment_root)

    assert list((experiment_root / "comparisons").iterdir()) == []
    assert not list(experiment_root.glob(".comparison-stage-*"))


def test_existing_objective_set_survives_judgment_publish_failure(
    tmp_path: Path,
) -> None:
    experiment_root, report = _mixed_report(tmp_path)
    artifacts = render_comparison(report, experiment_root)
    before = {
        path.name: path.read_bytes()
        for path in artifacts.facts.parent.iterdir()
    }

    def inject(point: str) -> None:
        if point == "publish:judgments.json":
            raise RuntimeError("injected judgment failure")

    with pytest.raises(RuntimeError, match="injected judgment failure"):
        render_comparison(
            report,
            experiment_root,
            judgment_document=b'{"validated":true}\n',
            fault_injector=inject,
        )

    assert {
        path.name: path.read_bytes()
        for path in artifacts.facts.parent.iterdir()
    } == before


def test_render_rejects_report_identity_mismatch_before_any_writes(
    tmp_path: Path,
) -> None:
    _, report = _mixed_report(tmp_path)
    wrong_root = tmp_path / "other-experiment"
    wrong_root.mkdir()

    with pytest.raises(SafetyError, match="identity does not match"):
        render_comparison(report, wrong_root)

    assert not (wrong_root / "comparisons").exists()


def _persist_cli_comparison(
    tmp_path: Path,
    *,
    experiments_root: Path | None = None,
) -> Path:
    store = ExperimentStore(experiments_root or (tmp_path / "experiments"))
    experiment_root = store.create("experiment-compare")
    output_path = experiment_root / "outputs/mixed/run-mixed/frame.png"
    content = _write_image(output_path)
    initial_run = _run(
        "mixed",
        (_output("outputs/mixed/run-mixed/frame.png", content),),
    )
    snapshot = dict(initial_run.variant_snapshot or {})
    graph = snapshot["materialized_graph"]
    draft = PlannedVariant(
        variant_id="mixed",
        source_workflow_id="workflow-mixed",
        source_hash="b" * 64,
        materialized_graph=graph,
        content_classification=ContentClassification.UNKNOWN,
        exact_run_hash="0" * 64,
        capability_hash=capability_hash(graph),
        expected_output_count=1,
        output_node_ids=("9",),
        expected_media_types=("image/png",),
        execution_parameters={
            "prompt_lines": ["portrait prompt for mixed"],
            "expected_output_count": 1,
            "output_node_ids": ["9"],
        },
    )
    variant = PlannedVariant.model_validate(
        {
            **draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(draft),
        }
    )
    approval_payload = {
        "source_hashes": [variant.source_hash],
        "variants": [variant.model_dump(mode="json")],
        "capability_coverage": [],
        "warnings": [],
    }
    plan_hash = hashlib.sha256(
        json.dumps(
            approval_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    plan = ExperimentPlan(
        plan_hash=plan_hash,
        source_hashes=(variant.source_hash,),
        variants=(variant,),
        capability_coverage=(),
        warnings=(),
        approval_payload=approval_payload,
    )
    run = initial_run.model_copy(
        update={
            "plan_hash": plan.plan_hash,
            "exact_run_hash": variant.exact_run_hash,
            "variant_snapshot": variant.model_dump(mode="json"),
            "expected_output_node_ids": variant.output_node_ids,
            "expected_media_types": variant.expected_media_types,
        }
    )
    store.write_json_new(
        "experiment-compare",
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    store.write_json_new(
        "experiment-compare",
        f"runs/{run.run_id}.json",
        run.model_dump(mode="json"),
    )
    return experiment_root


def test_compare_cli_rejects_tampered_classification_snapshot(
    tmp_path: Path,
    capsys,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    run_path = experiment_root / "runs/run-mixed.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["variant_snapshot"]["content_classification"] = "NSFW"
    run_path.write_text(json.dumps(payload), encoding="utf-8")

    code = main(["compare", "experiment-compare"], cwd=tmp_path)
    response = _cli_payload(capsys)

    assert code == ExitCode.SAFETY
    assert "snapshot" in response["error"]["message"]
    assert not (experiment_root / "comparisons").exists()


@pytest.mark.parametrize("stored_value", [None, "unexpected"])
def test_comparison_defaults_missing_or_unrecognized_classification_to_unknown(
    tmp_path: Path,
    stored_value: str | None,
) -> None:
    run = _run(
        "legacy",
        (),
        state=RunState.FAILED,
        expected_output_count=1,
        error={"kind": "legacy_failure"},
    )
    snapshot = dict(run.variant_snapshot or {})
    if stored_value is None:
        snapshot.pop("content_classification")
    else:
        snapshot["content_classification"] = stored_value
    legacy = run.model_copy(update={"variant_snapshot": snapshot})

    report = compare_outputs((legacy,), tmp_path / "experiment-compare")

    assert (
        report.variants["legacy"].content_classification
        is ContentClassification.UNKNOWN
    )


def _cli_payload(capsys) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _valid_judgments(experiment_root: Path) -> dict[str, object]:
    judgment_input = json.loads(
        (experiment_root / "comparisons/judgment-input.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": "1",
        "experiment_id": "experiment-compare",
        "facts_sha256": judgment_input["facts_sha256"],
        "judgments": [
            {
                "variant_id": "mixed",
                "identity_fidelity": 5,
                "prompt_adherence": 4,
                "visual_quality": 3,
                "creative_usefulness": 2,
                "reasons": ["Strong identity continuity", "Useful framing"],
                "referenced_output_ids": ["mixed/0"],
            }
        ],
    }


def _fresh_valid_judgments(experiment_root: Path) -> dict[str, object]:
    run_path = experiment_root / "runs/run-mixed.json"
    run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    report = compare_outputs((run,), experiment_root)
    return {
        "schema_version": "1",
        "experiment_id": "experiment-compare",
        "facts_sha256": comparison_facts_sha256(report),
        "judgments": [
            {
                "variant_id": "mixed",
                "identity_fidelity": 5,
                "prompt_adherence": 4,
                "visual_quality": 3,
                "creative_usefulness": 2,
                "reasons": ["Strong identity continuity"],
                "referenced_output_ids": ["mixed/0"],
            }
        ],
    }


def test_compare_cli_is_store_only_and_returns_judgment_input(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)

    def forbidden_executor():
        raise AssertionError("compare constructed RunningHubExecutor")

    monkeypatch.setattr(app, "RunningHubExecutor", forbidden_executor)
    code = main(["compare", "experiment-compare"], cwd=tmp_path)
    payload = _cli_payload(capsys)

    assert code == ExitCode.OK
    assert payload["command"] == "compare"
    assert payload["data"] == {
        "experiment_id": "experiment-compare",
        "experiment_root": str(experiment_root.resolve()),
        "variant_ids": ["mixed"],
        "output_ids": ["mixed/0"],
        "facts": str((experiment_root / "comparisons/facts.json").resolve()),
        "duplicate_groups": str(
            (experiment_root / "comparisons/duplicate-groups.json").resolve()
        ),
        "contact_sheet": str(
            (experiment_root / "comparisons/contact-sheet.jpg").resolve()
        ),
        "contact_sheets": [
            str((experiment_root / "comparisons/contact-sheet.jpg").resolve())
        ],
        "gallery": str((experiment_root / "comparisons/index.html").resolve()),
        "judgment_input": str(
            (experiment_root / "comparisons/judgment-input.json").resolve()
        ),
        "judgments": None,
    }
    assert [item["path"] for item in payload["artifacts"]] == [
        str((experiment_root / "comparisons/facts.json").resolve()),
        str((experiment_root / "comparisons/duplicate-groups.json").resolve()),
        str((experiment_root / "comparisons/contact-sheet.jpg").resolve()),
        str((experiment_root / "comparisons/index.html").resolve()),
        str((experiment_root / "comparisons/judgment-input.json").resolve()),
    ]
    assert (experiment_root / "comparisons/facts.json").exists()


def test_compare_cli_returns_rendered_absolute_paths_from_custom_store(
    tmp_path: Path,
    capsys,
) -> None:
    caller_root = tmp_path / "unrelated-caller"
    caller_root.mkdir()
    custom_store = tmp_path / "custom-artifact-store"
    (caller_root / "config.local.yaml").write_text(
        f"experiments_root: {json.dumps(str(custom_store))}\n",
        encoding="utf-8",
    )
    experiment_root = _persist_cli_comparison(
        tmp_path,
        experiments_root=custom_store,
    )

    assert main(["compare", "experiment-compare"], cwd=caller_root) == ExitCode.OK
    initial = _cli_payload(capsys)
    expected = {
        "experiment_root": experiment_root.resolve(),
        "facts": (experiment_root / "comparisons/facts.json").resolve(),
        "duplicate_groups": (
            experiment_root / "comparisons/duplicate-groups.json"
        ).resolve(),
        "contact_sheet": (
            experiment_root / "comparisons/contact-sheet.jpg"
        ).resolve(),
        "gallery": (experiment_root / "comparisons/index.html").resolve(),
        "judgment_input": (
            experiment_root / "comparisons/judgment-input.json"
        ).resolve(),
    }
    for key, path in expected.items():
        assert Path(initial["data"][key]) == path
        assert Path(initial["data"][key]).is_absolute()
    assert initial["data"]["judgments"] is None
    assert [Path(path) for path in initial["data"]["contact_sheets"]] == [
        expected["contact_sheet"]
    ]
    assert [Path(item["path"]) for item in initial["artifacts"]] == [
        expected["facts"],
        expected["duplicate_groups"],
        expected["contact_sheet"],
        expected["gallery"],
        expected["judgment_input"],
    ]

    submission = _valid_judgments(experiment_root)
    submission_path = caller_root / "judgments.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    assert main(
        ["compare", "experiment-compare", "--judgments", "judgments.json"],
        cwd=caller_root,
    ) == ExitCode.OK
    judged = _cli_payload(capsys)
    judgments = (experiment_root / "comparisons/judgments.json").resolve()

    assert Path(judged["data"]["judgments"]) == judgments
    assert Path(judged["artifacts"][-1]["path"]) == judgments


def test_compare_cli_validates_and_saves_subjective_judgments_separately(
    tmp_path: Path,
    capsys,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    assert main(["compare", "experiment-compare"], cwd=tmp_path) == ExitCode.OK
    capsys.readouterr()
    facts_before = (experiment_root / "comparisons/facts.json").read_bytes()
    submission = _valid_judgments(experiment_root)
    submission_path = tmp_path / "judgments.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    code = main(
        [
            "compare",
            "experiment-compare",
            "--judgments",
            str(submission_path),
        ],
        cwd=tmp_path,
    )
    payload = _cli_payload(capsys)

    assert code == ExitCode.OK
    judgments_path = (experiment_root / "comparisons/judgments.json").resolve()
    assert payload["data"]["judgments"] == str(judgments_path)
    assert payload["artifacts"][-1] == {
        "kind": "judgments",
        "path": str(judgments_path),
    }
    saved = json.loads(
        (experiment_root / "comparisons/judgments.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == submission
    assert (experiment_root / "comparisons/facts.json").read_bytes() == facts_before
    assert "identity_fidelity" not in json.loads(facts_before)


def test_fresh_compare_with_invalid_judgments_leaves_no_comparisons_directory(
    tmp_path: Path,
    capsys,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    assert not (experiment_root / "comparisons").exists()
    submission = _fresh_valid_judgments(experiment_root)
    submission["facts_sha256"] = "0" * 64
    submission_path = tmp_path / "invalid-fresh-judgments.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    code = main(
        ["compare", "experiment-compare", "--judgments", str(submission_path)],
        cwd=tmp_path,
    )
    payload = _cli_payload(capsys)

    assert code == ExitCode.USAGE
    assert payload["error"]["kind"] == "usage"
    assert not (experiment_root / "comparisons").exists()


def test_fresh_compare_stages_valid_judgments_with_objective_set(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    submission = _fresh_valid_judgments(experiment_root)
    submission_path = tmp_path / "valid-fresh-judgments.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    original_render = app.render_comparison
    observed: dict[str, bytes | None] = {}

    def recording_render(report, root, **kwargs):
        observed["judgment_document"] = kwargs.get("judgment_document")
        return original_render(report, root, **kwargs)

    monkeypatch.setattr(app, "render_comparison", recording_render)
    code = main(
        ["compare", "experiment-compare", "--judgments", str(submission_path)],
        cwd=tmp_path,
    )
    _cli_payload(capsys)

    assert code == ExitCode.OK
    assert observed["judgment_document"] is not None
    assert (experiment_root / "comparisons/judgments.json").exists()


def _invalid_submission(
    valid: dict[str, object],
    case: str,
) -> dict[str, object]:
    payload = json.loads(json.dumps(valid))
    judgment = payload["judgments"][0]
    if case == "experiment":
        payload["experiment_id"] = "different-experiment"
    elif case == "facts":
        payload["facts_sha256"] = "0" * 64
    elif case == "unknown_top":
        payload["surprise"] = True
    elif case == "unknown_nested":
        judgment["surprise"] = True
    elif case == "score_low":
        judgment["visual_quality"] = 0
    elif case == "score_high":
        judgment["visual_quality"] = 6
    elif case == "score_bool":
        judgment["visual_quality"] = True
    elif case == "score_string":
        judgment["visual_quality"] = "5"
    elif case == "variant":
        judgment["variant_id"] = "unknown"
    elif case == "output":
        judgment["referenced_output_ids"] = ["mixed/99"]
    elif case == "cross_variant_output":
        judgment["variant_id"] = "unknown"
        judgment["referenced_output_ids"] = ["mixed/0"]
    elif case == "duplicate_refs":
        judgment["referenced_output_ids"] = ["mixed/0", "mixed/0"]
    elif case == "duplicate_variant":
        payload["judgments"].append(json.loads(json.dumps(judgment)))
    elif case == "empty_reasons":
        judgment["reasons"] = []
    elif case == "blank_reason":
        judgment["reasons"] = ["   "]
    else:
        raise AssertionError(f"unknown invalid case {case}")
    return payload


@pytest.mark.parametrize(
    "case",
    [
        "experiment",
        "facts",
        "unknown_top",
        "unknown_nested",
        "score_low",
        "score_high",
        "score_bool",
        "score_string",
        "variant",
        "output",
        "cross_variant_output",
        "duplicate_refs",
        "duplicate_variant",
        "empty_reasons",
        "blank_reason",
    ],
)
def test_invalid_judgments_make_no_partial_writes_or_fact_changes(
    tmp_path: Path,
    capsys,
    case: str,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    assert main(["compare", "experiment-compare"], cwd=tmp_path) == ExitCode.OK
    capsys.readouterr()
    comparisons = experiment_root / "comparisons"
    before = {path.name: path.read_bytes() for path in comparisons.iterdir()}
    submission = _invalid_submission(_valid_judgments(experiment_root), case)
    submission_path = tmp_path / "invalid-judgments.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")

    code = main(
        [
            "compare",
            "experiment-compare",
            "--judgments",
            str(submission_path),
        ],
        cwd=tmp_path,
    )
    payload = _cli_payload(capsys)

    assert code == ExitCode.USAGE
    assert payload["error"]["kind"] == "usage"
    assert {path.name: path.read_bytes() for path in comparisons.iterdir()} == before
    assert not (comparisons / "judgments.json").exists()


def test_judgment_json_with_duplicate_object_keys_is_rejected(
    tmp_path: Path,
    capsys,
) -> None:
    experiment_root = _persist_cli_comparison(tmp_path)
    assert main(["compare", "experiment-compare"], cwd=tmp_path) == ExitCode.OK
    capsys.readouterr()
    valid = _valid_judgments(experiment_root)
    duplicate_keys = json.dumps(valid).replace(
        '"identity_fidelity": 5',
        '"identity_fidelity": 5, "identity_fidelity": 4',
    )
    submission_path = tmp_path / "duplicate-keys.json"
    submission_path.write_text(duplicate_keys, encoding="utf-8")

    code = main(
        [
            "compare",
            "experiment-compare",
            "--judgments",
            str(submission_path),
        ],
        cwd=tmp_path,
    )

    payload = _cli_payload(capsys)
    assert code == ExitCode.USAGE
    assert "duplicate JSON object key" in payload["error"]["message"]
    assert not (experiment_root / "comparisons/judgments.json").exists()
