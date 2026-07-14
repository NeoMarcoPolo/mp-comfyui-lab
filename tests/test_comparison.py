from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from mp_comfyui_lab.cli import app
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.comparison.metrics import (
    PHASH_HAMMING_THRESHOLD,
    compare_outputs,
    group_perceptual_duplicates,
)
from mp_comfyui_lab.comparison.render import (
    ComparisonArtifactConflict,
    render_comparison,
)
from mp_comfyui_lab.experiments.models import OutputRecord, RunRecord, RunState
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import SafetyError


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
) -> RunRecord:
    expected = len(outputs) if expected_output_count is None else expected_output_count
    exact_run_hash = hashlib.sha256(variant_id.encode("utf-8")).hexdigest()
    snapshot = {
        "variant_id": variant_id,
        "source_workflow_id": f"workflow-{variant_id}",
        "exact_run_hash": exact_run_hash,
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
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": variant_id},
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
        ("outputs/mixed/run-mixed/clip.mp4", b"video-bytes", "video/mp4"),
        ("outputs/mixed/run-mixed/sound.wav", b"audio-bytes", "audio/wav"),
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
        len(b"video-bytes"),
        len(b"audio-bytes"),
        len(b"unknown-bytes"),
    ]
    assert all(item.provider_metadata.run_id == "run-mixed" for item in facts)


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


def _mixed_report(tmp_path: Path):
    experiment_root = tmp_path / "experiment-compare"
    image_path = experiment_root / "outputs/mixed/run-mixed/frame.png"
    image_bytes = _write_image(image_path)
    media = (
        ("outputs/mixed/run-mixed/frame.png", image_bytes, "image/png"),
        ("outputs/mixed/run-mixed/clip.mp4", b"video-bytes", "video/mp4"),
        ("outputs/mixed/run-mixed/sound.wav", b"audio-bytes", "audio/wav"),
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
    judgment_schema = judgment_input["submission_schema"]["properties"][
        "judgments"
    ]["items"]
    assert judgment_schema["additionalProperties"] is False
    assert judgment_schema["properties"]["identity_fidelity"]["enum"] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert judgment_schema["properties"]["variant_id"]["enum"] == ["mixed"]
    assert judgment_schema["properties"]["referenced_output_ids"]["items"][
        "enum"
    ] == ["mixed/0", "mixed/1", "mixed/2", "mixed/3"]


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
    before = {
        path.name: path.read_bytes()
        for path in first_render.model_dump(mode="python").values()
    }
    second_render = render_comparison(report, experiment_root)
    after = {
        path.name: path.read_bytes()
        for path in second_render.model_dump(mode="python").values()
    }

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
    assert {
        path.name: path.read_bytes()
        for path in first_render.model_dump(mode="python").values()
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


def _persist_cli_comparison(tmp_path: Path) -> Path:
    store = ExperimentStore(tmp_path / "experiments")
    experiment_root = store.create("experiment-compare")
    output_path = experiment_root / "outputs/mixed/run-mixed/frame.png"
    content = _write_image(output_path)
    run = _run(
        "mixed",
        (_output("outputs/mixed/run-mixed/frame.png", content),),
    )
    store.write_json_new(
        "experiment-compare",
        f"runs/{run.run_id}.json",
        run.model_dump(mode="json"),
    )
    return experiment_root


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
        "variant_ids": ["mixed"],
        "output_ids": ["mixed/0"],
        "judgment_input": "comparisons/judgment-input.json",
        "judgments": None,
    }
    assert [item["path"] for item in payload["artifacts"]] == [
        "comparisons/facts.json",
        "comparisons/duplicate-groups.json",
        "comparisons/contact-sheet.jpg",
        "comparisons/index.html",
        "comparisons/judgment-input.json",
    ]
    assert (experiment_root / "comparisons/facts.json").exists()


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
    assert payload["data"]["judgments"] == "comparisons/judgments.json"
    assert payload["artifacts"][-1] == {
        "kind": "judgments",
        "path": "comparisons/judgments.json",
    }
    saved = json.loads(
        (experiment_root / "comparisons/judgments.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == submission
    assert (experiment_root / "comparisons/facts.json").read_bytes() == facts_before
    assert "identity_fidelity" not in json.loads(facts_before)


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
