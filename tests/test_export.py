from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import stat

from PIL import Image
import pytest

from mp_comfyui_lab.comparison.finalists import (
    ArtifactIntegrityError,
    StaleSelectionError,
    regenerate_review_artifacts,
    select_finalist,
)
from mp_comfyui_lab.comparison.judgments import JudgmentSubmission
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.experiments.models import (
    ApprovedSourceProvenance,
    DimensionName,
    ExperimentPlan,
    FixtureFingerprint,
    OutputRecord,
    PlannedVariant,
    PlanningConfidence,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.export.bundle import (
    BundleConflictError,
    BundleExporter,
    BundleIntegrityError,
    SelectionRequiredError,
    suggested_workflow_name,
)
import mp_comfyui_lab.export.bundle as bundle_module
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash


EXPERIMENT_ID = "exp-export"
VARIANT_ID = "candidate"


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _graph(checkpoint: str) -> dict[str, object]:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": "portrait.png"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "multiangle"},
        },
    }


def _snapshot_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _exported_review_contract(
    bundle: Path,
) -> tuple[Path, dict[str, object]]:
    manifest_path = (bundle / "review-manifest.json").resolve(strict=True)
    manifest_path.relative_to(bundle.resolve(strict=True))
    return manifest_path, json.loads(manifest_path.read_bytes())


def _write_png(path: Path, color: tuple[int, int, int]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path, format="PNG")
    return path.read_bytes()


def _persist_selected_experiment(tmp_path: Path) -> tuple[ExperimentStore, Path]:
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create(EXPERIMENT_ID)
    source_graph = _graph("qwen_image_edit_2511_base.safetensors")
    selected_graph = _graph("qwen_image_edit_2511.safetensors")
    fixture_source = tmp_path / "approved" / "portrait.png"
    fixture_bytes = _write_png(fixture_source, (20, 40, 60))
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    fixture = FixtureFingerprint(
        uri="portrait.png",
        binding_index=0,
        size=len(fixture_bytes),
        media_type="image/png",
        sha256=fixture_sha256,
    )
    semantic_patch = {
        "mutations": [
            {
                "op": "set_node_input",
                "node_id": "1",
                "field_name": "ckpt_name",
                "value": "qwen_image_edit_2511.safetensors",
            }
        ]
    }
    draft = PlannedVariant(
        variant_id=VARIANT_ID,
        source_workflow_id="1980860923768426497",
        source_hash=source_graph_hash(source_graph),
        semantic_patch=semantic_patch,
        diff={"nodes": []},
        materialized_graph=selected_graph,
        hypothesis=DimensionName.CHECKPOINT,
        evidence_ids=("evidence-checkpoint",),
        confidence=PlanningConfidence.OBSERVED,
        exact_run_hash="0" * 64,
        capability_hash=capability_hash(selected_graph),
        expected_output_count=1,
        output_node_ids=("9",),
        expected_media_types=("image/png",),
        execution_parameters={
            "prompt_lines": ["multiangle consistency"],
            "seed": 42,
            "expected_output_count": 1,
            "output_node_ids": ["9"],
        },
        fixture_fingerprints=(fixture,),
    )
    variant = PlannedVariant.model_validate(
        {
            **draft.model_dump(mode="python"),
            "exact_run_hash": recompute_exact_run_hash(draft),
        }
    )
    approval_payload = {
        "source_hashes": [variant.source_hash],
        "source_provenance": [
            {
                "workflow_id": variant.source_workflow_id,
                "canonical_url": (
                    "https://www.runninghub.ai/post/"
                    f"{variant.source_workflow_id}"
                ),
                "source_hash": variant.source_hash,
                "capability_hash": capability_hash(source_graph),
                "evidence_ids": [variant.source_workflow_id],
                "engagement": 0,
            }
        ],
        "variants": [variant.model_dump(mode="json")],
        "capability_coverage": [DimensionName.CHECKPOINT.value],
        "warnings": [],
        "brief": {
            "goal": "multiangle consistency",
            "partial_behavior_is_tested_capability": False,
        },
    }
    plan = ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=(variant.source_hash,),
        source_provenance=(
            ApprovedSourceProvenance.model_validate(
                approval_payload["source_provenance"][0]
            ),
        ),
        variants=(variant,),
        capability_coverage=(DimensionName.CHECKPOINT,),
        warnings=(),
        approval_payload=approval_payload,
    )
    store.write_text_new(
        EXPERIMENT_ID,
        "brief.yaml",
        "goal: multiangle consistency\n",
    )
    store.write_json_new(
        EXPERIMENT_ID,
        "sources/source-bundle.json",
        {
            "sources": [
                {
                    "workflow_id": variant.source_workflow_id,
                    "canonical_url": (
                        "https://www.runninghub.ai/post/"
                        f"{variant.source_workflow_id}"
                    ),
                    "source_hash": variant.source_hash,
                    "capability_hash": capability_hash(source_graph),
                    "raw_graph": source_graph,
                }
            ],
            "candidates": [],
        },
    )
    store.write_json_new(
        EXPERIMENT_ID,
        f"variants/{VARIANT_ID}.json",
        variant.model_dump(mode="json"),
    )
    store.write_json_new(
        EXPERIMENT_ID,
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    run_id = "run-selected"
    fixture_snapshot = (
        root
        / "runs"
        / run_id
        / "fixtures"
        / f"0000-{fixture_sha256[:16]}.png"
    )
    fixture_snapshot.parent.mkdir(parents=True, exist_ok=True)
    fixture_snapshot.write_bytes(fixture_bytes)
    output_relative = Path("outputs", VARIANT_ID, run_id, "0000.png")
    output_bytes = _write_png(root / output_relative, (220, 30, 90))
    output = OutputRecord(
        ordinal=0,
        provider_order=0,
        node_id="9",
        media_type="image/png",
        local_relative_path=output_relative.as_posix(),
        content_sha256=hashlib.sha256(output_bytes).hexdigest(),
        size=len(output_bytes),
    )
    run = RunRecord(
        run_id=run_id,
        experiment_id=EXPERIMENT_ID,
        variant_id=VARIANT_ID,
        state=RunState.SUCCESS,
        plan_hash=plan.plan_hash,
        exact_run_hash=variant.exact_run_hash,
        variant_snapshot=variant.model_dump(mode="json"),
        provider_task_id="provider-task-selected",
        expected_output_count=1,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
        actual_output_count=1,
        outputs=(output,),
    )
    store.write_json_new(
        EXPERIMENT_ID,
        f"runs/{run_id}.json",
        run.model_dump(mode="json"),
    )
    regenerate_review_artifacts(store, EXPERIMENT_ID, plan, (run,))
    select_finalist(
        store,
        EXPERIMENT_ID,
        VARIANT_ID,
        now=lambda: datetime(2026, 7, 14, 16, 0, tzinfo=UTC),
    )
    return store, root


def test_suggested_name_uses_base_type_and_ascii_short_description() -> None:
    assert suggested_workflow_name(
        base_model="Qwen Image Edit 2511.safetensors",
        workflow_type="I2I",
        description="Multi-angle consistency!",
    ) == "qwen_image_edit_2511_i2i_multi_angle_consistency"


def test_suggested_name_preserves_type_and_description_suffix_at_max_length() -> None:
    value = suggested_workflow_name(
        base_model="model-" + "x" * 300,
        workflow_type="i2i",
        description="multiangle-" + "y" * 300,
    )

    assert len(value) <= 160
    assert "_i2i_" in value
    assert value.rsplit("_i2i_", 1)[1]
    assert value == suggested_workflow_name(
        base_model="model-" + "x" * 300,
        workflow_type="i2i",
        description="multiangle-" + "y" * 300,
    )
    different = suggested_workflow_name(
        base_model="model-" + "x" * 300,
        workflow_type="i2i",
        description="multiangle-" + "y" * 299 + "z",
    )
    assert different != value


def test_export_requires_current_explicit_human_selection(tmp_path: Path) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    (root / "selection.json").unlink()

    with pytest.raises(SelectionRequiredError, match="selection"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_is_self_contained_byte_stable_and_production_read_only(
    tmp_path: Path,
) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    production_roots = (
        tmp_path / "mp-local-comfyui-runner",
        tmp_path / "mp-runninghub-api",
    )
    for root in production_roots:
        (root / "protected").mkdir(parents=True)
        (root / "protected" / "snapshot.txt").write_text(
            root.name, encoding="utf-8"
        )
    before = {root: _snapshot_tree(root) for root in production_roots}

    bundle = BundleExporter(store).export(EXPERIMENT_ID)
    first = _snapshot_tree(bundle)
    repeated = BundleExporter(store).export(EXPERIMENT_ID)

    assert bundle.is_absolute()
    assert repeated == bundle
    assert _snapshot_tree(bundle) == first
    assert {item.name for item in bundle.iterdir()} == {
        "source-workflow-api.json",
        "selected-workflow-api.json",
        "semantic-patch.json",
        "model-stack.json",
        "fixtures",
        "prompts.json",
        "execution.json",
        "outputs",
        "comparison.json",
        "comparisons",
        "finalists.json",
        "review-manifest.json",
        "selection.json",
        "handoff.json",
    }
    assert (bundle / "fixtures" / "manifest.json").is_file()
    fixture_manifest = json.loads(
        (bundle / "fixtures" / "manifest.json").read_text(encoding="utf-8")
    )
    fixture = fixture_manifest["fixtures"][0]
    assert hashlib.sha256((bundle / "fixtures" / fixture["path"]).read_bytes()).hexdigest() == fixture["sha256"]
    execution = json.loads((bundle / "execution.json").read_text(encoding="utf-8"))
    exported_output = execution["exported_outputs"][0]
    assert hashlib.sha256((bundle / exported_output["path"]).read_bytes()).hexdigest() == exported_output["sha256"]
    handoff = json.loads((bundle / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["suggested_workflow_name"] == (
        "qwen_image_edit_2511_i2i_multiangle_consistency"
    )
    assert handoff["instructions"] == [
        "Upload or import selected-workflow-api.json to RunningHub as a saved workflow.",
        "Obtain the URL of the saved RunningHub workflow.",
        "Invoke port-comfy-workflow with that URL through strip, audit, register, smoke, and promote.",
    ]
    assert all("command" not in key for key in handoff)
    selection = json.loads((bundle / "selection.json").read_text(encoding="utf-8"))
    assert selection["finalists_artifact_sha256"] == hashlib.sha256(
        (bundle / "finalists.json").read_bytes()
    ).hexdigest()
    canonical_manifest_path, review_manifest = (
        _exported_review_contract(bundle)
    )
    assert selection["review_generation_id"] == review_manifest["generation_id"]
    assert selection["review_manifest_sha256"] == hashlib.sha256(
        canonical_manifest_path.read_bytes()
    ).hexdigest()
    assert review_manifest["finalists_sha256"] == selection[
        "finalists_artifact_sha256"
    ]
    comparison = json.loads(
        (bundle / "comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["facts"]["experiment_id"] == EXPERIMENT_ID
    assert comparison["facts_sha256"] == review_manifest[
        "comparison_facts_sha256"
    ]
    assert "judgments" in comparison
    for artifact in review_manifest["artifacts"]:
        review_artifact = canonical_manifest_path.parent / artifact["path"]
        content = review_artifact.read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert len(content) == artifact["bytes"]
    assert {root: _snapshot_tree(root) for root in production_roots} == before


def test_exported_review_contract_resolves_every_manifest_artifact(
    tmp_path: Path,
) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    bundle = BundleExporter(store).export(EXPERIMENT_ID)

    manifest_path, manifest = _exported_review_contract(bundle)

    selection = json.loads(
        (bundle / "selection.json").read_text(encoding="utf-8")
    )
    assert manifest["generation_id"] == selection["review_generation_id"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == selection[
        "review_manifest_sha256"
    ]
    artifact_root = manifest_path.parent.resolve(strict=True)
    for artifact in manifest["artifacts"]:
        artifact_path = (artifact_root / artifact["path"]).resolve(strict=True)
        artifact_path.relative_to(artifact_root)
        content = artifact_path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
        assert len(content) == artifact["bytes"]


def test_export_stores_selected_media_once_for_review_and_execution(
    tmp_path: Path,
) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    bundle = BundleExporter(store).export(EXPERIMENT_ID)
    manifest_path, manifest = _exported_review_contract(bundle)
    execution = json.loads(
        (bundle / "execution.json").read_text(encoding="utf-8")
    )
    selected = execution["exported_outputs"][0]
    execution_path = (bundle / selected["path"]).resolve(strict=True)
    manifest_entry = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["sha256"] == selected["sha256"]
        and artifact["bytes"] == selected["bytes"]
        and artifact["media_type"] == selected["media_type"]
    )
    manifest_output_path = (
        manifest_path.parent / manifest_entry["path"]
    ).resolve(strict=True)

    assert execution_path == manifest_output_path
    payload_copies = []
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != selected["sha256"]:
            continue
        assert stat.S_ISREG(path.lstat().st_mode)
        payload_copies.append(path.relative_to(bundle).as_posix())
    assert payload_copies == [selected["path"]]


def test_export_rejects_tampered_recorded_output(tmp_path: Path) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    output = next((root / "outputs" / VARIANT_ID).rglob("*.png"))
    output.write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="output evidence"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_rejects_symlinked_fixture_snapshot(tmp_path: Path) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    fixture = next((root / "runs" / "run-selected" / "fixtures").iterdir())
    target = tmp_path / "outside.png"
    target.write_bytes(fixture.read_bytes())
    fixture.unlink()
    try:
        fixture.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("test environment does not permit symlinks")

    with pytest.raises(BundleIntegrityError, match="fixture"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_rejects_symlinked_fixture_ancestor_outside_experiment(
    tmp_path: Path,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    run_root = root / "runs" / "run-selected"
    outside_run = tmp_path / "outside-run-selected"
    run_root.rename(outside_run)
    try:
        run_root.symlink_to(outside_run, target_is_directory=True)
    except (OSError, NotImplementedError):
        outside_run.rename(run_root)
        pytest.skip("test environment does not permit directory symlinks")

    with pytest.raises(BundleIntegrityError, match="fixture"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_rejects_platform_link_fixture_ancestor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    run_root = root / "runs" / "run-selected"
    original = bundle_module.path_component_is_link_or_junction

    def reports_run_root_as_link(path, metadata=None):
        if path == run_root:
            return True
        return original(path, metadata)

    monkeypatch.setattr(
        bundle_module,
        "path_component_is_link_or_junction",
        reports_run_root_as_link,
    )

    with pytest.raises(BundleIntegrityError, match="fixture"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_existing_changed_bundle_conflicts_without_clobber(tmp_path: Path) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    exporter = BundleExporter(store)
    bundle = exporter.export(EXPERIMENT_ID)
    selected = bundle / "selected-workflow-api.json"
    selected.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BundleConflictError, match="different content"):
        exporter.export(EXPERIMENT_ID)

    assert selected.read_text(encoding="utf-8") == "{}\n"


def test_reselection_publishes_new_immutable_export_and_preserves_old_bundle(
    tmp_path: Path,
) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    exporter = BundleExporter(store)
    first = exporter.export(EXPERIMENT_ID)
    first_snapshot = _snapshot_tree(first)
    select_finalist(
        store,
        EXPERIMENT_ID,
        VARIANT_ID,
        now=lambda: datetime(2026, 7, 14, 17, 0, tzinfo=UTC),
    )

    second = exporter.export(EXPERIMENT_ID)

    assert second != first
    assert _snapshot_tree(first) == first_snapshot
    assert (second / "handoff.json").is_file()
    assert (second / "review-manifest.json").is_file()


def test_later_review_generation_gets_new_export_and_preserves_old_bundle(
    tmp_path: Path,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    exporter = BundleExporter(store)
    first = exporter.export(EXPERIMENT_ID)
    first_snapshot = _snapshot_tree(first)
    current = store.read_json(EXPERIMENT_ID, "finalists.json")
    plan = ExperimentPlan.model_validate_json(
        store.read_bytes(EXPERIMENT_ID, f"plans/{current['plan_hash']}.json")
    )
    run = RunRecord.model_validate_json(
        store.read_bytes(EXPERIMENT_ID, "runs/run-selected.json")
    )
    judgments = JudgmentSubmission.model_validate(
        {
            "schema_version": "1",
            "experiment_id": EXPERIMENT_ID,
            "facts_sha256": current["comparison_facts_sha256"],
            "judgments": [
                {
                    "variant_id": VARIANT_ID,
                    "identity_fidelity": 5,
                    "prompt_adherence": 5,
                    "visual_quality": 5,
                    "creative_usefulness": 5,
                    "reasons": ["selected after complete human review"],
                    "referenced_output_ids": [f"{VARIANT_ID}/0"],
                }
            ],
        }
    )
    later_review = regenerate_review_artifacts(
        store,
        EXPERIMENT_ID,
        plan,
        (run,),
        judgments=judgments,
    )
    assert later_review.finalists.review_generation_id != current[
        "review_generation_id"
    ]
    select_finalist(
        store,
        EXPERIMENT_ID,
        VARIANT_ID,
        now=lambda: datetime(2026, 7, 14, 18, 0, tzinfo=UTC),
    )

    second = exporter.export(EXPERIMENT_ID)

    assert second != first
    assert _snapshot_tree(first) == first_snapshot
    _, second_manifest = _exported_review_contract(second)
    assert second_manifest["generation_id"] == (
        later_review.finalists.review_generation_id
    )


def test_export_rejects_changed_human_review_artifact(tmp_path: Path) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    current = store.read_json(EXPERIMENT_ID, "finalists.json")
    gallery = (
        root
        / current["review_generation_path"]
        / "comparisons"
        / "index.html"
    )
    gallery.write_text("<html>changed</html>\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="review.*artifact"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_changed_finalists_make_selection_stale_for_export(tmp_path: Path) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    finalists = json.loads((root / "finalists.json").read_text(encoding="utf-8"))
    finalists["excluded"]["new"] = "run_status_failed"
    store.write_json(EXPERIMENT_ID, "finalists.json", finalists)

    with pytest.raises(StaleSelectionError, match="SHA-256 changed"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_missing_current_finalists_after_selection_is_typed_stale(
    tmp_path: Path,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    (root / "finalists.json").unlink()

    with pytest.raises(StaleSelectionError):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_rejects_coordinated_output_and_run_record_mutation(
    tmp_path: Path,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    run_path = root / "runs" / "run-selected.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    output_path = root / run["outputs"][0]["local_relative_path"]
    changed = b"coordinated changed output bytes"
    output_path.write_bytes(changed)
    run["outputs"][0]["content_sha256"] = hashlib.sha256(changed).hexdigest()
    run["outputs"][0]["size"] = len(changed)
    store.write_json(EXPERIMENT_ID, "runs/run-selected.json", run)

    with pytest.raises(ArtifactIntegrityError, match="evidence"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_uses_immutable_approved_source_url(
    tmp_path: Path,
) -> None:
    store, root = _persist_selected_experiment(tmp_path)
    source_bundle = store.read_json(EXPERIMENT_ID, "sources/source-bundle.json")
    source_bundle["sources"][0]["canonical_url"] = (
        "https://www.runninghub.ai/post/9999999999999999999"
    )
    store.write_json(EXPERIMENT_ID, "sources/source-bundle.json", source_bundle)

    bundle = BundleExporter(store).export(EXPERIMENT_ID)

    model_stack = json.loads(
        (bundle / "model-stack.json").read_text(encoding="utf-8")
    )
    assert model_stack["source_url"] == (
        "https://www.runninghub.ai/post/1980860923768426497"
    )


def test_export_rejects_selection_changed_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store, _ = _persist_selected_experiment(tmp_path)
    original = bundle_module.require_valid_selection

    def change_selection(experiment_store, experiment_id):
        selection = original(experiment_store, experiment_id)
        persisted = experiment_store.read_json(experiment_id, "selection.json")
        persisted["timestamp"] = "2026-07-14T17:00:00Z"
        experiment_store.write_json(experiment_id, "selection.json", persisted)
        return selection

    monkeypatch.setattr(
        bundle_module,
        "require_valid_selection",
        change_selection,
    )

    with pytest.raises(BundleIntegrityError, match="selection artifact changed"):
        BundleExporter(store).export(EXPERIMENT_ID)


def test_export_cli_is_noninteractive_and_returns_absolute_bundle_path(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _persist_selected_experiment(tmp_path)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLI prompted for input")
        ),
    )

    code = main(["export", EXPERIMENT_ID], cwd=tmp_path)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    bundle = Path(payload["data"]["bundle"])
    assert code == ExitCode.OK
    assert captured.err == ""
    assert bundle.is_absolute() and bundle.is_dir()
    assert payload["artifacts"] == [
        {"kind": "handoff_bundle", "path": str(bundle)}
    ]
