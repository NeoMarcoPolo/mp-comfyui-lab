from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from mp_comfyui_lab.comparison.finalists import (
    FinalistSelectionError,
    SelectionState,
    propose_finalists,
    regenerate_review_artifacts,
    select_finalist,
    selection_status,
    write_finalists,
)
from mp_comfyui_lab.comparison.judgments import JudgmentSubmission
from mp_comfyui_lab.comparison.metrics import compare_outputs
from mp_comfyui_lab.comparison.render import comparison_facts_sha256
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.experiments.models import (
    DimensionName,
    ExperimentPlan,
    OutputRecord,
    PlannedVariant,
    PlanningConfidence,
    RunRecord,
    RunState,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.graph.hashing import capability_hash


EXPERIMENT_ID = "exp-finalists"


def _canonical_hash(value: object) -> str:
    document = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _graph(variant_id: str) -> dict[str, object]:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": f"{variant_id}.safetensors"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": variant_id},
        },
    }


def _plan(
    variant_ids: tuple[str, ...],
    *,
    partial_behavior_is_tested_capability: bool = False,
) -> ExperimentPlan:
    variants = []
    for index, variant_id in enumerate(variant_ids):
        graph = _graph(variant_id)
        draft = PlannedVariant(
            variant_id=variant_id,
            source_workflow_id="workflow-source",
            source_hash="a" * 64,
            materialized_graph=graph,
            hypothesis=(
                None if index == 0 else DimensionName.CHECKPOINT
            ),
            evidence_ids=(f"evidence-{variant_id}",),
            confidence=(
                PlanningConfidence.OBSERVED
                if index == 0
                else PlanningConfidence.STRONGLY_INFERRED
            ),
            exact_run_hash="0" * 64,
            capability_hash=capability_hash(graph),
            expected_output_count=2 if variant_id == "partial" else 1,
            output_node_ids=("9",),
            expected_media_types=("image/png",),
            execution_parameters={
                "prompt_lines": ["multiangle consistency"],
                "expected_output_count": (
                    2 if variant_id == "partial" else 1
                ),
                "output_node_ids": ["9"],
            },
        )
        variants.append(
            PlannedVariant.model_validate(
                {
                    **draft.model_dump(mode="python"),
                    "exact_run_hash": recompute_exact_run_hash(draft),
                }
            )
        )
    approval_payload = {
        "source_hashes": ["a" * 64],
        "variants": [item.model_dump(mode="json") for item in variants],
        "capability_coverage": [DimensionName.CHECKPOINT.value],
        "warnings": [],
        "brief": {
            "goal": "multiangle consistency",
            "partial_behavior_is_tested_capability": (
                partial_behavior_is_tested_capability
            ),
        },
    }
    return ExperimentPlan(
        plan_hash=_canonical_hash(approval_payload),
        source_hashes=("a" * 64,),
        variants=tuple(variants),
        capability_coverage=(DimensionName.CHECKPOINT,),
        warnings=(),
        approval_payload=approval_payload,
    )


def _image_output(
    experiment_root: Path,
    variant: PlannedVariant,
    *,
    run_id: str,
    color: tuple[int, int, int],
) -> OutputRecord:
    relative_path = Path("outputs", variant.variant_id, run_id, "0000.png")
    path = experiment_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path, format="PNG")
    content = path.read_bytes()
    return OutputRecord(
        ordinal=0,
        provider_order=0,
        node_id="9",
        media_type="image/png",
        local_relative_path=relative_path.as_posix(),
        content_sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )


def _run(
    experiment_root: Path,
    variant: PlannedVariant,
    *,
    state: RunState,
    color: tuple[int, int, int] = (20, 40, 60),
) -> RunRecord:
    run_id = f"run-{variant.variant_id}"
    outputs = (
        ()
        if state in {RunState.FAILED, RunState.CANCELLED}
        else (_image_output(experiment_root, variant, run_id=run_id, color=color),)
    )
    error = None
    if state is RunState.INCOMPLETE:
        error = {
            "kind": "partial_outputs",
            "expected_output_count": variant.expected_output_count,
            "actual_output_count": len(outputs),
        }
    elif state is not RunState.SUCCESS:
        error = {"kind": state.value.casefold()}
    return RunRecord(
        run_id=run_id,
        experiment_id=EXPERIMENT_ID,
        variant_id=variant.variant_id,
        state=state,
        plan_hash="f" * 64,
        exact_run_hash=variant.exact_run_hash,
        variant_snapshot=variant.model_dump(mode="json"),
        provider_task_id=f"task-{variant.variant_id}",
        expected_output_count=variant.expected_output_count,
        expected_output_node_ids=variant.output_node_ids,
        expected_media_types=variant.expected_media_types,
        actual_output_count=len(outputs),
        outputs=outputs,
        error=error,
    )


def test_finalists_exclude_failed_cancelled_and_partial_batches_by_default(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / EXPERIMENT_ID
    plan = _plan(("complete", "partial", "failed", "cancelled"))
    variants = {item.variant_id: item for item in plan.variants}
    report = compare_outputs(
        (
            _run(experiment_root, variants["complete"], state=RunState.SUCCESS),
            _run(experiment_root, variants["partial"], state=RunState.INCOMPLETE),
            _run(experiment_root, variants["failed"], state=RunState.FAILED),
            _run(experiment_root, variants["cancelled"], state=RunState.CANCELLED),
        ),
        experiment_root,
        experiment_id=EXPERIMENT_ID,
    )

    finalists = propose_finalists(report, plan)

    assert finalists.variant_ids == ("complete",)
    assert finalists.requires_human_selection is True
    assert "selected_variant_id" not in finalists.model_dump(mode="json")
    assert finalists.excluded["partial"] == "partial_batch_not_approved"
    assert finalists.excluded["failed"] == "run_status_failed"
    assert finalists.excluded["cancelled"] == "run_status_cancelled"
    proposal = finalists.finalists[0]
    assert proposal.supporting_facts.objective_completion is True
    assert proposal.supporting_facts.perceptually_unique_output_count >= 0
    assert proposal.supporting_facts.capability_covered is True
    assert proposal.supporting_facts.evidence_confidence == "observed"


def test_partial_finalist_requires_immutable_approved_capability_and_is_visible(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / EXPERIMENT_ID
    plan = _plan(
        ("partial", "failed"),
        partial_behavior_is_tested_capability=True,
    )
    variants = {item.variant_id: item for item in plan.variants}
    report = compare_outputs(
        (
            _run(experiment_root, variants["partial"], state=RunState.INCOMPLETE),
            _run(experiment_root, variants["failed"], state=RunState.FAILED),
        ),
        experiment_root,
        experiment_id=EXPERIMENT_ID,
    )

    finalists = propose_finalists(report, plan)

    assert finalists.variant_ids == ("partial",)
    proposal = finalists.finalists[0]
    assert proposal.label.startswith("PARTIAL TEST CAPABILITY: ")
    assert proposal.supporting_facts.partial_batch is True
    assert proposal.supporting_facts.partial_behavior_approved is True
    assert proposal.supporting_facts.actual_output_count == 1
    assert proposal.supporting_facts.expected_output_count == 2
    assert "failed" not in finalists.variant_ids


def test_approved_partial_capability_does_not_admit_unrelated_incomplete_run(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / EXPERIMENT_ID
    plan = _plan(
        ("partial",),
        partial_behavior_is_tested_capability=True,
    )
    variant = plan.variants[0]
    run = RunRecord(
        run_id="run-unrelated-incomplete",
        experiment_id=EXPERIMENT_ID,
        variant_id=variant.variant_id,
        state=RunState.INCOMPLETE,
        plan_hash=plan.plan_hash,
        exact_run_hash=variant.exact_run_hash,
        variant_snapshot=variant.model_dump(mode="json"),
        provider_task_id="task-unrelated-incomplete",
        expected_output_count=variant.expected_output_count,
        expected_output_node_ids=variant.output_node_ids,
        expected_media_types=variant.expected_media_types,
        actual_output_count=0,
        error={"kind": "collection_error"},
    )
    report = compare_outputs(
        (run,),
        experiment_root,
        experiment_id=EXPERIMENT_ID,
    )

    finalists = propose_finalists(report, plan)

    assert finalists.variant_ids == ()
    assert finalists.excluded == {"partial": "run_status_incomplete"}


def test_finalists_persist_all_scoring_facts_and_validated_judgment_reasons(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / EXPERIMENT_ID
    plan = _plan(("baseline", "candidate"))
    variants = {item.variant_id: item for item in plan.variants}
    report = compare_outputs(
        (
            _run(
                experiment_root,
                variants["baseline"],
                state=RunState.SUCCESS,
                color=(20, 40, 60),
            ),
            _run(
                experiment_root,
                variants["candidate"],
                state=RunState.SUCCESS,
                color=(220, 30, 90),
            ),
        ),
        experiment_root,
        experiment_id=EXPERIMENT_ID,
    )
    judgments = JudgmentSubmission.model_validate(
        {
            "schema_version": "1",
            "experiment_id": EXPERIMENT_ID,
            "facts_sha256": comparison_facts_sha256(report),
            "judgments": [
                {
                    "variant_id": "baseline",
                    "identity_fidelity": 2,
                    "prompt_adherence": 2,
                    "visual_quality": 2,
                    "creative_usefulness": 2,
                    "reasons": ["baseline loses the requested angle"],
                    "referenced_output_ids": ["baseline/0"],
                },
                {
                    "variant_id": "candidate",
                    "identity_fidelity": 5,
                    "prompt_adherence": 5,
                    "visual_quality": 4,
                    "creative_usefulness": 5,
                    "reasons": ["identity remains stable across the new angle"],
                    "referenced_output_ids": ["candidate/0"],
                },
            ],
        }
    )

    finalists = propose_finalists(report, plan, judgments=judgments)

    assert finalists.variant_ids[0] == "candidate"
    proposal = finalists.finalists[0]
    assert proposal.judgment_reasons == (
        "identity remains stable across the new angle",
    )
    assert proposal.agent_judgment is not None
    assert proposal.agent_judgment.referenced_output_ids == ("candidate/0",)
    assert set(proposal.factor_scores) == {
        "objective_completion",
        "perceptual_uniqueness",
        "capability_coverage",
        "evidence_confidence",
        "agent_judgment",
    }


def test_regeneration_writes_empty_finalists_and_never_selection(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    experiment_root = store.create(EXPERIMENT_ID)
    plan = _plan(("failed",))
    variant = plan.variants[0]
    store.write_json_new(
        EXPERIMENT_ID,
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    run = _run(experiment_root, variant, state=RunState.FAILED)
    run = run.model_copy(update={"plan_hash": plan.plan_hash})
    store.write_json_new(
        EXPERIMENT_ID,
        f"runs/{run.run_id}.json",
        run.model_dump(mode="json"),
    )

    review = regenerate_review_artifacts(
        store,
        EXPERIMENT_ID,
        plan,
        (run,),
    )

    assert review.comparison_path.is_absolute()
    assert review.finalists_path.is_absolute()
    assert review.comparison_path.exists()
    assert review.finalists_path.exists()
    assert review.finalists.variant_ids == ()
    assert not (experiment_root / "selection.json").exists()


def test_select_is_explicit_and_changed_finalists_make_selection_stale(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path / "experiments")
    experiment_root = store.create(EXPERIMENT_ID)
    plan = _plan(("complete",))
    variant = plan.variants[0]
    store.write_json_new(
        EXPERIMENT_ID,
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    run = _run(experiment_root, variant, state=RunState.SUCCESS)
    run = run.model_copy(update={"plan_hash": plan.plan_hash})
    review = regenerate_review_artifacts(store, EXPERIMENT_ID, plan, (run,))
    selected_at = datetime(2026, 7, 14, 15, 30, tzinfo=UTC)

    selection = select_finalist(
        store,
        EXPERIMENT_ID,
        "complete",
        now=lambda: selected_at,
    )

    assert selection.selected_by == "human_cli_input"
    assert selection.selected_variant_id == "complete"
    assert selection.plan_hash == plan.plan_hash
    assert selection.finalists_artifact_sha256 == hashlib.sha256(
        review.finalists_path.read_bytes()
    ).hexdigest()
    assert selection.selected_at == selected_at
    persisted_selection = store.read_json(EXPERIMENT_ID, "selection.json")
    assert persisted_selection["variant_id"] == "complete"
    assert persisted_selection["timestamp"] == selected_at.isoformat().replace(
        "+00:00", "Z"
    )
    assert "selected_variant_id" not in persisted_selection
    assert "selected_at" not in persisted_selection
    assert selection_status(store, EXPERIMENT_ID).state is SelectionState.VALID

    empty_report = compare_outputs(
        (), experiment_root, experiment_id=EXPERIMENT_ID
    )
    write_finalists(store, propose_finalists(empty_report, plan))
    stale = selection_status(store, EXPERIMENT_ID)
    assert stale.state is SelectionState.STALE
    assert "finalists artifact SHA-256 changed" in (stale.reason or "")
    with pytest.raises(FinalistSelectionError, match="not a current finalist"):
        select_finalist(store, EXPERIMENT_ID, "complete")


def _cli_payload(capsys) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _persist_cli_review(tmp_path: Path) -> tuple[ExperimentStore, ExperimentPlan]:
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create(EXPERIMENT_ID)
    plan = _plan(("complete",))
    variant = plan.variants[0]
    store.write_json_new(
        EXPERIMENT_ID,
        f"plans/{plan.plan_hash}.json",
        plan.model_dump(mode="json"),
    )
    run = _run(root, variant, state=RunState.SUCCESS).model_copy(
        update={"plan_hash": plan.plan_hash}
    )
    store.write_json_new(
        EXPERIMENT_ID,
        f"runs/{run.run_id}.json",
        run.model_dump(mode="json"),
    )
    return store, plan


def test_finalists_and_select_cli_are_noninteractive_and_return_absolute_paths(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    store, _ = _persist_cli_review(tmp_path)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLI prompted for input")
        ),
    )

    assert main(["finalists", EXPERIMENT_ID], cwd=tmp_path) == ExitCode.OK
    finalists_payload = _cli_payload(capsys)
    comparison_path = Path(finalists_payload["data"]["comparison_path"])
    finalists_path = Path(finalists_payload["data"]["finalists_path"])
    assert comparison_path.is_absolute() and comparison_path.exists()
    assert finalists_path.is_absolute() and finalists_path.exists()
    assert finalists_payload["data"]["variant_ids"] == ["complete"]
    assert not (store.root(EXPERIMENT_ID) / "selection.json").exists()

    assert main(
        ["select", EXPERIMENT_ID, "complete"], cwd=tmp_path
    ) == ExitCode.OK
    selection_payload = _cli_payload(capsys)
    selection_path = Path(selection_payload["data"]["selection_path"])
    assert selection_path.is_absolute() and selection_path.exists()
    assert selection_payload["data"]["selected_by"] == "human_cli_input"


def test_select_and_export_cli_return_structured_stale_selection_errors(
    tmp_path: Path,
    capsys,
) -> None:
    store, _ = _persist_cli_review(tmp_path)
    assert main(["finalists", EXPERIMENT_ID], cwd=tmp_path) == ExitCode.OK
    capsys.readouterr()

    code = main(["select", EXPERIMENT_ID, "missing"], cwd=tmp_path)
    invalid = _cli_payload(capsys)
    assert code == ExitCode.SELECTION
    assert invalid["error"]["kind"] == "selection"

    assert main(
        ["select", EXPERIMENT_ID, "complete"], cwd=tmp_path
    ) == ExitCode.OK
    capsys.readouterr()
    finalists = store.read_json(EXPERIMENT_ID, "finalists.json")
    finalists["excluded"]["changed"] = "run_status_failed"
    store.write_json(EXPERIMENT_ID, "finalists.json", finalists)

    code = main(["export", EXPERIMENT_ID], cwd=tmp_path)
    stale = _cli_payload(capsys)
    assert code == ExitCode.SELECTION
    assert stale["error"]["kind"] == "stale_selection"
    assert "SHA-256 changed" in stale["error"]["message"]


def test_finalists_cli_rejects_noncanonical_wave_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    store, plan = _persist_cli_review(tmp_path)
    store.write_json_new(
        EXPERIMENT_ID,
        "plans/waves/not-an-ordinal.json",
        {"plan_hash": plan.plan_hash},
    )

    code = main(["finalists", EXPERIMENT_ID], cwd=tmp_path)

    payload = _cli_payload(capsys)
    assert code == ExitCode.SAFETY
    assert payload["error"]["kind"] == "safety"
    assert "wave artifact filename is not an ordinal" in payload["error"]["message"]
