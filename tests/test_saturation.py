from __future__ import annotations

import hashlib
import json

import pytest

import mp_comfyui_lab.experiments.saturation as saturation_module
from mp_comfyui_lab.graph.hashing import capability_hash as graph_capability_hash
from mp_comfyui_lab.experiments.models import (
    DimensionName,
    ExperimentPlan,
    PlannedVariant,
    PlanningConfidence,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.saturation import (
    ExperimentHistory,
    RerunReasonRequired,
    next_wave,
    request_rerun,
)


def _hash(value: object) -> str:
    document = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def _variant(
    variant_id: str,
    *,
    capability_hash: str,
    exact_run_hash: str,
    confidence: PlanningConfidence = PlanningConfidence.OBSERVED,
    hypothesis: DimensionName = DimensionName.CHECKPOINT,
    required_nodes: tuple[str, ...] = ("KSampler", "SaveImage"),
    corrective_hypotheses: dict[str, str] | None = None,
    correction: dict[str, object] | None = None,
    warnings: tuple[str, ...] = (),
    graph: dict[str, object] | None = None,
) -> PlannedVariant:
    graph = graph or {
        str(index): {"class_type": class_type, "inputs": {}}
        for index, class_type in enumerate(required_nodes, start=1)
    }
    output_node_id = next(reversed(graph))
    execution_parameters: dict[str, object] = {
        "prompt_lines": ["one"],
        "expected_output_count": 1,
        "output_node_ids": [output_node_id],
    }
    if corrective_hypotheses is not None:
        execution_parameters["corrective_hypotheses"] = corrective_hypotheses
    if correction is not None:
        execution_parameters["correction"] = correction
    return PlannedVariant(
        variant_id=variant_id,
        source_workflow_id="workflow-1",
        source_hash="a" * 64,
        materialized_graph=graph,
        hypothesis=hypothesis,
        evidence_ids=(f"evidence-{variant_id}",),
        confidence=confidence,
        warnings=warnings,
        exact_run_hash=exact_run_hash,
        capability_hash=capability_hash,
        expected_output_count=1,
        output_node_ids=(output_node_id,),
        expected_media_types=("image/png",),
        execution_parameters=execution_parameters,
    )


def _plan(
    *variants: PlannedVariant,
    candidate_scores: dict[str, tuple[int, int]] | None = None,
    coverage: tuple[DimensionName, ...] = (DimensionName.CHECKPOINT,),
) -> ExperimentPlan:
    scores = candidate_scores or {}
    candidates = [
        {
            "candidate_id": item.variant_id,
            "confidence": item.confidence.value if item.confidence else "observed",
            "evidence_ids": list(item.evidence_ids),
            "engagement": scores.get(item.variant_id, (0, 0))[0],
            "provenance_rank": scores.get(item.variant_id, (0, 0))[1],
        }
        for item in variants
    ]
    payload = {
        "source_hashes": ["a" * 64],
        "variants": [item.model_dump(mode="json") for item in variants],
        "capability_coverage": [item.value for item in coverage],
        "warnings": [],
        "candidates": candidates,
    }
    return ExperimentPlan(
        plan_hash=_hash(payload),
        source_hashes=("a" * 64,),
        variants=variants,
        capability_coverage=coverage,
        warnings=(),
        approval_payload=payload,
    )


def test_next_wave_selects_strongest_representative_per_novel_capability() -> None:
    repeated = _variant(
        "already-run",
        capability_hash="1" * 64,
        exact_run_hash="a" * 64,
    )
    weak = _variant(
        "weak-high-engagement",
        capability_hash="2" * 64,
        exact_run_hash="b" * 64,
        confidence=PlanningConfidence.WEAKLY_INFERRED,
    )
    observed = _variant(
        "observed-low-engagement",
        capability_hash="2" * 64,
        exact_run_hash="c" * 64,
    )
    other = _variant(
        "other-capability",
        capability_hash="3" * 64,
        exact_run_hash="d" * 64,
    )
    plan = _plan(
        repeated,
        weak,
        observed,
        other,
        candidate_scores={
            "weak-high-engagement": (99_999, 1),
            "observed-low-engagement": (1, 1),
        },
    )
    history = ExperimentHistory()
    history.record_success(repeated, run_id="run-original")

    wave = next_wave(plan, history)

    assert [item.variant_id for item in wave.variants] == [
        "observed-low-engagement",
        "other-capability",
    ]
    assert len({item.capability_hash for item in wave.variants}) == len(
        wave.variants
    )
    assert all(
        item.exact_run_hash not in history.exact_run_hashes
        for item in wave.variants
    )
    assert wave.reasons["already-run"] == "exact_run_already_recorded"
    assert wave.reasons["weak-high-engagement"] == (
        "weaker_representative_of_novel_capability"
    )


def test_candidate_evidence_breaks_ties_by_provenance_then_engagement() -> None:
    popular = _variant(
        "popular",
        capability_hash="4" * 64,
        exact_run_hash="e" * 64,
    )
    stronger_provenance = _variant(
        "stronger-provenance",
        capability_hash="4" * 64,
        exact_run_hash="f" * 64,
    )
    plan = _plan(
        popular,
        stronger_provenance,
        candidate_scores={
            "popular": (10_000, 1),
            "stronger-provenance": (1, 2),
        },
    )

    wave = next_wave(plan, ExperimentHistory())

    assert [item.variant_id for item in wave.variants] == [
        "stronger-provenance"
    ]


def test_representative_ranking_uses_engagement_cleanliness_and_stable_id() -> None:
    popular = _variant(
        "popular-with-warning",
        capability_hash="d" * 64,
        exact_run_hash="a" * 64,
        warnings=("structural warning",),
    )
    clean = _variant(
        "clean-but-less-popular",
        capability_hash="d" * 64,
        exact_run_hash="b" * 64,
    )
    warned_tie = _variant(
        "warned-tie",
        capability_hash="e" * 64,
        exact_run_hash="c" * 64,
        warnings=("structural warning",),
    )
    clean_tie = _variant(
        "clean-tie",
        capability_hash="e" * 64,
        exact_run_hash="d" * 64,
    )
    stable_b = _variant(
        "stable-b",
        capability_hash="f" * 64,
        exact_run_hash="e" * 64,
    )
    stable_a = _variant(
        "stable-a",
        capability_hash="f" * 64,
        exact_run_hash="f" * 64,
    )
    plan = _plan(
        popular,
        clean,
        warned_tie,
        clean_tie,
        stable_b,
        stable_a,
        candidate_scores={
            "popular-with-warning": (100, 1),
            "clean-but-less-popular": (10, 1),
            "warned-tie": (5, 1),
            "clean-tie": (5, 1),
            "stable-b": (1, 1),
            "stable-a": (1, 1),
        },
    )

    decision = next_wave(plan, ExperimentHistory())

    assert [item.variant_id for item in decision.variants] == [
        "popular-with-warning",
        "clean-tie",
        "stable-a",
    ]


def test_perceptually_equivalent_tested_capability_dominates_weaker_candidate() -> None:
    tested = _variant(
        "observed-high-engagement",
        capability_hash="5" * 64,
        exact_run_hash="1" * 64,
    )
    weaker = _variant(
        "unexplored-low-evidence",
        capability_hash="5" * 64,
        exact_run_hash="2" * 64,
        confidence=PlanningConfidence.UNEXPLORED,
    )
    plan = _plan(tested, weaker)
    history = ExperimentHistory()
    history.record_success(tested, run_id="run-tested")
    history.add_equivalent_group(
        ["observed-high-engagement", "unexplored-low-evidence"]
    )

    decision = next_wave(plan, history)

    assert "unexplored-low-evidence" in decision.skipped_variant_ids
    assert decision.reasons["unexplored-low-evidence"] == (
        "dominated_by_equivalent_tested_capability"
    )


def test_equivalent_output_cannot_dominate_a_different_capability_claim() -> None:
    tested_checkpoint = _variant(
        "tested-checkpoint",
        capability_hash="5" * 64,
        exact_run_hash="1" * 64,
    )
    untested_lora = _variant(
        "untested-lora",
        capability_hash="6" * 64,
        exact_run_hash="2" * 64,
        confidence=PlanningConfidence.UNEXPLORED,
        hypothesis=DimensionName.LORA_STACK,
    )
    plan = _plan(
        tested_checkpoint,
        untested_lora,
        coverage=(DimensionName.CHECKPOINT, DimensionName.LORA_STACK),
    )
    history = ExperimentHistory()
    history.record_success(tested_checkpoint, run_id="run-tested")
    history.add_equivalent_group(["tested-checkpoint", "untested-lora"])

    decision = next_wave(plan, history)

    assert [item.variant_id for item in decision.variants] == ["untested-lora"]


def test_weaker_tested_candidate_does_not_dominate_stronger_untested_claim() -> None:
    tested = _variant(
        "weak-tested",
        capability_hash="7" * 64,
        exact_run_hash="3" * 64,
        confidence=PlanningConfidence.WEAKLY_INFERRED,
    )
    stronger = _variant(
        "observed-untested",
        capability_hash="7" * 64,
        exact_run_hash="4" * 64,
    )
    plan = _plan(tested, stronger)
    history = ExperimentHistory()
    history.record_success(tested, run_id="run-weak")

    decision = next_wave(plan, history)

    assert [item.variant_id for item in decision.variants] == [
        "observed-untested"
    ]


def test_repeated_missing_node_failure_requires_corrective_hypothesis() -> None:
    unchanged = _variant(
        "still-requires-missing-node",
        capability_hash="8" * 64,
        exact_run_hash="5" * 64,
        required_nodes=("TextEncodeQwen", "SaveImage"),
    )
    corrected = _variant(
        "new-encoder-path",
        capability_hash="9" * 64,
        exact_run_hash="6" * 64,
        required_nodes=("CLIPTextEncode", "SaveImage"),
    )
    history = ExperimentHistory()
    history.record_failure(
        "variant-a",
        signature="missing-node:TextEncodeQwen",
        run_id="run-a",
    )
    history.record_failure(
        "variant-b",
        signature="missing-node:TextEncodeQwen",
        run_id="run-b",
    )

    decision = next_wave(_plan(unchanged, corrected), history)

    assert [item.variant_id for item in decision.variants] == [
        "new-encoder-path"
    ]
    assert decision.reasons["still-requires-missing-node"] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


def test_prose_only_corrective_hypothesis_cannot_bypass_repeated_failure() -> None:
    signature = "missing-node:TextEncodeQwen"
    corrected = _variant(
        "install-and-use-required-node",
        capability_hash="0" * 64,
        exact_run_hash="7" * 64,
        required_nodes=("TextEncodeQwen", "SaveImage"),
        corrective_hypotheses={
            signature: "executor capability audit confirms the node is installed"
        },
    )
    history = ExperimentHistory()
    history.record_failure("variant-a", signature=signature, run_id="run-a")
    history.record_failure("variant-b", signature=signature, run_id="run-b")

    decision = next_wave(_plan(corrected), history)

    assert decision.variants == ()
    assert decision.reasons[corrected.variant_id] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


def test_machine_verified_correction_selects_changed_candidate_and_affects_hashes() -> None:
    signature = "missing-node:TextEncodeQwen"
    failed_draft = _variant(
        "failed-encoder",
        capability_hash="0" * 64,
        exact_run_hash="5" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("TextEncodeQwen", "SaveImage"),
    )
    failed = failed_draft.model_copy(
        update={
            "capability_hash": graph_capability_hash(
                failed_draft.materialized_graph
            )
        }
    )
    correction = {
        "schema_version": 1,
        "failure_signature": signature,
        "failed_variant_id": failed.variant_id,
        "failed_exact_run_hash": failed.exact_run_hash,
        "failed_capability_hash": failed.capability_hash,
        "candidate_variant_id": "corrected-encoder",
        "changed_dimensions": [DimensionName.ENCODER.value],
        "rationale": "replace the unavailable encoder",
    }
    uncorrected = _variant(
        "corrected-encoder",
        capability_hash="9" * 64,
        exact_run_hash="0" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("CLIPTextEncode", "SaveImage"),
    )
    draft = _variant(
        "corrected-encoder",
        capability_hash="0" * 64,
        exact_run_hash="0" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("CLIPTextEncode", "SaveImage"),
        correction=correction,
    )
    corrected = draft.model_copy(
        update={
            "exact_run_hash": recompute_exact_run_hash(draft),
            "capability_hash": graph_capability_hash(
                draft.materialized_graph
            ),
        }
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(corrected, coverage=(DimensionName.ENCODER,)),
        history,
    )

    assert [item.variant_id for item in decision.variants] == ["corrected-encoder"]
    assert recompute_exact_run_hash(uncorrected) != corrected.exact_run_hash
    assert _plan(uncorrected).plan_hash != _plan(corrected).plan_hash


def _checkpoint_sampler_lora_graph(
    *,
    checkpoint: str = "base.safetensors",
    sampler: str = "euler",
    lora: str | None = None,
) -> dict[str, object]:
    graph: dict[str, object] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        }
    }
    model_node_id = "1"
    if lora is not None:
        graph["2"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        }
        model_node_id = "2"
    graph["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [model_node_id, 0],
            "sampler_name": sampler,
            "scheduler": "normal",
            "seed": 1,
        },
    }
    graph["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0]},
    }
    return graph


def test_sampler_only_correction_cannot_claim_checkpoint_dimension() -> None:
    signature = "provider-failed:unsupported sampler"
    failed_graph = _checkpoint_sampler_lora_graph(sampler="euler")
    failed = _variant(
        "failed",
        capability_hash=graph_capability_hash(failed_graph),
        exact_run_hash="1" * 64,
        graph=failed_graph,
    )
    candidate_graph = _checkpoint_sampler_lora_graph(sampler="dpmpp_2m")
    correction = {
        "schema_version": 1,
        "failure_signature": signature,
        "failed_variant_id": failed.variant_id,
        "failed_exact_run_hash": failed.exact_run_hash,
        "failed_capability_hash": failed.capability_hash,
        "candidate_variant_id": "candidate",
        "changed_dimensions": [DimensionName.CHECKPOINT.value],
    }
    candidate = _variant(
        "candidate",
        capability_hash=graph_capability_hash(candidate_graph),
        exact_run_hash="2" * 64,
        graph=candidate_graph,
        correction=correction,
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(_plan(candidate), history)

    assert decision.variants == ()
    assert decision.reasons[candidate.variant_id] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


def test_checkpoint_and_lora_correction_uses_actual_changed_dimensions() -> None:
    signature = "provider-failed:model combination"
    failed_graph = _checkpoint_sampler_lora_graph(
        checkpoint="base.safetensors",
        lora="old.safetensors",
    )
    failed = _variant(
        "failed",
        capability_hash=graph_capability_hash(failed_graph),
        exact_run_hash="3" * 64,
        graph=failed_graph,
    )
    candidate_graph = _checkpoint_sampler_lora_graph(
        checkpoint="revised.safetensors",
        lora="new.safetensors",
    )
    correction = {
        "schema_version": 1,
        "failure_signature": signature,
        "failed_variant_id": failed.variant_id,
        "failed_exact_run_hash": failed.exact_run_hash,
        "failed_capability_hash": failed.capability_hash,
        "candidate_variant_id": "candidate",
        "changed_dimensions": [
            DimensionName.CHECKPOINT.value,
            DimensionName.LORA_STACK.value,
        ],
    }
    candidate = _variant(
        "candidate",
        capability_hash=graph_capability_hash(candidate_graph),
        exact_run_hash="4" * 64,
        graph=candidate_graph,
        correction=correction,
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(
            candidate,
            coverage=(DimensionName.CHECKPOINT, DimensionName.LORA_STACK),
        ),
        history,
    )

    assert decision.variants == (candidate,)


def test_lora_correction_with_unknown_same_node_input_is_rejected() -> None:
    signature = "provider-failed:unapproved lora mode"
    failed_graph = _checkpoint_sampler_lora_graph(lora="old.safetensors")
    failed = _variant(
        "failed",
        capability_hash=graph_capability_hash(failed_graph),
        exact_run_hash="5" * 64,
        graph=failed_graph,
        hypothesis=DimensionName.LORA_STACK,
    )
    candidate_graph = _checkpoint_sampler_lora_graph(lora="new.safetensors")
    candidate_graph["2"]["inputs"]["unapproved_mode"] = "experimental"
    candidate = _variant(
        "candidate",
        capability_hash=graph_capability_hash(candidate_graph),
        exact_run_hash="6" * 64,
        graph=candidate_graph,
        hypothesis=DimensionName.LORA_STACK,
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [DimensionName.LORA_STACK.value],
        },
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(candidate, coverage=(DimensionName.LORA_STACK,)),
        history,
    )

    assert decision.variants == ()
    assert decision.reasons[candidate.variant_id] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


def test_checkpoint_correction_with_disconnected_custom_node_is_rejected() -> None:
    signature = "provider-failed:unexplained node"
    failed_graph = _checkpoint_sampler_lora_graph(
        checkpoint="base.safetensors",
    )
    failed = _variant(
        "failed",
        capability_hash=graph_capability_hash(failed_graph),
        exact_run_hash="5" * 64,
        graph=failed_graph,
    )
    candidate_graph = _checkpoint_sampler_lora_graph(
        checkpoint="revised.safetensors",
    )
    candidate_graph["90"] = {
        "class_type": "UnapprovedCustomNode",
        "inputs": {"mode": "checkpoint"},
    }
    candidate = _variant(
        "candidate",
        capability_hash=graph_capability_hash(candidate_graph),
        exact_run_hash="6" * 64,
        graph=candidate_graph,
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [DimensionName.CHECKPOINT.value],
        },
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(_plan(candidate), history)

    assert decision.variants == ()
    assert decision.reasons[candidate.variant_id] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


def test_same_graph_correction_cannot_fabricate_changed_capability_hash() -> None:
    signature = "provider-failed:unsupported configuration"
    failed_draft = _variant(
        "failed-baseline",
        capability_hash="0" * 64,
        exact_run_hash="5" * 64,
    )
    failed = failed_draft.model_copy(
        update={
            "capability_hash": graph_capability_hash(
                failed_draft.materialized_graph
            )
        }
    )
    candidate = _variant(
        "candidate",
        capability_hash="f" * 64,
        exact_run_hash="6" * 64,
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [DimensionName.CHECKPOINT.value],
        },
    )
    assert candidate.materialized_graph == failed.materialized_graph
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(_plan(candidate), history)

    assert decision.variants == ()
    assert decision.reasons[candidate.variant_id] == (
        "repeated_failure_requires_new_corrective_hypothesis"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("__record__", {}),
        ("candidate_variant_id", "different-candidate"),
        ("failed_exact_run_hash", "f" * 64),
        ("failed_capability_hash", "e" * 64),
        ("changed_dimensions", []),
    ],
)
def test_invalid_correction_identity_or_dimensions_cannot_select_candidate(
    field: str,
    value: object,
) -> None:
    signature = "missing-node:TextEncodeQwen"
    failed = _variant(
        "failed-encoder",
        capability_hash="8" * 64,
        exact_run_hash="5" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("TextEncodeQwen", "SaveImage"),
    )
    correction: dict[str, object] = {
        "schema_version": 1,
        "failure_signature": signature,
        "failed_variant_id": failed.variant_id,
        "failed_exact_run_hash": failed.exact_run_hash,
        "failed_capability_hash": failed.capability_hash,
        "candidate_variant_id": "candidate",
        "changed_dimensions": [DimensionName.ENCODER.value],
    }
    if field == "__record__":
        assert isinstance(value, dict)
        correction = {str(key): item for key, item in value.items()}
    else:
        correction[field] = value
    candidate = _variant(
        "candidate",
        capability_hash="9" * 64,
        exact_run_hash="6" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("CLIPTextEncode", "SaveImage"),
        correction=correction,
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(candidate, coverage=(DimensionName.ENCODER,)),
        history,
    )

    assert decision.variants == ()


def test_missing_node_correction_cannot_keep_failed_requirement() -> None:
    signature = "missing-node:TextEncodeQwen"
    failed = _variant(
        "failed-encoder",
        capability_hash="8" * 64,
        exact_run_hash="5" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("TextEncodeQwen", "SaveImage"),
    )
    candidate = _variant(
        "candidate",
        capability_hash="9" * 64,
        exact_run_hash="6" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("TextEncodeQwen", "SaveImage"),
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [DimensionName.ENCODER.value],
        },
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(candidate, coverage=(DimensionName.ENCODER,)),
        history,
    )

    assert decision.variants == ()


def test_correction_outside_plan_dimensions_cannot_select_candidate() -> None:
    signature = "missing-node:TextEncodeQwen"
    failed = _variant(
        "failed-encoder",
        capability_hash="8" * 64,
        exact_run_hash="5" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("TextEncodeQwen", "SaveImage"),
    )
    candidate = _variant(
        "candidate",
        capability_hash="9" * 64,
        exact_run_hash="6" * 64,
        hypothesis=DimensionName.ENCODER,
        required_nodes=("CLIPTextEncode", "SaveImage"),
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [
                DimensionName.ENCODER.value,
                DimensionName.SAMPLER.value,
            ],
        },
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(
        _plan(candidate, coverage=(DimensionName.ENCODER,)),
        history,
    )

    assert decision.variants == ()


def test_correction_must_change_failed_exact_and_capability_fingerprints() -> None:
    signature = "provider-failed:unsupported configuration"
    failed = _variant(
        "failed-baseline",
        capability_hash="8" * 64,
        exact_run_hash="5" * 64,
    )
    no_op = _variant(
        "candidate",
        capability_hash=failed.capability_hash,
        exact_run_hash=failed.exact_run_hash,
        correction={
            "schema_version": 1,
            "failure_signature": signature,
            "failed_variant_id": failed.variant_id,
            "failed_exact_run_hash": failed.exact_run_hash,
            "failed_capability_hash": failed.capability_hash,
            "candidate_variant_id": "candidate",
            "changed_dimensions": [DimensionName.CHECKPOINT.value],
        },
    )
    history = ExperimentHistory()
    history.record_failure(failed, signature=signature, run_id="run-a")
    history.record_failure(failed, signature=signature, run_id="run-b")

    decision = next_wave(_plan(no_op), history)

    assert decision.variants == ()


def test_exact_repeat_requires_recorded_rerun_reason_and_links_original_hash() -> None:
    repeated = _variant(
        "repeat-me",
        capability_hash="a" * 64,
        exact_run_hash="8" * 64,
    )

    with pytest.raises(RerunReasonRequired):
        request_rerun(repeated, reason="")

    approved = request_rerun(repeated, reason="measure output variance")

    assert approved.original_exact_run_hash == repeated.exact_run_hash
    assert approved.reason == "measure output variance"


def test_unapproved_dimension_is_not_submitted_from_an_old_plan() -> None:
    variant = _variant(
        "new-sampler",
        capability_hash="b" * 64,
        exact_run_hash="9" * 64,
        hypothesis=DimensionName.SAMPLER,
    )

    decision = next_wave(
        _plan(variant, coverage=(DimensionName.CHECKPOINT,)),
        ExperimentHistory(),
    )

    assert decision.variants == ()
    assert decision.reasons["new-sampler"] == "hypothesis_dimension_not_approved"
    assert decision.saturation_reason == (
        "no_approved_non_dominated_novel_capability_signature_remains"
    )


def test_next_wave_indexes_approval_evidence_once_for_512_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = tuple(
        _variant(
            f"variant-{index:03d}",
            capability_hash=_hash({"capability": index}),
            exact_run_hash=_hash({"exact": index}),
        )
        for index in range(512)
    )
    plan = _plan(*variants)
    calls = 0
    original = saturation_module._candidate_metadata

    def counted_metadata(candidate_plan: ExperimentPlan):
        nonlocal calls
        calls += 1
        return original(candidate_plan)

    monkeypatch.setattr(
        saturation_module,
        "_candidate_metadata",
        counted_metadata,
    )

    decision = next_wave(plan, ExperimentHistory())

    assert len(decision.variants) == 512
    assert calls == 1
