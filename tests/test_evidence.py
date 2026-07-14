from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
import pytest

from mp_comfyui_lab.discovery.models import (
    ContentClassification,
    CoverAsset,
    Engagement,
    WorkflowCandidate,
    WorkflowDetail,
)
from mp_comfyui_lab.evidence import build as evidence_build
from mp_comfyui_lab.evidence.build import assess_claim, build_evidence_graph
from mp_comfyui_lab.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    ClaimAssessment,
    ConfidenceLabel,
    CurrentEvidenceInput,
    EvidenceDimension,
    EvidenceGraph,
    EvidenceObservation,
    EvidenceSource,
    EvidenceSourceKind,
    EvidenceWarning,
    MutationClaim,
    PriorFinalistsArtifact,
    PriorHandoffArtifact,
)
from mp_comfyui_lab.executors.base import CapabilityReport
from mp_comfyui_lab.graph.models import strength_bucket
from mp_comfyui_lab.graph.normalize import normalize_graph


_DEFAULT_HASH = object()


def _node(class_type: str, **inputs: object) -> dict[str, object]:
    return {"class_type": class_type, "inputs": inputs}


def _workflow_graph(
    *,
    checkpoint: str = "qwen_image_edit_2511_bf16.safetensors",
    loras: tuple[str, ...] = (
        "multi-angle.safetensors",
        "consistency.safetensors",
    ),
    strengths: tuple[float, ...] = (0.25, 0.8),
    sampler: str = "sa_solver",
    scheduler: str = "simple",
    encoder: str = "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    reference_names: tuple[str, ...] = ("character.png", "style.png"),
    output_types: tuple[str, ...] = ("SaveImage",),
    prompt_fanout: bool = True,
) -> dict[str, dict[str, object]]:
    assert len(loras) == len(strengths)
    graph: dict[str, dict[str, object]] = {
        "10": _node("UNETLoader", unet_name=checkpoint),
        "20": _node("CLIPLoader", clip_name=encoder, type="qwen_image"),
        "30": _node("VAELoader", vae_name="qwen_image_vae.safetensors"),
    }

    model_source = "10"
    for index, (lora, strength) in enumerate(zip(loras, strengths, strict=True), 1):
        node_id = str(10 + index)
        graph[node_id] = _node(
            "LoraLoaderModelOnly",
            lora_name=lora,
            strength_model=strength,
            model=[model_source, 0],
        )
        model_source = node_id

    for index, reference_name in enumerate(reference_names, 1):
        graph[str(40 + index)] = _node("LoadImage", image=reference_name)

    if prompt_fanout:
        graph["50"] = _node(
            "easy promptLine",
            prompt="close-up | three-quarter | profile",
            start_index=0,
            max_rows=3,
        )
        prompt: object = ["50", 0]
    else:
        prompt = "close-up | three-quarter | profile"

    conditioning_inputs: dict[str, object] = {
        "clip": ["20", 0],
        "prompt": prompt,
    }
    for index in range(len(reference_names)):
        conditioning_inputs[f"image{index + 1}"] = [str(41 + index), 0]
    graph["60"] = _node("TextEncodeQwenImageEditPlus", **conditioning_inputs)
    graph["70"] = _node(
        "KSampler",
        model=[model_source, 0],
        positive=["60", 0],
        negative=["60", 0],
        latent_image=["30", 0],
        seed=7,
        sampler_name=sampler,
        scheduler=scheduler,
    )

    for index, output_type in enumerate(output_types, 1):
        graph[str(80 + index)] = _node(output_type, images=["70", 0])
    return graph


def _current_source(
    source_id: str = "workflow-1",
    *,
    graph: dict[str, dict[str, object]] | None = None,
    covers: bool = True,
    preview: bool = True,
    classification: ContentClassification = ContentClassification.SFW,
    engagement: Engagement | None = None,
    source_hash: str | None | object = _DEFAULT_HASH,
    capability_hash: str | None | object = _DEFAULT_HASH,
    source_kind: EvidenceSourceKind = EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH,
) -> CurrentEvidenceInput:
    raw_graph = graph or _workflow_graph()
    candidate = WorkflowCandidate(
        executor="runninghub",
        workflow_id=source_id,
        canonical_url=(
            f"https://www.runninghub.ai/post/{source_id}?token=must-not-persist"
        ),
        name=f"Evidence {source_id}",
        tags=("qwen", "multi-angle"),
        engagement=engagement or Engagement(uses=100, likes=25, collects=10),
        covers=(
            CoverAsset(
                url=f"https://assets.example/{source_id}.png?signature=secret"
            ),
        )
        if covers
        else (),
        content_classification=classification,
    )
    detail = WorkflowDetail(
        candidate=candidate,
        node_count=len(raw_graph),
        primitive_nodes=("KSampler", "SaveImage", "UNETLoader"),
        custom_nodes=("TextEncodeQwenImageEditPlus", "easy promptLine"),
        used_models=(
            "qwen_image_edit_2511_bf16.safetensors",
            "multi-angle.safetensors",
            "consistency.safetensors",
        ),
        preview=(
            CoverAsset(
                url=f"https://assets.example/{source_id}-preview.png?sig=secret"
            )
            if preview
            else None
        ),
    )
    return CurrentEvidenceInput(
        source_id=source_id,
        source_kind=source_kind,
        detail=detail,
        normalized_graph=normalize_graph(raw_graph),
        capability_report=CapabilityReport(
            required_nodes=("KSampler", "TextEncodeQwenImageEditPlus"),
            model_names=("qwen_image_edit_2511_bf16.safetensors",),
            lora_names=("multi-angle.safetensors", "consistency.safetensors"),
            output_node_ids=tuple(
                node_id
                for node_id, node in raw_graph.items()
                if str(node["class_type"]).startswith(("Save", "Preview"))
            ),
        ),
        source_hash=(
            (source_id.encode().hex() + "0" * 64)[:64]
            if source_hash is _DEFAULT_HASH
            else source_hash
        ),
        capability_hash=(
            (source_id.encode().hex() + "1" * 64)[:64]
            if capability_hash is _DEFAULT_HASH
            else capability_hash
        ),
    )


def _single_observation(source: CurrentEvidenceInput) -> EvidenceObservation:
    return build_evidence_graph((source,), prior_experiment_roots=()).observations[0]


def _claim(
    observation: EvidenceObservation,
    *,
    changed_dimensions: tuple[EvidenceDimension, ...] = (),
    **updates: object,
) -> MutationClaim:
    payload = {
        dimension.value: getattr(observation, dimension.value)
        for dimension in EvidenceDimension
    }
    payload.update(updates)
    payload["changed_dimensions"] = changed_dimensions
    return MutationClaim(**payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _prior_graph(source_id: str) -> EvidenceGraph:
    return build_evidence_graph(
        (_current_source(source_id),),
        prior_experiment_roots=(),
    )


def test_exact_qwen_multi_angle_consistency_with_visible_output_is_observed() -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    observation = graph.observations[0]

    assert observation.checkpoint == "qwen_image_edit_2511_bf16.safetensors"
    assert observation.loras == (
        "multi-angle.safetensors",
        "consistency.safetensors",
    )
    assert observation.visible_output.has_visible_output is True

    assessment = assess_claim(_claim(observation), graph)

    assert assessment.label is ConfidenceLabel.OBSERVED
    assert assessment.supporting_source_ids == ("workflow-1",)
    assert assessment.differing_dimensions == ()
    assert EvidenceDimension.CHECKPOINT in assessment.matched_dimensions
    assert "workflow-1" in assessment.explanation
    assert "current_normalized_graph" in assessment.explanation


def test_same_family_matching_topology_one_declared_change_is_strongly_inferred() -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    observation = graph.observations[0]
    claim = _claim(
        observation,
        checkpoint="qwen_image_edit_2511_fp8.safetensors",
        changed_dimensions=(EvidenceDimension.CHECKPOINT,),
    )

    assessment = assess_claim(claim, graph)

    assert assessment.label is ConfidenceLabel.STRONGLY_INFERRED
    assert assessment.differing_dimensions == (EvidenceDimension.CHECKPOINT,)
    assert assessment.supporting_source_ids == ("workflow-1",)
    assert "checkpoint" in assessment.explanation


def test_lora_order_and_loader_topology_mismatch_is_weakly_inferred() -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    observation = graph.observations[0]
    claim = _claim(
        observation,
        loras=tuple(reversed(observation.loras)),
        loader_topology=(*observation.loader_topology, "ExtraLoader.model"),
        changed_dimensions=(
            EvidenceDimension.LORAS,
            EvidenceDimension.LOADER_TOPOLOGY,
        ),
    )

    assessment = assess_claim(claim, graph)

    assert assessment.label is ConfidenceLabel.WEAKLY_INFERRED
    assert assessment.differing_dimensions == (
        EvidenceDimension.LORAS,
        EvidenceDimension.LOADER_TOPOLOGY,
    )


def test_no_model_family_support_is_unexplored_and_claim_is_retained() -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    observation = graph.observations[0]
    claim = _claim(
        observation,
        checkpoint="flux_2_dev_bf16.safetensors",
        model_family="flux_2_dev",
        changed_dimensions=(
            EvidenceDimension.CHECKPOINT,
            EvidenceDimension.MODEL_FAMILY,
        ),
    )

    assessment = assess_claim(claim, graph)

    assert assessment.label is ConfidenceLabel.UNEXPLORED
    assert assessment.supporting_source_ids == ()
    assert assessment.claim == claim
    assert "speculative claim retained" in assessment.explanation.casefold()


def test_unexplored_explanation_retains_considered_provenance_and_graph_warnings() -> None:
    shared_hash = "d" * 64
    flux_euler = _current_source(
        "flux-source",
        graph=_workflow_graph(
            checkpoint="flux_2_dev_bf16.safetensors",
            sampler="euler",
        ),
        source_hash=shared_hash,
    )
    flux_heun = _current_source(
        "flux-source",
        graph=_workflow_graph(
            checkpoint="flux_2_dev_bf16.safetensors",
            sampler="heun",
        ),
        source_hash=shared_hash,
    )
    graph = build_evidence_graph(
        (flux_euler, flux_heun),
        prior_experiment_roots=(),
    )
    claim = MutationClaim(
        checkpoint="qwen_image_edit_2511_bf16.safetensors",
        model_family="qwen_image_edit_2511",
    )

    assessment = assess_claim(claim, graph)

    assert assessment.label is ConfidenceLabel.UNEXPLORED
    assert assessment.supporting_source_ids == ()
    assert "flux-source" in assessment.explanation
    assert "current_normalized_graph" in assessment.explanation
    assert "checkpoint" in assessment.explanation
    assert "speculative claim retained" in assessment.explanation.casefold()
    assert any("conflicting payloads" in warning for warning in assessment.warnings)


def test_exact_stack_without_visible_output_is_downgraded() -> None:
    source = _current_source(covers=False, preview=False)
    graph = build_evidence_graph((source,), prior_experiment_roots=())
    observation = graph.observations[0]

    assert observation.output_node_count == 1
    assert observation.visible_output.output_role_node_ids
    assert observation.visible_output.has_visible_output is False
    assert assess_claim(_claim(observation), graph).label is ConfidenceLabel.WEAKLY_INFERRED


@pytest.mark.parametrize(
    ("covers", "preview", "output_types", "expected"),
    [
        (True, False, ("SaveImage",), True),
        (False, True, ("PreviewImage",), True),
        (True, True, (), False),
        (False, False, ("SaveImage",), False),
    ],
)
def test_visible_output_requires_visible_media_and_output_role_evidence(
    covers: bool,
    preview: bool,
    output_types: tuple[str, ...],
    expected: bool,
) -> None:
    observation = _single_observation(
        _current_source(
            graph=_workflow_graph(output_types=output_types),
            covers=covers,
            preview=preview,
        )
    )

    assert observation.visible_output.has_visible_output is expected


@pytest.mark.parametrize(
    ("strength", "bucket"),
    [
        (-2.000001, "negative_extreme"),
        (-2.0, "negative_strong"),
        (-1.25, "negative_standard"),
        (-0.75, "negative_light"),
        (-0.25, "negative_subtle"),
        (0.0, "zero"),
        (0.25, "positive_subtle"),
        (0.250001, "positive_light"),
        (0.75, "positive_light"),
        (0.750001, "positive_standard"),
        (1.25, "positive_standard"),
        (1.250001, "positive_strong"),
        (2.0, "positive_strong"),
        (2.000001, "positive_extreme"),
    ],
)
def test_lora_strength_ranges_use_stable_bucket_boundaries(
    strength: float,
    bucket: str,
) -> None:
    observation = _single_observation(
        _current_source(
            graph=_workflow_graph(
                loras=("multi-angle.safetensors",),
                strengths=(strength,),
            )
        )
    )

    record = observation.lora_strengths[0]
    assert (record.minimum, record.maximum, record.buckets) == (
        strength,
        strength,
        (bucket,),
    )
    assert strength_bucket(strength) == bucket


@pytest.mark.parametrize(
    ("dimension", "update"),
    [
        (EvidenceDimension.REFERENCE_COUNT, lambda o: o.reference_count + 1),
        (
            EvidenceDimension.REFERENCE_ROUTING,
            lambda o: tuple(reversed(o.reference_routing)),
        ),
        (EvidenceDimension.FANOUT, lambda o: (*o.fanout, "extra:linked_text:2")),
        (EvidenceDimension.OUTPUT_NODE_COUNT, lambda o: o.output_node_count + 1),
        (
            EvidenceDimension.OUTPUT_NODE_TYPES,
            lambda o: (*o.output_node_types, "PreviewImage"),
        ),
    ],
)
def test_reference_routing_fanout_and_output_dimensions_are_material(
    dimension: EvidenceDimension,
    update,
) -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    observation = graph.observations[0]
    assessment = assess_claim(
        _claim(
            observation,
            changed_dimensions=(dimension,),
            **{dimension.value: update(observation)},
        ),
        graph,
    )

    assert assessment.label is ConfidenceLabel.STRONGLY_INFERRED
    assert assessment.differing_dimensions == (dimension,)


def test_branch_and_slot_order_is_stable_and_material() -> None:
    forward = _single_observation(_current_source())
    reversed_stack = _single_observation(
        _current_source(
            source_id="workflow-2",
            graph=_workflow_graph(
                loras=("consistency.safetensors", "multi-angle.safetensors"),
            ),
        )
    )

    assert forward.loras == (
        "multi-angle.safetensors",
        "consistency.safetensors",
    )
    assert reversed_stack.loras == tuple(reversed(forward.loras))
    assert forward.loader_topology != reversed_stack.loader_topology


def test_multiple_best_supporting_sources_are_complete_and_sorted() -> None:
    sources = (
        _current_source("workflow-z"),
        _current_source("workflow-a"),
        _current_source("workflow-m"),
    )
    graph = build_evidence_graph(sources, prior_experiment_roots=())
    assessment = assess_claim(_claim(graph.observations[0]), graph)

    assert assessment.label is ConfidenceLabel.OBSERVED
    assert assessment.supporting_source_ids == (
        "workflow-a",
        "workflow-m",
        "workflow-z",
    )
    assert assessment.explanation.index("workflow-a") < assessment.explanation.index(
        "workflow-z"
    )


def test_many_supporting_sources_fit_the_declared_bounded_assessment() -> None:
    sources = tuple(
        _current_source(f"workflow-{index:03d}")
        for index in range(40)
    )
    graph = build_evidence_graph(sources, prior_experiment_roots=())

    assessment = assess_claim(_claim(graph.observations[0]), graph)

    assert len(assessment.supporting_source_ids) == 40
    assert assessment.supporting_source_ids == tuple(
        f"workflow-{index:03d}" for index in range(40)
    )
    assert all(
        source_id in assessment.explanation
        for source_id in assessment.supporting_source_ids
    )


def test_best_label_precedence_keeps_all_lower_confidence_support_ids() -> None:
    exact = _current_source("observed-source")
    reordered = _current_source(
        "weak-source",
        graph=_workflow_graph(
            loras=("consistency.safetensors", "multi-angle.safetensors"),
        ),
    )
    graph = build_evidence_graph((reordered, exact), prior_experiment_roots=())
    exact_observation = next(
        item for item in graph.observations if item.source_id == "observed-source"
    )

    assessment = assess_claim(_claim(exact_observation), graph)

    assert assessment.label is ConfidenceLabel.OBSERVED
    assert assessment.supporting_source_ids == ("observed-source", "weak-source")
    assert "weak-source" in assessment.explanation


def test_strong_inference_requires_claimed_loader_encoder_and_lora_topology() -> None:
    graph = build_evidence_graph((_current_source(),), prior_experiment_roots=())
    claim = MutationClaim(
        checkpoint="qwen_image_edit_2511_fp8.safetensors",
        changed_dimensions=(EvidenceDimension.CHECKPOINT,),
    )

    assessment = assess_claim(claim, graph)

    assert assessment.label is ConfidenceLabel.WEAKLY_INFERRED
    assert "topology" in assessment.explanation.casefold()


def test_nsfw_and_unknown_support_remains_visible() -> None:
    sources = (
        _current_source("nsfw-source", classification=ContentClassification.NSFW),
        _current_source(
            "unknown-source",
            classification=ContentClassification.UNKNOWN,
        ),
    )
    graph = build_evidence_graph(sources, prior_experiment_roots=())
    assessment = assess_claim(_claim(graph.observations[0]), graph)

    assert assessment.supporting_source_ids == ("nsfw-source", "unknown-source")
    assert {item.content_classification for item in graph.observations} == {
        ContentClassification.NSFW,
        ContentClassification.UNKNOWN,
    }
    assert any("NSFW" in warning for warning in assessment.warnings)


def test_popularity_alone_never_implies_model_compatibility() -> None:
    popular_flux = _current_source(
        graph=_workflow_graph(checkpoint="flux_2_dev_bf16.safetensors"),
        engagement=Engagement(uses=9_999_999, likes=9_999_999, collects=9_999_999),
    )
    graph = build_evidence_graph((popular_flux,), prior_experiment_roots=())
    observation = graph.observations[0]
    claim = _claim(
        observation,
        checkpoint="qwen_image_edit_2511_bf16.safetensors",
        model_family="qwen_image_edit_2511",
        changed_dimensions=(
            EvidenceDimension.CHECKPOINT,
            EvidenceDimension.MODEL_FAMILY,
        ),
    )

    assert assess_claim(claim, graph).label is ConfidenceLabel.UNEXPLORED


def test_assessment_is_guidance_and_does_not_manufacture_validation_errors() -> None:
    assessment = assess_claim(
        MutationClaim(
            checkpoint="never-seen.safetensors",
            changed_dimensions=(EvidenceDimension.CHECKPOINT,),
        ),
        EvidenceGraph(),
    )

    assert assessment.label is ConfidenceLabel.UNEXPLORED
    assert assessment.warnings == ()
    assert not hasattr(assessment, "errors")


def test_mutation_claim_requires_exact_unique_declared_dimensions() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        MutationClaim(
            checkpoint="qwen.safetensors",
            changed_dimensions=(
                EvidenceDimension.CHECKPOINT,
                EvidenceDimension.CHECKPOINT,
            ),
        )

    with pytest.raises(ValidationError, match="must have a claim value"):
        MutationClaim(changed_dimensions=(EvidenceDimension.CHECKPOINT,))

    with pytest.raises(ValidationError):
        MutationClaim(
            checkpoint="qwen.safetensors",
            changed_dimensions=("checkpoint",),
        )


def test_evidence_models_are_strict_deeply_immutable_and_detached() -> None:
    raw = _workflow_graph()
    source = _current_source(graph=raw)
    raw_before = deepcopy(raw)
    graph = build_evidence_graph((source,), prior_experiment_roots=())
    observation = graph.observations[0]

    raw["10"]["inputs"]["unet_name"] = "mutated.safetensors"

    assert observation.checkpoint == "qwen_image_edit_2511_bf16.safetensors"
    assert raw_before != raw
    with pytest.raises(ValidationError):
        graph.sources = ()
    with pytest.raises(ValidationError):
        observation.lora_strengths[0].buckets = ()
    with pytest.raises(ValidationError):
        EvidenceWarning(code="x", reason="y", unknown="nope")


def test_schema_source_kind_and_confidence_enums_are_exact() -> None:
    assert tuple(item.value for item in ConfidenceLabel) == (
        "observed",
        "strongly_inferred",
        "weakly_inferred",
        "unexplored",
    )
    assert {item.value for item in EvidenceSourceKind} == {
        "current_discovery_workflow",
        "current_workflow_detail",
        "current_normalized_graph",
        "prior_evidence_graph",
        "prior_finalists",
        "prior_handoff",
    }
    assert EVIDENCE_SCHEMA_VERSION == "1"


def test_current_source_uses_typed_detail_graph_capability_and_sanitized_provenance() -> None:
    source = _current_source(
        source_kind=EvidenceSourceKind.CURRENT_WORKFLOW_DETAIL,
    )
    graph = build_evidence_graph((source,), prior_experiment_roots=())
    provenance = graph.sources[0]

    assert isinstance(source.detail, WorkflowDetail)
    assert source.normalized_graph.__class__.__name__ == "NormalizedGraph"
    assert isinstance(source.capability_report, CapabilityReport)
    assert provenance.kind is EvidenceSourceKind.CURRENT_WORKFLOW_DETAIL
    assert provenance.source_hash == source.source_hash
    assert provenance.capability_hash == source.capability_hash
    assert provenance.canonical_url == "https://www.runninghub.ai/post/workflow-1"
    assert all("?" not in url for url in graph.observations[0].visible_output.cover_urls)


def test_duplicate_source_identity_prefers_hash_then_capability_then_source_id() -> None:
    same_source_hash = "a" * 64
    sources = (
        _current_source("z-id", source_hash=same_source_hash, capability_hash="1" * 64),
        _current_source("a-id", source_hash=same_source_hash, capability_hash="1" * 64),
        _current_source("cap-z", source_hash=None, capability_hash="b" * 64),
        _current_source("cap-a", source_hash=None, capability_hash="b" * 64),
    )

    graph = build_evidence_graph(sources, prior_experiment_roots=())

    assert [source.source_id for source in graph.sources] == ["a-id", "cap-a"]
    assert [observation.source_id for observation in graph.observations] == [
        "a-id",
        "cap-a",
    ]


def test_conflicting_duplicate_payload_warns_deterministically_never_last_wins() -> None:
    shared_hash = "c" * 64
    first = _current_source(
        "same-id",
        graph=_workflow_graph(sampler="euler"),
        source_hash=shared_hash,
    )
    second = _current_source(
        "same-id",
        graph=_workflow_graph(sampler="heun"),
        source_hash=shared_hash,
    )

    graph_a = build_evidence_graph((first, second), prior_experiment_roots=())
    graph_b = build_evidence_graph((second, first), prior_experiment_roots=())

    assert graph_a.canonical_json() == graph_b.canonical_json()
    assert len(graph_a.observations) == 1
    assert any(warning.code == "conflicting_duplicate" for warning in graph_a.warnings)
    assert "last" not in graph_a.warnings[0].reason.casefold()


def test_stable_graph_serialization_hash_and_order_ignore_input_order() -> None:
    sources = (
        _current_source("workflow-z"),
        _current_source("workflow-a"),
    )
    graph_a = build_evidence_graph(sources, prior_experiment_roots=())
    graph_b = build_evidence_graph(tuple(reversed(sources)), prior_experiment_roots=())

    assert [source.source_id for source in graph_a.sources] == [
        "workflow-a",
        "workflow-z",
    ]
    assert graph_a.canonical_json() == graph_b.canonical_json()
    assert graph_a.sha256() == graph_b.sha256()
    assert len(graph_a.sha256()) == 64
    assert json.loads(graph_a.canonical_json())["schema_version"] == "1"


def test_prior_evidence_graph_is_reinterpreted_with_relative_path_and_content_hash(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiments" / "prior-exp"
    artifact = experiment / "evidence" / "graph.json"
    _write_json(artifact, _prior_graph("prior-evidence"))

    graph = build_evidence_graph((), prior_experiment_roots=(tmp_path,))

    source = next(item for item in graph.sources if item.source_id == "prior-evidence")
    assert source.kind is EvidenceSourceKind.PRIOR_EVIDENCE_GRAPH
    assert source.artifact_path == "experiments/prior-exp/evidence/graph.json"
    assert source.content_sha256 is not None and len(source.content_sha256) == 64
    assert "prior-evidence" in {item.source_id for item in graph.observations}


@pytest.mark.parametrize("root_shape", ["project", "experiments", "experiment"])
def test_prior_finalists_and_handoff_support_allowed_root_shapes(
    tmp_path: Path,
    root_shape: str,
) -> None:
    experiments = tmp_path / "experiments"
    experiment = experiments / "prior-exp"
    finalist = _prior_graph("prior-finalist").observations[0]
    handoff = _prior_graph("prior-handoff").observations[0]
    _write_json(
        experiment / "finalists.json",
        PriorFinalistsArtifact(finalists=(finalist,)),
    )
    _write_json(
        experiment / "export" / "selected" / "handoff.json",
        PriorHandoffArtifact(selected=handoff),
    )
    root = {
        "project": tmp_path,
        "experiments": experiments,
        "experiment": experiment,
    }[root_shape]

    graph = build_evidence_graph((), prior_experiment_roots=(root,))

    assert {item.source_id for item in graph.observations} == {
        "prior-finalist",
        "prior-handoff",
    }
    assert {item.kind for item in graph.sources} == {
        EvidenceSourceKind.PRIOR_FINALISTS,
        EvidenceSourceKind.PRIOR_HANDOFF,
    }


def test_prior_scan_reads_only_known_schema_locations_and_creates_no_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "experiments" / "prior-exp"
    _write_json(experiment / "evidence" / "graph.json", _prior_graph("allowed"))
    unrelated = {
        tmp_path / "random.json",
        experiment / "notes.json",
        experiment / ".hidden" / "catalog.json",
        experiment / "compatibility.db",
        experiment / "catalog.json",
    }
    for path in unrelated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"schema_version":"1","secret":"do-not-read"}', encoding="utf-8")

    opened: list[Path] = []
    original_open = Path.open

    def open_spy(path: Path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_spy)

    graph = build_evidence_graph((), prior_experiment_roots=(tmp_path,))

    assert {item.source_id for item in graph.observations} == {"allowed"}
    assert set(opened) == {experiment / "evidence" / "graph.json"}
    assert not (tmp_path / "compatibility.db").exists()
    assert not any(path.name == "catalog.json" for path in opened)


def test_unknown_malformed_oversized_deep_and_nonfinite_prior_files_warn_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "experiments" / "prior-exp"
    unknown = experiment / "evidence" / "graph.json"
    malformed = experiment / "finalists.json"
    oversized = experiment / "export" / "oversized" / "handoff.json"
    deep = experiment / "export" / "deep" / "handoff.json"
    nonfinite = experiment / "export" / "nonfinite" / "handoff.json"
    _write_json(unknown, {"schema_version": "999", "token": "raw-secret"})
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text('{"schema_version":"1","token":"raw-secret",', encoding="utf-8")
    oversized.parent.mkdir(parents=True, exist_ok=True)
    oversized.write_text("x" * 257, encoding="utf-8")
    nested: object = "leaf"
    for _ in range(20):
        nested = [nested]
    _write_json(deep, {"schema_version": "1", "selected": nested})
    nonfinite.parent.mkdir(parents=True, exist_ok=True)
    nonfinite.write_text(
        '{"schema_version":"1","selected":{"score":NaN}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(evidence_build, "MAX_PRIOR_FILE_BYTES", 256)
    monkeypatch.setattr(evidence_build, "MAX_PRIOR_JSON_DEPTH", 12)

    graph = build_evidence_graph((), prior_experiment_roots=(tmp_path,))

    assert graph.observations == ()
    assert {warning.code for warning in graph.warnings} >= {
        "unknown_schema",
        "malformed_artifact",
        "oversized_artifact",
        "unsafe_json",
    }
    warning_text = " ".join(warning.reason for warning in graph.warnings)
    assert "raw-secret" not in warning_text
    assert "token" not in warning_text
    assert all(not Path(warning.path).is_absolute() for warning in graph.warnings)


def test_prior_json_with_duplicate_object_keys_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "experiments" / "prior-exp" / "finalists.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        '{"schema_version":"1","schema_version":"1","finalists":[]}',
        encoding="utf-8",
    )

    graph = build_evidence_graph((), prior_experiment_roots=(tmp_path,))

    assert graph.observations == ()
    assert len(graph.warnings) == 1
    assert graph.warnings[0].code == "malformed_artifact"
    assert "duplicate" in graph.warnings[0].reason.casefold()


def test_symlinked_artifact_and_export_directory_are_rejected_without_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "handoff.json"
    _write_json(
        external,
        PriorHandoffArtifact(selected=_prior_graph("escaped").observations[0]),
    )
    experiment = tmp_path / "experiments" / "prior-exp"
    direct_link = experiment / "evidence" / "graph.json"
    direct_link.parent.mkdir(parents=True)
    export_link = experiment / "export" / "linked"
    export_link.parent.mkdir(parents=True)
    try:
        direct_link.symlink_to(external)
        export_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    graph = build_evidence_graph((), prior_experiment_roots=(tmp_path,))

    assert graph.observations == ()
    assert any(warning.code == "unsafe_path" for warning in graph.warnings)
    assert "escaped" not in graph.canonical_json()


def test_root_and_file_count_limits_fail_closed(tmp_path: Path, monkeypatch) -> None:
    roots = tuple(tmp_path / f"root-{index:03d}" for index in range(3))
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(evidence_build, "MAX_PRIOR_ROOTS", 2)

    with pytest.raises(evidence_build.EvidenceScanError, match="root limit"):
        build_evidence_graph((), prior_experiment_roots=roots)

    experiment = tmp_path / "bounded" / "experiments" / "exp"
    for index in range(3):
        _write_json(
            experiment / "export" / str(index) / "handoff.json",
            PriorHandoffArtifact(
                selected=_prior_graph(f"prior-{index}").observations[0]
            ),
        )
    monkeypatch.setattr(evidence_build, "MAX_PRIOR_ROOTS", 10)
    monkeypatch.setattr(evidence_build, "MAX_PRIOR_FILES", 2)

    with pytest.raises(evidence_build.EvidenceScanError, match="file limit"):
        build_evidence_graph(
            (),
            prior_experiment_roots=(tmp_path / "bounded",),
        )


def test_prior_scan_is_deterministic_and_linear_across_many_roots(tmp_path: Path) -> None:
    roots: list[Path] = []
    for index in range(40):
        root = tmp_path / f"root-{index:03d}"
        roots.append(root)
        _write_json(
            root / "finalists.json",
            PriorFinalistsArtifact(
                finalists=(_prior_graph(f"prior-{index:03d}").observations[0],)
            ),
        )

    started = perf_counter()
    graph = build_evidence_graph((), prior_experiment_roots=tuple(reversed(roots)))
    elapsed = perf_counter() - started

    assert len(graph.observations) == 40
    assert [item.source_id for item in graph.observations] == sorted(
        item.source_id for item in graph.observations
    )
    assert elapsed < 2.0


def test_prior_artifact_models_reject_unknown_fields_and_wrong_schema() -> None:
    observation = _prior_graph("prior").observations[0]
    with pytest.raises(ValidationError):
        PriorFinalistsArtifact(
            schema_version="999",
            finalists=(observation,),
        )
    with pytest.raises(ValidationError):
        PriorHandoffArtifact(selected=observation, hidden_catalog="forbidden")


def test_graph_model_rejects_conflicting_duplicate_records() -> None:
    source = EvidenceSource(
        source_id="duplicate",
        kind=EvidenceSourceKind.PRIOR_HANDOFF,
        artifact_path="export/one/handoff.json",
        content_sha256="a" * 64,
    )
    other = source.model_copy(update={"artifact_path": "export/two/handoff.json"})

    with pytest.raises(ValidationError, match="duplicate source_id"):
        EvidenceGraph(sources=(source, other))


def test_claim_assessment_is_strict_and_frozen() -> None:
    claim = MutationClaim(checkpoint="qwen.safetensors")
    assessment = ClaimAssessment(
        claim=claim,
        label=ConfidenceLabel.UNEXPLORED,
        explanation="No matching support; speculative claim retained.",
    )

    with pytest.raises(ValidationError):
        assessment.label = ConfidenceLabel.OBSERVED
    with pytest.raises(ValidationError):
        ClaimAssessment(
            claim=claim,
            label=ConfidenceLabel.UNEXPLORED,
            explanation="ok",
            supporting_source_ids=("z", "a"),
        )
