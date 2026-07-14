from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.evidence.models import ConfidenceLabel
from mp_comfyui_lab.experiments.planning import (
    PlanningValidationError,
    build_plan,
    load_experiment_brief,
)
from mp_comfyui_lab.experiments import planning as planning_module
from mp_comfyui_lab.experiments.models import (
    Candidate,
    CandidateClaim,
    ExperimentBrief,
    PlanningSource,
    SourceBundle,
)
from mp_comfyui_lab.mutations import SemanticPatch, SetRoleInput
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import normalize_graph


def _graph(*, reference_count: int = 1, prompt_fanout: bool = True) -> dict:
    graph: dict[str, dict] = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "original.png"},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "source prompt", "clip": ["4", 0]},
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": "out"},
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["4", 2]}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0], "positive": ["2", 0], "negative": ["2", 0],
                "latent_image": ["7", 0], "seed": 1, "sampler_name": "euler", "scheduler": "normal", "steps": 10, "cfg": 7.0,
            },
        },
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    }
    for index in range(1, reference_count):
        graph[str(10 + index)] = {
            "class_type": "LoadImage",
            "inputs": {"image": f"original-{index}.png"},
        }
    if prompt_fanout:
        graph["8"] = {
            "class_type": "easy promptLine",
            "inputs": {"prompt": ["2", 0], "start_index": 0, "max_rows": 1000},
        }
    return graph


def _brief(**overrides: object) -> ExperimentBrief:
    values = {
        "goal": "Test a character variation",
        "sources": {"queries": ["qwen character"], "urls": []},
        "fixtures": {
            "reference_images": [{"uri": "portrait.png", "binding_index": 0}],
            "prompt_lines": ["wide shot", "close portrait", "night scene"],
        },
        "expectations": {"output_count": "auto", "media_types": ["image/png"]},
        "dimensions": {
            "checkpoint": "approved",
            "prompt_grammar": "approved",
            "sampler": "approved",
        },
    }
    values.update(overrides)
    return ExperimentBrief.model_validate(values)


def _source(graph: dict | None = None, *, workflow_id: str = "123") -> PlanningSource:
    raw_graph = graph or _graph()
    return PlanningSource(
        workflow_id=workflow_id,
        canonical_url=f"https://www.runninghub.ai/post/{workflow_id}",
        raw_graph=raw_graph,
        normalized_graph=normalize_graph(raw_graph),
        source_hash="a" * 64,
        capability_hash="b" * 64,
        evidence_ids=("evidence-1",),
        engagement=42,
    )


def _bundle(*, candidates: tuple[Candidate, ...] = (), graph: dict | None = None) -> SourceBundle:
    return SourceBundle(sources=(_source(graph),), candidates=candidates)


def _config(tmp_path: Path) -> LabConfig:
    return LabConfig.model_validate(
        {"experiments_root": tmp_path / "experiments"},
        context={"base_dir": tmp_path},
    )


def test_manifest_is_strict_and_forbids_budget_task_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _brief(max_tasks=3)
    with pytest.raises(ValidationError):
        _brief(budget=1)
    with pytest.raises(ValidationError):
        _brief(dimensions={"checkpoint": "approved", "unknown": "approved"})
    with pytest.raises(ValidationError):
        _brief(dimensions={"checkpoint": "anything"})


def test_load_manifest_rejects_duplicate_nonfinite_oversize_and_deep_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "brief.yaml"
    manifest.write_text("goal: one\ngoal: two\n", encoding="utf-8")
    with pytest.raises(PlanningValidationError, match="duplicate YAML key"):
        load_experiment_brief(manifest)
    manifest.write_text("goal: .nan\n", encoding="utf-8")
    with pytest.raises(PlanningValidationError):
        load_experiment_brief(manifest)
    manifest.write_text("x: " + "[" * 60 + "]" * 60, encoding="utf-8")
    with pytest.raises(PlanningValidationError):
        load_experiment_brief(manifest)
    manifest.write_bytes(b"x" * 1_048_577)
    with pytest.raises(PlanningValidationError):
        load_experiment_brief(manifest)


def test_reference_fixture_bindings_are_ordered_unique_and_prompt_lines_are_exact() -> None:
    brief = _brief(
        fixtures={
            "reference_images": [
                {"uri": "a.png", "binding_index": 2},
                {"uri": "b.png", "binding_index": 0},
            ],
            "prompt_lines": ["  exact first  ", "second"],
        }
    )
    assert [item.binding_index for item in brief.fixtures.reference_images] == [2, 0]
    assert brief.fixtures.prompt_lines == ("  exact first  ", "second")
    with pytest.raises(ValidationError):
        _brief(fixtures={"reference_images": [{"uri": "a", "binding_index": 0}, {"uri": "b", "binding_index": 0}], "prompt_lines": ["x"]})


def test_sources_canonicalize_runninghub_urls_and_strip_signed_query_data() -> None:
    brief = _brief(
        sources={"queries": [], "urls": ["https://www.runninghub.ai/post/123?token=secret"]}
    )
    assert brief.sources.urls == ("https://www.runninghub.ai/post/123",)
    with pytest.raises(ValidationError):
        _brief(sources={"queries": [], "urls": ["https://alice:secret@www.runninghub.ai/post/123"]})


def test_auto_three_prompt_lines_resolve_to_three_outputs(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    plan = build_plan(_brief(), _bundle(), config=_config(tmp_path))
    assert plan.variants[0].expected_output_count == 3
    assert plan.variants[0].expected_media_types == ("image/png",)


def test_unresolved_auto_cardinality_requires_explicit_integer(tmp_path: Path) -> None:
    with pytest.raises(PlanningValidationError, match="explicit integer"):
        build_plan(_brief(), _bundle(graph=_graph(prompt_fanout=False)), config=_config(tmp_path))


def test_fixture_fingerprints_change_plan_hash_without_persisting_absolute_path(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(b"first bytes")
    first = build_plan(_brief(), _bundle(), config=_config(tmp_path))
    portrait.write_bytes(b"second bytes")
    second = build_plan(_brief(), _bundle(), config=_config(tmp_path))
    assert first.plan_hash != second.plan_hash
    dumped = first.canonical_json()
    assert str(portrait) not in dumped
    assert "portrait.png" in dumped


def test_prompt_order_and_dimension_approval_participate_in_plan_hash(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    first = build_plan(_brief(), _bundle(), config=_config(tmp_path))
    reversed_prompts = _brief(fixtures={"reference_images": [{"uri": "portrait.png", "binding_index": 0}], "prompt_lines": ["night scene", "close portrait", "wide shot"]})
    changed_dimensions = _brief(dimensions={"checkpoint": "approved", "prompt_grammar": "approved", "sampler": "approved", "scheduler": "approved"})
    assert first.plan_hash != build_plan(reversed_prompts, _bundle(), config=_config(tmp_path)).plan_hash
    assert first.plan_hash != build_plan(changed_dimensions, _bundle(), config=_config(tmp_path)).plan_hash


def test_only_declared_reference_binding_is_overridden(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    graph = _graph(reference_count=3)
    plan = build_plan(_brief(), _bundle(graph=graph), config=_config(tmp_path))
    materialized = plan.variants[0].materialized_graph
    normalized = normalize_graph(materialized)
    references = normalized.bindings(SemanticRole.REFERENCE_IMAGE)
    assert references[0].value == "fixture://0"
    assert [item.value for item in references[1:]] == ["original-1.png", "original-2.png"]


def test_candidate_selection_prefers_confidence_and_one_capability_representative(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    patch = SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.CHECKPOINT, binding_index=0, field_name="ckpt_name", value="new.safetensors"),))
    candidates = (
        Candidate(candidate_id="weak", hypothesis="checkpoint", capability_signature="checkpoint:new", confidence=ConfidenceLabel.WEAKLY_INFERRED, evidence_ids=("e2",), semantic_patch=patch, engagement=999),
        Candidate(candidate_id="observed", hypothesis="checkpoint", capability_signature="checkpoint:new", confidence=ConfidenceLabel.OBSERVED, evidence_ids=("e1",), semantic_patch=patch, engagement=1),
    )
    plan = build_plan(_brief(), _bundle(candidates=candidates), config=_config(tmp_path))
    assert [item.variant_id for item in plan.variants] == ["baseline", "observed"]


def test_unexplored_candidate_is_retained_and_seed_candidate_requires_approval(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    unexplored = Candidate(candidate_id="unknown", hypothesis="sampler", capability_signature="sampler:new", confidence=ConfidenceLabel.UNEXPLORED, evidence_ids=("e1",), semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.SAMPLER, binding_index=0, field_name="sampler_name", value="dpmpp_2m"),)))
    seed = Candidate(candidate_id="seed", hypothesis="seed_variance", capability_signature="seed:2", confidence=ConfidenceLabel.OBSERVED, evidence_ids=("e1",), semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.SEED, binding_index=0, field_name="seed", value=2),)))
    plan = build_plan(_brief(), _bundle(candidates=(seed, unexplored)), config=_config(tmp_path))
    assert [item.variant_id for item in plan.variants] == ["baseline", "unknown"]
    assert "unexplored" in " ".join(plan.variants[-1].warnings)


def test_source_and_mapping_order_produce_byte_identical_canonical_plan_json(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    graph = _graph()
    reordered = dict(reversed(tuple(graph.items())))
    first = build_plan(_brief(), _bundle(graph=graph), config=_config(tmp_path))
    second = build_plan(_brief(), _bundle(graph=reordered), config=_config(tmp_path))
    assert first.canonical_json() == second.canonical_json()
    assert first.plan_hash == second.plan_hash


def test_fixture_resolution_rejects_symlink_special_and_changed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"fixture")
    link = tmp_path / "link.png"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    brief = _brief(fixtures={"reference_images": [{"uri": "link.png", "binding_index": 0}], "prompt_lines": ["x"]})
    with pytest.raises(Exception):
        build_plan(brief, _bundle(), config=_config(tmp_path))


def test_fixture_fingerprint_detects_a_change_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "portrait.png"
    fixture.write_bytes(b"fixture")
    original_fstat = planning_module.os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        result = original_fstat(descriptor)
        if calls == 1:
            return result
        return type("ChangedStat", (), {
            "st_mode": result.st_mode,
            "st_size": result.st_size,
            "st_dev": result.st_dev,
            "st_ino": result.st_ino,
            "st_mtime_ns": result.st_mtime_ns + 1,
        })()

    monkeypatch.setattr(planning_module.os, "fstat", changed_fstat)
    with pytest.raises(Exception, match="changed during read"):
        build_plan(_brief(), _bundle(), config=_config(tmp_path))


def test_candidate_claims_are_strict_and_explicit_when_no_safe_synthesis_exists() -> None:
    with pytest.raises(ValidationError):
        CandidateClaim.model_validate({"dimension": "checkpoint", "value": "x", "extra": "no"})
