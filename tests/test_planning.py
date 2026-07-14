from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types
import ctypes

import pytest
from pydantic import ValidationError

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.evidence.models import (
    ConfidenceLabel,
    EvidenceGraph,
    EvidenceObservation,
    EvidenceSource,
    EvidenceSourceKind,
)
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
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash


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
                "model": ["4", 0], "positive": ["8", 0] if prompt_fanout else ["2", 0], "negative": ["2", 0],
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
        source_hash=source_graph_hash(raw_graph),
        capability_hash=capability_hash(raw_graph),
        evidence_ids=(workflow_id,),
        engagement=42,
    )


def _bundle(*, candidates: tuple[Candidate, ...] = (), graph: dict | None = None) -> SourceBundle:
    source = _source(graph)
    return SourceBundle(
        sources=(source,),
        candidates=candidates,
        evidence_graph=EvidenceGraph(
            sources=(EvidenceSource(
                source_id=source.workflow_id,
                kind=EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH,
                canonical_url=source.canonical_url,
                source_hash=source.source_hash,
                capability_hash=source.capability_hash,
            ),),
            observations=(EvidenceObservation(source_id=source.workflow_id),),
        ),
    )


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
        Candidate(candidate_id="weak", hypothesis="checkpoint", capability_signature="checkpoint:new", confidence=ConfidenceLabel.WEAKLY_INFERRED, evidence_ids=("123",), semantic_patch=patch, engagement=999),
        Candidate(candidate_id="observed", hypothesis="checkpoint", capability_signature="checkpoint:new", confidence=ConfidenceLabel.OBSERVED, evidence_ids=("123",), semantic_patch=patch, engagement=1),
    )
    plan = build_plan(_brief(), _bundle(candidates=candidates), config=_config(tmp_path))
    assert [item.variant_id for item in plan.variants] == ["baseline", "observed"]


def test_unexplored_candidate_is_retained_and_seed_candidate_requires_approval(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    unexplored = Candidate(candidate_id="unknown", hypothesis="sampler", capability_signature="sampler:new", confidence=ConfidenceLabel.UNEXPLORED, evidence_ids=("123",), semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.SAMPLER, binding_index=0, field_name="sampler_name", value="dpmpp_2m"),)))
    seed = Candidate(candidate_id="seed", hypothesis="seed_variance", capability_signature="seed:2", confidence=ConfidenceLabel.OBSERVED, evidence_ids=("123",), semantic_patch=SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.SEED, binding_index=0, field_name="seed", value=2),)))
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


def test_source_and_candidate_order_produce_identical_plan_bytes(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    first_source = _source(workflow_id="123")
    second_source = _source(workflow_id="124")
    evidence = EvidenceGraph(
        sources=tuple(
            EvidenceSource(
                source_id=item.workflow_id,
                kind=EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH,
                canonical_url=item.canonical_url,
                source_hash=item.source_hash,
                capability_hash=item.capability_hash,
            )
            for item in (first_source, second_source)
        ),
        observations=tuple(EvidenceObservation(source_id=item.workflow_id) for item in (first_source, second_source)),
    )
    patch = SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.CHECKPOINT, binding_index=0, field_name="ckpt_name", value="next.safetensors"),))
    candidates = (
        Candidate(candidate_id="one", hypothesis="checkpoint", capability_signature="checkpoint:one", confidence="observed", evidence_ids=("123",), semantic_patch=patch),
        Candidate(candidate_id="two", hypothesis="checkpoint", capability_signature="checkpoint:two", confidence="observed", evidence_ids=("124",), semantic_patch=patch),
    )
    first = build_plan(_brief(), SourceBundle(sources=(first_source, second_source), candidates=candidates, evidence_graph=evidence), config=_config(tmp_path))
    second = build_plan(_brief(), SourceBundle(sources=(second_source, first_source), candidates=tuple(reversed(candidates)), evidence_graph=evidence), config=_config(tmp_path))
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


@pytest.mark.skipif(os.name == "nt" or not hasattr(os, "mkfifo"), reason="POSIX special-file coverage")
def test_fixture_descriptor_walk_rejects_fifo_and_intermediate_symlink(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "fixture.pipe")
    fifo_brief = _brief(fixtures={"reference_images": [{"uri": "fixture.pipe", "binding_index": 0}], "prompt_lines": ["x"]})
    with pytest.raises(Exception, match="regular file"):
        build_plan(fifo_brief, _bundle(), config=_config(tmp_path))
    target = tmp_path / "real"
    target.mkdir()
    (target / "portrait.png").write_bytes(b"fixture")
    (tmp_path / "swapped").symlink_to(target, target_is_directory=True)
    race_brief = _brief(fixtures={"reference_images": [{"uri": "swapped/portrait.png", "binding_index": 0}], "prompt_lines": ["x"]})
    with pytest.raises(Exception):
        build_plan(race_brief, _bundle(), config=_config(tmp_path))


class _Win32Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def _mock_win32_handles(monkeypatch: pytest.MonkeyPatch, *, reparse_handles=(), final_paths=None):
    opened: list[str] = []
    closed: list[int] = []
    handles = iter((10, 11, 12))
    final_paths = final_paths or {10: r"\\?\C:\fixtures", 11: r"\\?\C:\fixtures\parent", 12: r"\\?\C:\fixtures\parent\fixture.png"}
    reparse_handles = set(reparse_handles)

    def create(path, *_args):
        opened.append(path)
        return next(handles)

    def information(handle, _kind, pointer, _size):
        info = pointer._obj
        info.attributes = 0x00000400 if handle in reparse_handles else 0
        info.reparse_tag = 0xA000000C if handle in reparse_handles else 0
        return True

    def final(handle, buffer, _size, _flags):
        value = final_paths[handle]
        if buffer is None:
            return len(value)
        buffer.value = value
        return len(value)

    kernel32 = types.SimpleNamespace(
        CreateFileW=_Win32Function(create),
        GetFileInformationByHandleEx=_Win32Function(information),
        GetFinalPathNameByHandleW=_Win32Function(final),
        GetFileType=_Win32Function(lambda _handle: 1),
        CloseHandle=_Win32Function(lambda handle: closed.append(handle) or True),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32, raising=False)
    msvcrt = types.SimpleNamespace(open_osfhandle=lambda *_args: pytest.fail("fixture bytes must not be readable"))
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    return opened, closed


@pytest.mark.parametrize(("reparse_handle", "opened_count"), ((11, 2), (12, 3)))
def test_windows_opened_reparse_metadata_fails_before_fixture_read(
    monkeypatch: pytest.MonkeyPatch,
    reparse_handle: int,
    opened_count: int,
) -> None:
    opened, closed = _mock_win32_handles(monkeypatch, reparse_handles=(reparse_handle,))
    with pytest.raises(Exception, match="reparse"):
        planning_module._windows_open_fixture(Path(r"C:\fixtures"), ("parent", "fixture.png"))
    assert len(opened) == opened_count
    assert closed == list(range(10, 10 + opened_count))


def test_windows_parent_swap_final_handle_path_fails_before_fixture_read(monkeypatch: pytest.MonkeyPatch) -> None:
    opened, closed = _mock_win32_handles(
        monkeypatch,
        final_paths={10: r"\\?\C:\fixtures", 11: r"\\?\D:\swapped-parent"},
    )
    with pytest.raises(Exception, match="escapes configured containment"):
        planning_module._windows_open_fixture(Path(r"C:\fixtures"), ("parent", "fixture.png"))
    assert len(opened) == 2
    assert closed == [10, 11]


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


def test_planning_source_recomputes_graph_normalization_and_hashes() -> None:
    graph = _graph()
    with pytest.raises(ValidationError, match="source_hash"):
        PlanningSource(
            workflow_id="123",
            canonical_url="https://www.runninghub.ai/post/123",
            raw_graph=graph,
            normalized_graph=normalize_graph(graph),
            source_hash="a" * 64,
            capability_hash=capability_hash(graph),
            evidence_ids=("123",),
        )
    with pytest.raises(ValidationError, match="normalized_graph"):
        PlanningSource(
            workflow_id="123",
            canonical_url="https://www.runninghub.ai/post/123",
            raw_graph=graph,
            normalized_graph=normalize_graph({**graph, "99": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}}}),
            source_hash=source_graph_hash(graph),
            capability_hash=capability_hash(graph),
            evidence_ids=("123",),
        )


def test_auto_counts_reachable_multi_outputs_but_rejects_disconnected_fanout(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    graph = _graph()
    graph["9"] = {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}}
    plan = build_plan(_brief(), _bundle(graph=graph), config=_config(tmp_path))
    assert plan.variants[0].expected_output_count == 6
    disconnected = _graph()
    disconnected["6"]["inputs"]["positive"] = ["2", 0]
    with pytest.raises(PlanningValidationError, match="reachable"):
        build_plan(_brief(), _bundle(graph=disconnected), config=_config(tmp_path))


def test_explicit_count_keeps_all_valid_output_nodes_and_infers_media(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    graph = _graph()
    graph["9"] = {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}}
    brief = _brief(expectations={"output_count": 7, "media_types": []})
    variant = build_plan(brief, _bundle(graph=graph), config=_config(tmp_path)).variants[0]
    assert variant.expected_output_count == 7
    assert variant.output_node_ids == ("3", "9")
    assert variant.expected_media_types == ("image/png", "image/png")


def test_candidate_patch_roles_must_match_approved_hypothesis_and_seed_policy(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    seed_patch = SemanticPatch(mutations=(
        SetRoleInput(role=SemanticRole.SEED, binding_index=0, field_name="seed", value=2),
    ))
    mislabeled = Candidate(
        candidate_id="sampler-seed",
        hypothesis="sampler",
        capability_signature="sampler-seed",
        confidence=ConfidenceLabel.OBSERVED,
        evidence_ids=("123",),
        semantic_patch=seed_patch,
    )
    plan = build_plan(_brief(), _bundle(candidates=(mislabeled,)), config=_config(tmp_path))
    assert [item.variant_id for item in plan.variants] == ["baseline"]
    assert "seed_variance" in " ".join(plan.warnings)
    approved = _brief(dimensions={"checkpoint": "approved", "prompt_grammar": "approved", "sampler": "approved", "seed_variance": "approved"})
    seed = Candidate(
        candidate_id="seed",
        hypothesis="seed_variance",
        capability_signature="seed-variance",
        confidence=ConfidenceLabel.OBSERVED,
        evidence_ids=("123",),
        semantic_patch=seed_patch,
    )
    assert [item.variant_id for item in build_plan(approved, _bundle(candidates=(seed,)), config=_config(tmp_path)).variants] == ["baseline", "seed"]


def test_candidate_ids_and_fixture_uris_are_safe_and_canonical() -> None:
    patch = SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.CHECKPOINT, binding_index=0, field_name="ckpt_name", value="x"),))
    with pytest.raises(ValidationError):
        Candidate(candidate_id="baseline", hypothesis="checkpoint", capability_signature="x", confidence="observed", evidence_ids=("123",), semantic_patch=patch)
    with pytest.raises(ValidationError):
        _brief(fixtures={"reference_images": [{"uri": "../secret.png", "binding_index": 0}], "prompt_lines": ["x"]})
    with pytest.raises(ValidationError):
        _brief(fixtures={"reference_images": [{"uri": "C:/secret.png", "binding_index": 0}], "prompt_lines": ["x"]})
    brief = _brief(fixtures={"reference_images": [{"uri": "vault://characters/nested/hero.png", "binding_index": 0}], "prompt_lines": ["x"]})
    assert brief.fixtures.reference_images[0].uri == "vault://characters/nested/hero.png"
    for unsafe in ("folder/CON", "portrait.png.", "portrait.png ", "nul"):
        with pytest.raises(ValidationError):
            _brief(fixtures={"reference_images": [{"uri": unsafe, "binding_index": 0}], "prompt_lines": ["x"]})
    with pytest.raises(ValidationError):
        Candidate(candidate_id="con", hypothesis="checkpoint", capability_signature="x", confidence="observed", evidence_ids=("123",), semantic_patch=patch)


def test_candidate_tie_prefers_provenance_before_engagement(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    patch = SemanticPatch(mutations=(SetRoleInput(role=SemanticRole.CHECKPOINT, binding_index=0, field_name="ckpt_name", value="new"),))
    weak = Candidate(candidate_id="engagement", hypothesis="checkpoint", capability_signature="same", confidence="observed", evidence_ids=("123",), semantic_patch=patch, engagement=999, provenance_rank=1)
    strong = Candidate(candidate_id="provenance", hypothesis="checkpoint", capability_signature="same", confidence="observed", evidence_ids=("123",), semantic_patch=patch, engagement=1, provenance_rank=2)
    plan = build_plan(_brief(), _bundle(candidates=(weak, strong)), config=_config(tmp_path))
    assert [item.variant_id for item in plan.variants] == ["baseline", "provenance"]


def test_exact_hash_binds_media_and_every_output_field(tmp_path: Path) -> None:
    (tmp_path / "portrait.png").write_bytes(b"fixture")
    first = build_plan(_brief(), _bundle(), config=_config(tmp_path)).variants[0]
    assert first.expected_media_types == ("image/png",)
    with pytest.raises(PlanningValidationError, match="media types"):
        build_plan(
            _brief(expectations={"output_count": "auto", "media_types": ["image/jpeg"]}),
            _bundle(),
            config=_config(tmp_path),
        )
