from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
from math import isclose, isfinite, log1p
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import ValidationError
import pytest

from mp_comfyui_lab.config import LabConfig
from mp_comfyui_lab.discovery.models import (
    CoverAsset,
    Engagement,
    WorkflowCandidate,
    WorkflowDetail,
)
from mp_comfyui_lab.discovery.query import (
    ExpandedQuery,
    QueryExpansionConfigError,
    expand_query,
    load_query_expansions,
    parse_query_expansions_yaml,
    term_match,
)
from mp_comfyui_lab.discovery.ranking import (
    POSITIVE_WEIGHT_FIELDS,
    GraphEvidence,
    RankingConfigError,
    RankingWeights,
    load_default_ranking_weights,
    rank_candidates,
    select_group_representatives,
)
from mp_comfyui_lab.discovery import ranking as ranking_module
from mp_comfyui_lab.discovery import service as service_module
from mp_comfyui_lab.discovery.service import DiscoveryNotFoundError, DiscoveryService
from mp_comfyui_lab.discovery.urls import UnsupportedSourceError
from mp_comfyui_lab.executors.base import (
    CapabilityReport,
    FetchedWorkflow,
    SearchPage,
    SearchRequest,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.graph.hashing import source_graph_hash
from mp_comfyui_lab.json_types import FrozenDict


NEXT_SCENE_ID = "1980860923768426497"
QWEN_ID = "2004195228674387969"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "runninghub"


def _candidate(
    workflow_id: str,
    *,
    name: str,
    description: str | None = None,
    author: str | None = "creator-a",
    published_at: datetime | None = None,
    tags: tuple[str, ...] = (),
    uses: int = 0,
    likes: int = 0,
    collects: int = 0,
    covers: bool = True,
    provider_metadata: Mapping[str, Any] | None = None,
) -> WorkflowCandidate:
    return WorkflowCandidate(
        executor="runninghub",
        workflow_id=workflow_id,
        canonical_url=f"https://www.runninghub.ai/post/{workflow_id}",
        name=name,
        description=description,
        author_id=author,
        author_name=author,
        published_at=published_at,
        tags=tags,
        engagement=Engagement(uses=uses, likes=likes, collects=collects),
        covers=(
            CoverAsset(url=f"https://assets.example/{workflow_id}.png"),
        )
        if covers
        else (),
        provider_metadata=provider_metadata or {},
    )


def _detail(
    candidate: WorkflowCandidate,
    *,
    used_models: tuple[str, ...] = (),
    primitive_nodes: tuple[str, ...] = (),
    custom_nodes: tuple[str, ...] = (),
    preview: bool = False,
) -> WorkflowDetail:
    return WorkflowDetail(
        candidate=candidate,
        node_count=len(primitive_nodes) + len(custom_nodes),
        used_models=used_models,
        primitive_nodes=primitive_nodes,
        custom_nodes=custom_nodes,
        preview=(
            CoverAsset(url=f"https://assets.example/{candidate.workflow_id}-preview.png")
            if preview
            else None
        ),
    )


def _golden_candidates() -> tuple[
    tuple[WorkflowCandidate, ...], dict[str, WorkflowDetail]
]:
    next_scene = _candidate(
        NEXT_SCENE_ID,
        name="Qwen Next Scene cinematic storyboard",
        description="Character consistency across shots",
        author="creator-next",
        published_at=datetime(2025, 10, 22, tzinfo=UTC),
        tags=("角色一致性", "分镜"),
        uses=25_757,
        likes=512,
        collects=2_194,
    )
    newer_qwen = _candidate(
        QWEN_ID,
        name="Qwen image upscale utility",
        description="A general restoration workflow",
        author="creator-qwen",
        published_at=datetime(2026, 1, 22, tzinfo=UTC),
        tags=("upscale",),
        uses=11_977,
        likes=139,
        collects=688,
    )
    details = {
        NEXT_SCENE_ID: _detail(
            next_scene,
            used_models=(
                "Qwen-Image-Edit.safetensors",
                "next-scene_lora_v1-3000.safetensors",
            ),
            primitive_nodes=("KSampler", "SaveImage", "UNETLoader"),
            custom_nodes=("TextEncodeQwenImageEditPlus", "easy promptLine"),
            preview=True,
        ),
        QWEN_ID: _detail(
            newer_qwen,
            used_models=("Qwen-Upscale.safetensors",),
            primitive_nodes=("KSampler",),
        ),
    }
    return (next_scene, newer_qwen), details


def _graph_fixture() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "qwen_graph.json").read_text(encoding="utf-8"))


def test_query_expansion_loads_committed_bilingual_map_and_is_deterministic() -> None:
    concepts = load_query_expansions()
    expanded = expand_query("Qwen NEXT SCENE multi angle character consistency", concepts)

    assert expanded.queries == (
        "Qwen NEXT SCENE multi angle character consistency",
        "Qwen 下一场景 multi angle character consistency",
        "Qwen 分镜 multi angle character consistency",
        "Qwen NEXT SCENE 多角度 character consistency",
        "Qwen NEXT SCENE 多视角 character consistency",
        "Qwen NEXT SCENE multi angle 角色一致性",
        "Qwen NEXT SCENE multi angle 人物一致性",
    )
    assert [record.concept for record in expanded.expansions] == [
        "next_scene",
        "multi_angle",
        "character_consistency",
    ]
    assert expanded.expansions[0].source == "next scene"
    assert expanded.expansions[0].expanded_terms == (
        "next scene",
        "下一场景",
        "分镜",
    )
    assert len(expanded.queries) == len(set(expanded.queries))


@pytest.mark.parametrize(
    "query",
    [
        "triangle camera motion",
        "the next scenery is green",
        "inconsistent characters",
        "multiple angled brackets",
    ],
)
def test_query_expansion_activates_only_token_or_phrase_matches(query: str) -> None:
    expanded = expand_query(query)

    assert expanded.queries == (query,)
    assert expanded.expansions == ()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            "schema_version: 1\nschema_version: 1\nconcepts: []\n",
            "duplicate YAML key",
        ),
        (
            "schema_version: 1\nconcepts: []\nunexpected: true\n",
            "unexpected",
        ),
        (
            "schema_version: 1\nconcepts:\n"
            "  - concept: same\n    terms: [one, two]\n"
            "  - concept: same\n    terms: [three, four]\n",
            "duplicate concept",
        ),
        (
            "schema_version: 1\nconcepts:\n"
            "  - concept: first\n    terms: [shared, one]\n"
            "  - concept: second\n    terms: [SHARED, two]\n",
            "ambiguous term",
        ),
        (
            "schema_version: 1\nconcepts:\n"
            "  - concept: first\n    terms: [one]\n",
            "at least 2",
        ),
        (
            "schema_version: 1\nconcepts: []\n",
            "at least 1",
        ),
    ],
)
def test_query_expansion_schema_fails_closed_with_typed_errors(
    document: str,
    message: str,
) -> None:
    with pytest.raises(QueryExpansionConfigError, match=message):
        parse_query_expansions_yaml(document)


@pytest.mark.parametrize("field", ["concept", "terms"])
def test_query_expansion_rejects_duplicate_nested_keys_with_context(
    field: str,
) -> None:
    duplicate = (
        "    concept: second\n"
        if field == "concept"
        else "    terms: [three, four]\n"
    )
    document = (
        "schema_version: 1\n"
        "concepts:\n"
        "  - concept: first\n"
        "    terms: [one, two]\n"
        f"{duplicate}"
    )

    with pytest.raises(QueryExpansionConfigError) as raised:
        parse_query_expansions_yaml(document)

    message = str(raised.value)
    assert "query-expansions.yaml" in message
    assert "duplicate" in message
    assert f"concepts[0].{field}" in message
    assert "line 5" in message


def test_query_expansion_models_are_deeply_immutable_and_bounded() -> None:
    expanded = expand_query("next scene")

    with pytest.raises(ValidationError):
        expanded.queries = ("changed",)
    with pytest.raises(ValidationError):
        expand_query("x" * 513)


def test_default_weights_and_config_local_partial_override(tmp_path: Path) -> None:
    defaults = load_default_ranking_weights()
    assert defaults == RankingWeights(
        relevance=0.24,
        engagement=0.22,
        output_evidence=0.12,
        recency=0.08,
        creator_track_record=0.08,
        experimentability=0.12,
        novelty=0.14,
        duplicate_penalty=0.35,
    )

    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "ranking:\n"
        "  relevance: 0.4\n"
        "  novelty: 0.05\n",
        encoding="utf-8",
    )
    config = LabConfig.load(tmp_path)
    overridden = load_default_ranking_weights(config.ranking.model_dump(exclude_none=True))

    assert overridden.relevance == 0.4
    assert overridden.novelty == 0.05
    assert overridden.engagement == defaults.engagement


def test_default_ranking_resource_rejects_duplicate_weight_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = (
        "relevance: 0.24\n"
        "engagement: 0.22\n"
        "output_evidence: 0.12\n"
        "recency: 0.08\n"
        "creator_track_record: 0.08\n"
        "experimentability: 0.12\n"
        "novelty: 0.14\n"
        "duplicate_penalty: 0.35\n"
        "relevance: 0.4\n"
    )

    class Resource:
        def joinpath(self, _name: str) -> Resource:
            return self

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return document

    monkeypatch.setattr(ranking_module.resources, "files", lambda _package: Resource())

    with pytest.raises(RankingConfigError) as raised:
        load_default_ranking_weights()

    message = str(raised.value)
    assert "default-ranking.yaml" in message
    assert "duplicate" in message
    assert "relevance" in message
    assert "<root>" in message
    assert "line 9" in message


@pytest.mark.parametrize(
    "ranking_yaml",
    [
        "unknown: 0.1",
        "relevance: true",
        "relevance: .nan",
        "relevance: -0.1",
        "relevance: 1.1",
    ],
)
def test_config_local_rejects_unknown_bool_nonfinite_negative_or_large_weights(
    tmp_path: Path,
    ranking_yaml: str,
) -> None:
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\nranking:\n  "
        + ranking_yaml.replace("\n", "\n  ")
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        LabConfig.load(tmp_path)


def test_weight_loader_rejects_zero_positive_total_with_typed_error() -> None:
    with pytest.raises(RankingConfigError, match="positive weight total"):
        load_default_ranking_weights({field: 0.0 for field in POSITIVE_WEIGHT_FIELDS})


def test_rank_exposes_components_and_golden_next_scene_winner() -> None:
    candidates, details = _golden_candidates()
    ranked = rank_candidates(
        "qwen next scene character consistency",
        candidates,
        details,
        load_default_ranking_weights(),
    )

    winner = ranked[0]
    assert winner.candidate.workflow_id == NEXT_SCENE_ID
    assert set(winner.components.model_dump()) == {
        "relevance",
        "engagement",
        "output_evidence",
        "recency",
        "creator_track_record",
        "experimentability",
        "novelty",
        "penalty",
    }
    assert set(winner.components) == set(winner.components.model_dump())
    assert winner.total_score == winner.score
    assert "25,757 uses" in winner.explanation
    assert "relevance" in winner.explanation
    assert all(
        isfinite(value) and 0.0 <= value <= 1.0
        for value in winner.components.model_dump().values()
    )
    assert isfinite(winner.score) and 0.0 <= winner.score <= 1.0


def test_score_is_computed_exactly_from_weights_and_components() -> None:
    candidates, details = _golden_candidates()
    weights = load_default_ranking_weights()
    winner = rank_candidates("next scene", candidates, details, weights)[0]
    components = winner.components
    positive_total = sum(getattr(weights, field) for field in POSITIVE_WEIGHT_FIELDS)
    expected = sum(
        getattr(weights, field) * getattr(components, field)
        for field in POSITIVE_WEIGHT_FIELDS
    ) / positive_total
    expected = max(
        0.0,
        min(1.0, expected - weights.duplicate_penalty * components.penalty),
    )

    assert isclose(winner.score, expected, rel_tol=0, abs_tol=1e-12)


def test_relevance_uses_bilingual_terms_and_avoids_substring_traps() -> None:
    relevant = _candidate("101", name="电影分镜 workflow", covers=False)
    substring = _candidate("102", name="The next scenery workflow", covers=False)
    ranked = rank_candidates(
        "next scene",
        (substring, relevant),
        {
            "101": _detail(relevant),
            "102": _detail(substring),
        },
        load_default_ranking_weights(),
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["101"].components.relevance == 1.0
    assert by_id["102"].components.relevance == 0.0


def test_relevance_consumes_every_similar_source_query() -> None:
    relevant = _candidate("104", name="workflow", tags=("rare tag",), covers=False)
    unrelated = _candidate("105", name="workflow", covers=False)
    query_plan = ExpandedQuery(
        original="base model",
        source_queries=("base model", "rare tag"),
        queries=("base model", "rare tag"),
        expansions=(),
    )

    ranked = rank_candidates(
        query_plan,
        (unrelated, relevant),
        {"104": _detail(relevant), "105": _detail(unrelated)},
        load_default_ranking_weights(),
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["104"].components.relevance > by_id["105"].components.relevance


def test_hyphenated_query_phrase_is_one_conceptual_relevance_unit() -> None:
    relevant = _candidate("103", name="电影分镜 workflow", covers=False)
    ranked = rank_candidates(
        "next-scene",
        (relevant,),
        {"103": _detail(relevant)},
        load_default_ranking_weights(),
    )

    assert ranked[0].components.relevance == 1.0


def test_bilingual_tokenization_splits_latin_cjk_transitions() -> None:
    candidate = _candidate("106", name="Qwen 视频 workflow", covers=False)

    ranked = rank_candidates(
        "Qwen视频",
        (candidate,),
        {"106": _detail(candidate)},
        load_default_ranking_weights(),
    )

    assert ranked[0].components.relevance == 1.0
    assert term_match("Qwen视频", "Qwen 视频") is not None
    assert term_match("Qwencoder视频", "Qwen 视频") is None


def test_engagement_uses_documented_log1p_metric_mix() -> None:
    high = _candidate("201", name="alpha", uses=99, likes=24, collects=8)
    low = _candidate("202", name="beta", uses=9, likes=3, collects=0)
    ranked = rank_candidates(
        "unmatched",
        (high, low),
        {"201": _detail(high), "202": _detail(low)},
        load_default_ranking_weights(),
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}
    expected_low = (
        0.6 * log1p(9) / log1p(99)
        + 0.2 * log1p(3) / log1p(24)
        + 0.2 * 0.0
    )

    assert by_id["201"].components.engagement == 1.0
    assert isclose(
        by_id["202"].components.engagement,
        expected_low,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_engagement_normalization_handles_typical_and_multiple_huge_counts() -> None:
    counts = (0, 25_757, 10**1_000, 10**2_000)
    candidates = tuple(
        _candidate(
            str(210 + index),
            name="same",
            author=f"creator-{index}",
            uses=count,
            likes=count,
            collects=count,
            covers=False,
        )
        for index, count in enumerate(counts)
    )
    details = {item.workflow_id: _detail(item) for item in candidates}

    forward = rank_candidates(
        "unmatched",
        candidates,
        details,
        load_default_ranking_weights(),
    )
    reverse = rank_candidates(
        "unmatched",
        tuple(reversed(candidates)),
        details,
        load_default_ranking_weights(),
    )
    by_id = {item.candidate.workflow_id: item for item in forward}
    normalized = [
        by_id[str(210 + index)].components.engagement
        for index in range(len(counts))
    ]

    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert normalized == sorted(normalized)
    assert len(set(normalized)) == len(normalized)
    assert all(isfinite(value) and 0.0 <= value <= 1.0 for value in normalized)
    assert [item.candidate.workflow_id for item in forward] == [
        item.candidate.workflow_id for item in reverse
    ]


def test_all_equal_huge_engagement_is_finite_and_ties_by_workflow_id() -> None:
    huge = 10**1_000
    high_id = _candidate(
        "222",
        name="same",
        uses=huge,
        likes=huge,
        collects=huge,
        covers=False,
    )
    low_id = _candidate(
        "221",
        name="same",
        uses=huge,
        likes=huge,
        collects=huge,
        covers=False,
    )
    details = {item.workflow_id: _detail(item) for item in (high_id, low_id)}

    forward = rank_candidates(
        "unmatched",
        (high_id, low_id),
        details,
        load_default_ranking_weights(),
    )
    reverse = rank_candidates(
        "unmatched",
        (low_id, high_id),
        details,
        load_default_ranking_weights(),
    )

    assert [item.candidate.workflow_id for item in forward] == ["221", "222"]
    assert [item.candidate.workflow_id for item in reverse] == ["221", "222"]
    assert [item.components.engagement for item in forward] == [1.0, 1.0]
    assert all(isfinite(item.score) and 0.0 <= item.score <= 1.0 for item in forward)


def test_recency_and_creator_track_record_depend_only_on_active_set() -> None:
    old_popular = _candidate(
        "301",
        name="same",
        author="shared",
        published_at=datetime(2024, 1, 1, tzinfo=UTC),
        uses=1_000,
    )
    new_shared = _candidate(
        "302",
        name="same",
        author="shared",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        uses=0,
    )
    middle_solo = _candidate(
        "303",
        name="same",
        author="solo",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
        uses=10,
    )
    candidates = (old_popular, new_shared, middle_solo)
    details = {item.workflow_id: _detail(item) for item in candidates}
    first = rank_candidates("same", candidates, details, load_default_ranking_weights())
    second = rank_candidates(
        "same", tuple(reversed(candidates)), details, load_default_ranking_weights()
    )
    by_id = {item.candidate.workflow_id: item for item in first}

    assert [item.candidate.workflow_id for item in first] == [
        item.candidate.workflow_id for item in second
    ]
    assert by_id["301"].components.recency == 0.0
    assert by_id["302"].components.recency == 1.0
    assert 0.0 < by_id["303"].components.recency < 1.0
    assert (
        by_id["302"].components.creator_track_record
        > by_id["303"].components.creator_track_record
    )


def test_output_evidence_uses_explicit_class_tokens_and_normalized_role() -> None:
    node_classes = {
        "351": "SaveImage",
        "352": "Image Save",
        "353": "Image Saver Simple",
        "354": "UnsavedCondition",
        "355": "UnsavedCondition",
    }
    candidates = tuple(
        _candidate(workflow_id, name="same", covers=False)
        for workflow_id in node_classes
    )
    details = {
        item.workflow_id: _detail(
            item,
            primitive_nodes=(node_classes[item.workflow_id],),
        )
        for item in candidates
    }

    ranked = rank_candidates(
        "unmatched",
        candidates,
        details,
        load_default_ranking_weights(),
        graph_evidence={"355": GraphEvidence(roles=("output",))},
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["351"].components.output_evidence == 0.2
    assert by_id["352"].components.output_evidence == 0.2
    assert by_id["353"].components.output_evidence == 0.2
    assert by_id["354"].components.output_evidence == 0.0
    assert by_id["355"].components.output_evidence == 0.2


def test_coarse_duplicate_uses_ordered_models_and_sorted_node_inventory() -> None:
    first = _candidate("401", name="same", uses=100)
    second = _candidate("402", name="same", uses=10)
    details = {
        "401": _detail(
            first,
            used_models=("base", "lora"),
            primitive_nodes=("B", "A"),
            custom_nodes=("Z",),
        ),
        "402": _detail(
            second,
            used_models=("base", "lora"),
            primitive_nodes=("A", "B"),
            custom_nodes=("Z",),
        ),
    }
    ranked = rank_candidates(
        "same", (second, first), details, load_default_ranking_weights()
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["401"].components.penalty == 0.0
    assert by_id["402"].components.penalty == 1.0
    assert by_id["402"].duplicate_group.startswith("coarse:")
    assert "duplicate penalty" in by_id["402"].explanation


def test_intrinsic_representative_prefers_exact_relevance_before_engagement() -> None:
    relevant = _candidate("403", name="Next Scene workflow", uses=1, covers=False)
    popular = _candidate("404", name="Unrelated Qwen workflow", uses=100, covers=False)
    details = {
        candidate.workflow_id: _detail(
            candidate,
            used_models=("shared-model",),
            primitive_nodes=("KSampler",),
        )
        for candidate in (relevant, popular)
    }
    group_keys = {"403": "coarse:shared", "404": "coarse:shared"}
    plan = expand_query("next scene")
    weights = load_default_ranking_weights()

    selected = select_group_representatives(
        plan,
        tuple(reversed(tuple(details.values()))),
        weights,
        group_keys,
    )
    ranked = rank_candidates(
        plan,
        (popular, relevant),
        details,
        weights,
    )
    reversed_ranked = rank_candidates(
        plan,
        (relevant, popular),
        dict(reversed(tuple(details.items()))),
        weights,
    )

    assert selected == {"coarse:shared": "403"}
    assert {item.candidate.workflow_id: item.components.penalty for item in ranked} == {
        "403": 0.0,
        "404": 1.0,
    }
    assert [item.candidate.workflow_id for item in ranked] == [
        item.candidate.workflow_id for item in reversed_ranked
    ]


def test_graph_enrichment_uses_ranking_intrinsic_representative(tmp_path: Path) -> None:
    relevant = _candidate("405", name="Next Scene workflow", uses=1, covers=False)
    popular = _candidate("406", name="Unrelated Qwen workflow", uses=100, covers=False)
    details = {
        candidate.workflow_id: _detail(
            candidate,
            used_models=("shared-model",),
            primitive_nodes=("KSampler",),
        )
        for candidate in (relevant, popular)
    }
    executor = FakeDiscoveryExecutor((popular, relevant), details)

    result = _service(tmp_path, executor).search(
        "next scene",
        sort="HOTTEST",
        include_graphs=True,
    )

    assert [item.workflow_id for item in executor.fetch_calls] == ["405"]
    by_id = {item.candidate.workflow_id: item for item in result.results}
    assert by_id["405"].components.penalty == 0.0
    assert by_id["406"].components.penalty == 1.0


def test_creator_track_record_counts_each_duplicate_group_once() -> None:
    representative = _candidate(
        "4100",
        name="duplicate",
        author="creator-duplicate",
        uses=10,
        covers=False,
    )
    competitor = _candidate(
        "5100",
        name="distinct",
        author="creator-distinct",
        uses=20,
        covers=False,
    )
    baseline_details = {
        "4100": _detail(
            representative,
            used_models=("same-model",),
            primitive_nodes=("KSampler",),
        ),
        "5100": _detail(
            competitor,
            used_models=("different-model",),
            primitive_nodes=("KSampler",),
        ),
    }
    baseline = rank_candidates(
        "unmatched",
        (representative, competitor),
        baseline_details,
        load_default_ranking_weights(),
    )
    baseline_by_id = {item.candidate.workflow_id: item for item in baseline}
    duplicates = tuple(
        _candidate(
            str(4100 + index),
            name="duplicate",
            author="creator-duplicate",
            uses=10,
            covers=False,
        )
        for index in range(99)
    )
    candidates = (*duplicates, competitor)
    details = {
        item.workflow_id: _detail(
            item,
            used_models=("same-model",),
            primitive_nodes=("KSampler",),
        )
        for item in duplicates
    }
    details["5100"] = baseline_details["5100"]

    ranked = rank_candidates(
        "unmatched",
        tuple(reversed(candidates)),
        details,
        load_default_ranking_weights(),
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}
    duplicate_scores = {
        by_id[item.workflow_id].components.creator_track_record
        for item in duplicates
    }

    assert duplicate_scores == {
        baseline_by_id["4100"].components.creator_track_record
    }
    assert by_id["5100"].components.creator_track_record == (
        baseline_by_id["5100"].components.creator_track_record
    )
    assert by_id["4100"].components.creator_track_record < (
        by_id["5100"].components.creator_track_record
    )
    assert by_id["4100"].components.penalty == 0.0


@pytest.mark.parametrize(
    ("evidence", "expected_penalized"),
    [
        (
            {
                "501": GraphEvidence(source_hash="a" * 64, capability_hash="1" * 64),
                "502": GraphEvidence(source_hash="a" * 64, capability_hash="2" * 64),
            },
            "502",
        ),
        (
            {
                "501": GraphEvidence(source_hash="a" * 64, capability_hash="1" * 64),
                "502": GraphEvidence(source_hash="b" * 64, capability_hash="1" * 64),
            },
            "502",
        ),
    ],
)
def test_exact_source_or_capability_hash_supersedes_coarse_grouping(
    evidence: dict[str, GraphEvidence],
    expected_penalized: str,
) -> None:
    first = _candidate("501", name="same", uses=100)
    second = _candidate("502", name="same", uses=10)
    details = {
        item.workflow_id: _detail(item, used_models=("same",), primitive_nodes=("K",))
        for item in (first, second)
    }
    ranked = rank_candidates(
        "same",
        (second, first),
        details,
        load_default_ranking_weights(),
        graph_evidence=evidence,
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["501"].components.penalty == 0.0
    assert by_id[expected_penalized].components.penalty == 1.0
    assert not by_id[expected_penalized].duplicate_group.startswith("coarse:")


def test_different_enriched_hashes_do_not_fall_back_to_false_coarse_duplicate() -> None:
    first = _candidate("601", name="same", uses=1)
    second = _candidate("602", name="same", uses=1)
    details = {
        item.workflow_id: _detail(item, used_models=("same",), primitive_nodes=("K",))
        for item in (first, second)
    }
    evidence = {
        "601": GraphEvidence(source_hash="a" * 64, capability_hash="1" * 64),
        "602": GraphEvidence(source_hash="b" * 64, capability_hash="2" * 64),
    }
    ranked = rank_candidates(
        "same",
        (first, second),
        details,
        load_default_ranking_weights(),
        graph_evidence=evidence,
    )

    assert [item.components.penalty for item in ranked] == [0.0, 0.0]


def test_overlapping_source_and_capability_duplicates_form_one_exact_group() -> None:
    candidates = tuple(
        _candidate(workflow_id, name="same", uses=uses)
        for workflow_id, uses in (("611", 100), ("612", 50), ("613", 10))
    )
    details = {
        item.workflow_id: _detail(item, used_models=("same",), primitive_nodes=("K",))
        for item in candidates
    }
    evidence = {
        "611": GraphEvidence(source_hash="a" * 64, capability_hash="1" * 64),
        "612": GraphEvidence(source_hash="a" * 64, capability_hash="2" * 64),
        "613": GraphEvidence(source_hash="b" * 64, capability_hash="2" * 64),
    }

    ranked = rank_candidates(
        "same",
        tuple(reversed(candidates)),
        details,
        load_default_ranking_weights(),
        graph_evidence=evidence,
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["611"].components.penalty == 0.0
    assert by_id["612"].components.penalty == 1.0
    assert by_id["613"].components.penalty == 1.0
    assert len({by_id[item].duplicate_group for item in by_id}) == 1


def test_graph_canonicalization_error_is_visible_and_fail_closed_for_duplicates() -> None:
    first = _candidate("701", name="same")
    second = _candidate("702", name="same")
    details = {
        item.workflow_id: _detail(item, used_models=("same",), primitive_nodes=("K",))
        for item in (first, second)
    }
    evidence = {
        "701": GraphEvidence(canonicalization_error="graph is ambiguous"),
        "702": GraphEvidence(source_hash="b" * 64, capability_hash="2" * 64),
    }
    ranked = rank_candidates(
        "same",
        (first, second),
        details,
        load_default_ranking_weights(),
        graph_evidence=evidence,
    )
    by_id = {item.candidate.workflow_id: item for item in ranked}

    assert by_id["701"].canonicalization_error == "graph is ambiguous"
    assert by_id["701"].components.penalty == 0.0
    assert by_id["702"].components.penalty == 0.0


def test_tie_break_is_provider_order_independent_and_uses_workflow_id() -> None:
    high_id = _candidate("902", name="same", uses=5, covers=False)
    low_id = _candidate("901", name="same", uses=5, covers=False)
    details = {item.workflow_id: _detail(item) for item in (high_id, low_id)}
    forward = rank_candidates(
        "same", (high_id, low_id), details, load_default_ranking_weights()
    )
    reverse = rank_candidates(
        "same", (low_id, high_id), details, load_default_ranking_weights()
    )

    assert [item.candidate.workflow_id for item in forward] == ["901", "902"]
    assert [item.candidate.workflow_id for item in reverse] == ["901", "902"]


class FakeDiscoveryExecutor:
    name = "runninghub"

    def __init__(
        self,
        candidates: tuple[WorkflowCandidate, ...],
        details: Mapping[str, WorkflowDetail],
        *,
        graph: Mapping[str, Any] | None = None,
        search_errors: Mapping[str, Exception] | None = None,
        detail_errors: Mapping[str, Exception] | None = None,
        fetch_errors: Mapping[str, Exception] | None = None,
    ) -> None:
        self.candidates = candidates
        self.details = dict(details)
        self.graph = dict(graph or _graph_fixture())
        self.search_errors = dict(search_errors or {})
        self.detail_errors = dict(detail_errors or {})
        self.fetch_errors = dict(fetch_errors or {})
        self.search_calls: list[SearchRequest] = []
        self.detail_calls: list[tuple[str, str | None]] = []
        self.fetch_calls: list[WorkflowRef] = []

    def search(self, request: SearchRequest) -> SearchPage:
        self.search_calls.append(request)
        if error := self.search_errors.get(request.sort):
            raise error
        records = tuple(reversed(self.candidates)) if request.sort == "NEWEST" else self.candidates
        return SearchPage(
            records=records,
            total=len(records),
            page=request.page,
            size=request.size,
            has_next=False,
        )

    def detail(
        self,
        workflow_id: str,
        *,
        canonical_url: str | None = None,
    ) -> WorkflowDetail:
        self.detail_calls.append((workflow_id, canonical_url))
        if error := self.detail_errors.get(workflow_id):
            raise error
        try:
            detail = self.details[workflow_id]
        except KeyError:
            raise DiscoveryNotFoundError(f"workflow not found: {workflow_id}") from None
        if canonical_url is None:
            return detail
        return detail.model_copy(
            update={
                "candidate": detail.candidate.model_copy(
                    update={"canonical_url": canonical_url}
                )
            }
        )

    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        self.fetch_calls.append(ref)
        if error := self.fetch_errors.get(ref.workflow_id):
            raise error
        detail = self.detail(ref.workflow_id, canonical_url=ref.canonical_url)
        return FetchedWorkflow(ref=ref, detail=detail, graph=self.graph)

    def inspect_capabilities(self, workflow: FetchedWorkflow) -> CapabilityReport:
        return CapabilityReport(
            required_nodes=tuple(
                sorted(
                    set(
                        workflow.detail.primitive_nodes
                        + workflow.detail.custom_nodes
                    )
                )
            ),
            model_names=tuple(
                name
                for name in workflow.detail.used_models
                if "lora" not in name.casefold()
            ),
            lora_names=tuple(
                name
                for name in workflow.detail.used_models
                if "lora" in name.casefold()
            ),
        )


def _service(
    tmp_path: Path,
    executor: FakeDiscoveryExecutor,
) -> DiscoveryService:
    return DiscoveryService(
        executor=executor,
        config=LabConfig.model_validate(
            {"experiments_root": tmp_path / "experiments", "vaults": {}},
            context={"base_dir": tmp_path},
        ),
        store=ExperimentStore(tmp_path / "experiments"),
        clock=lambda: datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC),
        id_factory=lambda: "fixedid",
    )


def _read_attempts(path: Path) -> tuple[dict[str, Any], ...]:
    store = ExperimentStore(path.parent.parent)
    return store.read_jsonl(path.parent.name, path.name)


def test_service_implicit_sort_fanout_and_workflow_dedup_precede_public_detail(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates + (candidates[0],), details)
    service = _service(tmp_path, executor)

    result = service.search("qwen", page=2, size=7)

    assert [(call.sort, call.page, call.size) for call in executor.search_calls] == [
        ("", 2, 7),
        ("REPUTATION", 2, 7),
        ("HOTTEST", 2, 7),
        ("NEWEST", 2, 7),
    ]
    assert executor.detail_calls == [(NEXT_SCENE_ID, None), (QWEN_ID, None)]
    assert executor.fetch_calls == []
    assert len(result.results) == 2


def test_discovery_repeated_factory_id_uses_distinct_roots_without_overwrite(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates, details)
    service = _service(tmp_path, executor)

    first = service.search("first query", sort="HOTTEST")
    first_snapshot = next(
        Path(item.path)
        for item in first.artifacts
        if Path(item.path).name == "search-results.json"
    )
    first_bytes = first_snapshot.read_bytes()
    second = service.search("second query", sort="HOTTEST")

    assert first.experiment_id == "search-20260203t040506z-fixedid"
    assert second.experiment_id == "search-20260203t040506z-fixedid-2"
    assert first_snapshot.read_bytes() == first_bytes
    assert json.loads(first_snapshot.read_text(encoding="utf-8"))["query"] == (
        "first query"
    )


def test_service_explicit_sort_and_expansion_calls_are_unique(tmp_path: Path) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates, details)
    service = _service(tmp_path, executor)

    result = service.search("qwen next scene", sort="REPUTATION")

    assert [(call.query, call.sort) for call in executor.search_calls] == [
        ("qwen next scene", "REPUTATION"),
        ("qwen 下一场景", "REPUTATION"),
        ("qwen 分镜", "REPUTATION"),
    ]
    assert len(executor.search_calls) == len(set(executor.search_calls))
    assert result.query_plan.queries[0] == "qwen next scene"
    assert result.query_plan.expansions[0].concept == "next_scene"


def test_discovery_one_sort_failure_returns_partial_with_durable_attempts(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(
        candidates,
        details,
        search_errors={
            "HOTTEST": RuntimeError(
                "provider unavailable apiKey=search-secret auth_token=token-secret"
            )
        },
    )
    expected_snapshot = (
        tmp_path
        / "experiments"
        / "search-20260203t040506z-fixedid"
        / "search-results.json"
    )
    original_search = executor.search
    observed_checkpoints: list[dict[str, Any]] = []

    def observe_search(request: SearchRequest) -> SearchPage:
        assert expected_snapshot.is_file()
        observed_checkpoints.append(
            json.loads(expected_snapshot.read_text(encoding="utf-8"))
        )
        return original_search(request)

    executor.search = observe_search  # type: ignore[method-assign]

    result = _service(tmp_path, executor).search("qwen")

    assert observed_checkpoints[0]["attempt_count"] == 0
    assert result.incomplete is True
    assert len(result.results) == 2
    assert len(result.diagnostics) == 1
    assert result.diagnostic_count == 1
    assert result.diagnostics_artifact.kind == "attempts"
    failure = result.diagnostics[0]
    assert (failure.operation, failure.status, failure.sort) == (
        "search",
        "failed",
        "HOTTEST",
    )
    assert failure.error_kind == "RuntimeError"
    assert "search-secret" not in failure.message
    assert "token-secret" not in failure.message
    snapshot = json.loads(expected_snapshot.read_text(encoding="utf-8"))
    assert snapshot["status"] == "partial"
    assert snapshot["incomplete"] is True
    assert snapshot["attempt_count"] == 6
    assert snapshot["diagnostic_count"] == 1
    assert "attempts" not in snapshot
    attempts_path = Path(result.diagnostics_artifact.path)
    attempts = _read_attempts(attempts_path)
    assert [item["order"] for item in attempts] == list(range(1, 7))
    assert attempts[0]["result_ids"] == [NEXT_SCENE_ID, QWEN_ID]


def test_discovery_detail_failure_skips_candidate_with_diagnostic(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(
        candidates,
        details,
        detail_errors={
            QWEN_ID: RuntimeError("password=detail-secret provider detail failed")
        },
    )

    result = _service(tmp_path, executor).search(
        "qwen",
        sort="HOTTEST",
    )

    assert result.incomplete is True
    assert [item.candidate.workflow_id for item in result.results] == [NEXT_SCENE_ID]
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert (diagnostic.operation, diagnostic.workflow_id) == ("detail", QWEN_ID)
    assert "detail-secret" not in diagnostic.message
    snapshot = next(
        Path(item.path)
        for item in result.artifacts
        if Path(item.path).name == "search-results.json"
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["incomplete"] is True
    assert len(payload["results"]) == 1


def test_discovery_graph_failure_retains_coarse_candidates_without_false_duplicate(
    tmp_path: Path,
) -> None:
    first = _candidate("1095", name="first", uses=10, covers=False)
    second = _candidate("1096", name="second", uses=5, covers=False)
    details = {
        item.workflow_id: _detail(
            item,
            used_models=("shared-model",),
            primitive_nodes=("KSampler", "SaveImage"),
        )
        for item in (first, second)
    }
    executor = FakeDiscoveryExecutor(
        (second, first),
        details,
        fetch_errors={
            "1095": RuntimeError(
                "graph failed auth_token=graph-secret password=password-secret"
            )
        },
    )

    result = _service(tmp_path, executor).search(
        "workflow",
        sort="HOTTEST",
        include_graphs=True,
    )

    assert result.incomplete is True
    assert {item.candidate.workflow_id for item in result.results} == {"1095", "1096"}
    by_id = {item.candidate.workflow_id: item for item in result.results}
    assert by_id["1095"].canonicalization_error
    assert "graph-secret" not in by_id["1095"].canonicalization_error
    assert by_id["1095"].components.penalty == 0.0
    assert by_id["1096"].components.penalty == 0.0
    assert by_id["1095"].duplicate_group is None
    assert by_id["1096"].duplicate_group is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].operation == "graph"


def test_discovery_all_search_failures_raise_incomplete_with_artifact(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(
        candidates,
        details,
        search_errors={
            sort: RuntimeError(f"{sort or 'default'} failed secret=provider-secret")
            for sort in ("", "REPUTATION", "HOTTEST", "NEWEST")
        },
    )

    with pytest.raises(service_module.DiscoveryIncompleteError) as raised:
        _service(tmp_path, executor).search("qwen")

    error = raised.value
    assert error.experiment_id == "search-20260203t040506z-fixedid"
    assert {item.kind for item in error.artifacts} == {"search", "attempts"}
    snapshot = next(Path(item.path) for item in error.artifacts if item.kind == "search")
    attempts_path = next(
        Path(item.path) for item in error.artifacts if item.kind == "attempts"
    )
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["incomplete"] is True
    assert payload["results"] == []
    assert payload["attempt_count"] == 4
    assert payload["diagnostic_count"] == 4
    attempts = _read_attempts(attempts_path)
    assert {item["status"] for item in attempts} == {"failed"}
    assert "provider-secret" not in snapshot.read_text(encoding="utf-8")
    assert executor.detail_calls == []


def test_discovery_attempt_journal_is_linear_for_400_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = tuple(
        _candidate(str(20_000 + index), name=f"candidate {index}", covers=False)
        for index in range(400)
    )
    details = {item.workflow_id: _detail(item) for item in candidates}
    executor = FakeDiscoveryExecutor(tuple(reversed(candidates)), details)
    service = _service(tmp_path, executor)
    snapshot_writes = 0
    original_write_json = service.store.write_json

    def count_snapshot_writes(experiment_id, relative_path, value):
        nonlocal snapshot_writes
        if Path(relative_path) == Path("search-results.json"):
            snapshot_writes += 1
        return original_write_json(experiment_id, relative_path, value)

    monkeypatch.setattr(service.store, "write_json", count_snapshot_writes)
    started = perf_counter()

    result = service.search("candidate", sort="HOTTEST", size=100)

    elapsed = perf_counter() - started
    attempts_path = Path(result.diagnostics_artifact.path)
    attempts = _read_attempts(attempts_path)
    assert elapsed < 30.0
    assert snapshot_writes <= 5
    assert len(attempts) == 401
    assert attempts[0]["operation"] == "search"
    assert len(attempts[0]["result_ids"]) == 400
    assert all(item["artifact_paths"] for item in attempts[1:])
    assert len(result.results) == 400


def test_thousands_of_detail_failures_have_authoritative_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        _candidate(str(30_000 + index), name=f"failure {index}", covers=False)
        for index in range(2_000)
    )
    details = {item.workflow_id: _detail(item) for item in candidates}
    errors = {
        item.workflow_id: RuntimeError(
            f"detail {item.workflow_id} failed auth_token=provider-secret"
        )
        for item in candidates
    }
    executor = FakeDiscoveryExecutor(candidates, details, detail_errors=errors)

    result = _service(tmp_path, executor).search(
        "failure",
        sort="HOTTEST",
        size=100,
    )

    assert result.incomplete is True
    assert result.results == ()
    assert result.diagnostic_count == 2_000
    assert 0 < len(result.diagnostics) <= 16
    attempts_path = Path(result.diagnostics_artifact.path)
    attempts = _read_attempts(attempts_path)
    assert len(attempts) == 2_001
    assert sum(item["status"] == "failed" for item in attempts) == 2_000
    snapshot = next(Path(item.path) for item in result.artifacts if item.kind == "search")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["diagnostic_count"] == 2_000
    assert payload["attempt_count"] == 2_001
    assert len(payload["diagnostics_sample"]) <= 16
    assert payload["diagnostics_artifact"] == str(attempts_path)
    assert "provider-secret" not in attempts_path.read_text(encoding="utf-8")


def test_service_persists_sanitized_search_and_detail_without_graph(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        "1001",
        name="next scene",
        provider_metadata={
            "preview": "https://assets.example/a.png?X-Amz-Signature=secret-signature"
        },
    )
    detail = _detail(
        candidate,
        used_models=(
            "https://models.example/base.safetensors?sig=secret-model&sp=r",
        ),
    )
    executor = FakeDiscoveryExecutor((candidate,), {"1001": detail})
    service = _service(tmp_path, executor)

    result = service.search("next scene", sort="HOTTEST")

    artifact_paths = {Path(artifact.path) for artifact in result.artifacts}
    assert any(path.name == "search-results.json" for path in artifact_paths)
    assert any(path.name == "detail.json" for path in artifact_paths)
    assert not any(path.name == "graph.json" for path in artifact_paths)
    persisted = "\n".join(path.read_text(encoding="utf-8") for path in artifact_paths)
    assert "secret-signature" not in persisted
    assert "secret-model" not in persisted
    assert "X-Amz-Signature" not in persisted
    assert executor.fetch_calls == []


def test_include_graphs_enriches_once_and_persists_roles_and_hashes(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor((candidates[0],), details)
    service = _service(tmp_path, executor)

    result = service.search(
        "next scene", sort="REPUTATION", include_graphs=True
    )

    assert len(executor.fetch_calls) == 1
    assert result.results[0].source_hash is not None
    assert result.results[0].capability_hash is not None
    names = {Path(artifact.path).name for artifact in result.artifacts}
    assert {"graph.json", "normalized.json", "hashes.json"} <= names


def test_include_graphs_enriches_one_representative_per_coarse_group(
    tmp_path: Path,
) -> None:
    duplicate_a = _candidate("1071", name="duplicate a", uses=20, covers=False)
    duplicate_b = _candidate("1072", name="duplicate b", uses=10, covers=False)
    distinct = _candidate("1073", name="distinct", uses=5, covers=False)
    candidates = (duplicate_b, distinct, duplicate_a)
    details = {
        "1071": _detail(
            duplicate_a,
            used_models=("shared-model",),
            primitive_nodes=("KSampler", "SaveImage"),
        ),
        "1072": _detail(
            duplicate_b,
            used_models=("shared-model",),
            primitive_nodes=("SaveImage", "KSampler"),
        ),
        "1073": _detail(
            distinct,
            used_models=("distinct-model",),
            primitive_nodes=("KSampler",),
        ),
    }
    executor = FakeDiscoveryExecutor(candidates, details)

    result = _service(tmp_path, executor).search(
        "workflow",
        sort="HOTTEST",
        size=7,
        include_graphs=True,
    )

    assert [item.workflow_id for item in executor.fetch_calls] == ["1071", "1073"]
    assert all(call.size == 7 for call in executor.search_calls)
    by_id = {item.candidate.workflow_id: item for item in result.results}
    assert by_id["1071"].source_hash is not None
    assert by_id["1072"].source_hash is None
    assert by_id["1071"].duplicate_group == by_id["1072"].duplicate_group
    assert by_id["1071"].components.penalty == 0.0
    assert by_id["1072"].components.penalty == 1.0


def test_inspect_accepts_strict_url_fetches_once_and_uses_canonical_url(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor((candidates[0],), details)
    service = _service(tmp_path, executor)

    result = service.inspect(
        f"https://www.runninghub.ai/zh-cn/workflow/{NEXT_SCENE_ID}/?from=search"
    )

    assert len(executor.fetch_calls) == 1
    assert len(executor.detail_calls) == 1
    assert result.detail.candidate.canonical_url == (
        f"https://www.runninghub.ai/workflow/{NEXT_SCENE_ID}"
    )
    assert result.source_hash is not None
    assert result.capability_hash is not None
    assert {Path(artifact.path).name for artifact in result.artifacts} >= {
        "detail.json",
        "graph.json",
        "normalized.json",
        "hashes.json",
    }


def test_inspect_persists_secret_safe_graph_with_reproducible_hashes(
    tmp_path: Path,
) -> None:
    candidate = _candidate("1081", name="safe graph", covers=False)
    detail = _detail(candidate, primitive_nodes=("SaveImage",))
    graph = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {
                "endpoint": (
                    "https://alice:authority-secret@[2001:db8::1]:8443/base"
                    "?revision=1&format=fp16&X-Amz-Credential=aws-secret&"
                    "X-Amz-Signature=aws-signature#section=weights;"
                    "AcCeSsToKeN=fragment-secret"
                ),
                "documentation": (
                    "https://models.example/docs#section=weights;revision=1"
                ),
                "nested_endpoint": (
                    "https://models.example/base?revision=1"
                    "#section%253Dweights%2526access_token%253Dnested-secret"
                ),
                "password": "password-secret",
                "auth_token": "auth-secret",
                "apiKey": "api-secret",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 0],
                "text": "A secret garden, preserve this ordinary prompt text",
            },
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0]},
        },
    }
    executor = FakeDiscoveryExecutor((candidate,), {"1081": detail}, graph=graph)

    result = _service(tmp_path, executor).inspect(
        "https://www.runninghub.ai/post/1081"
    )

    paths = {Path(item.path).name: Path(item.path) for item in result.artifacts}
    persisted_graph = json.loads(paths["graph.json"].read_text(encoding="utf-8"))
    hashes = json.loads(paths["hashes.json"].read_text(encoding="utf-8"))
    serialized_artifacts = "\n".join(
        path.read_text(encoding="utf-8") for path in paths.values()
    )
    assert source_graph_hash(persisted_graph) == hashes["source_hash"]
    assert hashes["source_hash"] == result.source_hash
    assert "revision=1&format=fp16" in paths["graph.json"].read_text(
        encoding="utf-8"
    )
    assert (
        persisted_graph["1"]["inputs"]["endpoint"]
        == "https://[2001:db8::1]:8443/base?revision=1&format=fp16"
    )
    assert persisted_graph["1"]["inputs"]["documentation"] == (
        "https://models.example/docs#section=weights;revision=1"
    )
    assert persisted_graph["1"]["inputs"]["nested_endpoint"] == (
        "https://models.example/base?revision=1"
    )
    assert "section=weights;revision=1" in paths["normalized.json"].read_text(
        encoding="utf-8"
    )
    assert "A secret garden, preserve this ordinary prompt text" in serialized_artifacts
    for secret in (
        "password-secret",
        "auth-secret",
        "api-secret",
        "aws-secret",
        "aws-signature",
        "authority-secret",
        "fragment-secret",
        "nested-secret",
        "alice",
        "X-Amz-Credential",
        "X-Amz-Signature",
    ):
        assert secret not in serialized_artifacts


def test_inspect_rejects_non_runninghub_source_before_executor_call(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates, details)
    service = _service(tmp_path, executor)

    with pytest.raises(UnsupportedSourceError):
        service.inspect("https://example.com/workflow/123")

    assert executor.fetch_calls == []


def test_similar_inspects_source_and_records_bounded_derivations(
    tmp_path: Path,
) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates, details)
    service = _service(tmp_path, executor)

    result = service.similar(
        f"https://www.runninghub.ai/post/{NEXT_SCENE_ID}",
        sort="HOTTEST",
    )

    assert executor.fetch_calls == [
        WorkflowRef(
            executor="runninghub",
            workflow_id=NEXT_SCENE_ID,
            canonical_url=f"https://www.runninghub.ai/post/{NEXT_SCENE_ID}",
        )
    ]
    assert result.derivations
    assert {item.source for item in result.derivations} >= {
        "model",
        "lora",
        "tag",
        "prompt_grammar",
        "capability_role",
    }
    assert len(result.derivations) <= 16
    prompt_grammar = next(
        item.value for item in result.derivations if item.source == "prompt_grammar"
    )
    assert "lines" in prompt_grammar
    assert "commas" in prompt_grammar
    assert "placeholders" in prompt_grammar
    assert len(executor.search_calls) == len(set(executor.search_calls))
    assert all(call.sort == "HOTTEST" for call in executor.search_calls)
    snapshot = next(
        Path(artifact.path)
        for artifact in result.artifacts
        if Path(artifact.path).name == "search-results.json"
    )
    assert json.loads(snapshot.read_text(encoding="utf-8"))["derivations"]


def test_similar_many_evidence_items_stays_category_bounded(
    tmp_path: Path,
) -> None:
    source = _candidate(
        "1085",
        name="source",
        tags=(
            "next scene",
            "multi angle",
            *(f"tag{index}" for index in range(40)),
        ),
        covers=False,
    )
    models = tuple(f"model{index}.safetensors" for index in range(10))
    loras = tuple(f"style{index}_lora.safetensors" for index in range(8))
    detail = _detail(source, used_models=(*models, *loras))
    executor = FakeDiscoveryExecutor((source,), {"1085": detail})

    result = _service(tmp_path, executor).similar(
        "https://www.runninghub.ai/post/1085",
    )

    assert [item.source for item in result.derivations] == [
        "model",
        "lora",
        "tag",
        "prompt_grammar",
        "capability_role",
        "summary",
    ]
    assert len(executor.search_calls) == 24
    assert len(result.query_plan.queries) == 6
    assert len({(call.query.casefold(), call.sort) for call in executor.search_calls}) == 24
    assert {item.concept for item in result.query_plan.expansions} >= {
        "next_scene",
        "multi_angle",
    }
    assert all(call.query in result.query_plan.source_queries for call in executor.search_calls)
    assert all(len(call.query) <= 512 for call in executor.search_calls)
    assert len(executor.detail_calls) <= len({item.workflow_id for item in executor.candidates})


def test_similar_accepts_provider_safe_140_character_model_name(tmp_path: Path) -> None:
    source = _candidate("1086", name="source", covers=False)
    detail = _detail(source, used_models=(f"{'x' * 140}.safetensors",))
    executor = FakeDiscoveryExecutor((source,), {"1086": detail})

    result = _service(tmp_path, executor).similar(
        "https://www.runninghub.ai/post/1086",
        sort="HOTTEST",
    )

    model_query = next(item for item in result.derivations if item.source == "model")
    assert len(model_query.value) == 140
    assert model_query.omitted_count == 0
    assert model_query.value in {item.query for item in executor.search_calls}


def test_similar_fits_large_categories_and_records_omitted_evidence(
    tmp_path: Path,
) -> None:
    source = _candidate(
        "1087",
        name="source",
        tags=tuple(f"tag-{index:03d}-{'x' * 20}" for index in range(100)),
        covers=False,
    )
    executor = FakeDiscoveryExecutor((source,), {"1087": _detail(source)})

    result = _service(tmp_path, executor).similar(
        "https://www.runninghub.ai/post/1087",
        sort="HOTTEST",
    )

    tag_query = next(item for item in result.derivations if item.source == "tag")
    assert len(tag_query.value) <= 512
    assert tag_query.omitted_count > 0
    assert [call.query for call in executor.search_calls].count(tag_query.value) == 1


def test_similar_unrepresentable_evidence_fails_terminally_with_artifacts(
    tmp_path: Path,
) -> None:
    source = _candidate("1088", name="source", covers=False)
    detail = _detail(source, used_models=(f"{'x' * 513}.safetensors",))
    executor = FakeDiscoveryExecutor((source,), {"1088": detail})

    with pytest.raises(service_module.DiscoveryExecutionError) as raised:
        _service(tmp_path, executor).similar(
            "https://www.runninghub.ai/post/1088",
            sort="HOTTEST",
        )

    error = raised.value
    assert error.experiment_id == "similar-20260203t040506z-fixedid"
    assert {item.kind for item in error.artifacts} >= {"search", "attempts"}
    snapshot = next(Path(item.path) for item in error.artifacts if item.kind == "search")
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["incomplete"] is True
    assert payload["attempt_count"] == 1
    assert executor.search_calls == []


def test_similar_without_derivable_evidence_stops_before_provider_search(
    tmp_path: Path,
) -> None:
    source = _candidate("1091", name="plain workflow", tags=(), covers=False)
    executor = FakeDiscoveryExecutor(
        (source,),
        {"1091": _detail(source)},
        graph={"1": {"class_type": "Reroute", "inputs": {}}},
    )
    service = _service(tmp_path, executor)

    with pytest.raises(DiscoveryNotFoundError, match="no derivable public capability evidence"):
        service.similar("https://www.runninghub.ai/post/1091", sort="HOTTEST")

    assert executor.search_calls == []


def test_similar_non_prompt_capability_never_adds_zero_grammar_query(
    tmp_path: Path,
) -> None:
    source = _candidate("1092", name="output workflow", tags=(), covers=False)
    executor = FakeDiscoveryExecutor(
        (source,),
        {"1092": _detail(source, primitive_nodes=("SaveImage",))},
        graph={
            "1": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "result"},
            }
        },
    )
    service = _service(tmp_path, executor)

    result = service.similar(
        "https://www.runninghub.ai/post/1092",
        sort="HOTTEST",
    )

    assert {item.source for item in result.derivations} == {"capability_role"}
    assert [item.value for item in result.derivations] == ["output"]
    assert executor.search_calls
    assert all("prompt grammar" not in call.query for call in executor.search_calls)


def test_similar_include_graphs_reuses_inspected_source_graph(tmp_path: Path) -> None:
    candidates, details = _golden_candidates()
    executor = FakeDiscoveryExecutor(candidates, details)
    service = _service(tmp_path, executor)

    service.similar(
        f"https://www.runninghub.ai/post/{NEXT_SCENE_ID}",
        sort="HOTTEST",
        include_graphs=True,
    )

    fetched_ids = [call.workflow_id for call in executor.fetch_calls]
    assert fetched_ids.count(NEXT_SCENE_ID) == 1
    assert fetched_ids.count(QWEN_ID) == 1


def test_service_missing_detail_returns_partial_diagnostic(tmp_path: Path) -> None:
    candidate = _candidate("999", name="missing")
    executor = FakeDiscoveryExecutor((candidate,), {})
    service = _service(tmp_path, executor)

    result = service.search("missing", sort="REPUTATION")

    assert result.results == ()
    assert result.incomplete is True
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].operation == "detail"
    assert result.diagnostics[0].workflow_id == "999"


def test_service_does_not_mutate_provider_candidates_or_details(tmp_path: Path) -> None:
    candidates, details = _golden_candidates()
    candidate_dump = [candidate.model_dump(mode="json") for candidate in candidates]
    detail_dump = {
        key: detail.model_dump(mode="json") for key, detail in details.items()
    }
    executor = FakeDiscoveryExecutor(candidates, details)

    _service(tmp_path, executor).search("next scene", sort="REPUTATION")

    assert [candidate.model_dump(mode="json") for candidate in candidates] == candidate_dump
    assert {
        key: detail.model_dump(mode="json") for key, detail in details.items()
    } == detail_dump
