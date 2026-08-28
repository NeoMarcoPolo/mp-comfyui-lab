from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mp_comfyui_lab.community.models import (
    WorkflowIdentity,
    WorkflowRecord,
    WorkflowSource,
    WorkflowVariant,
)
from mp_comfyui_lab.community.ranking import (
    EvidenceSnapshot,
    score_workflows,
    select_test_workflows,
)


NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def _record(
    workflow_id: str,
    *,
    family: str,
    technique: str = "base",
    task: str = "text-to-image",
    author: str = "author",
    channel: str = "channel",
    timestamp: datetime | None = None,
    models: tuple[str, ...] = ("native_fp8.safetensors",),
) -> WorkflowRecord:
    source = WorkflowSource(
        guild_id="guild",
        channel_id=channel,
        message_id=workflow_id,
        message_url=f"https://discord.com/channels/guild/{channel}/{workflow_id}",
        author_id=author,
        timestamp=timestamp or NOW - timedelta(days=30),
        attachment_id=f"attachment-{workflow_id}",
        attachment_filename=f"{workflow_id}.png",
        attachment_sha256=f"sha-{workflow_id}",
    )
    identity = WorkflowIdentity(
        family=family,
        version="current",
        task=task,
        technique=technique,
        capability_hash=f"cap-{workflow_id}",
        core_models=models,
        loras=(),
        sampler_signature=("KSampler:euler",),
    )
    return WorkflowRecord(
        workflow_id=workflow_id,
        identity=identity,
        sources=(source,),
        api_graph_paths=(f"{workflow_id}.json",),
        ui_graph_paths=(),
        source_graph_hashes=(f"graph-{workflow_id}",),
        image_hashes=(f"image-{workflow_id}",),
        variants=(
            WorkflowVariant(
                variant_id=f"variant-{workflow_id}",
                source_key=source.source_key,
                api_graph_path=f"{workflow_id}.json",
                ui_graph_paths=(),
                source_graph_hash=f"graph-{workflow_id}",
                image_hash=f"image-{workflow_id}",
            ),
        ),
    )


def _message(
    record: WorkflowRecord,
    *,
    stars: int = 0,
    heart: int = 0,
    search: int = 0,
    unknown: int = 0,
    bot: bool = False,
) -> dict:
    source = record.sources[0]
    reactions = []
    for name, count in (("⭐", stars), ("❤️", heart), ("🔍", search), ("🤷", unknown)):
        if count:
            reactions.append({"emoji_id": None, "emoji_name": name, "count": count, "burst_count": 0})
    return {
        "guild_id": source.guild_id,
        "channel_id": source.channel_id,
        "message_id": source.message_id,
        "canonical_url": source.message_url,
        "author_id": source.author_id,
        "author_is_bot": bot,
        "timestamp": source.timestamp.isoformat(),
        "reactions": reactions,
        "referenced_message_id": None,
        "attachment_filenames": [source.attachment_filename],
    }


def _evidence(records: list[WorkflowRecord], messages: dict[str, dict], starboards: list[dict] | None = None) -> EvidenceSnapshot:
    return EvidenceSnapshot.model_validate(
        {
            "schema_version": 1,
            "guild_id": "guild",
            "complete": True,
            "targets": [f"guild/{item.sources[0].channel_id}/{item.workflow_id}" for item in records],
            "messages": messages,
            "starboard_snapshots": starboards or [],
            "errors": {},
        }
    )


def test_score_uses_peak_stars_explicit_reaction_weights_and_components() -> None:
    record = _record("one", family="krea-2")
    key = "guild/channel/one"
    evidence = _evidence(
        [record],
        {key: _message(record, stars=7, heart=2, search=2, unknown=99)},
        [{"starboard_message_id": "board", "source_message_id": "one", "historical_star_count": 9}],
    )

    ranked = score_workflows([record], evidence, NOW)[0]

    assert ranked.current_stars == 7
    assert ranked.historical_stars == 9
    assert ranked.star_count == 9
    assert ranked.weighted_reactions == 3.0
    assert ranked.starboard is True
    assert ranked.star_component == 1.0
    assert ranked.reaction_component == 1.0
    assert ranked.starboard_component == 1.0
    assert ranked.score == 0.85


def test_score_counts_each_independent_non_bot_author_once() -> None:
    original = _record("one", family="wan-2.2", author="creator")
    sources = list(original.sources)
    for message_id, author in (("two", "adopter"), ("three", "adopter"), ("four", "bot")):
        sources.append(
            original.sources[0].model_copy(
                update={
                    "message_id": message_id,
                    "message_url": f"https://discord.com/channels/guild/channel/{message_id}",
                    "attachment_id": f"attachment-{message_id}",
                    "author_id": author,
                }
            )
        )
    record = original.model_copy(
        update={
            "sources": tuple(sources),
            "variants": tuple(
                WorkflowVariant(
                    variant_id=f"variant-{source.message_id}",
                    source_key=source.source_key,
                    api_graph_path=f"{source.message_id}.json",
                    ui_graph_paths=(),
                    source_graph_hash=f"graph-{source.message_id}",
                    image_hash=f"image-{source.message_id}",
                )
                for source in sources
            ),
        }
    )
    messages = {
        f"guild/channel/{source.message_id}": {
            **_message(original),
            "message_id": source.message_id,
            "canonical_url": source.message_url,
            "author_id": source.author_id,
            "author_is_bot": source.author_id == "bot",
        }
        for source in sources
    }

    ranked = score_workflows([record], _evidence([record], messages), NOW)[0]

    assert ranked.independent_adoption_count == 1
    assert ranked.adoption_component == 1 / 3


def test_score_selects_representative_variant_from_highest_engagement_source() -> None:
    original = _record("one", family="krea-2", author="creator")
    repost_source = original.sources[0].model_copy(
        update={
            "message_id": "two",
            "message_url": "https://discord.com/channels/guild/channel/two",
            "attachment_id": "attachment-two",
            "author_id": "adopter",
        }
    )
    record = original.model_copy(
        update={
            "sources": (*original.sources, repost_source),
            "variants": (
                *original.variants,
                WorkflowVariant(
                    variant_id="variant-two",
                    source_key=repost_source.source_key,
                    api_graph_path="two.json",
                    ui_graph_paths=(),
                    source_graph_hash="graph-two",
                    image_hash="image-two",
                ),
            ),
        }
    )
    messages = {
        "guild/channel/one": _message(original, stars=1),
        "guild/channel/two": {
            **_message(original, stars=9),
            "message_id": "two",
            "canonical_url": repost_source.message_url,
            "author_id": "adopter",
        },
    }

    ranked = score_workflows([record], _evidence([record], messages), NOW)[0]

    assert ranked.representative_variant_id == "variant-two"


def test_new_post_confidence_floor_does_not_let_one_reaction_beat_established_post() -> None:
    recent = _record("recent", family="qwen-image", timestamp=NOW - timedelta(hours=6))
    established = _record("established", family="qwen-image", timestamp=NOW - timedelta(days=30))
    messages = {
        "guild/channel/recent": _message(recent, heart=1),
        "guild/channel/established": _message(established, heart=1),
    }

    ranked = {item.workflow.workflow_id: item for item in score_workflows([recent, established], _evidence([recent, established], messages), NOW)}

    assert ranked["recent"].reaction_component < ranked["established"].reaction_component


def test_selection_guarantees_family_winner_and_applies_additional_gates() -> None:
    family_low = _record("family-low", family="krea-2", channel="small")
    winner = _record("winner", family="wan-2.2")
    gate = _record("gate", family="wan-2.2")
    excluded = _record("excluded", family="wan-2.2")
    records = [family_low, winner, gate, excluded]
    messages = {
        "guild/small/family-low": _message(family_low),
        "guild/channel/winner": _message(winner, stars=8),
        "guild/channel/gate": _message(gate, heart=3),
        "guild/channel/excluded": _message(excluded),
    }

    result = select_test_workflows(score_workflows(records, _evidence(records, messages), NOW))
    selected = {item.workflow.workflow_id: item for item in result.selected}
    rejected = {item.workflow.workflow_id: item for item in result.excluded}

    assert "family-low" in selected
    assert "family_winner" in selected["family-low"].inclusion_reasons
    assert "winner" in selected
    assert "gate" in selected
    assert "weighted_reactions" in selected["gate"].inclusion_reasons
    assert rejected["excluded"].exclusion_reason == "engagement_gate_not_met"


def test_selection_excludes_obsolete_and_lower_quality_duplicates() -> None:
    obsolete = _record("old", family="sdxl")
    native = _record("native", family="flux-2", models=("flux2_fp8.safetensors",))
    gguf = _record("gguf", family="flux-2", models=("flux2_q4.gguf",))
    int8 = _record("int8", family="flux-2", models=("flux2_int8_convrot.safetensors",))
    records = [obsolete, native, gguf, int8]
    messages = {
        f"guild/channel/{record.workflow_id}": _message(record, stars=5)
        for record in records
    }

    result = select_test_workflows(score_workflows(records, _evidence(records, messages), NOW))
    excluded = {item.workflow.workflow_id: item.exclusion_reason for item in result.excluded}

    assert excluded["old"] == "obsolete_family"
    assert excluded["gguf"] == "lower_quality_duplicate"
    assert excluded["int8"] == "lower_quality_duplicate"
    assert "native" in {item.workflow.workflow_id for item in result.selected}


def test_selection_excludes_gguf_even_when_it_is_the_only_family_candidate() -> None:
    gguf = _record("gguf-only", family="qwen-image", models=("qwen-image-2512-Q8_0.gguf",))
    messages = {"guild/channel/gguf-only": _message(gguf, stars=12)}

    result = select_test_workflows(score_workflows([gguf], _evidence([gguf], messages), NOW))

    assert result.selected == ()
    assert result.excluded[0].exclusion_reason == "lower_quality_duplicate"
