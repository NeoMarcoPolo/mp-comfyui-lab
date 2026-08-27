"""Deterministic engagement scoring and workflow selection."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mp_comfyui_lab.community.models import WorkflowRecord


APPROVED_FAMILIES = frozenset(
    {
        "animate",
        "flux-2",
        "flux-2-klein",
        "krea-2",
        "minimax-h3",
        "qwen-image",
        "qwen-image-edit",
        "scail2",
        "wan-2.1",
        "wan-2.2",
        "z-image",
        "zit",
    }
)
OBSOLETE_FAMILIES = frozenset({"pony", "sdxl", "hunyuan"})
_FULL_WEIGHT_REACTIONS = frozenset(
    {
        "❤",
        "❤️",
        "🔥",
        "👍",
        "💯",
        "🎉",
        "approval",
        "great",
        "heart",
        "upvote",
        "yes",
    }
)
_HALF_WEIGHT_REACTIONS = frozenset(
    {
        "👀",
        "🔍",
        "🔖",
        "📌",
        "✉",
        "✉️",
        "bookmark",
        "inspect",
        "interest",
        "search",
    }
)


class ReactionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    emoji_id: str | None = None
    emoji_name: str
    count: int = Field(ge=0)
    burst_count: int = Field(default=0, ge=0)


class MessageEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    guild_id: str
    channel_id: str
    message_id: str
    canonical_url: str
    author_id: str
    author_is_bot: bool
    timestamp: datetime
    reactions: tuple[ReactionEvidence, ...] = ()
    referenced_message_id: str | None = None
    attachment_filenames: tuple[str, ...] = ()


class StarboardEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    starboard_message_id: str
    source_message_id: str
    historical_star_count: int = Field(ge=1)


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    guild_id: str
    complete: bool
    targets: tuple[str, ...]
    messages: dict[str, MessageEvidence]
    starboard_snapshots: tuple[StarboardEvidence, ...] = ()
    errors: dict[str, dict[str, str]] = {}


class RankedWorkflow(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow: WorkflowRecord
    primary_channel_id: str
    current_stars: int
    historical_stars: int
    star_count: int
    weighted_reactions: float
    independent_adoption_count: int
    starboard: bool
    opportunity_factor: float
    star_component: float
    reaction_component: float
    adoption_component: float
    starboard_component: float
    score: float
    inclusion_reasons: tuple[str, ...] = ()
    exclusion_reason: str | None = None


class SelectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected: tuple[RankedWorkflow, ...]
    excluded: tuple[RankedWorkflow, ...]


def _is_star(name: str) -> bool:
    folded = name.casefold()
    return name in {"⭐", "🌟"} or folded in {"star", "stars"} or folded.startswith("star_")


def _reaction_weight(name: str) -> float:
    folded = name.casefold()
    if name in _FULL_WEIGHT_REACTIONS or folded in _FULL_WEIGHT_REACTIONS:
        return 1.0
    if name in _HALF_WEIGHT_REACTIONS or folded in _HALF_WEIGHT_REACTIONS:
        return 0.5
    return 0.0


def _message_by_id(evidence: EvidenceSnapshot) -> dict[str, MessageEvidence]:
    result = {}
    for message in evidence.messages.values():
        result[message.message_id] = message
    return result


def _canonical_messages(
    workflow: WorkflowRecord,
    evidence: EvidenceSnapshot,
) -> list[MessageEvidence]:
    by_id = _message_by_id(evidence)
    found: dict[str, MessageEvidence] = {}
    for source in workflow.sources:
        message = evidence.messages.get(
            f"{source.guild_id}/{source.channel_id}/{source.message_id}"
        ) or by_id.get(source.message_id)
        visited = set()
        while (
            message is not None
            and message.referenced_message_id is not None
            and message.message_id not in visited
        ):
            visited.add(message.message_id)
            referenced = by_id.get(message.referenced_message_id)
            if referenced is None:
                break
            message = referenced
        if message is not None and not message.author_is_bot:
            found[message.message_id] = message
    return sorted(found.values(), key=lambda item: (item.timestamp, item.message_id))


def _aggregate(workflow: WorkflowRecord, evidence: EvidenceSnapshot, now: datetime) -> dict:
    canonical = _canonical_messages(workflow, evidence)
    if canonical:
        primary = canonical[0]
        primary_channel = primary.channel_id
        primary_timestamp = primary.timestamp
        primary_author = primary.author_id
    else:
        source = min(workflow.sources, key=lambda item: (item.timestamp, item.message_id))
        primary_channel = source.channel_id
        primary_timestamp = source.timestamp
        primary_author = source.author_id

    by_author: dict[str, tuple[int, float]] = {}
    for message in canonical:
        stars = sum(item.count for item in message.reactions if _is_star(item.emoji_name))
        weighted = sum(
            item.count * _reaction_weight(item.emoji_name)
            for item in message.reactions
            if not _is_star(item.emoji_name)
        )
        prior = by_author.get(message.author_id, (0, 0.0))
        by_author[message.author_id] = (max(prior[0], stars), max(prior[1], weighted))

    current_stars = max((value[0] for value in by_author.values()), default=0)
    weighted_reactions = sum(value[1] for value in by_author.values())
    canonical_ids = {item.message_id for item in canonical} | {
        source.message_id for source in workflow.sources
    }
    historical_stars = max(
        (
            item.historical_star_count
            for item in evidence.starboard_snapshots
            if item.source_message_id in canonical_ids
        ),
        default=0,
    )
    starboard = historical_stars > 0
    source_authors = {
        source.author_id
        for source in workflow.sources
        if source.author_id != primary_author
        and not any(
            message.message_id == source.message_id and message.author_is_bot
            for message in evidence.messages.values()
        )
    }
    canonical_authors = {
        message.author_id for message in canonical if message.author_id != primary_author
    }
    adoption_count = len(source_authors | canonical_authors)
    age_days = max(0.0, (now - primary_timestamp).total_seconds() / 86_400)
    opportunity_factor = min(1.0, max(0.25, age_days / 14.0))
    return {
        "primary_channel": primary_channel,
        "current_stars": current_stars,
        "historical_stars": historical_stars,
        "star_count": max(current_stars, historical_stars),
        "weighted_reactions": weighted_reactions,
        "adoption_count": adoption_count,
        "starboard": starboard,
        "opportunity_factor": opportunity_factor,
    }


def score_workflows(
    records: Sequence[WorkflowRecord],
    evidence: EvidenceSnapshot,
    now: datetime,
) -> list[RankedWorkflow]:
    """Calculate reproducible channel-normalized score components."""

    if not evidence.complete:
        raise ValueError("Engagement evidence must be complete before ranking")
    aggregates = {record.workflow_id: _aggregate(record, evidence, now) for record in records}
    raw_stars = {
        workflow_id: math.log1p(value["star_count"] * value["opportunity_factor"])
        for workflow_id, value in aggregates.items()
    }
    raw_reactions = {
        workflow_id: math.log1p(value["weighted_reactions"] * value["opportunity_factor"])
        for workflow_id, value in aggregates.items()
    }
    channel_star_max: dict[str, float] = defaultdict(float)
    channel_reaction_max: dict[str, float] = defaultdict(float)
    for workflow_id, aggregate in aggregates.items():
        channel = aggregate["primary_channel"]
        channel_star_max[channel] = max(channel_star_max[channel], raw_stars[workflow_id])
        channel_reaction_max[channel] = max(channel_reaction_max[channel], raw_reactions[workflow_id])

    ranked = []
    for record in records:
        aggregate = aggregates[record.workflow_id]
        channel = aggregate["primary_channel"]
        star_component = (
            raw_stars[record.workflow_id] / channel_star_max[channel]
            if channel_star_max[channel]
            else 0.0
        )
        reaction_component = (
            raw_reactions[record.workflow_id] / channel_reaction_max[channel]
            if channel_reaction_max[channel]
            else 0.0
        )
        adoption_component = min(1.0, aggregate["adoption_count"] / 3.0)
        starboard_component = 1.0 if aggregate["starboard"] else 0.0
        score = (
            0.50 * star_component
            + 0.25 * reaction_component
            + 0.15 * adoption_component
            + 0.10 * starboard_component
        )
        ranked.append(
            RankedWorkflow(
                workflow=record,
                primary_channel_id=channel,
                current_stars=aggregate["current_stars"],
                historical_stars=aggregate["historical_stars"],
                star_count=aggregate["star_count"],
                weighted_reactions=aggregate["weighted_reactions"],
                independent_adoption_count=aggregate["adoption_count"],
                starboard=aggregate["starboard"],
                opportunity_factor=aggregate["opportunity_factor"],
                star_component=star_component,
                reaction_component=reaction_component,
                adoption_component=adoption_component,
                starboard_component=starboard_component,
                score=score,
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.workflow.workflow_id))


def _is_low_bit(record: WorkflowRecord) -> bool:
    folded = " ".join(record.identity.core_models).casefold()
    return any(
        token in folded
        for token in (
            ".gguf",
            "int4",
            "int6",
            "int8",
            "nf4",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
        )
    )


def _eligibility_reasons(ranked: Sequence[RankedWorkflow]) -> dict[str, str | None]:
    reasons: dict[str, str | None] = {}
    quality_groups: dict[tuple[str, str, str], list[WorkflowRecord]] = defaultdict(list)
    for item in ranked:
        identity = item.workflow.identity
        quality_groups[(identity.family, identity.task, identity.technique)].append(item.workflow)
    for item in ranked:
        record = item.workflow
        family = record.identity.family
        if family in OBSOLETE_FAMILIES:
            reasons[record.workflow_id] = "obsolete_family"
        elif family not in APPROVED_FAMILIES:
            reasons[record.workflow_id] = "not_approved_family"
        else:
            group = quality_groups[(family, record.identity.task, record.identity.technique)]
            if _is_low_bit(record) and any(not _is_low_bit(candidate) for candidate in group):
                reasons[record.workflow_id] = "lower_quality_duplicate"
            else:
                reasons[record.workflow_id] = None
    return reasons


def select_test_workflows(ranked: Sequence[RankedWorkflow]) -> SelectionResult:
    """Apply the family guarantee and approved additional engagement gates."""

    eligibility = _eligibility_reasons(ranked)
    eligible = [item for item in ranked if eligibility[item.workflow.workflow_id] is None]
    family_winners: dict[str, str] = {}
    for item in eligible:
        family_winners.setdefault(item.workflow.identity.family, item.workflow.workflow_id)

    branch_top_half: set[str] = set()
    branches: dict[tuple[str, str], list[RankedWorkflow]] = defaultdict(list)
    for item in eligible:
        branches[(item.workflow.identity.family, item.workflow.identity.technique)].append(item)
    for branch in branches.values():
        ordered = sorted(branch, key=lambda item: (-item.score, item.workflow.workflow_id))
        cutoff = math.ceil(len(ordered) / 2)
        branch_top_half.update(item.workflow.workflow_id for item in ordered[:cutoff])

    selected = []
    excluded = []
    for item in ranked:
        workflow_id = item.workflow.workflow_id
        ineligible_reason = eligibility[workflow_id]
        if ineligible_reason is not None:
            excluded.append(item.model_copy(update={"exclusion_reason": ineligible_reason}))
            continue
        reasons = []
        if family_winners[item.workflow.identity.family] == workflow_id:
            reasons.append("family_winner")
        else:
            if item.starboard:
                reasons.append("starboard")
            if item.weighted_reactions >= 3:
                reasons.append("weighted_reactions")
            if item.independent_adoption_count > 0:
                reasons.append("independent_adoption")
            if workflow_id in branch_top_half:
                reasons.append("top_half_family_technique")
        if reasons:
            selected.append(item.model_copy(update={"inclusion_reasons": tuple(reasons)}))
        else:
            excluded.append(item.model_copy(update={"exclusion_reason": "engagement_gate_not_met"}))
    return SelectionResult(selected=tuple(selected), excluded=tuple(excluded))
