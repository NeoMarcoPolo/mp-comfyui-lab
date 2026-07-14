from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
from importlib import resources
import json
from math import isfinite, log, log1p
import re
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from mp_comfyui_lab.config import safe_load_yaml
from mp_comfyui_lab.discovery.models import Engagement, WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.discovery.query import (
    ExpandedQuery,
    expand_query,
    query_tokens,
    term_match,
)


POSITIVE_WEIGHT_FIELDS = (
    "relevance",
    "engagement",
    "output_evidence",
    "recency",
    "creator_track_record",
    "experimentability",
    "novelty",
)
# Stable public-evidence mix: reuse is the strongest signal, while likes and
# collections contribute equally. Each metric is log1p-normalized independently.
ENGAGEMENT_METRIC_MIX = {"uses": 0.6, "likes": 0.2, "collects": 0.2}
MAX_EXPLANATION_LENGTH = 1_024
_LOG_TWO = log(2.0)
_LOG_MANTISSA_BITS = 53


class RankingConfigError(ValueError):
    pass


def _bounded_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be a number")
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")
    return parsed


Score = Annotated[float, BeforeValidator(_bounded_score)]
HashValue = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
BoundedText = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class RankingWeights(_FrozenModel):
    relevance: Score
    engagement: Score
    output_evidence: Score
    recency: Score
    creator_track_record: Score
    experimentability: Score
    novelty: Score
    duplicate_penalty: Score

    @model_validator(mode="after")
    def validate_positive_total(self) -> RankingWeights:
        total = sum(getattr(self, field) for field in POSITIVE_WEIGHT_FIELDS)
        if not isfinite(total) or total <= 0.0:
            raise ValueError("positive weight total must be finite and greater than zero")
        return self


class RankingComponents(_FrozenModel):
    relevance: Score
    engagement: Score
    output_evidence: Score
    recency: Score
    creator_track_record: Score
    experimentability: Score
    novelty: Score
    penalty: Score

    def __iter__(self):
        return iter(type(self).model_fields)

    def __getitem__(self, field: str) -> float:
        if field not in type(self).model_fields:
            raise KeyError(field)
        return getattr(self, field)

    def keys(self) -> tuple[str, ...]:
        return tuple(type(self).model_fields)


class GraphEvidence(_FrozenModel):
    source_hash: HashValue | None = None
    capability_hash: HashValue | None = None
    roles: tuple[BoundedText, ...] = Field(default=(), max_length=64)
    canonicalization_error: BoundedText | None = None


class Artifact(_FrozenModel):
    kind: BoundedText
    path: BoundedText


class RankedWorkflow(_FrozenModel):
    candidate: WorkflowCandidate
    components: RankingComponents
    score: Score
    explanation: Annotated[
        str,
        StringConstraints(
            strict=True,
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_EXPLANATION_LENGTH,
        ),
    ]
    duplicate_group: str | None = None
    source_hash: HashValue | None = None
    capability_hash: HashValue | None = None
    canonicalization_error: BoundedText | None = None

    @property
    def total_score(self) -> float:
        return self.score


def load_default_ranking_weights(
    overrides: Mapping[str, Any] | None = None,
) -> RankingWeights:
    try:
        document = (
            resources.files("mp_comfyui_lab.resources")
            .joinpath("default-ranking.yaml")
            .read_text(encoding="utf-8")
        )
        payload = safe_load_yaml(
            document,
            source="default-ranking.yaml",
            error_type=RankingConfigError,
        )
    except RankingConfigError:
        raise
    except (OSError, TypeError) as error:
        raise RankingConfigError(f"could not load default ranking weights: {error}") from error
    if not isinstance(payload, dict):
        raise RankingConfigError("default ranking weights must be an object")
    merged = dict(payload)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise RankingConfigError("ranking overrides must be an object")
        merged.update(overrides)
    try:
        return RankingWeights.model_validate(merged)
    except ValidationError as error:
        raise RankingConfigError(str(error)) from error


def _tokens(value: str) -> tuple[str, ...]:
    return query_tokens(value)


def _query_units(plan: ExpandedQuery) -> tuple[tuple[str, ...], ...]:
    units: list[tuple[str, ...]] = []
    activated_concepts: set[str] = set()
    for source_query in plan.source_queries or (plan.original,):
        remainder = source_query
        for expansion in plan.expansions:
            matched = term_match(remainder, expansion.source)
            if matched is None:
                continue
            if expansion.concept not in activated_concepts:
                activated_concepts.add(expansion.concept)
                units.append(expansion.expanded_terms)
            remainder = remainder[: matched.start()] + " " + remainder[matched.end() :]
        units.extend((token,) for token in _tokens(remainder))
    unique: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for unit in units:
        canonical = tuple(term.casefold() for term in unit)
        if canonical not in seen:
            seen.add(canonical)
            unique.append(unit)
    return tuple(unique)


def _search_fields(detail: WorkflowDetail) -> tuple[str, ...]:
    candidate = detail.candidate
    return tuple(
        value
        for value in (
            candidate.name,
            candidate.description or "",
            *candidate.tags,
            *detail.used_models,
            *detail.primitive_nodes,
            *detail.custom_nodes,
        )
        if value
    )


def _relevance(plan: ExpandedQuery, detail: WorkflowDetail) -> float:
    units = _query_units(plan)
    if not units:
        return 0.0
    fields = _search_fields(detail)
    matched = sum(
        1
        for aliases in units
        if any(
            term_match(field, alias) is not None
            for alias in aliases
            for field in fields
        )
    )
    return matched / len(units)


def _count(engagement: Engagement, field: str) -> int:
    return int(getattr(engagement, field) or 0)


def _integer_log1p(value: int) -> float:
    if value < 0:
        raise ValueError("engagement count must be nonnegative")
    try:
        return log1p(value)
    except OverflowError:
        bit_length = value.bit_length()
        shift = max(0, bit_length - _LOG_MANTISSA_BITS)
        leading = value >> shift
        mantissa = leading / (1 << (bit_length - shift - 1))
        return (bit_length - 1) * _LOG_TWO + log(mantissa)


def _normalized_log1p(value: int, maximum: int) -> float:
    if maximum <= 0 or value <= 0:
        return 0.0
    if value >= maximum:
        return 1.0
    return max(0.0, min(1.0, _integer_log1p(value) / _integer_log1p(maximum)))


def _metric_score(
    engagement: Engagement,
    maxima: Mapping[str, int],
) -> float:
    score = 0.0
    for field, mix in ENGAGEMENT_METRIC_MIX.items():
        maximum = maxima[field]
        if maximum > 0:
            score += mix * _normalized_log1p(
                _count(engagement, field),
                maximum,
            )
    return max(0.0, min(1.0, score))


def _engagement_maxima(details: Sequence[WorkflowDetail]) -> dict[str, int]:
    return {
        field: max((_count(item.candidate.engagement, field) for item in details), default=0)
        for field in ENGAGEMENT_METRIC_MIX
    }


def _creator_scores(details: Sequence[WorkflowDetail]) -> dict[str, float]:
    aggregates: dict[str, dict[str, int]] = defaultdict(
        lambda: {field: 0 for field in ENGAGEMENT_METRIC_MIX}
    )
    for detail in details:
        candidate = detail.candidate
        creator = (
            candidate.author_id
            or candidate.author_name
            or f"workflow:{candidate.workflow_id}"
        )
        for field in ENGAGEMENT_METRIC_MIX:
            aggregates[creator][field] += _count(candidate.engagement, field)
    maxima = {
        field: max((counts[field] for counts in aggregates.values()), default=0)
        for field in ENGAGEMENT_METRIC_MIX
    }
    scores: dict[str, float] = {}
    for creator, counts in aggregates.items():
        score = 0.0
        for field, mix in ENGAGEMENT_METRIC_MIX.items():
            if maxima[field] > 0:
                score += mix * _normalized_log1p(counts[field], maxima[field])
        scores[creator] = max(0.0, min(1.0, score))
    return scores


def _recency_scores(details: Sequence[WorkflowDetail]) -> dict[str, float]:
    dated = [item.candidate.published_at for item in details if item.candidate.published_at]
    if not dated:
        return {item.candidate.workflow_id: 0.0 for item in details}
    earliest = min(dated)
    latest = max(dated)
    span = (latest - earliest).total_seconds()
    return {
        item.candidate.workflow_id: (
            0.0
            if item.candidate.published_at is None
            else 1.0
            if span == 0.0
            else (item.candidate.published_at - earliest).total_seconds() / span
        )
        for item in details
    }


def _class_tokens(value: str) -> tuple[str, ...]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", separated)
    return query_tokens(separated)


def _is_output_class(value: str) -> bool:
    tokens = set(_class_tokens(value))
    return bool(
        tokens & {"output", "preview"}
        or (tokens & {"save", "saver"} and "image" in tokens)
    )


def _output_evidence(
    detail: WorkflowDetail,
    evidence: GraphEvidence | None,
) -> float:
    output_nodes = tuple(
        node
        for node in detail.primitive_nodes + detail.custom_nodes
        if _is_output_class(node)
    )
    has_output_role = evidence is not None and "output" in evidence.roles
    return min(
        1.0,
        (0.4 if detail.candidate.covers else 0.0)
        + (0.4 if detail.preview is not None else 0.0)
        + (0.2 if output_nodes or has_output_role else 0.0),
    )


def _experimentability(detail: WorkflowDetail, evidence: GraphEvidence | None) -> float:
    inventory = detail.primitive_nodes + detail.custom_nodes
    return min(
        1.0,
        (0.35 if detail.used_models else 0.0)
        + (0.35 if inventory else 0.0)
        + (
            0.3
            if evidence is not None
            and evidence.canonicalization_error is None
            and (evidence.roles or evidence.source_hash or evidence.capability_hash)
            else 0.0
        ),
    )


def coarse_duplicate_group(detail: WorkflowDetail) -> str | None:
    if not (detail.used_models or detail.primitive_nodes or detail.custom_nodes):
        return None
    projection = {
        "models": list(detail.used_models),
        "primitive_nodes": sorted(detail.primitive_nodes),
        "custom_nodes": sorted(detail.custom_nodes),
    }
    serialized = json.dumps(projection, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "coarse:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _duplicate_keys(
    details: Sequence[WorkflowDetail],
    graph_evidence: Mapping[str, GraphEvidence],
) -> dict[str, str | None]:
    detail_ids = {detail.candidate.workflow_id for detail in details}
    successful = {
        workflow_id: evidence
        for workflow_id, evidence in graph_evidence.items()
        if workflow_id in detail_ids
        if evidence.canonicalization_error is None
        and evidence.source_hash is not None
        and evidence.capability_hash is not None
    }
    parents = {workflow_id: workflow_id for workflow_id in successful}

    def find(workflow_id: str) -> str:
        while parents[workflow_id] != workflow_id:
            parents[workflow_id] = parents[parents[workflow_id]]
            workflow_id = parents[workflow_id]
        return workflow_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_source: dict[str, list[str]] = defaultdict(list)
    by_capability: dict[str, list[str]] = defaultdict(list)
    for workflow_id, evidence in successful.items():
        by_source[evidence.source_hash].append(workflow_id)
        by_capability[evidence.capability_hash].append(workflow_id)
    for related in tuple(by_source.values()) + tuple(by_capability.values()):
        for workflow_id in related[1:]:
            union(related[0], workflow_id)

    exact_components: dict[str, list[str]] = defaultdict(list)
    for workflow_id in successful:
        exact_components[find(workflow_id)].append(workflow_id)
    exact_keys: dict[str, str | None] = {}
    for workflow_ids in exact_components.values():
        if len(workflow_ids) == 1:
            exact_keys[workflow_ids[0]] = None
            continue
        source_hashes = {successful[item].source_hash for item in workflow_ids}
        capability_hashes = {
            successful[item].capability_hash for item in workflow_ids
        }
        if len(source_hashes) == 1:
            group_key = f"source:{next(iter(source_hashes))}"
        elif len(capability_hashes) == 1:
            group_key = f"capability:{next(iter(capability_hashes))}"
        else:
            projection = [
                (
                    workflow_id,
                    successful[workflow_id].source_hash,
                    successful[workflow_id].capability_hash,
                )
                for workflow_id in sorted(workflow_ids)
            ]
            digest = hashlib.sha256(
                json.dumps(projection, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            group_key = f"exact:{digest}"
        for workflow_id in workflow_ids:
            exact_keys[workflow_id] = group_key

    keys: dict[str, str | None] = {}
    for detail in details:
        workflow_id = detail.candidate.workflow_id
        evidence = graph_evidence.get(workflow_id)
        if evidence is not None:
            if evidence.canonicalization_error is not None:
                keys[workflow_id] = None
                continue
            enriched = successful.get(workflow_id)
            if enriched is not None:
                keys[workflow_id] = exact_keys[workflow_id]
                continue
            keys[workflow_id] = None
            continue
        keys[workflow_id] = coarse_duplicate_group(detail)

    coarse_members: dict[str, list[str]] = defaultdict(list)
    for detail in details:
        coarse_key = coarse_duplicate_group(detail)
        if coarse_key is not None:
            coarse_members[coarse_key].append(detail.candidate.workflow_id)
    for coarse_key, workflow_ids in coarse_members.items():
        if any(
            graph_evidence.get(workflow_id) is not None
            and graph_evidence[workflow_id].canonicalization_error is not None
            for workflow_id in workflow_ids
        ):
            for workflow_id in workflow_ids:
                keys[workflow_id] = None
            continue
        enriched_ids = [
            workflow_id
            for workflow_id in workflow_ids
            if workflow_id in successful
        ]
        uninspected_ids = [
            workflow_id
            for workflow_id in workflow_ids
            if workflow_id not in graph_evidence
        ]
        if len(enriched_ids) == 1 and uninspected_ids:
            for workflow_id in workflow_ids:
                keys[workflow_id] = coarse_key
    return keys


def _positive_score(
    values: Mapping[str, float],
    weights: RankingWeights,
    *,
    include_novelty: bool = True,
) -> float:
    fields = POSITIVE_WEIGHT_FIELDS if include_novelty else POSITIVE_WEIGHT_FIELDS[:-1]
    total = sum(getattr(weights, field) for field in fields)
    if total == 0.0:
        return 0.0
    return sum(getattr(weights, field) * values[field] for field in fields) / total


def _explanation(
    components: RankingComponents,
    candidate: WorkflowCandidate,
    duplicate_group: str | None,
) -> str:
    strongest = max(
        POSITIVE_WEIGHT_FIELDS,
        key=lambda field: (getattr(components, field), -POSITIVE_WEIGHT_FIELDS.index(field)),
    )
    engagement = candidate.engagement
    sentence = (
        f"Strongest evidence: {strongest} ({getattr(components, strongest):.2f}); "
        f"{_format_count(_count(engagement, 'uses'))} uses, "
        f"{_format_count(_count(engagement, 'likes'))} likes, and "
        f"{_format_count(_count(engagement, 'collects'))} collects."
    )
    if components.penalty > 0.0:
        sentence += f" Applied duplicate penalty for {duplicate_group}."
    return sentence


def _format_count(value: int) -> str:
    if value < 1_000_000_000_000_000:
        return f"{value:,}"
    return f"~2^{value.bit_length() - 1}"


def _canonical_candidates(
    candidates: Sequence[WorkflowCandidate],
) -> dict[str, WorkflowCandidate]:
    grouped: dict[str, list[WorkflowCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.workflow_id].append(candidate)
    return {
        workflow_id: min(
            values,
            key=lambda value: value.model_dump_json(),
        )
        for workflow_id, values in grouped.items()
    }


def rank_candidates(
    query: str | ExpandedQuery,
    candidates: Sequence[WorkflowCandidate],
    details: Mapping[str, WorkflowDetail],
    weights: RankingWeights,
    *,
    graph_evidence: Mapping[str, GraphEvidence] | None = None,
) -> tuple[RankedWorkflow, ...]:
    if not isinstance(weights, RankingWeights):
        raise TypeError("weights must be RankingWeights")
    plan = expand_query(query) if isinstance(query, str) else query
    canonical = _canonical_candidates(candidates)
    ordered_details = tuple(
        details[workflow_id]
        for workflow_id in sorted(canonical)
        if workflow_id in details
    )
    evidence_by_id = dict(graph_evidence or {})
    duplicate_keys = _duplicate_keys(ordered_details, evidence_by_id)
    groups: dict[str, list[str]] = defaultdict(list)
    for workflow_id, key in duplicate_keys.items():
        if key is not None:
            groups[key].append(workflow_id)
    representatives: dict[str, str] = {}
    for key, workflow_ids in groups.items():
        if len(workflow_ids) < 2:
            duplicate_keys[workflow_ids[0]] = None
            continue
        representatives[key] = min(
            workflow_ids,
            key=lambda workflow_id: (
                -_count(details[workflow_id].candidate.engagement, "uses"),
                -_count(details[workflow_id].candidate.engagement, "likes"),
                -_count(details[workflow_id].candidate.engagement, "collects"),
                workflow_id,
                details[workflow_id].candidate.canonical_url,
            ),
        )
    representative_ids = {
        workflow_id
        for workflow_id, key in duplicate_keys.items()
        if key is None
    } | set(representatives.values())
    engagement_maxima = _engagement_maxima(ordered_details)
    creator_scores = _creator_scores(
        tuple(
            detail
            for detail in ordered_details
            if detail.candidate.workflow_id in representative_ids
        )
    )
    recency_scores = _recency_scores(ordered_details)

    raw: dict[str, dict[str, float]] = {}
    for detail in ordered_details:
        candidate = detail.candidate
        creator = (
            candidate.author_id
            or candidate.author_name
            or f"workflow:{candidate.workflow_id}"
        )
        raw[candidate.workflow_id] = {
            "relevance": _relevance(plan, detail),
            "engagement": _metric_score(candidate.engagement, engagement_maxima),
            "output_evidence": _output_evidence(
                detail,
                evidence_by_id.get(candidate.workflow_id),
            ),
            "recency": recency_scores[candidate.workflow_id],
            "creator_track_record": creator_scores.get(creator, 0.0),
            "experimentability": _experimentability(
                detail, evidence_by_id.get(candidate.workflow_id)
            ),
            "novelty": 1.0,
        }

    penalties = {detail.candidate.workflow_id: 0.0 for detail in ordered_details}
    for key, workflow_ids in groups.items():
        if len(workflow_ids) < 2:
            continue
        representative = representatives[key]
        for workflow_id in workflow_ids:
            raw[workflow_id]["novelty"] = 0.5 if workflow_id == representative else 0.0
            penalties[workflow_id] = 0.0 if workflow_id == representative else 1.0

    ranked: list[RankedWorkflow] = []
    for detail in ordered_details:
        candidate = detail.candidate
        workflow_id = candidate.workflow_id
        components = RankingComponents(
            **raw[workflow_id],
            penalty=penalties[workflow_id],
        )
        score = _positive_score(raw[workflow_id], weights)
        score = max(0.0, min(1.0, score - weights.duplicate_penalty * components.penalty))
        evidence = evidence_by_id.get(workflow_id)
        duplicate_group = duplicate_keys[workflow_id]
        ranked.append(
            RankedWorkflow(
                candidate=candidate,
                components=components,
                score=score,
                explanation=_explanation(components, candidate, duplicate_group),
                duplicate_group=duplicate_group,
                source_hash=evidence.source_hash if evidence is not None else None,
                capability_hash=evidence.capability_hash if evidence is not None else None,
                canonicalization_error=(
                    evidence.canonicalization_error if evidence is not None else None
                ),
            )
        )

    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                -item.score,
                -_count(item.candidate.engagement, "uses"),
                -_count(item.candidate.engagement, "likes"),
                -_count(item.candidate.engagement, "collects"),
                item.candidate.workflow_id,
                item.candidate.canonical_url,
            ),
        )
    )
