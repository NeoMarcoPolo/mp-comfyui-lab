from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from mp_comfyui_lab.json_types import SanitizedJsonObject, thaw_json

from .models import (
    DimensionName,
    ExperimentPlan,
    PlannedVariant,
    PlanningConfidence,
)


SATURATION_REASON = (
    "no_approved_non_dominated_novel_capability_signature_remains"
)
REPEATED_FAILURE_THRESHOLD = 2

_CONFIDENCE_STRENGTH = {
    PlanningConfidence.UNEXPLORED: 0,
    PlanningConfidence.WEAKLY_INFERRED: 1,
    PlanningConfidence.STRONGLY_INFERRED: 2,
    PlanningConfidence.OBSERVED: 3,
}


class RerunReasonRequired(ValueError):
    """Raised when an exact rerun has no explicit experimental reason."""


ArtifactId = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
ArtifactHash = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class WaveArtifact(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1
    ordinal: int = Field(ge=1)
    plan_hash: ArtifactHash
    completed: Literal[True] = True
    considered_variant_ids: tuple[ArtifactId, ...]
    selected_variant_ids: tuple[ArtifactId, ...] = ()
    pending_variant_ids: tuple[ArtifactId, ...] = ()
    skips: SanitizedJsonObject
    candidate_evidence: SanitizedJsonObject
    run_ids: tuple[ArtifactId, ...] = ()
    pending_run_ids: tuple[ArtifactId, ...] = ()
    objective_evidence: SanitizedJsonObject
    saturation_reason: ArtifactId | None = None

    @field_validator(
        "considered_variant_ids",
        "selected_variant_ids",
        "pending_variant_ids",
        "run_ids",
        "pending_run_ids",
        mode="before",
    )
    @classmethod
    def json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        considered = set(self.considered_variant_ids)
        selected = set(self.selected_variant_ids)
        pending = set(self.pending_variant_ids)
        skipped = set(self.skips)
        if len(considered) != len(self.considered_variant_ids):
            raise ValueError("considered variant IDs must be unique")
        if len(selected) != len(self.selected_variant_ids):
            raise ValueError("selected variant IDs must be unique")
        if len(pending) != len(self.pending_variant_ids):
            raise ValueError("pending variant IDs must be unique")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("wave run IDs must be unique")
        if len(set(self.pending_run_ids)) != len(self.pending_run_ids):
            raise ValueError("pending wave run IDs must be unique")
        if not selected.issubset(considered):
            raise ValueError("selected variants must be considered")
        if not pending.issubset(considered - selected):
            raise ValueError("pending variants must be unselected considered variants")
        if skipped != considered - selected:
            raise ValueError("wave skips must cover exactly the unselected variants")
        if self.saturation_reason is not None and selected:
            raise ValueError("a wave with selections cannot be saturated")
        if self.saturation_reason is not None and pending:
            raise ValueError("a wave with pending runs cannot be saturated")
        if self.saturation_reason is None and not selected and not pending:
            raise ValueError(
                "a wave without selections or pending runs requires saturation reason"
            )
        if len(self.run_ids) != len(self.selected_variant_ids):
            raise ValueError("every selected variant requires one durable run ID")
        if bool(self.pending_run_ids) != bool(self.pending_variant_ids):
            raise ValueError(
                "pending variants and durable pending run IDs must coexist"
            )
        return self


class CorrectionRecord(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal[1] = 1
    failure_signature: ArtifactId
    failed_variant_id: ArtifactId
    failed_exact_run_hash: ArtifactHash
    failed_capability_hash: ArtifactHash
    candidate_variant_id: ArtifactId
    changed_dimensions: tuple[DimensionName, ...] = Field(min_length=1)
    rationale: ArtifactId | None = None

    @field_validator("changed_dimensions", mode="before")
    @classmethod
    def json_dimensions(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(
            DimensionName(item) if isinstance(item, str) else item
            for item in value
        )

    @model_validator(mode="after")
    def unique_dimensions(self) -> Self:
        if len(set(self.changed_dimensions)) != len(self.changed_dimensions):
            raise ValueError("correction changed dimensions must be unique")
        return self


@dataclass(frozen=True, order=True)
class EvidenceStrength:
    confidence: int
    provenance_rank: int
    evidence_count: int
    engagement: int
    structural_cleanliness: int

    def as_dict(self) -> dict[str, int]:
        return {
            "confidence": self.confidence,
            "provenance_rank": self.provenance_rank,
            "evidence_count": self.evidence_count,
            "engagement": self.engagement,
            "structural_cleanliness": self.structural_cleanliness,
        }


@dataclass(frozen=True)
class RerunRequest:
    variant_id: str
    reason: str
    original_exact_run_hash: str
    original_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class WaveDecision:
    variants: tuple[PlannedVariant, ...]
    considered_variant_ids: tuple[str, ...]
    skipped_variant_ids: tuple[str, ...]
    pending_variant_ids: tuple[str, ...]
    reasons: Mapping[str, str]
    evidence: Mapping[str, Mapping[str, int]]
    saturation_reason: str | None


@dataclass
class ExperimentHistory:
    exact_run_hashes: set[str] = field(default_factory=set)
    pending_exact_run_hashes: set[str] = field(default_factory=set)
    pending_capability_hashes: dict[str, set[str]] = field(default_factory=dict)
    pending_run_ids_by_exact_hash: dict[str, set[str]] = field(default_factory=dict)
    successful_variant_ids: set[str] = field(default_factory=set)
    successful_capability_hashes: dict[str, set[str]] = field(default_factory=dict)
    run_ids_by_exact_hash: dict[str, set[str]] = field(default_factory=dict)
    equivalent_groups: list[frozenset[str]] = field(default_factory=list)
    failure_signatures: Counter[str] = field(default_factory=Counter)
    failure_capabilities: dict[str, set[str]] = field(default_factory=dict)
    failed_variants: dict[
        str,
        dict[tuple[str, str, str], PlannedVariant],
    ] = field(default_factory=dict)
    evidence_by_variant_id: dict[str, EvidenceStrength] = field(default_factory=dict)

    def record_success(
        self,
        variant: PlannedVariant,
        *,
        run_id: str,
        evidence: EvidenceStrength | None = None,
    ) -> None:
        self.record_exact_run(variant.exact_run_hash, run_id=run_id)
        self.successful_variant_ids.add(variant.variant_id)
        self.successful_capability_hashes.setdefault(
            variant.capability_hash,
            set(),
        ).add(variant.variant_id)
        if evidence is not None:
            self.evidence_by_variant_id[variant.variant_id] = evidence

    def record_exact_run(self, exact_run_hash: str, *, run_id: str) -> None:
        self.exact_run_hashes.add(exact_run_hash)
        self.run_ids_by_exact_hash.setdefault(exact_run_hash, set()).add(run_id)

    def record_pending(self, variant: PlannedVariant, *, run_id: str) -> None:
        self.record_pending_exact(variant.exact_run_hash, run_id=run_id)
        self.record_pending_capability(variant)

    def record_pending_exact(self, exact_run_hash: str, *, run_id: str) -> None:
        self.pending_exact_run_hashes.add(exact_run_hash)
        self.pending_run_ids_by_exact_hash.setdefault(
            exact_run_hash,
            set(),
        ).add(run_id)

    def record_pending_capability(self, variant: PlannedVariant) -> None:
        self.pending_capability_hashes.setdefault(
            variant.capability_hash,
            set(),
        ).add(variant.variant_id)

    def pending_run_ids(self, exact_run_hash: str) -> tuple[str, ...]:
        return tuple(
            sorted(self.pending_run_ids_by_exact_hash.get(exact_run_hash, set()))
        )

    def record_failure(
        self,
        variant: str | PlannedVariant,
        *,
        signature: str,
        run_id: str,
    ) -> None:
        normalized = signature.strip()
        if not normalized:
            raise ValueError("failure signature must be non-empty")
        self.failure_signatures[normalized] += 1
        if isinstance(variant, PlannedVariant):
            self.record_exact_run(variant.exact_run_hash, run_id=run_id)
            self.failure_capabilities.setdefault(normalized, set()).add(
                variant.capability_hash
            )
            self.failed_variants.setdefault(normalized, {})[
                (
                    variant.variant_id,
                    variant.exact_run_hash,
                    variant.capability_hash,
                )
            ] = variant

    def add_equivalent_group(self, variant_ids: Iterable[str]) -> None:
        group = frozenset(item.strip() for item in variant_ids if item.strip())
        if len(group) < 2:
            raise ValueError("an equivalence group requires at least two variants")
        if group not in self.equivalent_groups:
            self.equivalent_groups.append(group)

    def original_run_ids(self, exact_run_hash: str) -> tuple[str, ...]:
        return tuple(sorted(self.run_ids_by_exact_hash.get(exact_run_hash, set())))


def request_rerun(
    variant: PlannedVariant,
    *,
    reason: str,
    original_run_ids: Iterable[str] = (),
) -> RerunRequest:
    normalized = reason.strip() if isinstance(reason, str) else ""
    if not normalized:
        raise RerunReasonRequired("exact rerun requires a non-empty reason")
    return RerunRequest(
        variant_id=variant.variant_id,
        reason=normalized,
        original_exact_run_hash=variant.exact_run_hash,
        original_run_ids=tuple(sorted(set(original_run_ids))),
    )


def next_wave(plan: ExperimentPlan, history: ExperimentHistory) -> WaveDecision:
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("next_wave requires a validated ExperimentPlan")
    if not isinstance(history, ExperimentHistory):
        raise TypeError("next_wave requires ExperimentHistory")

    scores = {
        variant.variant_id: _evidence_strength(plan, variant)
        for variant in plan.variants
    }
    reasons: dict[str, str] = {}
    eligible: list[PlannedVariant] = []
    pending: list[PlannedVariant] = []
    represented_pending_hashes: set[str] = set()

    for variant in plan.variants:
        if (
            variant.hypothesis is not None
            and variant.hypothesis not in plan.capability_coverage
        ):
            reasons[variant.variant_id] = "hypothesis_dimension_not_approved"
            continue
        if _has_unapproved_correction_dimensions(
            variant,
            set(plan.capability_coverage),
        ):
            reasons[variant.variant_id] = "correction_dimension_not_approved"
            continue
        if variant.exact_run_hash in history.pending_exact_run_hashes:
            reasons[variant.variant_id] = "exact_run_pending"
            if variant.exact_run_hash not in represented_pending_hashes:
                pending.append(variant)
                represented_pending_hashes.add(variant.exact_run_hash)
            continue
        if variant.capability_hash in history.pending_capability_hashes:
            reasons[variant.variant_id] = "capability_run_pending"
            continue
        if variant.exact_run_hash in history.exact_run_hashes:
            reasons[variant.variant_id] = "exact_run_already_recorded"
            continue
        repeated_signature = _uncorrected_repeated_failure(variant, history)
        if repeated_signature is not None:
            reasons[variant.variant_id] = (
                "repeated_failure_requires_new_corrective_hypothesis"
            )
            continue
        if _dominated_by_equivalent_tested_variant(
            variant,
            history,
            scores,
        ):
            reasons[variant.variant_id] = (
                "dominated_by_equivalent_tested_capability"
            )
            continue
        if _dominated_by_tested_capability(variant, history, scores):
            reasons[variant.variant_id] = "dominated_by_tested_capability"
            continue
        eligible.append(variant)

    selected: list[PlannedVariant] = []
    grouped: dict[str, list[PlannedVariant]] = {}
    for variant in eligible:
        grouped.setdefault(variant.capability_hash, []).append(variant)
    for capability_hash in sorted(grouped):
        candidates = grouped[capability_hash]
        representative = min(
            candidates,
            key=lambda item: (
                tuple(-value for value in _score_tuple(scores[item.variant_id])),
                item.variant_id,
                item.exact_run_hash,
            ),
        )
        selected.append(representative)
        for candidate in candidates:
            if candidate.variant_id != representative.variant_id:
                reasons[candidate.variant_id] = (
                    "weaker_representative_of_novel_capability"
                )

    selected_tuple = tuple(selected)
    skipped = tuple(
        variant.variant_id
        for variant in plan.variants
        if variant.variant_id not in {item.variant_id for item in selected_tuple}
    )
    return WaveDecision(
        variants=selected_tuple,
        considered_variant_ids=tuple(item.variant_id for item in plan.variants),
        skipped_variant_ids=skipped,
        pending_variant_ids=tuple(item.variant_id for item in pending),
        reasons=MappingProxyType(dict(reasons)),
        evidence=MappingProxyType(
            {
                variant.variant_id: MappingProxyType(
                    scores[variant.variant_id].as_dict()
                )
                for variant in plan.variants
            }
        ),
        saturation_reason=(
            SATURATION_REASON if not selected_tuple and not pending else None
        ),
    )


def has_unapproved_correction_dimensions(plan: ExperimentPlan) -> bool:
    approved = set(plan.capability_coverage)
    return any(
        _has_unapproved_correction_dimensions(variant, approved)
        for variant in plan.variants
    )


def _has_unapproved_correction_dimensions(
    variant: PlannedVariant,
    approved: set[DimensionName],
) -> bool:
    parameters = thaw_json(variant.execution_parameters)
    raw = parameters.get("correction")
    if not isinstance(raw, Mapping):
        return False
    changed_dimensions = raw.get("changed_dimensions")
    if not isinstance(changed_dimensions, (list, tuple)):
        return False
    for item in changed_dimensions:
        try:
            dimension = DimensionName(item)
        except (TypeError, ValueError):
            return True
        if dimension not in approved:
            return True
    return False


def _evidence_strength(
    plan: ExperimentPlan,
    variant: PlannedVariant,
) -> EvidenceStrength:
    candidate = _candidate_metadata(plan).get(variant.variant_id, {})
    raw_confidence = candidate.get("confidence")
    try:
        confidence = PlanningConfidence(raw_confidence)
    except (TypeError, ValueError):
        confidence = variant.confidence or PlanningConfidence.OBSERVED
    evidence_ids = candidate.get("evidence_ids")
    evidence_count = (
        len({str(item) for item in evidence_ids})
        if isinstance(evidence_ids, (list, tuple))
        else len(set(variant.evidence_ids))
    )
    return EvidenceStrength(
        confidence=_CONFIDENCE_STRENGTH[confidence],
        provenance_rank=_nonnegative_int(candidate.get("provenance_rank")),
        evidence_count=evidence_count,
        engagement=_nonnegative_int(candidate.get("engagement")),
        structural_cleanliness=-len(variant.warnings),
    )


def _candidate_metadata(plan: ExperimentPlan) -> dict[str, Mapping[str, Any]]:
    payload = thaw_json(plan.approval_payload)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id not in result:
            result[candidate_id] = candidate
    return result


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _score_tuple(score: EvidenceStrength) -> tuple[int, ...]:
    return (
        score.confidence,
        score.provenance_rank,
        score.evidence_count,
        score.engagement,
        score.structural_cleanliness,
    )


def _score_for_history(
    variant_id: str,
    scores: Mapping[str, EvidenceStrength],
    history: ExperimentHistory,
) -> EvidenceStrength | None:
    return scores.get(variant_id) or history.evidence_by_variant_id.get(variant_id)


def _dominated_by_tested_capability(
    variant: PlannedVariant,
    history: ExperimentHistory,
    scores: Mapping[str, EvidenceStrength],
) -> bool:
    tested = history.successful_capability_hashes.get(
        variant.capability_hash,
        set(),
    )
    candidate_score = scores[variant.variant_id]
    return any(
        tested_score is not None and tested_score >= candidate_score
        for tested_id in tested
        if (tested_score := _score_for_history(tested_id, scores, history))
        is not None
    )


def _dominated_by_equivalent_tested_variant(
    variant: PlannedVariant,
    history: ExperimentHistory,
    scores: Mapping[str, EvidenceStrength],
) -> bool:
    candidate_score = scores[variant.variant_id]
    tested_capability = history.successful_capability_hashes.get(
        variant.capability_hash,
        set(),
    )
    for group in history.equivalent_groups:
        if variant.variant_id not in group:
            continue
        for tested_id in group & tested_capability:
            tested_score = _score_for_history(tested_id, scores, history)
            if tested_score is not None and tested_score >= candidate_score:
                return True
    return False


def _uncorrected_repeated_failure(
    variant: PlannedVariant,
    history: ExperimentHistory,
) -> str | None:
    parameters = thaw_json(variant.execution_parameters)
    correction_claimed = (
        "correction" in parameters or "corrective_hypotheses" in parameters
    )
    correction = _correction_record(parameters.get("correction"))
    if correction_claimed and (
        correction is None
        or history.failure_signatures[correction.failure_signature]
        < REPEATED_FAILURE_THRESHOLD
        or not _correction_addresses_failure(
            variant,
            correction,
            history,
        )
    ):
        return (
            correction.failure_signature
            if correction is not None
            else "invalid-correction-record"
        )
    required_nodes = {
        node.get("class_type")
        for node in thaw_json(variant.materialized_graph).values()
        if isinstance(node, Mapping) and isinstance(node.get("class_type"), str)
    }
    for signature, count in history.failure_signatures.items():
        if count < REPEATED_FAILURE_THRESHOLD:
            continue
        if (
            correction is not None
            and correction.failure_signature == signature
            and _correction_addresses_failure(variant, correction, history)
        ):
            continue
        if signature.startswith("missing-node:"):
            missing_node = signature.partition(":")[2]
            if (
                missing_node in required_nodes
                or variant.capability_hash
                in history.failure_capabilities.get(signature, set())
            ):
                return signature
            continue
        if variant.capability_hash in history.failure_capabilities.get(
            signature,
            set(),
        ):
            return signature
    return None


def _correction_record(raw: Any) -> CorrectionRecord | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return CorrectionRecord.model_validate(dict(raw))
    except (TypeError, ValueError, ValidationError):
        return None


def _correction_addresses_failure(
    variant: PlannedVariant,
    correction: CorrectionRecord,
    history: ExperimentHistory,
) -> bool:
    if correction.candidate_variant_id != variant.variant_id:
        return False
    if variant.hypothesis not in correction.changed_dimensions:
        return False
    if (
        variant.exact_run_hash == correction.failed_exact_run_hash
        or variant.capability_hash == correction.failed_capability_hash
    ):
        return False
    failed_key = (
        correction.failed_variant_id,
        correction.failed_exact_run_hash,
        correction.failed_capability_hash,
    )
    if failed_key not in history.failed_variants.get(
        correction.failure_signature,
        {},
    ):
        return False
    if correction.failure_signature.startswith("missing-node:"):
        missing_node = correction.failure_signature.partition(":")[2]
        required_nodes = {
            node.get("class_type")
            for node in thaw_json(variant.materialized_graph).values()
            if isinstance(node, Mapping)
            and isinstance(node.get("class_type"), str)
        }
        if missing_node in required_nodes:
            return False
    return True
