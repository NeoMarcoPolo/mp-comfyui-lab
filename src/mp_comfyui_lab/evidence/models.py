from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from mp_comfyui_lab.discovery.models import (
    ContentClassification,
    Engagement,
    WorkflowDetail,
)
from mp_comfyui_lab.executors.base import CapabilityReport
from mp_comfyui_lab.graph.models import NormalizedGraph
from mp_comfyui_lab.security import sanitize_for_persistence, sanitize_untrusted_diagnostic


EVIDENCE_SCHEMA_VERSION = "1"
MAX_EVIDENCE_TEXT_LENGTH = 512
MAX_EVIDENCE_ITEMS = 1_024
MAX_EVIDENCE_EXPLANATION_LENGTH = MAX_EVIDENCE_TEXT_LENGTH * MAX_EVIDENCE_ITEMS


def _sanitize_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return sanitize_untrusted_diagnostic(value)


def _sanitize_url(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    sanitized = sanitize_for_persistence(value)
    return sanitize_untrusted_diagnostic(sanitized)


BoundedText = Annotated[
    str,
    BeforeValidator(_sanitize_text),
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_EVIDENCE_TEXT_LENGTH,
    ),
]
BoundedExplanation = Annotated[
    str,
    BeforeValidator(_sanitize_text),
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_EVIDENCE_EXPLANATION_LENGTH,
    ),
]
SanitizedUrl = Annotated[
    str,
    BeforeValidator(_sanitize_url),
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=2_048,
    ),
]
HashValue = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ConfidenceLabel(StrEnum):
    OBSERVED = "observed"
    STRONGLY_INFERRED = "strongly_inferred"
    WEAKLY_INFERRED = "weakly_inferred"
    UNEXPLORED = "unexplored"


class EvidenceSourceKind(StrEnum):
    CURRENT_DISCOVERY_WORKFLOW = "current_discovery_workflow"
    CURRENT_WORKFLOW_DETAIL = "current_workflow_detail"
    CURRENT_NORMALIZED_GRAPH = "current_normalized_graph"
    PRIOR_EVIDENCE_GRAPH = "prior_evidence_graph"
    PRIOR_FINALISTS = "prior_finalists"
    PRIOR_HANDOFF = "prior_handoff"


CURRENT_SOURCE_KINDS = frozenset(
    {
        EvidenceSourceKind.CURRENT_DISCOVERY_WORKFLOW,
        EvidenceSourceKind.CURRENT_WORKFLOW_DETAIL,
        EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH,
    }
)


class EvidenceDimension(StrEnum):
    CHECKPOINT = "checkpoint"
    MODEL_FAMILY = "model_family"
    LORAS = "loras"
    LORA_STRENGTHS = "lora_strengths"
    LOADER_TOPOLOGY = "loader_topology"
    ENCODER_TOPOLOGY = "encoder_topology"
    SAMPLER_TOPOLOGY = "sampler_topology"
    SCHEDULER_TOPOLOGY = "scheduler_topology"
    VAE_TOPOLOGY = "vae_topology"
    PROMPT_GRAMMAR = "prompt_grammar"
    REFERENCE_COUNT = "reference_count"
    REFERENCE_ROUTING = "reference_routing"
    FANOUT = "fanout"
    OUTPUT_NODE_COUNT = "output_node_count"
    OUTPUT_NODE_TYPES = "output_node_types"


class LoraStrengthRange(_FrozenModel):
    identity: BoundedText
    minimum: float | None = None
    maximum: float | None = None
    buckets: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_range(self) -> LoraStrengthRange:
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("LoRA strength range requires both minimum and maximum")
        if self.minimum is not None:
            if not isfinite(self.minimum) or not isfinite(self.maximum):
                raise ValueError("LoRA strength range must be finite")
            if self.minimum > self.maximum:
                raise ValueError("LoRA strength minimum cannot exceed maximum")
        if len(self.buckets) != len(set(self.buckets)):
            raise ValueError("LoRA strength buckets must be unique")
        return self


class VisibleOutputEvidence(_FrozenModel):
    cover_urls: tuple[SanitizedUrl, ...] = Field(default=(), max_length=128)
    preview_url: SanitizedUrl | None = None
    output_role_node_ids: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=128,
    )

    @field_validator("cover_urls", "output_role_node_ids")
    @classmethod
    def sort_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(values)))

    @property
    def has_visible_output(self) -> bool:
        return bool(self.cover_urls or self.preview_url) and bool(
            self.output_role_node_ids
        )


class EvidenceSource(_FrozenModel):
    source_id: BoundedText
    kind: EvidenceSourceKind
    canonical_url: SanitizedUrl | None = None
    artifact_path: BoundedText | None = None
    source_hash: HashValue | None = None
    capability_hash: HashValue | None = None
    content_sha256: HashValue | None = None

    @field_validator("canonical_url")
    @classmethod
    def validate_canonical_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("canonical_url must be an absolute HTTP(S) URL")
        return value

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError("artifact_path must be a safe relative path")
        return str(path)

    @model_validator(mode="after")
    def validate_location(self) -> EvidenceSource:
        if self.kind in CURRENT_SOURCE_KINDS and self.canonical_url is None:
            raise ValueError("current evidence source requires canonical_url")
        if self.kind not in CURRENT_SOURCE_KINDS:
            if self.artifact_path is None or self.content_sha256 is None:
                raise ValueError(
                    "prior evidence source requires artifact_path and content_sha256"
                )
        return self


class EvidenceWarning(_FrozenModel):
    code: BoundedText
    reason: BoundedExplanation
    path: BoundedText | None = None


class _ComparableDimensions(_FrozenModel):
    checkpoint: BoundedText | None = None
    model_family: BoundedText | None = None
    loras: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=128,
    )
    lora_strengths: tuple[LoraStrengthRange, ...] | None = Field(
        default=None,
        max_length=128,
    )
    loader_topology: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    encoder_topology: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    sampler_topology: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    scheduler_topology: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    vae_topology: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    prompt_grammar: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    reference_count: int | None = Field(default=None, ge=0, le=MAX_EVIDENCE_ITEMS)
    reference_routing: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    fanout: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )
    output_node_count: int | None = Field(default=None, ge=0, le=MAX_EVIDENCE_ITEMS)
    output_node_types: tuple[BoundedText, ...] | None = Field(
        default=None,
        max_length=MAX_EVIDENCE_ITEMS,
    )


class EvidenceObservation(_ComparableDimensions):
    source_id: BoundedText
    loras: tuple[BoundedText, ...] = Field(default=(), max_length=128)
    lora_strengths: tuple[LoraStrengthRange, ...] = Field(
        default=(),
        max_length=128,
    )
    loader_topology: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    encoder_topology: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    sampler_topology: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    scheduler_topology: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    vae_topology: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    prompt_grammar: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    reference_count: int = Field(default=0, ge=0, le=MAX_EVIDENCE_ITEMS)
    reference_routing: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    fanout: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    output_node_count: int = Field(default=0, ge=0, le=MAX_EVIDENCE_ITEMS)
    output_node_types: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    visible_output: VisibleOutputEvidence = Field(
        default_factory=VisibleOutputEvidence
    )
    engagement: Engagement = Field(default_factory=Engagement)
    content_classification: ContentClassification = ContentClassification.UNKNOWN

    @model_validator(mode="after")
    def validate_lora_strength_alignment(self) -> EvidenceObservation:
        if len(self.loras) != len(self.lora_strengths):
            raise ValueError("every ordered LoRA requires one strength range record")
        if tuple(item.identity for item in self.lora_strengths) != self.loras:
            raise ValueError("LoRA strength records must preserve ordered LoRA identities")
        if self.reference_count != len(self.reference_routing):
            raise ValueError("reference_count must match reference_routing length")
        if self.output_node_count != len(self.output_node_types):
            raise ValueError("output_node_count must match output_node_types length")
        return self


class MutationClaim(_ComparableDimensions):
    changed_dimensions: tuple[EvidenceDimension, ...] = Field(
        default=(),
        max_length=len(EvidenceDimension),
    )

    @model_validator(mode="after")
    def validate_declared_dimensions(self) -> MutationClaim:
        if len(self.changed_dimensions) != len(set(self.changed_dimensions)):
            raise ValueError("changed_dimensions cannot contain duplicate dimensions")
        for dimension in self.changed_dimensions:
            if getattr(self, dimension.value) is None:
                raise ValueError(
                    f"declared dimension '{dimension.value}' must have a claim value"
                )
        expected = tuple(
            dimension
            for dimension in EvidenceDimension
            if dimension in self.changed_dimensions
        )
        if self.changed_dimensions != expected:
            raise ValueError("changed_dimensions must follow canonical dimension order")
        return self


class ClaimAssessment(_FrozenModel):
    claim: MutationClaim
    label: ConfidenceLabel
    supporting_source_ids: tuple[BoundedText, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    matched_dimensions: tuple[EvidenceDimension, ...] = Field(
        default=(),
        max_length=len(EvidenceDimension),
    )
    differing_dimensions: tuple[EvidenceDimension, ...] = Field(
        default=(),
        max_length=len(EvidenceDimension),
    )
    explanation: BoundedExplanation
    warnings: tuple[BoundedExplanation, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )

    @model_validator(mode="after")
    def validate_deterministic_order(self) -> ClaimAssessment:
        if self.supporting_source_ids != tuple(sorted(set(self.supporting_source_ids))):
            raise ValueError("supporting_source_ids must be unique and sorted")
        for field_name in ("matched_dimensions", "differing_dimensions"):
            values = getattr(self, field_name)
            expected = tuple(dimension for dimension in EvidenceDimension if dimension in values)
            if values != expected:
                raise ValueError(f"{field_name} must be unique and canonically ordered")
        return self


class EvidenceGraph(_FrozenModel):
    schema_version: Literal["1"] = EVIDENCE_SCHEMA_VERSION
    sources: tuple[EvidenceSource, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    observations: tuple[EvidenceObservation, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )
    warnings: tuple[EvidenceWarning, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )

    @field_validator("sources")
    @classmethod
    def sort_and_deduplicate_sources(
        cls,
        values: tuple[EvidenceSource, ...],
    ) -> tuple[EvidenceSource, ...]:
        ordered = sorted(
            values,
            key=lambda item: (
                item.source_id,
                item.kind.value,
                item.artifact_path or "",
                item.canonical_url or "",
            ),
        )
        unique: list[EvidenceSource] = []
        for item in ordered:
            if unique and unique[-1].source_id == item.source_id:
                if unique[-1] == item:
                    continue
                raise ValueError(f"duplicate source_id '{item.source_id}' has conflicts")
            unique.append(item)
        return tuple(unique)

    @field_validator("observations")
    @classmethod
    def sort_and_deduplicate_observations(
        cls,
        values: tuple[EvidenceObservation, ...],
    ) -> tuple[EvidenceObservation, ...]:
        ordered = sorted(values, key=lambda item: item.source_id)
        unique: list[EvidenceObservation] = []
        for item in ordered:
            if unique and unique[-1].source_id == item.source_id:
                if unique[-1] == item:
                    continue
                raise ValueError(
                    f"duplicate observation source_id '{item.source_id}' has conflicts"
                )
            unique.append(item)
        return tuple(unique)

    @field_validator("warnings")
    @classmethod
    def sort_and_deduplicate_warnings(
        cls,
        values: tuple[EvidenceWarning, ...],
    ) -> tuple[EvidenceWarning, ...]:
        return tuple(
            sorted(
                set(values),
                key=lambda item: (item.path or "", item.code, item.reason),
            )
        )

    @model_validator(mode="after")
    def validate_observation_sources(self) -> EvidenceGraph:
        source_ids = {source.source_id for source in self.sources}
        missing = sorted(
            observation.source_id
            for observation in self.observations
            if observation.source_id not in source_ids
        )
        if missing:
            raise ValueError(f"observations reference missing sources: {missing}")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class CurrentEvidenceInput(_FrozenModel):
    source_id: BoundedText
    source_kind: EvidenceSourceKind = EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH
    detail: WorkflowDetail
    normalized_graph: NormalizedGraph
    capability_report: CapabilityReport | None = None
    source_hash: HashValue | None = None
    capability_hash: HashValue | None = None

    @field_validator("source_kind")
    @classmethod
    def require_current_kind(cls, value: EvidenceSourceKind) -> EvidenceSourceKind:
        if value not in CURRENT_SOURCE_KINDS:
            raise ValueError("current evidence input requires a current source kind")
        return value

    @model_validator(mode="after")
    def validate_source_id(self) -> CurrentEvidenceInput:
        if self.source_id != self.detail.candidate.workflow_id:
            raise ValueError("source_id must match detail candidate workflow_id")
        return self


class PriorFinalistsArtifact(_FrozenModel):
    schema_version: Literal["1"] = EVIDENCE_SCHEMA_VERSION
    finalists: tuple[EvidenceObservation, ...] = Field(
        default=(),
        max_length=MAX_EVIDENCE_ITEMS,
    )


class PriorHandoffArtifact(_FrozenModel):
    schema_version: Literal["1"] = EVIDENCE_SCHEMA_VERSION
    selected: EvidenceObservation
