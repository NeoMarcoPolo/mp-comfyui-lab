from __future__ import annotations

from datetime import UTC
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    AfterValidator,
    AwareDatetime,
    ConfigDict,
    Field,
    StringConstraints,
)

from mp_comfyui_lab.json_types import FrozenDict, SanitizedJsonObject


NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
UtcDatetime = Annotated[
    AwareDatetime,
    AfterValidator(lambda value: value.astimezone(UTC)),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class ContentClassification(StrEnum):
    SFW = "SFW"
    NSFW = "NSFW"
    UNKNOWN = "UNKNOWN"


class Engagement(_FrozenModel):
    likes: int | None = Field(default=None, ge=0)
    uses: int | None = Field(default=None, ge=0)
    collects: int | None = Field(default=None, ge=0)
    downloads: int | None = Field(default=None, ge=0)
    views: int | None = Field(default=None, ge=0)
    comments: int | None = Field(default=None, ge=0)
    remixes: int | None = Field(default=None, ge=0)


class CoverAsset(_FrozenModel):
    url: NonBlankStr
    thumbnail_url: NonBlankStr | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class WorkflowCandidate(_FrozenModel):
    executor: NonBlankStr
    workflow_id: NonBlankStr
    canonical_url: NonBlankStr
    name: NonBlankStr
    description: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    published_at: UtcDatetime | None = None
    tags: tuple[str, ...] = ()
    engagement: Engagement = Field(default_factory=Engagement)
    covers: tuple[CoverAsset, ...] = ()
    content_classification: ContentClassification = ContentClassification.UNKNOWN
    classification_reasons: tuple[str, ...] = ()
    provider_metadata: SanitizedJsonObject = Field(default_factory=FrozenDict)


class WorkflowDetail(_FrozenModel):
    candidate: WorkflowCandidate
    node_count: int | None = Field(default=None, ge=0)
    primitive_nodes: tuple[str, ...] = ()
    custom_nodes: tuple[str, ...] = ()
    used_models: tuple[str, ...] = ()
    preview: CoverAsset | None = None
    webapp_id: str | None = None
