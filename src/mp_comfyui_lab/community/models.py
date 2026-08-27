"""Strict data contracts for community workflow evidence."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class WorkflowSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    guild_id: str
    channel_id: str
    message_id: str
    message_url: str
    author_id: str
    timestamp: datetime
    attachment_id: str
    attachment_filename: str
    attachment_sha256: str

    @property
    def source_key(self) -> str:
        return f"{self.channel_id}/{self.message_id}/{self.attachment_id}"


class WorkflowIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    family: str
    version: str
    task: str
    technique: str
    capability_hash: str
    core_models: tuple[str, ...]
    loras: tuple[str, ...]
    sampler_signature: tuple[str, ...]

    @property
    def key(self) -> str:
        parts = (
            self.family,
            self.version,
            self.task,
            self.technique,
            self.capability_hash,
            "|".join(self.core_models),
            "|".join(self.loras),
            "|".join(self.sampler_signature),
        )
        return "::".join(parts)


class WorkflowRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    workflow_id: str
    identity: WorkflowIdentity
    sources: tuple[WorkflowSource, ...]
    api_graph_paths: tuple[str, ...]
    ui_graph_paths: tuple[str, ...]
    source_graph_hashes: tuple[str, ...]
    image_hashes: tuple[str, ...]
    identity_review_required: bool = False
