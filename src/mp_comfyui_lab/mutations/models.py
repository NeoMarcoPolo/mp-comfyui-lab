from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from mp_comfyui_lab.evidence.models import ConfidenceLabel
from mp_comfyui_lab.graph.models import (
    ExactNodeId,
    FrozenJsonValue,
    SemanticRole,
    validate_serializable_json,
)
from mp_comfyui_lab.json_types import FrozenDict, FrozenJsonObject


MAX_MUTATIONS = 128
MAX_LORAS = 64
MAX_DONOR_NODES = 512
MAX_BOUNDARY_MAPPINGS = 1_024
MAX_MUTATION_TEXT_LENGTH = 2_048


BoundedText = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_MUTATION_TEXT_LENGTH,
    ),
]
MutationOp = Literal[
    "set_role_input",
    "replace_lora_stack",
    "replace_subgraph",
    "set_node_input",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _sorted_unique_text(values: Any) -> Any:
    values = _as_tuple(values)
    if isinstance(values, tuple):
        return tuple(sorted(set(values)))
    return values


def _validate_json_value(value: Any, path: str) -> Any:
    return validate_serializable_json(value, path)


class LoraSpec(_FrozenModel):
    name: BoundedText
    strength_model: float = Field(ge=-20.0, le=20.0)
    strength_clip: float | None = Field(default=None, ge=-20.0, le=20.0)


class DonorNode(_FrozenModel):
    node_id: ExactNodeId
    class_type: BoundedText
    inputs: FrozenJsonObject
    metadata: FrozenJsonObject = Field(default_factory=FrozenDict)

    @field_validator("inputs", "metadata", mode="before")
    @classmethod
    def validate_json_objects(cls, value: Any) -> Any:
        return _validate_json_value(value, "donor node JSON")

    @field_validator("metadata")
    @classmethod
    def reject_structural_metadata(
        cls,
        metadata: FrozenJsonObject,
    ) -> FrozenJsonObject:
        reserved = {"class_type", "inputs"}.intersection(metadata)
        if reserved:
            raise ValueError("donor metadata cannot redefine class_type or inputs")
        return metadata


class LoraLoaderTemplate(_FrozenModel):
    donor_node_id: ExactNodeId
    class_type: BoundedText
    inputs: FrozenJsonObject
    donor_workflow_id: BoundedText
    evidence_ids: tuple[BoundedText, ...] = Field(min_length=1, max_length=128)
    metadata: FrozenJsonObject = Field(default_factory=FrozenDict)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        return _sorted_unique_text(value)

    @field_validator("inputs", "metadata", mode="before")
    @classmethod
    def validate_json_objects(cls, value: Any) -> Any:
        return _validate_json_value(value, "LoRA loader template JSON")

    @field_validator("metadata")
    @classmethod
    def reject_structural_metadata(
        cls,
        metadata: FrozenJsonObject,
    ) -> FrozenJsonObject:
        reserved = {"class_type", "inputs"}.intersection(metadata)
        if reserved:
            raise ValueError("template metadata cannot redefine class_type or inputs")
        return metadata


class BoundaryInputMapping(_FrozenModel):
    donor_node_id: ExactNodeId
    donor_field_name: BoundedText
    source_node_id: ExactNodeId
    source_output_index: int = Field(ge=0)


class BoundaryOutputMapping(_FrozenModel):
    donor_node_id: ExactNodeId
    donor_output_index: int = Field(ge=0)
    consumer_node_id: ExactNodeId
    consumer_field_name: BoundedText


class SetRoleInput(_FrozenModel):
    op: Literal["set_role_input"] = "set_role_input"
    role: SemanticRole
    binding_index: int = Field(ge=0)
    field_name: BoundedText
    value: FrozenJsonValue

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        return _validate_json_value(value, "SetRoleInput value")


class ReplaceLoraStack(_FrozenModel):
    op: Literal["replace_lora_stack"] = "replace_lora_stack"
    loras: tuple[LoraSpec, ...] = Field(max_length=MAX_LORAS)
    evidence_ids: tuple[BoundedText, ...] = Field(min_length=1, max_length=128)
    confidence: ConfidenceLabel = ConfidenceLabel.OBSERVED
    loader_template: LoraLoaderTemplate | None = None

    @field_validator("loras", mode="before")
    @classmethod
    def normalize_loras(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        return _sorted_unique_text(value)

    @model_validator(mode="after")
    def require_unique_lora_names(self) -> ReplaceLoraStack:
        names = [lora.name for lora in self.loras]
        if len(names) != len(set(names)):
            raise ValueError("replacement LoRA names must be unique")
        return self


class ReplaceSubgraph(_FrozenModel):
    op: Literal["replace_subgraph"] = "replace_subgraph"
    target_node_ids: tuple[ExactNodeId, ...] = Field(min_length=1, max_length=512)
    donor_nodes: tuple[DonorNode, ...] = Field(
        min_length=1,
        max_length=MAX_DONOR_NODES,
    )
    boundary_inputs: tuple[BoundaryInputMapping, ...] = Field(
        default=(),
        max_length=MAX_BOUNDARY_MAPPINGS,
    )
    boundary_outputs: tuple[BoundaryOutputMapping, ...] = Field(
        min_length=1,
        max_length=MAX_BOUNDARY_MAPPINGS,
    )
    donor_workflow_id: BoundedText
    evidence_ids: tuple[BoundedText, ...] = Field(min_length=1, max_length=128)
    confidence: ConfidenceLabel = ConfidenceLabel.OBSERVED

    @field_validator(
        "target_node_ids",
        "donor_nodes",
        "boundary_inputs",
        "boundary_outputs",
        mode="before",
    )
    @classmethod
    def normalize_sequences(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("target_node_ids")
    @classmethod
    def sort_target_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("target_node_ids must be unique")
        return tuple(sorted(values))

    @field_validator("donor_nodes")
    @classmethod
    def sort_donor_nodes(cls, values: tuple[DonorNode, ...]) -> tuple[DonorNode, ...]:
        node_ids = [node.node_id for node in values]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("donor node IDs must be unique")
        return tuple(sorted(values, key=lambda node: node.node_id))

    @field_validator("boundary_inputs")
    @classmethod
    def sort_boundary_inputs(
        cls,
        values: tuple[BoundaryInputMapping, ...],
    ) -> tuple[BoundaryInputMapping, ...]:
        keys = [(item.donor_node_id, item.donor_field_name) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("boundary input donor fields must be unique")
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    item.donor_node_id,
                    item.donor_field_name,
                    item.source_node_id,
                    item.source_output_index,
                ),
            )
        )

    @field_validator("boundary_outputs")
    @classmethod
    def sort_boundary_outputs(
        cls,
        values: tuple[BoundaryOutputMapping, ...],
    ) -> tuple[BoundaryOutputMapping, ...]:
        keys = [(item.consumer_node_id, item.consumer_field_name) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("boundary output consumer fields must be unique")
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    item.consumer_node_id,
                    item.consumer_field_name,
                    item.donor_node_id,
                    item.donor_output_index,
                ),
            )
        )

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> Any:
        return _sorted_unique_text(value)


class SetNodeInput(_FrozenModel):
    op: Literal["set_node_input"] = "set_node_input"
    node_id: ExactNodeId
    field_name: BoundedText
    value: FrozenJsonValue

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        return _validate_json_value(value, "SetNodeInput value")


Mutation = Annotated[
    SetRoleInput | ReplaceLoraStack | ReplaceSubgraph | SetNodeInput,
    Field(discriminator="op"),
]


class SemanticPatch(_FrozenModel):
    mutations: tuple[Mutation, ...] = Field(min_length=1, max_length=MAX_MUTATIONS)

    @field_validator("mutations", mode="before")
    @classmethod
    def normalize_mutations(cls, value: Any) -> Any:
        return _as_tuple(value)

    @model_validator(mode="after")
    def reject_declared_conflicts(self) -> SemanticPatch:
        payloads: set[str] = set()
        role_targets: set[tuple[SemanticRole, int]] = set()
        node_targets: set[tuple[str, str]] = set()
        subgraph_targets: set[str] = set()
        has_lora_replacement = False

        for mutation in self.mutations:
            payload = json.dumps(
                mutation.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if payload in payloads:
                raise ValueError("semantic patch cannot contain duplicate mutations")
            payloads.add(payload)

            if isinstance(mutation, SetRoleInput):
                target = (mutation.role, mutation.binding_index)
                if target in role_targets:
                    raise ValueError("semantic patch has conflicting role mutations")
                role_targets.add(target)
            elif isinstance(mutation, SetNodeInput):
                target = (mutation.node_id, mutation.field_name)
                if target in node_targets:
                    raise ValueError("semantic patch has conflicting node input mutations")
                node_targets.add(target)
            elif isinstance(mutation, ReplaceLoraStack):
                if has_lora_replacement:
                    raise ValueError("semantic patch can replace the LoRA stack only once")
                has_lora_replacement = True
            elif isinstance(mutation, ReplaceSubgraph):
                overlap = subgraph_targets.intersection(mutation.target_node_ids)
                if overlap:
                    raise ValueError("replacement subgraphs cannot target the same nodes")
                subgraph_targets.update(mutation.target_node_ids)
        return self


class FieldDiff(_FrozenModel):
    path: tuple[BoundedText, ...]
    before_present: bool
    after_present: bool
    before: FrozenJsonValue | None = None
    after: FrozenJsonValue | None = None


class NodeDiff(_FrozenModel):
    node_id: ExactNodeId
    change: Literal["added", "removed", "modified"]
    fields: tuple[FieldDiff, ...] = Field(min_length=1, max_length=10_000)


class MutationDiff(_FrozenModel):
    mutation_index: int = Field(ge=0, lt=MAX_MUTATIONS)
    op: MutationOp
    evidence_ids: tuple[BoundedText, ...] = Field(default=(), max_length=128)
    confidence: ConfidenceLabel | None = None
    donor_workflow_id: BoundedText | None = None


class CanonicalDiff(_FrozenModel):
    nodes: tuple[NodeDiff, ...] = Field(default=(), max_length=4_096)
    mutations: tuple[MutationDiff, ...] = Field(
        min_length=1,
        max_length=MAX_MUTATIONS,
    )


class ValidationIssue(_FrozenModel):
    code: BoundedText
    message: BoundedText
    mutation_index: int | None = Field(default=None, ge=0, lt=MAX_MUTATIONS)


class ValidationReport(_FrozenModel):
    errors: tuple[ValidationIssue, ...] = Field(default=(), max_length=MAX_MUTATIONS)
    warnings: tuple[ValidationIssue, ...] = Field(default=(), max_length=MAX_MUTATIONS)

    @field_validator("errors", "warnings")
    @classmethod
    def canonicalize_issues(
        cls,
        issues: tuple[ValidationIssue, ...],
    ) -> tuple[ValidationIssue, ...]:
        unique = {
            (issue.code, issue.message, issue.mutation_index): issue for issue in issues
        }
        return tuple(unique[key] for key in sorted(unique, key=lambda key: (key[0], key[2] if key[2] is not None else -1, key[1])))

    @property
    def is_valid(self) -> bool:
        return not self.errors


class MaterializationResult(_FrozenModel):
    graph: FrozenJsonObject
    diff: CanonicalDiff
    validation: ValidationReport

    @field_validator("graph", mode="before")
    @classmethod
    def validate_graph_json(cls, value: Any) -> Any:
        return _validate_json_value(value, "materialized graph")
