import copy

from pydantic import TypeAdapter, ValidationError
import pytest

from mp_comfyui_lab.evidence.models import ConfidenceLabel
from mp_comfyui_lab.graph.models import SemanticRole
from mp_comfyui_lab.graph.normalize import normalize_graph
from mp_comfyui_lab.graph.patch import (
    PatchTopologyError,
    PatchValidationError,
    apply_patch,
    materialize_variant,
)
from mp_comfyui_lab.mutations.models import (
    BoundaryInputMapping,
    BoundaryOutputMapping,
    DonorNode,
    LoraLoaderTemplate,
    LoraSpec,
    Mutation,
    ReplaceLoraStack,
    ReplaceSubgraph,
    SemanticPatch,
    SetNodeInput,
    SetRoleInput,
)


def _node(class_type: str, **inputs: object) -> dict:
    return {"class_type": class_type, "inputs": inputs}


@pytest.fixture
def two_lora_graph() -> dict:
    return {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "upstream": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="consistency.safetensors",
            strength_model=0.65,
        ),
        "downstream": _node(
            "LoraLoaderModelOnly",
            model=["upstream", 0],
            lora_name="multi-angle.safetensors",
            strength_model=0.8,
        ),
        "sampler": _node(
            "KSampler",
            model=["downstream", 0],
            seed=7,
            sampler_name="euler",
            scheduler="normal",
        ),
    }


def test_mutation_is_discriminated_strict_and_deeply_immutable() -> None:
    mutation = TypeAdapter(Mutation).validate_python(
        {
            "op": "set_node_input",
            "node_id": "1",
            "field_name": "seed",
            "value": {"nested": [1, 2]},
        }
    )

    assert isinstance(mutation, SetNodeInput)
    assert mutation.value == {"nested": (1, 2)}
    with pytest.raises(ValidationError):
        SetNodeInput(
            op="set_node_input",
            node_id="1",
            field_name="seed",
            value=1,
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        mutation.node_id = "2"
    with pytest.raises(TypeError):
        mutation.value["nested"] = ()


def test_replace_lora_stack_reorders_chain_and_rewires_consumer(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(
                        name="multi-angle.safetensors",
                        strength_model=0.8,
                    ),
                    LoraSpec(
                        name="consistency.safetensors",
                        strength_model=0.65,
                    ),
                ],
                evidence_ids=["workflow-42"],
            )
        ]
    )

    materialized = apply_patch(two_lora_graph, patch)
    normalized = normalize_graph(materialized)

    assert [
        binding.value for binding in normalized.bindings(SemanticRole.LORA)
    ] == ["multi-angle.safetensors", "consistency.safetensors"]
    assert materialized["sampler"]["inputs"]["model"][0] == (
        normalized.bindings(SemanticRole.LORA)[-1].node_id
    )


def test_unexplored_compatibility_is_warning_not_block(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="experimental.safetensors", strength_model=0.7),
                ],
                evidence_ids=["workflow-99"],
                confidence=ConfidenceLabel.UNEXPLORED,
            )
        ]
    )

    result = materialize_variant(two_lora_graph, patch)

    assert result.graph
    assert result.validation.errors == ()
    assert [issue.code for issue in result.validation.warnings] == [
        "unexplored_compatibility"
    ]


def test_reference_override_changes_only_selected_binding() -> None:
    graph = {
        "ref-character": _node("LoadImage", image="original-character.png"),
        "ref-style": _node("LoadImage", image="original-style.png"),
        "ref-z-background": _node("LoadImage", image="original-background.png"),
    }
    patch = SemanticPatch(
        mutations=[
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=0,
                field_name="image",
                value="uploaded-character.png",
            )
        ]
    )

    materialized = apply_patch(graph, patch)

    assert materialized["ref-character"]["inputs"]["image"] == (
        "uploaded-character.png"
    )
    assert materialized["ref-style"]["inputs"]["image"] == "original-style.png"
    assert materialized["ref-z-background"]["inputs"]["image"] == (
        "original-background.png"
    )


def test_replace_subgraph_remaps_internal_and_boundary_links_deterministically() -> None:
    graph = {
        "1": _node("ModelProvider", model_name="base.safetensors"),
        "2": _node("OldConditioning", model=["1", 0], text="portrait"),
        "10": _node("Consumer", conditioning=["2", 0]),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["2"],
                donor_nodes=[
                    DonorNode(
                        node_id="encode",
                        class_type="DonorEncode",
                        inputs={"model": None, "text": "portrait"},
                    ),
                    DonorNode(
                        node_id="condition",
                        class_type="DonorConditioning",
                        inputs={"encoded": ["encode", 0]},
                    ),
                ],
                boundary_inputs=[
                    BoundaryInputMapping(
                        donor_node_id="encode",
                        donor_field_name="model",
                        source_node_id="1",
                        source_output_index=0,
                    )
                ],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="condition",
                        donor_output_index=0,
                        consumer_node_id="10",
                        consumer_field_name="conditioning",
                    )
                ],
                donor_workflow_id="donor-workflow-7",
                evidence_ids=["evidence-2", "evidence-1", "evidence-1"],
            )
        ]
    )

    result = materialize_variant(graph, patch)
    materialized = result.graph

    assert "2" not in materialized
    assert materialized["11"]["inputs"]["model"] == ("1", 0)
    assert materialized["12"]["inputs"]["encoded"] == ("11", 0)
    assert materialized["10"]["inputs"]["conditioning"] == ("12", 0)
    assert result.diff.mutations[0].donor_workflow_id == "donor-workflow-7"
    assert result.diff.mutations[0].evidence_ids == ("evidence-1", "evidence-2")


def test_set_node_input_is_explicit_escape_hatch() -> None:
    graph = {
        "sampler": _node("KSampler", seed=1, cfg=5.0),
    }
    patch = SemanticPatch(
        mutations=[
            SetNodeInput(
                node_id="sampler",
                field_name="cfg",
                value=6.5,
            )
        ]
    )

    result = materialize_variant(graph, patch)

    assert result.graph["sampler"]["inputs"]["cfg"] == 6.5
    assert [node.node_id for node in result.diff.nodes] == ["sampler"]
    assert result.diff.nodes[0].fields[0].path == ("inputs", "cfg")


def test_apply_patch_never_mutates_source_graph(two_lora_graph: dict) -> None:
    source = copy.deepcopy(two_lora_graph)
    patch = SemanticPatch(
        mutations=[
            SetNodeInput(
                node_id="sampler",
                field_name="seed",
                value=99,
            )
        ]
    )

    result = apply_patch(two_lora_graph, patch)

    assert two_lora_graph == source
    assert result is not two_lora_graph
    assert result["sampler"] is not two_lora_graph["sampler"]


def test_apply_patch_is_atomic_when_a_later_mutation_fails(
    two_lora_graph: dict,
) -> None:
    source = copy.deepcopy(two_lora_graph)
    patch = SemanticPatch(
        mutations=[
            SetNodeInput(node_id="sampler", field_name="seed", value=99),
            SetNodeInput(node_id="missing", field_name="seed", value=10),
        ]
    )

    with pytest.raises(PatchValidationError, match="missing"):
        apply_patch(two_lora_graph, patch)

    assert two_lora_graph == source


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_mutation_values_reject_nonfinite_json(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        SetNodeInput(node_id="sampler", field_name="cfg", value=bad_value)


def test_mutation_values_reject_excessive_depth() -> None:
    value: object = None
    for _ in range(50):
        value = [value]

    with pytest.raises(ValidationError, match="depth"):
        SetNodeInput(node_id="sampler", field_name="cfg", value=value)


def test_conflicting_mutations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="conflicting node"):
        SemanticPatch(
            mutations=[
                SetNodeInput(node_id="sampler", field_name="cfg", value=5),
                SetNodeInput(node_id="sampler", field_name="cfg", value=6),
            ]
        )


def test_set_role_input_requires_the_exact_bound_field() -> None:
    graph = {"reference": _node("LoadImage", image="reference.png")}
    patch = SemanticPatch(
        mutations=[
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=0,
                field_name="path",
                value="other.png",
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="field"):
        apply_patch(graph, patch)


def test_lora_growth_requires_a_provenance_bearing_loader_template(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="consistency.safetensors", strength_model=0.65),
                    LoraSpec(name="multi-angle.safetensors", strength_model=0.8),
                    LoraSpec(name="new.safetensors", strength_model=0.5),
                ],
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchTopologyError, match="template"):
        apply_patch(two_lora_graph, patch)


def test_lora_growth_uses_the_explicit_provenance_template(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="consistency.safetensors", strength_model=0.65),
                    LoraSpec(name="multi-angle.safetensors", strength_model=0.8),
                    LoraSpec(name="new.safetensors", strength_model=0.5),
                ],
                evidence_ids=["evidence"],
                loader_template=LoraLoaderTemplate(
                    donor_node_id="donor-loader",
                    class_type="LoraLoaderModelOnly",
                    inputs={
                        "model": ["unused", 0],
                        "lora_name": "donor.safetensors",
                        "strength_model": 1.0,
                    },
                    donor_workflow_id="donor-workflow",
                    evidence_ids=["donor-evidence"],
                ),
            )
        ]
    )

    result = apply_patch(two_lora_graph, patch)
    normalized = normalize_graph(result)

    assert [binding.value for binding in normalized.bindings(SemanticRole.LORA)] == [
        "consistency.safetensors",
        "multi-angle.safetensors",
        "new.safetensors",
    ]
    assert result["sampler"]["inputs"]["model"][0] == "1"


def test_lora_growth_reuses_spare_loaders_before_allocating_new_ones(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="consistency.safetensors", strength_model=0.65),
                    LoraSpec(name="new-one.safetensors", strength_model=0.5),
                    LoraSpec(name="new-two.safetensors", strength_model=0.4),
                ],
                evidence_ids=["evidence"],
                loader_template=LoraLoaderTemplate(
                    donor_node_id="donor-loader",
                    class_type="LoraLoaderModelOnly",
                    inputs={
                        "model": ["unused", 0],
                        "lora_name": "donor.safetensors",
                        "strength_model": 1.0,
                    },
                    donor_workflow_id="donor-workflow",
                    evidence_ids=["donor-evidence"],
                ),
            )
        ]
    )

    result = apply_patch(two_lora_graph, patch)

    assert result["sampler"]["inputs"]["model"][0] == "1"


def test_lora_patch_rejects_independent_chain_ambiguity() -> None:
    graph = {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "left": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="left.safetensors",
            strength_model=1.0,
        ),
        "right": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="right.safetensors",
            strength_model=1.0,
        ),
        "left-sampler": _node("KSampler", model=["left", 0], seed=1),
        "right-sampler": _node("KSampler", model=["right", 0], seed=1),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[LoraSpec(name="left.safetensors", strength_model=1.0)],
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchTopologyError, match="unambiguous"):
        apply_patch(graph, patch)


def test_lora_patch_rejects_branched_or_cyclic_chains() -> None:
    branched = {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "first": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="one.safetensors",
            strength_model=1.0,
        ),
        "left": _node(
            "LoraLoaderModelOnly",
            model=["first", 0],
            lora_name="two.safetensors",
            strength_model=1.0,
        ),
        "right": _node(
            "LoraLoaderModelOnly",
            model=["first", 0],
            lora_name="three.safetensors",
            strength_model=1.0,
        ),
        "sampler": _node("KSampler", model=["left", 0], seed=1),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[LoraSpec(name="one.safetensors", strength_model=1.0)],
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchTopologyError):
        apply_patch(branched, patch)


def test_lora_patch_rejects_cyclic_chain() -> None:
    graph = {
        "first": _node(
            "LoraLoaderModelOnly",
            model=["second", 0],
            lora_name="one.safetensors",
            strength_model=1.0,
        ),
        "second": _node(
            "LoraLoaderModelOnly",
            model=["first", 0],
            lora_name="two.safetensors",
            strength_model=1.0,
        ),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[LoraSpec(name="one.safetensors", strength_model=1.0)],
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchTopologyError, match="cycle"):
        apply_patch(graph, patch)


def test_subgraph_replacement_rejects_unmapped_outgoing_target_link() -> None:
    graph = {
        "1": _node("ModelProvider", model_name="base.safetensors"),
        "2": _node("OldConditioning", model=["1", 0]),
        "10": _node("Consumer", conditioning=["2", 0]),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["2"],
                donor_nodes=[
                    DonorNode(
                        node_id="donor",
                        class_type="DonorConditioning",
                        inputs={},
                    )
                ],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="donor",
                        donor_output_index=0,
                        consumer_node_id="10",
                        consumer_field_name="other",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="boundary"):
        apply_patch(graph, patch)


def test_dangling_links_are_rejected_before_patch_application() -> None:
    graph = {"sampler": _node("KSampler", model=["missing", 0], seed=1)}
    patch = SemanticPatch(
        mutations=[SetNodeInput(node_id="sampler", field_name="seed", value=2)]
    )

    with pytest.raises(PatchTopologyError, match="unknown"):
        apply_patch(graph, patch)


def test_diff_is_deterministic_across_input_mapping_order() -> None:
    first = {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "sampler": _node("KSampler", model=["loader", 0], seed=1, cfg=5.0),
    }
    second = {
        "sampler": _node("KSampler", cfg=5.0, seed=1, model=["loader", 0]),
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
    }
    patch = SemanticPatch(
        mutations=[SetNodeInput(node_id="sampler", field_name="seed", value=2)]
    )

    assert materialize_variant(first, patch).diff == materialize_variant(second, patch).diff


def _two_lora_model_clip_graph() -> dict:
    return {
        "base": _node("CheckpointLoaderSimple", ckpt_name="base.safetensors"),
        "first": _node(
            "LoraLoader",
            model=["base", 0],
            clip=["base", 1],
            lora_name="consistency.safetensors",
            strength_model=0.65,
            strength_clip=0.65,
        ),
        "second": _node(
            "LoraLoader",
            model=["first", 0],
            clip=["first", 1],
            lora_name="multi-angle.safetensors",
            strength_model=0.8,
            strength_clip=0.8,
        ),
        "sampler": _node("KSampler", model=["second", 0], seed=7),
        "encoder": _node("CLIPTextEncode", clip=["second", 1], text="portrait"),
    }


def test_lora_reorder_rewires_standard_model_and_clip_paths() -> None:
    graph = _two_lora_model_clip_graph()
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="multi-angle.safetensors", strength_model=0.8, strength_clip=0.8),
                    LoraSpec(name="consistency.safetensors", strength_model=0.65, strength_clip=0.65),
                ],
                evidence_ids=["evidence"],
            )
        ]
    )

    result = apply_patch(graph, patch)

    assert result["second"]["inputs"]["model"] == ["base", 0]
    assert result["second"]["inputs"]["clip"] == ["base", 1]
    assert result["first"]["inputs"]["model"] == ["second", 0]
    assert result["first"]["inputs"]["clip"] == ["second", 1]
    assert result["sampler"]["inputs"]["model"] == ["first", 0]
    assert result["encoder"]["inputs"]["clip"] == ["first", 1]


def test_lora_reorder_rejects_intermediate_clip_consumer() -> None:
    graph = _two_lora_model_clip_graph()
    graph["stale-encoder"] = _node("CLIPTextEncode", clip=["first", 1], text="stale")
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[LoraSpec(name="consistency.safetensors", strength_model=0.65)],
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchTopologyError, match="orphan"):
        apply_patch(graph, patch)


@pytest.mark.parametrize("mapping_kind", ["unknown", "unused", "incomplete"])
def test_subgraph_boundary_input_preflight_is_total(mapping_kind: str) -> None:
    graph = {
        "source": _node("ModelProvider", model_name="base.safetensors"),
        "old": _node("Old", model=["source", 0]),
        "consumer": _node("Consumer", value=["old", 0]),
    }
    boundary_inputs = []
    donor_inputs: dict[str, object] = {"model": ["source", 0]}
    if mapping_kind == "unknown":
        boundary_inputs = [
            BoundaryInputMapping(
                donor_node_id="missing",
                donor_field_name="model",
                source_node_id="source",
                source_output_index=0,
            )
        ]
    elif mapping_kind == "unused":
        donor_inputs = {"model": None, "unused": "literal"}
        boundary_inputs = [
            BoundaryInputMapping(
                donor_node_id="donor",
                donor_field_name="unused",
                source_node_id="source",
                source_output_index=0,
            )
        ]
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["old"],
                donor_nodes=[DonorNode(node_id="donor", class_type="Donor", inputs=donor_inputs)],
                boundary_inputs=boundary_inputs,
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="donor",
                        donor_output_index=0,
                        consumer_node_id="consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="boundary input"):
        apply_patch(graph, patch)


def test_subgraph_rejects_disconnected_donor_component() -> None:
    graph = {
        "old": _node("Old"),
        "consumer": _node("Consumer", value=["old", 0]),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["old"],
                donor_nodes=[
                    DonorNode(node_id="used", class_type="Used", inputs={}),
                    DonorNode(node_id="dead", class_type="Dead", inputs={}),
                ],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="used",
                        donor_output_index=0,
                        consumer_node_id="consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="path"):
        apply_patch(graph, patch)


def test_subgraph_rejects_unknown_donor_output_index() -> None:
    graph = {"old": _node("Old"), "consumer": _node("Consumer", value=["old", 0])}
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["old"],
                donor_nodes=[DonorNode(node_id="donor", class_type="Donor", inputs={})],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="donor",
                        donor_output_index=1,
                        consumer_node_id="consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="output"):
        apply_patch(graph, patch)


def test_set_role_input_must_remain_the_selected_role_binding() -> None:
    graph = {"reference": _node("LoadImage", image="reference.png")}
    patch = SemanticPatch(
        mutations=[
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=0,
                field_name="image",
                value="",
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="no longer"):
        apply_patch(graph, patch)


def test_set_role_input_rejects_out_of_range_but_allows_explicit_multiple_binding() -> None:
    graph = {
        "first": _node("LoadImage", image="first.png"),
        "second": _node("LoadImage", image="second.png"),
    }
    out_of_range = SemanticPatch(
        mutations=[
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=2,
                field_name="image",
                value="missing.png",
            )
        ]
    )
    selected = SemanticPatch(
        mutations=[
            SetRoleInput(
                role=SemanticRole.REFERENCE_IMAGE,
                binding_index=1,
                field_name="image",
                value="selected.png",
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="index"):
        apply_patch(graph, out_of_range)
    assert apply_patch(graph, selected)["second"]["inputs"]["image"] == "selected.png"


@pytest.mark.parametrize("reverse", [False, True])
def test_cross_operation_conflicts_are_rejected_symmetrically(reverse: bool) -> None:
    graph = {
        "old": _node("Old", value=1),
        "consumer": _node("Consumer", value=["old", 0]),
    }
    set_input = SetNodeInput(node_id="old", field_name="value", value=2)
    replacement = ReplaceSubgraph(
        target_node_ids=["old"],
        donor_nodes=[DonorNode(node_id="donor", class_type="Donor", inputs={})],
        boundary_outputs=[
            BoundaryOutputMapping(
                donor_node_id="donor",
                donor_output_index=0,
                consumer_node_id="consumer",
                consumer_field_name="value",
            )
        ],
        donor_workflow_id="donor",
        evidence_ids=["evidence"],
    )
    patch = SemanticPatch(mutations=[replacement, set_input] if reverse else [set_input, replacement])

    with pytest.raises(PatchValidationError, match="conflict"):
        apply_patch(graph, patch)


@pytest.mark.parametrize("reverse", [False, True])
def test_lora_controlled_nodes_conflict_with_direct_writes(
    two_lora_graph: dict,
    reverse: bool,
) -> None:
    direct = SetNodeInput(
        node_id="upstream",
        field_name="lora_name",
        value="direct.safetensors",
    )
    lora = ReplaceLoraStack(
        loras=[LoraSpec(name="consistency.safetensors", strength_model=0.65)],
        evidence_ids=["evidence"],
    )
    patch = SemanticPatch(mutations=[lora, direct] if reverse else [direct, lora])

    with pytest.raises(PatchValidationError, match="conflict"):
        apply_patch(two_lora_graph, patch)


def test_large_changed_mapping_has_the_same_apply_and_materialize_domain() -> None:
    graph = {"node": _node("Primitive", payload={str(index): 0 for index in range(10_100)})}
    patch = SemanticPatch(
        mutations=[
            SetNodeInput(
                node_id="node",
                field_name="payload",
                value={str(index): 1 for index in range(10_100)},
            )
        ]
    )

    assert apply_patch(graph, patch)["node"]["inputs"]["payload"]["0"] == 1
    result = materialize_variant(graph, patch)
    assert len(result.diff.nodes[0].fields) == 1


def test_lora_template_provenance_is_recorded_for_added_loaders(
    two_lora_graph: dict,
) -> None:
    patch = SemanticPatch(
        mutations=[
            ReplaceLoraStack(
                loras=[
                    LoraSpec(name="consistency.safetensors", strength_model=0.65),
                    LoraSpec(name="multi-angle.safetensors", strength_model=0.8),
                    LoraSpec(name="new.safetensors", strength_model=0.5),
                ],
                evidence_ids=["outer-evidence"],
                loader_template=LoraLoaderTemplate(
                    donor_node_id="template-node",
                    class_type="LoraLoaderModelOnly",
                    inputs={"model": ["unused", 0], "lora_name": "donor", "strength_model": 1.0},
                    donor_workflow_id="template-workflow",
                    evidence_ids=["template-b", "template-a"],
                ),
            )
        ]
    )

    mutation_diff = materialize_variant(two_lora_graph, patch).diff.mutations[0]

    assert mutation_diff.evidence_ids == ("outer-evidence",)
    assert mutation_diff.loader_template_donor_node_id == "template-node"
    assert mutation_diff.loader_template_donor_workflow_id == "template-workflow"
    assert mutation_diff.loader_template_evidence_ids == ("template-a", "template-b")
    assert mutation_diff.added_node_ids == ("1",)


def test_mutation_identifiers_and_nested_json_text_are_bounded() -> None:
    too_long = "x" * 2_049
    with pytest.raises(ValidationError, match="at most"):
        SetNodeInput(node_id=too_long, field_name="value", value=1)
    with pytest.raises(ValidationError, match="string"):
        SetNodeInput(node_id="node", field_name="value", value={"nested": too_long})
    with pytest.raises(ValidationError, match="key"):
        SetNodeInput(node_id="node", field_name="value", value={too_long: "value"})


def test_subgraph_rejects_boundary_mapping_on_literal_collection_field() -> None:
    graph = {
        "source": _node("Source"),
        "old": _node("Old"),
        "consumer": _node("Consumer", value=["old", 0]),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["old"],
                donor_nodes=[
                    DonorNode(
                        node_id="donor",
                        class_type="Donor",
                        inputs={"values": None},
                    )
                ],
                boundary_inputs=[
                    BoundaryInputMapping(
                        donor_node_id="donor",
                        donor_field_name="values",
                        source_node_id="source",
                        source_output_index=0,
                    )
                ],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="donor",
                        donor_output_index=0,
                        consumer_node_id="consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="materialize"):
        apply_patch(graph, patch)


def test_subgraph_literal_collection_does_not_create_donor_reachability_edge() -> None:
    graph = {"old": _node("Old"), "consumer": _node("Consumer", value=["old", 0])}
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["old"],
                donor_nodes=[
                    DonorNode(node_id="used", class_type="Used", inputs={}),
                    DonorNode(
                        node_id="literal-parent",
                        class_type="Literal",
                        inputs={"values": ["used", 0]},
                    ),
                ],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="literal-parent",
                        donor_output_index=0,
                        consumer_node_id="consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="donor",
                evidence_ids=["evidence"],
            )
        ]
    )

    with pytest.raises(PatchValidationError, match="path"):
        apply_patch(graph, patch)


def test_independent_subgraph_replacements_reserve_cumulative_ids_deterministically() -> None:
    graph = {
        "1": _node("Old"),
        "2": _node("Old"),
        "left-consumer": _node("Consumer", value=["1", 0]),
        "right-consumer": _node("Consumer", value=["2", 0]),
    }
    patch = SemanticPatch(
        mutations=[
            ReplaceSubgraph(
                target_node_ids=["1"],
                donor_nodes=[DonorNode(node_id="left", class_type="New", inputs={})],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="left",
                        donor_output_index=0,
                        consumer_node_id="left-consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="left",
                evidence_ids=["left-evidence"],
            ),
            ReplaceSubgraph(
                target_node_ids=["2"],
                donor_nodes=[DonorNode(node_id="right", class_type="New", inputs={})],
                boundary_outputs=[
                    BoundaryOutputMapping(
                        donor_node_id="right",
                        donor_output_index=0,
                        consumer_node_id="right-consumer",
                        consumer_field_name="value",
                    )
                ],
                donor_workflow_id="right",
                evidence_ids=["right-evidence"],
            ),
        ]
    )

    result = materialize_variant(graph, patch)

    assert result.graph["left-consumer"]["inputs"]["value"] == ("3", 0)
    assert result.graph["right-consumer"]["inputs"]["value"] == ("4", 0)
    assert tuple(item.added_node_ids for item in result.diff.mutations) == (("3",), ("4",))


def test_subgraph_target_identifiers_are_bounded() -> None:
    too_long = "x" * 2_049
    with pytest.raises(ValidationError, match="at most"):
        ReplaceSubgraph(
            target_node_ids=[too_long],
            donor_nodes=[DonorNode(node_id="donor", class_type="Donor", inputs={})],
            boundary_outputs=[
                BoundaryOutputMapping(
                    donor_node_id="donor",
                    donor_output_index=0,
                    consumer_node_id="consumer",
                    consumer_field_name="value",
                )
            ],
            donor_workflow_id="donor",
            evidence_ids=["evidence"],
        )
