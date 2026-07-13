import copy
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

import mp_comfyui_lab.graph.models as graph_models
import mp_comfyui_lab.graph.normalize as graph_normalize
from mp_comfyui_lab.graph.models import (
    AmbiguousRoleError,
    MissingRoleError,
    NormalizedGraph,
    NormalizedNode,
    RoleBinding,
    RoleBindings,
    SemanticRole,
)
from mp_comfyui_lab.graph.normalize import (
    GraphTopologyError,
    GraphValidationError,
    normalize_graph,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json"


@pytest.fixture
def qwen_graph() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _node(class_type: str, **inputs: object) -> dict:
    return {"class_type": class_type, "inputs": inputs}


def _lora_graph() -> dict:
    return {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "upstream": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="first.safetensors",
            strength_model=0.8,
        ),
        "downstream": _node(
            "LoraLoaderModelOnly",
            model=["upstream", 0],
            lora_name="second.safetensors",
            strength_model=1.0,
        ),
        "sampler": _node(
            "KSampler",
            model=["downstream", 0],
            seed=7,
            sampler_name="euler",
            scheduler="normal",
        ),
        "output": _node("SaveImage", images=["sampler", 0], filename_prefix="x"),
    }


def _nested_json(depth: int) -> dict:
    value: object = "leaf"
    for _ in range(depth):
        value = {"nested": value}
    assert isinstance(value, dict)
    return value


def test_qwen_graph_detects_golden_roles_in_topology_order(qwen_graph) -> None:
    source = copy.deepcopy(qwen_graph)

    graph = normalize_graph(qwen_graph)

    assert graph.require_one(SemanticRole.CHECKPOINT).value == (
        "Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors"
    )
    assert graph.require_one(SemanticRole.MODEL).field_name == "unet_name"
    assert graph.require_one(SemanticRole.UNET).field_name == "unet_name"
    assert [binding.value for binding in graph.bindings(SemanticRole.LORA)] == [
        "next-scene_lora_v1-3000.safetensors",
        "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
    ]
    assert graph.require_one(SemanticRole.REFERENCE_IMAGE).field_name == "image"
    assert graph.require_one(SemanticRole.REFERENCE_IMAGE).value == "reference.png"
    assert graph.require_one(SemanticRole.SEED).node_id == "56"
    assert graph.require_one(SemanticRole.SEED).field_name == "seed"
    assert graph.bindings(SemanticRole.OUTPUT)
    assert qwen_graph == source


def test_qwen_topology_supports_approved_newer_golden_identities(qwen_graph) -> None:
    qwen_graph["106"]["inputs"]["unet_name"] = (
        "Qwen-Rapid-AIO-SFW-v5.safetensors"
    )
    qwen_graph["133"]["inputs"]["lora_name"] = (
        "next-scene_lora-v2-3000.safetensors"
    )
    qwen_graph["109"]["inputs"]["lora_name"] = (
        "qe2511_consis_alpha_patched.safetensors"
    )

    graph = normalize_graph(qwen_graph)

    assert graph.require_one(SemanticRole.CHECKPOINT).value == (
        "Qwen-Rapid-AIO-SFW-v5.safetensors"
    )
    assert [binding.value for binding in graph.bindings(SemanticRole.LORA)] == [
        "next-scene_lora-v2-3000.safetensors",
        "qe2511_consis_alpha_patched.safetensors",
    ]


def test_qwen_graph_detects_prompt_encoder_conditioning_vae_and_fanout(
    qwen_graph,
) -> None:
    graph = normalize_graph(qwen_graph)

    assert any(
        binding.node_id == "95" and binding.field_name == "prompt"
        for binding in graph.bindings(SemanticRole.PROMPT_SOURCE)
    )
    assert any(
        binding.node_id == "76"
        for binding in graph.bindings(SemanticRole.PROMPT_FANOUT)
    )
    assert any(
        binding.value == "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        for binding in graph.bindings(SemanticRole.ENCODER)
    )
    assert {binding.node_id for binding in graph.bindings(SemanticRole.CONDITIONING)} >= {
        "61",
        "68",
    }
    assert graph.require_one(SemanticRole.VAE).value == "qwen_image_vae.safetensors"


def test_detects_checkpoint_loader_dimensions_noise_seed_and_output_types() -> None:
    source = {
        "checkpoint": _node(
            "CheckpointLoaderSimple", ckpt_name="photo-model.safetensors"
        ),
        "latent": _node("EmptyLatentImage", width=1024, height=768, batch_size=2),
        "sample": _node(
            "SamplerCustomAdvanced",
            model=["checkpoint", 0],
            latent_image=["latent", 0],
            noise_seed=42,
            sampler_name="dpmpp_2m",
            scheduler_name="karras",
        ),
        "preview": _node("PreviewImage", images=["sample", 0]),
        "animated": _node("SaveAnimatedWEBP", images=["sample", 0], fps=12),
        "video": _node("VHS_VideoCombine", images=["sample", 0], frame_rate=24),
    }

    graph = normalize_graph(source)

    assert graph.require_one(SemanticRole.CHECKPOINT).value == "photo-model.safetensors"
    assert graph.require_one(SemanticRole.SEED).field_name == "noise_seed"
    assert graph.require_one(SemanticRole.SAMPLER).value == "dpmpp_2m"
    assert graph.require_one(SemanticRole.SCHEDULER).value == "karras"
    assert [(item.field_name, item.value) for item in graph.bindings(SemanticRole.DIMENSIONS)] == [
        ("height", 768),
        ("width", 1024),
    ]
    assert [item.value for item in graph.bindings(SemanticRole.OUTPUT)] == [
        "SaveAnimatedWEBP",
        "PreviewImage",
        "VHS_VideoCombine",
    ]


def test_model_name_input_on_custom_model_provider_is_detected() -> None:
    graph = normalize_graph(
        {
            "model": _node(
                "CustomModelProvider",
                model_name="provider-model.safetensors",
            )
        }
    )

    assert graph.require_one(SemanticRole.CHECKPOINT).value == (
        "provider-model.safetensors"
    )
    assert graph.require_one(SemanticRole.MODEL).value == (
        "provider-model.safetensors"
    )


def test_normalized_nodes_and_bindings_are_deeply_immutable_and_raw() -> None:
    source = {
        "1": _node(
            "PrimitiveNode",
            values=[1, {"nested": [2, 3]}],
            api_key="secret",
            url="https://example.test/model?token=secret",
        )
    }

    graph = normalize_graph(source)
    node = graph.nodes[0]
    binding = RoleBinding(
        node_id="1",
        field_name="value",
        value={"nested": [1, 2]},
        confidence=0.9,
        reason="synthetic binding",
    )

    assert node.inputs["values"] == (1, {"nested": (2, 3)})
    assert node.inputs["api_key"] == "secret"
    assert node.inputs["url"] == "https://example.test/model?token=secret"
    assert binding.value["nested"] == (1, 2)
    with pytest.raises(TypeError):
        node.inputs["values"] = ()
    with pytest.raises(TypeError):
        node.inputs["values"][1]["nested"] = ()
    with pytest.raises(TypeError):
        binding.value["nested"] = ()
    with pytest.raises(ValidationError):
        graph.nodes = ()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"confidence": -0.01}, "greater than or equal"),
        ({"confidence": 1.01}, "less than or equal"),
        ({"reason": "   "}, "at least 1 character"),
        ({"field_name": ""}, "at least 1 character"),
        ({"extra_field": "nope"}, "Extra inputs"),
    ],
)
def test_role_binding_is_strict(updates, message) -> None:
    values = {
        "node_id": "1",
        "field_name": "seed",
        "value": 1,
        "confidence": 0.9,
        "reason": "seed input",
    }
    values.update(updates)

    with pytest.raises(ValidationError, match=message):
        RoleBinding(**values)


def test_normalized_graph_node_order_is_deterministic_not_source_order() -> None:
    graph = {
        "10": _node("PrimitiveNode", value=10),
        "alpha": _node("PrimitiveNode", value="a"),
        "2": _node("PrimitiveNode", value=2),
        "01": _node("PrimitiveNode", value=1),
    }

    normalized = normalize_graph(dict(reversed(list(graph.items()))))

    assert [node.node_id for node in normalized.nodes] == ["01", "2", "10", "alpha"]


def test_require_one_returns_only_binding_strictly_above_confidence_threshold() -> None:
    weak = RoleBinding(
        node_id="weak",
        field_name="ckpt_name",
        value="guess.safetensors",
        confidence=0.8,
        reason="weak heuristic",
    )
    strong = RoleBinding(
        node_id="strong",
        field_name="ckpt_name",
        value="known.safetensors",
        confidence=0.81,
        reason="known loader",
    )
    graph = NormalizedGraph(
        nodes=(),
        role_bindings=(
            RoleBindings(role=SemanticRole.CHECKPOINT, bindings=(weak, strong)),
        ),
    )

    assert graph.require_one(SemanticRole.CHECKPOINT) is strong


def test_require_one_reports_all_ambiguous_bindings_without_node_id_tiebreak() -> None:
    graph = normalize_graph(
        {
            "99": _node("UNETLoader", unet_name="second.safetensors"),
            "1": _node("CheckpointLoaderSimple", ckpt_name="first.safetensors"),
        }
    )

    with pytest.raises(AmbiguousRoleError) as raised:
        graph.require_one(SemanticRole.CHECKPOINT)

    assert [item.node_id for item in raised.value.bindings] == ["1", "99"]
    assert "first.safetensors" in str(raised.value)
    assert "second.safetensors" in str(raised.value)


def test_require_one_reports_low_confidence_candidates_when_role_is_missing() -> None:
    candidate = RoleBinding(
        node_id="maybe",
        field_name="model_name",
        value="uncertain.safetensors",
        confidence=0.8,
        reason="not confident enough",
    )
    graph = NormalizedGraph(
        nodes=(),
        role_bindings=(
            RoleBindings(role=SemanticRole.CHECKPOINT, bindings=(candidate,)),
        ),
    )

    with pytest.raises(MissingRoleError) as raised:
        graph.require_one(SemanticRole.CHECKPOINT)

    assert raised.value.bindings == (candidate,)
    assert "maybe" in str(raised.value)
    assert "0.8" in str(raised.value)


def test_literal_arrays_are_not_misclassified_as_links() -> None:
    source = {
        "1": _node("PrimitiveNode", choices=["red", "blue"], pair=[1, 2]),
        "2": _node("Consumer", source=["1", 0], nested=[["1", 0]]),
    }

    graph = normalize_graph(source)
    consumer = next(node for node in graph.nodes if node.node_id == "2")

    assert consumer.input_links == {"source": ("1", 0)}
    assert consumer.inputs["nested"] == (("1", 0),)
    assert next(node for node in graph.nodes if node.node_id == "1").input_links == {}
    assert source["2"]["inputs"]["source"] == ["1", 0]


@pytest.mark.parametrize(
    "field_name",
    ["choices", "values", "list", "indices", "colors", "ranges"],
)
def test_literal_pair_matching_real_node_id_stays_literal(field_name) -> None:
    source = {
        "1": _node("PrimitiveNode", value="upstream"),
        "2": _node("CollectionSelector", **{field_name: ["1", 0]}),
    }

    selector = normalize_graph(source).nodes[1]

    assert selector.inputs[field_name] == ("1", 0)
    assert selector.input_links == {}


@pytest.mark.parametrize(
    "field_name",
    [
        "custom_pair",
        "control_net",
        "audio",
        "detailer_pipe",
        "frame_rate",
        "VAE1",
    ],
)
def test_existing_node_pair_is_a_link_for_custom_comfy_fields(field_name) -> None:
    source = {
        "1": _node("PrimitiveNode", value="upstream"),
        "2": _node("CustomNode", **{field_name: ["1", 0]}),
    }

    consumer = normalize_graph(source).nodes[1]

    assert consumer.input_links[field_name] == ("1", 0)


def test_primitive_literal_feeding_text_encoder_is_prompt_source() -> None:
    source = {
        "source": _node("PrimitiveNode", value="portrait of a traveler"),
        "encoder": _node("CLIPTextEncode", text=["source", 0], clip=["clip", 0]),
        "clip": _node("CLIPLoader", clip_name="encoder.safetensors"),
    }

    binding = normalize_graph(source).require_one(SemanticRole.PROMPT_SOURCE)

    assert binding.node_id == "source"
    assert binding.field_name == "value"
    assert binding.value == "portrait of a traveler"
    assert binding.confidence > 0.8
    assert "consumer" in binding.reason


@pytest.mark.parametrize(
    "consumer_field",
    ["positive_prompt", "negative_prompt", "Positive Prompt"],
)
def test_primitive_prompt_source_and_fanout_follow_normalized_linked_prompt_fields(
    consumer_field,
) -> None:
    source = {
        "source": _node("PrimitiveNode", value="portrait of a traveler"),
        "consumer": _node(
            "CLIPTextEncode",
            **{consumer_field: ["source", 0]},
        ),
    }

    graph = normalize_graph(source)

    assert graph.require_one(SemanticRole.PROMPT_SOURCE).node_id == "source"
    assert any(
        binding.node_id == "consumer" and binding.field_name == consumer_field
        for binding in graph.bindings(SemanticRole.PROMPT_FANOUT)
    )


def test_class_only_scheduler_node_gets_scheduler_binding() -> None:
    graph = normalize_graph({"schedule": _node("KarrasScheduler", steps=24)})

    binding = graph.require_one(SemanticRole.SCHEDULER)

    assert binding.field_name == "$class_type"
    assert binding.value == "KarrasScheduler"
    assert binding.confidence > 0.8
    assert binding.reason == "scheduler node class"


def test_generic_reference_provider_is_detected_without_save_false_positive() -> None:
    graph = normalize_graph(
        {
            "reference": _node("ReferenceProvider", image_name="character.png"),
            "save": _node("SaveImage", filename="render.png"),
        }
    )

    assert [
        (binding.node_id, binding.field_name, binding.value)
        for binding in graph.bindings(SemanticRole.REFERENCE_IMAGE)
    ] == [("reference", "image_name", "character.png")]


def test_positive_and_negative_prompt_fields_are_prompt_sources() -> None:
    graph = normalize_graph(
        {
            "prompt": _node(
                "WanVideoPrompt",
                positive_prompt="Scene: {subject}",
                negative_prompt="blur, noise",
            )
        }
    )

    assert [
        (binding.field_name, binding.value)
        for binding in graph.bindings(SemanticRole.PROMPT_SOURCE)
    ] == [
        ("negative_prompt", "blur, noise"),
        ("positive_prompt", "Scene: {subject}"),
    ]


def test_image_save_class_spellings_are_outputs_without_selector_false_positive() -> None:
    graph = normalize_graph(
        {
            "save": _node("Image Save", images=["source", 0]),
            "simple": _node("Image Saver Simple", images=["source", 0]),
            "selector": _node("Sampler Selector (Image Saver)", value="euler"),
            "source": _node("PrimitiveNode", value="pixels"),
        }
    )

    assert [binding.node_id for binding in graph.bindings(SemanticRole.OUTPUT)] == [
        "save",
        "simple",
    ]


def test_encoder_loader_resolves_over_low_confidence_text_encode_hint() -> None:
    graph = normalize_graph(
        {
            "loader": _node("CLIPLoader", clip_name="clip.safetensors"),
            "encode": _node(
                "CLIPTextEncode",
                clip=["loader", 0],
                text="portrait",
            ),
        }
    )

    binding = graph.require_one(SemanticRole.ENCODER)
    hints = [
        item
        for item in graph.bindings(SemanticRole.ENCODER)
        if item.node_id == "encode"
    ]

    assert binding.node_id == "loader"
    assert binding.confidence > 0.8
    assert hints and all(item.confidence <= 0.8 for item in hints)
    assert graph.require_one(SemanticRole.CONDITIONING).node_id == "encode"


def test_only_meta_keys_are_removed_recursively_from_semantic_projection() -> None:
    source = {
        "_meta": {"workflow_title": "ignored top-level UI metadata"},
        "1": {
            "class_type": "PrimitiveNode",
            "inputs": {
                "_meta": {"widget": "hidden"},
                "config": {"_meta": {"ui": True}, "_private": "kept", "value": 3},
            },
            "_meta": {"title": "UI title"},
            "display_name": "kept source field",
        }
    }
    original = copy.deepcopy(source)

    node = normalize_graph(source).nodes[0]

    assert "_meta" not in node.inputs
    assert node.inputs["config"] == {"_private": "kept", "value": 3}
    assert source == original


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "nonempty"),
        ([], "object"),
        ({1: _node("PrimitiveNode")}, "string"),
        ({"": _node("PrimitiveNode")}, "blank"),
        ({"1": []}, "node '1'.*object"),
        ({"1": {"inputs": {}}}, "class_type"),
        ({"1": {"class_type": "", "inputs": {}}}, "class_type.*nonblank"),
        ({"1": {"class_type": 7, "inputs": {}}}, "class_type.*string"),
        ({"1": {"class_type": "Node"}}, "inputs"),
        ({"1": {"class_type": "Node", "inputs": []}}, "inputs.*object"),
        ({"1": {"class_type": "Node", "inputs": {1: "x"}}}, "input names.*string"),
        ({"1": _node("Node", value=float("nan"))}, "finite"),
        ({"1": _node("Node", value=(1, 2))}, "unsupported JSON"),
        (
            {
                "1": _node("Node"),
                "2": _node("Consumer", model=["1", True]),
            },
            "output index",
        ),
        ({"1": _node("Consumer", model=["missing", 0])}, "unknown node"),
        ({"1": _node("Consumer", model=["2", 0])}, "unknown node"),
    ],
)
def test_malformed_api_graphs_are_rejected_deterministically(source, message) -> None:
    with pytest.raises(GraphValidationError, match=message):
        normalize_graph(source)


def test_lora_order_follows_model_edges_not_numeric_or_source_order() -> None:
    graph = {
        "2": _node(
            "LoraLoaderModelOnly",
            model=["99", 0],
            lora_name="second.safetensors",
            strength_model=0.7,
        ),
        "50": _node("UNETLoader", unet_name="base.safetensors"),
        "1": _node("KSampler", model=["2", 0], sampler_name="euler", seed=1),
        "99": _node(
            "LoraLoaderModelOnly",
            model=["50", 0],
            lora_name="first.safetensors",
            strength_model=0.9,
        ),
    }

    bindings = normalize_graph(graph).bindings(SemanticRole.LORA)

    assert [(binding.node_id, binding.value) for binding in bindings] == [
        ("99", "first.safetensors"),
        ("2", "second.safetensors"),
    ]


def test_lora_order_follows_model_edges_through_passthrough_nodes() -> None:
    graph = {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "first": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="first.safetensors",
            strength_model=0.8,
        ),
        "sampling": _node("ModelSamplingFlux", model=["first", 0], shift=3.0),
        "second": _node(
            "LoraLoaderModelOnly",
            model=["sampling", 0],
            lora_name="second.safetensors",
            strength_model=0.8,
        ),
        "sampler": _node("KSampler", model=["second", 0], seed=1),
    }

    bindings = normalize_graph(graph).bindings(SemanticRole.LORA)

    assert [binding.node_id for binding in bindings] == ["first", "second"]


@pytest.mark.parametrize("mode", ["unused", "bypassed"])
def test_lora_chain_requires_final_downstream_model_consumer(mode) -> None:
    graph = _lora_graph()
    if mode == "unused":
        del graph["sampler"]
        del graph["output"]
    else:
        graph["sampler"]["inputs"]["model"] = ["upstream", 0]

    with pytest.raises(GraphTopologyError, match="LoRA.*(consumer|bypass|branch)"):
        normalize_graph(graph)


def test_power_lora_loader_slots_bind_in_numeric_order_with_strength_buckets() -> None:
    graph = normalize_graph(
        {
            "loader": _node("UNETLoader", unet_name="base.safetensors"),
            "power": _node(
                "Power Lora Loader (rgthree)",
                model=["loader", 0],
                lora_10={
                    "on": True,
                    "lora": "ten.safetensors",
                    "strength": 1.5,
                },
                lora_2={
                    "on": True,
                    "lora": "two.safetensors",
                    "strength": 0.5,
                },
                lora_1={
                    "on": True,
                    "lora": "one.safetensors",
                    "strength": 1.0,
                },
            ),
            "sampler": _node("KSampler", model=["power", 0], seed=1),
        }
    )

    assert [
        (
            binding.field_name,
            binding.value["name"],
            binding.value["strength_bucket"],
        )
        for binding in graph.bindings(SemanticRole.LORA)
    ] == [
        ("lora_1", "one.safetensors", "positive_standard"),
        ("lora_2", "two.safetensors", "positive_light"),
        ("lora_10", "ten.safetensors", "positive_strong"),
    ]


def test_final_lora_fanout_to_multiple_model_consumers_is_valid() -> None:
    source = _lora_graph()
    source["second_sampler"] = _node(
        "SamplerCustom",
        model=["downstream", 0],
        seed=8,
    )

    graph = normalize_graph(source)

    assert [binding.value for binding in graph.bindings(SemanticRole.LORA)] == [
        "first.safetensors",
        "second.safetensors",
    ]


def test_wan_video_selector_chain_follows_prev_lora_into_model_loader() -> None:
    graph = normalize_graph(
        {
            "first": _node(
                "WanVideoLoraSelect",
                lora="first.safetensors",
                strength=0.6,
            ),
            "second": _node(
                "WanVideoLoraSelect",
                lora="second.safetensors",
                strength=1.0,
                prev_lora=["first", 0],
            ),
            "loader": _node(
                "WanVideoModelLoader",
                model_name="wan.safetensors",
                lora=["second", 0],
            ),
            "sampler": _node("WanVideoSampler", model=["loader", 0], seed=1),
        }
    )

    assert [
        (binding.value["name"], binding.value["strength_bucket"])
        for binding in graph.bindings(SemanticRole.LORA)
    ] == [
        ("first.safetensors", "positive_light"),
        ("second.safetensors", "positive_standard"),
    ]


def test_wan_video_selector_can_feed_model_transform_node() -> None:
    graph = normalize_graph(
        {
            "loader": _node("UNETLoader", unet_name="base.safetensors"),
            "selector": _node(
                "WanVideoLoraSelect",
                lora="motion.safetensors",
                strength=1.0,
            ),
            "apply": _node(
                "WanVideoSetLoRAs",
                model=["loader", 0],
                lora=["selector", 0],
            ),
            "sampler": _node("WanVideoSampler", model=["apply", 0], seed=1),
        }
    )

    binding = graph.bindings(SemanticRole.LORA)[0]
    assert binding.value["name"] == "motion.safetensors"


def test_independent_model_lora_branches_are_valid_without_total_order() -> None:
    graph = normalize_graph(
        {
            "high_loader": _node("UNETLoader", unet_name="high.safetensors"),
            "low_loader": _node("UNETLoader", unet_name="low.safetensors"),
            "high_lora": _node(
                "LoraLoaderModelOnly",
                model=["high_loader", 0],
                lora_name="high-lora.safetensors",
                strength_model=1.0,
            ),
            "low_lora": _node(
                "LoraLoaderModelOnly",
                model=["low_loader", 0],
                lora_name="low-lora.safetensors",
                strength_model=1.0,
            ),
            "high_sampler": _node("KSampler", model=["high_lora", 0], seed=1),
            "low_sampler": _node("KSampler", model=["low_lora", 0], seed=2),
        }
    )

    assert {binding.value for binding in graph.bindings(SemanticRole.LORA)} == {
        "high-lora.safetensors",
        "low-lora.safetensors",
    }


def test_lora_branches_with_competing_orders_merging_to_one_channel_are_rejected() -> None:
    graph = {
        "loader": _node("UNETLoader", unet_name="base.safetensors"),
        "root": _node(
            "LoraLoaderModelOnly",
            model=["loader", 0],
            lora_name="root.safetensors",
            strength_model=1.0,
        ),
        "left": _node(
            "LoraLoaderModelOnly",
            model=["root", 0],
            lora_name="left.safetensors",
            strength_model=1.0,
        ),
        "right": _node(
            "LoraLoaderModelOnly",
            model=["root", 0],
            lora_name="right.safetensors",
            strength_model=1.0,
        ),
        "merge": _node(
            "ModelMerge",
            model=["left", 0],
            unet=["right", 0],
        ),
        "sampler": _node("KSampler", model=["merge", 0], seed=1),
    }

    with pytest.raises(GraphTopologyError, match="competing LoRA orders"):
        normalize_graph(graph)


@pytest.mark.parametrize("mode", ["unused", "cycle"])
def test_invalid_wan_selector_stacks_are_rejected(mode) -> None:
    graph = {
        "first": _node(
            "WanVideoLoraSelect",
            lora="first.safetensors",
            strength=1.0,
        ),
        "second": _node(
            "WanVideoLoraSelect",
            lora="second.safetensors",
            strength=1.0,
            prev_lora=["first", 0],
        ),
    }
    if mode == "cycle":
        graph["first"]["inputs"]["prev_lora"] = ["second", 0]

    with pytest.raises(GraphTopologyError, match="LoRA.*(consumer|cycle|unused)"):
        normalize_graph(graph)


@pytest.mark.parametrize("node_id", [" 1", "1 ", "1\x00", "1\ud800"])
def test_node_ids_are_exact_safe_unicode_identifiers(node_id) -> None:
    with pytest.raises(GraphValidationError, match="node ID"):
        normalize_graph({node_id: _node("PrimitiveNode", value=1)})


def test_unicode_format_control_in_node_id_is_a_graph_error() -> None:
    with pytest.raises(GraphValidationError, match="node ID.*control"):
        normalize_graph({"node\u200bname": _node("PrimitiveNode", value=1)})


def test_very_long_numeric_node_ids_sort_without_integer_conversion() -> None:
    long_id = "9" * 5_000
    graph = normalize_graph(
        {
            long_id: _node("PrimitiveNode", value="large"),
            "9": _node("PrimitiveNode", value="small"),
        }
    )

    assert [node.node_id for node in graph.nodes] == ["9", long_id]


def test_graph_depth_limit_precedes_recursive_model_validation() -> None:
    value: object = "leaf"
    for _ in range(601):
        value = {"nested": value}

    with pytest.raises(GraphValidationError, match="depth limit"):
        normalize_graph({"1": _node("PrimitiveNode", payload=value)})


def test_normalized_model_json_depth_contract_is_direct_and_serializable() -> None:
    max_depth = graph_models.MAX_SERIALIZABLE_JSON_DEPTH
    accepted = _nested_json(max_depth)
    rejected = _nested_json(max_depth + 1)

    node = NormalizedNode(
        node_id="node",
        class_type="PrimitiveNode",
        inputs=accepted,
        input_links={},
    )
    binding = RoleBinding(
        node_id="node",
        field_name="value",
        value=accepted,
        confidence=1.0,
        reason="depth boundary",
    )

    assert NormalizedNode.model_validate_json(node.model_dump_json()) == node
    assert RoleBinding.model_validate_json(binding.model_dump_json()) == binding
    assert node.model_dump(mode="json")["inputs"] == accepted
    assert binding.model_dump(mode="json")["value"] == accepted
    normalized = normalize_graph(
        {"node": {"class_type": "PrimitiveNode", "inputs": accepted}}
    )
    assert normalized.nodes[0].model_dump(mode="json")["inputs"] == accepted

    with pytest.raises(ValidationError, match="depth limit"):
        NormalizedNode(
            node_id="node",
            class_type="PrimitiveNode",
            inputs=rejected,
            input_links={},
        )
    with pytest.raises(ValidationError, match="depth limit"):
        RoleBinding(
            node_id="node",
            field_name="value",
            value=rejected,
            confidence=1.0,
            reason="depth overflow",
        )
    with pytest.raises(GraphValidationError, match="depth limit"):
        normalize_graph({"node": {"class_type": "PrimitiveNode", "inputs": rejected}})


def test_graph_node_and_value_limits_are_enforced_before_freezing(monkeypatch) -> None:
    monkeypatch.setattr(graph_normalize, "_MAX_GRAPH_NODES", 2, raising=False)
    with pytest.raises(GraphValidationError, match="node limit"):
        normalize_graph(
            {
                "1": _node("PrimitiveNode"),
                "2": _node("PrimitiveNode"),
                "3": _node("PrimitiveNode"),
            }
        )

    monkeypatch.setattr(graph_normalize, "_MAX_JSON_VALUES", 8, raising=False)
    with pytest.raises(GraphValidationError, match="value limit"):
        normalize_graph({"1": _node("PrimitiveNode", values=list(range(20)))})


@pytest.mark.parametrize("node_id", [" 1", "1 ", "1\x00", "1\ud800"])
def test_strict_models_reject_unsafe_node_ids_without_normalizing(node_id) -> None:
    with pytest.raises(ValidationError, match="node ID"):
        NormalizedNode(
            node_id=node_id,
            class_type="PrimitiveNode",
            inputs={},
            input_links={},
        )
    with pytest.raises(ValidationError, match="node ID"):
        RoleBinding(
            node_id=node_id,
            field_name="value",
            value=1,
            confidence=1.0,
            reason="test binding",
        )


@pytest.mark.parametrize(
    "graph",
    [
        {"1": _node("Class\ud800", value="ok")},
        {"1": {"class_type": "PrimitiveNode", "inputs": {"key\ud800": "ok"}}},
        {"1": _node("PrimitiveNode", value={"nested": ["bad\ud800"]})},
    ],
)
def test_graph_rejects_lone_surrogates_recursively(graph) -> None:
    with pytest.raises(GraphValidationError, match="surrogate"):
        normalize_graph(graph)


def test_lora_cycle_is_reported() -> None:
    graph = {
        "a": _node(
            "LoraLoaderModelOnly",
            model=["b", 0],
            lora_name="a.safetensors",
            strength_model=1.0,
        ),
        "b": _node(
            "LoraLoaderModelOnly",
            model=["a", 0],
            lora_name="b.safetensors",
            strength_model=1.0,
        ),
    }

    with pytest.raises(GraphTopologyError, match="LoRA cycle.*a.*b"):
        normalize_graph(graph)


def test_disconnected_lora_chains_are_reported() -> None:
    graph = _lora_graph()
    graph["other"] = _node(
        "LoraLoaderModelOnly",
        model=["loader", 0],
        lora_name="other.safetensors",
        strength_model=1.0,
    )

    with pytest.raises(GraphTopologyError, match="disconnected LoRA chains"):
        normalize_graph(graph)


def test_branched_lora_chain_is_reported_as_ambiguous() -> None:
    graph = _lora_graph()
    graph["branch"] = _node(
        "LoraLoaderModelOnly",
        model=["upstream", 0],
        lora_name="branch.safetensors",
        strength_model=1.0,
    )

    with pytest.raises(GraphTopologyError, match="ambiguous LoRA branch.*upstream"):
        normalize_graph(graph)
