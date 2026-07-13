import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import mp_comfyui_lab.graph.hashing as graph_hashing
from mp_comfyui_lab.graph.hashing import (
    capability_hash,
    exact_run_hash,
    source_graph_hash,
    strength_bucket,
)
from mp_comfyui_lab.graph.normalize import GraphValidationError, normalize_graph
from mp_comfyui_lab.json_types import freeze_json_object


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json"
VALID_FIXTURES = {
    "reference_image": "a" * 64,
    "mask": "b" * 64,
}


@pytest.fixture
def qwen_graph() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _reverse_objects(value):
    if isinstance(value, dict):
        return {
            key: _reverse_objects(item)
            for key, item in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_objects(item) for item in value]
    return value


def _without_meta(value):
    if isinstance(value, dict):
        return {
            key: _without_meta(item)
            for key, item in value.items()
            if key != "_meta"
        }
    if isinstance(value, list):
        return [_without_meta(item) for item in value]
    return value


def _renumber_graph(graph: dict) -> dict:
    ordered_ids = sorted(
        graph,
        key=lambda node_id: (0, int(node_id)) if node_id.isdigit() else (1, node_id),
    )
    replacements = {
        node_id: f"renumbered-{index:03d}"
        for index, node_id in enumerate(reversed(ordered_ids), start=1)
    }
    renumbered = {}
    for old_id, node in reversed(list(graph.items())):
        new_node = copy.deepcopy(node)
        for field_name, value in new_node["inputs"].items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and value[0] in replacements
                and isinstance(value[1], int)
                and not isinstance(value[1], bool)
            ):
                new_node["inputs"][field_name] = [replacements[value[0]], value[1]]
        renumbered[replacements[old_id]] = new_node
    return renumbered


def test_source_graph_hash_is_compact_sorted_utf8_sha256_without_meta() -> None:
    graph = {
        "节点": {
            "inputs": {
                "text": "café 猫",
                "nested": {"keep": 1, "_meta": {"widget": "x"}},
            },
            "class_type": "TextNode",
            "_meta": {"title": "ignored"},
        }
    }
    projection = _without_meta(graph)
    expected = hashlib.sha256(
        json.dumps(
            projection,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert source_graph_hash(graph) == expected
    assert source_graph_hash(graph) != source_graph_hash(
        {
            "节点": {
                "inputs": {
                    "text": "café 猫",
                    "nested": {"keep": 1},
                },
                "class_type": "TextNode",
            }
        }
    )


def test_source_graph_hash_ignores_mapping_order_and_meta_only(qwen_graph) -> None:
    original = copy.deepcopy(qwen_graph)
    reordered = _reverse_objects(qwen_graph)
    changed_meta = copy.deepcopy(qwen_graph)
    changed_meta["56"]["_meta"] = {"title": "different", "color": "blue"}
    changed_meta["_meta"] = {"workflow_title": "top-level UI metadata"}
    changed_real_field = copy.deepcopy(qwen_graph)
    changed_real_field["56"]["inputs"]["steps"] = 5
    changed_private_field = copy.deepcopy(qwen_graph)
    changed_private_field["56"]["inputs"]["_private"] = "semantic"

    digest = source_graph_hash(qwen_graph)

    assert source_graph_hash(reordered) == digest
    assert source_graph_hash(changed_meta) == digest
    assert source_graph_hash(changed_real_field) != digest
    assert source_graph_hash(changed_private_field) != digest
    assert qwen_graph == original


def test_exact_run_hash_binds_materialized_values_and_ordered_prompts(qwen_graph) -> None:
    baseline = exact_run_hash(
        qwen_graph,
        fixtures=VALID_FIXTURES,
        prompts=["first", "second"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )

    graph_seed = copy.deepcopy(qwen_graph)
    graph_seed["56"]["inputs"]["seed"] += 1
    filename = copy.deepcopy(qwen_graph)
    filename["105"]["inputs"]["filename_prefix"] = "other"

    assert baseline != exact_run_hash(
        graph_seed,
        fixtures=VALID_FIXTURES,
        prompts=["first", "second"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )
    assert baseline != exact_run_hash(
        filename,
        fixtures=VALID_FIXTURES,
        prompts=["first", "second"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )
    assert baseline != exact_run_hash(
        qwen_graph,
        fixtures=VALID_FIXTURES,
        prompts=["second", "first"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )
    assert baseline != exact_run_hash(
        qwen_graph,
        fixtures={**VALID_FIXTURES, "mask": "c" * 64},
        prompts=["first", "second"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )
    assert baseline != exact_run_hash(
        qwen_graph,
        fixtures=VALID_FIXTURES,
        prompts=["first", "second"],
        seed=43,
        execution_parameters={"instance_type": "plus", "timeout": 600},
    )
    assert baseline != exact_run_hash(
        qwen_graph,
        fixtures=VALID_FIXTURES,
        prompts=["first", "second"],
        seed=42,
        execution_parameters={"instance_type": "plus", "timeout": 601},
    )


def test_exact_run_hash_ignores_mapping_order_and_ui_meta(qwen_graph) -> None:
    baseline = exact_run_hash(
        qwen_graph,
        fixtures=VALID_FIXTURES,
        prompts=["one"],
        seed=9,
        execution_parameters={"z": {"b": 2, "a": 1}, "a": True},
    )
    changed_meta = copy.deepcopy(qwen_graph)
    changed_meta["56"]["_meta"] = {"title": "new UI title"}

    assert exact_run_hash(
        _reverse_objects(qwen_graph),
        fixtures=dict(reversed(list(VALID_FIXTURES.items()))),
        prompts=["one"],
        seed=9,
        execution_parameters={"a": True, "z": {"a": 1, "b": 2}},
    ) == baseline
    assert exact_run_hash(
        changed_meta,
        fixtures=VALID_FIXTURES,
        prompts=["one"],
        seed=9,
        execution_parameters={"z": {"b": 2, "a": 1}, "a": True},
    ) == baseline


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fixtures": []}, "fixtures.*mapping"),
        ({"fixtures": {"": "a" * 64}}, "fixture names.*nonblank"),
        ({"fixtures": {"reference": "A" * 64}}, "SHA-256"),
        ({"fixtures": {"reference": "abc"}}, "SHA-256"),
        ({"prompts": "one prompt"}, "prompts.*ordered"),
        ({"prompts": ["valid", 2]}, "prompt at index 1.*string"),
        ({"prompts": {"unordered"}}, "prompts.*ordered"),
        ({"seed": True}, "seed.*integer"),
        ({"seed": 1.0}, "seed.*integer"),
        ({"seed": -1}, "seed.*between"),
        ({"seed": 2**64}, "seed.*between"),
        ({"execution_parameters": []}, "execution_parameters.*mapping"),
        ({"execution_parameters": {1: "bad"}}, "keys.*strings"),
        ({"execution_parameters": {"cfg": math.inf}}, "finite"),
        ({"execution_parameters": {"opaque": object()}}, "unsupported JSON"),
    ],
)
def test_exact_run_hash_rejects_ambiguous_inputs(qwen_graph, overrides, message) -> None:
    values = {
        "fixtures": VALID_FIXTURES,
        "prompts": ["prompt"],
        "seed": 1,
        "execution_parameters": {},
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        exact_run_hash(qwen_graph, **values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_all_hashes_reject_nonfinite_graph_json(qwen_graph, value) -> None:
    qwen_graph["56"]["inputs"]["cfg"] = value

    with pytest.raises(ValueError, match="finite"):
        source_graph_hash(qwen_graph)
    with pytest.raises(ValueError, match="finite"):
        exact_run_hash(qwen_graph, fixtures={}, prompts=[], seed=1)
    with pytest.raises(ValueError, match="finite"):
        capability_hash(qwen_graph)


def test_capability_hash_ignores_seed_filenames_task_ids_and_timestamps(
    qwen_graph,
) -> None:
    changed = copy.deepcopy(qwen_graph)
    changed["56"]["inputs"]["seed"] += 100
    changed["95"]["inputs"]["seed"] += 100
    changed["105"]["inputs"]["filename_prefix"] = "other/output"
    changed["105"]["inputs"]["task_id"] = "provider-task-99"
    changed["105"]["inputs"]["created_at"] = "2030-01-01T00:00:00Z"

    assert capability_hash(qwen_graph) == capability_hash(changed)
    assert exact_run_hash(qwen_graph, fixtures={}, prompts=[], seed=1) != exact_run_hash(
        changed,
        fixtures={},
        prompts=[],
        seed=2,
    )


def test_capability_hash_ignores_generic_output_prefix_metadata(qwen_graph) -> None:
    changed = copy.deepcopy(qwen_graph)
    changed["105"]["inputs"]["prefix"] = "provider-specific-prefix"

    assert capability_hash(changed) == capability_hash(qwen_graph)
    assert exact_run_hash(changed, fixtures={}, prompts=[], seed=1) != exact_run_hash(
        qwen_graph,
        fixtures={},
        prompts=[],
        seed=1,
    )


def test_capability_hash_uses_prompt_grammar_not_incidental_wording(qwen_graph) -> None:
    wording = copy.deepcopy(qwen_graph)
    wording["95"]["inputs"]["prompt"] = "Render a coherent follow-up shot."
    grammar = copy.deepcopy(qwen_graph)
    grammar["95"]["inputs"]["prompt"] = "Scene: {next_action}\nStyle: [cinematic]"

    baseline = capability_hash(qwen_graph)

    assert capability_hash(wording) == baseline
    assert capability_hash(grammar) != baseline


def test_raw_query_parameters_remain_material_except_signed_credentials() -> None:
    revision_one = {
        "1": {
            "class_type": "ModelProvider",
            "inputs": {
                "endpoint": (
                    "https://models.example/model?revision=1&"
                    "X-Amz-Signature=secret-one&token=signed-one"
                )
            },
        }
    }
    different_signature = copy.deepcopy(revision_one)
    different_signature["1"]["inputs"]["endpoint"] = (
        "https://models.example/model?token=signed-two&revision=1&"
        "X-Amz-Signature=secret-two"
    )
    revision_two = copy.deepcopy(revision_one)
    revision_two["1"]["inputs"]["endpoint"] = (
        "https://models.example/model?revision=2&"
        "X-Amz-Signature=secret-one&token=signed-one"
    )

    assert normalize_graph(revision_one).nodes[0].inputs["endpoint"].endswith(
        "X-Amz-Signature=secret-one&token=signed-one"
    )
    assert source_graph_hash(revision_one) != source_graph_hash(different_signature)
    assert exact_run_hash(
        revision_one, fixtures={}, prompts=[], seed=1
    ) != exact_run_hash(different_signature, fixtures={}, prompts=[], seed=1)
    assert capability_hash(revision_one) == capability_hash(different_signature)
    assert capability_hash(revision_one) != capability_hash(revision_two)

    projection = graph_hashing._capability_projection(normalize_graph(revision_one))
    serialized = json.dumps(projection, ensure_ascii=False, sort_keys=True)
    assert "revision=1" in serialized
    assert "secret-one" not in serialized
    assert "signed-one" not in serialized


def test_unsigned_expires_query_parameter_remains_material() -> None:
    baseline = {
        "1": {
            "class_type": "ModelProvider",
            "inputs": {
                "endpoint": "https://models.example/model?expires=100&revision=1"
            },
        }
    }
    changed = copy.deepcopy(baseline)
    changed["1"]["inputs"]["endpoint"] = (
        "https://models.example/model?expires=200&revision=1"
    )

    assert capability_hash(baseline) != capability_hash(changed)


def test_capability_hash_ignores_reference_filename_but_not_strategy(qwen_graph) -> None:
    renamed = copy.deepcopy(qwen_graph)
    renamed["74"]["inputs"]["image"] = "different-name.png"
    second_reference = copy.deepcopy(qwen_graph)
    second_reference["75"] = {
        "class_type": "LoadImage",
        "inputs": {"image": "style-reference.png"},
    }
    second_reference["61"]["inputs"]["image1"] = ["75", 0]

    baseline = capability_hash(qwen_graph)

    assert capability_hash(renamed) == baseline
    assert capability_hash(second_reference) != baseline


def test_capability_hash_is_invariant_to_source_order_and_node_renumbering(
    qwen_graph,
) -> None:
    baseline = capability_hash(qwen_graph)
    renumbered = _renumber_graph(qwen_graph)

    assert capability_hash(_reverse_objects(qwen_graph)) == baseline
    assert capability_hash(renumbered) == baseline
    assert source_graph_hash(renumbered) != source_graph_hash(qwen_graph)
    assert exact_run_hash(renumbered, fixtures={}, prompts=[], seed=1) != exact_run_hash(
        qwen_graph,
        fixtures={},
        prompts=[],
        seed=1,
    )


def test_hashes_accept_repo_frozen_json_graphs(qwen_graph) -> None:
    frozen = freeze_json_object(qwen_graph)

    assert source_graph_hash(frozen) == source_graph_hash(qwen_graph)
    assert capability_hash(frozen) == capability_hash(qwen_graph)
    assert exact_run_hash(frozen, fixtures={}, prompts=[], seed=1) == exact_run_hash(
        qwen_graph,
        fixtures={},
        prompts=[],
        seed=1,
    )


@pytest.mark.parametrize(
    "hash_call",
    [
        lambda graph: source_graph_hash(graph),
        lambda graph: exact_run_hash(graph, fixtures={}, prompts=[], seed=1),
        lambda graph: capability_hash(graph),
    ],
)
def test_all_hashes_reject_dangling_links_before_serialization(hash_call) -> None:
    graph = {
        "1": {
            "class_type": "KSampler",
            "inputs": {"model": ["missing", 0], "seed": 1},
        }
    }

    with pytest.raises(GraphValidationError, match="unknown node 'missing'"):
        hash_call(graph)


@pytest.mark.parametrize(
    "graph",
    [
        {"bad\ud800": {"class_type": "PrimitiveNode", "inputs": {}}},
        {"1": {"class_type": "Bad\ud800", "inputs": {}}},
        {"1": {"class_type": "PrimitiveNode", "inputs": {"bad\ud800": 1}}},
        {"1": {"class_type": "PrimitiveNode", "inputs": {"nested": ["bad\ud800"]}}},
    ],
)
@pytest.mark.parametrize(
    "hash_call",
    [
        lambda graph: source_graph_hash(graph),
        lambda graph: exact_run_hash(graph, fixtures={}, prompts=[], seed=1),
        lambda graph: capability_hash(graph),
    ],
)
def test_all_hashes_reject_graph_surrogates_with_validation_error(
    graph,
    hash_call,
) -> None:
    with pytest.raises(GraphValidationError, match="surrogate"):
        hash_call(graph)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fixtures": {"reference": "a" * 63 + "\ud800"}},
        {"prompts": ["nested \ud800 prompt"]},
        {"execution_parameters": {"nested": {"value": "bad\ud800"}}},
        {"execution_parameters": {"bad\ud800": "value"}},
    ],
)
def test_exact_hash_rejects_surrogates_in_non_graph_inputs(qwen_graph, overrides) -> None:
    values = {
        "fixtures": {},
        "prompts": [],
        "seed": 1,
        "execution_parameters": {},
    }
    values.update(overrides)

    with pytest.raises(graph_hashing.HashValidationError, match="surrogate"):
        exact_run_hash(qwen_graph, **values)


@pytest.mark.parametrize(
    ("mutator", "description"),
    [
        (
            lambda graph: graph["106"]["inputs"].__setitem__(
                "unet_name", "different-model.safetensors"
            ),
            "model identity",
        ),
        (
            lambda graph: (
                graph["109"]["inputs"].__setitem__(
                    "lora_name", "next-scene_lora_v1-3000.safetensors"
                ),
                graph["133"]["inputs"].__setitem__(
                    "lora_name",
                    "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
                ),
            ),
            "LoRA order",
        ),
        (
            lambda graph: graph["133"]["inputs"].__setitem__("strength_model", 0.7),
            "LoRA strength bucket",
        ),
        (
            lambda graph: graph["56"]["inputs"].__setitem__(
                "sampler_name", "dpmpp_2m"
            ),
            "sampler",
        ),
        (
            lambda graph: graph["56"]["inputs"].__setitem__("scheduler", "karras"),
            "scheduler",
        ),
        (
            lambda graph: graph["107"]["inputs"].__setitem__(
                "clip_name", "different-encoder.safetensors"
            ),
            "encoder",
        ),
        (
            lambda graph: graph["68"]["inputs"].__setitem__("prompt", ["95", 0]),
            "prompt fanout topology",
        ),
        (
            lambda graph: graph.__setitem__(
                "205",
                {
                    "class_type": "SaveImage",
                    "inputs": {"filename_prefix": "second", "images": ["69", 0]},
                },
            ),
            "output cardinality",
        ),
    ],
)
def test_capability_hash_changes_for_material_capabilities(
    qwen_graph,
    mutator,
    description,
) -> None:
    changed = copy.deepcopy(qwen_graph)
    mutator(changed)

    assert capability_hash(changed) != capability_hash(qwen_graph), description


def test_capability_hash_uses_strength_buckets_not_incidental_float_noise(
    qwen_graph,
) -> None:
    same_bucket = copy.deepcopy(qwen_graph)
    same_bucket["133"]["inputs"]["strength_model"] = 0.9

    assert capability_hash(same_bucket) == capability_hash(qwen_graph)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-2.000001, "negative_extreme"),
        (-2.0, "negative_strong"),
        (-1.250001, "negative_strong"),
        (-1.25, "negative_standard"),
        (-0.750001, "negative_standard"),
        (-0.75, "negative_light"),
        (-0.250001, "negative_light"),
        (-0.25, "negative_subtle"),
        (-0.000001, "negative_subtle"),
        (0, "zero"),
        (0.000001, "positive_subtle"),
        (0.25, "positive_subtle"),
        (0.250001, "positive_light"),
        (0.75, "positive_light"),
        (0.750001, "positive_standard"),
        (1.25, "positive_standard"),
        (1.250001, "positive_strong"),
        (2.0, "positive_strong"),
        (2.000001, "positive_extreme"),
        (10**400, "positive_extreme"),
    ],
)
def test_strength_bucket_boundaries_are_explicit_and_stable(value, expected) -> None:
    assert strength_bucket(value) == expected


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf")])
def test_strength_bucket_rejects_non_numeric_or_nonfinite_values(value) -> None:
    with pytest.raises(ValueError, match="strength"):
        strength_bucket(value)
