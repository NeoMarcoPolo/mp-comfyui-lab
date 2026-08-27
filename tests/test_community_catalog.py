from __future__ import annotations

import copy
import json
from pathlib import Path

from mp_comfyui_lab.community.catalog import build_workflow_catalog


def _api_graph(model: str, prompt: str = "portrait", seed: int = 1, width: int = 768) -> dict:
    return {
        "model": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model},
        },
        "text": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": 1024},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["model", 0],
                "positive": ["text", 0],
                "latent_image": ["latent", 0],
                "seed": seed,
                "sampler_name": "euler",
                "scheduler": "simple",
            },
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["sampler", 0], "filename_prefix": "result"},
        },
    }


def _write_source(
    root: Path,
    *,
    message_id: str,
    filename: str,
    graph: dict | None,
    graph_name: str,
    content: str,
    author_id: str = "author",
    image_sha: str | None = None,
) -> dict:
    embedded = []
    if graph is not None:
        graph_path = root / graph_name
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        embedded.append(
            {
                "sha256_canonical_json": graph_name.split("/")[-1].split(".")[0],
                "graph_kind": "api",
                "relative_path": graph_name,
                "metadata_key": "info:prompt",
            }
        )
    ui_name = graph_name.replace("api", "ui")
    ui_path = root / ui_name
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    embedded.append(
        {
            "sha256_canonical_json": ui_name.split("/")[-1].split(".")[0],
            "graph_kind": "ui",
            "relative_path": ui_name,
            "metadata_key": "info:workflow",
        }
    )
    return {
        "attachment_id": f"a-{message_id}",
        "message_id": message_id,
        "channel_id": "channel",
        "timestamp": "2026-08-20T18:30:00+00:00",
        "author_id": author_id,
        "author_name": "creator",
        "message_content": content,
        "filename": filename,
        "message_url": f"https://discord.com/channels/guild/channel/{message_id}",
        "relative_path": f"sources/{message_id}/{filename}",
        "sha256": image_sha or f"image-{message_id}",
        "search_reasons": [],
        "search_queries": [],
        "embedded_graphs": embedded,
    }


def _manifest(root: Path, name: str, images: list[dict]) -> Path:
    path = root / name
    path.write_text(json.dumps({"guild_id": "guild", "images": images}), encoding="utf-8")
    return path


def test_catalog_merges_ui_api_pairs_reposts_and_prompt_only_variants(tmp_path: Path) -> None:
    first_graph = _api_graph("wan2.2_t2v_high.safetensors", prompt="red dress", seed=1, width=768)
    variant = copy.deepcopy(first_graph)
    variant["text"]["inputs"]["text"] = "blue dress, full body"
    variant["sampler"]["inputs"]["seed"] = 999
    variant["latent"]["inputs"]["width"] = 1024
    first = _write_source(
        tmp_path,
        message_id="1",
        filename="wan.png",
        graph=first_graph,
        graph_name="graphs/api-one.json",
        content="Wan 2.2 T2V",
        image_sha="same-image",
    )
    repost = _write_source(
        tmp_path,
        message_id="2",
        filename="wan-repost.png",
        graph=variant,
        graph_name="graphs/api-two.json",
        content="Wan 2.2 T2V repost",
        author_id="other-author",
        image_sha="same-image",
    )

    records = build_workflow_catalog(
        [_manifest(tmp_path, "one.json", [first]), _manifest(tmp_path, "two.json", [repost])],
        graph_root=tmp_path,
    )

    assert len(records) == 1
    assert records[0].identity.family == "wan-2.2"
    assert records[0].identity.task == "text-to-video"
    assert {source.message_id for source in records[0].sources} == {"1", "2"}
    assert len(records[0].api_graph_paths) == 2
    assert len(records[0].ui_graph_paths) == 2


def test_catalog_keeps_versions_tasks_and_material_techniques_distinct(tmp_path: Path) -> None:
    cases = [
        ("1", "wan2.1_t2v.safetensors", "Wan 2.1 T2V", "wan-2.1", "text-to-video", "base"),
        ("2", "wan2.2_t2v.safetensors", "Wan 2.2 T2V", "wan-2.2", "text-to-video", "base"),
        ("3", "wan2.2_i2v.safetensors", "Wan 2.2 I2V", "wan-2.2", "image-to-video", "base"),
        ("4", "wan2.2_t2v.safetensors", "Wan 2.2 SVI Pro", "wan-2.2", "text-to-video", "svi-pro"),
        ("5", "wan2.2_t2v.safetensors", "Wan 2.2 SVI Pro 2", "wan-2.2", "text-to-video", "svi-pro-2"),
        ("6", "anima.safetensors", "SCAIL2 native", "scail2", "text-to-image", "scail2"),
        ("7", "anima.safetensors", "Animate pipeline", "animate", "text-to-video", "animate"),
    ]
    images = [
        _write_source(
            tmp_path,
            message_id=message_id,
            filename=f"{message_id}.png",
            graph=_api_graph(model),
            graph_name=f"graphs/api-{message_id}.json",
            content=content,
        )
        for message_id, model, content, _, _, _ in cases
    ]

    records = build_workflow_catalog([_manifest(tmp_path, "all.json", images)], tmp_path)

    identities = {
        (record.identity.family, record.identity.task, record.identity.technique)
        for record in records
    }
    assert identities == {(family, task, technique) for _, _, _, family, task, technique in cases}


def test_catalog_keeps_ui_only_sources_distinct_and_flags_review(tmp_path: Path) -> None:
    first = _write_source(
        tmp_path,
        message_id="1",
        filename="one.png",
        graph=None,
        graph_name="graphs/api-one.json",
        content="Krea 2 workflow",
    )
    second = _write_source(
        tmp_path,
        message_id="2",
        filename="two.png",
        graph=None,
        graph_name="graphs/api-two.json",
        content="Krea 2 workflow",
    )

    records = build_workflow_catalog([_manifest(tmp_path, "all.json", [first, second])], tmp_path)

    assert len(records) == 2
    assert all(record.identity_review_required for record in records)


def test_catalog_preserves_graph_with_orphan_inactive_lora_branch_for_review(tmp_path: Path) -> None:
    graph = _api_graph("krea2_native_fp8.safetensors")
    graph["orphan"] = {
        "class_type": "WanVideoLoraSelect",
        "inputs": {"lora": "unused.safetensors", "strength": 1.0},
    }
    image = _write_source(
        tmp_path,
        message_id="1",
        filename="krea.png",
        graph=graph,
        graph_name="graphs/api-orphan.json",
        content="Krea 2 workflow",
    )

    records = build_workflow_catalog([_manifest(tmp_path, "all.json", [image])], tmp_path)

    assert len(records) == 1
    assert records[0].identity.family == "krea-2"
    assert records[0].identity.capability_hash.startswith("fallback:")
    assert records[0].identity_review_required is True
