from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mp_comfyui_lab.community.campaign import run_campaign


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph(model: str) -> dict:
    return {
        "model": {"class_type": "UNETLoader", "inputs": {"unet_name": model}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": "exact prompt"}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 1024}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {"model": ["model", 0], "positive": ["text", 0], "latent_image": ["latent", 0], "seed": 42},
        },
        "save": {"class_type": "SaveImage", "inputs": {"images": ["sampler", 0]}},
    }


def _write_inputs(root: Path) -> tuple[Path, Path]:
    graph_path = root / "graphs" / "api.json"
    ui_path = root / "graphs" / "ui.json"
    graph_path.parent.mkdir()
    graph_path.write_text(json.dumps(_graph("krea2_native_fp8.safetensors")), encoding="utf-8")
    ui_path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
    manifest = {
        "guild_id": "guild",
        "images": [
            {
                "attachment_id": "attachment",
                "message_id": "message",
                "channel_id": "channel",
                "timestamp": "2026-08-01T00:00:00+00:00",
                "author_id": "author",
                "author_name": "creator",
                "message_content": "Krea 2 exact workflow",
                "filename": "result.png",
                "message_url": "https://discord.com/channels/guild/channel/message",
                "relative_path": "sources/result.png",
                "sha256": "image-sha",
                "embedded_graphs": [
                    {"graph_kind": "api", "relative_path": "graphs/api.json", "sha256_canonical_json": "api"},
                    {"graph_kind": "ui", "relative_path": "graphs/ui.json", "sha256_canonical_json": "ui"},
                ],
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = {
        "schema_version": 1,
        "guild_id": "guild",
        "complete": True,
        "targets": ["guild/channel/message"],
        "messages": {
            "guild/channel/message": {
                "guild_id": "guild",
                "channel_id": "channel",
                "message_id": "message",
                "canonical_url": "https://discord.com/channels/guild/channel/message",
                "author_id": "author",
                "author_is_bot": False,
                "timestamp": "2026-08-01T00:00:00+00:00",
                "reactions": [{"emoji_id": None, "emoji_name": "⭐", "count": 4, "burst_count": 0}],
            }
        },
        "starboard_snapshots": [],
        "errors": {},
    }
    evidence_path = root / "engagement.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return manifest_path, evidence_path


def test_campaign_writes_deterministic_source_backed_artifacts(tmp_path: Path) -> None:
    manifest, engagement = _write_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    as_of = datetime(2026, 8, 27, tzinfo=timezone.utc)

    first_result = run_campaign([manifest], engagement, tmp_path, first, as_of)
    second_result = run_campaign([manifest], engagement, tmp_path, second, as_of)

    expected = {
        "source-evidence.json",
        "workflow-catalog.json",
        "ranked-workflows.json",
        "selected-test-manifest.json",
        "selection-summary.md",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {_sha(first / name) for name in expected} == {_sha(second / name) for name in expected}
    assert first_result["selected_count"] == 1
    selection = json.loads((first / "selected-test-manifest.json").read_text(encoding="utf-8"))
    assert selection["schema_version"] == 1
    assert selection["selected"][0]["workflow"]["identity"]["family"] == "krea-2"
    assert selection["selected"][0]["workflow"]["api_graph_paths"] == ["graphs/api.json"]
    assert selection["selected"][0]["workflow"]["ui_graph_paths"] == ["graphs/ui.json"]


def test_campaign_fails_closed_for_incomplete_evidence(tmp_path: Path) -> None:
    manifest, engagement = _write_inputs(tmp_path)
    payload = json.loads(engagement.read_text(encoding="utf-8"))
    payload["complete"] = False
    engagement.write_text(json.dumps(payload), encoding="utf-8")

    try:
        run_campaign(
            [manifest],
            engagement,
            tmp_path,
            tmp_path / "output",
            datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
    except ValueError as error:
        assert "complete" in str(error).casefold()
    else:
        raise AssertionError("incomplete evidence must fail closed")
