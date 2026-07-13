import json
import os
from pathlib import Path
import subprocess

import pytest

from mp_comfyui_lab.experiments import store as store_module
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import (
    SafetyError,
    safe_filename,
    sanitize_for_persistence,
)


EXPECTED_LAYOUT = {
    "sources",
    "evidence",
    "plans",
    "variants",
    "runs",
    "outputs",
    "comparisons",
    "export",
}


def test_store_creates_complete_layout_and_redacts_secrets(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("qwen-angle-001")
    store.write_json(
        "qwen-angle-001",
        "runs/run-1.json",
        {"apiKey": "secret", "taskId": "42"},
    )
    payload = json.loads((root / "runs/run-1.json").read_text(encoding="utf-8"))
    assert payload == {"apiKey": "[REDACTED]", "taskId": "42"}
    assert EXPECTED_LAYOUT <= {
        path.name for path in root.iterdir() if path.is_dir()
    }


def test_store_root_is_resolved(tmp_path):
    store = ExperimentStore(tmp_path / "parent" / ".." / "runs")

    assert store.root("abc") == (tmp_path / "runs" / "abc").resolve()


@pytest.mark.parametrize(
    "experiment_id",
    [
        "ab",
        "Uppercase",
        "-starts-with-dash",
        "contains/slash",
        "contains space",
        "abc.",
        "con",
        "aux",
        "nul",
        "com1",
        "lpt9",
        "a" * 97,
    ],
)
def test_store_rejects_invalid_experiment_ids(tmp_path, experiment_id):
    store = ExperimentStore(tmp_path)

    with pytest.raises(SafetyError, match="invalid experiment ID"):
        store.root(experiment_id)


def test_sanitize_for_persistence_redacts_nested_secrets_without_mutation():
    value = {
        "Authorization": "Bearer secret",
        "nested": [
            {"refresh_TOKEN": "refresh-secret"},
            ({"Cookie": "session-secret"},),
        ],
    }

    sanitized = sanitize_for_persistence(value)

    assert sanitized == {
        "Authorization": "[REDACTED]",
        "nested": [
            {"refresh_TOKEN": "[REDACTED]"},
            ({"Cookie": "[REDACTED]"},),
        ],
    }
    assert value["Authorization"] == "Bearer secret"
    assert value["nested"][0]["refresh_TOKEN"] == "refresh-secret"
    assert value["nested"][1][0]["Cookie"] == "session-secret"


def test_sanitize_for_persistence_strips_sensitive_url_query():
    value = {
        "outputUrl": (
            "https://cdn.example.test/frame.png?"
            "X-Amz-Credential=credential&X-Amz-Signature=secret#preview"
        )
    }

    assert sanitize_for_persistence(value) == {
        "outputUrl": "https://cdn.example.test/frame.png#preview"
    }


def test_sanitize_for_persistence_strips_blank_sensitive_url_query():
    value = {"outputUrl": "https://cdn.example.test/frame.png?TOKEN="}

    assert sanitize_for_persistence(value) == {
        "outputUrl": "https://cdn.example.test/frame.png"
    }


def test_sanitize_for_persistence_strips_signed_urls_in_collection():
    value = {"outputUrls": ["https://cdn/x?token=secret"]}

    assert sanitize_for_persistence(value) == {"outputUrls": ["https://cdn/x"]}


def test_sanitize_for_persistence_propagates_url_context_without_mutation():
    value = {
        "previewUrls": [
            {
                "items": (
                    "https://cdn/x?signature=secret",
                    ["https://cdn/stable?view=public"],
                )
            }
        ]
    }

    sanitized = sanitize_for_persistence(value)

    assert sanitized == {
        "previewUrls": [
            {
                "items": (
                    "https://cdn/x",
                    ["https://cdn/stable?view=public"],
                )
            }
        ]
    }
    assert value["previewUrls"][0]["items"][0].endswith("?signature=secret")
    assert value["previewUrls"][0]["items"][1][0].endswith("?view=public")


def test_sanitize_for_persistence_preserves_stable_url():
    value = {"source_url": "https://example.test/workflows/42?view=public"}

    assert sanitize_for_persistence(value) == value


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../secret.json",
        "nested/file.json",
        "nested\\file.json",
        "bad<name.json",
        "bad>name.json",
        "bad:name.json",
        'bad"name.json',
        "bad|name.json",
        "bad?name.json",
        "bad*name.json",
        "trailing.",
        "trailing ",
        "CON",
        "CON.txt",
        "bad\x00name",
        "bad\x85name",
    ],
)
def test_safe_filename_rejects_unsafe_names(name):
    with pytest.raises(SafetyError):
        safe_filename(name)


def test_safe_filename_preserves_safe_unicode_name():
    assert safe_filename("comparison-日本語.json") == "comparison-日本語.json"


@pytest.mark.parametrize(
    "component",
    [
        "bad<name.json",
        "bad>name.json",
        "bad:name.json",
        'bad"name.json',
        "bad|name.json",
        "bad?name.json",
        "bad*name.json",
        "trailing.",
        "trailing ",
        "CON.txt",
    ],
)
def test_store_rejects_windows_unsafe_artifact_components(
    tmp_path,
    monkeypatch,
    component,
):
    store = ExperimentStore(tmp_path)
    store.create("safe-001")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("unsafe path reached the filesystem"),
    )

    with pytest.raises(SafetyError, match="unsafe artifact path component"):
        store.read_json("safe-001", Path("runs") / component)


def test_store_validates_every_artifact_path_component(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    store.create("safe-001")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("unsafe path reached the filesystem"),
    )

    with pytest.raises(SafetyError, match="unsafe artifact path component"):
        store.read_json("safe-001", "plans/CON/result.json")


def test_write_json_persists_unicode_sorted_and_with_trailing_newline(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("unicode-001")

    destination = store.write_json(
        "unicode-001",
        "runs/result.json",
        {"z": "café", "a": "日本語"},
    )

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": "日本語",\n  "z": "café"\n}\n'
    )
    assert not list(destination.parent.glob("*.tmp"))


def test_write_text_creates_parent_directories_atomically(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("text-001")

    destination = store.write_text(
        "text-001",
        "plans/nested/plan.txt",
        "angle plan\n",
    )

    assert destination.read_text(encoding="utf-8") == "angle plan\n"
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("relative_path", ["../escape.json", "runs/../../escape.json"])
def test_store_rejects_relative_path_traversal(tmp_path, relative_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="traversal"):
        store.write_json("escape-001", relative_path, {})


def test_store_rejects_absolute_artifact_path(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="relative"):
        store.write_json("escape-001", tmp_path / "outside.json", {})


def test_read_json_rejects_path_escape(tmp_path):
    store = ExperimentStore(tmp_path)
    store.create("escape-001")

    with pytest.raises(SafetyError, match="traversal"):
        store.read_json("escape-001", "../outside.json")


def test_store_rejects_symlink_escape_when_supported(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("symlink-001")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "runs" / "outside"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("symlink-001", "runs/outside/result.txt", "unsafe")


def test_store_rejects_in_root_symlink_leaf_without_modifying_target(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("symlink-001")
    target = root / "sources" / "target.txt"
    target.write_text("original", encoding="utf-8")
    alias = root / "runs" / "alias.txt"
    try:
        alias.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("symlink-001", "runs/alias.txt", "replacement")

    assert target.read_text(encoding="utf-8") == "original"


def test_store_rejects_symlink_experiment_root(tmp_path):
    store = ExperimentStore(tmp_path / "experiments")
    target = store.create("target-001")
    alias = tmp_path / "experiments" / "alias-001"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.root("alias-001")


def test_store_rejects_in_root_junction_without_pathlib_helper(
    tmp_path,
    monkeypatch,
):
    if os.name != "nt":
        pytest.skip("directory junctions are Windows-specific")

    store = ExperimentStore(tmp_path / "experiments")
    root = store.create("junction-001")
    target = root / "sources"
    target_file = target / "target.txt"
    target_file.write_text("original", encoding="utf-8")
    alias = root / "runs" / "alias"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip("directory junctions are not supported")

    monkeypatch.setattr(Path, "is_junction", None, raising=False)

    with pytest.raises(SafetyError, match="symlink or junction"):
        store.write_text("junction-001", "runs/alias/target.txt", "replacement")

    assert target_file.read_text(encoding="utf-8") == "original"


def test_replace_failure_leaves_no_temporary_artifacts(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    root = store.create("replace-001")
    destination = root / "runs" / "result.json"
    destination.write_text("original\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.write_json("replace-001", "runs/result.json", {"ok": True})

    assert destination.read_text(encoding="utf-8") == "original\n"
    assert not list(root.rglob("*.tmp"))


def test_fsync_failure_preserves_destination_and_cleans_temporary_file(
    tmp_path,
    monkeypatch,
):
    store = ExperimentStore(tmp_path)
    root = store.create("fsync-001")
    destination = root / "runs" / "result.txt"
    destination.write_text("original", encoding="utf-8")

    def fail_fsync(file_descriptor):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(store_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated fsync failure"):
        store.write_text("fsync-001", "runs/result.txt", "replacement")

    assert destination.read_text(encoding="utf-8") == "original"
    assert not list(root.rglob("*.tmp"))


def test_serialization_failure_leaves_no_temporary_artifacts(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("serialize-001")

    with pytest.raises(TypeError):
        store.write_json("serialize-001", "runs/result.json", {"bad": object()})

    assert not list(root.rglob("*.tmp"))
