import json

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
        "a" * 97,
    ],
)
def test_store_rejects_invalid_experiment_ids(tmp_path, experiment_id):
    store = ExperimentStore(tmp_path)

    with pytest.raises(SafetyError, match="invalid experiment ID"):
        store.create(experiment_id)


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
        "bad\x00name",
        "bad\x85name",
    ],
)
def test_safe_filename_rejects_unsafe_names(name):
    with pytest.raises(SafetyError):
        safe_filename(name)


def test_safe_filename_preserves_safe_unicode_name():
    assert safe_filename("comparison-日本語.json") == "comparison-日本語.json"


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

    with pytest.raises(SafetyError, match="escapes experiment root"):
        store.write_text("symlink-001", "runs/outside/result.txt", "unsafe")


def test_replace_failure_leaves_no_temporary_artifacts(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    root = store.create("replace-001")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        store.write_json("replace-001", "runs/result.json", {"ok": True})

    assert not list(root.rglob("*.tmp"))


def test_serialization_failure_leaves_no_temporary_artifacts(tmp_path):
    store = ExperimentStore(tmp_path)
    root = store.create("serialize-001")

    with pytest.raises(TypeError):
        store.write_json("serialize-001", "runs/result.json", {"bad": object()})

    assert not list(root.rglob("*.tmp"))
