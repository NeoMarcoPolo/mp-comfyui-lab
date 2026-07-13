import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import mp_comfyui_lab
import pytest

from mp_comfyui_lab.cli import app
from mp_comfyui_lab.cli.app import ExitCode, main


VERSION_DOCUMENT = (
    '{"schema_version":"1","ok":true,"command":"version",'
    '"data":{"version":"0.1.0"},"artifacts":[]}\n'
)


def _single_json_document(captured):
    documents = captured.out.splitlines()
    assert len(documents) == 1
    assert captured.err == ""
    return json.loads(documents[0])


def test_version_emits_one_json_document(capsys):
    result = main(["version"])

    captured = capsys.readouterr()
    assert result == ExitCode.OK
    assert captured.out == VERSION_DOCUMENT
    assert captured.err == ""


def test_missing_command_emits_structured_usage_error(capsys):
    result = main([])

    captured = capsys.readouterr()
    document = _single_json_document(captured)
    assert result == ExitCode.USAGE
    assert document["ok"] is False
    assert document["error"]["kind"] == "usage"


def test_root_help_emits_structured_success(capsys):
    result = main(["--help"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.OK
    assert document["ok"] is True
    assert document["command"] == "help"
    assert "usage: mp-comfy-lab" in document["data"]["text"]


def test_version_help_emits_structured_success(capsys):
    result = main(["version", "--help"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.OK
    assert document["ok"] is True
    assert document["command"] == "help"
    assert "usage: mp-comfy-lab version" in document["data"]["text"]


def test_unknown_command_preserves_argparse_diagnostic(capsys):
    result = main(["unknown"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.USAGE
    assert document["error"]["message"] == (
        "argument command: invalid choice: 'unknown' (choose from version)"
    )


def test_root_unknown_option_does_not_become_command(capsys):
    result = main(["--bogus"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.USAGE
    assert document["command"] is None
    assert document["error"]["message"] == (
        "the following arguments are required: command"
    )


def test_entrypoint_exits_with_main_result(monkeypatch):
    monkeypatch.setattr(app, "main", lambda: ExitCode.SAFETY)

    with pytest.raises(SystemExit) as raised:
        app.entrypoint()

    assert raised.value.code == int(ExitCode.SAFETY)


def test_installed_launcher_emits_version_document():
    launcher_name = "mp-comfy-lab.exe" if sys.platform == "win32" else "mp-comfy-lab"
    launcher = Path(sys.executable).with_name(launcher_name)

    completed = subprocess.run(
        [launcher, "version"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == VERSION_DOCUMENT
    assert completed.stderr == ""


def test_package_version_is_loaded_from_distribution_metadata(monkeypatch):
    def fake_version(distribution_name):
        assert distribution_name == "mp-comfyui-lab"
        return "9.9.9"

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", fake_version)
        reloaded = importlib.reload(mp_comfyui_lab)
        assert reloaded.__version__ == "9.9.9"

    importlib.reload(mp_comfyui_lab)
