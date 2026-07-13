import json

from mp_comfyui_lab.cli.app import ExitCode, main


def test_version_emits_one_json_document(capsys):
    result = main(["version"])

    captured = capsys.readouterr()
    assert result == ExitCode.OK
    assert captured.out == (
        '{"schema_version":"1","ok":true,"command":"version",'
        '"data":{"version":"0.1.0"},"artifacts":[]}\n'
    )
    assert captured.err == ""


def test_missing_command_emits_structured_usage_error(capsys):
    result = main([])

    captured = capsys.readouterr()
    documents = captured.out.splitlines()
    assert result == ExitCode.USAGE
    assert len(documents) == 1
    assert json.loads(documents[0])["ok"] is False
    assert json.loads(documents[0])["error"]["kind"] == "usage"
    assert captured.err == ""
