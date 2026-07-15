from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mp_comfyui_lab.executors import runninghub_transport as transport


class _Response:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


class _Session:
    def __init__(self, *responses: _Response) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_key_resolution_prefers_environment_then_explicit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "runninghub.local.json"
    config_path.write_text(json.dumps({"apiKey": "config-secret"}), encoding="utf-8")
    monkeypatch.setenv("RUNNINGHUB_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "environment-secret")

    assert transport.load_default_config().apiKey == "environment-secret"

    monkeypatch.delenv("RUNNINGHUB_API_KEY")
    assert transport.load_default_config().apiKey == "config-secret"


def test_key_resolution_uses_lazy_mp_keys_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)
    monkeypatch.delenv("RUNNINGHUB_CONFIG_PATH", raising=False)
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        return SimpleNamespace(
            keys={"runninghub": {"api_key": "compatibility-secret"}}
        )

    monkeypatch.setattr(transport.importlib, "import_module", import_module)

    assert transport.load_default_config().apiKey == "compatibility-secret"
    assert imported == ["mp_keys"]


def test_key_resolution_fails_clearly_without_any_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)
    monkeypatch.delenv("RUNNINGHUB_CONFIG_PATH", raising=False)
    monkeypatch.setattr(
        transport.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )

    with pytest.raises(transport.RunningHubConfigError) as raised:
        transport.load_default_config()

    message = str(raised.value)
    assert "RUNNINGHUB_API_KEY" in message
    assert "RUNNINGHUB_CONFIG_PATH" in message


def test_transport_calls_are_strict_and_return_local_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "transport-secret")
    graph = '{"1":{"class_type":"SaveImage","inputs":{}}}'
    session = _Session(
        _Response({"code": 0, "data": {"prompt": graph}}),
        _Response({"code": "0", "data": "SUCCESS"}),
        _Response({"code": 0, "data": "SUCCESS", "taskId": "task-1"}),
        _Response(
            {
                "code": 0,
                "data": [
                    {
                        "fileUrl": "https://runninghub.ai/output.png",
                        "fileType": "png",
                        "nodeId": "9",
                        "taskCostTime": "2",
                        "consumeCoins": "3",
                    }
                ],
            }
        ),
    )

    assert transport.get_workflow_json(None, "workflow-1", session=session) == graph
    assert transport.get_task_status(None, "task-1", session=session) == "SUCCESS"
    assert transport.get_task_info(None, "task-1", session=session) == {
        "code": 0,
        "data": "SUCCESS",
        "taskId": "task-1",
    }
    outputs = transport.get_task_outputs(None, "task-1", session=session)

    assert outputs == [
        transport.TaskOutput(
            fileUrl="https://runninghub.ai/output.png",
            fileType="png",
            taskCostTime="2",
            nodeId="9",
            consumeCoins="3",
        )
    ]
    assert [call["url"] for call in session.calls] == [
        transport.GET_WORKFLOW_JSON_ENDPOINT,
        transport.TASK_STATUS_ENDPOINT,
        transport.TASK_STATUS_ENDPOINT,
        transport.TASK_OUTPUTS_ENDPOINT,
    ]
    assert session.calls[0]["json"] == {
        "workflowId": "workflow-1",
        "apiKey": "transport-secret",
    }
    assert all(call["allow_redirects"] is False for call in session.calls)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"code": True, "data": "SUCCESS"},
        {"code": 0.0, "data": "SUCCESS"},
        {"code": 17, "msg": "rejected", "data": None},
        {"code": 0, "data": None},
    ],
)
def test_status_transport_rejects_malformed_or_rejected_envelopes(
    payload: object,
) -> None:
    session = _Session(_Response(payload))

    with pytest.raises(transport.RunningHubApiError) as raised:
        transport.get_task_status("api-secret", "task-1", session=session)

    assert "api-secret" not in str(raised.value)


@pytest.mark.parametrize(
    "data",
    [
        {},
        ["not-an-object"],
        [{"fileUrl": 7, "fileType": "png", "nodeId": "9"}],
    ],
)
def test_outputs_transport_rejects_malformed_records(data: object) -> None:
    session = _Session(_Response({"code": 0, "data": data}))

    with pytest.raises(transport.RunningHubApiError):
        transport.get_task_outputs("api-secret", "task-1", session=session)
