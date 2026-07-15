from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from mp_comfyui_lab.executors.base import (
    AmbiguousSubmissionError,
    ExecutorError,
    ProviderSchemaError,
    ProviderTaskState,
)
from mp_comfyui_lab.executors.runninghub import (
    UPLOAD_ENDPOINT,
    RunningHubExecutor,
    upload_file_bytes,
)
from mp_comfyui_lab.executors.runninghub_transport import (
    NodeInfo,
    RunningHubApiError,
    RunningHubConfigError,
)
from mp_comfyui_lab.experiments.models import (
    MaterializedVariant,
    ResolvedFixture,
    ResolvedFixtures,
)


def _variant(
    *,
    graph: dict | None = None,
    instance_type: str | None = "plus",
) -> MaterializedVariant:
    return MaterializedVariant(
        variant_id="baseline",
        source_workflow_id="workflow-1",
        graph=graph
        or {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "fixture://0"},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"images": ["1", 0]},
            },
        },
        exact_run_hash="a" * 64,
        capability_hash="b" * 64,
        instance_type=instance_type,
        expected_output_count=1,
        expected_output_node_ids=("9",),
        expected_media_types=("image/png",),
    )


def _fixtures(tmp_path: Path) -> ResolvedFixtures:
    fixture = tmp_path / "portrait.png"
    content = b"fixture"
    fixture.write_bytes(content)
    return ResolvedFixtures(
        items=(
            ResolvedFixture(
                uri="portrait.png",
                path=fixture,
                sha256=hashlib.sha256(content).hexdigest(),
                media_type="image/png",
                binding_role="reference",
                binding_index=0,
            ),
        )
    )


@pytest.mark.parametrize("failure", ["timeout", "transient-status"])
def test_create_task_once_never_retries_ambiguous_transmission(
    failure: str,
) -> None:
    runninghub_module = importlib.import_module(
        "mp_comfyui_lab.executors.runninghub"
    )
    create_once = getattr(runninghub_module, "create_task_once", None)
    assert callable(create_once), "adapter must own a single-attempt create primitive"
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 503

        @staticmethod
        def json():
            return {"code": 503, "msg": "temporarily unavailable", "data": None}

    class Session:
        @staticmethod
        def post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            if failure == "timeout":
                raise requests.Timeout("response lost after transmission")
            return Response()

    with pytest.raises(AmbiguousSubmissionError):
        create_once(
            api_key=None,
            workflow_id="workflow-1",
            node_info_list=[],
            workflow='{"1": {}}',
            instance_type="plus",
            credential_loader=lambda: SimpleNamespace(apiKey="secret"),
            session=Session(),
        )

    assert len(calls) == 1


def test_create_task_once_uses_exact_payload_and_one_post() -> None:
    runninghub_module = importlib.import_module(
        "mp_comfyui_lab.executors.runninghub"
    )
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "code": 0,
                "msg": "success",
                "data": {"taskId": "provider-task-1", "taskStatus": "QUEUED"},
            }

    class Session:
        @staticmethod
        def post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    node = NodeInfo(nodeId="1", fieldName="image", fieldValue="uploaded.png")
    result = runninghub_module.create_task_once(
        api_key=None,
        workflow_id="workflow-1",
        node_info_list=[node],
        workflow='{"1": {}}',
        instance_type="plus",
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        session=Session(),
    )

    assert result.taskId == "provider-task-1"
    assert calls == [
        {
            "url": runninghub_module.CREATE_TASK_ENDPOINT,
            "json": {
                "workflowId": "workflow-1",
                "addMetadata": True,
                "usePersonalQueue": False,
                "apiKey": "secret",
                "nodeInfoList": [
                    {
                        "nodeId": "1",
                        "fieldName": "image",
                        "fieldValue": "uploaded.png",
                    }
                ],
                "workflow": '{"1": {}}',
                "instanceType": "plus",
            },
            "timeout": 60,
            "allow_redirects": False,
        }
    ]


def test_create_task_once_treats_explicit_business_rejection_as_confirmed() -> None:
    runninghub_module = importlib.import_module(
        "mp_comfyui_lab.executors.runninghub"
    )
    calls: list[str] = []

    class Response:
        status_code = 400

        @staticmethod
        def json():
            return {"code": 400, "msg": "invalid parameter", "data": None}

    class Session:
        @staticmethod
        def post(url, **_kwargs):
            calls.append(url)
            return Response()

    with pytest.raises(ExecutorError) as raised:
        runninghub_module.create_task_once(
            api_key="secret",
            workflow_id="workflow-1",
            session=Session(),
        )

    assert not isinstance(raised.value, AmbiguousSubmissionError)
    assert calls == [runninghub_module.CREATE_TASK_ENDPOINT]


def test_construction_and_validation_do_not_touch_credentials_or_paid_calls(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"validation touched {name}")

        return call

    executor = RunningHubExecutor(
        credential_loader=forbidden("credentials"),
        upload_call=forbidden("upload"),
        create_task_call=forbidden("create"),
        status_call=forbidden("status"),
        info_call=forbidden("info"),
        outputs_call=forbidden("outputs"),
    )

    report = executor.validate_variant(_variant(), _fixtures(tmp_path))

    assert report.is_valid
    assert report.errors == ()
    assert calls == []


def test_submit_surfaces_clear_local_missing_key_error(tmp_path: Path) -> None:
    def missing_credentials():
        raise RunningHubConfigError(
            "RunningHub API key is unavailable; set RUNNINGHUB_API_KEY or "
            "RUNNINGHUB_CONFIG_PATH"
        )

    executor = RunningHubExecutor(
        credential_loader=missing_credentials,
        upload_call=lambda **_kwargs: pytest.fail("missing key reached upload"),
        create_task_call=lambda **_kwargs: pytest.fail("missing key reached create"),
    )

    with pytest.raises(ExecutorError, match="RUNNINGHUB_API_KEY"):
        executor.submit(_variant(), _fixtures(tmp_path))


def test_submit_uploads_and_uses_exact_create_task_shape(tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []
    captured_create: dict[str, object] = {}

    def load_credentials():
        calls.append(("credentials", None))
        return SimpleNamespace(apiKey="credential-secret")

    def upload_call(**kwargs):
        calls.append(("upload", kwargs))
        assert kwargs == {
            "api_key": None,
            "content": b"fixture",
            "file_name": "portrait.png",
            "file_type": "image",
        }
        return "uploaded-portrait.png"

    def create_task_call(**kwargs):
        calls.append(("create", kwargs))
        captured_create.update(kwargs)
        return SimpleNamespace(taskId="provider-task-1", taskStatus="QUEUED")

    executor = RunningHubExecutor(
        credential_loader=load_credentials,
        upload_call=upload_call,
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    receipt = executor.submit(_variant(), _fixtures(tmp_path))

    assert receipt.provider_task_id == "provider-task-1"
    assert receipt.provider_state is ProviderTaskState.QUEUED
    assert [item[0] for item in calls] == ["credentials", "upload", "create"]
    assert set(captured_create) == {
        "api_key",
        "workflow_id",
        "node_info_list",
        "workflow",
        "instance_type",
    }
    assert captured_create["api_key"] is None
    assert captured_create["workflow_id"] == "workflow-1"
    assert captured_create["instance_type"] == "plus"
    assert captured_create["node_info_list"] == [
        NodeInfo(
            nodeId="1",
            fieldName="image",
            fieldValue="uploaded-portrait.png",
        )
    ]
    materialized = json.loads(captured_create["workflow"])
    assert materialized["1"]["inputs"]["image"] == "uploaded-portrait.png"
    assert "fixture://0" not in captured_create["workflow"]


@pytest.mark.parametrize("task_status", ["", 7, "FUTURE_PROVIDER_STATE"])
def test_submit_keeps_valid_task_id_when_optional_status_is_malformed(
    tmp_path: Path,
    task_status: object,
) -> None:
    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=lambda **_kwargs: SimpleNamespace(
            taskId="provider-task-1",
            taskStatus=task_status,
        ),
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    receipt = executor.submit(_variant(), _fixtures(tmp_path))

    assert receipt.provider_task_id == "provider-task-1"
    assert receipt.provider_state is None
    metadata = receipt.model_dump(mode="json")["provider_metadata"]
    assert metadata == {
        "warning": "RunningHub create returned an unrecognized optional task status"
    }


def test_submit_binds_snapshot_bytes_before_injected_upload_can_mutate_path(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures(tmp_path)
    approved_bytes = fixtures.items[0].path.read_bytes()
    captured_uploads: list[bytes] = []
    creates: list[str] = []

    def upload_call(*, api_key, content, file_name, file_type):
        assert api_key is None
        assert file_name == "portrait.png"
        assert file_type == "image"
        fixtures.items[0].path.write_bytes(b"attacker replacement")
        captured_uploads.append(content)
        fixtures.items[0].path.write_bytes(approved_bytes)
        return "uploaded-portrait.png"

    def create_task_call(**_kwargs):
        creates.append("create")
        return SimpleNamespace(taskId="provider-task-1", taskStatus="QUEUED")

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=upload_call,
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    receipt = executor.submit(_variant(), fixtures)

    assert receipt.provider_task_id == "provider-task-1"
    assert captured_uploads == [approved_bytes]
    assert creates == ["create"]


def test_submit_rejects_tampered_fixture_before_credentials_or_create(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures(tmp_path)
    fixtures.items[0].path.write_bytes(b"tampered snapshot")
    calls: list[str] = []

    def forbidden(name: str):
        def call(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"tampered snapshot touched {name}")

        return call

    executor = RunningHubExecutor(
        credential_loader=forbidden("credentials"),
        upload_call=forbidden("upload"),
        create_task_call=forbidden("create"),
        status_call=forbidden("status"),
        info_call=forbidden("info"),
        outputs_call=forbidden("outputs"),
    )

    with pytest.raises(ExecutorError, match="fixture snapshot"):
        executor.submit(_variant(), fixtures)

    assert calls == []


def test_submit_binds_snapshot_before_credentials_can_mutate_path(
    tmp_path: Path,
) -> None:
    fixtures = _fixtures(tmp_path)
    calls: list[str] = []
    uploaded: list[bytes] = []

    def load_credentials():
        calls.append("credentials")
        fixtures.items[0].path.write_bytes(b"changed after credential gate")
        return SimpleNamespace(apiKey="secret")

    def upload_call(**kwargs):
        calls.append("upload")
        uploaded.append(kwargs["content"])
        return "uploaded.png"

    def create_task_call(**_kwargs):
        calls.append("create")
        return SimpleNamespace(taskId="provider-task-1", taskStatus="QUEUED")

    executor = RunningHubExecutor(
        credential_loader=load_credentials,
        upload_call=upload_call,
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    receipt = executor.submit(_variant(), fixtures)

    assert receipt.provider_task_id == "provider-task-1"
    assert uploaded == [b"fixture"]
    assert calls == ["credentials", "upload", "create"]


def test_upload_file_bytes_sends_exact_multipart_content() -> None:
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"code": 0, "data": {"fileName": "provider-file.png"}}

    class Session:
        @staticmethod
        def post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    content = b"approved immutable bytes"

    uploaded_name = upload_file_bytes(
        "api-secret",
        content,
        "portrait.png",
        "image",
        session=Session(),
    )

    assert uploaded_name == "provider-file.png"
    assert calls == [
        {
            "url": UPLOAD_ENDPOINT,
            "data": {"apiKey": "api-secret", "fileType": "image"},
            "files": {"file": ("portrait.png", content)},
            "timeout": 300,
        }
    ]
    assert type(calls[0]["files"]["file"][1]) is bytes


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"fileName": "provider-file.png"}},
        {"code": True, "data": {"fileName": "provider-file.png"}},
        {"code": 0.0, "data": {"fileName": "provider-file.png"}},
        {"code": 0, "data": []},
        {"code": 0, "data": {"fileName": 7}},
        {"code": 0, "data": {"fileName": "../provider-file.png"}},
    ],
)
def test_upload_file_bytes_strictly_rejects_malformed_provider_response(
    payload: object,
) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return payload

    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response()

    with pytest.raises(RunningHubApiError) as raised:
        upload_file_bytes(
            "api-secret",
            b"approved immutable bytes",
            "portrait.png",
            "image",
            session=Session(),
        )

    assert "api-secret" not in str(raised.value)


def test_poll_maps_known_status_and_sanitizes_info() -> None:
    calls: list[tuple[str, object]] = []

    def load_credentials():
        calls.append(("credentials", None))
        return SimpleNamespace(apiKey="secret")

    def status_call(**kwargs):
        calls.append(("status", kwargs))
        return "RUNNING"

    def info_call(**kwargs):
        calls.append(("info", kwargs))
        return {
            "taskId": "provider-task-1",
            "msg": "preview https://cdn.example/a.png?token=live-secret",
            "authorization": "Bearer credential-secret",
        }

    executor = RunningHubExecutor(
        credential_loader=load_credentials,
        upload_call=lambda **_kwargs: None,
        create_task_call=lambda **_kwargs: None,
        status_call=status_call,
        info_call=info_call,
        outputs_call=lambda **_kwargs: None,
    )

    result = executor.poll("provider-task-1")

    assert result.provider_task_id == "provider-task-1"
    assert result.state is ProviderTaskState.RUNNING
    serialized = json.dumps(result.model_dump(mode="json"))
    assert "live-secret" not in serialized
    assert "credential-secret" not in serialized
    assert [item[0] for item in calls] == ["credentials", "status", "info"]
    assert calls[1][1] == {
        "api_key": None,
        "task_id": "provider-task-1",
    }
    assert calls[2][1] == calls[1][1]


def test_collect_outputs_preserves_provider_order_and_maps_media() -> None:
    calls: list[tuple[str, object]] = []
    raw_outputs = [
        SimpleNamespace(
            fileUrl="https://cdn.example/second.mp4?sig=two",
            fileType="mp4",
            nodeId="12",
            taskCostTime="2",
            consumeCoins="3",
        ),
        SimpleNamespace(
            fileUrl="https://cdn.example/first.png?sig=one",
            fileType="image/png",
            nodeId="9",
            taskCostTime="1",
            consumeCoins="2",
        ),
    ]

    def load_credentials():
        calls.append(("credentials", None))
        return SimpleNamespace(apiKey="secret")

    def outputs_call(**kwargs):
        calls.append(("outputs", kwargs))
        return raw_outputs

    executor = RunningHubExecutor(
        credential_loader=load_credentials,
        upload_call=lambda **_kwargs: None,
        create_task_call=lambda **_kwargs: None,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=outputs_call,
        output_host_suffixes=("cdn.example",),
    )

    outputs = executor.collect_outputs("provider-task-1")

    assert [item.node_id for item in outputs] == ["12", "9"]
    assert [item.provider_order for item in outputs] == [0, 1]
    assert [item.media_type for item in outputs] == ["video/mp4", "image/png"]
    assert [item.file_name for item in outputs] == ["second.mp4", "first.png"]
    assert outputs[0].url.endswith("?sig=two")
    assert calls == [
        ("credentials", None),
        (
            "outputs",
            {"api_key": None, "task_id": "provider-task-1"},
        ),
    ]


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.xiaoyaoyou.com/output.png",
        "https://user:pass@cdn.xiaoyaoyou.com/output.png",
        "https://127.0.0.1/output.png",
        "https://[::1]/output.png",
        "https://runninghub.ai.evil.example/output.png",
        "https://runninghub.ai%2eevil.example/output.png",
        "https://2130706433/output.png",
    ],
)
def test_collect_outputs_rejects_urls_outside_provider_host_policy(
    url: str,
) -> None:
    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: None,
        create_task_call=lambda **_kwargs: None,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: [
            SimpleNamespace(
                fileUrl=url,
                fileType="image/png",
                nodeId="9",
            )
        ],
    )

    with pytest.raises(ProviderSchemaError, match="output URL"):
        executor.collect_outputs("provider-task-1")


@pytest.mark.parametrize(
    "url",
    [
        "https://runninghub.ai/output.png",
        "https://cdn.runninghub.ai/output.png",
        "https://xiaoyaoyou.com/output.png",
        "https://assets.xiaoyaoyou.com/output.png",
    ],
)
def test_collect_outputs_accepts_default_provider_hosts(url: str) -> None:
    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: None,
        create_task_call=lambda **_kwargs: None,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: [
            SimpleNamespace(
                fileUrl=url,
                fileType="image/png",
                nodeId="9",
            )
        ],
    )

    assert executor.collect_outputs("provider-task-1")[0].url == url


def test_cancel_is_structured_unsupported_without_provider_call() -> None:
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("provider")
        raise AssertionError("cancel touched a provider callable")

    executor = RunningHubExecutor(
        credential_loader=forbidden,
        upload_call=forbidden,
        create_task_call=forbidden,
        status_call=forbidden,
        info_call=forbidden,
        outputs_call=forbidden,
    )

    result = executor.cancel("provider-task-1")

    assert result.supported is False
    assert result.cancelled is False
    assert result.diagnostic == (
        "RunningHub API does not expose a supported cancellation operation"
    )
    assert calls == []


def test_confirmed_http_rejection_is_not_ambiguous(tmp_path: Path) -> None:
    response = requests.Response()
    response.status_code = 400
    rejection = requests.HTTPError("bad request", response=response)
    creates: list[str] = []

    def create_task_call(**_kwargs):
        creates.append("create")
        raise rejection

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    with pytest.raises(ExecutorError) as raised:
        executor.submit(_variant(), _fixtures(tmp_path))

    assert not isinstance(raised.value, AmbiguousSubmissionError)
    assert creates == ["create"]


@pytest.mark.parametrize(
    ("graph", "expected_code"),
    [
        (
            {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}},
            "ambiguous_fixture_placeholder",
        ),
        (
            {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "fixture://0"},
                },
                "2": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "fixture://0"},
                },
            },
            "ambiguous_fixture_placeholder",
        ),
        (
            {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "fixture://1"},
                }
            },
            "missing_fixture_binding",
        ),
    ],
)
def test_placeholder_binding_errors_are_local_before_credentials(
    tmp_path: Path,
    graph,
    expected_code,
) -> None:
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("paid")
        raise AssertionError("invalid placeholder touched paid boundary")

    executor = RunningHubExecutor(
        credential_loader=forbidden,
        upload_call=forbidden,
        create_task_call=forbidden,
        status_call=forbidden,
        info_call=forbidden,
        outputs_call=forbidden,
    )

    report = executor.validate_variant(
        _variant(graph=graph),
        _fixtures(tmp_path),
    )

    assert expected_code in {item.code for item in report.errors}
    assert calls == []


def test_submit_timeout_is_ambiguous_and_create_is_not_retried(
    tmp_path: Path,
) -> None:
    creates: list[str] = []

    def create_task_call(**_kwargs):
        creates.append("create")
        raise requests.Timeout("request outcome unknown")

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    with pytest.raises(AmbiguousSubmissionError):
        executor.submit(_variant(), _fixtures(tmp_path))

    assert creates == ["create"]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(taskStatus="QUEUED"),
        SimpleNamespace(taskId=None, taskStatus="QUEUED"),
        SimpleNamespace(taskId="", taskStatus="QUEUED"),
        SimpleNamespace(taskId="   ", taskStatus="QUEUED"),
        SimpleNamespace(taskId="None", taskStatus="QUEUED"),
        SimpleNamespace(taskId="task\nid", taskStatus="QUEUED"),
        SimpleNamespace(taskId="task id", taskStatus="QUEUED"),
        SimpleNamespace(taskId="x" * 513, taskStatus="QUEUED"),
    ],
)
def test_submit_invalid_provider_task_id_is_ambiguous_after_create(
    tmp_path: Path,
    result: SimpleNamespace,
) -> None:
    creates: list[str] = []

    def create_task_call(**_kwargs):
        creates.append("create")
        return result

    executor = RunningHubExecutor(
        credential_loader=lambda: SimpleNamespace(apiKey="secret"),
        upload_call=lambda **_kwargs: "uploaded.png",
        create_task_call=create_task_call,
        status_call=lambda **_kwargs: None,
        info_call=lambda **_kwargs: None,
        outputs_call=lambda **_kwargs: None,
    )

    with pytest.raises(AmbiguousSubmissionError, match="no usable task ID"):
        executor.submit(_variant(), _fixtures(tmp_path))

    assert creates == ["create"]
