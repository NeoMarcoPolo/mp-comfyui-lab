from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.runninghub.ai"
GET_WORKFLOW_JSON_ENDPOINT = f"{BASE_URL}/api/openapi/getJsonApiFormat"
TASK_STATUS_ENDPOINT = f"{BASE_URL}/task/openapi/status"
TASK_OUTPUTS_ENDPOINT = f"{BASE_URL}/task/openapi/outputs"
REQUEST_TIMEOUT = (10, 60)


class RunningHubConfigError(RuntimeError):
    pass


class RunningHubApiError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RunningHubConfig:
    apiKey: str


@dataclass(frozen=True)
class NodeInfo:
    nodeId: str
    fieldName: str
    fieldValue: Any


@dataclass(frozen=True)
class TaskCreateResult:
    taskId: str | None
    taskStatus: str | None = None
    clientId: str | None = None
    promptTips: str | None = None


@dataclass(frozen=True)
class TaskOutput:
    fileUrl: str
    fileType: str
    nodeId: str
    taskCostTime: Any = None
    consumeCoins: Any = None


def _validated_api_key(value: object, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise RunningHubConfigError(f"{source} does not contain a valid API key")
    return value


def _load_explicit_config(path: Path) -> RunningHubConfig:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunningHubConfigError(
            "RUNNINGHUB_CONFIG_PATH could not be read as JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise RunningHubConfigError(
            "RUNNINGHUB_CONFIG_PATH must contain a JSON object"
        )
    return RunningHubConfig(
        apiKey=_validated_api_key(
            payload.get("apiKey"),
            "RUNNINGHUB_CONFIG_PATH",
        )
    )


def load_default_config() -> RunningHubConfig:
    """Resolve a key without writing it or requiring a private package."""
    environment_key = os.environ.get("RUNNINGHUB_API_KEY")
    if environment_key is not None:
        return RunningHubConfig(
            apiKey=_validated_api_key(
                environment_key,
                "RUNNINGHUB_API_KEY",
            )
        )

    config_path = os.environ.get("RUNNINGHUB_CONFIG_PATH")
    if config_path is not None:
        if not config_path.strip() or config_path != config_path.strip():
            raise RunningHubConfigError("RUNNINGHUB_CONFIG_PATH is invalid")
        return _load_explicit_config(Path(config_path))

    try:
        mp_keys = importlib.import_module("mp_keys")
    except (ImportError, ModuleNotFoundError):
        mp_keys = None
    if mp_keys is not None:
        try:
            compatibility_key = mp_keys.keys["runninghub"]["api_key"]
        except (AttributeError, KeyError, TypeError, ValueError):
            compatibility_key = None
        if compatibility_key is not None:
            return RunningHubConfig(
                apiKey=_validated_api_key(
                    compatibility_key,
                    "optional mp_keys compatibility",
                )
            )

    raise RunningHubConfigError(
        "RunningHub API key is unavailable; set RUNNINGHUB_API_KEY or "
        "RUNNINGHUB_CONFIG_PATH"
    )


def _resolve_api_key(api_key: str | None) -> str:
    if api_key is not None:
        return _validated_api_key(api_key, "RunningHub transport")
    return load_default_config().apiKey


def _request_identifier(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 512
    ):
        raise RunningHubApiError(f"RunningHub {field_name} is invalid")
    return value


def _post_json(
    endpoint: str,
    body: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout: tuple[int, int] = REQUEST_TIMEOUT,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], Any]:
    try:
        key = _resolve_api_key(api_key)
    except RunningHubConfigError as error:
        raise RunningHubApiError("RunningHub credentials could not be loaded") from error
    post = requests.post if session is None else session.post
    try:
        response = post(
            endpoint,
            json={**body, "apiKey": key},
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.RequestException:
        raise RunningHubApiError("RunningHub request failed") from None

    try:
        envelope = response.json()
    except (TypeError, ValueError):
        raise RunningHubApiError(
            f"RunningHub returned non-JSON response (HTTP {response.status_code})"
        ) from None
    if response.status_code != 200:
        raise RunningHubApiError(
            f"RunningHub request failed: HTTP {response.status_code}"
        )
    if not isinstance(envelope, Mapping):
        raise RunningHubApiError("RunningHub returned a malformed response")
    code = envelope.get("code")
    success_code = (type(code) is int and code == 0) or (
        type(code) is str and code == "0"
    )
    if not success_code:
        provider_code = code if type(code) is int else None
        raise RunningHubApiError(
            "RunningHub API rejected the request",
            code=provider_code,
        )
    normalized = dict(envelope)
    return normalized, normalized.get("data")


def get_workflow_json(
    api_key: str | None,
    workflow_id: str,
    *,
    session: requests.Session | None = None,
) -> str | Mapping[str, Any]:
    _, data = _post_json(
        GET_WORKFLOW_JSON_ENDPOINT,
        {"workflowId": _request_identifier(workflow_id, "workflow ID")},
        api_key=api_key,
        session=session,
    )
    if not isinstance(data, Mapping):
        raise RunningHubApiError("RunningHub workflow response is missing data")
    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    if isinstance(prompt, Mapping):
        return prompt
    raise RunningHubApiError("RunningHub workflow response is missing data.prompt")


def get_task_status(
    api_key: str | None,
    task_id: str,
    *,
    session: requests.Session | None = None,
) -> str:
    _, data = _post_json(
        TASK_STATUS_ENDPOINT,
        {"taskId": _request_identifier(task_id, "task ID")},
        api_key=api_key,
        session=session,
    )
    if not isinstance(data, str) or not data.strip():
        raise RunningHubApiError("RunningHub task status is malformed")
    return data


def get_task_info(
    api_key: str | None,
    task_id: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    envelope, _ = _post_json(
        TASK_STATUS_ENDPOINT,
        {"taskId": _request_identifier(task_id, "task ID")},
        api_key=api_key,
        session=session,
    )
    return envelope


def get_task_outputs(
    api_key: str | None,
    task_id: str,
    *,
    session: requests.Session | None = None,
) -> list[TaskOutput]:
    _, data = _post_json(
        TASK_OUTPUTS_ENDPOINT,
        {"taskId": _request_identifier(task_id, "task ID")},
        api_key=api_key,
        session=session,
    )
    if not isinstance(data, list):
        raise RunningHubApiError("RunningHub outputs must be an array")
    outputs: list[TaskOutput] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise RunningHubApiError("RunningHub output record is malformed")
        file_url = item.get("fileUrl")
        file_type = item.get("fileType")
        node_id = item.get("nodeId")
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (file_url, file_type, node_id)
        ):
            raise RunningHubApiError("RunningHub output record is malformed")
        outputs.append(
            TaskOutput(
                fileUrl=file_url,
                fileType=file_type,
                nodeId=node_id,
                taskCostTime=item.get("taskCostTime"),
                consumeCoins=item.get("consumeCoins"),
            )
        )
    return outputs
