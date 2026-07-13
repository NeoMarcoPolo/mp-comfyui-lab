from __future__ import annotations

from copy import deepcopy
from datetime import UTC
import json
from pathlib import Path
from typing import Any

import pytest
import requests
import responses

from mp_comfyui_lab.discovery.models import ContentClassification
from mp_comfyui_lab.discovery.urls import (
    ParsedRunningHubUrl,
    UnsupportedSourceError,
    parse_runninghub_url,
)
from mp_comfyui_lab.executors.base import (
    Executor,
    ExecutorError,
    ProviderSchemaError,
    SearchRequest,
    UnsupportedCapabilityError,
    WorkflowRef,
)
from mp_comfyui_lab.executors.runninghub import RunningHubExecutor
from mp_comfyui_lab.json_types import FrozenDict, thaw_json


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "runninghub"
WORKFLOW_ID = "1980860923768426497"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


class GraphFetcherSpy:
    def __init__(self, value: Any = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[Any, str]] = []

    def __call__(self, api_key: Any, workflow_id: str) -> Any:
        self.calls.append((api_key, workflow_id))
        if self.error is not None:
            raise self.error
        return self.value


def _executor(graph_fetcher: GraphFetcherSpy | None = None) -> RunningHubExecutor:
    return RunningHubExecutor(
        session=requests.Session(),
        graph_fetcher=graph_fetcher or GraphFetcherSpy(),
    )


def _add_json(
    mocked: responses.RequestsMock,
    url: str,
    payload: dict[str, Any],
) -> None:
    mocked.add(responses.POST, url, json=payload, status=200)


@pytest.mark.parametrize(
    ("url", "kind", "canonical"),
    [
        (
            f"https://www.runninghub.ai/post/{WORKFLOW_ID}",
            "post",
            f"https://www.runninghub.ai/post/{WORKFLOW_ID}",
        ),
        (
            f"https://www.runninghub.ai/zh-cn/post/{WORKFLOW_ID}",
            "post",
            f"https://www.runninghub.ai/post/{WORKFLOW_ID}",
        ),
        (
            f"https://www.runninghub.ai/workflow/{WORKFLOW_ID}",
            "workflow",
            f"https://www.runninghub.ai/workflow/{WORKFLOW_ID}",
        ),
        (
            f"https://runninghub.ai/en-us/workflow/{WORKFLOW_ID}/?from=search",
            "workflow",
            f"https://www.runninghub.ai/workflow/{WORKFLOW_ID}",
        ),
    ],
)
def test_parse_runninghub_url_accepts_exact_public_urls(
    url: str,
    kind: str,
    canonical: str,
) -> None:
    parsed = parse_runninghub_url(url)

    assert isinstance(parsed, ParsedRunningHubUrl)
    assert parsed.workflow_id == WORKFLOW_ID
    assert parsed.kind == kind
    assert parsed.canonical_url == canonical


@pytest.mark.parametrize(
    "url",
    [
        f"http://www.runninghub.ai/post/{WORKFLOW_ID}",
        f"https://example.com/post/{WORKFLOW_ID}",
        f"https://evil.runninghub.ai/post/{WORKFLOW_ID}",
        f"https://www.runninghub.ai:443/post/{WORKFLOW_ID}",
        f"https://user:password@www.runninghub.ai/post/{WORKFLOW_ID}",
        "https://www.runninghub.ai/user-center/saved",
        "https://www.runninghub.ai/account/workflows",
        "https://www.runninghub.ai/zh-cn/user-center/saved",
        "https://www.runninghub.ai/post/not-a-number",
        "https://www.runninghub.ai/post/",
        f"https://www.runninghub.ai/post/{WORKFLOW_ID}/edit",
        f"https://www.runninghub.ai/workflow/{WORKFLOW_ID}/extra",
        f"https://www.runninghub.ai/post/{WORKFLOW_ID}#comments",
        f"https://www.runninghub.ai/not-a-locale/post/{WORKFLOW_ID}",
        f" https://www.runninghub.ai/post/{WORKFLOW_ID}",
        f"https://www.runninghub.ai/post/{WORKFLOW_ID} ",
        f"\nhttps://www.runninghub.ai/post/{WORKFLOW_ID}",
        f"https://www.runninghub.ai/post/{WORKFLOW_ID}\r\n",
        f"https://www.runninghub.ai/\tpost/{WORKFLOW_ID}",
        f"\x00https://www.runninghub.ai/post/{WORKFLOW_ID}",
        f"https://www.runninghub.ai/post/\x1f{WORKFLOW_ID}",
        f"https://www.runninghub.ai/post/{WORKFLOW_ID}\x7f",
    ],
)
def test_parse_runninghub_url_rejects_nonpublic_or_malformed_urls(url: str) -> None:
    with pytest.raises(UnsupportedSourceError, match="public post or workflow"):
        parse_runninghub_url(url)


def test_search_posts_exact_shape_and_maps_observed_response() -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    graph_fetcher = GraphFetcherSpy(error=AssertionError("must not fetch graph"))
    executor = _executor(graph_fetcher)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        page = executor.search(
            SearchRequest(query="qwen", sort="REPUTATION", page=1, size=3)
        )

        request_body = json.loads(mocked.calls[0].request.body)

    assert request_body == {
        "size": 3,
        "current": 1,
        "search": "qwen",
        "tags": [],
        "sort": "REPUTATION",
    }
    assert page.total == 15072
    assert page.page == 1
    assert page.size == 3
    assert page.has_next is True
    assert len(page.records) == 3

    first = page.records[0]
    assert first.workflow_id == "2004195228674387969"
    assert first.engagement.uses == 11977
    assert first.engagement.likes == 139
    assert first.engagement.collects == 688
    assert first.covers[0].width == 1262
    assert first.author_name == "\u65e0\u6781"
    assert first.published_at.tzinfo is UTC
    assert first.canonical_url == (
        "https://www.runninghub.ai/post/2004195228674387969"
    )
    assert first.content_classification is ContentClassification.UNKNOWN
    assert first.classification_reasons == ("no explicit classification evidence",)
    assert graph_fetcher.calls == []


def test_public_detail_maps_inventory_and_strips_executable_html() -> None:
    payload = _load_fixture("detail_next_scene.json")
    payload["data"]["desc"] = (
        "<p>First &amp; safe</p><script>alert('secret')</script><p>Second</p>"
    )
    graph_fetcher = GraphFetcherSpy(error=AssertionError("must not fetch graph"))
    executor = _executor(graph_fetcher)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, payload)
        detail = executor.detail(WORKFLOW_ID)

    assert detail.candidate.description == "First & safe Second"
    assert "<" not in detail.candidate.description
    assert "alert" not in detail.candidate.description
    assert detail.node_count == 19
    assert "KSampler" in detail.primitive_nodes
    assert "ShellAgentPluginOutputText" in detail.custom_nodes
    assert "next-scene_lora_v1-3000.safetensors" in detail.used_models
    assert detail.preview is not None
    assert detail.preview.width == 4050
    assert detail.webapp_id == "1980884267800678402"
    assert detail.candidate.engagement.uses == 25757
    assert detail.candidate.published_at.tzinfo is UTC
    assert graph_fetcher.calls == []


@pytest.mark.parametrize(
    ("name", "description", "used_models", "expected"),
    [
        (
            "Safe for work portrait",
            "A public workflow",
            [],
            ContentClassification.SFW,
        ),
        (
            "Portrait workflow",
            "<p>Contains nudity</p>",
            [],
            ContentClassification.NSFW,
        ),
        (
            "Portrait workflow",
            "No classification label",
            [],
            ContentClassification.UNKNOWN,
        ),
        (
            "SFW portrait workflow",
            "No explicit unsafe prose",
            ["portrait_nsfw_v1.safetensors"],
            ContentClassification.NSFW,
        ),
    ],
)
def test_content_classification_uses_explicit_evidence_with_nsfw_precedence(
    name: str,
    description: str,
    used_models: list[str],
    expected: ContentClassification,
) -> None:
    payload = _load_fixture("detail_next_scene.json")
    payload["data"].update(
        name=name,
        desc=description,
        tags=[],
        usedModels=used_models,
    )
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, payload)
        candidate = executor.detail(WORKFLOW_ID).candidate

    assert candidate.content_classification is expected
    if expected is ContentClassification.UNKNOWN:
        assert candidate.classification_reasons == (
            "no explicit classification evidence",
        )
    else:
        assert candidate.classification_reasons
        assert any(expected.value in reason for reason in candidate.classification_reasons)


@pytest.mark.parametrize("as_string", [True, False])
def test_fetch_workflow_accepts_string_or_dict_graph(as_string: bool) -> None:
    graph = _load_fixture("qwen_graph.json")
    graph_value = json.dumps(graph) if as_string else graph
    graph_fetcher = GraphFetcherSpy(graph_value)
    executor = _executor(graph_fetcher)
    ref = WorkflowRef(
        executor="runninghub",
        workflow_id=WORKFLOW_ID,
        canonical_url=f"https://www.runninghub.ai/post/{WORKFLOW_ID}",
    )

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        fetched = executor.fetch_workflow(ref)

    assert isinstance(fetched.graph, FrozenDict)
    assert fetched.graph["56"]["class_type"] == "KSampler"
    assert fetched.ref == ref
    assert fetched.detail.candidate.canonical_url == ref.canonical_url
    assert graph_fetcher.calls == [(None, WORKFLOW_ID)]


@pytest.mark.parametrize(
    "graph_value",
    [None, [], "[]", "not-json", "null", 42],
)
def test_fetch_workflow_rejects_malformed_or_nondict_graphs(
    graph_value: Any,
) -> None:
    executor = _executor(GraphFetcherSpy(graph_value))
    ref = WorkflowRef(executor="runninghub", workflow_id=WORKFLOW_ID)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        with pytest.raises(
            ProviderSchemaError,
            match=r"openapi/getJsonApiFormat.*graph",
        ):
            executor.fetch_workflow(ref)


@pytest.mark.parametrize("kind", ["post", "workflow"])
def test_fetched_workflow_identity_validator_passes_for_public_url_kinds(
    kind: str,
) -> None:
    canonical_url = f"https://www.runninghub.ai/{kind}/{WORKFLOW_ID}"
    ref = WorkflowRef(
        executor="runninghub",
        workflow_id=WORKFLOW_ID,
        canonical_url=canonical_url,
    )
    executor = _executor(GraphFetcherSpy(_load_fixture("qwen_graph.json")))

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        fetched = executor.fetch_workflow(ref)

    assert fetched.detail.candidate.canonical_url == canonical_url
    assert fetched.ref.canonical_url == fetched.detail.candidate.canonical_url


@pytest.mark.parametrize(
    ("source_url", "canonical_url"),
    [
        (
            f"https://www.runninghub.ai/zh-cn/post/{WORKFLOW_ID}/"
            "?sv=2025-01-05&sp=r&sig=secret",
            f"https://www.runninghub.ai/post/{WORKFLOW_ID}",
        ),
        (
            f"https://runninghub.ai/en-us/workflow/{WORKFLOW_ID}"
            "?from=search&credential=secret",
            f"https://www.runninghub.ai/workflow/{WORKFLOW_ID}",
        ),
    ],
)
def test_fetch_workflow_canonicalizes_accepted_locale_and_query_source_urls(
    source_url: str,
    canonical_url: str,
) -> None:
    executor = _executor(GraphFetcherSpy(_load_fixture("qwen_graph.json")))
    ref = WorkflowRef(
        executor="runninghub",
        workflow_id=WORKFLOW_ID,
        canonical_url=source_url,
    )

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        fetched = executor.fetch_workflow(ref)

    assert fetched.ref.canonical_url == canonical_url
    assert fetched.detail.candidate.canonical_url == canonical_url
    assert "?" not in fetched.ref.canonical_url
    assert "secret" not in fetched.ref.canonical_url


@pytest.mark.parametrize(
    "canonical_url",
    [
        f"https://attacker.example/post/{WORKFLOW_ID}",
        f"http://www.runninghub.ai/post/{WORKFLOW_ID}",
        "https://www.runninghub.ai/workflow/9999999999999999999",
    ],
)
def test_fetch_workflow_rejects_untrusted_or_mismatched_canonical_url(
    canonical_url: str,
) -> None:
    graph_fetcher = GraphFetcherSpy(_load_fixture("qwen_graph.json"))
    executor = _executor(graph_fetcher)
    ref = WorkflowRef(
        executor="runninghub",
        workflow_id=WORKFLOW_ID,
        canonical_url=canonical_url,
    )

    with responses.RequestsMock(assert_all_requests_are_fired=False):
        with pytest.raises(UnsupportedSourceError, match="public post or workflow"):
            executor.fetch_workflow(ref)

    assert graph_fetcher.calls == []


def test_inspect_capabilities_uses_only_deterministic_public_inventories() -> None:
    executor = _executor(GraphFetcherSpy(_load_fixture("qwen_graph.json")))
    ref = WorkflowRef(executor="runninghub", workflow_id=WORKFLOW_ID)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        fetched = executor.fetch_workflow(ref)

    report = executor.inspect_capabilities(fetched)

    assert report.required_nodes == tuple(sorted(report.required_nodes))
    assert set(report.required_nodes) == set(
        fetched.detail.primitive_nodes + fetched.detail.custom_nodes
    )
    assert report.lora_names == ("next-scene_lora_v1-3000.safetensors",)
    assert "next-scene_lora_v1-3000.safetensors" not in report.model_names
    assert report.output_node_ids == ()
    assert thaw_json(report.provider_metadata) == {
        "inventory_source": "public detail",
        "output_node_class_hints": [
            "SaveImage",
            "ShellAgentPluginOutputText",
        ],
    }


def test_non_json_public_body_names_search_endpoint() -> None:
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        mocked.add(
            responses.POST,
            executor.search_url,
            body="<html>not json</html>",
            status=200,
            content_type="text/html",
        )
        with pytest.raises(
            ProviderSchemaError,
            match=r"api/search/workflow.*response JSON",
        ):
            executor.search(SearchRequest(query="qwen", size=3))


def test_nonzero_search_code_names_endpoint_and_code_field() -> None:
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(
            mocked,
            executor.search_url,
            {
                "code": 500,
                "msg": "failed",
                "errorMessages": ["provider failure"],
                "data": {},
            },
        )
        with pytest.raises(
            ProviderSchemaError,
            match=r"api/search/workflow.*code",
        ):
            executor.search(SearchRequest(query="qwen", size=3))


def test_nonzero_detail_code_names_endpoint_and_code_field() -> None:
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(
            mocked,
            executor.detail_url,
            {
                "code": "403",
                "msg": "forbidden",
                "errorMessages": None,
                "data": None,
            },
        )
        with pytest.raises(
            ProviderSchemaError,
            match=r"portal/workflow/detail.*code",
        ):
            executor.detail(WORKFLOW_ID)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda body: body.__setitem__("code", "0"), "code"),
        (lambda body: body.pop("msg"), "msg"),
        (lambda body: body.__setitem__("msg", 1), "msg"),
        (lambda body: body.pop("errorMessages"), "errorMessages"),
        (
            lambda body: body.__setitem__("errorMessages", "provider failure"),
            "errorMessages",
        ),
        (lambda body: body.__setitem__("errorMessages", [1]), "errorMessages[0]"),
        (lambda body: body.__setitem__("unexpected", True), "unexpected"),
    ],
)
def test_search_envelope_drift_names_endpoint_and_failing_field(
    mutation: Any,
    field: str,
) -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    mutation(payload)
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.search(SearchRequest(query="qwen", size=3))

    message = str(raised.value)
    assert "api/search/workflow" in message
    assert f"invalid field {field}:" in message


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda body: body["data"].pop("records"), "data.records"),
        (lambda body: body["data"].__setitem__("records", {}), "data.records"),
        (
            lambda body: body["data"]["records"][0].pop("name"),
            "data.records[0].name",
        ),
        (
            lambda body: body["data"]["records"][0]["statisticsInfo"].__setitem__(
                "useCount", "many"
            ),
            "data.records[0].statisticsInfo.useCount",
        ),
        (lambda body: body["data"].__setitem__("total", "many"), "data.total"),
        (
            lambda body: body["data"]["records"][0].__setitem__("tags", "tag"),
            "data.records[0].tags",
        ),
        (
            lambda body: body["data"]["records"][0]["owner"].__setitem__(
                "avatar", 123
            ),
            "data.records[0].owner.avatar",
        ),
        (
            lambda body: body["data"]["records"][0]["tags"][0].__setitem__(
                "nameEn", 123
            ),
            "data.records[0].tags[0].nameEn",
        ),
        (
            lambda body: body["data"]["records"][0]["covers"][0].__setitem__(
                "id", "not-an-id"
            ),
            "data.records[0].covers[0].id",
        ),
        (
            lambda body: body["data"]["records"][0].pop("systemWorkflow"),
            "data.records[0].systemWorkflow",
        ),
        (
            lambda body: body["data"]["records"][0]["owner"].__setitem__(
                "unexpected", True
            ),
            "data.records[0].owner.unexpected",
        ),
        (
            lambda body: body["data"]["records"][0].__setitem__(
                "apiKey", "must-not-normalize"
            ),
            "data.records[0].apiKey",
        ),
    ],
)
def test_search_schema_drift_names_endpoint_and_failing_field(
    mutation: Any,
    field: str,
) -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    mutation(payload)
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.search(SearchRequest(query="qwen", size=3))

    message = str(raised.value)
    assert "api/search/workflow" in message
    assert f"invalid field {field}:" in message


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda body: body["data"]["records"][0].__setitem__(
                "id", 2004195228674387969
            ),
            "data.records[0].id",
        ),
        (
            lambda body: body["data"]["records"][0]["statisticsInfo"].__setitem__(
                "useCount", 11977
            ),
            "data.records[0].statisticsInfo.useCount",
        ),
        (
            lambda body: body["data"]["records"][0]["covers"][0].__setitem__(
                "id", 2004197847220318209
            ),
            "data.records[0].covers[0].id",
        ),
        (lambda body: body["data"].__setitem__("size", 3), "data.size"),
        (lambda body: body["data"].__setitem__("nextCursor", 1), "data.nextCursor"),
    ],
)
def test_search_rejects_wrong_provider_numeric_representations(
    mutation: Any,
    field: str,
) -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    mutation(payload)
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.search(SearchRequest(query="qwen", size=3))

    message = str(raised.value)
    assert "api/search/workflow" in message
    assert f"invalid field {field}:" in message


@pytest.mark.parametrize(
    "url",
    [
        " https://assets.example/image.png",
        "https://bad host.example/image.png",
        "https://assets.example/image name.png",
        "https://assets.example/image.png\n?token=secret",
        "https://assets.\texample/image.png",
        "https://assets.example/image\x01.png",
        pytest.param(
            "https://assets.example/image\u0080.png",
            id="control-U+0080",
        ),
        pytest.param(
            "https://assets.example/image\u009f.png",
            id="control-U+009F",
        ),
        "https://user:password@assets.example/image.png",
        "https:///image.png",
        "https://-bad.example/image.png",
        "https://assets.example:not-a-port/image.png",
        "https://assets.example:70000/image.png",
    ],
)
def test_search_rejects_malformed_provider_http_urls(url: str) -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    payload["data"]["records"][0]["owner"]["avatar"] = url
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.search(SearchRequest(query="qwen", size=3))

    message = str(raised.value)
    assert "api/search/workflow" in message
    assert "invalid field data.records[0].owner.avatar:" in message


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda body: body["data"].pop("primitiveNodes"), "data.primitiveNodes"),
        (lambda body: body["data"].__setitem__("nodeCount", "many"), "data.nodeCount"),
        (
            lambda body: body["data"].__setitem__("customNodes", {}),
            "data.customNodes",
        ),
        (
            lambda body: body["data"]["preview"].__setitem__(
                "imageWidth", "wide"
            ),
            "data.preview.imageWidth",
        ),
        (lambda body: body["data"].pop("owner"), "data.owner"),
        (
            lambda body: body["data"].__setitem__("labels", 123),
            "data.labels",
        ),
        (lambda body: body["data"].pop("status"), "data.status"),
        (
            lambda body: body["data"]["publishAccess"].__setitem__(
                "unexpected", True
            ),
            "data.publishAccess.unexpected",
        ),
    ],
)
def test_detail_schema_drift_names_endpoint_and_failing_field(
    mutation: Any,
    field: str,
) -> None:
    payload = _load_fixture("detail_next_scene.json")
    mutation(payload)
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.detail(WORKFLOW_ID)

    message = str(raised.value)
    assert "portal/workflow/detail" in message
    assert f"invalid field {field}:" in message


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda body: body["data"].__setitem__(
                "id", 1980860923768426497
            ),
            "data.id",
        ),
        (
            lambda body: body["data"].__setitem__("nodeCount", "148"),
            "data.nodeCount",
        ),
        (lambda body: body["data"].__setitem__("status", "1"), "data.status"),
        (
            lambda body: body["data"]["publishAccess"].__setitem__(
                "accessType", "0"
            ),
            "data.publishAccess.accessType",
        ),
    ],
)
def test_detail_rejects_wrong_provider_numeric_representations(
    mutation: Any,
    field: str,
) -> None:
    payload = _load_fixture("detail_next_scene.json")
    mutation(payload)
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, payload)
        with pytest.raises(ProviderSchemaError) as raised:
            executor.detail(WORKFLOW_ID)

    message = str(raised.value)
    assert "portal/workflow/detail" in message
    assert f"invalid field {field}:" in message


def test_detail_rejects_mismatched_workflow_id_before_graph_fetch() -> None:
    payload = _load_fixture("detail_next_scene.json")
    payload["data"]["id"] = "1"
    graph_fetcher = GraphFetcherSpy(_load_fixture("qwen_graph.json"))
    executor = _executor(graph_fetcher)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, payload)
        with pytest.raises(
            ProviderSchemaError,
            match=r"portal/workflow/detail.*data.id",
        ):
            executor.fetch_workflow(
                WorkflowRef(executor="runninghub", workflow_id=WORKFLOW_ID)
            )

    assert graph_fetcher.calls == []


def test_network_errors_become_provider_errors_without_credentials() -> None:
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        mocked.add(
            responses.POST,
            executor.search_url,
            body=requests.ConnectionError(
                "failed with apiKey=super-secret and cookie=session-secret"
            ),
        )
        with pytest.raises(ExecutorError) as raised:
            executor.search(SearchRequest(query="qwen", size=3))

    message = str(raised.value)
    assert "api/search/workflow" in message
    assert "super-secret" not in message
    assert "session-secret" not in message


def test_provider_metadata_is_frozen_and_sanitizes_validated_url_credentials() -> None:
    payload = _load_fixture("search_qwen_reputation.json")
    first_record = payload["data"]["records"][0]
    signed_url = (
        "https://assets.example/image.png"
        "?X-Amz-Credential=url-secret&X-Amz-Signature=signed-secret"
    )
    first_record["owner"]["avatar"] = signed_url
    first_record["covers"][0]["url"] = signed_url
    first_record["covers"][0]["thumbnailUri"] = signed_url
    executor = _executor()

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.search_url, payload)
        candidate = executor.search(SearchRequest(query="qwen", size=3)).records[0]

    metadata = thaw_json(candidate.provider_metadata)
    serialized = json.dumps(metadata)
    assert isinstance(candidate.provider_metadata, FrozenDict)
    assert "url-secret" not in serialized
    assert "signed-secret" not in serialized
    assert metadata["record"]["owner"]["avatar"] == (
        "https://assets.example/image.png"
    )
    assert metadata["record"]["covers"][0]["url"] == (
        "https://assets.example/image.png"
    )
    assert metadata["record"]["covers"][0]["thumbnailUri"] == (
        "https://assets.example/image.png"
    )
    assert candidate.covers[0].url == "https://assets.example/image.png"
    assert candidate.covers[0].thumbnail_url == "https://assets.example/image.png"


def test_graph_fetch_errors_do_not_expose_provider_credentials() -> None:
    graph_fetcher = GraphFetcherSpy(
        error=RuntimeError("apiKey=super-secret cookie=session-secret")
    )
    executor = _executor(graph_fetcher)

    with responses.RequestsMock(assert_all_requests_are_fired=True) as mocked:
        _add_json(mocked, executor.detail_url, _load_fixture("detail_next_scene.json"))
        with pytest.raises(ExecutorError) as raised:
            executor.fetch_workflow(
                WorkflowRef(executor="runninghub", workflow_id=WORKFLOW_ID)
            )

    message = str(raised.value)
    assert "graph fetch" in message
    assert "super-secret" not in message
    assert "session-secret" not in message


def test_runninghub_executor_is_instantiable_and_paid_methods_are_structured() -> None:
    executor = _executor()
    message = "runninghub paid execution is not enabled"

    assert isinstance(executor, Executor)
    assert executor.name == "runninghub"

    with pytest.raises(UnsupportedCapabilityError, match=message):
        executor.validate_variant(object(), object())
    with pytest.raises(UnsupportedCapabilityError, match=message):
        executor.submit(object(), object())
    with pytest.raises(UnsupportedCapabilityError, match=message):
        executor.poll("task-1")
    with pytest.raises(UnsupportedCapabilityError, match=message):
        executor.collect_outputs("task-1")

    cancelled = executor.cancel("task-1")
    assert cancelled.supported is False
    assert cancelled.cancelled is False
    assert cancelled.diagnostic == message
