from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
import json
import re
from typing import Any, NoReturn

from mp_runninghub_api.runninghub import get_workflow_json
from pydantic import ValidationError
import requests

from mp_comfyui_lab.discovery.models import (
    ContentClassification,
    CoverAsset,
    Engagement,
    WorkflowCandidate,
    WorkflowDetail,
)
from mp_comfyui_lab.discovery.urls import (
    UnsupportedSourceError,
    parse_runninghub_url,
)
from mp_comfyui_lab.executors.base import (
    CancelResult,
    CapabilityReport,
    Executor,
    ExecutorError,
    FetchedWorkflow,
    PollResult,
    ProviderOutput,
    ProviderSchemaError,
    SearchPage,
    SearchRequest,
    SubmissionReceipt,
    UnsupportedCapabilityError,
    ValidationReport,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.models import MaterializedVariant, ResolvedFixtures
from mp_comfyui_lab.security import sanitize_for_persistence


BASE_URL = "https://www.runninghub.ai"
SEARCH_ENDPOINT = "/api/search/workflow"
DETAIL_ENDPOINT = "/api/portal/workflow/detail"
GRAPH_ENDPOINT = "/api/openapi/getJsonApiFormat"
PUBLIC_REQUEST_TIMEOUT = (10, 60)
PAID_EXECUTION_DISABLED = "runninghub paid execution is not enabled"

NSFW_TERMS = frozenset({"nsfw", "nude", "nudity", "\u88f8\u4f53", "\u6210\u4eba", "\u8272\u60c5"})
SFW_TERMS = frozenset({"sfw", "safe for work", "\u5b89\u5168"})


GraphFetcher = Callable[[str | None, str], Any]


class _PlainTextParser(HTMLParser):
    _HIDDEN_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag.casefold() in self._HIDDEN_TAGS:
            self.hidden_depth += 1
        elif self.hidden_depth == 0:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in self._HIDDEN_TAGS and self.hidden_depth:
            self.hidden_depth -= 1
        elif self.hidden_depth == 0:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0:
            self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def _schema_error(endpoint: str, field: str, reason: str) -> NoReturn:
    raise ProviderSchemaError(
        f"RunningHub {endpoint}: invalid field {field}: {reason}"
    )


def _required(mapping: dict[str, Any], key: str, endpoint: str, path: str) -> Any:
    if key not in mapping:
        _schema_error(endpoint, path, "field is missing")
    return mapping[key]


def _mapping(value: Any, endpoint: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _schema_error(endpoint, field, "expected an object")
    return value


def _list(value: Any, endpoint: str, field: str) -> list[Any]:
    if not isinstance(value, list):
        _schema_error(endpoint, field, "expected an array")
    return value


def _nonblank_string(value: Any, endpoint: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _schema_error(endpoint, field, "expected a non-empty string")
    return value.strip()


def _string(value: Any, endpoint: str, field: str) -> str:
    if not isinstance(value, str):
        _schema_error(endpoint, field, "expected a string")
    return value


def _error_messages(value: Any, endpoint: str, field: str) -> None:
    if value is None:
        return
    messages = _list(value, endpoint, field)
    for index, message in enumerate(messages):
        _string(message, endpoint, f"{field}[{index}]")


def _workflow_id(value: Any, endpoint: str, field: str) -> str:
    if isinstance(value, bool):
        _schema_error(endpoint, field, "expected a numeric workflow ID")
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        _schema_error(endpoint, field, "expected a numeric workflow ID")
    return value


def _integer(
    value: Any,
    endpoint: str,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        _schema_error(endpoint, field, "expected an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        parsed = int(value)
    else:
        _schema_error(endpoint, field, "expected an integer or numeric string")

    if minimum is not None and parsed < minimum:
        _schema_error(endpoint, field, f"expected a value of at least {minimum}")
    return parsed


def _boolean(value: Any, endpoint: str, field: str) -> bool:
    if not isinstance(value, bool):
        _schema_error(endpoint, field, "expected a boolean")
    return value


def _published_at(value: Any, endpoint: str, field: str) -> datetime:
    if not isinstance(value, str):
        _schema_error(endpoint, field, "expected an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _schema_error(endpoint, field, "expected an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _schema_error(endpoint, field, "timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _optional_string(value: Any, endpoint: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _schema_error(endpoint, field, "expected a string or null")
    return value


def _optional_identifier(value: Any, endpoint: str, field: str) -> str | None:
    if value is None:
        return None
    return _workflow_id(value, endpoint, field)


def _safe_url(value: Any, endpoint: str, field: str) -> str:
    url = _nonblank_string(value, endpoint, field)
    sanitized = sanitize_for_persistence({"url": url})
    return sanitized["url"]


def _sanitize_provider_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {
            key: _sanitize_provider_value(item)
            for key, item in value.items()
        }
        return sanitize_for_persistence(sanitized)
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_provider_value(item) for item in value)
    if isinstance(value, str):
        return sanitize_for_persistence({"url": value})["url"]
    return value


def _validated_canonical_url(
    canonical_url: str | None,
    workflow_id: str,
) -> str:
    if canonical_url is None:
        return f"https://www.runninghub.ai/post/{workflow_id}"

    parsed = parse_runninghub_url(canonical_url)
    if parsed.workflow_id != workflow_id:
        raise UnsupportedSourceError(
            "RunningHub canonical URL must identify the same public post or workflow"
        )
    return canonical_url


def _cover(value: Any, endpoint: str, field: str) -> CoverAsset:
    item = _mapping(value, endpoint, field)
    url = _safe_url(
        _required(item, "url", endpoint, f"{field}.url"),
        endpoint,
        f"{field}.url",
    )

    thumbnail_value = item.get("thumbnailUri")
    thumbnail_url = None
    if thumbnail_value is not None:
        thumbnail_url = _safe_url(
            thumbnail_value,
            endpoint,
            f"{field}.thumbnailUri",
        )

    width = None
    if "imageWidth" in item and item["imageWidth"] is not None:
        width = _integer(
            item["imageWidth"],
            endpoint,
            f"{field}.imageWidth",
            minimum=1,
        )

    height = None
    if "imageHeight" in item and item["imageHeight"] is not None:
        height = _integer(
            item["imageHeight"],
            endpoint,
            f"{field}.imageHeight",
            minimum=1,
        )

    try:
        return CoverAsset(
            url=url,
            thumbnail_url=thumbnail_url,
            width=width,
            height=height,
        )
    except ValidationError:
        _schema_error(endpoint, field, "cover metadata is malformed")


def _covers(value: Any, endpoint: str, field: str) -> tuple[CoverAsset, ...]:
    items = _list(value, endpoint, field)
    return tuple(
        _cover(item, endpoint, f"{field}[{index}]")
        for index, item in enumerate(items)
    )


def _tags(value: Any, endpoint: str, field: str) -> tuple[str, ...]:
    items = _list(value, endpoint, field)
    parsed: list[str] = []
    for index, value in enumerate(items):
        item_field = f"{field}[{index}]"
        item = _mapping(value, endpoint, item_field)
        parsed.append(
            _nonblank_string(
                _required(item, "name", endpoint, f"{item_field}.name"),
                endpoint,
                f"{item_field}.name",
            )
        )
    return tuple(parsed)


def _string_inventory(value: Any, endpoint: str, field: str) -> tuple[str, ...]:
    items = _list(value, endpoint, field)
    return tuple(
        _nonblank_string(item, endpoint, f"{field}[{index}]")
        for index, item in enumerate(items)
    )


def _owner(
    value: Any,
    endpoint: str,
    field: str,
) -> tuple[str, str]:
    owner = _mapping(value, endpoint, field)
    owner_id = _workflow_id(
        _required(owner, "id", endpoint, f"{field}.id"),
        endpoint,
        f"{field}.id",
    )
    owner_name = _nonblank_string(
        _required(owner, "name", endpoint, f"{field}.name"),
        endpoint,
        f"{field}.name",
    )
    return owner_id, owner_name


def _engagement(value: Any, endpoint: str, field: str) -> Engagement:
    statistics = _mapping(value, endpoint, field)

    def count(key: str) -> int:
        path = f"{field}.{key}"
        return _integer(
            _required(statistics, key, endpoint, path),
            endpoint,
            path,
            minimum=0,
        )

    return Engagement(
        likes=count("likeCount"),
        uses=count("useCount"),
        collects=count("collectCount"),
        downloads=count("downloadCount"),
        views=count("pv"),
    )


def _classification(
    *,
    name: str,
    description: str | None,
    tags: tuple[str, ...],
    used_models: tuple[str, ...],
) -> tuple[ContentClassification, tuple[str, ...]]:
    evidence = [("name", name)]
    if description:
        evidence.append(("description", description))
    evidence.extend((f"tag[{index}]", value) for index, value in enumerate(tags))
    evidence.extend(
        (f"usedModels[{index}]", value)
        for index, value in enumerate(used_models)
    )

    def matches(terms: frozenset[str], label: str) -> tuple[str, ...]:
        reasons: list[str] = []
        ordered_terms = sorted(terms, key=lambda term: (term.casefold(), term))
        for source, text in evidence:
            folded = text.casefold()
            for term in ordered_terms:
                if term.casefold() in folded:
                    reasons.append(f"{label} term '{term}' in {source}")
        return tuple(reasons)

    nsfw_reasons = matches(NSFW_TERMS, "NSFW")
    if nsfw_reasons:
        return ContentClassification.NSFW, nsfw_reasons

    sfw_reasons = matches(SFW_TERMS, "SFW")
    if sfw_reasons:
        return ContentClassification.SFW, sfw_reasons

    return (
        ContentClassification.UNKNOWN,
        ("no explicit classification evidence",),
    )


def _candidate(
    value: dict[str, Any],
    *,
    endpoint: str,
    field: str,
    canonical_url: str,
    used_models: tuple[str, ...] = (),
    require_labels: bool = False,
) -> WorkflowCandidate:
    workflow_id = _workflow_id(
        _required(value, "id", endpoint, f"{field}.id"),
        endpoint,
        f"{field}.id",
    )
    name = _nonblank_string(
        _required(value, "name", endpoint, f"{field}.name"),
        endpoint,
        f"{field}.name",
    )
    raw_description = _optional_string(
        _required(value, "desc", endpoint, f"{field}.desc"),
        endpoint,
        f"{field}.desc",
    )
    description = None if raw_description is None else _plain_text(raw_description)
    published_at = _published_at(
        _required(value, "publishTime", endpoint, f"{field}.publishTime"),
        endpoint,
        f"{field}.publishTime",
    )
    owner_id, owner_name = _owner(
        _required(value, "owner", endpoint, f"{field}.owner"),
        endpoint,
        f"{field}.owner",
    )
    engagement = _engagement(
        _required(
            value,
            "statisticsInfo",
            endpoint,
            f"{field}.statisticsInfo",
        ),
        endpoint,
        f"{field}.statisticsInfo",
    )
    tags = _tags(
        _required(value, "tags", endpoint, f"{field}.tags"),
        endpoint,
        f"{field}.tags",
    )
    covers = _covers(
        _required(value, "covers", endpoint, f"{field}.covers"),
        endpoint,
        f"{field}.covers",
    )

    if require_labels:
        labels = _required(value, "labels", endpoint, f"{field}.labels")
        if labels is not None and not isinstance(labels, str):
            _schema_error(endpoint, f"{field}.labels", "expected a string or null")

    classification, reasons = _classification(
        name=name,
        description=description,
        tags=tags,
        used_models=used_models,
    )

    try:
        return WorkflowCandidate(
            executor="runninghub",
            workflow_id=workflow_id,
            canonical_url=canonical_url,
            name=name,
            description=description,
            author_id=owner_id,
            author_name=owner_name,
            published_at=published_at,
            tags=tags,
            engagement=engagement,
            covers=covers,
            content_classification=classification,
            classification_reasons=reasons,
            provider_metadata={
                "endpoint": endpoint,
                "record": _sanitize_provider_value(value),
            },
        )
    except (ValidationError, ValueError):
        _schema_error(endpoint, field, "mapped candidate is malformed")


class RunningHubExecutor(Executor):
    search_url = f"{BASE_URL}{SEARCH_ENDPOINT}"
    detail_url = f"{BASE_URL}{DETAIL_ENDPOINT}"
    public_request_timeout = PUBLIC_REQUEST_TIMEOUT

    def __init__(
        self,
        session: requests.Session | None = None,
        graph_fetcher: GraphFetcher | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.graph_fetcher = graph_fetcher or get_workflow_json

    @property
    def name(self) -> str:
        return "runninghub"

    def _post_public(
        self,
        *,
        url: str,
        endpoint: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                url,
                json=body,
                timeout=self.public_request_timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            raise ExecutorError(f"RunningHub request failed for {endpoint}") from None

        try:
            payload = response.json()
        except (ValueError, requests.JSONDecodeError):
            _schema_error(endpoint, "response JSON", "body is not valid JSON")

        envelope = _mapping(payload, endpoint, "response JSON")
        code = _integer(
            _required(envelope, "code", endpoint, "code"),
            endpoint,
            "code",
        )
        _string(
            _required(envelope, "msg", endpoint, "msg"),
            endpoint,
            "msg",
        )
        _error_messages(
            _required(
                envelope,
                "errorMessages",
                endpoint,
                "errorMessages",
            ),
            endpoint,
            "errorMessages",
        )
        data = _required(envelope, "data", endpoint, "data")
        if code != 0:
            _schema_error(endpoint, "code", f"expected 0, received {code}")
        return _mapping(data, endpoint, "data")

    def search(self, request: SearchRequest) -> SearchPage:
        data = self._post_public(
            url=self.search_url,
            endpoint=SEARCH_ENDPOINT,
            body={
                "size": request.size,
                "current": request.page,
                "search": request.query,
                "tags": [],
                "sort": request.sort,
            },
        )
        records_value = _required(data, "records", SEARCH_ENDPOINT, "data.records")
        records = _list(records_value, SEARCH_ENDPOINT, "data.records")
        candidates = tuple(
            _candidate(
                _mapping(
                    record,
                    SEARCH_ENDPOINT,
                    f"data.records[{index}]",
                ),
                endpoint=SEARCH_ENDPOINT,
                field=f"data.records[{index}]",
                canonical_url=(
                    "https://www.runninghub.ai/post/"
                    + _workflow_id(
                        _required(
                            _mapping(
                                record,
                                SEARCH_ENDPOINT,
                                f"data.records[{index}]",
                            ),
                            "id",
                            SEARCH_ENDPOINT,
                            f"data.records[{index}].id",
                        ),
                        SEARCH_ENDPOINT,
                        f"data.records[{index}].id",
                    )
                ),
                require_labels=True,
            )
            for index, record in enumerate(records)
        )

        size = _integer(
            _required(data, "size", SEARCH_ENDPOINT, "data.size"),
            SEARCH_ENDPOINT,
            "data.size",
            minimum=1,
        )
        current = _integer(
            _required(data, "current", SEARCH_ENDPOINT, "data.current"),
            SEARCH_ENDPOINT,
            "data.current",
            minimum=1,
        )
        total = _integer(
            _required(data, "total", SEARCH_ENDPOINT, "data.total"),
            SEARCH_ENDPOINT,
            "data.total",
            minimum=0,
        )
        _integer(
            _required(data, "pages", SEARCH_ENDPOINT, "data.pages"),
            SEARCH_ENDPOINT,
            "data.pages",
            minimum=0,
        )
        has_next = _boolean(
            _required(data, "hasNext", SEARCH_ENDPOINT, "data.hasNext"),
            SEARCH_ENDPOINT,
            "data.hasNext",
        )
        _boolean(
            _required(
                data,
                "hasPrevious",
                SEARCH_ENDPOINT,
                "data.hasPrevious",
            ),
            SEARCH_ENDPOINT,
            "data.hasPrevious",
        )
        next_cursor = _required(
            data,
            "nextCursor",
            SEARCH_ENDPOINT,
            "data.nextCursor",
        )
        if next_cursor is not None and (
            isinstance(next_cursor, bool)
            or not isinstance(next_cursor, (str, int))
        ):
            _schema_error(
                SEARCH_ENDPOINT,
                "data.nextCursor",
                "expected a string, integer, or null",
            )

        return SearchPage(
            records=candidates,
            total=total,
            page=current,
            size=size,
            has_next=has_next,
        )

    def detail(
        self,
        workflow_id: str,
        *,
        canonical_url: str | None = None,
    ) -> WorkflowDetail:
        requested_id = _workflow_id(
            workflow_id,
            DETAIL_ENDPOINT,
            "request.workflowId",
        )
        candidate_url = _validated_canonical_url(canonical_url, requested_id)
        data = self._post_public(
            url=self.detail_url,
            endpoint=DETAIL_ENDPOINT,
            body={"workflowId": requested_id},
        )
        returned_id = _workflow_id(
            _required(data, "id", DETAIL_ENDPOINT, "data.id"),
            DETAIL_ENDPOINT,
            "data.id",
        )
        if returned_id != requested_id:
            _schema_error(
                DETAIL_ENDPOINT,
                "data.id",
                "does not match the requested workflow ID",
            )

        primitive_nodes = _string_inventory(
            _required(
                data,
                "primitiveNodes",
                DETAIL_ENDPOINT,
                "data.primitiveNodes",
            ),
            DETAIL_ENDPOINT,
            "data.primitiveNodes",
        )
        custom_nodes = _string_inventory(
            _required(
                data,
                "customNodes",
                DETAIL_ENDPOINT,
                "data.customNodes",
            ),
            DETAIL_ENDPOINT,
            "data.customNodes",
        )
        used_models = _string_inventory(
            _required(
                data,
                "usedModels",
                DETAIL_ENDPOINT,
                "data.usedModels",
            ),
            DETAIL_ENDPOINT,
            "data.usedModels",
        )
        node_count = _integer(
            _required(data, "nodeCount", DETAIL_ENDPOINT, "data.nodeCount"),
            DETAIL_ENDPOINT,
            "data.nodeCount",
            minimum=0,
        )
        preview_value = _required(
            data,
            "preview",
            DETAIL_ENDPOINT,
            "data.preview",
        )
        preview = (
            None
            if preview_value is None
            else _cover(preview_value, DETAIL_ENDPOINT, "data.preview")
        )
        webapp_id = _optional_identifier(
            _required(data, "webappId", DETAIL_ENDPOINT, "data.webappId"),
            DETAIL_ENDPOINT,
            "data.webappId",
        )
        candidate = _candidate(
            data,
            endpoint=DETAIL_ENDPOINT,
            field="data",
            canonical_url=candidate_url,
            used_models=used_models,
        )

        try:
            return WorkflowDetail(
                candidate=candidate,
                node_count=node_count,
                primitive_nodes=primitive_nodes,
                custom_nodes=custom_nodes,
                used_models=used_models,
                preview=preview,
                webapp_id=webapp_id,
            )
        except ValidationError:
            _schema_error(DETAIL_ENDPOINT, "data", "mapped detail is malformed")

    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        if ref.executor != self.name:
            raise ExecutorError("RunningHub workflow ref executor must be runninghub")

        detail = self.detail(
            ref.workflow_id,
            canonical_url=ref.canonical_url,
        )
        try:
            raw_graph = self.graph_fetcher(None, ref.workflow_id)
        except Exception:
            raise ExecutorError(
                f"RunningHub graph fetch failed for workflow {ref.workflow_id}"
            ) from None

        graph = raw_graph
        if isinstance(raw_graph, str):
            try:
                graph = json.loads(raw_graph)
            except json.JSONDecodeError:
                _schema_error(GRAPH_ENDPOINT, "graph", "expected a JSON object")
        if not isinstance(graph, dict):
            _schema_error(GRAPH_ENDPOINT, "graph", "expected a JSON object")

        try:
            return FetchedWorkflow(ref=ref, detail=detail, graph=graph)
        except ValidationError:
            _schema_error(
                GRAPH_ENDPOINT,
                "graph",
                "fetched workflow identity or graph is malformed",
            )

    def inspect_capabilities(self, workflow: FetchedWorkflow) -> CapabilityReport:
        detail = workflow.detail
        required_nodes = tuple(
            sorted(set(detail.primitive_nodes + detail.custom_nodes))
        )
        lora_names = tuple(
            sorted({name for name in detail.used_models if "lora" in name.casefold()})
        )
        model_names = tuple(
            sorted(set(detail.used_models).difference(lora_names))
        )
        output_hints = tuple(
            sorted(
                node
                for node in required_nodes
                if any(
                    term in node.casefold()
                    for term in ("output", "preview", "save")
                )
            )
        )
        return CapabilityReport(
            required_nodes=required_nodes,
            model_names=model_names,
            lora_names=lora_names,
            output_node_ids=(),
            provider_metadata={
                "inventory_source": "public detail",
                "output_node_class_hints": output_hints,
            },
        )

    def validate_variant(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> ValidationReport:
        del variant, fixtures
        raise UnsupportedCapabilityError(PAID_EXECUTION_DISABLED)

    def submit(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> SubmissionReceipt:
        del variant, fixtures
        raise UnsupportedCapabilityError(PAID_EXECUTION_DISABLED)

    def poll(self, provider_task_id: str) -> PollResult:
        del provider_task_id
        raise UnsupportedCapabilityError(PAID_EXECUTION_DISABLED)

    def collect_outputs(self, provider_task_id: str) -> tuple[ProviderOutput, ...]:
        del provider_task_id
        raise UnsupportedCapabilityError(PAID_EXECUTION_DISABLED)

    def cancel(self, provider_task_id: str) -> CancelResult:
        del provider_task_id
        return CancelResult(
            supported=False,
            cancelled=False,
            diagnostic=PAID_EXECUTION_DISABLED,
        )
