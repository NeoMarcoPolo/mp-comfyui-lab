from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
from ipaddress import ip_address
import json
import re
from typing import Annotated, Any, NoReturn, TypeVar
import unicodedata
from urllib.parse import urlsplit

from mp_runninghub_api.runninghub import get_workflow_json
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError
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
    ProviderResponseError,
    ProviderSchemaError,
    SearchPage,
    SearchRequest,
    SubmissionReceipt,
    UnsupportedCapabilityError,
    ValidationReport,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.models import MaterializedVariant, ResolvedFixtures
from mp_comfyui_lab.security import sanitize_embedded_urls, sanitize_for_persistence


BASE_URL = "https://www.runninghub.ai"
SEARCH_ENDPOINT = "/api/search/workflow"
DETAIL_ENDPOINT = "/api/portal/workflow/detail"
GRAPH_ENDPOINT = "/api/openapi/getJsonApiFormat"
PUBLIC_REQUEST_TIMEOUT = (10, 60)
PAID_EXECUTION_DISABLED = "runninghub paid execution is not enabled"
PROVIDER_NUMERIC_STRING_MAX_LENGTH = 64
MAX_GRAPH_DEPTH = 256
MAX_GRAPH_NODES = 100_000

NSFW_TERMS = frozenset({"nsfw", "nude", "nudity", "\u88f8\u4f53", "\u6210\u4eba", "\u8272\u60c5"})
SFW_TERMS = frozenset({"sfw", "safe for work", "\u5b89\u5168"})
SFW_NEGATION_TOKENS = ("not", "never", "non", "unsafe")
SFW_NEGATION_PAIRS = (
    ("no", "longer"),
    ("far", "from"),
)
LORA_INVENTORY_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])loras?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


GraphFetcher = Callable[[str | None, str], Any]


def _provider_aware_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        offset = parsed.utcoffset()
    except (ValueError, OverflowError):
        raise ValueError("expected an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or offset is None:
        raise ValueError("timestamp must include a timezone")
    return value


def _provider_http_url(value: str) -> str:
    if any(
        character.isspace()
        or unicodedata.category(character) == "Cc"
        for character in value
    ):
        raise ValueError("URL must not contain whitespace or control characters")

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        raise ValueError("expected a valid HTTP(S) URL") from None

    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
    ):
        raise ValueError("expected a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    if parsed.netloc.endswith(":") or port == 0:
        raise ValueError("expected a valid HTTP(S) URL port")

    try:
        ip_address(hostname)
    except ValueError:
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").rstrip(".")
        except UnicodeError:
            raise ValueError("expected a valid hostname") from None
        labels = ascii_hostname.split(".")
        if (
            not ascii_hostname
            or len(ascii_hostname) > 253
            or any(
                re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
                    label,
                )
                is None
                for label in labels
            )
        ):
            raise ValueError("expected a valid hostname")
    return value


ProviderNumericString = Annotated[
    str,
    Field(
        pattern=r"^[0-9]+$",
        max_length=PROVIDER_NUMERIC_STRING_MAX_LENGTH,
    ),
]
ProviderNonnegativeInt = Annotated[int, Field(ge=0)]
ProviderTimestamp = Annotated[str, AfterValidator(_provider_aware_timestamp)]
ProviderHttpUrl = Annotated[str, AfterValidator(_provider_http_url)]


class _ProviderModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class _ProviderOwner(_ProviderModel):
    id: ProviderNumericString
    avatar: ProviderHttpUrl
    name: str


class _ProviderStatistics(_ProviderModel):
    likeCount: ProviderNumericString
    downloadCount: ProviderNumericString
    useCount: ProviderNumericString
    pv: ProviderNumericString
    collectCount: ProviderNumericString


class _ProviderTag(_ProviderModel):
    id: ProviderNumericString
    name: str
    nameEn: str
    labels: None


class _ProviderCover(_ProviderModel):
    id: ProviderNumericString
    objName: str | None
    url: ProviderHttpUrl
    thumbnailUri: ProviderHttpUrl
    imageWidth: ProviderNumericString
    imageHeight: ProviderNumericString


class _ProviderEnvelope(_ProviderModel):
    code: ProviderNonnegativeInt
    msg: str
    errorMessages: list[str] | None
    data: Any


class _SearchRecord(_ProviderModel):
    id: ProviderNumericString
    name: str
    desc: str
    systemWorkflow: bool
    publishTime: ProviderTimestamp
    timestamp: ProviderNumericString
    preview: None
    owner: _ProviderOwner
    statisticsInfo: _ProviderStatistics
    nodeCount: None
    liked: ProviderNonnegativeInt
    covers: list[_ProviderCover]
    tags: list[_ProviderTag]
    labels: str
    seq: ProviderNumericString
    homeShow: None


class _SearchData(_ProviderModel):
    records: list[_SearchRecord]
    size: ProviderNumericString
    current: ProviderNumericString
    total: ProviderNumericString
    pages: ProviderNumericString
    hasNext: bool
    hasPrevious: bool
    nextCursor: None


class _SearchEnvelope(_ProviderModel):
    code: ProviderNonnegativeInt
    msg: str
    errorMessages: list[str] | None
    data: _SearchData


class _ProviderPublishAccess(_ProviderModel):
    accessType: ProviderNonnegativeInt
    publishScope: ProviderNonnegativeInt
    encrypted: bool
    granted: bool
    needPassword: bool
    owner: bool
    visibleUsers: Annotated[list[Any], Field(max_length=0)]


class _DetailData(_ProviderModel):
    id: ProviderNumericString
    name: str
    desc: str
    systemWorkflow: bool
    publishTime: ProviderTimestamp
    timestamp: ProviderNumericString
    md5: None
    imageSize: None
    preview: _ProviderCover
    owner: _ProviderOwner
    workflowContent: None
    nodeCount: ProviderNonnegativeInt
    statisticsInfo: _ProviderStatistics
    labels: str
    tags: list[_ProviderTag]
    covers: list[_ProviderCover]
    chineseCovers: list[_ProviderCover]
    englishCovers: list[_ProviderCover]
    primitiveNodes: list[str]
    customNodes: list[str]
    usedModels: list[str]
    liked: bool
    likeCount: ProviderNumericString
    collect: bool
    collectCount: ProviderNumericString
    webappId: ProviderNumericString
    status: ProviderNonnegativeInt
    workflowState: ProviderNonnegativeInt
    webappState: ProviderNonnegativeInt
    publishAccess: _ProviderPublishAccess
    canModifyPublishType: bool
    canModifyPublishScope: bool


class _DetailEnvelope(_ProviderModel):
    code: ProviderNonnegativeInt
    msg: str
    errorMessages: list[str] | None
    data: _DetailData


ProviderRecord = _SearchRecord | _DetailData
ProviderSchema = TypeVar("ProviderSchema", bound=_ProviderModel)


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


def _provider_field_path(location: tuple[int | str, ...]) -> str:
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else part
    return path or "response JSON"


def _validate_provider_schema(
    schema: type[ProviderSchema],
    value: Any,
    endpoint: str,
) -> ProviderSchema:
    try:
        return schema.model_validate(value)
    except ValidationError as error:
        issue = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )[0]
        _schema_error(
            endpoint,
            _provider_field_path(issue["loc"]),
            issue["msg"],
        )


def _normalized_model_error(
    endpoint: str,
    field: str,
    error: ValidationError,
) -> NoReturn:
    issue = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[0]
    location = _provider_field_path(issue["loc"])
    path = f"{field}.{location}" if location != "response JSON" else field
    _schema_error(endpoint, path, issue["msg"])


def _nonblank_string(value: str, endpoint: str, field: str) -> str:
    if not value.strip():
        _schema_error(endpoint, field, "expected a non-empty string")
    return value.strip()


def _workflow_id(value: Any, endpoint: str, field: str) -> str:
    if isinstance(value, bool):
        _schema_error(endpoint, field, "expected a numeric workflow ID")
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        _schema_error(endpoint, field, "expected a numeric workflow ID")
    return value


def _integer(
    value: str,
    endpoint: str,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    try:
        parsed = int(value)
    except (ValueError, OverflowError):
        _schema_error(endpoint, field, "numeric value cannot be normalized")

    if minimum is not None and parsed < minimum:
        _schema_error(endpoint, field, f"expected a value of at least {minimum}")
    return parsed


def _published_at(value: str, endpoint: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            _schema_error(endpoint, field, "timestamp must include a timezone")
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        _schema_error(
            endpoint,
            field,
            "timestamp cannot be normalized to UTC",
        )


def _sanitize_url_string(value: str) -> str:
    return sanitize_embedded_urls(value)


def _safe_url(value: str, endpoint: str, field: str) -> str:
    url = _nonblank_string(value, endpoint, field)
    return _sanitize_url_string(url)


def _sanitize_provider_value(value: Any) -> Any:
    return sanitize_for_persistence(value)


def _validate_graph_complexity(graph: dict[str, Any]) -> None:
    pending: list[tuple[Any, int]] = [(graph, 0)]
    seen_containers: set[int] = set()
    visited = 0

    while pending:
        value, depth = pending.pop()
        visited += 1
        if visited > MAX_GRAPH_NODES:
            _schema_error(
                GRAPH_ENDPOINT,
                "graph",
                "exceeds the maximum supported node count",
            )
        if depth > MAX_GRAPH_DEPTH:
            _schema_error(
                GRAPH_ENDPOINT,
                "graph",
                "exceeds the maximum supported depth",
            )

        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_containers:
                _schema_error(
                    GRAPH_ENDPOINT,
                    "graph",
                    "contains repeated or recursive containers",
                )
            seen_containers.add(identity)
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen_containers:
                _schema_error(
                    GRAPH_ENDPOINT,
                    "graph",
                    "contains repeated or recursive containers",
                )
            seen_containers.add(identity)
            pending.extend((item, depth + 1) for item in value)


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
    return parsed.canonical_url


def _cover(
    value: _ProviderCover,
    endpoint: str,
    field: str,
) -> CoverAsset:
    url = _safe_url(
        value.url,
        endpoint,
        f"{field}.url",
    )
    thumbnail_url = _safe_url(
        value.thumbnailUri,
        endpoint,
        f"{field}.thumbnailUri",
    )
    width = _integer(
        value.imageWidth,
        endpoint,
        f"{field}.imageWidth",
        minimum=1,
    )
    height = _integer(
        value.imageHeight,
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
    except ValidationError as error:
        _normalized_model_error(endpoint, field, error)


def _covers(
    values: list[_ProviderCover],
    endpoint: str,
    field: str,
) -> tuple[CoverAsset, ...]:
    return tuple(
        _cover(item, endpoint, f"{field}[{index}]")
        for index, item in enumerate(values)
    )


def _tags(
    values: list[_ProviderTag],
    endpoint: str,
    field: str,
) -> tuple[str, ...]:
    return tuple(
        _nonblank_string(
            value.name,
            endpoint,
            f"{field}[{index}].name",
        )
        for index, value in enumerate(values)
    )


def _string_inventory(
    values: list[str],
    endpoint: str,
    field: str,
    *,
    sanitize_urls: bool = False,
) -> tuple[str, ...]:
    parsed: list[str] = []
    for index, item in enumerate(values):
        normalized = _nonblank_string(
            item,
            endpoint,
            f"{field}[{index}]",
        )
        parsed.append(
            _sanitize_url_string(normalized)
            if sanitize_urls
            else normalized
        )
    return tuple(parsed)


def _engagement(
    statistics: _ProviderStatistics,
    endpoint: str,
    field: str,
) -> Engagement:
    def count(value: str, key: str) -> int:
        return _integer(
            value,
            endpoint,
            f"{field}.{key}",
            minimum=0,
        )

    try:
        return Engagement(
            likes=count(statistics.likeCount, "likeCount"),
            uses=count(statistics.useCount, "useCount"),
            collects=count(statistics.collectCount, "collectCount"),
            downloads=count(statistics.downloadCount, "downloadCount"),
            views=count(statistics.pv, "pv"),
        )
    except ValidationError as error:
        _normalized_model_error(endpoint, field, error)


def _classification_segments(value: str) -> tuple[str, ...]:
    segments: list[str] = []
    current: list[str] = []
    index = 0

    def finish_segment() -> None:
        segment = "".join(current)
        if segment:
            segments.append(segment)
        current.clear()

    while index < len(value):
        character = value[index]
        if character in "!?;|\r\n":
            finish_segment()
            index += 1
            continue
        if character == ".":
            run_end = index + 1
            while run_end < len(value) and value[run_end] == ".":
                run_end += 1
            if run_end - index == 1 and (
                run_end == len(value) or value[run_end].isspace()
            ):
                finish_segment()
            else:
                current.append(" ")
            index = run_end
            continue
        current.append(character)
        index += 1

    finish_segment()
    return tuple(segments)


def _alphanumeric_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in value.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _contains_token_pair(
    tokens: tuple[str, ...],
    pair: tuple[str, str],
) -> bool:
    return any(
        tokens[index : index + 2] == pair
        for index in range(len(tokens) - 1)
    )


def _negative_sfw_reasons(value: str) -> tuple[str, ...]:
    reasons: list[str] = []
    for segment in _classification_segments(value):
        tokens = _alphanumeric_tokens(segment)
        if any(
            tokens[index : index + 3] == ("unsafe", "for", "work")
            for index in range(len(tokens) - 2)
        ):
            reasons.append("unsafe for work")

        target_indexes = tuple(
            index
            for index, token in enumerate(tokens)
            if token == "sfw"
            or tokens[index : index + 3] == ("safe", "for", "work")
        )
        for target_index in target_indexes:
            preceding = tokens[:target_index]
            matched_reason = next(
                (
                    f"{token} before SFW"
                    for token in SFW_NEGATION_TOKENS
                    if token in preceding
                ),
                None,
            )
            if matched_reason is None:
                matched_reason = next(
                    (
                        f"{' '.join(pair)} before SFW"
                        for pair in SFW_NEGATION_PAIRS
                        if _contains_token_pair(preceding, pair)
                    ),
                    None,
                )
            if matched_reason is not None:
                reasons.append(matched_reason)

    return tuple(dict.fromkeys(reasons))


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

    phrase_reasons = tuple(
        f"NSFW phrase '{reason}' in {source}"
        for source, text in evidence
        for reason in _negative_sfw_reasons(text)
    )
    nsfw_reasons = phrase_reasons + matches(NSFW_TERMS, "NSFW")
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
    value: ProviderRecord,
    *,
    endpoint: str,
    field: str,
    canonical_url: str,
    used_models: tuple[str, ...] = (),
) -> WorkflowCandidate:
    name = _nonblank_string(
        value.name,
        endpoint,
        f"{field}.name",
    )
    description = _plain_text(value.desc)
    published_at = _published_at(
        value.publishTime,
        endpoint,
        f"{field}.publishTime",
    )
    owner_name = _nonblank_string(
        value.owner.name,
        endpoint,
        f"{field}.owner.name",
    )
    engagement = _engagement(
        value.statisticsInfo,
        endpoint,
        f"{field}.statisticsInfo",
    )
    tags = _tags(
        value.tags,
        endpoint,
        f"{field}.tags",
    )
    covers = _covers(
        value.covers,
        endpoint,
        f"{field}.covers",
    )

    classification, reasons = _classification(
        name=name,
        description=description,
        tags=tags,
        used_models=used_models,
    )

    try:
        return WorkflowCandidate(
            executor="runninghub",
            workflow_id=value.id,
            canonical_url=canonical_url,
            name=name,
            description=description,
            author_id=value.owner.id,
            author_name=owner_name,
            published_at=published_at,
            tags=tags,
            engagement=engagement,
            covers=covers,
            content_classification=classification,
            classification_reasons=reasons,
            provider_metadata={
                "endpoint": endpoint,
                "record": _sanitize_provider_value(
                    value.model_dump(mode="python")
                ),
            },
        )
    except ValidationError as error:
        _normalized_model_error(endpoint, field, error)


class RunningHubExecutor(Executor):
    search_url = f"{BASE_URL}{SEARCH_ENDPOINT}"
    detail_url = f"{BASE_URL}{DETAIL_ENDPOINT}"
    public_request_timeout = PUBLIC_REQUEST_TIMEOUT

    def __init__(
        self,
        session: requests.Session | None = None,
        graph_fetcher: GraphFetcher | None = None,
    ) -> None:
        self.session = requests.Session() if session is None else session
        self.graph_fetcher = (
            get_workflow_json if graph_fetcher is None else graph_fetcher
        )

    @property
    def name(self) -> str:
        return "runninghub"

    def _post_public(
        self,
        *,
        url: str,
        endpoint: str,
        body: dict[str, Any],
        schema: type[ProviderSchema],
    ) -> ProviderSchema:
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
        except ValueError:
            _schema_error(endpoint, "response JSON", "body is not valid JSON")

        envelope = _validate_provider_schema(
            _ProviderEnvelope,
            payload,
            endpoint,
        )
        if envelope.code != 0:
            raise ProviderResponseError(
                endpoint=endpoint,
                code=envelope.code,
                message=envelope.msg,
                error_messages=tuple(envelope.errorMessages or ()),
            )
        return _validate_provider_schema(schema, payload, endpoint)

    def search(self, request: SearchRequest) -> SearchPage:
        envelope = self._post_public(
            url=self.search_url,
            endpoint=SEARCH_ENDPOINT,
            body={
                "size": request.size,
                "current": request.page,
                "search": request.query,
                "tags": [],
                "sort": request.sort,
            },
            schema=_SearchEnvelope,
        )
        data = envelope.data
        candidates = tuple(
            _candidate(
                record,
                endpoint=SEARCH_ENDPOINT,
                field=f"data.records[{index}]",
                canonical_url=(
                    f"https://www.runninghub.ai/post/{record.id}"
                ),
            )
            for index, record in enumerate(data.records)
        )

        size = _integer(
            data.size,
            SEARCH_ENDPOINT,
            "data.size",
            minimum=1,
        )
        current = _integer(
            data.current,
            SEARCH_ENDPOINT,
            "data.current",
            minimum=1,
        )
        total = _integer(
            data.total,
            SEARCH_ENDPOINT,
            "data.total",
            minimum=0,
        )
        _integer(
            data.pages,
            SEARCH_ENDPOINT,
            "data.pages",
            minimum=0,
        )

        try:
            return SearchPage(
                records=candidates,
                total=total,
                page=current,
                size=size,
                has_next=data.hasNext,
            )
        except ValidationError as error:
            _normalized_model_error(SEARCH_ENDPOINT, "data", error)

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
        envelope = self._post_public(
            url=self.detail_url,
            endpoint=DETAIL_ENDPOINT,
            body={"workflowId": requested_id},
            schema=_DetailEnvelope,
        )
        data = envelope.data
        if data.id != requested_id:
            _schema_error(
                DETAIL_ENDPOINT,
                "data.id",
                "does not match the requested workflow ID",
            )

        primitive_nodes = _string_inventory(
            data.primitiveNodes,
            DETAIL_ENDPOINT,
            "data.primitiveNodes",
        )
        custom_nodes = _string_inventory(
            data.customNodes,
            DETAIL_ENDPOINT,
            "data.customNodes",
        )
        used_models = _string_inventory(
            data.usedModels,
            DETAIL_ENDPOINT,
            "data.usedModels",
            sanitize_urls=True,
        )
        preview = _cover(
            data.preview,
            DETAIL_ENDPOINT,
            "data.preview",
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
                node_count=data.nodeCount,
                primitive_nodes=primitive_nodes,
                custom_nodes=custom_nodes,
                used_models=used_models,
                preview=preview,
                webapp_id=data.webappId,
            )
        except ValidationError as error:
            _normalized_model_error(DETAIL_ENDPOINT, "data", error)

    def fetch_workflow(self, ref: WorkflowRef) -> FetchedWorkflow:
        if ref.executor != self.name:
            raise ExecutorError("RunningHub workflow ref executor must be runninghub")

        canonical_url = None
        fetched_ref = ref
        if ref.canonical_url is not None:
            canonical_url = _validated_canonical_url(
                ref.canonical_url,
                ref.workflow_id,
            )
            fetched_ref = WorkflowRef(
                executor=ref.executor,
                workflow_id=ref.workflow_id,
                canonical_url=canonical_url,
            )
        detail = self.detail(
            ref.workflow_id,
            canonical_url=canonical_url,
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
            except (RecursionError, ValueError):
                _schema_error(GRAPH_ENDPOINT, "graph", "expected a JSON object")
        if not isinstance(graph, dict):
            _schema_error(GRAPH_ENDPOINT, "graph", "expected a JSON object")
        _validate_graph_complexity(graph)

        try:
            return FetchedWorkflow(ref=fetched_ref, detail=detail, graph=graph)
        except (RecursionError, ValidationError):
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
        unique_models = set(detail.used_models)
        lora_names = tuple(
            sorted(
                name
                for name in unique_models
                if LORA_INVENTORY_TOKEN.search(name) is not None
            )
        )
        model_names = tuple(
            sorted(
                name
                for name in unique_models
                if LORA_INVENTORY_TOKEN.search(name) is None
            )
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
