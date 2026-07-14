from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import stat
from typing import Annotated, Any, NoReturn, TypeVar
import unicodedata
from urllib.parse import urlsplit

from mp_runninghub_api.config import load_default_config
from mp_runninghub_api.runninghub import (
    create_task,
    get_task_info,
    get_task_outputs,
    get_task_status,
    get_workflow_json,
)
from mp_runninghub_api.types import NodeInfo
from mp_runninghub_api.upload import upload_file
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
    AmbiguousSubmissionError,
    CancelResult,
    CapabilityReport,
    Executor,
    ExecutorError,
    FetchedWorkflow,
    PollResult,
    ProviderOutput,
    ProviderResponseError,
    ProviderSchemaError,
    ProviderTaskState,
    SearchPage,
    SearchRequest,
    SubmissionReceipt,
    TransientExecutorError,
    ValidationIssue,
    ValidationReport,
    WorkflowRef,
)
from mp_comfyui_lab.experiments.models import MaterializedVariant, ResolvedFixtures
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import SafetyError, safe_filename
from mp_comfyui_lab.security import (
    sanitize_embedded_urls,
    sanitize_for_persistence,
    sanitize_untrusted_diagnostic,
)


BASE_URL = "https://www.runninghub.ai"
SEARCH_ENDPOINT = "/api/search/workflow"
DETAIL_ENDPOINT = "/api/portal/workflow/detail"
GRAPH_ENDPOINT = "/api/openapi/getJsonApiFormat"
PUBLIC_REQUEST_TIMEOUT = (10, 60)
PROVIDER_NUMERIC_STRING_MAX_LENGTH = 64
MAX_GRAPH_DEPTH = 256
MAX_GRAPH_NODES = 100_000

NSFW_TERMS = frozenset({"nsfw", "nude", "nudity", "\u88f8\u4f53", "\u6210\u4eba", "\u8272\u60c5"})
SFW_CONTEXT_WINDOW = 8
SFW_CONTRAST_TOKENS = frozenset({"although", "but", "however", "though", "yet"})
SFW_NEGATIVE_IDIOMS = (
    ("anything", "but"),
    ("nowhere", "near"),
    ("no", "longer"),
    ("far", "from"),
)
SFW_AFTER_NEGATIVE_IDIOMS = (
    ("absolutely", "not"),
    ("certainly", "not"),
    ("definitely", "not"),
)
LORA_INVENTORY_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])loras?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


GraphFetcher = Callable[[str | None, str], Any]
CredentialLoader = Callable[[], Any]
UploadCall = Callable[..., str]
CreateTaskCall = Callable[..., Any]
StatusCall = Callable[..., Any]
InfoCall = Callable[..., Any]
OutputsCall = Callable[..., Any]

_UPLOAD_MEDIA_TYPES = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/webp": "image",
    "image/gif": "image",
    "video/mp4": "video",
    "video/webm": "video",
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/ogg": "audio",
    "application/zip": "zip",
}
_FIXTURE_PLACEHOLDER = re.compile(r"^fixture://([0-9]+)$")
_PROVIDER_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$")
_PAID_PROVIDER_STATES = {
    "SUBMITTED": "SUBMITTED",
    "QUEUED": "QUEUED",
    "RUNNING": "RUNNING",
    "SUCCESS": "SUCCESS",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
}
_OUTPUT_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "zip": "application/zip",
    **{media_type: media_type for media_type in _UPLOAD_MEDIA_TYPES},
}


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
        if character in ",;!|\r\n":
            finish_segment()
            index += 1
            continue
        if character == "?":
            current.append(" ")
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


def _contains_token_sequence(
    tokens: tuple[str, ...],
    sequence: tuple[str, ...],
) -> bool:
    return any(
        tokens[index : index + len(sequence)] == sequence
        for index in range(len(tokens) - len(sequence) + 1)
    )


def _sfw_target_length(tokens: tuple[str, ...], index: int) -> int:
    if tokens[index] in {"sfw", "\u5b89\u5168"}:
        return 1
    if tokens[index : index + 3] == ("safe", "for", "work"):
        return 3
    return 0


def _not_context(
    tokens: tuple[str, ...],
    before_start: int,
    target_index: int,
) -> str | None:
    not_index = next(
        (
            index
            for index in range(target_index - 1, before_start - 1, -1)
            if tokens[index] == "not"
        ),
        None,
    )
    if not_index is None:
        return None

    following = tokens[not_index + 1 : target_index]
    if following[:2] == ("a", "problem") or following[:1] == ("only",):
        return "affirmative"
    if following[:1] in (("sure",), ("certain",), ("necessarily",)):
        return "ambiguous"
    return "negative"


def _has_affirmative_sfw_evidence(value: str) -> bool:
    affirmative = False

    for segment in _classification_segments(value):
        tokens = _alphanumeric_tokens(segment)
        clause_start = 0
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in SFW_CONTRAST_TOKENS:
                if not (
                    token == "but"
                    and index > 0
                    and tokens[index - 1] == "anything"
                ):
                    clause_start = index + 1
                index += 1
                continue

            if tokens[index : index + 3] == ("unsafe", "for", "work"):
                index += 3
                continue

            target_length = _sfw_target_length(tokens, index)
            if target_length == 0:
                index += 1
                continue

            before_start = max(clause_start, index - SFW_CONTEXT_WINDOW)
            before = tokens[before_start:index]
            after_end = min(
                len(tokens),
                index + target_length + SFW_CONTEXT_WINDOW,
            )
            after = tokens[index + target_length : after_end]
            contrast_index = next(
                (
                    offset
                    for offset, after_token in enumerate(after)
                    if after_token in SFW_CONTRAST_TOKENS
                ),
                None,
            )
            if contrast_index is not None:
                after = after[:contrast_index]

            after_negative = next(
                (
                    " ".join(idiom)
                    for idiom in SFW_AFTER_NEGATIVE_IDIOMS
                    if _contains_token_sequence(after, idiom)
                ),
                None,
            )
            before_negative = next(
                (
                    " ".join(idiom)
                    for idiom in SFW_NEGATIVE_IDIOMS
                    if _contains_token_sequence(before, idiom)
                ),
                None,
            )
            if after_negative is not None or before_negative is not None:
                index += target_length
                continue
            if "hardly" in before or "never" in before:
                index += target_length
                continue
            if before[-1:] == ("non",):
                index += target_length
                continue

            if before[-4:] == ("could", "not", "be", "more"):
                affirmative = True
                index += target_length
                continue

            not_context = _not_context(tokens, before_start, index)
            if not_context == "affirmative":
                affirmative = True
            elif not (
                not_context in {"negative", "ambiguous"}
                or any(token in before for token in ("no", "non", "unsafe"))
                or "not" in after
            ):
                affirmative = True
            index += target_length

    return affirmative


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

    sfw_evidence = tuple(
        (source, _has_affirmative_sfw_evidence(text))
        for source, text in evidence
    )
    sfw_reasons = tuple(
        f"SFW evidence in {source}"
        for source, affirmative in sfw_evidence
        if affirmative
    )
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
        *,
        credential_loader: CredentialLoader = load_default_config,
        upload_call: UploadCall = upload_file,
        create_task_call: CreateTaskCall = create_task,
        status_call: StatusCall = get_task_status,
        info_call: InfoCall = get_task_info,
        outputs_call: OutputsCall = get_task_outputs,
    ) -> None:
        self.session = requests.Session() if session is None else session
        self.graph_fetcher = (
            get_workflow_json if graph_fetcher is None else graph_fetcher
        )
        self._credential_loader = credential_loader
        self._upload_call = upload_call
        self._create_task_call = create_task_call
        self._status_call = status_call
        self._info_call = info_call
        self._outputs_call = outputs_call

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
        errors, _ = self._fixture_bindings(variant, fixtures)
        return ValidationReport(errors=tuple(errors))

    @staticmethod
    def _fixture_bindings(
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> tuple[list[ValidationIssue], dict[int, tuple[str, str]]]:
        errors: list[ValidationIssue] = []
        fixture_by_index = {item.binding_index: item for item in fixtures.items}
        if len(fixture_by_index) != len(fixtures.items):
            errors.append(
                ValidationIssue(
                    code="duplicate_fixture_binding",
                    message="fixture binding indexes must be unique",
                )
            )

        occurrences: dict[int, list[tuple[str, str]]] = {}
        for node_id, node in variant.graph.items():
            if not isinstance(node, Mapping):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            for field_name, value in inputs.items():
                if not isinstance(value, str) or not value.startswith("fixture://"):
                    continue
                match = _FIXTURE_PLACEHOLDER.fullmatch(value)
                if match is None:
                    errors.append(
                        ValidationIssue(
                            code="malformed_fixture_placeholder",
                            message="fixture placeholders must use fixture://<binding-index>",
                        )
                    )
                    continue
                binding_index = int(match.group(1))
                occurrences.setdefault(binding_index, []).append(
                    (str(node_id), str(field_name))
                )

        for binding_index, fixture in fixture_by_index.items():
            matches = occurrences.get(binding_index, [])
            if len(matches) != 1:
                errors.append(
                    ValidationIssue(
                        code="ambiguous_fixture_placeholder",
                        message=(
                            f"fixture binding {binding_index} must occur exactly once"
                        ),
                    )
                )
            if fixture.media_type not in _UPLOAD_MEDIA_TYPES:
                errors.append(
                    ValidationIssue(
                        code="unsupported_fixture_media_type",
                        message=f"fixture binding {binding_index} has unsupported media type",
                    )
                )
            if not fixture.path.is_file():
                errors.append(
                    ValidationIssue(
                        code="missing_fixture_file",
                        message=f"fixture binding {binding_index} is not a regular file",
                    )
                )
        for binding_index in occurrences.keys() - fixture_by_index.keys():
            errors.append(
                ValidationIssue(
                    code="missing_fixture_binding",
                    message=f"placeholder binding {binding_index} has no resolved fixture",
                )
            )
        bindings = {
            binding_index: matches[0]
            for binding_index, matches in occurrences.items()
            if binding_index in fixture_by_index and len(matches) == 1
        }
        return errors, bindings

    def submit(
        self,
        variant: MaterializedVariant,
        fixtures: ResolvedFixtures,
    ) -> SubmissionReceipt:
        errors, bindings = self._fixture_bindings(variant, fixtures)
        if errors:
            raise ExecutorError("RunningHub variant validation failed before submit")
        for fixture in fixtures.items:
            self._verify_fixture_snapshot(fixture)
        try:
            config = self._credential_loader()
        except Exception as error:
            raise ExecutorError("RunningHub credentials could not be loaded") from error
        api_key = getattr(config, "apiKey", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ExecutorError("RunningHub credentials are missing an API key")

        materialized_graph = thaw_json(variant.graph)
        node_info_list: list[NodeInfo] = []
        for fixture in fixtures.items:
            node_id, field_name = bindings[fixture.binding_index]
            file_type = _UPLOAD_MEDIA_TYPES[fixture.media_type]
            self._verify_fixture_snapshot(fixture)
            try:
                uploaded_name = self._upload_call(
                    api_key=None,
                    file_path=str(fixture.path),
                    file_type=file_type,
                )
            except Exception as error:
                raise ExecutorError("RunningHub fixture upload failed") from error
            self._verify_fixture_snapshot(fixture)
            if not isinstance(uploaded_name, str):
                raise ExecutorError("RunningHub upload returned an invalid filename")
            uploaded_name = uploaded_name.strip()
            try:
                safe_filename(uploaded_name)
            except SafetyError as error:
                raise ExecutorError("RunningHub upload returned an unsafe filename") from error
            materialized_graph[node_id]["inputs"][field_name] = uploaded_name
            node_info_list.append(
                NodeInfo(
                    nodeId=node_id,
                    fieldName=field_name,
                    fieldValue=uploaded_name,
                )
            )

        try:
            result = self._create_task_call(
                api_key=None,
                workflow_id=variant.source_workflow_id,
                node_info_list=node_info_list,
                workflow=json.dumps(materialized_graph, ensure_ascii=False),
                instance_type=variant.instance_type,
            )
        except Exception as error:
            if self._ambiguous_request_error(error):
                raise AmbiguousSubmissionError(
                    "RunningHub submission outcome is ambiguous"
                ) from error
            raise ExecutorError("RunningHub submission was rejected") from error

        task_id = getattr(result, "taskId", None)
        if (
            not isinstance(task_id, str)
            or _PROVIDER_TASK_ID.fullmatch(task_id.strip()) is None
            or task_id.strip().casefold() == "none"
        ):
            raise AmbiguousSubmissionError(
                "RunningHub submission returned no usable task ID"
            )
        task_status = getattr(result, "taskStatus", None)
        provider_state = (
            None
            if task_status is None
            else self._provider_task_state(task_status)
        )
        return SubmissionReceipt(
            provider_task_id=task_id.strip(),
            provider_state=provider_state,
        )

    @staticmethod
    def _verify_fixture_snapshot(fixture: Any) -> None:
        try:
            descriptor = os.open(
                fixture.path,
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as error:
            raise ExecutorError("RunningHub fixture snapshot cannot be opened") from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ExecutorError("RunningHub fixture snapshot is not a regular file")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or digest.hexdigest() != fixture.sha256
        ):
            raise ExecutorError("RunningHub fixture snapshot failed integrity check")

    @staticmethod
    def _ambiguous_request_error(error: BaseException) -> bool:
        current: BaseException | None = error
        while current is not None:
            if isinstance(current, requests.RequestException):
                return current.response is None
            current = current.__cause__
        return False

    @staticmethod
    def _provider_task_state(value: Any) -> ProviderTaskState:
        if not isinstance(value, str) or not value.strip():
            raise ExecutorError("RunningHub returned a malformed task status")
        normalized = _PAID_PROVIDER_STATES.get(value.strip().upper())
        if normalized is None:
            return ProviderTaskState.UNKNOWN
        return ProviderTaskState(normalized)

    def poll(self, provider_task_id: str) -> PollResult:
        if not isinstance(provider_task_id, str) or not provider_task_id.strip():
            raise ExecutorError("RunningHub poll requires a nonblank task ID")
        task_id = provider_task_id.strip()
        try:
            config = self._credential_loader()
        except Exception as error:
            raise ExecutorError("RunningHub credentials could not be loaded") from error
        api_key = getattr(config, "apiKey", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ExecutorError("RunningHub credentials are missing an API key")
        try:
            raw_status = self._status_call(api_key=None, task_id=task_id)
            info = self._info_call(api_key=None, task_id=task_id)
        except Exception as error:
            raise TransientExecutorError("RunningHub poll request failed") from error
        if not isinstance(info, dict):
            raise ProviderSchemaError("RunningHub task info must be an object")
        observed_task_id = info.get("taskId")
        if observed_task_id is not None and observed_task_id != task_id:
            raise ProviderSchemaError("RunningHub task info returned a different task ID")
        state = self._provider_task_state(raw_status)
        metadata = sanitize_for_persistence(info)
        diagnostic = sanitize_untrusted_diagnostic(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        )
        return PollResult(
            provider_task_id=task_id,
            state=state,
            diagnostic=diagnostic,
            provider_metadata=metadata,
        )

    def collect_outputs(self, provider_task_id: str) -> tuple[ProviderOutput, ...]:
        if not isinstance(provider_task_id, str) or not provider_task_id.strip():
            raise ExecutorError("RunningHub output collection requires a task ID")
        task_id = provider_task_id.strip()
        try:
            config = self._credential_loader()
        except Exception as error:
            raise ExecutorError("RunningHub credentials could not be loaded") from error
        api_key = getattr(config, "apiKey", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ExecutorError("RunningHub credentials are missing an API key")
        try:
            raw_outputs = self._outputs_call(api_key=None, task_id=task_id)
        except Exception as error:
            raise TransientExecutorError(
                "RunningHub output collection failed"
            ) from error
        if not isinstance(raw_outputs, (list, tuple)):
            raise ProviderSchemaError("RunningHub outputs must be an ordered array")

        outputs: list[ProviderOutput] = []
        for provider_order, item in enumerate(raw_outputs):
            url = getattr(item, "fileUrl", None)
            node_id = getattr(item, "nodeId", None)
            file_type = getattr(item, "fileType", None)
            if not isinstance(url, str):
                raise ProviderSchemaError("RunningHub output URL must be a string")
            try:
                _provider_http_url(url)
            except ValueError as error:
                raise ProviderSchemaError("RunningHub output URL is invalid") from error
            if not isinstance(node_id, str) or not node_id.strip():
                raise ProviderSchemaError("RunningHub output node ID is missing")
            if not isinstance(file_type, str):
                raise ProviderSchemaError("RunningHub output media type is missing")
            media_type = _OUTPUT_MEDIA_TYPES.get(file_type.strip().casefold())
            if media_type is None:
                raise ProviderSchemaError("RunningHub output media type is unknown")
            file_name = Path(urlsplit(url).path).name
            try:
                safe_filename(file_name)
            except SafetyError as error:
                raise ProviderSchemaError("RunningHub output filename is unsafe") from error
            outputs.append(
                ProviderOutput(
                    node_id=node_id.strip(),
                    url=url,
                    media_type=media_type,
                    file_name=file_name,
                    provider_order=provider_order,
                    provider_metadata=sanitize_for_persistence(
                        {
                            "task_cost_time": getattr(item, "taskCostTime", None),
                            "consume_coins": getattr(item, "consumeCoins", None),
                        }
                    ),
                )
            )
        return tuple(outputs)

    def cancel(self, provider_task_id: str) -> CancelResult:
        del provider_task_id
        return CancelResult(
            supported=False,
            cancelled=False,
            diagnostic=(
                "RunningHub API does not expose a supported cancellation operation"
            ),
        )
