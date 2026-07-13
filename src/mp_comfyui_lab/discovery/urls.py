from __future__ import annotations

from enum import StrEnum
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict


_PUBLIC_HOSTS = frozenset({"runninghub.ai", "www.runninghub.ai"})
_PUBLIC_PATH = re.compile(
    r"/(?:[a-z]{2}(?:-[a-z]{2})?/)?(?P<kind>post|workflow)/"
    r"(?P<workflow_id>[0-9]+)/?"
)
_FORBIDDEN_RAW_CHARACTER = re.compile(r"[\s\x00-\x1f\x7f]")
_UNSUPPORTED_MESSAGE = (
    "RunningHub source must be an HTTPS public post or workflow URL"
)


class UnsupportedSourceError(ValueError):
    pass


class RunningHubUrlKind(StrEnum):
    POST = "post"
    WORKFLOW = "workflow"


class ParsedRunningHubUrl(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    workflow_id: str
    kind: RunningHubUrlKind
    canonical_url: str


def parse_runninghub_url(url: str) -> ParsedRunningHubUrl:
    if (
        not isinstance(url, str)
        or not url
        or _FORBIDDEN_RAW_CHARACTER.search(url) is not None
    ):
        raise UnsupportedSourceError(_UNSUPPORTED_MESSAGE)

    try:
        parsed = urlsplit(url)
    except ValueError:
        raise UnsupportedSourceError(_UNSUPPORTED_MESSAGE) from None

    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() not in _PUBLIC_HOSTS
        or parsed.fragment
    ):
        raise UnsupportedSourceError(_UNSUPPORTED_MESSAGE)

    matched = _PUBLIC_PATH.fullmatch(parsed.path)
    if matched is None:
        raise UnsupportedSourceError(_UNSUPPORTED_MESSAGE)

    kind = RunningHubUrlKind(matched.group("kind"))
    workflow_id = matched.group("workflow_id")
    return ParsedRunningHubUrl(
        workflow_id=workflow_id,
        kind=kind,
        canonical_url=(
            f"https://www.runninghub.ai/{kind.value}/{workflow_id}"
        ),
    )
