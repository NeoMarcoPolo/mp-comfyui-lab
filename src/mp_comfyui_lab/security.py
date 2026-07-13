import json
import re
from typing import Any
import unicodedata
from urllib.parse import urlsplit, urlunsplit


CANONICAL_SECRET_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "cookies",
        "credential",
        "refreshtoken",
        "sig",
        "signature",
        "signedurl",
        "token",
        "xamzcredential",
        "xamzsignature",
        "xapikey",
    }
)
REDACTED = "[REDACTED]"
MAX_DIAGNOSTIC_JSON_LENGTH = 65_536
MAX_URL_DESCRIPTOR_GAP = 256
WINDOWS_FORBIDDEN_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "conin$",
        "conout$",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
        *(f"com{number}" for number in ("\u00b9", "\u00b2", "\u00b3")),
        *(f"lpt{number}" for number in ("\u00b9", "\u00b2", "\u00b3")),
    }
)
HTTP_URL_START = re.compile(r"https?://", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = frozenset(".,;:!?")
TRAILING_URL_BRACKETS = ((")", "("), ("]", "["), ("}", "{"))
DIAGNOSTIC_ASSIGNMENT_START = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?P<key_quote>[\"']?)"
    r"(?P<key>[A-Za-z][A-Za-z0-9_.\-/]*)"
    r"(?P=key_quote)"
    r"(?P<separator>\s*[:=]\s*)",
)
DIAGNOSTIC_AUTHORIZATION_SCHEME = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?P<scheme>Bearer|Basic|Token|Digest|ApiKey|AWS4-HMAC-SHA256)"
    r"(?P<separator>\s+)",
    re.IGNORECASE,
)


class SafetyError(ValueError):
    pass


def _canonical_key(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if character.isalnum()
    )


def _is_secret_key(value: str) -> bool:
    return _canonical_key(value) in CANONICAL_SECRET_KEYS


def safe_filename(name: str) -> str:
    if not isinstance(name, str):
        raise SafetyError("filename must be a string")
    if not name.strip():
        raise SafetyError("filename must not be empty")
    if name in {".", ".."}:
        raise SafetyError("filename must not contain dot traversal")
    if any(character in WINDOWS_FORBIDDEN_FILENAME_CHARACTERS for character in name):
        raise SafetyError("filename contains a Windows-forbidden character")
    if any(unicodedata.category(character) == "Cc" for character in name):
        raise SafetyError("filename must not contain control characters")
    if name.endswith((".", " ")):
        raise SafetyError("filename must not end with a dot or space")
    normalized_basename = name.partition(".")[0].rstrip(" .").casefold()
    if normalized_basename in WINDOWS_RESERVED_BASENAMES:
        raise SafetyError("filename uses a reserved Windows device basename")
    return name


def _strip_url_query(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    before_fragment = value.partition("#")[0]
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or "?" not in before_fragment
    ):
        return value

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment))


def _split_trailing_url_punctuation(value: str) -> tuple[str, str]:
    suffix = ""
    while value and value[-1] in TRAILING_URL_PUNCTUATION:
        suffix = value[-1] + suffix
        value = value[:-1]

    changed = True
    while changed:
        changed = False
        for closing, opening in TRAILING_URL_BRACKETS:
            if value.endswith(closing) and value.count(closing) > value.count(opening):
                suffix = closing + suffix
                value = value[:-1]
                changed = True
    return value, suffix


def _separator_precedes_url(value: str, separator_index: int) -> bool:
    gap_start = separator_index + 1
    gap_end = min(len(value), gap_start + MAX_URL_DESCRIPTOR_GAP)
    next_url = HTTP_URL_START.search(value, gap_start, gap_end)
    if next_url is None:
        return False

    gap = value[gap_start : next_url.start()]
    return all(
        character.isalnum()
        or character.isspace()
        or character in "_-=[](){}.'\":/"
        for character in gap
    )


def _embedded_url_end(value: str, start: int) -> int:
    bracket_depth = {"(": 0, "[": 0, "{": 0}
    closing_brackets = {")": "(", "]": "[", "}": "{"}
    index = start + len(HTTP_URL_START.match(value, start).group(0))

    while index < len(value):
        character = value[index]
        if character.isspace() or character in "<>\"'":
            break
        if character in ",|;" and _separator_precedes_url(value, index):
            break
        if character in bracket_depth:
            bracket_depth[character] += 1
        elif character in closing_brackets:
            opening = closing_brackets[character]
            if bracket_depth[opening] == 0:
                break
            bracket_depth[opening] -= 1
        index += 1
    return index


def sanitize_embedded_urls(value: str) -> str:
    """Remove query components from HTTP(S) URLs embedded in untrusted text."""

    sanitized: list[str] = []
    cursor = 0
    while match := HTTP_URL_START.search(value, cursor):
        start = match.start()
        end = _embedded_url_end(value, start)
        url, suffix = _split_trailing_url_punctuation(value[start:end])
        sanitized.append(value[cursor:start])
        sanitized.append(f"{_strip_url_query(url)}{suffix}")
        cursor = end
    sanitized.append(value[cursor:])
    return "".join(sanitized)


def _quoted_diagnostic_value_end(value: str, start: int) -> int:
    quote = value[start]
    index = start + 1
    escaped = False
    while index < len(value):
        character = value[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            return index + 1
        index += 1
    return len(value)


def _diagnostic_value_replacement(
    value: str,
    start: int,
    *,
    authorization: bool,
) -> tuple[int, str]:
    if value.startswith(REDACTED, start):
        return start + len(REDACTED), REDACTED
    if start < len(value) and value[start] in {"'", '"'}:
        end = _quoted_diagnostic_value_end(value, start)
        quote = value[start]
        return end, f"{quote}{REDACTED}{quote}"

    delimiters = ",;|}]"
    end = start
    while end < len(value):
        character = value[end]
        if character in delimiters or (not authorization and character.isspace()):
            break
        end += 1

    content_end = end
    while content_end > start and value[content_end - 1].isspace():
        content_end -= 1
    return end, f"{REDACTED}{value[content_end:end]}"


def _redact_diagnostic_assignments(value: str) -> str:
    sanitized: list[str] = []
    cursor = 0
    search_from = 0

    while match := DIAGNOSTIC_ASSIGNMENT_START.search(value, search_from):
        if not _is_secret_key(match.group("key")):
            search_from = match.end()
            continue

        value_end, replacement = _diagnostic_value_replacement(
            value,
            match.end(),
            authorization=(
                _canonical_key(match.group("key")) == "authorization"
            ),
        )
        sanitized.append(value[cursor : match.end()])
        sanitized.append(replacement)
        cursor = value_end
        search_from = value_end

    sanitized.append(value[cursor:])
    return "".join(sanitized)


def _redact_authorization_schemes(value: str) -> str:
    sanitized: list[str] = []
    cursor = 0
    search_from = 0

    while match := DIAGNOSTIC_AUTHORIZATION_SCHEME.search(value, search_from):
        value_end, replacement = _diagnostic_value_replacement(
            value,
            match.end(),
            authorization=False,
        )
        sanitized.append(value[cursor : match.end()])
        sanitized.append(replacement)
        cursor = value_end
        search_from = value_end

    sanitized.append(value[cursor:])
    return "".join(sanitized)


def _sanitize_diagnostic_text(value: str) -> str:
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in value
    )
    normalized = re.sub(r"\s+", " ", without_controls).strip()
    without_url_queries = sanitize_embedded_urls(normalized)
    without_assignments = _redact_diagnostic_assignments(without_url_queries)
    return _redact_authorization_schemes(without_assignments)


def _sanitize_diagnostic_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and _is_secret_key(key)
                else _sanitize_diagnostic_json(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_diagnostic_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_diagnostic_text(value)
    return value


def sanitize_untrusted_diagnostic(value: str) -> str:
    """Redact credentials while retaining useful untrusted provider diagnostics."""

    stripped = value.strip()
    if (
        len(stripped) <= MAX_DIAGNOSTIC_JSON_LENGTH
        and stripped.startswith(("{", "["))
    ):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                sanitized = _sanitize_diagnostic_json(parsed)
                return json.dumps(
                    sanitized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
        except (RecursionError, ValueError):
            pass
    return _sanitize_diagnostic_text(value)


def _sanitize_for_persistence(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_secret_key(key):
                sanitized[key] = REDACTED
                continue
            sanitized[key] = _sanitize_for_persistence(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_persistence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_persistence(item) for item in value)
    if isinstance(value, str):
        return sanitize_embedded_urls(value)
    return value


def sanitize_for_persistence(value: Any) -> Any:
    return _sanitize_for_persistence(value)
