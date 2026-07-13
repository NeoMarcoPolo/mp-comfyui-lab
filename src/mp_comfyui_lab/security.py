import json
import re
from typing import Any
import unicodedata
from urllib.parse import urlsplit, urlunsplit


SECRET_KEYS = frozenset(
    {
        "apikey",
        "api_key",
        "authorization",
        "cookie",
        "cookies",
        "token",
        "access_token",
        "refresh_token",
        "signed_url",
    }
)
REDACTED = "[REDACTED]"
MAX_DIAGNOSTIC_JSON_LENGTH = 65_536
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
HTTP_URL_SUBSTRING = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
TRAILING_URL_PUNCTUATION = frozenset(".,;:!?")
TRAILING_URL_BRACKETS = ((")", "("), ("]", "["), ("}", "{"))
SECRET_KEY_PATTERN = "|".join(
    sorted((re.escape(key) for key in SECRET_KEYS), key=len, reverse=True)
)
DIAGNOSTIC_SECRET_ASSIGNMENT = re.compile(
    rf"(?P<prefix>(?P<key_quote>[\"']?)(?:{SECRET_KEY_PATTERN})"
    rf"(?P=key_quote)\s*[:=]\s*)"
    r"(?P<value>\[REDACTED\]|Bearer\s+[^\s,;}\]]+|\"(?:\\.|[^\"\\])*\"|"
    r"'(?:\\.|[^'\\])*'|[^\s,;}\]]+)",
    re.IGNORECASE,
)
DIAGNOSTIC_BEARER = re.compile(
    r"\bBearer\s+(?!\[REDACTED\])[^\s,;\"'}\]]+",
    re.IGNORECASE,
)


class SafetyError(ValueError):
    pass


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


def sanitize_embedded_urls(value: str) -> str:
    """Remove query components from HTTP(S) URLs embedded in untrusted text."""

    def sanitize_match(match: re.Match[str]) -> str:
        url, suffix = _split_trailing_url_punctuation(match.group(0))
        return f"{_strip_url_query(url)}{suffix}"

    return HTTP_URL_SUBSTRING.sub(sanitize_match, value)


def _redact_diagnostic_assignment(match: re.Match[str]) -> str:
    value = match.group("value")
    quote = (
        value[0]
        if len(value) >= 2 and value[0] in {"'", '"'} and value[0] == value[-1]
        else ""
    )
    return f"{match.group('prefix')}{quote}{REDACTED}{quote}"


def _sanitize_diagnostic_text(value: str) -> str:
    without_controls = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in value
    )
    normalized = re.sub(r"\s+", " ", without_controls).strip()
    without_url_queries = sanitize_embedded_urls(normalized)
    without_assignments = DIAGNOSTIC_SECRET_ASSIGNMENT.sub(
        _redact_diagnostic_assignment,
        without_url_queries,
    )
    return DIAGNOSTIC_BEARER.sub("Bearer [REDACTED]", without_assignments)


def _sanitize_diagnostic_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and key.casefold() in SECRET_KEYS
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
            if isinstance(key, str) and key.casefold() in SECRET_KEYS:
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
