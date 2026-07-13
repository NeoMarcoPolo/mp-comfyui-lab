from typing import Any
import unicodedata
from urllib.parse import parse_qsl, urlsplit, urlunsplit


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
SENSITIVE_QUERY_KEY_PARTS = ("token", "signature", "expires", "credential")
REDACTED = "[REDACTED]"
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


def _strip_sensitive_url_query(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.query:
        return value

    query_keys = (
        query_key.casefold()
        for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    )
    if not any(
        sensitive_part in query_key
        for query_key in query_keys
        for sensitive_part in SENSITIVE_QUERY_KEY_PARTS
    ):
        return value

    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment))


def _sanitize_for_persistence(value: Any, url_context: bool) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in SECRET_KEYS:
                sanitized[key] = REDACTED
                continue
            child_url_context = url_context or (
                isinstance(key, str) and "url" in key.casefold()
            )
            sanitized[key] = _sanitize_for_persistence(item, child_url_context)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_persistence(item, url_context) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_for_persistence(item, url_context) for item in value)
    if url_context:
        return _strip_sensitive_url_query(value)
    return value


def sanitize_for_persistence(value: Any) -> Any:
    return _sanitize_for_persistence(value, url_context=False)
