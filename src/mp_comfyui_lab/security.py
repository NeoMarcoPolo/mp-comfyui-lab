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


class SafetyError(ValueError):
    pass


def safe_filename(name: str) -> str:
    if not isinstance(name, str):
        raise SafetyError("filename must be a string")
    if not name.strip():
        raise SafetyError("filename must not be empty")
    if name in {".", ".."}:
        raise SafetyError("filename must not contain dot traversal")
    if "/" in name or "\\" in name:
        raise SafetyError("filename must not contain path separators")
    if any(unicodedata.category(character) == "Cc" for character in name):
        raise SafetyError("filename must not contain control characters")
    return name


def _strip_sensitive_url_query(key: Any, value: Any) -> Any:
    if not isinstance(key, str) or "url" not in key.casefold():
        return value
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


def sanitize_for_persistence(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in SECRET_KEYS:
                sanitized[key] = REDACTED
                continue
            sanitized_item = sanitize_for_persistence(item)
            sanitized[key] = _strip_sensitive_url_query(key, sanitized_item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_persistence(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_persistence(item) for item in value)
    return value
