"""Router security validation, allowlist enforcement, and redaction."""

from __future__ import annotations

import ipaddress
import urllib.parse
from collections.abc import Iterable

from crimson0020.translation_runtime.errors import TranslationRuntimeError

# Fixed HTTPS origin allowlist for AI Router providers and discovery endpoints
ALLOWED_ORIGINS: frozenset[str] = frozenset(
    {
        "opencode.ai",
        "api.kilo.ai",
        "hermes.ai.unturf.com",
        "duckduckgo.com",
        "html.duckduckgo.com",
        "stablehorde.net",
        "api.llm7.io",
        "blockrun.ai",
        "vireonix.ai",
        "ai-api.cehpoint.co.in",
        "cloudflare.com",
        "ai.cloudflare.com",
        "translate.google.com",
        "api.mymemory.translated.net",
    }
)

SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "api-key",
        "proxy-authorization",
    }
)


class SecurityValidationError(TranslationRuntimeError):
    """Raised when an outbound URL, origin, or header violates security rules."""


def validate_https_url(url: str, allowed_origins: frozenset[str] = ALLOWED_ORIGINS) -> urllib.parse.ParseResult:
    """Validates URL adheres to fixed HTTPS allowlist, default port, and no userinfo."""
    if not isinstance(url, str) or not url.strip():
        raise SecurityValidationError("URL must be a non-empty string.")

    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise SecurityValidationError(f"Forbidden scheme '{parsed.scheme}'. Only HTTPS is allowed.")

    if parsed.username or parsed.password:
        raise SecurityValidationError("Userinfo in URL is strictly prohibited.")

    if parsed.port is not None and parsed.port != 443:
        raise SecurityValidationError(f"Non-standard port '{parsed.port}' is strictly prohibited.")

    hostname = (parsed.hostname or "").lower()
    if hostname not in allowed_origins:
        raise SecurityValidationError(f"Host '{hostname}' is not in the approved router origin allowlist.")

    return parsed


def validate_public_dns_addresses(hostname: str, addresses: Iterable[str]) -> tuple[str, ...]:
    """Reject DNS answers that could route an approved hostname to non-public IP space."""
    resolved = tuple(dict.fromkeys(addresses))
    if not resolved:
        raise SecurityValidationError(f"Host '{hostname}' did not resolve to an address.")

    for address in resolved:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise SecurityValidationError(
                f"Host '{hostname}' resolved to an invalid IP address."
            ) from exc
        if not parsed.is_global:
            raise SecurityValidationError(
                f"Host '{hostname}' resolved to a non-public IP address."
            )
    return resolved


def redact_sensitive_content(headers: dict[str, str]) -> dict[str, str]:
    """Redacts authorization, cookies, and tokens from request/response headers."""
    redacted: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in SENSITIVE_HEADERS:
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted
