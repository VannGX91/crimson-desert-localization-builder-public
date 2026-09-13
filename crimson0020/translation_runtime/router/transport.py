"""Safe direct HTTP transport for router with timeouts, size bounds, and redirects check."""

from __future__ import annotations

import http.client
import socket
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass

from crimson0020.translation_runtime.contracts import FailureKind
from crimson0020.translation_runtime.job_control import CancellationToken
from crimson0020.translation_runtime.router.security import (
    SecurityValidationError,
    redact_sensitive_content,
    validate_https_url,
    validate_public_dns_addresses,
)

MAX_RESPONSE_BODY_BYTES = 2 * 1024 * 1024  # 2 MiB
MAX_REDIRECTS = 3
CONNECT_TIMEOUT_SECONDS = 10.0
# Free anonymous LLM routes that cannot answer inside this window are poor
# foreground translation lanes. Keeping the bound at 30s prevents one stale
# fallback from making the UI look dead for a full minute.
READ_TIMEOUT_SECONDS = 30.0

InjectedTransportCallable = Callable[
    [str, dict[str, str], bytes | None, float],
    tuple[int, dict[str, str], bytes],
]
ResolverCallable = Callable[[str], tuple[str, ...]]


def _resolve_public_host(hostname: str) -> tuple[str, ...]:
    addresses = tuple(
        entry[4][0]
        for entry in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    )
    return validate_public_dns_addresses(hostname, addresses)


def _perform_pinned_https_request(
    url: str,
    headers: dict[str, str],
    data: bytes | None,
    connect_timeout: float,
    read_timeout: float,
    pinned_ip: str,
) -> tuple[int, dict[str, str], bytes]:
    """Perform one HTTPS request while connecting only to a prevalidated IP."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    request_target = urllib.parse.urlunparse(
        ("", "", parsed.path or "/", parsed.params, parsed.query, "")
    )
    request_headers = {key: value for key, value in headers.items() if key.lower() != "host"}
    request_headers["Host"] = hostname

    connection = http.client.HTTPSConnection(hostname, port=443, timeout=connect_timeout)
    original_create_connection = connection._create_connection

    def create_pinned_connection(address, connect_timeout, source_address=None):
        return original_create_connection(
            (pinned_ip, address[1]),
            connect_timeout,
            source_address,
        )

    connection._create_connection = create_pinned_connection
    method = "POST" if data is not None else "GET"
    try:
        connection.request(method, request_target, body=data, headers=request_headers)
        if connection.sock is not None:
            connection.sock.settimeout(read_timeout)
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BODY_BYTES + 1)
        return response.status, dict(response.getheaders()), body
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class SafeHttpResponse:
    """Represents a validated, completed HTTP response."""

    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class TransportResult:
    """Outcome of safe HTTP request."""

    is_success: bool
    response: SafeHttpResponse | None = None
    failure_kind: FailureKind | None = None
    status_code: int | None = None
    error_message: str = ""
    duration_ms: float = 0.0


class SafeHttpTransport:
    """Safe HTTP transport boundary for AI Router."""

    def __init__(
        self,
        injected_transport: InjectedTransportCallable | None = None,
        resolver: ResolverCallable | None = None,
        read_timeout_seconds: float = READ_TIMEOUT_SECONDS,
    ) -> None:
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        self.injected_transport = injected_transport
        self.read_timeout_seconds = float(read_timeout_seconds)
        if resolver is not None:
            self.resolver = resolver
        elif injected_transport is not None:
            self.resolver = lambda _hostname: ("8.8.8.8",)
        else:
            self.resolver = _resolve_public_host

    def execute(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> TransportResult:
        """Executes safe HTTPS request adhering to allowlist, limits, and redirect restrictions."""
        start_time = time.time()
        current_url = url
        req_headers = dict(headers or {})
        redirect_count = 0

        while True:
            if cancellation_token and cancellation_token.is_cancelled:
                duration_ms = (time.time() - start_time) * 1000
                return TransportResult(
                    is_success=False,
                    failure_kind=FailureKind.TIMEOUT,
                    error_message="Operation cancelled by CancellationToken.",
                    duration_ms=duration_ms,
                )

            try:
                parsed_url = validate_https_url(current_url)
            except SecurityValidationError as exc:
                duration_ms = (time.time() - start_time) * 1000
                return TransportResult(
                    is_success=False,
                    failure_kind=FailureKind.UNAUTHORIZED,
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )

            hostname = parsed_url.hostname or ""
            try:
                resolved_addresses = validate_public_dns_addresses(
                    hostname,
                    self.resolver(hostname),
                )
            except SecurityValidationError as exc:
                duration_ms = (time.time() - start_time) * 1000
                return TransportResult(
                    is_success=False,
                    failure_kind=FailureKind.UNAUTHORIZED,
                    error_message=str(exc),
                    duration_ms=duration_ms,
                )
            except OSError as exc:
                duration_ms = (time.time() - start_time) * 1000
                return TransportResult(
                    is_success=False,
                    failure_kind=FailureKind.NETWORK,
                    error_message=f"DNS resolution failed: {exc}",
                    duration_ms=duration_ms,
                )

            if self.injected_transport is not None:
                try:
                    status, resp_headers, body = self.injected_transport(
                        current_url, req_headers, data, self.read_timeout_seconds
                    )
                except TimeoutError as exc:
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.TIMEOUT,
                        error_message=f"Timeout: {exc}",
                        duration_ms=duration_ms,
                    )
                except Exception as exc:
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.NETWORK,
                        error_message=f"Network error: {exc}",
                        duration_ms=duration_ms,
                    )
            else:
                try:
                    status, resp_headers, body = _perform_pinned_https_request(
                        current_url,
                        req_headers,
                        data,
                        CONNECT_TIMEOUT_SECONDS,
                        self.read_timeout_seconds,
                        resolved_addresses[0],
                    )
                except TimeoutError as error:
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.TIMEOUT,
                        error_message=f"Timeout: {error}",
                        duration_ms=duration_ms,
                    )
                except (OSError, http.client.HTTPException) as error:
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.NETWORK,
                        error_message=str(error),
                        duration_ms=duration_ms,
                    )

            # Handle redirects (301, 302, 307, 308)
            if status in (301, 302, 307, 308) and "Location" in resp_headers:
                redirect_count += 1
                if redirect_count > MAX_REDIRECTS:
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.UNAUTHORIZED,
                        error_message=f"Exceeded max redirects ({MAX_REDIRECTS}).",
                        duration_ms=duration_ms,
                    )
                next_url = urllib.parse.urljoin(current_url, resp_headers["Location"])
                next_parsed = urllib.parse.urlparse(next_url)
                # Same origin check
                if next_parsed.netloc.lower() != parsed_url.netloc.lower():
                    duration_ms = (time.time() - start_time) * 1000
                    return TransportResult(
                        is_success=False,
                        failure_kind=FailureKind.UNAUTHORIZED,
                        error_message="Cross-origin redirect strictly rejected.",
                        duration_ms=duration_ms,
                    )
                current_url = next_url
                continue

            # Check body size
            if len(body) > MAX_RESPONSE_BODY_BYTES:
                duration_ms = (time.time() - start_time) * 1000
                return TransportResult(
                    is_success=False,
                    status_code=status,
                    failure_kind=FailureKind.INVALID_STRUCTURE,
                    error_message=f"Response body exceeded 2 MiB limit ({len(body)} bytes).",
                    duration_ms=duration_ms,
                )

            duration_ms = (time.time() - start_time) * 1000
            safe_resp = SafeHttpResponse(
                status_code=status,
                headers=redact_sensitive_content(resp_headers),
                body=body,
            )
            return TransportResult(
                is_success=(200 <= status < 300),
                response=safe_resp,
                status_code=status,
                duration_ms=duration_ms,
            )
