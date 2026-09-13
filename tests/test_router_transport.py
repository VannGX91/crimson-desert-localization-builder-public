import unittest
from unittest.mock import patch

from crimson0020.translation_runtime.contracts import FailureKind
from crimson0020.translation_runtime.job_control import CancellationToken
from crimson0020.translation_runtime.router.security import (
    SecurityValidationError,
    redact_sensitive_content,
    validate_https_url,
)
from crimson0020.translation_runtime.router.transport import (
    SafeHttpTransport,
    _perform_pinned_https_request,
)


class TestRouterTransport(unittest.TestCase):
    def test_pinned_https_helper_connects_to_ip_but_preserves_https_host(self):
        create_calls = []
        response_reads = []
        socket_timeouts = []

        class FakeSocket:
            def settimeout(self, timeout):
                socket_timeouts.append(timeout)

        class FakeResponse:
            status = 201

            def getheaders(self):
                return [("X-Test", "ok")]

            def read(self, limit):
                response_reads.append(limit)
                return b"response"

        class FakeConnection:
            def __init__(self):
                self.requests = []
                self.closed = False
                self.sock = FakeSocket()
                self._create_connection = self._original_create_connection

            def _original_create_connection(self, address, timeout, source_address=None):
                create_calls.append((address, timeout, source_address))
                return object()

            def request(self, method, target, body=None, headers=None):
                self.requests.append((method, target, body, dict(headers or {})))
                self._create_connection(("opencode.ai", 443), 12.0, None)

            def getresponse(self):
                return FakeResponse()

            def close(self):
                self.closed = True

        fake_connection = FakeConnection()
        with patch(
            "crimson0020.translation_runtime.router.transport.http.client.HTTPSConnection",
            return_value=fake_connection,
        ) as connection_class:
            status, response_headers, body = _perform_pinned_https_request(
                "https://opencode.ai/v1/chat?mode=test",
                {"Host": "attacker.invalid", "X-Client": "test-client"},
                b"{}",
                12.0,
                34.0,
                "8.8.8.8",
            )

        connection_class.assert_called_once_with("opencode.ai", port=443, timeout=12.0)
        self.assertEqual(create_calls, [(('8.8.8.8', 443), 12.0, None)])
        self.assertEqual(
            fake_connection.requests,
            [("POST", "/v1/chat?mode=test", b"{}", {"X-Client": "test-client", "Host": "opencode.ai"})],
        )
        self.assertEqual(status, 201)
        self.assertEqual(response_headers, {"X-Test": "ok"})
        self.assertEqual(body, b"response")
        self.assertEqual(response_reads, [2 * 1024 * 1024 + 1])
        self.assertEqual(socket_timeouts, [34.0])
        self.assertTrue(fake_connection.closed)

    def test_real_transport_pins_connection_to_validated_dns_address(self):
        with patch(
            "crimson0020.translation_runtime.router.transport._perform_pinned_https_request",
            return_value=(200, {"Content-Type": "application/json"}, b"{}"),
        ) as perform_request:
            transport = SafeHttpTransport(
                resolver=lambda _host: ("8.8.8.8", "1.1.1.1"),
            )
            result = transport.execute("https://opencode.ai/v1/chat")

        self.assertTrue(result.is_success)
        self.assertEqual(perform_request.call_args.args[3], 10.0)
        self.assertEqual(perform_request.call_args.args[4], 30.0)
        self.assertEqual(perform_request.call_args.args[5], "8.8.8.8")

    def test_custom_read_timeout_is_forwarded_to_real_transport(self):
        with patch(
            "crimson0020.translation_runtime.router.transport._perform_pinned_https_request",
            return_value=(200, {"Content-Type": "application/json"}, b"{}"),
        ) as perform_request:
            transport = SafeHttpTransport(
                resolver=lambda _host: ("8.8.8.8",),
                read_timeout_seconds=12.0,
            )
            result = transport.execute("https://blockrun.ai/api/v1/chat/completions")

        self.assertTrue(result.is_success)
        self.assertEqual(perform_request.call_args.args[4], 12.0)

    def test_private_dns_resolution_is_rejected_before_network_call(self):
        calls = []

        def fake_network(req_url, headers, data, timeout):
            calls.append(req_url)
            return 200, {}, b"OK"

        transport = SafeHttpTransport(
            injected_transport=fake_network,
            resolver=lambda _host: ("127.0.0.1",),
        )
        result = transport.execute("https://opencode.ai/v1/chat")

        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.UNAUTHORIZED)
        self.assertEqual([], calls)

    def test_redirect_target_is_dns_validated_again(self):
        calls = []
        resolutions = iter((("8.8.8.8",), ("10.0.0.8",)))

        def fake_network(req_url, headers, data, timeout):
            calls.append(req_url)
            return 302, {"Location": "https://opencode.ai/v1/new"}, b""

        transport = SafeHttpTransport(
            injected_transport=fake_network,
            resolver=lambda _host: next(resolutions),
        )
        result = transport.execute("https://opencode.ai/v1/old")

        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.UNAUTHORIZED)
        self.assertEqual(["https://opencode.ai/v1/old"], calls)

    def test_http_url_and_userinfo_rejected(self):
        # Plain HTTP
        with self.assertRaises(SecurityValidationError):
            validate_https_url("http://opencode.ai/v1/chat")

        # Userinfo in URL
        with self.assertRaises(SecurityValidationError):
            validate_https_url("https://user:pass@opencode.ai/v1/chat")

        # Non-standard port
        with self.assertRaises(SecurityValidationError):
            validate_https_url("https://opencode.ai:8080/v1/chat")

        # Valid HTTPS URL from allowlist
        parsed = validate_https_url("https://opencode.ai/v1/chat")
        self.assertEqual(parsed.netloc, "opencode.ai")

    def test_unauthorized_origin_rejected(self):
        # Arbitrary origin outside fixed allowlist
        with self.assertRaises(SecurityValidationError):
            validate_https_url("https://evil-attacker.com/v1/chat")

    def test_live_verified_keyless_origins_are_allowlisted(self):
        for url in (
            "https://hermes.ai.unturf.com/v1/chat/completions",
            "https://stablehorde.net/api/v2/status/models?type=text",
            "https://api.llm7.io/v1/chat/completions",
            "https://blockrun.ai/api/v1/chat/completions",
            "https://vireonix.ai/v1/chat/completions",
            "https://ai-api.cehpoint.co.in/v1/chat/completions",
        ):
            assert validate_https_url(url).scheme == "https"

    def test_redirect_host_and_scheme_escape_rejected(self):
        def fake_network(req_url, headers, data, timeout):
            # Attempt redirect to external host
            if req_url == "https://opencode.ai/v1/chat":
                return 302, {"Location": "https://evil.com/steal"}, b""
            return 200, {}, b"OK"

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/chat")
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.UNAUTHORIZED)

    def test_oversized_body_rejected(self):
        def fake_network(req_url, headers, data, timeout):
            # Return body > 2 MiB
            return 200, {}, b"X" * (2 * 1024 * 1024 + 1)

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/chat")
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.INVALID_STRUCTURE)
        self.assertIn("limit", result.error_message.lower())

    def test_timeout_and_cancel(self):
        token = CancellationToken()
        token.cancel()

        transport = SafeHttpTransport()
        result = transport.execute("https://opencode.ai/v1/chat", cancellation_token=token)
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.TIMEOUT)

    def test_successful_request_and_response(self):
        def fake_network(req_url, headers, data, timeout):
            return 200, {"Content-Type": "application/json"}, b'{"models": []}'

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/models")
        self.assertTrue(result.is_success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.response.body, b'{"models": []}')

    def test_same_origin_redirect_success(self):
        def fake_network(req_url, headers, data, timeout):
            if req_url == "https://opencode.ai/v1/old":
                return 301, {"Location": "https://opencode.ai/v1/new"}, b""
            return 200, {}, b"Redirected OK"

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/old")
        self.assertTrue(result.is_success)
        self.assertEqual(result.response.body, b"Redirected OK")

    def test_max_redirects_exceeded(self):
        def fake_network(req_url, headers, data, timeout):
            return 302, {"Location": req_url + "/loop"}, b""

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/loop")
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.UNAUTHORIZED)
        self.assertIn("Exceeded max redirects", result.error_message)

    def test_generic_network_exception(self):
        def fake_network(req_url, headers, data, timeout):
            raise ConnectionResetError("Connection reset by peer")

        transport = SafeHttpTransport(injected_transport=fake_network)
        result = transport.execute("https://opencode.ai/v1/chat")
        self.assertFalse(result.is_success)
        self.assertEqual(result.failure_kind, FailureKind.NETWORK)

    def test_redaction_removes_auth_and_session(self):
        headers = {
            "Authorization": "Bearer secret-token-12345",
            "Cookie": "session=abcdef123456",
            "User-Agent": "Crimson0020/3.0",
        }
        redacted_hdrs = redact_sensitive_content(headers)
        self.assertEqual(redacted_hdrs["Authorization"], "[REDACTED]")
        self.assertEqual(redacted_hdrs["Cookie"], "[REDACTED]")
    def test_urllib_network_error(self):
        with patch(
            "crimson0020.translation_runtime.router.transport._perform_pinned_https_request",
            side_effect=OSError("Network unreachable"),
        ):
            transport = SafeHttpTransport()
            result = transport.execute("https://opencode.ai/v1/chat")
            self.assertFalse(result.is_success)
            self.assertEqual(result.failure_kind, FailureKind.NETWORK)


if __name__ == "__main__":
    unittest.main()
