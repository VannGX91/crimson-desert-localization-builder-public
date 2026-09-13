import concurrent.futures
import hashlib
import http.cookiejar
import json
import re
import socket
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen


class TranslationProviderError(RuntimeError):
    """Raised when a translation provider returns an unusable response."""


class RetryableTranslationProviderError(TranslationProviderError):
    """Raised for a temporary provider failure that may succeed after backoff."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = (
            max(0.0, float(retry_after_seconds))
            if retry_after_seconds is not None
            else None
        )


class PartialTranslationProviderError(TranslationProviderError):
    """Raised when a provider returns a safe subset of a requested batch."""

    def __init__(
        self,
        message: str,
        *,
        successes: Mapping[str, str],
        failures: Mapping[str, str],
    ) -> None:
        super().__init__(message)
        self.successes = MappingProxyType(dict(successes))
        self.failures = MappingProxyType(dict(failures))


class TranslationResponseFormatError(TranslationProviderError):
    """Raised when a provider response cannot be mapped safely to input IDs."""


class TranslationProvider(Protocol):
    key: str

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class TranslationProviderSpec:
    key: str
    name: str
    description: str
    endpoint: str
    external: bool
    free: bool
    requires_api_key: bool = False
    requires_model: bool = False
    configurable_endpoint: bool = False

    def to_state(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "endpoint": self.endpoint,
            "external": self.external,
            "free": self.free,
            "requires_api_key": self.requires_api_key,
            "requires_model": self.requires_model,
            "configurable_endpoint": self.configurable_endpoint,
        }


DEFAULT_TRANSLATION_PROVIDER = "google-web-free"
AI_ROUTER_TRANSLATION_PROVIDER = "ai_router_free"
_PROVIDER_SPECS = (
    TranslationProviderSpec(
        key="ai_router_free",
        name="AI Router (Free)",
        description=(
            "Tự động tìm và định tuyến qua các model AI miễn phí, ẩn danh, "
            "không cần API key."
        ),
        endpoint="https://router.local",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="google-web-free",
        name="Google Translate (Free)",
        description=(
            "Dịch tự động miễn phí qua Google Translate; hỗ trợ pygtrans và web endpoint gtx."
        ),
        endpoint="https://translate.google.com/translate_a/single",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="bing-web-free",
        name="Bing Translator (Free)",
        description=(
            "Dịch tự động miễn phí qua Microsoft Bing Translator; hỗ trợ LLM và browser session."
        ),
        endpoint="https://www.bing.com/ttranslatev3",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="alibaba-web-free",
        name="Alibaba Translate (Free)",
        description=(
            "Dịch tự động miễn phí qua Alibaba DAMO Academy; hạn ngạch lớn, tốc độ cao."
        ),
        endpoint="https://translate.alibaba.com",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="modernmt-web-free",
        name="ModernMT (Free)",
        description=(
            "Dịch máy thần kinh qua ModernMT (Translated); dịch văn cảnh tự nhiên, phù hợp game."
        ),
        endpoint="https://www.modernmt.com/translate",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="youdao-web-free",
        name="YouDao (Free)",
        description=(
            "Dịch tự động miễn phí qua NetEase YouDao; hỗ trợ dịch hội thoại nhanh."
        ),
        endpoint="https://fanyi.youdao.com",
        external=True,
        free=True,
    ),
    TranslationProviderSpec(
        key="openai-compatible",
        name="Custom (OpenAI-compatible)",
        description=(
            "Tùy chỉnh Base URL, model, batch, ký tự, concurrency và timeout; "
            "API key chỉ giữ trong phiên, localhost có thể không cần key."
        ),
        endpoint="https://api.openai.com/v1",
        external=True,
        free=False,
        requires_api_key=True,
        requires_model=True,
        configurable_endpoint=True,
    ),
)
_PROVIDER_MAP: dict[str, TranslationProviderSpec] = {
    spec.key: spec for spec in _PROVIDER_SPECS
}
# Backward-compatible and convenient aliases
_PROVIDER_MAP["google_free"] = TranslationProviderSpec(
    key="google_free",
    name="Google Translate (Free)",
    description=_PROVIDER_MAP["google-web-free"].description,
    endpoint=_PROVIDER_MAP["google-web-free"].endpoint,
    external=True,
    free=True,
)
_PROVIDER_MAP["bing_free"] = TranslationProviderSpec(
    key="bing_free",
    name="Bing Translator (Free)",
    description=_PROVIDER_MAP["bing-web-free"].description,
    endpoint=_PROVIDER_MAP["bing-web-free"].endpoint,
    external=True,
    free=True,
)
_PROVIDER_MAP["alibaba_free"] = TranslationProviderSpec(
    key="alibaba_free",
    name="Alibaba Translate (Free)",
    description=_PROVIDER_MAP["alibaba-web-free"].description,
    endpoint=_PROVIDER_MAP["alibaba-web-free"].endpoint,
    external=True,
    free=True,
)
_PROVIDER_MAP["modernmt_free"] = TranslationProviderSpec(
    key="modernmt_free",
    name="ModernMT (Free)",
    description=_PROVIDER_MAP["modernmt-web-free"].description,
    endpoint=_PROVIDER_MAP["modernmt-web-free"].endpoint,
    external=True,
    free=True,
)
_PROVIDER_MAP["youdao_free"] = TranslationProviderSpec(
    key="youdao_free",
    name="YouDao (Free)",
    description=_PROVIDER_MAP["youdao-web-free"].description,
    endpoint=_PROVIDER_MAP["youdao-web-free"].endpoint,
    external=True,
    free=True,
)
# Deprecated aliases for legacy test compatibility
_PROVIDER_MAP["mymemory-free"] = TranslationProviderSpec(
    key="mymemory-free",
    name="MyMemory (Free) [Deprecated]",
    description="Endpoint chính thức giới hạn 5000 từ/ngày.",
    endpoint="https://api.mymemory.translated.net/get",
    external=True,
    free=True,
)
_PROVIDER_MAP["mymemory_free"] = TranslationProviderSpec(
    key="mymemory_free",
    name="MyMemory (Free) [Deprecated]",
    description="Endpoint chính thức giới hạn 5000 từ/ngày.",
    endpoint="https://api.mymemory.translated.net/get",
    external=True,
    free=True,
)
_PROVIDER_MAP["libretranslate-cutie-free"] = TranslationProviderSpec(
    key="libretranslate-cutie-free",
    name="LibreTranslate Cutie (Free) [Deprecated]",
    description="Public mirror cộng đồng (đã ngừng hoạt động).",
    endpoint="https://translate.cutie.dating",
    external=True,
    free=True,
)
_PROVIDER_MAP["libretranslate-fedilab-free"] = TranslationProviderSpec(
    key="libretranslate-fedilab-free",
    name="LibreTranslate Fedilab (Free) [Deprecated]",
    description="Public mirror cộng đồng (chặn request bên ngoài).",
    endpoint="https://translate.fedilab.app",
    external=True,
    free=True,
)
TRANSLATION_PROVIDER_SPECS = MappingProxyType(_PROVIDER_MAP)

HttpTransport = Callable[
    [str, bytes, Mapping[str, str], float],
    tuple[int, bytes],
]
HttpGetTransport = Callable[
    [str, Mapping[str, str], float],
    tuple[int, bytes],
]


def get_translation_provider_spec(key: str) -> TranslationProviderSpec:
    if not isinstance(key, str):
        raise ValueError("Translation provider key must be text.")
    try:
        return TRANSLATION_PROVIDER_SPECS[key]
    except KeyError as error:
        raise ValueError(f"Unknown translation provider: {key}") from error


def _normalized_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Translation endpoint must be a non-empty URL.")
    parsed = urlsplit(endpoint.strip())
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Translation endpoint must not contain URL credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Translation endpoint must not contain a query or fragment.")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Translation endpoint must be an HTTP(S) URL.")
    is_loopback = parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not is_loopback:
        raise ValueError("Remote translation endpoints must use HTTPS.")
    if parsed.path.rstrip("/") not in {"", "/translate"}:
        raise ValueError("LibreTranslate endpoint path must be empty or /translate.")
    path = "/translate"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _default_transport(
    url: str,
    payload: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, bytes]:
    request = Request(url, data=payload, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(2 * 1024 * 1024)
    except HTTPError as error:
        return int(error.code), error.read(256 * 1024)
    except (TimeoutError, socket.timeout, URLError) as error:
        raise RetryableTranslationProviderError(
            f"Cannot reach translation provider: {error}"
        ) from error


def _default_get_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, bytes]:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(2 * 1024 * 1024)
    except HTTPError as error:
        return int(error.code), error.read(256 * 1024)
    except (TimeoutError, socket.timeout, URLError) as error:
        raise RetryableTranslationProviderError(
            f"Cannot reach translation provider: {error}"
        ) from error


def _status_error(status: int, body: bytes) -> TranslationProviderError:
    detail = ""
    try:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict):
            value = payload.get("error")
            if isinstance(value, str):
                detail = f": {value}"
            elif isinstance(value, dict) and isinstance(value.get("message"), str):
                detail = f": {value['message']}"
    except (UnicodeError, json.JSONDecodeError):
        pass
    message = f"Translation provider returned HTTP {status}{detail}"
    if status == 429 or status >= 500:
        return RetryableTranslationProviderError(message)
    return TranslationProviderError(message)


class LibreTranslateProvider:
    def __init__(
        self,
        key: str,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: float = 45.0,
        transport: HttpTransport | None = None,
    ) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("Translation provider key must be non-empty text.")
        if timeout <= 0 or timeout > 300:
            raise ValueError("Translation timeout must be between 0 and 300 seconds.")
        self.key = key
        self.endpoint = _normalized_endpoint(endpoint)
        self.api_key = api_key.strip() if isinstance(api_key, str) else None
        self.timeout = float(timeout)
        self._transport = transport or _default_transport

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        batch = tuple(texts)
        if not batch or any(not isinstance(text, str) for text in batch):
            raise ValueError("Translation batch must contain one or more text values.")
        if not source_language or not target_language:
            raise ValueError("Source and target language codes are required.")
        request_data: dict[str, object] = {
            "q": list(batch),
            "source": source_language,
            "target": target_language,
            "format": "text",
        }
        if self.api_key:
            request_data["api_key"] = self.api_key
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
        status, response_body = self._transport(
            self.endpoint,
            payload,
            {"Content-Type": "application/json", "Accept": "application/json"},
            self.timeout,
        )
        if status < 200 or status >= 300:
            raise _status_error(status, response_body)
        try:
            response = json.loads(response_body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TranslationResponseFormatError(
                "Translation provider returned invalid UTF-8 JSON."
            ) from error
        if not isinstance(response, dict):
            raise TranslationResponseFormatError(
                "Translation response must be an object."
            )
        translated = response.get("translatedText")
        if isinstance(translated, str) and len(batch) == 1:
            translated = [translated]
        if not isinstance(translated, list) or any(
            not isinstance(value, str) for value in translated
        ):
            raise TranslationResponseFormatError(
                "Translation response must contain translatedText strings."
            )
        if len(translated) != len(batch):
            raise TranslationResponseFormatError(
                "Translation provider result count does not match the request."
            )
        return tuple(translated)

    @staticmethod
    def _error_message(status: int, body: bytes) -> str:
        detail = ""
        try:
            payload = json.loads(body.decode("utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                detail = f": {payload['error']}"
        except (UnicodeError, json.JSONDecodeError):
            pass
        return f"Translation provider returned HTTP {status}{detail}"


SingleTextTranslator = Callable[[str, str, str], str]


class FreeWebTranslationProvider:
    """Run per-string free web translators concurrently while preserving order."""

    def __init__(
        self,
        key: str,
        translator: SingleTextTranslator,
        *,
        max_workers: int = 8,
    ) -> None:
        if not key:
            raise ValueError("Translation provider key must be non-empty text.")
        if max_workers <= 0 or max_workers > 16:
            raise ValueError("Free provider concurrency must be between 1 and 16.")
        self.key = key
        self._translator = translator
        self._max_workers = max_workers

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        batch = tuple(texts)
        if not batch or any(not isinstance(text, str) for text in batch):
            raise ValueError("Translation batch must contain one or more text values.")
        results = [""] * len(batch)
        workers = min(self._max_workers, len(batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._translator,
                    text,
                    source_language,
                    target_language,
                ): index
                for index, text in enumerate(batch)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    translated = future.result()
                except TranslationProviderError:
                    raise
                except Exception as error:
                    raise RetryableTranslationProviderError(
                        f"Free translation request failed: {error}"
                    ) from error
                if not isinstance(translated, str) or not translated.strip():
                    raise TranslationResponseFormatError(
                        "Free translation provider returned empty or invalid text."
                    )
                results[index] = translated
        return tuple(results)


class MyMemoryFreeProvider:
    """No-key MyMemory GET API with bounded per-row concurrency."""

    key = "mymemory-free"
    endpoint = "https://api.mymemory.translated.net/get"
    _MAX_QUERY_BYTES = 500

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_workers: int = 8,
        transport: HttpGetTransport | None = None,
    ) -> None:
        if timeout <= 0 or timeout > 300:
            raise ValueError("Translation timeout must be between 0 and 300 seconds.")
        self.timeout = float(timeout)
        self._transport = transport or _default_get_transport
        self._runner = FreeWebTranslationProvider(
            self.key,
            self._translate_one,
            max_workers=max_workers,
        )

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        return self._runner.translate(tuple(texts), source_language, target_language)

    def _translate_one(self, text: str, source: str, target: str) -> str:
        chunks = _split_utf8_chunks(text, self._MAX_QUERY_BYTES)
        translated = tuple(self._request_one(chunk, source, target) for chunk in chunks)
        return "".join(translated)

    def _request_one(self, text: str, source: str, target: str) -> str:
        query = urlencode({"q": text, "langpair": f"{source}|{target}"})
        status, body = self._transport(
            f"{self.endpoint}?{query}",
            {"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            self.timeout,
        )
        if status < 200 or status >= 300:
            raise _status_error(status, body)
        try:
            response = json.loads(body.decode("utf-8"))
            translated = response["responseData"]["translatedText"]
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise TranslationResponseFormatError(
                "MyMemory returned an invalid translation response."
            ) from error
        if not isinstance(translated, str) or not translated.strip():
            raise TranslationResponseFormatError("MyMemory returned no translation.")
        return translated


_PROTECTED_CHUNK_MARKER = re.compile(r"__CRIMSON_KEEP_\d{4}_[0-9A-F]{8}__")


def _split_utf8_chunks(text: str, max_bytes: int) -> tuple[str, ...]:
    """Split a long request without breaking protected markers or UTF-8."""

    if len(text.encode("utf-8")) <= max_bytes:
        return (text,)
    tokens = tuple(
        token
        for token in re.split(f"({_PROTECTED_CHUNK_MARKER.pattern}|\\s+)", text)
        if token
    )
    chunks: list[str] = []
    current = ""
    for token in tokens:
        token_parts = _split_oversized_token(token, max_bytes)
        for part in token_parts:
            candidate = current + part
            if current and len(candidate.encode("utf-8")) > max_bytes:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return tuple(chunks)


def _split_oversized_token(token: str, max_bytes: int) -> tuple[str, ...]:
    if len(token.encode("utf-8")) <= max_bytes:
        return (token,)
    parts: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and len(candidate.encode("utf-8")) > max_bytes:
            parts.append(current)
            current = character
        else:
            current = candidate
    if current:
        parts.append(current)
    return tuple(parts)


class GoogleFreeProvider:
    """Batch adapter for Google Translate (Free) supporting pygtrans and web endpoints."""

    key = "google-web-free"
    endpoint = "https://translate.google.com/translate_a/t"
    # pygtrans sends a list as repeated q fields in one request.  These limits
    # were live-tested against the same endpoint and keep the outer batcher
    # from needlessly fragmenting large catalogs.
    preferred_batch_items = 2_000
    preferred_batch_characters = 64_000
    preferred_retry_base_delay = 5.0

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        transport: HttpTransport | None = None,
        prefer_pygtrans: bool = True,
    ) -> None:
        if timeout <= 0 or timeout > 300:
            raise ValueError("Translation timeout must be between 0 and 300 seconds.")
        self.timeout = float(timeout)
        self._transport = transport
        self._prefer_pygtrans = prefer_pygtrans

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        batch = tuple(texts)
        if not batch or any(not isinstance(text, str) for text in batch):
            raise ValueError("Translation batch must contain one or more text values.")
        if not source_language or not target_language:
            raise ValueError("Source and target language codes are required.")

        if self._transport is None and self._prefer_pygtrans:
            try:
                import pygtrans
                client = pygtrans.Translate(fmt="text", timeout=self.timeout)
            except Exception:
                # Import or constructor incompatibility happens before any
                # translation request is dispatched, so the native transport
                # can safely take over without duplicating provider traffic.
                client = None

            if client is not None:
                # pygtrans 1.6.2 retries HTTP 429 internally in two nested loops,
                # which can keep a translation worker blocked for well over a
                # minute and prevents cooperative pause from reaching a safe
                # request boundary.  Use pygtrans' configured session and
                # endpoint directly for one bounded request; retry/backoff is
                # owned by the outer Auto Translate orchestrator.
                try:
                    response = client.session.post(
                        client.TRANSLATE_URL,
                        params={
                            "tl": target_language,
                            "sl": source_language or "auto",
                            "ie": "UTF-8",
                            "oe": "UTF-8",
                            "client": "at",
                            "dj": "1",
                            "format": "text",
                            "v": "1.0",
                        },
                        data={"q": list(batch)},
                        timeout=self.timeout,
                    )
                except (
                    RetryableTranslationProviderError,
                    TranslationProviderError,
                    TranslationResponseFormatError,
                ):
                    raise
                except Exception as error:
                    raise RetryableTranslationProviderError(
                        f"Cannot reach Google Translate provider: {error or type(error).__name__}"
                    ) from error

                try:
                    status = int(response.status_code)
                    body = bytes(getattr(response, "content", b""))
                except Exception as error:
                    raise TranslationResponseFormatError(
                        "Google pygtrans endpoint returned an invalid HTTP response."
                    ) from error
                if status < 200 or status >= 300:
                    raise _status_error(status, body)
                try:
                    payload = response.json()
                except Exception as error:
                    raise TranslationResponseFormatError(
                        "Google pygtrans endpoint returned invalid JSON."
                    ) from error
                if not isinstance(payload, list) or any(
                    not isinstance(value, str) or not value.strip() for value in payload
                ):
                    raise TranslationResponseFormatError(
                        "Google pygtrans endpoint returned invalid translations."
                    )
                if len(payload) != len(batch):
                    raise TranslationResponseFormatError(
                        "Google pygtrans endpoint result count does not match the request."
                    )
                return tuple(payload)

        transport = self._transport or _default_transport
        query = urlencode(
            {
                "client": "at",
                "sl": source_language,
                "tl": target_language,
                "ie": "UTF-8",
                "oe": "UTF-8",
                "dj": "1",
                "format": "text",
                "v": "1.0",
            }
        )
        payload = urlencode(
            [("q", text) for text in batch],
            doseq=True,
        ).encode("utf-8")
        status, body = transport(
            f"{self.endpoint}?{query}",
            payload,
            {
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            self.timeout,
        )
        if status < 200 or status >= 300:
            raise _status_error(status, body)
        try:
            response = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TranslationResponseFormatError(
                "Google free endpoint returned invalid UTF-8 JSON."
            ) from error
        if not isinstance(response, list) or any(
            not isinstance(value, str) or not value.strip() for value in response
        ):
            raise TranslationResponseFormatError(
                "Google free endpoint returned invalid translations."
            )
        if len(response) != len(batch):
            raise TranslationResponseFormatError(
                "Google free endpoint result count does not match the request."
            )
        return tuple(response)


class _BingWebTranslator:
    _HOST = "https://www.bing.com/Translator"
    _API = "https://www.bing.com/ttranslatev3"

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._lock = threading.Lock()
        self._opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self._credentials: tuple[str, str, str, str] | None = None

    @staticmethod
    def _headers(content_type: bool = False) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": _BingWebTranslator._HOST,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        if content_type:
            headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
            headers["Origin"] = "https://www.bing.com"
        return headers

    def _open(self, request: Request, limit: int) -> tuple[int, bytes]:
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return int(response.status), response.read(limit)
        except HTTPError as error:
            return int(error.code), error.read(min(limit, 256 * 1024))
        except (TimeoutError, socket.timeout, URLError) as error:
            raise RetryableTranslationProviderError(
                f"Cannot reach Bing Translator: {error}"
            ) from error

    def _initialize(self) -> tuple[str, str, str, str]:
        status, body = self._open(
            Request(self._HOST, headers=self._headers()), 2 * 1024 * 1024
        )
        if status < 200 or status >= 300:
            raise _status_error(status, body)
        try:
            html = body.decode("utf-8")
            raw_token = re.search(
                r"var\s+params_AbusePreventionHelper\s*=\s*(\[.*?\]);",
                html,
            ).group(1)
            token_data = json.loads(raw_token)
            key, token = str(token_data[0]), str(token_data[1])
            ig = re.search(r'IG:"(.*?)"', html).group(1)
            iid_match = re.search(
                r'id=["\']tta_outGDCont["\'][^>]*data-iid=["\']([^"\']+)',
                html,
                re.DOTALL,
            ) or re.search(
                r'data-iid=["\']([^"\']+)["\'][^>]*id=["\']tta_outGDCont["\']',
                html,
                re.DOTALL,
            )
            iid = iid_match.group(1)
        except (
            AttributeError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise TranslationProviderError(
                "Bing Translator page no longer exposes the expected web token."
            ) from error
        self._credentials = (key, token, ig, iid)
        return self._credentials

    def __call__(self, text: str, source: str, target: str) -> str:
        with self._lock:
            key, token, ig, iid = self._credentials or self._initialize()
        payload = urlencode(
            {
                "text": text,
                "fromLang": source or "auto-detect",
                "to": target,
                "tryFetchingGenderDebiasedTranslations": "true",
                "key": key,
                "token": token,
            }
        ).encode("utf-8")
        url = f"{self._API}?isVertical=1&&IG={ig}&IID={iid}"
        status, body = self._open(
            Request(url, data=payload, headers=self._headers(True), method="POST"),
            2 * 1024 * 1024,
        )
        if status < 200 or status >= 300:
            if status in {401, 403}:
                with self._lock:
                    self._credentials = None
            raise _status_error(status, body)
        try:
            data = json.loads(body.decode("utf-8"))
            translated = data[0]["translations"][0]["text"]
        except (
            UnicodeError,
            json.JSONDecodeError,
            IndexError,
            KeyError,
            TypeError,
        ) as error:
            raise TranslationResponseFormatError(
                "Bing free endpoint returned an invalid response."
            ) from error
        if not isinstance(translated, str) or not translated:
            raise TranslationResponseFormatError(
                "Bing free endpoint returned no translation."
            )
        return translated


class TranslatorsEngineProvider:
    """Web translator powered by ``translators`` with optional native fallback.

    Some web engines (notably Bing) accept substantially more than one short
    localization row per request, but expose a fairly small character limit.
    ``pack_characters`` groups rows behind stable sentinels so we can preserve
    row identity while reducing request count.  If an engine mutates a
    sentinel, that pack is retried row-by-row rather than returning a
    misaligned translation.
    """

    def __init__(
        self,
        key: str,
        engine_name: str,
        *,
        timeout: float = 30.0,
        max_workers: int = 6,
        translator_func: SingleTextTranslator | None = None,
        fallback_translator: SingleTextTranslator | None = None,
        pack_characters: int | None = None,
        preferred_batch_items: int | None = None,
        preferred_batch_characters: int | None = None,
    ) -> None:
        if timeout <= 0 or timeout > 300:
            raise ValueError("Translation timeout must be between 0 and 300 seconds.")
        if max_workers <= 0 or max_workers > 32:
            raise ValueError("Free provider concurrency must be between 1 and 32.")
        if pack_characters is not None and pack_characters < 128:
            raise ValueError("Packed translation limit must be at least 128 characters.")
        if preferred_batch_items is not None and preferred_batch_items <= 0:
            raise ValueError("Preferred batch item limit must be positive.")
        if preferred_batch_characters is not None and preferred_batch_characters <= 0:
            raise ValueError("Preferred batch character limit must be positive.")
        self.key = key
        self.engine_name = engine_name
        self.timeout = float(timeout)
        self._custom_translator = translator_func
        self._fallback_translator = fallback_translator
        self._max_workers = max_workers
        self._pack_characters = pack_characters
        if preferred_batch_items is not None:
            self.preferred_batch_items = preferred_batch_items
        if preferred_batch_characters is not None:
            self.preferred_batch_characters = preferred_batch_characters

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        batch = tuple(texts)
        if not batch or any(not isinstance(text, str) for text in batch):
            raise ValueError("Translation batch must contain one or more text values.")
        if not source_language or not target_language:
            raise ValueError("Source and target language codes are required.")

        packs = self._build_packs(batch)
        results = [""] * len(batch)
        workers = min(self._max_workers, len(packs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._translate_pack,
                    pack_index,
                    indices,
                    batch,
                    source_language,
                    target_language,
                ): indices
                for pack_index, indices in enumerate(packs)
            }
            for future in concurrent.futures.as_completed(futures):
                indices = futures[future]
                translated = future.result()
                for index, value in zip(indices, translated, strict=True):
                    results[index] = value
        return tuple(results)

    def _build_packs(self, batch: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
        if self._pack_characters is None:
            return tuple((index,) for index in range(len(batch)))

        packs: list[tuple[int, ...]] = []
        current: list[int] = []
        current_characters = 0
        for index, text in enumerate(batch):
            separator_cost = 32 if current else 0
            candidate = current_characters + separator_cost + len(text)
            if current and candidate > self._pack_characters:
                packs.append(tuple(current))
                current = []
                current_characters = 0
                separator_cost = 0
            current.append(index)
            current_characters += separator_cost + len(text)
        if current:
            packs.append(tuple(current))
        return tuple(packs)

    def _translate_pack(
        self,
        pack_index: int,
        indices: tuple[int, ...],
        batch: tuple[str, ...],
        source: str,
        target: str,
    ) -> tuple[str, ...]:
        if len(indices) == 1:
            return (self._translate_one(batch[indices[0]], source, target),)

        markers = tuple(
            f"__CRIMSON_SPLIT_{pack_index:04d}_{offset:04d}__"
            for offset in range(len(indices) - 1)
        )
        pieces: list[str] = []
        for offset, index in enumerate(indices):
            pieces.append(batch[index])
            if offset < len(markers):
                pieces.append(markers[offset])
        packed = "\n".join(pieces)
        translated = self._translate_one(packed, source, target)

        remaining = translated
        values: list[str] = []
        for marker in markers:
            if marker not in remaining:
                return tuple(
                    self._translate_one(batch[index], source, target) for index in indices
                )
            value, remaining = remaining.split(marker, 1)
            values.append(value.strip("\r\n"))
        values.append(remaining.strip("\r\n"))
        if len(values) != len(indices) or any(not value.strip() for value in values):
            return tuple(
                self._translate_one(batch[index], source, target) for index in indices
            )
        return tuple(values)

    def _translate_one(self, text: str, source: str, target: str) -> str:
        if self._custom_translator is not None:
            return self._custom_translator(text, source, target)
        try:
            import translators as ts
            ret = ts.translate_text(
                text,
                from_language=source or "auto",
                to_language=target,
                translator=self.engine_name,
                timeout=self.timeout,
            )
            if not isinstance(ret, str) or not ret.strip():
                raise TranslationResponseFormatError(
                    f"{self.engine_name.capitalize()} returned empty translation."
                )
            return ret
        except (TranslationResponseFormatError, TranslationProviderError):
            raise
        except Exception as error:
            if self._fallback_translator is not None:
                try:
                    ret = self._fallback_translator(text, source, target)
                    if isinstance(ret, str) and ret.strip():
                        return ret
                except Exception:
                    pass
            detail = str(error).strip() or type(error).__name__
            error_str = detail.lower()
            if "429" in error_str or "rate" in error_str or "limit" in error_str or "timeout" in error_str:
                raise RetryableTranslationProviderError(
                    f"{self.engine_name.capitalize()} temporary limit: {detail}"
                ) from error
            raise RetryableTranslationProviderError(
                f"{self.engine_name.capitalize()} translation request failed: {detail}"
            ) from error


class AlibabaWebProvider(TranslatorsEngineProvider):
    key = "alibaba-web-free"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_workers: int = 6,
        translator_func: SingleTextTranslator | None = None,
    ) -> None:
        super().__init__(
            key=self.key,
            engine_name="alibaba",
            timeout=timeout,
            max_workers=max_workers,
            translator_func=translator_func,
        )


class ModernMtWebProvider(TranslatorsEngineProvider):
    key = "modernmt-web-free"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_workers: int = 4,
        translator_func: SingleTextTranslator | None = None,
    ) -> None:
        super().__init__(
            key=self.key,
            engine_name="modernMt",
            timeout=timeout,
            max_workers=max_workers,
            translator_func=translator_func,
        )


class YouDaoWebProvider(TranslatorsEngineProvider):
    key = "youdao-web-free"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_workers: int = 6,
        translator_func: SingleTextTranslator | None = None,
    ) -> None:
        super().__init__(
            key=self.key,
            engine_name="youdao",
            timeout=timeout,
            max_workers=max_workers,
            translator_func=translator_func,
        )


def _normalized_openai_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("OpenAI-compatible Base URL is required.")
    parsed = urlsplit(base_url.strip())
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OpenAI-compatible endpoint must not contain credentials.")
    if parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("OpenAI-compatible endpoint must be a plain HTTP(S) URL.")
    loopback = parsed.hostname.casefold() in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme not in {"http", "https"} or (
        parsed.scheme != "https" and not loopback
    ):
        raise ValueError("Remote OpenAI-compatible endpoints must use HTTPS.")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        preferred_batch_items: int = 24,
        preferred_batch_characters: int = 4_000,
        preferred_outer_concurrency: int = 1,
        transport: HttpTransport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI-compatible model is required.")
        self.endpoint = _normalized_openai_endpoint(base_url)
        endpoint_host = urlsplit(self.endpoint).hostname or ""
        loopback = endpoint_host.casefold() in {"localhost", "127.0.0.1", "::1"}
        if not isinstance(api_key, str) or (not api_key.strip() and not loopback):
            raise ValueError(
                "OpenAI-compatible API key is required for remote endpoints."
            )
        if timeout <= 0 or timeout > 600:
            raise ValueError("OpenAI-compatible timeout must be between 0 and 600 seconds.")
        if (
            not isinstance(preferred_batch_items, int)
            or isinstance(preferred_batch_items, bool)
            or not 1 <= preferred_batch_items <= 2_048
        ):
            raise ValueError("OpenAI-compatible batch items must be between 1 and 2048.")
        if (
            not isinstance(preferred_batch_characters, int)
            or isinstance(preferred_batch_characters, bool)
            or not 128 <= preferred_batch_characters <= 1_000_000
        ):
            raise ValueError(
                "OpenAI-compatible batch characters must be between 128 and 1000000."
            )
        if (
            not isinstance(preferred_outer_concurrency, int)
            or isinstance(preferred_outer_concurrency, bool)
            or not 1 <= preferred_outer_concurrency <= 16
        ):
            raise ValueError("OpenAI-compatible concurrency must be between 1 and 16.")
        self.model = model.strip()
        self._api_key = api_key.strip()
        self.timeout = float(timeout)
        self.preferred_batch_items = preferred_batch_items
        self.preferred_batch_characters = preferred_batch_characters
        self.preferred_outer_concurrency = preferred_outer_concurrency
        self._transport = transport or _default_transport
        identity = hashlib.sha256(
            f"{self.endpoint}\n{self.model}".encode("utf-8")
        ).hexdigest()[:16]
        self.key = f"openai-compatible-{identity}"

    def translate(
        self,
        texts: Sequence[str],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        batch = tuple(texts)
        if not batch:
            raise ValueError("Translation batch must not be empty.")
        return self._translate_resilient(batch, source_language, target_language)

    def _translate_resilient(
        self,
        batch: tuple[str, ...],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        try:
            return self._translate_batch(batch, source_language, target_language)
        except TranslationResponseFormatError:
            if len(batch) < 4:
                raise
            middle = len(batch) // 2
            return self._translate_resilient(
                batch[:middle], source_language, target_language
            ) + self._translate_resilient(
                batch[middle:], source_language, target_language
            )

    def _translate_batch(
        self,
        batch: tuple[str, ...],
        source_language: str,
        target_language: str,
    ) -> tuple[str, ...]:
        indexed = {str(index): text for index, text in enumerate(batch)}
        request_data = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate localization text faithfully. Return only one JSON "
                        "object with exactly the same numeric keys. Preserve every "
                        "CRIMSON_KEEP token verbatim and in order."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source language: {source_language}\nTarget language: "
                        f"{target_language}\nJSON:\n"
                        + json.dumps(indexed, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            ],
        }
        payload = json.dumps(request_data, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        status, body = self._transport(
            self.endpoint,
            payload,
            headers,
            self.timeout,
        )
        if status < 200 or status >= 300:
            raise _status_error(status, body)
        try:
            response = json.loads(body.decode("utf-8"))
            content = response["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            translated = json.loads(content)
            expected = tuple(str(index) for index in range(len(batch)))
            if not isinstance(translated, dict) or set(translated) != set(expected):
                raise ValueError("response keys do not match")
            values = tuple(translated[key] for key in expected)
            if any(not isinstance(value, str) for value in values):
                raise ValueError("response values are not text")
            return values
        except (
            UnicodeError,
            json.JSONDecodeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise TranslationResponseFormatError(
                "OpenAI-compatible endpoint returned invalid indexed JSON."
            ) from error


_BING_TRANSLATOR = _BingWebTranslator()


def create_translation_provider(
    key: str,
    config: Mapping[str, object] | None = None,
) -> TranslationProvider:
    spec = get_translation_provider_spec(key)
    if key in {"google-web-free", "google_free"}:
        return GoogleFreeProvider()
    if key in {"bing-web-free", "bing_free"}:
        return TranslatorsEngineProvider(
            spec.key,
            "bing",
            max_workers=32,
            fallback_translator=_BING_TRANSLATOR,
            # translators' Bing adapter enforces a 1,000-character request
            # limit; leave headroom for row separators and encoding variance.
            pack_characters=900,
            # Let the outer orchestrator hand a large unit set to this provider;
            # the provider then packs it into ~900-character requests and runs
            # those requests concurrently.  The previous 24-item outer default
            # serialized hundreds of otherwise parallelizable Bing requests.
            preferred_batch_items=2_000,
            preferred_batch_characters=64_000,
        )
    if key in {"alibaba-web-free", "alibaba_free"}:
        return AlibabaWebProvider()
    if key in {"modernmt-web-free", "modernmt_free"}:
        return ModernMtWebProvider()
    if key in {"youdao-web-free", "youdao_free"}:
        return YouDaoWebProvider()
    if key in {"mymemory-free", "mymemory_free"}:
        return MyMemoryFreeProvider()
    if key in {"libretranslate-cutie-free", "libretranslate-fedilab-free"}:
        return LibreTranslateProvider(key=spec.key, endpoint=spec.endpoint)
    if key == "ai_router_free":
        from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
        from crimson0020.translation_runtime.router.discovery import ModelDiscoveryService
        from crimson0020.translation_runtime.router.health import RouteHealthTracker
        from crimson0020.translation_runtime.router.scheduler import (
            ConcurrencyManager,
            RouterScheduler,
        )

        # Router provider adapter conforming to TranslationProvider protocol
        class AIRouterProviderAdapter:
            key = "ai_router_free"
            # AI Router owns route/model fallback internally. Keep its outer
            # Auto Translate profile independent from standalone MT providers.
            preferred_batch_items = 48
            preferred_batch_characters = 12_000
            # Run several batches at once, but keep each anonymous route at one
            # in-flight request. The shared scheduler then spreads work across
            # independent healthy routes instead of sending concurrent requests
            # to the same endpoint.
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 4
            preferred_cooldown_deferred_retries = 4
            preferred_retryable_deferred_seconds = 5.0
            preferred_deferred_outer_concurrency = 2
            # Checkpoint + SQLite remain durable on every successful batch. The
            # full JSON output is a much larger O(catalog) snapshot, so persist
            # it less often while the router is running and always write the
            # complete output at normal termination.
            preferred_output_snapshot_batches = 4

            def __init__(self) -> None:
                self.db = KnowledgeDatabase()
                discovery = ModelDiscoveryService(database=self.db)
                health_tracker = RouteHealthTracker(self.db)
                self.scheduler = RouterScheduler(
                    discovery_service=discovery,
                    health_tracker=health_tracker,
                    concurrency_manager=ConcurrencyManager(
                        global_limit=4,
                        default_route_limit=1,
                        max_route_limit=2,
                        scalable_routes=frozenset({"opencode"}),
                    ),
                    max_attempts_per_batch=4,
                    max_attempts_per_route=2,
                    max_batch_duration_seconds=45.0,
                    max_bisection_attempts=2,
                    # A busy fast route must not stall another worker. Let that
                    # worker immediately consume a different healthy route lane.
                    primary_route_wait_seconds=0.0,
                )
                from crimson0020.translation_runtime.knowledge.terminology import TerminologyEngine
                self.terminology = TerminologyEngine(self.db)
                self._call_state = threading.local()

            def _reset_call_state(self) -> None:
                self._call_state.route_id = None
                self._call_state.model_id = None
                self._call_state.fallback_count = 0
                self._call_state.cooldown_seconds = 0.0
                self._call_state.failure_summary = None

            @property
            def last_route_id(self) -> str | None:
                return getattr(self._call_state, "route_id", None)

            @property
            def last_model_id(self) -> str | None:
                return getattr(self._call_state, "model_id", None)

            @property
            def last_fallback_count(self) -> int:
                return int(getattr(self._call_state, "fallback_count", 0))

            @property
            def last_cooldown_seconds(self) -> float:
                return float(getattr(self._call_state, "cooldown_seconds", 0.0))

            @property
            def last_failure_summary(self) -> str | None:
                return getattr(self._call_state, "failure_summary", None)

            def translate_items(
                self, items, source_lang: str, target_lang: str
            ) -> tuple[str, ...]:
                from crimson0020.translation_runtime.contracts import TranslationBatchRequest

                request_items = tuple(items)
                if not request_items:
                    return ()
                self._reset_call_state()
                req = TranslationBatchRequest(
                    job_id=f"job_sync_{request_items[0].work_id}",
                    source_language=source_lang,
                    target_language=target_lang,
                    items=request_items,
                )
                outcome = self.scheduler.schedule_batch(req)
                result_map = {r.work_id: r.translated_text for r in outcome.successes}
                failure_map = {f.work_id: f.message for f in outcome.failures}
                if outcome.successes:
                    last_success = outcome.successes[-1]
                    self._call_state.route_id = last_success.route_id
                    self._call_state.model_id = last_success.model_id
                elif outcome.attempts:
                    last_attempt = outcome.attempts[-1]
                    self._call_state.route_id = last_attempt.route_id
                    self._call_state.model_id = last_attempt.model_id
                self._call_state.fallback_count = max(0, len(outcome.attempts) - 1)
                self._call_state.cooldown_seconds = max(
                    (float(f.retry_after_seconds or 0.0) for f in outcome.failures),
                    default=0.0,
                )
                self._call_state.failure_summary = (
                    outcome.failures[0].message if outcome.failures else None
                )
                if len(result_map) != len(request_items):
                    for item in request_items:
                        if item.work_id not in result_map and item.work_id not in failure_map:
                            failure_map[item.work_id] = "AI Router returned no result for this item"
                    failure_msg = (
                        outcome.failures[0].message
                        if outcome.failures
                        else "AI Router could not translate all items"
                    )
                    if result_map:
                        raise PartialTranslationProviderError(
                            failure_msg, successes=result_map, failures=failure_map
                        )
                    if any(failure.retryable for failure in outcome.failures):
                        raise RetryableTranslationProviderError(
                            failure_msg,
                            retry_after_seconds=self._call_state.cooldown_seconds or None,
                        )
                    raise TranslationProviderError(failure_msg)
                return tuple(result_map[item.work_id] for item in request_items)

            def translate(
                self, texts: Sequence[str], source_lang: str, target_lang: str
            ) -> tuple[str, ...]:
                from crimson0020.translation_runtime.contracts import TranslationRequestItem
                scope_id = f"Crimson0020_0020_{source_lang}_{target_lang}"
                items = []
                for i, text in enumerate(texts):
                    constraints = ()
                    try:
                        constraints = self.terminology.extract_constraints(scope_id, text)
                    except Exception:
                        pass
                    items.append(
                        TranslationRequestItem(
                            work_id=f"w_{i}",
                            masked_text=text,
                            context=(),
                            context_signature="",
                            hard_terms=tuple(constraints),
                            soft_terms=(),
                        )
                    )
                return self.translate_items(items, source_lang, target_lang)

        return AIRouterProviderAdapter()
    values = dict(config or {})

    def integer_option(name: str, default: int) -> int:
        value = values.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"OpenAI-compatible {name} must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"OpenAI-compatible {name} must be an integer."
            ) from error

    def float_option(name: str, default: float) -> float:
        value = values.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"OpenAI-compatible {name} must be numeric.")
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"OpenAI-compatible {name} must be numeric."
            ) from error

    return OpenAICompatibleProvider(
        base_url=str(values.get("base_url", spec.endpoint)),
        model=str(values.get("model", "")),
        api_key=str(values.get("api_key", "")),
        timeout=float_option("timeout_seconds", 120.0),
        preferred_batch_items=integer_option("max_batch_items", 24),
        preferred_batch_characters=integer_option("max_batch_characters", 4_000),
        preferred_outer_concurrency=integer_option("concurrency", 1),
    )
