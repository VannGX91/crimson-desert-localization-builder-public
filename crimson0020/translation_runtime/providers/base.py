"""Base adapter definition for translation providers."""

from __future__ import annotations

import abc
import urllib.error
import urllib.request
from typing import Callable

from crimson0020.translation_runtime.contracts import (
    TranslationBatchOutcome,
    TranslationBatchRequest,
)

TransportCallable = Callable[
    [str, dict[str, str], bytes | str | None], tuple[int, dict[str, str], bytes]
]


class BaseTranslationAdapter(abc.ABC):
    """Abstract base class for translation adapters with deterministic transport injection."""

    def __init__(
        self,
        key: str,
        name: str,
        transport: TransportCallable | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self.transport = transport

    def execute_http(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, dict[str, str], bytes]:
        """Executes HTTP request via injected transport or standard urllib."""
        hdrs = headers or {}
        if self.transport is not None:
            return self.transport(url, hdrs, data)

        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                resp_headers = dict(resp.headers.items())
                body = resp.read()
                return status, resp_headers, body
        except urllib.error.HTTPError as error:
            resp_headers = dict(error.headers.items())
            body = error.read()
            return error.code, resp_headers, body

    @abc.abstractmethod
    def translate_batch(
        self, batch: TranslationBatchRequest
    ) -> TranslationBatchOutcome:
        """Translates a batch of items and returns typed TranslationBatchOutcome."""
