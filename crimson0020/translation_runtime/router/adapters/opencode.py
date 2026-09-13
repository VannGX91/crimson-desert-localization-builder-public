"""OpenCode router adapter supporting Chat Completions and Responses protocols."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
)
from crimson0020.translation_runtime.router.prompt import PromptBuilder
from crimson0020.translation_runtime.router.response_parser import RouterResponseParser
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

logger = logging.getLogger(__name__)

OPENCODE_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_RESPONSES_URL = "https://opencode.ai/zen/v1/responses"
APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"

# Current live-probe protocol hints. A hinted model uses its known-good protocol
# directly to avoid paying for a known-bad endpoint first. Unknown models still
# get one bounded alternate-protocol attempt on fast protocol-shaped failures.
OPENCODE_PROTOCOL_HINTS: dict[str, str] = {
    "muse-spark-1.3-contributor-free": "responses",
    "muse-spark-1.2-contributor-free": "responses",
    "ling-3.0-flash-fin-free": "chat_completions",
}
PROTOCOL_FALLBACK_STATUS_CODES: frozenset[int] = frozenset({400, 404, 405, 422, 500, 501, 502, 503})


def _extract_responses_text(response_json: dict[str, Any]) -> str:
    legacy_response = response_json.get("response")
    if isinstance(legacy_response, dict):
        legacy_content = legacy_response.get("content")
        if isinstance(legacy_content, str):
            return legacy_content

    output = response_json.get("output")
    if not isinstance(output, list):
        return ""

    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "output_text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "".join(text_parts)


class OpenCodeAdapter:
    """Adapter for anonymous OpenCode free models."""

    def __init__(
        self,
        model_id: str,
        protocol: str = "auto",
        transport: SafeHttpTransport | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
    ) -> None:
        self.model_id = model_id
        self.protocol = protocol
        self.transport = transport or SafeHttpTransport()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()

    def _protocol_order(self) -> tuple[str, ...]:
        if self.protocol in {"chat_completions", "responses"}:
            return (self.protocol,)
        if self.protocol != "auto":
            raise ValueError(f"Unsupported OpenCode protocol: {self.protocol}")

        hinted = OPENCODE_PROTOCOL_HINTS.get(self.model_id)
        if hinted:
            return (hinted,)
        return ("chat_completions", "responses")

    @staticmethod
    def _should_try_alternate(outcome: TranslationBatchOutcome) -> bool:
        if outcome.successes or not outcome.failures:
            return False
        if any(
            failure.kind in {FailureKind.RATE_LIMIT, FailureKind.TIMEOUT, FailureKind.UNAUTHORIZED}
            for failure in outcome.failures
        ):
            return False
        return any(
            attempt.status_code in PROTOCOL_FALLBACK_STATUS_CODES
            for attempt in outcome.attempts
            if attempt.status_code is not None
        )

    def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
        protocols = self._protocol_order()
        first = self._translate_batch_with_protocol(batch, protocols[0])
        if len(protocols) == 1 or not self._should_try_alternate(first):
            return first

        second = self._translate_batch_with_protocol(batch, protocols[1])
        return TranslationBatchOutcome(
            successes=second.successes,
            failures=second.failures,
            attempts=first.attempts + second.attempts,
        )

    def _translate_batch_with_protocol(
        self,
        batch: TranslationBatchRequest,
        protocol: str,
    ) -> TranslationBatchOutcome:
        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        t0 = time.time()
        try:
            envelope = self.prompt_builder.build_batch_prompt(batch)
        except Exception as e:
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message=f"Prompt build error: {e}",
                    )
                    for item in batch.items
                ),
                attempts=(),
            )

        request_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
            "x-opencode-client": "crimson0020",
            "x-opencode-project": "local",
            "x-opencode-request": request_id,
            "x-opencode-session": request_id,
        }

        if protocol == "responses":
            endpoint = OPENCODE_RESPONSES_URL
            payload: dict[str, Any] = {
                "model": self.model_id,
                "input": envelope.user_prompt,
                "instructions": envelope.system_instruction,
                "temperature": envelope.temperature,
            }
        else:
            endpoint = OPENCODE_COMPLETIONS_URL
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": envelope.system_instruction},
                    {"role": "user", "content": envelope.user_prompt},
                ],
                "temperature": envelope.temperature,
            }

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        trans_res = self.transport.execute(endpoint, headers=headers, data=body_bytes)
        duration_ms = (time.time() - t0) * 1000.0

        status_code = trans_res.status_code or (trans_res.response.status_code if trans_res.response else None)

        if not trans_res.is_success or (status_code and status_code != 200):
            kind = trans_res.failure_kind or FailureKind.NETWORK
            retryable = True
            retry_after = None

            if status_code == 429:
                kind = FailureKind.RATE_LIMIT
                retry_after = 30.0
            elif status_code in (401, 403):
                kind = FailureKind.UNAUTHORIZED
                retryable = False
            elif status_code == 404:
                kind = FailureKind.UNSUPPORTED_MODEL
                retryable = False
            elif status_code and status_code >= 500:
                kind = FailureKind.PROVIDER_INTERNAL

            attempt = RouteAttempt(
                route_id="opencode",
                model_id=self.model_id,
                duration_ms=duration_ms,
                error_kind=kind,
                status_code=status_code,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=kind,
                        retryable=retryable,
                        message=f"OpenCode HTTP error: {status_code or trans_res.error_message}",
                        retry_after_seconds=retry_after,
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        # Extract text content from JSON response
        try:
            assert trans_res.response is not None
            resp_json = json.loads(trans_res.response.body.decode("utf-8"))
            if protocol == "responses":
                raw_content = _extract_responses_text(resp_json)
            else:
                choices = resp_json.get("choices", [])
                raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
        except Exception as e:
            attempt = RouteAttempt(
                route_id="opencode",
                model_id=self.model_id,
                duration_ms=duration_ms,
                error_kind=FailureKind.INVALID_STRUCTURE,
                status_code=200,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message=f"Malformed OpenCode envelope JSON: {e}",
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        parsed = self.response_parser.parse_response(
            raw_text=raw_content,
            expected_ids=[item.work_id for item in batch.items],
        )

        successes: list[TranslationItemResult] = []
        failures: list[TranslationFailure] = []
        item_map = {item.work_id: item for item in batch.items}

        for wid, trans_text in parsed.translations.items():
            successes.append(
                TranslationItemResult(
                    work_id=wid,
                    translated_text=trans_text,
                    route_id="opencode",
                    model_id=self.model_id,
                )
            )

        for wid in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=wid,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="Translation item missing from OpenCode output",
                )
            )

        for wid in parsed.rejected_ids:
            if wid in item_map:
                failures.append(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Translation item ambiguous or rejected in OpenCode output",
                    )
                )

        attempt = RouteAttempt(
            route_id="opencode",
            model_id=self.model_id,
            duration_ms=duration_ms,
            error_kind=None if len(successes) == len(batch.items) else FailureKind.INVALID_STRUCTURE,
            status_code=200,
        )

        return TranslationBatchOutcome(
            successes=tuple(successes),
            failures=tuple(failures),
            attempts=(attempt,),
        )
