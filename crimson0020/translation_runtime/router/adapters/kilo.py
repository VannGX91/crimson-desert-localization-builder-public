"""Kilo Gateway and anonymous route adapters with shared parsing and isolated cooldown."""

from __future__ import annotations

import json
import logging
import time
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

KILO_GATEWAY_URL = "https://api.kilo.ai/api/gateway/chat/completions"
KILO_ANONYMOUS_URL = "https://api.kilo.ai/api/openrouter/chat/completions"
APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"
KILO_ANONYMOUS_BEARER = "Bearer anonymous"


class KiloAdapter:
    """Adapter for Kilo Gateway and anonymous AI routes."""

    def __init__(
        self,
        route_id: str = "kilo_gateway",
        model_id: str = "kilo/free-model",
        transport: SafeHttpTransport | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
    ) -> None:
        if route_id not in ("kilo_gateway", "kilo_anonymous"):
            raise ValueError(f"Unknown Kilo route_id: {route_id}")
        self.route_id = route_id
        self.model_id = model_id
        self.transport = transport or SafeHttpTransport()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()

    def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
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

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.route_id == "kilo_anonymous":
            headers["Authorization"] = KILO_ANONYMOUS_BEARER
            headers["X-KILOCODE-EDITORNAME"] = "Crimson0020"

        endpoint = KILO_GATEWAY_URL if self.route_id == "kilo_gateway" else KILO_ANONYMOUS_URL

        payload: dict[str, Any] = {
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
                retry_after = 45.0
            elif status_code in (401, 403):
                kind = FailureKind.UNAUTHORIZED
                retryable = False
            elif status_code == 404:
                kind = FailureKind.UNSUPPORTED_MODEL
                retryable = False
            elif status_code and status_code >= 500:
                kind = FailureKind.PROVIDER_INTERNAL

            attempt = RouteAttempt(
                route_id=self.route_id,
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
                        message=f"Kilo HTTP error: {status_code or trans_res.error_message}",
                        retry_after_seconds=retry_after,
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        # Parse model response
        try:
            assert trans_res.response is not None
            resp_json = json.loads(trans_res.response.body.decode("utf-8"))
            choices = resp_json.get("choices", [])
            raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
        except Exception as e:
            attempt = RouteAttempt(
                route_id=self.route_id,
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
                        message=f"Malformed Kilo JSON response: {e}",
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
                    route_id=self.route_id,
                    model_id=self.model_id,
                )
            )

        for wid in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=wid,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="Translation item missing from Kilo response",
                )
            )

        for wid in parsed.rejected_ids:
            if wid in item_map:
                failures.append(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Translation item ambiguous or rejected in Kilo response",
                    )
                )

        attempt = RouteAttempt(
            route_id=self.route_id,
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
