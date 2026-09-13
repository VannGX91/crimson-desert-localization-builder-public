"""UncloseAI router adapter supporting only discovered anonymous free models."""

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

UNCLOSE_COMPLETIONS_URL = "https://hermes.ai.unturf.com/v1/chat/completions"
APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"
UNCLOSE_MAX_BATCH_ITEMS = 16
UNCLOSE_MAX_BATCH_CHARACTERS = 4_000


class UncloseAdapter:
    """Adapter for anonymous UncloseAI free models with no user key UI/config."""

    scheduler_batch_limits = (UNCLOSE_MAX_BATCH_ITEMS, UNCLOSE_MAX_BATCH_CHARACTERS)

    def __init__(
        self,
        model_id: str = "Lorbus/Qwen3.6-27B-int4-AutoRound",
        transport: SafeHttpTransport | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
    ) -> None:
        self.model_id = model_id
        self.transport = transport or SafeHttpTransport(read_timeout_seconds=12.0)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()

    def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
        route_batches = self._route_batches(batch)
        if len(route_batches) == 1:
            return self._translate_batch_once(route_batches[0])

        successes: list[TranslationItemResult] = []
        failures: list[TranslationFailure] = []
        attempts: list[RouteAttempt] = []
        for index, route_batch in enumerate(route_batches):
            outcome = self._translate_batch_once(route_batch)
            successes.extend(outcome.successes)
            failures.extend(outcome.failures)
            attempts.extend(outcome.attempts)
            if outcome.failures and not outcome.successes:
                exemplar = outcome.failures[0]
                for later_batch in route_batches[index + 1 :]:
                    failures.extend(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=exemplar.kind,
                            retryable=exemplar.retryable,
                            message=exemplar.message,
                            retry_after_seconds=exemplar.retry_after_seconds,
                        )
                        for item in later_batch.items
                    )
                break

        return TranslationBatchOutcome(
            successes=tuple(successes),
            failures=tuple(failures),
            attempts=tuple(attempts),
        )

    @staticmethod
    def _route_batches(batch: TranslationBatchRequest) -> tuple[TranslationBatchRequest, ...]:
        if not batch.items:
            return (batch,)

        chunks: list[tuple] = []
        current: list = []
        current_characters = 0
        for item in batch.items:
            item_characters = len(item.masked_text)
            if current and (
                len(current) >= UNCLOSE_MAX_BATCH_ITEMS
                or current_characters + item_characters > UNCLOSE_MAX_BATCH_CHARACTERS
            ):
                chunks.append(tuple(current))
                current = []
                current_characters = 0
            current.append(item)
            current_characters += item_characters
        if current:
            chunks.append(tuple(current))

        if len(chunks) == 1:
            return (batch,)
        return tuple(
            TranslationBatchRequest(
                job_id=f"{batch.job_id}_unclose_{index + 1}",
                source_language=batch.source_language,
                target_language=batch.target_language,
                items=items,
            )
            for index, items in enumerate(chunks)
        )

    def _translate_batch_once(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
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

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": envelope.system_instruction},
                {"role": "user", "content": envelope.user_prompt},
            ],
            "temperature": envelope.temperature,
            "response_format": {"type": "json_object"},
        }

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        trans_res = self.transport.execute(UNCLOSE_COMPLETIONS_URL, headers=headers, data=body_bytes)
        duration_ms = (time.time() - t0) * 1000.0

        status_code = trans_res.status_code or (trans_res.response.status_code if trans_res.response else None)

        if not trans_res.is_success or (status_code and status_code != 200):
            kind = trans_res.failure_kind or FailureKind.NETWORK
            retryable = True
            retry_after = None

            if status_code == 429:
                kind = FailureKind.RATE_LIMIT
                retry_after = 60.0
            elif status_code in (401, 403):
                # Transition to key-required disables the route
                kind = FailureKind.UNAUTHORIZED
                retryable = False
            elif status_code == 404:
                kind = FailureKind.UNSUPPORTED_MODEL
                retryable = False
            elif status_code and status_code >= 500:
                kind = FailureKind.PROVIDER_INTERNAL

            attempt = RouteAttempt(
                route_id="unclose",
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
                        message=f"UncloseAI HTTP error: {status_code or trans_res.error_message}",
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
                route_id="unclose",
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
                        message=f"Malformed UncloseAI JSON response: {e}",
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
                    route_id="unclose",
                    model_id=self.model_id,
                )
            )

        for wid in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=wid,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="Translation item missing from Unclose response",
                )
            )

        for wid in parsed.rejected_ids:
            if wid in item_map:
                failures.append(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Translation item ambiguous or rejected in Unclose response",
                    )
                )

        attempt = RouteAttempt(
            route_id="unclose",
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
