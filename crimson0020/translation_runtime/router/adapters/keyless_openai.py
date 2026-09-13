"""Fixed, live-verified keyless OpenAI-compatible router adapters."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.prompt import PromptBuilder
from crimson0020.translation_runtime.router.response_parser import RouterResponseParser
from crimson0020.translation_runtime.router.transport import SafeHttpTransport


APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"
LLM7_MAX_BATCH_ITEMS = 24
LLM7_MAX_BATCH_CHARACTERS = 6_000
ROUTE_BATCH_LIMITS: dict[str, tuple[int, int]] = {
    "llm7": (LLM7_MAX_BATCH_ITEMS, LLM7_MAX_BATCH_CHARACTERS),
    # Current anonymous endpoints complete these sizes well inside the shared
    # 30-second transport timeout. Larger 48-item prompts can exceed it.
    "blockrun": (32, 6_000),
    "vireonix": (16, 4_000),
}

ROUTE_READ_TIMEOUT_SECONDS: dict[str, float] = {
    "blockrun": 12.0,
    "cehpoint": 12.0,
}


@dataclass(frozen=True, slots=True)
class KeylessOpenAIRoute:
    endpoint: str
    default_model: str
    dummy_authorization: str | None = None


KEYLESS_OPENAI_ROUTES: dict[str, KeylessOpenAIRoute] = {
    "llm7": KeylessOpenAIRoute(
        endpoint="https://api.llm7.io/v1/chat/completions",
        default_model="default",
        dummy_authorization="Bearer unused",
    ),
    "blockrun": KeylessOpenAIRoute(
        endpoint="https://blockrun.ai/api/v1/chat/completions",
        default_model="nvidia/gpt-oss-20b",
    ),
    "vireonix": KeylessOpenAIRoute(
        endpoint="https://vireonix.ai/v1/chat/completions",
        default_model="auto",
    ),
    "cehpoint": KeylessOpenAIRoute(
        endpoint="https://ai-api.cehpoint.co.in/v1/chat/completions",
        default_model="cehpoint-ai",
    ),
}


class KeylessOpenAIAdapter:
    """OpenAI-compatible adapter limited to a frozen set of anonymous routes."""

    def __init__(
        self,
        route_id: str,
        model_id: str | None = None,
        transport: SafeHttpTransport | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
    ) -> None:
        if route_id not in KEYLESS_OPENAI_ROUTES:
            raise ValueError(f"Unsupported keyless route: {route_id}")
        route = KEYLESS_OPENAI_ROUTES[route_id]
        self.route_id = route_id
        self.model_id = model_id or route.default_model
        self.route = route
        self.transport = transport or SafeHttpTransport(
            read_timeout_seconds=ROUTE_READ_TIMEOUT_SECONDS.get(route_id, 30.0)
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()

    @property
    def scheduler_batch_limits(self) -> tuple[int, int] | None:
        """Return the per-attempt item/character budget used by RouterScheduler."""
        return ROUTE_BATCH_LIMITS.get(self.route_id)

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

    def _route_batches(self, batch: TranslationBatchRequest) -> tuple[TranslationBatchRequest, ...]:
        limits = ROUTE_BATCH_LIMITS.get(self.route_id)
        if limits is None or not batch.items:
            return (batch,)
        max_items, max_characters = limits

        chunks: list[tuple[TranslationRequestItem, ...]] = []
        current: list[TranslationRequestItem] = []
        current_characters = 0
        for item in batch.items:
            item_characters = len(item.masked_text)
            if current and (
                len(current) >= max_items
                or current_characters + item_characters > max_characters
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
                job_id=f"{batch.job_id}_{self.route_id}_{index + 1}",
                source_language=batch.source_language,
                target_language=batch.target_language,
                items=items,
            )
            for index, items in enumerate(chunks)
        )

    def _translate_batch_once(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        try:
            envelope = self.prompt_builder.build_batch_prompt(batch)
        except Exception as exc:
            return self._failure_outcome(
                batch,
                FailureKind.INVALID_STRUCTURE,
                retryable=False,
                message=f"Prompt build error: {exc}",
            )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.route.dummy_authorization is not None:
            headers["Authorization"] = self.route.dummy_authorization

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": envelope.system_instruction},
                {"role": "user", "content": envelope.user_prompt},
            ],
            "temperature": envelope.temperature,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        t0 = time.time()
        result = self.transport.execute(self.route.endpoint, headers=headers, data=body)
        duration_ms = (time.time() - t0) * 1000.0
        status_code = result.status_code or (result.response.status_code if result.response else None)

        if not result.is_success or (status_code and status_code != 200):
            kind, retryable, retry_after = self._classify_failure(result.failure_kind, status_code)
            return self._failure_outcome(
                batch,
                kind,
                retryable=retryable,
                message=f"{self.route_id} HTTP error: {status_code or result.error_message}",
                retry_after=retry_after,
                duration_ms=duration_ms,
                status_code=status_code,
            )

        try:
            assert result.response is not None
            response_json = json.loads(result.response.body.decode("utf-8"))
            choices = response_json.get("choices", [])
            raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
        except Exception as exc:
            return self._failure_outcome(
                batch,
                FailureKind.INVALID_STRUCTURE,
                retryable=False,
                message=f"Malformed {self.route_id} JSON response: {exc}",
                duration_ms=duration_ms,
                status_code=200,
            )

        parsed = self.response_parser.parse_response(
            raw_text=raw_content,
            expected_ids=[item.work_id for item in batch.items],
        )
        successes = tuple(
            TranslationItemResult(
                work_id=work_id,
                translated_text=translated_text,
                route_id=self.route_id,
                model_id=self.model_id,
            )
            for work_id, translated_text in parsed.translations.items()
        )
        failures: list[TranslationFailure] = []
        item_ids = {item.work_id for item in batch.items}
        for work_id in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=work_id,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message=f"Translation item missing from {self.route_id} response",
                )
            )
        for work_id in parsed.rejected_ids:
            if work_id in item_ids:
                failures.append(
                    TranslationFailure(
                        work_id=work_id,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message=f"Translation item rejected in {self.route_id} response",
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
            successes=successes,
            failures=tuple(failures),
            attempts=(attempt,),
        )

    @staticmethod
    def _classify_failure(
        transport_kind: FailureKind | None,
        status_code: int | None,
    ) -> tuple[FailureKind, bool, float | None]:
        kind = transport_kind or FailureKind.NETWORK
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
        return kind, retryable, retry_after

    def _failure_outcome(
        self,
        batch: TranslationBatchRequest,
        kind: FailureKind,
        *,
        retryable: bool,
        message: str,
        retry_after: float | None = None,
        duration_ms: float = 0.0,
        status_code: int | None = None,
    ) -> TranslationBatchOutcome:
        attempts = ()
        if duration_ms or status_code is not None:
            attempts = (
                RouteAttempt(
                    route_id=self.route_id,
                    model_id=self.model_id,
                    duration_ms=duration_ms,
                    error_kind=kind,
                    status_code=status_code,
                ),
            )
        return TranslationBatchOutcome(
            successes=(),
            failures=tuple(
                TranslationFailure(
                    work_id=item.work_id,
                    kind=kind,
                    retryable=retryable,
                    message=message,
                    retry_after_seconds=retry_after,
                )
                for item in batch.items
            ),
            attempts=attempts,
        )
