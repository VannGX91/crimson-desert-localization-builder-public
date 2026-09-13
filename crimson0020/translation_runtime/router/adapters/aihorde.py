"""AI Horde asynchronous router adapter with state machine polling and bounds."""

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

HORDE_ASYNC_SUBMIT_URL = "https://stablehorde.net/api/v2/generate/text/async"
HORDE_STATUS_BASE_URL = "https://stablehorde.net/api/v2/generate/text/status"
HORDE_ANONYMOUS_KEY = "0000000000"
APP_CLIENT_AGENT = "Crimson0020:3.0:autonomous"
APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"


class AIHordeAdapter:
    """Adapter for AI Horde anonymous crowd-sourced asynchronous inference."""

    allow_structural_bisection = False

    def __init__(
        self,
        model_id: str = "aphrodite/free",
        transport: SafeHttpTransport | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
        poll_interval: float = 2.0,
        total_deadline_seconds: float = 15.0,
    ) -> None:
        self.model_id = model_id
        self.transport = transport or SafeHttpTransport()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()
        self.poll_interval = poll_interval
        self.total_deadline_seconds = total_deadline_seconds

    @property
    def scheduler_batch_limits(self) -> tuple[int, int]:
        return (8, 2_000)

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

        submit_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Client-Agent": APP_CLIENT_AGENT,
            "apikey": HORDE_ANONYMOUS_KEY,
            "User-Agent": APP_USER_AGENT,
        }

        submit_payload: dict[str, Any] = {
            "prompt": f"{envelope.system_instruction}\n\n{envelope.user_prompt}",
            "params": {
                "temperature": envelope.temperature,
                "max_length": 512,
                "max_context_length": 4096,
                "n": 1,
            },
            "models": [self.model_id],
        }

        body_bytes = json.dumps(submit_payload, ensure_ascii=False).encode("utf-8")
        submit_res = self.transport.execute(HORDE_ASYNC_SUBMIT_URL, headers=submit_headers, data=body_bytes)

        duration_ms = (time.time() - t0) * 1000.0
        status_code = submit_res.status_code or (submit_res.response.status_code if submit_res.response else None)

        if not submit_res.is_success or (status_code and status_code not in (200, 202)):
            kind = submit_res.failure_kind or FailureKind.NETWORK
            retryable = True
            retry_after = None

            if status_code == 429:
                kind = FailureKind.RATE_LIMIT
                retry_after = 60.0
            elif status_code in (401, 403):
                kind = FailureKind.UNAUTHORIZED
                retryable = False
            elif status_code == 404:
                kind = FailureKind.UNSUPPORTED_MODEL
                retryable = False
            elif status_code and status_code >= 500:
                kind = FailureKind.PROVIDER_INTERNAL

            attempt = RouteAttempt(
                route_id="aihorde",
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
                        message=f"AI Horde submit error: {status_code or submit_res.error_message}",
                        retry_after_seconds=retry_after,
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        try:
            assert submit_res.response is not None
            job_obj = json.loads(submit_res.response.body.decode("utf-8"))
            job_id = job_obj.get("id")
            if not job_id:
                raise ValueError("No job ID returned in Horde submit response")
        except Exception as e:
            attempt = RouteAttempt(
                route_id="aihorde",
                model_id=self.model_id,
                duration_ms=duration_ms,
                error_kind=FailureKind.INVALID_STRUCTURE,
                status_code=status_code,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message=f"Malformed Horde submit JSON: {e}",
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        # Polling state machine
        status_url = f"{HORDE_STATUS_BASE_URL}/{job_id}"
        poll_headers = {
            "Accept": "application/json",
            "Client-Agent": APP_CLIENT_AGENT,
            "apikey": HORDE_ANONYMOUS_KEY,
            "User-Agent": APP_USER_AGENT,
        }

        while True:
            elapsed = time.time() - t0
            if elapsed >= self.total_deadline_seconds:
                attempt = RouteAttempt(
                    route_id="aihorde",
                    model_id=self.model_id,
                    duration_ms=elapsed * 1000.0,
                    error_kind=FailureKind.TIMEOUT,
                    status_code=None,
                )
                return TranslationBatchOutcome(
                    successes=(),
                    failures=tuple(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.TIMEOUT,
                            retryable=True,
                            message=f"AI Horde polling exceeded deadline of {self.total_deadline_seconds}s",
                        )
                        for item in batch.items
                    ),
                    attempts=(attempt,),
                )

            time.sleep(self.poll_interval)
            poll_res = self.transport.execute(status_url, headers=poll_headers)
            poll_status = poll_res.status_code or (poll_res.response.status_code if poll_res.response else None)

            if not poll_res.is_success or (poll_status and poll_status != 200):
                continue

            try:
                assert poll_res.response is not None
                poll_data = json.loads(poll_res.response.body.decode("utf-8"))
                if poll_data.get("faulted") is True:
                    attempt = RouteAttempt(
                        route_id="aihorde",
                        model_id=self.model_id,
                        duration_ms=(time.time() - t0) * 1000.0,
                        error_kind=FailureKind.PROVIDER_INTERNAL,
                        status_code=200,
                    )
                    return TranslationBatchOutcome(
                        successes=(),
                        failures=tuple(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.PROVIDER_INTERNAL,
                                retryable=True,
                                message="AI Horde worker faulted during generation",
                            )
                            for item in batch.items
                        ),
                        attempts=(attempt,),
                    )

                if poll_data.get("done") is True:
                    gens = poll_data.get("generations", [])
                    raw_text = gens[0].get("text", "") if gens else ""
                    break
            except Exception:
                continue

        duration_ms = (time.time() - t0) * 1000.0
        parsed = self.response_parser.parse_response(
            raw_text=raw_text,
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
                    route_id="aihorde",
                    model_id=self.model_id,
                )
            )

        for wid in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=wid,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="Translation item missing from AI Horde generation",
                )
            )

        for wid in parsed.rejected_ids:
            if wid in item_map:
                failures.append(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Translation item ambiguous or rejected in AI Horde output",
                    )
                )

        attempt = RouteAttempt(
            route_id="aihorde",
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
