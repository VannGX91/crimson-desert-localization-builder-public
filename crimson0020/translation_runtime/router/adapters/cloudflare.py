"""Cloudflare AI isolated Edge browser adapter."""

from __future__ import annotations

import logging
import time

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
)
from crimson0020.translation_runtime.router.browser_worker import (
    IsolatedBrowserWorker,
    find_system_edge_executable,
)
from crimson0020.translation_runtime.router.prompt import PromptBuilder
from crimson0020.translation_runtime.router.response_parser import RouterResponseParser

logger = logging.getLogger(__name__)


class CloudflareAdapter:
    """Adapter for Cloudflare AI using process-isolated system Edge browser worker."""

    def __init__(
        self,
        model_id: str = "@cf/meta/llama-3.1-8b-instruct",
        edge_path: str | None = None,
        worker: IsolatedBrowserWorker | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: RouterResponseParser | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or RouterResponseParser()
        if worker is not None:
            self.worker = worker
        else:
            edge_exec = find_system_edge_executable(custom_path=edge_path)
            self.worker = IsolatedBrowserWorker(edge_executable=edge_exec) if edge_exec else None

    def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        t0 = time.time()
        if self.worker is None or (self.worker.edge_executable is None and not self.worker.runner):
            attempt = RouteAttempt(
                route_id="cloudflare",
                model_id=self.model_id,
                duration_ms=0.0,
                error_kind=FailureKind.UNSUPPORTED_MODEL,
                status_code=None,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.UNSUPPORTED_MODEL,
                        retryable=False,
                        message="Microsoft Edge browser is not available on this system for Cloudflare Edge worker",
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

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

        try:
            worker_res = self.worker.execute_inference(
                prompt=f"{envelope.system_instruction}\n\n{envelope.user_prompt}",
                allowed_origin="ai.cloudflare.com",
            )
        except Exception as e:
            attempt = RouteAttempt(
                route_id="cloudflare",
                model_id=self.model_id,
                duration_ms=(time.time() - t0) * 1000.0,
                error_kind=FailureKind.PROVIDER_INTERNAL,
                status_code=None,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=False,
                        message=f"Browser worker error: {e}",
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        duration_ms = (time.time() - t0) * 1000.0

        if worker_res.get("status") != "success":
            attempt = RouteAttempt(
                route_id="cloudflare",
                model_id=self.model_id,
                duration_ms=duration_ms,
                error_kind=FailureKind.PROVIDER_INTERNAL,
                status_code=None,
            )
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=False,
                        message=f"Worker failure: {worker_res.get('error', 'unknown error')}",
                    )
                    for item in batch.items
                ),
                attempts=(attempt,),
            )

        raw_result = worker_res.get("result", "")
        parsed = self.response_parser.parse_response(
            raw_text=raw_result,
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
                    route_id="cloudflare",
                    model_id=self.model_id,
                )
            )

        for wid in parsed.missing_ids:
            failures.append(
                TranslationFailure(
                    work_id=wid,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="Translation item missing from Cloudflare worker output",
                )
            )

        for wid in parsed.rejected_ids:
            if wid in item_map:
                failures.append(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Translation item ambiguous or rejected in Cloudflare worker output",
                    )
                )

        attempt = RouteAttempt(
            route_id="cloudflare",
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
