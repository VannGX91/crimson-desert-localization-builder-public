"""Google Translate Free web adapter implementation."""

from __future__ import annotations

import json
import time
import urllib.parse

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
)
from crimson0020.translation_runtime.protection import (
    mask_protected_content,
    restore_protected_content,
)
from crimson0020.translation_runtime.providers.base import (
    BaseTranslationAdapter,
    TransportCallable,
)


class GoogleFreeAdapter(BaseTranslationAdapter):
    """Adapter for Google Translate's free web endpoint."""

    ENDPOINT = "https://translate.google.com/translate_a/t"
    MAX_ITEMS = 2_000
    MAX_CHARACTERS = 64_000

    def __init__(self, transport: TransportCallable | None = None) -> None:
        super().__init__(
            key="google_free",
            name="Google Translate (Free)",
            transport=transport,
        )

    def _parse_retry_after(self, headers: dict[str, str]) -> float | None:
        val = headers.get("Retry-After") or headers.get("retry-after")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return 30.0

    def translate_batch(
        self, batch: TranslationBatchRequest
    ) -> TranslationBatchOutcome:
        start_time = time.time()
        successes: list[TranslationItemResult] = []
        failures: list[TranslationFailure] = []
        attempts: list[RouteAttempt] = []

        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        # Prepare request items and masks
        masks = [mask_protected_content(item.masked_text) for item in batch.items]
        query_params = {
            "client": "te",
            "format": "html",
            "v": "1.0",
            "sl": batch.source_language,
            "tl": batch.target_language,
        }
        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(query_params)}"
        # Build POST body with repeated q fields
        post_data = urllib.parse.urlencode(
            [("q", m.masked_text) for m in masks]
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        }

        try:
            status, resp_headers, body = self.execute_http(
                url=url, headers=headers, data=post_data
            )
            duration_ms = (time.time() - start_time) * 1000

            if status == 200:
                try:
                    data = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    attempts.append(
                        RouteAttempt(
                            self.key,
                            "web",
                            duration_ms,
                            FailureKind.INVALID_STRUCTURE,
                            status,
                        )
                    )
                    for item in batch.items:
                        failures.append(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.INVALID_STRUCTURE,
                                retryable=False,
                                message=f"Malformed response JSON from Google: {exc}",
                            )
                        )
                    return TranslationBatchOutcome(
                        successes=(),
                        failures=tuple(failures),
                        attempts=tuple(attempts),
                    )

                # Extract translated strings
                translations: list[str] = []
                if isinstance(data, list):
                    for row in data:
                        if isinstance(row, list) and row:
                            translations.append(str(row[0]))
                        elif isinstance(row, str):
                            translations.append(row)

                if len(translations) != len(batch.items):
                    attempts.append(
                        RouteAttempt(
                            self.key,
                            "web",
                            duration_ms,
                            FailureKind.MISSING_ITEM,
                            status,
                        )
                    )
                    for item in batch.items:
                        failures.append(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.MISSING_ITEM,
                                retryable=True,
                                message="Provider returned mismatched number of translated items.",
                            )
                        )
                    return TranslationBatchOutcome(
                        successes=(),
                        failures=tuple(failures),
                        attempts=tuple(attempts),
                    )

                # Sibling-isolated restoration
                for item, trans_text, mask in zip(
                    batch.items, translations, masks, strict=True
                ):
                    restored, err = restore_protected_content(
                        trans_text, mask, work_id=item.work_id
                    )
                    if err is None:
                        successes.append(
                            TranslationItemResult(
                                work_id=item.work_id,
                                translated_text=restored,
                                route_id=self.key,
                                model_id="web",
                            )
                        )
                    else:
                        failures.append(err)

                attempts.append(
                    RouteAttempt(self.key, "web", duration_ms, status_code=200)
                )

            elif status == 429:
                retry_after = self._parse_retry_after(resp_headers)
                attempts.append(
                    RouteAttempt(
                        self.key,
                        "web",
                        duration_ms,
                        FailureKind.RATE_LIMIT,
                        status,
                    )
                )
                for item in batch.items:
                    failures.append(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.RATE_LIMIT,
                            retryable=True,
                            message="Rate limit exceeded on Google Translate web.",
                            retry_after_seconds=retry_after,
                        )
                    )

            elif status in (401, 403):
                attempts.append(
                    RouteAttempt(
                        self.key,
                        "web",
                        duration_ms,
                        FailureKind.UNAUTHORIZED,
                        status,
                    )
                )
                for item in batch.items:
                    failures.append(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.UNAUTHORIZED,
                            retryable=False,
                            message=f"Unauthorized/Forbidden HTTP {status} from Google.",
                        )
                    )

            else:
                attempts.append(
                    RouteAttempt(
                        self.key,
                        "web",
                        duration_ms,
                        FailureKind.NETWORK,
                        status,
                    )
                )
                for item in batch.items:
                    failures.append(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.NETWORK,
                            retryable=True,
                            message=f"HTTP {status} from Google Translate.",
                        )
                    )

        except TimeoutError as exc:
            duration_ms = (time.time() - start_time) * 1000
            attempts.append(
                RouteAttempt(
                    self.key, "web", duration_ms, FailureKind.TIMEOUT
                )
            )
            for item in batch.items:
                failures.append(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.TIMEOUT,
                        retryable=True,
                        message=f"Request to Google timed out: {exc}",
                    )
                )

        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            attempts.append(
                RouteAttempt(
                    self.key, "web", duration_ms, FailureKind.PROVIDER_INTERNAL
                )
            )
            for item in batch.items:
                failures.append(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=False,
                        message=f"Unexpected error communicating with Google: {exc}",
                    )
                )

        return TranslationBatchOutcome(
            successes=tuple(successes),
            failures=tuple(failures),
            attempts=tuple(attempts),
        )
