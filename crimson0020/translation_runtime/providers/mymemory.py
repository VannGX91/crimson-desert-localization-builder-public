"""MyMemory Free translation adapter with Unicode-safe chunking."""

from __future__ import annotations

import json
import re
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


class MyMemoryFreeAdapter(BaseTranslationAdapter):
    """Adapter for MyMemory API with 500-byte Unicode-safe chunking."""

    ENDPOINT = "https://api.mymemory.translated.net/get"
    MAX_CHUNK_BYTES = 450  # Safe margin below official 500 byte limit

    def __init__(self, transport: TransportCallable | None = None) -> None:
        super().__init__(
            key="mymemory_free",
            name="MyMemory (Free)",
            transport=transport,
        )

    def _split_into_safe_chunks(self, text: str) -> list[str]:
        """Splits text into chunks <= MAX_CHUNK_BYTES UTF-8 without cutting <xN> tokens or multibyte chars."""
        encoded = text.encode("utf-8")
        if len(encoded) <= self.MAX_CHUNK_BYTES:
            return [text]

        # Token protection pattern to avoid slicing <xN>
        token_pattern = re.compile(r"<x\d+>")
        # Match tokens or words or characters
        parts = []
        last = 0
        for m in token_pattern.finditer(text):
            start, end = m.span()
            if start > last:
                parts.extend(text[last:start].split(" "))
            parts.append(m.group(0))
            last = end
        if last < len(text):
            parts.extend(text[last:].split(" "))

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_bytes = 0

        for part in parts:
            if not part:
                continue
            part_bytes = len(part.encode("utf-8")) + (1 if current_chunk else 0)
            if current_bytes + part_bytes > self.MAX_CHUNK_BYTES and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [part]
                current_bytes = len(part.encode("utf-8"))
            else:
                current_chunk.append(part)
                current_bytes += part_bytes

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def translate_batch(
        self, batch: TranslationBatchRequest
    ) -> TranslationBatchOutcome:
        start_time = time.time()
        successes: list[TranslationItemResult] = []
        failures: list[TranslationFailure] = []
        attempts: list[RouteAttempt] = []

        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        langpair = f"{batch.source_language}|{batch.target_language}"

        for item in batch.items:
            mask = mask_protected_content(item.masked_text)
            chunks = self._split_into_safe_chunks(mask.masked_text)
            translated_chunks: list[str] = []
            item_failed = False

            for chunk in chunks:
                query_params = {
                    "q": chunk,
                    "langpair": langpair,
                }
                url = f"{self.ENDPOINT}?{urllib.parse.urlencode(query_params)}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                }

                try:
                    status, resp_headers, body = self.execute_http(
                        url=url, headers=headers
                    )
                    duration_ms = (time.time() - start_time) * 1000

                    if status == 200:
                        data = json.loads(body.decode("utf-8"))
                        resp_data = data.get("responseData") or {}
                        trans_text = resp_data.get("translatedText")
                        if not isinstance(trans_text, str):
                            failures.append(
                                TranslationFailure(
                                    work_id=item.work_id,
                                    kind=FailureKind.INVALID_STRUCTURE,
                                    retryable=False,
                                    message="MyMemory response missing translatedText.",
                                )
                            )
                            item_failed = True
                            break
                        translated_chunks.append(trans_text)
                    elif status == 429:
                        failures.append(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.RATE_LIMIT,
                                retryable=True,
                                message="Rate limit reached on MyMemory API.",
                            )
                        )
                        item_failed = True
                        break
                    else:
                        failures.append(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.NETWORK,
                                retryable=True,
                                message=f"HTTP {status} from MyMemory.",
                            )
                        )
                        item_failed = True
                        break

                except TimeoutError as exc:
                    failures.append(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.TIMEOUT,
                            retryable=True,
                            message=f"MyMemory timeout: {exc}",
                        )
                    )
                    item_failed = True
                    break
                except Exception as exc:
                    failures.append(
                        TranslationFailure(
                            work_id=item.work_id,
                            kind=FailureKind.PROVIDER_INTERNAL,
                            retryable=False,
                            message=f"MyMemory error: {exc}",
                        )
                    )
                    item_failed = True
                    break

            if not item_failed:
                full_translated_masked = " ".join(translated_chunks)
                restored, err = restore_protected_content(
                    full_translated_masked, mask, work_id=item.work_id
                )
                if err is None:
                    successes.append(
                        TranslationItemResult(
                            work_id=item.work_id,
                            translated_text=restored,
                            route_id=self.key,
                            model_id="api",
                        )
                    )
                else:
                    failures.append(err)

        duration_ms = (time.time() - start_time) * 1000
        attempts.append(RouteAttempt(self.key, "api", duration_ms, status_code=200))

        return TranslationBatchOutcome(
            successes=tuple(successes),
            failures=tuple(failures),
            attempts=tuple(attempts),
        )
