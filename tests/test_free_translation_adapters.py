import unittest

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.providers.google import GoogleFreeAdapter
from crimson0020.translation_runtime.providers.mymemory import MyMemoryFreeAdapter


class TestFreeTranslationAdapters(unittest.TestCase):
    def test_google_adapter_success_and_protection(self):
        def fake_transport(url, headers, body):
            # Google returns JSON list of translated strings
            # For 2 items: ["Xin chào <x0>", "Tạm biệt"]
            resp = b'[["Xin ch\xc3\xa0o <x0>"], ["T\xe1\xba\xa1m bi\xe1\xbb\x87t"]]'
            return 200, {"Content-Type": "application/json"}, resp

        adapter = GoogleFreeAdapter(transport=fake_transport)
        item1 = TranslationRequestItem("w1", "Hello <x0>", (), "sig1", (), ())
        item2 = TranslationRequestItem("w2", "Goodbye", (), "sig2", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item1, item2))

        outcome = adapter.translate_batch(batch)
        self.assertEqual(len(outcome.successes), 2)
        self.assertEqual(len(outcome.failures), 0)
        self.assertEqual(outcome.successes[0].translated_text, "Xin chào <x0>")
        self.assertEqual(outcome.successes[1].translated_text, "Tạm biệt")

    def test_google_adapter_rate_limit_and_retry_after(self):
        def fake_transport(url, headers, body):
            return 429, {"Retry-After": "45"}, b"Rate limit exceeded"

        adapter = GoogleFreeAdapter(transport=fake_transport)
        item = TranslationRequestItem("w1", "Hello", (), "sig", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item,))

        outcome = adapter.translate_batch(batch)
        self.assertEqual(len(outcome.successes), 0)
        self.assertEqual(len(outcome.failures), 1)
        fail = outcome.failures[0]
        self.assertEqual(fail.kind, FailureKind.RATE_LIMIT)
        self.assertTrue(fail.retryable)
        self.assertEqual(fail.retry_after_seconds, 45.0)

    def test_google_adapter_partial_item_isolation(self):
        def fake_transport(url, headers, body):
            # First item translated with broken token, second item translated cleanly
            resp = b'[["Xin ch\xc3\xa0o <X0>"], ["Th\xe1\xba\xbf gi\xe1\xbb\x9bi"]]'
            return 200, {"Content-Type": "application/json"}, resp

        adapter = GoogleFreeAdapter(transport=fake_transport)
        item1 = TranslationRequestItem("w1", "Hello <x0>", (), "sig1", (), ())
        item2 = TranslationRequestItem("w2", "World", (), "sig2", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item1, item2))

        outcome = adapter.translate_batch(batch)
        # Sibling isolation: item 1 failed, item 2 succeeded!
        self.assertEqual(len(outcome.successes), 1)
        self.assertEqual(outcome.successes[0].work_id, "w2")
        self.assertEqual(len(outcome.failures), 1)
        self.assertEqual(outcome.failures[0].work_id, "w1")
        self.assertEqual(outcome.failures[0].kind, FailureKind.PROTECTED_CONTENT_VIOLATION)

    def test_mymemory_adapter_chunking_and_success(self):
        def fake_transport(url, headers, body):
            # MyMemory response JSON format: {"responseData": {"translatedText": "Xin chào"}}
            resp = b'{"responseData": {"translatedText": "Xin ch\xc3\xa0o <x0>"}, "responseStatus": 200}'
            return 200, {"Content-Type": "application/json"}, resp

        adapter = MyMemoryFreeAdapter(transport=fake_transport)
        item = TranslationRequestItem("w1", "Hello <x0>", (), "sig", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item,))

        outcome = adapter.translate_batch(batch)
        self.assertEqual(len(outcome.successes), 1)
        self.assertEqual(outcome.successes[0].translated_text, "Xin chào <x0>")

    def test_mymemory_chunking_unicode_safe_boundary(self):
        # Text exceeding 500 bytes with multibyte characters
        long_text = "Tiếng Việt " * 60  # ~660 bytes UTF-8
        calls = []

        def fake_transport(url, headers, body):
            calls.append(url)
            return 200, {"Content-Type": "application/json"}, b'{"responseData": {"translatedText": "ok"}, "responseStatus": 200}'

        adapter = MyMemoryFreeAdapter(transport=fake_transport)
        item = TranslationRequestItem("w1", long_text, (), "sig", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item,))

        outcome = adapter.translate_batch(batch)
        self.assertGreater(len(calls), 1)  # Confirms it was chunked
        self.assertEqual(len(outcome.successes), 1)

    def test_adapter_5xx_and_timeout(self):
        def timeout_transport(url, headers, body):
            raise TimeoutError("Connection timed out")

        adapter = GoogleFreeAdapter(transport=timeout_transport)
        item = TranslationRequestItem("w1", "Hello", (), "sig", (), ())
        batch = TranslationBatchRequest("j1", "en", "vi", (item,))

        outcome = adapter.translate_batch(batch)
        self.assertEqual(len(outcome.failures), 1)
        self.assertEqual(outcome.failures[0].kind, FailureKind.TIMEOUT)
        self.assertTrue(outcome.failures[0].retryable)

    def test_empty_batch(self):
        adapter = GoogleFreeAdapter()
        outcome = adapter.translate_batch(TranslationBatchRequest("j1", "en", "vi", ()))
        self.assertEqual(len(outcome.successes), 0)
        self.assertEqual(len(outcome.failures), 0)

        my_mem = MyMemoryFreeAdapter()
        outcome2 = my_mem.translate_batch(TranslationBatchRequest("j1", "en", "vi", ()))
        self.assertEqual(len(outcome2.successes), 0)
        self.assertEqual(len(outcome2.failures), 0)

    def test_google_errors_and_malformed_responses(self):
        # 401 Unauthorized
        adapter = GoogleFreeAdapter(transport=lambda u, h, b: (401, {}, b"Unauthorized"))
        batch = TranslationBatchRequest("j1", "en", "vi", (TranslationRequestItem("w1", "Hi", (), "s", (), ()),))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.UNAUTHORIZED)

        # 500 Network
        adapter = GoogleFreeAdapter(transport=lambda u, h, b: (500, {}, b"Server error"))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.NETWORK)

        # Invalid JSON
        adapter = GoogleFreeAdapter(transport=lambda u, h, b: (200, {}, b"Not JSON"))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.INVALID_STRUCTURE)

        # Mismatched items length
        adapter = GoogleFreeAdapter(transport=lambda u, h, b: (200, {}, b'[["one"]]'))
        batch2 = TranslationBatchRequest("j1", "en", "vi", (
            TranslationRequestItem("w1", "Hi", (), "s", (), ()),
            TranslationRequestItem("w2", "Bye", (), "s2", (), ())
        ))
        out = adapter.translate_batch(batch2)
        self.assertEqual(out.failures[0].kind, FailureKind.MISSING_ITEM)

        # Unexpected exception
        def failing_transport(u, h, b):
            raise RuntimeError("Boom")
        adapter = GoogleFreeAdapter(transport=failing_transport)
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.PROVIDER_INTERNAL)

        # Non-numeric retry-after header
        adapter = GoogleFreeAdapter(transport=lambda u, h, b: (429, {"Retry-After": "invalid"}, b"Rate limit"))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].retry_after_seconds, 30.0)

    def test_mymemory_errors_and_failures(self):
        batch = TranslationBatchRequest("j1", "en", "vi", (TranslationRequestItem("w1", "Hi", (), "s", (), ()),))

        # Missing translatedText
        adapter = MyMemoryFreeAdapter(transport=lambda u, h, b: (200, {}, b'{"responseData": {}}'))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.INVALID_STRUCTURE)

        # 429 Rate limit
        adapter = MyMemoryFreeAdapter(transport=lambda u, h, b: (429, {}, b"Rate limited"))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.RATE_LIMIT)

        # 500 Network
        adapter = MyMemoryFreeAdapter(transport=lambda u, h, b: (500, {}, b"500 Internal"))
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.NETWORK)

        # Timeout
        def timeout_transport(u, h, b):
            raise TimeoutError("Timeout")
        adapter = MyMemoryFreeAdapter(transport=timeout_transport)
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.TIMEOUT)

        # Generic exception
        def boom_transport(u, h, b):
            raise RuntimeError("Generic error")
        adapter = MyMemoryFreeAdapter(transport=boom_transport)
        out = adapter.translate_batch(batch)
        self.assertEqual(out.failures[0].kind, FailureKind.PROVIDER_INTERNAL)

        # Corrupted token restoration
        adapter = MyMemoryFreeAdapter(transport=lambda u, h, b: (200, {}, b'{"responseData": {"translatedText": "Corrupted <X0>"}}'))
        batch_prot = TranslationBatchRequest("j1", "en", "vi", (TranslationRequestItem("w1", "Hello <x0>", (), "s", (), ()),))
        out = adapter.translate_batch(batch_prot)
        self.assertEqual(out.failures[0].kind, FailureKind.PROTECTED_CONTENT_VIOLATION)


if __name__ == "__main__":
    unittest.main()
