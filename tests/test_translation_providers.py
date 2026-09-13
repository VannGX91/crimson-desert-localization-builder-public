import json
import time
import unittest
from threading import Lock
from urllib.parse import parse_qs, urlparse

from crimson0020.translation_providers import (
    DEFAULT_TRANSLATION_PROVIDER,
    AlibabaWebProvider,
    FreeWebTranslationProvider,
    GoogleFreeProvider,
    LibreTranslateProvider,
    ModernMtWebProvider,
    MyMemoryFreeProvider,
    OpenAICompatibleProvider,
    RetryableTranslationProviderError,
    TranslationProviderError,
    YouDaoWebProvider,
    create_translation_provider,
    get_translation_provider_spec,
)


class TranslationProviderTests(unittest.TestCase):
    def test_default_and_provider_registry_prioritize_free_no_key_services(self) -> None:
        google = get_translation_provider_spec(DEFAULT_TRANSLATION_PROVIDER)
        mymemory = get_translation_provider_spec("mymemory-free")
        bing = get_translation_provider_spec("bing-web-free")
        alibaba = get_translation_provider_spec("alibaba-web-free")
        modernmt = get_translation_provider_spec("modernmt-web-free")
        youdao = get_translation_provider_spec("youdao-web-free")
        custom = get_translation_provider_spec("openai-compatible")

        self.assertEqual("google-web-free", google.key)
        self.assertTrue(google.free)
        self.assertIn("(Free)", google.name)
        self.assertTrue(bing.free)
        self.assertTrue(alibaba.free)
        self.assertIn("(Free)", alibaba.name)
        self.assertTrue(modernmt.free)
        self.assertIn("(Free)", modernmt.name)
        self.assertTrue(youdao.free)
        self.assertIn("(Free)", youdao.name)
        self.assertTrue(mymemory.free)
        self.assertFalse(google.requires_api_key)
        self.assertFalse(google.requires_model)
        self.assertFalse(custom.free)
        self.assertTrue(custom.requires_api_key)
        self.assertTrue(custom.requires_model)
        self.assertTrue(custom.configurable_endpoint)
        self.assertEqual("https://api.openai.com/v1", custom.endpoint)
        self.assertEqual(custom.endpoint, custom.to_state()["endpoint"])

    def test_mymemory_uses_documented_get_endpoint_without_key(self) -> None:
        calls: list[tuple[str, dict[str, str], float]] = []

        def transport(url, headers, timeout):
            calls.append((url, dict(headers), timeout))
            return 200, json.dumps(
                {"responseData": {"translatedText": "Xin chào"}}
            ).encode("utf-8")

        provider = MyMemoryFreeProvider(transport=transport)

        translated = provider.translate(("Hello",), "en", "vi")

        self.assertEqual(("Xin chào",), translated)
        self.assertIn("api.mymemory.translated.net/get?", calls[0][0])
        self.assertIn("q=Hello", calls[0][0])
        self.assertIn("langpair=en%7Cvi", calls[0][0])
        self.assertNotIn("key=", calls[0][0])

    def test_google_posts_an_ordered_batch_in_one_request(self) -> None:
        calls: list[tuple[str, bytes, dict[str, str], float]] = []

        def transport(url, payload, headers, timeout):
            calls.append((url, payload, dict(headers), timeout))
            source = parse_qs(payload.decode("utf-8"), keep_blank_values=True)["q"]
            return 200, json.dumps([f"vi:{text}" for text in source]).encode("utf-8")

        provider = GoogleFreeProvider(transport=transport)

        translated = provider.translate(("One", "Two", "Three"), "en", "vi")

        self.assertEqual(("vi:One", "vi:Two", "vi:Three"), translated)
        self.assertEqual(1, len(calls))
        self.assertEqual(
            ["One", "Two", "Three"],
            parse_qs(calls[0][1].decode("utf-8"), keep_blank_values=True)["q"],
        )
        query = parse_qs(urlparse(calls[0][0]).query)
        self.assertEqual(["at"], query["client"])
        self.assertEqual(["1"], query["dj"])
        self.assertEqual("/translate_a/t", urlparse(calls[0][0]).path)

    def test_mymemory_splits_text_over_500_bytes_instead_of_aborting_job(self) -> None:
        calls: list[str] = []

        def transport(url, _headers, _timeout):
            source = parse_qs(urlparse(url).query)["q"][0]
            calls.append(source)
            return 200, json.dumps(
                {"responseData": {"translatedText": f"VI:{source}"}}
            ).encode("utf-8")

        provider = MyMemoryFreeProvider(transport=transport)

        translated = provider.translate((("word " * 130).strip(),), "en", "vi")

        self.assertGreater(len(calls), 1)
        self.assertTrue(all(len(text.encode("utf-8")) <= 500 for text in calls))
        self.assertEqual(len(calls), translated[0].count("VI:"))

    def test_mymemory_chunking_preserves_spacing_and_protected_markers(self) -> None:
        calls: list[str] = []

        def transport(url, _headers, _timeout):
            source = parse_qs(
                urlparse(url).query,
                keep_blank_values=True,
            )["q"][0]
            calls.append(source)
            return 200, json.dumps(
                {"responseData": {"translatedText": source}}
            ).encode("utf-8")

        provider = MyMemoryFreeProvider(transport=transport)
        marker = "__CRIMSON_KEEP_0000_1234ABCD__"
        source = f"  {('word  ' * 85)}{marker}  tail  "

        translated = provider.translate((source,), "en", "vi")

        self.assertEqual(source, translated[0])
        self.assertEqual(1, sum(call.count(marker) for call in calls))
        self.assertTrue(all(len(call.encode("utf-8")) <= 500 for call in calls))

    def test_libretranslate_posts_an_ordered_batch_and_validates_response(self) -> None:
        calls: list[tuple[str, bytes, dict[str, str], float]] = []

        def transport(url, payload, headers, timeout):
            calls.append((url, payload, dict(headers), timeout))
            return 200, json.dumps(
                {"translatedText": ["Xin chào", "Thế giới"]}
            ).encode("utf-8")

        provider = LibreTranslateProvider(
            key="test",
            endpoint="https://translate.example.test",
            timeout=17.0,
            transport=transport,
        )

        translated = provider.translate(("Hello", "World"), "en", "vi")

        self.assertEqual(("Xin chào", "Thế giới"), translated)
        self.assertEqual("https://translate.example.test/translate", calls[0][0])
        self.assertEqual(
            {
                "q": ["Hello", "World"],
                "source": "en",
                "target": "vi",
                "format": "text",
            },
            json.loads(calls[0][1]),
        )
        self.assertEqual("application/json", calls[0][2]["Content-Type"])
        self.assertEqual(17.0, calls[0][3])

    def test_libretranslate_rejects_wrong_result_count(self) -> None:
        provider = LibreTranslateProvider(
            key="test",
            endpoint="https://translate.example.test",
            transport=lambda *_args: (
                200,
                json.dumps({"translatedText": ["Một"]}).encode("utf-8"),
            ),
        )

        with self.assertRaisesRegex(TranslationProviderError, "result count"):
            provider.translate(("One", "Two"), "en", "vi")

    def test_retryable_http_status_is_typed(self) -> None:
        provider = LibreTranslateProvider(
            key="test",
            endpoint="https://translate.example.test",
            transport=lambda *_args: (429, b'{"error":"slow down"}'),
        )

        with self.assertRaises(RetryableTranslationProviderError):
            provider.translate(("Hello",), "en", "vi")

    def test_endpoint_policy_rejects_remote_plain_http_and_url_credentials(self) -> None:
        for endpoint in (
            "http://translate.example.test",
            "https://user:password@translate.example.test",
            "file:///tmp/translate",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    LibreTranslateProvider(key="test", endpoint=endpoint)

    def test_provider_registry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown translation provider"):
            create_translation_provider("arbitrary-python")

    def test_free_web_provider_runs_bounded_parallel_requests_and_keeps_order(self) -> None:
        lock = Lock()
        active = 0
        peak = 0

        def translate_one(text: str, source: str, target: str) -> str:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                self.assertEqual("en", source)
                self.assertEqual("vi", target)
                time.sleep(0.01)
                return f"vi:{text}"
            finally:
                with lock:
                    active -= 1

        provider = FreeWebTranslationProvider(
            key="synthetic-free",
            translator=translate_one,
            max_workers=3,
        )

        translated = provider.translate(tuple(str(i) for i in range(8)), "en", "vi")

        self.assertEqual(tuple(f"vi:{i}" for i in range(8)), translated)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 3)

    def test_openai_compatible_uses_configured_endpoint_model_and_indexed_json(self) -> None:
        captured: dict[str, object] = {}

        def transport(url, payload, headers, timeout):
            captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
            response = {
                "choices": [
                    {"message": {"content": '{"1":"Thế giới","0":"Xin chào"}'}}
                ]
            }
            return 200, json.dumps(response).encode("utf-8")

        provider = OpenAICompatibleProvider(
            base_url="https://gateway.example.test/v1",
            model="user-model",
            api_key="session-secret",
            transport=transport,
        )

        translated = provider.translate(("Hello", "World"), "en", "vi")

        self.assertEqual(("Xin chào", "Thế giới"), translated)
        self.assertEqual(
            "/v1/chat/completions", urlparse(str(captured["url"])).path
        )
        request = json.loads(captured["payload"])
        self.assertEqual("user-model", request["model"])
        self.assertIn('\"0\":\"Hello\"', request["messages"][1]["content"])
        self.assertEqual(
            "Bearer session-secret", captured["headers"]["Authorization"]
        )

    def test_openai_compatible_requires_user_model_and_key(self) -> None:
        for config in (
            {"base_url": "https://api.example.test/v1", "model": "", "api_key": "k"},
            {"base_url": "https://api.example.test/v1", "model": "m", "api_key": ""},
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    create_translation_provider("openai-compatible", config)

    def test_openai_compatible_localhost_allows_no_api_key_and_omits_auth_header(self) -> None:
        captured: dict[str, object] = {}

        def transport(url, payload, headers, timeout):
            captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
            response = {"choices": [{"message": {"content": '{"0":"Xin chào"}'}}]}
            return 200, json.dumps(response).encode("utf-8")

        provider = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:1234/v1",
            model="local-model",
            api_key="",
            transport=transport,
        )

        self.assertEqual(("Xin chào",), provider.translate(("Hello",), "en", "vi"))
        self.assertNotIn("Authorization", captured["headers"])

    def test_openai_compatible_factory_applies_custom_performance_profile(self) -> None:
        provider = create_translation_provider(
            "openai-compatible",
            {
                "base_url": "http://localhost:1234/v1",
                "model": "local-model",
                "api_key": "",
                "max_batch_items": 128,
                "max_batch_characters": 48000,
                "concurrency": 4,
                "timeout_seconds": 240,
            },
        )

        self.assertEqual(128, provider.preferred_batch_items)
        self.assertEqual(48000, provider.preferred_batch_characters)
        self.assertEqual(4, provider.preferred_outer_concurrency)
        self.assertEqual(240.0, provider.timeout)

    def test_openai_compatible_splits_large_batch_after_bad_indexed_json(self) -> None:
        calls: list[dict[str, object]] = []

        def transport(_url, payload, _headers, _timeout):
            request = json.loads(payload)
            source = json.loads(request["messages"][1]["content"].split("JSON:\n", 1)[1])
            calls.append(source)
            if len(source) == 4:
                content = "not-json"
            else:
                content = json.dumps(
                    {key: f"vi:{value}" for key, value in source.items()}
                )
            return 200, json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")

        provider = OpenAICompatibleProvider(
            base_url="https://gateway.example.test/v1",
            model="user-model",
            api_key="session-secret",
            transport=transport,
        )

        translated = provider.translate(("a", "b", "c", "d"), "en", "vi")

        self.assertEqual(("vi:a", "vi:b", "vi:c", "vi:d"), translated)
        self.assertEqual([4, 2, 2], [len(call) for call in calls])

    def test_alibaba_modernmt_youdao_providers_translate_and_keep_order(self) -> None:
        def mock_translator(text: str, source: str, target: str) -> str:
            return f"{target}:{text}"

        for provider_cls, key in (
            (AlibabaWebProvider, "alibaba-web-free"),
            (ModernMtWebProvider, "modernmt-web-free"),
            (YouDaoWebProvider, "youdao-web-free"),
        ):
            with self.subTest(provider=key):
                provider = provider_cls(translator_func=mock_translator)
                self.assertEqual(key, provider.key)
                translated = provider.translate(("hello", "world"), "en", "vi")
                self.assertEqual(("vi:hello", "vi:world"), translated)

    def test_factory_creates_all_ordered_providers(self) -> None:
        for key in (
            "google-web-free",
            "google_free",
            "bing-web-free",
            "alibaba-web-free",
            "modernmt-web-free",
            "youdao-web-free",
        ):
            with self.subTest(key=key):
                provider = create_translation_provider(key)
                self.assertIsNotNone(provider)
                self.assertTrue(hasattr(provider, "translate"))


if __name__ == "__main__":
    unittest.main()
