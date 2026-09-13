import unittest

import cloudscraper


class CloudscraperCompatibilityTests(unittest.TestCase):
    def test_sync_and_async_cloudscraper_transports_fail_closed(self) -> None:
        for factory in (
            cloudscraper.create_scraper,
            cloudscraper.create_async_scraper,
        ):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(RuntimeError, "disabled"):
                    factory()


if __name__ == "__main__":
    unittest.main()
