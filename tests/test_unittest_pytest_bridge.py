from __future__ import annotations

import os
import unittest
from pathlib import Path

import pytest


TESTS = Path(__file__).resolve().parent
PYTEST_ONLY_MODULES = (
    "test_router_aihorde.py",
    "test_router_cloudflare.py",
    "test_router_discovery.py",
    "test_router_duckduckgo.py",
    "test_router_health.py",
    "test_router_keyless_openai.py",
    "test_router_kilo.py",
    "test_router_opencode.py",
    "test_router_prompt.py",
    "test_router_scheduler.py",
    "test_router_unclose.py",
    "test_translation_unit_builder.py",
    "test_translation_work_knowledge.py",
)


class PytestCoverageBridgeTests(unittest.TestCase):
    def test_pytest_only_modules_are_exercised_by_unittest_discovery(self) -> None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            self.skipTest("bridge is only needed by unittest discovery")

        args = [str(TESTS / name) for name in PYTEST_ONLY_MODULES]
        exit_code = pytest.main([*args, "-q", "-p", "no:cacheprovider"])
        self.assertEqual(pytest.ExitCode.OK, exit_code)


if __name__ == "__main__":
    unittest.main()
