import unittest
from unittest.mock import patch

import app


class AppEntrypointTests(unittest.TestCase):
    def test_smoke_test_argument_checks_runtime_without_opening_main_ui(self) -> None:
        with (
            patch.object(app, "run_smoke_test", return_value=None) as smoke,
            patch.object(app, "run_app") as gui,
        ):
            result = app.main(["--smoke-test"])

        self.assertEqual(0, result)
        smoke.assert_called_once_with()
        gui.assert_not_called()

    def test_normal_launch_opens_the_main_ui(self) -> None:
        with (
            patch.object(app, "run_smoke_test") as smoke,
            patch.object(app, "run_app") as gui,
        ):
            result = app.main([])

        self.assertEqual(0, result)
        gui.assert_called_once_with()
        smoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
