import os
import subprocess
import unittest

from crimson0020 import console_window


class NoConsoleWindowTests(unittest.TestCase):
    def test_creation_flags_add_create_no_window_on_windows(self) -> None:
        merged = console_window.no_window_creationflags(0)
        if os.name == "nt":
            self.assertTrue(merged & 0x08000000)
        else:
            self.assertEqual(0, merged)

    def test_creation_flags_preserve_existing_flags(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows-only creation flags")
        existing = subprocess.CREATE_NEW_PROCESS_GROUP
        merged = console_window.no_window_creationflags(existing)
        self.assertTrue(merged & existing)
        self.assertTrue(merged & 0x08000000)

    def test_crimson0020_import_installs_and_marks_popen(self) -> None:
        console_window.install_no_window_subprocess()
        if os.name != "nt":
            self.skipTest("Windows-only creation flags")
        self.assertTrue(getattr(subprocess.Popen, "_crimson_no_window", False))


if __name__ == "__main__":
    unittest.main()
