import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.checksums import pa_checksum, pa_checksum_file
from crimson0020.native_checksum import native_checksum_available


class ChecksumTests(unittest.TestCase):
    def test_native_checksum_library_is_available_for_release_builds(self) -> None:
        self.assertTrue(native_checksum_available())

    def test_streaming_file_checksum_matches_in_memory_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for length in (0, 1, 11, 12, 13, 24, 25, 4097, 1024 * 1024 + 7):
                data = bytes((index * 37 + 11) & 0xFF for index in range(length))
                path = root / f"payload-{length}.bin"
                path.write_bytes(data)

                self.assertEqual(pa_checksum(data), pa_checksum_file(path))

    def test_python_fallback_matches_when_native_library_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data = bytes((index * 19 + 7) & 0xFF for index in range(8193))
            path = Path(temp_dir) / "fallback.bin"
            path.write_bytes(data)

            with patch(
                "crimson0020.checksums.native_pa_checksum_file",
                return_value=None,
            ):
                self.assertEqual(pa_checksum(data), pa_checksum_file(path))


if __name__ == "__main__":
    unittest.main()
