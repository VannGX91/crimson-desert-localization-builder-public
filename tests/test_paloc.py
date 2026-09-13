import json
import tempfile
import unittest
from pathlib import Path

from crimson0020.paloc import (
    PalocFormatError,
    export_paloc_json,
    json_data_to_paloc,
    paloc_to_json_data,
)


class PalocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_data = {
            "metadata": {"tail_bytes": "aabb"},
            "entries": [
                {
                    "marker": "0102030405060708",
                    "key": "UI_HELLO",
                    "original": "Hello",
                    "translation": "Xin chào",
                },
                {
                    "marker": "1112131415161718",
                    "key": "UI_COUNT",
                    "original": "Count: {0}",
                    "translation": "Số lượng: {0}",
                },
            ],
        }

    def test_json_paloc_round_trip_preserves_existing_schema(self) -> None:
        paloc_bytes = json_data_to_paloc(self.sample_data)

        unpacked = paloc_to_json_data(paloc_bytes)

        self.assertEqual(self.sample_data["metadata"], unpacked["metadata"])
        self.assertEqual(2, len(unpacked["entries"]))
        self.assertEqual("UI_HELLO", unpacked["entries"][0]["key"])
        self.assertEqual("Xin chào", unpacked["entries"][0]["original"])
        self.assertEqual("Xin chào", unpacked["entries"][0]["translation"])

    def test_export_writes_utf8_json_with_vietnamese_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "localizationstring_eng.paloc"
            output = root / "localizationstring_eng.paloc.json"
            source.write_bytes(json_data_to_paloc(self.sample_data))

            entry_count = export_paloc_json(source, output)

            decoded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(2, entry_count)
            self.assertEqual("Xin chào", decoded["entries"][0]["translation"])
            self.assertFalse(output.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_invalid_marker_is_rejected(self) -> None:
        invalid = {
            **self.sample_data,
            "entries": [
                {
                    **self.sample_data["entries"][0],
                    "marker": "1234",
                }
            ],
        }

        with self.assertRaisesRegex(PalocFormatError, "marker"):
            json_data_to_paloc(invalid)

    def test_missing_translation_is_rejected(self) -> None:
        entry = dict(self.sample_data["entries"][0])
        del entry["translation"]
        invalid = {"metadata": {"tail_bytes": ""}, "entries": [entry]}

        with self.assertRaisesRegex(PalocFormatError, "translation"):
            json_data_to_paloc(invalid)

    def test_invalid_utf8_is_not_silently_discarded(self) -> None:
        malformed = (
            b"12345678"
            + (1).to_bytes(4, "little")
            + b"\xff"
            + (1).to_bytes(4, "little")
            + b"A"
        )

        with self.assertRaisesRegex(PalocFormatError, "UTF-8"):
            paloc_to_json_data(malformed)


if __name__ == "__main__":
    unittest.main()
