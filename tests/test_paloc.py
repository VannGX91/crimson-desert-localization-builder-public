import json
import struct
import tempfile
import unittest
from pathlib import Path

from crimson0020.crypto import compress_lz4
from crimson0020.paloc import (
    PalocFormatError,
    export_paloc_json,
    json_data_to_paloc,
    paloc_to_json_data,
)


def _wrap_203_paloc(raw_paloc: bytes) -> bytes:
    payload = compress_lz4(raw_paloc)
    header = b"paloc\x00\x00\x00\x00" + struct.pack(
        "<II",
        len(payload),
        len(raw_paloc),
    )
    return header + (b"\x00" * (512 - len(header))) + payload


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

    def test_203_lz4_container_round_trip_preserves_storage_format(self) -> None:
        raw_paloc = json_data_to_paloc(self.sample_data)
        wrapped_paloc = _wrap_203_paloc(raw_paloc)

        unpacked = paloc_to_json_data(wrapped_paloc)

        self.assertEqual("lz4-container-v1", unpacked["metadata"]["storage_format"])
        self.assertEqual("aabb", unpacked["metadata"]["tail_bytes"])
        self.assertEqual("Xin chào", unpacked["entries"][0]["original"])
        self.assertEqual(wrapped_paloc, json_data_to_paloc(unpacked))

    def test_203_lz4_container_rejects_nonzero_reserved_header_bytes(self) -> None:
        wrapped_paloc = bytearray(_wrap_203_paloc(json_data_to_paloc(self.sample_data)))
        wrapped_paloc[128] = 1

        with self.assertRaisesRegex(PalocFormatError, "reserved header"):
            paloc_to_json_data(bytes(wrapped_paloc))

    def test_203_lz4_container_rejects_stored_size_mismatch(self) -> None:
        wrapped_paloc = bytearray(_wrap_203_paloc(json_data_to_paloc(self.sample_data)))
        struct.pack_into("<I", wrapped_paloc, 9, len(wrapped_paloc))

        with self.assertRaisesRegex(PalocFormatError, "stored size"):
            paloc_to_json_data(bytes(wrapped_paloc))

    def test_203_lz4_container_rejects_truncated_header(self) -> None:
        truncated = b"paloc\x00\x00\x00\x00" + b"\x00" * 32

        with self.assertRaisesRegex(PalocFormatError, "header is truncated"):
            paloc_to_json_data(truncated)


if __name__ == "__main__":
    unittest.main()
